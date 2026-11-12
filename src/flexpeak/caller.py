"""Peak calling orchestration: coverage -> background -> segmentation -> NestedPeakSet."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from .background.nb import BackgroundModel, check_window_scale
from .peaks import NestedPeakSet, Region, SubPeak
from .segment.hmm import NBHMM, _runs, segment_chromosome
from .segment.subpeak import DEFAULT_BANDWIDTHS, find_subpeaks
from .signal.coverage import Coverage
from .stats.nbtest import (
    aggregate_nb_params, ar1_factor, autocorrelation_length_bp, nb_neglog10_sf,
    qvalue_from_neglog10p, rho_from_inflation,
)

__all__ = ["CallParams", "call_peaks", "PRESETS", "robust_start_enabled",
           "weak_domain_window_bp"]


@dataclass
class CallParams:
    """Everything that controls a call.  Serialised into every output header."""

    bin_size: int = 25
    windows: Tuple[int, ...] = (1000, 5000, 10000)
    bandwidths: Tuple[int, ...] = DEFAULT_BANDWIDTHS
    posterior_cutoff: float = 0.5
    min_width: int = 100
    max_gap: int = 0
    """Connect regions separated by <= max_gap bp *when the gap carries similar
    enrichment* (see ``gap_fold_fraction``).  0 disables.  The quick scan
    (``flexpeak.scan``) sets it from the sample's own inter-run gap
    distribution; the HMM already infers boundaries, so this only joins the
    pieces of one feature that a dip in coverage split."""
    gap_fold_fraction: float = 0.5
    """Two neighbouring regions within ``max_gap`` are merged only if the fold
    enrichment of the gap between them is at least this fraction of the
    weaker neighbour's fold (and at least ``min_fold``).  That is what keeps a
    distance rule from fusing two features across genuine background -- the
    SICER/epic2 failure mode -- while still connecting nearby similar
    measurements.  <= 0 restores a plain distance merge."""
    min_fold: float = 1.2
    """Effect-size floor on a region's observed/expected fold.

    It was 1.5, which silently erased weak broad marks: the background is a
    max over windows and so sits a little above the true level, and a 1.6x
    domain measures ~1.35x against it.  On the low-quality benchmark (1.6x
    domains, dispersion 0.4) that took base-pair F1 from 96% at 1.2 to 13% at
    1.5 with nothing else changed.  Significance, not this floor, is what keeps
    noise out -- see ``fdr_scope``."""
    qvalue: float = 0.05
    fdr_scope: str = "genome"
    """Multiplicity the region-level BH correction is taken over.

    ``"genome"``: as many tests as candidate-sized tiles fit in the genome
    (genome size / median candidate width).  The HMM *scanned* the whole
    genome before handing over its candidates, so correcting over only the
    handful it returned is selection without paying for it -- on the null
    benchmark that let 16 noise regions through at ``min_fold=1.2``; with the
    genome-wide count it is 2, below the 3 the old 1.5 floor allowed.
    ``"candidates"``: BH over the returned regions only (the old behaviour)."""
    min_persistence: int = 2
    subpeak_descent_level: int = 1
    """Which smoothing level sub-peak boundaries are walked on; see
    ``find_subpeaks(descent_level=...)``.  0 is the raw-signal ablation."""
    subpeak_vs_domain: bool = True
    """Inside a domain, judge sub-peaks against the domain's own level.

    A sub-peak nested in a domain claims focal structure *on top of* the
    domain, so that is what it must be tested against; tested against genome
    background every bump on a 2.5x plateau is "enriched".  The domain level
    is the median of the 1 kb-smoothed fold inside the region, and the fold
    floor and the NB test both use ``mu`` scaled by it.  Measured on the
    benchmark: the 1.6x low-quality domains, which contain no focal peaks,
    went from ~150 reported sub-peaks to none, while all 18 true focal peaks
    in the mixed domains stayed (-log10 p 15-29 against the domain, versus a
    95th percentile of 1.3-3.9 for the noise).

    The test is deliberately run *without* the genome-wide autocorrelation
    inflation.  That factor is estimated at domain scale and conditioning on
    the domain level is what removes it; applied anyway it is ~14x on the
    mixed benchmark and buries the true focal peaks at -log10 p ~2.  Off:
    everything is tested against genome background (ablation)."""
    subpeak_domain_z: float = 4.0
    """Outlier gate for sub-peaks inside a domain, in robust sigmas.

    Sub-peak candidates are local maxima, so their p-values are not uniform
    under the null -- they are the upper tail by construction, and BH on them
    passed a third of the noise bumps on the low-quality domains.  But a
    domain supplies its own null: nearly all of its candidate maxima are
    noise, so the median and MAD of their log fold over the domain level
    measure that noise directly, and a real focal peak is an outlier against
    it.  Measured on the benchmark, every true focal peak scored z >= 5.1 and
    none of ~2,400 noise candidates exceeded 3.8.  The gate self-calibrates:
    a noisier library widens its own null.

    The null is pooled over every domain in the sample: each candidate's fold
    is already relative to its own domain's level, so they are comparable,
    and pooling is what lets a 3 kb domain with four candidates be judged
    at all.  Needs ``subpeak_domain_min_candidates`` candidates in the pool;
    0 disables the gate."""
    subpeak_domain_min_candidates: int = 10
    min_subpeak_width: Optional[int] = None
    """Narrowest sub-peak reported, bp.  None means one detection bandwidth
    at the descent level (150 bp by default): a footprint narrower than the
    smoothing that found it is a single-bin spike, not a feature."""
    call_subpeaks: bool = True
    domain_min_width: int = 2000
    """Regions at least this wide are labelled ``domain`` rather than ``peak``."""
    background_model: str = "nb"  # "nb" | "poisson" (ablation, DESIGN.md 11)
    boundary_fraction: float = 0.5
    hmm_stay_background: float = 0.995
    hmm_stay_enriched: float = 0.98
    background_exclude_fold: float = 2.0
    """Bins above this fold are held out when fitting the local background.

    Without it a peak sets its own background: the windowed mean it is divided
    by includes the peak, so fold enrichment collapses toward 1 in proportion to
    how much of the window the feature fills.  0 restores the plain windowed
    mean (DESIGN.md 11 ablation)."""
    background_exclude_iters: int = 3
    dispersion: Optional[Tuple[float, float]] = None
    """Pin the per-bin dispersion trend ``(a0, a1)`` (``alpha = a0 + a1 / mu``)
    instead of fitting it on this sample.  The fit is one number pair per
    library, and libraries of one experiment should not be tested under
    different noise models because their estimates differ: fit each, take
    the median (``<name>_qc.json``: ``dispersion_a0`` / ``dispersion_a1``),
    and pin it for all of them."""
    segment_dispersion: Optional[Tuple[float, float]] = None
    """The same pin for the trend on segmentation-bin sums, which is the
    dispersion the HMM segments with (``segment_dispersion_a0`` / ``_a1`` in
    the QC)."""
    max_width: Optional[int] = None
    """Widest region reported, bp; None: no cap.  A region wider than this is
    reported as its sub-peaks (each a standalone ``peak``) or, if it has
    none, dropped.  For assays whose features are narrow by nature (the
    ``atac`` and ``tf`` presets): a 100 kb "accessible domain" is an
    amplified or otherwise elevated-background stretch, not open
    chromatin."""
    background_depth_aware: bool = True
    """Size the background windows and the signal exclusion by the fragments
    they hold, on count data of known depth (``effective_windows`` and
    ``_signal_threshold`` in ``background.nb``): each window is widened to
    hold ~20 fragments, and a stretch is held out as signal only when it is
    also 3 SD of fragment noise above the background.  A deep library is
    unaffected; off restores the windows and the plain fold rule (ablation)."""
    background_robust_start: Optional[bool] = None
    """Start the background's signal exclusion from a running median instead
    of the plain windowed mean (see ``local_lambda(robust_start=...)``).
    None: on when every background window is >= ``DOMAIN_WINDOW_BP`` (the
    domain-mark setting), where moderate enrichment across the window
    otherwise becomes part of the background."""
    weak_domain_bp: Optional[int] = None
    """Window of the weak-domain pass, bp; 0 disables, None picks
    max(10 kb, 2 x domain_min_width) when the background windows are
    domain-scale (>= ``DOMAIN_WINDOW_BP``) and disables it otherwise.

    The HMM decides state bin by bin at the segmentation scale, and a long,
    low (1.5-2.5x) stretch full of dips drops back to background at every
    dip: it comes out as scattered pieces or not at all.  This pass tests
    sliding sums over
    the window against the wide background with the same NB + AR(1) model
    the region test uses, against the wide background, Bonferroni over
    genome / window independent windows, and adds significant runs at
    ``weak_domain_min_fold`` as candidate regions.  They then face what any
    region faces: the region test and FDR, and -- holding focal peaks --
    the residual test that dissolves a "domain" that is only its peaks."""
    weak_domain_min_fold: float = 1.5
    background_exclude_smooth_bp: Optional[int] = None
    """Scale at which signal is judged before being held out of the background.

    None derives it from the windows.  It is coupled to
    ``background_exclude_fold``: judging a wider average makes noise cross the
    threshold far less often, which is what lets the threshold come down far
    enough to see a broad domain sitting at only ~1.5x background."""
    hmm_fit_fraction: float = 1.0
    """Share of each chromosome the HMM's Baum-Welch iterations run on
    (evenly spaced blocks; see ``NBHMM.fit_fraction``).  The posterior the
    calls are made from is always computed on the whole chromosome."""
    hmm_fit_state_dispersion: bool = False
    """Fit each HMM state's dispersion instead of reusing the background trend.
    Off: measured to cost more sensitivity than it buys (see NBHMM)."""
    segment_bin_size: Optional[int] = None
    """Bin size the HMM segments at, independent of the reporting bin size.

    Per-bin state inference needs enough signal in a bin to tell the states
    apart.  These libraries are 80-95% empty at 100 bp, so inside a real domain
    the model meets long runs of zeros, reads them as background, and cuts the
    domain there -- which is why a broad mark comes back as tens of thousands of
    ~400 bp fragments.  Coarsening only the *segmentation* fixes that without
    giving up fine-resolution sub-peaks: on chr4 H3K9me3, going from 100 bp to
    500 bp took 10,169 regions of median 600 bp down to 1,407 of median 3,500 bp
    while called territory stayed put (5.7% -> 5.2%) -- the same signal, merged
    rather than shredded.

    ``None`` picks it from the data (see ``_segment_bin_size``); set it
    explicitly, or equal to ``bin_size``, to segment at reporting resolution."""
    dwell_from_autocorrelation: bool = True
    """Floor the HMM's enriched dwell time at the signal's own autocorrelation
    length, so a free Baum-Welch fit cannot shred domains into fragments far
    below the scale the data has.  See NBHMM.min_stay_enriched."""
    segment_bins_per_scale: int = 8
    """How many segmentation bins span one autocorrelation length."""
    min_segment_bin_bp: int = 100
    """Floor on the *automatically* chosen segmentation bin.

    The autocorrelation length collapses to a single bin on a weak or
    overdispersed library -- sparsity decorrelates a signal that is really
    continuous -- and then the HMM decides a state per 25 bp bin, which no
    sparse bin can support: a 1.6x domain came back as ~50 fragments and pure
    noise grew a "weak" state at 1.7x.  Four bins is the smallest unit that
    carries a decision.  Measured: free on every full-size benchmark set
    (mixed +0.7 F1), 6x fewer false calls on a small null genome, and the
    shredded weak domains went from F1 0.50 to 0.78 (the tuner takes them the
    rest of the way).  200 bp already costs TF-scale peaks (0.93 -> 0.88).
    Not applied to an explicit ``segment_bin_size``."""
    max_dwell: int = 100_000
    """Cap on that floor, in bp -- a degenerate autocorrelation estimate on a
    short or repetitive contig must not turn the chromosome into one region."""
    focal_windows: Tuple[int, ...] = (1_000, 5_000, 10_000)
    """MACS-style local windows for the focal pass (``focal_pass``).  Plain
    local means (no signal exclusion) maxed with the wide background, so the
    focal background rises to the local level inside a promoter cluster or a
    domain and only the summits stand out -- which is how MACS keeps H3K4me3
    peaks narrow, and what a wide-window background alone cannot do: with
    150/750 kb windows the valleys between promoters sit far above
    background and a whole cluster fused into one 40-85 kb region (measured
    on GSE285245 H3K4me3 / H3K27ac at Havcr2 and Tcf7).  Without a control
    the windows below 5 kb are dropped, as MACS drops its 1 kb window: a
    1 kb plain mean of the treatment at a summit *is* the summit, every
    peak reads as fold 1 and the cluster fuses again (measured)."""
    focal_pass: bool = True
    """Run a second, fine-scale segmentation against the focal background
    whenever ``windows`` are all wider than ``focal_windows``.  Its regions
    become the sub-peaks of the domains they fall in (tested against the
    focal background, i.e. relative to the local level) and standalone
    ``peak`` regions elsewhere; the scale-space sub-peak search is then
    skipped.  Off, or when the windows already are the focal ones, the
    scale-space search runs as before."""
    focal_max_segment_bp: int = 500
    """Cap on the focal pass's segmentation bin."""
    focal_min_fold: float = 1.3
    """Fold floor of a focal region over its local background (MACS's
    ``--fe-cutoff`` idea; MACS's own default is 1.0): with deep coverage a
    1.1x bump over the local mean is "significant", and that is not a
    peak.  1.3 rather than 1.5 because the local mean includes the peak
    itself, so a 2x summit over a plateau measures ~1.4x."""
    focal_gap_bins: int = 2
    """Significant focal bins separated by at most this many segmentation
    bins are one region."""
    focal_test_window_bp: Optional[int] = None
    """Width of the sliding sum the focal test is run on, stepped at the
    segmentation bin.  None: the focal feature scale, clipped to 200-1000 bp.
    MACS's per-base pileup is a sum over the fragment length, and that sum
    is where its power comes from; a single 25 bp bin of a normalised track
    is so overdispersed (alpha 1-1.6 on GSE285245 bigWigs) that a 2.5x
    summit is 1.5 SD and never significant, which left every promoter
    cluster as one block."""
    focal_scales: Tuple[int, ...] = ()
    """Sliding-window widths (bp) the focal test is run at, finest first.

    Empty: one window, ``focal_test_window_bp`` (or the focal feature scale
    clipped to 200-1000 bp).  The quick scan sets a ladder from the mark's
    own run-width distribution (``ScanRecommendation.focal_scales``): the
    median bin-level run, the 90th percentile run, and for domain marks the
    median run after 5 kb smoothing.  One window cannot serve a 200 bp
    H3K9me3 spike and a 5 kb cluster: a 1 kb window dilutes the spike
    ~3x, and a 200 bp window has no power on the cluster.  Each scale gets
    its own background and dispersion; scales are then assembled fine to
    coarse (see ``_assemble_scales``)."""
    focal_background: str = "flank"
    """Focal background at each scale.

    ``"flank"``: MACS's local lambda with the tested window left out -- the
    mean of the flanks over ``focal_flank_factors`` x the window (at least
    ``focal_min_flank_bp``) on each side, excluding a guard of half a window
    around it, max over the flank sizes, floored at the robust background
    level (the median of the smoothed track).  Leaving the window out is
    what lets a sub-peak inside a domain be tested against the domain
    around it rather than against a mean it is part of; the median floor is
    what stops a quiet neighbourhood from being judged against the genome
    MEAN (measured at Pdcd1 H3K9me3: focal background 100/kb against a
    true level of 5-15/kb, so a 10x spike read 1.2x).
    ``"plain"``: the previous behaviour -- plain 10 kb mean (peak included)
    maxed with the wide background (ablation)."""
    domain_residual_share: float = 0.5
    """A first-pass region holding focal peaks stands as a domain only if
    the signal BETWEEN its peaks carries at least this share of its excess
    over the wide background; otherwise it is reported as its peaks.  A
    Polycomb or H3K9me3 domain is mostly between-peak territory; a promoter
    cluster is mostly its summits.  0 disables (residual fold and p only)."""
    focal_flank_factors: Tuple[float, ...] = (10.0, 50.0)
    focal_min_flank_bp: int = 5_000
    focal_floor_fraction: float = 0.5
    """The focal background never drops below this fraction of the robust
    background level (median of the smoothed track).  1.0 is MACS's rule
    (local lambda never below the genome level); lower lets a quiet
    neighbourhood be its own background.  0.5 chosen on pooled H3K9me3
    chr1 against replicate support (spikes >= 4x pooled and >= 2x in >= 3
    of 4 replicates): with clipped dispersion, recall 0.62 at 1.0, 0.81 at
    0.5, 0.84 at 0.25; share of narrow calls with replicate support 0.99 /
    0.97 / 0.94; simulated null 3 calls at all three."""
    focal_dispersion: str = "clipped"
    """How the window-sum dispersion is estimated on background windows:
    ``"moments"`` (variance of the windows below the 95th percentile, per mean
    group) or ``"clipped"`` (iteratively refitted with the windows the fit
    itself calls enriched removed -- see ``_clipped_dispersion``)."""
    focal_fdr: str = "pooled"
    """``"pooled"``: one BH over every scale; ``"per_scale"``: each window
    size its own family."""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["windows"] = list(self.windows)
        d["bandwidths"] = list(self.bandwidths)
        d["focal_scales"] = list(self.focal_scales)
        d["focal_flank_factors"] = list(self.focal_flank_factors)
        for k in ("dispersion", "segment_dispersion"):
            d[k] = list(d[k]) if d[k] is not None else None
        return d

    def replace(self, **kw) -> "CallParams":
        d = asdict(self)
        d.update(kw)
        return CallParams(**d)


PRESETS: Dict[str, Dict] = {
    # Derived from the classic narrow/broad split; the runtime tuner (flexpeak.tune)
    # normally supersedes these.  They exist as the deterministic fallback.
    "tf": dict(bin_size=10, windows=(1000, 5000, 10000), bandwidths=(50, 150, 500),
               min_width=100, domain_min_width=1500, min_fold=2.0, posterior_cutoff=0.6),
    "h3k4me3": dict(bin_size=25, windows=(1000, 5000, 10000),
                    bandwidths=(150, 500, 1500), min_width=200, domain_min_width=3000),
    "h3k27ac": dict(bin_size=25, windows=(10_000, 50_000),
                    bandwidths=(150, 500, 1500, 5000), min_width=250, domain_min_width=3000),
    # Broad-mark windows are 1 Mb / 5 Mb, not 200 kb / 1 Mb: measured on
    # GSE285245 H3K27me3 (chr11) the weak Polycomb neighbourhood around Havcr2
    # (2.4x over the chromosome background, several hundred kb wide) became
    # its own background at 200 kb and was never called; at 1 Mb it is.
    "h3k27me3": dict(bin_size=100, windows=(1_000_000, 5_000_000),
                     bandwidths=(500, 1500, 5000, 15000), min_width=500,
                     domain_min_width=5000, min_fold=1.2, posterior_cutoff=0.45),
    # No preset pins segment_bin_size, and that is a measured decision, not an
    # oversight.  Coarse segmentation helps a real broad mark -- on CUT&RUN
    # H3K27me3, 900 bp -> 10 kb raised territory 17.2% -> 19.3% against a 20.2%
    # envelope with FRiP flat.  But pinning 10 kb into the preset cost 87.5% ->
    # 50.0% recall on the mixed narrow+broad benchmark, because a 10 kb
    # segmentation bin cannot resolve a 300 bp peak.  The right scale is a
    # property of the library, not of the mark, so it is swept by the tuner
    # (see tune.default_grid) and settable per run with --segment-bin-size.
    "h3k9me3": dict(bin_size=100, windows=(1_000_000, 5_000_000),
                    bandwidths=(500, 1500, 5000, 15000), min_width=300,
                    domain_min_width=5000, min_fold=1.2, posterior_cutoff=0.45),
    "h3k36me3": dict(bin_size=100, windows=(1_000_000, 5_000_000),
                     bandwidths=(500, 1500, 5000), min_width=500,
                     domain_min_width=5000, min_fold=1.2),
    # max_width: accessible elements are sub-kb to a few kb, clusters of them
    # (super-enhancers, HOX) up to ~20 kb.  On 18 SNU449 libraries the widest
    # region called with a sound background was 12.6 kb; the cap only ever
    # acts on what a broken background lets through (amplified stretches of
    # an aneuploid genome read as 100 kb - 2 Mb "domains").
    # hmm_fit_fraction: measured on six of those libraries against a full
    # fit -- region Jaccard 0.994-0.995, recall of MACS3 peaks, overlap with
    # the pooled-replicate set and replicate support all unchanged to +/-
    # 0.001, segmentation 10-20 s -> 6-9 s.  Only this preset samples by
    # default; elsewhere it has not been measured.
    "atac": dict(bin_size=10, windows=(1000, 5000, 10000), bandwidths=(50, 150, 500),
                 min_width=100, domain_min_width=1500, min_fold=2.0, max_width=50_000,
                 hmm_fit_fraction=0.25),
}


def preset_params(name: str, **overrides) -> CallParams:
    key = name.lower().replace("-", "").replace("_", "")
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    d = dict(PRESETS[key])
    d.update(overrides)
    return CallParams(**d)


@dataclass
class CallResult:
    peaks: NestedPeakSet
    background: BackgroundModel
    params: CallParams
    diagnostics: Dict = field(default_factory=dict)


def call_peaks(
    treatment: Coverage,
    control: Optional[Coverage] = None,
    params: Optional[CallParams] = None,
    name: str = "flexpeak",
    covariates: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    blacklist: Optional[Sequence[Tuple[str, int, int]]] = None,
    return_details: bool = False,
    background: Optional[BackgroundModel] = None,
    segmentation_cache: Optional[Dict] = None,
    n_jobs: Optional[int] = None,
):
    """Call nested peaks on binned coverage.

    Returns a :class:`NestedPeakSet`, or a :class:`CallResult` if
    ``return_details`` is set (used by QC and by the tuning sweep).

    ``background`` reuses an already fitted model for this exact treatment,
    control and bin size -- the tuning sweep varies nothing the background
    depends on, so refitting it at every grid point is pure waste.

    ``segmentation_cache`` is a dict, owned by the caller, that keeps each
    chromosome's HMM fit keyed on the settings the fit depends on.  The
    posterior cutoff, ``min_fold`` and everything downstream of the posterior
    leave the fit unchanged, so a sweep over them fits the HMM once instead of
    at every grid point.  Pass one dict per (treatment, background) pair.

    ``n_jobs`` calls the chromosomes in that many worker processes
    (``<= 0``: every CPU allocated to this process; ``None``/1: serial).  The
    result is identical to the serial call.
    """
    params = params or CallParams()

    # Coverage can be coarsened but never refined: resolution not read from the
    # source file cannot be invented.  Asking for a finer bin than the loaded
    # coverage is a common, recoverable mistake (a preset tuned for TFs applied
    # to coverage binned for broad marks), so adapt and say so rather than
    # failing after the expensive load.
    if params.bin_size < treatment.bin_size:
        warnings.warn(
            f"preset/parameters request {params.bin_size} bp bins but the coverage "
            f"is binned at {treatment.bin_size} bp; using {treatment.bin_size} bp. "
            f"Re-load the source file with bin_size={params.bin_size} for full "
            f"resolution.",
            stacklevel=2,
        )
        params = params.replace(bin_size=treatment.bin_size)

    if treatment.bin_size != params.bin_size:
        treatment = treatment.rebin(params.bin_size)
    if control is not None and control.bin_size != params.bin_size:
        control = control.rebin(params.bin_size)

    msg = check_window_scale(params.windows, params.domain_min_width,
                             control is not None, params.background_exclude_fold)
    if msg:
        warnings.warn(msg, stacklevel=2)

    import time as _time

    timing: Dict[str, float] = {}
    _t = _time.perf_counter()

    def _lap(label: str) -> None:
        nonlocal _t
        now = _time.perf_counter()
        timing[label] = timing.get(label, 0.0) + now - _t
        _t = now

    bg = background if background is not None else fit_background(
        treatment, control, params, covariates, n_jobs=n_jobs)
    if params.dispersion is not None:
        import dataclasses as _dc

        bg = _dc.replace(bg, a0=float(params.dispersion[0]), a1=float(params.dispersion[1]))
    _lap("background")

    bs = treatment.bin_size
    min_bins = max(1, params.min_width // bs)
    regions: List[Region] = []
    diagnostics: Dict = {"chroms": {}, "timing": timing}

    # One scale estimate for the whole sample, used for both the segmentation
    # bin and the HMM dwell floor.  Per chromosome it is too unstable to trust
    # (see signal_scale_bp), and a setting that silently switches off on half
    # the chromosomes is worse than one that is simply wrong.
    cache = segmentation_cache if segmentation_cache is not None else {}
    if "signal_scale_bp" not in cache:
        cache["signal_scale_bp"] = signal_scale_bp(treatment, n_jobs=n_jobs)
    scale_bp = cache["signal_scale_bp"]
    _lap("signal_scale")
    seg_bs = _segment_bin_size(treatment, params, scale_bp)
    factor = max(1, seg_bs // bs)
    diagnostics["signal_scale_bp"] = int(scale_bp)
    diagnostics["segment_bin_size"] = int(factor * bs)

    # The HMM's dispersion is measured at the scale the HMM works at: on the
    # segmentation-bin sums, not derived from the per-bin trend.  A sparse
    # pileup at 10-25 bp is zero-inflated and its per-bin trend degenerates
    # (measured on a CUT&RUN CTCF replicate: a0 0, a1 at the 100 clip, so
    # alpha = 100/mu, and 1.6 Gb "called"); the sums are not.  The same
    # choice the focal test makes for its window sums.
    seg_key = ("seg_dispersion", factor, params.background_model)
    if seg_key not in cache:
        from .background.nb import estimate_dispersion

        cs, ms = [], []
        for chrom in sorted(treatment.bins):
            c, m = treatment.bins[chrom], bg.mu[chrom]
            n = min(c.size, m.size)
            if factor > 1:
                cs.append(_coarsen(c[:n], factor))
                ms.append(_coarsen(m[:n], factor))
            else:
                cs.append(c[:n].astype(np.float64))
                ms.append(m[:n].astype(np.float64))
        c_all, m_all = np.concatenate(cs), np.concatenate(ms)
        if robust_start_enabled(params) and params.background_exclude_fold > 0:
            # Background segmentation bins only, as the background's own
            # dispersion is fitted with the anchored estimate (see
            # BackgroundModel.fit): with the 95th-percentile trim the
            # moderate enrichment widens the emissions until a 45 kb empty
            # gap (Pdcd1 gene body, H3K27me3) no longer drops to background.
            bgk = c_all <= params.background_exclude_fold * m_all
            if bgk.sum() >= 2_000:
                c_all, m_all = c_all[bgk], m_all[bgk]
        if c_all.size > 4_000_000:
            step = int(np.ceil(c_all.size / 4_000_000))
            c_all, m_all = c_all[::step], m_all[::step]
        cache[seg_key] = estimate_dispersion(c_all, m_all)
    seg_a0, seg_a1 = cache[seg_key]
    diagnostics["segment_dispersion_fitted"] = {"a0": float(seg_a0), "a1": float(seg_a1)}
    if params.segment_dispersion is not None:
        seg_a0, seg_a1 = (float(v) for v in params.segment_dispersion)
    diagnostics["segment_dispersion"] = {"a0": float(seg_a0), "a1": float(seg_a1)}

    _lap("segment_dispersion")
    weak_bp = weak_domain_window_bp(params)
    genome_bp = int(sum(treatment.chrom_sizes.get(c, a.size * bs)
                        for c, a in treatment.bins.items()))
    diag_weak = 0
    skip_sub = (params.focal_pass and bool(params.focal_windows)
                and bool(params.windows)
                and min(params.windows) > max(params.focal_windows))
    ctx = _ChromContext(treatment=treatment, bg=bg, params=params, factor=factor,
                        scale_bp=scale_bp, seg_a0=seg_a0, seg_a1=seg_a1,
                        weak_bp=weak_bp, genome_bp=genome_bp, skip_subpeaks=skip_sub,
                        cache=cache, keep_fit=segmentation_cache is not None)
    order = sorted(treatment.bins)
    for chrom, (regs, diag, n_weak, key, fit) in zip(
            order, _map_chromosomes(ctx, order, n_jobs)):
        regions.extend(regs)
        diagnostics["chroms"][chrom] = diag
        diag_weak += n_weak
        if fit is not None:
            cache[key] = fit
    _lap("segmentation")

    diagnostics["weak_domain_bp"] = int(weak_bp)
    diagnostics["n_weak_domain_candidates"] = int(diag_weak)

    focal_ran = False
    if params.focal_pass and params.focal_windows and params.windows and \
            min(params.windows) > max(params.focal_windows):
        regions, focal_diag = _focal_pass(treatment, control, params, bg, regions,
                                          covariates, cache)
        diagnostics["focal"] = focal_diag
        focal_ran = True

    peaks = NestedPeakSet(
        regions,
        params={
            "flexpeak_version": _version(),
            "call": params.to_dict(),
            "treatment": treatment.source,
            "control": control.source if control is not None else None,
            "background": {
                "a0": bg.a0, "a1": bg.a1, "used_control": bg.used_control,
                "genome_mean": bg.genome_mean,
            },
            "segment_bin_size": int(factor * bs),
            "segment_dispersion": {"a0": float(seg_a0), "a1": float(seg_a1)},
            "signal_scale_bp": int(scale_bp),
            "is_count_data": treatment.is_count,
            "approximate_pvalues": not treatment.is_count,
        },
        chrom_sizes=dict(treatment.chrom_sizes),
    )

    _lap("focal" if focal_ran else "assemble")
    diagnostics["n_candidates"] = len(peaks)
    if focal_ran:
        # Focal-pass sub-peaks are HMM-segmented, NB-tested regions in their
        # own right; the robust-z gate exists for scale-space noise bumps.
        peaks = _apply_fdr(peaks, params)
    else:
        # Sub-peaks are searched only in the regions that survive the
        # region-level FDR.  They used to be searched in every HMM candidate
        # first: on a sparse ATAC library that is ~350,000 candidates for
        # ~25,000 calls, and the search was most of the run.
        def _subpeaks(kept: NestedPeakSet) -> None:
            if params.call_subpeaks and not skip_sub:
                _fill_subpeaks(ctx, kept.regions, n_jobs)
                _gate_domain_subpeaks(kept, params)

        peaks = _apply_fdr(peaks, params, subpeaks=_subpeaks)
    if params.max_width:
        peaks, n_capped = _cap_width(peaks, int(params.max_width))
        diagnostics["regions_over_max_width"] = n_capped
        peaks.params["regions_over_max_width"] = n_capped
    _lap("fdr_subpeaks")
    if blacklist:
        before = len(peaks)
        peaks = peaks.exclude(blacklist)
        diagnostics["blacklisted_regions_dropped"] = before - len(peaks)
        peaks.params["blacklist_regions_dropped"] = before - len(peaks)
    peaks.name_all(name)
    _lap("blacklist")

    if return_details:
        return CallResult(peaks=peaks, background=bg, params=params, diagnostics=diagnostics)
    return peaks


@dataclass
class _ChromContext:
    """Everything one chromosome's call needs; shared with forked workers."""

    treatment: Coverage
    bg: BackgroundModel
    params: CallParams
    factor: int
    scale_bp: int
    seg_a0: float
    seg_a1: float
    weak_bp: int
    genome_bp: int
    skip_subpeaks: bool
    cache: Dict
    keep_fit: bool


def _call_chromosome(ctx: _ChromContext, chrom: str):
    """Segment one chromosome and build its regions.

    Returns ``(regions, diagnostics, n_weak_candidates, cache_key, fit)``;
    ``fit`` is the HMM result when it was fitted here and the caller keeps a
    segmentation cache, else None.  Its posterior is at SEGMENTATION
    resolution (``_map_chromosomes`` expands it), which is all the fit has.
    """
    treatment, bg, params, factor = ctx.treatment, ctx.bg, ctx.params, ctx.factor
    bs = treatment.bin_size
    min_bins = max(1, params.min_width // bs)
    counts32, mu32 = treatment.bins[chrom], bg.mu[chrom]
    if counts32.size != mu32.size:  # pragma: no cover - defensive
        m = min(counts32.size, mu32.size)
        counts32, mu32 = counts32[:m], mu32[:m]
    n = counts32.size

    # Segment on coarsened bins, then map the posterior back to reporting
    # resolution.  Only the segmentation is coarsened: region statistics,
    # the region-level test and sub-peak detection all still run on the
    # fine bins, so nothing but the boundary quantisation is given up.
    mu = mu32.astype(np.float64)
    if factor > 1:
        c_seg = _coarsen(counts32, factor)
        mu_seg = _coarsen(mu32, factor)
    else:
        c_seg = counts32.astype(np.float64)
        mu_seg = mu
    alpha = bg.alpha(mu)
    # The dispersion of a SUM of bins is not the per-bin trend evaluated at
    # the summed mean -- that is the same mistake aggregate_nb_params exists
    # to prevent, and it is far from harmless here.  Var(sum) = sum(mu_i) +
    # sum(alpha_i mu_i^2), inflated for autocorrelation; for homogeneous
    # bins that is alpha/k, so reading the trend at k*mu instead leaves the
    # dispersion a factor of k too large and the states become
    # indistinguishable exactly when the bins get coarse enough to be
    # useful.  Measured on H3K9me3, it made called territory *fall* from
    # 5.7% to 0.8% as the segmentation coarsened, when it should have risen.
    alpha_seg = (np.maximum(ctx.seg_a0 + ctx.seg_a1 / np.maximum(mu_seg, 1e-9), 1e-8)
                 if factor > 1 else alpha)

    if params.background_model == "poisson":
        alpha = np.full_like(mu, 1e-9)  # NB -> Poisson, for the ablation
        alpha_seg = np.full_like(mu_seg, 1e-9)

    hmm = NBHMM(
        stay_background=params.hmm_stay_background,
        stay_enriched=params.hmm_stay_enriched,
        min_stay_enriched=_dwell_floor(
            ctx.scale_bp if params.dwell_from_autocorrelation else 0,
            factor * bs, params),
        fit_state_dispersion=params.hmm_fit_state_dispersion,
        decode_path=False,
        fit_fraction=params.hmm_fit_fraction,
    )
    key = (chrom, factor, hmm.stay_background, hmm.stay_enriched,
           hmm.min_stay_enriched, hmm.fit_state_dispersion,
           params.background_model, hmm.fit_fraction)
    res = ctx.cache.get(key)
    fitted = res is None
    if fitted:
        _, res = segment_chromosome(
            c_seg, mu_seg, alpha_seg,
            posterior_cutoff=params.posterior_cutoff,
            min_bins=1,
            hmm=hmm,
        )
        post_seg = res.posterior
    else:
        # A cached fit holds the posterior at reporting resolution, where it
        # is the segmentation posterior repeated.
        post_seg = res.posterior[::factor] if factor > 1 else res.posterior

    # Runs of enriched segmentation bins, as reporting-bin intervals.  The
    # last segmentation bin may run past the chromosome end; clip it.
    seg_runs = _runs(post_seg >= params.posterior_cutoff, 1)
    intervals = [(a * factor, min(b * factor, n)) for a, b in seg_runs]
    intervals = [(a, b) for a, b in intervals if b - a >= min_bins]

    chrom_len = treatment.chrom_sizes.get(chrom, n * bs)
    n_weak = 0
    need_fine = bool(params.max_gap) or bool(ctx.weak_bp) or factor <= 1
    if need_fine:
        counts = counts32.astype(np.float64)
        posterior = (np.repeat(post_seg, factor)[:n] if factor > 1 else post_seg)
        intervals = _merge_similar(
            intervals, params.max_gap // bs if params.max_gap else 0, counts, mu,
            params.gap_fold_fraction, params.min_fold)
        if ctx.weak_bp:
            weak = _weak_domains(counts, mu, bg, bs, ctx.weak_bp, params, ctx.genome_bp)
            n_weak = len(weak)
            if weak:
                intervals = _union_intervals(intervals, weak)
        regions = _build_regions(
            chrom, intervals, counts, mu, alpha, posterior, bs, chrom_len,
            params, bg.autocorr_inflation, skip_subpeaks=True,
        )
    else:
        # Every interval starts and ends on a segmentation-bin edge, so the
        # region sums are sums of segmentation-bin sums: the prefix arrays
        # are built at that resolution (10-100x shorter) and index the same
        # totals the reporting-bin prefix sums would.
        weights = np.full(post_seg.size, float(factor))
        weights[-1] = float(n - (post_seg.size - 1) * factor)
        prefix = (
            np.concatenate([[0.0], np.cumsum(c_seg)]),
            np.concatenate([[0.0], np.cumsum(mu_seg)]),
            np.concatenate([[0.0], np.cumsum(_coarsen(alpha * mu * mu, factor))]),
            np.concatenate([[0.0], np.cumsum(post_seg * weights)]),
            factor,
        )
        regions = _build_regions(
            chrom, intervals, None, None, None, None, bs, chrom_len,
            params, bg.autocorr_inflation, skip_subpeaks=True, prefix=prefix,
        )
    diag = {
        "n_intervals": len(intervals),
        "multipliers": res.multipliers.tolist(),
        "hmm_iter": res.n_iter,
        "hmm_converged": res.converged,
        "min_stay_enriched": hmm.min_stay_enriched,
        "dwell_floor_bp": (
            int(round(factor * bs / (1.0 - hmm.min_stay_enriched)))
            if 0.0 < hmm.min_stay_enriched < 1.0 else 0
        ),
        "dispersion_scale": (
            np.round(res.dispersion_scale, 2).tolist()
            if res.dispersion_scale is not None else None
        ),
        "fitted_dwell_bp": [
            int(round(factor * bs / max(1.0 - d, 1e-9)))
            for d in np.diag(res.transitions)
        ],
    }
    return regions, diag, n_weak, key, (res if fitted and ctx.keep_fit else None)


#: Set in the parent right before the worker pool forks, so the workers see
#: the coverage and the background through copy-on-write memory instead of
#: having ~10 GB of arrays pickled to them.
_FORK_CONTEXT: Optional[_ChromContext] = None


def _chrom_task(chrom: str):
    return _call_chromosome(_FORK_CONTEXT, chrom)


#: Regions (as bin intervals) per chromosome for the sub-peak pass; set with
#: ``_FORK_CONTEXT`` before the pool forks.
_FORK_INTERVALS: Optional[Dict[str, List[Tuple[int, int]]]] = None


def _subpeak_chromosome(ctx: _ChromContext, chrom: str,
                        intervals: Sequence[Tuple[int, int]]) -> List[List[SubPeak]]:
    treatment, bg, params = ctx.treatment, ctx.bg, ctx.params
    bs = treatment.bin_size
    counts = treatment.bins[chrom].astype(np.float64)
    mu = bg.mu[chrom].astype(np.float64)
    if counts.size != mu.size:  # pragma: no cover - defensive
        m = min(counts.size, mu.size)
        counts, mu = counts[:m], mu[:m]
    alpha = (np.full_like(mu, 1e-9) if params.background_model == "poisson"
             else bg.alpha(mu))
    chrom_len = treatment.chrom_sizes.get(chrom, counts.size * bs)
    return [_call_subpeaks(chrom, b0, b1, counts, mu, alpha, bs, chrom_len,
                           params, bg.autocorr_inflation)
            for b0, b1 in intervals]


def _subpeak_task(chrom: str):
    return _subpeak_chromosome(_FORK_CONTEXT, chrom, _FORK_INTERVALS[chrom])


def _fill_subpeaks(ctx: _ChromContext, regions: Sequence[Region],
                   n_jobs: Optional[int]) -> None:
    """Search sub-peaks in ``regions`` (in place), one chromosome per worker."""
    global _FORK_CONTEXT, _FORK_INTERVALS
    bs = ctx.treatment.bin_size
    by_chrom: Dict[str, List[Region]] = {}
    for r in regions:
        by_chrom.setdefault(r.chrom, []).append(r)
    if not by_chrom:
        return
    n_bins = {c: ctx.treatment.bins[c].size for c in by_chrom}
    intervals = {c: [(r.start // bs, min(-(-r.end // bs), n_bins[c])) for r in rs]
                 for c, rs in by_chrom.items()}
    order = sorted(by_chrom, key=lambda c: -len(by_chrom[c]))
    jobs = resolve_call_jobs(n_jobs, len(order))
    if jobs > 1 and len(regions) >= 2000:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        _FORK_CONTEXT, _FORK_INTERVALS = ctx, intervals
        try:
            with ProcessPoolExecutor(max_workers=jobs,
                                     mp_context=mp.get_context("fork")) as pool:
                found = dict(zip(order, pool.map(_subpeak_task, order)))
        finally:
            _FORK_CONTEXT = _FORK_INTERVALS = None
    else:
        found = {c: _subpeak_chromosome(ctx, c, intervals[c]) for c in order}
    for c, rs in by_chrom.items():
        for r, subs in zip(rs, found[c]):
            r.subpeaks = subs


def resolve_call_jobs(n_jobs: Optional[int], n_chroms: int) -> int:
    """Worker processes for the per-chromosome call.

    ``None`` or 1: serial.  ``<= 0``: automatic -- the CPUs this process may
    use (the scheduler's allocation, not the machine's core count), capped at
    the number of chromosomes.  Needs the ``fork`` start method (Linux); on
    other platforms the call stays serial.
    """
    import multiprocessing as mp

    if n_jobs is None or n_jobs == 1 or n_chroms <= 1:
        return 1
    if "fork" not in mp.get_all_start_methods():
        return 1
    if n_jobs <= 0:
        try:
            n_jobs = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - not Linux
            n_jobs = os.cpu_count() or 1
    return max(1, min(int(n_jobs), n_chroms))


def _map_chromosomes(ctx: _ChromContext, order: Sequence[str], n_jobs: Optional[int]):
    """``_call_chromosome`` for every chromosome, results in ``order``.

    Chromosomes are independent once the background, the dispersion and the
    signal scale are fixed (all three are genome-wide and fitted before this
    point), so they are called in worker processes.  The call was
    single-threaded before: measured on a 7.5M-fragment ATAC library at
    10 bp bins it used 58% of one core with eight allocated.
    """
    global _FORK_CONTEXT
    jobs = resolve_call_jobs(n_jobs, len(order))
    sizes = {c: ctx.treatment.bins[c].size for c in order}
    if jobs <= 1:
        done = {c: _call_chromosome(ctx, c) for c in order}
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from .segment._kernels import warm_up

        warm_up()  # compiled once here, inherited by every worker
        # Largest first, so the pool does not finish on the longest task.
        todo = sorted(order, key=lambda c: -sizes[c])
        _FORK_CONTEXT = ctx
        try:
            with ProcessPoolExecutor(max_workers=jobs,
                                     mp_context=mp.get_context("fork")) as pool:
                done = dict(zip(todo, pool.map(_chrom_task, todo)))
        finally:
            _FORK_CONTEXT = None
    out = []
    for c in order:
        regs, diag, n_weak, key, fit = done[c]
        if fit is not None and ctx.factor > 1:
            fit.posterior = np.repeat(fit.posterior, ctx.factor)[: sizes[c]]
        out.append((regs, diag, n_weak, key, fit))
    return out


#: CallParams fields the background model depends on.  A fitted background can
#: be reused across settings that differ only in fields outside this set.
BACKGROUND_FIELDS = frozenset({
    "bin_size", "windows", "background_exclude_fold",
    "background_exclude_iters", "background_exclude_smooth_bp",
    "background_robust_start", "background_depth_aware",
})


DOMAIN_WINDOW_BP = 500_000
"""Background windows at least this wide mark the domain-scale setting (the
scan and presets use 1 Mb / 5 Mb for domain marks), where the anchored
background and the weak-domain pass are on by default.  Not at 100-200 kb,
the mixed-mark windows: measured on GSE285245 H3K27ac (165 / 823 kb) the
anchored background and its background-only dispersion cost replicate
support (0.92 -> 0.89) and recall (0.52 -> 0.49)."""


def robust_start_enabled(params: "CallParams") -> bool:
    if params.background_robust_start is not None:
        return bool(params.background_robust_start)
    return bool(params.windows) and min(params.windows) >= DOMAIN_WINDOW_BP


def weak_domain_window_bp(params: "CallParams") -> int:
    if params.weak_domain_bp is not None:
        return max(0, int(params.weak_domain_bp))
    if params.windows and min(params.windows) >= DOMAIN_WINDOW_BP:
        return int(max(10_000, 2 * params.domain_min_width))
    return 0


def fit_background(treatment: Coverage, control: Optional[Coverage],
                   params: CallParams, covariates=None,
                   n_jobs: Optional[int] = None) -> BackgroundModel:
    """The background ``call_peaks`` would fit for these inputs and parameters.

    Coverage must already be at ``params.bin_size``.
    """
    return BackgroundModel.fit(
        treatment, control=control, windows=params.windows, covariates=covariates,
        exclude_fold=params.background_exclude_fold,
        exclude_iters=params.background_exclude_iters,
        exclude_smooth_bp=params.background_exclude_smooth_bp,
        robust_start=robust_start_enabled(params),
        depth_aware=params.background_depth_aware,
        n_jobs=n_jobs,
    )


def _aggregate_alpha(mu: np.ndarray, alpha: np.ndarray, factor: int,
                     inflation: float = 1.0) -> np.ndarray:
    """Dispersion of block sums of ``factor`` bins, from the per-bin trend.

    ``inflation`` is the background's asymptotic lag-1 factor; the factor
    actually applied is the exact one for a sum of ``factor`` bins
    (``ar1_factor``), which is far smaller for short blocks."""
    mu_seg = _coarsen(mu, factor)
    if factor <= 1:
        return np.maximum(alpha, 1e-10)
    f = float(ar1_factor(factor, rho_from_inflation(inflation)))
    var = mu_seg + f * _coarsen(alpha * mu * mu, factor)
    return np.maximum((var - mu_seg) / np.maximum(mu_seg * mu_seg, 1e-12), 1e-10)


def _coarsen(a: np.ndarray, factor: int) -> np.ndarray:
    """Sum adjacent bins in blocks of ``factor``, zero-padding the tail."""
    if factor <= 1:
        return a
    pad = (-a.size) % factor
    if pad:
        a = np.concatenate([a, np.zeros(pad, dtype=a.dtype)])
    # Summed in float64 whatever the input, so float32 coverage does not
    # need a float64 copy of the whole chromosome first.
    return a.reshape(-1, factor).sum(axis=1, dtype=np.float64)


def signal_scale_bp(treatment: Coverage, min_bins: int = 1000,
                    n_jobs: Optional[int] = None) -> int:
    """The mark's own scale in bp: a robust autocorrelation length.

    Estimated per chromosome and combined at the 75th percentile rather than the
    median.  That is not arbitrary: the estimator is biased *downward* by
    sparsity -- empty bins decorrelate a signal that is really continuous -- so
    the low readings are coverage artefacts, not shorter biology.  Measured on
    the H3K9me3 sample the per-chromosome estimates were 200 bp on five
    chromosomes and 1,600-2,300 bp on two, a 12x spread; H3K27me3 and H3K27ac
    were stable to within 2x.  Taking a high quantile is what stops one sparse
    chromosome from silently disabling every scale-derived setting on it.
    """
    arrays = [a for a in treatment.bins.values() if a.size >= min_bins]
    jobs = resolve_call_jobs(n_jobs, len(arrays))
    if jobs > 1:
        # Threads, not processes: the work is one FFT per chromosome, which
        # runs outside the interpreter lock.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            lens = list(pool.map(
                lambda a: autocorrelation_length_bp(a, treatment.bin_size), arrays))
    else:
        lens = [autocorrelation_length_bp(a, treatment.bin_size) for a in arrays]
    if not lens:
        return treatment.bin_size
    return int(max(np.percentile(lens, 75), treatment.bin_size))


def _segment_bin_size(treatment: Coverage, params: CallParams,
                      scale_bp: int) -> int:
    """Bin size for the HMM, from the mark's scale.

    A fraction of the autocorrelation length: fine enough to place a boundary
    well inside a feature, coarse enough that a bin carries a decision.  Chosen
    this way rather than from how empty the coverage is, which was tried and is
    exactly backwards -- H3K27ac is the *emptiest* of the three test samples at
    its reporting bin size (96.7% of 25 bp bins) and is the one that must not be
    coarsened, because its features are only a few hundred bp wide.

    Swept on the three CUT&RUN samples, ``scale / 8`` lands on the best or
    near-best setting for each by signal-per-called-base:

    ===========  ==========  =========  ==========  ==============
    sample       scale       seg bin    median      FRiP/territory
    ===========  ==========  =========  ==========  ==============
    H3K27ac        ~675 bp      75 bp      ~450 bp        ~50x
    H3K27me3      ~8400 bp    1000 bp      ~18 kb          6.7x
    H3K9me3        ~900 bp     100 bp       400 bp         11x
    ===========  ==========  =========  ==========  ==============

    H3K9me3 comes out uncoarsened, and that is the right answer for it: its
    signal is punctate, and forcing it into domains cost more than half the
    signal-in-peaks (FRiP 0.45 -> 0.21 at 800 bp) for fewer, wider calls.
    """
    if params.segment_bin_size:
        return max(int(params.segment_bin_size), treatment.bin_size)

    bs = treatment.bin_size
    cap = max(bs, int(params.domain_min_width) // 4)
    seg = (int(scale_bp / max(params.segment_bins_per_scale, 1)) // bs) * bs
    floor = max(bs, (int(params.min_segment_bin_bp) // bs) * bs)
    return int(np.clip(seg, floor, cap))


def _dwell_floor(scale_bp: float, bin_size: int, params: CallParams) -> float:
    """Minimum enriched dwell time for the HMM, as a self-transition probability.

    The scale is the larger of the narrowest region we are willing to emit
    (``min_width`` -- below it the model would be optimising for runs that are
    then discarded) and the signal's own scale, which is what tells a punctate
    mark from a domain mark without being told.  Capped by ``max_dwell`` so a
    degenerate estimate cannot swallow a chromosome.
    """
    scale = max(float(params.min_width), float(scale_bp))
    scale = min(scale, float(params.max_dwell))
    if scale <= bin_size:
        return 0.0
    return float(np.clip(1.0 - bin_size / scale, 0.0, 0.9999))


def _focal_pass(treatment: Coverage, control, params: CallParams, bg_wide: BackgroundModel,
                regions: List[Region], covariates, cache: Dict):
    """Multi-scale sliding-window test against a local background per scale.

    For every window width in ``focal_scales`` (finest first): sliding sums
    stepped at a fine segmentation bin, tested (NB tail) against the
    MACS-style local level with the window itself left out of its flanks
    (``focal_background="flank"``), with the window dispersion fitted on the
    sums separately inside and outside first-pass regions.  One BH over all
    positions and scales.  Significant windows form intervals per scale;
    ``_assemble_scales`` nests them fine to coarse.  Each resulting focal
    region contributes its sub-peaks (or itself) to the first-pass region it
    overlaps, or is reported as a standalone region with its sub-peaks.
    """
    bs = treatment.bin_size
    # Scale of the focal features: autocorrelation of the high-passed signal.
    if "focal_scale_bp" not in cache:
        k_hp = max(1, int(round(20_000 / bs)))
        lens = []
        for arr in treatment.bins.values():
            if arr.size < 1000:
                continue
            x = arr.astype(np.float64)
            hp = x - uniform_filter1d(x, size=k_hp, mode="nearest") if k_hp < x.size else x
            lens.append(autocorrelation_length_bp(hp, bs, max_lag_bp=20_000))
        cache["focal_scale_bp"] = int(max(np.percentile(lens, 75), bs)) if lens else bs
    scale_bp = cache["focal_scale_bp"]
    seg = int(np.clip((scale_bp // 8 // bs) * bs, bs, params.focal_max_segment_bp))
    seg = max(seg, (int(params.min_segment_bin_bp) // bs) * bs, bs)
    raw_scales = (tuple(int(w) for w in params.focal_scales) if params.focal_scales else
                  ((int(params.focal_test_window_bp) if params.focal_test_window_bp
                    else int(np.clip(scale_bp, 200, 1000))),))
    # The segmentation bin must resolve the finest window: at most half of it.
    seg = int(min(seg, max(bs, (min(raw_scales) // 2 // bs) * bs)))
    factor = max(1, seg // bs)
    seg = factor * bs
    scales: List[int] = []
    for w in sorted(raw_scales):
        w = int(max(seg, round(w / seg) * seg))
        if w not in scales:
            scales.append(w)
    min_bins = max(1, params.min_width // bs)
    mode = params.focal_background
    if mode not in ("flank", "plain"):
        raise ValueError(f"focal_background must be 'flank' or 'plain', got {mode!r}")
    diag = {"scales_bp": scales, "scale_bp": int(scale_bp), "segment_bin_size": int(seg),
            "background": mode,
            "test": "NB tail of sliding sums at each scale, pooled BH over scales and positions"}

    plain_mu = None
    if mode == "plain":
        # The previous focal background: plain local means (peak included)
        # maxed with the wide background.  Kept as an ablation.
        windows = tuple(sorted(set(params.focal_windows)))
        if control is None:
            windows = tuple(w for w in windows if w >= 10_000) or (max(windows),)
        bg_local = BackgroundModel.fit(treatment, control=control, windows=windows,
                                       covariates=covariates, exclude_fold=0.0)
        plain_mu = {c: np.maximum(bg_local.mu[c][: bg_wide.mu[c].size],
                                  bg_wide.mu[c][: bg_local.mu[c].size]) for c in bg_wide.mu}
        diag["windows"] = list(windows)

    chroms = sorted(treatment.bins)
    # First-pass territory, at segmentation resolution: the dispersion of a
    # window inside a domain is the domain's texture, outside it the
    # background's, and one fit over both mixes them (measured: window alpha
    # 0.25 on H3K27me3, a sub-peak then needed ~6x over its domain).
    by_chrom_regions: Dict[str, List[Region]] = {}
    for r in regions:
        by_chrom_regions.setdefault(r.chrom, []).append(r)

    per = {}  # (chrom, scale) -> dict(c_w, mu_w, inside, mean_bin)
    for chrom in chroms:
        counts = treatment.bins[chrom].astype(np.float64)
        m = min(counts.size, bg_wide.mu[chrom].size)
        counts = counts[:m]
        c_seg = _coarsen(counts, factor)
        n = c_seg.size
        inside = np.zeros(n, dtype=bool)
        for r in by_chrom_regions.get(chrom, []):
            inside[r.start // seg: -(-r.end // seg)] = True
        cum = np.concatenate([[0.0], np.cumsum(c_seg)])
        idx = np.arange(n)
        floor = _robust_floor(c_seg, max(1, int(round(min(scales) / seg))),
                              max(1, int(round(max(params.focal_windows or (10_000,)) / seg))))
        wide_seg = (_coarsen(bg_wide.mu[chrom][:m].astype(np.float64), factor)
                    if (control is not None and mode == "flank") else None)
        plain_seg = (_coarsen(plain_mu[chrom][:m].astype(np.float64), factor)
                     if plain_mu is not None else None)
        for w in scales:
            k = max(1, w // seg)
            lo0 = idx - k // 2
            lo, hi = np.clip(lo0, 0, n), np.clip(lo0 + k, 0, n)
            c_w = cum[hi] - cum[lo]
            n_w = np.maximum(hi - lo, 1)
            if mode == "plain":
                mean_bin = (np.concatenate([[0.0], np.cumsum(plain_seg)]))
                mu_w = mean_bin[hi] - mean_bin[lo]
                mean_bin = plain_seg
            else:
                g = max(1, k // 2)
                ex_lo, ex_hi = np.clip(lo0 - g, 0, n), np.clip(lo0 + k + g, 0, n)
                ex_s, ex_n = cum[ex_hi] - cum[ex_lo], ex_hi - ex_lo
                mean_bin = np.full(n, floor * float(params.focal_floor_fraction))
                for fct in params.focal_flank_factors:
                    L = max(int(round(fct * k)), int(round(params.focal_min_flank_bp / seg)), 1)
                    o_lo, o_hi = np.clip(lo0 - g - L, 0, n), np.clip(lo0 + k + g + L, 0, n)
                    fl_n = (o_hi - o_lo) - ex_n
                    fl = np.where(fl_n > 0, (cum[o_hi] - cum[o_lo] - ex_s) / np.maximum(fl_n, 1), 0.0)
                    np.maximum(mean_bin, fl, out=mean_bin)
                if wide_seg is not None:
                    # With a control the wide model is the control's own
                    # level, which the flanks of the treatment cannot see.
                    np.maximum(mean_bin, wide_seg, out=mean_bin)
                mu_w = mean_bin * n_w
            per[(chrom, w)] = dict(c_w=c_w, mu_w=np.maximum(mu_w, 1e-9), inside=inside,
                                   mean_bin=mean_bin)

    # Dispersion of the window sums, per scale, measured on the sums of
    # windows OUTSIDE first-pass regions.  Inside them the windows are the
    # signal: a fit there measured the peaks, not the noise (H3K4me3: alpha
    # ~2 + 93/mu inside against 0.03 + 5/mu outside, and no promoter summit
    # in a first-pass region was ever significant).
    from .background.nb import estimate_dispersion

    disp = {}
    nlps = {}
    for w in scales:
        cs = np.concatenate([per[(c, w)]["c_w"][~per[(c, w)]["inside"]] for c in chroms])
        ms = np.concatenate([per[(c, w)]["mu_w"][~per[(c, w)]["inside"]] for c in chroms])
        if cs.size < 2_000:
            cs = np.concatenate([per[(c, w)]["c_w"] for c in chroms])
            ms = np.concatenate([per[(c, w)]["mu_w"] for c in chroms])
        if cs.size > 4_000_000:
            step = int(np.ceil(cs.size / 4_000_000))
            cs, ms = cs[::step], ms[::step]
        fit = (_clipped_dispersion(cs, ms) if params.focal_dispersion == "clipped"
               else estimate_dispersion(cs, ms))
        if params.background_model == "poisson":
            fit = (1e-9, 0.0)
        disp[w] = [float(fit[0]), float(fit[1])]
        for c in chroms:
            d = per[(c, w)]
            nlps[(c, w)] = nb_neglog10_sf(d["c_w"], d["mu_w"],
                                          np.maximum(fit[0] + fit[1] / d["mu_w"], 1e-10))
    diag["window_dispersion"] = {str(w): v for w, v in disp.items()}

    # FDR over the tests actually made.  Windows stepped at the segmentation
    # bin overlap: a 3.6 kb window stepped at 100 bp is the same test 36
    # times, and BH over every step counted it 36 times.  So the BH cut is
    # set on non-overlapping windows (genome / window tests per scale, all
    # scales pooled) and then applied to every position.
    keys = [(c, w) for w in scales for c in chroms]
    cut = -np.log10(params.qvalue) if params.qvalue > 0 else 0.0
    families = ([[k for k in keys if k[1] == w] for w in scales]
                if params.focal_fdr == "per_scale" else [keys])
    p_cuts = {}
    for fam in families:
        sampled = np.concatenate([nlps[(c, w)][::max(1, w // seg)] for (c, w) in fam])
        q_s = qvalue_from_neglog10p(sampled)
        passing = sampled[q_s >= cut]
        pc_ = float(passing.min()) if passing.size else float("inf")
        for k in fam:
            p_cuts[k] = pc_
        del sampled, q_s, passing
    diag["p_cut_neglog10"] = {str(w): p_cuts[(chroms[0], w)] for w in scales}
    sig = {}
    dbg_loci = cache.get("focal_debug") or []
    dbg_out = []
    for k in keys:
        d = per[k]
        nl = nlps[k]
        # Significant and over the fold floor of the local level.
        p_cut = p_cuts[k]
        sig[k] = (nl >= p_cut) & (d["c_w"] >= params.focal_min_fold * d["mu_w"])
        for (c, s0, e0) in dbg_loci:
            if c == k[0]:
                j0, j1 = s0 // seg, min(nl.size, -(-e0 // seg))
                dbg_out.append({"chrom": c, "scale": k[1], "start_bin": j0, "seg": seg,
                                "p_cut": p_cut,
                                "c_w": d["c_w"][j0:j1].copy(), "mu_w": d["mu_w"][j0:j1].copy(),
                                "inside": d["inside"][j0:j1].copy(),
                                "nlp": nl[j0:j1].copy(), "sig": sig[k][j0:j1].copy()})
    if dbg_loci:
        cache["focal_debug_out"] = dbg_out

    focal: List[Region] = []
    n_per_scale = {w: 0 for w in scales}
    n_parents = 0
    for chrom in chroms:
        counts = treatment.bins[chrom].astype(np.float64)
        m = min(counts.size, bg_wide.mu[chrom].size)
        counts = counts[:m]
        chrom_len = treatment.chrom_sizes.get(chrom, counts.size * bs)
        pc = np.concatenate([[0.0], np.cumsum(counts)])
        layers = []
        for w in scales:
            k = max(1, w // seg)
            n_seg = sig[(chrom, w)].size
            mu_fine = np.repeat(per[(chrom, w)]["mean_bin"] / factor, factor)[:m]
            ivs = _runs(sig[(chrom, w)], 1)
            # A run of significant window CENTRES means the windows around
            # them are enriched: report the footprint (centres +- half a
            # window), trimmed at each end to the bins above the local level.
            # The centres alone made a 200 bp hit one 100 bp bin, which the
            # min_width filter then dropped.
            ivs = [(max(0, a - k // 2), min(n_seg, b - k // 2 + k)) for a, b in ivs]
            ivs = _merge_close(ivs, params.focal_gap_bins)
            ivs = [_trim_to_level(a * factor, min(b * factor, m), counts, mu_fine)
                   for a, b in ivs]
            ivs = [(a, b) for a, b in ivs if b - a >= min_bins]
            n_per_scale[w] += len(ivs)
            layers.append((w, ivs, mu_fine, disp[w]))
        nodes, n_par = _assemble_scales(layers, pc, params.focal_min_fold)
        n_parents += n_par
        for (a, b, li), leaves in nodes:
            w, _, mu_fine, fit = layers[li]
            reg = _focal_region(chrom, a, b, counts, mu_fine, fit, bs, chrom_len, params)
            if reg is None:
                continue
            for (la, lb, lj) in leaves:
                sp = _focal_region(chrom, la, lb, counts, layers[lj][2], layers[lj][3], bs,
                                   chrom_len, params)
                if sp is not None:
                    reg.subpeaks.append(_as_subpeak(sp, treatment.bins[chrom], bs))
            focal.append(reg)
    diag["n_per_scale"] = {str(w): int(v) for w, v in n_per_scale.items()}
    diag["n_scale_parents"] = int(n_parents)
    diag["n_focal"] = len(focal)

    # Attach to first-pass regions, or keep as standalone peaks.  A focal
    # region with its own finer sub-peaks contributes those; one without
    # contributes itself.
    by_chrom: Dict[str, List[Region]] = {}
    for r in regions:
        by_chrom.setdefault(r.chrom, []).append(r)
    for lst in by_chrom.values():
        lst.sort(key=lambda r: r.start)
    starts_by = {c: np.array([r.start for r in lst]) for c, lst in by_chrom.items()}
    out = list(regions)
    n_attached = 0
    attached: Dict[int, List[Region]] = {}
    for f in focal:
        parents = by_chrom.get(f.chrom, [])
        parent = None
        if parents:
            i = int(np.searchsorted(starts_by[f.chrom], f.start, side="right")) - 1
            for j in (i, i + 1):
                if 0 <= j < len(parents) and parents[j].end > f.start and parents[j].start < f.end:
                    parent = parents[j]
                    break
        if parent is None:
            out.append(f)
            continue
        pieces = f.subpeaks or [_as_subpeak(f, treatment.bins[f.chrom], bs)]
        added = []
        for sp in pieces:
            s0, e0 = max(sp.start, parent.start), min(sp.end, parent.end)
            if e0 <= s0:
                continue
            summit = int(min(max(sp.summit, s0), e0 - 1))
            parent.subpeaks.append(SubPeak(
                chrom=sp.chrom, start=int(s0), end=int(e0), summit=summit, score=sp.score,
                neg_log10_p=sp.neg_log10_p, fold_enrichment=sp.fold_enrichment,
                persistence=0))
            added.append((s0, e0))
        if not added:
            out.append(f)
            continue
        attached.setdefault(id(parent), []).append((f, added))
        n_attached += 1

    # A region that holds focal peaks is a claim about the level BETWEEN
    # them: the domain must stand on its residual, with the focal bins taken
    # out.  Otherwise a run of narrow peaks over a slightly drifted
    # background is reported as one "domain" whose fold is entirely the
    # peaks' (measured on the simulated narrow set with the h3k27me3 preset:
    # 79 kb regions at 1.5x average fold, 0.17 precision).  A region whose
    # residual fails the same fold and p-value cut as any region is
    # dissolved into its focal peaks, which are reported standalone.  The
    # residual is judged against the WIDE background: the domain's claim is
    # enrichment over the genome, and against the local focal level (which
    # inside a domain is the domain itself) every domain failed (measured:
    # 115 of 174 H3K9me3 domains holding focal peaks dissolved on chr1+chr11).
    cut = -np.log10(params.qvalue) if params.qvalue > 0 else 0.0
    rho = rho_from_inflation(bg_wide.autocorr_inflation)
    n_dissolved = 0
    kept: List[Region] = []
    for r in out:
        fs = attached.get(id(r))
        if not fs:
            kept.append(r)
            continue
        # Slices only: a whole-chromosome float copy per region was most of
        # the run once thousands of regions held focal peaks (H3K4me3: 117 s
        # -> 484 s).
        arr_c, arr_m = treatment.bins[r.chrom], bg_wide.mu[r.chrom]
        m = min(arr_c.size, arr_m.size)
        b0, b1 = r.start // bs, min(m, -(-r.end // bs))
        c_all = arr_c[b0:b1].astype(np.float64)
        m_all = arr_m[b0:b1].astype(np.float64)
        # Only the leaf sub-peaks come out of the residual: a coarse focal
        # parent (a sub-domain) is territory of the domain, not a peak in it.
        mask = np.ones(max(b1 - b0, 0), dtype=bool)
        for _f, pcs in fs:
            for ps, pe in pcs:
                mask[max(ps // bs, b0) - b0: max(min(-(-pe // bs), b1) - b0, 0)] = False
        n_res = int(mask.sum())
        ok = n_res > 0
        if ok:
            c_r, m_r = c_all[mask], m_all[mask]
            a_r = bg_wide.alpha(m_r) if params.background_model != "poisson" else np.full_like(m_r, 1e-9)
            obs, exp = float(c_r.sum()), float(max(m_r.sum(), 1e-9))
            var = float(ar1_factor(n_res, rho)) * (exp + float((a_r * m_r * m_r).sum()))
            a_eff = max((var - exp) / (exp * exp), 1e-10)
            nlp = float(nb_neglog10_sf(np.array([obs]), np.array([exp]), np.array([a_eff]))[0])
            ok = obs / exp >= params.min_fold and nlp >= cut
            if ok and params.domain_residual_share > 0:
                # ... and the between-peak level must carry its share of the
                # region's excess.  A promoter cluster's shoulders are
                # enriched over a wide background, but the region is its
                # peaks (H3K4me3 at Havcr2: 5-17 kb "domains" whose excess
                # sat almost entirely in the summits).
                tot_obs = float(c_all.sum())
                tot_exp = float(m_all.sum())
                excess = tot_obs - tot_exp
                ok = excess <= 0 or (obs - exp) >= params.domain_residual_share * excess
        if ok:
            kept.append(r)
        else:
            kept.extend(f for f, _ in fs)
            n_dissolved += 1
    out = kept
    for r in out:
        r.subpeaks.sort(key=lambda sp: sp.start)
    diag["n_attached"] = n_attached
    diag["n_standalone"] = len(focal) - n_attached
    diag["n_dissolved"] = n_dissolved
    out.sort(key=lambda r: (r.chrom, r.start))
    return out, diag


def _clipped_dispersion(counts: np.ndarray, mu: np.ndarray, clip_neglog10p: float = 2.0,
                        n_iter: int = 6) -> Tuple[float, float]:
    """``(a0, a1)`` fitted on background windows with real peaks removed.

    The moment fit trims the top 5% of raw sums, which is not the same as
    removing the windows that are enriched for their mean.  On a mark whose
    "background" holds thousands of real narrow peaks that inflates the
    dispersion until those very peaks cannot pass: measured on pooled
    H3K9me3 chr1, 83% of the spikes at >= 4x their flanks were >= 2x in at
    least 3 of the 4 replicates (random background maxima: 0%), yet a0 came
    out at 0.11 and they scored p ~ 1e-3.  So: fit, drop the windows the
    fit itself calls enriched at p < 10**-clip_neglog10p, refit, repeat to
    convergence.  Clipping 1% of a true null tail costs a few percent of
    variance; the simulated null set is the check on that.
    """
    from .background.nb import estimate_dispersion

    c = np.asarray(counts, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    fit = estimate_dispersion(c, m)
    for _ in range(n_iter):
        nl = nb_neglog10_sf(c, m, np.maximum(fit[0] + fit[1] / m, 1e-10))
        keep = nl < clip_neglog10p
        if keep.sum() < 1000:
            break
        new = estimate_dispersion(c[keep], m[keep], background_quantile=1.0)
        done = abs(new[0] - fit[0]) <= 0.01 * max(fit[0], 1e-6) and \
            abs(new[1] - fit[1]) <= 0.01 * max(fit[1], 1e-6)
        fit = new
        if done:
            break
    return fit


def _robust_floor(x: np.ndarray, k0: int, cap: int) -> float:
    """Background level of a track, per bin: the median of the track smoothed
    over ``k0`` bins, doubling the smoothing until the median is non-zero
    (sparse tracks), capped at ``cap`` bins; never above the mean.  The same
    statistic ``local_lambda`` floors the wide background with."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 1e-9
    k = max(1, int(k0))
    sm = uniform_filter1d(x, size=k, mode="nearest") if k > 1 else x
    while float(np.median(sm)) <= 0.0 and k < cap:
        k = min(k * 2, cap)
        sm = uniform_filter1d(x, size=k, mode="nearest")
    return float(max(min(float(np.median(sm)), float(x.mean())), 1e-9))


def _assemble_scales(layers, pc: np.ndarray, min_fold: float):
    """Nest the significant intervals of each scale, fine to coarse.

    ``layers`` is ``[(window_bp, intervals, mu_fine), ...]`` finest first,
    intervals in fine bins.  An interval at a coarser scale that overlaps
    nothing finer is new territory (a feature only the wider window has
    power for).  One that does overlap finer intervals becomes their parent
    only if its *residual* -- its bins outside those finer intervals --
    is still at ``min_fold`` over its own scale's local level: then it is a
    cluster or sub-domain the fine peaks sit in.  Otherwise the fine peaks
    explain it and it is dropped, which is what keeps a 3 kb window around
    a 200 bp spike from widening the spike to 3 kb.

    Returns ``([((start, end, layer), leaves), ...], n_parents)`` where
    ``leaves`` are ``(start, end, layer)`` sub-intervals (empty when the node
    is a single-scale interval).
    """
    tops: List[list] = []  # [start, end, layer, leaves]
    n_parents = 0
    for li, (w, ivs, mu_fine, _fit) in enumerate(layers):
        if not ivs:
            continue
        pm = np.concatenate([[0.0], np.cumsum(mu_fine)])
        if not tops:
            tops = [[a, b, li, []] for a, b in ivs]
            continue
        tops.sort(key=lambda t: t[0])
        t_start = np.array([t[0] for t in tops])
        t_end = np.array([t[1] for t in tops])
        claimed = np.zeros(len(tops), dtype=bool)
        new: List[list] = []
        for a, b in ivs:
            lo = int(np.searchsorted(t_end, a, side="right"))
            hi = int(np.searchsorted(t_start, b, side="left"))
            ov = [j for j in range(lo, hi) if not claimed[j]]
            if not ov:
                if lo >= hi:
                    new.append([a, b, li, []])
                continue
            obs = pc[b] - pc[a]
            exp = pm[b] - pm[a]
            for j in ov:
                s0, e0 = max(a, tops[j][0]), min(b, tops[j][1])
                obs -= pc[e0] - pc[s0]
                exp -= pm[e0] - pm[s0]
            if exp > 0 and obs / exp >= min_fold:
                leaves = []
                for j in ov:
                    leaves.extend(tops[j][3] or [(tops[j][0], tops[j][1], tops[j][2])])
                    claimed[j] = True
                s = min(a, min(tops[j][0] for j in ov))
                e = max(b, max(tops[j][1] for j in ov))
                new.append([s, e, li, leaves])
                n_parents += 1
        tops = [t for j, t in enumerate(tops) if not claimed[j]] + new
        tops = _merge_overlapping_nodes(tops)
    tops.sort(key=lambda t: t[0])
    return [((t[0], t[1], t[2]), t[3]) for t in tops], n_parents


def _merge_overlapping_nodes(tops: List[list]) -> List[list]:
    """Two parents of one layer can share an edge with the same finer node;
    union them so the output never holds overlapping regions."""
    tops = sorted(tops, key=lambda t: t[0])
    out: List[list] = []
    for t in tops:
        if out and t[0] < out[-1][1]:
            p = out[-1]
            lp = p[3] or [(p[0], p[1], p[2])]
            lt = t[3] or [(t[0], t[1], t[2])]
            out[-1] = [p[0], max(p[1], t[1]), max(p[2], t[2]), sorted(set(lp + lt))]
        else:
            out.append(t)
    return out


def _focal_region(chrom, b0, b1, counts, mu, fit, bin_size, chrom_len,
                  params: CallParams) -> Optional[Region]:
    """Region statistics for one focal interval (fine bins) against its own
    scale's local level, with the dispersion of that scale's window sums
    (``fit`` = (a0, a1), alpha = a0 + a1 / expected sum) -- the same model
    that detected it.  Re-testing with the per-bin trend and an AR(1)
    inflation undid the window test: a 200 bp H3K9me3 spike at 3.6x passed
    as a window and failed as a region.  None when it fails the focal fold
    floor."""
    obs = float(counts[b0:b1].sum())
    m = mu[b0:b1]
    exp = float(m.sum())
    if exp <= 0 or obs / exp < params.focal_min_fold:
        return None
    start, end = min(b0 * bin_size, chrom_len), min(b1 * bin_size, chrom_len)
    if end <= start:
        return None
    a_eff = max(float(fit[0]) + float(fit[1]) / exp, 1e-10)
    nlp = float(nb_neglog10_sf(np.array([obs]), np.array([exp]), np.array([a_eff]))[0])
    width = end - start
    return Region(chrom=chrom, start=int(start), end=int(end),
                  kind="domain" if width >= params.domain_min_width else "peak",
                  score=obs, neg_log10_p=nlp, fold_enrichment=obs / exp,
                  mean_posterior=1.0, source="focal")


def _trim_to_level(a: int, b: int, counts: np.ndarray, mu: np.ndarray) -> Tuple[int, int]:
    """Shrink [a, b) from both ends past bins at or below the local level."""
    above = np.flatnonzero(counts[a:b] > mu[a:b])
    if above.size == 0:
        return a, b
    return a + int(above[0]), a + int(above[-1]) + 1


def _as_subpeak(r: Region, arr: np.ndarray, bin_size: int) -> SubPeak:
    b0 = r.start // bin_size
    b1 = max(b0 + 1, -(-r.end // bin_size))
    summit = (b0 + int(np.argmax(arr[b0:b1]))) * bin_size + bin_size // 2
    summit = int(min(max(summit, r.start), r.end - 1))
    return SubPeak(chrom=r.chrom, start=r.start, end=r.end, summit=summit, score=r.score,
                   neg_log10_p=r.neg_log10_p, fold_enrichment=r.fold_enrichment,
                   persistence=0)


def _build_regions(chrom, intervals, counts, mu, alpha, posterior, bin_size,
                   chrom_len, params: CallParams, inflation: float = 1.0,
                   skip_subpeaks: bool = False, prefix=None) -> List[Region]:
    """Every region of a chromosome in one pass: prefix sums for the
    aggregates and a single vectorised NB test.

    The per-region version (``_build_region``) costs a SciPy call per region;
    a punctate mark has tens of thousands of regions per chromosome and the
    tuning sweep builds them eight times over, which on a CUT&RUN CTCF
    replicate at 10 bp took longer than 30 minutes for one chromosome.
    Same arithmetic as ``aggregate_nb_params`` with ``inflation``.
    """
    if not intervals:
        return []
    iv = np.asarray(intervals, dtype=np.int64).reshape(-1, 2)
    b0, b1 = iv[:, 0], iv[:, 1]
    if prefix is not None:
        # Prefix sums of counts, mu, alpha*mu^2 and posterior (weighted by
        # reporting bins) over blocks of ``f`` reporting bins; the intervals
        # start and end on block edges (the last may be clipped to the
        # chromosome end, where the padding is zero).
        pc, pm, pv, pp, blk = prefix
        i0, i1 = b0 // blk, -(-b1 // blk)
    else:
        pc = np.concatenate([[0.0], np.cumsum(counts, dtype=np.float64)])
        pm = np.concatenate([[0.0], np.cumsum(mu, dtype=np.float64)])
        pv = np.concatenate([[0.0], np.cumsum(
            np.asarray(alpha, dtype=np.float64) * mu * mu, dtype=np.float64)])
        pp = np.concatenate([[0.0], np.cumsum(posterior, dtype=np.float64)])
        i0, i1 = b0, b1
    obs = pc[i1] - pc[i0]
    exp = pm[i1] - pm[i0]
    ok = exp > 0
    exp_safe = np.where(ok, exp, 1.0)
    fold = np.where(ok, obs / exp_safe, 0.0)
    ok &= fold >= params.min_fold
    start = np.minimum(b0 * bin_size, chrom_len)
    end = np.minimum(b1 * bin_size, chrom_len)
    ok &= end > start
    if not ok.any():
        return []
    f = ar1_factor(b1 - b0, rho_from_inflation(inflation))
    var = f * (exp_safe + pv[i1] - pv[i0])
    a_eff = np.maximum((var - exp_safe) / (exp_safe * exp_safe), 1e-10)
    nlp = nb_neglog10_sf(obs, exp_safe, a_eff)
    mean_post = (pp[i1] - pp[i0]) / np.maximum(b1 - b0, 1)

    out: List[Region] = []
    for i in np.flatnonzero(ok):
        width = int(end[i] - start[i])
        region = Region(
            chrom=chrom, start=int(start[i]), end=int(end[i]),
            kind="domain" if width >= params.domain_min_width else "peak",
            score=float(obs[i]), neg_log10_p=float(nlp[i]), fold_enrichment=float(fold[i]),
            mean_posterior=float(mean_post[i]),
        )
        if params.call_subpeaks and not skip_subpeaks:
            region.subpeaks = _call_subpeaks(
                chrom, int(b0[i]), int(b1[i]), counts, mu, alpha, bin_size, chrom_len,
                params, inflation,
            )
        out.append(region)
    return out


def _build_region(chrom, b0, b1, counts, mu, alpha, posterior, bin_size,
                  chrom_len, params: CallParams, inflation: float = 1.0) -> Optional[Region]:
    obs = float(counts[b0:b1].sum())
    exp = float(mu[b0:b1].sum())
    if exp <= 0:
        return None
    fold = obs / exp
    if fold < params.min_fold:
        return None

    # Region-level test on the aggregate.  The dispersion of a SUM of bins is
    # not the per-bin dispersion -- see aggregate_nb_params.
    mu_tot, a_region = aggregate_nb_params(
        mu[b0:b1], alpha[b0:b1], rho=rho_from_inflation(inflation))
    nlp = float(nb_neglog10_sf(np.array([obs]), np.array([mu_tot]), np.array([a_region]))[0])

    start = min(b0 * bin_size, chrom_len)
    end = min(b1 * bin_size, chrom_len)
    if end <= start:
        return None

    width = end - start
    kind = "domain" if width >= params.domain_min_width else "peak"

    region = Region(
        chrom=chrom, start=start, end=end, kind=kind,
        score=obs, neg_log10_p=nlp, fold_enrichment=fold,
        mean_posterior=float(np.mean(posterior[b0:b1])),
    )

    if params.call_subpeaks:
        region.subpeaks = _call_subpeaks(
            chrom, b0, b1, counts, mu, alpha, bin_size, chrom_len, params, inflation
        )
    return region


def _call_subpeaks(chrom, b0, b1, counts, mu, alpha, bin_size, chrom_len,
                   params: CallParams, inflation: float = 1.0) -> List[SubPeak]:
    # Pad the window so a summit near a region edge still has context to descend into.
    pad = max(1, int(max(params.bandwidths) / bin_size))
    lo = max(0, b0 - pad)
    hi = min(counts.size, b1 + pad)
    seg = counts[lo:hi]
    if seg.size < 3:
        return []

    # Inside a domain the reference level is the domain, not the genome.
    # Everything below works on slices: a whole-chromosome product or copy
    # per region made this O(regions x chromosome) -- 11 ms per region on a
    # 10 bp chromosome, 15 minutes for one CUT&RUN CTCF replicate.
    is_domain = (b1 - b0) * bin_size >= params.domain_min_width
    level = 1.0
    if params.subpeak_vs_domain and is_domain:
        k = max(1, int(1000 / bin_size))
        local = (uniform_filter1d(counts[b0:b1], k, mode="nearest")
                 / np.maximum(uniform_filter1d(mu[b0:b1], k, mode="nearest"), 1e-9))
        level = max(1.0, float(np.median(local)))
        inflation = 1.0
    seg_bg = mu[lo:hi] * level if level != 1.0 else mu[lo:hi]

    maxima = find_subpeaks(
        seg, bin_size, background=seg_bg,
        bandwidths=params.bandwidths,
        min_persistence=params.min_persistence,
        min_fold=params.min_fold,
        boundary_fraction=params.boundary_fraction,
        offset=lo,
        descent_level=params.subpeak_descent_level,
    )

    # Keep only summits inside the parent region, clipped to it.
    maxima = [m for m in maxima
              if b0 <= m.index < b1 and min(m.end, b1) > max(m.start, b0)]
    if not maxima:
        return []
    s0 = np.array([max(m.start, b0) for m in maxima]) - b0
    s1 = np.array([min(m.end, b1) for m in maxima]) - b0

    # Every candidate's statistics from prefix sums and one vectorised NB
    # test.  A large domain can hold thousands of candidates, and testing them
    # one call at a time was ~90% of the whole call.  Same arithmetic as
    # aggregate_nb_params, applied to all candidates at once.
    def _prefix(a):
        return np.concatenate([[0.0], np.cumsum(a, dtype=np.float64)])

    mu_r = mu[b0:b1] * level if level != 1.0 else mu[b0:b1]
    pc, pm = _prefix(counts[b0:b1]), _prefix(mu_r)
    pv = _prefix(np.asarray(alpha[b0:b1], dtype=np.float64) * mu_r * mu_r)
    obs = pc[s1] - pc[s0]
    exp = pm[s1] - pm[s0]
    ok = exp > 0
    exp_safe = np.where(ok, exp, 1.0)
    f = ar1_factor(s1 - s0, rho_from_inflation(inflation if inflation is not None else 1.0))
    var = f * (exp_safe + pv[s1] - pv[s0])
    a_eff = np.maximum((var - exp_safe) / (exp_safe * exp_safe), 1e-10)
    nlp = nb_neglog10_sf(obs, exp_safe, a_eff)

    min_w = params.min_subpeak_width
    if min_w is None:
        lvl = int(np.clip(params.subpeak_descent_level, 0, len(params.bandwidths) - 1))
        min_w = int(params.bandwidths[lvl])
    keep = ok & ((s1 - s0) * bin_size >= min_w)

    subs: List[SubPeak] = []
    for i, m in enumerate(maxima):
        if not keep[i]:
            continue
        start = min((s0[i] + b0) * bin_size, chrom_len)
        end = min((s1[i] + b0) * bin_size, chrom_len)
        if end <= start:
            continue
        summit = min(m.index * bin_size + bin_size // 2, chrom_len - 1)
        summit = int(min(max(summit, start), end - 1))
        subs.append(
            SubPeak(chrom=chrom, start=int(start), end=int(end), summit=summit,
                    score=float(obs[i]), neg_log10_p=float(nlp[i]),
                    fold_enrichment=float(obs[i] / exp[i]),
                    persistence=m.persistence)
        )
    return subs


def _gate_domain_subpeaks(peaks: NestedPeakSet, params: CallParams) -> None:
    """Drop domain sub-peaks that are not outliers against the pooled null.

    See ``CallParams.subpeak_domain_z``.  Operates in place, before FDR, on
    the folds ``_call_subpeaks`` measured against each domain's own level.
    """
    if not (params.subpeak_vs_domain and params.subpeak_domain_z > 0):
        return
    doms = [r for r in peaks.regions if r.kind == "domain" and r.subpeaks]
    folds = np.array([s.fold_enrichment for r in doms for s in r.subpeaks])
    if folds.size < params.subpeak_domain_min_candidates:
        return
    lf = np.log(np.maximum(folds, 1e-9))
    med = np.median(lf)
    mad = 1.4826 * np.median(np.abs(lf - med))
    cut = np.exp(med + params.subpeak_domain_z * max(mad, 1e-9))
    for r in doms:
        r.subpeaks = [s for s in r.subpeaks if s.fold_enrichment >= cut]


def _apply_fdr(peaks: NestedPeakSet, params: CallParams,
               subpeaks=None) -> NestedPeakSet:
    """BH across all regions, then across all sub-peaks, then threshold.

    ``subpeaks``, if given, is called with the regions that passed (a
    :class:`NestedPeakSet`) and fills in their sub-peaks; the sub-peak BH
    family is then the sub-peaks of the reported regions.  Without it the
    regions arrive with their sub-peaks and the family is every sub-peak
    passed in (the focal pass).
    """
    if len(peaks) == 0:
        return peaks

    if params.fdr_scope not in ("genome", "candidates"):
        raise ValueError(f"fdr_scope must be 'genome' or 'candidates', "
                         f"got {params.fdr_scope!r}")
    # Focal-pass peaks reported standalone were already corrected over every
    # position in the genome by the focal test; correcting them again as
    # regions, with the genome divided by a median width that they
    # themselves pull down to ~1 kb, is a second genome-wide correction
    # (measured on H3K4me3 chr11: two of the four Havcr2 promoter peaks
    # dropped).  They are corrected among themselves, as sub-peaks are.
    hmm_regions = [r for r in peaks.regions if r.source != "focal"]
    focal_regions = [r for r in peaks.regions if r.source == "focal"]
    for group, scope in ((hmm_regions, params.fdr_scope), (focal_regions, "candidates")):
        if not group:
            continue
        n_tests = None
        if scope == "genome":
            genome = sum(peaks.chrom_sizes.values())
            width = float(np.median([r.end - r.start for r in group]))
            n_tests = genome / max(width, 1.0)
        qs = qvalue_from_neglog10p(np.array([r.neg_log10_p for r in group]), n_tests=n_tests)
        for r, q in zip(group, qs):
            r.neg_log10_q = float(q)

    cutoff = -np.log10(params.qvalue) if params.qvalue > 0 else 0.0

    def _subpeak_fdr(regions) -> None:
        subs = [s for r in regions for s in r.subpeaks]
        if subs:
            sq = qvalue_from_neglog10p(np.array([s.neg_log10_p for s in subs]))
            for s, q in zip(subs, sq):
                s.neg_log10_q = float(q)

    if subpeaks is None:
        _subpeak_fdr(peaks.regions)
    kept = peaks.filter(min_qvalue_score=cutoff, min_width=params.min_width,
                        min_fold=params.min_fold)
    if subpeaks is not None:
        subpeaks(kept)
        _subpeak_fdr(kept.regions)
    # Drop sub-peaks that individually fail; the parent region can still pass.
    for r in kept.regions:
        r.subpeaks = [s for s in r.subpeaks if s.neg_log10_q >= cutoff]
    return kept


def _cap_width(peaks: NestedPeakSet, max_width: int) -> Tuple[NestedPeakSet, int]:
    """Replace regions wider than ``max_width`` by their sub-peaks.

    See ``CallParams.max_width``.  Each sub-peak becomes a standalone
    ``peak`` region carrying its own statistics (it passed the sub-peak FDR
    against the region's level, so it is a focal feature in its own right);
    a wide region with no sub-peaks is dropped.  Returns the new set and the
    number of regions the cap acted on.
    """
    out: List[Region] = []
    n = 0
    for r in peaks.regions:
        if r.width <= max_width:
            out.append(r)
            continue
        n += 1
        for sp in r.subpeaks:
            out.append(Region(
                chrom=sp.chrom, start=sp.start, end=sp.end, kind="peak",
                score=sp.score, neg_log10_p=sp.neg_log10_p, neg_log10_q=sp.neg_log10_q,
                fold_enrichment=sp.fold_enrichment, mean_posterior=r.mean_posterior,
                subpeaks=[sp], source=r.source))
    return NestedPeakSet(out, peaks.params, peaks.chrom_sizes), n


def _merge_similar(intervals: Sequence[Tuple[int, int]], max_gap_bins: int,
                   counts: np.ndarray, mu: np.ndarray, gap_fold_fraction: float,
                   min_fold: float):
    """Connect nearby similar measurements.

    Consecutive intervals closer than ``max_gap_bins`` are joined when the gap
    between them is itself enriched: its observed/expected fold must reach
    ``gap_fold_fraction`` of the weaker neighbour's fold and ``min_fold``.  A
    gap at background keeps the two apart whatever its length.
    """
    if max_gap_bins <= 0 or len(intervals) < 2:
        return list(intervals)
    if gap_fold_fraction <= 0:
        return _merge_close(intervals, max_gap_bins)
    pc = np.concatenate([[0.0], np.cumsum(counts, dtype=np.float64)])
    pm = np.concatenate([[0.0], np.cumsum(mu, dtype=np.float64)])

    def fold(a, b):
        e = pm[b] - pm[a]
        return (pc[b] - pc[a]) / e if e > 0 else 0.0

    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if 0 < s - pe <= max_gap_bins:
            f_gap = fold(pe, s)
            need = max(min_fold, gap_fold_fraction * min(fold(ps, pe), fold(s, e)))
            if f_gap >= need:
                out[-1][1] = max(pe, e)
                continue
        elif s - pe <= 0:
            out[-1][1] = max(pe, e)
            continue
        out.append([s, e])
    return [(a, b) for a, b in out]


def _weak_domains(counts: np.ndarray, mu: np.ndarray, bg: BackgroundModel, bin_size: int,
                  window_bp: int, params: CallParams, genome_bp: int) -> List[Tuple[int, int]]:
    """Candidate weak domains: runs of sliding ``window_bp`` sums significant
    against the wide background.  See ``CallParams.weak_domain_bp``.

    The reference is the wide background on purpose.  A local-contrast
    reference (the lower of the two flanks) was tried and calls every step
    in the background: at Pdcd1 the H3K9me3 track sits at 1.9 over
    93.985-94.085 Mb and at 5.5-6.6 from 94.085 to past 94.6 Mb -- not a
    domain, the neighbourhood's ordinary level (chr1 median bin 4.2) with
    Pdcd1 in a euchromatic dip -- and the min-flank test called its edge.

    Windows are stepped at a quarter window; the cut is Bonferroni over
    genome / window non-overlapping windows (the steps are the same test four
    times over, as in the focal pass).  Each run's footprint is trimmed at
    both ends to the 1 kb-smoothed bins at the fold floor, so a domain does
    not extend half a window into background.  Returns ``(start_bin,
    end_bin)`` intervals.
    """
    k = max(2, int(window_bp // bin_size))
    n = counts.size
    if n < 2 * k:
        return []
    step = max(1, k // 4)
    centres = np.arange(0, n, step)
    lo = np.clip(centres - k // 2, 0, n)
    hi = np.clip(lo + k, 0, n)
    alpha = bg.alpha(mu)
    pc = np.concatenate([[0.0], np.cumsum(counts, dtype=np.float64)])
    pm = np.concatenate([[0.0], np.cumsum(mu, dtype=np.float64)])
    pv = np.concatenate([[0.0], np.cumsum(alpha * mu * mu, dtype=np.float64)])
    obs, exp = pc[hi] - pc[lo], pm[hi] - pm[lo]
    ok = exp > 0
    exp_s = np.where(ok, exp, 1.0)
    f = ar1_factor(hi - lo, rho_from_inflation(bg.autocorr_inflation))
    var = f * (exp_s + pv[hi] - pv[lo])
    a_eff = np.maximum((var - exp_s) / (exp_s * exp_s), 1e-10)
    nlp = nb_neglog10_sf(obs, exp_s, a_eff)
    n_tests = max(1.0, genome_bp / float(window_bp))
    cut = (-np.log10(params.qvalue) if params.qvalue > 0 else 0.0) + np.log10(n_tests)
    fmin = max(float(params.weak_domain_min_fold), float(params.min_fold))
    sig = ok & (nlp >= cut) & (obs >= fmin * exp_s)
    if not sig.any():
        return []
    k_s = max(1, int(1000 // bin_size))
    sm_c = uniform_filter1d(counts, k_s, mode="nearest") if k_s > 1 else counts
    sm_m = uniform_filter1d(mu, k_s, mode="nearest") if k_s > 1 else mu
    above = sm_c >= params.min_fold * sm_m
    out: List[Tuple[int, int]] = []
    for a, b in _runs(sig, 1):
        s0, e0 = int(lo[a]), int(hi[b - 1])
        idx = np.flatnonzero(above[s0:e0])
        if idx.size == 0:
            continue
        s1, e1 = s0 + int(idx[0]), s0 + int(idx[-1]) + 1
        if out and s1 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e1))
        else:
            out.append((s1, e1))
    return out


def _union_intervals(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]):
    """Union of two interval lists (bins), overlapping or touching intervals joined."""
    allv = sorted([tuple(x) for x in a] + [tuple(x) for x in b])
    out: List[List[int]] = []
    for s0, e0 in allv:
        if out and s0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e0)
        else:
            out.append([int(s0), int(e0)])
    return [(x, y) for x, y in out]


def _merge_close(intervals: Sequence[Tuple[int, int]], max_gap_bins: int):
    if max_gap_bins <= 0 or not intervals:
        return list(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - out[-1][1] <= max_gap_bins:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def _version() -> str:
    from . import __version__

    return __version__
