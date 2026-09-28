"""Peak calling orchestration: coverage -> background -> segmentation -> NestedPeakSet."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .background.nb import BackgroundModel, check_window_scale
from .peaks import NestedPeakSet, Region, SubPeak
from .segment.hmm import NBHMM, _runs, segment_chromosome
from .segment.subpeak import DEFAULT_BANDWIDTHS, find_subpeaks
from .signal.coverage import Coverage
from .stats.nbtest import (
    aggregate_nb_params, autocorrelation_length_bp, nb_neglog10_sf,
    qvalue_from_neglog10p,
)

__all__ = ["CallParams", "call_peaks", "PRESETS"]


@dataclass
class CallParams:
    """Everything that controls a call.  Serialised into every output header."""

    bin_size: int = 25
    windows: Tuple[int, ...] = (1000, 5000, 10000)
    bandwidths: Tuple[int, ...] = DEFAULT_BANDWIDTHS
    posterior_cutoff: float = 0.5
    min_width: int = 100
    max_gap: int = 0
    """Merge regions separated by <= max_gap bp.  0 disables (the HMM already
    infers boundaries; this exists for compatibility, not as a tuning knob)."""
    min_fold: float = 1.5
    qvalue: float = 0.05
    min_persistence: int = 2
    call_subpeaks: bool = True
    domain_min_width: int = 2000
    """Regions at least this wide are labelled ``domain`` rather than ``peak``."""
    background_model: str = "nb"  # "nb" | "poisson" (ablation, DESIGN 9.5)
    boundary_fraction: float = 0.5
    hmm_stay_background: float = 0.995
    hmm_stay_enriched: float = 0.98
    background_exclude_fold: float = 2.0
    """Bins above this fold are held out when fitting the local background.

    Without it a peak sets its own background: the windowed mean it is divided
    by includes the peak, so fold enrichment collapses toward 1 in proportion to
    how much of the window the feature fills.  0 restores the plain windowed
    mean (DESIGN 9.5 ablation)."""
    background_exclude_iters: int = 3
    background_exclude_smooth_bp: Optional[int] = None
    """Scale at which signal is judged before being held out of the background.

    None derives it from the windows.  It is coupled to
    ``background_exclude_fold``: judging a wider average makes noise cross the
    threshold far less often, which is what lets the threshold come down far
    enough to see a broad domain sitting at only ~1.5x background."""
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
    max_dwell: int = 100_000
    """Cap on that floor, in bp -- a degenerate autocorrelation estimate on a
    short or repetitive contig must not turn the chromosome into one region."""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["windows"] = list(self.windows)
        d["bandwidths"] = list(self.bandwidths)
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
    "h3k27me3": dict(bin_size=100, windows=(200_000, 1_000_000),
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
    "h3k9me3": dict(bin_size=100, windows=(200_000, 1_000_000),
                    bandwidths=(500, 1500, 5000, 15000), min_width=300,
                    domain_min_width=5000, min_fold=1.2, posterior_cutoff=0.45),
    "h3k36me3": dict(bin_size=100, windows=(100_000, 500_000),
                     bandwidths=(500, 1500, 5000), min_width=500,
                     domain_min_width=5000, min_fold=1.2),
    "atac": dict(bin_size=10, windows=(1000, 5000, 10000), bandwidths=(50, 150, 500),
                 min_width=100, domain_min_width=1500, min_fold=2.0),
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
):
    """Call nested peaks on binned coverage.

    Returns a :class:`NestedPeakSet`, or a :class:`CallResult` if
    ``return_details`` is set (used by QC and by the tuning sweep).
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

    bg = BackgroundModel.fit(
        treatment, control=control, windows=params.windows, covariates=covariates,
        exclude_fold=params.background_exclude_fold,
        exclude_iters=params.background_exclude_iters,
        exclude_smooth_bp=params.background_exclude_smooth_bp,
    )

    bs = treatment.bin_size
    min_bins = max(1, params.min_width // bs)
    regions: List[Region] = []
    diagnostics: Dict = {"chroms": {}}

    # One scale estimate for the whole sample, used for both the segmentation
    # bin and the HMM dwell floor.  Per chromosome it is too unstable to trust
    # (see signal_scale_bp), and a setting that silently switches off on half
    # the chromosomes is worse than one that is simply wrong.
    scale_bp = signal_scale_bp(treatment)
    seg_bs = _segment_bin_size(treatment, params, scale_bp)
    factor = max(1, seg_bs // bs)
    diagnostics["signal_scale_bp"] = int(scale_bp)
    diagnostics["segment_bin_size"] = int(factor * bs)

    for chrom in sorted(treatment.bins):
        counts = treatment.bins[chrom].astype(np.float64)
        mu = bg.mu[chrom].astype(np.float64)
        if counts.size != mu.size:  # pragma: no cover - defensive
            m = min(counts.size, mu.size)
            counts, mu = counts[:m], mu[:m]

        # Segment on coarsened bins, then map the posterior back to reporting
        # resolution.  Only the segmentation is coarsened: region statistics,
        # the region-level test and sub-peak detection all still run on the
        # fine bins, so nothing but the boundary quantisation is given up.
        c_seg = _coarsen(counts, factor)
        mu_seg = _coarsen(mu, factor)
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
        alpha_seg = _aggregate_alpha(mu, alpha, factor, bg.autocorr_inflation)

        if params.background_model == "poisson":
            alpha = np.full_like(mu, 1e-9)  # NB -> Poisson, for the ablation
            alpha_seg = np.full_like(mu_seg, 1e-9)

        hmm = NBHMM(
            stay_background=params.hmm_stay_background,
            stay_enriched=params.hmm_stay_enriched,
            min_stay_enriched=_dwell_floor(
                scale_bp if params.dwell_from_autocorrelation else 0,
                factor * bs, params),
            fit_state_dispersion=params.hmm_fit_state_dispersion,
        )
        _, res = segment_chromosome(
            c_seg, mu_seg, alpha_seg,
            posterior_cutoff=params.posterior_cutoff,
            min_bins=1,
            hmm=hmm,
        )
        posterior = (np.repeat(res.posterior, factor)[: counts.size]
                     if factor > 1 else res.posterior)
        res.posterior = posterior
        intervals = _runs(posterior >= params.posterior_cutoff, min_bins)
        intervals = _merge_close(intervals, params.max_gap // bs if params.max_gap else 0)
        diagnostics["chroms"][chrom] = {
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

        chrom_len = treatment.chrom_sizes.get(chrom, counts.size * bs)
        for b0, b1 in intervals:
            region = _build_region(
                chrom, b0, b1, counts, mu, alpha, res.posterior, bs, chrom_len,
                params, bg.autocorr_inflation,
            )
            if region is not None:
                regions.append(region)

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
            "signal_scale_bp": int(scale_bp),
            "is_count_data": treatment.is_count,
            "approximate_pvalues": not treatment.is_count,
        },
        chrom_sizes=dict(treatment.chrom_sizes),
    )

    peaks = _apply_fdr(peaks, params)
    if blacklist:
        before = len(peaks)
        peaks = peaks.exclude(blacklist)
        diagnostics["blacklisted_regions_dropped"] = before - len(peaks)
        peaks.params["blacklist_regions_dropped"] = before - len(peaks)
    peaks.name_all(name)

    if return_details:
        return CallResult(peaks=peaks, background=bg, params=params, diagnostics=diagnostics)
    return peaks


def _aggregate_alpha(mu: np.ndarray, alpha: np.ndarray, factor: int,
                     inflation: float = 1.0) -> np.ndarray:
    """Dispersion of block sums of ``factor`` bins, from the per-bin trend."""
    mu_seg = _coarsen(mu, factor)
    if factor <= 1:
        return np.maximum(alpha * inflation, 1e-10)
    var = mu_seg + max(inflation, 1.0) * _coarsen(alpha * mu * mu, factor)
    return np.maximum((var - mu_seg) / np.maximum(mu_seg * mu_seg, 1e-12), 1e-10)


def _coarsen(a: np.ndarray, factor: int) -> np.ndarray:
    """Sum adjacent bins in blocks of ``factor``, zero-padding the tail."""
    if factor <= 1:
        return a
    pad = (-a.size) % factor
    if pad:
        a = np.concatenate([a, np.zeros(pad, dtype=a.dtype)])
    return a.reshape(-1, factor).sum(axis=1)


def signal_scale_bp(treatment: Coverage, min_bins: int = 1000) -> int:
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
    lens = [
        autocorrelation_length_bp(a, treatment.bin_size)
        for a in treatment.bins.values() if a.size >= min_bins
    ]
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
    return int(np.clip(seg, bs, cap))


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
        mu[b0:b1], alpha[b0:b1], inflation=inflation)
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
    seg_bg = mu[lo:hi]
    if seg.size < 3:
        return []

    maxima = find_subpeaks(
        seg, bin_size, background=seg_bg,
        bandwidths=params.bandwidths,
        min_persistence=params.min_persistence,
        min_fold=params.min_fold,
        boundary_fraction=params.boundary_fraction,
        offset=lo,
    )

    subs: List[SubPeak] = []
    for m in maxima:
        # Keep only summits inside the parent region.
        if not (b0 <= m.index < b1):
            continue
        s0, s1 = max(m.start, b0), min(m.end, b1)
        if s1 <= s0:
            continue
        obs = float(counts[s0:s1].sum())
        exp = float(mu[s0:s1].sum())
        if exp <= 0:
            continue
        fold = obs / exp
        mu_tot, a = aggregate_nb_params(mu[s0:s1], alpha[s0:s1], inflation=inflation)
        nlp = float(nb_neglog10_sf(np.array([obs]), np.array([mu_tot]), np.array([a]))[0])

        start = min(s0 * bin_size, chrom_len)
        end = min(s1 * bin_size, chrom_len)
        summit = min(m.index * bin_size + bin_size // 2, chrom_len - 1)
        if end <= start:
            continue
        summit = int(min(max(summit, start), end - 1))
        subs.append(
            SubPeak(chrom=chrom, start=start, end=end, summit=summit,
                    score=obs, neg_log10_p=nlp, fold_enrichment=fold,
                    persistence=m.persistence)
        )
    return subs


def _apply_fdr(peaks: NestedPeakSet, params: CallParams) -> NestedPeakSet:
    """BH across all regions, then across all sub-peaks, then threshold."""
    if len(peaks) == 0:
        return peaks

    region_p = np.array([r.neg_log10_p for r in peaks.regions])
    region_q = qvalue_from_neglog10p(region_p)
    for r, q in zip(peaks.regions, region_q):
        r.neg_log10_q = float(q)

    subs = [s for r in peaks.regions for s in r.subpeaks]
    if subs:
        sq = qvalue_from_neglog10p(np.array([s.neg_log10_p for s in subs]))
        for s, q in zip(subs, sq):
            s.neg_log10_q = float(q)

    cutoff = -np.log10(params.qvalue) if params.qvalue > 0 else 0.0
    kept = peaks.filter(min_qvalue_score=cutoff, min_width=params.min_width,
                        min_fold=params.min_fold)
    # Drop sub-peaks that individually fail; the parent region can still pass.
    for r in kept.regions:
        r.subpeaks = [s for s in r.subpeaks if s.neg_log10_q >= cutoff]
    return kept


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
