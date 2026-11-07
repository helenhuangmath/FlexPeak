"""Quick quantification pre-pass: profile one chromosome before calling anything.

The idea is MACS-like in spirit -- look at the data before choosing the model
-- but it answers the questions FlexPeak actually has to answer before a call:

  * at what resolution does a bin carry a decision?           -> bin size
  * how wide are the features, and how much genome do they fill?
                                           -> windows, segmentation scale,
                                              min_width, domain_min_width
  * how close together are the pieces of one feature?         -> max_gap
  * narrow, broad, or both?                                   -> tuning grid

It is cheap by construction: one chromosome, read once at a fine base bin and
coarsened to each candidate bin size (10, 25, 50 bp by default), with every
statistic an array operation.  On a 200 Mb chromosome it takes seconds, which
is what lets it run before the full load and shrink the tuning sweep from a
27-point grid on three chromosomes to a handful of points on one.

Background and signal are separated by iterative sigma clipping: start from
the median and the MAD, clip everything above ``mean + n_sd * sd`` (5 SD by
default), re-estimate mean and sd on what is left, repeat.  Bins beyond the
threshold are *candidate* signal.  That is deliberately crude at a single bin:
a sparse library at 10 bp has a long Poisson/NB tail that crosses 5 SD on
noise alone.  So the observed tail is compared with the tail a negative
binomial with the clipped mean and dispersion would produce, and the **excess**
is reported as the signal fraction.  The same clip is applied to the signal
smoothed at 1 kb and 5 kb, where noise has been averaged down and only broad
enrichment survives; the ratio of the excess at 5 kb to the excess at the
finest bin is what tells a domain mark from a punctate one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d

from .signal.coverage import Coverage
from .stats.nbtest import autocorrelation_length_bp

__all__ = ["BinProfile", "ScanProfile", "ScanRecommendation", "scan_coverage",
           "scan_file", "scan_window", "plan_scan_windows", "sigma_clip",
           "DEFAULT_SCAN_BINS", "DEFAULT_SMOOTH_BP", "DEFAULT_SCAN_MAX_BP",
           "DEFAULT_SCAN_WINDOWS"]

DEFAULT_SCAN_BINS: Tuple[int, ...] = (10, 25, 50)
DEFAULT_SMOOTH_BP: Tuple[int, ...] = (1_000, 5_000)


# -- the clip ---------------------------------------------------------------

def sigma_clip(x: np.ndarray, n_sd: float = 5.0, max_iter: int = 20,
               tol: float = 1e-3, averaged_over: int = 1) -> Tuple[float, float, np.ndarray]:
    """Iterative upper sigma clip.  Returns ``(mean, sd, keep_mask)``.

    Starts from the median and 1.4826 * MAD so the first threshold is not set
    by the signal it is meant to exclude; converges in a few iterations because
    each pass can only lower the threshold.  Zero-inflated data makes the MAD
    collapse to 0, so the sd is floored at the Poisson sd of the mean --
    divided by ``sqrt(averaged_over)`` when ``x`` is a running mean over that
    many bins, otherwise the floor sits above a smoothed domain and the clip
    never finds it.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0, 0.0, np.zeros(0, dtype=bool)
    k = max(1.0, float(averaged_over))
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    # Zero-majority data (sparse CUT&RUN at a fine bin) has median 0 and
    # MAD 0, and a clip started there keeps only the zeros.  Start from the
    # mean instead and floor the sd at the Poisson sd of that mean.
    centre = med if med > 0 else float(x.mean())
    sd = max(mad, np.sqrt(max(centre, 1e-9) / k), 1e-9)
    mean = centre
    keep = x <= mean + n_sd * sd
    for _ in range(max_iter):
        if keep.sum() < 10:
            break
        kept = x[keep]
        new_mean = float(kept.mean())
        new_sd = max(float(kept.std()), np.sqrt(max(new_mean, 1e-9) / k) * 0.5, 1e-9)
        new_keep = x <= new_mean + n_sd * new_sd
        done = (abs(new_mean - mean) <= tol * max(mean, 1e-9)
                and abs(new_sd - sd) <= tol * max(sd, 1e-9))
        mean, sd, keep = new_mean, new_sd, new_keep
        if done:
            break
    return mean, sd, keep


def _nb_tail(mean: float, sd: float, thr: float) -> float:
    """P(X > thr) under NB with that mean and variance (Poisson if var<=mean)."""
    if mean <= 0:
        return 0.0
    var = sd * sd
    k = np.floor(thr)
    if var <= mean * (1 + 1e-6):
        return float(stats.poisson.sf(k, mean))
    alpha = (var - mean) / (mean * mean)
    n = 1.0 / alpha
    p = n / (n + mean)
    return float(stats.nbinom.sf(k, n, p))


def _runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lengths of True runs and of the gaps between consecutive runs (bins)."""
    if mask.size == 0 or not mask.any():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    return e - s, (s[1:] - e[:-1]) if s.size > 1 else np.zeros(0, dtype=np.int64)


def _q(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else 0.0


# -- data contracts ---------------------------------------------------------

@dataclass
class BinProfile:
    """What one chromosome looks like at one bin size."""

    bin_size: int
    n_bins: int
    mean: float
    median: float
    robust_sd: float
    frac_zero: float
    q90: float
    q99: float
    q999: float
    bg_mean: float
    """Mean of the bins the clip kept (the background)."""
    bg_sd: float
    bg_frac: float
    """Fraction of bins the clip kept."""
    threshold: float
    """``bg_mean + n_sd * bg_sd`` in signal units."""
    dispersion: float
    """NB dispersion of the background bins, (var - mean) / mean^2."""
    tail_observed: float
    """Fraction of bins above the threshold."""
    tail_expected: float
    """Fraction an NB background with the clipped moments would put there."""
    signal_excess: float
    """``max(tail_observed - tail_expected, 0)``: the estimated signal fraction."""
    snr: float
    """(99.9th percentile - bg_mean) / bg_sd: how far the strongest bins stand out."""
    autocorr_bp: int
    n_runs: int
    run_median_bp: float
    run_p90_bp: float
    run_max_bp: int
    gap_median_bp: float
    gap_p25_bp: float
    smoothed_excess: Dict[int, float] = field(default_factory=dict)
    """Signal excess after smoothing at each ``DEFAULT_SMOOTH_BP`` scale."""
    smoothed_run_median_bp: Dict[int, float] = field(default_factory=dict)
    """Median width of the excess runs at each smoothing scale: a focal peak
    smeared by a 5 kb filter gives ~5 kb runs, a domain gives its own width."""
    smoothed_run_p90_bp: Dict[int, float] = field(default_factory=dict)


@dataclass
class ScanRecommendation:
    """Parameter choices read off the profile, and the reasons for them."""

    signal_class: str
    """'narrow' | 'broad' | 'mixed' | 'null'."""
    bin_size: int
    windows: Tuple[int, ...]
    segment_bin_size: Optional[int]
    min_width: int
    domain_min_width: int
    max_gap: int
    posterior_cutoff: float
    min_fold: float
    scale_bp: int
    """The feature scale the choices were derived from (autocorrelation length)."""
    focal_scales: Tuple[int, ...] = ()
    """Sliding-window widths for the focal test (``CallParams.focal_scales``),
    read off the run-width distributions: median bin-level run, 90th
    percentile run, and (domain marks) median run after 5 kb smoothing."""
    domain_dominated: bool = False
    """Smoothed excess dwarfs the bin-level excess: domains with focal cores."""
    grid: Dict[str, Sequence] = field(default_factory=dict)
    """A small tuning grid for ``flexpeak.tune`` -- the full 27-point grid is
    not needed once the scan has settled the scale questions."""
    reasons: List[str] = field(default_factory=list)

    def overrides(self) -> Dict:
        """``CallParams.replace(**overrides)`` arguments."""
        d = dict(bin_size=self.bin_size, windows=tuple(self.windows),
                 min_width=self.min_width, domain_min_width=self.domain_min_width,
                 max_gap=self.max_gap, posterior_cutoff=self.posterior_cutoff,
                 min_fold=self.min_fold)
        if self.segment_bin_size:
            d["segment_bin_size"] = int(self.segment_bin_size)
        if self.focal_scales:
            d["focal_scales"] = tuple(int(w) for w in self.focal_scales)
        return d


@dataclass
class ScanProfile:
    chrom: str
    chrom_size: int
    source: str
    n_sd: float
    is_count: bool
    profiles: Dict[int, BinProfile]
    elapsed_s: float = 0.0
    window: Tuple[int, int] = (0, 0)
    """The profiled stretch of the chromosome (bp); the first window when
    several chromosomes were profiled (see ``regions``)."""
    regions: List[Tuple[str, int, int]] = field(default_factory=list)
    """Every profiled ``(chrom, start, end)``.  The default scan takes one
    window from each of several chromosomes spread over the size ranking
    (``plan_scan_windows``), so the scale it reads is not one chromosome's."""

    window_votes: List[Tuple[str, str, bool]] = field(default_factory=list)
    """Per-window ``(region, class, domain_dominated)`` when several windows
    were profiled: each window's own classification, used to keep a
    borderline pooled call on the safe side (see ``_consensus``)."""

    @property
    def bin_sizes(self) -> List[int]:
        return sorted(self.profiles)

    @property
    def profiled_bp(self) -> int:
        if self.regions:
            return int(sum(e - s for _, s, e in self.regions))
        return int(self.window[1] - self.window[0]) if self.window[1] > self.window[0] \
            else int(self.chrom_size)

    # -- recommendation -----------------------------------------------------

    def recommend(self, n_sd: Optional[float] = None,
                  _force: Optional[Tuple[str, bool]] = None) -> ScanRecommendation:
        """Turn the profile into parameters.  Every rule is stated in ``reasons``."""
        profs = [self.profiles[b] for b in self.bin_sizes]
        finest, coarsest = profs[0], profs[-1]
        reasons: List[str] = []

        # Focal structure: the high-passed autocorrelation length and the
        # bin-level excess.  Domain structure: the excess that survives
        # smoothing, and how wide its runs are compared with the filter.
        acorr = int(max(np.median([p.autocorr_bp for p in profs]), finest.bin_size))
        excess_fine = max(p.signal_excess for p in profs)
        scales = sorted(coarsest.smoothed_excess)
        s_hi = scales[-1] if scales else 0
        s_lo = scales[0] if scales else 0
        excess_hi = coarsest.smoothed_excess.get(s_hi, 0.0)
        excess_lo = coarsest.smoothed_excess.get(s_lo, 0.0)
        run_hi = coarsest.smoothed_run_median_bp.get(s_hi, 0.0)
        run_hi90 = coarsest.smoothed_run_p90_bp.get(s_hi, 0.0)
        reasons.append(f"high-passed autocorrelation length {acorr:,} bp "
                       f"({', '.join(f'{p.bin_size}bp={p.autocorr_bp:,}' for p in profs)})")
        reasons.append(f"signal excess beyond {self.n_sd:g} SD: {excess_fine:.4f} at the bin "
                       f"level (SNR {max(p.snr for p in profs):.0f}), {excess_lo:.4f} at "
                       f"{s_lo:,} bp, {excess_hi:.4f} at {s_hi:,} bp")
        reasons.append(f"excess runs at {s_hi:,} bp smoothing: median {run_hi:,.0f} bp, "
                       f"p90 {run_hi90:,.0f} bp; bin-level runs at {coarsest.bin_size} bp: "
                       f"n={coarsest.n_runs:,}, p90 {coarsest.run_p90_bp:,.0f} bp, "
                       f"gap p25 {coarsest.gap_p25_bp:,.0f} bp")

        # Focal structure needs bins that stand out, not just bins that cross
        # 5 SD somewhere: a weak domain mark puts a few strong cores beyond the
        # threshold too (H3K9me3 CUT&RUN: excess 0.001 at SNR 6, against CTCF
        # at SNR 45 and H3K27me3 at SNR 21), and reading those as peaks sends
        # the caller to a fine bin and a tenfold slower run for nothing.
        snr = max(p.snr for p in profs)
        focal = excess_fine >= 1e-3 and snr >= 10.0
        # p90, not the median: a handful of domains among many focal peaks is
        # still a domain mark, and the median of run widths would not see them.
        wide = run_hi90 >= 3.0 * s_hi
        smooth_only = excess_hi >= 2e-3 or excess_lo >= 2e-3
        # Smoothed excess that the focal peaks cannot account for.  A focal
        # peak of width w smeared by an s-wide filter occupies ~w + s, so the
        # bin-level runs predict how much of the window focal structure alone
        # would light up at the coarse scale; excess well beyond that is
        # domain territory, whether or not the label says "focal".  This is
        # what makes a borderline domain mark safe: calling it narrow puts
        # 1-10 kb background windows under it and the domains vanish
        # (measured on H3K9me3 CUT&RUN: 3,880 regions -> 70, FRiP 0.12 ->
        # 0.0004), whereas calling a punctate mark broad only costs resolution.
        span = float(self.profiled_bp)
        expected_focal = coarsest.n_runs * (coarsest.run_median_bp + s_hi) / max(span, 1.0)
        # 1.5x, not 2x: measured on H3K9me3 CUT&RUN the window excess was 1.7x
        # the focal expectation and the narrow setting erased the mark.  CTCF
        # (0.4x), H3K27ac (0.5x) and the simulated narrow set (0.5x) stay narrow.
        unexplained = excess_hi >= 2e-3 and excess_hi > 1.5 * expected_focal
        reasons.append(f"smoothed excess {excess_hi:.3f} vs {expected_focal:.3f} expected from "
                       f"{coarsest.n_runs:,} focal runs smeared by the {s_hi:,} bp filter"
                       + (": domains beyond the focal structure" if unexplained else ""))
        wide = wide or unexplained
        # Domain-dominated: the smoothed excess dwarfs the bin-level excess,
        # so the bins beyond 5 SD are the strong cores of domains, not
        # separate focal peaks (measured: H3K27me3 CUT&RUN 0.023 at the bin
        # level against 0.25 at 5 kb; CTCF 0.012 against 0.12).
        dominated = excess_hi >= 8.0 * max(excess_fine, 1e-6)
        if not focal and not smooth_only:
            cls = "null"
            reasons.append("no excess at any scale: no reproducible enrichment on this chromosome")
        elif focal and wide:
            cls = "mixed"
            reasons.append(f"bin-level peaks and smoothed runs up to {run_hi90 / max(s_hi, 1):.1f}x "
                           f"wider than the {s_hi:,} bp filter: domains with focal structure"
                           + (" (domain-dominated: smoothed excess "
                              f"{excess_hi / max(excess_fine, 1e-6):.0f}x the bin-level excess)"
                              if dominated else ""))
        elif not focal:
            cls = "broad"
            reasons.append(f"no bin-level excess but {excess_hi:.3f} of the chromosome in smoothed "
                           f"excess (runs up to {run_hi90:,.0f} bp): domain mark"
                           + ("" if wide else " with short domains"))
        else:
            cls = "narrow"
            reasons.append("excess at the bin level, and smoothed runs no wider than the "
                           "filter: punctate mark")

        if _force is not None:
            cls, dominated = _force
        elif self.window_votes:
            c2, d2, note = _consensus(cls, dominated, self.window_votes)
            if note:
                reasons.append(note)
            cls, dominated = c2, d2

        if cls in ("broad", "mixed"):
            scale = int(max(run_hi90, acorr, coarsest.bin_size))
        else:
            scale = int(max(acorr, coarsest.run_p90_bp, 200))

        if cls == "narrow":
            # About 20 bins across one feature: finer than that buys nothing
            # the sub-peak summit does not already give, and every pass
            # downstream scales with the bin count (a 10 bp genome-wide call
            # is ~10x a 100 bp one).  Candidates that are mostly empty are
            # skipped for the same reason.
            usable = [p for p in profs if p.frac_zero < 0.97 and p.bin_size <= max(scale // 20, 10)]
            bin_size = (usable[-1].bin_size if usable else
                        min(profs, key=lambda p: p.bin_size).bin_size)
            reasons.append(f"bin {bin_size} bp: largest candidate with >= 20 bins per "
                           f"feature ({scale:,} bp scale) and < 97% empty bins "
                           f"(empty: {', '.join(f'{p.bin_size}bp={p.frac_zero:.0%}' for p in profs)})")
        elif cls == "broad":
            bin_size = max(coarsest.bin_size, 100 if scale >= 5_000 else 50)
            reasons.append(f"bin {bin_size} bp: broad mark, reporting resolution is not the "
                           "limit and every downstream pass scales with the bin count")
        elif cls == "mixed" and dominated:
            bin_size = max(coarsest.bin_size, 100 if scale >= 5_000 else 50)
            reasons.append(f"bin {bin_size} bp: domain-dominated mixed mark, binned like a "
                           "broad one (the sub-peaks are still found at the finer scale)")
        elif cls == "mixed":
            bin_size = 25 if 25 in self.profiles else profs[len(profs) // 2].bin_size
            reasons.append(f"bin {bin_size} bp: mixed mark, middle candidate")
        else:
            bin_size = 25 if 25 in self.profiles else profs[len(profs) // 2].bin_size

        # Domain marks: the background windows must dwarf not just a domain
        # but the weak, several-hundred-kb neighbourhoods domains sit in,
        # or the neighbourhood becomes its own background (measured on
        # H3K27me3 at Havcr2: not called at 200 kb / 1 Mb, called at
        # 1 Mb / 5 Mb).  The focal pass (1-10 kb windows) keeps the
        # sub-structure sharp regardless.
        if cls == "broad" or (cls == "mixed" and dominated):
            windows = (1_000_000, 5_000_000)
        elif cls == "mixed":
            w1 = int(np.clip(10 * scale, 10_000, 200_000))
            windows = (w1, 5 * w1)
        else:
            windows = (1_000, 5_000, 10_000)
        reasons.append(f"background windows {', '.join(f'{w:,}' for w in windows)} bp"
                       + (" (domain mark: wide enough for weak neighbourhoods)"
                          if windows[0] >= 1_000_000 else ""))

        seg = None
        if cls == "broad" or (cls == "mixed" and dominated):
            seg = int(np.clip((scale // 20 // bin_size) * bin_size, bin_size, 5_000))
            reasons.append(f"segmentation bin {seg:,} bp (domain scale / 20)")
        min_width = {"narrow": 100, "mixed": 150, "broad": 500, "null": 100}[cls]
        domain_min = {"narrow": 1_500, "mixed": 3_000, "broad": 5_000, "null": 2_000}[cls]
        post = {"narrow": 0.5, "mixed": 0.5, "broad": 0.45, "null": 0.5}[cls]
        min_fold = {"narrow": 1.5, "mixed": 1.2, "broad": 1.2, "null": 1.5}[cls]

        # Connect nearby similar measurements: pieces of one feature are
        # separated by gaps shorter than the gaps between features.  The
        # lower quartile of inter-run gaps, capped by the feature scale, is the
        # distance below which two runs are more likely one feature than two.
        gap = coarsest.gap_p25_bp if coarsest.n_runs >= 20 else 0.0
        cap = {"narrow": 2 * scale, "mixed": 5 * scale, "broad": 10 * scale, "null": 0}[cls]
        max_gap = int(min(gap, cap)) if cap else 0
        max_gap = (max_gap // bin_size) * bin_size
        reasons.append(f"max_gap {max_gap:,} bp (gap p25 {gap:,.0f} bp, cap {cap:,} bp); "
                       "regions closer than this are merged only when the gap "
                       "carries similar enrichment")

        focal_scales = focal_scale_ladder(profs, bin_size, cls, domain_min)
        reasons.append("focal sliding windows " + ", ".join(f"{w:,}" for w in focal_scales)
                       + " bp (median run, p90 run"
                       + (", 5 kb-smoothed median run" if cls in ("broad", "mixed") else "")
                       + "; each with its own left-out flank background)")

        # The axes that set what a call looks like, not only how strict it is.
        # Swept one at a time (tune(search="coordinate")), so five two-value
        # axes cost ~6-10 evaluations rather than 32.
        #   max_gap               the scan's gap quartile, and 4x it: how
        #                         far apart two pieces may be and still be
        #                         one feature (similarity-gated either way)
        #   focal_floor_fraction  how far below the robust background level
        #                         a quiet neighbourhood may set the focal
        #                         background (MACS's rule is 1.0)
        #   weak_domain_bp        the long-weak-domain pass off, or at its
        #                         window (domain marks only)
        if cls == "narrow":
            grid = {"min_fold": (1.5, 2.0), "posterior_cutoff": (0.5, 0.6)}
        else:
            grid = {"min_fold": (1.2, 1.5), "posterior_cutoff": (0.4, 0.5)}
        if max_gap > 0:
            grid["max_gap"] = (max_gap, int(min(4 * max_gap, max(cap, max_gap)) // bin_size * bin_size))
            if grid["max_gap"][1] <= max_gap:
                grid.pop("max_gap")
        domain_scale = cls == "broad" or (cls == "mixed" and dominated)
        if cls in ("broad", "mixed") and min(windows) > 10_000:
            grid["focal_floor_fraction"] = (0.5, 1.0)
        if domain_scale:
            grid["weak_domain_bp"] = (0, int(max(10_000, 2 * domain_min)))

        return ScanRecommendation(
            signal_class=cls, bin_size=int(bin_size), windows=tuple(windows),
            segment_bin_size=seg, min_width=min_width, domain_min_width=domain_min,
            max_gap=max_gap, posterior_cutoff=post, min_fold=min_fold,
            scale_bp=scale, focal_scales=tuple(focal_scales), grid=grid, reasons=reasons,
            domain_dominated=bool(dominated),
        )

    # -- reporting ----------------------------------------------------------

    def report(self, rec: Optional[ScanRecommendation] = None) -> str:
        rec = rec or self.recommend()
        lines = [
            "FlexPeak quick scan",
            "=" * 72,
            f"  source        {self.source}",
            (f"  chromosomes   {len(self.regions)} windows, {self.profiled_bp / 1e6:,.0f} Mb: "
             + ", ".join(f"{c}:{s / 1e6:.0f}-{e / 1e6:.0f}Mb" for c, s, e in self.regions)
             if len(self.regions) > 1 else
             f"  chromosome    {self.chrom} ({self.chrom_size:,} bp; profiled "
             f"{self.window[0]:,}-{self.window[1]:,})")
            + ("" if self.is_count else "   [normalised signal, not counts]"),
            f"  clip          {self.n_sd:g} SD, iterative",
            f"  elapsed       {self.elapsed_s:.1f} s",
            "",
            f"  {'bin':>5} {'empty':>6} {'mean':>8} {'bg mean':>8} {'bg sd':>7} "
            f"{'disp':>6} {'thr':>7} {'tail obs':>9} {'tail exp':>9} {'excess':>8} "
            f"{'snr':>6} {'acorr':>7} {'runs':>7} {'run50':>7} {'run90':>8} {'gap25':>7}",
        ]
        for b in self.bin_sizes:
            p = self.profiles[b]
            lines.append(
                f"  {b:>5} {p.frac_zero:>6.1%} {p.mean:>8.3f} {p.bg_mean:>8.3f} {p.bg_sd:>7.3f} "
                f"{p.dispersion:>6.2f} {p.threshold:>7.2f} {p.tail_observed:>9.4f} "
                f"{p.tail_expected:>9.4f} {p.signal_excess:>8.4f} {p.snr:>6.1f} "
                f"{p.autocorr_bp:>7,} {p.n_runs:>7,} {p.run_median_bp:>7,.0f} "
                f"{p.run_p90_bp:>8,.0f} {p.gap_p25_bp:>7,.0f}")
        last = self.profiles[self.bin_sizes[-1]]
        if last.smoothed_excess:
            lines.append("  smoothed excess (run median) at " + ", ".join(
                f"{k:,} bp: {v:.4f} ({last.smoothed_run_median_bp.get(k, 0):,.0f} bp)"
                for k, v in sorted(last.smoothed_excess.items())))
        lines += ["", f"  class         {rec.signal_class}",
                  f"  recommended   bin={rec.bin_size} windows={list(rec.windows)} "
                  f"segment_bin={rec.segment_bin_size} min_width={rec.min_width} "
                  f"domain_min_width={rec.domain_min_width} max_gap={rec.max_gap} "
                  f"posterior={rec.posterior_cutoff} min_fold={rec.min_fold}",
                  f"  focal windows {list(rec.focal_scales)} bp",
                  f"  tuning grid   {rec.grid}", "  why:"]
        lines += [f"    - {r}" for r in rec.reasons]
        if self.window_votes:
            lines.append("  per window:   " + ", ".join(
                f"{r} {c}{'*' if d else ''}" for r, c, d in self.window_votes)
                + "   (* domain-dominated)")
        return "\n".join(lines)

    def to_dict(self, rec: Optional[ScanRecommendation] = None) -> Dict:
        rec = rec or self.recommend()
        d = asdict(self)
        d["profiles"] = {str(k): asdict(v) for k, v in self.profiles.items()}
        for v in d["profiles"].values():
            for key in ("smoothed_excess", "smoothed_run_median_bp", "smoothed_run_p90_bp"):
                v[key] = {str(k): x for k, x in v[key].items()}
        d["recommendation"] = asdict(rec)
        d["recommendation"]["grid"] = {k: list(v) for k, v in rec.grid.items()}
        d["recommendation"]["focal_scales"] = list(rec.focal_scales)
        return d

    def to_json(self, path=None, rec: Optional[ScanRecommendation] = None) -> str:
        text = json.dumps(self.to_dict(rec), indent=2, default=str)
        if path:
            with open(path, "w") as fh:
                fh.write(text + "\n")
        return text


def _consensus(cls: str, dominated: bool, votes: Sequence[Tuple[str, str, bool]]):
    """Keep a pooled classification on the safe side of the windows' own.

    Pooling 60 Mb from six chromosomes can tip a borderline mark across a
    class boundary that most of its chromosomes do not cross.  The two
    errors are not symmetric: a punctate mark called mixed only loses
    resolution, a domain mark called narrow gets 1-10 kb background windows
    and loses its domains (GSE285245 H3K4me3, pooled windows "narrow": 789
    regions on six chromosomes).  So: narrow (or null) becomes mixed when a
    third of the windows see domains, and a mixed mark is domain-dominated
    when half of the windows say so.  Returns ``(cls, dominated, note)``.
    """
    n = len(votes)
    if n < 2:
        return cls, dominated, ""
    wide = sum(c in ("mixed", "broad") for _, c, _ in votes) / n
    dom = sum(bool(d) for _, c, d in votes if c in ("mixed", "broad")) / n
    new_cls, new_dom = cls, dominated
    if cls in ("narrow", "null") and wide >= 1 / 3:
        new_cls = "mixed"
    if new_cls in ("mixed", "broad") and dom >= 0.5:
        new_dom = True
    if (new_cls, new_dom) == (cls, dominated):
        return cls, dominated, ""
    return new_cls, new_dom, (
        f"windows vote {wide:.0%} mixed/broad, {dom:.0%} domain-dominated: pooled "
        f"'{cls}{' dominated' if dominated else ''}' -> '{new_cls}"
        f"{' dominated' if new_dom else ''}' (the safe side: a domain mark called "
        "narrow loses its domains, a punctate mark called mixed only resolution)")


def focal_scale_ladder(profs: Sequence[BinProfile], bin_size: int, cls: str,
                       domain_min_width: int, max_scale: int = 10_000,
                       min_ratio: float = 1.8) -> List[int]:
    """Sliding-window widths for the focal test, from the run-width distributions.

    The bin-level runs (bins beyond the clip) are stable across 10/25/50 bp
    -- measured on GSE285245: CTCF median 130-150 bp / p90 680-750 bp,
    H3K9me3 200 / 730, H3K27me3 300 / 1,800, H3K4me3 460 / 2,700 -- so their
    median is the width of a typical focal feature and their p90 the width
    of a wide one.  Domain marks add the median run after 5 kb smoothing
    (clusters and sub-domains, 3-5 kb).  The high-passed autocorrelation is
    no use here: it came out at ~9 bins at every bin size, i.e. it measures
    the bin, not the feature.  Scales within ``min_ratio`` of the previous
    one add nothing and are dropped; all are capped at ``max_scale`` (wider
    structure is the HMM's job) and snapped to the bin.
    """
    if not profs:
        return []
    by_bin = {p.bin_size: p for p in profs}
    p = by_bin.get(bin_size) or min(profs, key=lambda q: abs(q.bin_size - bin_size))
    finest = min(profs, key=lambda q: q.bin_size)
    cand = [p.run_median_bp or finest.run_median_bp, p.run_p90_bp or finest.run_p90_bp]
    if cls in ("broad", "mixed"):
        coarsest = max(profs, key=lambda q: q.bin_size)
        sm = coarsest.smoothed_run_median_bp
        if sm:
            cand.append(sm.get(max(sm), 0.0))
    lo = max(2 * bin_size, 100)
    hi = int(min(max_scale, max(domain_min_width, lo) * 4))
    out: List[int] = []
    for w in sorted(c for c in cand if c and c > 0):
        w = int(np.clip(round(w / bin_size) * bin_size, lo, hi))
        if not out or w >= min_ratio * out[-1]:
            out.append(w)
    return out


# -- the scan ---------------------------------------------------------------

def profile_bins(arr: np.ndarray, bin_size: int, n_sd: float = 5.0,
                 smooth_bp: Sequence[int] = DEFAULT_SMOOTH_BP,
                 smooth_min_fold: float = 1.5) -> BinProfile:
    """All the statistics for one array at one bin size."""
    x = np.asarray(arr, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("empty coverage array")
    bg_mean, bg_sd, keep = sigma_clip(x, n_sd)
    thr = bg_mean + n_sd * bg_sd
    above = x > thr
    tail_obs = float(above.mean())
    tail_exp = _nb_tail(bg_mean, bg_sd, thr)
    disp = max((bg_sd * bg_sd - bg_mean) / max(bg_mean * bg_mean, 1e-12), 0.0)
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    q90, q99, q999 = np.percentile(x, [90, 99, 99.9])

    runs, gaps = _runs(above)
    smoothed: Dict[int, float] = {}
    sm_run_med: Dict[int, float] = {}
    sm_run_p90: Dict[int, float] = {}
    # Autocorrelation of the high-passed signal: slow baseline drift makes
    # every sample autocorrelated over tens of kb, and it is the features'
    # own width, not the drift's, that sets the segmentation scale.
    k_hp = max(1, int(round(20_000 / bin_size)))
    hp = x - uniform_filter1d(x, size=k_hp, mode="nearest") if k_hp < x.size else x
    for s_bp in smooth_bp:
        k = max(1, int(round(s_bp / bin_size)))
        if k <= 1:
            smoothed[int(s_bp)] = float(max(tail_obs - tail_exp, 0.0))
            continue
        sm = uniform_filter1d(x, size=k, mode="nearest")
        # Judge the smoothed track against a *local* baseline (the moving
        # window MACS uses for its lambda, here a window 20x the smoothing
        # scale with apparent signal held out), not against one global level:
        # real coverage drifts over tens of kb by more than a weak domain
        # rises above it, and that drift would otherwise inflate the clipped
        # sd until no domain clears 5 SD.
        dc, base = _drift_corrected(x, sm, bin_size, s_bp)
        # The SD that "5 SD" refers to here is the measurement noise of a
        # k-bin mean -- the bin-level clipped sd over sqrt(k) -- floored at the
        # lower-tail spread of the corrected track, which absorbs whatever
        # drift the baseline could not remove without ever seeing the signal
        # side of the distribution.  A fold gate over the *local* baseline
        # keeps a shallow swell that clears the noise from counting as a
        # domain: enrichment is a ratio, not an offset.
        sd_noise = bg_sd / np.sqrt(k)
        q5, q25, med = np.percentile(dc, [5, 25, 50])
        sd_eff = max(sd_noise, (q25 - q5) / 1.2816, 1e-9)
        mask = (dc > med + n_sd * sd_eff) & (sm >= smooth_min_fold * base)
        exp_tail = float(stats.norm.sf(n_sd))
        smoothed[int(s_bp)] = float(max(mask.mean() - exp_tail, 0.0))
        r_s, _ = _runs(mask)
        sm_run_med[int(s_bp)] = float(np.median(r_s) * bin_size) if r_s.size else 0.0
        sm_run_p90[int(s_bp)] = _q(r_s, 90) * bin_size

    return BinProfile(
        bin_size=int(bin_size), n_bins=int(n), mean=float(x.mean()), median=med,
        robust_sd=float(mad), frac_zero=float((x <= 0).mean()),
        q90=float(q90), q99=float(q99), q999=float(q999),
        bg_mean=float(bg_mean), bg_sd=float(bg_sd), bg_frac=float(keep.mean()),
        threshold=float(thr), dispersion=float(disp),
        tail_observed=tail_obs, tail_expected=float(tail_exp),
        signal_excess=float(max(tail_obs - tail_exp, 0.0)),
        snr=float((q999 - bg_mean) / max(bg_sd, 1e-9)),
        autocorr_bp=int(autocorrelation_length_bp(hp, bin_size)),
        n_runs=int(runs.size),
        run_median_bp=float(np.median(runs) * bin_size) if runs.size else 0.0,
        run_p90_bp=_q(runs, 90) * bin_size, run_max_bp=int(runs.max() * bin_size) if runs.size else 0,
        gap_median_bp=float(np.median(gaps) * bin_size) if gaps.size else 0.0,
        gap_p25_bp=_q(gaps, 25) * bin_size,
        smoothed_excess=smoothed, smoothed_run_median_bp=sm_run_med,
        smoothed_run_p90_bp=sm_run_p90,
    )


def _drift_corrected(x: np.ndarray, sm: np.ndarray, bin_size: int, s_bp: int,
                     chunk_bp: int = 200_000, quantile: float = 20.0):
    """Remove slow baseline drift from a smoothed track.  Returns ``(corrected, baseline)``.

    The baseline is the lower ``quantile`` of the smoothed track in each
    ``chunk_bp`` chunk, linearly interpolated between chunk centres: a domain
    narrower than ~80% of a chunk cannot pull it up, and a quantile over
    thousands of bins is nearly noise-free, so subtracting it (and adding the
    median back) removes drift without adding tail noise of its own.  Drift at
    scales below the chunk is not removed -- nothing scale-based can separate
    it from a domain of the same width -- so weak domains riding on strong
    sub-chunk drift are found only if their amplitude clears the 5 SD rule
    against that drift, which is the honest answer.
    """
    C = max(2, int(round(chunk_bp / bin_size)))
    n = sm.size
    if n < 2 * C:
        base = np.full(n, max(float(np.percentile(sm, quantile)), 1e-9))
        return sm - base + float(np.median(base)), base
    n_chunks = n // C
    body = sm[: n_chunks * C].reshape(n_chunks, C)
    q = np.percentile(body, quantile, axis=1)
    centres = (np.arange(n_chunks) + 0.5) * C
    if n_chunks * C < n:
        q = np.append(q, np.percentile(sm[n_chunks * C:], quantile))
        centres = np.append(centres, (n_chunks * C + n) / 2.0)
    base = np.maximum(np.interp(np.arange(n), centres, q), 1e-9)
    return sm - base + float(np.median(base)), base


def pick_scan_chrom(cov_or_sizes, prefer: Optional[str] = "chr1") -> str:
    """``chr1`` when present, else the largest chromosome."""
    sizes = cov_or_sizes.chrom_sizes if isinstance(cov_or_sizes, Coverage) else dict(cov_or_sizes)
    if prefer and prefer in sizes:
        return prefer
    return max(sizes, key=lambda c: sizes[c])


DEFAULT_SCAN_MAX_BP = 60_000_000
"""The scan profiles at most this much of the chromosome.  Distributions,
run lengths and the autocorrelation are settled long before 60 Mb, and the
cost of the percentiles and FFTs grows with the length: on mm10 chr1 (195 Mb)
the full chromosome took ~55 s."""


def scan_window(chrom_size: int, max_bp: int = DEFAULT_SCAN_MAX_BP) -> Tuple[int, int]:
    """``(start, end)`` of the profiled window: a central stretch, so the
    telomeric/centromeric gap at the start of a chromosome is not in it."""
    if max_bp <= 0 or chrom_size <= max_bp:
        return 0, int(chrom_size)
    start = int(0.15 * chrom_size)
    if start + max_bp > chrom_size:
        start = chrom_size - max_bp
    return start, start + max_bp


DEFAULT_SCAN_WINDOWS = 6
"""Chromosomes the default scan samples, one window each, splitting
``DEFAULT_SCAN_MAX_BP`` between them.  The total read is the same 60 Mb as
the single-chromosome scan, so the cost is too; what changes is that the
scale, the class and the widths are read off large, mid-sized and small
chromosomes at once instead of off chr1 alone."""


def _is_primary(name: str) -> bool:
    from .io.bam import DEFAULT_EXCLUDE

    low = name.lower()
    return (name not in DEFAULT_EXCLUDE and "_" not in name and "random" not in low
            and "un" not in low.replace("chr", "") and low not in ("chry", "y"))


def plan_scan_windows(sizes: Dict[str, int], max_bp: int = DEFAULT_SCAN_MAX_BP,
                      n_windows: int = DEFAULT_SCAN_WINDOWS) -> List[Tuple[str, int, int]]:
    """Stratified scan windows: ``n_windows`` chromosomes evenly spaced over
    the size ranking (largest to smallest primary chromosome), one central
    window of ``max_bp / n`` from each.  Unplaced contigs and chrY are
    skipped; with fewer chromosomes than ``n_windows`` each gets a larger
    share, and a chromosome smaller than its share is taken whole."""
    prim = {c: int(s) for c, s in sizes.items() if _is_primary(c) and s > 0}
    if not prim:
        prim = {c: int(s) for c, s in sizes.items() if s > 0}
    order = sorted(prim, key=lambda c: (-prim[c], c))
    n = max(1, min(int(n_windows), len(order)))
    idx = sorted(set(int(round(i)) for i in np.linspace(0, len(order) - 1, n)))
    picks = [order[i] for i in idx]
    per = max_bp // len(picks) if max_bp > 0 else 0
    out = []
    for c in picks:
        s, e = scan_window(prim[c], per)
        out.append((c, s, e))
    return out


def _profile_pieces(pieces: Sequence[np.ndarray], base_bin: int, bin_sizes: Sequence[int],
                    n_sd: float, smooth_bp: Sequence[int]) -> Dict[int, BinProfile]:
    """Coarsen each window to every bin size and profile their concatenation.
    Windows are coarsened separately so no bin straddles two chromosomes; the
    handful of junctions in 60 Mb do not move a distribution, a run-length
    quantile or an autocorrelation."""
    profiles: Dict[int, BinProfile] = {}
    for b in sorted(set(int(x) for x in bin_sizes)):
        if b % base_bin:
            raise ValueError(f"bin size {b} is not a multiple of the loaded {base_bin} bp")
        k = b // base_bin
        parts = []
        for base in pieces:
            if k > 1:
                pad = (-base.size) % k
                parts.append(np.concatenate([base, np.zeros(pad, base.dtype)]).reshape(-1, k).sum(axis=1))
            else:
                parts.append(base)
        profiles[b] = profile_bins(np.concatenate(parts), b, n_sd=n_sd, smooth_bp=smooth_bp)
    return profiles


def scan_coverage(cov: Coverage, bin_sizes: Sequence[int] = DEFAULT_SCAN_BINS,
                  chrom: Optional[str] = None, n_sd: float = 5.0,
                  smooth_bp: Sequence[int] = DEFAULT_SMOOTH_BP,
                  max_bp: int = DEFAULT_SCAN_MAX_BP,
                  n_windows: int = DEFAULT_SCAN_WINDOWS) -> ScanProfile:
    """Profile an already-loaded coverage at several bin sizes.

    ``chrom`` profiles one window of that chromosome (``scan_window``);
    otherwise ``n_windows`` stratified windows over the loaded chromosomes
    (``plan_scan_windows``), ``max_bp`` in total.  ``cov`` must be binned at a
    divisor of every requested bin size (load at their GCD).
    """
    import time

    t0 = time.perf_counter()
    if chrom is not None:
        if chrom not in cov.bins:
            raise ValueError(f"{chrom!r} not in coverage ({list(cov.bins)[:5]}...)")
        plan = [(chrom,) + scan_window(int(cov.chrom_sizes[chrom]), max_bp)]
    else:
        plan = plan_scan_windows({c: cov.chrom_sizes.get(c, cov.bins[c].size * cov.bin_size)
                                  for c in cov.bins}, max_bp, n_windows)
    bs = cov.bin_size
    pieces = [cov.bins[c][s // bs: -(-e // bs)] for c, s, e in plan]
    profiles = _profile_pieces(pieces, bs, bin_sizes, n_sd, smooth_bp)
    c0, w0, w1 = plan[0]
    return ScanProfile(chrom=c0 if len(plan) == 1 else ",".join(c for c, _, _ in plan),
                       chrom_size=int(cov.chrom_sizes.get(c0, 0)),
                       source=cov.source, n_sd=float(n_sd), is_count=cov.is_count,
                       profiles=profiles, elapsed_s=time.perf_counter() - t0,
                       window=(int(w0), int(w1)), regions=[tuple(r) for r in plan],
                       window_votes=_window_votes(pieces, plan, bs, bin_sizes, n_sd,
                                                  smooth_bp, cov.is_count))


def scan_file(path, bin_sizes: Sequence[int] = DEFAULT_SCAN_BINS,
              chrom: Optional[str] = None, n_sd: float = 5.0,
              smooth_bp: Sequence[int] = DEFAULT_SMOOTH_BP, cache: bool = True,
              cache_dir=None, max_bp: int = DEFAULT_SCAN_MAX_BP,
              n_windows: int = DEFAULT_SCAN_WINDOWS, **load_kw) -> ScanProfile:
    """Scan a BAM/bigWig at the GCD of the bin sizes.

    Default: ``n_windows`` stratified windows (``plan_scan_windows``), read
    directly -- only the windowed bases are touched, so a six-chromosome scan
    reads the same 60 Mb the one-chromosome scan did.  ``chrom`` (or
    ``n_windows=1``) scans one chromosome as before.
    """
    import math
    import time

    t0 = time.perf_counter()
    base = 0
    for b in bin_sizes:
        base = math.gcd(base, int(b))
    if chrom is not None or n_windows <= 1:
        from .io.load import load_signal

        if chrom is None:
            chrom = _file_scan_chrom(path, **load_kw)
        cov = load_signal(path, bin_size=base, chroms=[chrom], cache=cache,
                          cache_dir=cache_dir, **load_kw)
        prof = scan_coverage(cov, bin_sizes, chrom=chrom, n_sd=n_sd, smooth_bp=smooth_bp,
                             max_bp=max_bp)
        prof.elapsed_s = time.perf_counter() - t0
        return prof

    sizes = _file_chrom_sizes(path, **load_kw)
    plan = plan_scan_windows(sizes, max_bp, n_windows)
    pieces, is_count = read_windows(path, plan, base, cache=cache, cache_dir=cache_dir,
                                    **load_kw)
    profiles = _profile_pieces(pieces, base, bin_sizes, n_sd, smooth_bp)
    votes = _window_votes(pieces, plan, base, bin_sizes, n_sd, smooth_bp, is_count)
    c0, w0, w1 = plan[0]
    return ScanProfile(chrom=",".join(c for c, _, _ in plan), chrom_size=int(sizes[c0]),
                       source=str(path), n_sd=float(n_sd), is_count=is_count,
                       profiles=profiles, elapsed_s=time.perf_counter() - t0,
                       window=(int(w0), int(w1)), regions=[tuple(r) for r in plan],
                       window_votes=votes)


def _window_votes(pieces, plan, base_bin, bin_sizes, n_sd, smooth_bp, is_count):
    """Each window classified on its own.  Doubles the profiling work (the
    file is read once), which is ~1 s on the 60 Mb default."""
    if len(plan) < 2:
        return []
    votes = []
    for arr, (c, s, e) in zip(pieces, plan):
        prof = ScanProfile(chrom=c, chrom_size=e, source="", n_sd=float(n_sd),
                           is_count=is_count,
                           profiles=_profile_pieces([arr], base_bin, bin_sizes, n_sd, smooth_bp),
                           window=(s, e))
        r = prof.recommend()
        votes.append((c, r.signal_class, bool(r.domain_dominated)))
    return votes


def read_windows(path, plan: Sequence[Tuple[str, int, int]], bin_size: int,
                 cache: bool = True, cache_dir=None, **load_kw):
    """Binned signal over each ``(chrom, start, end)`` window, read directly
    from the file.  Returns ``([array per window], is_count)``.  bigWig
    values are rescaled to a mean of 10 over the windows, as ``load_bigwig``
    does genome-wide.  Cached like a load (keyed on the window list)."""
    from .io.load import cache_path, infer_format

    keys = [f"{c}:{s}-{e}" for c, s, e in plan]
    cp = cache_path(path, bin_size, ["scanwin"] + keys, cache_dir) if cache else None
    if cp is not None and cp.exists():
        try:
            cov = Coverage.load(cp)
            return [cov.bins[k] for k in keys], cov.is_count
        except Exception:
            pass
    fmt = infer_format(path)
    if fmt == "bigwig":
        import pyBigWig

        bw = pyBigWig.open(str(path))
        try:
            pieces = []
            for c, s, e in plan:
                n = (e - s) // bin_size
                v = np.asarray(bw.values(c, s, s + n * bin_size, numpy=True), dtype=np.float32)
                np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.clip(v, 0.0, None, out=v)
                pieces.append(v.reshape(-1, bin_size).mean(axis=1))
        finally:
            bw.close()
        tot = sum(float(p.sum()) for p in pieces)
        nb = sum(p.size for p in pieces)
        if tot > 0:
            sc = 10.0 * nb / tot
            pieces = [(p * sc).astype(np.float32) for p in pieces]
        is_count = False
    elif fmt == "bam":
        from .io.bam import load_bam_windows

        pieces = load_bam_windows(str(path), plan, bin_size, **load_kw)
        is_count = True
    else:
        full = Coverage.load(path)
        bs = full.bin_size
        if bin_size % bs:
            raise ValueError(f"cache is binned at {bs} bp, cannot scan at {bin_size} bp")
        full = full.rebin(bin_size) if bin_size != bs else full
        pieces = [full.bins[c][s // bin_size: e // bin_size] for c, s, e in plan]
        is_count = full.is_count
    if cp is not None:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            Coverage(bins=dict(zip(keys, pieces)), bin_size=bin_size,
                     chrom_sizes={k: p.size * bin_size for k, p in zip(keys, pieces)},
                     source=str(path), is_count=is_count).save(cp)
        except OSError:
            pass
    return pieces, is_count


def _file_chrom_sizes(path, **load_kw) -> Dict[str, int]:
    """Chromosome names and sizes without reading any coverage."""
    p = str(path).lower()
    if p.endswith((".bam", ".cram", ".sam")):
        import pysam

        with pysam.AlignmentFile(str(path), reference_filename=load_kw.get("reference")) as af:
            sizes = dict(zip(af.references, af.lengths))
    elif p.endswith((".bw", ".bigwig")):
        import pyBigWig

        bw = pyBigWig.open(str(path))
        try:
            sizes = dict(bw.chroms())
        finally:
            bw.close()
    else:
        sizes = Coverage.load(path).chrom_sizes
    from .io.bam import DEFAULT_EXCLUDE

    return {c: int(s) for c, s in sizes.items() if c not in DEFAULT_EXCLUDE}


def _file_scan_chrom(path, **load_kw) -> str:
    return pick_scan_chrom(_file_chrom_sizes(path, **load_kw))
