"""Negative binomial background model.

DESIGN.md section 3.  MACS2's local Poisson lambda assumes mean == variance.
Real ChIP/CUT&RUN background is overdispersed, so on noisy or shallow libraries
a Poisson test reports wildly inflated significance and the reported FDR stops
meaning anything.  Replacing it with a negative binomial whose dispersion is
estimated from the data is the single highest-value component in the plan, and
it involves no machine learning at all.

The testable consequence is p-value calibration on negative controls
(input-vs-input should yield ~0 peaks, with uniform p-values): see
``tests/test_background.py::test_null_calibration``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..signal.coverage import Coverage
from ..stats.nbtest import autocorrelation_inflation

__all__ = ["BackgroundModel", "estimate_dispersion", "local_lambda",
           "check_window_scale", "exclusion_smoothing_bp", "effective_windows"]


#: Bins used to fit the dispersion trend and the autocorrelation inflation.
_FIT_MAX_BINS = 4_000_000

#: Chromosomes shorter than this many bins are fitted at full resolution
#: (see the block-mean fit in ``local_lambda``): nothing to gain, and small
#: test genomes keep their exact per-bin estimate.
_COARSEN_MIN_BINS = 1_000_000

#: Bins a per-chromosome summary statistic (a median, a quantile) is read
#: from; longer arrays are strided down to about this many.
_STAT_MAX_BINS = 2_000_000


def _median(x: np.ndarray) -> float:
    if x.size > _STAT_MAX_BINS:
        x = x[:: int(np.ceil(x.size / _STAT_MAX_BINS))]
    return float(np.median(x))


#: Cap on the trimming-bias correction.  A larger value than this means the
#: exclusion is removing most of the distribution, not its signal tail.
_MAX_TRIM_CORRECTION = 3.0


def _trim_correction(plain: np.ndarray, masked: np.ndarray, den: np.ndarray,
                     quiet_quantile: float = 0.75,
                     quiet_min_den: Optional[float] = None) -> float:
    """Undo the downward bias that one-sided exclusion introduces.

    Excluding only bins *above* a threshold trims one tail, so the mean of what
    is left sits below the true mean even when there is no signal at all.  Left
    uncorrected that is not a small effect: on simulated null data at dispersion
    0.5 it took the false-positive territory from 1.3% of the genome to 12.9%,
    because every window's background had been quietly deflated.

    The size of the bias is measurable from the sample.  Windows where the
    exclusion removed the least are the ones with the least signal in them, so
    there the plain mean is already a good background estimate and the ratio
    ``plain / masked`` is the trimming bias and nothing else.  Its median over
    those windows is the correction, applied everywhere.  On pure noise every
    window is a quiet window, the correction lands at exactly the bias, and the
    estimator collapses back to the plain mean -- which is what keeps the null
    calibration intact.
    """
    # The correction is one number per chromosome: a strided sample of a
    # few million bins fixes it to three digits, and the quantile / median
    # over every bin of a 10 bp chromosome was a third of the whole fit.
    if den.size > _STAT_MAX_BINS:
        step = int(np.ceil(den.size / _STAT_MAX_BINS))
        plain, masked, den = plain[::step], masked[::step], den[::step]
    if quiet_min_den is not None:
        # Absolute rather than relative quietness.  At megabase windows on a
        # domain mark no window is free of signal, and "the quietest quarter"
        # still had a fifth of its bins held out as real enrichment: the
        # ratio then measured the signal fraction, not the trimming bias
        # (GSE285245 H3K27me3 chr11, 5 Mb: 2.2x, which lifted the whole
        # background to 2-3x the true level).  Only windows that lost at most
        # 1 - quiet_min_den of their bins estimate the bias; none -> no
        # correction (with a 5 kb judging scale, noise almost never crosses
        # 2x, so the bias it would measure is ~1 anyway).
        quiet = den >= quiet_min_den
    else:
        quiet = den >= np.quantile(den, quiet_quantile)
    usable = quiet & (masked > 0) & np.isfinite(masked) & np.isfinite(plain)
    if not usable.any():
        return 1.0
    ratio = float(np.median(plain[usable] / masked[usable]))
    if not np.isfinite(ratio):
        return 1.0
    return float(np.clip(ratio, 1.0, _MAX_TRIM_CORRECTION))


def _window_max(arr: np.ndarray, weights: Optional[np.ndarray], bin_size: int,
                windows: Sequence[int], floor: float,
                min_weight: float = 0.05,
                quiet_min_den: Optional[float] = None) -> np.ndarray:
    """Max over window scales of the (optionally weighted) local mean."""
    out = np.full(arr.shape, floor, dtype=np.float32)
    for w in windows:
        k = max(1, int(round(w / bin_size)))
        if k >= arr.size:
            k = max(1, arr.size // 2)
        if weights is None:
            sm = uniform_filter1d(arr, size=k, mode="nearest")
        else:
            plain = uniform_filter1d(arr, size=k, mode="nearest")
            num = uniform_filter1d(arr * weights, size=k, mode="nearest")
            den = uniform_filter1d(weights, size=k, mode="nearest")
            # A window with almost nothing left unmasked carries no usable
            # background estimate; fall back to the floor rather than to the
            # ratio of two near-zero numbers.
            sm = np.where(den > min_weight, num / np.maximum(den, 1e-6), floor)
            sm = np.nan_to_num(sm, nan=floor, posinf=floor, neginf=floor)
            sm = sm * _trim_correction(plain, sm, den, quiet_min_den=quiet_min_den)
        np.maximum(out, sm.astype(np.float32), out=out)
    return out


def _window_median(x: np.ndarray, bin_size: int, windows: Sequence[int],
                   floor: float, blocks_per_window: int = 8) -> np.ndarray:
    """Max over window scales of a running median, per bin.

    Approximated as the median of block medians (``blocks_per_window`` blocks
    per window, a running median of ``blocks_per_window + 1`` blocks),
    interpolated between block centres: O(n) and seconds on a whole genome,
    where an exact running median over a 1 Mb window is not.
    """
    from scipy.ndimage import median_filter

    x = np.asarray(x, dtype=np.float32)
    n = x.size
    out = np.full(n, floor, dtype=np.float32)
    for w in windows:
        k = min(max(1, int(round(w / bin_size))), max(1, n // 2))
        B = max(1, k // blocks_per_window)
        nb = n // B
        if nb < 3:
            np.maximum(out, np.float32(np.median(x)), out=out)
            continue
        blk = np.median(x[: nb * B].reshape(nb, B), axis=1)
        centres = (np.arange(nb) + 0.5) * B
        if nb * B < n:
            blk = np.append(blk, np.median(x[nb * B:]))
            centres = np.append(centres, (nb * B + n) / 2.0)
        sm = median_filter(blk, size=blocks_per_window + 1, mode="nearest")
        np.maximum(out, np.interp(np.arange(n), centres, sm).astype(np.float32), out=out)
    return out


def exclusion_smoothing_bp(bin_size: int, windows: Sequence[int],
                           max_bp: int = 5000) -> int:
    """Scale at which signal is judged before being excluded from the background.

    Thresholding single bins is what makes one-sided exclusion dangerous: at
    realistic dispersion a large slice of pure noise sits above twice its own
    mean, so the mask eats the background's upper tail and deflates it.  Judging
    a *smoothed* signal removes that: averaging over n bins shrinks the noise's
    standard deviation by sqrt(n) while leaving a genuine domain at its true
    fold, so noise stops crossing the threshold and enrichment still does.

    The scale is a tenth of the narrowest background window -- small enough to
    stay far below the background estimate's own resolution, wide enough to
    average noise down -- floored at a few bins and capped so a megabase-window
    preset does not smooth away the features it is trying to find.
    """
    if not windows:
        return max(4 * bin_size, bin_size)
    return int(max(4 * bin_size, min(min(windows) // 10, max_bp)))


def local_lambda(
    arr: np.ndarray,
    bin_size: int,
    windows: Sequence[int] = (1000, 5000, 10000),
    genome_mean: float = 0.0,
    exclude_fold: float = 2.0,
    exclude_iters: int = 3,
    exclude_smooth_bp: Optional[int] = None,
    robust_start: bool = False,
    unit: Optional[float] = None,
    noise_z: float = 3.0,
    info: Optional[dict] = None,
    coarsen: bool = True,
) -> np.ndarray:
    """MACS-style local background: max over several window scales.

    ``unit`` is the signal one fragment contributes (the track's total over
    its fragment count; ~fragment length / bin size for a pileup).  It makes
    the signal exclusion and the background floor count-aware -- see
    ``_signal_threshold`` and the floor below.  None (a normalised track of
    unknown depth) keeps the plain fold rule.  ``info``, if given, receives
    ``held_out`` (bins the final pass set aside as signal) and ``n``.

    Taking the maximum is deliberately conservative -- it is what stops a caller
    from turning a broad swell of background into thousands of "peaks".

    ``exclude_fold`` fixes the failure that the plain windowed mean has when the
    background is estimated from the treatment: **a peak otherwise defines its
    own background.**  A window is a mean over its bins, so a peak that fills an
    appreciable fraction of the window raises the very number it is about to be
    divided by, and its fold enrichment collapses toward 1.  The wider the
    feature, the worse it gets -- which is exactly backwards for a caller whose
    reason to exist is broad marks.

    So the estimate is iterated: fit, mark bins above ``exclude_fold`` times the
    current estimate as apparent signal, refit the window means from the
    remaining bins only, repeat.  Two or three passes is enough; it converges
    quickly because each pass can only lower the estimate.  Set
    ``exclude_fold=0`` for the plain unmasked mean.

    Measured on CUT&RUN H3K27ac and H3K27me3 samples: median fold enrichment in
    the top 5% of bins rose 2.5x -> 8.0x (H3K27ac) and 3.6x -> 26.6x (H3K27me3),
    and H3K27me3 domains stopped being shredded into ~800 bp fragments.

    ``robust_start`` anchors the exclusion on a running median of the
    smoothed track: bins are held out above ``exclude_fold`` x min(current
    estimate, median), so the threshold can fall but never rise above twice
    the median.  The plain iteration has a self-consistent fixed point on a
    mark with a lot of moderate enrichment: everything below twice the
    estimate is kept, which lifts the estimate, which keeps more.  Measured
    on GSE285245 H3K27me3 chr11 (100 bp, 1 Mb / 5 Mb): the gap next to
    Havcr2 is at 2.4 and the chromosome median bin 3.0, but the background
    converged to 5.8 there (chromosome median 7.8), so a 23 kb block at 9.7
    -- 4x the true level -- read 1.7x.  A median does not move until
    enrichment fills half the window.  On background alone the smoothed
    track is near-symmetric, median ~ mean, nothing crosses the threshold
    either way, and the masked mean with its trimming correction is still
    the estimate.
    """
    arr32 = np.ascontiguousarray(arr, dtype=np.float32)
    floor = max(float(genome_mean), 1e-9)
    if not exclude_fold or exclude_fold <= 0 or exclude_iters < 1:
        return _window_max(arr32, None, bin_size, windows, floor)

    if exclude_smooth_bp is None:
        exclude_smooth_bp = exclusion_smoothing_bp(bin_size, windows)
    k_s = max(1, int(round(exclude_smooth_bp / bin_size)))

    # Nothing below resolves anything finer than the judging scale: signal
    # is judged on a k_s-bin mean and the background is a mean over windows
    # at least ten times that.  On a long chromosome the fit is therefore
    # run on block means of half the judging scale and interpolated back --
    # the same estimate, k_s/2 times fewer bins through a dozen filters and
    # six exclusion passes (a 10 bp human genome: 280 s of the call, serial).
    c = k_s // 2
    if coarsen and c >= 2 and arr32.size >= _COARSEN_MIN_BINS:
        n = arr32.size
        nb = n // c
        blocks = arr32[: nb * c].reshape(nb, c).mean(axis=1)
        if nb * c < n:
            blocks = np.append(blocks, np.float32(arr32[nb * c:].mean()))
        lam_c = local_lambda(
            blocks, bin_size * c, windows, genome_mean,
            exclude_fold=exclude_fold, exclude_iters=exclude_iters,
            exclude_smooth_bp=exclude_smooth_bp, robust_start=robust_start,
            # per-fragment signal, in units of the block MEAN
            unit=(unit / c) if unit else unit, noise_z=noise_z, info=info,
            coarsen=False)
        centres = (np.arange(lam_c.size, dtype=np.float64) + 0.5) * c - 0.5
        if nb * c < n:
            centres[-1] = (nb * c + n - 1) / 2.0
        return np.interp(np.arange(n, dtype=np.float64), centres, lam_c).astype(np.float32)
    judged = (uniform_filter1d(arr32, size=k_s, mode="nearest")
              if k_s > 1 else arr32)

    # The floor is meant to be "the background level", the guard that stops
    # peaks being called in anomalously quiet regions.  Three statistics were
    # measured for it on GSE285245 H3K27me3 (chr11, 100 bp; median bin 2.3,
    # lower quartile 0.6) and a per-replicate CUT&RUN CTCF bigWig at 10 bp:
    #
    # * the mean of the bins a 2x exclusion keeps: 7.4 on H3K27me3, so a 4x
    #   domain at Havcr2 measured 1.1x and was never called;
    # * a 5-SD sigma-clipped centre of the smoothed track: 8.2 on H3K27me3
    #   -- a broad mark's domains sit well inside 5 SD, so the clip removes
    #   nothing and returns the mean of background plus domains; and on the
    #   sparse CTCF track it slid into the zero mass (chrX: 0.58 against a
    #   rate of 6.3, with 113 Mb of chrX then "called").  Re-reading it on
    #   the kept bins at every exclusion pass made that a runaway (chr1:
    #   5.2 -> 0.001);
    # * the median of the smoothed track, read at the first smoothing scale
    #   at which the track is not zero-inflated (median > 0): 2-3 on
    #   H3K27me3, the rate on the sparse track.  That is the number the
    #   other two were being judged against, so it is the floor.
    #
    # The exclusion then iterates the window means against this fixed floor.
    #
    # "Not zero-inflated" needs more than a positive median on a sparse
    # library.  Measured on a 7.5M-fragment ATAC library (hg38, 10 bp; 2.4
    # fragments per kb): the median first turns positive at a smoothing
    # scale holding about one fragment, where it reads 0.2 against a rate of
    # 0.5 -- the median of a count with mean ~1 is not its mean.  With
    # ``unit`` known the scale is widened until the median window holds
    # ``_FLOOR_MIN_FRAGMENTS`` fragments, where median and mean agree to a
    # few percent.
    k_f, judged_f = k_s, judged
    cap = max(1, int(max(windows) // bin_size)) if windows else k_s
    if unit:
        cap = max(cap, int(np.ceil(4 * _FLOOR_MIN_FRAGMENTS * unit
                                   / max(float(genome_mean), 1e-9))))
        cap = max(1, min(cap, arr32.size // 4))
    need = _FLOOR_MIN_FRAGMENTS * unit if unit else 0.0
    med = _median(judged_f)
    while k_f < cap and (med <= 0.0 or med * k_f < need):
        k_f = min(k_f * 2, cap)
        judged_f = uniform_filter1d(arr32, size=k_f, mode="nearest")
        med = _median(judged_f)
    floor = max(min(floor, med), 1e-9)
    anchor = None
    if robust_start:
        anchor = _window_median(judged, bin_size, windows, floor)
        lam = anchor
    else:
        lam = _window_max(arr32, None, bin_size, windows, floor)
    keep = None
    for _ in range(max(int(exclude_iters), 1) * 2):
        thr = lam if anchor is None else np.minimum(lam, anchor)
        keep = (judged <= _signal_threshold(thr, exclude_fold, k_s, unit, noise_z)
                ).astype(np.float32)
        new = _window_max(arr32, keep, bin_size, windows, floor,
                          quiet_min_den=0.95 if robust_start else None)
        done = np.allclose(new, lam, rtol=0.01, atol=1e-6)
        lam = new
        if done:
            break
    if info is not None and keep is not None:
        info["held_out"] = info.get("held_out", 0) + int(keep.size - keep.sum())
        info["n"] = info.get("n", 0) + int(keep.size)
    return lam


#: Fragments the median smoothing window must hold before that median is read
#: as the background level (see the floor in ``local_lambda``).
_FLOOR_MIN_FRAGMENTS = 10.0

#: Fragments a local-background window must be expected to hold (see
#: ``effective_windows``).
_WINDOW_MIN_FRAGMENTS = 20.0


def _signal_threshold(level: np.ndarray, fold: float, k_s: int,
                      unit: Optional[float], noise_z: float = 3.0) -> np.ndarray:
    """Level above which the ``k_s``-bin smoothed track is held out as signal.

    ``fold`` x the background alone is a statement about enrichment only when
    the judged window holds enough fragments for noise not to reach it.  On a
    sparse library it does not: at 2.4 fragments per kb a 100 bp window
    expects 0.65 overlapping fragments, so ONE fragment is already twice the
    background, every covered stretch is "signal", and the background is
    fitted on the empty bins alone.  Measured on SNU449 ATAC (7-8M fragments,
    hg38, 10 bp bins, 1/5/10 kb windows): 15-18% of the genome held out, the
    fitted background at a median of 0.2 against a rate of 0.5, and the
    genome then reads as 1.7-2x "domains" 10-600 kb wide.

    So the threshold is the larger of the fold rule and the level plus
    ``noise_z`` standard deviations of the judged mean under Poisson
    fragments: a window of n expected fragments has sd sqrt(n) fragments,
    i.e. ``sqrt(unit * level / k_s)`` in track units.  With nine or more
    fragments in the window the fold rule is the larger one and nothing
    changes; below that, what is held out has to be more than one fragment
    of luck.
    """
    level = np.asarray(level, dtype=np.float32)
    thr = np.float32(fold) * level
    if unit and noise_z > 0:
        noisy = level + np.float32(noise_z) * np.sqrt(
            np.float32(unit / max(k_s, 1)) * np.maximum(level, 0))
        thr = np.maximum(thr, noisy)
    return thr


def effective_windows(windows: Sequence[int], fragments_per_bp: Optional[float],
                      bin_size: int,
                      min_fragments: float = _WINDOW_MIN_FRAGMENTS) -> Tuple[int, ...]:
    """The background windows widened to hold enough fragments to be a mean.

    A local window is a rate estimate, and its error is set by the fragments
    in it, not by its width in bp.  The 1 / 5 / 10 kb ladder is MACS's, sized
    for a control library tens of millions of reads deep; MACS itself drops
    the 1 kb window when there is no control.  At 2.4 fragments per kb a 1 kb
    window holds two or three fragments: its mean is 0, 0.4, 0.8 ... x the
    rate, and the max over windows takes whichever is highest, so the
    background is noise that follows the reads.  Every window is therefore
    raised to the width expected to hold ``min_fragments`` fragments at the
    library's own rate; windows that collapse onto each other (within 25%)
    are merged.  A deep library is unchanged.
    """
    wins = tuple(int(w) for w in windows)
    if not wins or not fragments_per_bp or fragments_per_bp <= 0:
        return wins
    w_min = int(np.ceil(min_fragments / fragments_per_bp / bin_size)) * bin_size
    out: list = []
    for w in sorted(max(w, w_min) for w in wins):
        if not out or w > 1.25 * out[-1]:
            out.append(int(w))
    return tuple(out)


def estimate_dispersion(
    counts: np.ndarray,
    mu: np.ndarray,
    n_quantiles: int = 40,
    background_quantile: float = 0.95,
    min_bins_per_group: int = 50,
) -> Tuple[float, float]:
    """Fit a parametric mean-dispersion trend on background bins.

    Model (DESeq2's parametric trend):  ``var = mu + alpha(mu) * mu^2``  with
    ``alpha(mu) = a0 + a1 / mu``.

    Fitting only on bins below ``background_quantile`` keeps genuine enrichment
    from inflating the dispersion estimate -- otherwise strong peaks make the
    background look noisy and the caller loses sensitivity.

    Returns ``(a0, a1)``.  Both are clipped to be non-negative.
    """
    counts = np.asarray(counts, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    finite = np.isfinite(counts) & np.isfinite(mu) & (mu > 0)
    counts, mu = counts[finite], mu[finite]
    if counts.size < min_bins_per_group * 2:
        return 0.1, 0.0

    cutoff = np.quantile(counts, background_quantile)
    keep = counts <= cutoff
    if keep.sum() < min_bins_per_group * 2:
        keep = np.ones_like(counts, dtype=bool)
    c, m = counts[keep], mu[keep]

    order = np.argsort(m)
    c, m = c[order], m[order]
    groups = np.array_split(np.arange(c.size),
                            max(2, min(n_quantiles, c.size // min_bins_per_group)))

    mus, alphas = [], []
    for g in groups:
        if g.size < min_bins_per_group:
            continue
        gm = float(m[g].mean())
        gv = float(c[g].var(ddof=1))
        if gm <= 0:
            continue
        a = (gv - gm) / (gm * gm)
        mus.append(gm)
        alphas.append(max(a, 0.0))

    if len(mus) < 3:
        gm = float(c.mean())
        gv = float(c.var(ddof=1))
        a = max((gv - gm) / (gm * gm), 0.0) if gm > 0 else 0.1
        return float(min(a, 10.0)), 0.0

    mus_a = np.array(mus)
    alphas_a = np.array(alphas)
    # alpha = a0 + a1 * (1/mu)  -- ordinary least squares in 1/mu
    X = np.column_stack([np.ones_like(mus_a), 1.0 / mus_a])
    try:
        coef, *_ = np.linalg.lstsq(X, alphas_a, rcond=None)
        a0, a1 = float(coef[0]), float(coef[1])
    except np.linalg.LinAlgError:  # pragma: no cover
        a0, a1 = float(np.median(alphas_a)), 0.0

    a0 = float(np.clip(a0, 0.0, 10.0))
    a1 = float(np.clip(a1, 0.0, 100.0))
    if a0 == 0.0 and a1 == 0.0:
        a0 = 1e-4  # keep it strictly NB; a pure Poisson fallback is a decision, not an accident
    return a0, a1


def check_window_scale(windows: Sequence[int], feature_width: int,
                       has_control: bool, exclude_fold: float = 0.0) -> Optional[str]:
    """Warn when local-background windows are too small for the target features.

    In no-control mode the background is estimated from the treatment itself, so
    a window narrower than the feature absorbs the feature: fold enrichment
    collapses toward 1 and broad domains silently disappear.  The rule of thumb
    is that the largest window should exceed the widest expected feature by at
    least 3x.

    Signal exclusion (``local_lambda(exclude_fold=...)``) is what actually
    defends against this, and it defends against clustered features too, which
    no window size can.  With it on, this stays as a scale sanity check rather
    than the only line of defence, so it is not raised.
    """
    if has_control or not windows or feature_width <= 0 or exclude_fold > 0:
        return None
    largest = max(windows)
    if largest < 3 * feature_width:
        return (
            f"local background windows (max {largest:,} bp) are not much larger "
            f"than the features being called ({feature_width:,} bp) and there is "
            f"no control: the background estimate will absorb the signal and "
            f"broad regions may be missed. Use windows >= {3 * feature_width:,} bp, "
            f"or supply a control."
        )
    return None


@dataclass
class BackgroundModel:
    """Per-bin expected background ``mu`` plus a mean-dispersion trend.

    Fitted per chromosome but with a genome-wide dispersion trend, so sparse
    chromosomes borrow strength from the whole genome.
    """

    mu: Dict[str, np.ndarray]
    a0: float
    a1: float
    genome_mean: float
    windows: Tuple[int, ...]
    used_control: bool
    covariate_coef: Optional[np.ndarray] = None
    autocorr_inflation: float = 1.0
    """Global AR(1) variance inflation, estimated on background bins."""
    meta: Dict = field(default_factory=dict)

    def alpha(self, mu: np.ndarray) -> np.ndarray:
        """Dispersion at a given mean, from the fitted trend."""
        mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-9)
        return np.maximum(self.a0 + self.a1 / mu, 1e-8)

    def fold_enrichment(self, cov: Coverage) -> Dict[str, np.ndarray]:
        return {
            c: (cov.bins[c] / np.maximum(self.mu[c], 1e-9)).astype(np.float32)
            for c in cov.bins
        }

    @classmethod
    def fit(
        cls,
        treatment: Coverage,
        control: Optional[Coverage] = None,
        windows: Sequence[int] = (1000, 5000, 10000),
        covariates: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        min_mu: float = 1e-3,
        exclude_fold: float = 2.0,
        exclude_iters: int = 3,
        exclude_smooth_bp: Optional[int] = None,
        robust_start: bool = False,
        depth_aware: bool = True,
        n_jobs: Optional[int] = None,
    ) -> "BackgroundModel":
        """Fit the background.

        Parameters
        ----------
        control : matched input/IgG.  Optional by design -- no-control mode is
            first class, because most CUT&Tag data has no matched input
            (DESIGN.md section 3).
        covariates : ``{chrom: {name: array}}`` aligned to the treatment bins,
            e.g. mappability or GC.  Fitted by log-linear regression on
            background bins and applied multiplicatively.
        depth_aware : size the windows and the signal exclusion by the
            fragments they hold (``effective_windows``, ``_signal_threshold``)
            when the track is a fragment pileup of known depth.  Off restores
            the windows and the plain fold rule exactly as given (ablation).
        n_jobs : worker processes, one chromosome each (``<= 0``: every CPU
            allocated to this process; None/1: serial).
        """
        gmean = treatment.mean_signal()
        ctrl = control.scaled_to(treatment) if control is not None else None

        # The track the background is read from: the control where there is
        # one, else the treatment.  Its depth decides how wide a window has
        # to be and what one fragment looks like in track units.
        requested = tuple(int(w) for w in windows)
        src = ctrl if ctrl is not None else treatment
        unit = frag_rate = None
        if depth_aware and src.is_count and src.n_fragments:
            tot, gbp = src.total_signal, float(src.effective_genome_size)
            if tot > 0 and gbp > 0:
                unit = tot / float(src.n_fragments)
                frag_rate = float(src.n_fragments) / gbp
        windows = effective_windows(requested, frag_rate, treatment.bin_size)
        # The exclusion's judging scale follows the windows the caller asked
        # for, not the widened ones: it is what keeps a held-out peak tight.
        if exclude_smooth_bp is None:
            exclude_smooth_bp = exclusion_smoothing_bp(treatment.bin_size, requested)

        # With a control, the local background comes from the CONTROL, never
        # from the treatment.  Deriving it from the treatment makes every
        # region its own background: for a domain wider than the largest
        # window the estimate converges on the domain's own level and the
        # fold enrichment collapses to ~1.  That is exactly how broad marks
        # get lost.  In no-control mode the treatment is the only estimate
        # available, so the windows MUST be substantially wider than the
        # features being called; presets for broad marks set them
        # accordingly, and ``check_window_scale`` warns when they do not.
        global _FIT_STATE
        order = list(treatment.bins)
        _FIT_STATE = dict(
            treatment=treatment, ctrl=ctrl, windows=windows, gmean=gmean,
            kw=dict(exclude_fold=exclude_fold, exclude_iters=exclude_iters,
                    exclude_smooth_bp=exclude_smooth_bp, robust_start=robust_start,
                    unit=unit),
            min_mu=min_mu)
        try:
            jobs = _fit_jobs(n_jobs, len(order))
            if jobs > 1:
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor

                todo = sorted(order, key=lambda c: -treatment.bins[c].size)
                with ProcessPoolExecutor(max_workers=jobs,
                                         mp_context=mp.get_context("fork")) as pool:
                    done = dict(zip(todo, pool.map(_fit_chrom, todo)))
            else:
                done = {c: _fit_chrom(c) for c in order}
        finally:
            _FIT_STATE = None
        mu: Dict[str, np.ndarray] = {c: done[c][0] for c in order}
        held = sum(done[c][1].get("held_out", 0) for c in order)
        held_n = sum(done[c][1].get("n", 0) for c in order)

        coef = None
        if covariates:
            mu, coef = _apply_covariates(treatment, mu, covariates)

        counts_all = np.concatenate([treatment.bins[c] for c in sorted(treatment.bins)])
        mu_all = np.concatenate([mu[c] for c in sorted(mu)])
        if robust_start and exclude_fold and exclude_fold > 0:
            # The dispersion is a property of the background, so with the
            # anchored estimate it is fitted on the bins the exclusion kept.
            # Fitted on everything below the 95th percentile, as otherwise,
            # the moderate enrichment the lower background now stands under
            # is read as noise: GSE285245 H3K27me3 a0 went 0.0 -> 1.29, the
            # HMM's weak and background states blurred, and a 23 kb block at
            # 3x (Havcr2) was not entered.
            ks = max(1, int(round((exclude_smooth_bp or exclusion_smoothing_bp(
                treatment.bin_size, windows)) / treatment.bin_size)))
            kept = []
            for c in sorted(treatment.bins):
                a = treatment.bins[c].astype(np.float32)
                j = uniform_filter1d(a, size=ks, mode="nearest") if ks > 1 else a
                kept.append(j <= _signal_threshold(mu[c][: j.size], exclude_fold, ks, unit))
            kept = np.concatenate(kept)
            if kept.sum() >= 10_000:
                counts_bg, mu_bg = counts_all[kept], mu_all[kept]
            else:
                counts_bg, mu_bg = counts_all, mu_all
        else:
            counts_bg, mu_bg = counts_all, mu_all
        # A few million bins settle the mean-dispersion trend and the lag-1
        # inflation; fitting them on every bin of a 10 bp mouse genome (263M)
        # cost minutes and ~12 GB of float64 temporaries and sort indices.
        # The trend uses a strided subsample (adjacency irrelevant); the
        # inflation needs contiguous bins, so it takes a central block.
        n_all = counts_all.size
        if counts_bg.size > _FIT_MAX_BINS:
            step = int(np.ceil(counts_bg.size / _FIT_MAX_BINS))
            c_fit, m_fit = counts_bg[::step], mu_bg[::step]
        else:
            c_fit, m_fit = counts_bg, mu_bg
        if n_all > _FIT_MAX_BINS:
            mid = (n_all - _FIT_MAX_BINS) // 2
            c_blk, m_blk = counts_all[mid: mid + _FIT_MAX_BINS], mu_all[mid: mid + _FIT_MAX_BINS]
        else:
            c_blk, m_blk = counts_all, mu_all
        a0, a1 = estimate_dispersion(c_fit, m_fit)
        inflation = autocorrelation_inflation(c_blk, m_blk)

        return cls(
            mu=mu,
            a0=a0,
            a1=a1,
            genome_mean=gmean,
            windows=tuple(windows),
            used_control=ctrl is not None,
            covariate_coef=coef,
            autocorr_inflation=inflation,
            meta={
                "n_bins": int(mu_all.size),
                "bin_size": treatment.bin_size,
                "is_count": treatment.is_count,
                "exclude_fold": float(exclude_fold),
                "robust_start": bool(robust_start),
                # What fraction of the genome the background fit set aside as
                # apparent signal.  A value near 0 means the exclusion did
                # nothing; a value near 1 means it has eaten the background and
                # the windows or the fold are wrong.
                # Counted on the mask the last exclusion pass actually used
                # (the smoothed track against its threshold), not on single
                # bins against ``exclude_fold`` x mu: on a sparse library one
                # fragment puts a bin over that, and the figure then reported
                # the covered fraction of the genome (11-18% on ATAC whose
                # open chromatin is 1-3%).
                "excluded_fraction": (float(held / held_n) if held_n else float(
                    (c_fit > exclude_fold * m_fit).mean()
                )) if exclude_fold > 0 else 0.0,
                "exclude_smooth_bp": int(
                    exclude_smooth_bp
                    or exclusion_smoothing_bp(treatment.bin_size, windows)
                ) if exclude_fold > 0 else 0,
                "requested_windows": list(requested),
                "windows": list(windows),
                "signal_per_fragment": unit,
                "fragments_per_kb": (frag_rate * 1000.0) if frag_rate else None,
            },
        )


#: Set in the parent right before the worker pool forks (copy-on-write
#: access to the coverage; only the fitted background comes back).
_FIT_STATE: Optional[dict] = None


def _fit_chrom(chrom: str):
    st = _FIT_STATE
    treatment, ctrl = st["treatment"], st["ctrl"]
    arr = treatment.bins[chrom]
    if ctrl is not None and chrom in ctrl.bins:
        src = ctrl.bins[chrom]
        if src.size != arr.size:  # defensive: differing chrom lengths
            src = np.resize(src, arr.size)
    else:
        src = arr
    info: dict = {}
    lam = local_lambda(src, treatment.bin_size, st["windows"], st["gmean"],
                       info=info, **st["kw"])
    return np.maximum(lam, st["min_mu"]).astype(np.float32), info


def _fit_jobs(n_jobs: Optional[int], n_chroms: int) -> int:
    import multiprocessing as mp
    import os

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


def _apply_covariates(
    treatment: Coverage,
    mu: Dict[str, np.ndarray],
    covariates: Dict[str, Dict[str, np.ndarray]],
    background_quantile: float = 0.90,
):
    """Log-linear covariate correction fitted on background bins only."""
    names = sorted({n for d in covariates.values() for n in d})
    if not names:
        return mu, None

    ys, Xs = [], []
    for chrom, arr in treatment.bins.items():
        cov_c = covariates.get(chrom)
        if not cov_c:
            continue
        cols = [np.asarray(cov_c[n], dtype=np.float64)[: arr.size] for n in names if n in cov_c]
        if len(cols) != len(names):
            continue
        cut = np.quantile(arr, background_quantile)
        keep = (arr <= cut) & (arr > 0)
        if keep.sum() < 100:
            continue
        ys.append(np.log(arr[keep] / np.maximum(mu[chrom][keep], 1e-9)))
        Xs.append(np.column_stack([c[keep] for c in cols]))

    if not ys:
        return mu, None

    y = np.concatenate(ys)
    X = np.vstack(Xs)
    X = np.column_stack([np.ones(X.shape[0]), X])
    good = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if good.sum() < 100:
        return mu, None
    try:
        coef, *_ = np.linalg.lstsq(X[good], y[good], rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover
        return mu, None

    out = {}
    for chrom, m in mu.items():
        cov_c = covariates.get(chrom)
        if not cov_c or any(n not in cov_c for n in names):
            out[chrom] = m
            continue
        cols = [np.asarray(cov_c[n], dtype=np.float64)[: m.size] for n in names]
        Xc = np.column_stack([np.ones(m.size)] + cols)
        adj = np.exp(np.clip(Xc @ coef, -5, 5))
        out[chrom] = np.maximum(m * adj, 1e-3).astype(np.float32)
    return out, coef
