"""Negative binomial background model.

DESIGN.md section 3.2.  MACS2's local Poisson lambda assumes mean == variance.
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
           "check_window_scale", "exclusion_smoothing_bp"]


#: Bins used to fit the dispersion trend and the autocorrelation inflation.
_FIT_MAX_BINS = 4_000_000

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
) -> np.ndarray:
    """MACS-style local background: max over several window scales.

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

    Measured on the CUT&RUN samples in ``testing/``: median fold enrichment in
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
    k_f, judged_f = k_s, judged
    cap = max(1, int(max(windows) // bin_size)) if windows else k_s
    while float(np.median(judged_f)) <= 0.0 and k_f < cap:
        k_f = min(k_f * 2, cap)
        judged_f = uniform_filter1d(arr32, size=k_f, mode="nearest")
    floor = max(min(floor, float(np.median(judged_f))), 1e-9)
    anchor = None
    if robust_start:
        anchor = _window_median(judged, bin_size, windows, floor)
        lam = anchor
    else:
        lam = _window_max(arr32, None, bin_size, windows, floor)
    for _ in range(max(int(exclude_iters), 1) * 2):
        thr = lam if anchor is None else np.minimum(lam, anchor)
        keep = (judged <= exclude_fold * thr).astype(np.float32)
        new = _window_max(arr32, keep, bin_size, windows, floor,
                          quiet_min_den=0.95 if robust_start else None)
        done = np.allclose(new, lam, rtol=0.01, atol=1e-6)
        lam = new
        if done:
            break
    return lam


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
    ) -> "BackgroundModel":
        """Fit the background.

        Parameters
        ----------
        control : matched input/IgG.  Optional by design -- no-control mode is
            first class, because most CUT&Tag data has no matched input
            (DESIGN.md section 3.2).
        covariates : ``{chrom: {name: array}}`` aligned to the treatment bins,
            e.g. mappability or GC.  Fitted by log-linear regression on
            background bins and applied multiplicatively.
        """
        gmean = treatment.mean_signal()
        ctrl = control.scaled_to(treatment) if control is not None else None

        mu: Dict[str, np.ndarray] = {}
        for chrom, arr in treatment.bins.items():
            if ctrl is not None and chrom in ctrl.bins:
                # With a control, the local background comes from the CONTROL,
                # never from the treatment.  Deriving it from the treatment
                # makes every region its own background: for a domain wider
                # than the largest window the estimate converges on the domain's
                # own level and the fold enrichment collapses to ~1.  That is
                # exactly how broad marks get lost.
                c = ctrl.bins[chrom]
                if c.size != arr.size:  # defensive: differing chrom lengths
                    c = np.resize(c, arr.size)
                lam = local_lambda(c, ctrl.bin_size, windows, gmean,
                                   exclude_fold=exclude_fold,
                                   exclude_iters=exclude_iters,
                                   exclude_smooth_bp=exclude_smooth_bp,
                                   robust_start=robust_start)
            else:
                # No-control mode.  The treatment is the only background
                # estimate available, so the windows MUST be substantially
                # wider than the features being called or the same collapse
                # occurs.  Presets for broad marks set windows accordingly, and
                # ``check_window_scale`` warns when they do not.
                lam = local_lambda(arr, treatment.bin_size, windows, gmean,
                                   exclude_fold=exclude_fold,
                                   exclude_iters=exclude_iters,
                                   exclude_smooth_bp=exclude_smooth_bp,
                                   robust_start=robust_start)
            mu[chrom] = np.maximum(lam, min_mu).astype(np.float32)

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
                kept.append(j <= exclude_fold * mu[c][: j.size])
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
                "excluded_fraction": float(
                    (c_fit > exclude_fold * m_fit).mean()
                ) if exclude_fold > 0 else 0.0,
                "exclude_smooth_bp": int(
                    exclude_smooth_bp
                    or exclusion_smoothing_bp(treatment.bin_size, windows)
                ) if exclude_fold > 0 else 0,
            },
        )


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
