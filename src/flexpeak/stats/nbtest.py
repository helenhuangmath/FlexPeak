"""Negative binomial tail probabilities and FDR control."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats

__all__ = ["nb_neglog10_sf", "poisson_neglog10_sf", "benjamini_hochberg",
           "qvalue_from_neglog10p", "aggregate_nb_params",
           "autocorrelation_inflation", "autocorrelation_length_bp"]

_LOG10 = np.log(10.0)


def _nb_params(mu: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (mean, dispersion) to scipy's (n, p) parameterisation.

    var = mu + alpha * mu^2  =>  n = 1/alpha,  p = n / (n + mu)
    """
    alpha = np.maximum(alpha, 1e-10)
    n = 1.0 / alpha
    p = n / (n + mu)
    return n, np.clip(p, 1e-12, 1.0 - 1e-15)


def nb_neglog10_sf(counts, mu, alpha) -> np.ndarray:
    """-log10 P(X >= counts) under NB(mu, alpha), vectorised.

    Uses the survival function at ``counts - 1`` so the test is inclusive of the
    observed value, matching the convention MACS uses for its Poisson test.
    """
    counts = np.asarray(counts, dtype=np.float64)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-12)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape != mu.shape:
        alpha = np.broadcast_to(alpha, mu.shape)

    k = np.maximum(np.floor(counts) - 1.0, -1.0)
    n, p = _nb_params(mu, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        logsf = stats.nbinom.logsf(k, n, p)
    out = -logsf / _LOG10
    # k == -1 means "P(X >= 0)" == 1 exactly; guard the numerical edge.
    out[k < 0] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=323.0, neginf=0.0)


def poisson_neglog10_sf(counts, mu) -> np.ndarray:
    """-log10 P(X >= counts) under Poisson(mu).  Present for the §9.5 ablation."""
    counts = np.asarray(counts, dtype=np.float64)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-12)
    k = np.maximum(np.floor(counts) - 1.0, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logsf = stats.poisson.logsf(k, mu)
    out = -logsf / _LOG10
    out[k < 0] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=323.0, neginf=0.0)


def autocorrelation_inflation(counts, mu, background_quantile: float = 0.90,
                              max_rho: float = 0.9) -> float:
    """Variance inflation from coverage autocorrelation, estimated on BACKGROUND.

    Adjacent bins are correlated, so N bins carry fewer than N independent
    observations and an uncorrected aggregate test is anticonservative.  The
    AR(1) effective-sample-size factor is ``(1+rho)/(1-rho)``.

    Estimating ``rho`` from the region under test would be backwards twice over:
    inside a real peak the smoothness *is* the signal, so genuine peaks would be
    penalised hardest; and deeper libraries are smoother, so the penalty would
    grow with depth and make more data produce fewer calls.  It is therefore
    estimated once, genome-wide, from background bins only.
    """
    c = np.asarray(counts, dtype=np.float64).ravel()
    m = np.maximum(np.asarray(mu, dtype=np.float64).ravel(), 1e-9)
    if c.size < 100:
        return 1.0
    cut = np.quantile(c, background_quantile)
    keep = c <= cut
    if keep.sum() < 100:
        return 1.0
    # Work on residuals so large-scale background structure is not mistaken
    # for short-range noise correlation.
    r = (c[keep] - m[keep]) / np.sqrt(m[keep])
    r = r - r.mean()
    denom = float((r * r).sum())
    if denom <= 0:
        return 1.0
    rho = float(np.dot(r[:-1], r[1:]) / denom)
    rho = min(max(rho, 0.0), max_rho)
    return (1.0 + rho) / (1.0 - rho)


def autocorrelation_length_bp(signal, bin_size: int, max_lag_bp: int = 200_000,
                              max_bins: int = 4_000_000) -> int:
    """Lag at which the signal's autocorrelation first falls below 1/e, in bp.

    This is the mark's own scale, read off the sample rather than declared: a
    transcription factor gives a few hundred bp, H3K27me3 gives kilobases.  QC
    already reports it as a punctate-vs-domain diagnostic; the segmentation uses
    it as the floor on the HMM's enriched dwell time, so the model is not free
    to shred a domain into fragments an order of magnitude below the scale the
    data plainly has.

    Computed by FFT: the direct lag-by-lag loop is O(n * max_lag), which on a
    chromosome at 25 bp bins is billions of operations for a number used once.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size > max_bins:  # a strided sample is plenty for a scale estimate
        x = x[:: int(np.ceil(x.size / max_bins))]
    if x.size < 32:
        return int(bin_size)
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return int(bin_size)

    max_lag = int(min(max(1, max_lag_bp // bin_size), x.size // 2))
    n_fft = 1 << int(np.ceil(np.log2(x.size + max_lag + 1)))
    f = np.fft.rfft(x, n_fft)
    ac = np.fft.irfft(f * np.conjugate(f), n_fft)[: max_lag + 1] / denom
    below = np.flatnonzero(ac[1:] < 1.0 / np.e)
    lag = int(below[0]) + 1 if below.size else max_lag
    return int(max(lag, 1) * bin_size)


def aggregate_nb_params(mu_bins, alpha_bins, counts=None, max_rho: float = 0.95,
                        inflation: Optional[float] = None):
    """Dispersion for the SUM of several bins.  Returns ``(mu_total, alpha_eff)``.

    Testing a whole region means testing a sum, and the per-bin dispersion is
    not the dispersion of that sum.  For independent bins,

        Var(sum) = sum_i (mu_i + alpha_i * mu_i^2)

    so the effective dispersion of the aggregate is

        alpha_eff = (Var(sum) - mu_total) / mu_total^2

    which for homogeneous bins is ``alpha_bin / N``.  Applying the per-bin
    alpha directly to ``mu_total`` instead inflates the variance by a factor of
    N, and therefore destroys significance in proportion to region width --
    narrow peaks survive it, broad domains do not.  That asymmetry is precisely
    the failure mode FlexPeak exists to fix, so getting this right matters more
    here than in a narrow-only caller.

    Bins are not truly independent: real coverage is autocorrelated, so treating
    N bins as N independent observations is anticonservative.  Pass ``inflation``
    from :func:`autocorrelation_inflation`, which estimates the correction once
    from background bins.  The ``counts`` argument estimates it from the region
    itself instead; that is retained only for diagnostics, because it penalises
    real peaks hardest and grows with sequencing depth.
    """
    mu = np.asarray(mu_bins, dtype=np.float64).ravel()
    alpha = np.broadcast_to(np.asarray(alpha_bins, dtype=np.float64), mu.shape)
    mu_total = float(mu.sum())
    if mu_total <= 0:
        return 1e-9, 1.0

    var_indep = mu_total + float((alpha * mu * mu).sum())

    if inflation is None:
        inflation = 1.0
    if inflation == 1.0 and counts is not None:
        c = np.asarray(counts, dtype=np.float64).ravel()
        if c.size >= 4:
            d = c - c.mean()
            denom = float((d * d).sum())
            if denom > 0:
                rho = float(np.dot(d[:-1], d[1:]) / denom)
                rho = min(max(rho, 0.0), max_rho)
                inflation = (1.0 + rho) / (1.0 - rho)

    var = inflation * var_indep
    alpha_eff = (var - mu_total) / (mu_total * mu_total)
    return mu_total, float(max(alpha_eff, 1e-10))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, monotonicity enforced."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def qvalue_from_neglog10p(neglog10p: np.ndarray) -> np.ndarray:
    """BH on the -log10 scale, staying in log space to avoid underflow.

    Very significant bins have p far below float64's smallest normal, so
    exponentiating first would collapse them all to zero and destroy the
    ordering that BH depends on.
    """
    s = np.asarray(neglog10p, dtype=np.float64)
    n = s.size
    if n == 0:
        return s
    order = np.argsort(-s)  # most significant first
    ranks = np.arange(1, n + 1)
    # log10(q) = log10(p) + log10(n) - log10(rank)
    logq = -s[order] + np.log10(n) - np.log10(ranks)
    logq = np.minimum.accumulate(logq[::-1])[::-1]  # monotone in the same direction as BH
    # Deliberately no upper clip: the whole point of staying in log space is to
    # keep the ordering of hits whose p-values are below float64's smallest
    # normal.  Capping the score here would tie them all together again.
    out_sorted = np.maximum(-logq, 0.0)
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out
