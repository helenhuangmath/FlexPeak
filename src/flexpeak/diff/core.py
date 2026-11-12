"""Differential regions between two conditions, THOR-style.

THOR (Allhoff et al., NAR 2016) segments the *pair* of signals with a hidden
Markov model whose states are "no difference", "higher in condition 1" and
"higher in condition 2", so a differential region is found as a run of bins
that agree, not as a thresholded window.  FlexPeak keeps that structure and
changes what the states emit.

**Emissions are conditional on the total.**  Given the normalised counts of
the two conditions in a bin, ``a`` and ``b``, the information about *which*
condition is higher is in the split ``a / (a + b)``, not in the depth.  So each
state emits ``a`` from a beta-binomial on ``n = a + b`` with a state-specific
proportion: 1/2 for "same", above it for "gain", below it for "loss".  That
makes the model free of any background model -- a region that is deep in both
conditions and a region that is shallow in both both sit at 1/2 -- and it
makes an empty bin uninformative (every state emits it with probability 1), so
the chain carries its state across a dip instead of being pushed to "same".

**Overdispersion comes from the replicates.**  With two or more replicates per
condition the split between replicates of the *same* condition should sit at
their depth ratio; how far it scatters beyond binomial is the biological
variability, and it is read off directly as the beta-binomial ``rho``.  With
single samples ``rho`` is re-estimated inside the EM from the bins the chain
assigns to "same", which is the no-replicate approximation THOR also makes.

**Normalisation is a choice, stated.**  Scaling by library size (``total``)
calls a global gain of signal in one condition as loss everywhere else.  The
default matches the sigma-clipped *background* level of every sample (the
quick scan's estimate), so a sample that simply has more signal is not
normalised into looking like it has less background.  ``tmm`` is the edgeR
trimmed mean of M-values over informative bins, which is what THOR uses.

The region test is the same beta-binomial on the region's aggregate counts,
two-sided, BH-corrected over as many region-sized tiles as fit in the genome
(the chain scanned the whole genome to find them).  Each replicate's counts are
reported so a call carried by one replicate is visible as such.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.special import betaln, gammaln

from ..segment._kernels import forward_backward, viterbi
from ..signal.coverage import Coverage
from ..stats.nbtest import autocorrelation_length_bp, qvalue_from_neglog10p

__all__ = ["DiffParams", "DiffRegion", "DiffResult", "call_differential",
           "differential_from_coverage", "normalise", "estimate_rho"]


# -- parameters --------------------------------------------------------------

@dataclass
class DiffParams:
    bin_size: Optional[int] = None
    """HMM bin.  None: from the quick scan (narrow 50, mixed 100, broad
    >= 200 bp), else 100 bp."""
    qvalue: float = 0.05
    min_fold: float = 1.5
    """Minimum |A/B| fold change for a region, and the floor on the
    separation of the gain/loss states from 1/2 in the EM."""
    min_count: float = 10.0
    """Minimum normalised counts summed over both conditions in a region."""
    min_width: Optional[int] = None
    """None: two bins."""
    posterior_cutoff: float = 0.5
    norm: str = "background"  # background | tmm | total
    within_peaks: bool = False
    max_gap: Optional[int] = None
    """Connect same-direction regions closer than this when the gap leans the
    same way.  None: the scan's gap scale, else two bins."""
    gap_fold_fraction: float = 0.5
    stay_same: float = 0.99
    stay_diff: float = 0.95
    dwell_from_autocorrelation: bool = True
    max_dwell: int = 100_000
    max_iter: int = 15
    tol: float = 1e-4
    rho: Optional[float] = None
    """Fix the beta-binomial overdispersion instead of estimating it."""
    rho_floor_no_replicates: float = 0.02
    """Without replicates the EM re-estimates rho from the bins it assigns to
    "same", and that is degenerate: every bin that deviates is explained by
    a differential state instead, so rho collapses (measured on pooled
    H3K27me3 bigWigs: rho 0.002, 390,588 "differential" regions).  This floor
    is the edgeR no-replicate convention -- assume a biological coefficient of
    variation of ~0.2 -- expressed as beta-binomial overdispersion.  Ignored
    when either condition has replicates or ``rho`` is fixed."""
    region_rho: Optional[float] = None
    """Overdispersion for the *region-level* test.  Between-sample variation
    is correlated across the bins of a region -- a whole peak is higher in
    one replicate -- so summing independent per-bin beta-binomials
    understates the variance of a region sum in proportion to its width, and
    any long run at a modest fold comes out "significant" (measured on pooled
    H3K4me3 bigWigs: 543,181 regions at q <= 0.05).  None: estimated from
    the replicate scatter of the candidate regions' sums within each
    condition (the edgeR/DESeq2 approach, at region scale) when replicates
    exist, else ``rho_floor_no_replicates`` applied at region scale."""
    pseudo_depth: float = 0.0
    """For normalised (bigWig) input, which the loader rescales to arbitrary
    pseudo-counts.  0 (default): recover fragment-scale counts from the
    track's own noise -- on sigma-clipped background bins a Poisson-like
    track has variance/mean equal to its units per fragment, so dividing by
    that ratio puts the sample on the scale the beta-binomial assumes
    (measured on pooled GEO H3K4me3 at 100 bp: a fixed depth of 10 per bin
    overstated precision ~10x and passed 98,680 regions).  > 0: rescale every
    sample to this mean per HMM bin instead."""
    control_smooth_bp: int = 1_000
    min_fold_over_control: float = 1.5
    require_enrichment: bool = True
    """Without an input, gate candidates on FlexPeak's own no-control local
    background of the pooled signal (max over 10 kb and 50 kb windows with
    apparent signal held out): a region must reach ``min_fold_over_control``
    over it in at least one condition.  Measured on CUT&RUN CTCF, 4 vs 4:
    without the gate the chain tracked regional background differences
    between the pooled conditions and returned 23,099 regions of median
    10 kb with both states pinned at the 1.5x floor."""
    enrichment_windows: Tuple[int, ...] = (100_000, 500_000)
    """Windows for that local background.  They must dwarf the widest
    domain being tested or the domain becomes its own background (measured:
    with 10/50 kb windows a 40 kb domain at 2.5x came out at 1.4x over
    "background" and was dropped; at 100/500 kb it is excluded from the
    estimate and passes at 2.5x).  ``call_differential`` widens them further
    to the scan's windows for a broad mark."""
    """With an input/IgG, a region must exceed this fold over the scaled,
    smoothed control in at least one condition.  The control is never
    subtracted from the counts: on a split-conditional model that only
    manufactures extreme splits wherever the subtraction floors one condition
    at zero (measured: 47 -> 4,679 calls, F1 98% -> 15%)."""
    fit_max_bins: int = 4_000_000
    """The EM fits on at most this many bins (the largest chromosomes); every
    chromosome is then decoded with the fitted parameters."""
    pseudocount: float = 0.5
    min_depth_per_bin: float = 10.0
    """After normalisation the two conditions together must average at least
    this many fragments per HMM bin; if not, every sample is coarsened by the
    smallest integer factor that gets there (at most ``max_depth_rebin``).
    A per-bin split of 0.4 fragments is noise (measured on CUT&RUN CTCF at
    50 bp: 2.7 million HMM candidates), and a 300 bp peak is still resolved
    at 200-500 bp bins.  0 disables."""
    max_depth_rebin: int = 20
    fdr_scope: str = "candidates"
    """Multiplicity for the BH correction of the region test.  ``candidates``
    (default): over the regions the HMM returned, as count-based differential
    testing does over the regions it counts -- the HMM posterior is the
    genome-wide screen and the region test with region-scale overdispersion
    is the guard.  ``genome``: over genome-sized tiles as the peak caller
    does; measured with two replicates on simulated data it removed every
    true call (0 of 38) while ``candidates`` kept 38 at 93% precision and 0
    on the null dataset."""

    def to_dict(self) -> Dict:
        return asdict(self)

    def replace(self, **kw) -> "DiffParams":
        d = asdict(self)
        d.update(kw)
        return DiffParams(**d)


# -- output contract ---------------------------------------------------------

@dataclass
class DiffRegion:
    chrom: str
    start: int
    end: int
    direction: str
    """'gain' (higher in A) or 'loss' (higher in B)."""
    log2fc: float
    counts_a: List[float]
    counts_b: List[float]
    mean_posterior: float
    neg_log10_p: float = 0.0
    neg_log10_q: float = 0.0
    name: str = ""

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def total(self) -> float:
        return float(sum(self.counts_a) + sum(self.counts_b))


@dataclass
class DiffResult:
    regions: List[DiffRegion]
    params: Dict = field(default_factory=dict)
    chrom_sizes: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict = field(default_factory=dict)

    @property
    def n_gain(self) -> int:
        return sum(1 for r in self.regions if r.direction == "gain")

    @property
    def n_loss(self) -> int:
        return sum(1 for r in self.regions if r.direction == "loss")

    def __len__(self) -> int:
        return len(self.regions)

    def intervals(self, direction: Optional[str] = None) -> List[Tuple[str, int, int]]:
        return [(r.chrom, r.start, r.end) for r in self.regions
                if direction is None or r.direction == direction]

    def _header(self, fmt: str) -> str:
        return "# " + json.dumps({"format": fmt, "params": self.params},
                                 sort_keys=True, default=str) + "\n"

    def to_narrowpeak(self, path=None) -> str:
        """ENCODE narrowPeak; strand + = higher in A (gain), - = higher in B."""
        lines = [self._header("flexpeak-diff-narrowPeak")]
        for r in self.regions:
            score = int(min(1000, round(10 * r.neg_log10_q)))
            strand = "+" if r.direction == "gain" else "-"
            lines.append(f"{r.chrom}\t{r.start}\t{r.end}\t{r.name}\t{score}\t{strand}\t"
                         f"{r.log2fc:.4f}\t{r.neg_log10_p:.4f}\t{r.neg_log10_q:.4f}\t-1\n")
        return self._emit("".join(lines), path)

    def to_tsv(self, path=None) -> str:
        cols = ["chrom", "start", "end", "name", "direction", "width", "log2fc",
                "counts_a", "counts_b", "mean_posterior", "neg_log10_p", "neg_log10_q"]
        lines = [self._header("flexpeak-diff-tsv"), "\t".join(cols) + "\n"]
        for r in self.regions:
            lines.append("\t".join([
                r.chrom, str(r.start), str(r.end), r.name, r.direction, str(r.width),
                f"{r.log2fc:.4f}", ";".join(f"{c:.1f}" for c in r.counts_a),
                ";".join(f"{c:.1f}" for c in r.counts_b), f"{r.mean_posterior:.3f}",
                f"{r.neg_log10_p:.4f}", f"{r.neg_log10_q:.4f}"]) + "\n")
        return self._emit("".join(lines), path)

    def to_bed(self, direction: str, path=None) -> str:
        lines = [f"{r.chrom}\t{r.start}\t{r.end}\t{r.name}\t{r.log2fc:.3f}\t"
                 f"{'+' if r.direction == 'gain' else '-'}\n"
                 for r in self.regions if r.direction == direction]
        return self._emit("".join(lines), path)

    @staticmethod
    def _emit(text: str, path) -> str:
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def write_all(self, prefix: str) -> Dict[str, str]:
        paths = {
            "narrowPeak": f"{prefix}_diffpeaks.narrowPeak",
            "tsv": f"{prefix}_diffpeaks.tsv",
            "gain": f"{prefix}_gain.bed",
            "loss": f"{prefix}_loss.bed",
        }
        self.to_narrowpeak(paths["narrowPeak"])
        self.to_tsv(paths["tsv"])
        self.to_bed("gain", paths["gain"])
        self.to_bed("loss", paths["loss"])
        return paths

    def report(self) -> str:
        d = self.diagnostics
        w = [r.width for r in self.regions]
        lines = [
            "FlexPeak differential report",
            "=" * 60,
            f"  bin size            {d.get('bin_size')} bp"
            + (f"   (scan: {d['scan_class']} mark, scale {d['scan_scale_bp']:,} bp)"
               if d.get("scan_class") else "")
            + (f"\n  depth rebin         {d['rebin_note']}" if d.get("rebin_note") else ""),
            f"  normalisation       {d.get('norm')}: factors A {d.get('norm_a')}, B {d.get('norm_b')}",
            f"  overdispersion rho  {d.get('rho', 0):.4f} per bin ({d.get('rho_source')}); "
            f"{d.get('region_rho', 0):.4f} per region ({d.get('region_rho_source')})",
            f"  input               {'fragment counts' if d.get('is_count', True) else 'normalised signal (bigWig): p-values approximate; ' + str(d.get('scale_note'))}",
            f"  state proportions   gain {d.get('p_gain', 0):.3f} "
            f"(fold {d.get('fold_gain', 0):.2f}), loss {d.get('p_loss', 0):.3f} "
            f"(fold {d.get('fold_loss', 0):.2f})",
            f"  dwell floor         {d.get('dwell_floor_bp', 0):,} bp; fitted dwell "
            f"{d.get('fitted_dwell_bp')} bp",
            f"  EM                  {d.get('n_iter')} iterations, "
            f"{'converged' if d.get('converged') else 'not converged'}, "
            f"fit on {d.get('fit_bins', 0):,} bins",
            f"  enrichment gate     {d.get('enrichment_gate')}: {d.get('control_dropped', 0):,} candidates dropped",
            f"  candidates          {d.get('n_candidates', 0):,} -> "
            f"{len(self.regions):,} at q <= {self.params.get('call', {}).get('qvalue')}",
            f"  regions             {len(self.regions):,} "
            f"({self.n_gain:,} gain in A, {self.n_loss:,} loss in A)",
            f"  median width        {np.median(w) if w else 0:,.0f} bp",
            f"  elapsed             {d.get('elapsed_s', 0):.1f} s",
        ]
        return "\n".join(lines)


# -- normalisation -----------------------------------------------------------

def _concat(cov: Coverage) -> np.ndarray:
    return cov.concat()


def normalise(samples: Sequence[Coverage], method: str = "background",
              n_sd: float = 5.0) -> List[float]:
    """Per-sample scale factors that bring the samples onto a common level.

    ``total``: library size.  ``background``: the sigma-clipped background mean
    (so enrichment differences are not mistaken for depth differences).
    ``tmm``: edgeR's trimmed mean of M-values against the sample closest to
    the geometric-mean library, over bins with counts in every sample.
    """
    from ..scan import sigma_clip

    arrays = [_concat(s) for s in samples]
    if method == "total":
        tot = np.array([float(a.sum()) for a in arrays])
        ref = tot.mean()
        return [float(ref / t) if t > 0 else 1.0 for t in tot]
    if method == "background":
        bg = np.array([sigma_clip(a, n_sd)[0] for a in arrays])
        ref = bg.mean()
        return [float(ref / b) if b > 0 else 1.0 for b in bg]
    if method == "tmm":
        tot = np.array([float(a.sum()) for a in arrays])
        lib = tot / tot.mean()
        ref_idx = int(np.argmin(np.abs(np.log(lib))))
        ref = arrays[ref_idx]
        out = []
        for i, a in enumerate(arrays):
            if i == ref_idx:
                out.append(1.0)
                continue
            ok = (a > 0) & (ref > 0)
            if ok.sum() < 100:
                out.append(float(tot[ref_idx] / tot[i]) if tot[i] > 0 else 1.0)
                continue
            x, y = a[ok] / tot[i], ref[ok] / tot[ref_idx]
            m = np.log2(y / x)
            av = 0.5 * (np.log2(x) + np.log2(y))
            lo_m, hi_m = np.quantile(m, [0.30, 0.70])
            lo_a, hi_a = np.quantile(av, [0.05, 0.95])
            keep = (m >= lo_m) & (m <= hi_m) & (av >= lo_a) & (av <= hi_a)
            if keep.sum() < 50:
                keep = np.ones_like(m, dtype=bool)
            w = 1.0 / ((1.0 - x[keep]) / (a[ok][keep] + 1e-9)
                       + (1.0 - y[keep]) / (ref[ok][keep] + 1e-9))
            tmm = float(np.sum(w * m[keep]) / np.sum(w))
            out.append(float(2.0 ** tmm * tot[ref_idx] / tot[i]))
        return out
    raise ValueError(f"unknown normalisation {method!r}")


# -- beta-binomial -----------------------------------------------------------

def _bb_logpmf(a: np.ndarray, n: np.ndarray, p: float, rho: float) -> np.ndarray:
    """log BetaBinomial(a | n, p, rho); Binomial when rho -> 0.  n == 0 -> 0."""
    a = np.asarray(a, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    comb = gammaln(n + 1) - gammaln(a + 1) - gammaln(n - a + 1)
    if rho <= 1e-9:
        out = comb + a * np.log(p) + (n - a) * np.log1p(-p)
    else:
        al = p * (1.0 - rho) / rho
        be = (1.0 - p) * (1.0 - rho) / rho
        out = comb + betaln(a + al, n - a + be) - betaln(al, be)
    out = np.where(n > 0, out, 0.0)
    return np.nan_to_num(out, nan=-745.0, neginf=-745.0)


def _bb_sf_two_sided(a: float, n: float, p: float, rho: float) -> float:
    """Two-sided tail probability of the split ``a`` of ``n`` under (p, rho)."""
    if n <= 0:
        return 1.0
    n_i = int(round(n))
    a_i = int(min(max(round(a), 0), n_i))
    if rho <= 1e-9:
        lo = stats.binom.cdf(a_i, n_i, p)
        hi = stats.binom.sf(a_i - 1, n_i, p)
    else:
        al = p * (1.0 - rho) / rho
        be = (1.0 - p) * (1.0 - rho) / rho
        lo = stats.betabinom.cdf(a_i, n_i, al, be)
        hi = stats.betabinom.sf(a_i - 1, n_i, al, be)
    return float(min(1.0, 2.0 * min(lo, hi)))


def _aggregate_rho(n_bins: np.ndarray, rho: float, p: float = 0.5) -> float:
    """Effective beta-binomial rho of a sum of independent bins."""
    n_bins = np.asarray(n_bins, dtype=np.float64)
    N = float(n_bins.sum())
    if N <= 1:
        return rho
    pq = p * (1.0 - p)
    var = float(np.sum(n_bins * pq * (1.0 + (n_bins - 1.0) * rho)))
    return float(np.clip((var / (N * pq) - 1.0) / (N - 1.0), 0.0, 0.5))


def estimate_rho(reps: Sequence[np.ndarray], min_total: float = 20.0,
                 max_bins: int = 2_000_000, min_bins: int = 100) -> Optional[float]:
    """Beta-binomial overdispersion from replicate-to-replicate scatter.

    For each pair of replicates of one condition, the split of their summed
    count should be binomial at their depth ratio; the excess variance,
    ``(var - n p q) / (n p q (n - 1))``, is ``rho``.  Method of moments over
    bins with enough counts, pooled over pairs; None if no pair exists.
    """
    vals = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            x, y = reps[i].astype(np.float64), reps[j].astype(np.float64)
            n = np.rint(x + y)
            ok = n >= min_total
            if ok.sum() < min_bins:
                continue
            idx = np.flatnonzero(ok)
            if idx.size > max_bins:
                idx = idx[:: int(np.ceil(idx.size / max_bins))]
            xa, na = np.rint(x[idx]), n[idx]
            p = float(xa.sum() / na.sum())
            q = 1.0 - p
            resid = (xa - na * p) ** 2 - na * p * q
            denom = na * p * q * (na - 1.0)
            r = float(resid.sum() / max(denom.sum(), 1e-9))
            vals.append(float(np.clip(r, 0.0, 0.5)))
    return float(np.median(vals)) if vals else None


# -- the HMM -----------------------------------------------------------------

def _transitions(stay_same: float, stay_diff: float, floor: float) -> np.ndarray:
    A = np.zeros((3, 3))
    for i in range(3):
        stay = stay_same if i == 0 else max(stay_diff, floor)
        A[i, :] = (1.0 - stay) / 2.0
        A[i, i] = stay
    return A


def _apply_floor(A: np.ndarray, floor: float) -> np.ndarray:
    A = A.copy()
    for i in (1, 2):
        if A[i, i] < floor:
            off = A[i].sum() - A[i, i]
            if off > 0:
                A[i] *= (1.0 - floor) / off
            A[i, i] = floor
    return A


@dataclass
class _Fit:
    p_gain: float
    p_loss: float
    rho: float
    A: np.ndarray
    pi: np.ndarray
    n_iter: int
    converged: bool


def _fit_hmm(a: np.ndarray, n: np.ndarray, params: DiffParams, rho0: Optional[float],
             floor: float) -> _Fit:
    """Baum-Welch on the state proportions, rho (if free) and transitions."""
    f = max(params.min_fold, 1.01)
    p_min = f / (1.0 + f)                   # gain state at least min_fold
    p_gain, p_loss = max(p_min, 0.7), min(1.0 - p_min, 0.3)
    rho = rho0 if rho0 is not None else 0.02
    rho_free = params.rho is None and rho0 is None
    if params.rho is not None:
        rho = float(params.rho)
    A = _transitions(params.stay_same, params.stay_diff, floor)
    pi = np.array([0.98, 0.01, 0.01])
    prev = -np.inf
    converged = False
    it = 0
    for it in range(1, params.max_iter + 1):
        log_e = np.vstack([_bb_logpmf(a, n, 0.5, rho),
                           _bb_logpmf(a, n, p_gain, rho),
                           _bb_logpmf(a, n, p_loss, rho)])
        gamma, xi, ll = forward_backward(log_e, A, pi)
        # M-step: weighted proportions, floored away from 1/2 by min_fold.
        wn = gamma @ n
        wa = gamma @ a
        # Bounded away from 0 and 1: a state at p = 0.999 claims every bin
        # where one condition happens to be empty (measured on sparse CUT&RUN
        # CTCF: fold 914, 5.5 million "regions").
        if wn[1] > 0:
            p_gain = float(np.clip(wa[1] / wn[1], p_min, 0.95))
        if wn[2] > 0:
            p_loss = float(np.clip(wa[2] / wn[2], 0.05, 1.0 - p_min))
        if rho_free:
            g = gamma[0]
            ok = n > 1
            pq = 0.25
            resid = g[ok] * ((a[ok] - n[ok] * 0.5) ** 2 - n[ok] * pq)
            denom = g[ok] * (n[ok] * pq * (n[ok] - 1.0))
            if denom.sum() > 0:
                rho = float(np.clip(resid.sum() / denom.sum(), 1e-4, 0.5))
        A = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
        A = np.clip(A, 1e-6, 1.0)
        A /= A.sum(axis=1, keepdims=True)
        A = _apply_floor(A, floor)
        pi = np.maximum(gamma[:, 0], 1e-6)
        pi /= pi.sum()
        if abs(ll - prev) < params.tol * max(1.0, abs(prev)):
            converged = True
            break
        prev = ll
    return _Fit(p_gain, p_loss, rho, A, pi, it, converged)


def _decode(a: np.ndarray, n: np.ndarray, fit: _Fit) -> np.ndarray:
    log_e = np.vstack([_bb_logpmf(a, n, 0.5, fit.rho),
                       _bb_logpmf(a, n, fit.p_gain, fit.rho),
                       _bb_logpmf(a, n, fit.p_loss, fit.rho)])
    gamma, _, _ = forward_backward(log_e, fit.A, fit.pi)
    return gamma


def _runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(x), int(y)) for x, y in zip(s, e) if y - x >= min_len]


def _merge_same_direction(ivs: List[Tuple[int, int]], max_gap: int, ratio: np.ndarray,
                          sign: float, frac: float, min_log_fold: float):
    """Join runs closer than ``max_gap`` whose gap leans the same way."""
    if max_gap <= 0 or len(ivs) < 2:
        return ivs
    pr = np.concatenate([[0.0], np.cumsum(ratio, dtype=np.float64)])
    mean = lambda s, e: (pr[e] - pr[s]) / max(e - s, 1)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        ps, pe = out[-1]
        if 0 < s - pe <= max_gap:
            g = sign * mean(pe, s)
            need = max(min_log_fold * 0.5, frac * min(sign * mean(ps, pe), sign * mean(s, e)))
            if g >= need:
                out[-1][1] = e
                continue
        out.append([s, e])
    return [(x, y) for x, y in out]


# -- the pipeline ------------------------------------------------------------

def _control_level(ref: Coverage, ctrls: Sequence[Coverage], smooth_bp: int,
                   chroms: Sequence[str]) -> Dict[str, np.ndarray]:
    """Depth-scaled, smoothed control level per bin, averaged over controls."""
    k = max(1, int(round(smooth_bp / ref.bin_size)))
    out: Dict[str, np.ndarray] = {}
    for c in chroms:
        n = ref.bins[c].size
        lvl = np.zeros(n, dtype=np.float64)
        for ct in ctrls:
            sc = ct.scaled_to(ref)
            if c in sc.bins:
                v = sc.bins[c].astype(np.float64)
                lvl[: min(v.size, n)] += v[:n] / len(ctrls)
        out[c] = uniform_filter1d(lvl, size=k, mode="nearest") if k > 1 else lvl
    return out


def differential_from_coverage(
    reps_a: Sequence[Coverage], reps_b: Sequence[Coverage],
    controls_a: Optional[Sequence[Coverage]] = None,
    controls_b: Optional[Sequence[Coverage]] = None,
    params: Optional[DiffParams] = None, name: str = "flexpeak_diff",
    scan_info: Optional[Dict] = None, log: Callable = lambda *a: None,
) -> DiffResult:
    """Differential regions from already-loaded, equally-binned coverage."""
    t0 = time.perf_counter()
    params = params or DiffParams()
    bs = reps_a[0].bin_size
    if any(r.bin_size != bs for r in list(reps_a) + list(reps_b)):
        raise ValueError("all samples must share one bin size")
    chroms = sorted(set.intersection(*(set(r.bins) for r in list(reps_a) + list(reps_b))))
    if not chroms:
        raise ValueError("no chromosome is present in every sample")
    reps_a = [r.subset(chroms) for r in reps_a]
    reps_b = [r.subset(chroms) for r in reps_b]
    # Loaders and the cache's rebin path round a chromosome's bin count
    # differently by one; line every sample up on the shortest.
    for c in chroms:
        n_min = min(r.bins[c].size for r in list(reps_a) + list(reps_b))
        for r in list(reps_a) + list(reps_b):
            if r.bins[c].size != n_min:
                r.bins[c] = r.bins[c][:n_min]
    ctrl_a = (_control_level(reps_a[0], controls_a, params.control_smooth_bp, chroms)
              if controls_a else None)
    ctrl_b = (_control_level(reps_b[0], controls_b, params.control_smooth_bp, chroms)
              if controls_b else None)

    # Normalised input: put the pseudo-counts on a stated depth before anything
    # else, so the test's power is a parameter and not an accident of the loader.
    is_count = all(r.is_count for r in list(reps_a) + list(reps_b))
    scale_note = None
    if not is_count:
        if params.pseudo_depth > 0:
            reps_a = [_rescale_mean(r, params.pseudo_depth) for r in reps_a]
            reps_b = [_rescale_mean(r, params.pseudo_depth) for r in reps_b]
            scale_note = f"rescaled to a mean of {params.pseudo_depth:g} per {bs} bp bin"
        else:
            units = [_poisson_units(r) for r in list(reps_a) + list(reps_b)]
            reps_a = [_rescale_mean(r, r.mean_signal() / u) for r, u in zip(reps_a, units[: len(reps_a)])]
            reps_b = [_rescale_mean(r, r.mean_signal() / u) for r, u in zip(reps_b, units[len(reps_a):])]
            scale_note = ("fragment scale recovered from background variance/mean: units per "
                          f"fragment {', '.join(f'{u:.2f}' for u in units)}; mean per {bs} bp bin now "
                          f"{', '.join(f'{r.mean_signal():.2f}' for r in list(reps_a) + list(reps_b))}")
        log(f"[flexpeak diff] normalised (non-count) input: {scale_note}; p-values are approximate")

    # Depth-adaptive bin: coarsen until a bin carries enough fragments for
    # its split to be informative.
    rebin_note = None
    if params.min_depth_per_bin > 0:
        depth = float(np.mean([r.mean_signal() for r in reps_a])
                      + np.mean([r.mean_signal() for r in reps_b]))
        if 0 < depth < params.min_depth_per_bin:
            factor = int(min(np.ceil(params.min_depth_per_bin / depth), params.max_depth_rebin))
            if factor > 1:
                reps_a = [r.rebin(bs * factor) for r in reps_a]
                reps_b = [r.rebin(bs * factor) for r in reps_b]
                if ctrl_a is not None:
                    ctrl_a = {c: _coarsen_sum(v, factor) for c, v in ctrl_a.items()}
                if ctrl_b is not None:
                    ctrl_b = {c: _coarsen_sum(v, factor) for c, v in ctrl_b.items()}
                bs = bs * factor
                rebin_note = (f"{depth:.2f} fragments per bin across both conditions is below "
                              f"{params.min_depth_per_bin:g}: coarsened x{factor} to {bs} bp")
                log(f"[flexpeak diff] {rebin_note}")

    # Normalise every sample onto one level, then average within condition so
    # the two conditions carry equal weight whatever their replicate counts.
    factors = normalise(list(reps_a) + list(reps_b), params.norm)
    fa, fb = factors[: len(reps_a)], factors[len(reps_a):]
    log(f"[flexpeak diff] normalisation ({params.norm}): A {np.round(fa, 3).tolist()} "
        f"B {np.round(fb, 3).tolist()}")

    # Within-condition pairs only: a cross-condition pair would read the very
    # differences being looked for as replicate scatter.
    rho0 = None
    if params.rho is None:
        vals = [v for v in (estimate_rho(_rep_arrays(reps_a, fa, chroms)),
                            estimate_rho(_rep_arrays(reps_b, fb, chroms))) if v is not None]
        rho0 = float(np.median(vals)) if vals else None
    rho_source = ("fixed" if params.rho is not None else
                  "replicates" if rho0 is not None else
                  f"EM on the 'same' state, floored at {params.rho_floor_no_replicates:g} "
                  "(no replicates)")
    if rho0 is not None:
        log(f"[flexpeak diff] replicate overdispersion rho = {rho0:.4f}")

    per_chrom: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for c in chroms:
        a = np.mean([r.bins[c].astype(np.float64) * f for r, f in zip(reps_a, fa)], axis=0)
        b = np.mean([r.bins[c].astype(np.float64) * f for r, f in zip(reps_b, fb)], axis=0)
        n = np.rint(a + b)
        ai = np.minimum(np.rint(a), n)
        per_chrom[c] = (a, b, ai, n)

    # Dwell floor from the scale of the *difference*, not of the signal: shared
    # domains make the pooled signal autocorrelated over tens of kb, and a
    # floor read from that would forbid a 600 bp differential peak outright.
    # The log ratio is ~0 wherever the conditions agree, so its correlation
    # length is the width of the differential features themselves.  Capped at
    # 50 bins: connecting runs (``max_gap``) is what joins a broad domain, the
    # floor only has to stop bin-level flicker.
    pc = params.pseudocount
    scale_bp = int(np.percentile([
        autocorrelation_length_bp(np.log2((per_chrom[c][0] + pc) / (per_chrom[c][1] + pc)), bs)
        for c in chroms if per_chrom[c][0].size >= 1000] or [bs], 75))
    floor = 0.0
    if params.dwell_from_autocorrelation:
        scale = min(max(float(scale_bp), 2.0 * bs), 50.0 * bs, float(params.max_dwell))
        floor = float(np.clip(1.0 - bs / scale, 0.0, 0.9999))

    # Fit on the largest chromosomes up to fit_max_bins, decode everything.
    order = sorted(chroms, key=lambda c: per_chrom[c][3].size, reverse=True)
    fit_c, tot = [], 0
    for c in order:
        fit_c.append(c)
        tot += per_chrom[c][3].size
        if tot >= params.fit_max_bins:
            break
    a_fit = np.concatenate([per_chrom[c][2] for c in fit_c])
    n_fit = np.concatenate([per_chrom[c][3] for c in fit_c])
    if a_fit.size > params.fit_max_bins:
        a_fit, n_fit = a_fit[: params.fit_max_bins], n_fit[: params.fit_max_bins]
    fit = _fit_hmm(a_fit, n_fit, params, rho0, floor)
    if params.rho is None and rho0 is None and fit.rho < params.rho_floor_no_replicates:
        fit = _fit_hmm(a_fit, n_fit, params.replace(rho=params.rho_floor_no_replicates),
                       None, floor)
    log(f"[flexpeak diff] HMM: p_gain {fit.p_gain:.3f} p_loss {fit.p_loss:.3f} "
        f"rho {fit.rho:.4f} ({fit.n_iter} it, {'ok' if fit.converged else 'not converged'})")

    min_bins = max(2, int(params.min_width // bs)) if params.min_width else 2
    max_gap_bins = (int(params.max_gap // bs) if params.max_gap is not None else 2)
    log_min_fold = float(np.log2(max(params.min_fold, 1.0)))

    # Enrichment gate without an input: the pooled signal's local background.
    if params.require_enrichment and ctrl_a is None and ctrl_b is None:
        from ..background.nb import local_lambda

        pooled_mean = float(np.mean([0.5 * (per_chrom[c][0].mean() + per_chrom[c][1].mean())
                                     for c in chroms]))
        lam_pooled = {
            c: local_lambda(0.5 * (per_chrom[c][0] + per_chrom[c][1]), bs,
                            params.enrichment_windows, pooled_mean, exclude_fold=2.0)
            for c in chroms
        }
        ctrl_a = ctrl_b = lam_pooled
        fa_gate = fb_gate = [1.0]
    else:
        fa_gate, fb_gate = fa, fb

    cands: List[DiffRegion] = []
    n_control_dropped = 0
    for c in chroms:
        a, b, ai, n = per_chrom[c]
        gamma = _decode(ai, n, fit)
        ratio = np.log2((a + pc) / (b + pc))
        chrom_len = reps_a[0].chrom_sizes.get(c, n.size * bs)
        for direction, k, sign in (("gain", 1, 1.0), ("loss", 2, -1.0)):
            post = gamma[k]
            ivs = _runs(post >= params.posterior_cutoff, min_bins)
            ivs = _merge_same_direction(ivs, max_gap_bins, ratio, sign,
                                        params.gap_fold_fraction, log_min_fold)
            for s, e in ivs:
                ca = [float(r.bins[c][s:e].sum() * f) for r, f in zip(reps_a, fa)]
                cb = [float(r.bins[c][s:e].sum() * f) for r, f in zip(reps_b, fb)]
                ma, mb = float(np.mean(ca)), float(np.mean(cb))
                # Input gate: enriched over the control in at least one condition.
                if ctrl_a is not None or ctrl_b is not None:
                    need = params.min_fold_over_control
                    ok_a = ctrl_a is None or ma >= need * float(ctrl_a[c][s:e].sum()) * np.mean(fa_gate)
                    ok_b = ctrl_b is None or mb >= need * float(ctrl_b[c][s:e].sum()) * np.mean(fb_gate)
                    if not (ok_a or ok_b):
                        n_control_dropped += 1
                        continue
                l2 = float(np.log2((ma + pc) / (mb + pc)))
                cands.append(DiffRegion(
                    chrom=c, start=int(min(s * bs, chrom_len)), end=int(min(e * bs, chrom_len)),
                    direction=direction, log2fc=l2, counts_a=ca, counts_b=cb,
                    mean_posterior=float(post[s:e].mean())))
    n_cand = len(cands)

    # Region-level test: two-sided beta-binomial on the aggregate split with a
    # region-scale overdispersion (see DiffParams.region_rho).
    rho_region, rho_region_source = _region_rho(cands, len(reps_a), len(reps_b), params)
    for r in cands:
        ma, mb = float(np.mean(r.counts_a)), float(np.mean(r.counts_b))
        pval = _bb_sf_two_sided(ma, ma + mb, 0.5, rho_region)
        r.neg_log10_p = float(-np.log10(max(pval, 1e-300)))
    keep = [r for r in cands
            if r.total >= params.min_count and abs(r.log2fc) >= log_min_fold
            and (r.direction == "gain") == (r.log2fc > 0)]
    if keep:
        p = np.array([r.neg_log10_p for r in keep])
        n_tests = None
        if params.fdr_scope == "genome":
            genome = sum(reps_a[0].chrom_sizes[c] for c in chroms)
            n_tests = genome / max(float(np.median([r.width for r in keep])), 1.0)
        q = qvalue_from_neglog10p(p, n_tests=n_tests)
        for r, qq in zip(keep, q):
            r.neg_log10_q = float(qq)
        cut = -np.log10(params.qvalue) if params.qvalue > 0 else 0.0
        keep = [r for r in keep if r.neg_log10_q >= cut]
    keep.sort(key=lambda r: (r.chrom, r.start))
    for i, r in enumerate(keep, 1):
        r.name = f"{name}_{i:05d}"

    diag = {
        "bin_size": bs, "norm": params.norm, "norm_a": [round(x, 3) for x in fa],
        "norm_b": [round(x, 3) for x in fb], "rho": fit.rho, "rho_source": rho_source,
        "p_gain": fit.p_gain, "p_loss": fit.p_loss,
        "fold_gain": fit.p_gain / (1 - fit.p_gain), "fold_loss": (1 - fit.p_loss) / fit.p_loss,
        "dwell_floor_bp": int(round(bs / (1 - floor))) if 0 < floor < 1 else 0,
        "fitted_dwell_bp": [int(round(bs / max(1 - d, 1e-9))) for d in np.diag(fit.A)],
        "n_iter": fit.n_iter, "converged": fit.converged, "fit_bins": int(a_fit.size),
        "n_candidates": n_cand, "scale_bp": scale_bp,
        "control_dropped": n_control_dropped, "is_count": is_count, "scale_note": scale_note,
        "enrichment_gate": ("input" if controls_a or controls_b else
                            "pooled local background" if params.require_enrichment else "none"),
        "rebin_note": rebin_note,
        "region_rho": rho_region, "region_rho_source": rho_region_source,
        "elapsed_s": time.perf_counter() - t0,
    }
    if scan_info:
        diag.update(scan_info)
    return DiffResult(
        regions=keep,
        params={"call": params.to_dict(), "cond_a": [r.source for r in reps_a],
                "cond_b": [r.source for r in reps_b], "norm_factors": {"a": fa, "b": fb},
                "rho": fit.rho, "p_gain": fit.p_gain, "p_loss": fit.p_loss},
        chrom_sizes={c: int(reps_a[0].chrom_sizes[c]) for c in chroms},
        diagnostics=diag,
    )


def _region_rho(cands: Sequence["DiffRegion"], n_a: int, n_b: int,
                params: DiffParams) -> Tuple[float, str]:
    """Region-scale beta-binomial overdispersion for the region test.

    With replicates: method of moments on the split of region sums between
    replicates of the same condition, pooled over the candidate regions
    (within-condition pairs only, so the differences being tested never enter).
    Without: the no-replicate prior.
    """
    if params.region_rho is not None:
        return float(params.region_rho), "fixed"
    vals = []
    prior = float(params.rho_floor_no_replicates)
    if cands:
        for k, n in (("counts_a", n_a), ("counts_b", n_b)):
            if n < 2:
                continue
            reps = [np.array([getattr(r, k)[i] for r in cands], dtype=np.float64) for i in range(n)]
            v = estimate_rho(reps, min_total=20.0, min_bins=10)
            if v is not None:
                vals.append(v)
    if vals:
        # Shrink a few-region estimate toward the prior (edgeR-style
        # moderation): with 20 regions the data and the prior weigh equally.
        est = float(np.median(vals))
        w = len(cands) / (len(cands) + 20.0)
        return float(max(w * est + (1 - w) * prior, 1e-4)), (
            f"replicate scatter of {len(cands)} region sums, shrunk toward {prior:g}")
    return prior, "no replicates: BCV ~0.2 prior at region scale"


def _poisson_units(cov: Coverage, n_sd: float = 5.0, max_bins: int = 4_000_000) -> float:
    """Track units per fragment: variance / mean of the sigma-clipped background.

    For counts this is 1 + alpha * mean (>= 1); for a track scaled by ``c`` it
    is ``c`` times that.  The NB inflation makes the estimate slightly too
    large, i.e. the recovered counts slightly too small -- conservative.
    """
    from ..scan import sigma_clip

    x = cov.concat()
    if x.size > max_bins:
        x = x[:: int(np.ceil(x.size / max_bins))]
    m, sd, keep = sigma_clip(x, n_sd)
    if m <= 0 or keep.sum() < 100:
        return 1.0
    kept = x[keep]
    clipped = kept.var() / max(kept.mean(), 1e-9)
    # Weak domains survive a 5 SD clip and inflate the variance (measured:
    # 1013 units for a track whose true factor was 37).  The MAD-based
    # variance barely sees a small enriched fraction, and contamination can
    # only inflate the ratio, so the smaller estimate is the better one; a
    # sparse track with MAD 0 falls back to the clipped ratio.
    med = float(np.median(kept))
    mad = 1.4826 * float(np.median(np.abs(kept - med)))
    robust = (mad * mad) / med if (mad > 0 and med > 0) else clipped
    return float(max(min(clipped, robust), 1e-6))


def _coarsen_sum(a: np.ndarray, factor: int) -> np.ndarray:
    pad = (-a.size) % factor
    if pad:
        a = np.concatenate([a, np.zeros(pad, dtype=a.dtype)])
    return a.reshape(-1, factor).sum(axis=1)


def _rescale_mean(cov: Coverage, target_mean: float) -> Coverage:
    """Scale a coverage so its mean per bin is ``target_mean``."""
    m = cov.mean_signal()
    if m <= 0:
        return cov
    f = target_mean / m
    return Coverage(bins={c: (a * f).astype(np.float32) for c, a in cov.bins.items()},
                    bin_size=cov.bin_size, chrom_sizes=dict(cov.chrom_sizes),
                    source=cov.source, is_count=cov.is_count, n_fragments=cov.n_fragments,
                    fragment_length=cov.fragment_length, meta=dict(cov.meta))


def _rep_arrays(reps: Sequence[Coverage], factors: Sequence[float], chroms) -> List[np.ndarray]:
    return [np.concatenate([r.bins[c] for c in chroms]).astype(np.float64) * f
            for r, f in zip(reps, factors)]


def _within_peaks(res: DiffResult, reps_a, reps_b, params: DiffParams, log) -> DiffResult:
    """Keep differential regions overlapping a FlexPeak peak in either pooled condition."""
    from ..caller import CallParams, call_peaks
    from ..peaks import interval_overlap_bp

    def pooled(reps):
        bins = {c: np.sum([r.bins[c] for r in reps], axis=0).astype(np.float32)
                for c in reps[0].bins}
        return Coverage(bins=bins, bin_size=reps[0].bin_size,
                        chrom_sizes=dict(reps[0].chrom_sizes), source="pooled")

    cp = CallParams(bin_size=reps_a[0].bin_size, call_subpeaks=False)
    union = call_peaks(pooled(reps_a), None, cp).intervals() + \
        call_peaks(pooled(reps_b), None, cp).intervals()
    before = len(res.regions)
    res.regions = [r for r in res.regions
                   if interval_overlap_bp([(r.chrom, r.start, r.end)], union) > 0]
    for i, r in enumerate(res.regions, 1):
        r.name = r.name.rsplit("_", 1)[0] + f"_{i:05d}"
    res.diagnostics["within_peaks_dropped"] = before - len(res.regions)
    log(f"[flexpeak diff] within-peaks filter: {before} -> {len(res.regions)}")
    return res


def call_differential(
    paths_a: Sequence[str], paths_b: Sequence[str],
    controls_a: Optional[Sequence[str]] = None, controls_b: Optional[Sequence[str]] = None,
    params: Optional[DiffParams] = None, loader: Optional[Callable] = None,
    name: str = "flexpeak_diff", scan: bool = True, log: Callable = lambda *a: None,
) -> DiffResult:
    """Load two conditions from files and call differential regions.

    ``loader(path, bin_size) -> Coverage``; the default is
    :func:`flexpeak.load_signal` with caching.  The quick scan on the first
    sample of condition A picks the bin size and the gap scale unless the
    parameters fix them.
    """
    from ..io.load import load_signal

    params = params or DiffParams()
    loader = loader or (lambda p, b: load_signal(p, bin_size=b))
    scan_info = None
    if scan and (params.bin_size is None or params.max_gap is None):
        from ..scan import scan_file

        prof = scan_file(paths_a[0])
        rec = prof.recommend()
        bs = params.bin_size or {"narrow": 50, "mixed": 100, "broad": max(200, rec.segment_bin_size or 0),
                                 "null": 100}[rec.signal_class]
        params = params.replace(bin_size=int(bs))
        if params.max_gap is None:
            params = params.replace(max_gap=int(max(rec.max_gap, 2 * bs)))
        if max(rec.windows) > max(params.enrichment_windows):
            params = params.replace(enrichment_windows=tuple(rec.windows[-2:]))
        scan_info = {"scan_class": rec.signal_class, "scan_scale_bp": rec.scale_bp}
        log(f"[flexpeak diff] scan: {rec.signal_class} mark -> bin {params.bin_size} bp, "
            f"max_gap {params.max_gap} bp ({prof.elapsed_s:.1f} s)")
    if params.bin_size is None:
        params = params.replace(bin_size=100)
    bs = params.bin_size
    log(f"[flexpeak diff] loading {len(paths_a)} + {len(paths_b)} samples at {bs} bp")
    reps_a = [loader(p, bs) for p in paths_a]
    reps_b = [loader(p, bs) for p in paths_b]
    ca = [loader(p, bs) for p in (controls_a or [])]
    cb = [loader(p, bs) for p in (controls_b or [])]
    res = differential_from_coverage(reps_a, reps_b, ca or None, cb or None, params,
                                     name=name, scan_info=scan_info, log=log)
    if params.within_peaks:
        res = _within_peaks(res, reps_a, reps_b, params, log)
    return res
