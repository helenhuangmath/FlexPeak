"""Synthetic ChIP-like data with known ground truth.

Used by the test suite and by the worked examples.  It is also the seed of the
evaluation harness (DESIGN.md section 9): a simulator where truth is known
exactly is the only place where precision and recall are unambiguous, so it is
the right place to catch regressions before touching real data.

Three region types are generated, matching the three cases the caller must
handle without being told which it is looking at:

  narrow  focal peaks, a few hundred bp                  (TF / H3K4me3-like)
  broad   wide low-amplitude domains                     (H3K27me3-like)
  mixed   a broad domain containing focal sub-peaks      (H3K9me3-like)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .signal.coverage import Coverage

__all__ = ["TruthRegion", "SimulatedDataset", "simulate", "write_bam", "write_bigwig"]


@dataclass
class TruthRegion:
    chrom: str
    start: int
    end: int
    kind: str  # "narrow" | "broad" | "mixed"
    fold: float
    subpeaks: List[Tuple[int, int, int]] = field(default_factory=list)
    """(start, end, summit) of focal maxima inside a mixed domain."""

    @property
    def width(self) -> int:
        return self.end - self.start


@dataclass
class SimulatedDataset:
    treatment: Coverage
    control: Coverage
    truth: List[TruthRegion]
    profile: Dict[str, np.ndarray]
    """Noise-free expected signal, for debugging and for plotting."""
    params: Dict

    def truth_intervals(self) -> List[Tuple[str, int, int]]:
        return [(r.chrom, r.start, r.end) for r in self.truth]

    def subpeak_intervals(self) -> List[Tuple[str, int, int]]:
        return [(r.chrom, s, e) for r in self.truth for s, e, _ in r.subpeaks]


def _gauss(n: int, center: float, sigma: float) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    return np.exp(-0.5 * ((x - center) / max(sigma, 1e-9)) ** 2)


def simulate(
    chrom_sizes: Optional[Dict[str, int]] = None,
    bin_size: int = 25,
    background_level: float = 8.0,
    dispersion: float = 0.15,
    n_narrow: int = 40,
    n_broad: int = 6,
    n_mixed: int = 4,
    narrow_width: int = 600,
    broad_width: int = 60000,
    narrow_fold: float = 12.0,
    broad_fold: float = 2.5,
    mixed_subpeak_fold: float = 6.0,
    subpeaks_per_domain: int = 3,
    depth_factor: float = 1.0,
    seed: int = 0,
) -> SimulatedDataset:
    """Generate a treatment/control pair with known truth.

    Parameters
    ----------
    dispersion : NB dispersion of the background.  0 gives Poisson noise; the
        default is a realistically overdispersed library.  Raising it is how the
        "low quality data" scenario is simulated.
    depth_factor : multiplies all signal, simulating sequencing depth.
    """
    rng = np.random.default_rng(seed)
    chrom_sizes = chrom_sizes or {"chr1": 4_000_000, "chr2": 2_000_000}

    profile: Dict[str, np.ndarray] = {}
    baseline: Dict[str, np.ndarray] = {}
    truth: List[TruthRegion] = []

    for chrom, size in chrom_sizes.items():
        nb = size // bin_size

        # Slow undulating background: real coverage is never flat, and a caller
        # that assumes it is will call the swells.  The treatment and its input
        # control SHARE this drift -- that is the entire point of an input
        # control, and simulating independent drift would both flatter
        # no-control mode and unfairly penalise control-based background
        # estimation.
        drift = _smooth_noise(nb, rng, scale=max(4, nb // 50))
        base = 1.0 + 0.35 * drift
        baseline[chrom] = base
        prof = base.copy()

        placed: List[Tuple[int, int]] = []

        def place(width_bp: int, tries: int = 200) -> Optional[Tuple[int, int]]:
            w = max(2, width_bp // bin_size)
            for _ in range(tries):
                s = int(rng.integers(5, max(6, nb - w - 5)))
                e = s + w
                if all(e + 20 < ps or s > pe + 20 for ps, pe in placed):
                    placed.append((s, e))
                    return s, e
            return None

        for _ in range(n_narrow):
            got = place(narrow_width)
            if not got:
                continue
            s, e = got
            center = (s + e) / 2.0
            sigma = max(1.0, (e - s) / 4.0)
            prof += (narrow_fold - 1.0) * _gauss(nb, center, sigma)
            truth.append(TruthRegion(chrom, s * bin_size, e * bin_size, "narrow", narrow_fold))

        for _ in range(n_broad):
            got = place(broad_width)
            if not got:
                continue
            s, e = got
            block = np.zeros(nb)
            block[s:e] = broad_fold - 1.0
            # Soften the edges: real domains do not have square boundaries.
            block = _smooth(block, max(1, (e - s) // 20))
            prof += block
            truth.append(TruthRegion(chrom, s * bin_size, e * bin_size, "broad", broad_fold))

        for _ in range(n_mixed):
            got = place(broad_width)
            if not got:
                continue
            s, e = got
            block = np.zeros(nb)
            block[s:e] = broad_fold - 1.0
            block = _smooth(block, max(1, (e - s) // 20))
            prof += block

            subs: List[Tuple[int, int, int]] = []
            span = e - s
            for k in range(subpeaks_per_domain):
                frac = (k + 1) / (subpeaks_per_domain + 1)
                c = s + int(span * frac)
                sigma = max(1.0, (narrow_width / bin_size) / 4.0)
                prof += (mixed_subpeak_fold - 1.0) * _gauss(nb, c, sigma)
                half = max(1, int(2 * sigma))
                subs.append(
                    ((c - half) * bin_size, (c + half) * bin_size, c * bin_size + bin_size // 2)
                )
            truth.append(
                TruthRegion(chrom, s * bin_size, e * bin_size, "mixed", broad_fold, subs)
            )

        profile[chrom] = prof

    lam = {c: p * background_level * depth_factor for c, p in profile.items()}
    # The control sees the same baseline drift but none of the enrichment.
    ctrl_lam = {c: b * background_level * depth_factor for c, b in baseline.items()}
    treat_bins = {c: _nb_sample(rng, v, dispersion).astype(np.float32) for c, v in lam.items()}
    ctrl_bins = {c: _nb_sample(rng, v, dispersion).astype(np.float32)
                 for c, v in ctrl_lam.items()}

    common = dict(bin_size=bin_size, chrom_sizes=dict(chrom_sizes), is_count=True)
    params = dict(
        bin_size=bin_size, background_level=background_level, dispersion=dispersion,
        n_narrow=n_narrow, n_broad=n_broad, n_mixed=n_mixed, seed=seed,
        depth_factor=depth_factor, narrow_fold=narrow_fold, broad_fold=broad_fold,
    )

    return SimulatedDataset(
        treatment=Coverage(bins=treat_bins, source=f"sim:treat:seed{seed}", **common),
        control=Coverage(bins=ctrl_bins, source=f"sim:ctrl:seed{seed}", **common),
        truth=sorted(truth, key=lambda r: (r.chrom, r.start)),
        profile=profile,
        params=params,
    )


def _nb_sample(rng, mu: np.ndarray, dispersion: float) -> np.ndarray:
    """Sample NB with mean mu and var = mu + dispersion*mu^2 (Poisson if 0)."""
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-9)
    if dispersion <= 0:
        return rng.poisson(mu)
    n = 1.0 / dispersion
    p = n / (n + mu)
    return rng.negative_binomial(n, p)


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(x, size=max(1, int(k)), mode="nearest")


def _smooth_noise(n: int, rng, scale: int) -> np.ndarray:
    raw = rng.standard_normal(n)
    sm = _smooth(raw, scale)
    s = sm.std()
    return sm / s if s > 0 else sm


# -- file writers -----------------------------------------------------------


def write_bam(path, cov: Coverage, read_length: int = 50, seed: int = 0,
              paired: bool = False) -> str:
    """Write binned coverage out as a sorted, indexed BAM of synthetic reads.

    Reads are placed so that re-reading the BAM reproduces the binned profile.
    Requires pysam.
    """
    try:
        import pysam
    except ImportError as e:  # pragma: no cover
        raise ImportError("write_bam requires pysam (pip install pysam)") from e

    rng = np.random.default_rng(seed)
    path = str(path)
    chroms = sorted(cov.bins)
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": c, "LN": int(cov.chrom_sizes[c])} for c in chroms],
    }

    unsorted = path + ".unsorted.bam"
    n = 0
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for tid, chrom in enumerate(chroms):
            arr = cov.bins[chrom]
            counts = np.rint(np.maximum(arr, 0)).astype(np.int64)
            size = int(cov.chrom_sizes[chrom])
            for b in np.flatnonzero(counts):
                k = int(counts[b])
                base = int(b) * cov.bin_size
                offs = rng.integers(0, cov.bin_size, size=k)
                for j in range(k):
                    pos = int(min(max(base + int(offs[j]), 0), size - read_length - 1))
                    a = pysam.AlignedSegment()
                    a.query_name = f"r{n}"
                    a.query_sequence = "A" * read_length
                    a.query_qualities = pysam.qualitystring_to_array("I" * read_length)
                    a.flag = 16 if rng.random() < 0.5 else 0
                    a.reference_id = tid
                    a.reference_start = pos
                    a.mapping_quality = 60
                    a.cigartuples = [(0, read_length)]
                    out.write(a)
                    n += 1

    pysam.sort("-o", path, unsorted)
    pysam.index(path)
    import os

    os.remove(unsorted)
    return path


def write_bigwig(path, cov: Coverage) -> str:
    """Write binned coverage as a bigWig.  Requires pyBigWig."""
    try:
        import pyBigWig
    except ImportError as e:  # pragma: no cover
        raise ImportError("write_bigwig requires pyBigWig (pip install pyBigWig)") from e

    path = str(path)
    chroms = sorted(cov.bins)
    bw = pyBigWig.open(path, "w")
    try:
        bw.addHeader([(c, int(cov.chrom_sizes[c])) for c in chroms])
        for c in chroms:
            arr = cov.bins[c].astype(np.float64)
            starts = np.arange(arr.size, dtype=np.int64) * cov.bin_size
            ends = np.minimum(starts + cov.bin_size, int(cov.chrom_sizes[c]))
            keep = ends > starts
            bw.addEntries(
                [c] * int(keep.sum()),
                starts[keep].astype(np.int64).tolist(),
                ends=ends[keep].astype(np.int64).tolist(),
                values=arr[keep].tolist(),
            )
    finally:
        bw.close()
    return path
