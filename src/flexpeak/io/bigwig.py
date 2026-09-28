"""bigWig -> binned coverage.

bigWig input lets users re-call peaks without keeping BAMs around, but it comes
with a real statistical caveat that the caller must know about: a bigWig usually
holds a *normalised* score (RPKM, CPM, ratio), not counts.  The negative
binomial test needs counts.  We therefore rescale to pseudo-counts and set
``Coverage.is_count = False`` so downstream code can flag the approximation in
the output header rather than silently pretending the p-values are exact.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..signal.coverage import Coverage
from .bam import DEFAULT_EXCLUDE

__all__ = ["load_bigwig"]


def _require_pybigwig():
    try:
        import pyBigWig  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading bigWig requires pyBigWig.  Install it with:\n"
            "    pip install 'flexpeak[bigwig]'   (or: pip install pyBigWig)"
        ) from e
    return __import__("pyBigWig")


#: Bases read per pyBigWig call.  20 Mb is ~80 MB of float32 in flight, which
#: bounds peak memory on a 200 Mb chromosome while keeping the call count low.
CHUNK_BP = 20_000_000


def _binned_from_values(bw, chrom: str, n_bins: int, bin_size: int,
                        chunk_bp: int = CHUNK_BP) -> np.ndarray:
    """Mean signal per bin, read through ``bw.values`` in chunks.

    Why not ``bw.stats(..., nBins=n)`` in one call, which is the obvious API:
    its cost grows sharply with ``nBins``, and a whole chromosome at 25 bp is
    millions of bins -- measured at ~5 minutes for mm10 chr19 against ~0.2 s
    here.  Reading raw values and reshaping is the same arithmetic done in
    numpy instead of per-bin inside the library.

    Bases the file leaves unset come back NaN and are counted as zero coverage,
    which is what they mean in a coverage track (and matches what the per-bin
    path did for wholly-empty bins).
    """
    length = int(bw.chroms()[chrom])
    out = np.empty(n_bins, dtype=np.float32)
    bins_per_chunk = max(1, chunk_bp // bin_size)
    for b0 in range(0, n_bins, bins_per_chunk):
        b1 = min(b0 + bins_per_chunk, n_bins)
        want = (b1 - b0) * bin_size
        # A contig shorter than one bin still gets a bin; reading past the end
        # is an error, so short-read it and treat the missing tail as zero.
        stop = min(b0 * bin_size + want, length)
        vals = np.asarray(bw.values(chrom, b0 * bin_size, stop, numpy=True),
                          dtype=np.float32)
        np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if vals.size < want:
            vals = np.concatenate([vals, np.zeros(want - vals.size, np.float32)])
        out[b0:b1] = vals.reshape(-1, bin_size).mean(axis=1)
    return out


def _binned_from_stats(bw, chrom: str, n_bins: int, bin_size: int,
                       chunk_bp: int = CHUNK_BP) -> np.ndarray:
    """Fallback for a pyBigWig built without numpy support.

    Still chunked: it is ``nBins`` per call, not the total, that makes ``stats``
    slow.  ``type="sum"`` divided by the bin width, rather than ``type="mean"``:
    the built-in mean divides by *covered* bases, so a half-covered bin comes
    back twice as high as the value path returns for the same bin.  The two
    readers have to agree to the last bit, or a cache written by one and read
    under the other would silently shift every call.
    """
    length = int(bw.chroms()[chrom])
    out = np.empty(n_bins, dtype=np.float32)
    bins_per_chunk = max(1, chunk_bp // bin_size)
    for b0 in range(0, n_bins, bins_per_chunk):
        b1 = min(b0 + bins_per_chunk, n_bins)
        stop = min(b1 * bin_size, length)
        vals = bw.stats(chrom, b0 * bin_size, stop, type="sum", nBins=b1 - b0)
        out[b0:b1] = [0.0 if v is None else float(v) / bin_size for v in vals]
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def load_bigwig(
    path: str,
    bin_size: int = 25,
    chroms: Optional[Sequence[str]] = None,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    scale_to_counts: bool = True,
    target_mean_count: float = 10.0,
    fragment_length: Optional[int] = None,
) -> Coverage:
    """Read a bigWig into binned coverage.

    Parameters
    ----------
    scale_to_counts : rescale so the genome-wide mean bin value is
        ``target_mean_count``.  This preserves the *shape* of the signal (all
        that segmentation needs) and puts the values on a scale where negative
        binomial testing is meaningful.  Set False to keep raw values.
    """
    bw_mod = _require_pybigwig()
    bw = bw_mod.open(str(path))
    try:
        if not bw.isBigWig():
            raise ValueError(f"{path} is not a bigWig file")
        sizes: Dict[str, int] = dict(bw.chroms())
        wanted = [c for c in (chroms or sizes) if c in sizes and c not in exclude]
        if not wanted:
            raise ValueError(f"no usable chromosomes in {path} (requested {chroms})")

        # Probe the fast path once rather than per chromosome: numpy support is
        # a build-time property of pyBigWig, not a per-call one.
        reader = _binned_from_values
        try:
            bw.values(wanted[0], 0, min(2, int(sizes[wanted[0]])), numpy=True)
        except (TypeError, RuntimeError):  # pragma: no cover - build dependent
            reader = _binned_from_stats

        bins: Dict[str, np.ndarray] = {}
        for chrom in wanted:
            length = int(sizes[chrom])
            # The trailing partial bin is dropped, so every bin covers exactly
            # bin_size bases and a bin mean is never diluted by a short tail.
            nb = max(1, length // bin_size)
            arr = reader(bw, chrom, nb, bin_size)
            # bigWigs may carry negative values (log ratios); enrichment calling
            # is only defined on non-negative signal.
            np.clip(arr, 0.0, None, out=arr)
            bins[chrom] = arr
    finally:
        bw.close()

    scale = 1.0
    if scale_to_counts:
        total = sum(float(a.sum()) for a in bins.values())
        n = sum(a.size for a in bins.values())
        mean = total / n if n else 0.0
        if mean > 0:
            scale = target_mean_count / mean
            for c in bins:
                bins[c] = (bins[c] * scale).astype(np.float32)

    return Coverage(
        bins=bins,
        bin_size=bin_size,
        chrom_sizes={c: int(sizes[c]) for c in wanted},
        source=str(path),
        is_count=False,
        n_fragments=None,
        fragment_length=fragment_length,
        meta={"bigwig_scale": scale, "scaled_to_counts": bool(scale_to_counts)},
    )
