"""BAM/CRAM -> binned fragment coverage.

Paired-end reads become real fragments.  Single-end reads are extended by a
fragment length estimated from strand cross-correlation, which is also the
NSC/RSC QC statistic (DESIGN.md section 5).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..signal.coverage import Coverage

__all__ = ["load_bam", "estimate_fragment_length", "DEFAULT_EXCLUDE"]

DEFAULT_EXCLUDE = ("chrM", "chrEBV", "M", "MT")

# 0x4 unmapped | 0x100 secondary | 0x200 vendor fail | 0x400 duplicate | 0x800 supplementary
_BAD_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800


def _require_pysam():
    try:
        import pysam  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading BAM/CRAM requires pysam.  Install it with:\n"
            "    pip install 'flexpeak[bam]'   (or: pip install pysam)"
        ) from e
    return __import__("pysam")


def _accumulate(diff: np.ndarray, starts: np.ndarray, ends: np.ndarray, bin_size: int) -> None:
    """Add fragment spans into a difference array (cumsum gives binned coverage).

    Using a difference array keeps this O(n_fragments) instead of
    O(total_fragment_bp), which is the difference between seconds and minutes.
    """
    if starts.size == 0:
        return
    sb = np.floor_divide(starts, bin_size)
    eb = np.floor_divide(ends - 1, bin_size) + 1
    np.clip(sb, 0, diff.size - 1, out=sb)
    np.clip(eb, 0, diff.size - 1, out=eb)
    np.add.at(diff, sb, 1.0)
    np.add.at(diff, eb, -1.0)


def estimate_fragment_length(
    path: str,
    chrom: Optional[str] = None,
    max_shift: int = 500,
    resolution: int = 5,
    mapq: int = 30,
    max_reads: int = 2_000_000,
) -> Tuple[int, float]:
    """Estimate single-end fragment length by strand cross-correlation.

    Returns ``(fragment_length, nsc)`` where NSC is the normalised strand
    coefficient -- the ratio of the correlation at the chosen shift to the
    background correlation.  NSC is reported in QC as an enrichment measure.
    """
    pysam = _require_pysam()
    with pysam.AlignmentFile(path) as af:
        if chrom is None:
            candidates = [
                (n, l) for n, l in zip(af.references, af.lengths)
                if n not in DEFAULT_EXCLUDE
            ]
            if not candidates:
                return 200, 0.0
            chrom = max(candidates, key=lambda x: x[1])[0]
        length = af.get_reference_length(chrom)
        nb = length // resolution + 2
        plus = np.zeros(nb, dtype=np.float32)
        minus = np.zeros(nb, dtype=np.float32)
        n = 0
        for read in af.fetch(chrom):
            if read.flag & _BAD_FLAGS or read.mapping_quality < mapq:
                continue
            if read.is_reverse:
                end = read.reference_end or read.reference_start
                minus[min(end // resolution, nb - 1)] += 1
            else:
                plus[min(read.reference_start // resolution, nb - 1)] += 1
            n += 1
            if n >= max_reads:
                break

    if n < 1000:
        return 200, 0.0

    plus -= plus.mean()
    minus -= minus.mean()
    denom = np.sqrt((plus ** 2).sum() * (minus ** 2).sum())
    if denom == 0:
        return 200, 0.0

    shifts = np.arange(0, max_shift // resolution + 1)
    cc = np.empty(shifts.size, dtype=np.float64)
    for i, s in enumerate(shifts):
        if s:
            cc[i] = float(np.dot(plus[:plus.size - s], minus[s:])) / denom
        else:
            cc[i] = float(np.dot(plus, minus)) / denom

    # Ignore the zero-shift "phantom" peak from read length.
    lo = max(1, 50 // resolution)
    best = int(np.argmax(cc[lo:]) + lo)
    frag = int(best * resolution)
    background = float(np.median(cc[-max(3, cc.size // 5):]))
    nsc = float(cc[best] / background) if background > 0 else 0.0
    return max(50, min(frag, max_shift)), nsc


def _detect_paired(af, chroms: Sequence[str], probe: int = 5000) -> bool:
    """Auto-detect layout from the first reads rather than trusting the user."""
    seen = paired = 0
    for chrom in chroms:
        for read in af.fetch(chrom):
            if read.flag & _BAD_FLAGS:
                continue
            seen += 1
            if read.is_paired:
                paired += 1
            if seen >= probe:
                break
        if seen >= probe:
            break
    return bool(seen) and paired > seen // 2


def load_bam(
    path: str,
    bin_size: int = 25,
    mapq: int = 30,
    chroms: Optional[Sequence[str]] = None,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    paired: Optional[bool] = None,
    fragment_length: Optional[int] = None,
    max_fragment: int = 1000,
    keep_duplicates: bool = False,
    reference: Optional[str] = None,
) -> Coverage:
    """Read a BAM/CRAM into binned fragment coverage.

    Parameters
    ----------
    paired : None auto-detects from the first reads; True/False forces a mode.
    fragment_length : single-end extension length; estimated if None.
    """
    pysam = _require_pysam()
    bad = (_BAD_FLAGS & ~0x400) if keep_duplicates else _BAD_FLAGS

    with pysam.AlignmentFile(path, reference_filename=reference) as af:
        all_chroms = list(af.references)
        sizes: Dict[str, int] = dict(zip(all_chroms, af.lengths))
        wanted = [c for c in (chroms or all_chroms)
                  if c in sizes and c not in exclude]
        if not wanted:
            raise ValueError(f"no usable chromosomes in {path} (requested {chroms})")
        if paired is None:
            paired = _detect_paired(af, wanted)

    frag_len = fragment_length
    nsc = 0.0
    if not paired and frag_len is None:
        frag_len, nsc = estimate_fragment_length(path, mapq=mapq)
    if not paired:
        frag_len = frag_len or 200

    bins: Dict[str, np.ndarray] = {}
    n_frag = 0
    with pysam.AlignmentFile(path, reference_filename=reference) as af:
        for chrom in wanted:
            nb = sizes[chrom] // bin_size + 2
            diff = np.zeros(nb + 1, dtype=np.float32)
            starts: List[int] = []
            ends: List[int] = []
            for read in af.fetch(chrom):
                if read.flag & bad or read.mapping_quality < mapq:
                    continue
                if paired:
                    # One read per pair carries the positive template length,
                    # so counting only those counts each fragment once.
                    if not read.is_proper_pair or read.template_length <= 0:
                        continue
                    tlen = read.template_length
                    if tlen > max_fragment:
                        continue
                    s = read.reference_start
                    e = s + tlen
                elif read.is_reverse:
                    e = read.reference_end or read.reference_start
                    s = e - frag_len
                else:
                    s = read.reference_start
                    e = s + frag_len
                starts.append(max(0, s))
                ends.append(min(sizes[chrom], e))
                if len(starts) >= 1_000_000:
                    _accumulate(diff, np.array(starts), np.array(ends), bin_size)
                    n_frag += len(starts)
                    starts, ends = [], []
            if starts:
                _accumulate(diff, np.array(starts), np.array(ends), bin_size)
                n_frag += len(starts)
            bins[chrom] = np.cumsum(diff[:-1])[:nb].astype(np.float32)

    return Coverage(
        bins=bins,
        bin_size=bin_size,
        chrom_sizes={c: int(sizes[c]) for c in wanted},
        source=str(path),
        is_count=True,
        n_fragments=n_frag,
        fragment_length=frag_len,
        meta={"paired": bool(paired), "mapq": mapq, "nsc": nsc},
    )
