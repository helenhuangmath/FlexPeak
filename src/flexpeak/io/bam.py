"""BAM/CRAM -> binned fragment coverage.

Paired-end reads become real fragments.  Single-end reads are extended by a
fragment length estimated from strand cross-correlation, which is also the
NSC/RSC QC statistic (DESIGN.md section 5).

The file is read **once**.  Each chromosome is scanned into two integer arrays
of fragment anchors -- forward starts and reverse ends for single-end data,
fragment starts and ends for paired-end -- and everything after that is NumPy:
the fragment-length estimate, the extension and the binning.  The per-read
Python loop does nothing but read four fields and append an integer, which is
the floor pysam allows.  Measured on the 1.5M-read benchmark BAMs this took a
load from 5.8 s to under 1.5 s; most of the old cost was a second full pass for
the fragment-length estimate that indexed a NumPy array once per read.

Chromosomes are independent, so on large files they are scanned in worker
processes (DESIGN.md section 7).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..signal.coverage import Coverage

__all__ = ["load_bam", "estimate_fragment_length", "DEFAULT_EXCLUDE"]

DEFAULT_EXCLUDE = ("chrM", "chrEBV", "M", "MT")

# 0x4 unmapped | 0x100 secondary | 0x200 vendor fail | 0x400 duplicate | 0x800 supplementary
_BAD_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800
_PROPER_PAIR = 0x2
_REVERSE = 0x10

#: Below this many mapped reads the process pool costs more than it saves
#: (each worker pays 1-2 s to start and import its dependencies).
_PARALLEL_MIN_READS = 10_000_000


def _require_pysam():
    try:
        import pysam  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading BAM/CRAM requires pysam.  Install it with:\n"
            "    pip install 'flexpeak[bam]'   (or: pip install pysam)"
        ) from e
    return __import__("pysam")


def _scan_chrom(path: str, chrom: str, bad: int, mapq: int, paired: bool,
                max_fragment: int = 1000,
                reference: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """One pass over a chromosome -> fragment anchor arrays.

    Single-end: ``(forward read starts, reverse read ends)``.
    Paired-end: ``(fragment starts, fragment ends)``, one per proper pair --
    only the mate carrying the positive template length is kept, so each
    fragment is counted once.
    """
    pysam = _require_pysam()
    a: List[int] = []
    b: List[int] = []
    push_a, push_b = a.append, b.append
    with pysam.AlignmentFile(path, reference_filename=reference) as af:
        if paired:
            for read in af.fetch(chrom):
                f = read.flag
                if f & bad or not f & _PROPER_PAIR or read.mapping_quality < mapq:
                    continue
                tlen = read.template_length
                if tlen <= 0 or tlen > max_fragment:
                    continue
                s = read.reference_start
                push_a(s)
                push_b(s + tlen)
        else:
            for read in af.fetch(chrom):
                f = read.flag
                if f & bad or read.mapping_quality < mapq:
                    continue
                if f & _REVERSE:
                    e = read.reference_end
                    push_b(e if e is not None else read.reference_start)
                else:
                    push_a(read.reference_start)
    return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)


def _scan_task(args) -> Tuple[str, np.ndarray, np.ndarray]:
    """Process-pool entry point: top-level so it pickles."""
    chrom = args[1]
    a, b = _scan_chrom(*args)
    return chrom, a, b


def _bin_fragments(starts: np.ndarray, ends: np.ndarray, size: int,
                   bin_size: int) -> np.ndarray:
    """Binned fragment coverage via a difference array.

    Each fragment adds 1 to every bin it touches.  ``bincount`` on the start
    and end bins replaces a per-fragment loop (and ``np.add.at``, which is
    ~10x slower than ``bincount`` for this).
    """
    nb = size // bin_size + 2
    if starts.size == 0:
        return np.zeros(nb, dtype=np.float32)
    s = np.maximum(starts, 0)
    e = np.minimum(ends, size)
    sb = np.clip(s // bin_size, 0, nb)
    eb = np.clip((e - 1) // bin_size + 1, 0, nb)
    diff = (np.bincount(sb, minlength=nb + 1)
            - np.bincount(eb, minlength=nb + 1))
    return np.cumsum(diff[:-1])[:nb].astype(np.float32)


def _fragments(a: np.ndarray, b: np.ndarray, paired: bool,
               frag_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Anchors -> (starts, ends).  Single-end reads are extended 5' -> 3'."""
    if paired:
        return a, b
    starts = np.concatenate([a, b - frag_len])
    ends = np.concatenate([a + frag_len, b])
    return starts, ends


def _strand_xcorr(plus: np.ndarray, minus: np.ndarray, length: int,
                  max_shift: int = 500, resolution: int = 5,
                  max_reads: int = 2_000_000) -> Tuple[int, float]:
    """Fragment length and NSC from forward-start / reverse-end positions."""
    n = plus.size + minus.size
    if n > max_reads:
        # Reads arrive sorted by position, and the estimate has always used
        # the first ``max_reads`` of them; keep that same leading slice.
        cut = np.partition(np.concatenate([plus, minus]), max_reads - 1)[max_reads - 1]
        plus, minus = plus[plus <= cut], minus[minus <= cut]
        n = plus.size + minus.size
    if n < 1000:
        return 200, 0.0

    nb = length // resolution + 2
    p = np.bincount(np.minimum(plus // resolution, nb - 1), minlength=nb).astype(np.float64)
    m = np.bincount(np.minimum(minus // resolution, nb - 1), minlength=nb).astype(np.float64)
    p -= p.mean()
    m -= m.mean()
    denom = np.sqrt((p ** 2).sum() * (m ** 2).sum())
    if denom == 0:
        return 200, 0.0

    shifts = np.arange(0, max_shift // resolution + 1)
    cc = np.empty(shifts.size, dtype=np.float64)
    for i, s in enumerate(shifts):
        cc[i] = float(np.dot(p[:p.size - s], m[s:]) if s else np.dot(p, m)) / denom

    # Ignore the zero-shift "phantom" peak from read length.
    lo = max(1, 50 // resolution)
    best = int(np.argmax(cc[lo:]) + lo)
    frag = int(best * resolution)
    background = float(np.median(cc[-max(3, cc.size // 5):]))
    nsc = float(cc[best] / background) if background > 0 else 0.0
    return max(50, min(frag, max_shift)), nsc


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
    plus, minus = _scan_chrom(path, chrom, _BAD_FLAGS, mapq, paired=False)
    return _strand_xcorr(plus, minus, length, max_shift, resolution, max_reads)


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


def _mapped_reads(af, chroms: Sequence[str]) -> int:
    """Mapped reads on ``chroms`` from the index; 0 if the index has no stats."""
    try:
        stats = {s.contig: s.mapped for s in af.get_index_statistics()}
    except Exception:  # CRAM / old index: no per-contig counts
        return 0
    return int(sum(stats.get(c, 0) for c in chroms))


def _resolve_jobs(n_jobs: Optional[int], n_chroms: int, n_reads: int) -> int:
    """Worker count.  See ``load_bam(n_jobs=...)`` for the three modes.

    ``None`` only goes parallel where the default start method is ``fork``.
    Under ``spawn`` (macOS, Windows) every worker re-imports the caller's
    ``__main__``, so a script without an ``if __name__ == "__main__"`` guard
    would crash -- or worse, re-run itself in every worker.  A library must not
    impose that on a plain script, so there it is opt-in.
    """
    if n_chroms <= 1:
        return 1
    if n_jobs is None or n_jobs <= 0:
        if n_reads < _PARALLEL_MIN_READS:
            return 1
        if n_jobs is None:
            import multiprocessing as mp

            if mp.get_start_method(allow_none=True) not in (None, "fork") or (
                    mp.get_start_method(allow_none=True) is None
                    and mp.get_context().get_start_method() != "fork"):
                return 1
        n_jobs = min(os.cpu_count() or 1, 8)
    return max(1, min(int(n_jobs), n_chroms))


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
    n_jobs: Optional[int] = None,
) -> Coverage:
    """Read a BAM/CRAM into binned fragment coverage.

    Parameters
    ----------
    paired : None auto-detects from the first reads; True/False forces a mode.
    fragment_length : single-end extension length; estimated if None, from the
        largest loaded chromosome.
    n_jobs : worker processes for the per-chromosome scan.

        * ``None`` (default) -- automatic, and only where it is safe without a
          ``__main__`` guard (fork-based platforms such as Linux).
        * ``<= 0`` -- automatic on any platform: up to 8 workers, but serial
          below a few million reads, where pool start-up outweighs the gain.
          On macOS/Windows the calling script needs an
          ``if __name__ == "__main__":`` guard.  The CLI uses this mode.
        * ``>= 1`` -- exactly that many workers.
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
        jobs = _resolve_jobs(n_jobs, len(wanted), _mapped_reads(af, wanted))

    # Largest first: it is the one the fragment-length estimate needs, and the
    # longest task should start first so the pool does not finish on it.
    order = sorted(wanted, key=lambda c: sizes[c], reverse=True)
    tasks = [(str(path), c, bad, mapq, bool(paired), max_fragment, reference)
             for c in order]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            scans = {c: (a, b) for c, a, b in pool.map(_scan_task, tasks)}
    else:
        scans = {c: (a, b) for c, a, b in map(_scan_task, tasks)}

    frag_len = fragment_length
    nsc = 0.0
    if not paired and frag_len is None:
        plus, minus = scans[order[0]]
        frag_len, nsc = _strand_xcorr(plus, minus, sizes[order[0]])
    if not paired:
        frag_len = frag_len or 200

    bins: Dict[str, np.ndarray] = {}
    n_frag = 0
    for chrom in wanted:
        a, b = scans.pop(chrom)
        starts, ends = _fragments(a, b, bool(paired), int(frag_len or 0))
        bins[chrom] = _bin_fragments(starts, ends, sizes[chrom], bin_size)
        n_frag += int(starts.size)

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
