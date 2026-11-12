"""Format dispatch and the coverage cache."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..signal.coverage import Coverage

__all__ = ["load_signal", "infer_format", "cache_path", "primary_chromosomes",
           "remove_caches"]

_BAM_EXT = {".bam", ".cram", ".sam"}
_BW_EXT = {".bw", ".bigwig", ".bigWig"}
_NPZ_EXT = {".npz"}


def infer_format(path) -> str:
    ext = Path(str(path)).suffix.lower()
    if ext in _BAM_EXT:
        return "bam"
    if ext in {e.lower() for e in _BW_EXT}:
        return "bigwig"
    if ext in _NPZ_EXT:
        return "cache"
    raise ValueError(
        f"cannot infer format from {path!r}; expected one of "
        f"{sorted(_BAM_EXT | _BW_EXT | _NPZ_EXT)}"
    )


def cache_path(path, bin_size: int, chroms: Optional[Sequence[str]], cache_dir=None) -> Path:
    """Deterministic cache filename keyed on source, mtime, bin size, chroms."""
    p = Path(str(path))
    try:
        stamp = f"{p.stat().st_size}:{int(p.stat().st_mtime)}"
    except OSError:
        stamp = "0:0"
    key = f"{p.resolve()}|{stamp}|{bin_size}|{','.join(sorted(chroms)) if chroms else 'all'}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    base = Path(cache_dir) if cache_dir else p.parent / ".flexpeak_cache"
    return base / f"{p.stem}.{bin_size}bp.{digest}.npz"


def _stamp(path) -> str:
    p = Path(str(path))
    try:
        return f"{p.resolve()}|{p.stat().st_size}:{int(p.stat().st_mtime)}"
    except OSError:
        return f"{p}|0:0"


def _siblings(path, bin_size: int, cache_dir=None):
    """``(file, header)`` of every cache of this source at this bin size."""
    p = Path(str(path))
    base = Path(cache_dir) if cache_dir else p.parent / ".flexpeak_cache"
    if not base.is_dir():
        return
    for f in sorted(base.glob(f"{p.stem}.{bin_size}bp.*.npz")):
        try:
            yield f, Coverage.read_meta(f)
        except Exception:
            continue


def _superset_cache(path, bin_size: int, chroms: Sequence[str],
                    cache_dir=None) -> Optional[Coverage]:
    """This file's coverage for ``chroms`` out of a cache that holds more
    chromosomes (typically the whole-genome one), instead of reading the
    source again and leaving a second cache beside the first.  Count data
    only: a bigWig is rescaled over the chromosomes it was loaded on."""
    want, stamp = set(chroms), _stamp(path)
    for f, meta in _siblings(path, bin_size, cache_dir):
        inner = meta.get("meta") or {}
        if inner.get("cache_stamp") != stamp or not meta.get("is_count", True):
            continue
        if want < set(meta.get("chroms") or []):
            try:
                return Coverage.load(f, chroms=chroms)
            except Exception:
                continue
    return None


def _prune_caches(path, bin_size: int, keep: Path, chroms: Sequence[str],
                  cache_dir=None) -> None:
    """Drop this source's caches that ``keep`` makes redundant: ones written
    from an older version of the file, and ones holding a subset of the
    chromosomes ``keep`` holds.  One cache per input is the steady state."""
    have, stamp = set(chroms), _stamp(path)
    for f, meta in _siblings(path, bin_size, cache_dir):
        if f == keep:
            continue
        inner = meta.get("meta") or {}
        if inner.get("cache_stamp") is None:
            continue  # not written by load_signal (a scan-window cache): leave it
        stale = inner.get("cache_stamp") != stamp
        subset = set(meta.get("chroms") or []) <= have
        if stale or subset:
            try:
                os.remove(f)
            except OSError:
                pass


def remove_caches(path, cache_dir=None) -> List[Path]:
    """Delete every coverage cache written for ``path``.

    Covers all bin sizes, chromosome subsets and the quick scan's window
    caches.  A file is removed only when its header names ``path`` as its
    source, so caches of other inputs sharing the directory (or the stem)
    are left alone.  The default ``.flexpeak_cache/`` directory is removed
    too once it is empty; a directory given as ``cache_dir`` is kept.
    Returns the files removed.
    """
    p = Path(str(path))
    base = Path(cache_dir) if cache_dir else p.parent / ".flexpeak_cache"
    if not base.is_dir():
        return []
    try:
        want = p.resolve()
    except OSError:
        want = p
    removed: List[Path] = []
    for f in sorted(base.glob(f"{p.stem}.*bp.*.npz")):
        try:
            src = Coverage.read_meta(f).get("source") or ""
            src = src.split("#", 1)[0]  # derived copies carry a '#suffix'
            if not src or Path(src).resolve() != want:
                continue
            f.unlink()
            removed.append(f)
        except Exception:
            continue
    if cache_dir is None:
        try:
            base.rmdir()  # only succeeds when nothing else is left in it
        except OSError:
            pass
    return removed


def _finer_cache(path, bin_size: int, chroms: Optional[Sequence[str]],
                 cache_dir=None) -> Optional[Coverage]:
    """The largest cached bin size that divides ``bin_size``, for this exact
    source file (same stamp) and chromosome set; None if there is none."""
    for b in sorted({d for d in range(1, bin_size) if bin_size % d == 0}, reverse=True):
        want = cache_path(path, b, chroms, cache_dir)
        if want.exists():
            try:
                return Coverage.load(want)
            except Exception:
                continue
    return None


#: Sequence names that are not assembled chromosomes: unplaced / unlocalised
#: scaffolds, alternate haplotypes, fix patches and decoys (``chrUn_...``,
#: ``chr1_KI270706v1_random``, ``chr6_GL000250v2_alt``, ``GL000220.1``,
#: ``KI270728.1``, ``JH584299.1``, ``HLA-...``), the mitochondrion, and the
#: spike-in / viral sequences commonly appended to a reference.
_NOT_PRIMARY = re.compile(
    r"(_|\.|^hla-|^(chr)?(un|m|mt|mito|ebv|phix|lambda)$)", re.IGNORECASE)


def primary_chromosomes(sizes: Dict[str, int]) -> List[str]:
    """The assembled chromosomes of a reference, in the file's order.

    A hg38 BAM header carries 190-odd sequences beyond chr1-22, X and Y.
    They hold almost no usable signal but they are not free: each is one more
    chromosome to fit, chrM alone is thousands-fold over-covered in ATAC, and
    the alt / random contigs are where multi-mapping artefacts pile up and get
    called.  Peak calling is restricted to the assembled chromosomes unless
    told otherwise.

    The rule is by name (see ``_NOT_PRIMARY``), and it is only trusted when
    what it keeps is most of the genome: a reference named some other way
    (RefSeq ``NC_000001.11``, an assembly that is all scaffolds) keeps
    everything.
    """
    keep = [c for c in sizes if not _NOT_PRIMARY.search(c)]
    total = float(sum(sizes.values())) or 1.0
    if not keep or sum(sizes[c] for c in keep) < 0.5 * total:
        return list(sizes)
    return keep


def _chrom_sizes(path, fmt: str) -> Dict[str, int]:
    if fmt == "bam":
        from .bam import _require_pysam

        with _require_pysam().AlignmentFile(str(path)) as af:
            return dict(zip(af.references, af.lengths))
    from .bigwig import _require_pybigwig

    bw = _require_pybigwig().open(str(path))
    try:
        return dict(bw.chroms())
    finally:
        bw.close()


def load_signal(
    path,
    bin_size: int = 25,
    chroms: Optional[Sequence[str]] = None,
    fmt: Optional[str] = None,
    cache: bool = True,
    cache_dir=None,
    primary_only: bool = True,
    **kwargs,
) -> Coverage:
    """Load BAM/CRAM or bigWig into a :class:`Coverage`, with on-disk caching.

    The cache is what makes re-running with different parameters, and the
    runtime tuning sweep, effectively free -- the source file is parsed once.

    With no ``chroms`` given, only the assembled chromosomes are loaded
    (:func:`primary_chromosomes`); ``primary_only=False`` loads every
    sequence in the file except the mitochondrion and EBV.
    """
    fmt = fmt or infer_format(path)

    if fmt == "cache":
        return Coverage.load(path)

    if chroms is None and primary_only:
        try:
            sizes = _chrom_sizes(path, fmt)
        except (OSError, ValueError):
            sizes = {}
        prim = primary_chromosomes(sizes) if sizes else []
        if prim and len(prim) < len(sizes):
            chroms = prim

    cp = cache_path(path, bin_size, chroms, cache_dir) if cache else None
    if cp is not None and cp.exists():
        try:
            return Coverage.load(cp)
        except Exception:  # corrupt or version-mismatched cache: rebuild
            try:
                os.remove(cp)
            except OSError:
                pass
    if cp is not None:
        # A finer cache of the same file and chromosomes can be coarsened
        # instead of re-reading the source: the quick scan and a preset often
        # ask for different bins of one file, and the read is the slow part.
        # Rebinned in memory and not written back: coarsening a cache takes
        # about a second, and saving it left one more file per bin size.
        finer = _finer_cache(path, bin_size, chroms, cache_dir)
        if finer is not None:
            return finer.rebin(bin_size)
        if chroms and fmt == "bam":
            sup = _superset_cache(path, bin_size, chroms, cache_dir)
            if sup is not None:
                return sup

    if fmt == "bam":
        from .bam import load_bam

        cov = load_bam(str(path), bin_size=bin_size, chroms=chroms, **kwargs)
    elif fmt == "bigwig":
        from .bigwig import load_bigwig

        cov = load_bigwig(str(path), bin_size=bin_size, chroms=chroms, **kwargs)
    else:
        raise ValueError(f"unknown format {fmt!r}")

    if cp is not None:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cov.meta["cache_stamp"] = _stamp(path)
            cov.meta["chrom_signal"] = {c: float(a.sum()) for c, a in cov.bins.items()}
            cov.save(cp)
            _prune_caches(path, bin_size, cp, list(cov.bins), cache_dir)
        except OSError:
            pass  # a read-only input directory must not break peak calling
    return cov
