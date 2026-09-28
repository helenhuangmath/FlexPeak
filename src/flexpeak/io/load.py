"""Format dispatch and the coverage cache."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence

from ..signal.coverage import Coverage

__all__ = ["load_signal", "infer_format", "cache_path"]

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


def load_signal(
    path,
    bin_size: int = 25,
    chroms: Optional[Sequence[str]] = None,
    fmt: Optional[str] = None,
    cache: bool = True,
    cache_dir=None,
    **kwargs,
) -> Coverage:
    """Load BAM/CRAM or bigWig into a :class:`Coverage`, with on-disk caching.

    The cache is what makes re-running with different parameters, and the
    runtime tuning sweep, effectively free -- the source file is parsed once.
    """
    fmt = fmt or infer_format(path)

    if fmt == "cache":
        return Coverage.load(path)

    cp = cache_path(path, bin_size, chroms, cache_dir) if cache else None
    if cp is not None and cp.exists():
        try:
            return Coverage.load(cp)
        except Exception:  # corrupt or version-mismatched cache: rebuild
            try:
                os.remove(cp)
            except OSError:
                pass

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
            cov.save(cp)
        except OSError:
            pass  # a read-only input directory must not break peak calling
    return cov
