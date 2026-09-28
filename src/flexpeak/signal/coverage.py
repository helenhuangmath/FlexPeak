"""Binned coverage: the single in-memory representation everything else uses.

DESIGN.md section 3.1 / 7.1: bin once, never iterate per-base-pair.  Base-pair
resolution is recovered only inside candidate regions, for summit refinement.
The cache is what makes the runtime parameter sweep (section 8) affordable --
the BAM is parsed once and every sweep point is an array operation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

__all__ = ["Coverage"]


@dataclass
class Coverage:
    """Binned signal for one sample.

    Attributes
    ----------
    bins : per-chromosome float32 arrays of binned signal
    bin_size : bp per bin
    chrom_sizes : full chromosome lengths in bp
    source : provenance string, echoed into output headers
    is_count : True if values are fragment counts (BAM), False if they are an
        already-normalised score (bigWig).  Statistical testing requires counts;
        bigWig input is rescaled to pseudo-counts on load, and this flag records
        that the result is approximate.
    """

    bins: Dict[str, np.ndarray]
    bin_size: int
    chrom_sizes: Dict[str, int]
    source: str = ""
    is_count: bool = True
    n_fragments: Optional[int] = None
    fragment_length: Optional[int] = None
    meta: Dict = field(default_factory=dict)

    # -- basic properties ----------------------------------------------------

    @property
    def chroms(self) -> List[str]:
        return list(self.bins.keys())

    @property
    def total_signal(self) -> float:
        return float(sum(float(a.sum()) for a in self.bins.values()))

    @property
    def n_bins(self) -> int:
        return int(sum(a.size for a in self.bins.values()))

    @property
    def effective_genome_size(self) -> int:
        return int(sum(self.chrom_sizes[c] for c in self.bins))

    def __repr__(self) -> str:
        return (
            f"<Coverage {len(self.bins)} chroms, {self.n_bins:,} bins @ {self.bin_size}bp, "
            f"total={self.total_signal:,.0f}, source={self.source!r}>"
        )

    def mean_signal(self) -> float:
        n = self.n_bins
        return self.total_signal / n if n else 0.0

    # -- coordinate helpers --------------------------------------------------

    def bin_to_bp(self, index: int) -> int:
        return int(index) * self.bin_size

    def bp_to_bin(self, chrom: str, pos: int) -> int:
        return int(pos) // self.bin_size

    def clip(self, chrom: str, pos: int) -> int:
        """Clamp a coordinate into ``[0, chrom_size]``."""
        return int(min(max(int(pos), 0), self.chrom_sizes.get(chrom, pos)))

    def concat(self) -> np.ndarray:
        """All bins end to end -- for genome-wide statistics only.

        Sorted by chromosome name, so the result lines up with anything else
        built the same way (the background model concatenates its ``mu`` in
        sorted order too); insertion order would silently misalign them.
        """
        return np.concatenate([self.bins[c] for c in sorted(self.bins)])

    # -- derived views -------------------------------------------------------

    def subset(self, chroms: Iterable[str]) -> "Coverage":
        """Restrict to a chromosome subset -- the basis of fast tuning (section 7.5)."""
        chroms = [c for c in chroms if c in self.bins]
        if not chroms:
            raise ValueError("no requested chromosomes present in coverage")
        return Coverage(
            bins={c: self.bins[c] for c in chroms},
            bin_size=self.bin_size,
            chrom_sizes={c: self.chrom_sizes[c] for c in chroms},
            source=self.source,
            is_count=self.is_count,
            n_fragments=self.n_fragments,
            fragment_length=self.fragment_length,
            meta=dict(self.meta),
        )

    def rebin(self, bin_size: int) -> "Coverage":
        """Coarsen to a larger bin size (must be an integer multiple)."""
        if bin_size == self.bin_size:
            return self
        if bin_size % self.bin_size:
            raise ValueError(
                f"target bin_size {bin_size} is not a multiple of {self.bin_size}; "
                "re-load from source to go finer"
            )
        k = bin_size // self.bin_size
        out = {}
        for c, a in self.bins.items():
            pad = (-a.size) % k
            if pad:
                a = np.concatenate([a, np.zeros(pad, dtype=a.dtype)])
            out[c] = a.reshape(-1, k).sum(axis=1).astype(np.float32)
        return Coverage(
            bins=out,
            bin_size=bin_size,
            chrom_sizes=dict(self.chrom_sizes),
            source=self.source,
            is_count=self.is_count,
            n_fragments=self.n_fragments,
            fragment_length=self.fragment_length,
            meta=dict(self.meta),
        )

    def thin(self, fraction: float, seed: int = 0) -> "Coverage":
        """Binomial thinning: a depth-subsampled copy of this sample.

        This is what makes the corpus-free tuner possible (DESIGN.md 8.1): every
        library is deeper than a subsample of itself, so a degraded copy can
        stand in for a shallower experiment without any external reference.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        rng = np.random.default_rng(seed)
        out = {}
        for c, a in self.bins.items():
            n = np.rint(np.maximum(a, 0)).astype(np.int64)
            out[c] = rng.binomial(n, fraction).astype(np.float32)
        return Coverage(
            bins=out,
            bin_size=self.bin_size,
            chrom_sizes=dict(self.chrom_sizes),
            source=f"{self.source}#thin{fraction}s{seed}",
            is_count=self.is_count,
            n_fragments=int(self.n_fragments * fraction) if self.n_fragments else None,
            fragment_length=self.fragment_length,
            meta=dict(self.meta),
        )

    def scaled_to(self, other: "Coverage") -> "Coverage":
        """Depth-scale this sample (typically a control) onto another's depth.

        Comparing a treatment against an unscaled control of different depth
        would report the depth difference as enrichment.
        """
        mine = self.total_signal
        theirs = other.total_signal
        scale = (theirs / mine) if mine > 0 else 1.0
        return Coverage(
            bins={c: (a * scale).astype(np.float32) for c, a in self.bins.items()},
            bin_size=self.bin_size,
            chrom_sizes=dict(self.chrom_sizes),
            source=f"{self.source}#scaled{round(scale, 3)}",
            is_count=self.is_count,
            n_fragments=self.n_fragments,
            fragment_length=self.fragment_length,
            meta=dict(self.meta),
        )

    # -- construction and the on-disk cache ----------------------------------

    @classmethod
    def from_arrays(cls, arrays: Dict[str, np.ndarray], bin_size: int, **kw) -> "Coverage":
        """Build from raw arrays, inferring chromosome sizes from their lengths."""
        bins = {c: np.asarray(a, dtype=np.float32) for c, a in arrays.items()}
        kw.setdefault("chrom_sizes", {c: a.size * bin_size for c, a in bins.items()})
        return cls(bins=bins, bin_size=bin_size, **kw)

    def save(self, path) -> None:
        """Write to a compressed npz.  This is the coverage cache."""
        meta = {
            "bin_size": self.bin_size,
            "chrom_sizes": self.chrom_sizes,
            "source": self.source,
            "is_count": self.is_count,
            "n_fragments": self.n_fragments,
            "fragment_length": self.fragment_length,
            "meta": self.meta,
            "chroms": list(self.bins),
        }
        np.savez_compressed(
            str(path),
            __meta__=np.array([str(meta)], dtype=object),
            **self.bins,
        )

    @classmethod
    def load(cls, path) -> "Coverage":
        """Read a cache written by :meth:`save`."""
        z = np.load(str(path), allow_pickle=True)
        try:
            meta = ast.literal_eval(str(z["__meta__"][0]))
            chroms = meta.get("chroms") or [k for k in z.files if k != "__meta__"]
            bins = {c: np.asarray(z[c], dtype=np.float32) for c in chroms}
            return cls(
                bins=bins,
                bin_size=int(meta["bin_size"]),
                chrom_sizes=dict(meta["chrom_sizes"]),
                source=meta.get("source", ""),
                is_count=bool(meta.get("is_count", True)),
                n_fragments=meta.get("n_fragments"),
                fragment_length=meta.get("fragment_length"),
                meta=dict(meta.get("meta") or {}),
            )
        finally:
            z.close()
