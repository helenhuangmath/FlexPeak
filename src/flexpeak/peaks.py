"""The nested peak data contract.

This module is deliberately the first thing defined in the package: every other
module reads or writes these objects, so its shape is the most expensive thing
to change later (DESIGN.md section 3.3).

A ``Region`` is one contiguous enriched interval. It is either:

  * ``kind="peak"``   -- a standalone focal peak with no broader enclosing domain
  * ``kind="domain"`` -- a broad enriched domain, which may contain ``subpeaks``

The nesting is what lets FlexPeak represent mixed marks (a megabase H3K9me3
domain that also contains focal ZNF peaks) without asking the user to declare
"narrow" or "broad" up front.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Sequence

__all__ = ["SubPeak", "Region", "NestedPeakSet", "read_nested_tsv",
           "read_bed_intervals"]


def _open_out(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


@dataclass
class SubPeak:
    """A focal maximum inside (or coincident with) a Region."""

    chrom: str
    start: int
    end: int
    summit: int
    score: float = 0.0
    neg_log10_p: float = 0.0
    neg_log10_q: float = 0.0
    fold_enrichment: float = 0.0
    persistence: int = 0
    """Number of smoothing scales at which this maximum survived (DESIGN 3.3)."""
    name: str = ""

    @property
    def width(self) -> int:
        return self.end - self.start

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(f"empty/inverted subpeak {self.chrom}:{self.start}-{self.end}")
        if not (self.start <= self.summit < self.end):
            raise ValueError(
                f"summit {self.summit} outside {self.chrom}:{self.start}-{self.end}"
            )


@dataclass
class Region:
    """One called enriched interval, possibly containing sub-peaks."""

    chrom: str
    start: int
    end: int
    kind: str = "peak"  # "peak" | "domain"
    score: float = 0.0
    neg_log10_p: float = 0.0
    neg_log10_q: float = 0.0
    fold_enrichment: float = 0.0
    mean_posterior: float = 0.0
    """Mean HMM posterior of being enriched across the region."""
    subpeaks: List[SubPeak] = field(default_factory=list)
    name: str = ""

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def n_subpeaks(self) -> int:
        return len(self.subpeaks)

    @property
    def summit(self) -> int:
        """Best summit: the highest-scoring sub-peak, else the interval midpoint."""
        if self.subpeaks:
            return max(self.subpeaks, key=lambda s: s.score).summit
        return (self.start + self.end) // 2

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(f"empty/inverted region {self.chrom}:{self.start}-{self.end}")
        if self.kind not in ("peak", "domain"):
            raise ValueError(f"kind must be 'peak' or 'domain', got {self.kind!r}")


class NestedPeakSet:
    """An ordered collection of Regions plus the parameters that produced them.

    Every output format in FlexPeak is a projection of this object, so a tool
    that only understands narrowPeak still gets sensible results.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        regions: Optional[Sequence[Region]] = None,
        params: Optional[Dict] = None,
        chrom_sizes: Optional[Dict[str, int]] = None,
    ):
        self.regions: List[Region] = list(regions or [])
        self.params: Dict = dict(params or {})
        self.chrom_sizes: Dict[str, int] = dict(chrom_sizes or {})
        self.sort()

    # -- collection protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.regions)

    def __iter__(self) -> Iterator[Region]:
        return iter(self.regions)

    def __getitem__(self, i):
        return self.regions[i]

    def __repr__(self) -> str:
        return (
            f"<NestedPeakSet {len(self.regions)} regions "
            f"({self.n_domains} domains, {self.n_subpeaks} subpeaks), "
            f"{self.total_bp:,} bp>"
        )

    # -- summaries -----------------------------------------------------------

    @property
    def n_domains(self) -> int:
        return sum(1 for r in self.regions if r.kind == "domain")

    @property
    def n_subpeaks(self) -> int:
        return sum(r.n_subpeaks for r in self.regions)

    @property
    def total_bp(self) -> int:
        """Total genomic territory covered. The denominator of the over-merge check."""
        return sum(r.width for r in self.regions)

    def widths(self) -> List[int]:
        return [r.width for r in self.regions]

    def sort(self) -> "NestedPeakSet":
        self.regions.sort(key=lambda r: (r.chrom, r.start, r.end))
        for r in self.regions:
            r.subpeaks.sort(key=lambda s: s.start)
        return self

    def filter(self, min_qvalue_score: float = 0.0, min_width: int = 0,
               min_fold: float = 0.0) -> "NestedPeakSet":
        """Return a new set keeping regions passing all thresholds.

        ``min_qvalue_score`` is on the -log10(q) scale, so 2.0 means q <= 0.01.
        """
        kept = [
            r for r in self.regions
            if r.neg_log10_q >= min_qvalue_score
            and r.width >= min_width
            and r.fold_enrichment >= min_fold
        ]
        return NestedPeakSet(kept, self.params, self.chrom_sizes)

    def exclude(self, intervals: Sequence[tuple],
                min_overlap_frac: float = 0.0) -> "NestedPeakSet":
        """Drop regions overlapping ``intervals`` -- blacklist filtering.

        ``min_overlap_frac=0`` drops a region on any overlap at all, which is
        what ENCODE's own pipelines do (``bedtools intersect -v``): an artefact
        region contaminates the whole call it sits in, and a partly-artefactual
        peak is not worth trying to rescue by trimming.
        """
        if not intervals:
            return self
        keep = []
        for r in self.regions:
            ov = interval_overlap_bp([(r.chrom, r.start, r.end)], intervals)
            if ov > min_overlap_frac * r.width:
                continue
            keep.append(r)
        return NestedPeakSet(keep, self.params, self.chrom_sizes)

    def name_all(self, prefix: str = "flexpeak") -> "NestedPeakSet":
        """Assign stable hierarchical names: ``prefix_00001`` / ``prefix_00001.2``."""
        for i, r in enumerate(self.regions, 1):
            r.name = f"{prefix}_{i:05d}"
            for j, s in enumerate(r.subpeaks, 1):
                s.name = f"{r.name}.{j}"
        return self

    # -- output projections --------------------------------------------------

    def _header(self, kind: str, **extra) -> str:
        meta = {
            "format": "flexpeak-" + kind,
            "format_version": self.FORMAT_VERSION,
            "params": self.params,
        }
        meta.update(extra)
        return "# " + json.dumps(meta, sort_keys=True, default=str) + "\n"

    def to_narrowpeak(self, path=None) -> str:
        """ENCODE narrowPeak: one line per sub-peak (or per region if it has none)."""
        buf = io.StringIO()
        buf.write(self._header("narrowPeak"))
        for r in self.regions:
            if r.subpeaks:
                for s in r.subpeaks:
                    buf.write(
                        f"{s.chrom}\t{s.start}\t{s.end}\t{s.name or '.'}\t"
                        f"{int(min(1000, 10 * s.neg_log10_q))}\t.\t"
                        f"{s.fold_enrichment:.5f}\t{s.neg_log10_p:.5f}\t"
                        f"{s.neg_log10_q:.5f}\t{s.summit - s.start}\n"
                    )
            else:
                buf.write(
                    f"{r.chrom}\t{r.start}\t{r.end}\t{r.name or '.'}\t"
                    f"{int(min(1000, 10 * r.neg_log10_q))}\t.\t"
                    f"{r.fold_enrichment:.5f}\t{r.neg_log10_p:.5f}\t"
                    f"{r.neg_log10_q:.5f}\t{r.summit - r.start}\n"
                )
        return self._emit(buf.getvalue(), path)

    def to_broadpeak(self, path=None) -> str:
        """ENCODE broadPeak: one line per region (domain or standalone peak)."""
        buf = io.StringIO()
        buf.write(self._header("broadPeak"))
        for r in self.regions:
            buf.write(
                f"{r.chrom}\t{r.start}\t{r.end}\t{r.name or '.'}\t"
                f"{int(min(1000, 10 * r.neg_log10_q))}\t.\t"
                f"{r.fold_enrichment:.5f}\t{r.neg_log10_p:.5f}\t{r.neg_log10_q:.5f}\n"
            )
        return self._emit(buf.getvalue(), path)

    def to_bed12(self, path=None) -> str:
        """BED12 where each block is a sub-peak -- renders the nesting in IGV/UCSC."""
        buf = io.StringIO()
        for r in self.regions:
            subs = r.subpeaks or [
                SubPeak(r.chrom, r.start, r.end, r.summit, r.score)
            ]
            sizes = ",".join(str(s.width) for s in subs)
            starts = ",".join(str(s.start - r.start) for s in subs)
            buf.write(
                f"{r.chrom}\t{r.start}\t{r.end}\t{r.name or '.'}\t"
                f"{int(min(1000, 10 * r.neg_log10_q))}\t.\t{r.start}\t{r.end}\t"
                f"{'150,50,50' if r.kind == 'domain' else '50,50,150'}\t"
                f"{len(subs)}\t{sizes}\t{starts}\n"
            )
        return self._emit(buf.getvalue(), path)

    def to_nested_tsv(self, path=None) -> str:
        """Native format: parent/child link preserved explicitly."""
        buf = io.StringIO()
        # Only the nested projection round-trips, so it is the one that has to
        # carry chrom_sizes: genome_fraction is meaningless without them.
        buf.write(self._header("nested", chrom_sizes=self.chrom_sizes))
        buf.write(
            "chrom\tstart\tend\tname\tkind\tparent\tsummit\tfold_enrichment\t"
            "neg_log10_p\tneg_log10_q\tn_subpeaks\tposterior\tpersistence\n"
        )
        for r in self.regions:
            buf.write(
                f"{r.chrom}\t{r.start}\t{r.end}\t{r.name or '.'}\t{r.kind}\t.\t"
                f"{r.summit}\t{r.fold_enrichment:.5f}\t{r.neg_log10_p:.5f}\t"
                f"{r.neg_log10_q:.5f}\t{r.n_subpeaks}\t{r.mean_posterior:.5f}\t.\n"
            )
            for s in r.subpeaks:
                buf.write(
                    f"{s.chrom}\t{s.start}\t{s.end}\t{s.name or '.'}\tsubpeak\t"
                    f"{r.name or '.'}\t{s.summit}\t{s.fold_enrichment:.5f}\t"
                    f"{s.neg_log10_p:.5f}\t{s.neg_log10_q:.5f}\t0\t.\t{s.persistence}\n"
                )
        return self._emit(buf.getvalue(), path)

    @staticmethod
    def _emit(text: str, path) -> str:
        if path is not None:
            with _open_out(path) as fh:
                fh.write(text)
        return text

    def write_all(self, prefix: str) -> Dict[str, str]:
        """Write every projection, MACS-style, and return the paths."""
        paths = {
            "narrowPeak": f"{prefix}_peaks.narrowPeak",
            "broadPeak": f"{prefix}_peaks.broadPeak",
            # BED12 content, but a plain .bed extension: genome browsers and
            # most interval tools dispatch on the extension and do not know
            # ".bed12".  The key stays "bed12" because the *format* is what it
            # is; only the filename changes.
            "bed12": f"{prefix}_nested.bed",
            "nested": f"{prefix}_peaks.nested.tsv",
        }
        self.to_narrowpeak(paths["narrowPeak"])
        self.to_broadpeak(paths["broadPeak"])
        self.to_bed12(paths["bed12"])
        self.to_nested_tsv(paths["nested"])
        return paths

    # -- interval utilities used by tuning and evaluation --------------------

    def intervals(self) -> List[tuple]:
        return [(r.chrom, r.start, r.end) for r in self.regions]

    def jaccard(self, other: "NestedPeakSet") -> float:
        """Base-pair Jaccard against another set. Used by the tuning objective."""
        return interval_jaccard(self.intervals(), other.intervals())

    def to_dict(self) -> Dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "params": self.params,
            "chrom_sizes": self.chrom_sizes,
            "regions": [asdict(r) for r in self.regions],
        }


def _merge_sorted(intervals: Sequence[tuple]) -> Dict[str, List[tuple]]:
    by_chrom: Dict[str, List[tuple]] = {}
    for chrom, start, end in intervals:
        by_chrom.setdefault(chrom, []).append((start, end))
    for chrom, ivs in by_chrom.items():
        ivs.sort()
        merged = []
        for s, e in ivs:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        by_chrom[chrom] = merged
    return by_chrom


def interval_overlap_bp(a: Sequence[tuple], b: Sequence[tuple]) -> int:
    """Total overlapping base pairs between two interval sets."""
    A, B = _merge_sorted(a), _merge_sorted(b)
    total = 0
    for chrom in set(A) & set(B):
        ia = ib = 0
        xs, ys = A[chrom], B[chrom]
        while ia < len(xs) and ib < len(ys):
            lo = max(xs[ia][0], ys[ib][0])
            hi = min(xs[ia][1], ys[ib][1])
            if hi > lo:
                total += hi - lo
            if xs[ia][1] < ys[ib][1]:
                ia += 1
            else:
                ib += 1
    return total


def interval_total_bp(a: Sequence[tuple]) -> int:
    return sum(e - s for ivs in _merge_sorted(a).values() for s, e in ivs)


def interval_jaccard(a: Sequence[tuple], b: Sequence[tuple]) -> float:
    inter = interval_overlap_bp(a, b)
    union = interval_total_bp(a) + interval_total_bp(b) - inter
    return inter / union if union > 0 else 0.0


def read_nested_tsv(path) -> "NestedPeakSet":
    """Read back a file written by :meth:`NestedPeakSet.to_nested_tsv`.

    The nested TSV is the only projection that preserves the parent/child link,
    so it is the only one that round-trips.  This exists so QC and figures can
    be re-run against a call set months later without re-calling peaks.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    params: Dict = {}
    chrom_sizes: Dict[str, int] = {}
    header_sizes: Dict[str, int] = {}
    regions: List[Region] = []
    by_name: Dict[str, Region] = {}
    last_region: Optional[Region] = None

    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                try:
                    meta = json.loads(line[1:])
                except ValueError:
                    continue
                params = meta.get("params") or {}
                hdr = meta.get("chrom_sizes")
                if isinstance(hdr, dict):
                    header_sizes.update({c: int(v) for c, v in hdr.items()})
                continue
            if not line.strip() or line.startswith("chrom\t"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 13:
                raise ValueError(f"{path}: expected 13 columns, got {len(f)}")
            chrom, start, end, name, kind, parent = f[0], int(f[1]), int(f[2]), f[3], f[4], f[5]
            summit, fold, nlp, nlq = int(f[6]), float(f[7]), float(f[8]), float(f[9])
            if kind == "subpeak":
                sub = SubPeak(chrom=chrom, start=start, end=end, summit=summit,
                              neg_log10_p=nlp, neg_log10_q=nlq, fold_enrichment=fold,
                              persistence=int(f[12]) if f[12] not in (".", "") else 0,
                              name=name)
                # An unnamed set writes "." for every parent, so fall back to
                # the region the sub-peak was written under -- the writer always
                # emits a region immediately before its own sub-peaks.
                owner = by_name.get(parent) if parent != "." else last_region
                if owner is None:
                    raise ValueError(f"{path}: sub-peak {name} references unknown "
                                     f"parent {parent!r}")
                owner.subpeaks.append(sub)
            else:
                r = Region(chrom=chrom, start=start, end=end, kind=kind,
                           neg_log10_p=nlp, neg_log10_q=nlq, fold_enrichment=fold,
                           mean_posterior=float(f[11]) if f[11] not in (".", "") else 0.0,
                           name=name)
                regions.append(r)
                last_region = r
                if name != ".":
                    if name in by_name:
                        raise ValueError(f"{path}: duplicate region name {name!r}; "
                                         "sub-peaks cannot be attached unambiguously")
                    by_name[name] = r
            chrom_sizes.setdefault(chrom, 0)
            chrom_sizes[chrom] = max(chrom_sizes[chrom], end)

    # Prefer the true chromosome sizes the header carried; the running maximum
    # above is only a lower bound on each chromosome, and genome_fraction --
    # the territory denominator -- is wrong if it is used instead.
    return NestedPeakSet(regions, params, header_sizes or chrom_sizes)


def read_bed_intervals(path) -> List[tuple]:
    """Read ``(chrom, start, end)`` from a BED file, plain or gzipped.

    Deliberately minimal: blacklists and reference peak sets are the only things
    read this way, and only their coordinates matter.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    out: List[tuple] = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")) or not line.strip():
                continue
            f = line.split("\t")
            if len(f) < 3:
                raise ValueError(f"{path}: expected at least 3 BED columns")
            out.append((f[0], int(f[1]), int(f[2])))
    return out
