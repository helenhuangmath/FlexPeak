"""QC metrics (DESIGN.md section 5).

Deliberately a subset for v0: the library/enrichment/peak statistics that can be
computed from binned coverage plus a call set.  The HTML report and the
replicate/IDR section land in P4.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..peaks import NestedPeakSet
from ..signal.coverage import Coverage

__all__ = ["qc_metrics", "format_qc"]


def _gini(x) -> float:
    """Coverage inequality. Near 0 = flat (no enrichment), near 1 = highly focal."""
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * (idx * x).sum() / (n * x.sum()) - (n + 1) / n)


def _autocorrelation_length(x, max_lag: int = 400) -> int:
    """Lag (in bins) at which autocorrelation first drops below 1/e.

    A direct read on whether the signal is punctate or domain-like, and the
    reason a mark's natural scale can be inferred without being told.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0 or x.size < 10:
        return 0
    max_lag = int(min(max_lag, x.size // 2))
    thresh = 1.0 / np.e
    for lag in range(1, max_lag):
        r = float(np.dot(x[:-lag], x[lag:])) / denom
        if r < thresh:
            return lag
    return max_lag


def _frip(cov: Coverage, peaks: NestedPeakSet) -> float:
    """Fraction of signal in peaks."""
    total = cov.total_signal
    if total <= 0:
        return 0.0
    bs = cov.bin_size
    inside = 0.0
    for r in peaks:
        a = cov.bins.get(r.chrom)
        if a is None:
            continue
        s = min(r.start // bs, a.size)
        e = min(r.end // bs, a.size)
        inside += float(a[s:e].sum())
    return float(inside / total)


def _bimodality(widths: List[int]) -> float:
    """Sarle's bimodality coefficient on log widths.

    Above ~0.555 suggests two width populations -- i.e. a genuinely mixed mark
    with both focal peaks and broad domains (DESIGN.md section 5).
    """
    if len(widths) < 8:
        return 0.0
    x = np.log10(np.asarray(widths, dtype=np.float64))
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    z = (x - x.mean()) / s
    n = x.size
    g1 = float((z ** 3).mean())
    g2 = float((z ** 4).mean() - 3.0)
    denom = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom <= 0:
        return 0.0
    return float((g1 * g1 + 1.0) / denom)


def qc_metrics(
    treatment: Coverage,
    peaks: Optional[NestedPeakSet] = None,
    control: Optional[Coverage] = None,
    background=None,
    segment_bin_size: Optional[int] = None,
) -> Dict:
    """Compute QC metrics and a coarse quality grade."""
    arr = treatment.concat()
    total = float(arr.sum())
    m: Dict = {
        "source": treatment.source,
        "bin_size": treatment.bin_size,
        "n_bins": int(arr.size),
        "n_chroms": len(treatment.bins),
        "total_signal": total,
        "mean_signal": float(arr.mean()) if arr.size else 0.0,
        "n_fragments": treatment.n_fragments,
        "fragment_length": treatment.fragment_length,
        "is_count_data": treatment.is_count,
        "zero_bin_fraction": float((arr == 0).mean()) if arr.size else 0.0,
        "gini": _gini(arr),
        "autocorrelation_bins": _autocorrelation_length(arr),
        "nsc": treatment.meta.get("nsc"),
        "paired": treatment.meta.get("paired"),
    }
    m["autocorrelation_bp"] = m["autocorrelation_bins"] * treatment.bin_size

    if background is not None:
        m["dispersion_a0"] = background.a0
        m["dispersion_a1"] = background.a1
        m["used_control"] = background.used_control
        m["dispersion_at_mean"] = float(
            background.alpha(np.array([max(background.genome_mean, 1e-9)]))[0]
        )
        meta = getattr(background, "meta", None) or {}
        m["background_excluded_fraction"] = meta.get("excluded_fraction")
        m["background_exclude_fold"] = meta.get("exclude_fold")

    if peaks is not None:
        widths = peaks.widths()
        genome = sum(treatment.chrom_sizes.values()) or 1
        m.update({
            "n_regions": len(peaks),
            "n_domains": peaks.n_domains,
            "n_subpeaks": peaks.n_subpeaks,
            "total_peak_bp": peaks.total_bp,
            "genome_fraction": peaks.total_bp / float(genome),
            "median_width": float(np.median(widths)) if widths else 0.0,
            "max_width": int(max(widths)) if widths else 0,
            "frip": _frip(treatment, peaks),
            "width_bimodality": _bimodality(widths),
        })
        seg = segment_bin_size or (peaks.params or {}).get("segment_bin_size")
        if seg:
            m["segment_bin_size"] = int(seg)
            # Regions that are exactly one segmentation bin wide are not domains
            # the model joined up -- they are single blocks, and their width is
            # the bin size showing through.  A call set where that is the
            # typical outcome has width by quantisation, and reporting a median
            # width without saying so would be misleading.
            m["single_bin_fraction"] = float(
                np.mean(np.asarray(widths, dtype=np.float64) <= int(seg))
            )

    m["grade"], m["grade_reasons"] = _grade(m)
    return m


def _grade(m: Dict):
    """Coarse A-D grade with explicit reasons.

    QC as an explanation of the caller's behaviour, not a wall of numbers.
    """
    score = 0
    reasons: List[str] = []

    frip = m.get("frip")
    if frip is not None:
        if frip >= 0.20:
            score += 2
        elif frip >= 0.05:
            score += 1
            reasons.append(
                f"FRiP {frip:.3f} is modest (>=0.05 acceptable, >=0.20 good)")
        else:
            reasons.append(f"FRiP {frip:.3f} is low: weak enrichment or failed IP")

    gini = m.get("gini", 0.0)
    if gini >= 0.6:
        score += 2
    elif gini >= 0.35:
        score += 1
        reasons.append(f"coverage Gini {gini:.2f}: enrichment present but diffuse")
    else:
        reasons.append(f"coverage Gini {gini:.2f}: signal is nearly flat")

    disp = m.get("dispersion_at_mean")
    if disp is not None and disp > 0.5:
        reasons.append(
            f"background dispersion alpha={disp:.2f} is high: noisy library, "
            f"the NB model is doing real work here (a Poisson caller would "
            f"report inflated significance)")

    sbf = m.get("single_bin_fraction")
    if sbf is not None and sbf >= 0.5 and m.get("segment_bin_size", 0) > 1000:
        reasons.append(
            f"{sbf:.0%} of regions are one segmentation bin "
            f"({m['segment_bin_size']:,} bp) or narrower: their width is the "
            f"segmentation scale showing through, not domains the model joined "
            f"up. Treat the median width as an upper bound and try a finer "
            f"--segment-bin-size")

    excl = m.get("background_excluded_fraction")
    if excl is not None and excl > 0.5:
        reasons.append(
            f"{excl:.0%} of the genome was held out of the background fit as "
            f"apparent signal: the local-background windows are probably too "
            f"narrow for this mark, and what is left is not a background")

    zero = m.get("zero_bin_fraction")
    if zero is not None and zero > 0.9:
        reasons.append(f"{zero:.1%} of bins are empty: consider a larger bin size")

    bim = m.get("width_bimodality")
    if bim is not None and bim >= 0.555:
        reasons.append(
            f"peak-width bimodality {bim:.2f}: this looks like a MIXED mark "
            f"(both focal peaks and broad domains) -- nested output is "
            f"meaningful here")

    grade = "A" if score >= 5 else "B" if score >= 3 else "C" if score >= 1 else "D"
    return grade, reasons


_ROWS = (
    ("source", "source"),
    ("n_fragments", "fragments"),
    ("fragment_length", "fragment length (bp)"),
    ("paired", "paired-end"),
    ("bin_size", "bin size (bp)"),
    ("n_bins", "bins"),
    ("mean_signal", "mean signal/bin"),
    ("zero_bin_fraction", "empty bins"),
    ("gini", "coverage Gini"),
    ("autocorrelation_bp", "autocorrelation length (bp)"),
    ("nsc", "NSC"),
    ("dispersion_at_mean", "NB dispersion at mean"),
    ("used_control", "control used"),
    ("background_exclude_fold", "background exclude fold"),
    ("background_excluded_fraction", "  genome held out as signal"),
    ("n_regions", "regions"),
    ("n_domains", "  domains"),
    ("n_subpeaks", "  subpeaks"),
    ("segment_bin_size", "segmentation bin (bp)"),
    ("single_bin_fraction", "  regions <= 1 segment bin"),
    ("median_width", "median width (bp)"),
    ("max_width", "max width (bp)"),
    ("genome_fraction", "genome fraction"),
    ("frip", "FRiP"),
    ("width_bimodality", "width bimodality"),
)


def format_qc(m: Dict) -> str:
    """Plain-text QC report."""
    lines = ["FlexPeak QC", "=" * 60]
    for key, label in _ROWS:
        v = m.get(key)
        if v is None:
            continue
        if isinstance(v, float):
            txt = f"{v:,.4f}"
        elif isinstance(v, int):
            txt = f"{v:,}"
        else:
            txt = str(v)
        lines.append("  " + f"{label:<28}" + " " + txt)
    lines.append("")
    lines.append("  QUALITY GRADE: " + m.get("grade", "?"))
    for r in m.get("grade_reasons", []):
        lines.append("    - " + r)
    return "\n".join(lines)
