"""Basic statistical figures for a call set.

These are the descriptive figures you want in front of you the moment a run
finishes: how many peaks, how wide, how enriched, how significant, and where
they sit in the genome.  They describe *this* call set -- they are not a
benchmark and make no claim about correctness (for correctness against known
truth see ``examples/figures.py``, which runs on simulated data).

Everything is drawn from a :class:`~flexpeak.peaks.NestedPeakSet`, so the same
figures come out of a fresh run or of a nested TSV read back off disk months
later.  All output is set in Arial with black text (vector PDF by default) (see :mod:`flexpeak.plots.style`).

    from flexpeak.plots import peak_stat_figures
    peak_stat_figures(peaks, outdir="figures", prefix="H3K27ac")
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..peaks import NestedPeakSet
from .style import (
    FILL, FS_ANNOT, FS_LABEL, FS_NOTE, FS_SUB, FS_TICK, FS_TITLE, INK, INK2,
    MUTED, S1, S2, S3, SURFACE, TITLE_PAD, apply_style, finish, footnote, savefig,
)

__all__ = ["peak_stats", "peak_stat_figures", "format_stats"]


# -- numbers ----------------------------------------------------------------

def _chrom_key(c: str):
    """Natural chromosome order: chr1 < chr2 < ... < chr10 < chrX < chrUn_x."""
    s = c[3:] if c.lower().startswith("chr") else c
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


def _call_param(peaks: NestedPeakSet, key: str, default):
    call = (peaks.params or {}).get("call") or {}
    v = call.get(key, default)
    return default if v is None else v


def _finite(a: np.ndarray) -> np.ndarray:
    """Drop non-finite values.  ``-log10 q`` is infinite for q == 0."""
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def _cap_infinite(a: np.ndarray) -> np.ndarray:
    """Replace +inf with the largest finite value, so a histogram still bins.

    A q-value that underflows to 0 gives ``-log10 q = inf``.  Dropping those
    points would silently delete the *most* significant peaks from the figure,
    so they are pinned to the top of the axis and the count is reported in the
    caption instead.
    """
    a = np.asarray(a, dtype=np.float64)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return np.zeros_like(a)
    return np.where(np.isfinite(a), a, fin.max())


def _log_bins(vals: np.ndarray, n: int = 34):
    """Log-spaced bin edges that survive a degenerate (single-value) input."""
    v = _finite(vals)
    v = v[v > 0]
    if v.size == 0:
        return np.linspace(0.0, 1.0, n)
    lo, hi = float(v.min()) * 0.7, float(v.max()) * 1.4
    if not hi > lo:
        lo, hi = lo * 0.5, max(hi * 2.0, lo * 4.0)
    return np.logspace(np.log10(lo), np.log10(hi), n)


def _quantiles(vals) -> Dict[str, float]:
    v = _finite(np.asarray(vals, dtype=np.float64))
    if v.size == 0:
        return {"n": 0, "min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0,
                "mean": 0.0, "max": 0.0}
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "q25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "q75": float(np.percentile(v, 75)),
        "mean": float(v.mean()),
        "max": float(v.max()),
    }


def peak_stats(peaks: NestedPeakSet) -> Dict:
    """Descriptive statistics for a call set, as a plain dict.

    Every number drawn on a figure is read from here, so the figures and any
    table you build cannot disagree.
    """
    regions = list(peaks)
    widths = np.array([r.width for r in regions], dtype=np.float64)
    sub_widths = np.array([s.width for r in regions for s in r.subpeaks],
                          dtype=np.float64)
    folds = np.array([r.fold_enrichment for r in regions], dtype=np.float64)
    qs = np.array([r.neg_log10_q for r in regions], dtype=np.float64)
    n_sub_per = np.array([r.n_subpeaks for r in regions], dtype=np.float64)

    genome = float(sum(peaks.chrom_sizes.values())) or float("nan")
    per_chrom: Dict[str, Dict[str, float]] = {}
    for r in regions:
        d = per_chrom.setdefault(r.chrom, {"n": 0, "bp": 0, "n_domains": 0,
                                           "n_subpeaks": 0})
        d["n"] += 1
        d["bp"] += r.width
        d["n_domains"] += int(r.kind == "domain")
        d["n_subpeaks"] += r.n_subpeaks
    for c, d in per_chrom.items():
        size = peaks.chrom_sizes.get(c, 0)
        d["chrom_size"] = size
        d["fraction"] = d["bp"] / size if size else 0.0

    return {
        "n_regions": len(regions),
        "n_domains": peaks.n_domains,
        "n_focal_peaks": len(regions) - peaks.n_domains,
        "n_subpeaks": peaks.n_subpeaks,
        "n_regions_with_subpeaks": int((n_sub_per > 0).sum()),
        "total_peak_bp": int(peaks.total_bp),
        "genome_bp": int(genome) if np.isfinite(genome) else 0,
        "genome_fraction": (peaks.total_bp / genome) if np.isfinite(genome) else 0.0,
        "n_chroms": len(per_chrom),
        "width": _quantiles(widths),
        "subpeak_width": _quantiles(sub_widths),
        "fold_enrichment": _quantiles(folds),
        "neg_log10_q": _quantiles(qs),
        "n_infinite_q": int((~np.isfinite(qs)).sum()),
        "subpeaks_per_region": _quantiles(n_sub_per),
        "per_chrom": per_chrom,
        "domain_min_width": _call_param(peaks, "domain_min_width", 2000),
        "min_fold": _call_param(peaks, "min_fold", 0.0),
        "qvalue": _call_param(peaks, "qvalue", 0.05),
        "treatment": (peaks.params or {}).get("treatment"),
        "approximate_pvalues": bool((peaks.params or {}).get("approximate_pvalues")),
    }


_SUMMARY_ROWS = (
    ("regions called", lambda s: f"{s['n_regions']:,}"),
    ("  broad domains", lambda s: f"{s['n_domains']:,}"),
    ("  focal peaks", lambda s: f"{s['n_focal_peaks']:,}"),
    ("nested sub-peaks", lambda s: f"{s['n_subpeaks']:,}"),
    ("chromosomes with peaks", lambda s: f"{s['n_chroms']:,}"),
    ("total peak territory", lambda s: f"{s['total_peak_bp'] / 1e6:,.2f} Mb"),
    ("genome fraction", lambda s: f"{s['genome_fraction']:.2%}"),
    ("median width", lambda s: f"{s['width']['median']:,.0f} bp"),
    ("width IQR", lambda s: f"{s['width']['q25']:,.0f}-{s['width']['q75']:,.0f} bp"),
    ("max width", lambda s: f"{s['width']['max'] / 1000:,.1f} kb"),
    ("median fold enrichment", lambda s: f"{s['fold_enrichment']['median']:.2f}"),
    ("median -log10 q", lambda s: f"{s['neg_log10_q']['median']:.2f}"),
)


def format_stats(s: Dict) -> str:
    """Plain-text version of the summary panel, for a log or a sidecar file."""
    lines = ["FlexPeak peak statistics", "=" * 60]
    if s.get("treatment"):
        lines.append(f"  {'source':<26} {s['treatment']}")
    for label, fn in _SUMMARY_ROWS:
        lines.append(f"  {label:<26} {fn(s)}")
    if s["subpeak_width"]["n"]:
        lines.append(f"  {'median sub-peak width':<26} "
                     f"{s['subpeak_width']['median']:,.0f} bp")
    if s["approximate_pvalues"]:
        lines.append("")
        lines.append("  NOTE: input was not raw counts; p/q-values are approximate.")
    return "\n".join(lines)


# -- figures ----------------------------------------------------------------

def _plain_log(ax, axis: str = "x"):
    """Label a log axis 1000 / 10000 rather than 10^3 / 10^4.

    Widths and fold enrichments are read as numbers, not as exponents.  Minor
    ticks are labelled only when the axis spans less than ~2 decades; over a
    wider range they would collide.  Call this after the limits are settled.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axobj = ax.xaxis if axis == "x" else ax.yaxis
    lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
    lo, hi = max(float(lo), 1e-12), max(float(hi), 1e-12)
    decades = np.log10(hi) - np.log10(lo)
    fmt = FuncFormatter(lambda v, _pos: f"{v:g}")
    axobj.set_major_formatter(fmt)
    if decades < 2.2:
        axobj.set_minor_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
        axobj.set_minor_formatter(fmt)
    else:
        axobj.set_minor_formatter(NullFormatter())


def _threshold_label(ax, x, text, y_frac: float = 0.99, ha: str = "left"):
    """Label a vertical threshold line."""
    ax.text(x, y_frac, text, transform=ax.get_xaxis_transform(), ha=ha,
            va="top", fontsize=FS_NOTE, color=INK, zorder=6)


def _annotate_quantiles(ax, q: Dict, unit: str = "bp"):
    """Median line plus an n / median / IQR readout in the corner."""
    if not q["n"]:
        return
    ax.axvline(q["median"], color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=5)
    fmt = (lambda v: f"{v:,.0f}") if unit == "bp" else (lambda v: f"{v:,.2f}")
    ax.text(0.985, 0.95,
            f"n = {q['n']:,}\nmedian {fmt(q['median'])} {unit}\n"
            f"IQR {fmt(q['q25'])}-{fmt(q['q75'])} {unit}",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_NOTE,
            color=INK, linespacing=1.5, zorder=6)


def fig_width_distribution(peaks, s, plt, path):
    """Peak width distribution, regions and sub-peaks on the same log axis."""
    widths = np.array([r.width for r in peaks], dtype=np.float64)
    subs = np.array([sp.width for r in peaks for sp in r.subpeaks], dtype=np.float64)
    thresh = float(s["domain_min_width"])

    both = np.concatenate([w for w in (widths, subs) if w.size]) if (
        widths.size or subs.size) else np.array([1.0])
    bins = _log_bins(both)

    n_panels = 2 if subs.size else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10.0, 3.2 * n_panels + 1.5),
                             sharex=True, gridspec_kw={"hspace": 0.30})
    axes = np.atleast_1d(axes)

    series = [(widths, S1, "called regions")]
    if subs.size:
        series.append((subs, S2, "nested sub-peaks"))

    for ax, (vals, color, label) in zip(axes, series):
        ax.hist(vals, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.8)
        ax.set_xscale("log")
        ax.axvline(thresh, color=MUTED, lw=1.2, ls=(0, (4, 3)))
        ax.grid(axis="x", visible=False)
        finish(ax, ylabel="count")
        # Series label upper-left, readout upper-right: the mode of a width
        # distribution sits left of centre, so neither corner is over a bar.
        ax.text(0.015, 0.95, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_ANNOT, color=INK, zorder=6)
        _annotate_quantiles(ax, _quantiles(vals), "bp")

    _threshold_label(axes[0], thresh, f" domain threshold ({thresh / 1000:g} kb)",
                     y_frac=0.62)
    _plain_log(axes[-1], "x")
    axes[-1].set_xlabel("width (bp, log scale)")
    axes[0].set_title("Peak width distribution", loc="left", pad=TITLE_PAD)
    axes[0].text(0, 1.012,
                 f"{s['n_domains']:,} domains and {s['n_focal_peaks']:,} focal "
                 f"peaks, plus {s['n_subpeaks']:,} sub-peaks nested inside them",
                 transform=axes[0].transAxes, fontsize=FS_SUB, color=INK,
                 va="bottom")
    footnote(axes[-1],
             "Width is plotted on a log axis because a mixed mark spans orders "
             "of magnitude; on a linear axis every focal peak collapses into "
             "the first bin.\n"
             "Two separated modes here mean the sample carries both focal peaks "
             "and broad domains -- the case the nested output exists for.",
             y=-0.30)
    return savefig(fig, path)


def fig_peak_counts(peaks, s, plt, path):
    """How many peaks, and how much genome they occupy."""
    fig = plt.figure(figsize=(12.6, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1],
                          wspace=0.28, hspace=0.75)
    ax = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    labels = ["all\nregions", "broad\ndomains", "focal\npeaks", "nested\nsub-peaks"]
    vals = [s["n_regions"], s["n_domains"], s["n_focal_peaks"], s["n_subpeaks"]]
    colors = [INK2, S1, S3, S2]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    top = max(vals) if max(vals) else 1
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, f"{v:,}",
                ha="center", va="bottom", fontsize=FS_ANNOT, color=INK)
    ax.set_ylim(0, top * 1.16)
    ax.grid(axis="x", visible=False)
    finish(ax, ylabel="count",
           title="Peaks called",
           subtitle="a region is a broad domain or a focal peak;\n"
                    "sub-peaks nest inside regions")

    def _stacked(a, parts, total_label):
        """One horizontal 100%-style bar, in Mb, with the parts labelled."""
        left = 0.0
        for value, color, label in parts:
            a.barh([0], [value], left=[left], color=color, height=0.46,
                   label=label)
            left += value
        a.set_yticks([])
        a.set_ylim(-0.5, 0.85)
        a.set_xlim(0, max(left, 1e-9))
        a.grid(visible=False)
        for side in ("top", "right", "bottom", "left"):
            a.spines[side].set_visible(True)
            a.spines[side].set_color(INK)
            a.spines[side].set_linewidth(0.8)
        a.text(0, 0.30, total_label, fontsize=FS_NOTE, color=INK, va="bottom")
        a.set_xlabel("Mb")

    covered = s["total_peak_bp"] / 1e6
    rest = max(s["genome_bp"] / 1e6 - covered, 0.0)
    _stacked(ax2,
             [(covered, S1, "peaks"), (rest, FILL, "rest of genome")],
             f"{covered:,.1f} Mb in peaks - {s['genome_fraction']:.2%} of the "
             f"{s['genome_bp'] / 1e6:,.0f} Mb assayed")
    ax2.set_title("Genome territory", loc="left", pad=26)

    dom_bp = sum(r.width for r in peaks if r.kind == "domain") / 1e6
    focal_bp = max(covered - dom_bp, 0.0)
    _stacked(ax3,
             [(dom_bp, S1, "domains"), (focal_bp, S3, "focal peaks")],
             f"{dom_bp:,.1f} Mb in domains, {focal_bp:,.1f} Mb in focal peaks")
    ax3.set_title("Composition of the called bp", loc="left", pad=26)
    ax3.legend(loc="lower left", bbox_to_anchor=(0, -0.95), ncol=2,
               labelcolor=INK, fontsize=FS_NOTE)
    ax2.legend(loc="lower left", bbox_to_anchor=(0, -0.95), ncol=2,
               labelcolor=INK, fontsize=FS_NOTE)

    footnote(ax,
             "Territory is the denominator for every recall-style claim: calling "
             "more genome always buys more overlap, so read it next to the counts, "
             "never on its own.\n"
             "A handful of domains holding most of the called bp is the "
             "over-merge signature -- the lower-right bar is where it shows up.",
             y=-0.14)
    return savefig(fig, path)


def fig_per_chromosome(peaks, s, plt, path):
    """Peak count and peak density per chromosome."""
    chroms = sorted(s["per_chrom"], key=_chrom_key)
    counts = [s["per_chrom"][c]["n"] for c in chroms]
    fracs = [s["per_chrom"][c]["fraction"] * 100 for c in chroms]
    x = np.arange(len(chroms))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(max(9.0, 0.45 * len(chroms) + 4.0), 8.2),
                                  sharex=True, gridspec_kw={"hspace": 0.20})
    ax.bar(x, counts, color=S1, width=0.68)
    ax.grid(axis="x", visible=False)
    finish(ax, ylabel="regions called")

    ax2.bar(x, fracs, color=S2, width=0.68)
    mean_frac = s["genome_fraction"] * 100
    ax2.axhline(mean_frac, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax2.text(0.998, mean_frac, f"genome-wide {mean_frac:.2f}% ",
             transform=ax2.get_yaxis_transform(), fontsize=FS_NOTE, color=INK,
             va="bottom", ha="right", zorder=6)
    ax2.grid(axis="x", visible=False)
    finish(ax2, ylabel="% of chromosome in peaks")

    ax2.set_xticks(x)
    ax2.set_xticklabels(chroms, rotation=90 if len(chroms) > 12 else 0,
                        fontsize=FS_TICK if len(chroms) <= 24 else FS_NOTE)
    ax.set_title("Peaks by chromosome", loc="left", pad=TITLE_PAD)
    ax.text(0, 1.012,
            f"{s['n_regions']:,} regions across {len(chroms)} chromosomes",
            transform=ax.transAxes, fontsize=FS_SUB, color=INK, va="bottom")
    footnote(ax2,
             "Counts follow chromosome length, so the lower panel -- fraction of "
             "each chromosome called -- is the one that shows real imbalance.\n"
             "A single chromosome far above the dashed genome-wide line usually "
             "means a copy-number or blacklist artefact, not biology.",
             y=-0.30 if len(chroms) <= 12 else -0.55)
    return savefig(fig, path)


def fig_enrichment(peaks, s, plt, path):
    """Fold enrichment and significance distributions."""
    folds = np.array([r.fold_enrichment for r in peaks], dtype=np.float64)
    qs = _cap_infinite(np.array([r.neg_log10_q for r in peaks], dtype=np.float64))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                  gridspec_kw={"wspace": 0.26})

    ax.hist(folds, bins=_log_bins(folds), color=S1, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xscale("log")
    min_fold = float(s["min_fold"])
    if min_fold > 0:
        ax.axvline(min_fold, color=MUTED, lw=1.2, ls=(0, (4, 3)))
        _threshold_label(ax, min_fold, f" min_fold {min_fold:g}", y_frac=0.72)
    ax.grid(axis="x", visible=False)
    finish(ax, xlabel="fold enrichment over background (log scale)", ylabel="regions")
    ax.set_title("Enrichment", loc="left", pad=12)
    _plain_log(ax, "x")
    _annotate_quantiles(ax, s["fold_enrichment"], "x")

    nb = 36
    ax2.hist(qs, bins=np.linspace(0, max(float(qs.max()) if qs.size else 1.0, 1.0), nb),
             color=S2, edgecolor=SURFACE, linewidth=0.8)
    cutoff = -np.log10(float(s["qvalue"])) if float(s["qvalue"]) > 0 else 0.0
    ax2.axvline(cutoff, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    _threshold_label(ax2, cutoff, f" FDR {float(s['qvalue']):g}", y_frac=0.72)
    ax2.grid(axis="x", visible=False)
    # Plain "log10": Arial has no U+2080/U+2081 subscript digits.
    finish(ax2, xlabel="-log10 q", ylabel="regions")
    ax2.set_title("Significance", loc="left", pad=12)
    _annotate_quantiles(ax2, s["neg_log10_q"], "")

    note = ("Both distributions are truncated by construction: nothing below "
            "min_fold or below the FDR cutoff was ever emitted,\n"
            "so the left edge of each panel is the threshold, not the data.")
    if s["n_infinite_q"]:
        note += (f"\n{s['n_infinite_q']:,} region(s) had q underflow to 0 "
                 f"(-log10 q = infinity) and are drawn at the right edge.")
    if s["approximate_pvalues"]:
        note += ("\nInput was not raw counts (bigWig), so these p/q-values are "
                 "approximate -- rank them, do not read them as exact.")
    footnote(ax, note, y=-0.26)
    return savefig(fig, path)


def fig_width_vs_enrichment(peaks, s, plt, path):
    """Are the wide calls the strong ones, or the weak ones?"""
    regions = list(peaks)
    w = np.array([r.width for r in regions], dtype=np.float64)
    f = np.array([max(r.fold_enrichment, 1e-3) for r in regions], dtype=np.float64)
    is_dom = np.array([r.kind == "domain" for r in regions], dtype=bool)

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    if w.size > 20_000:
        hb = ax.hexbin(w, f, xscale="log", yscale="log", gridsize=52,
                       cmap="Blues", mincnt=1, linewidths=0)
        cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("regions per cell", color=INK, fontsize=FS_LABEL)
        cb.outline.set_visible(True)
        cb.outline.set_edgecolor(INK)
        cb.outline.set_linewidth(0.8)
        density_note = "drawn as a density (hexbin): too many regions to plot individually"
    else:
        alpha = 0.55 if w.size < 4000 else 0.25
        ax.scatter(w[~is_dom], f[~is_dom], s=12, color=S3, alpha=alpha,
                   linewidths=0, label="focal peaks")
        ax.scatter(w[is_dom], f[is_dom], s=12, color=S1, alpha=alpha,
                   linewidths=0, label="broad domains")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="upper right", labelcolor=INK, markerscale=2.0)
        density_note = ""

    ax.axvline(float(s["domain_min_width"]), color=MUTED, lw=1.2, ls=(0, (4, 3)))
    if float(s["min_fold"]) > 0:
        ax.axhline(float(s["min_fold"]), color=MUTED, lw=1.2, ls=(0, (4, 3)))
    _plain_log(ax, "x")
    _plain_log(ax, "y")
    finish(ax, xlabel="region width (bp, log scale)",
           ylabel="fold enrichment (log scale)",
           title="Width against enrichment",
           subtitle=f"{s['n_regions']:,} regions"
                    + (f" -- {density_note}" if density_note else ""))
    footnote(ax,
             "The dashed lines are the domain-width threshold and min_fold. A "
             "cloud hugging the min_fold line is the shape to be suspicious of:\n"
             "those calls exist because the threshold let them through, not "
             "because the signal is strong.",
             y=-0.20)
    return savefig(fig, path)


def fig_cumulative_territory(peaks, s, plt, path):
    """How concentrated is the called territory in the widest peaks?"""
    w = np.sort(np.array([r.width for r in peaks], dtype=np.float64))[::-1]
    if w.size == 0:
        return None
    cum = np.cumsum(w) / w.sum()
    rank = np.arange(1, w.size + 1) / w.size

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 3)),
            label="equal widths (reference)")
    ax.plot(rank, cum, color=S1, lw=2.2, label="observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", labelcolor=INK)

    marks = []
    for frac in (0.01, 0.05, 0.10):
        i = max(int(np.ceil(frac * w.size)) - 1, 0)
        marks.append((frac, float(cum[i])))
        ax.plot([frac], [cum[i]], marker="o", color=S2, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.text(0.5, 0.28,
            "\n".join(f"top {f:.0%} of regions hold {c:.0%} of peak bp"
                      for f, c in marks),
            transform=ax.transAxes, fontsize=FS_ANNOT, color=INK, linespacing=1.7)

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    finish(ax, xlabel="regions, widest first",
           ylabel="cumulative share of called bp",
           title="Concentration of called territory",
           subtitle=f"{s['total_peak_bp'] / 1e6:,.1f} Mb across "
                    f"{s['n_regions']:,} regions")
    footnote(ax,
             "A curve pinned to the top-left says a handful of very wide calls "
             "own the territory -- often one over-merged domain, not many peaks.\n"
             "A curve near the diagonal says width is evenly spread.",
             y=-0.20)
    return savefig(fig, path)


def fig_subpeaks_per_region(peaks, s, plt, path):
    """How many focal maxima each region carries."""
    n = np.array([r.n_subpeaks for r in peaks], dtype=int)
    if n.size == 0 or n.max() == 0:
        return None

    top = int(min(n.max(), 20))
    edges = np.arange(-0.5, top + 1.5, 1.0)
    clipped = np.clip(n, 0, top)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.hist(clipped, bins=edges, color=S2, edgecolor=SURFACE, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(0, top + 1, max(1, top // 10)))
    ax.grid(axis="x", visible=False)
    _plain_log(ax, "y")
    finish(ax, xlabel="sub-peaks per region"
                      + (f" ({top}+ pooled in the last bar)" if n.max() > top else ""),
           ylabel="regions (log scale)",
           title="Sub-peak structure",
           subtitle=f"{s['n_regions_with_subpeaks']:,} of {s['n_regions']:,} "
                    f"regions carry at least one sub-peak "
                    f"({s['n_subpeaks']:,} sub-peaks total)")
    ax.text(0.985, 0.95,
            f"mean {n.mean():.2f} per region\nmax {n.max():,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_NOTE,
            color=INK, linespacing=1.5, zorder=6)
    footnote(ax,
             "The count axis is logarithmic: most regions carry zero or one "
             "sub-peak and would otherwise flatten the rest of the distribution.\n"
             "Many sub-peaks inside a flat broad domain is the known false "
             "sub-peak mode -- min_persistence is the knob that trades it "
             "against sensitivity.",
             y=-0.24)
    return savefig(fig, path)


def fig_summary(peaks, s, plt, path, title=None):
    """One page: the numbers plus the four distributions that matter most."""
    fig = plt.figure(figsize=(15.0, 9.0))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.28,
                          left=0.06, right=0.975, top=0.855, bottom=0.09)

    # -- numbers panel
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    y = 1.0
    for label, fn in _SUMMARY_ROWS:
        ax.text(0.0, y, label, fontsize=FS_NOTE, color=INK, va="top",
                transform=ax.transAxes)
        ax.text(1.0, y, fn(s), fontsize=FS_NOTE, color=INK, va="top", ha="right",
                transform=ax.transAxes)
        y -= 0.083
    ax.set_title("Summary", loc="left", pad=12)

    # -- width
    ax = fig.add_subplot(gs[0, 1])
    widths = np.array([r.width for r in peaks], dtype=np.float64)
    ax.hist(widths, bins=_log_bins(widths, 28), color=S1, edgecolor=SURFACE,
            linewidth=0.7)
    ax.set_xscale("log")
    ax.axvline(float(s["domain_min_width"]), color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.grid(axis="x", visible=False)
    _plain_log(ax, "x")
    finish(ax, xlabel="region width (bp)", ylabel="count")
    ax.set_title("Width", loc="left", pad=12)

    # -- fold
    ax = fig.add_subplot(gs[0, 2])
    folds = np.array([r.fold_enrichment for r in peaks], dtype=np.float64)
    ax.hist(folds, bins=_log_bins(folds, 28), color=S3, edgecolor=SURFACE,
            linewidth=0.7)
    ax.set_xscale("log")
    ax.grid(axis="x", visible=False)
    _plain_log(ax, "x")
    finish(ax, xlabel="fold enrichment", ylabel="count")
    ax.set_title("Enrichment", loc="left", pad=12)

    # -- significance
    ax = fig.add_subplot(gs[1, 0])
    qs = _cap_infinite(np.array([r.neg_log10_q for r in peaks], dtype=np.float64))
    ax.hist(qs, bins=np.linspace(0, max(float(qs.max()) if qs.size else 1.0, 1.0), 28),
            color=S2, edgecolor=SURFACE, linewidth=0.7)
    ax.grid(axis="x", visible=False)
    # Say so on the axis: the right-edge spike is q underflowing to zero, not a
    # real mode, and this panel has no room for a footnote.
    xlab = "-log10 q"
    if s["n_infinite_q"]:
        xlab += f"   ({s['n_infinite_q']:,} at q=0, pinned right)"
    finish(ax, xlabel=xlab, ylabel="count")
    ax.set_title("Significance", loc="left", pad=12)

    # -- per chromosome
    ax = fig.add_subplot(gs[1, 1:])
    chroms = sorted(s["per_chrom"], key=_chrom_key)
    x = np.arange(len(chroms))
    ax.bar(x, [s["per_chrom"][c]["n"] for c in chroms], color=S1, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(chroms, rotation=90 if len(chroms) > 12 else 0,
                       fontsize=FS_NOTE if len(chroms) > 24 else FS_TICK)
    ax.grid(axis="x", visible=False)
    finish(ax, ylabel="regions called")
    ax.set_title("By chromosome", loc="left", pad=12)

    fig.text(0.06, 0.965, title or "FlexPeak call statistics", ha="left",
             fontsize=FS_TITLE, color=INK, va="bottom")
    sub = s.get("treatment") or ""
    if sub:
        sub = os.path.basename(str(sub))
    extra = ("  -- bigWig input, p/q-values approximate"
             if s["approximate_pvalues"] else "")
    fig.text(0.06, 0.925, f"{sub}{extra}", ha="left", fontsize=FS_SUB,
             color=INK, va="bottom")
    return savefig(fig, path)


_FIGURES = (
    ("width_distribution", fig_width_distribution),
    ("peak_counts", fig_peak_counts),
    ("per_chromosome", fig_per_chromosome),
    ("enrichment", fig_enrichment),
    ("width_vs_enrichment", fig_width_vs_enrichment),
    ("cumulative_territory", fig_cumulative_territory),
    ("subpeaks_per_region", fig_subpeaks_per_region),
)


def peak_stat_figures(
    peaks: NestedPeakSet,
    outdir: str = ".",
    prefix: str = "flexpeak",
    title: Optional[str] = None,
    only: Optional[List[str]] = None,
    verbose: bool = False,
    fmt: str = "pdf",
) -> Dict[str, str]:
    """Write the basic statistics figures for ``peaks``.

    ``fmt`` is the file extension: ``"pdf"`` (default, vector and editable) or
    ``"png"`` for embedding in Markdown, where a PDF will not render.

    Returns a ``{figure name: path}`` map.  Figures that do not apply to this
    call set (sub-peak structure when nothing has sub-peaks) are skipped rather
    than written empty, and are simply absent from the returned map.
    """
    if len(peaks) == 0:
        raise ValueError("no regions in the call set: nothing to plot")

    plt = apply_style()
    s = peak_stats(peaks)
    os.makedirs(outdir, exist_ok=True)

    wanted = set(only) if only else None
    paths: Dict[str, str] = {}
    for name, fn in _FIGURES:
        if wanted is not None and name not in wanted:
            continue
        p = fn(peaks, s, plt, os.path.join(outdir, f"{prefix}_{name}.{fmt}"))
        if p:
            paths[name] = p
            if verbose:
                print(f"  wrote {p}")
    if wanted is None or "summary" in wanted:
        p = fig_summary(peaks, s, plt, os.path.join(outdir, f"{prefix}_summary.{fmt}"),
                        title=title)
        paths["summary"] = p
        if verbose:
            print(f"  wrote {p}")

    stats_path = os.path.join(outdir, f"{prefix}_stats.txt")
    with open(stats_path, "w") as fh:
        fh.write(format_stats(s) + "\n")
    paths["stats"] = stats_path
    return paths
