#!/usr/bin/env python3
"""Render the comparison figures from Benchmark/results/metrics.tsv.

    python Benchmark/run_benchmark.py     # first: produce metrics.tsv
    python Benchmark/make_figures.py      # then: render Benchmark/figures/

Style matches examples/figures.py (Arial, black text, outlined panels with no
grid lines, no bold, validated colourblind-safe palette).  Kept self-contained so the Benchmark directory can be moved or
shared on its own.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import OrderedDict
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.join(HERE, "figures")

SURFACE = "#ffffff"
INK = "#000000"   # all text and panel outlines
INK2 = "#52514e"
MUTED = "#8b8a85"
GRID = "#c9c7c1"
S1 = "#2a78d6"   # blue   - FlexPeak
S2 = "#eb6834"   # orange - MACS
S3 = "#1baf7a"   # aqua   - re-implemented baselines
NEUTRAL = "#6f6e69"  # ablation: a reference, not a competitor

FAMILY_STYLE = {
    "FlexPeak": (S1, "o"),
    "MACS": (S2, "s"),
    "baseline": (S3, "^"),
    "ablation": (NEUTRAL, "D"),
}

FS_TITLE = 15
FS_SUB = 12.5
FS_LABEL = 13
FS_TICK = 12
FS_LEGEND = 12.5
FS_ANNOT = 12
FS_NOTE = 11.5
TITLE_PAD = 34

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Arimo",
                        "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": "none",
    "savefig.facecolor": SURFACE, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": FS_TITLE, "axes.titleweight": "normal",
    "axes.labelsize": FS_LABEL, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "legend.labelcolor": INK,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "legend.frameon": False, "legend.fontsize": FS_LEGEND,
    "lines.linewidth": 2.0, "font.size": FS_TICK,
    "font.weight": "normal", "figure.titleweight": "normal",
    "figure.dpi": 200,
})

DATASET_ORDER = ["narrow", "broad", "mixed", "lowqual"]
DATASET_LABEL = {
    "narrow": "narrow\n(focal peaks)",
    "broad": "broad\n(domains)",
    "mixed": "mixed\n(both)",
    "lowqual": "low quality\n(weak, noisy)",
    "null": "null\n(no signal)",
}


def _finish(ax, xlabel=None, ylabel=None, title=None, subtitle=None):
    ax.set_facecolor("none")
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(False)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=TITLE_PAD if subtitle else 12)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=FS_SUB,
                color=INK, va="bottom")


def finalize(fig):
    """Black text, outlined unfilled panels, no grid lines -- enforced at save.

    White text is kept: it only sits on dark heatmap cells.  Mirrors
    flexpeak.plots.style.finalize; duplicated to keep Benchmark/ standalone.
    """
    from matplotlib.colors import to_hex
    from matplotlib.text import Text

    fig.patch.set_facecolor(SURFACE)
    for ax in fig.get_axes():
        if not ax.axison:
            continue
        ax.grid(False)
        ax.set_facecolor("none")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(INK)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(which="both", color=INK, labelcolor=INK)
    for t in fig.findobj(Text):
        if to_hex(t.get_color()) != "#ffffff":
            t.set_color(INK)
    return fig


def _save(fig, name):
    finalize(fig)
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, os.path.dirname(HERE))}")


def load_rows() -> List[Dict]:
    path = os.path.join(RESULTS, "metrics.tsv")
    if not os.path.exists(path):
        sys.exit(f"missing {path} -- run Benchmark/run_benchmark.py first")
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            for k, v in list(r.items()):
                if k in ("dataset", "method", "key", "family"):
                    continue
                r[k] = float(v) if v not in ("", None) else float("nan")
            rows.append(r)
    return rows


def method_order(rows) -> "OrderedDict[str, str]":
    """Methods in registry order, mapping key -> label."""
    out: "OrderedDict[str, str]" = OrderedDict()
    for r in rows:
        out.setdefault(r["key"], r["method"])
    return out


def family_of(rows) -> Dict[str, str]:
    return {r["key"]: r["family"] for r in rows}


def cell(rows, dataset, key, field):
    for r in rows:
        if r["dataset"] == dataset and r["key"] == key:
            return r[field]
    return float("nan")


def _legend_handles(families=None):
    names = {"FlexPeak": "FlexPeak", "MACS": "MACS2 / MACS3",
             "baseline": "re-implemented baseline", "ablation": "ablation"}
    return [plt.Line2D([], [], color=c, marker=m, linestyle="none",
                       markersize=11, markeredgecolor=SURFACE, label=names[f])
            for f, (c, m) in FAMILY_STYLE.items()
            if families is None or f in families]


def _panel_grid(dsets, title, subtitle):
    """2x2 panel grid with room reserved for a title block and a bottom legend.

    A 1x4 strip puts the figure title, the legend and the axis labels on top of
    each other once the type is this size; the grid keeps each panel square
    enough to read a scatter in.
    """
    nrow = 2 if len(dsets) > 2 else 1
    ncol = int(np.ceil(len(dsets) / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.2 * nrow),
                             sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(dsets):]:
        ax.set_visible(False)
    fig.subplots_adjust(top=0.86, bottom=0.15, hspace=0.42, wspace=0.12)
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=FS_TITLE, color=INK)
    fig.text(0.02, 0.935, subtitle, fontsize=FS_SUB, color=INK, va="bottom")
    return fig, axes[:len(dsets)]


# -- B1: F1 heatmap ---------------------------------------------------------

def fig_b1(rows):
    keys = method_order(rows)
    dsets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
    Z = np.array([[cell(rows, d, k, "bp_f1") for d in dsets] for k in keys])

    fig, ax = plt.subplots(figsize=(1.7 * len(dsets) + 7.5, 0.52 * len(keys) + 3.0))
    im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isnan(Z[i, j]):
                continue
            ax.text(j, i, f"{Z[i, j]:.0%}", ha="center", va="center",
                    fontsize=FS_ANNOT,
                    color="#ffffff" if Z[i, j] > 0.62 else INK)
    ax.set_xticks(range(len(dsets)), [DATASET_LABEL[d] for d in dsets])
    ax.set_yticks(range(len(keys)), list(keys.values()))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("base-pair F1", color=INK, fontsize=FS_LABEL)
    cb.outline.set_visible(True)
    cb.outline.set_edgecolor(INK)
    cb.outline.set_linewidth(0.8)
    _finish(ax, title="Base-pair F1 against known truth",
            subtitle="same simulated data and same control for every method; "
                     "each run at its own nominal FDR 5%")
    _save(fig, "B1_f1_heatmap.png")


# -- B2: precision vs recall ------------------------------------------------

def fig_b2(rows):
    dsets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
    fam = family_of(rows)
    keys = method_order(rows)

    fig, axes = _panel_grid(
        dsets, "Precision against recall, per dataset",
        "Top right is better. Points on the lower right buy recall with "
        "territory; points on the upper left are conservative.")

    for ax, d in zip(axes, dsets):
        for k in keys:
            p = cell(rows, d, k, "bp_precision")
            r = cell(rows, d, k, "bp_recall")
            if np.isnan(p) or np.isnan(r):
                continue
            color, marker = FAMILY_STYLE[fam[k]]
            ax.plot(r * 100, p * 100, marker=marker, color=color, markersize=11,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle="none")
        ax.set_xlim(-4, 104)
        ax.set_ylim(-4, 104)
        _finish(ax, xlabel="base-pair recall (%)", ylabel="base-pair precision (%)")
        ax.set_title(DATASET_LABEL[d].replace("\n", " "), loc="left", pad=12)

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    _save(fig, "B2_precision_recall.png")


# -- B3: fusion vs fragmentation -------------------------------------------

def fig_b3(rows):
    dsets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
    fam = family_of(rows)
    keys = method_order(rows)

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    seen = set()
    for d in dsets:
        for k in keys:
            fu = cell(rows, d, k, "fusion")
            fr = cell(rows, d, k, "fragmentation")
            if np.isnan(fu) or np.isnan(fr):
                continue
            color, marker = FAMILY_STYLE[fam[k]]
            ax.plot(fr * 100, fu * 100, marker=marker, color=color,
                    markersize=11, alpha=0.75, markeredgecolor=SURFACE,
                    markeredgewidth=1.4, linestyle="none")
            seen.add(fam[k])
    ax.set_xlim(-4, 104)
    ax.set_ylim(-2, max(12, ax.get_ylim()[1]))
    ax.annotate("one truth region split\nacross several calls",
                xy=(96, 1.2), xytext=(58, 6.5), fontsize=FS_NOTE, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.annotate("one call spanning\nseveral truth regions",
                xy=(6, 10.6), xytext=(16, 8.2), fontsize=FS_NOTE, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.legend(handles=_legend_handles(seen), loc="upper right", labelcolor=INK)
    _finish(ax, xlabel="fragmentation index (%)", ylabel="fusion index (%)",
            title="The two boundary failure modes, plotted against each other",
            subtitle="every method on every non-null dataset; the origin is "
                     "correct, both axes are errors")
    ax.text(0, -0.20,
            "MACS in narrow mode sits at the right edge on broad data: it "
            "recovers the territory but reports each domain as many peaks.\n"
            "Fixed-gap island merging moves up the y-axis as the window grows. "
            "Neither index alone is sufficient — a method can be good at one "
            "by being bad at the other.",
            transform=ax.transAxes, fontsize=FS_NOTE, color=INK, va="top",
            linespacing=1.5)
    _save(fig, "B3_fusion_vs_fragmentation.png")


# -- B4: specificity on the null -------------------------------------------

def fig_b4(rows):
    if not any(r["dataset"] == "null" for r in rows):
        print("  B4 skipped: no null dataset in results")
        return
    keys = method_order(rows)
    fam = family_of(rows)
    labels, terr, colors = [], [], []
    for k, label in keys.items():
        t = cell(rows, "null", k, "territory")
        if np.isnan(t):
            continue
        labels.append(label)
        terr.append(t * 100)
        colors.append(FAMILY_STYLE[fam[k]][0])

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.0, 0.52 * len(labels) + 3.4))
    ax.barh(y, terr, color=colors, height=0.62)
    for yy, v in zip(y, terr):
        ax.text(v + max(terr) * 0.012, yy, f"{v:.2f}%", va="center",
                fontsize=FS_ANNOT, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(terr) * 1.16)
    ax.grid(axis="y", visible=False)
    _finish(ax, xlabel="genome fraction called on data with no signal (%)",
            title="Specificity: calls on a null dataset",
            subtitle="input versus input; the correct answer is zero, so every "
                     "bar is entirely false positives")
    ax.text(0, -0.14 - 0.010 * len(labels),
            "The simulator draws background from the negative binomial family "
            "FlexPeak assumes, which favours FlexPeak here by construction.\n"
            "The comparison that does not depend on that assumption is "
            "FlexPeak against its own Poisson ablation: identical code, one "
            "distribution swapped.",
            transform=ax.transAxes, fontsize=FS_NOTE, color=INK, va="top",
            linespacing=1.5)
    _save(fig, "B4_null_specificity.png")


# -- B5: recall against territory ------------------------------------------

def fig_b5(rows):
    dsets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
    fam = family_of(rows)
    keys = method_order(rows)

    fig, axes = _panel_grid(
        dsets, "Recall against the territory it cost",
        "Upper left is better. Recall alone can always be raised by calling "
        "more genome, so it is only interpretable next to territory.")

    for ax, d in zip(axes, dsets):
        for k in keys:
            t = cell(rows, d, k, "territory")
            r = cell(rows, d, k, "region_recall")
            if np.isnan(t) or np.isnan(r):
                continue
            color, marker = FAMILY_STYLE[fam[k]]
            ax.plot(t * 100, r * 100, marker=marker, color=color, markersize=11,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle="none")
        ax.set_ylim(-4, 104)
        ax.set_xlim(left=-0.6)
        _finish(ax, xlabel="genome fraction called (%)", ylabel="region recall (%)")
        ax.set_title(DATASET_LABEL[d].replace("\n", " "), loc="left", pad=12)

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    _save(fig, "B5_recall_vs_territory.png")


# -- B6: runtime ------------------------------------------------------------

def fig_b6(rows):
    keys = method_order(rows)
    fam = family_of(rows)
    labels, times, colors = [], [], []
    for k, label in keys.items():
        vals = [r["runtime_s"] for r in rows if r["key"] == k]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        labels.append(label)
        times.append(float(np.median(vals)))
        colors.append(FAMILY_STYLE[fam[k]][0])

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 0.52 * len(labels) + 3.4))
    ax.barh(y, times, color=colors, height=0.62)
    left = max(min(times) * 0.5, 0.001)
    for yy, v in zip(y, times):
        txt = f"{v:.1f} s" if v >= 0.1 else f"{v * 1000:.0f} ms"
        ax.text(max(v, left) * 1.25, yy, txt, va="center", fontsize=FS_ANNOT,
                color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(left=left, right=max(times) * 5)
    ax.grid(axis="y", visible=False)
    _finish(ax, xlabel="median wall-clock per dataset (s, log scale)",
            title="Runtime on a 4.5 Mb synthetic genome",
            subtitle="single-threaded; not a scaling result — see the caveat below")
    ax.text(0, -0.14 - 0.010 * len(labels),
            "MACS parses the BAM on every run and FlexPeak is timed the same "
            "way (cache disabled), so I/O is included for both.\n"
            "The re-implemented baselines operate on coverage already in "
            "memory and are therefore NOT comparable on time; their bars "
            "measure arithmetic only.\n"
            "A 4.5 Mb genome is far below the scale where these differences "
            "matter — treat this as a smoke test, not a benchmark.",
            transform=ax.transAxes, fontsize=FS_NOTE, color=INK, va="top",
            linespacing=1.5)
    _save(fig, "B6_runtime.png")


FIGURES = OrderedDict([
    ("B1", fig_b1), ("B2", fig_b2), ("B3", fig_b3),
    ("B4", fig_b4), ("B5", fig_b5), ("B6", fig_b6),
])


def main(argv):
    rows = load_rows()
    wanted = [a.upper() for a in argv[1:]] or list(FIGURES)
    for name in wanted:
        if name not in FIGURES:
            print(f"unknown figure {name}; available: {', '.join(FIGURES)}")
            return 2
        FIGURES[name](rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
