#!/usr/bin/env python3
"""Render the differential-calling figures from Benchmark/results_diff/metrics.tsv.

    python Benchmark/run_diff_benchmark.py
    python Benchmark/make_diff_figures.py

D1  F1 heatmap (overall / gain / loss) per method and dataset
D2  precision vs recall per dataset, with direction accuracy as marker size
D3  null-dataset false calls and runtime
"""

from __future__ import annotations

import csv
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as MF  # noqa: E402  (shared style helpers)

import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_diff")
FIGDIR = os.path.join(HERE, "figures")
MF.FIGDIR = FIGDIR

DSETS = ["diff_mixed", "diff_depth", "diff_single"]
LABEL = {"diff_mixed": "equal depth,\n2 replicates", "diff_depth": "B at 50% depth,\n2 replicates",
         "diff_single": "1 replicate\neach", "diff_null": "null\n(no difference)"}


def load_rows():
    rows = []
    with open(os.path.join(RESULTS, "metrics.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            for k, v in list(r.items()):
                if k not in ("dataset", "method", "key", "family"):
                    r[k] = float(v) if v not in ("", None) else float("nan")
            rows.append(r)
    return rows


def _cell(rows, d, k, f):
    for r in rows:
        if r["dataset"] == d and r["key"] == k:
            return r[f]
    return float("nan")


def fig_d1(rows):
    keys = MF.method_order(rows)
    dsets = [d for d in DSETS if any(r["dataset"] == d for r in rows)]
    fields = [("bp_f1", "all"), ("gain_f1", "gain (higher in A)"), ("loss_f1", "loss (higher in B)")]
    fig, axes = plt.subplots(1, 3, figsize=(1.55 * len(dsets) * 3 + 8, 0.52 * len(keys) + 3.2),
                             sharey=True)
    for ax, (f, title) in zip(axes, fields):
        Z = np.array([[_cell(rows, d, k, f) for d in dsets] for k in keys])
        im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.0%}", ha="center", va="center",
                            fontsize=MF.FS_ANNOT, color="#ffffff" if Z[i, j] > 0.62 else MF.INK)
        ax.set_xticks(range(len(dsets)), [LABEL[d] for d in dsets])
        ax.set_yticks(range(len(keys)), list(keys.values()))
        ax.set_title(title, loc="left", pad=10)
        MF._finish(ax)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("base-pair F1", color=MF.INK, fontsize=MF.FS_LABEL)
    cb.outline.set_edgecolor(MF.INK)
    fig.suptitle("Differential regions: base-pair F1 against known truth, by direction",
                 x=0.01, ha="left", fontsize=MF.FS_TITLE)
    MF._save(fig, "D1_diff_f1_heatmap.png")


def fig_d2(rows):
    keys = MF.method_order(rows)
    fam = MF.family_of(rows)
    dsets = [d for d in DSETS if any(r["dataset"] == d for r in rows)]
    fig, axes = plt.subplots(1, len(dsets), figsize=(5.2 * len(dsets) + 1, 5.6))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dsets):
        for k in keys:
            p, r = _cell(rows, d, k, "bp_precision"), _cell(rows, d, k, "bp_recall")
            acc = _cell(rows, d, k, "direction_accuracy")
            if np.isnan(p) or np.isnan(r):
                continue
            color, marker = MF.FAMILY_STYLE.get(fam[k], (MF.NEUTRAL, "o"))
            size = 6 + 9 * (0 if np.isnan(acc) else acc)
            ax.plot(r * 100, p * 100, marker=marker, color=color, markersize=size,
                    markeredgecolor=MF.SURFACE, markeredgewidth=1.2, linestyle="none")
        ax.set_xlim(-4, 104)
        ax.set_ylim(-4, 104)
        ax.set_title(LABEL[d].replace("\n", " "), loc="left", pad=10)
        MF._finish(ax, xlabel="base-pair recall (%)", ylabel="base-pair precision (%)")
    fig.suptitle("Precision against recall for differential regions "
                 "(marker size = direction accuracy)", x=0.01, ha="left", fontsize=MF.FS_TITLE)
    handles = [plt.Line2D([], [], marker=m, color=c, linestyle="none", markersize=10, label=f)
               for f, (c, m) in MF.FAMILY_STYLE.items() if f in set(fam.values())]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    MF._save(fig, "D2_diff_precision_recall.png")


def fig_d3(rows):
    keys = MF.method_order(rows)
    fam = MF.family_of(rows)
    labels, calls, times, colors = [], [], [], []
    for k, lab in keys.items():
        n = _cell(rows, "diff_null", k, "n_calls")
        t = [r["runtime_s"] for r in rows if r["key"] == k and not np.isnan(r["runtime_s"])]
        if np.isnan(n) and not t:
            continue
        labels.append(lab); calls.append(0 if np.isnan(n) else n)
        times.append(float(np.median(t)) if t else np.nan)
        colors.append(MF.FAMILY_STYLE.get(fam[k], (MF.NEUTRAL, "o"))[0])
    y = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 0.5 * len(labels) + 3.2), sharey=True)
    a1.barh(y, calls, color=colors, height=0.62)
    for yy, v in zip(y, calls):
        a1.text(v + max(calls + [1]) * 0.02, yy, f"{int(v)}", va="center", fontsize=MF.FS_ANNOT)
    a1.set_yticks(y, labels); a1.invert_yaxis()
    MF._finish(a1, xlabel="regions called on the null dataset (correct answer: 0)",
               title="False differential calls")
    a2.barh(y, np.nan_to_num(times), color=colors, height=0.62)
    for yy, v in zip(y, times):
        if not np.isnan(v):
            a2.text(v * 1.15, yy, f"{v:.1f} s", va="center", fontsize=MF.FS_ANNOT)
    a2.set_xscale("log")
    MF._finish(a2, xlabel="median wall-clock per dataset (s, log)", title="Runtime, 6 Mb genome")
    fig.tight_layout()
    MF._save(fig, "D3_diff_null_runtime.png")


FIGURES = OrderedDict([("D1", fig_d1), ("D2", fig_d2), ("D3", fig_d3)])


def main(argv):
    rows = load_rows()
    for name in [a.upper() for a in argv[1:]] or list(FIGURES):
        FIGURES[name](rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
