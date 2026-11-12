#!/usr/bin/env python3
"""Locus views: signal tracks with the called regions and sub-peaks under them.

    python examples/locus_tracks.py --out figs \\
        --track "H3K9me3 Cl13" signal.bw calls_peaks.broadPeak calls_peaks.narrowPeak \\
        --track "Arm vs Cl13" arm.bw diff_diffpeaks.narrowPeak - \\
        --locus Pdcd1 chr1:94038305-94052553 20000

Each ``--track`` is a label, a bigWig, a region file (broadPeak / BED / signed
narrowPeak from ``flexpeak diff``, or ``-``) and a sub-peak file (narrowPeak,
or ``-``).  One figure per locus; one panel per track, same x axis.  Signal
is drawn as a filled area; regions as bars under it (domains thick, peaks
thin; gain/loss of a differential file in blue/orange); sub-peaks as ticks.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from flexpeak.plots.style import (INK, INK2, MUTED, S1, S2, FS_ANNOT, FS_LABEL,
                                  FS_TICK, FS_TITLE, apply_style, finalize)

Interval = Tuple[str, int, int, str]


def read_regions(path: str) -> List[Interval]:
    """(chrom, start, end, tag): 'domain'/'peak' by width, or 'gain'/'loss' from
    the strand column of a differential narrowPeak."""
    out: List[Interval] = []
    if not path or path == "-" or not os.path.exists(path):
        return out
    is_diff = "diff" in os.path.basename(path)
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            c, s, e = f[0], int(f[1]), int(f[2])
            if is_diff and len(f) >= 6 and f[5] in ("+", "-"):
                tag = "gain" if f[5] == "+" else "loss"
            else:
                tag = "domain" if e - s >= 2000 else "peak"
            out.append((c, s, e, tag))
    return out


def signal(bw_path: str, chrom: str, start: int, end: int, n: int = 900):
    import pyBigWig

    bw = pyBigWig.open(bw_path)
    try:
        if chrom not in bw.chroms():
            return None, None
        end = min(end, bw.chroms()[chrom])
        vals = bw.stats(chrom, start, end, type="mean", nBins=n)
    finally:
        bw.close()
    y = np.array([0.0 if v is None else v for v in vals], dtype=float)
    return np.linspace(start, end, n), y


def draw_locus(tracks, name, chrom, start, end, out_path, gene_span=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    apply_style()
    n = len(tracks)
    fig, axes = plt.subplots(n + 1, 1, figsize=(11, 1.6 * n + 1.6), sharex=True,
                             gridspec_kw={"hspace": 0.45, "height_ratios": [1.0] * n + [0.28]})
    axes = np.atleast_1d(axes)
    for ax, (label, bw, regions, subpeaks) in zip(axes[:n], tracks):
        x, y = signal(bw, chrom, start, end)
        ymax = 1.0
        if x is not None:
            ax.fill_between(x, 0, y, color=S1, alpha=0.35, linewidth=0)
            ax.plot(x, y, color=S1, linewidth=0.9)
            ymax = float(np.nanmax(y)) if np.nanmax(y) > 0 else 1.0
        ax.set_ylim(-0.42 * ymax, 1.08 * ymax)
        for c, s, e, tag in regions:
            if c != chrom or e < start or s > end:
                continue
            colour = {"gain": S1, "loss": S2}.get(tag, INK2)
            h = 0.16 if tag in ("domain", "gain", "loss") else 0.10
            ax.add_patch(plt.Rectangle((max(s, start), -0.24 * ymax), min(e, end) - max(s, start),
                                       h * ymax, facecolor=colour, edgecolor="none"))
        for c, s, e, _ in subpeaks:
            if c != chrom or e < start or s > end:
                continue
            ax.add_patch(plt.Rectangle((max(s, start), -0.40 * ymax), min(e, end) - max(s, start),
                                       0.11 * ymax, facecolor=S2, edgecolor="none"))
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=FS_LABEL, labelpad=8)
        ax.set_yticks([0, round(ymax, 1)])
        ax.tick_params(labelsize=FS_TICK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # Gene strip: its own axis, so nothing collides with the coordinate label.
    g = axes[n]
    g.set_ylim(0, 1)
    g.set_yticks([])
    for side in ("top", "right", "left"):
        g.spines[side].set_visible(False)
    if gene_span:
        gs, ge = gene_span
        g.plot([gs, ge], [0.55, 0.55], color=INK, lw=3.0, solid_capstyle="butt")
        g.text((gs + ge) / 2, 0.15, name, ha="center", va="top", fontsize=FS_ANNOT, color=INK)
    g.set_xlim(start, end)
    g.set_xlabel(f"{chrom}:{start:,}-{end:,}", fontsize=FS_LABEL)
    g.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.2f} Mb"))
    g.tick_params(labelsize=FS_TICK)
    fig.suptitle(f"{name}: signal with called regions (bars) and sub-peaks (ticks)",
                 x=0.01, ha="left", fontsize=FS_TITLE)
    handles = [Patch(facecolor=INK2, label="region / domain"), Patch(facecolor=S1, label="gain in A"),
               Patch(facecolor=S2, label="loss in A / sub-peak")]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=FS_ANNOT - 1,
               bbox_to_anchor=(0.99, 0.985))
    finalize(fig)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.19, right=0.98)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", nargs=4, action="append", required=True,
                    metavar=("LABEL", "BIGWIG", "REGIONS", "SUBPEAKS"))
    ap.add_argument("--locus", nargs=3, action="append", required=True,
                    metavar=("NAME", "CHR:START-END", "FLANK_BP"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="locus")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    tracks = [(lab, bw, read_regions(reg), read_regions(sub) if sub != "-" else [])
              for lab, bw, reg, sub in args.track]
    for name, span, flank in args.locus:
        chrom, rng = span.split(":")
        gs, ge = (int(v.replace(",", "")) for v in rng.split("-"))
        path = os.path.join(args.out, f"{args.prefix}_{name}.png")
        draw_locus(tracks, name, chrom, max(0, gs - int(flank)), ge + int(flank), path,
                   gene_span=(gs, ge))
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
