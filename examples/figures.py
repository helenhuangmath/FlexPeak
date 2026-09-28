#!/usr/bin/env python3
"""Generate the figures in docs/figures/ from synthetic data with known truth.

    python examples/figures.py            # all figures
    python examples/figures.py fig2 fig3  # just those

Every figure is produced by running FlexPeak, never by hand-entering numbers,
so re-running this after a code change re-measures rather than re-draws.  Truth
is known exactly here because the data is simulated (DESIGN.md section 9) --
these are correctness figures, not performance claims about real data.

Requires matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from flexpeak import call_peaks, preset_params, tune
from flexpeak.background.nb import BackgroundModel
from flexpeak.peaks import interval_overlap_bp
from flexpeak.plots.style import finalize
from flexpeak.simulate import simulate
from flexpeak.stats.nbtest import (
    nb_neglog10_sf, poisson_neglog10_sf, qvalue_from_neglog10p,
)

warnings.simplefilter("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "docs", "figures")

# -- palette ----------------------------------------------------------------
# Categorical slots 1-3 of a validated colourblind-safe order; the first three
# slots clear the all-pairs CVD and normal-vision floors.  Figures are rendered
# for a light surface only: they are static PNGs in the docs, one look.
SURFACE = "#ffffff"
INK = "#000000"   # all text and panel outlines
INK2 = "#52514e"
MUTED = "#8b8a85"
GRID = "#c9c7c1"
S1 = "#2a78d6"  # blue
S2 = "#eb6834"  # orange
S3 = "#1baf7a"  # aqua
FILL = "#d9d8d3"

# -- type scale -------------------------------------------------------------
# One scale, used everywhere, so a size change is a one-line change.  Nothing
# is bold: hierarchy comes from size and ink weight, not from weight.
FS_TITLE = 15
FS_SUB = 12.5
FS_LABEL = 13
FS_TICK = 12
FS_LEGEND = 12.5
FS_ANNOT = 12       # labels drawn on the plot (bar values, lane names)
FS_NOTE = 11.5      # footnotes under the axes

# Titles need room: the pad has to clear the title's own descenders plus the
# subtitle line beneath it, and both grew with the type scale.
TITLE_PAD = 34

plt.rcParams.update({
    # Arial, with fallbacks so the script still renders on a machine that does
    # not ship it (Linux/CI): Liberation Sans and Arimo are metric-compatible.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Arimo",
                        "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": "none",
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": FS_TITLE,
    "axes.titleweight": "normal",
    "axes.labelsize": FS_LABEL,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "legend.labelcolor": INK,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.frameon": False,
    "legend.fontsize": FS_LEGEND,
    "lines.linewidth": 2.0,
    "font.size": FS_TICK,
    "font.weight": "normal",
    "figure.titleweight": "normal",
    "figure.dpi": 200,
})


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


def _footnote(ax, text, y=-0.32):
    """A caveat under the axes."""
    ax.text(0, y, text, transform=ax.transAxes, fontsize=FS_NOTE, color=INK,
            va="top", linespacing=1.5)


def _save(fig, name):
    finalize(fig)
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, os.path.dirname(HERE))}")
    return path


# -- shared data ------------------------------------------------------------

_CACHE = {}


def mixed_data():
    """A mixed dataset: narrow peaks, broad domains, and domains with sub-peaks."""
    if "mixed" not in _CACHE:
        _CACHE["mixed"] = simulate(
            chrom_sizes={"chr1": 3_000_000, "chr2": 1_500_000}, bin_size=25,
            n_narrow=30, n_broad=4, n_mixed=3, seed=7,
        )
    return _CACHE["mixed"]


def mixed_params():
    """The settings the tuner selects on this dataset (see fig6)."""
    return preset_params("h3k9me3").replace(
        bin_size=25, posterior_cutoff=0.5, min_fold=1.5)


def mixed_call():
    if "call" not in _CACHE:
        d = mixed_data()
        _CACHE["call"] = call_peaks(d.treatment, d.control, mixed_params(),
                                    name="demo", return_details=True)
    return _CACHE["call"]


def recall_of(peaks, truth):
    called = peaks.intervals()
    hits = sum(1 for (c, s, e) in truth
               if interval_overlap_bp([(c, s, e)], called) >= 0.4 * (e - s))
    return hits / len(truth) if truth else 0.0


# -- fig 1: nested calls ----------------------------------------------------

def fig1():
    """A broad domain and its focal sub-peaks, called in one pass."""
    data = mixed_data()
    call = mixed_call()
    peaks = call.peaks

    # The mixed truth region whose call carries the most sub-peaks.
    best, best_region = None, None
    for t in [r for r in data.truth if r.kind == "mixed"]:
        for r in peaks:
            if r.chrom == t.chrom and r.start < t.end and r.end > t.start:
                if best is None or r.n_subpeaks > best.n_subpeaks:
                    best, best_region = r, t
    if best is None:
        print("  fig1: no called region overlaps a mixed domain; skipping")
        return

    chrom = best.chrom
    bs = data.treatment.bin_size
    pad = 10_000
    lo = max(0, min(best.start, best_region.start) - pad)
    hi = max(best.end, best_region.end) + pad
    b0, b1 = lo // bs, hi // bs

    counts = data.treatment.bins[chrom][b0:b1]
    mu = call.background.mu[chrom][b0:b1]
    x = (np.arange(b0, b1) * bs) / 1000.0  # kb

    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(counts.astype(float), size=9, mode="nearest")

    fig, axes = plt.subplots(
        3, 1, figsize=(11.5, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [5, 1.1, 1.1], "hspace": 0.28})
    ax, ax_call, ax_truth = axes

    ax.fill_between(x, 0, counts, color=FILL, linewidth=0, label="coverage, 25 bp bins")
    ax.plot(x, smooth, color=S1, lw=2.0, label="smoothed signal")
    ax.plot(x, mu, color=S2, lw=2.0, label="NB background µ")
    ax.set_ylim(0, float(counts.max()) * 1.30)  # headroom for the legend
    ax.legend(loc="upper right", ncol=3, labelcolor=INK)
    _finish(ax, ylabel="fragments / bin",
            title="A broad domain and its focal peaks, called in one pass",
            subtitle=f"{best.name}  {chrom}:{best.start:,}-{best.end:,} — "
                     f"{best.width / 1000:.0f} kb, fold {best.fold_enrichment:.2f}. "
                     f"Neither 'narrow' nor 'broad' was declared anywhere.")

    for a in (ax_call, ax_truth):
        a.grid(False)
        a.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            a.spines[side].set_visible(True)
            a.spines[side].set_color(INK)
            a.spines[side].set_linewidth(0.8)
        a.set_ylim(0, 1)

    ax_call.add_patch(Rectangle((best.start / 1000, 0.55), best.width / 1000, 0.35,
                                facecolor=S1, alpha=0.30, linewidth=0))
    for s in best.subpeaks:
        ax_call.add_patch(Rectangle((s.start / 1000, 0.08), max(s.width, 200) / 1000, 0.38,
                                    facecolor=S1, linewidth=0))
        ax_call.plot([s.summit / 1000], [0.55], marker="v", color=S1, markersize=5)
    ax_call.set_ylabel("FlexPeak", rotation=0, ha="right", va="center",
                       color=INK, fontsize=FS_ANNOT)

    ax_truth.add_patch(Rectangle((best_region.start / 1000, 0.55),
                                 best_region.width / 1000, 0.35,
                                 facecolor=S3, alpha=0.30, linewidth=0))
    for s, e, summit in best_region.subpeaks:
        ax_truth.add_patch(Rectangle((s / 1000, 0.08), max(e - s, 200) / 1000, 0.38,
                                     facecolor=S3, linewidth=0))
    ax_truth.set_ylabel("truth", rotation=0, ha="right", va="center",
                        color=INK, fontsize=FS_ANNOT)
    ax_truth.set_xlabel(f"{chrom} position (kb)")
    ax_truth.set_xlim(lo / 1000, hi / 1000)

    ax_call.text(0.995, 0.72, "domain", transform=ax_call.transAxes, ha="right",
                 va="center", fontsize=FS_NOTE, color=INK)
    ax_call.text(0.995, 0.27, "sub-peaks", transform=ax_call.transAxes, ha="right",
                 va="center", fontsize=FS_NOTE, color=INK)

    fig.subplots_adjust(bottom=0.21)
    fig.text(0.013, 0.10,
             f"The domain boundaries and the {len(best_region.subpeaks)} true "
             f"sub-peaks are all recovered — but {best.n_subpeaks} sub-peaks "
             f"were called in total.\n"
             f"The scale-space detector also fires on background fluctuation "
             f"inside a flat domain. False sub-peak rate inside domains is a "
             f"known\nopen item (DESIGN.md §9.4), not a solved one; "
             f"min_persistence is the knob that trades it against sensitivity.",
             fontsize=FS_NOTE, color=INK, va="top", linespacing=1.5)
    _save(fig, "fig1_nested_calls.png")


# -- fig 2: negative control ------------------------------------------------

def fig2():
    """Calibration on data containing no signal: NB vs the Poisson ablation."""
    null = simulate(chrom_sizes={"chr1": 2_000_000}, n_narrow=0, n_broad=0,
                    n_mixed=0, dispersion=0.25, seed=11)
    bg = BackgroundModel.fit(null.treatment, null.control)
    counts = null.treatment.concat()
    mu = np.concatenate([bg.mu[c] for c in sorted(bg.mu)])

    nb_p = nb_neglog10_sf(counts, mu, bg.alpha(mu))
    po_p = poisson_neglog10_sf(counts, mu)
    cutoff = -np.log10(0.05)
    nb_hits = int((qvalue_from_neglog10p(nb_p) >= cutoff).sum())
    po_hits = int((qvalue_from_neglog10p(po_p) >= cutoff).sum())

    n = counts.size
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.30})

    top = float(max(np.sort(po_p)[-1], expected[0])) * 1.05
    ax.plot([0, top], [0, top], color=MUTED, lw=1.2, ls=(0, (4, 3)),
            label="uniform (correct)")
    ax.plot(expected, np.sort(po_p)[::-1], color=S2, lw=2.0, label="Poisson")
    ax.plot(expected, np.sort(nb_p)[::-1], color=S1, lw=2.0, label="negative binomial")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.legend(loc="upper left", labelcolor=INK)
    ax.annotate("above the line = inflated significance",
                xy=(3.0, 10.5), xytext=(5.8, 6.8), fontsize=FS_NOTE, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.annotate("below = conservative", xy=(4.2, 4.0), xytext=(5.6, 2.2),
                fontsize=FS_NOTE, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    # Plain "log10", not the subscript characters: Arial has no U+2080/U+2081.
    _finish(ax, xlabel="expected  −log10 p", ylabel="observed  −log10 p",
            title="p-values on data with no signal",
            subtitle=f"input vs input, {n:,} bins, fitted dispersion "
                     f"α = {bg.alpha(np.array([bg.genome_mean]))[0]:.2f}")

    labels = ["negative\nbinomial", "Poisson\n(ablation)"]
    vals = [nb_hits, po_hits]
    bars = ax2.bar(labels, vals, color=[S1, S2], width=0.55)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03, f"{v:,}",
                 ha="center", va="bottom", fontsize=FS_ANNOT, color=INK)
    ax2.set_ylim(0, max(vals) * 1.18)
    ax2.grid(axis="x", visible=False)
    _finish(ax2, ylabel="bins called significant",
            title="False positives at FDR 5%",
            subtitle=f"out of {n:,} bins; every one is wrong")

    _footnote(ax, "Poisson assumes mean = variance. Real background is "
                  "overdispersed, so a Poisson caller reports inflated "
                  "significance\nand its FDR stops meaning anything — which is "
                  "why this test is run on every commit "
                  "(tests/test_background.py).")

    _save(fig, "fig2_negative_control.png")


# -- fig 3: depth degradation -----------------------------------------------

def fig3():
    """Recall and territory as sequencing depth falls."""
    hard = simulate(
        chrom_sizes={"chr1": 3_000_000}, n_narrow=40, n_broad=3, n_mixed=0,
        narrow_fold=3.0, broad_fold=1.6, background_level=6.0, dispersion=0.4,
        seed=41,
    )
    truth = hard.truth_intervals()
    genome = sum(hard.treatment.chrom_sizes.values())
    params = preset_params("h3k27me3").replace(bin_size=25)

    fracs = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)
    seeds = (0, 1, 2)
    recall = np.zeros((len(fracs), len(seeds)))
    territory = np.zeros_like(recall)

    for i, f in enumerate(fracs):
        for j, s in enumerate(seeds):
            if f < 1.0:
                t = hard.treatment.thin(f, seed=s)
                c = hard.control.thin(f, seed=s + 100)
            else:
                t, c = hard.treatment, hard.control
            p = call_peaks(t, c, params)
            recall[i, j] = recall_of(p, truth)
            territory[i, j] = p.total_bp / genome
        print(f"    depth {f:5.0%}  recall {recall[i].mean():5.1%}  "
              f"territory {territory[i].mean():5.1%}")

    x = np.array(fracs) * 100
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.6, 7.6), sharex=True,
                                  gridspec_kw={"hspace": 0.22})

    for ax_, mat, color, ylab in ((ax, recall, S1, "recall of true regions (%)"),
                                  (ax2, territory, S2, "genome fraction called (%)")):
        ax_.plot(x, mat.mean(axis=1) * 100, color=color, lw=2.0, marker="o",
                 markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        for j in range(mat.shape[1]):
            ax_.plot(x, mat[:, j] * 100, color=color, lw=0, marker="o",
                     markersize=3, alpha=0.35)
        ax_.set_xscale("log")
        ax_.set_ylim(0, max(mat.max() * 100 * 1.25, 5))
        _finish(ax_, ylabel=ylab)

    ax.set_title("Recall falls with depth. Territory does not follow it down.",
                 loc="left", pad=TITLE_PAD)
    ax.text(0, 1.012,
            "hard case: 3× enrichment, overdispersed library, treatment and "
            "control thinned together; 3 seeds per depth",
            transform=ax.transAxes, fontsize=FS_SUB, color=INK, va="bottom")
    ax2.set_xlabel("sequencing depth retained (%, log scale)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"{v:g}" for v in x])
    ax2.minorticks_off()

    r_full, r_low = recall[0].mean(), recall[-1].mean()
    t_full, t_low = territory[0].mean(), territory[-1].mean()
    _footnote(ax2,
              f"Read the panels together: recall alone can always be bought by "
              f"calling more genome.\n"
              f"At 2% depth recall has fallen {r_full:.0%} → {r_low:.0%} while "
              f"territory has barely moved ({t_full:.0%} → {t_low:.0%}), so the "
              f"surviving calls are worse, not merely fewer.\n"
              f"The curve is also not monotone — 50% scores at or above full "
              f"depth. On this dataset the differences between adjacent depths "
              f"are within seed noise.")

    _save(fig, "fig3_depth_degradation.png")


# -- fig 4: recall by region type -------------------------------------------

def fig4():
    """Recall by region type, from a single run with a single parameter set."""
    data = mixed_data()
    peaks = mixed_call().peaks
    all_truth = data.truth_intervals()

    # Recall is the only metric that decomposes by truth class.  Precision and
    # the fusion index are properties of the CALLS, so they are single global
    # numbers -- splitting them per class would score a correctly-called narrow
    # peak as a false positive of the broad class.
    classes = [
        ("narrow\npeaks", [r for r in data.truth if r.kind == "narrow"]),
        ("broad\ndomains", [r for r in data.truth if r.kind == "broad"]),
        ("mixed\ndomains", [r for r in data.truth if r.kind == "mixed"]),
        ("all\nregions", list(data.truth)),
    ]
    names, vals = [], []
    for label, regions in classes:
        truth = [(r.chrom, r.start, r.end) for r in regions]
        names.append(f"{label}\nn={len(truth)}")
        vals.append(recall_of(peaks, truth) * 100)

    precision = 100 * sum(
        1 for r in peaks
        if interval_overlap_bp([(r.chrom, r.start, r.end)], all_truth) >= 0.25 * r.width
    ) / max(len(peaks), 1)
    fused = sum(
        1 for r in peaks
        if sum(1 for (c, s, e) in all_truth
               if c == r.chrom
               and interval_overlap_bp([(c, s, e)], [(r.chrom, r.start, r.end)]) >= 200) > 1
    )
    fusion = 100 * fused / max(len(peaks), 1)

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    bars = ax.bar(names, vals, width=0.55, color=S1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=FS_ANNOT, color=INK)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    _finish(ax, ylabel="recall of true regions (%)",
            title="One parameter set, every region type",
            subtitle=f"{len(peaks)} regions called; narrow and broad were never "
                     f"called separately")
    _footnote(ax,
              f"Precision {precision:.0f}% and fusion index {fusion:.0f}% are "
              f"global — they are properties of the calls, not of a truth class, "
              f"so they do not split by bar.\n"
              f"The fusion index (calls spanning more than one true region) is "
              f"the anti-over-merge check; it also guards FlexPeak's own tuner "
              f"against drifting that way.")

    _save(fig, "fig4_recall_by_type.png")


# -- fig 5: width bimodality ------------------------------------------------

def fig5():
    """The two width populations a mixed mark produces from a single run."""
    peaks = mixed_call().peaks
    region_w = np.array([r.width for r in peaks], dtype=float)
    sub_w = np.array([s.width for r in peaks for s in r.subpeaks], dtype=float)

    lo = min(region_w.min(), sub_w.min()) * 0.7
    hi = max(region_w.max(), sub_w.max()) * 1.4
    bins = np.logspace(np.log10(lo), np.log10(hi), 34)

    # Small multiples rather than overlaid histograms: the two populations
    # differ by two orders of magnitude in count, so overlaying them buries the
    # domains completely.
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.0, 6.6), sharex=True,
                                  gridspec_kw={"hspace": 0.32})
    thresh = mixed_params().domain_min_width

    for a, vals, color, label in ((ax, region_w, S1, "called regions"),
                                  (ax2, sub_w, S2, "nested sub-peaks")):
        a.hist(vals, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.8)
        a.set_xscale("log")
        a.axvline(thresh, color=MUTED, lw=1.2, ls=(0, (4, 3)))
        a.grid(axis="x", visible=False)
        _finish(a, ylabel="count")
        a.text(0.995, 0.92, label, transform=a.transAxes, ha="right", va="top",
               fontsize=FS_ANNOT, color=INK)

    ax.text(thresh * 1.15, ax.get_ylim()[1] * 0.94,
            f"domain threshold\n({thresh / 1000:g} kb)",
            fontsize=FS_NOTE, color=INK, va="top")
    ax2.set_xlabel("width (bp, log scale)")
    ax.set_title("Two width populations, from one run", loc="left", pad=TITLE_PAD)
    ax.text(0, 1.012,
            f"{peaks.n_domains} domains and {len(peaks) - peaks.n_domains} focal "
            f"peaks, plus {peaks.n_subpeaks} sub-peaks nested inside them",
            transform=ax.transAxes, fontsize=FS_SUB, color=INK, va="bottom")
    _footnote(ax2,
              "A caller forced to pick one scale produces one of these "
              "populations and misses the other.\n"
              "QC reports this separation as a width-bimodality score and uses "
              "it to flag the sample as a mixed mark.")

    _save(fig, "fig5_width_bimodality.png")


# -- fig 6: the tuning sweep ------------------------------------------------

def fig6():
    """What the self-tuner looked at, and what it chose."""
    data = mixed_data()
    grid = {"posterior_cutoff": (0.4, 0.5, 0.6), "min_fold": (1.2, 1.5, 2.0)}
    result = tune(
        data.treatment, data.control,
        base=preset_params("h3k9me3").replace(bin_size=25),
        grid=grid, n_chroms=2, thin_fractions=(0.5,), seeds=(0,),
    )

    cuts = list(grid["posterior_cutoff"])
    folds = list(grid["min_fold"])
    Z = np.full((len(folds), len(cuts)), np.nan)
    for point, score in result.curve:
        Z[folds.index(point["min_fold"]), cuts.index(point["posterior_cutoff"])] = score.total

    chosen = result.params.to_dict()
    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    im = ax.imshow(Z, cmap="Blues", origin="lower", aspect="auto",
                   vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    for i in range(len(folds)):
        for j in range(len(cuts)):
            if np.isnan(Z[i, j]):
                continue
            frac = (Z[i, j] - np.nanmin(Z)) / max(np.nanmax(Z) - np.nanmin(Z), 1e-12)
            ax.text(j, i, f"{Z[i, j]:.4f}", ha="center", va="center",
                    fontsize=FS_ANNOT, color="#ffffff" if frac > 0.62 else INK)

    def _cell_ink(i, j):
        frac = (Z[i, j] - np.nanmin(Z)) / max(np.nanmax(Z) - np.nanmin(Z), 1e-12)
        return "#ffffff" if frac > 0.62 else INK

    # Label the two cells in place: an arrow drawn across neighbouring cells
    # reads as pointing at whatever it crosses.
    ci, fi = cuts.index(chosen["posterior_cutoff"]), folds.index(chosen["min_fold"])
    ax.add_patch(Rectangle((ci - 0.47, fi - 0.47), 0.94, 0.94, fill=False,
                           edgecolor=S2, linewidth=2.5))
    ax.text(ci, fi + 0.26, "selected", ha="center", va="center", fontsize=FS_NOTE,
            color=_cell_ink(fi, ci))

    bi, bj = np.unravel_index(np.nanargmax(Z), Z.shape)
    if (bi, bj) != (fi, ci):
        ax.add_patch(Rectangle((bj - 0.47, bi - 0.47), 0.94, 0.94, fill=False,
                               edgecolor=MUTED, linewidth=1.8, linestyle=(0, (4, 3))))
        ax.text(bj, bi + 0.26, "argmax", ha="center", va="center", fontsize=FS_NOTE,
                color=_cell_ink(bi, bj))

    ax.set_xticks(range(len(cuts)), [str(c) for c in cuts])
    ax.set_yticks(range(len(folds)), [str(f) for f in folds])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("objective (stability − territory penalty)", color=INK,
                 fontsize=FS_LABEL)
    cb.outline.set_visible(True)
    cb.outline.set_edgecolor(INK)
    cb.outline.set_linewidth(0.8)

    spread = float(np.nanmax(Z) - np.nanmin(Z))
    _finish(ax, xlabel="posterior_cutoff", ylabel="min_fold",
            title="The tuner reports its curve, not just its answer",
            subtitle=f"{len(result.curve)} grid points, no pretrained model; "
                     f"objective spread {spread:.4f}")

    # The caveat is read off the run, never assumed: whether the sample can
    # distinguish these settings is exactly what the tuner is asked to report.
    if result.stable:
        note = ("The plateau is taken, not the argmax: the best-scoring point can "
                "sit on a noise spike at the edge of a flat region.\n"
                "Selecting the middle of the near-best set is what stops the "
                "tuner overfitting its own metric, exactly as eyeballing one "
                "locus overfits that locus.")
    else:
        note = ("The objective barely varies across this grid, so the sample "
                "cannot distinguish these settings and the pick is arbitrary.\n"
                "FlexPeak reports that rather than presenting it as a decision: "
                + "; ".join(result.notes[:1]))
    _footnote(ax, note, y=-0.22)

    _save(fig, "fig6_tuning_sweep.png")


# -- fig 7: bin size diagnostic ---------------------------------------------

BIN_SIZES = (25, 50, 100, 200, 500)


def _truth_mask(data, chrom, bin_size, kinds=None):
    """Bins whose span overlaps a truth region of the requested kind(s)."""
    n = data.treatment.bins[chrom].size * data.treatment.bin_size
    n_bins = -(-n // bin_size)
    mask = np.zeros(n_bins, dtype=bool)
    for r in data.truth:
        if r.chrom != chrom or (kinds and r.kind not in kinds):
            continue
        mask[r.start // bin_size: -(-r.end // bin_size)] = True
    return mask


def _auroc(pos, neg):
    """Rank-based AUROC: P(a random peak bin outscores a random background bin)."""
    from scipy.stats import rankdata

    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def fig7():
    """Peak signal vs background as the bin size changes: the bin_size diagnosis."""
    data = mixed_data()
    windows = mixed_params().windows

    panels, auc_narrow, auc_broad = [], [], []
    for bs in BIN_SIZES:
        treat = data.treatment.rebin(bs)
        ctrl = data.control.rebin(bs)
        bg = BackgroundModel.fit(treat, ctrl, windows=windows)

        lf, m_any, m_narrow, m_broad = [], [], [], []
        for chrom in sorted(treat.bins):
            counts = treat.bins[chrom].astype(np.float64)
            mu = bg.mu[chrom].astype(np.float64)[: counts.size]
            lf.append(np.log2((counts + 1.0) / (mu + 1.0)))
            for dest, kinds in ((m_any, None), (m_narrow, ("narrow",)),
                                (m_broad, ("broad", "mixed"))):
                dest.append(_truth_mask(data, chrom, bs, kinds)[: counts.size])
        lf = np.concatenate(lf)
        m_any = np.concatenate(m_any)
        m_narrow = np.concatenate(m_narrow)
        m_broad = np.concatenate(m_broad)

        bgmask = ~m_any  # background = outside every truth region, at every bin size
        panels.append((bs, lf[m_any], lf[bgmask]))
        auc_narrow.append(_auroc(lf[m_narrow], lf[bgmask]))
        auc_broad.append(_auroc(lf[m_broad], lf[bgmask]))
        print(f"    {bs:4d} bp  AUROC narrow {auc_narrow[-1]:.3f}  "
              f"broad+mixed {auc_broad[-1]:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.26})
    flat = axes.ravel()
    edges = np.linspace(-1.5, 3.5, 46)

    for k, (bs, peak_lf, bg_lf) in enumerate(panels):
        a = flat[k]
        a.hist(bg_lf, bins=edges, density=True, color=S2, alpha=0.85,
               label="background bins")
        a.hist(peak_lf, bins=edges, density=True, color=S1, alpha=0.75,
               label="bins inside true regions")
        a.axvline(0, color=MUTED, lw=1.1, ls=(0, (4, 3)))
        a.set_xlim(edges[0], edges[-1])
        a.grid(axis="x", visible=False)
        _finish(a, xlabel="log2 fold over background", ylabel="density")
        a.set_title(f"{bs} bp bins", loc="left", pad=12)
        if k == 0:
            a.legend(loc="upper right", labelcolor=INK, fontsize=FS_NOTE)

    a = flat[5]
    for vals, color, label in ((auc_narrow, S1, "narrow peaks"),
                               (auc_broad, S2, "broad + mixed domains")):
        a.plot(BIN_SIZES, vals, color=color, lw=2.0, marker="o", markersize=8,
               markeredgecolor=SURFACE, markeredgewidth=1.6, label=label)
    best_n = BIN_SIZES[int(np.nanargmax(auc_narrow))]
    best_b = BIN_SIZES[int(np.nanargmax(auc_broad))]
    a.set_xscale("log")
    a.set_xticks(list(BIN_SIZES))
    a.set_xticklabels([str(b) for b in BIN_SIZES])
    a.minorticks_off()
    a.set_ylim(0.5, 1.02)
    a.axhline(0.5, color=MUTED, lw=1.1, ls=(0, (4, 3)))
    a.legend(loc="lower left", labelcolor=INK, fontsize=FS_NOTE)
    _finish(a, xlabel="bin size (bp)", ylabel="AUROC vs background")
    a.set_title("separability by region type", loc="left", pad=12)

    fig.suptitle("Peak signal distribution by bin size", x=0.088, y=1.00,
                 ha="left", fontsize=FS_TITLE, color=INK)
    fig.text(0.088, 0.947,
             "Same data, re-binned. Bins are labelled from truth, not from "
             "calls, so the peak/background split is identical in every panel.",
             fontsize=FS_SUB, color=INK, va="bottom")
    fig.text(0.088, 0.028,
             f"Broad+mixed separability climbs steeply with bin size "
             f"({min(auc_broad):.2f} at 25 bp → {max(auc_broad):.2f} at 500 bp): "
             f"fine bins leave a domain as weak signal spread over thousands of "
             f"bins.\n"
             f"Narrow is nearly flat by comparison "
             f"({min(auc_narrow):.2f}–{max(auc_narrow):.2f}), best at {best_n} bp "
             f"and slightly worse either side. On this data the two optima differ "
             f"({best_n} vs {best_b} bp) but the cost to narrow peaks of\n"
             f"binning coarsely is small — the trade-off is real and asymmetric, "
             f"not symmetric. This is the figure to read before choosing "
             f"bin_size or deciding what the tuner should sweep.",
             fontsize=FS_NOTE, color=INK, va="top", linespacing=1.5)
    fig.subplots_adjust(top=0.86, bottom=0.17)

    _save(fig, "fig7_bin_size_diagnostic.png")


FIGURES = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
           "fig5": fig5, "fig6": fig6, "fig7": fig7}


def main(argv):
    wanted = argv[1:] or list(FIGURES)
    unknown = [a for a in wanted if a not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}; "
              f"available: {', '.join(FIGURES)}")
        return 2
    for name in wanted:
        print(f"{name}: {FIGURES[name].__doc__.splitlines()[0]}")
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
