"""Shared figure style: Arial type, vector PDF output, one palette.

Every figure FlexPeak writes goes through :func:`apply_style` so a change to the
type scale or the palette is a one-line change here rather than a sweep through
the plotting code.  Two choices are load-bearing and deliberate:

* **Arial.**  Requested for all figures.  Fallbacks are listed for machines that
  do not ship it (Linux/CI); Liberation Sans and Arimo are metric-compatible, so
  a figure laid out against Arial does not reflow.
* **PDF, with ``pdf.fonttype = 42``.**  Type 42 embeds TrueType outlines and
  keeps glyphs as *text*, so the output stays vector and the labels remain
  selectable and editable in Illustrator/Inkscape.  The matplotlib default
  (type 3) converts text to a form most vector editors cannot re-edit.

The look is deliberately plain: every piece of text is black, every panel is an
outlined box with no fill, and there are no grid or other background lines
inside the panels.  :func:`finalize` enforces that at save time, so a plotting
function that sets a grey label or turns a grid on cannot leak it into a figure.
"""

from __future__ import annotations

__all__ = [
    "apply_style", "savefig", "finish", "footnote", "finalize",
    "SURFACE", "INK", "INK2", "MUTED", "GRID", "S1", "S2", "S3", "FILL",
    "FS_TITLE", "FS_SUB", "FS_LABEL", "FS_TICK", "FS_LEGEND", "FS_ANNOT",
    "FS_NOTE", "TITLE_PAD",
]

# -- palette ----------------------------------------------------------------
# Categorical slots 1-3 of a colourblind-safe order, matching examples/figures.py
# so the docs figures and the per-run statistics figures read as one system.
SURFACE = "#ffffff"
INK = "#000000"   # all text and panel outlines
INK2 = "#52514e"
MUTED = "#8b8a85"
GRID = "#c9c7c1"
S1 = "#2a78d6"  # blue   -- regions / domains
S2 = "#eb6834"  # orange -- sub-peaks
S3 = "#1baf7a"  # aqua   -- third series
FILL = "#d9d8d3"

# -- type scale -------------------------------------------------------------
FS_TITLE = 15
FS_SUB = 12.5
FS_LABEL = 13
FS_TICK = 12
FS_LEGEND = 12.5
FS_ANNOT = 12
FS_NOTE = 11.5

TITLE_PAD = 34

#: Arial first, then metric-compatible stand-ins for machines without it.
FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "Arimo", "DejaVu Sans"]

_APPLIED = False


def apply_style(force: bool = False):
    """Install the FlexPeak rcParams and return the ``pyplot`` module.

    Importing matplotlib is deferred to call time: it is an optional dependency
    (``pip install 'flexpeak[figures]'``), and importing the package must not
    require it.
    """
    global _APPLIED
    try:
        import matplotlib
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Figures require matplotlib.  Install it with:\n"
            "    pip install 'flexpeak[figures]'   (or: pip install matplotlib)"
        ) from e

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if _APPLIED and not force:
        return plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        # Keep text as text in the PDF (see module docstring).
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 6,
        "savefig.format": "pdf",
        "figure.facecolor": SURFACE,
        "axes.facecolor": "none",
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
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
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.frameon": False,
        "legend.fontsize": FS_LEGEND,
        "legend.labelcolor": INK,
        "lines.linewidth": 2.0,
        "font.size": FS_TICK,
        "font.weight": "normal",
        "figure.titleweight": "normal",
        "figure.dpi": 200,
    })
    _APPLIED = True
    return plt


def finish(ax, xlabel=None, ylabel=None, title=None, subtitle=None):
    """Use an outlined, transparent plotting panel and set labels in one call."""
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


def footnote(ax, text, y=-0.30):
    """A caveat under the axes."""
    ax.text(0, y, text, transform=ax.transAxes, fontsize=FS_NOTE, color=INK,
            va="top", linespacing=1.5)


def finalize(fig):
    """Enforce the house look on a finished figure.

    Every text artist becomes black, every visible panel becomes an unfilled box
    with a black outline, and grid lines are removed.  White text is left alone:
    it is only ever used on top of a dark heatmap cell, where black would vanish.
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


def savefig(fig, path):
    """Write the figure (format from the extension, PDF by default) and close it."""
    import os

    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ext = os.path.splitext(str(path))[1].lstrip(".").lower() or "pdf"
    finalize(fig)
    fig.savefig(path, format=ext, bbox_inches="tight")
    plt.close(fig)
    return str(path)
