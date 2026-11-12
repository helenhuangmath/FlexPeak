"""Figures drawn from a call set.

Optional: this subpackage imports matplotlib lazily, so ``import flexpeak``
still works without it (``pip install 'flexpeak[figures]'`` to enable).
"""

from .stats import format_stats, peak_stat_figures, peak_stats, qc_figure
from .style import apply_style

__all__ = ["peak_stats", "peak_stat_figures", "format_stats", "qc_figure", "apply_style"]
