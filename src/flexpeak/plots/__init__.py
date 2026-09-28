"""Figures drawn from a call set.

Optional: this subpackage imports matplotlib lazily, so ``import flexpeak``
still works without it (``pip install 'flexpeak[figures]'`` to enable).
"""

from .stats import format_stats, peak_stat_figures, peak_stats
from .style import apply_style

__all__ = ["peak_stats", "peak_stat_figures", "format_stats", "apply_style"]
