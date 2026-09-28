"""FlexPeak: flexible, adaptive peak calling.

No pretrained model, no training corpus, no ML dependency -- adaptivity comes
from per-sample self-tuning at runtime.  See docs/DESIGN.md.
"""

__version__ = "0.1.0"

from .peaks import NestedPeakSet, Region, SubPeak
from .signal.coverage import Coverage
from .caller import CallParams, call_peaks, preset_params, PRESETS
from .background.nb import BackgroundModel
from .io.load import load_signal
from .tune.sweep import tune, TuneResult
from .tune.objective import TuningObjective

__all__ = [
    "__version__",
    "NestedPeakSet", "Region", "SubPeak",
    "Coverage", "load_signal",
    "CallParams", "call_peaks", "preset_params", "PRESETS",
    "BackgroundModel",
    "tune", "TuneResult", "TuningObjective",
]
