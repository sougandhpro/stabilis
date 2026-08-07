"""stabilis -- frequency stability analysis.

The Allan variance family, wavelet variance, and confidence intervals that
do not require guessing the noise type first.

The core imports numpy and nothing else. scipy and matplotlib are optional
and are never required for a correct answer.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stabilis")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

# __version__ must be bound before these import, since provenance reads it.
from .provenance import Provenance, Step, new_provenance  # noqa: E402
from .types import FreqSeries, PhaseSeries, to_frequency, to_phase  # noqa: E402

__all__ = [
    "__version__",
    "FreqSeries",
    "PhaseSeries",
    "Provenance",
    "Step",
    "new_provenance",
    "to_frequency",
    "to_phase",
]
