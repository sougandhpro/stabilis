"""stabilis -- frequency stability analysis.

The Allan variance family, wavelet variance, and confidence intervals that
do not require guessing the noise type first.

The core imports numpy and nothing else. scipy and matplotlib are optional
and are never required for a correct answer.

Everything a caller normally needs is re-exported here, so consuming code
writes `from stabilis import oadev, confidence` and never has to know which
submodule something lives in. That is a promise: **the names in `__all__`
below are the public API** and will not move without a version bump, while
anything reached through a submodule path may.

`fit` is the function, `stabilis.fitting` is the package it lives in. The
package is not called `fit` precisely so that the two never collide: a
top-level name that shadows a submodule makes `import stabilis.fit` and
`from stabilis.fit import ...` mean different things, which is the sort of
ambiguity that costs somebody an afternoon.

Four subsystems are deliberately *not* re-exported, and their absence is
the documentation:

* `stabilis.bench` -- the measurement setup. Optional, and `stats/` is
  forbidden to import it, so keeping it off the top level keeps that
  separation visible.
* `stabilis.plot` -- needs matplotlib. Importing it here would drag an
  optional dependency into every `import stabilis`.
* `stabilis.report` -- the customer-facing measurement report. Its figure
  path wants matplotlib for the same reason, and it is an output layer
  rather than part of the analysis API.
* `stabilis.export` -- sigma-tau tables and report bundles, the same output
  layer. `save_phase` and `save_frequency` *are* re-exported, because
  writing a data file is the mirror of reading one and belongs beside
  `load_phase`; writing a *result* is a different job.

    import numpy as np
    from stabilis import load_phase, oadev, confidence

    series = load_phase("run.dat", rate_hz=1.0)
    result = oadev(series, taus="octave")
    ci = confidence(result)
    for tau, dev, lo, hi in zip(result.tau_s, result.dev, ci.dev_lo, ci.dev_hi):
        print(f"{tau:8.3g} s  {dev:.4e}  [{lo:.4e}, {hi:.4e}]")
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stabilis")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

# __version__ must be bound before these import, since provenance reads it.
from .provenance import Provenance, Step, new_provenance  # noqa: E402
from .types import FreqSeries, PhaseSeries, to_frequency, to_phase  # noqa: E402
from .io import (  # noqa: E402
    WrittenFile,
    load_frequency,
    load_phase,
    save_frequency,
    save_phase,
)
from .edit import drop_outliers, select_range, summary  # noqa: E402
from .preprocess import basic_stats  # noqa: E402
from .stats import (  # noqa: E402
    ONE_SIGMA,
    TWO_SIGMA,
    Confidence,
    StabilityResult,
    adev,
    confidence,
    decompose,
    identify,
    identify_curve,
    mdev,
    mtie,
    oadev,
    ohdev,
    tdev,
    tierms,
    totdev,
    theo1,
    theobr,
    theoh,
    wdev,
    welch,
    Spectrum,
    script_l,
    Spur,
    SpurTable,
    detect_spurs,
    notch_spurs,
    sinusoid_dev,
)
from .multi import (  # noqa: E402
    CrossMatrix,
    CrossResult,
    DataSet,
    Rejection,
    cross_matrix,
    cross_variance,
    dataset,
    overlap,
    reject,
)
from .fitting import FitResult, fit  # noqa: E402
from .synth import power_law_freq, power_law_phase  # noqa: E402

__all__ = [
    "__version__",
    # data
    "FreqSeries",
    "PhaseSeries",
    "to_frequency",
    "to_phase",
    "load_phase",
    "load_frequency",
    "save_phase",
    "save_frequency",
    "WrittenFile",
    "basic_stats",
    # preparation
    "select_range",
    "drop_outliers",
    "summary",
    # estimators
    "adev",
    "oadev",
    "mdev",
    "tdev",
    "ohdev",
    "mtie",
    "tierms",
    "totdev",
    "theo1",
    "theobr",
    "theoh",
    "wdev",
    "StabilityResult",
    # spectra
    "welch",
    "Spectrum",
    "script_l",
    "Spur",
    "SpurTable",
    "detect_spurs",
    "notch_spurs",
    "sinusoid_dev",
    # uncertainty
    "confidence",
    "Confidence",
    "ONE_SIGMA",
    "TWO_SIGMA",
    # analysis
    "fit",
    "FitResult",
    "decompose",
    "identify",
    "identify_curve",
    "cross_variance",
    "CrossResult",
    "cross_matrix",
    "CrossMatrix",
    "reject",
    "Rejection",
    "dataset",
    "DataSet",
    "overlap",
    # synthesis
    "power_law_freq",
    "power_law_phase",
    # provenance
    "Provenance",
    "Step",
    "new_provenance",
]
