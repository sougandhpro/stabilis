"""Immutable containers for phase and fractional-frequency data.

Two conventions here are load-bearing.

**Phase is stored as offsets from an explicit epoch.** float64 has ~15-16
significant digits, so a phase value of magnitude 1e5 s has a spacing of
~1.5e-11 s. Storing absolute MJD-derived seconds would silently quantize
away exactly the picoseconds that sub-nanosecond work cares about.
`precision_ratio` measures this and construction warns when it bites.

**NaN marks a gap.** It propagates correctly through differencing, so an
estimator that drops non-finite terms is automatically gap-aware. NaN
cannot, however, distinguish "sample missing on a known grid" from
"counter was off and phase resumed with an unknown offset" -- the second
needs segment-wise analysis, or the phase step becomes a giant fake
frequency excursion. See `preprocess.segments`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .log import warn
from .provenance import Provenance, new_provenance

__all__ = [
    "PhaseSeries",
    "FreqSeries",
    "to_frequency",
    "to_phase",
    "precision_ratio",
    "PRECISION_WARN_RATIO",
]

# Warn when float64 spacing at the largest phase magnitude exceeds this
# fraction of the smallest resolved step in the data. At a ratio of 1 the
# measurement resolution has been entirely consumed by the representation.
PRECISION_WARN_RATIO = 1e-3


def _as_readonly(a) -> np.ndarray:
    """Coerce to a read-only 1-D float64 array. Immutable means immutable."""
    arr = np.asarray(a, dtype=np.float64).reshape(-1)
    arr = arr.copy() if arr.flags.owndata is False else arr
    arr.flags.writeable = False
    return arr


def precision_ratio(x: np.ndarray) -> float:
    """float64 spacing at peak magnitude, over the smallest resolved step.

        ratio = spacing(max|x|) / min(|diff x| > 0)

    A ratio near 1 means the representation has quantized away the
    measurement. Returns 0.0 when the question is not answerable (fewer
    than two finite points, or a perfectly constant series).
    """
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return 0.0
    steps = np.abs(np.diff(finite))
    steps = steps[steps > 0.0]
    if steps.size == 0:
        return 0.0
    return float(np.spacing(np.abs(finite).max()) / steps.min())


@dataclass(frozen=True)
class PhaseSeries:
    """Time error x(t) in seconds, sampled uniformly at `rate_hz`.

    `x[i]` is the phase at absolute time `epoch + i / rate_hz`, expressed as
    an offset from the phase at `epoch`. Gaps are NaN.
    """

    x: np.ndarray
    rate_hz: float
    epoch: float = 0.0
    provenance: Provenance = field(
        default_factory=lambda: new_provenance("in-memory array")
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _as_readonly(self.x))
        if self.rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {self.rate_hz}")
        if self.x.size == 0:
            raise ValueError("phase series is empty")
        ratio = precision_ratio(self.x)
        if ratio > PRECISION_WARN_RATIO:
            object.__setattr__(
                self,
                "provenance",
                warn(
                    self.provenance,
                    "types",
                    "precision_check",
                    {"ratio": f"{ratio:.2e}", "peak_s": f"{np.nanmax(np.abs(self.x)):.3e}"},
                    "float64 spacing is a significant fraction of the resolved "
                    "step; subtract an epoch or store smaller offsets",
                ),
            )

    @property
    def tau0_s(self) -> float:
        """Sample interval."""
        return 1.0 / self.rate_hz

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def duration_s(self) -> float:
        """Span of the record, first sample to last."""
        return (self.n - 1) * self.tau0_s

    @property
    def n_gaps(self) -> int:
        return int(np.count_nonzero(~np.isfinite(self.x)))


@dataclass(frozen=True)
class FreqSeries:
    """Fractional frequency y(t), dimensionless, sampled at `rate_hz`.

    `y[i]` is the average fractional frequency over
    `[epoch + i/rate_hz, epoch + (i+1)/rate_hz]`. Gaps are NaN.
    """

    y: np.ndarray
    rate_hz: float
    epoch: float = 0.0
    provenance: Provenance = field(
        default_factory=lambda: new_provenance("in-memory array")
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "y", _as_readonly(self.y))
        if self.rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {self.rate_hz}")
        if self.y.size == 0:
            raise ValueError("frequency series is empty")

    @property
    def tau0_s(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def n(self) -> int:
        return int(self.y.size)

    @property
    def duration_s(self) -> float:
        """Total averaging span covered by the record."""
        return self.n * self.tau0_s

    @property
    def n_gaps(self) -> int:
        return int(np.count_nonzero(~np.isfinite(self.y)))


def to_frequency(ps: PhaseSeries) -> FreqSeries:
    """Fractional frequency from phase, exactly.

        y[i] = (x[i+1] - x[i]) / tau0

    Length drops by one. NaN in x propagates into the two y samples that
    straddle it, which is the correct gap behaviour.
    """
    y = np.diff(ps.x) * ps.rate_hz
    return FreqSeries(
        y=y,
        rate_hz=ps.rate_hz,
        epoch=ps.epoch,
        provenance=_note(ps.provenance, "to_frequency", ps.n, y.size),
    )


def to_phase(fs: FreqSeries) -> PhaseSeries:
    """Phase from fractional frequency, by cumulative sum.

        x[0] = 0;  x[k] = sum_{i<k} y[i] * tau0

    Length grows by one. The integration constant is unrecoverable, so
    x[0] is set to zero and that choice is recorded -- every stability
    statistic here is invariant to it, but a caller comparing absolute
    phase is not.

    Round-tripping x -> y -> x returns the offset-removed phase to within a
    few ULP, not bit-exactly: a cumulative sum of differences does not
    telescope exactly in floating point. Measured accumulation is 0.2 ULP at
    n=10, 1.0 at n=1e3, 2.5 at n=1e5. Bounded in test_types.py.
    """
    x = np.concatenate(([0.0], np.cumsum(fs.y) * fs.tau0_s))
    return PhaseSeries(
        x=x,
        rate_hz=fs.rate_hz,
        epoch=fs.epoch,
        provenance=_note(fs.provenance, "to_phase", fs.n, x.size),
    )


def _note(prov: Provenance, action: str, n_in: int, n_out: int) -> Provenance:
    """Record a domain conversion. Length change is the auditable part."""
    from .log import record

    return record(prov, "types", action, {"n_in": n_in}, f"n_out={n_out}")
