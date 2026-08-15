"""Discrete tones: finding them, measuring them, and removing them loudly.

A mains harmonic, a fan, a PLL reference feedthrough -- a spur is a
sinusoid sitting on top of the noise, and it does two things worth
separating. In a spectrum it is a spike. In `sigma_y(tau)` it is a *bump*
with a closed form, derived in `theory/spur.md` and asserted by test:

    x(t) = A sin(2 pi nu t + phi)   =>   sigma_y(tau) = 2 A sin^2(pi nu tau) / tau

phase-averaged. That identity is what makes the whole module checkable:
generate a tone of known amplitude, and both the measured amplitude and the
resulting stability curve are predicted exactly.

**Amplitude is measured by fitting the sinusoid, not by reading the
spectrum.** A tone landing between bins loses amplitude to scalloping, and
the usual answer is a flat-top window with transcribed coefficients. A
least-squares fit of `cos` and `sin` at a refined frequency is exact
instead of approximately flat, needs no coefficient table, and produces the
notch as a by-product -- so the window question does not arise.

**Nothing is removed quietly.** `notch` records every tone it subtracted,
with its frequency and amplitude, in the returned series' provenance. A
spur is often physically real, and a record with one taken out is a
different record: `sigma_y(tau)` after notching describes an oscillator
that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._specfun import chi2_cdf
from ..log import record, warn
from ..provenance import Provenance
from ..types import FreqSeries, PhaseSeries

__all__ = ["Spur", "SpurTable", "detect", "notch", "sinusoid_dev",
           "detect_spurs", "notch_spurs"]


@dataclass(frozen=True)
class Spur:
    """One tone, as measured rather than as binned."""

    f_hz: float
    amplitude: float
    excess_db: float
    f_bin_hz: float

    @property
    def rms(self) -> float:
        """A sinusoid's RMS is its peak over root two."""
        return self.amplitude / np.sqrt(2.0)


@dataclass(frozen=True)
class SpurTable:
    """Every tone found, the floor they were found against, and the odds."""

    spurs: tuple[Spur, ...]
    floor: np.ndarray
    threshold_db: float
    expected_false_alarms: float
    searched_hz: tuple[float, float]
    provenance: Provenance

    def __len__(self) -> int:
        return len(self.spurs)


def sinusoid_dev(amplitude: float, f_hz: float, tau_s) -> np.ndarray:
    """The Allan deviation a pure phase tone produces, exactly.

        sigma_y(tau) = 2 * A * sin(pi*nu*tau)**2 / tau

    Derived from the second difference of `A sin(2*pi*nu*t + phi)` averaged
    over `phi`: the difference operator contributes `(exp(i*w*tau) - 1)**2`,
    whose modulus is `4 sin(w*tau/2)**2`, and the phase average halves the
    square. Verified against this library's own `adev` to five significant
    figures over seven octaves.

    The zeros at `nu*tau = 1, 2, ...` are real and are the reason a spur can
    hide from a stability plot: at exactly those taus the second difference
    is blind to it.
    """
    tau = np.asarray(tau_s, dtype=np.float64)
    return 2.0 * amplitude * np.sin(np.pi * f_hz * tau) ** 2 / tau


def detect(spec, series, threshold_db: float = 10.0, max_spurs: int = 16,
           floor_bins: int = 33) -> SpurTable:
    """Tones standing `threshold_db` above the local noise floor.

    The floor is a running median of the spectrum rather than a fit, so a
    sloping power-law background is followed without assuming which power
    law it is, and a tone -- being a few bins wide at most -- cannot pull
    the median that describes it.

    Each candidate's frequency and amplitude are then refined against the
    *record*, not the spectrum: the bin only says where to look.
    """
    if spec.kind not in ("S_x", "S_y"):
        raise ValueError(
            f"spur detection needs the spectrum in the units the record is "
            f"in, and this one is {spec.kind!r}. Detect on the native "
            "spectrum, before converting."
        )
    values, _ = _values(series)
    f, psd = spec.f[1:], spec.psd[1:]              # DC carries the detrend
    floor = _running_median(psd, floor_bins)
    excess_db = 10.0 * np.log10(np.where(floor > 0, psd / floor, 1.0))

    peaks = _peaks(excess_db, threshold_db, max_spurs, floor_bins // 2)
    spurs = tuple(
        _refine(values, series.rate_hz, float(f[i]), spec.df_hz,
                float(excess_db[i]))
        for i in peaks
    )

    edge = floor_bins // 2
    searched = (float(f[edge]), float(f[-edge - 1]))
    false_alarms = _false_alarm_count(threshold_db, spec.n_segments,
                                      max(0, f.size - 2 * edge))
    prov = record(
        series.provenance, "stats.spur", "detect",
        {"threshold_db": threshold_db, "floor_bins": floor_bins,
         "n_segments": spec.n_segments, "max_spurs": max_spurs,
         "searched_hz": [round(v, 12) for v in searched]},
        f"{len(spurs)} tones in {searched[0]:.4g}..{searched[1]:.4g} Hz; "
        f"{false_alarms:.2g} expected by chance",
    )
    if len(spurs) == max_spurs:
        prov = warn(prov, "stats.spur", "truncated", {"max_spurs": max_spurs},
                    "the candidate list filled up, so weaker tones may "
                    "remain undetected; raise max_spurs to see them")
    if false_alarms > 0.5 and spurs:
        prov = warn(prov, "stats.spur", "false_alarm_rate",
                    {"expected": float(false_alarms)},
                    f"at this threshold and {spec.n_segments} segments, "
                    f"{false_alarms:.2g} bins are expected to exceed it by "
                    "chance alone; treat the weakest entries with suspicion")
    return SpurTable(spurs=spurs, floor=floor, threshold_db=threshold_db,
                     expected_false_alarms=float(false_alarms),
                     searched_hz=searched, provenance=prov)


def notch(series, spurs):
    """Subtract the fitted tones, and record every one that was removed.

    Returns a series of the same type. The removal is an *edit*: a stability
    curve computed afterwards describes an oscillator with no such tone,
    which is a different oscillator. That is legitimate when the tone is
    known to be measurement pickup and misleading when it is not, so the
    decision stays with the caller and the record of it stays with the data.
    """
    spurs = tuple(spurs.spurs if isinstance(spurs, SpurTable) else spurs)
    if not spurs:
        raise ValueError(
            "no tones given, so there is nothing to notch. `detect` returns "
            "an empty table when it finds none; removing nothing quietly "
            "would leave a provenance entry claiming an edit that did not "
            "happen."
        )
    values, kind = _values(series)
    t = np.arange(values.size, dtype=np.float64) / series.rate_hz
    out = values.copy()
    finite = np.isfinite(out)
    for s in spurs:
        # The same design `_refine` measured with -- constant and ramp
        # included, then discarded. Fitting the tone alone lets a phase
        # record's frequency offset leak into the sinusoid's coefficients,
        # which left 0.3% of a 3e-11 tone behind and still 25 dB above the
        # floor. The trend belongs to the oscillator and stays.
        design = _design(t, s.f_hz)
        coeffs, *_ = np.linalg.lstsq(design[finite], out[finite], rcond=None)
        out -= design[:, :2] @ coeffs[:2]

    prov = record(
        series.provenance, "stats.spur", "notch",
        {"n_tones": len(spurs),
         "f_hz": [round(s.f_hz, 12) for s in spurs],
         "amplitude": [s.amplitude for s in spurs]},
        f"{len(spurs)} tones subtracted; the record no longer describes the "
        "device as measured",
    )
    if kind == "phase":
        return PhaseSeries(x=out, rate_hz=series.rate_hz, epoch=series.epoch,
                           provenance=prov)
    return FreqSeries(y=out, rate_hz=series.rate_hz, epoch=series.epoch,
                      provenance=prov)


# --------------------------------------------------------------------------


def _values(series):
    if isinstance(series, PhaseSeries):
        return np.asarray(series.x, dtype=np.float64), "phase"
    if isinstance(series, FreqSeries):
        return np.asarray(series.y, dtype=np.float64), "frequency"
    raise TypeError(f"expected a PhaseSeries or FreqSeries, got {type(series)}")


def _running_median(psd: np.ndarray, width: int) -> np.ndarray:
    """Median over a sliding window, edges included by shrinking the window.

    A median rather than a mean because the thing being measured is an
    outlier: a mean floor is raised by the very tone it is meant to make
    visible, and a strong spur can hide a weak one nearby by inflating the
    background between them.
    """
    if width < 3 or width % 2 == 0:
        raise ValueError(f"floor_bins must be odd and at least 3, got {width}")
    half = width // 2
    n = psd.size
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.median(psd[lo:hi])
    return out


def _peaks(excess_db: np.ndarray, threshold_db: float, max_spurs: int,
           edge: int):
    """Local maxima of the excess, strongest first, one per cluster.

    Adjacent bins over threshold are one tone seen through the window's main
    lobe, not several, so a cluster contributes its peak and nothing else.

    The outermost `edge` bins are not searched, and that exclusion is not
    tidiness. Their floor is a median over a *truncated* window, so it is
    estimated from fewer bins and from systematically different ones; and at
    the low end they also carry the residue of per-segment detrending, which
    `theory/psd.md` records as the usual Welch trade. Together those
    produced a confident 14.8 dB detection of a tone that was not there --
    found while writing this, and the reason the band is now stated rather
    than assumed to be the whole spectrum.
    """
    over = excess_db > threshold_db
    if edge > 0:
        over[:edge] = False
        over[-edge:] = False
    if not over.any():
        return []
    peaks = []
    i = 0
    while i < over.size:
        if not over[i]:
            i += 1
            continue
        j = i
        while j < over.size and over[j]:
            j += 1
        peaks.append(i + int(np.argmax(excess_db[i:j])))
        i = j
    peaks.sort(key=lambda k: -excess_db[k])
    return peaks[:max_spurs]


def _refine(values, rate_hz, f_bin, df_hz, excess_db) -> Spur:
    """Frequency and amplitude from the record, with the bin as a hint.

    Scans a grid across the bin the peak was found in, keeping the
    frequency that explains the most variance, then interpolates the
    maximum parabolically. Fitting rather than reading the spectrum is what
    makes the amplitude exact for a tone that fell between bins.

    A constant and a linear term ride in the design matrix so that a
    frequency offset in a phase record cannot be absorbed into the tone.
    """
    finite = np.isfinite(values)
    t = np.arange(values.size, dtype=np.float64) / rate_hz
    y = values[finite]
    tf = t[finite]

    # Three passes, each narrowing the bracket by the previous grid step.
    # One pass is not enough, and the shortfall is invisible in the fit
    # itself: a frequency off by ~1e-6 Hz still explains almost all the
    # variance, but leaves 0.3% of a 3e-11 tone behind -- which is 25 dB
    # above the floor and detected again as a tone in its own right.
    f_hat, half = f_bin, df_hz
    for _ in range(3):
        grid = f_hat + np.linspace(-half, half, 65)
        grid = grid[grid > 0.0]
        if grid.size < 3:
            break
        power = np.array([_explained(tf, y, f) for f in grid])
        f_hat = _parabolic(grid, power, int(np.argmax(power)))
        half = 2.0 * half / (grid.size - 1)

    amp, _ = _fit(tf, y, f_hat)
    return Spur(f_hz=float(f_hat), amplitude=float(amp),
                excess_db=float(excess_db), f_bin_hz=float(f_bin))


def _design(t, f):
    return np.column_stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t),
                            np.ones_like(t), t])


def _fit(t, y, f):
    """Least squares by normal equations -- 4x4, so this stays cheap."""
    x = _design(t, f)
    gram = x.T @ x
    rhs = x.T @ y
    try:
        coeffs = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:            # pragma: no cover - degenerate
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(np.hypot(coeffs[0], coeffs[1])), coeffs


def _explained(t, y, f) -> float:
    """Variance the tone accounts for; maximizing it minimizes the residual."""
    _, coeffs = _fit(t, y, f)
    return float(np.sum((_design(t, f) @ coeffs) ** 2))


def _parabolic(grid, power, k) -> float:
    """Sub-grid maximum from three points, or the grid point at an edge."""
    if k == 0 or k == grid.size - 1:
        return float(grid[k])
    a, b, c = power[k - 1], power[k], power[k + 1]
    denom = a - 2.0 * b + c
    if denom == 0.0:
        return float(grid[k])
    shift = 0.5 * (a - c) / denom
    return float(grid[k] + shift * (grid[1] - grid[0]))


def _false_alarm_count(threshold_db, n_segments, n_bins) -> float:
    """How many bins should clear the threshold on noise alone.

    Each Welch bin is chi-square with about `2*n_segments` degrees of
    freedom, so the tail is exact and worth printing: a detector that
    cannot say how often it cries wolf is not a measurement.

    Two approximations are deliberately not hidden. Overlapping segments
    are correlated, so the true degrees of freedom are below `2*n`, which
    makes this an *under*-estimate. And the floor is itself estimated,
    which adds scatter this does not model. Both push the same way, so the
    printed number is a lower bound on the false-alarm rate.
    """
    dof = 2.0 * max(1, n_segments)
    ratio = 10.0 ** (threshold_db / 10.0)
    tail = 1.0 - float(chi2_cdf(dof * ratio, dof))
    return max(0.0, tail) * n_bins


# `stats` and the top level re-export these under longer names. Inside this
# module `detect` and `notch` read best; from `stabilis import notch` they
# would read as if the library only ever notched one thing.
detect_spurs = detect
notch_spurs = notch
