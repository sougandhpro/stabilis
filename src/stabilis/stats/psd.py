"""Power spectral density: the frequency-domain view of the same noise.

Welch's method, one-sided, with the window's noise-equivalent bandwidth
divided out so the result is a density rather than a periodogram in
disguise.

Three things this module refuses to guess, each because the mistake is
invisible afterwards:

* **the carrier frequency.** `S_phi` and `L(f)` are `S_x` scaled by
  `(2*pi*nu0)**2`. A spectrum converted with an assumed carrier is wrong by
  a constant, and nothing downstream can tell. `nu0_hz` is required and
  recorded.
* **the small-angle approximation.** `L(f) = S_phi(f)/2` holds only while
  the integrated phase modulation above `f` is small. Close to the carrier
  it is not, and the honest output there is no number at all.
* **what was detrended.** A phase record carries a frequency offset that
  dominates every low-frequency bin. Removing it is necessary and is a
  decision, so it goes in provenance rather than in a default nobody sees.

The conversions, with `x` the time error in seconds and `y = dx/dt`:

    S_y(f)   = (2*pi*f)**2   * S_x(f)      1/Hz
    S_phi(f) = (2*pi*nu0)**2 * S_x(f)      rad**2/Hz
    L(f)     = S_phi(f) / 2                rad**2/Hz, reported in dBc/Hz
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..log import record, warn
from ..provenance import Provenance
from ..types import FreqSeries, PhaseSeries
from .window import WINDOWS, window

__all__ = ["Spectrum", "KINDS", "DETRENDS", "SMALL_ANGLE_RAD2", "welch",
           "convert", "script_l", "discrete_integration_gain"]

KINDS = ("S_x", "S_y", "S_phi")
DETRENDS = ("none", "mean", "linear")

# Mean-square phase deviation above `f`, in rad^2, beyond which
# `L = S_phi/2` is no longer the small-angle result. 0.1 rad^2 is one tenth
# of a radian squared -- a 1.6% error in the Bessel expansion, and the
# conventional place the approximation is said to fail.
SMALL_ANGLE_RAD2 = 0.1


@dataclass(frozen=True)
class Spectrum:
    """A one-sided power spectral density, and how it was measured."""

    f: np.ndarray
    psd: np.ndarray
    kind: str
    rate_hz: float
    n_segments: int
    nperseg: int
    window: str
    enbw_hz: float
    detrend: str
    provenance: Provenance
    nu0_hz: float | None = None

    @property
    def df_hz(self) -> float:
        """Bin spacing. *Not* the resolution -- that is `enbw_hz`."""
        return self.rate_hz / self.nperseg

    @property
    def units(self) -> str:
        return {"S_x": "s^2/Hz", "S_y": "1/Hz", "S_phi": "rad^2/Hz"}[self.kind]


def welch(series, nperseg: int | None = None, overlap: float = 0.5,
          window_name: str = "hann", detrend: str | None = None,
          nu0_hz: float | None = None) -> Spectrum:
    """One-sided PSD by Welch's method: `S_x` from phase, `S_y` from frequency.

        S(f_k) = 2 * |FFT(w * x)|**2 / (rate * sum(w**2))

    averaged over overlapping segments, with the DC and Nyquist bins not
    doubled since they have no negative-frequency twin. The `sum(w**2)`
    denominator is what makes this a density: it divides out the window's
    power, so Hann and rectangular give the same level for the same noise.

    Gaps segment the record rather than being masked -- the same rule the
    variance estimators follow, and for the same reason: an FFT of a record
    with a NaN in it is entirely NaN, and filling the hole would invent
    data. Segments shorter than `nperseg` are dropped and counted.
    """
    values, kind = _series_view(series)
    detrend = _default_detrend(detrend, kind)
    if detrend not in DETRENDS:
        raise ValueError(f"unknown detrend {detrend!r}; expected {DETRENDS}")
    if window_name not in WINDOWS:
        raise ValueError(f"unknown window {window_name!r}; expected {WINDOWS}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    runs = _gap_free_runs(values)
    nperseg = _choose_nperseg(nperseg, runs)
    w = window(window_name, nperseg)
    step = max(1, int(round(nperseg * (1.0 - overlap))))

    total = np.zeros(nperseg // 2 + 1)
    n_segments = dropped = 0
    for start, stop in runs:
        if stop - start < nperseg:
            dropped += 1
            continue
        for a in range(start, stop - nperseg + 1, step):
            total += _segment_psd(values[a:a + nperseg], w, series.rate_hz,
                                  detrend)
            n_segments += 1
    if n_segments == 0:
        raise ValueError(
            f"no gap-free run reaches {nperseg} samples, so no segment could "
            f"be transformed. The longest run is "
            f"{max((b - a) for a, b in runs)} samples; pass a smaller "
            "nperseg, or select a clean span with edit.select_range."
        )

    psd = total / n_segments
    f = np.fft.rfftfreq(nperseg, d=1.0 / series.rate_hz)
    enbw = series.rate_hz * float(np.sum(w * w)) / float(np.sum(w)) ** 2

    prov = record(
        series.provenance, "stats.psd", "welch",
        {"nperseg": nperseg, "overlap": overlap, "window": window_name,
         "detrend": detrend, "n_segments": n_segments,
         "runs_too_short": dropped},
        f"{kind} over {n_segments} segments, enbw {enbw:.4g} Hz",
    )
    if dropped:
        prov = warn(prov, "stats.psd", "runs_dropped",
                    {"dropped": dropped, "nperseg": nperseg},
                    f"{dropped} gap-free runs were shorter than nperseg and "
                    "contributed nothing; the spectrum is of the rest")
    return Spectrum(f=f, psd=psd, kind=kind, rate_hz=series.rate_hz,
                    n_segments=n_segments, nperseg=nperseg,
                    window=window_name, enbw_hz=enbw, detrend=detrend,
                    provenance=prov, nu0_hz=nu0_hz)


def convert(spec: Spectrum, kind: str, nu0_hz: float | None = None) -> Spectrum:
    """Between `S_x`, `S_y` and `S_phi`, exactly.

        S_y = (2*pi*f)**2 * S_x        S_phi = (2*pi*nu0)**2 * S_x

    The DC bin of `S_y` is identically zero and of `S_x` is infinite, since
    the two differ by `f**2`. Rather than emit an infinity that will be
    plotted, the DC bin of a spectrum converted *to* `S_x` is dropped, and
    the fact is recorded.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if kind == spec.kind:
        return spec
    nu0 = nu0_hz if nu0_hz is not None else spec.nu0_hz
    if "S_phi" in (kind, spec.kind) and not nu0:
        raise ValueError(
            "S_phi is S_x scaled by (2*pi*nu0)**2, and the carrier frequency "
            "is not in the data. Pass nu0_hz: a spectrum converted with an "
            "assumed carrier is wrong by a constant that nothing downstream "
            "can detect."
        )

    sx, f, prov = _to_sx(spec, nu0)
    if kind == "S_x":
        psd = sx
    elif kind == "S_y":
        psd = (2.0 * np.pi * f) ** 2 * sx
    else:
        psd = (2.0 * np.pi * nu0) ** 2 * sx
    prov = record(prov, "stats.psd", "convert",
                  {"from": spec.kind, "to": kind, "nu0_hz": nu0}, "ok")
    return replace(spec, f=f, psd=psd, kind=kind, provenance=prov,
                   nu0_hz=nu0 if nu0 else spec.nu0_hz)


def discrete_integration_gain(f: np.ndarray, tau0_s: float) -> np.ndarray:
    """How much a cumulative sum exceeds an ideal integrator, by frequency.

        gain(f) = (pi*f*tau0)**2 / sin(pi*f*tau0)**2

    `convert` uses `S_y = (2*pi*f)**2 * S_x`, which is the *continuous-time*
    definition and is what IEEE 1139 means. `to_phase` integrates by
    cumulative sum, whose transfer function is `tau0 / (1 - exp(-j*w*tau0))`
    rather than `1/(j*w)`, so the phase record it produces has more
    high-frequency content than the continuous identity predicts -- by this
    factor, rising to `(pi/2)**2 = 2.467` at Nyquist.

    Measured against synthesized white FM it holds: 2.272 against 2.246
    predicted in the top half-decade, 1.075 against 1.081 near f/10. It is
    a property of sampled data, not an error, and it is the reason a
    spectrum taken from phase and one taken from the same record's
    frequency do not agree near Nyquist.

    Returned rather than applied. A caller comparing two spectra of one
    record wants it; a caller reporting `S_y` to a datasheet does not, and
    a switch inside `convert` would let the wrong one happen quietly.
    """
    u = np.pi * np.asarray(f, dtype=np.float64) * tau0_s
    return np.where(u == 0.0, 1.0, u ** 2 / np.sin(np.where(u == 0.0, 1.0, u)) ** 2)


def script_l(spec: Spectrum, nu0_hz: float | None = None):
    """`L(f)` in dBc/Hz, and the mask of where the small-angle result holds.

        L(f) = S_phi(f) / 2,   valid while  integral_f^fN S_phi df << 1 rad^2

    Returns `(f, dbc, valid)`. The invalid entries are NaN rather than a
    plausible number: close to the carrier the phase excursion is not small,
    `L` is not `S_phi/2`, and printing the formula's output anyway produces
    exactly the near-carrier curve that a datasheet would be written from.

    The integral runs to Nyquist because that is as far as the measurement
    goes; a record sampled faster would push the boundary further out.
    """
    sphi = convert(spec, "S_phi", nu0_hz)
    f, psd = sphi.f, sphi.psd

    # Mean-square phase above each f: the reverse cumulative integral.
    df = np.diff(f, prepend=f[0] - sphi.df_hz)
    above = np.cumsum((psd * df)[::-1])[::-1]
    valid = above < SMALL_ANGLE_RAD2

    dbc = np.full(f.size, np.nan)
    ok = valid & (psd > 0.0)
    dbc[ok] = 10.0 * np.log10(psd[ok] / 2.0)
    return f, dbc, valid


# --------------------------------------------------------------------------


def _series_view(series):
    """The data column and which spectrum it natively produces."""
    if isinstance(series, PhaseSeries):
        return np.asarray(series.x, dtype=np.float64), "S_x"
    if isinstance(series, FreqSeries):
        return np.asarray(series.y, dtype=np.float64), "S_y"
    raise TypeError(f"expected a PhaseSeries or FreqSeries, got {type(series)}")


def _default_detrend(detrend, kind: str) -> str:
    """Linear for phase, mean for frequency -- and never silently.

    A phase record's linear term *is* its frequency offset, which is
    physically real and spectrally enormous: leaving it in puts a 1/f**2
    skirt over every low-frequency bin. Removing it is right and is a
    decision, so the choice is recorded either way.
    """
    if detrend is not None:
        return detrend
    return "linear" if kind == "S_x" else "mean"


def _gap_free_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges of contiguous finite samples."""
    finite = np.isfinite(values)
    if finite.all():
        return [(0, values.size)]
    edges = np.diff(finite.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if finite[0]:
        starts.insert(0, 0)
    if finite[-1]:
        stops.append(values.size)
    runs = [(a, b) for a, b in zip(starts, stops) if b > a]
    if not runs:
        raise ValueError("the record has no finite samples")
    return runs


def _choose_nperseg(nperseg, runs) -> int:
    """A power of two giving roughly eight segments of the longest run.

    Eight is the usual compromise: the variance of a Welch estimate falls
    as 1/segments while the lowest resolvable frequency rises with them.
    Chosen from the longest *gap-free* run rather than the record length,
    since a segment that spans a gap cannot be transformed at all.
    """
    longest = max(b - a for a, b in runs)
    if nperseg is not None:
        if nperseg < 2:
            raise ValueError(f"nperseg must be at least 2, got {nperseg}")
        return int(nperseg)
    target = max(16, longest // 8)
    return int(2 ** int(np.floor(np.log2(target))))


def _segment_psd(seg: np.ndarray, w: np.ndarray, rate_hz: float,
                 detrend: str) -> np.ndarray:
    """One segment's one-sided periodogram, normalized to a density."""
    seg = _detrend(seg, detrend)
    spec = np.fft.rfft(seg * w)
    psd = 2.0 * (spec.real ** 2 + spec.imag ** 2) / (rate_hz * np.sum(w * w))
    # DC has no negative-frequency twin, nor does Nyquist on an even length.
    psd[0] *= 0.5
    if seg.size % 2 == 0:
        psd[-1] *= 0.5
    return psd


def _detrend(seg: np.ndarray, how: str) -> np.ndarray:
    if how == "none":
        return seg
    if how == "mean":
        return seg - seg.mean()
    k = np.arange(seg.size, dtype=np.float64)
    slope, intercept = np.polyfit(k, seg, 1)
    return seg - (slope * k + intercept)


def _to_sx(spec: Spectrum, nu0):
    """`S_x` and its frequency axis, dropping DC where the conversion blows up."""
    f, psd = spec.f, spec.psd
    if spec.kind == "S_x":
        return psd, f, spec.provenance
    prov = spec.provenance
    if f[0] == 0.0:
        f, psd = f[1:], psd[1:]
        prov = record(prov, "stats.psd", "drop_dc", {"from": spec.kind},
                      "S_x is S_y/(2*pi*f)**2 and diverges at f=0; the DC "
                      "bin is dropped rather than returned as infinity")
    if spec.kind == "S_y":
        return psd / (2.0 * np.pi * f) ** 2, f, prov
    return psd / (2.0 * np.pi * nu0) ** 2, f, prov
