"""Power-law noise synthesis (Kasdin & Walter).

Generates a fractional-frequency series whose one-sided PSD follows
``S_y(f) ~ f^alpha``, by convolving white noise with the impulse response

    h[0] = 1;   h[k] = h[k-1] * (k - 1 + beta/2) / k,    beta = -alpha

which is the discrete fractional integrator of order beta/2. Two sanity
checks fall straight out of the recursion, and both are asserted in the
tests:

* ``beta = 2`` (alpha = -2, random-walk FM) gives ``h = [1, 1, 1, ...]``,
  a cumulative sum. Integrating white noise gives a random walk.
* ``beta = -2`` (alpha = +2, white PM) truncates to ``h = [1, -1]``, plain
  differencing. Differencing white noise gives f^2 noise.

The RNG is always passed in explicitly. Nothing here touches numpy's global
random state, because a validation report that cannot be reproduced bit for
bit is not evidence.

**Amplitude is calibrated.** `h_alpha` is a settable parameter, not an
arbitrary scale: the generated series has ``S_y(f) = h_alpha * f**alpha``
with the `h_alpha` you asked for. See `_unit_h_alpha` for the derivation
and `theory/noise-types.md` for what it was checked against -- three of
those checks are exact rather than statistical.
"""

from __future__ import annotations

import numpy as np

from .log import record
from .provenance import new_provenance
from .types import FreqSeries, PhaseSeries, to_phase

__all__ = ["ALPHA_NAMES", "power_law_noise", "power_law_freq", "power_law_phase"]

# IEEE 1139 power-law noise types, keyed by the exponent of S_y(f).
ALPHA_NAMES = {
    2: "white PM",
    1: "flicker PM",
    0: "white FM",
    -1: "flicker FM",
    -2: "random-walk FM",
}


def power_law_noise(n: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """`n` samples of fractional frequency with ``S_y(f) ~ f^alpha``.

    Unit-scale, in the sense that the white driving sequence has unit
    variance -- *not* in the sense that the amplitude is unknown. The
    resulting `h_alpha` is `_unit_h_alpha(alpha, tau0)`, exactly. Callers
    wanting a chosen `h_alpha` use `power_law_freq`, which does the one
    division.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    h = _kasdin_walter_impulse(n, beta=-alpha)
    w = rng.standard_normal(n)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    y = np.fft.irfft(np.fft.rfft(h, nfft) * np.fft.rfft(w, nfft), nfft)
    return y[:n]


def _kasdin_walter_impulse(n: int, beta: float) -> np.ndarray:
    """The h[k] recursion above, as a cumulative product rather than a loop.

        h[k] = prod_{j=1..k} (j - 1 + beta/2) / j
    """
    j = np.arange(1, n, dtype=np.float64)
    return np.concatenate(([1.0], np.cumprod((j - 1.0 + beta / 2.0) / j)))


def _unit_h_alpha(alpha: float, tau0: float) -> float:
    """The `h_alpha` that `power_law_noise` produces at unit driving variance.

        h_unit = 2 * tau0 * (2 * pi * tau0)**alpha

    Derivation. The generator drives a unit-variance white sequence through
    `H(z) = (1 - 1/z)**(-beta/2)`, `beta = -alpha`. A unit-variance white
    sequence sampled at `tau0` has the flat one-sided PSD `2 * tau0`, and
    on the unit circle ``|1 - exp(-2j*pi*f*tau0)| = 2*sin(pi*f*tau0)``, so

        S_y(f) = 2 * tau0 * (2*sin(pi*f*tau0))**alpha

    exactly, at every `f` up to Nyquist. For `f << 1/tau0` the sine
    linearizes to `pi*f*tau0` and that becomes `h_unit * f**alpha`.

    **The sine is not a detail.** The exact discrete PSD is what the
    estimators actually see, and it departs from the ideal power law near
    Nyquist -- which is why the closed forms this is validated against hold
    exactly at some `alpha` and only asymptotically in `m` at others. The
    departures are exact, known, and tabulated in `theory/noise-types.md`;
    they are a property of discrete-time noise, not an error here.
    """
    return 2.0 * tau0 * (2.0 * np.pi * tau0) ** alpha


def power_law_freq(
    n: int,
    alpha: float,
    rate_hz: float,
    rng: np.random.Generator,
    h_alpha: float = 1.0,
) -> FreqSeries:
    """A `FreqSeries` with ``S_y(f) = h_alpha * f**alpha``, with provenance.

    `h_alpha` is the IEEE 1139 coefficient in SI units, not a multiplier:
    asking for `h_alpha=1e-26` at `alpha=0` gets you white FM whose
    `sigma_y(tau)` is `sqrt(1e-26/(2*tau))`.
    """
    if h_alpha < 0.0:
        raise ValueError(f"h_alpha is a noise power and must be >= 0, got {h_alpha}")
    tau0 = 1.0 / rate_hz
    scale = np.sqrt(h_alpha / _unit_h_alpha(alpha, tau0))
    y = power_law_noise(n, alpha, rng) * scale
    prov = record(
        new_provenance(f"synth.power_law_freq(alpha={alpha})"),
        "synth",
        "power_law_noise",
        {"n": n, "alpha": alpha, "rate_hz": rate_hz,
         "h_alpha": h_alpha, "scale": float(scale)},
        ALPHA_NAMES.get(int(alpha), f"alpha={alpha}"),
    )
    return FreqSeries(y=y, rate_hz=rate_hz, provenance=prov)


def power_law_phase(
    n: int,
    alpha: float,
    rate_hz: float,
    rng: np.random.Generator,
    h_alpha: float = 1.0,
) -> PhaseSeries:
    """A `PhaseSeries` of synthetic power-law noise.

    Generated in the frequency domain then integrated, so `n` here is the
    number of *frequency* samples and the returned phase has ``n + 1``.
    """
    return to_phase(power_law_freq(n, alpha, rate_hz, rng, h_alpha))
