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

    Unit-scale: the output has no calibrated `h_alpha`. Amplitude
    calibration against the IEEE 1139 coefficients arrives with the
    Monte-Carlo unbiasedness tests, which is the only place it matters.
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


def power_law_freq(
    n: int,
    alpha: float,
    rate_hz: float,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> FreqSeries:
    """A `FreqSeries` of synthetic power-law noise, with provenance."""
    y = power_law_noise(n, alpha, rng) * scale
    prov = record(
        new_provenance(f"synth.power_law_freq(alpha={alpha})"),
        "synth",
        "power_law_noise",
        {"n": n, "alpha": alpha, "rate_hz": rate_hz, "scale": scale},
        ALPHA_NAMES.get(int(alpha), f"alpha={alpha}"),
    )
    return FreqSeries(y=y, rate_hz=rate_hz, provenance=prov)


def power_law_phase(
    n: int,
    alpha: float,
    rate_hz: float,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> PhaseSeries:
    """A `PhaseSeries` of synthetic power-law noise.

    Generated in the frequency domain then integrated, so `n` here is the
    number of *frequency* samples and the returned phase has ``n + 1``.
    """
    return to_phase(power_law_freq(n, alpha, rate_hz, rng, scale))
