"""Exact `E[sigma^2(tau)]` from the generator's impulse response.

The synthesizer drives a unit-variance white sequence `w` through the
Kasdin-Walter recursion, `y = h * w`. Any estimator whose terms are a
*convolution* of the record -- the Allan family, the wavelet family -- then
forms each term as one fixed linear functional of `w`:

    c_t = sum_l k_l y[t - l] = sum_s g_s w[t - s],    g = k * h

so `E[c_t^2] = g . g` for every `t` past the transient, and

    E[sigma^2(tau)] = scale * (g . g)

by the `variance = scale * mean(c**2)` contract every estimator in this
library honours. **No Monte Carlo, no tolerance** -- the only approximation
is truncating `h` at `n` terms, and the truncated tail is dropped rather
than counted (see `expected_variance`).

**What this shares, and what it does not.** It reuses two things that are
independently pinned: the impulse recursion, which has exact tests at
beta = 0, +-2 and 1 (`tests/test_synth.py`), and the Daubechies filters,
which the defining algebra certifies in `tests/test_wavelet.py`. It shares
nothing with `modwt` or with any estimator -- the equivalent wavelet filter
here is built by the textbook upsampling cascade, so comparing it against
what `modwt` produces is a check of two paths, not one.

**Why it exists.** Before it, the non-Haar wavelet variances had no anchor
on their *level*: every non-Haar test in the suite was scale-free, so a
constant multiplicative error, or a wrong `tau_j <-> level` assignment,
would have passed the whole suite (VALIDATION-DEBT item 11). Haar was
anchored only because `2 nu^2 = oavar` ties it to a certified path.

**What it does not cover.** TOTVAR and the Theo family are linear
functionals of the record but *not* convolutions -- both extend the record
by reflection, which is time-varying -- so they need the full `A H H^T A^T`
form rather than a kernel, and are not handled here. MTIE is out
permanently: an extreme value is not a linear functional at all.
"""

from __future__ import annotations

from math import sqrt

import numpy as np

from ..stats.wavelet import FILTERS, wavelet_filters
from ..synth import _kasdin_walter_impulse, _unit_h_alpha

__all__ = ["IMPULSE_TERMS", "allan_kernel", "wavelet_kernel",
           "expected_variance", "expected_avar", "expected_wvar"]

# Terms of `h` retained. The recursion decays like k**(beta/2 - 1), so the
# steepest case this library generates (beta = 2, random-walk FM) does not
# decay at all and is the one that sets this: 2**16 terms put the residual
# of the checks below at 1e-11 relative, an order under the tightest
# tolerance any caller asserts at.
IMPULSE_TERMS = 1 << 16


def allan_kernel(m: int) -> np.ndarray:
    """The Allan kernel at averaging factor `m`, in convolution orientation.

        c_t = ybar_2 - ybar_1 = sum_l k_l y[t - l],
        k = [+1/m]*m + [-1/m]*m

    The most recent `m` samples carry `+1/m` because `y[t - l]` runs
    backwards in `l`, so reading `k` left to right is reading the record
    right to left. It sums to zero, which is what makes `c_t` stationary
    even when `y` is not.
    """
    if m < 1:
        raise ValueError(f"allan_kernel: m must be >= 1, got {m}")
    return np.concatenate((np.full(m, 1.0 / m), np.full(m, -1.0 / m)))


def _upsample(f: np.ndarray, step: int) -> np.ndarray:
    """`f` with `step - 1` zeros between taps: `L -> (L - 1) * step + 1`."""
    if step == 1:
        return f
    out = np.zeros((f.size - 1) * step + 1)
    out[::step] = f
    return out


def wavelet_kernel(name: str, level: int) -> np.ndarray:
    """The level-`j` MODWT equivalent wavelet filter, by the cascade.

        a_j = up(h~, 2^{j-1}) * c_{j-1},
        c_j = up(g~, 2^{j-1}) * c_{j-1},    c_0 = [1]

    with `g~ = g/sqrt(2)`, `h~ = h/sqrt(2)`, `up(f, s)` inserting `s - 1`
    zeros between taps. This is the definition of the equivalent filter
    unrolled -- `W_j` is `V_{j-1}` filtered by `h~` at stride `2^{j-1}`,
    and `V_{j-1}` is the record filtered by the scaling cascade -- and it is
    written here rather than read off `modwt` precisely so that the two can
    be compared.

    Length is `(2^j - 1)(L - 1) + 1` for an `L`-tap filter, and the norm is
    `2^{-j}` exactly, both of which are asserted in the tests: the norm is
    the orthonormality of the transform, and it is what fixes the *level*
    of the wavelet variance rather than only its shape.

    Percival & Walden, *Wavelet Methods for Time Series Analysis*, section
    4.5 (equivalent filters) and 5.3 (the MODWT variance).
    """
    if name not in FILTERS:
        raise ValueError(
            f"unknown wavelet filter {name!r}; have {sorted(FILTERS)}")
    if level < 1:
        raise ValueError(f"wavelet_kernel: level must be >= 1, got {level}")
    g, h = wavelet_filters(name)
    g, h = g / sqrt(2.0), h / sqrt(2.0)

    scaling = np.array([1.0])
    for j in range(1, level):
        scaling = np.convolve(_upsample(g, 1 << (j - 1)), scaling)
    return np.convolve(_upsample(h, 1 << (level - 1)), scaling)


def expected_variance(kernel: np.ndarray, alpha: float, scale: float,
                      n: int = IMPULSE_TERMS) -> float:
    """`E[sigma^2]` of `scale * mean(c**2)` at unit driving variance.

        g = kernel * h,    E[sigma^2] = scale * (g . g)

    The trailing `len(kernel) - 1` entries of the convolution are dropped:
    their sums run off the end of the *truncated* `h`, where the true
    process still has support and a zero-sum kernel cancels it. Keeping
    them doubles the alpha = -2 answer -- which is how the sign of that
    mistake was found, not by inspection.

    The value does not depend on `tau0`; it is a property of the sample
    sequence. Callers wanting SI units divide by `_unit_h_alpha` and
    multiply by the `h_alpha` they asked the generator for, which is what
    `expected_avar` and `expected_wvar` do.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    h = _kasdin_walter_impulse(n, beta=-alpha)
    g = np.convolve(h, kernel)[: -(kernel.size - 1)]
    return scale * float(g @ g)


def expected_avar(alpha: float, m: int, tau0: float = 1.0,
                  h_alpha: float = 1.0, n: int = IMPULSE_TERMS) -> float:
    """`E[avar(m*tau0)]` for a record generated at this `tau0` and `h_alpha`."""
    unit = expected_variance(allan_kernel(m), alpha, 0.5, n)
    return unit * h_alpha / _unit_h_alpha(alpha, tau0)


def expected_wvar(name: str, level: int, alpha: float, tau0: float = 1.0,
                  h_alpha: float = 1.0, n: int = IMPULSE_TERMS) -> float:
    """`E[nu^2(tau_j)]`, the wavelet variance at `tau_j = 2^{j-1} tau0`.

    `scale` is 1: the wavelet variance *is* the mean square of the
    coefficients. `wdev` reports `sqrt(2 nu^2)` on the Allan scale, so
    double this to compare against a deviation.
    """
    unit = expected_variance(wavelet_kernel(name, level), alpha, 1.0, n)
    return unit * h_alpha / _unit_h_alpha(alpha, tau0)
