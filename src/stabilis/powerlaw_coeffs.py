"""Power-law conversion coefficients: `sigma2_y(tau)` from `S_y(f) = h_a f^a`.

Data, not behaviour. Kept in its own module so the table can be diffed
against the source by eye without reading the solver around it.

**Provenance of every number here.** The Allan rows are transcribed from
NIST SP 1065 (Riley, *Handbook of Frequency Stability Analysis*), section
7.1 "Power Law Domain Conversions", eq. (67) and the table on p. 74, read
from the digital original at `tf.nist.gov/general/pdf/2220.pdf`. The
handbook gives **no** power-law conversions for the modified Allan
variance -- section 7.1 says its formulae extend to ThetaH, and stops --
so the modified rows are *derived* here by quadrature of the handbook's own
eq. (66), and each carries the evidence for its closed form. See
`theory/powerlaw.md`.

Two rows depend on `f_h`, the upper cutoff frequency of the measuring
system. `f_h` is **not identifiable from tau-domain data at all**: white PM
and flicker PM have the same Allan tau-slope, which is why eq. (67)'s own
introduction says the time-to-frequency conversion "is not unique". It is a
recorded input, never inferred, and every row is valid only where
`2*pi*f_h*tau >> 1`.

The modified rows carry no `f_h` at all. That is not an omission -- it is
the whole reason the decomposition stacks AVAR and MVAR rows. The modified
estimator's bandwidth scales as `1/tau`, so its white-PM row falls as
`tau**-3` where flicker PM falls as `tau**-2`, and the degeneracy that
makes AVAR alone unable to separate them is broken.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ALPHAS", "FH_DEPENDENT", "MIN_FH_TAU_PRODUCT", "avar_coeff",
           "mvar_coeff", "drift_coeff", "MVAR_FLICKER_FM_COEFF"]

# The five IEEE 1139 power laws, most-blue first, which is the column order
# of the decomposition's design matrix.
ALPHAS = (2, 1, 0, -1, -2)

# Rows whose coefficient depends on the measurement bandwidth. Only the
# unmodified Allan rows for the two PM noises.
FH_DEPENDENT = frozenset({(2, False), (1, False)})

# A row is used only where `2*pi*f_h*tau` clears this. The conversions are
# asymptotic in that product, and SP 1065 states they "can be applied only
# when the dominant noise process decreases toward higher sideband
# frequencies". Ten is the conventional floor; below it the coefficient is
# not merely imprecise but wrong in a way no error bar reports.
MIN_FH_TAU_PRODUCT = 10.0

# Mod AVAR for flicker FM, as a pure number: `Mod sigma2_y = K * h_-1`.
#
# Derived, not transcribed, and the only coefficient here with no closed
# form. Quadrature of SP 1065 eq. (66) gives K(m) = 0.9352449 at m=128,
# 0.9352320 at 256, 0.9352288 at 512, 0.9352280 at 1024, converging as
# 1/m**2 to the value below. A search over `a*ln2 + b*ln3` with small
# rational a, b found nothing within 2e-9, so the number is shipped as a
# number with its derivation rather than dressed up as a closed form it
# does not have. `27*ln(2)/20 = 0.9357487` is close and is **not** it --
# it misses in the fourth figure, well outside the quadrature's own
# convergence.
MVAR_FLICKER_FM_COEFF = 0.9352277


def avar_coeff(alpha: int, tau: np.ndarray, f_h: float) -> np.ndarray:
    """Coefficient of `h_alpha` in `sigma2_y(tau)`, unmodified Allan.

        alpha = -2   (2*pi**2/3) * tau
        alpha = -1   2*ln(2)
        alpha =  0   1/(2*tau)
        alpha = +1   [1.038 + 3*ln(2*pi*f_h*tau)] / (4*pi**2*tau**2)
        alpha = +2   3*f_h / (4*pi**2*tau**2)

    SP 1065 eq. (67) and p. 74. **The flicker-PM row is printed two
    different ways in the source and they disagree**; the form above is the
    one the handbook's own eq. (65) supports. See `theory/powerlaw.md` for
    the arithmetic that settles it -- this is the single place in this
    library where a published formula is not reproduced verbatim, so it is
    documented twice and tested three ways.
    """
    tau = np.asarray(tau, dtype=np.float64)
    if alpha == -2:
        return 2.0 * np.pi**2 / 3.0 * tau
    if alpha == -1:
        return np.full_like(tau, 2.0 * np.log(2.0))
    if alpha == 0:
        return 1.0 / (2.0 * tau)
    if alpha == 1:
        return (1.038 + 3.0 * np.log(2.0 * np.pi * f_h * tau)) / (
            4.0 * np.pi**2 * tau**2)
    if alpha == 2:
        return 3.0 * f_h / (4.0 * np.pi**2 * tau**2)
    raise ValueError(f"alpha must be one of {ALPHAS}, got {alpha}")


def mvar_coeff(alpha: int, tau: np.ndarray) -> np.ndarray:
    """Coefficient of `h_alpha` in `Mod sigma2_y(tau)`. No `f_h` anywhere.

        alpha = -2   (11*pi**2/20) * tau
        alpha = -1   MVAR_FLICKER_FM_COEFF
        alpha =  0   1/(4*tau)
        alpha = +1   (24*ln(2) - 9*ln(3)) / (8*pi**2*tau**2)
        alpha = +2   3 / (8*pi**2*tau**3)

    Derived by quadrature of SP 1065 eq. (66); the handbook tabulates no
    modified conversions. Each was checked against the quadrature at
    m = 32, 128, 512 and at two `tau0`, and all five against the shipped
    `mdev` over 300 realizations of calibrated synthetic noise.

    The `tau**-3` in the white-PM row against `tau**-2` in flicker PM is
    the degeneracy-breaking difference the whole stacked fit rests on.
    """
    tau = np.asarray(tau, dtype=np.float64)
    if alpha == -2:
        return 11.0 * np.pi**2 / 20.0 * tau
    if alpha == -1:
        return np.full_like(tau, MVAR_FLICKER_FM_COEFF)
    if alpha == 0:
        return 1.0 / (4.0 * tau)
    if alpha == 1:
        return (24.0 * np.log(2.0) - 9.0 * np.log(3.0)) / (8.0 * np.pi**2 * tau**2)
    if alpha == 2:
        return 3.0 / (8.0 * np.pi**2 * tau**3)
    raise ValueError(f"alpha must be one of {ALPHAS}, got {alpha}")


def drift_coeff(tau: np.ndarray) -> np.ndarray:
    """Coefficient of `D**2` in `sigma2_y(tau)` for linear frequency drift.

        sigma2_y(tau) = D**2 * tau**2 / 2

    Exact, not asymptotic, and the same for the modified variance. This is
    the identity `tests/test_identities.py` already asserts at 1e-13, which
    is why drift enters the fit as `D**2` and keeps the model linear.
    """
    tau = np.asarray(tau, dtype=np.float64)
    return tau**2 / 2.0
