"""Bias in the reported deviation, and how to remove it.

**Scope note.** The plan lists B1/B2/B3 and the TOTDEV and Theo1
corrections under this module. Those are deferred to M3 on purpose: every
one of them exists to correct an estimator this library does not yet have,
and B1 is a noise-identification aid that would sit beside two methods that
already agree to within a tenth of a power law. What is here is the
correction that changes numbers being reported *today*.

The variance estimators are unbiased. The deviation is not, and the reason
is elementary -- the square root is concave, so by Jensen's inequality
E[sqrt(V)] < sqrt(E[V]). For a chi-square sampling law with eta degrees of
freedom the factor is exact:

    E[sigma_hat] / sigma = sqrt(2/eta) * Gamma((eta+1)/2) / Gamma(eta/2)

At eta = 2 that is 0.8862: **the reported deviation is 11.4% low on
average**. At eta = 100 it is 0.9975. So the bias is negligible at short
tau and severe at long tau -- which is precisely where an oscillator
specification is written and where a customer reads the number off the
plot.

Two consequences run through the rest of the library:

* Anything fitted here is fitted to **variances**, never to deviations.
  Fitting a model to biased points biases the parameters, and the bias
  varies point to point because eta does.
* The correction is offered, not applied silently. An uncorrected sigma is
  what every other tool reports, so changing it by default would make this
  library disagree with Stable32 for a reason the user never asked about.

See theory/bias.md.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "THEO1_BIAS_MEASURED",
    "THEO1_BIAS_PARAMS",
    "theo1_bias",
    "totvar_bias",
    "TOTVAR_BIAS_A","deviation_bias_factor", "unbias_deviation"]


def deviation_bias_factor(eta):
    """E[sigma_hat]/sigma for a chi-square sampling law with `eta` dof.

        c4(eta) = sqrt(2/eta) * Gamma((eta+1)/2) / Gamma(eta/2)

    Known in quality control as the c4 constant. Evaluated through
    `lgamma` and a difference of logarithms rather than as a ratio of
    gamma functions, which would overflow above eta ~ 340 -- well inside
    the range of a long record.

    Always in (0, 1], rising to 1 as eta grows. The asymptotic form is
    1 - 1/(4*eta), which is a useful sanity check: at eta = 100 that
    predicts 0.9975 and the exact value is 0.99750.
    """
    eta = np.asarray(eta, dtype=np.float64)
    if np.any(eta <= 0.0):
        raise ValueError("deviation_bias_factor: eta must be positive")
    flat = np.ravel(eta)
    out = np.array(
        [
            math.exp(
                0.5 * math.log(2.0 / e)
                + math.lgamma(0.5 * (e + 1.0))
                - math.lgamma(0.5 * e)
            )
            for e in flat
        ],
        dtype=np.float64,
    )
    return float(out[0]) if eta.shape == () else out.reshape(eta.shape)


def unbias_deviation(dev, eta):
    """Remove the concavity bias from a deviation.

        sigma_corrected = sigma_hat / c4(eta)

    This *raises* the value, by 11.4% at eta = 2 and by 0.25% at eta = 100.

    Opt-in by design. Every other tool in the field reports the uncorrected
    deviation, so applying this silently would make `stabilis` disagree
    with Stable32 by an amount that looks like a bug. When it is applied it
    must be stated -- IEEE Std 1139 clause 4.2 requires the pretreatment of
    the data to be reported, and this is pretreatment of the result.
    """
    return np.asarray(dev, dtype=np.float64) / deviation_bias_factor(eta)


# TOTVAR is biased low for the steeper FM noises. NIST SP 1065 section
# 5.2.11: "Bias corrections of the form 1/[1-a(tau/T)], where T is the
# record length, need to be applied for flicker and random walk FM noise,
# where a = 0.481 and 0.750, respectively." Its expected value equals
# AVAR's for white and flicker PM and for white FM, so those are zero --
# recorded explicitly rather than left to a missing key.
TOTVAR_BIAS_A = {2: 0.0, 1: 0.0, 0: 0.0, -1: 0.481, -2: 0.750}


def totvar_bias(alpha: int, tau_s, record_s: float) -> np.ndarray:
    """Multiplier turning TOTVAR into an unbiased estimate of AVAR.

        corrected = totvar / (1 - a * tau / T)

    `a` depends on the noise type, which is why TOTVAR is coupled to noise
    identification in a way the classical estimators are not. Returned as a
    factor rather than applied, so a caller can see and record what was
    done to their number.

    The correction grows without bound as `tau` approaches `T/a`, so it is
    refused past the point where it would exceed a factor of two: beyond
    there the "correction" is larger than the thing being corrected, and a
    number produced that way is not a measurement.
    """
    if alpha not in TOTVAR_BIAS_A:
        raise ValueError(
            f"totvar_bias: no published coefficient for alpha={alpha}; "
            f"SP 1065 gives {sorted(TOTVAR_BIAS_A)}")
    a = TOTVAR_BIAS_A[alpha]
    tau = np.asarray(tau_s, dtype=np.float64)
    if record_s <= 0.0:
        raise ValueError(f"record length must be positive, got {record_s}")
    denominator = 1.0 - a * tau / record_s
    if np.any(denominator < 0.5):
        worst = float(np.asarray(tau).ravel()[np.argmin(denominator)])
        raise ValueError(
            f"totvar_bias: at tau={worst:g} s of a {record_s:g} s record the "
            f"correction for alpha={alpha} exceeds a factor of two. Past that "
            "the correction is bigger than the quantity being corrected; drop "
            "the long-tau points instead."
        )
    return 1.0 / denominator


# SP 1065 Table 2, eq. (31): Thêo1's bias as AVAR/Thêo1 = a + b/m^c, per
# noise type. Published, unlike THEO1_BIAS_MEASURED below -- which is why
# both are here. White FM is the (1, 0, 0) row, the identity that
# identified the missing 0.75 in eq. (30).
THEO1_BIAS_PARAMS = {
    2: (0.09, 0.74, 0.40),
    1: (0.14, 0.82, 0.30),
    0: (1.00, 0.00, 0.00),
    -1: (1.87, -1.05, 0.79),
    -2: (2.70, -1.53, 0.85),
}


def theo1_bias(alpha: int, m):
    """`AVAR/Thêo1` for a known noise type, SP 1065 eq. (31).

        bias = a + b / m^c

    Multiply a Thêo1 variance by this to get an estimate of AVAR. It needs
    the noise type, which is the whole reason ThêoBR exists -- `theobr`
    measures the same ratio off the data instead, and needs no `alpha`.

    The asymptotes are a useful check on the measured table below: `a` is
    the large-m limit, 1.87 for flicker FM and 2.70 for random-walk FM,
    against measured Thêo1/AVAR of 0.55 and 0.37 -- 1/1.82 and 1/2.70.
    """
    if alpha not in THEO1_BIAS_PARAMS:
        raise ValueError(
            f"theo1_bias: SP 1065 Table 2 covers alpha in "
            f"{sorted(THEO1_BIAS_PARAMS)}, not {alpha}")
    a, b, c = THEO1_BIAS_PARAMS[alpha]
    m = np.asarray(m, dtype=np.float64)
    if np.any(m < 1):
        raise ValueError("theo1_bias: averaging factor must be at least 1")
    return a + b / m**c


# Thêo1's bias against AVAR, at the same tau, for m >~ 100.
#
# **Measured here, not published.** 60 realizations at N=8192 of each power
# law, Thêo1 against an interpolated overlapping AVAR at Thêo1's own tau.
# White FM comes out at 1.00, which is the property that identified the
# missing 0.75 in SP 1065 eq. (30) -- see stats/theo1.py. The FM biases are
# real and are what Howe's TheoBR exists to remove by averaging Thêo1
# against AVAR over the range where both are defined.
#
# TheoBR is not implemented, so these are here to be *reported*, not
# silently applied: a number divided by a ratio measured on synthetic noise
# is not the same thing as a number corrected by a published estimator.
THEO1_BIAS_MEASURED = {0: 1.00, -1: 0.55, -2: 0.37}
