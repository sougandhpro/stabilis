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

__all__ = ["deviation_bias_factor", "unbias_deviation"]


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
