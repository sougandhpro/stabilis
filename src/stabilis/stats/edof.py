"""Equivalent degrees of freedom, estimated from the data itself.

This is the differentiator. Stable32 looks the noise type up in a table and
applies a closed form fitted to that power law; when the noise is not one of
its five types it has no answer at all. Real oscillators mix noise types,
sit on a drift, and pass through a measurement system that adds its own --
so "which of the five is it" is often the wrong question.

The estimators here ask a different one. The EDOF is *defined* by the
sampling variance of the estimate,

    Var(sigma_hat^2) = 2 * sigma^4 / eta

and for a mean of M squared coefficients drawn from a stationary Gaussian
process that is

    Var(sigma_hat^2) = (2 * sigma^4 / M) * sum_k (1 - |k|/M) * rho(k)^2

where rho(k) is the autocorrelation of the *coefficients*. Equating the two,

    eta = M / (1 + 2 * sum_{k>=1} (1 - k/M) * rho(k)^2)

Everything needed is in the coefficient array every estimator already
returns, and nothing needed is a noise type. See theory/edof.md.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "totvar_edof",
    "TOTVAR_EDOF_BC",
    "eta1",
    "eta3",
    "edof",
    "EDOF_METHODS",
    "DATA_DRIVEN_METHODS",
    "MIN_TERMS_FOR_ETA1",
]

# "greenhall" is dispatched by stats.ci, not here: it is a function of the
# noise type and the record geometry rather than of the coefficients, so it
# shares none of this module's inputs. It is listed here because this tuple
# is the single source of truth for what a caller -- and the CLI -- may ask
# for. `auto` deliberately never resolves to it; see the module docstring.
EDOF_METHODS = ("auto", "eta1", "eta3", "greenhall")
DATA_DRIVEN_METHODS = ("auto", "eta1", "eta3")

# Below this many terms the sample autocorrelation is mostly noise and eta1
# stops being informative -- it can even exceed M. Fall back to eta3, which
# is conservative by construction and cannot mislead in the optimistic
# direction. The threshold is deliberately generous; long tau is exactly
# where an over-narrow error bar does the most damage.
MIN_TERMS_FOR_ETA1 = 8

# Lag truncation, as a multiple of the averaging factor. The coefficients
# of an overlapping estimator are correlated out to roughly the window
# width -- 2m or 3m samples -- so lags past ~4m carry signal only by
# accident. See the docstring of eta1 for why not truncating is worse than
# it sounds.
DEFAULT_LAG_LIMIT = 4


def _autocorrelation(c: np.ndarray, kmax: int) -> np.ndarray:
    """Biased sample autocorrelation of `c` at lags 0..kmax, via FFT.

    Zero-padded past 2M so the circular correlation cannot wrap. O(M log M)
    rather than the O(kmax*M) of the direct form, which matters: at a large
    averaging factor kmax and M are both O(N), so the direct sum is
    quadratic in the record length.

    Divides by M rather than by (M-k) -- the biased normalization. It shrinks
    the high-lag estimates towards zero, which is the desired behaviour here
    (those are the noisiest), and it keeps the sequence positive
    semi-definite, so the sum below cannot produce a negative eta.

    The coefficients are *not* demeaned. A second or third difference of
    phase has zero mean by construction for any process without a
    deterministic polynomial term, so subtracting a sample mean would remove
    signal rather than a nuisance parameter.
    """
    m_terms = c.size
    n = 1 << (2 * m_terms - 1).bit_length()
    f = np.fft.rfft(c, n)
    acov = np.fft.irfft(f * np.conjugate(f), n)[: kmax + 1] / m_terms
    if acov[0] <= 0.0:
        return np.zeros_like(acov)
    return acov / acov[0]


def eta1(coeffs: np.ndarray, m: int, lag_limit: int = DEFAULT_LAG_LIMIT) -> float:
    """Data-driven EDOF from the coefficient autocorrelation.

        eta1 = M / (1 + 2 * sum_{k=1}^{kmax} (1 - k/M) * rho_hat(k)^2)
        kmax = min(M - 1, lag_limit * m)

    The weight `1 - k/M` is not a tuning choice -- it falls out of the
    variance of a mean over M correlated terms, and is the Bartlett factor
    in the expression quoted at the top of this module. The **truncation**
    is the choice, and it is the one that matters.

    **Why truncate.** Summing rho_hat(k)^2 over all M-1 lags adds a
    spurious contribution even when the true autocorrelation is zero: each
    rho_hat(k) has variance ~1/M, so its square has expectation ~1/M, and
    M-1 of them sum to ~1. That fake 1 sits beside the real 1 in the
    denominator and roughly halves the estimate. Overlapping coefficients
    are correlated only out to about the window width, 2m or 3m samples, so
    lags past ~4m carry noise and nothing else. Discarding them removes the
    bias and loses no signal.

    **Why this weight and not a taper down to kmax.** The obvious-looking
    `1 - k/(kmax+1)`, which sends the last retained lag to zero, was tried
    first. It shrinks real correlation as well as noise, and measured
    against Monte-Carlo truth it ran 6-25% *high* across every power law --
    error bars too narrow, which is the direction that matters. With the
    theoretical `1 - k/M`, the same measurement centres near 1.0. There is
    no positive-definiteness argument for the taper here either, since the
    sum is over *squares* and cannot go negative.

    Measured against `2*sigma^4/Var(sigma_hat^2)` over 500 realizations of
    N=512 at each of alpha = 2, 1, 0, -1, -2: the ratio stays within about
    +/-10% for m up to N/32 and degrades to ~30% at the last one or two
    points, where the estimate has only a handful of degrees of freedom and
    the Monte-Carlo truth is itself uncertain. See tests/test_edof.py.
    """
    c = np.asarray(coeffs, dtype=np.float64)
    m_terms = c.size
    if m_terms == 0:
        raise ValueError("eta1: no coefficients")
    if m_terms < MIN_TERMS_FOR_ETA1:
        raise ValueError(
            f"eta1: {m_terms} terms is below MIN_TERMS_FOR_ETA1="
            f"{MIN_TERMS_FOR_ETA1}; use eta3"
        )
    kmax = min(m_terms - 1, lag_limit * int(m))
    rho = _autocorrelation(c, kmax)
    k = np.arange(1, kmax + 1, dtype=np.float64)
    weights = 1.0 - k / m_terms
    total = float(np.sum(weights * rho[1:] ** 2))
    return float(np.clip(m_terms / (1.0 + 2.0 * total), 1.0, m_terms))


def eta3(n_terms: int, m: int, span_factor: int = 2) -> float:
    """Conservative EDOF: one independent term per estimator window.

        eta3 = max(M / (span_factor * m), 1)

    `span_factor * m` is the width of one term in phase samples -- 2m for
    the Allan family, 3m for Hadamard and modified. Terms further apart than
    that share no data, so this counts the non-overlapping terms and ignores
    the information overlapping ones do carry.

    It is therefore a lower bound, not an estimate: for white FM it is about
    three times too small. That is the point. It is the fallback when there
    are too few terms for the autocorrelation to mean anything, and a
    fallback must err towards error bars that are too wide.
    """
    return max(float(n_terms) / (span_factor * int(m)), 1.0)


def edof(
    coeffs: np.ndarray,
    m: int,
    span_factor: int = 2,
    method: str = "auto",
    lag_limit: int = DEFAULT_LAG_LIMIT,
) -> tuple[float, str]:
    """EDOF and the method actually used, which are not always the requested one.

    Returns `(eta, method_used)`. The second element exists because
    "silently fell back" is exactly the kind of decision an ISO-17025 report
    has to be able to answer for, so the caller records it rather than
    discovering it later.
    """
    if method == "greenhall":
        raise ValueError(
            "edof: 'greenhall' is not a coefficient-based estimator; it needs "
            "the noise type and the record geometry. Call "
            "stats.greenhall.greenhall_edof, or stats.ci.confidence with "
            "method='greenhall'"
        )
    if method not in DATA_DRIVEN_METHODS:
        raise ValueError(
            f"edof: method must be one of {DATA_DRIVEN_METHODS}, got {method!r}"
        )
    n_terms = int(np.asarray(coeffs).size)
    if method == "eta3":
        return eta3(n_terms, m, span_factor), "eta3"
    if method == "eta1":
        return eta1(coeffs, m, lag_limit), "eta1"
    if n_terms < MIN_TERMS_FOR_ETA1:
        return eta3(n_terms, m, span_factor), "eta3"
    return eta1(coeffs, m, lag_limit), "eta1"


# TOTVAR's own degrees of freedom. NIST SP 1065 section 5.2.11: "The number
# of equivalent chi^2 degrees of freedom for TOTVAR can be estimated for
# white FM, flicker FM and random walk FM noise by the expression
# b(T/tau)-c, where b = 1.500, 1.168 and 0.927, and c = 0, 0.222 and 0.358,
# respectively."
TOTVAR_EDOF_BC = {0: (1.500, 0.0), -1: (1.168, 0.222), -2: (0.927, 0.358)}


def totvar_edof(alpha: int, tau_s: float, record_s: float,
                overlapping_edof: float | None = None) -> float:
    """Equivalent chi-square degrees of freedom for a TOTVAR estimate.

        edf = b*(T/tau) - c

    **This is not optional dressing.** TOTVAR keeps `N-2` terms at every
    tau by reusing reflected data, so the coefficient array stays long even
    where the real information has run out. A data-driven estimator handed
    those coefficients sees many terms and reports a confidence far better
    than the record can support. The published formula counts what is
    actually there.

    For white and flicker PM, SP 1065 says the edf equals the overlapping
    ADEV's "with the number of chi-square degrees of freedom increased by
    2" -- so that case needs the Allan edf passed in rather than derived
    here.
    """
    if alpha in (2, 1):
        if overlapping_edof is None:
            raise ValueError(
                f"totvar_edof: alpha={alpha} is defined against the "
                "overlapping ADEV's edf plus two, so pass overlapping_edof=")
        return float(overlapping_edof) + 2.0
    if alpha not in TOTVAR_EDOF_BC:
        raise ValueError(
            f"totvar_edof: no published coefficients for alpha={alpha}; "
            f"SP 1065 gives {sorted(TOTVAR_EDOF_BC)} plus 2 and 1")
    if tau_s <= 0.0 or record_s <= 0.0:
        raise ValueError("tau and record length must be positive")
    b, c = TOTVAR_EDOF_BC[alpha]
    return max(1.0, b * (record_s / tau_s) - c)
