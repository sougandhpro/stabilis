"""Greenhall EDOF -- degrees of freedom from an assumed power law.

The other half of the story from `edof.py`. There, the degrees of freedom
are read out of the estimator's own coefficient autocovariance and no noise
type is ever named. Here they come from a *model*: the power law, the
differencing order, the averaging factor and the record length, and no
look at the data at all.

This is the routine behind Stable32's `Min Sigma` and `Max Sigma` columns,
so it is what makes an old report and a new one diffable line for line.
A parity feature, not the default -- `auto` never selects it, because
assuming a power law has to be an explicit, recorded act.

Everything below is a way of evaluating

    1/edf = (1/M) * [ 1 + (2/s_z(0)^2) * sum_{j=1}^{M-1} (1 - j/M)*s_z(j*delta)^2 ]

accurately and cheaply, where `s_z` is the autocovariance of the estimator's
own differenced process. Transcribed from Greenhall & Riley, 35th PTTI,
2003; the equation numbers in the docstrings are the paper's. See
theory/greenhall.md, which carries the equations in full.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "greenhall_edof",
    "greenhall_edof_simple",
    "greenhall_edof_pooled",
    "term_count",
    "GREENHALL_RUN_POLICIES",
    "GREENHALL_ALPHAS",
    "GREENHALL_ORDERS",
    "J_MAX",
]

# Truncation threshold for the full algorithm: past this many summands the
# closed-form integral replaces the sum. 100 is the paper's own suggestion
# (section 4.3), not a value tuned here.
J_MAX = 100

# Section 4.6.3: "m must be less than about 1e6 to avoid roundoff error".
_FLPM_ROUNDOFF_LIMIT = 1e6

GREENHALL_ALPHAS = (2, 1, 0, -1, -2, -3, -4)
GREENHALL_ORDERS = (1, 2, 3)

# What to do when the record arrived in more than one contiguous run.
#
# "pool"     -- run the algorithm on each run at its own N, then combine by
#               Satterthwaite. Assumes the runs are independent.
# "fallback" -- refuse, and report it, so the caller can use eta1, which
#               needs no such assumption. Strictly faithful to the paper.
GREENHALL_RUN_POLICIES = ("pool", "fallback")

# Generalized autocovariance of the power-law process w(t), eq. (7):
#
#   alpha       2       1        0        -1        -2       -3       -4
#   sw(t)     -|t|   t^2 ln|t|  |t|^3  -t^4 ln|t|  -|t|^5  t^6 ln|t|  |t|^7
#
# split into sign, power and whether a log factor is present. The log
# entries evaluate to 0 at t = 0.
_SW_SIGN = {2: -1.0, 1: 1.0, 0: 1.0, -1: -1.0, -2: -1.0, -3: 1.0, -4: 1.0}
_SW_POWER = {2: 1, 1: 2, 0: 3, -1: 4, -2: 5, -3: 6, -4: 7}
_SW_LOG = {2: False, 1: True, 0: False, -1: True, -2: False, -3: True, -4: False}

# Coefficients of (Delta_1 Delta_-1)^d at offsets 0, 1, .. d, as printed in
# eq. (9). They are (-1)^k * C(2d, d-k); test_greenhall.py asserts that
# rather than leaving the transcription unchecked.
_SZ_COEFFS = {
    1: (2.0, -1.0),
    2: (6.0, -4.0, 1.0),
    3: (20.0, -15.0, 6.0, -1.0),
}

# Table 1 -- large-r coefficients (a0, a1) for modified variances, keyed
# (alpha, d). Written as the paper prints them: exact fractions where it
# gives fractions, three decimals where it gives decimals. A missing key is
# a blank cell in the source, and every blank is a combination that
# violates alpha + 2d > 1.
_TABLE1 = {
    (2, 1): (2 / 3, 1 / 3), (2, 2): (7 / 9, 1 / 2), (2, 3): (22 / 25, 2 / 3),
    (1, 1): (0.840, 0.345), (1, 2): (0.997, 0.616), (1, 3): (1.141, 0.843),
    (0, 1): (1.079, 0.368), (0, 2): (1.033, 0.607), (0, 3): (1.184, 0.848),
    (-1, 2): (1.048, 0.534), (-1, 3): (1.180, 0.816),
    (-2, 2): (1.302, 0.535), (-2, 3): (1.175, 0.777),
    (-3, 3): (1.194, 0.703),
    (-4, 3): (1.489, 0.702),
}

# Table 2 -- the same for unmodified variances. The alpha = 2 row is also
# the Case 4 closed form, C(4d,2d)/C(2d,d)^2 and d/2, which is asserted in
# the tests: it checks a whole transcribed row against arithmetic.
_TABLE2 = {
    (2, 1): (3 / 2, 1 / 2), (2, 2): (35 / 18, 1.0), (2, 3): (231 / 100, 3 / 2),
    (1, 1): (78.6, 25.2), (1, 2): (790.0, 410.0), (1, 3): (9950.0, 6520.0),
    (0, 1): (2 / 3, 1 / 6), (0, 2): (2 / 3, 1 / 3), (0, 3): (7 / 9, 1 / 2),
    (-1, 2): (0.852, 0.375), (-1, 3): (0.997, 0.617),
    (-2, 2): (1.079, 0.368), (-2, 3): (1.033, 0.607),
    (-3, 3): (1.053, 0.553),
    (-4, 3): (1.302, 0.535),
}

# Table 3 -- (b0, b1) of the logarithmic denominator b0 + b1*ln(m) that
# stands in for s_z(0) in the unmodified flicker-PM case. Keyed by d. The
# source's caption says "FLFM" where the text says FLPM; the numbers are
# used as printed and the slip is recorded in theory/greenhall.md.
_TABLE3 = {1: (6.0, 4.0), 2: (15.23, 12.0), 3: (47.8, 40.0)}


def _sw(t, alpha: int) -> np.ndarray:
    """Generalized autocovariance of the power-law process w(t), eq. (7).

        sw(t, alpha) = sign * |t|**power           (alpha even-numbered rows)
                     = sign * |t|**power * ln|t|   (alpha = 1, -1, -3)

    The log rows must evaluate to 0 at t = 0, where the polynomial factor
    wins; `np.where` guards the log rather than the result, so no NaN is
    ever formed to be discarded afterwards.
    """
    a = np.abs(np.asarray(t, dtype=np.float64))
    val = a ** _SW_POWER[alpha]
    if _SW_LOG[alpha]:
        val = np.where(a > 0.0, val * np.log(np.where(a > 0.0, a, 1.0)), 0.0)
    return _SW_SIGN[alpha] * val


def _sx(t, f: float, alpha: int) -> np.ndarray:
    """Autocovariance of the (possibly averaged) phase, eq. (8).

        sx(t, F, alpha) = F**2 * [ 2*sw(t) - sw(t - 1/F) - sw(t + 1/F) ]
        sx(t, inf, alpha) = sw(t, alpha + 2),        -4 <= alpha <= 0

    The infinite-F limit is -sw''(t): it treats the phase as the
    mean-square derivative of w. It exists only for alpha <= 0, and it is
    what keeps the unmodified large-m case from differencing large sw
    values that cancel to something small -- the roundoff hazard the paper
    warns about in its appendix.
    """
    if math.isinf(f):
        if alpha > 0:
            raise ValueError(f"greenhall: the F=inf limit needs alpha <= 0, got {alpha}")
        return _sw(t, alpha + 2)
    step = 1.0 / f
    return f * f * (2.0 * _sw(t, alpha) - _sw(t - step, alpha) - _sw(t + step, alpha))


def _sz(t, f: float, alpha: int, d: int) -> np.ndarray:
    """Autocovariance of the d-th difference, eq. (9).

        sz(t, F, alpha, d) = (Delta_1 Delta_-1)**d sx(t, F, alpha)
                           = sum_k c[k] * (sx(t-k) + sx(t+k)),  k = 1..d
                             plus c[0]*sx(t)

    with c = (2,-1), (6,-4,1), (20,-15,6,-1) for d = 1, 2, 3.
    """
    t = np.asarray(t, dtype=np.float64)
    c = _SZ_COEFFS[d]
    out = c[0] * _sx(t, f, alpha)
    for k in range(1, d + 1):
        out = out + c[k] * (_sx(t - k, f, alpha) + _sx(t + k, f, alpha))
    return out


def _basic_sum(j_top: float, m_terms: float, s: float, f: float, alpha: int, d: int) -> float:
    """The truncated autocovariance sum, eq. (10).

        BasicSum(J, M, S, F, alpha, d)
            = sz(0)**2
            + (1 - J/M) * sz(J/S)**2
            + 2 * sum_{j=1}^{J-1} (1 - j/M) * sz(j/S)**2

    The isolated j = J term is deliberate: it turns the sum into a
    trapezoidal approximation of the integral that the large-record
    branches evaluate in closed form, so the two agree where they meet.
    """
    j = np.arange(1, int(j_top), dtype=np.float64)
    total = float(_sz(0.0, f, alpha, d)) ** 2
    total += (1.0 - j_top / m_terms) * float(_sz(j_top / s, f, alpha, d)) ** 2
    if j.size:
        total += 2.0 * float(np.sum((1.0 - j / m_terms) * _sz(j / s, f, alpha, d) ** 2))
    return total


def term_count(n: int, m: int, d: int, modified: bool, overlapped: bool) -> int:
    """Number of terms the estimator forms, eq. (11).

        L = m*(1/F + d)
        M = 1 + floor(S*(N - L)/m),      requires N >= L

    Evaluated in integer arithmetic because `L` is an integer either way --
    `m*(1+d)` when modified, `1 + d*m` when not -- and going through the
    literal `1/F` lets a floor land on the wrong side of an integer. The
    result agrees exactly with the coefficient-array lengths in
    classical.py, which is what test_greenhall.py checks.
    """
    if d not in GREENHALL_ORDERS:
        raise ValueError(f"greenhall: d must be one of {GREENHALL_ORDERS}, got {d}")
    if m < 1:
        raise ValueError(f"greenhall: m must be a positive integer, got {m}")
    length = m * (1 + d) if modified else 1 + d * m
    if n < length:
        raise ValueError(
            f"greenhall: N={n} is below L={length} for m={m}, d={d}, "
            f"modified={modified}; not enough data for a single term"
        )
    return 1 + (n - length if overlapped else (n - length) // m)


def _check(alpha: int, d: int) -> None:
    """The stationarity restriction of section 4.1: alpha + 2d > 1.

    A refusal rather than a clamp. Below it the d-th phase differences are
    not stationary, the variance being estimated does not exist, and any
    number returned would be a fiction. Every blank cell in Tables 1 and 2
    is exactly a combination that fails this.
    """
    if alpha not in GREENHALL_ALPHAS:
        raise ValueError(f"greenhall: alpha must be one of {GREENHALL_ALPHAS}, got {alpha}")
    if d not in GREENHALL_ORDERS:
        raise ValueError(f"greenhall: d must be one of {GREENHALL_ORDERS}, got {d}")
    if alpha + 2 * d <= 1:
        raise ValueError(
            f"greenhall: alpha={alpha} with d={d} violates alpha + 2d > 1; the "
            f"{d}-order phase differences are not stationary, so the variance "
            "does not exist. Use a higher-order estimator."
        )


def greenhall_edof_simple(
    alpha: int, d: int, m: int, n: int, modified: bool, overlapped: bool,
    j_factor: int | None = None,
) -> float:
    """The simplified algorithm, eq. (13): sum (6) truncated at J.

        J = min(M, j_factor*S)
        1/edf = BasicSum(J, M, S, F, alpha, d) / (sz(0, F, alpha, d)**2 * M)

    `j_factor` defaults to `d + 1`, past which sz has decayed to nothing.
    The paper suggests re-running at `6S` to see what the truncation cost;
    that is what the argument is for, and a test does exactly that.
    """
    _check(alpha, d)
    f = 1.0 if modified else float(m)
    s = float(m) if overlapped else 1.0
    m_terms = float(term_count(n, m, d, modified, overlapped))
    j_top = min(m_terms, (d + 1 if j_factor is None else j_factor) * s)
    inv = _basic_sum(j_top, m_terms, s, f, alpha, d) / (
        float(_sz(0.0, f, alpha, d)) ** 2 * m_terms
    )
    return 1.0 / inv


def _large_r(a0: float, a1: float, r: float) -> float:
    """The long-record branch shared by cases 1-3: 1/edf = (a0 - a1/r)/r.

    It is the integral the sum converges to, `a0 = 2*int_0^(d+1) sz^2 dt`
    and `a1 = 2*int_0^(d+1) sz^2 t dt`, evaluated in advance by the authors
    and printed in Tables 1 and 2.
    """
    return r / (a0 - a1 / r)


def _case4(d: int, m_terms: float, r: float) -> float:
    """Case 4 -- unmodified white PM. Exact, section 4.6.4.

        K = ceil(r)
        K <= d:  1/edf = (1/M)*[1 + 2*C(2d,d)**-2 * sum_{k=1}^{K-1} (1-k/r)*C(2d,d-k)**2]
        K >  d:  1/edf = (a0 - a1/r)/M,  a0 = C(4d,2d)/C(2d,d)**2, a1 = d/2

    Exact because the phase samples really are uncorrelated when w(t) is a
    Wiener process. Note the divisor is **M, not r** -- this is the one
    case where they differ, and swapping them gives a plausible-looking
    answer wrong by the stride factor.
    """
    centre = math.comb(2 * d, d)
    k_top = math.ceil(r)
    if k_top <= d:
        tail = sum(
            (1.0 - k / r) * math.comb(2 * d, d - k) ** 2 for k in range(1, k_top)
        )
        inv = (1.0 + 2.0 * tail / centre**2) / m_terms
    else:
        a0 = math.comb(4 * d, 2 * d) / centre**2
        inv = (a0 - 0.5 * d / r) / m_terms
    return 1.0 / inv


def greenhall_edof(
    alpha: int, d: int, m: int, n: int, modified: bool, overlapped: bool
) -> float:
    """Equivalent degrees of freedom by the full algorithm, section 4.6.

    Four cases, each with the same three-branch shape: sum directly when
    the sum is short, use the precomputed integral when the record is long,
    and otherwise re-sum with the stride rescaled so the term count lands
    on `J_MAX`. `alpha` is the *frequency* exponent, S_y(f) ~ f**alpha, the
    same convention as `noiseid.ALPHA_NAMES`.

    Which case applies is decided by the filter factor and the noise:
    modified variances (F = 1) are case 1 whatever the noise, and the
    unmodified ones split at alpha <= 0, alpha = 1 and alpha = 2 because
    the kernel degenerates differently in each.
    """
    _check(alpha, d)
    f = 1.0 if modified else float(m)
    s = float(m) if overlapped else 1.0
    m_terms = float(term_count(n, m, d, modified, overlapped))
    j_top = min(m_terms, (d + 1) * s)
    r = m_terms / s

    if f == 1.0:
        return _case1(j_top, m_terms, s, r, alpha, d)
    if alpha == 2:
        return _case4(d, m_terms, r)
    if alpha == 1:
        return _case3(j_top, m_terms, s, r, m, d)
    return _case2(j_top, m_terms, s, r, m, alpha, d)


def _case1(j_top: float, m_terms: float, s: float, r: float, alpha: int, d: int) -> float:
    """Case 1 -- modified variances, F = 1, any alpha. Section 4.6.1.

        J <= J_MAX          1/edf = BasicSum(J, M, S, F)/(sz(0,F)**2 * M)
        J > J_MAX, r >= d+1 1/edf = (a0 - a1/r)/r
        otherwise           1/edf = BasicSum(Jmax, Jmax, Jmax/r, F)
                                    / (sz(0,F)**2 * Jmax)

    Also covers unmodified variances at m = 1, where F = m = 1 and the two
    families coincide.
    """
    zero = float(_sz(0.0, 1.0, alpha, d)) ** 2
    if j_top <= J_MAX:
        return m_terms * zero / _basic_sum(j_top, m_terms, s, 1.0, alpha, d)
    if r >= d + 1:
        a0, a1 = _TABLE1[(alpha, d)]
        return _large_r(a0, a1, r)
    return J_MAX * zero / _basic_sum(J_MAX, J_MAX, J_MAX / r, 1.0, alpha, d)


def _case2(j_top: float, m_terms: float, s: float, r: float, m: int, alpha: int, d: int) -> float:
    """Case 2 -- unmodified variances, alpha <= 0. Section 4.6.2.

    Identical in shape to case 1, except that the kernel switches to its
    F -> infinity limit once `m*(d+1)` would exceed `J_MAX`, and always in
    the third branch. That switch is the whole point of the case: at large
    m the finite-F difference in eq. (8) is a cancellation of large numbers.
    """
    if j_top <= J_MAX:
        f = float(m) if m * (d + 1) <= J_MAX else math.inf
        inv = _basic_sum(j_top, m_terms, s, f, alpha, d) / (
            float(_sz(0.0, f, alpha, d)) ** 2 * m_terms
        )
        return 1.0 / inv
    if r >= d + 1:
        a0, a1 = _TABLE2[(alpha, d)]
        return _large_r(a0, a1, r)
    inv = _basic_sum(J_MAX, J_MAX, J_MAX / r, math.inf, alpha, d) / (
        float(_sz(0.0, math.inf, alpha, d)) ** 2 * J_MAX
    )
    return 1.0 / inv


def _case3(j_top: float, m_terms: float, s: float, r: float, m: int, d: int) -> float:
    """Case 3 -- unmodified flicker PM, alpha = 1. Section 4.6.3.

    As S = m grows the kernel develops logarithmic singularities, so in the
    long-record branches `sz(0)` is replaced by its asymptotic form
    `b0 + b1*ln(m)` from Table 3:

        J > J_MAX, r >= d+1  1/edf = (a0 - a1/r)/((b0 + b1 ln m)**2 * r)
        otherwise            1/edf = BasicSum(Jmax, Jmax, Jmax/r, Jmax/r)
                                     / ((b0 + b1 ln m)**2 * Jmax)

    The paper declines to add a further large-m subcase, on the grounds
    that flicker PM is not the dominant noise when m is large. That
    judgement is inherited; the roundoff limit it implies is checked below
    rather than left silent.
    """
    if j_top <= J_MAX:
        if m > _FLPM_ROUNDOFF_LIMIT:
            raise ValueError(
                f"greenhall: unmodified flicker PM at m={m} exceeds the "
                f"published roundoff limit of {_FLPM_ROUNDOFF_LIMIT:.0e}; the "
                "sz(0) cancellation is not reliable in double precision there"
            )
        inv = _basic_sum(j_top, m_terms, s, float(m), 1, d) / (
            float(_sz(0.0, float(m), 1, d)) ** 2 * m_terms
        )
        return 1.0 / inv
    b0, b1 = _TABLE3[d]
    log_term = (b0 + b1 * math.log(m)) ** 2
    if r >= d + 1:
        a0, a1 = _TABLE2[(1, d)]
        return log_term * r / (a0 - a1 / r)
    m_prime = J_MAX / r
    inv = _basic_sum(J_MAX, J_MAX, m_prime, m_prime, 1, d) / (log_term * J_MAX)
    return 1.0 / inv


def greenhall_edof_pooled(
    alpha: int, d: int, m: int, run_lengths, run_terms,
    modified: bool, overlapped: bool, policy: str = "pool",
) -> tuple[float, str]:
    """EDOF for a record that arrived in several contiguous runs.

        eta = (sum_j M_j)**2 / sum_j (M_j**2 / eta_j)

    Satterthwaite's combination for a weighted mean of independent
    chi-square terms, with the same `M_j` weights `run_estimator` used to
    pool the coefficients themselves. Exact for one run.

    `run_terms` is the count of terms that *survived* in each run, which is
    below eq. (11)'s nominal count when a gap-masking policy removed some.
    Each run's edf is scaled by that surviving fraction, since 1/edf in
    eq. (6) carries a factor 1/M: dropping terms uniformly costs degrees of
    freedom in proportion.

    Returns `(eta, method_used)`. Under `policy="fallback"` a segmented
    record yields `(nan, "greenhall-unavailable")` rather than an
    exception, so the caller can record the refusal and use eta1 instead.
    """
    if policy not in GREENHALL_RUN_POLICIES:
        raise ValueError(
            f"greenhall: policy must be one of {GREENHALL_RUN_POLICIES}, got {policy!r}"
        )
    lengths = [int(v) for v in run_lengths]
    survived = [int(v) for v in run_terms]
    if len(lengths) != len(survived):
        raise ValueError(
            f"greenhall: {len(lengths)} run lengths but {len(survived)} term counts"
        )
    if len(lengths) > 1 and policy == "fallback":
        return float("nan"), "greenhall-unavailable"

    weights, etas = [], []
    for n_run, kept in zip(lengths, survived):
        if kept <= 0:
            continue
        nominal = term_count(n_run, m, d, modified, overlapped)
        eta = greenhall_edof(alpha, d, m, n_run, modified, overlapped)
        weights.append(float(kept))
        etas.append(eta * min(kept / nominal, 1.0))
    if not weights:
        raise ValueError("greenhall: no run contributed a term")

    total = math.fsum(weights)
    spread = math.fsum(w * w / e for w, e in zip(weights, etas))
    return total * total / spread, "greenhall"
