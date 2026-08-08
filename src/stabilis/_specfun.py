"""Special functions for confidence intervals, without scipy.

scipy has every function here and is faster at all of them. It is an
optional dependency by design -- a calibration lab embedding this engine in
a test rig should not have to ship a Fortran toolchain -- so these have to
exist and have to be right.

**Deliberately scalar, with a thin vectorizing wrapper.** Every call site
evaluates these over a tau grid: tens of points, not millions. A Python
loop costs microseconds there, while the masked-iteration numpy form of a
regularized incomplete gamma is sixty lines that nobody can check by eye.
The style contract says the equation and the code must be diffable, and
that constraint binds harder here than speed does.

Accuracy is asserted against closed forms rather than against a table:
`chi2_ppf(p, 2) == -2*ln(1-p)` exactly, `chi2_cdf(x, 1) == erf(sqrt(x/2))`,
and the finite-sum CDF for even degrees of freedom. The normal quantile is
checked against a bisection on `math.erfc`, which is slow, obviously
correct, and entirely independent of the rational approximation it checks.
See tests/test_specfun.py.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "gammainc_p",
    "gammainc_q",
    "norm_ppf",
    "chi2_cdf",
    "chi2_ppf",
]

_EPS = 3.0e-16
_FPMIN = 1.0e-300

# The series needs ~sqrt(72a) terms at x = a, so 1e4 covers a up to ~1e6 --
# far beyond any plausible EDOF. Non-convergence raises rather than
# returning a partial sum: a silently truncated series is a wrong number
# that looks right, which is the failure mode this library exists to avoid.
_ITMAX = 10_000

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# Stirling remainder series r(a) ~ sum B_2n / (2n(2n-1) a^(2n-1)), as
# coefficients of 1/a^(2n-1): 1/12, -1/360, 1/1260, -1/1680, 1/1188, ...
_STIRLING = (
    1.0 / 12.0, -1.0 / 360.0, 1.0 / 1260.0, -1.0 / 1680.0,
    1.0 / 1188.0, -691.0 / 360360.0, 1.0 / 156.0,
)


def _stirling_remainder(a: float) -> float:
    """r(a) = lnΓ(a) - [(a-1/2)·ln a - a + ln(2π)/2].

    Below a = 20 the two terms are O(50) or smaller, so taking the
    difference directly costs ~1e-15 absolute and is simplest. Above it the
    difference would cancel catastrophically, but the asymptotic series is
    then accurate to ~1e-18, so the regimes swap cleanly.
    """
    if a < 20.0:
        return math.lgamma(a) - ((a - 0.5) * math.log(a) - a + _LOG_SQRT_2PI)
    u = 1.0 / (a * a)
    total = 0.0
    for c in reversed(_STIRLING):
        total = total * u + c
    return total / a


def _log_prefactor(a: float, x: float) -> float:
    """ln( x^a · e^-x / Γ(a) ), regrouped so nothing large cancels.

        a·ln x - x - lnΓ(a)
          = ½·ln a - ½·ln(2π) - r(a) + [ a·ln(x/a) - (x - a) ]

    Written directly, the three terms are each ~a·ln a: at a = 5000 that is
    4e4, where float64 spacing is 7e-12, so the exponent carries an absolute
    error that becomes a relative error in the result. Measured against an
    exact Decimal evaluation of the integer-order case, the direct form gives
    11.0 correct digits at a = 5000; this one gives ~15.

    The bracketed term is the only one that is not O(ln a), and it is
    written as `a·ln(x/a) - (x-a)` rather than the tidier-looking
    `a·(log1p(t) - t)` with `t = (x-a)/a`. Those are algebraically the same
    and numerically are not: for x << a, t approaches -1 and `1 + t` loses
    everything to cancellation, since t carries absolute error 2e-16 while
    1 + t is x/a. At a = 1/2, x = 1e-6 that alone cost five digits. Forming
    the ratio directly never loses it.

    Near the peak the bracket does cancel -- both terms approach zero and
    their difference is ~-(x-a)²/(2a) -- but harmlessly, because what
    cancels is large only in proportion to how small the answer is.
    """
    return (
        0.5 * math.log(a)
        - _LOG_SQRT_2PI
        - _stirling_remainder(a)
        + a * math.log(x / a)
        - (x - a)
    )


def _apply(fn, *args):
    """Evaluate a scalar function elementwise over broadcast arguments.

    Returns a float for scalar input and an array otherwise, so callers can
    write `chi2_ppf(0.16, 12.0)` and get a number back.
    """
    arrays = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in args))
    shape = arrays[0].shape
    flat = [np.ravel(a) for a in arrays]
    out = np.array([fn(*vals) for vals in zip(*flat)], dtype=np.float64)
    return float(out[0]) if shape == () else out.reshape(shape)


# --------------------------------------------------------------------------
# regularized incomplete gamma
# --------------------------------------------------------------------------


def _series(a: float, x: float) -> float:
    """Lower regularized incomplete gamma P(a, x) by its series.

        P(a, x) = x^a e^-x / Γ(a) · Σ_{n≥0} x^n / (a(a+1)···(a+n))

    Converges quickly for x < a+1 and slowly beyond it, which is why the
    caller switches to the continued fraction there. Getting that switch
    wrong is the classic failure of this pair, so the tests sweep it.
    """
    total = 1.0 / a
    delta = total
    ap = a
    for _ in range(_ITMAX):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            return total * math.exp(_log_prefactor(a, x))
    raise RuntimeError(f"gammainc series failed to converge at a={a}, x={x}")


def _continued_fraction(a: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(a, x) by continued fraction.

        Q(a, x) = x^a e^-x / Γ(a) · 1/(x+1-a- 1·(1-a)/(x+3-a- 2·(2-a)/(...)))

    Evaluated by the modified Lentz algorithm, which carries the numerator
    and denominator recurrences separately so neither can overflow.
    """
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b if b != 0.0 else 1.0 / _FPMIN
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return math.exp(_log_prefactor(a, x)) * h
    raise RuntimeError(f"gammainc continued fraction failed at a={a}, x={x}")


def _check(a: float, x: float) -> None:
    if a <= 0.0:
        raise ValueError(f"gammainc: shape a must be positive, got {a}")
    if x < 0.0:
        raise ValueError(f"gammainc: x must be non-negative, got {x}")


def _p_scalar(a: float, x: float) -> float:
    _check(a, x)
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _series(a, x)
    return 1.0 - _continued_fraction(a, x)


def _q_scalar(a: float, x: float) -> float:
    """Q computed on its own branch, never as `1 - P`.

    Deriving it by subtraction throws the answer away in the far tail: at
    a = 1000, x = 2000 the true Q is ~1e-16 while P rounds to 1.0, so
    `1 - P` returns exactly zero. Evaluating the continued fraction
    directly keeps full relative accuracy there, and the tail is precisely
    where a high-confidence interval lives.
    """
    _check(a, x)
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _series(a, x)
    return _continued_fraction(a, x)


def gammainc_p(a, x):
    """Lower regularized incomplete gamma P(a, x) = γ(a, x) / Γ(a)."""
    return _apply(_p_scalar, a, x)


def gammainc_q(a, x):
    """Upper regularized incomplete gamma Q(a, x) = Γ(a, x) / Γ(a)."""
    return _apply(_q_scalar, a, x)


# --------------------------------------------------------------------------
# normal quantile -- Wichura AS241 (PPND16)
# --------------------------------------------------------------------------

# Acklam's approximation is shorter and reaches ~1e-9. That is not enough
# for a library whose selling point is metrology-grade numbers, so this is
# AS241, accurate to ~1e-16 across the whole range.

_A = (
    3.3871328727963666080e0, 1.3314166789178437745e2, 1.9715909503065514427e3,
    1.3731693765509461125e4, 4.5921953931549871457e4, 6.7265770927008700853e4,
    3.3430575583588128105e4, 2.5090809287301226727e3,
)
_B = (
    1.0, 4.2313330701600911252e1, 6.8718700749205790830e2,
    5.3941960214247511077e3, 2.1213794301586595867e4, 3.9307895800092710610e4,
    2.8729085735721942674e4, 5.2264952788528545610e3,
)
_C = (
    1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
    3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
    2.27238449892691845833e-2, 7.74545014278341407640e-4,
)
_D = (
    1.0, 2.05319162663775882187e0, 1.67638483018380384940e0,
    6.89767334985100004550e-1, 1.48103976427480074590e-1,
    1.51986665636164571966e-2, 5.47593808499534494600e-4,
    1.05075007164441684324e-9,
)
_E = (
    6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
    2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
    2.71155556874348757815e-5, 2.01033439929228813265e-7,
)
_F = (
    1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1,
    1.48753612908506148525e-2, 7.86869131145613259100e-4,
    1.84631831751005468180e-5, 1.42151175831644588870e-7,
    2.04426310338993978564e-15,
)


def _poly(coeffs, r: float) -> float:
    """Horner, ascending order -- the order the AS241 tables are printed in."""
    out = 0.0
    for c in reversed(coeffs):
        out = out * r + c
    return out


def _norm_ppf_scalar(p: float) -> float:
    """Inverse standard normal CDF, three rational branches.

    Central for |p - 1/2| <= 0.425, then two tail branches split at
    sqrt(-ln(min(p, 1-p))) = 5, which is p ~ 1.4e-12.
    """
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"norm_ppf: p must be in [0, 1], got {p}")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * _poly(_A, r) / _poly(_B, r)
    r = p if q < 0.0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        value = _poly(_C, r - 1.6) / _poly(_D, r - 1.6)
    else:
        value = _poly(_E, r - 5.0) / _poly(_F, r - 5.0)
    return -value if q < 0.0 else value


def norm_ppf(p):
    """Quantile of the standard normal distribution."""
    return _apply(_norm_ppf_scalar, p)


# --------------------------------------------------------------------------
# chi-square
# --------------------------------------------------------------------------


def chi2_cdf(x, nu):
    """CDF of chi-square with `nu` degrees of freedom: P(nu/2, x/2)."""
    return _apply(lambda xx, nn: _p_scalar(0.5 * nn, 0.5 * xx), x, nu)


def _chi2_ppf_scalar(p: float, nu: float) -> float:
    """Inverse chi-square CDF: Wilson-Hilferty start, then Newton.

    Bisection alone is wrong here. `nu` reaches 1e4 on a long record, where
    the distribution is narrow and far from the origin, so bracketing costs
    dozens of iterations for every value. Wilson-Hilferty,

        x0 = nu · (1 - 2/(9nu) + z·sqrt(2/(9nu)))^3,   z = Φ⁻¹(p)

    lands within a percent or so and Newton then converges in three or four
    steps, using the exact density as the derivative:

        f(x) = x^(nu/2-1) e^(-x/2) / (2^(nu/2) Γ(nu/2))

    A maintained bracket catches the cases Newton cannot handle on its own
    -- small `nu` with `p` near zero, where the density underflows and the
    step would be meaningless.
    """
    if nu <= 0.0:
        raise ValueError(f"chi2_ppf: nu must be positive, got {nu}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"chi2_ppf: p must be in [0, 1], got {p}")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return math.inf

    a = 0.5 * nu
    h = 2.0 / (9.0 * nu)
    x = nu * (1.0 - h + _norm_ppf_scalar(p) * math.sqrt(h)) ** 3
    if not math.isfinite(x) or x <= 0.0:
        x = 0.5 * nu  # Wilson-Hilferty can go negative at tiny nu and p.

    lo, hi = 0.0, math.inf
    for _ in range(200):
        err = _p_scalar(a, 0.5 * x) - p
        if err > 0.0:
            hi = min(hi, x)
        else:
            lo = max(lo, x)
        if abs(err) <= 1e-15 * p:
            break
        # pdf(x) = x^(a-1) e^(-x/2) / (2^a Γ(a)) = P'(a, x/2) / 2, and the
        # prefactor above already computes that logarithm stably.
        log_pdf = _log_prefactor(a, 0.5 * x) - math.log(x)
        nxt = x - err / math.exp(log_pdf) if log_pdf > -700.0 else math.inf
        if not (lo < nxt < hi):  # Newton escaped, or the density underflowed.
            nxt = 0.5 * (lo + hi) if math.isfinite(hi) else 2.0 * x
        if nxt == x:
            break
        x = nxt
    return x


def chi2_ppf(p, nu):
    """Quantile of the chi-square distribution with `nu` degrees of freedom."""
    return _apply(_chi2_ppf_scalar, p, nu)
