"""Certified values by exact rational arithmetic.

This is a deliberate *second implementation*, and its independence is the
whole point. It is never imported by the production path: nothing in
`stabilis.stats` may import this module, and a test enforces that.

The trick that makes it possible is that every classical variance is a sum
of squares of integer linear combinations of the phase samples, divided by
an integer. So if the input data are rational -- and both NBS14 reference
sets are, being integers or integers over 2147483647 -- then the variance is
an **exact rational**, computable in `fractions.Fraction` with no floating
point anywhere. The deviation is a square root, so it is certified in
`decimal.Decimal` at 50 significant digits.

That gives certified values we compute ourselves, to far more digits than
any published table, without depending on a transcription being correct.
Accuracy of the production path is then reported the way NIST's Statistical
Reference Datasets report it: as LRE, the number of correct digits.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction

__all__ = [
    "exact_avar",
    "exact_oavar",
    "exact_mvar",
    "exact_tvar",
    "exact_ohvar",
    "exact_phase_from_freq",
    "certified_deviation",
    "lre",
    "CERTIFIED_DIGITS",
]

CERTIFIED_DIGITS = 50
_ONE = Fraction(1)


def exact_phase_from_freq(y: list[Fraction], tau0: Fraction = _ONE) -> list[Fraction]:
    """Exact phase from fractional frequency: x[0] = 0, x[k] = sum y[i]*tau0."""
    x = [Fraction(0)]
    for value in y:
        x.append(x[-1] + value * tau0)
    return x


def exact_oavar(x: list[Fraction], m: int, tau0: Fraction = _ONE) -> Fraction:
    """Exact overlapping Allan variance.

        avar = sum_{i=0}^{N-2m-1} (x[i+2m] - 2x[i+m] + x[i])**2
               / (2 * (N-2m) * (m*tau0)**2)
    """
    n = len(x)
    count = n - 2 * m
    if count < 1:
        raise ValueError(f"need N > 2m; got N={n}, m={m}")
    total = sum(
        (x[i + 2 * m] - 2 * x[i + m] + x[i]) ** 2 for i in range(count)
    )
    return total / (2 * count * (m * tau0) ** 2)


def exact_avar(x: list[Fraction], m: int, tau0: Fraction = _ONE) -> Fraction:
    """Exact non-overlapping Allan variance, on the decimated phase."""
    xd = x[::m]
    count = len(xd) - 2
    if count < 1:
        raise ValueError(f"need at least 3 decimated points; got {len(xd)}")
    total = sum((xd[k + 2] - 2 * xd[k + 1] + xd[k]) ** 2 for k in range(count))
    return total / (2 * count * (m * tau0) ** 2)


def exact_mvar(x: list[Fraction], m: int, tau0: Fraction = _ONE) -> Fraction:
    """Exact modified Allan variance.

        mvar = sum_j ( sum_{i=j}^{j+m-1} (x[i+2m] - 2x[i+m] + x[i]) )**2
               / (2 * m**4 * tau0**2 * (N-3m+1))
    """
    n = len(x)
    count = n - 3 * m + 1
    if count < 1:
        raise ValueError(f"need N >= 3m; got N={n}, m={m}")
    z = [x[i + 2 * m] - 2 * x[i + m] + x[i] for i in range(n - 2 * m)]
    total = sum(sum(z[j : j + m]) ** 2 for j in range(count))
    return total / (2 * m**4 * tau0**2 * count)


def exact_tvar(x: list[Fraction], m: int, tau0: Fraction = _ONE) -> Fraction:
    """Exact time variance: tvar = (m*tau0)**2 / 3 * mvar."""
    return (m * tau0) ** 2 * exact_mvar(x, m, tau0) / 3


def exact_ohvar(x: list[Fraction], m: int, tau0: Fraction = _ONE) -> Fraction:
    """Exact overlapping Hadamard variance, from the third difference.

        hvar = sum (x[i+3m] - 3x[i+2m] + 3x[i+m] - x[i])**2
               / (6 * (N-3m) * (m*tau0)**2)
    """
    n = len(x)
    count = n - 3 * m
    if count < 1:
        raise ValueError(f"need N > 3m; got N={n}, m={m}")
    total = sum(
        (x[i + 3 * m] - 3 * x[i + 2 * m] + 3 * x[i + m] - x[i]) ** 2
        for i in range(count)
    )
    return total / (6 * count * (m * tau0) ** 2)


def certified_deviation(var: Fraction, digits: int = CERTIFIED_DIGITS) -> Decimal:
    """Square root of an exact rational, to `digits` significant figures."""
    getcontext().prec = digits + 10
    value = Decimal(var.numerator) / Decimal(var.denominator)
    getcontext().prec = digits
    return +value.sqrt()


def lre(computed: float, certified: Decimal | Fraction) -> float:
    """Log relative error: the number of correct significant digits.

        LRE = -log10( |computed - certified| / |certified| )

    This is the NIST Statistical Reference Datasets metric. Reporting a
    number of correct digits says far more than a pass/fail tick -- a
    degradation to 9 digits on a deliberately ill-conditioned dataset, with
    an explanation, is more credible evidence than a green light.

    Returns `float("inf")` when the values agree exactly, and 0.0 when the
    certified value is zero but the computed one is not.
    """
    certified_dec = (
        certified
        if isinstance(certified, Decimal)
        else Decimal(certified.numerator) / Decimal(certified.denominator)
    )
    getcontext().prec = CERTIFIED_DIGITS + 10
    diff = abs(Decimal(repr(computed)) - certified_dec)
    if diff == 0:
        return float("inf")
    if certified_dec == 0:
        return 0.0
    return float(-(diff / abs(certified_dec)).log10())
