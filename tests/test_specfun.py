"""Special functions against closed forms, not against a table.

Every assertion here compares a rational approximation or an iterative
scheme with something that can be written down exactly. That is the point:
a table of expected values from another library would only tell us we agree
with it, and these functions exist precisely so that library is optional.

The normal quantile gets an independent oracle -- bisection on
`math.erfc`, which is slow, obviously correct, and shares no code with the
AS241 coefficients it checks. A mistyped coefficient is the realistic
failure mode for that function, and only an independent path catches it.
"""

import math
from decimal import Decimal, getcontext

import numpy as np
import pytest

from stabilis._specfun import (
    chi2_cdf,
    chi2_ppf,
    gammainc_p,
    gammainc_q,
    norm_ppf,
)


def _norm_cdf(z):
    """Φ(z) from the standard library, for the independent inverse below."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _norm_ppf_by_bisection(p, tol=1e-15):
    """Φ⁻¹ the slow, obvious way. 60 iterations, no coefficients to mistype."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# --- incomplete gamma ---------------------------------------------------


@pytest.mark.parametrize("a", [0.5, 1.0, 1.5, 2.0, 5.0, 50.0, 500.0])
def test_p_and_q_sum_to_one(a):
    x = np.linspace(0.01, 4.0 * a, 40)
    assert np.allclose(gammainc_p(a, x) + gammainc_q(a, x), 1.0, rtol=0, atol=1e-15)


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 7.5, 40.0])
def test_p_is_monotone_and_bounded(a):
    """P rises from 0 to 1 without ever stepping backwards.

    The upper limit has to clear the distribution, not merely scale with
    `a`: P(1/2, 3) is 0.986, nowhere near 1. Mean + 8 standard deviations
    (a + 8*sqrt(a)) plus a floor of 60 reaches it at every `a` here.
    """
    x = np.linspace(0.0, a + 8.0 * math.sqrt(a) + 60.0, 300)
    p = gammainc_p(a, x)
    assert p[0] == 0.0
    assert np.all(np.diff(p) >= 0.0)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert p[-1] == pytest.approx(1.0, abs=1e-12)


def test_p_at_a_equals_one_is_the_exponential_cdf():
    """P(1, x) = 1 - e^-x exactly, which pins the normalization."""
    x = np.linspace(0.0, 30.0, 100)
    assert np.allclose(gammainc_p(1.0, x), -np.expm1(-x), rtol=1e-14, atol=1e-16)


def test_p_at_half_integer_a_matches_the_error_function():
    """P(1/2, x) = erf(sqrt(x)), an independent closed form."""
    x = np.linspace(1e-6, 25.0, 60)
    want = np.array([math.erf(math.sqrt(v)) for v in x])
    assert np.allclose(gammainc_p(0.5, x), want, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize("a", [1, 2, 3, 5, 8, 13, 30])
def test_q_at_integer_a_matches_the_finite_poisson_sum(a):
    """For integer a, Q(a, x) = e^-x · Σ_{k<a} x^k/k! -- exactly summable."""
    for x in (0.1, 0.5, 1.0, 2.0, float(a), a + 0.5, a + 1.0, 3.0 * a):
        term, total = 1.0, 1.0
        for k in range(1, a):
            term *= x / k
            total += term
        want = math.exp(-x) * total
        assert gammainc_q(float(a), x) == pytest.approx(want, rel=1e-13, abs=1e-16)


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 4.5, 10.0, 100.0, 5000.0])
def test_the_series_and_continued_fraction_agree_where_they_meet(a):
    """Both branches evaluated at the same x, at the switch point x = a+1.

    A step at the boundary is the classic Numerical-Recipes `gammp` bug and
    it hides well, because each branch is individually accurate. Comparing
    the branches directly is the only way to see it -- sampling the public
    function either side of the switch measures its slope instead, which is
    a real 1e-10 effect that swamps the 1e-14 one being looked for.
    """
    from stabilis._specfun import _continued_fraction, _series

    x = a + 1.0
    assert _series(a, x) == pytest.approx(1.0 - _continued_fraction(a, x), rel=1e-13)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="must be positive"):
        gammainc_p(0.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        gammainc_p(1.0, -1.0)


# --- certified: exact Decimal evaluation at integer order ----------------
#
# For integer `a` the upper incomplete gamma is a finite sum, so it can be
# evaluated to arbitrary precision with `decimal` and used as a certified
# oracle -- the same device as `validate/certified.py`, applied here. This
# is what turns "the code runs" into a measured number of correct digits.

CERTIFIED_PRECISION = 60

# Measured floor over the grid below, not a round number picked to pass.
# The worst case is a=5000, x=5500 at 12.44 digits, and it is understood:
# the exponent contains `a·ln(x/a) - (x-a)`, whose terms are ~2500 there,
# so float64 spacing of 4.5e-13 on each puts an absolute ~1e-12 into the
# exponent and hence a relative ~1e-12 into the result. Carrying the
# exponent in extended precision would recover it, and would be effort
# spent on the twelfth digit of a tail probability of 3.6e-12. Every value
# a confidence interval actually uses lands at 14 digits or better.
MIN_LRE_SPECFUN = 12.4


def _exact_q(a: int, x: float) -> Decimal:
    """Q(a, x) = e^-x · Σ_{k<a} x^k/k!, exact for integer a."""
    getcontext().prec = CERTIFIED_PRECISION
    xd = Decimal(repr(x))
    term = Decimal(1)
    total = Decimal(1)
    for k in range(1, a):
        term = term * xd / k
        total += term
    return total * (-xd).exp()


def _lre(got: float, want: Decimal) -> float:
    """Correct significant digits, NIST StRD convention."""
    getcontext().prec = CERTIFIED_PRECISION
    g = Decimal(repr(got))
    if g == want:
        return float(CERTIFIED_PRECISION)
    return float(-(abs(g - want) / abs(want)).log10())


@pytest.mark.parametrize("a", [1, 10, 100, 1000, 5000])
@pytest.mark.parametrize("frac", [0.5, 0.9, 1.0, 1.1, 1.5, 2.0])
def test_certified_gammainc_q_at_integer_order(a, frac):
    """At least MIN_LRE_SPECFUN correct digits against exact arithmetic.

    This is the test that drove two fixes. Computing Q as `1 - P` returned
    exactly 0.0 at a=1000, x=2000 where the true value is 6.8e-136 -- total
    cancellation in the far tail, which is where a high-confidence bound
    lives. And the direct `a·ln x - x - lnΓ(a)` prefactor subtracts three
    numbers of size a·ln a, giving 11.0 digits at a=5000 against ~15 for the
    regrouped Stirling form.
    """
    x = float(a) * frac
    got = gammainc_q(float(a), x)
    if got == 0.0:
        # Genuine float64 underflow: at a=5000, x=1e4 the answer is ~1e-665.
        # Zero is the honest result, not a defect. Confirm it really is
        # unrepresentable rather than a cancellation artefact.
        assert _exact_q(a, x) < Decimal("1e-300")
        return
    assert _lre(got, _exact_q(a, x)) >= MIN_LRE_SPECFUN


# --- normal quantile ----------------------------------------------------


def test_norm_ppf_is_symmetric_and_zero_at_the_median():
    assert norm_ppf(0.5) == 0.0
    for p in (0.001, 0.01, 0.1, 0.25, 0.4):
        assert norm_ppf(p) == pytest.approx(-norm_ppf(1.0 - p), rel=1e-15)


@pytest.mark.parametrize(
    "p", [1e-14, 1e-13, 1.4e-12, 1e-11, 1e-10, 1e-6, 1e-3, 0.01, 0.075,
          0.1, 0.3, 0.4249, 0.4251, 0.5]
)
def test_norm_ppf_against_independent_bisection(p):
    """AS241 vs bisection on erfc -- no shared code, no shared coefficients.

    A mistyped coefficient is the realistic failure mode, and only an
    independent path catches it. Sweeps both branch boundaries:
    |p - 1/2| = 0.425, and the tail split at sqrt(-ln p) = 5, i.e.
    p ≈ 1.4e-12.

    Restricted to the lower half deliberately. The oracle is
    `0.5*erfc(-z/sqrt(2))`, which loses ~10 digits to cancellation as it
    approaches 1, so it is not fit to check the upper tail. Symmetry covers
    that half instead, and is asserted separately at 1e-15.
    """
    assert norm_ppf(p) == pytest.approx(_norm_ppf_by_bisection(p), rel=1e-9, abs=1e-9)


def test_norm_ppf_round_trips_through_the_cdf():
    """Φ(Φ⁻¹(p)) = p is the strongest single check available."""
    p = np.array([1e-12, 1e-6, 0.01, 0.16, 0.5, 0.84, 0.975, 0.999, 1 - 1e-9])
    back = np.array([_norm_cdf(z) for z in norm_ppf(p)])
    assert np.allclose(back, p, rtol=1e-13, atol=0)


def test_norm_ppf_edges_and_errors():
    assert norm_ppf(0.0) == -math.inf
    assert norm_ppf(1.0) == math.inf
    with pytest.raises(ValueError, match="must be in"):
        norm_ppf(1.5)


# --- chi-square ---------------------------------------------------------


def test_chi2_cdf_at_two_dof_is_the_exponential():
    """F(x; 2) = 1 - e^(-x/2), exactly."""
    x = np.linspace(0.0, 40.0, 100)
    assert np.allclose(chi2_cdf(x, 2.0), -np.expm1(-0.5 * x), rtol=1e-14, atol=1e-16)


def test_chi2_cdf_at_one_dof_is_the_error_function():
    """F(x; 1) = erf(sqrt(x/2))."""
    x = np.linspace(1e-8, 30.0, 80)
    want = np.array([math.erf(math.sqrt(0.5 * v)) for v in x])
    assert np.allclose(chi2_cdf(x, 1.0), want, rtol=1e-13, atol=1e-16)


@pytest.mark.parametrize("p", [1e-10, 1e-4, 0.025, 0.1587, 0.5, 0.8413, 0.975, 0.9999])
def test_chi2_ppf_at_two_dof_has_a_closed_form(p):
    """F⁻¹(p; 2) = -2·ln(1-p). The one case needing no iteration at all."""
    assert chi2_ppf(p, 2.0) == pytest.approx(-2.0 * math.log1p(-p), rel=1e-13)


@pytest.mark.parametrize("nu", [0.5, 1.0, 2.0, 3.0, 7.0, 25.0, 100.0, 1000.0, 10000.0])
@pytest.mark.parametrize("p", [1e-6, 0.001, 0.025, 0.1587, 0.5, 0.8413, 0.975, 0.999])
def test_chi2_ppf_round_trips_through_the_cdf(nu, p):
    """The inversion is only as good as its round trip, at every scale of nu."""
    x = chi2_ppf(p, nu)
    assert chi2_cdf(x, nu) == pytest.approx(p, rel=1e-11, abs=1e-15)


@pytest.mark.parametrize("nu", [1.0, 2.0, 9.0, 60.0, 5000.0])
def test_chi2_ppf_is_monotone_in_p(nu):
    p = np.linspace(1e-6, 1 - 1e-6, 500)
    x = chi2_ppf(p, nu)
    assert np.all(np.diff(x) > 0.0)


def test_chi2_ppf_mean_and_edges():
    """The median of chi-square(nu) sits just below nu; the edges are exact."""
    for nu in (1.0, 4.0, 20.0, 200.0):
        assert 0.0 < chi2_ppf(0.5, nu) < nu
    assert chi2_ppf(0.0, 5.0) == 0.0
    assert chi2_ppf(1.0, 5.0) == math.inf
    with pytest.raises(ValueError, match="nu must be positive"):
        chi2_ppf(0.5, 0.0)


def test_vectorization_and_broadcasting():
    """Scalars come back as floats; arrays broadcast against each other."""
    assert isinstance(chi2_ppf(0.5, 4.0), float)
    nu = np.array([2.0, 10.0, 100.0])
    got = chi2_ppf(0.975, nu)
    assert got.shape == (3,)
    assert np.allclose(got, [chi2_ppf(0.975, float(v)) for v in nu], rtol=0, atol=0)
    grid = chi2_ppf(np.array([[0.1], [0.9]]), nu)
    assert grid.shape == (2, 3)


def test_no_scipy_import():
    """These exist so scipy stays optional. Guard that they never reach for it."""
    import ast
    import inspect

    import stabilis._specfun as mod

    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("scipy") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("scipy")
