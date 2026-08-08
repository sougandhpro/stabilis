"""The classical estimators against certified values and published tables.

Two distinct kinds of assertion live here and the difference matters:

* **Certified** comparisons are against exact rational arithmetic computed
  by `validate.certified`, an independent implementation. These are
  correctness assertions, reported as LRE -- correct significant digits.
* **Published** comparisons are against Stable32-produced tables. These are
  *interop* assertions. Matching them proves this library agrees with the
  incumbent, which is commercially necessary and epistemically weak.
"""

from fractions import Fraction

import numpy as np
import pytest

from stabilis.reference import (
    EXPECTED_NBS14_9POINT,
    INTEROP_RTOL,
    KNOWN_DISCREPANCIES,
    nbs14_1000point,
    nbs14_9point,
)
from stabilis.stats import adev, mdev, oadev, ohdev, tdev
from stabilis.types import FreqSeries
from stabilis.validate.certified import (
    certified_deviation,
    exact_avar,
    exact_mvar,
    exact_ohvar,
    exact_oavar,
    exact_phase_from_freq,
    exact_tvar,
    lre,
)

# (name, float64 estimator, exact oracle)
ESTIMATORS = [
    ("adev", adev, exact_avar),
    ("oadev", oadev, exact_oavar),
    ("mdev", mdev, exact_mvar),
    ("tdev", tdev, exact_tvar),
    ("ohdev", ohdev, exact_ohvar),
]

# float64 carries ~15.95 decimal digits. Requiring 13 leaves headroom for
# the accumulation in a 1000-point sum while still failing on a real defect.
MIN_LRE = 13.0


def _nbs14_9_exact():
    return exact_phase_from_freq([Fraction(int(v)) for v in nbs14_9point()])


def _nbs14_1000_exact(n=1000):
    from stabilis.reference import NBS14_LEHMER_MODULUS, nbs14_1000point_integers

    y = [Fraction(k, NBS14_LEHMER_MODULUS) for k in nbs14_1000point_integers(n)]
    return exact_phase_from_freq(y)


@pytest.mark.parametrize("name,fn,exact", ESTIMATORS)
@pytest.mark.parametrize("af", [1, 2])
def test_certified_nbs14_9point(name, fn, exact, af):
    """Correct to at least MIN_LRE digits against exact rational arithmetic."""
    result = fn(FreqSeries(nbs14_9point(), rate_hz=1.0), taus="all")
    certified = certified_deviation(exact(_nbs14_9_exact(), af))
    assert lre(result.dev_at(af), certified) >= MIN_LRE


@pytest.mark.parametrize("name,fn,exact", ESTIMATORS)
@pytest.mark.parametrize("af", [1, 10, 100])
def test_certified_nbs14_1000point(name, fn, exact, af):
    """Same, on the 1000-point Lehmer set, at the published averaging factors."""
    result = fn(FreqSeries(nbs14_1000point(), rate_hz=1.0), taus="all", max_af=100)
    certified = certified_deviation(exact(_nbs14_1000_exact(), af))
    assert lre(result.dev_at(af), certified) >= MIN_LRE


@pytest.mark.parametrize("name,fn,_exact", ESTIMATORS)
def test_published_nbs14_9point_interop(name, fn, _exact):
    """INTEROP: agrees with the Stable32-produced table to its 7 published figures.

    No entry is exempted and no entry has been edited. `INTEROP_RTOL` is the
    tightest tolerance seven published figures support, so asserting against
    the table as printed is both possible and required.
    """
    expected = EXPECTED_NBS14_9POINT.get(name)
    if not expected:
        pytest.skip(f"no published values for {name}")
    result = fn(FreqSeries(nbs14_9point(), rate_hz=1.0), taus="all")
    for af, want in expected.items():
        assert result.dev_at(af) == pytest.approx(want, rel=INTEROP_RTOL)


def test_recorded_discrepancies_stay_below_the_published_tolerance():
    """Declared source-quality notes must still be notes, not conflicts.

    `KNOWN_DISCREPANCIES` exists so a table entry that disagrees with our
    certified value gets *recorded* rather than quietly rewritten to match
    us. Each one has to stay finer than the source's own precision -- if a
    disagreement ever grows past `INTEROP_RTOL` it is a real conflict and
    this fails, which is the point.
    """
    fs = FreqSeries(nbs14_9point(), rate_hz=1.0)
    for (name, af), (published, certified, note) in KNOWN_DISCREPANCIES.items():
        assert EXPECTED_NBS14_9POINT[name][af] == published, (
            f"{name}({af}): the published value in the table was edited; "
            f"reference tables record the source, not our own output"
        )
        assert published != certified, f"{name}({af}) no longer disagrees"
        rel = abs(published - certified) / certified
        assert rel < INTEROP_RTOL, (
            f"{name}({af}): disagreement {rel:.2e} now exceeds the published "
            f"tolerance {INTEROP_RTOL:.0e} -- this is a real conflict. {note}"
        )
        # And our own value must be the certified one, not the published one.
        got = ohdev(fs, taus="all").dev_at(af) if name in ("hdev", "ohdev") else None
        if got is not None:
            assert got == pytest.approx(certified, rel=1e-12)


@pytest.mark.parametrize("name,fn,_exact", ESTIMATORS)
def test_coefficient_contract(name, fn, _exact):
    """variance == scale * mean(coeffs**2), for every averaging factor.

    This makes the single-CI-path contract self-enforcing: the confidence
    interval is computed from `coeffs`, so if they ever stop reproducing the
    reported variance the error bars would be silently wrong.
    """
    rng = np.random.default_rng(11)
    result = fn(FreqSeries(rng.standard_normal(500), rate_hz=1.0), taus="octave")
    assert len(result.coeffs) == result.af.size
    for var, c in zip(result.var, result.coeffs):
        assert result.scale * np.mean(c**2) == pytest.approx(var, rel=1e-15)


@pytest.mark.parametrize(
    "name,fn,expected_terms",
    [
        ("oadev", oadev, lambda n, m: n - 2 * m),
        ("mdev", mdev, lambda n, m: n - 3 * m + 1),
        ("ohdev", ohdev, lambda n, m: n - 3 * m),
    ],
)
def test_term_counts(name, fn, expected_terms):
    """The term count is the first thing to break on an indexing error."""
    n_phase = 257
    ps_result = fn(FreqSeries(np.zeros(n_phase - 1), rate_hz=1.0), taus="all")
    for m, got in zip(ps_result.af, ps_result.n_terms):
        assert got == expected_terms(n_phase, int(m)), f"{name} at m={m}"


def test_gaps_reduce_the_term_count():
    """A gap invalidates only the terms that span it, and says so."""
    x = np.arange(101.0)
    clean = oadev(FreqSeries(np.diff(x), rate_hz=1.0), taus="all")
    gapped_y = np.diff(x)
    gapped_y[50] = np.nan
    gapped = oadev(FreqSeries(gapped_y, rate_hz=1.0), taus="all")
    assert np.all(gapped.n_terms < clean.n_terms)


def test_oracle_is_independent_of_the_production_path():
    """The production path must never import the oracle.

    Its independence is the entire evidential value: a shared helper would
    quietly turn the certified comparison into a tautology.
    """
    import ast
    import inspect

    import stabilis.stats.classical as classical
    import stabilis.stats.core as core

    for module in (classical, core):
        tree = ast.parse(inspect.getsource(module))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(f"{node.module or ''}.{a.name}" for a in node.names)
        offenders = [n for n in names if "validate" in n or "certified" in n]
        assert not offenders, f"{module.__name__} imports the oracle: {offenders}"
