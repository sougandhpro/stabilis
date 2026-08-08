"""Averaging-factor selection.

Reportedly the largest single cause of two tools disagreeing while computing
identical mathematics, so the grids are pinned exactly.
"""

import numpy as np
import pytest

from stabilis.tau import averaging_factors, taus_from_afs


def test_octave():
    assert list(averaging_factors(4097, "octave")) == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def test_decade_is_1_2_4_per_decade():
    """The Stable32 convention, which is not 1-2-5."""
    assert list(averaging_factors(4001, "decade")) == [1, 2, 4, 10, 20, 40, 100, 200, 400, 1000]


def test_all():
    assert list(averaging_factors(41, "all")) == list(range(1, 11))


def test_log_is_deduplicated_and_sorted():
    afs = averaging_factors(4001, "log", n_log=30)
    assert afs[0] == 1
    assert list(afs) == sorted(set(afs))
    assert afs[-1] <= 1000


def test_default_max_af_is_quarter_of_n():
    """Documented policy: m <= n/4 unless the caller overrides."""
    assert averaging_factors(400, "all")[-1] == 100
    assert averaging_factors(401, "all")[-1] == 100


def test_explicit_max_af_is_honoured():
    assert averaging_factors(1000, "octave", max_af=10)[-1] == 8
    assert averaging_factors(1000, "all", max_af=3).tolist() == [1, 2, 3]


def test_max_af_clamped_to_n_minus_one():
    """Never propose an averaging factor the record cannot support."""
    assert averaging_factors(10, "all", max_af=10**6)[-1] == 9


def test_taus_from_afs():
    taus = taus_from_afs(np.array([1, 2, 4]), rate_hz=10.0)
    assert np.allclose(taus, [0.1, 0.2, 0.4])


def test_tiny_records_give_empty_grid_not_a_crash():
    assert averaging_factors(3, "octave").size == 0


def test_invalid_inputs():
    with pytest.raises(ValueError, match="unknown tau mode"):
        averaging_factors(100, "fortnightly")
    with pytest.raises(ValueError, match="at least 2 points"):
        averaging_factors(1, "octave")


# ---------------------------------------------------------------------------
# Bounded grids -- a memory bug, not a preference
# ---------------------------------------------------------------------------


def test_all_is_every_integer_while_that_is_affordable():
    """Short records lose nothing: "all" still means all."""
    from stabilis.tau import MAX_TAUS, averaging_factors

    afs = averaging_factors(1200, "all")          # max_af 300, under the cap
    assert afs.tolist() == list(range(1, 301))
    assert afs.size <= MAX_TAUS


@pytest.mark.parametrize("n", [200_000, 1_000_000, 20_000_000])
def test_all_stays_bounded_on_long_records(n):
    """The bug this exists to stop.

    Unbounded, `all` returns n/4 factors and `run_estimator` keeps a
    coefficient array of order n for each -- 60 GB at 200k points, 1500 GB
    at a million, and the process is killed rather than slowed. The count
    is asserted rather than the runtime, because a memory bug is not caught
    by a slow test.
    """
    from stabilis.tau import MAX_TAUS, averaging_factors

    afs = averaging_factors(n, "all")
    assert afs.size <= MAX_TAUS
    assert afs[0] == 1
    assert afs[-1] == n // 4, "the grid must still reach the intended max_af"
    assert np.all(np.diff(afs) > 0), "strictly increasing, no duplicates"


def test_the_thinned_grid_has_stable32_s_shape():
    """Dense at the start, geometric after -- what the incumbent does.

    Stable32's own all-tau export for a 4097-point record is 189 rows
    reaching AF 1006: consecutive at first, about 2% apart at the end. That
    file is committed under tests/data/stable32/, so this is a shape check
    against measured behaviour rather than an invention.
    """
    from stabilis.tau import DENSE_AF, averaging_factors

    afs = averaging_factors(4097, "all", max_af=1006, max_taus=200)
    assert afs[:DENSE_AF].tolist() == list(range(1, DENSE_AF + 1))
    assert afs[-1] == 1006
    # Growth in the tail is a few percent, not a factor of two (octave) and
    # not one (exhaustive).
    ratios = afs[1:] / afs[:-1]
    assert 1.005 < np.median(ratios[DENSE_AF:]) < 1.10


def test_an_over_large_grid_is_refused_rather_than_truncated():
    """Silently returning fewer taus than asked for would be worse."""
    from stabilis.tau import averaging_factors

    with pytest.raises(ValueError, match="over the 10 limit"):
        averaging_factors(10_000, "log", n_log=50, max_taus=10)


def test_explicit_averaging_factors_are_taken_exactly():
    """For reproducing someone else's tau list, which must not be thinned."""
    from stabilis.tau import averaging_factors

    want = [1, 7, 13, 100, 999]
    assert averaging_factors(5000, want).tolist() == want
    # sorted and deduplicated, but never resampled
    assert averaging_factors(5000, [9, 3, 3, 1]).tolist() == [1, 3, 9]

    with pytest.raises(ValueError, match="more than"):
        averaging_factors(50, [1, 200])
    with pytest.raises(ValueError, match=">= 1"):
        averaging_factors(50, [0, 5])


def test_suggest_names_a_grid_and_a_reason():
    from stabilis.tau import suggest

    small = suggest(2_000, rate_hz=1.0)
    assert small["mode"] == "all", "every integer is affordable here"

    big = suggest(10_000_000, rate_hz=1.0)
    assert big["mode"] == "octave"
    assert big["n_taus"] < 40
    assert "N/4" in big["why"]
    # The octave grid's last point is the largest power of two under
    # max_af = N/4, not max_af itself.
    assert big["tau_max_s"] == 2 ** 21


def test_keeping_coefficients_that_would_not_fit_is_refused_with_the_fix():
    """The second half of the memory bound, past the grid itself."""
    from stabilis.stats import oadev
    from stabilis.types import PhaseSeries

    x = np.cumsum(np.random.default_rng(0).standard_normal(400_000)) * 1e-12
    ps = PhaseSeries(x, rate_hz=1.0)
    with pytest.raises(MemoryError, match="keep_coeffs=False"):
        oadev(ps, taus="all")
    # And the same call is fine once the arrays are not retained.
    assert oadev(ps, taus="all", keep_coeffs=False).af.size <= 500
