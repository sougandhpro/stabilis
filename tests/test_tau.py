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
