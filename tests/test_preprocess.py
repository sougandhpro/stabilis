"""Descriptive statistics and gap segmentation."""

import numpy as np
import pytest

from stabilis.preprocess import basic_stats, longest_segment, segments


def test_basic_stats():
    s = basic_stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["n"] == 4 and s["n_finite"] == 4 and s["n_gaps"] == 0
    assert s["min"] == 1.0 and s["max"] == 4.0
    assert s["mean"] == 2.5 and s["median"] == 2.5
    assert s["std"] == pytest.approx(np.sqrt(5.0 / 3.0))


def test_basic_stats_excludes_gaps():
    s = basic_stats(np.array([1.0, np.nan, 3.0]))
    assert s["n"] == 3 and s["n_finite"] == 2 and s["n_gaps"] == 1
    assert s["mean"] == 2.0


def test_basic_stats_json_safe():
    """Values must be builtins, not numpy scalars, before serialization."""
    s = basic_stats(np.array([1.0, 2.0]))
    assert all(isinstance(v, (int, float)) for v in s.values())
    assert not any(isinstance(v, np.generic) for v in s.values())


def test_basic_stats_single_point_std_is_nan():
    assert np.isnan(basic_stats(np.array([7.0]))["std"])


def test_basic_stats_all_gaps_rejected():
    with pytest.raises(ValueError, match="no finite values"):
        basic_stats(np.array([np.nan, np.nan]))


def test_segments_no_gaps():
    assert segments(np.arange(5.0)) == [(0, 5)]


def test_segments_interior_gap():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert segments(a) == [(0, 2), (3, 5)]


def test_segments_leading_and_trailing_gaps():
    a = np.array([np.nan, 1.0, 2.0, np.nan])
    assert segments(a) == [(1, 3)]


def test_segments_min_length_filter():
    a = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    assert segments(a, min_length=2) == [(2, 5)]


def test_segments_empty_and_all_gap():
    assert segments(np.array([])) == []
    assert segments(np.array([np.nan, np.nan])) == []


def test_longest_segment():
    a = np.array([1.0, np.nan, 2.0, 3.0, 4.0, np.nan, 9.0])
    assert longest_segment(a) == (2, 5)
    with pytest.raises(ValueError, match="no finite values"):
        longest_segment(np.array([np.nan]))
