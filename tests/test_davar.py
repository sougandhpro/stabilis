"""The dynamic Allan variance: the identity, the conventions, the refusals.

The central assertion is `window == len(record)` giving back the batch
result **bit for bit**, not to a tolerance. That is only possible because
the surface is built by calling the registered estimator rather than
reimplementing it, so the test is really a check that no shortcut has
crept into the windowing -- a re-zeroed window, a re-derived tau grid, a
reused coefficient array -- each of which would leave the values equal to
1e-15 and fail this.

The second group covers what a surface has that a curve does not: window
placement, the centre-time convention, and what happens to a tau one
window can estimate and the next cannot.

`test_the_sliding_update_loses_significance_permanently` is here rather
than in a docstring because it is the measured justification for the
locked decision to recompute. It reproduces Galleani and Tavella's
eq. (13) and shows what it costs.
"""

import numpy as np
import pytest

from stabilis.stats import oadev, ohdev
from stabilis.stream import MAX_WINDOWS, STEP_DIVISOR, davar
from stabilis.synth import power_law_freq, power_law_phase
from stabilis.types import FreqSeries, PhaseSeries, to_phase

RATE = 1.0
H = 1e-24


def record(n, seed, alpha=0):
    return power_law_phase(n, alpha, RATE, np.random.default_rng(seed), H)


def disturbed(n, seed, lo, hi, factor):
    """White FM with its noise multiplied by `factor` over `[lo, hi)`."""
    fs = power_law_freq(n, 0, RATE, np.random.default_rng(seed), H)
    y = np.array(fs.y)
    y[lo:hi] *= factor
    return to_phase(FreqSeries(y=y, rate_hz=RATE))


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------

def test_one_window_over_the_whole_record_is_the_batch_call_bit_for_bit():
    ps = record(4095, 3)
    batch = oadev(ps, taus="octave")
    surface = davar(ps, window=ps.n)

    assert surface.n_windows == 1
    assert np.array_equal(surface.af, batch.af)
    assert np.array_equal(surface.tau_s, batch.tau_s)
    assert np.array_equal(surface.var[0], batch.var)
    assert np.array_equal(surface.n_terms[0], batch.n_terms)


def test_the_identity_holds_for_another_estimator_and_for_frequency_input():
    ps = record(2047, 4)
    surface = davar(ps, window=ps.n, kind="ohdev")
    assert np.array_equal(surface.var[0], ohdev(ps, taus="octave").var)

    fs = power_law_freq(1023, 0, RATE, np.random.default_rng(9), H)
    assert np.array_equal(davar(fs, window=fs.n).var[0], oadev(fs, taus="octave").var)


def test_the_identity_survives_gaps():
    x = np.array(record(2047, 5).x)
    x[400:406] = np.nan
    series = PhaseSeries(x=x, rate_hz=RATE)
    assert np.array_equal(davar(series, window=series.n).var[0],
                          oadev(series, taus="octave").var)


def test_each_window_is_the_batch_call_on_that_slice():
    """Not only the whole record: every row is exactly its own window."""
    ps = record(2047, 6)
    surface = davar(ps, window=512, step=256, taus=[1, 2, 4, 8])
    for k, start in enumerate(surface.start_index.tolist()):
        slice_ = PhaseSeries(x=ps.x[start:start + 512], rate_hz=RATE)
        assert np.array_equal(surface.var[k], oadev(slice_, taus=[1, 2, 4, 8]).var)


# ---------------------------------------------------------------------------
# What the surface is for
# ---------------------------------------------------------------------------

def test_a_disturbance_invisible_in_the_curve_is_a_ridge_in_the_surface():
    """The whole-record number describes neither state the clock was in."""
    ps = disturbed(8192, 19, 3000, 4000, factor=5.0)
    whole = oadev(ps, taus=[1])
    surface = davar(ps, window=1024, step=256, taus=[1])

    dev = surface.dev[:, 0]
    quiet, peak = np.nanmin(dev), np.nanmax(dev)
    assert peak / quiet > 4.0
    # The single number sits between the two states and matches neither.
    assert quiet < whole.dev[0] < peak
    assert whole.dev[0] / quiet > 1.5
    # And the ridge is where the disturbance was, to within the window.
    t_peak = surface.t_s[int(np.nanargmax(dev))]
    assert 3000 <= t_peak <= 4100


def test_at_af_is_the_other_cut_through_the_surface():
    ps = record(2047, 7)
    surface = davar(ps, window=512, step=128, taus=[1, 4])
    assert np.array_equal(surface.at_af(4), surface.var[:, 1])
    with pytest.raises(KeyError, match="not in this surface"):
        surface.at_af(3)


def test_window_result_is_an_ordinary_result_carrying_no_coefficients():
    from stabilis.stats import confidence

    ps = record(2047, 8)
    surface = davar(ps, window=512, step=256)
    got = surface.window_result(2)
    assert got.kind == "oadev"
    assert np.array_equal(got.var, surface.var[2])
    with pytest.raises(ValueError, match="keep_coeffs=False"):
        confidence(got)


def test_intervals_are_computed_per_window_when_asked_for():
    ps = record(2047, 9)
    surface = davar(ps, window=512, step=256, taus=[1, 2, 4], level=0.6827)
    assert surface.level == 0.6827
    assert np.all(surface.var_lo < surface.var)
    assert np.all(surface.var < surface.var_hi)
    assert np.all(surface.edof > 0.0)


def test_without_a_level_there_are_no_intervals_at_all():
    surface = davar(record(1023, 10), window=256)
    assert surface.var_lo is None and surface.var_hi is None


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

def test_windows_are_whole_and_the_uncovered_ends_are_recorded():
    ps = record(1000, 11)
    surface = davar(ps, window=400, step=100)
    starts = surface.start_index
    assert starts[0] == 0
    assert int(starts[-1]) + 400 <= ps.n
    assert int(starts[-1]) + 400 + 100 > ps.n
    step = next(s for s in surface.provenance.steps if s.action == "evaluate")
    assert step.params["uncovered_head_s"] == pytest.approx(199.5)


def test_the_centre_time_is_the_centroid_of_the_window():
    ps = record(1023, 12)
    surface = davar(ps, window=400, step=100)
    assert surface.t_s[0] == pytest.approx((400 - 1) / 2.0 / RATE)
    assert np.allclose(np.diff(surface.t_s), 100.0 / RATE)


def test_a_frequency_window_centre_sits_half_a_sample_later():
    """A frequency sample is an interval; a phase sample is an instant."""
    fs = power_law_freq(1023, 0, RATE, np.random.default_rng(13), H)
    surface = davar(fs, window=400, step=100)
    assert surface.t_s[0] == pytest.approx(400 / 2.0 / RATE)


def test_the_default_step_is_a_tenth_of_the_window_and_is_recorded():
    surface = davar(record(2047, 14), window=500)
    assert surface.step == 500 // STEP_DIVISOR
    step = next(s for s in surface.provenance.steps if s.action == "evaluate")
    assert step.params["step_default"] is True


def test_a_tau_one_window_cannot_reach_is_nan_rather_than_missing():
    """A gap shortens a window's longest run, so its longest tau goes."""
    x = np.array(record(2047, 15).x)
    # Long enough that a window straddling it has no run left that reaches
    # af 128, while windows clear of it do.
    x[850:1200] = np.nan
    series = PhaseSeries(x=x, rate_hz=RATE)
    surface = davar(series, window=512, step=128, taus=[1, 2, 4, 8, 16, 128],
                    gap_policy="segment")
    assert surface.var.shape[1] == surface.af.size
    holes = ~np.isfinite(surface.var)
    assert holes.any()
    # Every column that survived is estimable by at least one window.
    assert np.isfinite(surface.var).any(axis=0).all()


def test_a_tau_no_window_can_reach_is_dropped_and_counted():
    ps = record(1023, 16)
    surface = davar(ps, window=64, step=32, taus=[1, 2, 4, 40])
    assert 40 not in surface.af.tolist()
    step = next(s for s in surface.provenance.steps if s.action == "evaluate")
    assert step.params["taus_dropped_everywhere"] == 1


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_window_longer_than_the_record_is_refused():
    ps = record(255, 17)
    with pytest.raises(ValueError, match="exceeds the"):
        davar(ps, window=ps.n + 1)


def test_a_degenerate_window_or_step_is_refused():
    ps = record(255, 18)
    with pytest.raises(ValueError, match="at least 3 samples"):
        davar(ps, window=2)
    with pytest.raises(ValueError, match="at least 1 sample"):
        davar(ps, window=64, step=0)


def test_too_many_windows_is_refused_with_the_knob_named():
    ps = record(4095, 20)
    with pytest.raises(ValueError, match="Raise step"):
        davar(ps, window=64, step=1, max_windows=8)
    assert MAX_WINDOWS > 8


def test_an_estimator_that_ignores_the_tau_list_is_refused():
    """`wdev`'s scales are dyadic, so its columns cannot always be aligned.

    On an octave grid they happen to land on it and the surface is built
    normally; asked for anything else, the mismatch is caught rather than
    silently filed under whichever column `searchsorted` picked.
    """
    ps = record(2047, 21)
    assert davar(ps, window=1024, step=512, kind="wdev").af[0] == 1
    with pytest.raises(ValueError, match="not a subset of the requested grid"):
        davar(ps, window=1024, step=512, kind="wdev", taus=[1, 3, 5])


def test_a_window_too_short_for_any_tau_says_which_estimator_refused():
    """A grid the record allows but the estimator's algebra does not."""
    ps = record(255, 22)
    with pytest.raises(ValueError, match="no window of .* ohdev said"):
        davar(ps, window=64, step=32, kind="ohdev", taus=[40])


# ---------------------------------------------------------------------------
# The measured justification for recomputing
# ---------------------------------------------------------------------------

def test_the_sliding_update_loses_significance_permanently():
    """Galleani & Tavella eq. (13), measured. See theory/davar.md.

    Add the entering term and subtract the expiring one, and one 1 µs
    phase step -- an ordinary counter glitch -- leaves the running sum
    wrong by parts in 10⁶ for the rest of the record, long after the step
    has left the window. Ordinary noise costs parts in 10¹⁴ over the same
    number of slides, which is the point: the hazard is dynamic range, not
    run length, so a fast update that looks fine on a test record is not
    evidence that it is fine.
    """
    ps = record(19999, 23)
    x = np.array(ps.x)
    width, k = 500, 1

    def final_error(data):
        d2 = (data[2 * k:] - 2.0 * data[k:-k] + data[:-2 * k]) ** 2
        terms = width - 2 * k
        running = float(np.sum(d2[:terms]))
        for s in range(1, d2.size - terms + 1):
            running += d2[s + terms - 1] - d2[s - 1]
        true = float(np.sum(d2[-terms:]))
        return abs(running - true) / abs(true)

    clean = final_error(x)
    x[10000:10010] += 1e-6
    glitched = final_error(x)

    assert clean < 1e-12
    assert glitched > 1e-8
    assert glitched / clean > 1e3

    # And the recomputing implementation is exact on the same record,
    # because the last window is simply evaluated.
    surface = davar(PhaseSeries(x=x, rate_hz=RATE), window=width,
                    step=width, taus=[k])
    last = PhaseSeries(x=x[int(surface.start_index[-1]):
                           int(surface.start_index[-1]) + width], rate_hz=RATE)
    assert surface.var[-1, 0] == oadev(last, taus=[k]).var[0]
