"""Discrete tones: detection, measurement, and removal.

The identity underneath all of it is derived rather than looked up, and
checked against this library's own estimator:

    x(t) = A sin(2*pi*nu*t + phi)  =>  sigma_y(tau) = 2*A*sin(pi*nu*tau)**2/tau

phase-averaged. Because it is exact, a tone of known amplitude predicts
both what the detector should measure and what the stability curve should
do -- so the module can be checked end to end without a reference table.
"""

import numpy as np
import pytest

from stabilis import PhaseSeries, adev, power_law_phase
from stabilis.stats.psd import welch
from stabilis.stats.spur import detect, notch, sinusoid_dev

RATE = 1.0
TONE_HZ = 0.0731234          # deliberately not on a bin centre
TONE_A = 3e-11


def _record(amplitude=TONE_A, f_hz=TONE_HZ, n=2**15, seed=4, h_alpha=1e-26,
            phase=0.7, extra=()):
    base = power_law_phase(n, 0.0, RATE, np.random.default_rng(seed),
                           h_alpha=h_alpha)
    t = np.arange(base.n) / RATE
    x = np.asarray(base.x, dtype=np.float64).copy()
    if amplitude:
        x += amplitude * np.sin(2 * np.pi * f_hz * t + phase)
    for a, f in extra:
        x += a * np.sin(2 * np.pi * f * t)
    return PhaseSeries(x=x, rate_hz=RATE)


# --------------------------------------------------------------------------
# The identity
# --------------------------------------------------------------------------


def test_the_closed_form_matches_the_shipped_estimator():
    """A Tier-1 identity: derived here, checked against `adev`.

    Averaged over the tone's phase, since a single realization has one.
    """
    taus = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    rng = np.random.default_rng(0)
    t = np.arange(2**14) / RATE
    acc = np.zeros_like(taus)
    trials = 40
    for _ in range(trials):
        x = TONE_A * np.sin(2 * np.pi * TONE_HZ * t + rng.uniform(0, 2 * np.pi))
        acc += adev(PhaseSeries(x=x, rate_hz=RATE),
                    taus=[int(v) for v in taus]).var
    measured = np.sqrt(acc / trials)
    assert measured == pytest.approx(sinusoid_dev(TONE_A, TONE_HZ, taus),
                                     rel=2e-3)


def test_a_tone_is_invisible_at_the_taus_where_the_identity_has_zeros():
    """`sin(pi*nu*tau)**2` vanishes at `nu*tau = 1, 2, ...`, so a stability
    plot sampled only there would show nothing at all.

    That is the reason to look for spurs in the spectrum rather than
    inferring them from a sigma-tau curve.
    """
    period = 1.0 / TONE_HZ
    assert sinusoid_dev(TONE_A, TONE_HZ, np.array([period])) < 1e-24
    assert sinusoid_dev(TONE_A, TONE_HZ, np.array([period / 2])) > 1e-12


# --------------------------------------------------------------------------
# Detection and measurement
# --------------------------------------------------------------------------


def test_a_tone_between_bins_is_measured_to_a_fraction_of_a_percent():
    """The point of fitting the record rather than reading the spectrum.

    The tone sits deliberately off a bin centre, where scalloping would
    cost a windowed periodogram several percent of its amplitude.
    """
    series = _record()
    spurs = detect(welch(series), series).spurs
    assert len(spurs) == 1
    assert spurs[0].f_hz == pytest.approx(TONE_HZ, rel=1e-4)
    assert spurs[0].amplitude == pytest.approx(TONE_A, rel=2e-3)
    assert spurs[0].f_bin_hz != spurs[0].f_hz      # the bin was only a hint


def test_the_reported_rms_is_the_amplitude_over_root_two():
    series = _record()
    spur = detect(welch(series), series).spurs[0]
    assert spur.rms == pytest.approx(spur.amplitude / np.sqrt(2.0), rel=1e-12)


def test_a_clean_record_yields_no_tones():
    """The false-alarm rate has to be believable in the direction that
    matters: a detector that finds spurs in pure noise is worthless."""
    series = _record(amplitude=0.0, seed=9)
    assert len(detect(welch(series), series)) == 0


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 11])
def test_pure_noise_stays_clean_across_seeds(seed):
    series = _record(amplitude=0.0, seed=seed)
    assert len(detect(welch(series), series)) == 0


def test_two_tones_are_found_separately_and_measured_separately():
    series = _record(extra=((1e-11, 0.21),))
    spurs = sorted(detect(welch(series), series).spurs, key=lambda s: s.f_hz)
    assert len(spurs) == 2
    assert spurs[0].amplitude == pytest.approx(TONE_A, rel=5e-3)
    assert spurs[1].f_hz == pytest.approx(0.21, rel=1e-4)
    assert spurs[1].amplitude == pytest.approx(1e-11, rel=5e-3)


def test_the_threshold_is_what_decides_and_it_is_stated_in_dB():
    """Raising the threshold past a tone's measured excess hides it, which
    is the check that the reported excess and the decision agree."""
    series = _record()
    found = detect(welch(series), series).spurs[0]
    assert len(detect(welch(series), series,
                      threshold_db=found.excess_db - 3.0)) == 1
    assert len(detect(welch(series), series,
                      threshold_db=found.excess_db + 3.0)) == 0


def test_the_searched_band_excludes_the_edges_and_says_so():
    """Their floor is a median over a truncated window, and at the low end
    they carry the residue of per-segment detrending -- which produced a
    confident 14.8 dB detection of a tone that was not there."""
    series = _record()
    table = detect(welch(series), series)
    spec = welch(series)
    lo, hi = table.searched_hz
    assert lo > spec.f[1]
    assert hi < spec.f[-1]
    step = [s for s in table.provenance.steps if s.action == "detect"][-1]
    assert "searched_hz" in step.params


def test_the_expected_false_alarm_count_is_reported():
    """A detector that cannot say how often it cries wolf is not a
    measurement. The chi-square tail is exact, so it is printed."""
    series = _record()
    strict = detect(welch(series), series, threshold_db=20.0)
    loose = detect(welch(series), series, threshold_db=3.0)
    assert strict.expected_false_alarms <= loose.expected_false_alarms
    assert loose.expected_false_alarms > 0.0


def test_a_full_candidate_list_is_flagged_rather_than_silently_truncated():
    series = _record(extra=((1e-11, 0.21), (1e-11, 0.33)))
    table = detect(welch(series), series, max_spurs=1)
    assert len(table) == 1
    assert any(s.action == "truncated" for s in table.provenance.steps)


def test_detection_refuses_a_spectrum_in_the_wrong_units():
    """`S_phi` is scaled by the carrier; an amplitude read from it would be
    in radians while the record is in seconds."""
    from stabilis.stats.psd import convert

    series = _record()
    with pytest.raises(ValueError, match="units the record is in"):
        detect(convert(welch(series), "S_phi", nu0_hz=10e6), series)


def test_an_even_floor_window_is_refused():
    series = _record()
    with pytest.raises(ValueError, match="odd"):
        detect(welch(series), series, floor_bins=32)


# --------------------------------------------------------------------------
# Notching, which is an edit
# --------------------------------------------------------------------------


def test_notching_removes_the_tone_and_leaves_the_noise():
    series = _record()
    table = detect(welch(series), series)
    cleaned = notch(series, table)

    t = np.arange(series.n) / RATE
    residual = 2.0 * abs(np.vdot(np.exp(-2j * np.pi * TONE_HZ * t),
                                 cleaned.x)) / series.n
    assert residual < TONE_A / 1000.0
    # The record is otherwise intact -- same samples, minus one sinusoid.
    assert cleaned.n == series.n
    assert np.isfinite(cleaned.x).all()
    removed = np.asarray(series.x) - np.asarray(cleaned.x)
    assert np.std(removed) == pytest.approx(TONE_A / np.sqrt(2.0), rel=0.02)


def test_notching_makes_the_spur_undetectable_afterwards():
    series = _record()
    cleaned = notch(series, detect(welch(series), series))
    assert len(detect(welch(cleaned), cleaned)) == 0


def test_every_removal_is_recorded_with_its_frequency_and_amplitude():
    """A record with a tone taken out is a different record, and the file
    has to say which one."""
    series = _record()
    table = detect(welch(series), series)
    cleaned = notch(series, table)
    step = [s for s in cleaned.provenance.steps if s.action == "notch"][-1]
    assert step.params["n_tones"] == 1
    assert step.params["f_hz"][0] == pytest.approx(TONE_HZ, rel=1e-4)
    assert step.params["amplitude"][0] == pytest.approx(TONE_A, rel=2e-3)
    assert "no longer describes the device as measured" in step.outcome


def test_notching_nothing_is_refused_rather_than_recorded_as_an_edit():
    """It would leave a provenance entry claiming an edit that did not
    happen, which is worse than an error."""
    series = _record(amplitude=0.0, seed=9)
    with pytest.raises(ValueError, match="nothing to notch"):
        notch(series, detect(welch(series), series))


def test_notching_preserves_the_series_type_and_its_epoch():
    series = PhaseSeries(x=_record().x, rate_hz=RATE, epoch=1.7e9)
    cleaned = notch(series, detect(welch(series), series))
    assert isinstance(cleaned, PhaseSeries)
    assert cleaned.epoch == 1.7e9


def test_the_stability_curve_loses_exactly_the_predicted_bump():
    """The end-to-end closure: the difference between the curves before and
    after notching is the closed form for the tone that was removed."""
    series = _record(h_alpha=1e-30)          # tone-dominated, so the bump is it
    taus = [2, 4, 8, 16]
    before = adev(series, taus=taus)
    after = adev(notch(series, detect(welch(series), series)), taus=taus)
    predicted = sinusoid_dev(TONE_A, TONE_HZ, np.asarray(taus, dtype=float))

    assert before.dev == pytest.approx(predicted, rel=0.05)
    assert (after.dev < before.dev / 100).all()


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def test_cli_spur_reports_the_tone_and_the_false_alarm_count(tmp_path, capsys):
    from stabilis import save_phase
    from stabilis.cli import main

    path = tmp_path / "x.dat"
    save_phase(_record(), path)
    assert main(["spur", str(path)]) == 0
    out = capsys.readouterr().out
    assert "expected by chance" in out
    assert "searched" in out
    assert "0.0731" in out


def test_cli_spur_notch_writes_a_file_that_says_what_was_removed(
        tmp_path, capsys):
    import json

    from stabilis import load_phase, save_phase
    from stabilis.cli import main

    path, out = tmp_path / "x.dat", tmp_path / "clean.dat"
    save_phase(_record(), path)
    assert main(["spur", str(path), "--notch-out", str(out)]) == 0
    assert "no longer describes the device as measured" in capsys.readouterr().out

    # The tone is gone from the file that was written.
    assert main(["spur", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["spurs"] == []


def test_cli_spur_refuses_to_notch_when_nothing_was_found(tmp_path):
    from stabilis import save_phase
    from stabilis.cli import main

    path = tmp_path / "x.dat"
    save_phase(_record(amplitude=0.0, seed=9), path)
    assert main(["spur", str(path), "--notch-out",
                 str(tmp_path / "no.dat")]) == 1
    assert not (tmp_path / "no.dat").exists()
