"""Power spectral density, and the identities that tie it to sigma-tau.

The interesting tests here are the ones that cross a boundary: Parseval,
which ties the spectrum to the record's own variance at machine precision,
and the time-domain identity, which ties this module's output to
`powerlaw.py`'s conversion coefficients through two entirely separate code
paths. A PSD that agrees with an ADEV computed from the same samples is
evidence about both.
"""

import numpy as np
import pytest

from stabilis import FreqSeries, PhaseSeries, oadev, power_law_freq, to_phase
from stabilis.stats.psd import (
    SMALL_ANGLE_RAD2,
    convert,
    discrete_integration_gain,
    script_l,
    welch,
)
from stabilis.stats.window import coherent_gain, enbw_bins, window


def _white_fm(h0=1e-24, n=2**16, seed=3, rate=1.0):
    return power_law_freq(n, 0.0, rate, np.random.default_rng(seed), h_alpha=h0)


# --------------------------------------------------------------------------
# Windows and bandwidth
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,want", [("rectangular", 1.0), ("hann", 1.5)])
def test_the_noise_equivalent_bandwidth_is_exact(name, want):
    """Analytic, not tabulated: 1 for rectangular and 3/2 for Hann.

    It is the number a spur amplitude is divided by to become a density,
    so an error here turns a discrete tone into a noise level.
    """
    assert enbw_bins(window(name, 4096)) == pytest.approx(want, rel=1e-12)


def test_the_hann_window_keeps_half_a_sinusoids_amplitude():
    """Coherent gain 1/2, exactly, from `0.5 - 0.5*cos` summing to n/2."""
    assert coherent_gain(window("hann", 4096)) == pytest.approx(0.5, rel=1e-12)


def test_an_unknown_window_is_refused_with_the_choices():
    with pytest.raises(ValueError, match="unknown window"):
        window("flattop", 64)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def test_parseval_holds_to_machine_precision():
    """The spectrum integrates to the record's own mean square.

    One rectangular-windowed segment with no detrend, so nothing has been
    removed and the identity is exact rather than approximate. This is what
    makes the level meaningful: get the `sum(w**2)` normalization wrong and
    every number in the file is off by a constant no later step can find.
    """
    y = np.random.default_rng(0).normal(0.0, 1.0, 4096)
    spec = welch(FreqSeries(y=y, rate_hz=1.0), nperseg=4096, overlap=0.0,
                 window_name="rectangular", detrend="none")
    assert spec.psd.sum() * spec.df_hz == pytest.approx((y ** 2).mean(),
                                                        rel=1e-12)


@pytest.mark.parametrize("window_name", ["rectangular", "hann"])
def test_the_level_does_not_depend_on_the_window(window_name):
    """A density divided by the window's own power is window-independent."""
    spec = welch(_white_fm(), window_name=window_name)
    assert spec.psd.mean() == pytest.approx(1e-24, rel=0.05)


@pytest.mark.parametrize("h0", [1e-24, 4e-22, 1e-18])
def test_white_fm_comes_back_at_the_level_it_was_generated_at(h0):
    """End to end against noise of known amplitude, at three levels.

    The synthesizer's `h_alpha` is calibrated in SI units, so this is a
    real closure and not a self-comparison: the generator works in the time
    domain and the estimator in the frequency domain.
    """
    spec = welch(_white_fm(h0=h0))
    assert spec.psd.mean() == pytest.approx(h0, rel=0.05)


def test_the_mean_is_the_unbiased_statistic_not_the_median():
    """Each bin is chi-square distributed, so its median sits below its mean.

    Recorded because the median looks like the robust choice and is
    systematically 3.4% low here at 15 segments -- which would read as a
    calibration error in the estimator rather than a property of chi-square.
    """
    spec = welch(_white_fm())
    assert spec.psd.mean() == pytest.approx(1e-24, rel=0.05)
    assert np.median(spec.psd) < 0.98 * spec.psd.mean()


@pytest.mark.parametrize("alpha,slope", [(0.0, 0.0), (-2.0, -2.0), (2.0, 2.0)])
def test_the_power_law_slope_comes_back(alpha, slope):
    """`S_y(f) = h_alpha * f**alpha`, fitted over the middle three decades."""
    series = power_law_freq(2**16, alpha, 1.0, np.random.default_rng(7),
                            h_alpha=1e-24)
    spec = welch(series)
    f, psd = spec.f[1:], spec.psd[1:]
    band = (f > 3e-3) & (f < 0.3)
    fit = np.polyfit(np.log10(f[band]), np.log10(psd[band]), 1)[0]
    assert fit == pytest.approx(slope, abs=0.15)


# --------------------------------------------------------------------------
# The identity that closes the loop with the time domain
# --------------------------------------------------------------------------


def test_the_spectrum_and_the_allan_deviation_agree_on_white_fm():
    """`sigma_y(tau) = sqrt(h0 / (2*tau))` where `h0` is the flat `S_y`.

    Two entirely separate code paths over the same samples -- an FFT and a
    second difference -- reaching the same physical quantity through the
    published conversion. This is the strongest test in the file: neither
    path can be adjusted to please the other without breaking something
    that is checked elsewhere.
    """
    series = _white_fm(h0=1e-24, n=2**17)
    h0 = welch(series).psd.mean()

    result = oadev(to_phase(series), taus="octave")
    for tau, dev in zip(result.tau_s[:6], result.dev[:6]):
        assert dev == pytest.approx(np.sqrt(h0 / (2.0 * tau)), rel=0.10)


def test_a_cumulative_sum_is_not_an_ideal_integrator_and_by_how_much():
    """`(pi*f*tau0)**2 / sin(...)**2`, reaching 2.467 at Nyquist.

    `convert` uses the continuous-time identity `S_y = (2*pi*f)**2 * S_x`,
    which is what IEEE 1139 defines. `to_phase` integrates by cumulative
    sum, which is not that operator. The gap between them is real, it is
    predictable, and it is why a spectrum taken from phase and one taken
    from the same record's frequency separate near Nyquist.
    """
    series = _white_fm(h0=1e-24)
    spec = convert(welch(to_phase(series), detrend="none"), "S_y")
    gain = discrete_integration_gain(spec.f, 1.0)

    top = spec.f > 0.25
    assert (spec.psd[top] / gain[top]).mean() == pytest.approx(1e-24, rel=0.05)
    assert spec.psd[top].mean() > 1.5e-24            # uncorrected, plainly high
    assert gain[-1] == pytest.approx((np.pi / 2) ** 2, rel=1e-12)


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------


def test_the_conversions_are_exact_and_reversible():
    spec = welch(_white_fm(n=2**12))
    back = convert(convert(spec, "S_x"), "S_y")
    sx = convert(spec, "S_x")
    assert np.allclose(back.psd, spec.psd[1:], rtol=1e-14)
    assert np.allclose(sx.psd * (2.0 * np.pi * sx.f) ** 2, spec.psd[1:],
                       rtol=1e-14)


def test_converting_to_s_x_drops_dc_rather_than_returning_infinity():
    """`S_x = S_y/(2*pi*f)**2` diverges at f = 0, and a plotted infinity is
    worse than an absent point."""
    spec = welch(_white_fm(n=2**12))
    sx = convert(spec, "S_x")
    assert spec.f[0] == 0.0 and sx.f[0] > 0.0
    assert np.isfinite(sx.psd).all()
    assert any(s.action == "drop_dc" for s in sx.provenance.steps)


def test_s_phi_refuses_to_assume_a_carrier():
    """A spectrum converted with a guessed nu0 is wrong by a constant that
    nothing downstream can detect."""
    spec = welch(_white_fm(n=2**12))
    with pytest.raises(ValueError, match="carrier frequency is not in the data"):
        convert(spec, "S_phi")
    assert convert(spec, "S_phi", nu0_hz=10e6).nu0_hz == 10e6


def test_s_phi_is_s_x_times_the_carrier_squared():
    spec = welch(_white_fm(n=2**12))
    sx = convert(spec, "S_x")
    sphi = convert(spec, "S_phi", nu0_hz=5e6)
    assert np.allclose(sphi.psd, (2.0 * np.pi * 5e6) ** 2 * sx.psd, rtol=1e-14)


# --------------------------------------------------------------------------
# L(f), and where it stops being true
# --------------------------------------------------------------------------


def test_script_l_is_half_s_phi_where_the_angle_is_small():
    spec = welch(to_phase(_white_fm(h0=1e-30, n=2**14)), detrend="none")
    f, dbc, valid = script_l(spec, nu0_hz=10e6)
    sphi = convert(spec, "S_phi", nu0_hz=10e6)
    assert valid.all()
    ok = f > 0
    assert np.allclose(dbc[ok], 10.0 * np.log10(sphi.psd[ok] / 2.0), rtol=1e-12)


def test_script_l_is_withheld_where_the_phase_excursion_is_not_small():
    """Near the carrier of a noisy unit the approximation fails, and the
    honest output is no number rather than a plausible one.

    That near-carrier region is exactly the part of the curve a datasheet
    gets written from, which is why it is refused rather than flagged.
    """
    spec = welch(to_phase(_white_fm(h0=1e-20, n=2**14)), detrend="none")
    f, dbc, valid = script_l(spec, nu0_hz=10e9)
    assert not valid.all()
    assert np.isnan(dbc[~valid]).all()
    # The refused region is the low-frequency end, contiguously.
    assert not valid[1]
    assert valid[-1]


def test_a_unit_too_noisy_to_have_a_near_carrier_figure_gets_none():
    spec = welch(to_phase(_white_fm(h0=1e-16, n=2**14)), detrend="none")
    _, dbc, valid = script_l(spec, nu0_hz=10e9)
    assert not valid.any()
    assert np.isnan(dbc).all()


def test_the_small_angle_threshold_is_a_named_constant():
    """So that changing it is a visible act rather than an edit to a literal."""
    assert 0.0 < SMALL_ANGLE_RAD2 < 1.0


# --------------------------------------------------------------------------
# Gaps, defaults and refusals
# --------------------------------------------------------------------------


def test_gaps_segment_the_record_rather_than_poisoning_the_fft():
    """One NaN makes an entire FFT NaN, so runs are transformed separately."""
    y = np.random.default_rng(1).normal(0.0, 1.0, 4096)
    y[2000] = np.nan
    spec = welch(FreqSeries(y=y, rate_hz=1.0), nperseg=512)
    assert np.isfinite(spec.psd).all()
    assert spec.n_segments > 0


def test_runs_too_short_to_transform_are_reported_not_ignored():
    y = np.random.default_rng(1).normal(0.0, 1.0, 4096)
    y[600:1200] = np.nan                     # leaves a 600-sample island
    spec = welch(FreqSeries(y=y, rate_hz=1.0), nperseg=1024)
    assert any(s.action == "runs_dropped" for s in spec.provenance.steps)


def test_a_record_with_no_run_long_enough_is_refused_with_the_longest():
    y = np.random.default_rng(1).normal(0.0, 1.0, 300)
    with pytest.raises(ValueError, match="no gap-free run reaches"):
        welch(FreqSeries(y=y, rate_hz=1.0), nperseg=1024)


def test_phase_is_detrended_linearly_and_frequency_by_its_mean():
    """A phase record's linear term is its frequency offset, which is real
    and spectrally enormous. Removing it is a decision, so it is recorded."""
    phase = welch(to_phase(_white_fm(n=2**12)))
    freq = welch(_white_fm(n=2**12))
    assert phase.detrend == "linear"
    assert freq.detrend == "mean"
    step = [s for s in phase.provenance.steps if s.action == "welch"][-1]
    assert step.params["detrend"] == "linear"


def test_the_detrend_and_window_choices_are_refused_if_unknown():
    series = _white_fm(n=2**12)
    with pytest.raises(ValueError, match="unknown detrend"):
        welch(series, detrend="quadratic")
    with pytest.raises(ValueError, match="unknown window"):
        welch(series, window_name="kaiser")
    with pytest.raises(ValueError, match="overlap must be"):
        welch(series, overlap=1.0)


def test_the_segment_length_is_chosen_from_the_longest_gap_free_run():
    """Not from the record length -- a segment spanning a gap cannot be
    transformed at all."""
    y = np.random.default_rng(1).normal(0.0, 1.0, 2**15)
    y[2**11:] = np.nan
    spec = welch(FreqSeries(y=y, rate_hz=1.0))
    assert spec.nperseg <= 2**11


def test_an_array_that_is_not_a_series_is_refused():
    with pytest.raises(TypeError, match="PhaseSeries or FreqSeries"):
        welch(np.zeros(1024))


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def test_cli_psd_json_carries_the_bandwidth_and_the_provenance(tmp_path, capsys):
    import json

    from stabilis import save_frequency
    from stabilis.cli import main

    path = tmp_path / "y.dat"
    save_frequency(_white_fm(n=2**13), path)
    assert main(["psd", str(path), "--kind", "frequency", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "S_y"
    assert payload["units"] == "1/Hz"
    assert payload["enbw_hz"] > 0
    assert payload["provenance"]["steps"]


def test_cli_script_l_writes_null_where_it_withholds(tmp_path, capsys):
    import json

    from stabilis import save_phase
    from stabilis.cli import main

    path = tmp_path / "x.dat"
    save_phase(to_phase(_white_fm(h0=1e-20, n=2**13)), path)
    assert main(["psd", str(path), "--script-l", "--nu0", "1e10",
                 "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert None in payload["script_l_dbc"]        # NaN is not JSON
    assert not all(payload["small_angle_valid"])
