"""Plotting. matplotlib is optional, so every test here skips without it.

The plots are checked for the things that can be wrong in a way a glance
would not catch: log axes, asymmetric error bars drawn asymmetrically,
zeros dropped rather than floored, and spec points styled differently from
measurements.
"""

import numpy as np
import pytest

from stabilis.bench import Bench, Channel, Device, Measurement, Source
from stabilis.bench.solve import solve
from stabilis.stats import confidence, oadev
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries

plt = pytest.importorskip("matplotlib.pyplot")
from stabilis import plot  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def result():
    fs = power_law_freq(2048, 0, rate_hz=1.0, rng=np.random.default_rng(1),
                        h_alpha=4e-24)
    return oadev(fs, taus="octave")


def test_sigma_tau_is_log_log_with_the_data_on_it(result):
    ax = plot.sigma_tau(result, label="A vs B")
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    line = ax.get_lines()[0]
    assert np.allclose(line.get_xdata(), result.tau_s)
    assert np.allclose(line.get_ydata(), result.dev)
    assert ax.get_legend() is not None


def test_error_bars_are_drawn_asymmetric_because_the_interval_is(result):
    """A chi-square interval on a variance is not symmetric.

    Drawn as a symmetric +/- it would understate the upper bound badly at
    small degrees of freedom, which is exactly where a reader most needs
    to see it.
    """
    ci = confidence(result)
    ax = plot.sigma_tau(result, confidence=ci)
    # errorbar draws the caps as LineCollections; the segment extents carry
    # the bounds we asked for.
    segments = ax.collections[0].get_segments()
    lows = np.array([s[0, 1] for s in segments])
    highs = np.array([s[1, 1] for s in segments])
    assert np.allclose(lows, ci.dev_lo, rtol=1e-9, atol=0.0)
    assert np.allclose(highs, ci.dev_hi, rtol=1e-9, atol=0.0)

    # The arms differ, and by more at long tau where eta is small. Compared
    # as a ratio with atol=0 on purpose: `np.allclose`'s default atol of
    # 1e-8 is far larger than these deviations of order 1e-14, so the
    # default would call any two of them equal -- the same absolute-cutoff
    # trap `_nnls._gradient_tolerance` documents.
    upper = highs - result.dev
    lower = result.dev - lows
    asymmetry = upper / lower
    assert np.all(asymmetry > 1.0), "the upper arm of a chi-square interval is longer"
    assert asymmetry[-1] > asymmetry[0] * 1.5, (
        "asymmetry must grow as the degrees of freedom fall"
    )


def test_two_curves_compose_on_one_axis(result):
    ax = plot.sigma_tau(result, label="first")
    plot.sigma_tau(result, label="second", ax=ax)
    assert len(ax.get_lines()) == 2
    assert [t.get_text() for t in ax.get_legend().get_texts()] == \
        ["first", "second"]


def test_slope_guides_pass_through_the_anchor_with_the_right_exponents(result):
    ax = plot.sigma_tau(result, label="data")
    plot.slope_guides(ax)
    guides = {line.get_label(): line for line in ax.get_lines()
              if "tau^" in str(line.get_label())}
    assert len(guides) == len(plot.SLOPES)

    tau0, sigma0 = result.tau_s[0], result.dev[0]
    for name, exponent in plot.SLOPES.items():
        line = next(v for k, v in guides.items() if k.startswith(name))
        x, y = line.get_xdata(), line.get_ydata()
        assert np.allclose(y, sigma0 * (x / tau0) ** exponent, rtol=1e-9)


def test_slope_guides_refuse_an_empty_axis():
    """Anchoring to nothing would put the guides somewhere meaningless."""
    _fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="nothing plotted yet"):
        plot.slope_guides(ax)


def test_white_and_flicker_pm_share_one_guide_line():
    """They have the same ADEV slope, and pretending otherwise would lie.

    This is the degeneracy `powerlaw.py` stacks modified rows to break; on
    a sigma-tau plot the two are simply not distinguishable, so one line
    carries both names.
    """
    assert "W/F PM" in plot.SLOPES
    assert sum(1 for v in plot.SLOPES.values() if v == -1.0) == 1


def test_spec_points_are_styled_as_a_claim_not_a_measurement():
    """A datasheet curve must never look like your own data."""
    source = Source(id="OCXO", spec_adev=((1.0, 1e-12), (10.0, 3e-13)))
    ax = plot.spec_points(source)
    line = ax.get_lines()[0]
    assert line.get_linestyle() == "--"
    assert line.get_markerfacecolor() == "none"
    assert "spec, not measured" in line.get_label()


def test_spec_points_refuses_a_source_with_no_spec():
    with pytest.raises(ValueError, match="no spec_adev"):
        plot.spec_points(Source(id="bare"))


def test_phase_and_frequency_use_real_time_on_the_x_axis():
    fs = power_law_freq(512, 0, rate_hz=4.0, rng=np.random.default_rng(2))
    ax = plot.frequency(fs, label="y")
    x = ax.get_lines()[0].get_xdata()
    assert x[1] - x[0] == pytest.approx(0.25), "tau0 = 1/rate, not 1"
    assert ax.get_xlabel() == "time (s)"


def test_bench_solution_drops_zeros_instead_of_flooring_them():
    """On a log axis zero is absent, not small.

    The hat pins a source at exactly zero when its noise is below what the
    bench can resolve. Plotting that at the bottom of the axis would invent
    a measurement; leaving a gap says what actually happened.
    """
    rng = np.random.default_rng(4)
    h = {"A": 1.0e-26, "B": 9.0e-24, "C": 1.0e-24}
    y = {k: power_law_freq(4096, 0, rate_hz=1.0, rng=rng, h_alpha=v).y
         for k, v in h.items()}
    pairs = (("A", "B"), ("B", "C"), ("C", "A"))
    results = {f"m{i}": oadev(FreqSeries(y[a] - y[b], rate_hz=1.0),
                              taus="octave")
               for i, (a, b) in enumerate(pairs)}
    bench = Bench(
        sources=tuple(Source(id=s) for s in "ABC"),
        devices=(Device(id="D1", channels=(Channel(id="A-1", floor=0.0),)),),
        measurements=tuple(
            Measurement(id=f"m{i}", channel_id="A-1", source_a=a, source_b=b)
            for i, (a, b) in enumerate(pairs)),
    )
    sol = solve(bench, results)
    assert np.any(sol.source_var == 0.0), "fixture should pin the quiet source"

    ax = plot.bench_solution(sol)
    quiet = next(line for line in ax.get_lines() if line.get_label() == "A")
    assert len(quiet.get_xdata()) < sol.tau_s.size
    assert np.all(np.asarray(quiet.get_ydata()) > 0.0)


def test_save_writes_a_file(tmp_path, result):
    out = plot.save(plot.sigma_tau(result), tmp_path / "curve.png")
    assert out.exists() and out.stat().st_size > 1000


def test_slope_guides_do_not_rescale_the_axis(result):
    """A guide should pass through the plot, not define it.

    A tau^-1 line drawn across three decades of tau spans three decades of
    sigma. Letting that autoscale squashes the very data the guide was
    added to help read.
    """
    ax = plot.sigma_tau(result, label="data")
    before_x, before_y = ax.get_xlim(), ax.get_ylim()
    plot.slope_guides(ax)
    assert ax.get_xlim() == before_x
    assert ax.get_ylim() == before_y
