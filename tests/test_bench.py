"""The bench model: identifiability, and the solve that subsumes three features.

The identifiability tests are the ones worth reading. They assert a graph
theorem — a connected set of measurements determines its sources' individual
variances iff it contains an odd cycle — which is why the technique is
called *three*-cornered hat and why a ring of four determines nothing.
"""

import numpy as np
import pytest

from stabilis.bench import Bench, Channel, Device, Measurement, Source, check
from stabilis.bench.graph import design_matrix, suggest_edge
from stabilis.bench.solve import common_mode_warnings, solve
from stabilis.stats import oadev
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries


def make_bench(pairs, floor=None, channels=None, devices=None):
    """A bench from `[(source_a, source_b), ...]`, one channel unless told."""
    sources = sorted({s for pair in pairs for s in pair})
    channels = ["A-1"] * len(pairs) if channels is None else channels
    unique = sorted(set(channels))
    if devices is None:
        devs = (Device(id="D1", channels=tuple(
            Channel(id=c, floor=floor) for c in unique)),)
    else:
        devs = tuple(
            Device(id=d, channels=tuple(
                Channel(id=c, floor=floor) for c in unique if c in cs))
            for d, cs in devices.items()
        )
    ms = tuple(
        Measurement(id=f"m{i}", channel_id=channels[i], source_a=a, source_b=b)
        for i, (a, b) in enumerate(pairs)
    )
    return Bench(sources=tuple(Source(id=s) for s in sources),
                 devices=devs, measurements=ms)


# ---------------------------------------------------------------------------
# Identifiability -- the graph theorem
# ---------------------------------------------------------------------------


def test_the_triangle_is_identifiable_and_that_is_why_it_is_three_cornered():
    """Three sources, an odd cycle, and a declared channel floor.

    Declaring `floor=0.0` is the assumption the classical three-cornered
    hat makes silently; here it is written down, and it is what turns four
    unknowns against three rows into three against three.
    """
    ident = check(make_bench([("A", "B"), ("B", "C"), ("C", "A")], floor=0.0))
    assert ident.identifiable
    assert ident.rank == ident.n_unknowns == 3
    assert ident.bipartite == (False,), "a triangle is not bipartite"


def test_a_ring_of_four_determines_nothing_however_long_it_runs():
    """The result that matters most, because it looks fine and is not.

    A-B, B-C, C-D, D-A is a 4-cycle: bipartite, so the vector that raises
    {A, C} and lowers {B, D} is invisible to every measurement. No amount
    of averaging touches it.
    """
    ident = check(make_bench(
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")], floor=0.0))
    assert not ident.identifiable
    assert ident.bipartite == (True,)
    assert ident.deficiency == 1

    # And the null direction is exactly the alternating one.
    null = ident.null_space[0]
    sign = np.sign(null[: len(ident.sources)])
    order = {s: i for i, s in enumerate(ident.sources)}
    assert sign[order["A"]] == sign[order["C"]]
    assert sign[order["B"]] == sign[order["D"]]
    assert sign[order["A"]] != sign[order["B"]]


def test_one_diagonal_fixes_the_ring_and_both_are_offered():
    """The actionable half: which single measurement to go and take."""
    ring = make_bench([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
                      floor=0.0)
    assert set(suggest_edge(ring)) == {("A", "C"), ("B", "D")}

    fixed = make_bench(
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")], floor=0.0)
    assert check(fixed).identifiable


def test_two_records_against_a_common_reference_determine_all_three():
    """The odd-cycle theorem is about the variances, not about the records.

    `x_ab = x_ac - x_bc` identically, so a pair measured against a shared
    third is one subtraction away from the edge that closes the triangle.
    This is how the three-cornered hat is normally run -- two channels
    against the house clock -- and the check has to say so rather than send
    the user off to buy a counter.
    """
    from stabilis.bench.graph import derivable_edges

    chain = make_bench([("A", "C"), ("B", "C")], floor=0.0)
    ident = check(chain)
    assert not ident.identifiable            # from the variances alone
    assert ("A", "B") in ident.suggestions

    assert derivable_edges(chain) == (("A", "B", "C", "m0", "m1"),)
    assert ident.derivable == (("A", "B", "C", "m0", "m1"),)
    assert "no new data" in ident.report()
    assert "m0 - m1" in ident.report()

    # And it is the edge that fixes it.
    assert check(make_bench([("A", "C"), ("B", "C"), ("A", "B")],
                            floor=0.0)).identifiable


def test_a_ring_of_four_needs_no_new_data_either():
    """Both diagonals of the 4-cycle are subtractions of adjacent sides."""
    from stabilis.bench.graph import derivable_edges

    ring = make_bench([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
                      floor=0.0)
    pairs = {(a, b) for a, b, _via, _ma, _mb in derivable_edges(ring)}
    assert pairs == {("A", "C"), ("B", "D")}
    assert {(e[0], e[1]) for e in check(ring).derivable} == pairs


def test_an_edge_already_measured_is_not_offered_as_derivable():
    """A triangle is done. Nothing to construct, and nothing to suggest."""
    from stabilis.bench.graph import derivable_edges

    triangle = make_bench([("A", "B"), ("B", "C"), ("A", "C")], floor=0.0)
    assert derivable_edges(triangle) == ()
    assert check(triangle).derivable == ()
    assert check(triangle).identifiable


def test_a_back_to_back_run_derives_nothing():
    """Both ends on one clock says nothing about any *pair*, so it cannot
    stand in as the shared reference of a subtraction."""
    from stabilis.bench.graph import derivable_edges

    bench = make_bench([("A", "A"), ("B", "B")], floor=0.0)
    assert derivable_edges(bench) == ()


def _two_record_bench(seed=3, n=200_000):
    """Two records against a common reference, and the third by subtraction."""
    from stabilis.types import PhaseSeries

    rng = np.random.default_rng(seed)
    levels = {"A": 3e-12, "B": 1e-12, "C": 6e-12}
    x = {k: np.cumsum(rng.standard_normal(n)) * v for k, v in levels.items()}
    ac = x["A"] - x["C"]
    bc = x["B"] - x["C"]
    ab = ac - bc                      # constructed; no third counter exists
    np.testing.assert_allclose(ab, x["A"] - x["B"], rtol=0, atol=1e-24)

    bench = make_bench([("A", "C"), ("B", "C"), ("A", "B")], floor=0.0)
    results = {
        "m0": oadev(PhaseSeries(ac, rate_hz=1.0), taus="octave"),
        "m1": oadev(PhaseSeries(bc, rate_hz=1.0), taus="octave"),
        "m2": oadev(PhaseSeries(ab, rate_hz=1.0), taus="octave"),
    }
    return bench, results, levels


def test_the_derived_triangle_inverts_to_the_closed_form_exactly():
    """The hat from two records is an identity, and holds to machine precision.

    With the channel floors declared zero the 3x3 system has a closed-form
    inverse — `sigma2_C = (V_ac + V_bc - V_ab)/2` and its two rotations — so
    this needs no tolerance and no seed argument. It is the sharpest
    statement available that the derived edge is real information rather
    than a restatement of the two rows it came from: if it were dependent,
    the system would be singular and there would be nothing to invert.

    Asserted against the **unconstrained** solution, because that is the one
    the closed form describes. At the longest tau here the formula goes
    slightly negative — a quiet clock differenced against loud ones, with
    two degrees of freedom left — and NNLS is obliged to return zero there.
    Checking the constrained answer would therefore be testing the clip, not
    the algebra; the two agree wherever the physics permits, which is
    asserted separately below.
    """
    bench, results, _levels = _two_record_bench()
    solution = solve(bench, results)

    v_ac, v_bc, v_ab = (results[k].var for k in ("m0", "m1", "m2"))
    expected = {"A": (v_ac + v_ab - v_bc) / 2,
                "B": (v_bc + v_ab - v_ac) / 2,
                "C": (v_ac + v_bc - v_ab) / 2}
    for name, want in expected.items():
        np.testing.assert_allclose(solution.signed_var(name), want,
                                   rtol=1e-12, atol=0)

    # Where the closed form is physical for *every* clock, the constrained
    # answer is the closed form. The mask is over all three together rather
    # than per clock: NNLS is one joint solve, so pinning a negative source
    # at zero moves its partners at that same tau -- they are not
    # independent columns to be checked one at a time.
    physical = np.all([want >= 0.0 for want in expected.values()], axis=0)
    assert physical.any() and not physical.all()
    for name, want in expected.items():
        np.testing.assert_allclose(
            solution.source_var[physical, solution.sources.index(name)],
            want[physical], rtol=1e-12, atol=0)
    assert solution.any_negative                    # reported, not hidden


def test_the_hat_recovers_three_clocks_from_two_measured_records():
    """End to end, with the third row constructed rather than measured.

    White FM at three levels, so `sigma(tau) = level/sqrt(tau)` is the
    closed-form truth and no reference table is involved.

    **B gets a looser bound than A and C, and that is the physics rather
    than slack.** It is the quietest clock by a factor of three and six, so
    the hat determines it as a small difference of large numbers and its
    error is dominated by its partners' noise -- the effect measured and
    propagated in `theory/bench.md`, where the quiet source of a comparable
    triple came out at 53 % against a predicted 77 %. A single tolerance
    across all three would be scoring the choice of oscillators.
    """
    bench, results, levels = _two_record_bench()
    solution = solve(bench, results)

    for name, rtol in (("A", 0.03), ("C", 0.03), ("B", 0.10)):
        got = solution.dev(name)[:4]
        truth = levels[name] / np.sqrt(solution.tau_s[:4])
        np.testing.assert_allclose(got, truth, rtol=rtol)


def test_a_chain_is_a_tree_and_is_equally_hopeless():
    ident = check(make_bench([("A", "B"), ("B", "C")], floor=0.0))
    assert not ident.identifiable
    assert ident.bipartite == (True,)


def test_back_to_back_runs_pin_a_channel_and_separate_nothing():
    """A bench of only noise-floor runs measures the instrument, not the clocks.

    Well determined in its channel block, useless in its source block --
    and a real mistake people make, so the report says so in words.
    """
    bench = make_bench([("A", "A"), ("B", "B")])
    ident = check(bench)
    assert not ident.identifiable
    assert any("measures the instrument" in n for n in ident.notes)

    # And such a row contributes nothing to any source column.
    a, sources, _channels, _known = design_matrix(bench)
    assert np.all(a[:, : len(sources)] == 0.0)


def test_a_back_to_back_run_makes_the_triangle_work_without_assuming_a_floor():
    """The better experiment: measure the channel instead of assuming it."""
    bench = make_bench([("A", "B"), ("B", "C"), ("C", "A"), ("A", "A")])
    ident = check(bench)
    assert ident.identifiable
    assert ident.n_unknowns == 4, "three sources plus one free channel floor"


def test_check_reads_no_data_files():
    """The diagnostic has to run while the experiment is still being planned.

    Every measurement here names a file that does not exist, and `check`
    must not care -- if it did, it could not be used before data collection,
    which is the only time its answer is cheap to act on.
    """
    bench = Bench(
        sources=tuple(Source(id=s) for s in "ABC"),
        devices=(Device(id="D1", channels=(Channel(id="A-1", floor=0.0),)),),
        measurements=tuple(
            Measurement(id=f"m{i}", channel_id="A-1", source_a=a, source_b=b,
                        path="/nonexistent/nowhere.dat", sha256="0" * 64)
            for i, (a, b) in enumerate([("A", "B"), ("B", "C"), ("C", "A")])
        ),
    )
    assert check(bench).identifiable


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def three_oscillators(h_a, h_b, h_c, n=8192, seed=5):
    """Three independent white-FM sources at known levels, pairwise measured."""
    rng = np.random.default_rng(seed)
    y = {name: power_law_freq(n, 0, rate_hz=1.0, rng=rng, h_alpha=h).y
         for name, h in (("A", h_a), ("B", h_b), ("C", h_c))}
    results = {}
    for mid, (first, second) in zip(
            ("m0", "m1", "m2"), (("A", "B"), ("B", "C"), ("C", "A"))):
        diff = FreqSeries(y[first] - y[second], rate_hz=1.0)
        results[mid] = oadev(diff, taus="octave")
    return results


def test_the_hat_recovers_three_known_oscillators():
    """End to end, with an ideal noiseless channel.

    Each source's variance is `h/(2 tau)`, so the truth is known in closed
    form rather than assumed from the fit. Compared at short tau where the
    estimate has real degrees of freedom.
    """
    h = {"A": 4.0e-24, "B": 1.0e-24, "C": 9.0e-24}
    results = three_oscillators(h["A"], h["B"], h["C"])
    bench = make_bench([("A", "B"), ("B", "C"), ("C", "A")], floor=0.0)
    sol = solve(bench, results)

    assert sol.sources == ("A", "B", "C")
    assert sol.channels == ()
    for name in "ABC":
        want = h[name] / (2.0 * sol.tau_s)
        got = sol.source_var[:, sol.sources.index(name)]
        near = sol.tau_s <= 64.0
        assert np.allclose(got[near], want[near], rtol=0.35), name


def test_the_auto_variance_cannot_do_what_the_hat_does():
    """The point of the whole exercise, stated as a comparison.

    A single pair measurement sees `h_A + h_B` and cannot say which is
    which. The hat separates them, and this asserts the separation is real
    rather than a coincidence of the numbers chosen.
    """
    h = {"A": 4.0e-24, "B": 1.0e-24, "C": 9.0e-24}
    results = three_oscillators(h["A"], h["B"], h["C"])
    bench = make_bench([("A", "B"), ("B", "C"), ("C", "A")], floor=0.0)
    sol = solve(bench, results)

    pair_var = results["m0"].var          # A against B, the raw measurement
    a_only = sol.source_var[:, 0]
    near = sol.tau_s <= 64.0
    # The raw pair is the sum; the hat's A alone is the smaller part of it.
    assert np.all(a_only[near] < pair_var[near])
    assert np.allclose(pair_var[near],
                       (sol.source_var[:, 0] + sol.source_var[:, 1])[near],
                       rtol=0.05)


def test_a_channel_floor_is_recovered_from_a_back_to_back_run():
    """Noise-floor subtraction, which is the same solve with one more row."""
    rng = np.random.default_rng(11)
    n, h_src, h_floor = 8192, 4.0e-24, 1.0e-24
    y = {s: power_law_freq(n, 0, rate_hz=1.0, rng=rng, h_alpha=h_src).y
         for s in "ABC"}
    floor = {k: power_law_freq(n, 0, rate_hz=1.0, rng=rng, h_alpha=h_floor).y
             for k in range(4)}

    results = {}
    for k, (mid, (first, second)) in enumerate(zip(
            ("m0", "m1", "m2"), (("A", "B"), ("B", "C"), ("C", "A")))):
        series = FreqSeries(y[first] - y[second] + floor[k], rate_hz=1.0)
        results[mid] = oadev(series, taus="octave")
    results["m3"] = oadev(FreqSeries(floor[3], rate_hz=1.0), taus="octave")

    bench = make_bench([("A", "B"), ("B", "C"), ("C", "A"), ("A", "A")])
    sol = solve(bench, results)
    assert sol.channels == ("A-1",)
    near = sol.tau_s <= 64.0
    assert np.allclose(sol.channel_var[near, 0],
                       (h_floor / (2.0 * sol.tau_s))[near], rtol=0.4)
    assert np.allclose(sol.source_var[near, 0],
                       (h_src / (2.0 * sol.tau_s))[near], rtol=0.5)


def test_an_unidentifiable_bench_is_refused_with_the_fix_in_the_message():
    results = three_oscillators(1e-24, 1e-24, 1e-24)
    ring = Bench(
        sources=tuple(Source(id=s) for s in "ABCD"),
        devices=(Device(id="D1", channels=(Channel(id="A-1", floor=0.0),)),),
        measurements=tuple(
            Measurement(id=f"m{i}", channel_id="A-1", source_a=a, source_b=b)
            for i, (a, b) in enumerate(
                [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
        ),
    )
    with pytest.raises(ValueError, match="cannot determine its unknowns"):
        solve(ring, {"m0": results["m0"], "m1": results["m1"],
                     "m2": results["m2"], "m3": results["m0"]})


def test_negative_variances_are_reported_signed_not_clipped():
    """The most useful diagnostic the method produces, kept intact.

    Two sources are made deliberately correlated -- B is A plus a little
    extra -- so their difference is far quieter than independence predicts
    and the hat is pushed to a negative variance. The constrained answer is
    obliged to be non-negative; the signed one says what really happened.
    """
    rng = np.random.default_rng(3)
    n = 4096
    a = power_law_freq(n, 0, rate_hz=1.0, rng=rng, h_alpha=9.0e-24).y
    b = a + power_law_freq(n, 0, rate_hz=1.0, rng=rng, h_alpha=1.0e-25).y
    c = power_law_freq(n, 0, rate_hz=1.0, rng=rng, h_alpha=9.0e-24).y

    results = {
        "m0": oadev(FreqSeries(a - b, rate_hz=1.0), taus="octave"),
        "m1": oadev(FreqSeries(b - c, rate_hz=1.0), taus="octave"),
        "m2": oadev(FreqSeries(c - a, rate_hz=1.0), taus="octave"),
    }
    sol = solve(make_bench([("A", "B"), ("B", "C"), ("C", "A")], floor=0.0),
                results)
    assert sol.any_negative, "correlated sources must show up as negative"
    assert np.all(sol.source_var >= 0.0), "the constrained answer stays physical"
    assert any(s.action == "negative_variance" for s in sol.provenance.steps)


def test_non_variance_statistics_are_refused():
    """The sum rule holds for variances, and these are not variances.

    Keyed on the `is_variance` flag rather than the name, so an estimator
    this module has never heard of is caught too -- hence the invented
    kind here. The message gives the reason *for each* of the two the
    library ships: they fail differently, and calling both "extreme-value
    statistics" would be wrong about TIE rms.
    """
    from dataclasses import replace

    results = three_oscillators(1e-24, 1e-24, 1e-24)
    disguised = {k: replace(v, kind="peak-to-peak-thing", is_variance=False)
                 for k, v in results.items()}
    with pytest.raises(ValueError) as exc:
        solve(make_bench([("A", "B"), ("B", "C"), ("C", "A")], floor=0.0),
              disguised)
    assert "maximum of a sum" in str(exc.value)
    assert "frequency offset" in str(exc.value)


def test_mixing_estimators_in_one_solve_is_refused():
    from dataclasses import replace

    results = three_oscillators(1e-24, 1e-24, 1e-24)
    results["m1"] = replace(results["m1"], kind="mdev")
    with pytest.raises(ValueError, match="mix estimators"):
        solve(make_bench([("A", "B"), ("B", "C"), ("C", "A")], floor=0.0),
              results)


def test_records_with_no_common_tau_are_refused_rather_than_interpolated():
    """An interpolated sigma(tau) is not a measurement."""
    rng = np.random.default_rng(2)
    fast = oadev(power_law_freq(4096, 0, rate_hz=1000.0, rng=rng), taus="octave")
    slow = oadev(power_law_freq(4096, 0, rate_hz=1.0, rng=rng), taus="octave")
    results = {"m0": fast, "m1": slow, "m2": fast}
    with pytest.raises(ValueError, match="share no averaging time"):
        solve(make_bench([("A", "B"), ("B", "C"), ("C", "A")], floor=0.0),
              results)


# ---------------------------------------------------------------------------
# Correlation warnings
# ---------------------------------------------------------------------------


def test_sharing_one_channel_in_turn_is_normal_and_said_to_be():
    """Measuring three pairs through one channel sequentially is standard.

    An earlier version called this "maximally correlated", which is wrong:
    that is only true of records taken at the same time, and one channel
    cannot make two simultaneous measurements. What is true, and what the
    warning now says, is that the channel's noise is common to every row
    and cannot be attributed to any one of them.
    """
    bench = make_bench([("A", "B"), ("B", "C"), ("C", "A")], floor=0.0)
    texts = " ".join(common_mode_warnings(bench))
    assert "normal practice" in texts
    assert "maximally correlated" not in texts


def test_the_same_channel_at_the_same_epoch_is_one_record_described_twice():
    """That case *is* a real duplicate, and is distinguished from the above."""
    bench = Bench(
        sources=(Source(id="A"), Source(id="B"), Source(id="C")),
        devices=(Device(id="D1", channels=(Channel(id="A-1", floor=0.0),)),),
        measurements=(
            Measurement(id="m0", channel_id="A-1", source_a="A", source_b="B",
                        epoch="2026-08-08T00:00:00Z"),
            Measurement(id="m1", channel_id="A-1", source_a="B", source_b="C",
                        epoch="2026-08-08T00:00:00Z"),
        ),
    )
    texts = " ".join(common_mode_warnings(bench))
    assert "same record described twice" in texts


def test_two_channels_of_one_device_is_warned_about():
    """The dangerous case: shared timebase, and the answer still looks fine."""
    bench = make_bench([("A", "B"), ("B", "C"), ("C", "A")], floor=0.0,
                       channels=["A-1", "A-2", "A-3"])
    texts = " ".join(common_mode_warnings(bench))
    assert "ONE instrument" in texts
    assert "does **not** cancel" in texts


def test_independent_devices_draw_no_common_mode_warning():
    bench = make_bench(
        [("A", "B"), ("B", "C"), ("C", "A")], floor=0.0,
        channels=["A-1", "B-1", "C-1"],
        devices={"D1": {"A-1"}, "D2": {"B-1"}, "D3": {"C-1"}},
    )
    texts = " ".join(common_mode_warnings(bench))
    assert "ONE instrument" not in texts


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


def test_a_measurement_naming_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="is not a source"):
        Bench(
            sources=(Source(id="A"),),
            devices=(Device(id="D1", channels=(Channel(id="A-1"),)),),
            measurements=(Measurement(id="m0", channel_id="A-1",
                                      source_a="A", source_b="Z"),),
        )


def test_a_measurement_naming_an_unknown_channel_is_refused():
    with pytest.raises(ValueError, match="is not a channel"):
        Bench(
            sources=(Source(id="A"),),
            devices=(Device(id="D1", channels=(Channel(id="A-1"),)),),
            measurements=(Measurement(id="m0", channel_id="B-9",
                                      source_a="A", source_b="A"),),
        )


def test_duplicate_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate source id"):
        Bench(sources=(Source(id="A"), Source(id="A")))


def test_a_negative_channel_floor_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        Channel(id="A-1", floor=-1.0)


# ---------------------------------------------------------------------------
# Persistence, and the independence discipline
# ---------------------------------------------------------------------------


def test_bench_round_trips_through_json(tmp_path):
    from stabilis.bench import store

    bench = Bench(
        sources=(Source(id="A", name="Maser 1", kind="hmaser",
                        nominal_hz=1e7, spec_adev=((1.0, 1e-13), (10.0, 3e-14)),
                        spec_floor=2e-16),
                 Source(id="B", name="OCXO")),
        devices=(Device(id="D1", model="53230A", serial="MY123",
                        channels=(Channel(id="A-1", label="chan 1", floor=None),
                                  Channel(id="A-2", floor=1e-26))),),
        measurements=(Measurement(id="m0", channel_id="A-1", source_a="A",
                                  source_b="B", path="ab.dat", kind="phase",
                                  tau0_s=1.0, reason="nightly"),),
        notes="test bench",
    )
    written = store.save(bench, tmp_path / "bench.json")
    back = store.load(tmp_path / "bench.json")
    assert back.sources[0].spec_adev == ((1.0, 1e-13), (10.0, 3e-14))
    assert back.devices[0].channels[1].floor == 1e-26
    assert back.measurements[0].tau0_s == 1.0
    assert back.notes == "test bench"
    # Equality ignores `path`, which is where it was loaded from rather than
    # part of what a bench is. `save` returns the bench as written, because
    # it stamps `created` -- so that is what the round trip is against.
    assert back == written
    assert written.created and not bench.created

    # And a stamp-free save is byte-reproducible, which is what fixtures
    # and provenance hashing need.
    a = store.save(bench, tmp_path / "a.json", stamp_created=False)
    b = store.save(bench, tmp_path / "b.json", stamp_created=False)
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert a.created == b.created == ""


def test_an_unknown_schema_is_refused_rather_than_guessed(tmp_path):
    from stabilis.bench import store

    p = tmp_path / "bench.json"
    p.write_text('{"schema": 99, "sources": [], "devices": [], '
                 '"measurements": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="not readable by this version"):
        store.load(p)


def test_changed_data_is_refused_when_hashes_are_verified(tmp_path):
    """A measurement whose bytes changed is a different measurement."""
    from stabilis.provenance import file_sha256
    from stabilis.bench import store

    data = tmp_path / "ab.dat"
    data.write_text("1.0\n2.0\n3.0\n", encoding="ascii")
    bench = Bench(
        sources=(Source(id="A"), Source(id="B")),
        devices=(Device(id="D1", channels=(Channel(id="A-1"),)),),
        measurements=(Measurement(id="m0", channel_id="A-1", source_a="A",
                                  source_b="B", path="ab.dat",
                                  sha256=file_sha256(data)),),
    )
    store.save(bench, tmp_path / "bench.json")
    path = tmp_path / "bench.json"
    assert store.verify(store.load(path)) == []
    store.load(path, verify_hashes=True)          # passes

    data.write_text("1.0\n2.0\n4.0\n", encoding="ascii")
    with pytest.raises(store.HashMismatch, match="does not match"):
        store.load(path, verify_hashes=True)


def test_an_unhashed_measurement_is_skipped_not_called_verified(tmp_path):
    """Absence of a hash is not agreement with one."""
    from stabilis.bench import store

    bench = Bench(
        sources=(Source(id="A"),),
        devices=(Device(id="D1", channels=(Channel(id="A-1"),)),),
        measurements=(Measurement(id="m0", channel_id="A-1", source_a="A",
                                  source_b="A", path="missing.dat"),),
    )
    store.save(bench, tmp_path / "bench.json")
    assert store.verify(store.load(tmp_path / "bench.json")) == []


def test_stats_never_imports_bench():
    """The same independence discipline the certified oracle has.

    The estimators must stay usable, and testable, with no notion of where
    their data came from. If `stats/` ever reaches into `bench/`, the bench
    model stops being an optional layer and becomes a dependency of every
    ADEV anyone computes.
    """
    import ast
    import inspect
    import pkgutil
    import importlib

    import stabilis.stats as stats

    checked = 0
    for info in pkgutil.iter_modules(stats.__path__):
        module = importlib.import_module(f"stabilis.stats.{info.name}")
        tree = ast.parse(inspect.getsource(module))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(f"{node.module or ''}.{a.name}" for a in node.names)
        offenders = [n for n in names if "bench" in n]
        assert not offenders, f"stats.{info.name} imports bench: {offenders}"
        checked += 1
    assert checked >= 5, "expected to have scanned the whole stats package"
