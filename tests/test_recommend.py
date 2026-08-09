"""Which technique a bench supports, and the correction behind it.

The load-bearing test is the classical three-cornered hat: three sources
through one channel is four unknowns against three rows and the textbook
fix is to assert that the channel is silent. Measure one of the three
pairs again through an *independent* channel and the same system closes
having measured that instead. `would_identify` is the assertion that the
substitution works.

The second is the correction. Two channels of one instrument must be
reported as *capped*, not clean, and must not close the system -- they
share a timebase, an LO, a supply and an oven, and that part passes
straight through a cross-variance. A version of this library said the
opposite in a warning string.
"""

import pytest

from stabilis.bench import (Bench, Channel, Device, Measurement, Source,
                            check, cross_opportunities, recommend)

TRIANGLE = (("A", "B"), ("B", "C"), ("C", "A"))


def triangle_bench(devices, extra=()):
    """Three sources measured pairwise, plus whatever `extra` rows are given."""
    rows = tuple(
        Measurement(id=f"m{i}", channel_id="A-1", source_a=a, source_b=b)
        for i, (a, b) in enumerate(TRIANGLE)
    )
    return Bench(sources=tuple(Source(id=s) for s in "ABC"),
                 devices=devices, measurements=rows + tuple(extra))


ONE_CHANNEL = (Device(id="D1", channels=(Channel(id="A-1"),)),)
TWO_CHANNELS = (Device(id="D1", channels=(Channel(id="A-1"),
                                          Channel(id="A-2"))),)
TWO_DEVICES = (Device(id="D1", channels=(Channel(id="A-1"),)),
               Device(id="D2", channels=(Channel(id="B-1"),)))


def test_the_classical_hat_alone_does_not_close():
    """Four unknowns, three rows. The premise of everything below."""
    bench = triangle_bench(ONE_CHANNEL)
    rec = recommend(bench)
    assert not rec.hat_identifiable
    assert rec.hat_deficiency == 1
    assert not rec.has_cross
    assert not rec.would_identify
    # And it says what the options are rather than only that it failed.
    assert any("declare the floor" in n for n in rec.notes)
    assert any("back-to-back" in n for n in rec.notes)


def test_a_repeated_edge_on_a_second_instrument_closes_it():
    """The measured substitute for the assumption the textbook makes."""
    extra = (Measurement(id="m3", channel_id="B-1", source_a="A", source_b="B"),)
    rec = recommend(triangle_bench(TWO_DEVICES, extra))
    assert not rec.hat_identifiable, "the hat alone still cannot"
    assert rec.would_identify
    assert [o.quality for o in rec.opportunities] == ["clean"]
    assert rec.channels_determined == ("A-1", "B-1")
    assert any("three-cornered hat's missing row" in n for n in rec.notes)


def test_declaring_the_floor_closes_the_same_system():
    """The two routes are equivalent; only one of them measured anything.

    Asserted so the claim in `theory/bench.md` -- that a repeated edge
    substitutes for the declaration -- is checked rather than stated. Both
    reach rank 3 over three sources.
    """
    declared = (Device(id="D1", channels=(Channel(id="A-1", floor=0.0),)),)
    assert check(triangle_bench(declared)).identifiable
    extra = (Measurement(id="m3", channel_id="B-1", source_a="A", source_b="B"),)
    assert recommend(triangle_bench(TWO_DEVICES, extra)).would_identify


def test_two_channels_of_one_instrument_are_capped_not_clean():
    """The correction. Common mode passes straight through a cross-variance.

    The opportunity is still reported -- it is a real configuration and it
    does reject the independent part -- but it is not clean, it does not
    close the system, and the note says why in terms of the hardware.
    """
    extra = (Measurement(id="m3", channel_id="A-2", source_a="A", source_b="B"),)
    rec = recommend(triangle_bench(TWO_CHANNELS, extra))
    assert rec.has_cross
    assert [o.quality for o in rec.opportunities] == ["common-mode"]
    assert not rec.opportunities[0].clean
    assert not rec.would_identify, "common mode does not remove the channel term"
    assert rec.channels_determined == ()
    note = " ".join(rec.notes)
    assert "one timebase" in note and "one instrument" in note
    assert "Two instruments would reject all of it" in note


def test_two_records_through_one_channel_are_not_an_opportunity():
    """One channel cannot make two simultaneous measurements."""
    extra = (Measurement(id="m3", channel_id="A-1", source_a="A", source_b="B"),)
    assert cross_opportunities(triangle_bench(ONE_CHANNEL, extra)) == ()


def test_a_back_to_back_pair_on_two_devices_is_an_opportunity():
    """The purest form: both ends on one source, seen two ways."""
    rows = (Measurement(id="b1", channel_id="A-1", source_a="A", source_b="A"),
            Measurement(id="b2", channel_id="B-1", source_a="A", source_b="A"))
    bench = Bench(sources=(Source(id="A"),), devices=TWO_DEVICES,
                  measurements=rows)
    (opportunity,) = cross_opportunities(bench)
    assert opportunity.source_a == opportunity.source_b == "A"
    assert opportunity.clean
    assert opportunity.measurements == ("b1", "b2")


def test_an_identifiable_bench_with_cross_is_told_to_run_both():
    """They answer different questions, so a disagreement is informative."""
    declared = (Device(id="D1", channels=(Channel(id="A-1", floor=0.0),)),
                Device(id="D2", channels=(Channel(id="B-1", floor=0.0),)))
    extra = (Measurement(id="m3", channel_id="B-1", source_a="A", source_b="B"),)
    rec = recommend(triangle_bench(declared, extra))
    assert rec.hat_identifiable and rec.has_cross
    assert not rec.would_identify, "nothing to close, so nothing is claimed"
    assert any("run both and compare" in n for n in rec.notes)


def test_the_report_names_the_measurements_to_cross():
    """Advice that does not name the rows is not actionable."""
    extra = (Measurement(id="m3", channel_id="B-1", source_a="A", source_b="B"),)
    text = recommend(triangle_bench(TWO_DEVICES, extra)).report()
    assert "A vs B via m0, m3" in text
    assert "independent instruments" in text
    assert "N-cornered hat: no" in text


def test_the_hypothetical_bench_never_leaks_into_the_real_one():
    """`would_identify` adds a synthetic channel; the caller must not see it."""
    extra = (Measurement(id="m3", channel_id="B-1", source_a="A", source_b="B"),)
    bench = triangle_bench(TWO_DEVICES, extra)
    before = (len(bench.devices), len(bench.measurements))
    rec = recommend(bench)
    assert rec.would_identify
    assert (len(bench.devices), len(bench.measurements)) == before
    assert all("__cross__" not in c for c in rec.channels_determined)


def test_a_four_clock_ring_is_not_rescued_by_cross_correlation():
    """Cross-correlation removes channel terms, never a source degeneracy.

    A ring of four is bipartite: only sums across the split are ever
    determined, however many instruments watch it. The recommendation must
    not imply otherwise -- that is precisely the confusion the two-by-two
    exists to prevent.
    """
    devices = (Device(id="D1", channels=(Channel(id="A-1"),)),
               Device(id="D2", channels=(Channel(id="B-1"),)))
    ring = (("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"))
    rows = tuple(
        Measurement(id=f"m{i}", channel_id="A-1", source_a=a, source_b=b)
        for i, (a, b) in enumerate(ring)
    ) + (Measurement(id="x", channel_id="B-1", source_a="A", source_b="B"),)
    bench = Bench(sources=tuple(Source(id=s) for s in "ABCD"),
                  devices=devices, measurements=rows)
    rec = recommend(bench)
    assert rec.has_cross and rec.opportunities[0].clean
    assert not rec.would_identify


@pytest.mark.parametrize("bench_devices,extra,status", [
    (ONE_CHANNEL, (), 2),
    (TWO_DEVICES,
     (Measurement(id="m3", channel_id="B-1", source_a="A", source_b="B"),), 0),
    (TWO_CHANNELS,
     (Measurement(id="m3", channel_id="A-2", source_a="A", source_b="B"),), 2),
])
def test_cli_bench_recommend_exit_status(tmp_path, capsys, bench_devices,
                                         extra, status):
    """0 when something determines the sources, 2 when nothing does."""
    from stabilis.bench import save
    from stabilis.cli import main

    path = tmp_path / "bench.json"
    save(triangle_bench(bench_devices, extra), path)
    assert main(["bench", "recommend", str(path)]) == status
    assert "technique:" in capsys.readouterr().out


def test_cli_bench_recommend_json(tmp_path, capsys):
    import json

    from stabilis.bench import save
    from stabilis.cli import main

    path = tmp_path / "bench.json"
    extra = (Measurement(id="m3", channel_id="B-1", source_a="A", source_b="B"),)
    save(triangle_bench(TWO_DEVICES, extra), path)
    assert main(["bench", "recommend", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["would_identify"] is True
    assert payload["hat_identifiable"] is False
    assert payload["opportunities"][0]["quality"] == "clean"
    assert payload["channels_determined"] == ["A-1", "B-1"]
