"""Realtime ingest and the online Allan variance.

Three kinds of check here.

*The identity*, which is what this whole subsystem is validated by:
replaying a record through the streaming path must return what the batch
estimator returns. The coefficients are bit-identical by construction (the
association order is copied from the batch expression), so what is
measured is the summation difference alone -- one ulp, and flat in `N`.
A test asserts both the tightness and the flatness, because the second is
the evidence that the Neumaier compensation is doing anything.

*The gap rule*, which is the one that would be silently wrong. A stream
that omits its dropped samples entirely -- exactly what a counter does --
must rebuild the original record NaN for NaN. Appending arriving values to
a list instead passes every other test in this file and fails this one,
which is why it is written as an equality against the original array and
not as a count.

*Refusals*, one per way a timebase can be quietly corrupted: a rate that
does not match the timestamps, a sample arriving out of order, a duplicate,
and an outage too long to materialize.
"""

import socket
import threading

import numpy as np
import pytest

from stabilis.stats import oadev
from stabilis.stream import (
    GRID_TOLERANCE,
    collect,
    feed,
    grid_slot,
    on_grid,
    online_avar,
    online_push,
    online_push_many,
    online_result,
    parse_lines,
    replay,
    tail_file,
    tcp_lines,
    udp_lines,
)
from stabilis.stream.online import MAX_RING_BYTES
from stabilis.synth import power_law_phase
from stabilis.types import PhaseSeries

RATE = 1.0
H = 1e-24


def record(n, seed, alpha=0):
    return power_law_phase(n, alpha, RATE, np.random.default_rng(seed), H)


# ---------------------------------------------------------------------------
# The identity against the batch estimator
# ---------------------------------------------------------------------------

def test_replayed_record_matches_the_batch_estimator():
    ps = record(4095, 3)
    batch = oadev(ps, taus="octave")
    acc = online_avar(RATE, batch.af)
    online_push_many(acc, ps.x)
    got = online_result(acc)

    assert np.array_equal(got.af, batch.af)
    assert np.array_equal(got.n_terms, batch.n_terms)
    assert got.var == pytest.approx(batch.var, rel=2e-15)


def test_the_disagreement_is_one_ulp_and_does_not_grow_with_n():
    """The compensation's whole point: O(1) ulp, not O(N).

    A plain sequential sum measures 15 ulp at N=4096 and 74 at N=65536 on
    these records (theory/stream.md tabulates it), so a regression that
    dropped the compensation would fail this by an order of magnitude at
    the longer length while still passing at the shorter one. Both are
    checked for that reason.
    """
    worst = {}
    for n in (1023, 16383):
        ps = record(n, 5)
        batch = oadev(ps, taus="octave")
        acc = online_avar(RATE, batch.af)
        online_push_many(acc, ps.x)
        rel = np.abs(online_result(acc).var - batch.var) / np.abs(batch.var)
        worst[ps.n] = rel.max() / np.finfo(float).eps
    assert worst[1024] < 4.0
    assert worst[16384] < 4.0


def test_a_gapped_record_streams_to_the_same_answer():
    """Gap handling is not a parallel rule; it is the same `isfinite` drop."""
    x = np.array(record(2047, 8).x)
    x[100:105] = np.nan
    x[1500] = np.nan
    series = PhaseSeries(x=x, rate_hz=RATE)
    batch = oadev(series, taus="octave")

    acc = online_avar(RATE, batch.af)
    online_push_many(acc, x)
    got = online_result(acc)

    assert np.array_equal(got.n_terms, batch.n_terms)
    assert got.dropped_terms == batch.dropped_terms
    assert got.var == pytest.approx(batch.var, rel=2e-15)


def test_chunking_the_feed_cannot_change_the_answer():
    ps = record(1023, 12)
    afs = [1, 2, 4, 8, 16]
    one = online_avar(RATE, afs)
    online_push_many(one, ps.x)

    chunked = online_avar(RATE, afs)
    for start in range(0, ps.n, 37):
        online_push_many(chunked, ps.x[start:start + 37])

    assert np.array_equal(online_result(one).var, online_result(chunked).var)


def test_a_result_before_the_first_complete_term_is_refused():
    acc = online_avar(RATE, [4])
    online_push_many(acc, np.zeros(5))
    with pytest.raises(ValueError, match="no complete term yet"):
        online_result(acc)


def test_an_online_result_carries_no_coefficients_so_confidence_refuses_it():
    from stabilis.stats import confidence

    ps = record(511, 2)
    acc = online_avar(RATE, [1, 2, 4])
    online_push_many(acc, ps.x)
    with pytest.raises(ValueError, match="keep_coeffs=False"):
        confidence(online_result(acc))


def test_the_history_a_stream_would_need_is_bounded():
    too_big = MAX_RING_BYTES // 8
    with pytest.raises(MemoryError, match="phase history"):
        online_avar(RATE, [too_big])


def test_averaging_factors_are_required_and_validated():
    with pytest.raises(ValueError, match="no averaging factors"):
        online_avar(RATE, [])
    with pytest.raises(ValueError, match=">= 1"):
        online_avar(RATE, [0, 4])
    with pytest.raises(ValueError, match="rate_hz"):
        online_avar(0.0, [1])


# ---------------------------------------------------------------------------
# The gap rule
# ---------------------------------------------------------------------------

def test_a_stream_that_omits_its_dropped_samples_rebuilds_the_record():
    """The one that catches the natural wrong implementation.

    `replay(..., skip_gaps=True)` emits nothing at all for a NaN, which is
    what a counter does when it fails to produce a reading. Appending
    arriving values to a list would return a record five samples shorter
    with every later sample slid earlier -- and perfectly smooth.
    """
    x = np.array(record(1023, 21).x)
    x[300:305] = np.nan
    x[700] = np.nan

    got = collect(replay(x, RATE, skip_gaps=True), RATE)

    assert got.n == x.size
    assert got.n_gaps == 6
    assert np.array_equal(np.isnan(got.x), np.isnan(x))
    assert np.array_equal(got.x[~np.isnan(x)], x[~np.isnan(x)])


def test_the_reconstructed_record_gives_the_gapped_batch_answer():
    x = np.array(record(2047, 22).x)
    x[900:906] = np.nan
    reference = oadev(PhaseSeries(x=x, rate_hz=RATE), taus="octave")

    got = collect(replay(x, RATE, skip_gaps=True), RATE)
    assert np.array_equal(np.asarray(got.x), x, equal_nan=True)
    assert oadev(got, taus="octave").var == pytest.approx(reference.var, rel=0, abs=0)


def test_feeding_a_gapped_stream_matches_the_gapped_batch_estimator():
    x = np.array(record(2047, 23).x)
    x[1200:1204] = np.nan
    batch = oadev(PhaseSeries(x=x, rate_hz=RATE), taus="octave")

    acc = online_avar(RATE, batch.af)
    feed(acc, replay(x, RATE, skip_gaps=True))
    got = online_result(acc)

    assert acc.n_seen == x.size
    assert np.array_equal(got.n_terms, batch.n_terms)
    assert got.var == pytest.approx(batch.var, rel=2e-15)


def test_a_feed_resumes_where_the_last_one_stopped():
    """Two streams in sequence must equal one uninterrupted stream."""
    ps = record(1023, 24)
    afs = [1, 2, 4, 8]
    whole = online_avar(RATE, afs)
    feed(whole, replay(ps.x, RATE))

    split = online_avar(RATE, afs)
    feed(split, replay(ps.x[:600], RATE))
    feed(split, ((600 / RATE + i / RATE, float(v))
                 for i, v in enumerate(ps.x[600:])))

    assert split.n_seen == whole.n_seen
    assert np.array_equal(online_result(split).var, online_result(whole).var)


def test_an_epoch_earlier_than_the_first_sample_leaves_leading_gaps():
    ps = record(255, 25)
    got = collect(replay(ps.x, RATE, epoch=10.0), RATE, epoch=0.0)
    assert got.n == ps.n + 10
    assert np.isnan(got.x[:10]).all()
    assert got.x[10] == ps.x[0]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_rate_that_does_not_match_the_timestamps_is_refused():
    ps = record(255, 31)
    with pytest.raises(ValueError, match="samples from grid slot"):
        collect(replay(ps.x, RATE), rate_hz=1.7)


def test_a_uniform_offset_is_an_epoch_rather_than_jitter():
    """Every sample 0.2 of an interval late is not a defect; it is `epoch`.

    `collect` takes the epoch from the first timestamp by default, so a
    constant shift is absorbed and nothing lands off grid. Reporting that
    as jitter would cry wolf on every record whose first sample is not at
    t = 0, which is all of them.
    """
    ps = record(255, 32)
    shifted = ((t + 0.2 / RATE, v) for t, v in replay(ps.x, RATE))
    got = collect(shifted, RATE)
    assert got.n == ps.n
    assert got.epoch == pytest.approx(0.2)
    step = next(s for s in got.provenance.steps if s.action == "collect")
    assert step.params["worst_grid_offset_samples"] == 0.0


def test_jitter_inside_the_tolerance_is_accepted_and_reported():
    ps = record(255, 32)
    jitter = np.where(np.arange(ps.n) % 2, 0.2, -0.1) / RATE
    nudged = ((t + jitter[i], v) for i, (t, v) in enumerate(replay(ps.x, RATE)))
    got = collect(nudged, RATE, epoch=0.0)
    assert got.n == ps.n
    step = next(s for s in got.provenance.steps if s.action == "collect")
    assert step.params["worst_grid_offset_samples"] == pytest.approx(0.2, abs=1e-6)
    assert GRID_TOLERANCE == 0.25


def test_an_out_of_order_sample_is_refused_rather_than_sorted():
    pairs = [(0.0, 1.0), (1.0, 2.0), (3.0, 3.0), (2.0, 4.0)]
    with pytest.raises(ValueError, match="at or before slot"):
        collect(iter(pairs), RATE)


def test_a_duplicate_timestamp_is_refused():
    pairs = [(0.0, 1.0), (1.0, 2.0), (1.0, 3.0)]
    with pytest.raises(ValueError, match="at or before slot"):
        collect(iter(pairs), RATE)


def test_an_outage_too_long_to_materialize_is_refused():
    pairs = [(0.0, 1.0), (5e6, 2.0)]
    with pytest.raises(ValueError, match="outage rather than a gap"):
        collect(iter(pairs), RATE)


def test_an_empty_stream_says_so():
    with pytest.raises(ValueError, match="no samples"):
        collect(iter(()), RATE)


def test_grid_slot_reports_the_offset_that_placement_destroys():
    index, offset = grid_slot(t=10.2, epoch=0.0, rate_hz=1.0)
    assert index == 10
    assert offset == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Parsing and the missing-timestamp honesty rule
# ---------------------------------------------------------------------------

def test_parse_lines_reads_a_timetag_column():
    lines = ["# header", "", "0.0 1.5", "1.0 2.5", "junk", "2.0 3.5"]
    got = list(parse_lines(lines, column=1, time_column=0))
    assert got == [(0.0, 1.5), (1.0, 2.5), (2.0, 3.5)]


def test_without_a_time_column_timestamps_come_from_arrival_order():
    lines = ["1.5", "2.5", "3.5"]
    got = list(parse_lines(lines, rate_hz=2.0))
    assert got == [(0.0, 1.5), (0.5, 2.5), (1.0, 3.5)]


def test_synthesized_timestamps_are_recorded_as_such():
    """A dropped sample is undetectable without a timetag, and it says so."""
    got = collect(replay(np.arange(10.0), RATE), RATE, timestamps=False)
    step = next(s for s in got.provenance.steps if s.action == "collect")
    assert step.params["timestamps"] == "synthesized"
    assert "floor rather than a count" in step.outcome


def test_parse_lines_needs_a_rate_when_it_must_invent_the_time():
    with pytest.raises(ValueError, match="time_column"):
        list(parse_lines(["1.0"]))


def test_a_gap_value_becomes_a_gap():
    got = list(parse_lines(["1.0", "0.0", "3.0"], rate_hz=1.0, gap_value=0.0))
    assert np.isnan(got[1][1])


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def test_tail_file_reads_what_is_there_then_stops_at_the_idle_timeout(tmp_path):
    path = tmp_path / "live.dat"
    path.write_text("1.0\n2.0\n3.0\n")
    assert list(tail_file(path, poll_s=0.01, timeout_s=0.05)) == ["1.0", "2.0", "3.0"]


def test_tail_file_follows_an_append_and_never_yields_half_a_line(tmp_path):
    """A read landing mid-line must wait for the rest of it."""
    path = tmp_path / "live.dat"
    path.write_text("1.0\n")

    def writer():
        with open(path, "a") as fh:
            fh.write("2.")
            fh.flush()
            fh.write("0\n3.0\n")
            fh.flush()

    thread = threading.Thread(target=writer)
    thread.start()
    got = list(tail_file(path, poll_s=0.01, timeout_s=0.3))
    thread.join()
    assert got == ["1.0", "2.0", "3.0"]


def test_tail_file_from_start_false_skips_what_was_already_there(tmp_path):
    path = tmp_path / "live.dat"
    path.write_text("1.0\n2.0\n")
    assert list(tail_file(path, poll_s=0.01, timeout_s=0.05,
                          from_start=False)) == []


def test_a_replayed_file_streams_to_the_batch_answer(tmp_path):
    """The end-to-end path: text on disk, parsed, gridded, accumulated."""
    ps = record(1023, 41)
    path = tmp_path / "run.dat"
    path.write_text("".join(f"{i} {v:.17e}\n" for i, v in enumerate(ps.x)))
    batch = oadev(ps, taus="octave")

    acc = online_avar(RATE, batch.af)
    with open(path) as fh:
        feed(acc, parse_lines(fh, column=1, time_column=0))

    assert online_result(acc).var == pytest.approx(batch.var, rel=2e-15)


def test_tcp_lines_reads_a_loopback_stream():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        conn, _ = server.accept()
        conn.sendall(b"1.0\n2.")
        conn.sendall(b"0\n3.0\n")
        conn.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        assert list(tcp_lines("127.0.0.1", port, timeout_s=2.0)) == \
            ["1.0", "2.0", "3.0"]
    finally:
        thread.join()
        server.close()


def test_udp_lines_reads_a_loopback_datagram():
    ready = threading.Event()
    got = []

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    def receive_on_port():
        ready.set()
        got.extend(udp_lines(port=port, timeout_s=1.0))

    thread = threading.Thread(target=receive_on_port)
    thread.start()
    ready.wait(timeout=1.0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(20):
        sender.sendto(b"1.0\n2.0\n", ("127.0.0.1", port))
        if got:
            break
        threading.Event().wait(0.05)
    sender.close()
    thread.join(timeout=3.0)
    assert got[:2] == ["1.0", "2.0"]


# ---------------------------------------------------------------------------
# on_grid, directly
# ---------------------------------------------------------------------------

def test_on_grid_fills_every_slot_exactly_once():
    pairs = [(0.0, 1.0), (3.0, 2.0), (4.0, 3.0)]
    got = list(on_grid(iter(pairs), RATE, epoch=0.0))
    assert [i for i, _, _ in got] == [0, 1, 2, 3, 4]
    values = [v for _, v, _ in got]
    assert values[0] == 1.0 and np.isnan(values[1]) and np.isnan(values[2])
    assert np.isnan([o for _, _, o in got][1])


def test_online_push_returns_the_accumulator_for_chaining():
    acc = online_avar(RATE, [1])
    assert online_push(acc, 1.0) is acc
