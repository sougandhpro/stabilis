"""Stage 4: the bench page, over real HTTP against a real server.

Two groups carry the weight.

**The identifiability tests** are assertions about graph theory, not about
this library's arithmetic, and they are the reason the stage exists: a
triangle separates its clocks, a chain and a four-ring do not however long
they run, and the page must say so before any file is read.

**The by-hand test** is the same contract stages 1-3 hold to. The page
prints the calls it made; those calls are run here by hand and the answer
has to be bit-identical. It is what stops the page from growing a second
opinion about a number, and it matters more here than anywhere else,
because a bench solution is several results combined and there is more room
for a discrepancy to hide.
"""

import json
import pathlib
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from stabilis.gui import Handler, cache_clear

# The truth the synthetic bench is built from: three clocks of genuinely
# different white-FM levels, and a measurement channel of its own.
LEVELS = {"A": 3e-12, "B": 1e-12, "C": 6e-12}
CHANNEL = 2e-13
N = 4096


@pytest.fixture(autouse=True)
def _fresh_cache():
    cache_clear()
    yield
    cache_clear()


@pytest.fixture
def rig(tmp_path):
    """Pairwise records for A, B, C plus a back-to-back run, in nanoseconds.

    `x_ab = x_a - x_b + channel`, which is the model the solver inverts, so
    the recovered variances have a known answer to be checked against
    rather than only being checked for self-consistency.
    """
    rng = np.random.default_rng(7)
    x = {k: np.cumsum(rng.standard_normal(N)) * v for k, v in LEVELS.items()}

    def write(name, values):
        with open(tmp_path / name, "w") as fh:
            fh.write("t,dt_ns\n")
            for i, v in enumerate(values):
                fh.write(f"{float(i):.1f},{v:.9f}\n")

    for a, b in (("A", "B"), ("B", "C"), ("A", "C")):
        chan = np.cumsum(rng.standard_normal(N)) * CHANNEL
        write(f"{a}{b}.csv", (x[a] - x[b] + chan) * 1e9)
    # Both ends on one clock: this row pins the channel from data instead of
    # assuming it, and is what makes the triangle four rows for four
    # unknowns rather than three for four.
    write("AA.csv", np.cumsum(rng.standard_normal(N)) * CHANNEL * 1e9)
    return tmp_path


@pytest.fixture
def server(rig):
    Handler.root = rig.resolve()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def source(id_):
    return {"id": id_, "name": "", "kind": "", "nominal_hz": None,
            "spec_adev": [], "spec_floor": None, "notes": ""}


def device(id_, channels, floor=None):
    return {"id": id_, "model": "", "serial": "", "notes": "",
            "channels": [{"id": c, "label": "", "floor": floor, "notes": ""}
                         for c in channels]}


def meas(id_, a, b, channel="CH1", path="", kind="phase"):
    return {"id": id_, "channel_id": channel, "source_a": a, "source_b": b,
            "path": path, "sha256": "", "kind": kind, "tau0_s": None,
            "epoch": "", "reason": ""}


TRIANGLE = {
    "sources": [source(s) for s in "ABC"],
    "devices": [device("CNT1", ["CH1"])],
    "measurements": [
        meas("M_AB", "A", "B", path="AB.csv"),
        meas("M_BC", "B", "C", path="BC.csv"),
        meas("M_AC", "A", "C", path="AC.csv"),
        meas("M_FLOOR", "A", "A", path="AA.csv"),
    ],
}

DATA = {m["id"]: {"path": m["path"], "expr": "dt_ns", "unit": "ns",
                  "rate": 1.0}
        for m in TRIANGLE["measurements"]}


def without(bench, *ids):
    """The same bench with some measurements removed."""
    return {**bench,
            "measurements": [m for m in bench["measurements"]
                             if m["id"] not in ids]}


# ---------------------------------------------------------------------------
# what the graph can and cannot determine, before any file is read
# ---------------------------------------------------------------------------

def test_a_triangle_with_a_measured_channel_is_identifiable(server):
    code, body = post(server, "/api/bench/check", {"bench": TRIANGLE})
    assert code == 200
    assert body["identifiable"]
    assert body["rank"] == body["n_unknowns"] == 4     # 3 clocks + 1 channel
    assert body["bipartite"] == [False]                # the odd cycle
    assert "IDENTIFIABLE" in body["report"]


def test_the_classical_triangle_is_one_row_short_and_says_so(server):
    """Three clocks through one channel is four unknowns against three rows.

    This is the trap the classical three-cornered hat hides by assuming the
    instrument contributes nothing. Here the assumption has to be written
    down -- as a declared floor, or as the back-to-back run that measures
    it -- and until it is, the page refuses to pretend.
    """
    code, body = post(server, "/api/bench/check",
                      {"bench": without(TRIANGLE, "M_FLOOR")})
    assert code == 200
    assert not body["identifiable"]
    assert body["rank"] == 3 and body["n_unknowns"] == 4
    assert body["free_channels"] == ["CH1"]


def test_declaring_the_channel_floor_is_what_makes_the_classical_hat_work(server):
    """Zero is a claim about the instrument, and writing it makes it visible."""
    bench = {**without(TRIANGLE, "M_FLOOR"),
             "devices": [device("CNT1", ["CH1"], floor=0.0)]}
    code, body = post(server, "/api/bench/check", {"bench": bench})
    assert code == 200
    assert body["identifiable"]
    assert body["free_channels"] == []           # it moved to the known side
    assert body["fixed_channels"] == ["CH1"]


def test_a_chain_is_bipartite_and_names_the_edge_that_fixes_it(server):
    code, body = post(server, "/api/bench/check",
                      {"bench": without(TRIANGLE, "M_AC", "M_FLOOR")})
    assert not body["identifiable"]
    assert body["bipartite"] == [True]
    assert ["A", "C"] in body["suggestions"]


def test_four_clocks_in_a_ring_determine_nothing_and_both_diagonals_fix_it(server):
    """The 4-cycle: a setup people build, and no other tool warns about."""
    ring = {
        "sources": [source(s) for s in "ABCD"],
        "devices": [device("CNT1", ["CH1"])],
        "measurements": [meas(f"M{a}{b}", a, b) for a, b in
                         (("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"))],
    }
    code, body = post(server, "/api/bench/check", {"bench": ring})
    assert not body["identifiable"]
    assert body["bipartite"] == [True]
    assert sorted(body["suggestions"]) == [["A", "C"], ["B", "D"]]


def test_a_pair_reachable_by_subtraction_is_marked_as_needing_no_new_data(server):
    """Two records against a common reference is the usual hat, not a dead end.

    `x_ab = x_ac - x_bc` identically, so the page has to distinguish "write
    a subtraction" from "buy a counter and log for a fortnight". Both come
    back, tagged differently.
    """
    bench = {
        "sources": [source(s) for s in "ABC"],
        "devices": [device("CNT1", ["CH1"], floor=0.0)],
        "measurements": [meas("M_AC", "A", "C"), meas("M_BC", "B", "C")],
    }
    code, body = post(server, "/api/bench/check", {"bench": bench})
    assert code == 200
    assert not body["identifiable"]
    assert ["A", "B"] in body["suggestions"]
    assert body["derivable"] == [["A", "B", "C", "M_AC", "M_BC"]]
    assert "no new data" in body["report"]


def test_a_pair_needing_real_data_is_not_marked_derivable(server):
    """Two disjoint pairs share no reference, so nothing can be constructed."""
    bench = {
        "sources": [source(s) for s in "ABCD"],
        "devices": [device("CNT1", ["CH1"], floor=0.0)],
        "measurements": [meas("M_AB", "A", "B"), meas("M_CD", "C", "D")],
    }
    code, body = post(server, "/api/bench/check", {"bench": bench})
    assert code == 200
    assert body["derivable"] == []


def test_an_empty_bench_reports_that_it_has_nothing_to_determine(server):
    """Rank 0 of 0 unknowns is trivially consistent and means nothing.

    The library is right to call it identifiable; the page would be wrong to
    show a tick, so the count it needs to say otherwise comes back too.
    """
    code, body = post(server, "/api/bench/check",
                      {"bench": {"sources": [], "devices": [],
                                 "measurements": []}})
    assert code == 200
    assert body["identifiable"] and body["n_measurements"] == 0


def test_a_clock_nobody_measures_is_named_rather_than_silently_dropped(server):
    bench = {**TRIANGLE, "sources": [source(s) for s in "ABCD"]}
    code, body = post(server, "/api/bench/check", {"bench": bench})
    assert body["identifiable"]
    assert body["unmeasured"] == ["D"]
    assert "D" not in body["sources"]


def test_a_bench_that_does_not_refer_to_itself_is_refused(server):
    bench = {**TRIANGLE, "measurements": [meas("M1", "A", "Z")]}
    code, body = post(server, "/api/bench/check", {"bench": bench})
    assert code == 400
    assert "not a source in this bench" in body["error"]


def test_the_check_returns_the_bench_as_the_library_would_write_it(server):
    """What the Save button hands to the browser is `store.to_dict` output.

    Round-tripped through the library rather than serialized in JavaScript,
    so the page cannot invent a layout that `stabilis bench check` then
    fails to read.
    """
    from stabilis.bench.store import from_dict, to_dict

    code, body = post(server, "/api/bench/check", {"bench": TRIANGLE})
    assert code == 200
    assert to_dict(from_dict(body["bench"])) == body["bench"]
    assert [m["id"] for m in body["bench"]["measurements"]] == \
        [m["id"] for m in TRIANGLE["measurements"]]


# ---------------------------------------------------------------------------
# the solve
# ---------------------------------------------------------------------------

def test_the_hat_recovers_clocks_of_known_level(server):
    """The one test that checks the answer rather than the plumbing.

    Loose tolerances on purpose: this is one realization of a stochastic
    process, and the ADEV of 4096 points at tau0 carries its own scatter.
    A tight bound here would be a test of the seed.
    """
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": DATA, "estimator": "oadev"})
    assert code == 200, body
    assert body["sources"] == ["A", "B", "C"] and body["channels"] == ["CH1"]
    for name, truth in LEVELS.items():
        assert body["source_dev"][name][0] == pytest.approx(truth, rel=0.25)
    assert body["channel_dev"]["CH1"][0] == pytest.approx(CHANNEL, rel=0.25)


def test_the_page_matches_the_library_called_by_hand(server, rig):
    """The page prints its own recipe; run it and demand the same numbers.

    Every step the endpoint took is reproduced here from the printed calls
    -- including the pedestal subtraction and the unit scaling, which is why
    the recipe uses `read_table` rather than the shorter `load_phase`.
    """
    import stabilis
    from stabilis.bench.solve import solve
    from stabilis.bench.store import from_dict
    from stabilis.io.text import read_table

    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": DATA, "estimator": "oadev"})
    assert code == 200, body

    results = {}
    for m in TRIANGLE["measurements"]:
        _labels, rows, _defects = read_table(rig / m["path"])
        rows = rows[1:]                     # the header line, as the page drops it
        values = rows[:, 1]
        values = (values - values[np.isfinite(values)][0]) * 1e-09
        series = stabilis.PhaseSeries(values, rate_hz=1.0)
        results[m["id"]] = stabilis.oadev(series, taus="octave", max_af=None)
    solution = solve(from_dict({**TRIANGLE, "schema": 1}), results,
                     require_identifiable=True)

    np.testing.assert_array_equal(body["tau_s"], solution.tau_s)
    for name in solution.sources:
        np.testing.assert_array_equal(body["source_dev"][name],
                                      solution.dev(name))


def test_the_recipe_the_page_prints_is_the_calls_it_made(server):
    """The printed script has to name every measurement and the solve.

    A recipe that quietly omitted a row would still look like a recipe, and
    would reproduce a different answer.
    """
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": DATA, "estimator": "oadev"})
    script = "\n".join(body["calls"])
    for m in TRIANGLE["measurements"]:
        assert f"results[{m['id']!r}]" in script
        assert repr(m["path"]) in script
    assert "values[np.isfinite(values)][0]" in script      # the pedestal
    assert "* 1e-09" in script                             # the unit scaling
    assert "require_identifiable=True" in script


def test_an_unidentifiable_bench_is_refused_until_it_is_forced(server):
    bench = without(TRIANGLE, "M_AC", "M_FLOOR")
    code, body = post(server, "/api/bench/solve",
                      {"bench": bench, "data": DATA, "estimator": "oadev"})
    assert code == 400
    assert "cannot determine its unknowns" in body["error"]

    code, body = post(server, "/api/bench/solve",
                      {"bench": bench, "data": DATA, "estimator": "oadev",
                       "allow_unidentifiable": True})
    assert code == 200
    assert "require_identifiable=False" in "\n".join(body["calls"])


def test_a_statistic_that_is_not_a_variance_is_refused(server):
    """MTIE: the maximum of a sum is not the sum of maxima."""
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": DATA, "estimator": "mtie"})
    assert code == 400
    assert "cannot be combined this way" in body["error"]


def test_an_unknown_estimator_is_refused_with_the_valid_list(server):
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": DATA, "estimator": "nope"})
    assert code == 400
    assert "oadev" in body["error"]


def test_a_measurement_with_no_data_is_left_out_and_said_to_be(server):
    """Absence is reported, never guessed at. The bench still counts the row
    when deciding what is identifiable, which is the point of writing a
    bench before the data exists."""
    data = {k: v for k, v in DATA.items() if k != "M_FLOOR"}
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": data, "estimator": "oadev"})
    assert code == 200
    assert any("M_FLOOR" in note and "no data assigned" in note
               for note in body["notes"])
    assert [m["id"] for m in body["measurements"]] == \
        ["M_AB", "M_BC", "M_AC"]


def test_a_file_with_no_column_named_is_refused_with_the_columns_listed(server):
    data = {**DATA, "M_AB": {"path": "AB.csv", "expr": "", "unit": "ns",
                             "rate": 1.0}}
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": data, "estimator": "oadev"})
    assert code == 400
    assert "dt_ns" in body["error"]


def test_the_benchs_tau0_wins_over_a_rate_typed_on_the_page(server):
    """`tau0_s` is the field `bench.json` carries to the command line.

    If a control on this page could override it, the page and
    `stabilis bench solve` would disagree about the same file -- so the
    bench is authoritative and the page says so.
    """
    bench = {**TRIANGLE,
             "measurements": [{**m, "tau0_s": 10.0}
                              for m in TRIANGLE["measurements"]]}
    code, body = post(server, "/api/bench/solve",
                      {"bench": bench, "data": DATA, "estimator": "oadev"})
    assert code == 200
    assert body["tau_s"][0] == pytest.approx(10.0)
    assert all(m["rate_hz"] == pytest.approx(0.1) for m in body["measurements"])
    assert all("from the bench" in m["rate_note"] for m in body["measurements"])


def test_units_are_the_pages_business_and_scale_the_answer_exactly(server):
    """The bench says nothing about what unit a file is in, so this does."""
    in_ns = post(server, "/api/bench/solve",
                 {"bench": TRIANGLE, "data": DATA, "estimator": "oadev"})[1]
    as_s = {k: {**v, "unit": "s"} for k, v in DATA.items()}
    in_s = post(server, "/api/bench/solve",
                {"bench": TRIANGLE, "data": as_s, "estimator": "oadev"})[1]
    np.testing.assert_allclose(
        np.array(in_s["source_dev"]["A"]),
        np.array(in_ns["source_dev"]["A"]) * 1e9, rtol=1e-12)


def test_a_crop_reaches_the_solve_and_is_recorded(server):
    data = {k: {**v, "windows": [[100, 1100]]} for k, v in DATA.items()}
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": data, "estimator": "oadev"})
    assert code == 200
    assert all(m["n"] == 1000 for m in body["measurements"])
    assert "edit.select_range(series, 100, 1100" in "\n".join(body["calls"])


def test_the_step_log_reaches_the_page(server):
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": DATA, "estimator": "oadev"})
    actions = [s["action"] for s in body["steps"]]
    assert "solve" in actions
    assert any(s["module"] == "bench" for s in body["steps"])


def test_a_correlation_the_bench_can_see_is_warned_about(server):
    """Every row through one channel: its noise is common and cannot be
    attributed to any single measurement."""
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": DATA, "estimator": "oadev"})
    assert any("CH1" in w for w in body["warnings"])


def test_the_unconstrained_solution_comes_back_signed(server):
    """A negative variance is the single most useful diagnostic the method
    produces, so it is reported rather than clipped."""
    code, body = post(server, "/api/bench/solve",
                      {"bench": TRIANGLE, "data": DATA, "estimator": "oadev"})
    assert set(body["source_var_signed"]) == {"A", "B", "C"}
    assert "CH1" in body["channel_var_signed"]
    signed = np.array([body["source_var_signed"][s] for s in "ABC"])
    constrained = np.array([body["source_dev"][s] for s in "ABC"]) ** 2
    assert np.all(constrained >= 0.0)
    assert body["any_negative"] == bool(np.any(signed < 0.0))


# ---------------------------------------------------------------------------
# six counters, six formats, and none of them clean
# ---------------------------------------------------------------------------

@pytest.fixture
def messy(tmp_path):
    """One bench, four files, four ways of writing a number down.

    This is the case the stage exists for and the one the rest of the page
    never sees: every other stage looks at a single file the user is already
    looking at. Here each row is a different instrument's export -- its own
    delimiter, its own timebase, its own dropouts, its own glitch.
    """
    rng = np.random.default_rng(11)
    x = {k: np.cumsum(rng.standard_normal(N)) * v for k, v in LEVELS.items()}
    t = np.arange(N, dtype=float)

    # comma, header, nanoseconds, and a warm-up transient in the first 400
    ab = (x["A"] - x["B"]) * 1e9
    ab[:400] += np.linspace(500.0, 0.0, 400)
    with open(tmp_path / "ab.csv", "w") as fh:
        fh.write("t,dt_ns\n")
        for i in range(N):
            fh.write(f"{t[i]:.1f},{ab[i]:.9f}\n")

    # tab separated, no header, and a single wild sample
    bc = (x["B"] - x["C"]) * 1e9
    bc[2500] = 9.9e6
    np.savetxt(tmp_path / "bc.tsv", np.column_stack([t, bc]),
               delimiter="\t", fmt="%.9f")

    # MJD timestamps at 1 Hz, which is 1/86400 of a day per sample
    ac = (x["A"] - x["C"]) * 1e9
    np.savetxt(tmp_path / "ac.dat",
               np.column_stack([59000.0 + t / 86400.0, ac]), fmt="%.12f")

    # back-to-back, with genuine dropouts written as blanks
    aa = np.cumsum(rng.standard_normal(N)) * CHANNEL * 1e9
    with open(tmp_path / "aa.csv", "w") as fh:
        fh.write("t,dt_ns\n")
        for i in range(N):
            fh.write(f"{t[i]:.1f},"
                     + ("nan" if 2000 <= i < 2050 else f"{aa[i]:.9f}") + "\n")
    return tmp_path


@pytest.fixture
def messy_server(messy):
    Handler.root = messy.resolve()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


MESSY_DATA = {
    "M_AB": {"path": "ab.csv", "expr": "dt_ns", "unit": "ns", "rate": 1.0,
             "windows": [[400, N]], "reason": "after warm-up"},
    "M_BC": {"path": "bc.tsv", "expr": "#1", "unit": "ns", "rate": 1.0,
             "delimiter": "\t", "outlier_threshold": 6.0,
             "reason": "counter glitch"},
    # An explicit rate, which is what the "no common averaging time" message
    # tells you to do and what `test_an_mjd_timebase_needs_an_explicit_rate`
    # exercises the other half of.
    "M_AC": {"path": "ac.dat", "expr": "#1", "unit": "ns", "rate": 1.0,
             "time_column": 0, "time_format": "mjd"},
    "M_FLOOR": {"path": "aa.csv", "expr": "dt_ns", "unit": "ns", "rate": 1.0},
}
MESSY_BENCH = {
    "sources": [source(s) for s in "ABC"],
    "devices": [device("CNT1", ["CH1"])],
    "measurements": [
        meas("M_AB", "A", "B", path="ab.csv"),
        meas("M_BC", "B", "C", path="bc.tsv"),
        meas("M_AC", "A", "C", path="ac.dat"),
        meas("M_FLOOR", "A", "A", path="aa.csv"),
    ],
}


def test_four_instruments_four_formats_one_solve(messy_server):
    """Each measurement carries its own delimiter, timebase and units."""
    code, body = post(messy_server, "/api/bench/solve",
                      {"bench": MESSY_BENCH, "data": MESSY_DATA,
                       "estimator": "oadev"})
    assert code == 200, body
    assert body["sources"] == ["A", "B", "C"]
    for name, truth in LEVELS.items():
        assert body["source_dev"][name][0] == pytest.approx(truth, rel=0.3)
    # every row was read, each from its own file
    assert {m["path"] for m in body["measurements"]} == \
        {"ab.csv", "bc.tsv", "ac.dat", "aa.csv"}


def test_an_mjd_timebase_is_not_read_as_seconds(messy_server):
    """1/86400 of a day per sample. Reading it as seconds puts every tau out
    by that factor, plausibly and silently."""
    from stabilis.gui import files

    table = files.load_table(pathlib.Path(files.safe_path(
        pathlib.Path(Handler.root), "ac.dat")))
    rate, _jitter, note = files.timebase(table, 0, "mjd")
    assert rate == pytest.approx(1.0, rel=1e-6)
    assert "mjd" in note


def test_an_mjd_timebase_needs_an_explicit_rate_and_the_page_says_why(
        messy_server):
    """A real trap for a bench, and the reason is float64, not the counter.

    MJD 59000 has a float64 spacing of 6.3e-7 s, so a 1 Hz timebase infers
    as 0.99999976 Hz and `af/rate` misses an exact 1.000 s by far more than
    the nanosecond tolerance `_common_taus` allows. Against a row given an
    explicit 1 Hz the two share no tau at all, and the solver is right to
    refuse. What the page adds is *why*, and what to do.
    """
    inferred = {**MESSY_DATA, "M_AC": {"path": "ac.dat", "expr": "#1",
                                       "unit": "ns", "time_column": 0,
                                       "time_format": "mjd"}}
    code, body = post(messy_server, "/api/bench/solve",
                      {"bench": MESSY_BENCH, "data": inferred,
                       "estimator": "oadev"})
    assert code == 400
    message = body["error"]
    assert "share no averaging time" in message      # the library's refusal
    assert "Rates the page used" in message          # what the page added
    assert "M_AC" in message and "quantized by float64" in message
    assert "explicit rate" in message

    # And with the rate given, the same bench solves.
    code, _ok = post(messy_server, "/api/bench/solve",
                     {"bench": MESSY_BENCH, "data": MESSY_DATA,
                      "estimator": "oadev"})
    assert code == 200


def test_a_wrong_delimiter_is_refused_rather_than_half_read(messy_server):
    data = {**MESSY_DATA, "M_BC": {**MESSY_DATA["M_BC"], "delimiter": ","}}
    code, body = post(messy_server, "/api/bench/solve",
                      {"bench": MESSY_BENCH, "data": data,
                       "estimator": "oadev"})
    assert code == 400


def test_the_outlier_rejection_is_counted_and_reported(messy_server):
    """A threshold that quietly took a third of the record is the thing
    worth seeing, so what it removed comes back as a number."""
    code, body = post(messy_server, "/api/bench/solve",
                      {"bench": MESSY_BENCH, "data": MESSY_DATA,
                       "estimator": "oadev"})
    bc = next(m for m in body["measurements"] if m["id"] == "M_BC")
    assert bc["n_rejected"] >= 1
    assert any("outlier threshold removed" in note for note in body["notes"])

    # And with no threshold the wild sample stays, which is what makes the
    # rejection worth having: the deviation is visibly worse.
    data = {**MESSY_DATA, "M_BC": {**MESSY_DATA["M_BC"],
                                   "outlier_threshold": ""}}
    _c, raw = post(messy_server, "/api/bench/solve",
                   {"bench": MESSY_BENCH, "data": data, "estimator": "oadev"})
    kept = next(m for m in raw["measurements"] if m["id"] == "M_BC")
    assert kept["n_rejected"] == 0
    assert kept["dev"][0] > bc["dev"][0] * 5


def test_gaps_in_a_record_are_reported_per_measurement(messy_server):
    """Which of six counters dropped out is actionable; "somewhere" is not."""
    code, body = post(messy_server, "/api/bench/solve",
                      {"bench": MESSY_BENCH, "data": MESSY_DATA,
                       "estimator": "oadev"})
    floor = next(m for m in body["measurements"] if m["id"] == "M_FLOOR")
    assert floor["n_gaps"] == 50
    assert floor["runs"] == 2
    assert any("M_FLOOR" in n and "gaps" in n for n in body["notes"])


def test_a_crop_keeps_the_warm_up_out(messy_server):
    """The window is a real edit, and it is recorded as one."""
    code, body = post(messy_server, "/api/bench/solve",
                      {"bench": MESSY_BENCH, "data": MESSY_DATA,
                       "estimator": "oadev"})
    ab = next(m for m in body["measurements"] if m["id"] == "M_AB")
    assert ab["window"] == [400, N] and ab["n"] == N - 400
    assert "reason='after warm-up'" in "\n".join(body["calls"])


def test_several_windows_become_one_gapped_record_never_two_curves(messy_server):
    """A measurement is one row of the bench, so it cannot become two
    results. The stretches between kept windows are gaps."""
    data = {**MESSY_DATA,
            "M_AB": {**MESSY_DATA["M_AB"], "windows": [[500, 1500],
                                                       [3000, 4000]]}}
    code, body = post(messy_server, "/api/bench/solve",
                      {"bench": MESSY_BENCH, "data": data,
                       "estimator": "oadev"})
    assert code == 200
    ab = next(m for m in body["measurements"] if m["id"] == "M_AB")
    assert ab["n_windows"] == 2
    assert ab["window"] == [500, 4000]
    assert ab["n"] == 3500 and ab["n_gaps"] == 1500     # the stretch between
    assert "nothing interpolated" in "\n".join(body["calls"])


def test_counters_running_at_different_rates_are_named(messy_server):
    """`bench.solve` intersects tau grids and interpolates nothing, so a rate
    mismatch has to be visible before it becomes "no common averaging time"."""
    data = {**MESSY_DATA, "M_FLOOR": {**MESSY_DATA["M_FLOOR"], "rate": 0.5}}
    code, body = post(messy_server, "/api/bench/solve",
                      {"bench": MESSY_BENCH, "data": data,
                       "estimator": "oadev"})
    assert code == 200
    assert any("different rates" in note for note in body["notes"])
    assert body["tau_s"][0] == pytest.approx(2.0)      # the common grid


def test_no_common_tau_is_refused_with_the_spans_named(messy_server):
    data = {**MESSY_DATA, "M_FLOOR": {**MESSY_DATA["M_FLOOR"], "rate": 3.7}}
    code, body = post(messy_server, "/api/bench/solve",
                      {"bench": MESSY_BENCH, "data": data,
                       "estimator": "oadev"})
    assert code == 400
    assert "share no averaging time" in body["error"]


def test_the_record_can_be_previewed_before_it_is_solved(messy_server):
    """The editor draws through the same `/api/traces` stages 1 and 2 use,
    so a measurement's assignment is posted as the request object it is."""
    code, body = post(messy_server, "/api/traces", {
        "path": "bc.tsv", "series": [{"expr": "#1", "label": "M_BC"}],
        "delimiter": "\t", "data_kind": "phase", "unit": "ns", "rate": 1.0,
        "scaled": True, "target": 400})
    assert code == 200
    series = body["series"][0]
    assert series["kind"] == "envelope"
    # the wild sample survives decimation, which is the whole point of a
    # min/max envelope over stride sampling
    assert max(v for v in series["hi"] if v is not None) > 1e-3


# ---------------------------------------------------------------------------
# many files at once
# ---------------------------------------------------------------------------

def test_more_files_than_the_old_four_slot_cache_stay_resident(tmp_path):
    """A bench wants every one of its files on every solve.

    The cache used to hold four, so a six-measurement bench evicted the file
    it was about to need and re-read the lot each time. The budget is in
    bytes now, because four 2 GB records is not the same commitment as four
    2 MB ones.
    """
    from stabilis.gui import files

    files.cache_clear()
    paths = []
    for i in range(8):
        p = tmp_path / f"c{i}.dat"
        np.savetxt(p, np.arange(500, dtype=float))
        paths.append(p)
    for p in paths:
        files.load_table(p)

    reads = []
    original = files._read
    files._read = lambda path, delim: (reads.append(path)
                                       or original(path, delim))
    try:
        for p in paths:
            files.load_table(p)
    finally:
        files._read = original
    assert reads == [], f"re-read {reads} that should have been cached"
    assert files.cache_bytes() == 8 * 500 * 8


def test_the_cache_still_has_a_ceiling(tmp_path, monkeypatch):
    """Bounded, or a page left open overnight holds a gigabyte hostage."""
    from stabilis.gui import files

    files.cache_clear()
    monkeypatch.setattr(files, "CACHE_BYTES", 4000)     # 500 float64s
    for i in range(6):
        p = tmp_path / f"d{i}.dat"
        np.savetxt(p, np.arange(500, dtype=float))
        files.load_table(p)
    assert files.cache_bytes() <= 4000


def test_one_record_larger_than_the_budget_is_still_cached(tmp_path, monkeypatch):
    """Evicting it on arrival would re-read it on every keystroke instead."""
    from stabilis.gui import files

    files.cache_clear()
    monkeypatch.setattr(files, "CACHE_BYTES", 8)        # one float64
    p = tmp_path / "big.dat"
    np.savetxt(p, np.arange(500, dtype=float))
    files.load_table(p)
    assert files.cache_bytes() == 500 * 8


# ---------------------------------------------------------------------------
# opening a bench file
# ---------------------------------------------------------------------------

def test_a_bench_file_round_trips_through_the_page(server, rig):
    from stabilis.bench.store import from_dict, save

    save(from_dict({**TRIANGLE, "schema": 1}), rig / "bench.json")
    code, body = post(server, "/api/bench/load", {"path": "bench.json"})
    assert code == 200
    assert [m["id"] for m in body["bench"]["measurements"]] == \
        [m["id"] for m in TRIANGLE["measurements"]]
    assert body["hash_problems"] == []


def test_a_changed_file_is_reported_rather_than_refused(server, rig):
    """An unsolvable bench is still one worth opening and correcting."""
    from stabilis.bench.store import from_dict, save

    bench = {**TRIANGLE,
             "measurements": [{**m, "sha256": "0" * 64}
                              for m in TRIANGLE["measurements"]]}
    save(from_dict({**bench, "schema": 1}), rig / "bench.json")
    code, body = post(server, "/api/bench/load", {"path": "bench.json"})
    assert code == 200
    assert len(body["hash_problems"]) == len(TRIANGLE["measurements"])
    assert "AB.csv" in " ".join(body["hash_problems"])


def test_a_bench_path_cannot_climb_out_of_the_root(server):
    code, body = post(server, "/api/bench/load",
                      {"path": "../../etc/passwd"})
    assert code == 400
    assert "outside" in body["error"]


def test_a_schema_the_reader_does_not_know_is_refused(server, rig):
    (rig / "future.json").write_text(json.dumps({"schema": 99}), encoding="utf-8")
    code, body = post(server, "/api/bench/load", {"path": "future.json"})
    assert code == 400
    assert "not readable by this version" in body["error"]


# ---------------------------------------------------------------------------
# the boundary
# ---------------------------------------------------------------------------

def test_the_bench_stage_needs_no_matplotlib(server):
    import sys

    before = set(sys.modules)
    post(server, "/api/bench/solve",
         {"bench": TRIANGLE, "data": DATA, "estimator": "oadev"})
    for module in set(sys.modules) - before:
        assert not module.startswith("matplotlib"), module


# The two command-line modules that sit on the boundary with the page.
_CLI_GUI_EDGE = ("stabilis.cli", "stabilis.cli.gui")


def test_nothing_in_the_library_imports_the_page(server):
    """`gui/` is a client of `stabilis`, and the arrow points one way.

    The same discipline `stats/` has towards `bench/`: the page may reach
    into the library freely, and the day the library reaches back, the page
    has stopped being something that could be lifted out and shipped
    separately.

    Two names on the command line are exempt, and both are small enough to
    read. `stabilis.cli.gui` dispatches `stabilis gui`, which is its whole
    job; `stabilis.cli` itself writes `from .gui import add_gui` to assemble
    the parser, and an `ast.ImportFrom` for a relative import records
    `module` as the bare token `gui`, which the scan below cannot tell from
    the package.

    Only `serve` is lazy. `DEFAULT_PORT` is imported at module scope and
    always was, because argparse captures a default when the parser is
    built -- so `import stabilis.cli` loads the whole page package for one
    integer. That is a real wart, and naming it here is better than the
    older wording, which claimed the import was inside the command function
    and was not true of the old `cli.py` either. What the exemption buys is that
    every *other* module in the package is now proved not to reach the page;
    the single blanket exemption for a 2054-line file proved nothing.
    """
    import ast
    import importlib
    import inspect
    import pkgutil

    import stabilis

    checked = 0
    for info in pkgutil.walk_packages(stabilis.__path__, "stabilis."):
        if info.name.startswith("stabilis.gui") or info.name in _CLI_GUI_EDGE:
            continue
        module = importlib.import_module(info.name)
        try:
            tree = ast.parse(inspect.getsource(module))
        except (OSError, TypeError):
            continue
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(f"{node.module or ''}.{a.name}"
                             for a in node.names)
        assert not [n for n in names if "gui" in n.split(".")], \
            f"{info.name} imports the page"
        checked += 1
    assert checked >= 10, "expected to have scanned the whole package"
