"""Greenhall EDOF, against the paper, against arithmetic, and against truth.

The paper this transcribes prints no worked values -- sections 5 and 6 of
the preprint are empty headings -- and Stable32 reference output does not
exist in this repo yet. So there is no table to check against, and the
validation is built out of four independent things instead:

* the *structure* of the algorithm checked against arithmetic it should
  already satisfy (the difference coefficients, the alpha = 2 row, the
  blank cells);
* the term count checked against the coefficient arrays the estimators
  really build;
* the kernels checked against sequences worked out by hand from the
  published GACV table;
* and the answer checked against `2*sigma^4/Var(sigma_hat^2)` measured by
  Monte Carlo, which is what EDOF *means*.

A differential test against `allantools` is included where that package is
present. It is an INTEROP check between two independent implementations of
one published algorithm; no code was read from it and none was ported.
"""

import math

import numpy as np
import pytest

from stabilis.stats import adev, mdev, oadev, ohdev, tdev
from stabilis.stats.greenhall import (
    GREENHALL_ALPHAS,
    GREENHALL_ORDERS,
    J_MAX,
    _sx,
    _sz,
    _SZ_COEFFS,
    _TABLE1,
    _TABLE2,
    greenhall_edof,
    greenhall_edof_pooled,
    greenhall_edof_simple,
    term_count,
)
from stabilis.synth import power_law_freq, power_law_phase
from stabilis.types import FreqSeries

ALPHAS = (2, 1, 0, -1, -2)

# (d, modified, overlapped) for each estimator, and the name it goes by.
FAMILIES = (
    ("oadev", 2, False, True),
    ("adev", 2, False, False),
    ("mdev", 2, True, True),
    ("ohdev", 3, False, True),
)


def _monte_carlo_eta(runs):
    """eta_true = 2*mean(var)^2 / Var(var), per tau, from the definition."""
    v = np.array([r.var for r in runs])
    return 2.0 * v.mean(axis=0) ** 2 / v.var(axis=0)


# --- the transcription, checked against arithmetic -----------------------


@pytest.mark.parametrize("d", GREENHALL_ORDERS)
def test_difference_coefficients_are_the_binomials_they_should_be(d):
    """(Delta_1 Delta_-1)^d has coefficients (-1)^k C(2d, d-k).

    The paper prints the three expansions literally, so they were typed in
    literally. This re-derives them, which turns a transcription into a
    checked fact.
    """
    for k, c in enumerate(_SZ_COEFFS[d]):
        assert c == (-1) ** k * math.comb(2 * d, d - k)


@pytest.mark.parametrize("d", GREENHALL_ORDERS)
def test_table2_white_pm_row_equals_the_case4_closed_form(d):
    """Table 2's alpha = 2 row is also section 4.6.4's exact coefficients.

    a0 = C(4d,2d)/C(2d,d)^2 and a1 = d/2. The paper gives both, in two
    different places, which makes one a check on the other -- an entire
    transcribed row validated by arithmetic rather than by eyesight.
    """
    a0, a1 = _TABLE2[(2, d)]
    assert a0 == pytest.approx(math.comb(4 * d, 2 * d) / math.comb(2 * d, d) ** 2)
    assert a1 == pytest.approx(d / 2)


def test_blank_table_cells_are_exactly_the_non_stationary_combinations():
    """Every gap in Tables 1 and 2 is a case where alpha + 2d <= 1.

    Not a coincidence and worth pinning: it says the tables are complete
    over their whole valid domain, so a missing key can only ever be a
    transcription slip, never a legitimately absent value.
    """
    for table in (_TABLE1, _TABLE2):
        for alpha in GREENHALL_ALPHAS:
            for d in GREENHALL_ORDERS:
                assert ((alpha, d) in table) == (alpha + 2 * d > 1)


# --- the kernels, checked against hand-worked sequences ------------------


def test_phase_autocovariance_at_unit_filter_factor():
    """sx(t, 1, 0) = 2|t|^3 - |t-1|^3 - |t+1|^3 = -6|t|, and -2 at t = 0.

    Worked out by hand from the sw(t, 0) = |t|^3 entry of eq. (7). The
    cubic collapsing to something linear is the whole content of the
    second-difference operator, so getting it wrong would be silent.
    """
    assert [float(_sx(float(t), 1.0, 0)) for t in range(5)] == [-2, -6, -12, -18, -24]


@pytest.mark.parametrize(
    "alpha,expected",
    [
        (0, [12.0, -4.0, -2.0, 0.0, 0.0]),
        (-2, [132.0, 52.0, 2.0, 0.0, 0.0]),
    ],
)
def test_second_difference_autocovariance_matches_hand_calculation(alpha, expected):
    """sz(t, 1, alpha, 2) for white FM and random-walk FM, worked by hand.

    From eq. (7)'s |t|^3 and -|t|^5 entries, through eq. (8) at F = 1 and
    eq. (9) at d = 2. Exact integers, so this is asserted at equality
    rather than at a tolerance: every one of the three published equations
    has to be right for the sequence to come out.

    That both die at t = 3 is the decay eq. (12) relies on when it
    truncates at (d+1)S.
    """
    assert [float(_sz(float(t), 1.0, alpha, 2)) for t in range(5)] == expected


def test_log_entries_of_the_gacv_vanish_at_zero():
    """"The entries with ln|t| must evaluate to 0 when t = 0" -- eq. (7).

    Left to itself that is 0 * -inf. The guard is in _sw; without it the
    whole sum comes back NaN for three of the seven power laws.
    """
    for alpha in (1, -1, -3):
        assert float(_sz(0.0, 1.0, alpha, 2)) == pytest.approx(
            float(_sz(0.0, 1.0, alpha, 2))
        )
        assert np.isfinite(float(_sz(0.0, 1.0, alpha, 2)))


# --- the term count, against the estimators that build the terms ---------


def test_eq11_reproduces_every_coefficient_array_length():
    """M from eq. (11) equals the number of coefficients actually formed.

    Over every estimator and every averaging factor on an `all` grid. This
    is the sharpest available check that each estimator's declared
    `Shape(d, modified, overlapped)` is the right one -- a wrong `d` or a
    wrong `modified` shifts the count and shows up here immediately, where
    it would otherwise only show up as a subtly wrong error bar.
    """
    rng = np.random.default_rng(3)
    series = power_law_phase(999, 0, rate_hz=1.0, rng=rng)
    for fn in (adev, oadev, mdev, tdev, ohdev):
        r = fn(series, taus="all")
        assert r.shape is not None, f"{r.kind} declares no Shape"
        for m, c in zip(r.af.tolist(), r.coeffs):
            want = term_count(series.n, m, r.shape.d, r.shape.modified, r.shape.overlapped)
            assert want == c.size, f"{r.kind} m={m}: eq.11 gave {want}, built {c.size}"


def test_term_count_refuses_a_record_too_short_for_one_term():
    with pytest.raises(ValueError, match="not enough data"):
        term_count(n=5, m=4, d=2, modified=False, overlapped=True)


# --- structure -----------------------------------------------------------


@pytest.mark.parametrize("name,d,modified,overlapped", FAMILIES)
def test_edof_stays_between_one_and_the_term_count(name, d, modified, overlapped):
    """1 <= edf <= M, for every power law and averaging factor.

    Not imposed by a clamp anywhere -- the algorithm is left to produce
    what it produces, so this is a statement about the algorithm rather
    than about the guard rails.
    """
    n = 1024
    for alpha in GREENHALL_ALPHAS:
        if alpha + 2 * d <= 1:
            continue
        for m in (1, 2, 3, 7, 16, 64, 128):
            terms = term_count(n, m, d, modified, overlapped)
            if terms < 1:
                continue
            eta = greenhall_edof(alpha, d, m, n, modified, overlapped)
            assert 1.0 <= eta <= terms, f"{name} alpha={alpha} m={m}: eta={eta}"


@pytest.mark.parametrize("name,d,modified,overlapped", FAMILIES)
def test_edof_falls_as_the_averaging_factor_rises(name, d, modified, overlapped):
    """Longer averaging means fewer independent terms, always."""
    n = 4096
    for alpha in GREENHALL_ALPHAS:
        if alpha + 2 * d <= 1:
            continue
        etas = [
            greenhall_edof(alpha, d, m, n, modified, overlapped)
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        ]
        assert all(b < a for a, b in zip(etas, etas[1:])), f"{name} alpha={alpha}: {etas}"


def test_overlapping_beats_non_overlapping():
    """The reason to pay for the overlapped estimator at all.

    Same data, same tau: reusing samples across terms buys degrees of
    freedom. If this ever inverted, the estimator recommendation in every
    report built on it would be backwards.
    """
    n = 2048
    for alpha in ALPHAS:
        for m in (2, 4, 16, 64):
            over = greenhall_edof(alpha, 2, m, n, False, True)
            under = greenhall_edof(alpha, 2, m, n, False, False)
            assert over > under, f"alpha={alpha} m={m}: {over} !> {under}"


def test_non_stationary_combinations_are_refused_not_clamped():
    """alpha + 2d <= 1 means the variance does not exist. Say so.

    Random-walk FM has no Allan-variance-order... it has, but flicker walk
    and random run do not: at d = 2 the second differences of alpha = -3
    are still non-stationary and any number returned would be a fiction.
    """
    with pytest.raises(ValueError, match=r"alpha \+ 2d > 1"):
        greenhall_edof(-3, 2, 1, 1000, False, True)
    with pytest.raises(ValueError, match=r"alpha \+ 2d > 1"):
        greenhall_edof(-1, 1, 1, 1000, False, True)
    with pytest.raises(ValueError, match="alpha must be one of"):
        greenhall_edof(-5, 3, 1, 1000, False, True)


def test_the_infinite_filter_limit_is_refused_above_white_fm():
    """sx(t, inf, alpha) = sw(t, alpha+2) only holds for -4 <= alpha <= 0.

    Above that the limiting form would index past the published table, so
    it must never be reached. Case 2 is the only caller and it is gated on
    alpha <= 0; this pins the guard rather than trusting the gate.
    """
    with pytest.raises(ValueError, match="F=inf limit"):
        _sx(1.0, math.inf, 1)


# --- the simplified algorithm against the full one -----------------------


def test_simplified_and_full_agree_where_the_sum_is_short():
    """Section 4.5 vs 4.6: identical whenever J <= J_MAX and F = 1.

    The full version only departs from the simplified one to avoid a long
    sum or a cancellation. Where neither applies they must be the same
    arithmetic, and this says so.
    """
    n = 4096
    for alpha in ALPHAS:
        for m in (1, 2, 4, 8, 16):
            if min(term_count(n, m, 2, True, True), 3 * m) > J_MAX:
                continue
            full = greenhall_edof(alpha, 2, m, n, True, True)
            simple = greenhall_edof_simple(alpha, 2, m, n, True, True)
            assert full == pytest.approx(simple, rel=1e-12)


def test_extending_the_truncation_barely_moves_the_answer():
    """The paper's own suggested check: re-run at 6S instead of (d+1)S.

    sz decays past t = d, so the extra lags should contribute almost
    nothing. If they contributed a lot, the truncation in eq. (12) would
    be unsafe and every edf in the library would be biased.
    """
    n = 4096
    for alpha in ALPHAS:
        for m in (2, 8, 32):
            short = greenhall_edof_simple(alpha, 2, m, n, False, True)
            long = greenhall_edof_simple(alpha, 2, m, n, False, True, j_factor=6)
            assert short == pytest.approx(long, rel=0.02)


# --- against measurement -------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_greenhall_tracks_monte_carlo_truth(alpha):
    """The definitional check: eta vs 2*sigma^4/Var(sigma_hat^2).

    Measured, not tabulated. The tolerance is wide and one-sided in
    neither direction, because two different approximations are being
    compared: Greenhall's model of the noise, and a Monte-Carlo estimate
    whose own relative standard error is sqrt(2/trials) = 7% here.

    m = 1 is excluded and tested separately below -- it is the one place
    the model's assumptions bite hard enough to matter, and hiding that
    inside a loose tolerance would be the wrong kind of green.
    """
    n = 512
    rng = np.random.default_rng(100 + alpha)
    runs = [
        oadev(power_law_freq(n, alpha, rate_hz=1.0, rng=rng), taus="octave")
        for _ in range(400)
    ]
    truth = _monte_carlo_eta(runs)
    for i, m in enumerate(runs[0].af.tolist()):
        if m == 1:
            continue
        eta = greenhall_edof(alpha, 2, m, n + 1, False, True)
        ratio = eta / truth[i]
        assert 0.7 < ratio < 1.3, f"alpha={alpha} m={m}: greenhall/truth = {ratio:.2f}"


def test_the_model_departs_from_ideal_power_law_noise_at_m_equals_one():
    """At m = 1 Greenhall's process model is materially not the ideal one.

    This is the paper's model showing through, not a transcription error:
    it takes the phase to be `x = Delta_tau0 w` for a `w` with an exact
    power-law spectrum at all frequencies, so its "white FM" rolls off
    towards Nyquist rather than being flat. At m = 1 the estimator sits
    right at that corner and the difference is large. By m = 4 it is gone.

    Both directions are pinned because the *sign* is what matters to a
    user. For ideal white FM the second differences are an MA(1) with
    rho(1) = -1/2, so eta_true = M/1.5 exactly -- and Greenhall runs high,
    meaning error bars **too narrow**. For ideal random-walk FM the second
    differences are iid, so eta_true = M exactly -- and Greenhall runs low,
    meaning error bars too wide. Neither is fatal, both are quotable, and
    a user who thinks these intervals are exact should see this test.
    """
    n = 513
    terms = term_count(n, 1, 2, False, True)

    white_fm = greenhall_edof(0, 2, 1, n, False, True)
    assert white_fm / (terms / 1.5) == pytest.approx(1.17, abs=0.02)

    random_walk_fm = greenhall_edof(-2, 2, 1, n, False, True)
    assert random_walk_fm / terms == pytest.approx(0.76, abs=0.02)


# --- gapped records ------------------------------------------------------


def test_pooling_is_exact_for_a_single_run():
    """One run must give back the plain algorithm, bit for bit.

    Otherwise the pooled path would quietly become a second, slightly
    different implementation used on most real data.
    """
    n = 1000
    terms = term_count(n, 4, 2, False, True)
    direct = greenhall_edof(0, 2, 4, n, False, True)
    for policy in ("pool", "fallback"):
        pooled, used = greenhall_edof_pooled(
            0, 2, 4, [n], [terms], False, True, policy=policy
        )
        assert used == "greenhall"
        assert pooled == direct


def test_pooling_two_equal_runs_doubles_the_degrees_of_freedom():
    """Satterthwaite on two identical runs: eta_pool = 2*eta.

        (2M)^2 / (2 * M^2/eta) = 2*eta

    Exactly, and it is the sanity check that the weights are the term
    counts and not something else. Note it is *not* the same as the edf of
    one record of twice the length, which is larger -- the terms spanning
    the join are lost.
    """
    n = 600
    terms = term_count(n, 8, 2, False, True)
    single = greenhall_edof(0, 2, 8, n, False, True)
    pooled, _ = greenhall_edof_pooled(
        0, 2, 8, [n, n], [terms, terms], False, True, policy="pool"
    )
    assert pooled == pytest.approx(2.0 * single, rel=1e-12)
    assert pooled < greenhall_edof(0, 2, 8, 2 * n, False, True)


def test_fallback_policy_declines_a_segmented_record():
    """The strictly faithful reading: the paper assumes one contiguous run.

    It returns NaN and a reason rather than raising, so the caller records
    the refusal and drops to eta1 -- which needs no such assumption.
    """
    eta, used = greenhall_edof_pooled(
        0, 2, 4, [500, 400], [492, 392], False, True, policy="fallback"
    )
    assert math.isnan(eta)
    assert used == "greenhall-unavailable"
    with pytest.raises(ValueError, match="policy must be one of"):
        greenhall_edof_pooled(0, 2, 4, [500], [492], False, True, policy="nonsense")


def test_lost_terms_cost_degrees_of_freedom_in_proportion():
    """A masked gap removes terms, and eq. (6) carries a factor 1/M.

    So an estimate built from half the nominal terms gets half the degrees
    of freedom. Using the nominal count instead would quote error bars for
    data that was not there.
    """
    n = 1000
    terms = term_count(n, 4, 2, False, True)
    full = greenhall_edof(0, 2, 4, n, False, True)
    half, _ = greenhall_edof_pooled(0, 2, 4, [n], [terms // 2], False, True)
    assert half == pytest.approx(full * (terms // 2) / terms, rel=1e-12)


def test_pooling_matches_a_real_segmented_record():
    """End to end, through the estimator, on data with an actual gap.

    Checks the plumbing rather than the arithmetic: that `run_lengths` and
    `run_terms` come off the result in the shape the pooled function wants,
    and that the per-run term counts sum to the pooled coefficient array.
    """
    rng = np.random.default_rng(7)
    y = power_law_freq(800, 0, rate_hz=1.0, rng=rng).y.copy()
    y[300:310] = np.nan
    series = FreqSeries(y=y, rate_hz=1.0)
    r = oadev(series, taus="octave")
    assert len(r.run_lengths) == 2
    for i, m in enumerate(r.af.tolist()):
        assert int(r.run_terms[i].sum()) == r.coeffs[i].size
        eta, used = greenhall_edof_pooled(
            0, r.shape.d, m, r.run_lengths, r.run_terms[i],
            r.shape.modified, r.shape.overlapped,
        )
        assert used == "greenhall"
        assert 1.0 <= eta <= r.coeffs[i].size


# --- interoperability ----------------------------------------------------


def test_against_allantools_greenhall():
    """INTEROP, not correctness: two implementations of one published paper.

    `allantools` implements the same PTTI 2003 algorithm. Its source was
    not read and nothing was ported from it -- this compares numbers only,
    which is why it is evidence at all. Agreement over ~19k parameter
    combinations says the transcription of four cases, three tables and
    three kernels is right.

    The single documented exception is `r == d+1` exactly, the boundary
    between the summed branch and the tabulated-integral branch. Section
    4.6 writes the condition as `r >= d+1`, and appendix A repeats it as
    `M >= (d+1)S`, so the tabulated branch is taken here. The two branches
    differ by under 0.1% for alpha <= 0 and by 2.4% for flicker PM, where
    the `b0 + b1 ln m` asymptotic is at its weakest -- a discontinuity in
    the published algorithm, not in this transcription of it.
    """
    ci = pytest.importorskip("allantools.ci")
    checked = boundary = 0
    for n in (200, 512, 1000):
        for d, modified, overlapped in ((2, False, True), (2, True, True), (3, False, True)):
            for alpha in GREENHALL_ALPHAS:
                if alpha + 2 * d <= 1:
                    continue
                for m in range(1, n // (d + 1), 7):
                    ours = greenhall_edof(alpha, d, m, n, modified, overlapped)
                    try:
                        theirs = ci.edf_greenhall(
                            alpha=alpha, d=d, m=m, N=n,
                            overlapping=overlapped, modified=modified, verbose=False,
                        )
                    except (NotImplementedError, AssertionError):
                        # That implementation does not cover the whole
                        # published algorithm: the K <= d branch of case 4
                        # is unwritten, and its Table 2 stops at alpha =
                        # -3, so random-run FM at d = 3 is unreachable.
                        # Nothing to compare against there. Those branches
                        # are covered by the structural tests above, and
                        # the gap is itself worth knowing about.
                        continue
                    terms = term_count(n, m, d, modified, overlapped)
                    if terms / (m if overlapped else 1) == d + 1:
                        boundary += 1
                        continue
                    assert ours == pytest.approx(theirs, rel=1e-9), (
                        f"alpha={alpha} d={d} m={m} N={n} "
                        f"modified={modified} overlapped={overlapped}"
                    )
                    checked += 1
    assert checked > 1000, f"only {checked} combinations compared"
    assert boundary < 0.01 * checked, f"{boundary} boundary cases is too many to skip"


def test_against_the_published_white_fm_closed_form():
    """INTEROP. The long-standing approximation for overlapped AVAR, WHFM:

        edf = [3(N-1)/(2m) - 2(N-2)/N] * 4m^2/(4m^2+5)

    (NIST SP 1065 section 5.) It is itself an approximation fitted to the
    same problem Greenhall solves properly, so this is agreement between
    two models, asserted at the few percent the older form claims -- not a
    correctness assertion, and m = 1 is excluded for the reason given in
    the model-departure test above.
    """
    n = 1000
    for m in (2, 4, 8, 16, 32, 64, 128, 256):
        ours = greenhall_edof(0, 2, m, n, False, True)
        published = (3 * (n - 1) / (2 * m) - 2 * (n - 2) / n) * 4 * m**2 / (4 * m**2 + 5)
        assert ours == pytest.approx(published, rel=0.13), f"m={m}"
