"""Cross-variance: alignment, the identities, and the error bar's calibration.

Three kinds of check here, in rising cost.

*Identities*, which are exact and cheap. `cross_variance(a, a)` must
return the auto-variance bit for bit and an `sem` equal to the
chi-square one, because the Isserlis expression reduces to `2 sigma^4 /
eta` when the two channels are one channel. Getting that wrong by a
factor would otherwise be invisible.

*Refusals*, one per way an alignment can be silently wrong: a rate
difference that accumulates past half a sample, an epoch off the grid, no
overlap, mismatched domains, and a statistic that is not a variance.

*Calibration*, which is the expensive one and the only real evidence that
the error bar means anything. `sem` claims to be the standard deviation of
the estimate; that is measurable by generating many realizations and
taking the scatter, and this file measures it. The rejected
Bartlett-on-the-products estimator is kept here and measured beside it, so
the reason it was rejected stays a live number rather than a paragraph in
a docstring.
"""

import numpy as np
import pytest

from stabilis.multi import common_support, cross_sem, cross_variance, overlap
from stabilis.multi.align import MAX_GRID_OFFSET_SAMPLES, MAX_SLIP_SAMPLES
from stabilis.stats import oadev
from stabilis.stats.edof import eta1
from stabilis.synth import power_law_phase
from stabilis.types import FreqSeries, PhaseSeries

RATE = 1.0
H_CLOCK = 1e-24
H_INSTRUMENT = 1e-22


def two_channels(rng, n, alpha_clock=0, alpha_inst=2,
                 h_clock=H_CLOCK, h_inst=H_INSTRUMENT):
    """One clock seen through two channels with independent noise.

    Returns `(a, b, clock)` so a test can compare the recovered common part
    against the record that actually generated it -- the one comparison a
    cross-variance is really making.
    """
    clock = power_law_phase(n, alpha_clock, RATE, rng, h_clock)
    na = power_law_phase(n, alpha_inst, RATE, rng, h_inst)
    nb = power_law_phase(n, alpha_inst, RATE, rng, h_inst)
    return (PhaseSeries(x=clock.x + na.x, rate_hz=RATE),
            PhaseSeries(x=clock.x + nb.x, rate_hz=RATE),
            clock)


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

def test_crossing_a_record_with_itself_is_the_auto_variance():
    """Exact, to the last bit. `mean(c*c)` is `mean(c**2)`, same array."""
    rng = np.random.default_rng(1)
    a, _, _ = two_channels(rng, 1 << 11)
    cross = cross_variance(a, a)
    auto = oadev(a)
    assert np.array_equal(cross.af, auto.af)
    assert np.array_equal(cross.var, auto.var)
    assert np.allclose(cross.correlation, 1.0)
    assert not cross.negative.any()


def test_the_self_cross_error_bar_is_the_chi_square_one():
    """`sem` reduces to `sigma^2 sqrt(2/eta)` when b is a.

    The cross error bar is not a parallel construction to the chi-square
    interval -- it is the same one, seen through Isserlis. If this drifts,
    the two paths have stopped agreeing about what a degree of freedom is.
    """
    rng = np.random.default_rng(2)
    a, _, _ = two_channels(rng, 1 << 11)
    cross = cross_variance(a, a)
    auto = oadev(a)
    eta = np.array([eta1(auto.coeffs[i], int(m)) for i, m in enumerate(auto.af)])
    assert np.allclose(cross.sem, auto.var * np.sqrt(2.0 / eta), rtol=1e-12)
    assert np.allclose(cross.eta, eta, rtol=1e-12)


def test_cross_sem_is_symmetric_in_its_two_channels():
    rng = np.random.default_rng(3)
    a, b, _ = two_channels(rng, 1 << 11)
    ab, ba = cross_variance(a, b), cross_variance(b, a)
    assert np.allclose(ab.var, ba.var, rtol=1e-14)
    assert np.allclose(ab.sem, ba.sem, rtol=1e-14)
    assert np.allclose(ab.auto_a, ba.auto_b, rtol=1e-14)


def test_the_cross_estimate_sees_under_the_channel_noise_floor():
    """The whole point, stated as an assertion.

    A clock well below its channels' noise is invisible to the
    auto-variance and recovered by the cross. Asserted only where the
    rejection is comfortable, since at long tau the two converge -- there
    the instrument noise has averaged down anyway and there is nothing
    left to reject.
    """
    rng = np.random.default_rng(4)
    a, b, clock = two_channels(rng, 1 << 13)
    cross = cross_variance(a, b)
    truth = oadev(clock)
    short = cross.af <= 8

    # The single-channel answer is high by a factor the cross removes.
    assert np.all(cross.auto_a[short] > 1.5 * truth.var[short])
    assert np.allclose(cross.var[short], truth.var[short], rtol=0.15)
    # And it says so: 10 dB of rejection or better where that holds.
    assert np.all(cross.rejection_db[short] > 10.0)


def test_independent_records_cross_to_nothing_and_go_negative():
    """No common clock: the estimate scatters about zero, and says so.

    Negative points are the expected behaviour, not a defect, so the test
    demands them. It also demands `dev` be NaN there rather than zero --
    the failure mode is a log-axis plot growing a noise floor that does
    not exist.
    """
    rng = np.random.default_rng(5)
    a = power_law_phase(1 << 12, 0, RATE, rng, H_CLOCK)
    b = power_law_phase(1 << 12, 0, RATE, rng, H_CLOCK)
    cross = cross_variance(a, b)
    assert cross.negative.any()
    assert np.all(np.isnan(cross.dev[cross.negative]))
    # Consistent with zero: every point inside three of its own sems. That
    # is the statement that survives to the last tau, where eta is single
    # figures and a lone realization can wander a long way in correlation.
    assert np.all(np.abs(cross.var) < 3.0 * cross.sem)
    # Where the estimate is well determined it is also tiny beside the
    # autos it came from.
    well_determined = cross.eta > 100.0
    assert np.all(np.abs(cross.correlation[well_determined]) < 0.15)


def test_the_rejection_reduces_to_five_log_ten_eta():
    """`rejection_db` is `5 log10(eta)` when the channels dominate.

    Not a separate formula -- a consequence of the sem. Substituting it
    into the definition gives the identity asserted first, exactly:

        rejection = 5 log10(eta) - 5 log10(1 + rho^2)

    and the second term vanishes for uncorrelated channels, leaving the
    classic cross-spectrum law. Both are checked, because the exact
    identity is what catches an sem that changed meaning and the limit is
    what says the number means what a metrologist expects.
    """
    rng = np.random.default_rng(6)
    a = power_law_phase(1 << 12, 0, RATE, rng, H_CLOCK)
    b = power_law_phase(1 << 12, 0, RATE, rng, H_CLOCK)
    cross = cross_variance(a, b)
    exact = 5.0 * np.log10(cross.eta) - 5.0 * np.log10(1.0 + cross.correlation**2)
    assert np.allclose(cross.rejection_db, exact, rtol=1e-12)
    weak = np.abs(cross.correlation) < 0.1
    assert weak.any()
    assert np.allclose(cross.rejection_db[weak],
                       5.0 * np.log10(cross.eta[weak]), atol=0.05)


def test_cross_sem_matches_its_closed_form():
    assert cross_sem(4.0, 9.0, 0.0, 2.0) == pytest.approx(np.sqrt(18.0))
    # Sign of the cross term cannot matter; only its square appears.
    assert cross_sem(4.0, 9.0, -3.0, 2.0) == cross_sem(4.0, 9.0, 3.0, 2.0)
    # The auto case, written out.
    assert cross_sem(4.0, 4.0, 4.0, 8.0) == pytest.approx(4.0 * np.sqrt(2.0 / 8.0))


@pytest.mark.parametrize("estimator", ["oadev", "adev", "mdev", "tdev", "ohdev"])
def test_the_whole_family_gets_a_cross_form_for_free(estimator):
    """No estimator-specific code exists, so this is the test that says so."""
    rng = np.random.default_rng(7)
    a, b, _ = two_channels(rng, 1 << 10)
    cross = cross_variance(a, b, estimator, max_af=32)
    auto = cross_variance(a, a, estimator, max_af=32)
    assert cross.kind == f"cross_{estimator}"
    assert np.array_equal(auto.var, getattr(
        __import__("stabilis.stats", fromlist=[estimator]), estimator)(
            a, max_af=32).var)
    assert np.all(np.abs(cross.correlation) <= 1.5)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def test_overlap_crops_to_the_shared_window():
    rng = np.random.default_rng(8)
    x = power_law_phase(1 << 10, 0, RATE, rng).x
    a = PhaseSeries(x=x, rate_hz=RATE, epoch=0.0)
    b = PhaseSeries(x=x, rate_hz=RATE, epoch=10.0)
    al = overlap([a, b])
    assert al.epoch == 10.0
    assert al.offsets == (10, 0)
    assert al.trimmed == (0, 10)
    assert al.n == a.n - 10
    assert np.array_equal(al.series[0].x, x[10:])
    assert np.array_equal(al.series[1].x, x[:-10])
    assert all(s.epoch == 10.0 for s in al.series)


def test_overlap_of_identical_records_changes_nothing():
    """`cross_variance` calls it unconditionally, so it must be a no-op."""
    rng = np.random.default_rng(9)
    a = power_law_phase(1 << 8, 0, RATE, rng)
    al = overlap([a, a])
    assert al.offsets == (0, 0) and al.trimmed == (0, 0)
    assert al.notes == ()
    assert np.array_equal(al.series[0].x, a.x)


def test_a_shifted_record_still_cross_correlates_with_itself():
    """The end-to-end statement that cropping puts the terms back in step.

    Two views of one record offset by a whole number of samples must cross
    to the auto-variance of their common stretch. Nothing else in the file
    tests that `overlap` shifts in the right *direction*, which is a bug
    that would leave every number plausible.
    """
    rng = np.random.default_rng(10)
    x = power_law_phase(1 << 11, 0, RATE, rng).x
    a = PhaseSeries(x=x[:-32], rate_hz=RATE, epoch=0.0)
    b = PhaseSeries(x=x[32:], rate_hz=RATE, epoch=32.0)
    cross = cross_variance(a, b, max_af=64)
    common = PhaseSeries(x=x[32:-32], rate_hz=RATE)
    assert np.allclose(cross.var, oadev(common, max_af=64).var, rtol=1e-12)


def test_a_rate_difference_is_judged_by_the_slip_it_accumulates():
    """The same 2.4e-7 is harmless in a short record and fatal in a long one."""
    rng = np.random.default_rng(11)
    off_rate = RATE * (1.0 + 2.4e-7)
    short = power_law_phase(1 << 10, 0, RATE, rng).x
    ok = overlap([PhaseSeries(x=short, rate_hz=RATE),
                  PhaseSeries(x=short, rate_hz=off_rate)])
    assert ok.n == short.size
    assert any("slip" in note for note in ok.notes)

    n_long = int(MAX_SLIP_SAMPLES / 2.4e-7) + 1000
    long = np.zeros(n_long)
    with pytest.raises(ValueError, match="drift of"):
        overlap([PhaseSeries(x=long, rate_hz=RATE),
                 PhaseSeries(x=long, rate_hz=off_rate)])


def test_a_sub_sample_epoch_offset_is_refused_not_rounded():
    rng = np.random.default_rng(12)
    x = power_law_phase(1 << 8, 0, RATE, rng).x
    a = PhaseSeries(x=x, rate_hz=RATE, epoch=0.0)
    b = PhaseSeries(x=x, rate_hz=RATE, epoch=10.4)
    with pytest.raises(ValueError, match="off the common grid"):
        overlap([a, b])
    # Float64 slop on an MJD-scale epoch is not a sub-sample offset.
    fine = PhaseSeries(x=x, rate_hz=RATE,
                       epoch=10.0 + MAX_GRID_OFFSET_SAMPLES / 10.0)
    assert overlap([a, fine]).offsets == (10, 0)


def test_records_that_never_overlapped_are_refused():
    rng = np.random.default_rng(13)
    x = power_law_phase(1 << 8, 0, RATE, rng).x
    with pytest.raises(ValueError, match="do not overlap"):
        overlap([PhaseSeries(x=x, rate_hz=RATE, epoch=0.0),
                 PhaseSeries(x=x, rate_hz=RATE, epoch=1e6)])


def test_phase_and_frequency_cannot_be_aligned_with_each_other():
    rng = np.random.default_rng(14)
    with pytest.raises(TypeError, match="across domains"):
        overlap([power_law_phase(64, 0, RATE, rng),
                 FreqSeries(y=np.zeros(64), rate_hz=RATE)])


def test_common_support_makes_the_gaps_the_union_of_both():
    a = PhaseSeries(x=np.array([0.0, 1.0, np.nan, 3.0, 4.0]), rate_hz=RATE)
    b = PhaseSeries(x=np.array([0.0, np.nan, 2.0, 3.0, 4.0]), rate_hz=RATE)
    ca, cb = common_support([a, b])
    missing = ~np.isfinite(ca.x)
    assert list(missing) == [False, True, True, False, False]
    assert np.array_equal(missing, ~np.isfinite(cb.x))
    assert any(step.action == "common_support" for step in ca.provenance.steps)


def test_common_support_needs_equal_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        common_support([PhaseSeries(x=np.zeros(8), rate_hz=RATE),
                        PhaseSeries(x=np.zeros(9), rate_hz=RATE)])


def test_differing_gaps_do_not_derail_the_pairing():
    """Two records with unrelated dropouts still produce pairable terms.

    Without `common_support` the two coefficient arrays come out different
    lengths and the estimator raises -- or worse, they come out the same
    length holding different terms. The check is that the answer equals
    what the union-gapped records give when done by hand.
    """
    rng = np.random.default_rng(15)
    a, b, _ = two_channels(rng, 1 << 10)
    xa, xb = np.array(a.x), np.array(b.x)
    xa[[100, 101, 500]] = np.nan
    xb[[300, 700, 701]] = np.nan
    cross = cross_variance(PhaseSeries(x=xa, rate_hz=RATE),
                           PhaseSeries(x=xb, rate_hz=RATE), max_af=32)
    union = ~np.isfinite(xa) | ~np.isfinite(xb)
    ua, ub = np.array(xa), np.array(xb)
    ua[union] = np.nan
    ub[union] = np.nan
    ra = oadev(PhaseSeries(x=ua, rate_hz=RATE), max_af=32)
    rb = oadev(PhaseSeries(x=ub, rate_hz=RATE), max_af=32)
    by_hand = [ra.scale * float((ra.coeffs[i] * rb.coeffs[i]).mean())
               for i in range(ra.af.size)]
    assert np.allclose(cross.var, by_hand, rtol=1e-14)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_statistic_that_is_not_a_variance_has_no_cross_form():
    rng = np.random.default_rng(16)
    a, b, _ = two_channels(rng, 1 << 9)
    with pytest.raises(ValueError, match="no cross form"):
        cross_variance(a, b, "mtie", max_af=16)


def test_a_noise_type_edof_is_refused():
    """greenhall and totvar are derived for a sum of squares, not products."""
    rng = np.random.default_rng(17)
    a, b, _ = two_channels(rng, 1 << 9)
    with pytest.raises(ValueError, match="edof_method must be one of"):
        cross_variance(a, b, max_af=16, edof_method="greenhall")


def test_stats_never_imports_multi():
    """The same one-way dependency `bench/` has, for the same reason.

    `multi/` is a client of the estimators: it calls them and multiplies
    what they return. The moment `stats/` reaches back, every ADEV anyone
    computes acquires a dependency on the multi-channel layer, and the
    coefficient contract stops being the interface between them.
    """
    import ast
    import importlib
    import inspect
    import pkgutil

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
        offenders = [n for n in names if "multi" in n]
        assert not offenders, f"stats.{info.name} imports multi: {offenders}"
        checked += 1
    assert checked >= 5, "expected to have scanned the whole stats package"


def test_the_independence_assumption_is_recorded_on_every_call():
    """It cannot be checked from the data, so it is carried instead."""
    rng = np.random.default_rng(18)
    a, b, _ = two_channels(rng, 1 << 9)
    cross = cross_variance(a, b, max_af=16)
    steps = [s.action for s in cross.provenance.steps]
    assert "independence_assumed" in steps
    assert "cross_variance" in steps


def test_a_short_record_falls_back_to_eta3_and_says_which():
    """Past the point where a sample autocorrelation means anything.

    `max_af` is given explicitly to push past the tau grid's own policy
    limit, which otherwise keeps every point comfortably above
    `MIN_TERMS_FOR_ETA1` -- the fallback is reachable but not by accident,
    which is the intended behaviour and worth pinning.
    """
    rng = np.random.default_rng(19)
    a, b, _ = two_channels(rng, 40)
    cross = cross_variance(a, b, taus="all", max_af=20)
    assert set(cross.edof_method) <= {"eta1", "eta3"}
    assert cross.edof_method[0] == "eta1"
    assert cross.edof_method[-1] == "eta3"
    # eta3 is a lower bound on the degrees of freedom, so the fallback must
    # widen the bar rather than narrow it.
    assert cross.eta[-1] <= cross.n_terms[-1]


# ---------------------------------------------------------------------------
# Calibration of the error bar
# ---------------------------------------------------------------------------

def bartlett_sem_on_products(p, m, lag_limit=4):
    """The estimator that was rejected. Kept here so the rejection is measured.

    Bartlett-weighted sum of the *products'* own autocovariance,

        Var(mean p) = (1/M) sum_{k=-K}^{K} (1 - |k|/M) gamma(k),  K = 4m

    which is the Newey-West form `eta1` embodies and which assumes nothing
    about the distribution. It is the obvious thing to do and it is
    optimistic by a factor of three at the long-tau end; see
    `theory/cross.md` and the test below.
    """
    mm = int(p.size)
    kmax = int(min(lag_limit * m, mm - 1))
    d = p - p.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * mm)))
    spec = np.fft.rfft(d, nfft)
    gamma = np.fft.irfft(spec * np.conj(spec), nfft)[: kmax + 1] / mm
    weights = 1.0 - np.arange(1, kmax + 1) / mm
    total = gamma[0] + 2.0 * float(np.dot(weights, gamma[1:]))
    return float(np.sqrt(max(total, 0.0) / mm))


def measure_scatter(seed, n, trials, max_af, alpha_clock=0, alpha_inst=2,
                    h_clock=H_CLOCK, h_inst=H_INSTRUMENT):
    """Monte-Carlo truth for the sem, and the rejected estimator beside it.

    Returns `(af, ratio_isserlis, ratio_bartlett)`, each ratio being the
    mean predicted sem over the measured scatter of the estimate. One means
    the error bar is right; below one means it is too narrow.
    """
    rng = np.random.default_rng(seed)
    var, isserlis, bartlett = [], [], []
    for _ in range(trials):
        a, b, _ = two_channels(rng, n, alpha_clock, alpha_inst, h_clock, h_inst)
        cross = cross_variance(a, b, max_af=max_af)
        ra = oadev(a, max_af=max_af)
        rb = oadev(b, max_af=max_af)
        var.append(cross.var)
        isserlis.append(cross.sem)
        bartlett.append([
            ra.scale * bartlett_sem_on_products(ra.coeffs[i] * rb.coeffs[i],
                                                int(m))
            for i, m in enumerate(ra.af)])
        af = cross.af
    var = np.array(var)
    scatter = var.std(axis=0, ddof=1)
    return af, np.array(isserlis).mean(axis=0) / scatter, \
        np.array(bartlett).mean(axis=0) / scatter


@pytest.mark.slow
def test_the_error_bar_is_the_scatter_of_the_estimate():
    """`sem` against Monte-Carlo truth, which is what it claims to be.

    Tolerance widens with the measurement's own standard error: 300 trials
    determines a standard deviation to about 4%, so a band of +/-20% is
    several times that and still tight enough to catch a factor.
    """
    af, isserlis, _ = measure_scatter(seed=101, n=1 << 12, trials=300,
                                      max_af=256)
    for m, ratio in zip(af, isserlis):
        assert 0.8 < ratio < 1.25, f"af={m}: sem/scatter = {ratio:.3f}"


@pytest.mark.slow
def test_the_bartlett_estimator_is_optimistic_where_isserlis_is_not():
    """The measurement behind the choice in `theory/cross.md`.

    Both are computed on the same realizations, so this is a paired
    comparison and not two separate claims. The assertion is deliberately
    two-sided: the rejected estimator must still be *right at short tau*,
    because if it were wrong everywhere the interesting finding -- that it
    fails only where the products' tails matter -- would not be the reason.
    """
    af, isserlis, bartlett = measure_scatter(seed=102, n=1 << 12, trials=300,
                                             max_af=256)
    short, long = af <= 8, af >= 256
    assert np.all(np.abs(bartlett[short] - 1.0) < 0.2)
    assert np.all(bartlett[long] < 0.75), f"bartlett {bartlett}"
    assert np.all(isserlis[long] > 0.85), f"isserlis {isserlis}"
    # The gap opens as tau grows; it is not a constant offset.
    assert (isserlis - bartlett)[long].min() > (isserlis - bartlett)[short].max()


@pytest.mark.slow
def test_asymmetric_channels_widen_the_bar_rather_than_narrowing_it():
    """The equal-shape approximation at its worst, and which way it errs.

    White PM in one channel and flicker PM in the other, three decades
    apart in level, is the case the Isserlis collapse is least entitled to.
    What matters is not that it stays exact but that it fails
    conservatively.
    """
    rng = np.random.default_rng(103)
    var, sem = [], []
    for _ in range(200):
        clock = power_law_phase(1 << 12, 0, RATE, rng, H_CLOCK)
        a = PhaseSeries(x=clock.x + power_law_phase(1 << 12, 2, RATE, rng, 1e-21).x,
                        rate_hz=RATE)
        b = PhaseSeries(x=clock.x + power_law_phase(1 << 12, -1, RATE, rng, 1e-25).x,
                        rate_hz=RATE)
        cross = cross_variance(a, b, max_af=256)
        var.append(cross.var)
        sem.append(cross.sem)
    ratio = np.array(sem).mean(axis=0) / np.array(var).std(axis=0, ddof=1)
    assert np.all(ratio > 0.85), f"error bars too narrow: {ratio}"
    assert np.all(ratio < 1.5), f"error bars uselessly wide: {ratio}"
