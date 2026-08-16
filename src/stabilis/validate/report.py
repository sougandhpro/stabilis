"""`stabilis validate` -- validation as a deliverable, not just green CI.

ISO/IEC 17025:2017 clause 7.11.2 requires software to be validated by the
laboratory. NOTE 2 exempts commercial off-the-shelf software "in general
use within its designed application range", which a new open-core tool
fails twice over: it is neither commercial nor in general use. So an
accredited lab cannot adopt this library on its own say-so, and shipped
validation evidence is not a nicety -- it is the thing that makes adoption
possible at all.

What makes this evidence rather than decoration:

* **Every claim states its tolerance and why that tolerance.** A pass at an
  unstated tolerance says nothing.
* **It can fail.** `run(self_test=True)` injects a check that is designed
  to fail, and a test asserts the report then reports failure. A validation
  report that cannot fail is worthless.
* **It is reproducible.** Fixed seeds, no network, and every input either
  generated from a published recursion or hashed. The digest covers the
  claims and the measured values, so a third party regenerating from a
  clean checkout can compare one string.
* **Interop and correctness are labelled differently.** Every published
  ADEV table traces back to Stable32, so matching one proves
  interoperability. Correctness rests on the analytic and certified tiers.

See theory pages, and `references.md` for the epistemic caveat in full.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np

from .. import __version__
from .._specfun import chi2_ppf, gammainc_q
from ..reference import (
    EXPECTED_NBS14_1000POINT,
    INTEROP_RTOL,
    NBS14_LEHMER_MODULUS,
    nbs14_1000point,
    nbs14_1000point_integers,
    nbs14_9point,
)
from ..stats import ONE_SIGMA, TWO_SIGMA, adev, confidence, mdev, oadev, ohdev, tdev
from ..stats.bias import deviation_bias_factor
from ..stats.core import StabilityResult
from ..stats.edof import eta1
from ..stats.greenhall import greenhall_edof
from ..synth import (_kasdin_walter_impulse, _unit_h_alpha, power_law_freq,
                     power_law_phase)
from ..types import FreqSeries, PhaseSeries
from .certified import (
    certified_deviation,
    exact_avar,
    exact_mvar,
    exact_ohvar,
    exact_oavar,
    exact_phase_from_freq,
    exact_tvar,
    lre,
)

__all__ = ["Check", "Report", "run", "to_json", "to_html"]

TIERS = ("identity", "certified", "coverage", "interop", "self-test")

ESTIMATORS = [
    ("adev", adev, exact_avar),
    ("oadev", oadev, exact_oavar),
    ("mdev", mdev, exact_mvar),
    ("tdev", tdev, exact_tvar),
    ("ohdev", ohdev, exact_ohvar),
]

# Published so the Monte-Carlo tiers can be repeated by a third party.
SEEDS = {"coverage": 22, "gaps": 7, "stream": 3}

# Coverage cannot be measured meaningfully with fewer than this many
# realizations -- the estimate's own standard error would swamp the effect
# being checked. A lower request is raised to this rather than producing a
# number that looks measured and is not.
MIN_COVERAGE_TRIALS = 300


@dataclass(frozen=True)
class Check:
    """One assertion, with the tolerance it was made at and why."""

    tier: str
    name: str
    claim: str
    tolerance: str
    measured: str
    passed: bool

    def __post_init__(self) -> None:
        # numpy comparisons yield np.bool_, which json refuses. The report
        # is serialized and hashed, so coerce once here rather than at
        # every call site and hope none is missed.
        object.__setattr__(self, "passed", bool(self.passed))
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}")


@dataclass(frozen=True)
class Report:
    checks: tuple[Check, ...]
    environment: dict
    seeds: dict
    digest: str
    generated: str

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def by_tier(self, tier: str) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.tier == tier)


# --------------------------------------------------------------------------
# tiers
# --------------------------------------------------------------------------


def _identities() -> list[Check]:
    """Tier 1. Machine precision, no external data. Correctness lives here."""
    out = []
    n = 1024
    i = np.arange(n, dtype=np.float64)
    ps = PhaseSeries(i**2 / 2.0, rate_hz=1.0)  # y = t, drift D = 1

    a = oadev(ps, taus="octave", keep_coeffs=False)
    out.append(Check(
        "identity", "linear drift: avar = D^2 tau^2 / 2",
        "A pure linear frequency drift has a quadratic phase, so the second "
        "difference is the constant D*tau^2 and the variance is exactly "
        "D^2*tau^2/2. Validates the Allan normalization and the drift "
        "convention together.",
        "bit-exact. The second differences are integers here, so any "
        "discrepancy at all is a defect rather than rounding.",
        "equal to the last bit" if np.array_equal(a.var, a.tau_s**2 / 2.0)
        else f"max rel {np.max(np.abs(a.var/(a.tau_s**2/2.0) - 1)):.2e}",
        bool(np.array_equal(a.var, a.tau_s**2 / 2.0)),
    ))

    h = ohdev(ps, taus="octave", keep_coeffs=False)
    out.append(Check(
        "identity", "linear drift: hvar = 0",
        "The third difference annihilates linear frequency drift exactly, "
        "which is why the Hadamard variance needs no detrending.",
        "bit-exact zero, not merely small. The third differences of an "
        "exactly quadratic phase are integers that cancel to zero, so any "
        "nonzero result is a coefficient error rather than rounding.",
        "exactly zero" if np.array_equal(h.var, np.zeros_like(h.var))
        else f"max {np.max(np.abs(h.var)):.3e}",
        bool(np.array_equal(h.var, np.zeros_like(h.var))),
    ))

    y = np.array([3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0, -5.0, 3.0, 5.0, 8.0])
    base = oadev(FreqSeries(y, rate_hz=1.0), taus="all", keep_coeffs=False).var
    shifted = oadev(FreqSeries(y + 1000.0, rate_hz=1.0), taus="all",
                    keep_coeffs=False).var
    out.append(Check(
        "identity", "constant frequency offset leaves avar unchanged",
        "AVAR measures instability, not offset. A constant added to y must "
        "cancel in the first difference.",
        "bit-exact. The offset is 1000 against data of order 1, so a "
        "cancellation that were merely approximate would show at the "
        "third digit rather than the fifteenth.",
        "unchanged" if np.array_equal(base, shifted) else "changed",
        bool(np.array_equal(base, shifted)),
    ))

    rng = np.random.default_rng(8)
    x = np.cumsum(rng.standard_normal(300))
    b = oadev(PhaseSeries(x, rate_hz=1.0), taus="octave", keep_coeffs=False).var
    s = oadev(PhaseSeries(x * 8.0, rate_hz=1.0), taus="octave", keep_coeffs=False).var
    out.append(Check(
        "identity", "scale equivariance: x -> 8x gives var -> 64 var",
        "Scaling the phase scales the variance by the square. A power of "
        "two is used so the expected result is exactly representable.",
        "bit-exact, because 8 and 64 are powers of two and so scale a "
        "float64 without touching its mantissa. Any other factor would "
        "make this a rounding test rather than an algebraic one.",
        "exact" if np.array_equal(s, b * 64.0) else "differs",
        bool(np.array_equal(s, b * 64.0)),
    ))

    rng = np.random.default_rng(6)
    fs = FreqSeries(rng.standard_normal(400), rate_hz=4.0)
    md = mdev(fs, taus="octave", keep_coeffs=False)
    td = tdev(fs, taus="octave", keep_coeffs=False)
    rel = float(np.max(np.abs(td.dev / (td.tau_s / np.sqrt(3.0) * md.dev) - 1.0)))
    out.append(Check(
        "identity", "tdev = tau/sqrt(3) * mdev",
        "TDEV is a rescaling of MDEV by definition, so this checks the two "
        "scale factors against each other.",
        "1e-15 relative. Not bit-exact because sqrt(3) is irrational, so "
        "the two sides cannot round identically however correct they are.",
        f"max rel {rel:.2e}",
        rel < 1e-15,
    ))

    rng = np.random.default_rng(5)
    fs = FreqSeries(rng.standard_normal(300), rate_hz=1.0)
    a1 = adev(fs, taus="all", keep_coeffs=False).dev_at(1)
    o1 = oadev(fs, taus="all", keep_coeffs=False).dev_at(1)
    m1 = mdev(fs, taus="all", keep_coeffs=False).dev_at(1)
    worst = max(abs(o1 - a1), abs(m1 - a1)) / a1
    out.append(Check(
        "identity", "adev = oadev = mdev at averaging factor 1",
        "At m=1 there is no overlap to exploit and no inner average to "
        "take, so the three estimators must coincide.",
        "1e-15 relative -- about one ulp. The three reach the same "
        "arithmetic by different code paths, so anything looser would "
        "hide a real divergence between them.",
        f"max rel {worst:.2e}",
        worst < 1e-15,
    ))

    rng = np.random.default_rng(seed=SEEDS["gaps"])
    y = power_law_freq(200, 0, rate_hz=1.0, rng=rng).y
    gapped = y.copy()
    gapped[100] = np.nan
    clean = oadev(FreqSeries(y, rate_hz=1.0), taus="all", keep_coeffs=False)
    holed = oadev(FreqSeries(gapped, rate_hz=1.0), taus="all", keep_coeffs=False)
    losses = {int(m): int(clean.n_terms[list(clean.af).index(m)] - t)
              for m, t in zip(holed.af, holed.n_terms)}
    ok = all(v == 2 * k for k, v in losses.items())
    out.append(Check(
        "identity", "one gap costs exactly 2m terms",
        "A missing frequency sample invalidates only the terms whose window "
        "spans it. Anything more means the record is being truncated.",
        "exact integer equality at every averaging factor. Term counts "
        "are integers, so there is no tolerance to choose: either the "
        "arithmetic of the gap is right or it is not.",
        "exactly 2m at every m" if ok else f"losses {losses}",
        ok,
    ))

    p = 0.16
    got = chi2_ppf(p, 2.0)
    want = -2.0 * np.log1p(-p)
    out.append(Check(
        "identity", "chi2_ppf(p, 2) = -2 ln(1-p)",
        "The one chi-square quantile with a closed form. Validates the "
        "inverse-CDF iteration against arithmetic that needs no iteration.",
        "1e-13 relative. Two orders looser than machine precision because "
        "the inversion terminates on the CDF rather than on the quantile, "
        "and the density carries that difference across.",
        f"rel {abs(got/want - 1):.2e}",
        abs(got / want - 1.0) < 1e-13,
    ))

    c4 = deviation_bias_factor(2.0)
    out.append(Check(
        "identity", "deviation bias c4(2) = sqrt(pi)/2",
        "E[sigma_hat]/sigma at two degrees of freedom, the closed form that "
        "quantifies the 11.4% shortfall of the last ADEV point.",
        "1e-15 relative. Both sides reduce to elementary constants, so "
        "nothing here should cost more than an ulp.",
        f"{c4:.10f}, rel {abs(c4/(np.sqrt(np.pi)/2) - 1):.2e}",
        abs(c4 / (np.sqrt(np.pi) / 2) - 1.0) < 1e-15,
    ))
    out.append(_synth_amplitude_identity())
    out.extend(_streaming_identities())
    return out


def _streaming_identities() -> list[Check]:
    """The two M6 identities: the surface and the accumulator against batch.

    Both are checks that a *different* code path returns what the batch
    estimator returns, which is the only validation either subsystem can
    have -- neither Stable32 nor `allantools` publishes a dynamic Allan
    variance, and a streaming estimator that agreed with a reference table
    but not with its own batch form would be the wrong kind of right.

    The first is asserted at exactly zero because it is the same call on
    the same data; the second cannot be, because an accumulator sums
    sequentially where `np.mean` sums pairwise. See `theory/davar.md` and
    `theory/stream.md`.
    """
    from ..stream import davar, feed, online_avar, online_result, replay

    out = []
    ps = power_law_phase(4095, 0, 1.0, np.random.default_rng(SEEDS["stream"]), 1e-24)
    batch = oadev(ps, taus="octave")

    surface = davar(ps, window=ps.n)
    same = np.array_equal(surface.var[0], batch.var)
    out.append(Check(
        "identity", "davar over the whole record is the batch estimator",
        "A dynamic Allan variance whose window is the entire record must "
        "be the ordinary overlapping Allan variance of that record. It is "
        "built by calling the registered estimator on each window rather "
        "than reimplementing it, so this holds bit for bit -- and any "
        "shortcut in the windowing (a re-zeroed window, a re-derived tau "
        "grid) would break exactness while still agreeing to 1e-15.",
        "exact. Zero difference in every bit of every point; nothing "
        "looser would detect the failure modes this is here to catch.",
        f"{surface.af.size} taus, "
        + ("identical in every bit" if same
           else f"max rel {np.abs(surface.var[0] / batch.var - 1).max():.2e}"),
        same,
    ))

    acc = online_avar(1.0, batch.af)
    feed(acc, replay(ps.x, 1.0))
    online = online_result(acc)
    rel = np.abs(online.var / batch.var - 1.0)
    ulp = rel.max() / np.finfo(float).eps
    out.append(Check(
        "identity", "a replayed stream is the batch estimator",
        "The record fed sample by sample through the O(max tau) "
        "accumulator, against the same record analysed in batch. The "
        "coefficients are bit-identical by construction -- the "
        "association order is copied from the batch expression -- so what "
        "is measured here is summation alone: Neumaier-compensated in the "
        "stream against pairwise in numpy.",
        "4 ulp, and independent of record length. An uncompensated "
        "accumulator measures 15 ulp at N = 4096 and 74 at N = 65536, "
        "growing as O(N); this bound is what distinguishes the two.",
        f"max {ulp:.2f} ulp over {online.af.size} taus, "
        f"{int((online.var == batch.var).sum())} of them identical",
        ulp < 4.0 and np.array_equal(online.n_terms, batch.n_terms),
    ))

    x = np.array(ps.x)
    x[900:906] = np.nan
    rebuilt = _collected(x)
    exact = np.array_equal(rebuilt, x, equal_nan=True)
    out.append(Check(
        "identity", "a stream that drops samples rebuilds them as gaps",
        "Six samples removed from the stream entirely -- not sent as NaN, "
        "simply never emitted, which is what a counter does when it fails "
        "to produce a reading. Placing arriving samples by timestamp must "
        "return the original record NaN for NaN. Appending them to a list "
        "instead returns a record six samples shorter, with every later "
        "sample slid earlier and the whole timebase wrong from there on, "
        "and it looks *smoother* than the truth.",
        "exact, including the position of every gap. A count of gaps is "
        "not enough: the failure this catches produces the right count "
        "and the wrong timebase.",
        f"{x.size} slots rebuilt, "
        + ("every value and every gap in place" if exact else "MISMATCH"),
        exact,
    ))
    return out


def _collected(x: np.ndarray) -> np.ndarray:
    """The record a stream of `x` with its gaps omitted reconstructs."""
    from ..stream import collect, replay

    return np.asarray(collect(replay(x, 1.0, skip_gaps=True), 1.0).x)


def _synth_amplitude_identity() -> Check:
    """The synthesizer's calibrated amplitude, against exact closed forms.

    `E[avar]` of the generator is computable in closed form without running
    it, because the series is one linear functional of the driving white
    noise. Three power laws have exact expressions -- see
    `theory/noise-types.md` -- and the calibration either reproduces all
    three or is wrong. This is what lets the power-law decomposition be
    tested end to end against a *known* h_alpha rather than only
    structurally.
    """
    worst, tau0 = 0.0, 1.0
    for alpha, want in (
        # white FM: h0/(2 tau), exact at every m because y is iid.
        (0.0, lambda m: 1.0 / (2.0 * m * tau0)),
        # white PM: a brick-wall process at exactly Nyquist.
        (2.0, lambda m: 3.0 / (8.0 * np.pi**2 * tau0 * (m * tau0) ** 2)),
        # random-walk FM: the continuum form times an exact 1 + 1/(2m^2).
        (-2.0, lambda m: 2.0 * np.pi**2 * (m * tau0) / 3.0
         * (1.0 + 1.0 / (2.0 * m**2))),
    ):
        for m in (1, 2, 4, 16):
            got = _exact_generator_avar(alpha, m) / _unit_h_alpha(alpha, tau0)
            worst = max(worst, abs(got / want(m) - 1.0))
    return Check(
        "identity", "synthesizer amplitude against its exact closed forms",
        "The generator's E[avar] computed from its impulse response, against "
        "the closed forms for white FM, white PM and random-walk FM. Fixes "
        "h_alpha as a calibrated quantity in SI units rather than an "
        "arbitrary scale -- without this the power-law decomposition has no "
        "end-to-end test, only a structural one.",
        "1e-10 relative. All three are exact identities, not asymptotic "
        "limits: the random-walk form carries its 1/(2 m^2) discrete "
        "correction explicitly, which is 50% at m = 1.",
        f"worst deviation {worst:.2e} over three power laws and m = 1..16",
        worst < 1e-10,
    )


def _exact_generator_avar(alpha: float, m: int, n: int = 1 << 16) -> float:
    """`E[avar(m*tau0)]` at unit driving variance, from the impulse response.

        ybar2 - ybar1 = sum_j g_j w_j,   g = correlate(a, h),
        a = [-1/m]*m + [+1/m]*m,         E[avar] = (g @ g)/2

    The trailing `len(a)-1` entries are dropped: their sums run off the end
    of the truncated `h`, where the true process still has support and `a`
    summing to zero cancels them. Keeping them doubles the alpha = -2 answer.
    """
    h = _kasdin_walter_impulse(n, beta=-alpha)
    a = np.concatenate((np.full(m, -1.0 / m), np.full(m, 1.0 / m)))
    g = np.convolve(h, a[::-1])[: -(a.size - 1)]
    return float(g @ g) / 2.0


def _certified() -> list[Check]:
    """Tier 1b. Against exact rational arithmetic -- an independent oracle."""
    out = []
    x9 = exact_phase_from_freq([Fraction(int(v)) for v in nbs14_9point()])
    x1k = exact_phase_from_freq(
        [Fraction(k, NBS14_LEHMER_MODULUS) for k in nbs14_1000point_integers(1000)]
    )
    fs9 = FreqSeries(nbs14_9point(), rate_hz=1.0)
    fs1k = FreqSeries(nbs14_1000point(), rate_hz=1.0)

    for name, fn, exact in ESTIMATORS:
        digits = []
        for series, xq, afs, kw in (
            (fs9, x9, (1, 2), {}),
            (fs1k, x1k, (1, 10, 100), {"max_af": 100}),
        ):
            r = fn(series, taus="all", keep_coeffs=False, **kw)
            for af in afs:
                digits.append(lre(r.dev_at(af), certified_deviation(exact(xq, af))))
        worst = min(digits)
        out.append(Check(
            "certified", f"{name}: correct digits vs exact rational arithmetic",
            "Every value recomputed with `fractions.Fraction` and reduced to "
            "50 digits with `decimal`, by code that shares nothing with the "
            "production path -- a test enforces that `stats/` never imports "
            "the oracle. This is a correctness assertion, not an interop one.",
            "at least 13 of ~15.95 available digits. The margin covers "
            "accumulation over a 1000-point sum.",
            f"{worst:.1f} digits (worst of 5 points)",
            worst >= 13.0,
        ))

    worst = _incomplete_gamma_digits()
    out.append(Check(
        "certified", "incomplete gamma vs exact Decimal evaluation",
        "For integer order the upper incomplete gamma is a finite sum, so it "
        "can be evaluated exactly at 60 digits and used as an oracle. This "
        "found two real defects: Q computed as 1-P returned 0.0 where the "
        "answer was 6.8e-136, and the direct log-prefactor lost four digits "
        "at large order.",
        "at least 12.4 digits, the measured floor at a=5000, x=5500. The "
        "residue there is the float64 spacing of the exponent's terms; "
        "every value a confidence interval actually uses lands at 14 or "
        "better.",
        f"{worst:.1f} digits (worst over a = 1..5000, x/a = 0.5..1.5)",
        worst >= 12.4,
    ))
    return out


def _incomplete_gamma_digits() -> float:
    """Worst correct-digit count of Q(a, x) against exact Decimal arithmetic.

    Measured here rather than asserted, because a check whose result is
    hardcoded is not evidence. Restricted to integer `a`, where the upper
    incomplete gamma is the finite sum e^-x * sum_{k<a} x^k/k! and can be
    evaluated to arbitrary precision.
    """
    from decimal import Decimal, getcontext

    getcontext().prec = 60
    worst = float("inf")
    for a in (1, 10, 100, 1000, 5000):
        for frac in (0.5, 1.0, 1.5):
            x = float(a) * frac
            got = gammainc_q(float(a), x)
            if got == 0.0:  # genuine underflow, not a cancellation artefact
                continue
            xd = Decimal(repr(x))
            term = total = Decimal(1)
            for k in range(1, a):
                term = term * xd / k
                total += term
            want = total * (-xd).exp()
            g = Decimal(repr(got))
            digits = 60.0 if g == want else float(-(abs(g - want) / abs(want)).log10())
            worst = min(worst, digits)
    return worst


def _coverage(trials: int) -> list[Check]:
    """Tier 3. The measurement no competing tool publishes.

    The acceptance band widens with the Monte-Carlo standard error, because
    a measured coverage is itself an estimate. Fixing the band at the
    published spread would make the report fail spuriously whenever it was
    run at a lower trial count -- and a report that cries wolf is not
    evidence any more than one that cannot fail.
    """
    out = []
    trials = max(trials, MIN_COVERAGE_TRIALS)

    # Generate each ensemble once and score both confidence levels against
    # it. Regenerating per level would double the cost for nothing -- the
    # realizations do not depend on the level.
    ensembles = {}
    for alpha in (2, 0, -2):
        rng = np.random.default_rng(SEEDS["coverage"] + alpha)
        runs = [oadev(power_law_freq(512, alpha, rate_hz=1.0, rng=rng),
                      taus="octave") for _ in range(trials)]
        ensembles[alpha] = (runs, np.mean([r.var for r in runs], axis=0))

    for level, label in ((ONE_SIGMA, "68.27%"), (TWO_SIGMA, "95.45%")):
        worst_lo, worst_hi = 1.0, 0.0
        for runs, truth in ensembles.values():
            hits = np.zeros(runs[0].af.size)
            for r in runs:
                c = confidence(r, level=level)
                hits += (c.var_lo <= truth) & (truth <= c.var_hi)
            cov = hits / trials
            worst_lo = min(worst_lo, float(cov.min()))
            worst_hi = max(worst_hi, float(cov.max()))

        # Systematic allowance (how far the estimator itself strays at the
        # last octave) plus four Monte-Carlo standard errors.
        systematic = 0.09 if level == ONE_SIGMA else 0.07
        se = float(np.sqrt(level * (1.0 - level) / trials))
        lo_bound = level - systematic - 4.0 * se
        hi_bound = min(1.0, level + systematic + 4.0 * se)
        ok = lo_bound <= worst_lo and worst_hi <= hi_bound
        vacuous = " The upper bound saturates at 1.0 here, so overcoverage " \
                  "is caught by the 68.27% check rather than this one." \
                  if hi_bound >= 1.0 else ""
        out.append(Check(
            "coverage", f"measured containment at nominal {label}",
            "Fraction of realizations whose interval contains the ensemble "
            "truth, over white PM, white FM and random-walk FM at N=512. "
            "This validates the EDOF estimator and the chi-square inversion "
            "jointly, end to end -- either being wrong shows up here and "
            "nowhere else.",
            f"within [{lo_bound:.3f}, {hi_bound:.3f}] = nominal +/- "
            f"{systematic:.2f} systematic +/- 4 Monte-Carlo standard errors "
            f"({se:.3f} at {trials} trials). The systematic part is how far "
            "the estimator strays at the last octave, where eta is small "
            f"and the autocorrelation is barely resolved.{vacuous}",
            f"{worst_lo:.3f} to {worst_hi:.3f} against nominal {label}",
            ok,
        ))

    runs = ensembles[0][0]  # white FM, already generated above
    v = np.array([r.var for r in runs])
    eta_true = 2.0 * v.mean(axis=0) ** 2 / v.var(axis=0)
    ratios = []
    for i, af in enumerate(runs[0].af):
        if int(af) > 512 // 32:
            continue
        est = np.median([eta1(r.coeffs[i], int(af)) for r in runs])
        ratios.append(est / eta_true[i])
    worst = max(abs(r - 1.0) for r in ratios)
    # eta_true is itself a variance-of-a-variance, whose relative standard
    # error is sqrt(2/trials) for a Gaussian sample. The estimator's own
    # error and the truth's uncertainty add in quadrature.
    truth_se = float(np.sqrt(2.0 / trials))
    bound = float(np.hypot(0.10, 4.0 * truth_se))
    out.append(Check(
        "coverage", "eta1 against Monte-Carlo truth",
        "EDOF is defined by Var(sigma_hat^2) = 2 sigma^4 / eta, and both "
        "sides are measurable. No published table is involved -- every such "
        "table assumes a noise type, which is the assumption this estimator "
        "exists to avoid.",
        f"within {bound:.0%} for m up to N/32, being 10% for the estimator "
        f"combined in quadrature with four standard errors of the "
        f"Monte-Carlo truth ({truth_se:.0%} at {trials} trials). The last "
        "octave is looser and is covered by the coverage checks above.",
        f"worst deviation {worst:.0%} over m = 1..16",
        worst < bound,
    ))
    out += _greenhall_against_truth(ensembles, trials, truth_se)
    out.append(_h_alpha_round_trip(ensembles, trials))
    out.append(_decomposition_round_trip(trials))
    out.append(_bench_hat_round_trip())
    out.append(_cross_sees_under_the_instrument(trials))
    out.append(_rejection_gain_from_extra_channels(trials))
    out.extend(_fitted_parameter_coverage(trials))
    return out


def _fitted_parameter_coverage(trials: int) -> list[Check]:
    """Fit a known curve many times; do the stated intervals contain it?

    Two claims scored on one ensemble, because they are the same
    measurement read twice.

    **The M4 pass criterion**: a maximum-likelihood fit recovers a known
    parameter from synthesized noise within its stated uncertainty. White FM
    at a known `h_alpha` has `oadev` variance `h0/(2 tau)` exactly, so both
    fitted parameters have a truth to be scored against and no reference
    table is involved.

    **What the cross-tau correlation is worth**, which is the claim no
    competing tool can make: the same intervals recomputed with the
    correlation deleted under-cover, and by how much. That number is the
    argument for `stats/taucov.py` existing, and stating it as a measured
    coverage rather than as a ratio of error bars is what makes it evidence.

    Scored on the *slope*, not on the amplitude. Both are fitted, but the
    slope is the parameter a customer reads off a specification -- "is this
    oscillator white-FM limited out to a day" -- and it is where the
    correlation bites hardest, since a coherent tilt of the whole curve is
    exactly what correlated points produce.
    """
    from ..fitting import fit as run_fit
    from .._specfun import norm_ppf

    reps = max(trials, MIN_COVERAGE_TRIALS)
    truth = {"A": 1.0e-22, "b": -1.0}
    h_alpha = 2.0 * truth["A"]
    level = ONE_SIGMA
    z = float(norm_ppf(0.5 * (1.0 + level)))

    rng = np.random.default_rng(SEEDS["coverage"] + 900)
    inside = {"A": 0, "b": 0}
    inside_naive = {"A": 0, "b": 0}
    ratios, failures = [], 0
    for _ in range(reps):
        fs = power_law_freq(4096, 0, rate_hz=1.0, rng=rng, h_alpha=h_alpha)
        got = run_fit(oadev(fs, taus="octave", max_af=256), "powerlaw")
        if not got.converged:
            failures += 1
            continue
        for name in ("A", "b"):
            miss = abs(got.params[name] - truth[name])
            inside[name] += miss <= z * got.stderr[name]
            inside_naive[name] += miss <= z * got.stderr_naive[name]
        ratios.append(got.understatement["b"])

    scored = reps - failures
    se = float(np.sqrt(level * (1.0 - level) / scored))
    # Four Monte-Carlo standard errors and no systematic allowance, unlike
    # the per-point coverage row above. That row needs one because the EDOF
    # estimator itself strays at the last octave; this one was measured at
    # 1500 trials and came back at +0.50 and -1.00 standard errors, so
    # there is nothing to allow for -- and adding an allowance anyway would
    # have made the row unable to fail. Deleting the correlation moves the
    # slope's coverage by 5.8 standard errors, which this band catches and
    # a wider one did not.
    band = 4.0 * se
    measured = {k: inside[k] / scored for k in inside}
    naive_b = inside_naive["b"] / scored

    covered = Check(
        "coverage", "a maximum-likelihood fit recovers a known parameter",
        "White FM synthesized at a known IEEE 1139 coefficient has "
        "`oadev` variance h0/(2 tau) exactly, so a two-parameter power-law "
        "fit has a truth for both its amplitude and its slope. Scored as "
        f"containment of that truth by the stated interval over {scored} "
        "realizations at N=4096 -- the whole chain, from the calibrated "
        "synthesizer through the estimator, the degrees of freedom, the "
        "cross-tau covariance and the chi-square likelihood, back to the "
        "number that went in. Every converged fit is scored, including the "
        "ones whose goodness of fit came out poor: dropping those would be "
        "selecting on the outcome being measured.",
        f"both parameters contained {level:.4f} of the time, within "
        f"{band:.3f} (four Monte-Carlo standard errors)",
        "; ".join(f"{k} {v:.3f}" for k, v in measured.items())
        + (f"; {failures} fits did not converge" if failures else ""),
        all(abs(v - level) <= band for v in measured.values()),
    )

    # Two conditions, not one. "The naive interval under-covers" is only
    # evidence for this machinery if the corrected interval does *not* --
    # otherwise a run in which both were broken in the same way would pass
    # while proving nothing.
    margin = 3.0 * se
    deficit = Check(
        "coverage", "ignoring the cross-tau correlation under-covers",
        "The same fits, with the same fitted values, scored against the "
        "interval that would be quoted if the taus were treated as "
        "independent -- which is what every other tool reports. The points "
        "of an overlapping estimator come from the same phase samples, so "
        "that interval is too narrow, and this measures by how much rather "
        "than asserting it.",
        f"the uncorrelated interval contains the true slope less than "
        f"{level - margin:.3f} of the time, and the correlation-aware one "
        f"is within {band:.3f} of {level:.4f}",
        f"slope containment {naive_b:.3f} uncorrelated against "
        f"{measured['b']:.3f} corrected; mean error-bar ratio "
        f"{float(np.mean(ratios)):.2f}",
        naive_b < level - margin and abs(measured["b"] - level) <= band,
    )
    return [covered, deficit]


def _rejection_gain_from_extra_channels(trials: int) -> Check:
    """Four channels reject deeper than two, by the amount claimed.

    The commercial question this answers is "is a third and fourth channel
    worth buying", so the number that matters is the *gain* over two, and
    the honest version of it is bounded above as well as below. Averaging
    P = 6 pairs would win 5*log10(6) = 3.9 dB if the pairs were
    independent; they are not, since (c0,c1) and (c0,c2) share a channel,
    and the exact covariance says how much of that is real.

    Both bounds are scored. A gain that vanished would mean the averaging
    does nothing; a gain that reached the naive figure would mean the
    shared-channel terms had been dropped somewhere.
    """
    from ..multi import cross_matrix, reject

    rng = np.random.default_rng(SEEDS["coverage"] + 1700)
    n, rate = 4096, 1.0
    runs = max(trials // 20, 10)
    gains, bias, worst_sigma = [], [], 0.0
    for _ in range(runs):
        clock = power_law_phase(n, 0, rate, rng, 1.0e-24)
        ch = {
            f"c{i}": PhaseSeries(
                x=clock.x + power_law_phase(n, 2, rate, rng, 1.0e-22).x,
                rate_hz=rate)
            for i in range(4)
        }
        four = reject(cross_matrix(ch, max_af=8))
        two = reject(cross_matrix({k: ch[k] for k in ("c0", "c1")}, max_af=8))
        truth = oadev(clock, max_af=8).var
        gains.append(four.rejection_db - two.rejection_db)
        bias.append(four.var / truth - 1.0)
        worst_sigma = max(worst_sigma,
                          float(np.max(np.abs(four.var - truth) / four.sem)))

    gain = np.asarray(gains).mean(axis=0)
    err = np.asarray(bias)
    mean_bias = float(err.mean())
    bias_se = float(err.std(ddof=1) / np.sqrt(err.size))
    naive = 5.0 * np.log10(6.0)
    ok = (float(gain.min()) > 0.3 and float(gain.max()) < naive
          and abs(mean_bias) < 4.0 * bias_se and worst_sigma < 4.0)
    return Check(
        "coverage", "extra channels reject deeper, by less than naive counting",
        "One white-FM clock through four white-PM channels, averaged over "
        "all six pairs, against the same clock through two of them. The "
        "pairs are not independent -- (c0,c1) and (c0,c2) share a channel -- "
        "so the exact Isserlis covariance over pairs of pairs is what sets "
        "the error bar, and the gain it predicts is smaller than treating "
        "the six as six independent estimates would claim.",
        f"the averaging gain is positive at every tau and strictly below "
        f"the naive 5*log10(6) = {naive:.2f} dB; the four-channel estimate "
        "is unbiased against the clock record within four standard errors "
        "of the ensemble mean, and every realization inside four of its own "
        "reported standard errors.",
        f"gain {gain.min():.2f}-{gain.max():.2f} dB against {naive:.2f} dB "
        f"naive; bias {mean_bias:+.1%} against {bias_se:.1%} standard error "
        f"over {runs} realizations; worst miss {worst_sigma:.1f} sigma",
        ok,
    )


def _cross_sees_under_the_instrument(trials: int) -> Check:
    """A clock buried under its channels' noise, recovered by the product.

    The claim that distinguishes this library from every other tool in the
    field, so it is scored against the record that actually generated the
    common part rather than against a fit. Two things are measured at once:
    how far the auto-variance is wrong, which is the size of the problem,
    and how far the cross is wrong, which is the size of the answer.

    Scored at the shortest taus only. That is not a convenience -- it is
    where the instrument dominates and therefore the only place the
    rejection is being tested at all. By long tau the clock has emerged
    from the noise and both estimates are right, which would flatter the
    method by averaging in the region where it does nothing.
    """
    from ..multi import cross_variance

    rng = np.random.default_rng(SEEDS["coverage"] + 1300)
    n, rate = 4096, 1.0
    h_clock, h_inst = 1.0e-24, 1.0e-22
    runs = max(trials // 10, 20)
    auto_err, cross_err, worst_sigma, rejection = [], [], 0.0, []
    for _ in range(runs):
        clock = power_law_phase(n, 0, rate, rng, h_clock)
        a = PhaseSeries(x=clock.x + power_law_phase(n, 2, rate, rng, h_inst).x,
                        rate_hz=rate)
        b = PhaseSeries(x=clock.x + power_law_phase(n, 2, rate, rng, h_inst).x,
                        rate_hz=rate)
        cross = cross_variance(a, b, max_af=8)
        truth = oadev(clock, max_af=8).var
        auto_err.append(float(np.median(cross.auto_a / truth - 1.0)))
        cross_err.append(cross.var / truth - 1.0)
        # The error bar has to cover the miss, not merely be small.
        worst_sigma = max(worst_sigma,
                          float(np.max(np.abs(cross.var - truth) / cross.sem)))
        rejection.append(float(np.min(cross.rejection_db)))

    # Scored on the bias and on the coverage, not on a flat percentage. A
    # single realization at 15 dB of rejection carries about 30% of its own
    # standard error, so "within 10%" would be a statement about luck; what
    # the method promises is that it is *centred* on the truth and that it
    # says how far off it might be. Both are measurable.
    err = np.asarray(cross_err)
    bias = float(err.mean())
    bias_se = float(err.std(ddof=1) / np.sqrt(err.size))
    median_auto = float(np.median(auto_err))
    ok = abs(bias) < 4.0 * bias_se and worst_sigma < 4.0
    return Check(
        "coverage", "cross-variance recovers a clock the auto-variance cannot",
        "One synthetic white-FM clock seen through two channels with "
        "independent white-PM noise, at a level that puts the clock about a "
        "factor of three under the single-channel floor in deviation. The "
        "truth is the generated clock record's own oadev, not a fit, so this "
        "measures the rejection rather than its self-consistency. Scored "
        "over tau = 1..8 s, where the instrument dominates and the rejection "
        "is actually being tested; by long tau the clock has emerged anyway "
        "and both estimates are right.",
        "unbiased to within four standard errors of the ensemble mean, and "
        "every single-realization miss inside four of its own reported "
        "standard errors. Not a flat percentage: at this rejection one "
        "realization carries tens of percent of sampling scatter, so a tight "
        "band would score luck. The auto-variance is reported beside it as "
        "the size of the problem it is solving.",
        f"bias {bias:+.1%} against {bias_se:.1%} standard error over {runs} "
        f"realizations, worst miss {worst_sigma:.1f} sigma, auto median "
        f"{median_auto:+.0%} high, rejection {np.min(rejection):.1f} dB or "
        "better",
        ok,
    )


def _bench_hat_round_trip() -> Check:
    """Three known oscillators in, three recovered variances out.

    The N-cornered hat's whole claim is that it separates what a single
    measurement cannot: a pair record sees `h_i + h_j` and no amount of
    averaging splits it. Three pair records containing an odd cycle do
    split it, and this measures how well.

    Deliberately built with levels that differ by 30x, because the hat's
    known weakness is the *quietest* source -- there it is a small
    difference of large numbers and the estimate can go negative. Scoring
    only the two louder ones would flatter the method; the quietest is
    reported separately rather than dropped.
    """
    from ..bench import Bench, Channel, Device, Measurement, Source
    from ..bench.graph import design_matrix
    from ..bench.solve import solve
    from ..stats.edof import edof

    h = {"A": 4.0e-24, "B": 1.0e-24, "C": 9.0e-24}
    rng = np.random.default_rng(SEEDS["coverage"] + 900)
    y = {k: power_law_freq(8192, 0, rate_hz=1.0, rng=rng, h_alpha=v).y
         for k, v in h.items()}
    pairs = (("A", "B"), ("B", "C"), ("C", "A"))
    results = {
        f"m{i}": oadev(FreqSeries(y[a] - y[b], rate_hz=1.0), taus="octave")
        for i, (a, b) in enumerate(pairs)
    }
    bench = Bench(
        sources=tuple(Source(id=s) for s in "ABC"),
        devices=(Device(id="D1", channels=(Channel(id="A-1", floor=0.0),)),),
        measurements=tuple(
            Measurement(id=f"m{i}", channel_id="A-1", source_a=a, source_b=b)
            for i, (a, b) in enumerate(pairs)
        ),
    )
    sol = solve(bench, results)

    # The tolerance is propagated, not chosen. Each measured variance has
    # Var(sigma2_hat) = 2 sigma^4 / eta, so the solution's covariance is
    # A+ diag(...) A+^T -- and for a source far quieter than its partners
    # that predicts a large relative error, because the hat is then a small
    # difference of large numbers. Scoring the quiet source against the
    # same flat percentage as the loud ones would be scoring the choice of
    # oscillators rather than the method.
    a, _sources, _channels, _known = design_matrix(bench)
    pinv = np.linalg.pinv(a)
    near = sol.tau_s <= 64.0
    worst_ratio, per = 0.0, []
    for name in "ABC":
        col = sol.sources.index(name)
        truth = h[name] / (2.0 * sol.tau_s)
        got = sol.source_var[:, col]
        predicted = np.zeros(sol.tau_s.size)
        for k in range(sol.tau_s.size):
            var_b = []
            for i, mid in enumerate(("m0", "m1", "m2")):
                r = results[mid]
                eta, _ = edof(r.coeffs[k], int(r.af[k]), r.span_factor)
                var_b.append(2.0 * r.var[k] ** 2 / max(eta, 1.0))
            cov = pinv @ np.diag(var_b) @ pinv.T
            predicted[k] = np.sqrt(cov[col, col])
        # How many of its own predicted standard errors is it out by?
        sigmas = np.abs(got[near] - truth[near]) / predicted[near]
        worst_ratio = max(worst_ratio, float(np.max(sigmas)))
        rel = float(np.max(np.abs(got[near] / truth[near] - 1.0)))
        expect = float(np.max(predicted[near] / truth[near]))
        per.append(f"{name} ({h[name]:.0e}): {rel:.0%} against {expect:.0%} predicted")

    bound = 3.0
    return Check(
        "coverage", "N-cornered hat recovers three known oscillators",
        "Three independent white-FM sources at known h_alpha, measured "
        "pairwise, solved as one non-negative least-squares system per tau. "
        "The truth is the closed form h/(2 tau), not a value taken from the "
        "fit, so this measures the method rather than its self-consistency. "
        "It is also the claim no single measurement can support: one pair "
        "record sees h_i + h_j and cannot split it.",
        f"within {bound:.0f} standard errors, where the standard error is "
        "propagated through the solve from each row's own EDOF rather than "
        "chosen. The levels differ by 30x on purpose: for the quietest "
        "source the hat is a small difference of large numbers, and its "
        "predicted error is correspondingly large. A flat percentage would "
        "score the choice of oscillators instead of the method.",
        f"worst {worst_ratio:.1f} sigma; per source " + ", ".join(per),
        worst_ratio < bound,
    )


def _decomposition_round_trip(trials: int) -> Check:
    """Generate a known h_alpha, decompose the measured curve, compare.

    The end-to-end claim for the power-law fit: not that it inverts its own
    design matrix -- a test does that at 1e-8 -- but that the whole chain
    holds, from the IEEE 1139 coefficient through the synthesizer, the
    estimators, the conversion coefficients and the constrained solve, back
    to the same number.

    Only the three `f_h`-free power laws are scored. The two PM rows depend
    on a bandwidth that is not identifiable from tau-domain data, and the
    synthesizer's flicker PM is not brick-wall limited, so scoring them
    here would measure that known offset rather than this fit. It is
    recorded in VALIDATION-DEBT.md instead of being quietly averaged in.
    """
    from ..stats.powerlaw import decompose

    tau0, h_true, n = 1.0, 5.0e-24, 4096
    reps = max(trials // 8, 40)
    worst, seen = 0.0, []
    for alpha in (0, -1, -2):
        rng = np.random.default_rng(SEEDS["coverage"] + 500 + alpha)
        acc_a = acc_m = None
        for _ in range(reps):
            fs = power_law_freq(n, alpha, rate_hz=1.0 / tau0, rng=rng,
                                h_alpha=h_true)
            ra, rm = oadev(fs, taus="octave"), mdev(fs, taus="octave")
            acc_a = ra.var if acc_a is None else acc_a + ra.var
            acc_m = rm.var if acc_m is None else acc_m + rm.var
        keep = (ra.af >= 4) & (ra.af <= n // 32)
        curves = [
            StabilityResult(kind=r.kind, af=r.af[keep], tau_s=r.tau_s[keep],
                            var=(acc / reps)[keep], n_terms=r.n_terms[keep],
                            rate_hz=r.rate_hz, provenance=r.provenance,
                            shape=r.shape)
            for r, acc in ((ra, acc_a), (rm, acc_m))
        ]
        got = decompose(curves, f_h=0.5 / tau0, weights="none").h[alpha]
        seen.append(f"alpha={alpha}: {got / h_true:.3f}")
        worst = max(worst, abs(got / h_true - 1.0))

    bound = 0.20
    return Check(
        "coverage", "power-law decomposition recovers a known h_alpha",
        "Synthetic noise generated at a known IEEE 1139 coefficient, "
        "estimated with oadev and mdev, and decomposed. Closes the loop that "
        "the conversion coefficients, the calibrated synthesizer and the "
        "non-negative solve all sit on -- no step of which is checked by "
        f"this measurement alone. Averaged over {reps} realizations per "
        "power law at N=4096.",
        f"within {bound:.0%} of the generated level for white FM, flicker FM "
        "and random-walk FM. The two PM rows are excluded because their "
        "coefficients depend on a bandwidth that tau-domain data cannot "
        "identify -- see VALIDATION-DEBT.md item 2.",
        f"recovered/true {'; '.join(seen)}",
        worst < bound,
    )


def _h_alpha_round_trip(ensembles, trials: int) -> Check:
    """Ask for an h_alpha, generate, estimate, and see what comes back.

    The identity tier proves the calibration formula against exact closed
    forms. This proves the *path*: the real generator through the real
    estimator, so a mistake anywhere between the two shows up here. The
    ensembles are the ones the coverage checks already built, generated at
    `h_alpha = 1` -- which is not a special case, since the calibration is a
    single multiplication and `test_synth.py` pins its linearity separately.
    """
    worst, tau0 = 0.0, 1.0
    for alpha, (runs, mean_var) in ensembles.items():
        first = runs[0]
        for i, af in enumerate(first.af.tolist()):
            if af > 512 // 16:
                continue
            want = _exact_generator_avar(float(alpha), af) / _unit_h_alpha(alpha, tau0)
            worst = max(worst, abs(mean_var[i] / want - 1.0))
    # The ensemble mean of a variance has relative standard error
    # sqrt(2/(trials*eta)); eta >= 1, so sqrt(2/trials) is the floor and the
    # band is four of those plus an allowance for the estimator's own spread.
    se = float(np.sqrt(2.0 / trials))
    bound = float(np.hypot(0.05, 4.0 * se))
    return Check(
        "coverage", "requested h_alpha recovered end to end",
        "The ensemble mean of sigma_hat^2 against the amplitude that was "
        "asked for, over white PM, white FM and random-walk FM at N=512. "
        "Compared against the generator's exact discrete E[avar], not the "
        "continuum closed form, so the known 1/(2m^2) and Nyquist "
        "corrections are not charged to the Monte Carlo.",
        f"within {bound:.0%}, being 5% for the estimator combined in "
        f"quadrature with four standard errors of an ensemble-mean variance "
        f"({se:.0%} at {trials} trials), for m up to N/16.",
        f"worst deviation {worst:.1%} over three power laws",
        worst < bound,
    )


def _greenhall_against_truth(ensembles, trials: int, truth_se: float) -> list[Check]:
    """The model-based EDOF measured the same way eta1 is, and compared to it.

    Two checks, deliberately separate. The first asks whether Greenhall's
    algorithm predicts the sampling variance it claims to. The second asks
    what its assumption costs against an estimator that makes none -- and
    that is the whole argument for the default this library ships, so it is
    measured rather than asserted in prose.
    """
    worst_model, worst_gap, at_m1 = 0.0, 0.0, {}
    for alpha, (runs, _) in ensembles.items():
        v = np.array([r.var for r in runs])
        eta_true = 2.0 * v.mean(axis=0) ** 2 / v.var(axis=0)
        first = runs[0]
        for i, af in enumerate(first.af.tolist()):
            eta = greenhall_edof(alpha, first.shape.d, af, 513,
                                 first.shape.modified, first.shape.overlapped)
            if af == 1:
                at_m1[alpha] = eta / eta_true[i]
                continue
            if af > 512 // 32:
                continue
            worst_model = max(worst_model, abs(eta / eta_true[i] - 1.0))
            data_driven = float(np.median([eta1(r.coeffs[i], af) for r in runs]))
            worst_gap = max(worst_gap, abs(eta / data_driven - 1.0))

    bound = float(np.hypot(0.20, 4.0 * truth_se))
    span = ", ".join(f"alpha={a}: {r:.2f}" for a, r in sorted(at_m1.items()))
    return [
        Check(
            "coverage", "Greenhall EDOF against Monte-Carlo truth",
            "The same definitional measurement applied to the model-based "
            "algorithm: it is told the power law and the record geometry, "
            "never shown the data, and its answer is compared to "
            "2 sigma^4 / Var(sigma_hat^2) measured over the ensemble. This is "
            "the only correctness evidence for the Stable32 parity path that "
            "does not itself come from Stable32.",
            f"within {bound:.0%} for m = 2..16, being 20% for the model "
            f"combined in quadrature with four standard errors of the "
            f"Monte-Carlo truth ({truth_se:.0%} at {trials} trials). Wider "
            "than eta1's band because two approximations are being compared, "
            "not one estimate against a truth.",
            f"worst deviation {worst_model:.0%} over m = 2..16",
            worst_model < bound,
        ),
        Check(
            "coverage", "what assuming a power law costs, measured",
            "Greenhall against eta1 on the same realizations. They answer "
            "different questions -- one is told the noise type, the other "
            "reads the data -- so this is not a correctness check on either. "
            "It is the honest statement of the trade the user is making, and "
            "the reason `auto` never selects the model-based route.",
            "the two agree within 40% for m = 2..16 across white PM, white FM "
            "and random-walk FM. At m = 1 they diverge sharply and by design: "
            "the paper models phase as a difference of a band-unlimited power "
            "law, so its noise is not flat to Nyquist. The ratios there are "
            f"reported rather than bounded.",
            f"worst disagreement {worst_gap:.0%} for m = 2..16; "
            f"greenhall/eta1 at m = 1 is {span}",
            worst_gap < 0.40,
        ),
    ]


def _interop() -> list[Check]:
    """Tier 2. Proves agreement with the incumbent. Not a correctness claim."""
    out = []
    fs = FreqSeries(nbs14_1000point(), rate_hz=1.0)
    for name, fn, _ in ESTIMATORS:
        expected = EXPECTED_NBS14_1000POINT.get(name)
        if not expected:
            continue
        r = fn(fs, taus="all", max_af=100, keep_coeffs=False)
        worst = max(abs(r.dev_at(af) / want - 1.0) for af, want in expected.items())
        out.append(Check(
            "interop", f"{name} vs the published NBS14 1000-point table",
            "INTEROP, not correctness. These values were produced by "
            "Stable32 and reproduced in NIST SP 1065, so agreement shows "
            "this library matches the incumbent. Correctness is the "
            "certified tier above.",
            f"{INTEROP_RTOL:.0e} relative -- the tightest the seven "
            "published significant figures support. Asserting harder would "
            "test our transcription, not our arithmetic.",
            f"worst rel {worst:.1e}",
            worst < INTEROP_RTOL,
        ))
    out += _stable32_parity()
    return out


# Where the reference session's exports live, relative to this file. Absent
# from an installed wheel, which is fine -- the checks that read it say how
# many tables they found and pass vacuously on none, so a partially
# collected session is still worth running.
STABLE32_OUTPUTS = (
    Path(__file__).resolve().parents[3] / "tests" / "data" / "stable32"
)

# Stable32 exports five significant figures, so a value we compute at full
# precision can differ from the printed one by up to half an ulp of the
# fifth digit -- 5e-5 when the mantissa is near 1. Anything inside this is
# agreement to the last digit the export actually carries; asserting
# tighter would be asserting against digits Stable32 never wrote.
STABLE32_PRINT_RTOL = 1e-4


def _stable32_parity() -> list[Check]:
    """Against Stable32's own exported tables, where a session has produced them.

    This is the claim the whole Greenhall path exists to support, and until
    a session happens there is nothing here to check. Both checks report
    the number of tables scored, so "passed on nothing" is visible rather
    than silently reassuring.
    """
    from ..io import load_phase
    from ..registry import get
    from .stable32 import load_dir

    tables = [t for t in load_dir(STABLE32_OUTPUTS / "outputs")
              if t.estimator is not None]
    if not tables:
        return [Check(
            "interop", "Stable32 output parity",
            "Direct comparison against Stable32's own exported tables -- "
            "the strongest interoperability claim available, and the one an "
            "accredited lab replacing Stable32 will check first.",
            "not applicable: no reference outputs are present.",
            "NOT RUN -- no exports in tests/data/stable32/outputs. See the "
            "run sheet there; this is item 1 of VALIDATION-DEBT.md.",
            True,
        )]

    worst_sigma, worst_bound, worst_edof = 0.0, 0.0, 0.0
    n_sigma, n_bound, files = 0, 0, set()
    # Five estimators over three tau grids means the same few curves are
    # asked for repeatedly; computing each once keeps the report's runtime
    # proportional to the data rather than to the number of tables.
    cache: dict = {}
    # One curve per (record, estimator), computed to the largest averaging
    # factor any of that pair's tables needs. The three tau grids are
    # subsets of the same `all` grid, so keying on each grid's own maximum
    # would compute the same curve three times.
    needed: dict = {}
    for t in tables:
        key = (t.input_file, t.estimator)
        needed[key] = sorted(set(needed.get(key, [])) | set(t.af.tolist()))
    for t in tables:
        files.add(t.input_file)
        key = (t.input_file, t.estimator)
        if key not in cache:
            series = load_phase(STABLE32_OUTPUTS / "inputs" / t.input_file,
                                rate_hz=t.af[0] / t.tau_s[0])
            cache[key] = get("estimator", t.estimator)(
                series, taus=needed[key])
        r = cache[key].select_afs(t.af)

        # Hadamard of a pure ramp is ~1e-23 against the Allan family's
        # 1e-13: both programs are differencing away the entire signal, so
        # what is left is arithmetic on the file's own rounding. Compared
        # anyway -- they read the same bytes -- but it is not a claim about
        # the estimator, so it cannot dominate the reported worst case.
        scale = float(np.max(cache[key].dev))
        keep = (t.sigma > 0.0) & (t.sigma >= 1e-6 * scale)
        if keep.any():
            worst_sigma = max(worst_sigma, float(np.max(
                np.abs(r.dev[keep] / t.sigma[keep] - 1.0))))
            n_sigma += int(keep.sum())

        if not t.has_bounds:
            continue
        # Their Alpha column feeds our Greenhall path, so this compares the
        # EDOF algorithm alone rather than confounding it with noise ID.
        c = confidence(r, level=t.confidence, method="greenhall",
                       alpha=t.alpha.astype(int))
        usable = np.array([
            bool(np.isfinite(t.sigma_lo[k]) and t.sigma[k] >= 1e-6 * scale)
            for k in range(t.af.size)
        ])
        if usable.any():
            worst_bound = max(
                worst_bound,
                float(np.max(np.abs(c.dev_lo[usable] / t.sigma_lo[usable] - 1.0))),
                float(np.max(np.abs(c.dev_hi[usable] / t.sigma_hi[usable] - 1.0))),
            )
            implied = _implied_edof(t.sigma_hi[usable] / t.sigma[usable],
                                    t.confidence)
            worst_edof = max(worst_edof,
                             float(np.max(np.abs(implied / c.edof[usable] - 1.0))))
            n_bound += int(usable.sum())

    where = f"{len(tables)} tables over {len(files)} record(s)"
    return [
        Check(
            "interop", "sigma columns against Stable32's own export",
            "Every estimator this library implements, against the same "
            "statistic computed by Stable32 1.62 on byte-identical input. "
            "INTEROP, not correctness -- but it is the comparison a lab "
            "migrating off Stable32 actually performs.",
            f"{STABLE32_PRINT_RTOL:.0e} relative, which is the half-ulp of "
            "the fifth significant figure Stable32 prints. Agreement inside "
            "this means agreement to the last digit the export carries.",
            f"worst rel {worst_sigma:.2e} over {n_sigma} points, {where}",
            worst_sigma < STABLE32_PRINT_RTOL,
        ),
        Check(
            "interop", "Min/Max Sigma against Stable32, and the EDOF behind them",
            "INTEROP, not correctness. Stable32's error bars, reproduced by "
            "feeding its own Alpha column into this library's Greenhall "
            "path -- so this measures the EDOF algorithm alone, not the "
            "noise identification. Inverting the printed ratio also "
            "recovers the EDOF Stable32 used, which is the quantity "
            "actually in dispute: it runs about 1.2% above ours for m up to "
            "32, where an independent implementation of the same paper "
            "returns our value to every printed digit. See "
            "theory/greenhall.md.",
            "1% on the bounds. The bounds are far less sensitive than the "
            "EDOF behind them: a 1.3% EDOF difference moves a 68.27% "
            "deviation bound by under 0.06%, because the chi-square "
            "quantile ratio flattens at large degrees of freedom.",
            f"bounds worst {worst_bound:.2e} over {n_bound} points; implied "
            f"EDOF differs by up to {worst_edof:.1%} ({where})",
            worst_bound < 0.01,
        ),
    ]


def _implied_edof(ratio_hi, level: float):
    """The EDOF Stable32 must have used, from its own printed columns.

    `sigma_hi/sigma` depends on nothing but the degrees of freedom and the
    confidence level, so it inverts. Bisection, because the ratio is
    monotone in `eta` and nothing here needs a derivative.

    **Vectorized over the whole column deliberately.** `chi2_ppf` is itself
    an iterative inversion, so a scalar bisection per row costs one root
    find per step per row -- for one report that was 25 000 of them and
    thirty seconds. Bisecting every row together makes it 32 array calls.

    32 steps takes the bracket from 2e4 down to 5e-6, far finer than the
    0.1% this is reported to.
    """
    ratio = np.asarray(ratio_hi, dtype=np.float64)
    p = (1.0 - level) / 2.0
    lo = np.full(ratio.shape, 1.0)
    hi = np.full(ratio.shape, 2.0e4)
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        too_wide = np.sqrt(mid / chi2_ppf(p, mid)) > ratio
        lo = np.where(too_wide, mid, lo)
        hi = np.where(too_wide, hi, mid)
    return 0.5 * (lo + hi)


def _self_test() -> list[Check]:
    """A check designed to fail. A report that cannot fail is worthless."""
    return [Check(
        "self-test", "deliberate failure",
        "Asserts 1 == 2, so that the rendering, the exit status and the "
        "overall verdict are all exercised in the failing direction.",
        "none. This check exists to fail.",
        "1 != 2, as intended",
        False,
    )]


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def run(trials: int = MIN_COVERAGE_TRIALS, self_test: bool = False) -> Report:
    """Run the whole hierarchy and return the report.

    `trials` sets the Monte-Carlo depth. The default is quick enough to run
    interactively; the published numbers use 800.
    """
    checks = _identities() + _certified() + _coverage(trials) + _interop()
    if self_test:
        checks += _self_test()
    checks = tuple(checks)

    environment = {
        "stabilis": __version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "monte_carlo_trials": max(trials, MIN_COVERAGE_TRIALS),
    }
    # The digest covers the claims and the measured values -- the evidence --
    # and deliberately not the timestamp or the platform, which vary without
    # the evidence changing. A different digest on the same library and
    # numpy version means the numbers differ, which is information.
    payload = json.dumps([asdict(c) for c in checks], sort_keys=True)
    return Report(
        checks=checks,
        environment=environment,
        seeds=dict(SEEDS),
        digest=hashlib.sha256(payload.encode()).hexdigest(),
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def to_json(report: Report) -> str:
    return json.dumps(
        {
            "passed": report.passed,
            "digest": report.digest,
            "generated": report.generated,
            "environment": report.environment,
            "seeds": report.seeds,
            "checks": [asdict(c) for c in report.checks],
        },
        indent=2,
    )


_CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:60rem;
margin:2rem auto;padding:0 1rem;color:#182026}
h1{margin-bottom:.2rem}h2{margin-top:2rem;border-bottom:1px solid #dde;padding-bottom:.2rem}
.sub{color:#5c7080;margin-top:0}
table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:14px}
th,td{border:1px solid #dde;padding:.45rem .6rem;text-align:left;vertical-align:top}
th{background:#f5f7fa}
.pass{color:#0a6640;font-weight:600}.fail{color:#a82a2a;font-weight:600}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.verdict{padding:.7rem 1rem;border-radius:4px;font-weight:600;margin:1rem 0}
.ok{background:#e8f5ee;color:#0a6640}.bad{background:#fdeaea;color:#a82a2a}
.note{background:#f5f7fa;border-left:3px solid #8a9ba8;padding:.6rem 1rem;margin:1rem 0}
"""

_TIER_TITLES = {
    "identity": ("Analytic identities",
                 "Machine precision, no external data. Correctness lives here."),
    "certified": ("Certified values",
                  "Against exact rational arithmetic computed by an independent "
                  "implementation that the production path is forbidden to import."),
    "coverage": ("Monte-Carlo coverage",
                 "Measured containment of the confidence intervals. No competing "
                 "tool publishes this."),
    "interop": ("Interoperability",
                "Agreement with Stable32-produced tables. Commercially necessary, "
                "epistemically weak -- these prove we match the incumbent, not "
                "that either of us is right."),
    "self-test": ("Self-test",
                  "A check designed to fail, proving this report can."),
}


def _esc(v) -> str:
    return (str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def to_html(report: Report) -> str:
    verdict = ("ok", "PASS") if report.passed else ("bad", "FAIL")
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>stabilis validation report</title>",
        f"<style>{_CSS}</style>",
        "<h1>stabilis validation report</h1>",
        f"<p class='sub'>Generated {_esc(report.generated)} &middot; "
        f"digest <span class='mono'>{report.digest[:16]}…</span></p>",
        f"<div class='verdict {verdict[0]}'>{verdict[1]} &mdash; "
        f"{sum(c.passed for c in report.checks)} of {len(report.checks)} "
        "checks passed</div>",
        "<div class='note'><strong>Why this document exists.</strong> "
        "ISO/IEC 17025:2017 clause 7.11.2 requires software to be validated "
        "by the laboratory. NOTE 2 exempts commercial software in general "
        "use; a new open-core tool fails that twice over, so this evidence "
        "is what makes adoption possible rather than a nicety. Every claim "
        "below states its tolerance and the reason for it.</div>",
    ]
    for tier in TIERS:
        rows = report.by_tier(tier)
        if not rows:
            continue
        title, blurb = _TIER_TITLES[tier]
        parts.append(f"<h2>{title}</h2><p class='sub'>{blurb}</p>")
        parts.append("<table><tr><th>Check</th><th>Claim</th>"
                     "<th>Tolerance and why</th><th>Measured</th><th></th></tr>")
        for c in rows:
            mark = ("pass", "PASS") if c.passed else ("fail", "FAIL")
            parts.append(
                f"<tr><td>{_esc(c.name)}</td><td>{_esc(c.claim)}</td>"
                f"<td>{_esc(c.tolerance)}</td>"
                f"<td class='mono'>{_esc(c.measured)}</td>"
                f"<td class='{mark[0]}'>{mark[1]}</td></tr>"
            )
        parts.append("</table>")

    parts.append("<h2>Reproducing this report</h2>")
    parts.append("<p>No network access is used and every input is either "
                 "generated from a published recursion or fixed by seed. "
                 "The digest covers the claims and the measured values, not "
                 "the timestamp, so a regeneration on the same library and "
                 "numpy version should reproduce it exactly.</p>")
    parts.append("<table><tr><th>Item</th><th>Value</th></tr>")
    for k, v in {**report.environment, "seeds": report.seeds,
                 "digest": report.digest}.items():
        parts.append(f"<tr><td>{_esc(k)}</td><td class='mono'>{_esc(v)}</td></tr>")
    parts.append("</table>")
    return "\n".join(parts)
