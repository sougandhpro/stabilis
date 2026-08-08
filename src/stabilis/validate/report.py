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
from ..synth import _kasdin_walter_impulse, _unit_h_alpha, power_law_freq
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
SEEDS = {"coverage": 22, "gaps": 7}

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
    return out


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
    return out


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
    return out


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
