"""The identity and certified tiers: claims that hold exactly.

Analytic identities that must hold to rounding, and agreement with the
exact-rational oracle in `certified.py`. No Monte-Carlo anywhere here, so
every check in this module either holds or is a bug.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .._specfun import chi2_ppf, gammainc_q
from ..reference import (
    NBS14_LEHMER_MODULUS,
    nbs14_1000point,
    nbs14_1000point_integers,
    nbs14_9point,
)
from ..stats import adev, mdev, oadev, ohdev, tdev
from ..stats.bias import deviation_bias_factor
from ..stats.wavelet import modwt
from ..synth import power_law_freq, power_law_phase
from ..types import FreqSeries, PhaseSeries
from .expected import expected_avar, wavelet_kernel
from .certified import certified_deviation, exact_phase_from_freq, lre
from .check import ESTIMATORS, SEEDS, Check

__all__ = []


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
    out.append(_wavelet_level_identity())
    out.extend(_streaming_identities())
    return out


def _wavelet_level_identity() -> Check:
    """The absolute level of the wavelet variance, which nothing else pins.

    Every non-Haar wavelet check in the suite used to be *scale-free*: the
    filter algebra certifies the filters, drift annihilation is a ratio at
    one filter, the EDOF score is `2 mean**2 / var`. Only Haar was anchored,
    by `2 nu**2 = oavar`. So a constant multiplicative error in the D4/D6/D8
    variance, or a wrong `tau_j <-> level` assignment, would have passed
    everything. This is the check that closes it, and it is exact.

    Two claims, scored together because either alone is insufficient:

    * `modwt` applied to a unit impulse returns the level-`j` equivalent
      filter, and that must equal the textbook upsampling cascade built in
      `validate/expected.py`. This pins the level loop, the `1/sqrt(2)`
      normalization and the `2**(j-1)` stride -- the shape.
    * that filter has squared norm `2**-j` exactly, the orthonormality of
      the transform. This pins the level, since for white FM
      `E[nu_j**2] = var(y) * ||a_j||**2`, and `tau_j = 2**(j-1) tau0` then
      makes `2 nu_j**2` the Allan variance.
    """
    worst_shape, worst_norm = 0.0, 0.0
    for name in ("haar", "d4", "d6", "d8"):
        for level in range(1, 7):
            a = wavelet_kernel(name, level)
            x = np.zeros(1 << 12)
            x[0] = 1.0
            got = modwt(x, name, levels=level)[level - 1]
            worst_shape = max(worst_shape, float(np.abs(got[:a.size] - a).max()),
                              float(np.abs(got[a.size:]).max()))
            worst_norm = max(worst_norm,
                             abs(float(a @ a) * (1 << level) - 1.0))
    ok = worst_shape < 1e-15 and worst_norm < 1e-14
    return Check(
        "identity", "MODWT equivalent filters, shape and level",
        "The transform's response at each level against the equivalent "
        "filter built independently by the upsampling cascade, and that "
        "filter's squared norm against 2**-j. Over haar, d4, d6, d8 and "
        "levels 1..6. Without this the non-Haar wavelet variances have no "
        "check on their absolute level at all -- see VALIDATION-DEBT "
        "item 11 for what that did and did not expose.",
        "1e-15 absolute on the taps and 1e-14 relative on the norm. Both "
        "are exact identities evaluated in floating point, so the only "
        "budget is accumulation over the cascade.",
        f"taps {worst_shape:.1e}, norm {worst_norm:.1e}",
        ok,
    )


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
            got = expected_avar(alpha, m, tau0)
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

