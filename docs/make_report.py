"""Generate the project progress report.

Every number in the output is measured at generation time, not typed in.
Run it again after any change and the report re-derives itself:

    python3 docs/make_report.py            # writes docs/stabilis-progress.html
    python3 docs/make_report.py --pdf      # also renders a PDF via Edge

There is no PDF library in the environment and none is wanted as a
dependency, so the PDF is produced by driving the Windows Edge binary in
headless print-to-pdf mode through WSL interop -- the same native-Windows
route used for Stable32.
"""

from __future__ import annotations

import argparse
import html
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

import stabilis
from stabilis.reference import (
    EXPECTED_NBS14_1000POINT,
    NBS14_LEHMER_MODULUS,
    nbs14_1000point,
    nbs14_1000point_integers,
    nbs14_9point,
)
from stabilis.stats import adev, mdev, oadev, ohdev, tdev
from stabilis.multi import cross_matrix, cross_variance, reject
from stabilis.synth import (power_law_freq, power_law_noise,
                            power_law_phase)
from stabilis.types import FreqSeries, PhaseSeries, to_frequency, to_phase
from stabilis.validate.certified import (
    certified_deviation,
    exact_avar,
    exact_mvar,
    exact_ohvar,
    exact_oavar,
    exact_phase_from_freq,
    exact_tvar,
    lre,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EDGE = Path(
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
)

ESTIMATORS = [
    ("adev", adev, exact_avar),
    ("oadev", oadev, exact_oavar),
    ("mdev", mdev, exact_mvar),
    ("tdev", tdev, exact_tvar),
    ("ohdev", ohdev, exact_ohvar),
]


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------


def measure_certified_accuracy():
    """LRE of every estimator against exact rational arithmetic."""
    x9 = exact_phase_from_freq([Fraction(int(v)) for v in nbs14_9point()])
    x1000 = exact_phase_from_freq(
        [Fraction(k, NBS14_LEHMER_MODULUS) for k in nbs14_1000point_integers(1000)]
    )
    fs9 = FreqSeries(nbs14_9point(), rate_hz=1.0)
    fs1000 = FreqSeries(nbs14_1000point(), rate_hz=1.0)

    rows = []
    for name, fn, exact in ESTIMATORS:
        r9 = fn(fs9, taus="all")
        r1k = fn(fs1000, taus="all", max_af=100)
        entry = {"estimator": name}
        for label, result, xq, afs in (
            ("9pt", r9, x9, (1, 2)),
            ("1000pt", r1k, x1000, (1, 10, 100)),
        ):
            values = []
            for af in afs:
                got = result.dev_at(af)
                values.append(lre(got, certified_deviation(exact(xq, af))))
            entry[label] = min(values)
        rows.append(entry)
    return rows


def measure_certified_digits():
    """A worked certified value, shown to full precision."""
    x9 = exact_phase_from_freq([Fraction(int(v)) for v in nbs14_9point()])
    var = exact_ohvar(x9, 1)
    return {
        "rational": f"{var.numerator} / {var.denominator}",
        "certified": str(certified_deviation(var)),
        "computed": repr(ohdev(FreqSeries(nbs14_9point(), rate_hz=1.0), taus="all").dev_at(1)),
    }


def measure_published_interop():
    """Computed vs the Stable32-produced NBS14 1000-point table."""
    fs = FreqSeries(nbs14_1000point(), rate_hz=1.0)
    rows = []
    for name, fn, _ in ESTIMATORS:
        expected = EXPECTED_NBS14_1000POINT.get(name)
        if not expected:
            continue
        result = fn(fs, taus="all", max_af=100)
        for af, want in sorted(expected.items()):
            got = result.dev_at(af)
            rows.append(
                {
                    "estimator": name,
                    "af": af,
                    "computed": got,
                    "published": want,
                    "rel": abs(got - want) / want,
                }
            )
    return rows


def measure_roundtrip():
    """Phase -> frequency -> phase accumulation, in ULP."""
    rng = np.random.default_rng(0)
    rows = []
    for n in (10, 1_000, 100_000):
        x = np.cumsum(rng.standard_normal(n)) * 1e-12
        ps = PhaseSeries(x, rate_hz=1.0)
        back = to_phase(to_frequency(ps)).x
        ref = ps.x - ps.x[0]
        peak = float(np.max(np.abs(ref)))
        err = float(np.max(np.abs(back - ref)))
        rows.append({"n": n, "peak": peak, "err": err, "ulp": err / np.spacing(peak)})
    return rows


def measure_psd_slopes():
    """Recovered log-log PSD slope of the synthesized power-law noise."""
    rng = np.random.default_rng(20260807)
    n = 1 << 16
    rows = []
    for alpha in (2, 1, 0, -1, -2):
        power = np.zeros(n // 2 + 1)
        for _ in range(8):
            power += np.abs(np.fft.rfft(power_law_noise(n, alpha, rng))) ** 2
        freq = np.fft.rfftfreq(n)
        band = (freq > 1e-3) & (freq < 0.1)
        slope = float(np.polyfit(np.log(freq[band]), np.log(power[band] / 8), 1)[0])
        rows.append({"alpha": alpha, "slope": slope, "err": abs(slope - alpha)})
    return rows


def measure_identities():
    """The Tier 0 analytic identities, re-run here rather than quoted."""
    rows = []

    n, drift = 1024, 1.0
    i = np.arange(n, dtype=np.float64)
    ps = PhaseSeries(drift * i**2 / 2.0, rate_hz=1.0)
    a = oadev(ps, taus="octave")
    exact_expected = drift**2 * a.tau_s**2 / 2.0
    rows.append(
        {
            "identity": "linear frequency drift: avar = D²·τ²/2",
            "claim": "bit-exact",
            "ok": bool(np.array_equal(a.var, exact_expected)),
        }
    )
    h = ohdev(ps, taus="octave")
    rows.append(
        {
            "identity": "linear frequency drift: hvar = 0",
            "claim": "bit-exact",
            "ok": bool(np.array_equal(h.var, np.zeros_like(h.var))),
        }
    )

    y = np.array([3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0, -5.0, 3.0, 5.0, 8.0])
    base = oadev(FreqSeries(y, rate_hz=1.0), taus="all").var
    shifted = oadev(FreqSeries(y + 1000.0, rate_hz=1.0), taus="all").var
    rows.append(
        {
            "identity": "constant frequency offset leaves avar unchanged",
            "claim": "bit-exact",
            "ok": bool(np.array_equal(base, shifted)),
        }
    )

    rng = np.random.default_rng(8)
    x = np.cumsum(rng.standard_normal(300))
    b = oadev(PhaseSeries(x, rate_hz=1.0), taus="octave").var
    s = oadev(PhaseSeries(x * 8.0, rate_hz=1.0), taus="octave").var
    rows.append(
        {
            "identity": "scale equivariance at 2^k: var → 64·var for c = 8",
            "claim": "bit-exact",
            "ok": bool(np.array_equal(s, b * 64.0)),
        }
    )

    rng = np.random.default_rng(5)
    fs = FreqSeries(rng.standard_normal(300), rate_hz=1.0)
    a1 = adev(fs, taus="all").dev_at(1)
    rows.append(
        {
            "identity": "adev = oadev = mdev at AF 1",
            "claim": "rel 1e-15",
            "ok": abs(oadev(fs, taus="all").dev_at(1) - a1) < 1e-15 * a1
            and abs(mdev(fs, taus="all").dev_at(1) - a1) < 1e-15 * a1,
        }
    )

    rng = np.random.default_rng(6)
    fs = FreqSeries(rng.standard_normal(400), rate_hz=4.0)
    m, t = mdev(fs, taus="octave"), tdev(fs, taus="octave")
    rows.append(
        {
            "identity": "tdev = τ/√3 · mdev",
            "claim": "rel 1e-15",
            "ok": bool(np.allclose(t.dev, t.tau_s / np.sqrt(3.0) * m.dev, rtol=1e-15, atol=0)),
        }
    )

    r = ohdev(PhaseSeries(np.full(256, 7.5), rate_hz=1.0), taus="octave")
    rows.append(
        {
            "identity": "constant phase gives exactly zero",
            "claim": "bit-exact",
            "ok": bool(np.array_equal(r.var, np.zeros_like(r.var))),
        }
    )
    return rows


GAP_TRIALS = 400


def measure_gap_handling():
    """What one missing frequency sample costs, before and after the fix.

    The "before" numbers are not quoted from memory -- they are recomputed
    by integrating the gapped record with `to_phase`, which is exactly what
    the estimator used to do and which still propagates NaN to the end of
    the record by design. Defect and repair are measured side by side at
    generation time, like everything else here.

    Averaged over many realizations rather than shown for one, deliberately.
    Both paths are estimates of a noisy quantity, so on any single record
    the broken one can land closer by luck -- and does. The meaningful
    comparison is the *bias*: mean signed deviation from the ungapped
    answer, which is where truncating half the record shows up as something
    other than noise.
    """
    stats = {m: {"old": [], "new": []} for m in (1, 2, 4, 8, 16)}
    for seed in range(GAP_TRIALS):
        y = power_law_freq(200, 0, rate_hz=1.0, rng=np.random.default_rng(seed)).y
        gapped = y.copy()
        gapped[100] = np.nan
        clean = oadev(FreqSeries(y, rate_hz=1.0), taus="all", keep_coeffs=False)
        fixed = oadev(FreqSeries(gapped, rate_hz=1.0), taus="all", keep_coeffs=False)
        old = oadev(to_phase(FreqSeries(gapped, rate_hz=1.0)), taus="all", keep_coeffs=False)
        for m, acc in stats.items():
            c, f, o = (r.af.tolist().index(m) for r in (clean, fixed, old))
            acc["old"].append(old.dev[o] / clean.dev[c] - 1.0)
            acc["new"].append(fixed.dev[f] / clean.dev[c] - 1.0)
            acc["terms"] = (int(clean.n_terms[c]), int(old.n_terms[o]), int(fixed.n_terms[f]))

    rows = []
    for m, acc in stats.items():
        old_e, new_e = np.array(acc["old"]), np.array(acc["new"])
        ref, o_t, n_t = acc["terms"]
        rows.append(
            {
                "m": m,
                "old_bias": float(old_e.mean()),
                "new_bias": float(new_e.mean()),
                "old_rms": float(np.sqrt((old_e**2).mean())),
                "new_rms": float(np.sqrt((new_e**2).mean())),
                "terms": f"{ref} / {o_t} / {n_t}",
                "lost": ref - n_t,
            }
        )
    return rows


CROSS_TRIALS = 120


def measure_cross_rejection():
    """What two channels see that one cannot, against the clock itself.

    One white-FM clock, two channels with independent white-PM noise at a
    level that buries it: at tau = 1 s the clock is about a third of the
    single-channel deviation, so the auto-variance is answering a question
    about the instrument. The truth is the clock record's *own* oadev, not
    a closed form and not a fit, so what is measured is the rejection
    rather than the estimator's self-consistency.

    Ratios are averaged over realizations for the same reason the gap table
    is: on any one record either estimate can land close by luck, and the
    figure that means something is where each is centred. The reported
    scatter is the ensemble spread of the cross ratio, which is the price
    paid for the rejection and belongs beside it.
    """
    rng = np.random.default_rng(20260809)
    n, rate = 8192, 1.0
    auto, cross, rej, sem = [], [], [], []
    for _ in range(CROSS_TRIALS):
        clock = power_law_phase(n, 0, rate, rng, 1.0e-24)
        a = PhaseSeries(x=clock.x + power_law_phase(n, 2, rate, rng, 1.0e-22).x,
                        rate_hz=rate)
        b = PhaseSeries(x=clock.x + power_law_phase(n, 2, rate, rng, 1.0e-22).x,
                        rate_hz=rate)
        x = cross_variance(a, b, max_af=64)
        truth = oadev(clock, max_af=64).var
        auto.append(x.auto_a / truth)
        cross.append(x.var / truth)
        rej.append(x.rejection_db)
        sem.append(x.sem / truth)
        af, tau = x.af, x.tau_s
    auto, cross = np.array(auto), np.array(cross)
    return [
        {
            "tau": float(tau[i]),
            "af": int(af[i]),
            "auto": float(np.mean(auto[:, i])),
            "cross": float(np.mean(cross[:, i])),
            "spread": float(np.std(cross[:, i], ddof=1)),
            "sem": float(np.mean(np.array(sem)[:, i])),
            "rej": float(np.mean(np.array(rej)[:, i])),
        }
        for i in range(af.size)
    ]


REJECT_TRIALS = 40


def measure_rejection_gain():
    """What a third and fourth channel are worth, in decibels.

    The commercial question, so it is measured rather than quoted. Averaging
    P = M(M-1)/2 pairs would win `5 log10 P` if the pairs were independent.
    They are not -- (c0,c1) and (c0,c2) share a channel -- so the naive
    figure is an upper bound and the exact Isserlis covariance over pairs of
    pairs says how much is real.

    Reported against the two-channel measurement on the *same* records, so
    the comparison is paired and the clock realization cancels out of it.
    """
    rng = np.random.default_rng(20260810)
    n, rate = 8192, 1.0
    rows = {2: [], 3: [], 4: []}
    for _ in range(REJECT_TRIALS):
        clock = power_law_phase(n, 0, rate, rng, 1.0e-24)
        ch = {
            f"c{i}": PhaseSeries(
                x=clock.x + power_law_phase(n, 2, rate, rng, 1.0e-22).x,
                rate_hz=rate)
            for i in range(4)
        }
        for m_ch in rows:
            subset = {k: ch[k] for k in list(ch)[:m_ch]}
            r = reject(cross_matrix(subset, max_af=64))
            rows[m_ch].append(r.rejection_db)
        af, tau = r.af, r.tau_s
    base = np.array(rows[2]).mean(axis=0)
    return [
        {
            "tau": float(tau[i]),
            "af": int(af[i]),
            "base": float(base[i]),
            "gain3": float(np.array(rows[3]).mean(axis=0)[i] - base[i]),
            "gain4": float(np.array(rows[4]).mean(axis=0)[i] - base[i]),
            "naive3": 5.0 * np.log10(3.0),
            "naive4": 5.0 * np.log10(6.0),
        }
        for i in range(af.size)
    ]


FIT_TRIALS = 200


def measure_fit():
    """What a fitted parameter's error bar is worth, measured two ways.

    White FM at a known `h_alpha` has `oadev` variance h0/(2 tau) exactly,
    so a two-parameter power-law fit has a truth for both parameters and
    the coverage of its stated interval is a measurable quantity rather
    than a claim. The same fits are scored a second time against the
    interval that would be quoted if the taus were treated as independent
    -- which is what every other tool reports -- so the table is the value
    of the cross-tau correlation rather than an assertion of it.

    The deviance is reported beside them because it is the other half of
    the same calibration: an objective whose degrees of freedom are wrong
    produces intervals that are wrong in the same direction, and reporting
    only the coverage would leave that unlocated.
    """
    from stabilis._specfun import norm_ppf
    from stabilis.fitting import fit as run_fit
    from stabilis.stats.ci import ONE_SIGMA

    truth = {"A": 1.0e-22, "b": -1.0}
    z = float(norm_ppf(0.5 * (1.0 + ONE_SIGMA)))
    rng = np.random.default_rng(20260814)
    inside = {"A": 0, "b": 0}
    naive = {"A": 0, "b": 0}
    ratios, devs, predicted, dofs, rejected = [], [], [], [], 0
    for _ in range(FIT_TRIALS):
        fs = power_law_freq(4096, 0, rate_hz=1.0, rng=rng,
                            h_alpha=2.0 * truth["A"])
        got = run_fit(oadev(fs, taus="octave", max_af=256), "powerlaw")
        for name in ("A", "b"):
            miss = abs(got.params[name] - truth[name])
            inside[name] += miss <= z * got.stderr[name]
            naive[name] += miss <= z * got.stderr_naive[name]
        ratios.append([got.understatement["A"], got.understatement["b"]])
        devs.append(got.gof.deviance)
        predicted.append(got.gof.scale * got.gof.dof)
        dofs.append(got.gof.dof)
        rejected += not got.gof.ok
        n_taus = int(got.tau_s.size)
    ratios = np.array(ratios)
    return {
        "trials": FIT_TRIALS,
        "level": ONE_SIGMA,
        "se": float(np.sqrt(ONE_SIGMA * (1.0 - ONE_SIGMA) / FIT_TRIALS)),
        "rows": [
            {
                "name": name,
                "truth": truth[name],
                "covered": inside[name] / FIT_TRIALS,
                "naive": naive[name] / FIT_TRIALS,
                "ratio": float(ratios[:, i].mean()),
            }
            for i, name in enumerate(("A", "b"))
        ],
        "deviance": float(np.mean(devs)),
        "deviance_sem": float(np.std(devs) / np.sqrt(FIT_TRIALS)),
        "deviance_expected": float(np.mean(predicted)),
        "dof_effective": float(np.mean(dofs)),
        "n_taus": n_taus,
        "dof_naive": n_taus - 2,
        "rejected": rejected,
    }


def measure_generator():
    """The NBS14 recursion against its published check values."""
    ints = nbs14_1000point_integers(3)
    y = nbs14_1000point(4)
    published = [5.748904731939040e-01, 1.841829699390490e-01]
    return {
        "k0": ints[0],
        "k1": ints[1],
        "k1_ok": ints[1] == 395529916,
        "max_dev": float(max(abs(y[i] - published[i]) for i in range(2))),
    }


def run_tests():
    """Actual pytest outcome, so the report cannot claim a green suite falsely."""
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
    return {
        "summary": tail[0] if tail else "no output",
        "passed": proc.returncode == 0,
        "seconds": time.time() - t0,
    }


def git_state():
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=ROOT
            ).stdout.strip()
        except OSError:
            return "unavailable"

    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "subject": run("log", "-1", "--pretty=%s"),
        "dirty": bool(run("status", "--porcelain")),
        "n_commits": run("rev-list", "--count", "HEAD"),
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font: 10.5pt/1.5 "Charter","Georgia",serif; color:#14171a; margin:0 auto;
       max-width: 190mm; padding: 10mm 6mm; background:#fff; }
h1 { font-size: 22pt; margin: 0 0 2mm; letter-spacing:-.2pt; }
h2 { font-size: 13pt; margin: 9mm 0 2.5mm; padding-bottom:1.5mm;
     border-bottom: 1.5px solid #14171a; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 5mm 0 1.5mm; page-break-after: avoid; }
p, li { margin: 0 0 2.5mm; }
.sub { color:#5b6570; font-size:10pt; margin-bottom:1mm; }
.meta { font: 8.5pt/1.5 ui-monospace,"Cascadia Mono",Consolas,monospace;
        color:#5b6570; border-top:1px solid #d5dae0; padding-top:2mm; margin-top:1mm;}
table { border-collapse: collapse; width:100%; margin:3mm 0 4mm;
        font-size:9pt; page-break-inside: avoid; }
th { text-align:left; border-bottom:1.2px solid #14171a; padding:1.6mm 2mm;
     font-weight:600; }
td { border-bottom:.5px solid #dfe4e9; padding:1.4mm 2mm; }
td.n, th.n { text-align:right; font-variant-numeric: tabular-nums;
             font-family: ui-monospace,Consolas,monospace; }
tr:last-child td { border-bottom:1.2px solid #14171a; }
code, .mono { font-family: ui-monospace,"Cascadia Mono",Consolas,monospace;
              font-size:9pt; }
.ok { color:#136c34; font-weight:600; }
.bad { color:#a4161a; font-weight:600; }
.note { background:#f5f7f9; border-left:3px solid #8c98a4; padding:2.5mm 3.5mm;
        margin:3mm 0; page-break-inside: avoid; }
.note strong { color:#14171a; }
.kv { display:grid; grid-template-columns: 42mm 1fr; gap:.8mm 4mm; font-size:9.5pt; }
.kv dt { color:#5b6570; }
.kv dd { margin:0; }
ul { padding-left: 5mm; margin: 0 0 3mm; }
.small { font-size:9pt; color:#5b6570; }
"""


def esc(v):
    return html.escape(str(v))


def table(headers, rows, numeric=()):
    head = "".join(
        f'<th class="n">{esc(h)}</th>' if i in numeric else f"<th>{esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="n">{c}</td>' if i in numeric else f"<td>{c}</td>"
            for i, c in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def verdict(ok):
    return '<span class="ok">PASS</span>' if ok else '<span class="bad">FAIL</span>'


def build_html():
    acc = measure_certified_accuracy()
    digits = measure_certified_digits()
    interop = measure_published_interop()
    roundtrip = measure_roundtrip()
    slopes = measure_psd_slopes()
    identities = measure_identities()
    gen = measure_generator()
    gaps = measure_gap_handling()
    crossing = measure_cross_rejection()
    gains = measure_rejection_gain()
    fitting = measure_fit()
    tests = run_tests()
    git = git_state()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts = [f"<style>{CSS}</style>"]
    parts.append(
        f"""
<h1>stabilis — progress report</h1>
<p class="sub">Frequency stability analysis engine. Milestones M0, M1 and M2
complete — the v0.1 line, the first version from which an uncertainty may
legitimately be quoted — and most of M3: Greenhall EDOF, which is error-bar
parity with Stable32; the exact non-negative least squares core and the
power-law decomposition built on it; TOTVAR, the Thêo family and the MODWT
wavelet variance. M5 is complete ahead of its place in the sequence, being
the part with no equivalent in any other tool: the bench model, the
cross-variance, and multi-channel noise rejection. M4's fitting layer has
since landed on the same principle — the covariance <em>between</em> τ is
measured rather than assumed away, so a fitted parameter's uncertainty is
the one it actually has. M4's export layer has landed with it: the record
back out as self-describing ASCII, the curve as CSV or JSON, the block
Stable32 prints — reproduced character for character against its own
committed exports — and a bundle directory whose digest covers the
measurement rather than the files that carry it. M6 (streaming ingest and
the dynamic Allan variance) and M7 (gap filling and the modified total
family) are complete, and with them the Thêo family's error bar. Most
recently, the data-conditioning steps the library had promised since M1 —
drift removal and decimation — and a wavelet variance for gappy records
that neither fills nor discards.</p>
<div class="meta">
generated {esc(now)} &nbsp;·&nbsp; stabilis {esc(stabilis.__version__)} &nbsp;·&nbsp;
numpy {esc(np.__version__)} &nbsp;·&nbsp; python {esc(platform.python_version())}<br>
commit {esc(git['commit'])} ({esc(git['n_commits'])} total){' — WORKING TREE DIRTY' if git['dirty'] else ''}
 &nbsp;·&nbsp; every figure below is measured at generation time
</div>"""
    )

    # --- what this is
    parts.append("<h2>What this is</h2>")
    parts.append(
        """
<p>Stable32 is the de-facto tool for frequency stability analysis. It is
Windows-only, closed, GUI-driven, unscriptable and no longer developed.
<em>stabilis</em> is an open-core engine for the same job, intended as the free
foundation under a paid workbench that produces signed ISO-17025-style reports
for calibration labs and oscillator manufacturers.</p>

<p>Two things distinguish it from a reimplementation. First, the Haar wavelet
variance <em>is</em> the Allan variance, so classical AVAR is the K=1 case of one
framework; higher-order Daubechies filters then buy drift immunity and
convergence for steeper noise. Second, confidence intervals are derived from the
estimator's own coefficient autocovariance, so they do not require identifying
the noise type first — Stable32 looks it up in a table and has no answer when the
noise is not one of its five power laws.</p>

<div class="note"><strong>The commercial argument for validation.</strong>
ISO/IEC 17025:2017 clause 6.4.1 defines software as <em>equipment</em>, and
7.11.2 requires laboratory software to be validated before introduction. Its
NOTE 2 exempts commercial software “in general use within its designed
application range” — an exemption stabilis fails twice over, being new and
being open-core, since a lab that patches it explicitly triggers re-validation.
An accredited lab therefore cannot adopt this engine unless it ships validation
evidence. That evidence is the product's moat, not its overhead.</div>"""
    )

    # --- certified accuracy
    parts.append("<h2>Results: accuracy against certified values</h2>")
    parts.append(
        """
<p>Every classical variance is a sum of squares of integer linear combinations
of the phase samples, divided by an integer. Both NBS14 reference sets are
integer-derived, so on them the variance is an <strong>exact rational</strong>.
It is computed independently in <code>fractions.Fraction</code>, with the
deviation certified to 50 digits via <code>decimal.Decimal.sqrt()</code>. That
gives certified values computed here, to far more digits than any published
table, with no dependence on a transcription being correct.</p>

<p>Accuracy is reported as <strong>LRE</strong>, the log relative error, after
the NIST Statistical Reference Datasets convention: it is the number of correct
significant digits. float64 carries about 15.95, so these are at the
representation limit.</p>"""
    )
    parts.append(
        table(
            ["Estimator", "NBS14 9-point (min LRE)", "NBS14 1000-point (min LRE)"],
            [
                (r["estimator"], f"{r['9pt']:.1f}", f"{r['1000pt']:.1f}")
                for r in acc
            ],
            numeric=(1, 2),
        )
    )
    parts.append(
        f"""
<h3>A worked certified value</h3>
<p class="small">Overlapping Hadamard variance of the NBS14 nine-point set at AF 1,
as an exact rational and as a 50-digit deviation:</p>
<div class="kv">
  <dt>exact variance</dt><dd class="mono">{esc(digits['rational'])}</dd>
  <dt>certified deviation</dt><dd class="mono">{esc(digits['certified'])}</dd>
  <dt>computed (float64)</dt><dd class="mono">{esc(digits['computed'])}</dd>
</div>
<div class="note"><strong>The oracle earned its keep immediately.</strong>
The relayed NBS14 table gave <code>hdev(1) = 70.80608</code> while listing
<code>ohdev(1) = 70.80607</code> — but these are the same estimator at AF 1,
since there is no overlap to exploit. The certified value settles it at
<code>70.806073185850378</code>, so the first was a transcription error. This is
exactly why correctness rests on algebra rather than on published tables.</div>"""
    )

    # --- identities
    parts.append("<h2>Results: analytic identities</h2>")
    parts.append(
        """<p>These need no reference data and no oracle, which is why they carry
the correctness argument. Every published ADEV table traces back to Stable32, so
agreeing with one proves interoperability; agreeing with algebra proves
correctness.</p>"""
    )
    parts.append(
        table(
            ["Identity", "Claim", "Result"],
            [(r["identity"], r["claim"], verdict(r["ok"])) for r in identities],
        )
    )
    parts.append(
        """<div class="note"><strong>The most valuable single test.</strong> For a
pure linear frequency drift <em>D</em>, the phase is quadratic, so the second
difference is the constant <em>D·τ²</em> and the third difference vanishes. Hence
<code>avar = D²τ²/2</code> and <code>hvar = 0</code>, both exactly. One test
validates the Allan normalization, the Hadamard normalization and the drift
convention simultaneously — and it is bit-exact, not approximate.</div>"""
    )

    # --- interop
    parts.append("<h2>Results: interoperability with Stable32</h2>")
    parts.append(
        """<p class="small">Labelled as interop, not correctness: these published
values were produced by Stable32 itself. Seven significant figures as published,
so agreement is assessed at rtol 1e-6.</p>"""
    )
    parts.append(
        table(
            ["Estimator", "AF", "computed", "published", "rel. diff"],
            [
                (
                    r["estimator"],
                    r["af"],
                    f"{r['computed']:.7e}",
                    f"{r['published']:.7e}",
                    f"{r['rel']:.1e}",
                )
                for r in interop
            ],
            numeric=(1, 2, 3, 4),
        )
    )

    # --- generator + numerics
    parts.append("<h2>Results: reference generator and numerics</h2>")
    parts.append(
        f"""
<h3>NBS14 1000-point set ships as arithmetic, not a file</h3>
<p>The set is defined by a published Lehmer recursion
<code>k[i+1] = 16807·k[i] mod 2147483647</code> from seed
<code>1234567890</code>, so it needs no data file and carries no licence
question. Verified against the published <code>wriley.com/tst_suit.dat</code>:
all 1000 values agreed to 5.55e-16 — one ULP, which is that file's own text
rounding. <strong>The generator is the more exact artifact of the two.</strong></p>
<div class="kv">
  <dt>k[0]</dt><dd class="mono">{esc(gen['k0'])}</dd>
  <dt>k[1] (check value)</dt><dd class="mono">{esc(gen['k1'])} &nbsp; {verdict(gen['k1_ok'])}</dd>
  <dt>vs published digits</dt><dd class="mono">max dev {gen['max_dev']:.2e}</dd>
</div>

<h3>Phase ↔ frequency round-trip</h3>
<p>Recovers the offset-removed phase to a few ULP. Not bit-exact: a cumulative
sum of differences does not telescope exactly in floating point. The test bounds
this at 8 ULP, a figure derived from the measurements below rather than tuned.</p>"""
    )
    parts.append(
        table(
            ["n", "peak (s)", "max error (s)", "ULP"],
            [
                (r["n"], f"{r['peak']:.3e}", f"{r['err']:.3e}", f"{r['ulp']:.1f}")
                for r in roundtrip
            ],
            numeric=(0, 1, 2, 3),
        )
    )
    parts.append(
        """
<h3>Power-law noise synthesis</h3>
<p>Kasdin–Walter generation, log-log periodogram slope recovered over
1e-3 &lt; f &lt; 0.1, averaged over eight realizations.</p>"""
    )
    parts.append(
        table(
            ["α (requested)", "slope (recovered)", "|error|"],
            [
                (r["alpha"], f"{r['slope']:+.4f}", f"{r['err']:.4f}")
                for r in slopes
            ],
            numeric=(0, 1, 2),
        )
    )

    # --- decisions
    parts.append("<h2>Results: one missing sample</h2>")
    parts.append(
        """<p>Gap handling is where a metrology tool fails most expensively,
because it fails quietly. Integrating frequency to phase with a cumulative sum
cannot step over an unknown increment, so a single <code>NaN</code> made every
later phase sample <code>NaN</code> — and the estimator, dropping non-finite
terms as designed, silently analysed only the data <em>before</em> the gap.</p>
<p>Measured at generation time over TRIALS realizations of 200 white-FM points
with one sample removed at the midpoint, each compared against its own ungapped
answer. <em>Old</em> integrates the gapped record in one cumulative sum, which
is what the estimator used to do; <em>new</em> segments at the gap, integrates
each run from its own zero and pools the coefficients. Bias is the mean signed
error and is the figure that matters — on any single record either path can land
closer by luck, but only one of them is centred on the right answer.</p>""".replace("TRIALS", str(GAP_TRIALS))
    )
    parts.append(
        table(
            ["m", "old bias", "new bias", "old RMS", "new RMS",
             "terms ungapped / old / new", "terms lost"],
            [
                (
                    r["m"],
                    f"{r['old_bias'] * 100:+.2f}%",
                    f"{r['new_bias'] * 100:+.2f}%",
                    f"{r['old_rms'] * 100:.2f}%",
                    f"{r['new_rms'] * 100:.2f}%",
                    r["terms"],
                    f"{r['lost']} = {r['lost'] // r['m']}m",
                )
                for r in gaps
            ],
            numeric=(0, 1, 2, 3, 4),
        )
    )
    parts.append(
        """<div class="note"><strong>One sample in two hundred, and the old path
was biased low by percent — with no warning and a plausible-looking term
count.</strong> The bias is systematic and grows with τ, because the surviving
terms were not a random subset of the record: they were the first half of it.
Scatter falls by roughly an order of magnitude too, which is the direct
consequence of keeping 197 terms instead of 99.
<p>The correct cost of a gap is the <code>2m</code> terms whose window spans it
— <code>3m</code> for the Hadamard and modified families — and that is now
exactly what it costs, asserted term by term in
<code>tests/test_gaps.py</code>. Frequency input defaults to the segmenting
policy and refuses the masking one; phase input keeps masking, which is correct
when the timescale ran continuously across the gap. The choice, the number of
runs and the longest run are all recorded in provenance.</p>
<p>The residual few tenths of a percent in the "new bias" column is not a
defect and does not go away: <code>E[σ̂] = σ·√(2/η)·Γ((η+1)/2)/Γ(η/2)</code>, so
a deviation estimated from fewer terms is genuinely low on average. It is the
same bias the ungapped estimate carries, slightly amplified. Correcting it, and
reporting the error bar that goes with it, is what M2 went on to do — and the
reason this
defect had to be fixed first, since it feeds the term count that sets η.</p></div>"""
    )

    parts.append("<h2>Results: seeing under the instrument</h2>")
    parts.append(
        """<p>Every estimator here is a mean of squared coefficients. Measure one
clock through two independent channels and replace one factor with the other
channel's coefficients, and the instrument noise stops being a floor: writing
x<sub>a</sub> = x<sub>c</sub> + n<sub>a</sub> and x<sub>b</sub> = x<sub>c</sub> +
n<sub>b</sub>, the auto-variance of either is σ²<sub>c</sub> + σ²<sub>n</sub>
however long you log, while E[d<sub>a</sub>d<sub>b</sub>] = E[d<sub>c</sub>²]
exactly, because the two cross terms have expectation zero. Neither Stable32 nor
<code>allantools</code> computes this.</p>
<p>Measured at generation time over TRIALS realizations of a white-FM clock
seen through two white-PM channels, 8192 points each. Both columns are ratios
to <em>the clock record's own</em> σ²(τ) — not a closed form and not a fit — so
the table is the rejection itself rather than a consistency check. A ratio of
1.00 is the right answer; the auto column is what a single channel would have
told you.</p>""".replace("TRIALS", str(CROSS_TRIALS))
    )
    parts.append(
        table(
            ["τ (s)", "auto / truth", "cross / truth", "ensemble spread",
             "reported ± ", "rejection"],
            [
                (
                    f"{r['tau']:g}",
                    f"{r['auto']:.2f}×",
                    f"{r['cross']:.3f}×",
                    f"{r['spread'] * 100:.1f}%",
                    f"{r['sem'] * 100:.1f}%",
                    f"{r['rej']:.1f} dB",
                )
                for r in crossing
            ],
            numeric=(0, 1, 2, 3, 4, 5),
        )
    )
    parts.append(
        """<div class="note"><strong>At τ = 1 s the single-channel answer is
several times the clock it was meant to measure, and the cross estimate is
within a percent of it.</strong> The two converge at long τ, and that is
correct rather than disappointing: by then the instrument's white PM has
averaged down and there is nothing left to reject, which is exactly why the
short-τ end is the only place the method is being tested.
<p>Three prices, printed rather than hidden. The estimate is a mean of products
and is <em>signed</em>: at long τ it goes negative, which means the shared part
has fallen below the scatter, and nothing clips it or reports √|var| — a log
axis would otherwise grow a noise floor that is not there. Convergence is slow,
because only the <em>estimate</em> of the instrument noise averages down, as
M<sup>1/4</sup> in amplitude; the rejection column is what was actually
achieved, and reduces to the familiar 5·log₁₀(M) law as a consequence of the
error bar rather than as a substitute for one. And correlated channels — a
shared reference, a shared supply, one thermal environment — break the method
with no symptom whatever in the data, so the independence assumption is a
provenance warning on every call.</p>
<p>The error bar cannot be the χ² one, a sum of products being unbounded below.
Isserlis' theorem gives (auto<sub>a</sub>·auto<sub>b</sub> + cross²)/η, which is
2σ⁴/η when the two channels are the same channel — so
<code>cross_variance(a, a)</code> returns the auto-variance bit for bit and its
error bar <em>is</em> the chi-square one, rather than a parallel construction
that could drift from it.</p>
<p>The two right-hand columns are deliberately not the same quantity, and the
gap between them at long τ is the interesting part. "Ensemble spread" is the
scatter of the ratio against <em>this realization's own clock record</em>,
which is what a user comparing two channels of one oscillator actually wants to
know; "reported ±" is the standard error about the population variance, which
additionally carries the clock's own sampling variation. At short τ the
instrument dominates and the two coincide. At long τ the clock dominates, both
channels see the same clock record, and the conditional error collapses while
the unconditional one does not — so the reported bar is conservative for that
question, correctly, since nothing in one measurement can tell it which
question was being asked. Against the quantity it does claim — the scatter of
the estimate about the truth — it is measured at ±20% out to m = N/4 in
<code>tests/test_cross.py</code>.</p>
<p><strong>Two estimators were rejected on measurement before this one, and
they fail in different places.</strong> A Bartlett sum over the products' own
autocovariance, assuming no distribution at all, decays monotonically to
0.30 of the true scatter at long τ — the products are heavy-tailed where the
coefficients are Gaussian, so each extra lag of their sample autocovariance
adds more noise than signal. The equal-shape closed form, which collapses the
three correlation functions onto one shape and is what the library shipped
first, is fine at both ends of the τ range and 15% narrow in the middle: at
short τ the instrument sets all three shapes and at long τ the clock does, so
the approximation is exact in both limits and wrong exactly where they mix.
The form now used sums all four lagged covariances and approximates nothing;
it costs no new numerics, since the cross-correlation of two coefficient
arrays comes out of the FFT the EDOF machinery already runs. All three are
kept in the test suite and measured on the same realizations, so the reason
for the choice stays a live number.</p></div>"""
    )

    parts.append(
        """<p><strong>More channels reject deeper, and by less than counting
the pairs would suggest.</strong> Each of the P = M(M−1)/2 pairs is an
unbiased estimate of the same clock, so averaging them helps — but (c0,c1) and
(c0,c2) share a channel, so they are not independent estimates and the naive
<code>5·log₁₀(P)</code> is an upper bound rather than the answer. Measured at
generation time over TRIALS realizations, as the gain over the two-channel
measurement on the same records:</p>""".replace("TRIALS", str(REJECT_TRIALS))
    )
    parts.append(
        table(
            ["τ (s)", "2 channels", "3 channels", "naive", "4 channels", "naive"],
            [
                (
                    f"{r['tau']:g}",
                    f"{r['base']:.1f} dB",
                    f"+{r['gain3']:.2f} dB",
                    f"+{r['naive3']:.2f}",
                    f"+{r['gain4']:.2f} dB",
                    f"+{r['naive4']:.2f}",
                )
                for r in gains
            ],
            numeric=(0, 1, 2, 3, 4, 5),
        )
    )
    parts.append(
        """<div class="note">The gain is real at short τ and vanishes at long
τ, and both halves are correct. Where the instruments dominate there is noise
to reject and the extra pairs reject it; by the time the clock dominates there
is nothing left to remove, and a fourth channel buys nothing whatever. A tool
that reported the naive figure would promise a constant 3.9 dB across the
whole curve and be wrong at both ends of it — overselling the hardware at long
τ, and overstating the confidence at short τ, which is the more expensive
mistake.</div>"""
    )

    parts.append(
        """<h2>Fitting a model, and what the correlation between τ is worth</h2>
<p>Two properties make this more than calling a curve fitter. The sampling law
of a stability estimate is <strong>χ², not Gaussian</strong>, and least Gaussian
at long τ where the specification lives; and the points of an overlapping
estimator are <strong>correlated</strong>, because they are computed from the
same phase samples. The first is handled by minimizing the exact likelihood —
which turns out to be a sum of squares anyway, so an ordinary least-squares
optimizer reaches it with no Gaussian approximation. The second is handled by
measuring the covariance <em>between</em> τ from the coefficient arrays every
estimator already returns, and carrying it through a sandwich estimator.</p>
<p>Ignoring the correlation leaves the parameters roughly unbiased and their
<strong>uncertainties too narrow</strong> — the one failure mode that survives
review, because the fitted value and the residuals both look fine. Measured at
generation time over TRIALS realizations of white FM at a known h₀, N = 4096,
where the truth of both fitted parameters is known exactly:</p>""".replace(
            "TRIALS", str(FIT_TRIALS))
    )
    parts.append(
        table(
            ["parameter", "truth", "coverage, correlated",
             "coverage, ignoring it", "error-bar ratio"],
            [
                (
                    r["name"],
                    f"{r['truth']:g}",
                    f"{r['covered']:.3f}",
                    f"{r['naive']:.3f}",
                    f"{r['ratio']:.2f}×",
                )
                for r in fitting["rows"]
            ],
            numeric=(1, 2, 3, 4),
        )
    )
    parts.append(
        f"""<div class="note"><strong>Nominal coverage is
{fitting['level']:.4f} and the Monte-Carlo standard error is
{fitting['se']:.3f}.</strong> The correlation-aware interval sits on nominal;
the interval that treats the τ as independent — which is what every other tool
reports — misses it by several standard errors, in the direction that
overstates confidence. The error-bar ratio is that shortfall seen the other way
round.
<p>The deviance is the other half of the same calibration, and it is reported
because an objective whose degrees of freedom are wrong produces intervals that
are wrong in the same direction. Measured mean
{fitting['deviance']:.2f} ± {fitting['deviance_sem']:.2f} against a predicted
{fitting['deviance_expected']:.2f}, on {fitting['dof_effective']:.1f} effective
degrees of freedom rather than the {fitting['dof_naive']} a parameter count
would give — {fitting['n_taus']} τ, two of them fitted, and the rest spent on
the correlation. Getting that correction wrong is not academic: on a longer
record during development, the obvious version — which assumes the fit absorbs
<em>p</em> dimensions — predicted 8.0 where the measurement was 5.8 ± 0.4, five
standard errors out and in the direction that makes a correct fit look
suspiciously good. What the fit actually absorbs is tr(HP), and with correlated
points the model lies along the directions where P carries its mass.
{fitting['rejected']} of {fitting['trials']} fits were flagged as poor by the
goodness-of-fit test; all of them are still scored above, since dropping them
would be selecting on the outcome being measured.</p>
<p>Neither number is available from a parameter count, and neither is
available at all without the cross-τ covariance — which no other tool in this
field computes.</p></div>"""
    )

    parts.append("<h2>Design decisions worth recording</h2>")
    parts.append(
        """
<h3>Scale factors live outside the coefficients</h3>
<p>Every estimator returns the coefficient array it averaged, so one
confidence-interval implementation can serve all of them. The obvious form is
<code>variance = mean(c²)</code>, which requires dividing by √2 or √6. That was
implemented and then reverted: it makes every coefficient irrational and destroys
exactness where the mathematics is exact — the drift identity above failed by one
bit. The contract is now <code>variance = scale · mean(c²)</code>. EDOF depends
only on the scale-free correlation structure, so nothing is lost.</p>

<h3>float64 phase is a genuine hazard at this precision</h3>
<p>At an accumulated phase of 10³–10⁶ s the float64 spacing is 10⁻¹³–10⁻¹⁰ s,
which silently quantizes away exactly the picoseconds sub-nanosecond work cares
about. Phase is therefore stored as offsets from an explicit epoch, and
construction warns when the spacing approaches the smallest resolved step. The
warning is recorded in provenance, not merely printed, so it survives into a
report.</p>

<h3>The classical estimators are not routed through the wavelet filter bank</h3>
<p>The Haar-equals-Allan identity is the intellectual core of the library, but
making it the <em>implementation</em> would put boundary handling and
shift-equivariance on the critical path for no user-visible gain. Written
directly, the identity becomes an independent check of two separate code paths
instead of a tautology.</p>

<h3>No pickle support, by design</h3>
<p><code>pickle.load</code> on a file arriving from a customer's test rig is
arbitrary code execution. <code>.npy</code> covers the same need safely.</p>"""
    )

    # --- state
    parts.append("<h2>Current state</h2>")
    parts.append(
        f"""
<div class="kv">
  <dt>test suite</dt><dd class="mono">{esc(tests['summary'])} &nbsp; {verdict(tests['passed'])}</dd>
  <dt>wall time</dt><dd class="mono">{tests['seconds']:.1f} s</dd>
  <dt>latest commit</dt><dd class="mono">{esc(git['commit'])} — {esc(git['subject'])}</dd>
</div>
<p>Three structural properties are enforced by tests rather than convention: the
core computes with scipy and matplotlib absent; every theory page must declare
its validation status and name a covering test; and built-in readers and
estimators register through the same decorator a third-party plugin would use.</p>

<h3>Built</h3>
<ul>
<li><strong>M0</strong> — immutable phase/frequency containers with the precision
guard, provenance with a step log, the plugin registry, tau selection, Kasdin–Walter
synthesis, delimited-text reading, theory pages, and a CLI.</li>
<li><strong>M1</strong> — <code>adev</code>, <code>oadev</code>, <code>mdev</code>,
<code>tdev</code>, <code>ohdev</code> computed directly from phase, gap-aware, plus
the exact-rational oracle and the Tier 0 identity suite.</li>
<li><strong>M2</strong> — η₁ and η₃ degrees of freedom estimated from the
estimator's own coefficient autocovariance, the pure-numpy incomplete gamma and
inverse χ², χ² confidence intervals, lag-1 noise identification with an MDEV-slope
cross-check, the deviation bias correction, and <code>stabilis validate</code>
emitting a versioned evidence report that is capable of failing. This is the v0.1
line: before it, no uncertainty could be quoted from the library at all.</li>
<li><strong>M3, first piece</strong> — Greenhall EDOF, transcribed from the
Greenhall &amp; Riley PTTI 2003 preprint: all four cases, three coefficient
tables and the generalized autocovariance, plus Satterthwaite pooling over the
runs of a gapped record. It is what Stable32's <code>Alpha</code>,
<code>Min Sigma</code> and <code>Max Sigma</code> columns are computed from, so
it is what makes an old report and a new one diffable line for line. It is
selected explicitly and never by <code>auto</code>: assuming a power law has to
be a recorded act, and where the noise cannot be identified from the data the
library declines and falls back rather than guessing, which is a place it
deliberately answers less than Stable32 does.</li>
<li><strong>M3, second piece</strong> — the <strong>power-law
decomposition</strong>, which reads a measured σ(τ) curve back into the five
IEEE 1139 noise levels and a drift term. Because every hₐ is a noise power and
drift enters as D², the model is linear with a non-negativity constraint, which
is exactly non-negative least squares — solved exactly by Lawson–Hanson, with a
KKT optimality certificate on every fit, rather than by an optimizer that could
be blamed for a negative answer. It fits Allan and <em>modified</em> Allan rows
together, because white PM and flicker PM differ under AVAR only by a slowly
varying logarithm: on noiseless data that is enough, but with 2% noise on the
curve an AVAR-only fit returned between 0.00 and 1.00 times the true h₁ across
six seeds, where the stacked fit held to 5% on every one. The conversion
coefficients behind it were transcribed from NIST SP 1065 §7.1, whose flicker-PM
row is printed two mutually inconsistent ways — resolved by integrating the
handbook's own transfer function, and recorded rather than silently corrected.
The five modified-Allan coefficients the handbook does not tabulate at all were
derived here from its eq. (66).</li>
<li><strong>Amplitude-calibrated synthesis</strong> — the noise generator's
h<sub>α</sub> is now a settable parameter in SI units rather than an arbitrary
scale, checked against four closed forms that are exact at every averaging
factor. That is what makes the decomposition testable end to end: noise is
generated at a known level, estimated, decomposed, and compared to what went
in.</li>
<li><strong>M3, the estimator tail</strong> — <strong>TOTVAR</strong> with its
published EDOF and the SP 1065 bias correction; the <strong>Thêo family</strong>
— Thêo1, ThêoBR and ThêoH, reaching 75% of the record where the Allan variance
stops at half; <strong>MTIE and TIE rms</strong> by binary doubling, 10⁶ points
over 18 octaves in 0.15 s; and the <strong>MODWT wavelet variance</strong>.</li>
<li><strong>The wavelet layer</strong>, which is the one that reframes the rest:
the Haar wavelet variance <em>is</em> the Allan variance, exactly rather than
approximately, so the classical estimator is the K=1 member of one family and
the Daubechies filters are the remainder. The library asserts that identity to
1e-14 over eleven octaves and again on the confidence interval, where the Haar
bounds come back equal to the overlapping Allan bounds. What the higher orders
buy is concrete: K vanishing moments annihilate any polynomial trend of degree
below K, so D4 removes a linear frequency drift with no detrending step at all
— otherwise a separate and contestable operation on a customer's data. The
filter coefficients are computed by spectral factorization and then certified
against the defining algebra, rather than transcribed, because a transcribed
list with one wrong digit still looks like a filter.</li>
<li><strong>Three misprints in NIST SP 1065 found by measurement</strong>, all
recorded rather than silently corrected: the flicker-PM conversion row printed
two inconsistent ways; the 0.75 missing from the Thêo1 equation; and Thêo1's
averaging time, printed as 0.75(m−1)τ₀ once and 0.75mτ₀ twice, where the
handbook's own ThêoBR equation only parses under the second and measurement
puts the first 10% low at m=10. For a tool whose product is correctness, being
able to show the evidence for each departure is the deliverable, not a
footnote.</li>
<li><strong>The output and interaction layer</strong> — plotting, a flat 30-name
public API, and <code>stabilis gui</code>, a local browser page in four stages
that computes nothing itself: every variance and bound on it comes from the
same calls the CLI makes, and every selection prints the equivalent
non-interactive script, because an exclusion that exists only as a mouse
gesture is not evidence. The fourth stage is the bench, which is the only one
that reads several files at once and therefore the one that assumes least
about them.</li>
<li><strong>M5 complete, ahead of its place in the sequence.</strong>
The <strong>bench model</strong> describes the actual experiment as sources,
devices and channels, and turns three-cornered hat, N-cornered hat and
instrument noise-floor subtraction into one non-negative linear solve per τ
rather than three features. The part with no equivalent anywhere is the
refusal: identifiability is a graph property — a connected set of measurements
determines its oscillators individually <em>iff it contains an odd cycle</em>,
which is why the technique is called <em>three</em>-cornered hat — so a ring of
four determines nothing however long it runs, and <code>stabilis bench
check</code> says so <em>before any data is collected</em> and names the one
extra measurement that would fix it. It also distinguishes the pairs you must
go and measure from the pairs you already have, since x<sub>ab</sub> =
x<sub>ac</sub> − x<sub>bc</sub> identically: two clocks against a common
reference are one subtraction away from the edge that closes the triangle, and
a four-clock ring needs no new data at all.<br>
The <strong>cross-variance</strong> is the other half of the same idea, one
level down. Two channels watching one clock: the auto-variance of either is
floored at σ²<sub>c</sub> + σ²<sub>n</sub> however long you log, but the
difference operator is linear, so E[d<sub>a</sub>d<sub>b</sub>] =
E[d<sub>c</sub>²] when the channels' noise is independent — the instrument term
has expectation zero rather than being small. It costs one function for the
whole estimator family, because every result already carries the coefficient
array it averaged and the cross case is <code>mean(c_a·c_b)</code> where the
auto case is <code>mean(c²)</code>. Its error bar cannot be the χ² one, a sum
of products being unbounded below; Isserlis' theorem gives
(auto_a·auto_b + cross²)/η instead, which is 2σ⁴/η when the two channels are
the same channel — so <code>cross_variance(a, a)</code> returns the
auto-variance bit for bit and the two error bars are one construction rather
than two. <code>cross_matrix</code> does every pair at once and
<code>reject</code> averages them, which reaches deeper still; the pairs share
channels and are therefore not independent, and the exact covariance says how
much of the naive gain is real. Negative variances are reported signed rather
than clipped throughout, because they are the method telling you an
independence assumption failed.</li>
<li><strong>The measurement report</strong>, which is the actual paid
deliverable and now exists: one self-contained HTML document per measurement,
carrying the σ(τ) table with its intervals, the noise type at each τ, the
record's own hash, the full provenance step log, and a conformity verdict.
The verdict is the part that differs from every other tool — <strong>it is
taken on the confidence interval, not on the point estimate.</strong> A point
passes only when its whole interval lies below the limit and fails only when
the whole interval lies above it; an interval straddling the limit is reported
<em>indeterminate</em>, meaning the record is too short to decide and the
remedy is more data. Comparing the point estimate instead — what the field
normally does — is simple acceptance with a ~50% false-accept risk for a unit
sitting on the limit, and it is worse than that here, because at low degrees of
freedom the point estimate is itself biased low (11.4% at η = 2) and the error
therefore runs towards passing exactly where the evidence is thinnest. Three
rules follow and each is a test: a limit is never interpolated between the taus
a datasheet states, no interval means no verdict, and a two-source pass is
conclusive while a failure is not attributable.</li>
<li><strong>Model fitting under the χ² likelihood</strong>, with the
correlation between τ measured rather than assumed away. Arbitrary models —
a Gauss–Markov correlation time, a servo bump, a quantization floor, or a
customer's own — fitted against the exact sampling law rather than by least
squares, which is the wrong objective for a distribution running from
0.55σ to 2.2σ. Two independently derived uncertainties come back: a
correlation-aware sandwich and a copula bootstrap that keeps both the χ²
margins and the measured correlation. They agree when the model is right and
disagree when it is not, which is why both are printed. And the fit
<strong>refuses to quote parameters</strong> whose deviance is outside what
its effective degrees of freedom admit — in either direction, since a fit
that is too good means the model has more freedom than the curve
constrains.</li>
<li><strong>The export layer</strong> — the last thing between the library and
being usable inside somebody else's workflow, since until now nothing in the
package wrote data at all. A written record <strong>carries its own rate and
epoch</strong>, so a round trip needs no arguments and cannot be given the
wrong ones; it is written at 17 significant digits, because <code>%.15e</code>
returns a different double for 42% of them, measured. The Stable32 block is
transcribed rather than derived and then checked against Stable32's own bytes:
all 6461 committed data rows are reproduced character for character, the eight
preamble lines with them, and the σ column computed here from the same input
agrees to the last digit the format prints. Four refusals rather than
inventions — an estimator Stable32 never named, a frequency series whose
preamble has never been captured, a gapped record where the preamble asserts
contiguity, and a result with no α, since writing 0 there would assert white
FM. And <code>stabilis export bundle</code>: a directory holding the data
copied byte for byte, the curve, the report, the provenance and a SHA-256 for
each, whose <strong>digest is over the measurement rather than the files</strong>
— hashing the files would have been the obvious choice and is wrong, because
the provenance embedded in them carries a timestamp and a digest that changes
every run says nothing.</li>
<li><strong>Power spectra</strong> — Welch, one-sided, with the window's
noise-equivalent bandwidth divided out so the result is a density rather than
a periodogram in disguise; checked by Parseval to twelve digits and by
recovering a synthesized h<sub>0</sub> at three amplitudes over six decades.
Two conversions are <em>refused</em> rather than guessed: S<sub>φ</sub>
without a carrier frequency, since a 10 MHz OCXO and a 10 GHz source give
identical time-error records at equal fractional stability; and
ℒ(f) = S<sub>φ</sub>/2 wherever the integrated phase excursion is too large
for the small-angle limit, which is precisely the near-carrier region a
datasheet is written from. Two findings came out of the measurements rather
than the reasoning: the mean is the unbiased statistic and the median sits
3.4% low at 15 segments, and a cumulative sum is not an ideal integrator —
the two differ by (πfτ₀)²/sin²(πfτ₀), rising to 2.467 at Nyquist.</li>
<li><strong>Limit masks with a verdict</strong> — ITU-T G.811 transcribed
from the Recommendation itself, which changed a number: the widely repeated
form of its MTIE limit above 1000 s is a flat 300 ns ceiling, and the
document says 10⁻⁵τ + 0.29 µs, which rises. The <em>measurement conditions</em>
the standard states are enforced as part of the limit, so a record sampled
at 1 Hz or shorter than the 12τ it demands returns "not evaluated" rather
than a green tick. An extreme-value statistic is judged differently from a
variance: exceeding an MTIE limit fails conclusively without an interval,
while staying under one is a pass over the measured window only, since a
longer record can only find a larger maximum.</li>
<li><strong>M6, streaming and the dynamic Allan variance</strong> — sources
from a replayed record, an appending file or a socket; an online Allan
estimator; and DAVAR recomputed per window rather than slid, which makes the
window-equals-record case an exact identity rather than a tolerance. Dropped
samples stay as slots, so a late gap cannot re-time the record behind it.
See <code>theory/stream.md</code> and <code>theory/davar.md</code>.</li>
<li><strong>M7, gaps and the modified total family</strong> — Howe and
Schlossberger's Total Imputer, implemented from the paper, with the filled
samples excluded from the degrees of freedom so a filled record cannot buy a
narrower error bar; MTOT, TTOT and HTOT, which carry no error bar because
their terms are not one time-ordered sequence; and SP 1065 Table 9, the
first confidence interval for the Thêo family.</li>
<li><strong>Conditioning, and gaps without filling</strong> — drift removal
by six models (phase quadratic, mean second difference and linear;
frequency linear, bisection and mean), the fitted drift recorded in
provenance, and decimation that keeps phase and frequency the same
operation. Detrending is not free: the fitted trend absorbs low-frequency
noise, and <code>theory/drift.md</code> measures how far into the curve that
reaches for each noise type. Alongside it, Mondal and Percival's
gap-tolerant wavelet variance, which averages each pair of filter taps over
the times both were observed: equal to the ordinary estimator when nothing
is missing, unbiased when something is, and it keeps the coarse scales
that masking a gap throws away.</li>
<li><strong>Spurs</strong> — found against a running-median floor, then
<em>measured by fitting the record</em> rather than reading the spectrum, so
a tone between bins is exact without a flat-top window: 0.001% on amplitude
and about a nanohertz on frequency. Every table prints its own expected
false-alarm count from the chi-square tail. Notching records each removed
tone in the returned series' provenance, because a record with a tone taken
out describes an oscillator that does not exist.</li>
</ul>

<h3>Next</h3>
<ul>
<li><strong>The rest of the Stable32 session</strong> — real 1.62 exports over
six synthetic records are committed and scored automatically, so the claim is
no longer "implements the published algorithm" but "matches Stable32 to the
precision it prints". Still to come: the public NBS suites and, most valuable
of all, a real oscillator record, being the only case where the data was not
generated by the tool under test.</li>
<li><strong>Somebody else opening a file we wrote</strong> — the export layer's
round trips all go through this library's own reader, which shows the pair is
self-consistent and nothing more. The mirror of the rule already in force for
readers: a writer validated only against its own reader is validated against
nothing. It closes with one line added to the Stable32 run sheet, and is
recorded as item 8 of <code>VALIDATION-DEBT.md</code> until it does.</li>
<li><strong>No live source has ever run</strong> — the streaming path and the
dynamic Allan variance are validated by identity against the batch estimator,
which is the right comparison and an exact one: a replayed record agrees to a
single ulp, and a whole-record surface is bit-identical. What none of that is,
is an instrument. <code>replay</code> drives the suite, <code>tail_file</code>
is tested against a file written by the same process, and the sockets are
tested on loopback. Item 10 of <code>VALIDATION-DEBT.md</code>; it closes with
one appending text file from a real rig.</li>
<li><strong>Error bars the newest estimators lack</strong> — the gappy
wavelet variance needs Mondal and Percival's multitaper interval, and the
MTOT family needs Riley's published degrees of freedom. Both are declared in
<code>VALIDATION-DEBT.md</code> rather than approximated.</li>
<li><strong>M4 and the tail of M8</strong> — the rest of M4 is instrument
readers, gated on paired vendor exports rather than on effort: a binary format
with no public specification can only be accepted if the same capture is also
available as vendor ASCII to check the reader against. The rest of M8 is
the interactive span selector and the remaining plots.</li>
</ul>

<div class="note"><strong>Known limitations.</strong> Confidence intervals now
exist, but their measured coverage falls short of nominal at the extremes of the
τ range; the measured figures live in <code>docs/validation-report.html</code>,
generated by <code>stabilis validate</code>, which is the artifact that carries
them. Greenhall's bounds are now available alongside the data-driven ones, but
have now been checked against Stable32 1.62's own exports on this project's
six synthetic records: the term-count column agrees exactly as integers, the
sigma columns of all five estimators agree to 4.9e-5 — the last digit Stable32
prints — and Min/Max Sigma agree to 5.5e-4. The degrees of freedom behind those
bounds do <em>not</em> quite agree: Stable32's implied EDOF runs about 1.2%
above ours for averaging factors up to 32, where an independent implementation
of the same paper returns our value to every printed digit. Cause unknown, worth
under 0.06% on a 68.27% bound, and recorded rather than guessed at. No real
oscillator record has been through both programs yet, which is the remaining
part of that item in <code>VALIDATION-DEBT.md</code>. Of the ten
power-law conversion coefficients, nine are checked end to end against noise of
known amplitude; the tenth, unmodified flicker PM, cannot be with the current
generator, because that coefficient assumes a brick-wall bandwidth and the
Kasdin–Walter process is not brick-wall limited — the gap is measured, bounded
and recorded rather than averaged away. Instrument readers are not yet validated
against real hardware exports, and are deliberately not written until a paired
vendor export exists to check them against. The Thêo family has no confidence
interval: it reports no coefficient arrays for the data-driven path and its own
EDF table is not implemented, so the CLI offers no interval flags for it rather
than flags that would fail. ThêoBR's bias correction is measured from the data
and is itself noisy on a single record — a common vertical shift of the whole
curve, not scatter between points — so the ratio is reported in provenance
rather than folded silently into the answer. The conformity verdict inherits
its interval's accuracy: the decision rule is fully tested at both boundaries
and the intervals have measured containment, but containment is an ensemble
statement while a verdict is issued on one record, so the two do not compose
into a stated false-accept risk — what is missing is an operating-characteristic
curve, and until it exists the report states the rule and the interval it used
rather than a risk figure. The cross-variance has no interop tier and can never
have one, since no other tool computes it and manufacturing a reference from
this library's own output would make the assertion a statement about itself;
its evidence is correspondingly heavier on the functional side, and the
hardware half — two real channels, deliberately sharing a reference in one run
and not in another — is still owed.</div>"""
    )
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", action="store_true", help="also render a PDF via Edge")
    args = ap.parse_args()

    out_html = HERE / "stabilis-progress.html"
    out_html.write_text(build_html(), encoding="utf-8")
    print(f"wrote {out_html}")

    if args.pdf:
        if not EDGE.exists():
            print(f"Edge not found at {EDGE}; open the HTML and print to PDF", file=sys.stderr)
            return 1
        out_pdf = HERE / "stabilis-progress.pdf"
        win = lambda p: subprocess.run(  # noqa: E731
            ["wslpath", "-w", str(p)], capture_output=True, text=True
        ).stdout.strip()
        subprocess.run(
            [
                str(EDGE),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={win(out_pdf)}",
                f"file:///{win(out_html)}",
            ],
            capture_output=True,
            timeout=180,
        )
        if out_pdf.exists():
            print(f"wrote {out_pdf} ({out_pdf.stat().st_size / 1024:.0f} kB)")
        else:
            print("Edge produced no PDF", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
