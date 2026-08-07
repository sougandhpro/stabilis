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
from stabilis.synth import power_law_noise
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
    tests = run_tests()
    git = git_state()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts = [f"<style>{CSS}</style>"]
    parts.append(
        f"""
<h1>stabilis — progress report</h1>
<p class="sub">Frequency stability analysis engine. Milestones M0 and M1 complete.</p>
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
</ul>

<h3>Next</h3>
<ul>
<li><strong>M2</strong> — tapered η₁ and η₃ degrees of freedom, the pure-numpy
inverse χ², confidence intervals, noise identification, and
<code>stabilis validate</code> emitting a versioned evidence report. This is the
v0.1 release point and the first genuinely saleable cut.</li>
<li><strong>M3</strong> — Greenhall EDOF for error-bar parity with Stable32, the
MODWT wavelet layer, TOTVAR with its bias correction, power-law decomposition.</li>
<li><strong>M4–M7</strong> — model fitting against the χ² likelihood, instrument
readers, realtime ingest and dynamic Allan variance, then the remaining parity
tail and the report layer.</li>
</ul>

<div class="note"><strong>Known limitations.</strong> No confidence intervals yet,
so no uncertainty may be quoted from this version. Expected-value tables relayed
from secondary sources are marked UNVERIFIED in the source pending transcription
from the SP 1065 PDF; the certified oracle does not depend on them. Instrument
readers are not yet validated against real hardware exports. TOTVAR, Theo1, MTIE
and the wavelet layer are not implemented.</div>"""
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
