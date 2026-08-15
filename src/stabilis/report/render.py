"""Rendering a `MeasurementReport` to HTML and to JSON.

Pure formatting: nothing here decides anything, so a disagreement about a
verdict is always a question about `build.py`. Kept numpy-only and
self-contained -- one file with its style inlined, no external assets --
because the document has to survive being emailed to an assessor and
opened three years later.

The prose sections are part of the deliverable rather than decoration.
A report that prints a conformity verdict without stating the decision
rule that produced it is not usable evidence, and one that does not say
what the measurement cannot distinguish invites the reader to assume it
distinguishes everything.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from ..provenance import to_dict as prov_to_dict

__all__ = ["to_html", "to_json"]

_CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:62rem;
margin:2rem auto;padding:0 1rem;color:#182026}
h1{margin-bottom:.2rem}h2{margin-top:2rem;border-bottom:1px solid #dde;padding-bottom:.2rem}
.sub{color:#5c7080;margin-top:0}
table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:14px}
table.narrow{width:auto;min-width:22rem}
th,td{border:1px solid #dde;padding:.4rem .55rem;text-align:left;vertical-align:top}
th{background:#f5f7fa}
td.num{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.pass{color:#0a6640;font-weight:600}.fail{color:#a82a2a;font-weight:600}
.indeterminate{color:#a66321;font-weight:600}.skip{color:#5c7080}
.not_evaluated{color:#5c7080;font-weight:600}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.verdict{padding:.7rem 1rem;border-radius:4px;font-weight:600;margin:1rem 0}
.v-pass{background:#e8f5ee;color:#0a6640}
.v-fail{background:#fdeaea;color:#a82a2a}
.v-indeterminate{background:#fdf3e7;color:#a66321}
.v-none{background:#f5f7fa;color:#394b59}
.note{background:#f5f7fa;border-left:3px solid #8a9ba8;padding:.6rem 1rem;margin:1rem 0}
figure{margin:1rem 0}figure img{max-width:100%}
"""

_VERDICT_TEXT = {
    "pass": "PASS",
    "fail": "FAIL",
    "indeterminate": "INDETERMINATE",
    "none": "NO SPECIFICATION EVALUATED",
}


def _esc(v) -> str:
    return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _g(v, fmt: str = "{:.4g}") -> str:
    """A number, or an em dash where there is not one. Never a bare 'nan'."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    return "&mdash;" if not np.isfinite(x) else fmt.format(x)


def _pct(x: float) -> str:
    """A guarded margin as a percentage above or below the limit."""
    if not np.isfinite(x):
        return "&mdash;"
    return f"{(x - 1.0) * 100:+.1f} %"


def _banner(report) -> str:
    v = report.verdict
    text = _VERDICT_TEXT[v]
    if report.worst is None and report.masks:
        # A mask-only report still has a verdict; its closest approach lives
        # in the mask's own section, since a Recommendation and a datasheet
        # claim are not interchangeable evidence.
        judged = [m for m in report.masks if m.verdict != "not_evaluated"]
        detail = (f"From {len(judged)} published limit(s); see the "
                  "conformity section below for the margin at each."
                  if judged else
                  "No published limit could be evaluated under the "
                  "measurement conditions it states.")
    elif report.worst is None:
        detail = ("No specification was supplied, or none could be evaluated "
                  "at a tau this measurement visited.")
    else:
        w = report.worst
        detail = (
            f"Worst margin {_pct(w.margin)} at &tau; = {_g(w.tau_used_s)} s: "
            f"limit {_g(w.limit, '{:.3e}')}, measured {_g(w.dev, '{:.3e}')}, "
            f"upper bound {_g(w.dev_hi, '{:.3e}')}."
        )
    return f"<div class='verdict v-{v}'>{text} &mdash; {detail}</div>"


def _record_table(report) -> str:
    r = report.record
    frac = r.get("gap_fraction", 0.0)
    rows = [
        ("Source", _esc(r["source"])),
        ("SHA-256", f"<span class='mono'>{_esc(r['source_sha256'])}</span>"),
        ("Quantity", _esc(r["kind"])),
        ("Sample rate", f"{_g(r['rate_hz'])} Hz (&tau;<sub>0</sub> = "
                        f"{_g(r['tau0_s'])} s)"),
        ("Samples", f"{r['n']}"),
        ("Duration", f"{_g(r['duration_s'])} s"),
        ("Gaps", f"{r['n_gaps']} ({frac * 100:.2f} % of the record), "
                 f"{r['runs']} contiguous run(s), longest {r['longest_run']}"),
        ("Estimator", _esc(report.estimator)),
        ("Confidence level", _g(report.level, "{:.4f}")),
    ]
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _mask_tables(report) -> str:
    """One section per published limit, each naming the document it came from.

    Kept separate from the specification table on purpose. A manufacturer's
    datasheet claim and a published Recommendation are both limits, and
    they are not interchangeable evidence -- merging them into one table
    would let a reader lose which is which.
    """
    if not report.masks:
        return ""
    parts = ["<h2>Conformity against published limits</h2>"]
    for m in report.masks:
        parts.append(
            f"<h3>{_esc(m.mask)} &mdash; <span class='{m.verdict}'>"
            f"{m.verdict.upper().replace('_', ' ')}</span></h3>")
        parts.append("<table><tr><th>&tau; (s)</th><th>Measured</th>"
                     "<th>Limit</th><th>Margin</th><th>Verdict</th>"
                     "<th>Note</th></tr>")
        for c in m.checks:
            limit = "&mdash;" if c.limit is None else _g(c.limit, "{:.3e}")
            margin = "&mdash;" if c.margin is None else f"{c.margin:.2f}&times;"
            parts.append(
                f"<tr><td class='num'>{_g(c.tau_s)}</td>"
                f"<td class='num'>{_g(c.measured, '{:.3e}')}</td>"
                f"<td class='num'>{limit}</td>"
                f"<td class='num'>{margin}</td>"
                f"<td class='{c.verdict}'>"
                f"{c.verdict.upper().replace('_', ' ')}</td>"
                f"<td class='sub'>{_esc(c.reason)}</td></tr>")
        parts.append("</table>")
        worst = m.worst
        if worst is not None:
            parts.append(
                f"<p class='sub'>Closest approach: {worst.margin:.2f}&times; "
                f"at &tau; = {_g(worst.tau_s)} s.</p>")
        else:
            parts.append("<p class='sub'>No &tau; could be compared with "
                         "this limit.</p>")
    parts.append(
        "<div class='note'><strong>What a published limit adds.</strong> "
        "These are transcribed from the standards named above, with the "
        "measurement conditions each states enforced as part of the limit: "
        "a point measured outside them is reported <em>not evaluated</em> "
        "rather than passed. A limit is never extrapolated beyond the "
        "&tau; range its document states.</div>")
    return "\n".join(parts)


def _spec_table(report) -> str:
    if not report.specs:
        return ""
    head = ("<table><tr><th>&tau; specified</th><th>&tau; used</th>"
            "<th>Limit</th><th>Measured</th><th>Interval</th>"
            "<th>Guarded margin</th><th>On the point estimate</th>"
            "<th>Verdict</th></tr>")
    parts = [head]
    for s in report.specs:
        if not s.evaluated:
            parts.append(
                f"<tr><td class='num'>{_g(s.tau_spec_s)}</td>"
                f"<td colspan='6' class='skip'>{_esc(s.reason)}</td>"
                f"<td class='skip'>not evaluated</td></tr>"
            )
            continue
        parts.append(
            f"<tr><td class='num'>{_g(s.tau_spec_s)}</td>"
            f"<td class='num'>{_g(s.tau_used_s)}</td>"
            f"<td class='num'>{_g(s.limit, '{:.3e}')}</td>"
            f"<td class='num'>{_g(s.dev, '{:.3e}')}</td>"
            f"<td class='num'>{_g(s.dev_lo, '{:.3e}')} .. "
            f"{_g(s.dev_hi, '{:.3e}')}</td>"
            f"<td class='num'>{_pct(s.margin)}</td>"
            f"<td class='num'>{_pct(s.margin_point)}</td>"
            f"<td class='{s.verdict}'>{s.verdict.upper()}</td></tr>"
        )
    parts.append("</table>")
    parts.append(
        "<div class='note'><strong>Decision rule.</strong> A point passes "
        "only when its whole confidence interval lies below the limit, and "
        "fails only when the whole interval lies above it. An interval "
        "straddling the limit is reported <em>indeterminate</em>: the "
        "record is too short to resolve conformity at this confidence, and "
        "the remedy is more data rather than a rounder answer. The "
        "<em>guarded margin</em> is the limit over the upper bound; the "
        "final column repeats it against the point estimate, which is what "
        "tools that ignore uncertainty report, and carries a 50 % "
        "false-accept risk for a unit sitting on the limit.</div>"
    )
    return "".join(parts)


def _rows_table(report) -> str:
    head = ("<table><tr><th>AF</th><th>&tau; (s)</th><th>Terms</th>"
            "<th>&sigma;(&tau;)</th><th>Lower</th><th>Upper</th>"
            "<th>EDOF</th><th>Method</th><th>Noise</th>"
            "<th>E[&sigma;&#770;]/&sigma;</th></tr>")
    parts = [head]
    for r in report.rows:
        noise = _esc(r.alpha_name) if r.alpha_name else "&mdash;"
        if r.alpha_confidence:
            noise += f" <span class='skip'>({_esc(r.alpha_confidence)})</span>"
        method = _esc(r.method)
        if r.declined:
            method += f" <span class='skip'>&mdash; {_esc(r.declined)}</span>"
        parts.append(
            f"<tr><td class='num'>{r.af}</td>"
            f"<td class='num'>{_g(r.tau_s)}</td>"
            f"<td class='num'>{r.n_terms}</td>"
            f"<td class='num'>{_g(r.dev, '{:.4e}')}</td>"
            f"<td class='num'>{_g(r.dev_lo, '{:.4e}')}</td>"
            f"<td class='num'>{_g(r.dev_hi, '{:.4e}')}</td>"
            f"<td class='num'>{_g(r.edof, '{:.1f}')}</td>"
            f"<td>{method}</td><td>{noise}</td>"
            f"<td class='num'>{_g(r.bias, '{:.3f}')}</td></tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _fit_table(report) -> str:
    fit = report.fit
    if fit is None:
        return ""
    rows = "".join(
        f"<tr><td class='num'>{a}</td><td class='num'>{_g(h, '{:.4e}')}</td></tr>"
        for a, h in sorted(fit.h.items(), reverse=True)
    )
    dom = "none resolved" if fit.dominant is None else f"&alpha; = {fit.dominant}"
    return (
        "<h2>Noise decomposition</h2>"
        "<p class='sub'>Non-negative least squares over the measured curve. "
        "Coefficients are of the one-sided S<sub>y</sub>(f) = h<sub>&alpha;</sub> "
        "f<sup>&alpha;</sup> model.</p>"
        "<table class='narrow'><tr><th>&alpha;</th>"
        f"<th>h<sub>&alpha;</sub></th></tr>{rows}"
        f"<tr><th>drift D</th><td class='num'>{_g(fit.drift, '{:.4e}')}</td></tr>"
        "</table>"
        f"<p>Dominant at the shortest fitted &tau;: <strong>{dom}</strong>. "
        f"Rows fitted {fit.n_rows}, dropped {fit.dropped_rows}; "
        f"f<sub>h</sub> = {_g(fit.f_h)} Hz"
        f"{' (assumed, not measured)' if fit.assumed_fh else ''}; "
        f"relative residual {_g(fit.residual, '{:.3e}')}.</p>"
    )


def _uncertainty_prose(report) -> str:
    methods = sorted({r.method for r in report.rows})
    return (
        "<h2>Basis of the stated uncertainty</h2>"
        "<p>Each interval is a chi-square interval on the variance, "
        "transformed to the deviation by taking square roots &mdash; a "
        "monotone map, so the stated coverage is preserved exactly. The "
        "interval is asymmetric because the sampling law of a variance is, "
        "and reporting it as a symmetric &plusmn; would understate the "
        "upper bound at exactly the long averaging times where the degrees "
        "of freedom are fewest.</p>"
        f"<p>Effective degrees of freedom per point came from: "
        f"<span class='mono'>{_esc(', '.join(methods))}</span>. Where a "
        "method declined, the reason is printed in the table row rather "
        "than being silently replaced by a fallback.</p>"
        "<p>The last column of the stability table is "
        "E[&sigma;&#770;]/&sigma; at that row's degrees of freedom. The "
        "square root of an unbiased variance is a biased deviation: at two "
        "degrees of freedom the reported point is 11.4 % low on average. "
        "The values here are <em>not</em> corrected for it, because every "
        "other tool in the field reports them uncorrected and a silent "
        "correction would look like a defect; the factor is printed so the "
        "reader can apply it, and the conformity decision above uses the "
        "upper bound, which is unaffected by the argument.</p>"
    )


def _limitations(report) -> str:
    return (
        "<h2>What this measurement does not establish</h2>"
        "<ul>"
        "<li><strong>A two-source comparison bounds the device under test "
        "from above.</strong> The measured variance contains both sources "
        "and the measurement channel's own floor, so a pass is conclusive "
        "&mdash; the unit is at least as good as reported &mdash; while a "
        "failure may belong to the reference or to the channel. Separating "
        "them needs three or more sources; that is what the bench solve "
        "does.</li>"
        "<li><strong>Noise type is identified per &tau;, and identification "
        "can be wrong.</strong> Where it drives the degrees of freedom, a "
        "misidentification moves the error bar without moving the point "
        "estimate. Rows carrying a low-confidence identification are "
        "labelled as such.</li>"
        "<li><strong>Gaps are excluded, never filled.</strong> Terms "
        "touching a gap are dropped and the surviving count is what reaches "
        "the degrees of freedom, so a gappy record widens its own error "
        "bars rather than quietly keeping them.</li>"
        "<li><strong>This document is a measurement report, not a "
        "calibration certificate.</strong> It states no traceability chain "
        "and makes no accreditation claim. The software's own validation "
        "evidence is a separate document, produced by "
        "<span class='mono'>stabilis validate</span>.</li>"
        "</ul>"
    )


def _stats_table(report) -> str:
    s = report.stats
    order = ("n", "n_finite", "n_gaps", "min", "max", "mean", "median", "std")
    body = "".join(
        f"<tr><th>{k}</th><td class='num'>{_g(s[k], '{:.6g}')}</td></tr>"
        for k in order if k in s
    )
    return (
        "<h2>Descriptive statistics</h2>"
        "<p class='sub'>Of the record as loaded. The sample standard "
        "deviation does not converge for flicker or random-walk noise and "
        "must not be read as an uncertainty on a clock &mdash; it is here "
        "because it is a sanity check on the file.</p>"
        f"<table>{body}</table>"
    )


def _provenance_table(report) -> str:
    steps = report.provenance.steps
    rows = "".join(
        f"<tr><td class='mono'>{_esc(s.t_utc)}</td><td>{_esc(s.module)}</td>"
        f"<td>{_esc(s.action)}</td>"
        f"<td class='mono'>{_esc(json.dumps(s.params, default=str))}</td>"
        f"<td>{_esc(s.outcome)}</td></tr>"
        for s in steps
    )
    return (
        "<h2>Provenance</h2>"
        "<p class='sub'>Every recorded decision, including the ones taken "
        "by default on the operator's behalf. IEEE Std 1139 requires the "
        "pretreatment of the data to be reported; this is that record.</p>"
        "<table><tr><th>UTC</th><th>Module</th><th>Action</th>"
        f"<th>Parameters</th><th>Outcome</th></tr>{rows}</table>"
    )


def to_html(report, figure_uri: str = "") -> str:
    """The whole document, as one self-contained HTML string.

    `figure_uri` is an optional `data:` URI for the sigma-tau plot; see
    `report.figure`. Passed in rather than drawn here so that rendering
    never needs matplotlib.
    """
    head = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{_esc(report.title)}</title>",
        f"<style>{_CSS}</style>",
        f"<h1>{_esc(report.title)}</h1>",
        f"<p class='sub'>{_esc(report.subject) or 'unnamed measurement'}"
        f"{' &middot; ' + _esc(report.operator) if report.operator else ''}"
        f" &middot; generated {_esc(report.generated)}"
        f" &middot; digest <span class='mono'>{report.digest[:16]}…</span></p>",
        _banner(report),
    ]
    if report.notes:
        head.append(f"<div class='note'>{_esc(report.notes)}</div>")
    if figure_uri:
        head.append(
            f"<figure><img src='{figure_uri}' alt='sigma-tau curve'>"
            "<figcaption class='sub'>Error bars are the interval tabulated "
            "below, not a symmetric standard error.</figcaption></figure>"
        )
    spec = _spec_table(report)
    if spec:
        label = _esc(report.spec_label) or "supplied specification"
        spec = f"<h2>Conformity against {label}</h2>{spec}"
    return "\n".join(head + [
        "<h2>The measurement</h2>",
        _record_table(report),
        spec,
        _mask_tables(report),
        "<h2>Stability</h2>",
        _rows_table(report),
        _fit_table(report),
        _uncertainty_prose(report),
        _limitations(report),
        _stats_table(report),
        _provenance_table(report),
        "<h2>Environment</h2>",
        "<table>" + "".join(
            f"<tr><th>{_esc(k)}</th><td class='mono'>{_esc(v)}</td></tr>"
            for k, v in {**report.environment, "digest": report.digest}.items()
        ) + "</table>",
    ])


def to_json(report, indent: int = 2) -> str:
    """The same content as data. The digest is over this, not over the HTML."""
    return json.dumps(
        {
            "title": report.title,
            "subject": report.subject,
            "operator": report.operator,
            "notes": report.notes,
            "generated": report.generated,
            "digest": report.digest,
            "estimator": report.estimator,
            "level": report.level,
            "verdict": report.verdict,
            "worst": None if report.worst is None else asdict(report.worst),
            "record": report.record,
            "stats": report.stats,
            "rows": [asdict(r) for r in report.rows],
            "specs": [asdict(s) for s in report.specs],
            "masks": [{"mask": m.mask, "quantity": m.quantity,
                       "verdict": m.verdict,
                       "checks": [asdict(c) for c in m.checks],
                       "worst": None if m.worst is None else asdict(m.worst)}
                      for m in report.masks],
            "environment": report.environment,
            "provenance": prov_to_dict(report.provenance),
        },
        indent=indent,
        default=str,
    )
