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

This module assembles and renders. The checks live one module per tier:
`identity.py` (identity and certified), `coverage.py` and `roundtrip.py`
(coverage), `interop.py`; `check.py` holds the records they return.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from .. import __version__
from .check import (MIN_COVERAGE_TRIALS, SEEDS, TIERS, Check,  # noqa: F401
                    Report)
from .coverage import _coverage
from .identity import _certified, _identities
from .interop import (STABLE32_OUTPUTS, STABLE32_PRINT_RTOL,  # noqa: F401
                      _interop)

__all__ = ["Check", "Report", "run", "to_json", "to_html"]


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
    """The report as JSON -- the machine-readable evidence artefact."""
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
    """The report as a standalone HTML page, styles included."""
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

