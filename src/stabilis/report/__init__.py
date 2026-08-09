"""The measurement report -- the document a laboratory hands to a customer.

Distinct from `stabilis.validate`, and the distinction is the whole point:
`validate` is evidence that *the software* computes what it claims, and is
produced once per release; this is evidence about *one measurement*, and is
produced once per unit. A customer needs both, and conflating them is how a
tool ends up asserting that a good oscillator proves a correct library.

    from stabilis import load_phase, oadev, confidence
    from stabilis.report import build, to_html, figure_uri, spec_afs

    series = load_phase("dut.dat", rate_hz=1.0)
    spec = ((1.0, 2e-12), (10.0, 8e-13), (100.0, 3e-13))
    result = oadev(series, taus=spec_afs(spec, series.tau0_s))
    ci = confidence(result)
    report = build(series, result, ci, subject="OCXO 42 vs H-maser",
                   spec=spec, spec_label="datasheet")
    open("report.html", "w").write(to_html(report, figure_uri(result, ci, spec)))

`build` computes nothing the caller could have passed in, so the report is
never a second place where estimator options get chosen. It is kept off
`stabilis.__all__` for the same reason `plot` is: the figure path wants
matplotlib, and the top-level import stays numpy-only.
"""

from .build import (
    SPEC_TAU_RTOL,
    VERDICTS,
    MeasurementReport,
    Row,
    SpecCheck,
    build,
    spec_afs,
)
from .figure import figure_uri
from .render import to_html, to_json

__all__ = [
    "SPEC_TAU_RTOL",
    "VERDICTS",
    "MeasurementReport",
    "Row",
    "SpecCheck",
    "build",
    "spec_afs",
    "figure_uri",
    "to_html",
    "to_json",
]
