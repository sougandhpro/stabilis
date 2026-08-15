"""Getting results back out: a stability curve as CSV, JSON or Stable32 text.

The data half of the job is `stabilis.io` -- `save_phase` and
`save_frequency` write a record. This package writes what the analysis
*produced*, which is a different thing with different consumers, and it
sits apart for the same reason `validate/stable32.py` sits outside `io/`:
one moves data, the other moves results.

    from stabilis import load_phase, oadev, confidence
    from stabilis.export import to_csv, to_json, to_stable32

    series = load_phase("dut.dat", rate_hz=1.0)
    result = oadev(series, taus="octave")
    ci = confidence(result, edof_method="greenhall")

    open("curve.csv", "w").write(to_csv(result, ci))
    open("curve.json", "w").write(to_json(result, ci))
    open("curve.txt", "w", newline="").write(to_stable32(series, result, ci))

Nothing here computes. Every number written was decided by whoever chose
the estimator and the interval -- the rule `report.build` follows, applied
to the export layer, so a curve can never be exported with options the
caller did not pick.

Kept off `stabilis.__all__` because it is an output layer, alongside
`plot`, `bench` and `report`.
"""

from .stable32 import ABSENT, HEADERS, NEWLINE, ROW, SIGMA_LABELS, to_stable32
from .table import COLUMNS, to_csv, to_dict, to_json

# What the command line offers. The order is the order of increasing
# fidelity: csv drops the per-tau method, stable32 drops eleven digits.
TABLE_FORMATS = ("csv", "json", "stable32")

__all__ = [
    "TABLE_FORMATS",
    "COLUMNS",
    "to_dict",
    "to_json",
    "to_csv",
    "SIGMA_LABELS",
    "ROW",
    "HEADERS",
    "ABSENT",
    "NEWLINE",
    "to_stable32",
]
