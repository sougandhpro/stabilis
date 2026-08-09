"""Assembling a measurement report from a result and its uncertainty.

This is the deliverable a calibration laboratory actually pays for, so the
two things it must not do are state a conformity verdict the measurement
cannot support, and hide the assumptions that produced the error bar.

**The conformity decision is made on the interval, not on the point.**
Comparing a point estimate to a limit is "simple acceptance" in ILAC-G8
terms and carries a ~50 % false-accept risk for a unit sitting on the
limit. Here a spec point is *passed* only when the whole interval lies
below the limit, *failed* only when the whole interval lies above it, and
otherwise reported `indeterminate` -- the measurement was too short to
resolve conformity, which is a fact about the measurement and belongs in
the report rather than being rounded away to a verdict.

That distinction has teeth here in a way it does not for a voltmeter,
because the chi-square interval on a variance is wildly asymmetric at the
long-tau end where the customer's specification usually lives. At two
degrees of freedom the point estimate is itself 11.4 % low on average
(`bias.deviation_bias_factor`), so a naive comparison is biased *towards*
passing exactly where the evidence is thinnest. `Row.bias` carries that
factor per tau so the reader can see it.

**A two-source measurement bounds the device under test from above.** The
measured variance is the sum of the two sources' variances plus the
channel floor, so a pass is conclusive -- the unit is at least as good as
measured -- while a failure may belong to the reference. The report says
so; separating them is what `bench.solve` is for.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass

import numpy as np

from .. import __version__
from ..edit import summary as edit_summary
from ..preprocess import basic_stats
from ..provenance import Provenance
from ..provenance import utc_now
from ..stats import identify_curve
from ..stats.bias import deviation_bias_factor

__all__ = [
    "SPEC_TAU_RTOL",
    "VERDICTS",
    "Row",
    "SpecCheck",
    "MeasurementReport",
    "spec_afs",
    "build",
]

# How far a measured tau may sit from the tau a specification is stated at
# before the comparison is refused rather than made. A datasheet limit at
# 1 s is not a limit at 1.28 s, and an octave grid will happily offer the
# second. `spec_afs` exists so a caller can avoid the situation entirely.
SPEC_TAU_RTOL = 0.02

# Ordered worst-first: the overall verdict is the worst of the individual
# ones, and "indeterminate" is worse than "pass" because it means the
# question was not answered.
VERDICTS = ("fail", "indeterminate", "pass")


def spec_afs(spec, tau0_s: float) -> np.ndarray:
    """Averaging factors that land a measured point on each spec tau.

        m = round(tau_spec / tau0)

    Union this with the analysis grid before estimating and every
    specification point gets evaluated at its own tau rather than at
    whichever grid point happened to fall nearby. Averaging factors are
    integers, so a spec tau that is not a multiple of tau0 still lands
    slightly off; `build` reports the residual mismatch.
    """
    afs = {max(1, int(round(float(tau) / tau0_s))) for tau, _ in spec}
    return np.array(sorted(afs), dtype=np.int64)


@dataclass(frozen=True)
class Row:
    """One averaging time of the reported sigma-tau table."""

    af: int
    tau_s: float
    n_terms: int
    dev: float
    dev_lo: float
    dev_hi: float
    edof: float
    method: str
    # Why the requested EDOF method was not the one used, empty where it
    # was. A reader asking "why is this row different" gets an answer.
    declined: str
    alpha: int | None
    alpha_name: str
    alpha_confidence: str
    # E[sigma_hat]/sigma at this row's degrees of freedom. Below ~0.99 the
    # point estimate is visibly low and the reader should be reading the
    # upper bound instead.
    bias: float


@dataclass(frozen=True)
class SpecCheck:
    """One specification point, and whether the measurement clears it."""

    tau_spec_s: float
    limit: float
    evaluated: bool
    reason: str
    tau_used_s: float
    dev: float
    dev_lo: float
    dev_hi: float
    verdict: str
    # Guarded margin: the limit over the *upper* bound. Above 1 the unit
    # conforms at the stated confidence.
    margin: float
    # The same ratio taken against the point estimate, which is what every
    # other tool reports. Shown alongside so the gap between the two is
    # visible rather than argued about.
    margin_point: float


@dataclass(frozen=True)
class MeasurementReport:
    """Everything the rendered document states, and nothing it computes."""

    title: str
    subject: str
    operator: str
    notes: str
    estimator: str
    level: float
    rows: tuple[Row, ...]
    specs: tuple[SpecCheck, ...]
    spec_label: str
    verdict: str
    worst: SpecCheck | None
    record: dict
    stats: dict
    fit: object | None
    provenance: Provenance
    environment: dict
    digest: str
    generated: str

    @property
    def has_intervals(self) -> bool:
        return bool(self.rows) and np.isfinite(self.rows[0].edof)


def _verdict(dev_lo: float, dev_hi: float, limit: float) -> str:
    """Conformity of one interval against one limit.

    Guarded acceptance both ways. The middle case is not a failure and not
    a pass; it is the interval straddling the limit, and the remedy is a
    longer record rather than a rounder number.
    """
    if dev_hi <= limit:
        return "pass"
    if dev_lo > limit:
        return "fail"
    return "indeterminate"


def _noise_rows(series, afs, identify_noise: bool):
    """Noise type per averaging factor, or a row of blanks if not asked."""
    if not identify_noise:
        return [None] * len(afs)
    return identify_curve(series, afs)


def _rows(result, ci, noise) -> tuple[Row, ...]:
    """One `Row` per tau, folding the interval and the noise ID together."""
    dev = np.asarray(result.dev, dtype=np.float64)
    n = dev.size
    if ci is None:
        lo = hi = np.full(n, np.nan)
        edof = np.full(n, np.nan)
        method = ("no interval",) * n
        declined = ("",) * n
        bias = np.full(n, np.nan)
    else:
        lo, hi = np.asarray(ci.dev_lo), np.asarray(ci.dev_hi)
        edof = np.asarray(ci.edof, dtype=np.float64)
        method = tuple(ci.method)
        declined = tuple(ci.declined) if ci.declined else ("",) * n
        good = np.isfinite(edof) & (edof > 0.0)
        bias = np.full(n, np.nan)
        bias[good] = deviation_bias_factor(edof[good])
    return tuple(
        Row(
            af=int(result.af[i]),
            tau_s=float(result.tau_s[i]),
            n_terms=int(result.n_terms[i]),
            dev=float(dev[i]),
            dev_lo=float(lo[i]),
            dev_hi=float(hi[i]),
            edof=float(edof[i]),
            method=str(method[i]),
            declined=str(declined[i]),
            alpha=None if noise[i] is None else int(noise[i].alpha),
            alpha_name="" if noise[i] is None else noise[i].name,
            alpha_confidence="" if noise[i] is None else noise[i].confidence,
            bias=float(bias[i]),
        )
        for i in range(n)
    )


def _check_one(tau_spec: float, limit: float, rows: tuple[Row, ...]) -> SpecCheck:
    """Evaluate one specification point against the nearest measured tau."""
    blank = {
        "tau_spec_s": float(tau_spec),
        "limit": float(limit),
        "tau_used_s": float("nan"),
        "dev": float("nan"),
        "dev_lo": float("nan"),
        "dev_hi": float("nan"),
        "verdict": "not evaluated",
        "margin": float("nan"),
        "margin_point": float("nan"),
    }
    if not rows:
        return SpecCheck(evaluated=False, reason="no measured points", **blank)
    taus = np.array([r.tau_s for r in rows])
    i = int(np.argmin(np.abs(np.log(taus / tau_spec))))
    mismatch = abs(taus[i] / tau_spec - 1.0)
    if mismatch > SPEC_TAU_RTOL:
        return SpecCheck(
            evaluated=False,
            reason=(
                f"nearest measured tau is {taus[i]:.6g} s, "
                f"{mismatch * 100:.1f} % from the specified {tau_spec:.6g} s; "
                "the limit was not interpolated"
            ),
            **blank,
        )
    row = rows[i]
    if not np.isfinite(row.dev_hi):
        blank["tau_used_s"] = row.tau_s
        blank["dev"] = row.dev
        return SpecCheck(
            evaluated=False,
            reason=(
                "no confidence interval at this tau, so conformity cannot be "
                "decided; the point estimate alone would carry a 50 % "
                "false-accept risk at the limit"
            ),
            **blank,
        )
    return SpecCheck(
        tau_spec_s=float(tau_spec),
        limit=float(limit),
        evaluated=True,
        reason="",
        tau_used_s=row.tau_s,
        dev=row.dev,
        dev_lo=row.dev_lo,
        dev_hi=row.dev_hi,
        verdict=_verdict(row.dev_lo, row.dev_hi, float(limit)),
        margin=float(limit) / row.dev_hi,
        margin_point=float(limit) / row.dev,
    )


def _overall(specs: tuple[SpecCheck, ...]):
    """Worst verdict, and the check that earned it.

    "Worst" is by verdict class first and by guarded margin second, so a
    report on a unit that passes everywhere still names the tau it came
    closest at -- which is the number a customer reads before the verdict.
    """
    done = [s for s in specs if s.evaluated]
    if not done:
        return "none", None
    verdict = min((s.verdict for s in done), key=VERDICTS.index)
    worst = min(done, key=lambda s: (VERDICTS.index(s.verdict), s.margin))
    return verdict, worst


def _record(series) -> dict:
    """The auditable description of the record itself."""
    from ..types import PhaseSeries

    out = dict(edit_summary(series))
    out["kind"] = "phase" if isinstance(series, PhaseSeries) else "frequency"
    out["rate_hz"] = float(series.rate_hz)
    out["tau0_s"] = float(series.tau0_s)
    out["epoch"] = float(series.epoch)
    out["source"] = series.provenance.source
    out["source_sha256"] = series.provenance.source_sha256 or "not a file"
    return out


def build(
    series,
    result,
    ci=None,
    title: str = "Frequency stability measurement",
    subject: str = "",
    operator: str = "",
    notes: str = "",
    spec=(),
    spec_label: str = "",
    fit=None,
    identify_noise: bool = True,
) -> MeasurementReport:
    """Assemble a measurement report. Computes nothing the caller can pass in.

    `result` and `ci` are made by the caller so the report never becomes a
    second, divergent place where estimator options are chosen. `spec` is a
    sequence of `(tau_s, sigma_limit)` pairs -- the same shape as a bench
    `Source.spec_adev` -- and is evaluated only at taus the measurement
    actually visited; see `spec_afs`.

    `ci=None` produces a report with no error bars and therefore no
    conformity verdict, which is the honest outcome for MTIE and TIE rms.
    """
    values = np.asarray(series.x if hasattr(series, "x") else series.y)
    noise = _noise_rows(series, np.asarray(result.af).tolist(), identify_noise)
    rows = _rows(result, ci, noise)
    specs = tuple(_check_one(float(t), float(v), rows) for t, v in spec)
    verdict, worst = _overall(specs)
    environment = {
        "stabilis": __version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    # The digest covers the evidence -- the rows and the conformity
    # decisions -- and not the timestamp, the operator or the platform.
    # Re-running the same analysis on the same file must reproduce it, so
    # that two copies of a report can be compared by sixteen characters.
    payload = json.dumps(
        {"rows": [asdict(r) for r in rows],
         "specs": [asdict(s) for s in specs],
         "verdict": verdict,
         "estimator": result.kind},
        sort_keys=True,
        default=str,
    )
    return MeasurementReport(
        title=title,
        subject=subject,
        operator=operator,
        notes=notes,
        estimator=result.kind,
        level=float(ci.level) if ci is not None else float("nan"),
        rows=rows,
        specs=specs,
        spec_label=spec_label,
        verdict=verdict,
        worst=worst,
        record=_record(series),
        stats=basic_stats(values),
        fit=fit,
        provenance=result.provenance,
        environment=environment,
        digest=hashlib.sha256(payload.encode()).hexdigest(),
        generated=utc_now(),
    )
