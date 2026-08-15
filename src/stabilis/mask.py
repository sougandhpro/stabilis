"""Limit masks, and the verdict a unit gets against one.

"Does this unit meet spec" is the question a calibration report exists to
answer, and a mask is how a telecom standard asks it: a piecewise limit on
`MTIE(tau)` or `TDEV(tau)`, and a curve that must stay under it.

Three things here are not obvious and all three change verdicts.

**A maximum and a variance are not judged the same way.** For a variance
the verdict is taken on the confidence interval, exactly as
`theory/report.md` sets out. MTIE is not a variance: it is an observed
extreme, and observing an excursion is conclusive while failing to observe
one is not. So exceeding an MTIE limit is a `fail` on the spot, whereas
staying under it is only a pass *over the window actually measured* -- and
the window is therefore part of the verdict rather than a footnote.

**A standard's measurement conditions are part of its limit.** G.811 does
not merely give numbers; it requires a sampling time no coarser than 1/30 s
and, for TDEV, a record at least twelve times the integration period. A
curve that violates those has not been compared with the standard, whatever
the numbers say, so those points come back `not_evaluated` rather than
`pass`.

**A limit is never interpolated, and never extrapolated.** Each mask states
the tau range over which it applies. Outside it there is no limit to
compare against -- not a limit obtained by continuing the last segment.

Every catalogued mask is transcribed from the standard's own text, with the
source and the verbatim wording in this file, and converted to SI seconds
at transcription time so that nothing downstream handles microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .log import record, warn
from .provenance import Provenance

__all__ = ["Segment", "Mask", "MaskCheck", "MaskResult", "VERDICTS",
           "CATALOG", "evaluate", "limit_at", "available"]

# `not_evaluated` is first because it is not a degree of failure -- it is
# the absence of a comparison, and sorting it with the verdicts would let a
# summary quietly average it in.
VERDICTS = ("not_evaluated", "fail", "indeterminate", "pass")


@dataclass(frozen=True)
class Segment:
    """One piece of a piecewise limit, over `tau_lo < tau <= tau_hi`.

        limit(tau) = offset + coeff * tau**power

    Three terms rather than a general polynomial because that is the shape
    every telecom mask actually takes -- a floor, a slope, and occasionally
    a square root -- and a general form would invite fitting a mask instead
    of transcribing one.
    """

    tau_lo: float
    tau_hi: float
    offset: float
    coeff: float = 0.0
    power: float = 1.0

    def limit(self, tau: float) -> float:
        return self.offset + self.coeff * tau ** self.power


@dataclass(frozen=True)
class Mask:
    """A named limit and the conditions under which it means anything."""

    name: str
    quantity: str
    segments: tuple[Segment, ...]
    source: str
    units: str = "s"
    # Conditions the standard states. Violating one does not make the unit
    # fail; it makes the comparison invalid, which is a different answer.
    max_tau0_s: float | None = None
    min_record_factor: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def tau_min(self) -> float:
        return min(s.tau_lo for s in self.segments)

    @property
    def tau_max(self) -> float:
        return max(s.tau_hi for s in self.segments)


@dataclass(frozen=True)
class MaskCheck:
    """One tau, judged."""

    tau_s: float
    measured: float
    limit: float | None
    verdict: str
    margin: float | None
    reason: str = ""


@dataclass(frozen=True)
class MaskResult:
    """A curve against a mask, and the one point that decides the outcome."""

    mask: str
    quantity: str
    verdict: str
    checks: tuple[MaskCheck, ...]
    provenance: Provenance

    @property
    def worst(self) -> MaskCheck | None:
        """The evaluated point with the least headroom, or None if none was.

        Reported instead of a bare pass/fail because a unit that passes by
        1% and one that passes by 40% are not the same unit, and the
        milestone this closes asks for the margin at the worst point.
        """
        scored = [c for c in self.checks if c.margin is not None]
        return min(scored, key=lambda c: c.margin) if scored else None


def limit_at(mask: Mask, tau: float) -> float | None:
    """The limit at `tau`, or None where the mask does not reach.

    Segment bounds are half-open on the left (`tau_lo < tau <= tau_hi`),
    which is how the standards write them and which matters at the
    breakpoints: G.811's TDEV mask changes form exactly at tau = 100 s and
    the two forms agree there, so the choice is invisible when transcription
    is right and loud when it is not.
    """
    for seg in mask.segments:
        if seg.tau_lo < tau <= seg.tau_hi:
            return seg.limit(tau)
    return None


def evaluate(result, mask: Mask, ci=None, series=None) -> MaskResult:
    """Judge a curve against a mask, one tau at a time.

    `ci` is used when the statistic is a variance, where the verdict is
    taken on the interval. It is ignored for an extreme-value statistic,
    which has no such interval and does not need one to fail.

    `series` supplies the record length and sample interval, which some
    standards constrain. Without it those conditions cannot be checked and
    the points they govern are reported `not_evaluated` -- absence of
    evidence, rather than the benefit of the doubt.
    """
    if result.kind != mask.quantity:
        raise ValueError(
            f"mask {mask.name!r} limits {mask.quantity!r} and this is a "
            f"{result.kind!r} curve. A limit on one statistic says nothing "
            "about another; compute the one the mask names."
        )

    is_extreme = not result.is_variance
    lo, hi = _bounds(result, ci, is_extreme)
    checks = [
        _judge(float(t), float(m), float(l), float(h), mask, series, is_extreme)
        for t, m, l, h in zip(result.tau_s, result.dev, lo, hi)
    ]

    prov = record(
        (ci or result).provenance, "mask", "evaluate",
        {"mask": mask.name, "quantity": mask.quantity,
         "on_interval": bool(ci is not None and not is_extreme),
         "n_evaluated": sum(c.verdict != "not_evaluated" for c in checks)},
        _overall(checks),
    )
    if all(c.verdict == "not_evaluated" for c in checks):
        prov = warn(prov, "mask", "nothing_evaluated", {"mask": mask.name},
                    "no tau met this mask's stated range and measurement "
                    "conditions, so the curve has not been compared with it")
    return MaskResult(mask=mask.name, quantity=mask.quantity,
                      verdict=_overall(checks), checks=tuple(checks),
                      provenance=prov)


def available() -> list[str]:
    return sorted(CATALOG)


# --------------------------------------------------------------------------


def _bounds(result, ci, is_extreme):
    """What gets compared: the interval for a variance, the point otherwise."""
    dev = np.asarray(result.dev, dtype=np.float64)
    if is_extreme or ci is None:
        return dev, dev
    return np.asarray(ci.dev_lo), np.asarray(ci.dev_hi)


def _judge(tau, measured, lo, hi, mask, series, is_extreme) -> MaskCheck:
    """One point. Conditions first, then the limit."""
    limit = limit_at(mask, tau)
    if limit is None:
        return MaskCheck(tau, measured, None, "not_evaluated", None,
                         f"tau is outside the mask's range "
                         f"[{mask.tau_min:g}, {mask.tau_max:g}] s")

    unmet = _conditions_unmet(tau, mask, series)
    if unmet:
        return MaskCheck(tau, measured, limit, "not_evaluated", None, unmet)

    margin = limit / measured if measured > 0 else np.inf
    if is_extreme:
        # An observed excursion is conclusive; its absence is not. A record
        # twice as long can only find a larger maximum, never a smaller one.
        verdict = "fail" if measured > limit else "pass"
        reason = ("" if verdict == "fail" else
                  "over the window measured; a longer record can only "
                  "increase an observed maximum")
        return MaskCheck(tau, measured, limit, verdict, margin, reason)

    if hi <= limit:
        return MaskCheck(tau, measured, limit, "pass", margin)
    if lo > limit:
        return MaskCheck(tau, measured, limit, "fail", margin)
    return MaskCheck(tau, measured, limit, "indeterminate", margin,
                     "the interval straddles the limit; the record is too "
                     "short to decide at this confidence")


def _conditions_unmet(tau, mask, series) -> str:
    """The standard's own measurement conditions, checked before its numbers.

    A curve that violates them has not been compared with the standard,
    whatever the numbers say.
    """
    if mask.max_tau0_s is None and mask.min_record_factor <= 0.0:
        return ""
    if series is None:
        return ("this mask states measurement conditions and no series was "
                "supplied to check them against")
    if mask.max_tau0_s is not None and series.tau0_s > mask.max_tau0_s:
        return (f"the record is sampled at {series.tau0_s:g} s, coarser than "
                f"the {mask.max_tau0_s:g} s this standard requires")
    if mask.min_record_factor > 0.0:
        need = mask.min_record_factor * tau
        if series.duration_s < need:
            return (f"the standard requires a record of at least "
                    f"{mask.min_record_factor:g}*tau = {need:g} s here, and "
                    f"this one is {series.duration_s:g} s")
    return ""


def _overall(checks) -> str:
    """The worst verdict present. Absence of a comparison is not a pass."""
    seen = {c.verdict for c in checks}
    for verdict in ("fail", "indeterminate", "pass"):
        if verdict in seen:
            return verdict
    return "not_evaluated"


# --------------------------------------------------------------------------
# The catalogue.
#
# Transcribed from the standards' own text, fetched rather than recalled --
# see references.md. Converted to SI seconds here, at transcription time,
# so that nothing downstream ever handles microseconds.
# --------------------------------------------------------------------------

_G811 = ("ITU-T Recommendation G.811 (09/97), 'Timing characteristics of "
         "primary reference clocks', clause 6.1")

_G811_CONDITIONS = (
    "MTIE and TDEV are measured through an equivalent 10 Hz, first-order, "
    "low-pass measurement filter -- which this library cannot verify was "
    "applied, and does not apply itself.",
    "Measured using the independent clock configuration of Figure 2a/G.810.",
)

CATALOG: dict[str, Mask] = {
    # Verbatim from clause 6.1:
    #   MTIE:  0.275 x 10^-3 tau + 0.025 us   for 0.1 < tau <= 1000 s
    #          10^-5 tau + 0.29 us            for tau > 1000 s
    # Both expressions are in microseconds; 1e-6 converts each to seconds.
    # They agree at the breakpoint -- 0.3 us from either side -- which is
    # the arithmetic check that the transcription is right.
    "itu-t-g.811-mtie": Mask(
        name="itu-t-g.811-mtie",
        quantity="mtie",
        segments=(
            Segment(tau_lo=0.1, tau_hi=1000.0,
                    offset=0.025e-6, coeff=0.275e-3 * 1e-6, power=1.0),
            Segment(tau_lo=1000.0, tau_hi=np.inf,
                    offset=0.29e-6, coeff=1e-5 * 1e-6, power=1.0),
        ),
        source=_G811,
        max_tau0_s=1.0 / 30.0,
        notes=_G811_CONDITIONS,
    ),
    # Verbatim from clause 6.1:
    #   TDEV:  3 ns        for 0.1 < tau <= 100 s
    #          0.03 tau ns for 100 < tau <= 1000 s
    #          30 ns       for 1000 < tau < 10 000 s
    # Continuous at both breakpoints (3 ns at 100 s, 30 ns at 1000 s).
    # Note the mask *ends* at 10 000 s: beyond it G.811 states no limit, so
    # neither does this.
    "itu-t-g.811-tdev": Mask(
        name="itu-t-g.811-tdev",
        quantity="tdev",
        segments=(
            Segment(tau_lo=0.1, tau_hi=100.0, offset=3e-9, coeff=0.0),
            Segment(tau_lo=100.0, tau_hi=1000.0,
                    offset=0.0, coeff=0.03e-9, power=1.0),
            Segment(tau_lo=1000.0, tau_hi=10_000.0, offset=30e-9, coeff=0.0),
        ),
        source=_G811,
        max_tau0_s=1.0 / 30.0,
        min_record_factor=12.0,   # "minimum measurement period ... T = 12 tau"
        notes=_G811_CONDITIONS,
    ),
}
