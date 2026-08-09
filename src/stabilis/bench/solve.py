"""One linear solve per tau, subsuming three features that are usually three.

    sigma2_meas = sigma2_i + sigma2_j + sigma2_channel

is linear in the variances and non-negative by physics, so the whole bench
is `A @ v = b`, `v >= 0` -- exactly non-negative least squares, solved
exactly by `_nnls`. Three named techniques are special cases of the same
solve rather than separate code:

* **three-cornered hat** is the 3-source triangle;
* **N-cornered hat** is any larger graph containing an odd cycle;
* **instrument noise-floor subtraction** is what a back-to-back row does --
  both ends on one source, so it drops out of every source column and pins
  only its channel.

**Any estimator that is a variance works for free**, because the sum rule
holds for anything that is one. What the solver refuses is anything whose
result says `is_variance=False` -- MTIE, because the maximum of a sum is
not the sum of maxima, and TIE rms, because it retains a frequency offset
that is deterministic rather than adding in quadrature. Two different
failures, checked by one flag.

**Negative variances are reported, not clipped.** The unconstrained
least-squares solution is computed alongside the constrained one and
returned signed. A negative source variance is the system saying an
assumption is violated -- the sources are not independent, or one is
misattributed, or a channel is worse than believed. Clipping it to zero
destroys the single most useful diagnostic the method produces, which is
why the NNLS answer and the signed one are both on the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._nnls import kkt_residual, nnls
from ..log import record, warn
from ..provenance import Provenance, new_provenance
from .graph import check, design_matrix
from .model import Bench

__all__ = ["BenchSolution", "REFUSED_KINDS", "solve", "common_mode_warnings"]

# Statistics that are not variances, detected by the result's own
# `is_variance` flag rather than by name so a plugin estimator cannot slip
# past. The names below are for the error message only.
#
# The two this library ships fail for *different* reasons, and saying
# "extreme-value statistics" of both would be wrong. MTIE is one: the
# maximum of a sum is not the sum of maxima. TIE rms is a mean square and
# its stochastic part does obey the sum rule -- it is refused because it
# retains the frequency offset between the sources, which is deterministic,
# so two clocks 1e-12 off can differ by 2e-12 or by nothing depending on
# sign. Neither combines the way this solver assumes.
REFUSED_KINDS = frozenset({"mtie", "tierms"})


@dataclass(frozen=True)
class BenchSolution:
    """Per-tau variances for every source and every free channel."""

    tau_s: np.ndarray
    sources: tuple[str, ...]
    channels: tuple[str, ...]
    source_var: np.ndarray          # (n_tau, n_sources), non-negative
    channel_var: np.ndarray         # (n_tau, n_channels), non-negative
    unconstrained: np.ndarray       # (n_tau, n_unknowns), signed
    residual: np.ndarray
    kkt: np.ndarray
    estimator: str
    warnings: tuple[str, ...]
    provenance: Provenance

    def dev(self, source_id: str) -> np.ndarray:
        """Deviation for one source. Zero where the fit pinned it at zero."""
        return np.sqrt(self.source_var[:, self.sources.index(source_id)])

    def signed_var(self, source_id: str) -> np.ndarray:
        """The *unconstrained* variance for one source, sign intact.

        Read this before trusting the constrained answer. A negative value
        is not a numerical artefact, it is the measurement telling you the
        independence assumption failed somewhere -- and the constrained
        solve, being obliged to return something non-negative, cannot say
        so on its own.
        """
        return self.unconstrained[:, self.sources.index(source_id)]

    @property
    def any_negative(self) -> bool:
        return bool(np.any(self.unconstrained < 0.0))


def common_mode_warnings(bench: Bench) -> tuple[str, ...]:
    """Correlations the bench can see, each naming its own evidence.

    The whole system rests on the off-diagonal terms vanishing, and these
    are the common ways they do not. Warnings rather than refusals: they
    are statements about the physics that the graph cannot settle on its
    own, and the user may know better.
    """
    out = []
    owner = bench.channel_index()
    by_device: dict[str, list] = {}
    for m in bench.measurements:
        by_device.setdefault(owner[m.channel_id], []).append(m)

    for device_id, group in sorted(by_device.items()):
        if len(group) < 2:
            continue
        channels = {m.channel_id for m in group}
        if len(channels) == 1 and len(group) > 1:
            one = next(iter(channels))
            same_epoch = _overlapping_epochs([m for m in group if m.epoch])
            if same_epoch:
                out.append(
                    f"channel {one}: measurements {', '.join(same_epoch)} "
                    "share a channel AND an epoch. One channel cannot make "
                    "two simultaneous measurements, so these are the same "
                    "record described twice -- not independent rows, and the "
                    "solve will over-count them."
                )
            else:
                out.append(
                    f"channel {one}: all {len(group)} measurements run through "
                    "it. That is normal practice for pairs measured in turn, "
                    "and it is why this bench has one channel floor rather "
                    "than several -- its noise is common to every row and "
                    "cannot be attributed to any one of them. Add a "
                    "back-to-back run to measure that floor rather than "
                    "assume it."
                )
        elif len(channels) > 1:
            out.append(
                f"device {device_id}: {len(group)} measurements across "
                f"{len(channels)} channels of ONE instrument. They share a "
                "timebase, LO, supply and thermal environment, so the "
                "instrument noise is common and does **not** cancel in the "
                "hat. This is the dangerous case, because the answer still "
                "looks reasonable. Independent instruments are what the "
                "method assumes -- and are better for cross-correlation "
                "too, which rejects only the part two channels do NOT "
                "share. See `bench recommend`."
            )

    shared: dict[str, list[str]] = {}
    for m in bench.measurements:
        if m.is_back_to_back:
            continue
        for s in (m.source_a, m.source_b):
            shared.setdefault(s, []).append(m.id)
    for source_id, ids in sorted(shared.items()):
        if len(ids) > 2:
            out.append(
                f"source {source_id} appears in {len(ids)} measurements "
                f"({', '.join(sorted(ids))}). Their *estimates* are then "
                "correlated even when the physics is clean, which inflates "
                "the uncertainty on the solution without biasing it. The "
                "solve does not yet whiten for this; treat the spread "
                "between taus as optimistic."
            )
    return tuple(out)


def _overlapping_epochs(measurements) -> list[str]:
    """Ids of measurements claiming the same epoch on one channel.

    Only exact string equality, deliberately. A real overlap test needs a
    start and a duration, which a `Measurement` does not carry; claiming to
    detect partial overlap from an epoch alone would be inventing a
    capability. Identical epochs are unambiguous, and that is the case
    worth catching.
    """
    seen: dict[str, list[str]] = {}
    for m in measurements:
        seen.setdefault(m.epoch, []).append(m.id)
    clashes = [ids for ids in seen.values() if len(ids) > 1]
    return sorted(i for group in clashes for i in group)


def solve(
    bench: Bench,
    results: dict,
    require_identifiable: bool = True,
) -> BenchSolution:
    """Solve the bench at every tau the measurements have in common.

    `results` maps `Measurement.id` to a `StabilityResult` -- computed by
    the caller, with whatever estimator and settings it likes, which is how
    "any variance works" stays true without this module knowing anything
    about estimators.

    `require_identifiable=False` solves anyway and lets NNLS return the
    minimum-norm-ish answer among the many that fit. It exists because a
    user may want to see the fit for a bench they know is degenerate, and
    it warns loudly. The default refuses, with the diagnostic that names
    the measurement which would fix it.
    """
    if not results:
        raise ValueError("bench.solve: no results given")

    kinds = {r.kind for r in results.values()}
    bad = sorted({r.kind for r in results.values() if not r.is_variance})
    if bad:
        raise ValueError(
            f"bench.solve: {bad} cannot be combined this way. The sum rule "
            "sigma2_meas = sigma2_i + sigma2_j + sigma2_c holds for "
            "variances, and these are not variances -- for MTIE the maximum "
            "of a sum is not the sum of maxima, and TIE rms retains a "
            "frequency offset that is deterministic rather than adding in "
            "quadrature. Refused rather than fitted."
        )
    if len(kinds) > 1:
        raise ValueError(
            f"bench.solve: results mix estimators {sorted(kinds)}. The sum "
            "rule holds within one statistic, not across them -- an oadev "
            "row and an mdev row are not the same quantity and cannot share "
            "a solve."
        )
    estimator = next(iter(kinds))

    ident = check(bench)
    if require_identifiable and not ident.identifiable:
        raise ValueError(
            "bench.solve: this bench cannot determine its unknowns.\n"
            + ident.report()
            + "\n\nPass require_identifiable=False to solve anyway, knowing "
            "the answer is one of infinitely many that fit equally well."
        )

    used = [m for m in bench.measurements if m.id in results]
    missing = [m.id for m in bench.measurements if m.id not in results]
    if not used:
        raise ValueError(
            f"bench.solve: none of the bench's measurements are in `results` "
            f"(bench has {[m.id for m in bench.measurements]})"
        )

    taus = _common_taus([results[m.id] for m in used])
    a_full, sources, channels, known = design_matrix(bench)
    keep = [i for i, m in enumerate(bench.measurements) if m.id in results]
    a = a_full[keep]
    known = known[keep]

    n_tau, n_unknown = taus.size, a.shape[1]
    src_var = np.zeros((n_tau, len(sources)))
    chan_var = np.zeros((n_tau, len(channels)))
    signed = np.zeros((n_tau, n_unknown))
    residual = np.zeros(n_tau)
    kkt = np.zeros(n_tau)

    for k, tau in enumerate(taus):
        b = np.array([_var_at(results[m.id], tau) for m in used]) - known
        x, res = nnls(a, b)
        residual[k] = res
        kkt[k] = kkt_residual(a, b, x)
        src_var[k] = x[: len(sources)]
        chan_var[k] = x[len(sources):]
        # The unconstrained solution, sign intact. lstsq rather than a
        # solve: the system may be over- or under-determined and this is a
        # diagnostic, not the answer.
        signed[k] = np.linalg.lstsq(a, b, rcond=None)[0]

    warnings = common_mode_warnings(bench)
    prov = record(
        new_provenance("bench.solve"),
        "bench",
        "solve",
        {"estimator": estimator, "rows": len(used), "taus": int(n_tau),
         "sources": list(sources), "free_channels": list(channels),
         "identifiable": ident.identifiable},
        "ok",
    )
    if missing:
        prov = record(prov, "bench", "measurements_omitted",
                      {"ids": missing}, "no result supplied for these rows")
    if not ident.identifiable:
        prov = warn(prov, "bench", "not_identifiable",
                    {"rank": ident.rank, "unknowns": ident.n_unknowns},
                    "solved anyway at the caller's request; the answer is "
                    "one of infinitely many that fit equally well")
    for text in warnings:
        prov = warn(prov, "bench", "correlation_warning", {}, text)
    if np.any(signed < 0.0):
        worst = float(signed.min())
        prov = warn(prov, "bench", "negative_variance", {"most_negative": worst},
                    "the unconstrained solution went negative, so an "
                    "independence assumption is violated somewhere. The "
                    "constrained answer is still non-negative and still "
                    "wrong in the same place")

    return BenchSolution(
        tau_s=taus,
        sources=sources,
        channels=channels,
        source_var=src_var,
        channel_var=chan_var,
        unconstrained=signed,
        residual=residual,
        kkt=kkt,
        estimator=estimator,
        warnings=warnings,
        provenance=prov,
    )


def _common_taus(results: list) -> np.ndarray:
    """Taus every result has, as floats, in increasing order.

    Intersecting rather than interpolating is deliberate: an interpolated
    sigma(tau) is not a measurement, and the whole bench solve would then
    be partly about the interpolation. If two records have no tau in
    common, that is a fact about the experiment and it is reported as one.
    """
    common = None
    for r in results:
        # Round to nanoseconds before intersecting: taus computed as
        # af/rate from different records agree to floating point, not to
        # the bit, and an exact set intersection would silently return
        # nothing.
        keys = {round(float(t), 9) for t in r.tau_s}
        common = keys if common is None else (common & keys)
    if not common:
        spans = ", ".join(
            f"{r.kind} {r.tau_s[0]:g}..{r.tau_s[-1]:g}s" for r in results
        )
        raise ValueError(
            "bench.solve: the measurements share no averaging time. Nothing "
            "is interpolated here, so they must be analysed on a common tau "
            f"grid -- check tau0 and the tau mode. Spans: {spans}"
        )
    return np.array(sorted(common))


def _var_at(result, tau: float) -> float:
    idx = np.flatnonzero(np.round(result.tau_s, 9) == tau)
    if idx.size == 0:
        raise KeyError(f"tau {tau} not in {result.kind} result")
    return float(result.var[idx[0]])
