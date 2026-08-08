"""Power-law decomposition: measured sigma(tau) curves into `h_alpha`.

Fits

    sigma2_row(tau) = sum_alpha c_alpha(row, tau, f_h) * h_alpha
                      + (tau**2/2) * D**2

over stacked rows, where a *row* is one (estimator, tau) pair. The
coefficients are `powerlaw_coeffs`; the unknowns are the five IEEE 1139
noise powers and the drift.

**Why this is a linear solve and not an optimizer.** Every `h_alpha` is a
noise power and cannot be negative, and drift enters as `D**2`, so the
model is linear in its parameters with a non-negativity constraint on all
six -- which is exactly non-negative least squares. `_nnls` terminates at
the true constrained minimum in finitely many steps, so a zero coming back
for some `h_alpha` is a statement about the data rather than possibly a
statement about a solver that stopped early. Every fit carries its own KKT
optimality certificate.

**Why AVAR rows alone are not enough.** White PM and flicker PM have the
same Allan tau-slope; SP 1065 says so in its own introduction to domain
conversions. Stacking modified rows breaks the degeneracy, because the
modified estimator's bandwidth scales as `1/tau` and its white-PM row falls
as `tau**-3` against flicker PM's `tau**-2`. Hand this function AVAR rows
only and it will say so rather than return two numbers whose sum is
determined and whose split is not.

See theory/powerlaw.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._nnls import kkt_residual, nnls
from ..log import record, warn
from ..powerlaw_coeffs import (
    ALPHAS,
    MIN_FH_TAU_PRODUCT,
    avar_coeff,
    drift_coeff,
    mvar_coeff,
)
from ..provenance import Provenance, new_provenance
from .core import StabilityResult
from .edof import edof

__all__ = ["PowerLawFit", "WEIGHT_METHODS", "REFUSED_KINDS", "decompose"]

WEIGHT_METHODS = ("edof", "none")

# Statistics that are not variances. The whole model is a sum of variances,
# and these break the sum rule rather than merely fitting it badly, so they
# are refused instead of down-weighted. Same rule the M5 bench solver needs.
REFUSED_KINDS = frozenset({"mtie", "tierms"})

# Which coefficient table a result's `kind` uses. Anything whose `shape`
# says `modified` takes the modified column.
_MODIFIED_KINDS = frozenset({"mdev", "mvar", "tdev", "tvar"})


@dataclass(frozen=True)
class PowerLawFit:
    """A decomposition, with everything needed to defend it."""

    h: dict[int, float]
    drift: float
    tau_s: np.ndarray
    fitted: np.ndarray
    measured: np.ndarray
    residual: float
    kkt: float
    n_rows: int
    dropped_rows: int
    f_h: float
    assumed_fh: bool
    weights: str
    provenance: Provenance

    @property
    def dominant(self) -> int | None:
        """The alpha carrying the most power at the shortest fitted tau.

        `None` when every `h_alpha` came back zero, which happens for a
        record that is pure drift. Reported rather than raising: "no
        stochastic noise resolved" is a real answer.
        """
        if not any(v > 0.0 for v in self.h.values()):
            return None
        tau = float(self.tau_s.min())
        power = {a: self.h[a] * float(avar_coeff(a, np.array([tau]), self.f_h)[0])
                 for a in ALPHAS}
        return max(power, key=power.get)


def _rows(result: StabilityResult, f_h: float):
    """Design rows, measured variances and the keep mask for one curve.

    Rows failing `2*pi*f_h*tau >= MIN_FH_TAU_PRODUCT` are dropped, not
    fitted. The conversions are asymptotic in that product; below it the
    coefficient is wrong in a way that no residual reports, and a fitted
    number from an invalid row is worse than a missing one.
    """
    if result.kind in REFUSED_KINDS:
        raise ValueError(
            f"decompose: {result.kind!r} is not a variance -- it is an "
            "extreme-value statistic, and the sum over power laws that this "
            f"fit inverts does not hold for it. Refused rather than fitted. "
            f"Supply Allan-family results instead ({sorted(REFUSED_KINDS)} "
            "are excluded)"
        )
    modified = (result.shape.modified if result.shape is not None
                else result.kind in _MODIFIED_KINDS)
    keep = 2.0 * np.pi * f_h * result.tau_s >= MIN_FH_TAU_PRODUCT
    tau = result.tau_s[keep]
    cols = [mvar_coeff(a, tau) if modified else avar_coeff(a, tau, f_h)
            for a in ALPHAS]
    cols.append(drift_coeff(tau))
    return np.column_stack(cols), result.var[keep], tau, keep


def decompose(
    results: StabilityResult | list[StabilityResult],
    f_h: float | None = None,
    weights: str = "edof",
) -> PowerLawFit:
    """Fit `h_alpha` and drift to one or more measured stability curves.

    `results` is one `StabilityResult` or a list of them; stacking an
    `oadev` and an `mdev` of the same data is the intended use and the only
    way the two PM noises separate.

    `f_h` is the measurement system's upper cutoff in hertz. It is **not
    identifiable from tau-domain data** -- it is an input. Omitted, it
    defaults to the Nyquist frequency of the fastest result and the fit is
    flagged `assumed_fh`, which propagates into provenance and must be
    shown wherever the result is.

    `weights="edof"` weights each row by its own sampling uncertainty,
    `Var(sigma2_hat) = 2*sigma**4/eta`, so that a long-tau point with three
    degrees of freedom does not outvote a short-tau point with three
    hundred. `weights="none"` fits the raw variances, which is dominated by
    whichever rows are numerically largest and is offered only for
    diagnostics.
    """
    if isinstance(results, StabilityResult):
        results = [results]
    if not results:
        raise ValueError("decompose: no results given")
    if weights not in WEIGHT_METHODS:
        raise ValueError(
            f"decompose: weights must be one of {WEIGHT_METHODS}, got {weights!r}")

    assumed_fh = f_h is None
    if assumed_fh:
        f_h = max(r.rate_hz for r in results) / 2.0
    if f_h <= 0.0:
        raise ValueError(f"decompose: f_h must be positive, got {f_h}")

    blocks, rhs, taus, sigmas, dropped = [], [], [], [], 0
    for r in results:
        a, b, tau, keep = _rows(r, f_h)
        dropped += int((~keep).sum())
        if b.size:
            blocks.append(a)
            rhs.append(b)
            taus.append(tau)
            sigmas.append(_row_sigma(r, keep) if weights == "edof"
                          else np.ones(b.size))
    if not blocks:
        raise ValueError(
            f"decompose: every row was dropped -- none satisfies "
            f"2*pi*f_h*tau >= {MIN_FH_TAU_PRODUCT} at f_h = {f_h:.6g} Hz. "
            "Either f_h is far smaller than the measurement bandwidth, or "
            "the tau grid is entirely below the valid range"
        )

    a = np.vstack(blocks)
    b = np.concatenate(rhs)
    w = 1.0 / np.concatenate(sigmas)
    n_unknowns = a.shape[1]
    if a.shape[0] < n_unknowns:
        raise ValueError(
            f"decompose: {a.shape[0]} usable rows for {n_unknowns} unknowns. "
            "Add taus, or stack a modified estimator alongside the "
            "unmodified one -- see theory/powerlaw.md"
        )

    x, residual = nnls(a * w[:, None], b * w)
    kkt = kkt_residual(a * w[:, None], b * w, x)

    prov = new_provenance("stats.powerlaw.decompose")
    prov = record(prov, "stats.powerlaw", "decompose",
                  {"kinds": [r.kind for r in results], "rows": int(a.shape[0]),
                   "dropped": dropped, "f_h": float(f_h), "weights": weights},
                  "ok")
    if assumed_fh:
        prov = warn(prov, "stats.powerlaw", "assumed_fh", {"f_h": float(f_h)},
                    "f_h was not supplied and is not identifiable from the "
                    "data; Nyquist assumed. The white-PM and flicker-PM "
                    "levels scale with it and must not be quoted without it")
    if dropped:
        prov = record(prov, "stats.powerlaw", "dropped_rows",
                      {"count": dropped, "rule": f"2*pi*f_h*tau >= "
                       f"{MIN_FH_TAU_PRODUCT}"}, "outside the valid range")

    return PowerLawFit(
        h={alpha: float(x[i]) for i, alpha in enumerate(ALPHAS)},
        drift=float(np.sqrt(x[-1])),
        tau_s=np.concatenate(taus),
        fitted=a @ x,
        measured=b,
        residual=float(residual),
        kkt=float(kkt),
        n_rows=int(a.shape[0]),
        dropped_rows=dropped,
        f_h=float(f_h),
        assumed_fh=assumed_fh,
        weights=weights,
        provenance=prov,
    )


def _row_sigma(result: StabilityResult, keep: np.ndarray) -> np.ndarray:
    """Standard error of each kept row's measured variance.

        Var(sigma2_hat) = 2 * sigma**4 / eta

    The definition of equivalent degrees of freedom, run backwards. `keep`
    is the boolean mask over the full tau grid, so nothing here depends on
    the dropped rows happening to be contiguous.

    Falls back to the measured variance itself where the estimator kept no
    coefficient arrays, which makes the fit relative rather than
    uncertainty-weighted -- still far better than not weighting at all,
    since the rows span decades.
    """
    idx = np.flatnonzero(keep)
    var = np.abs(result.var[idx])
    if not result.coeffs:
        return var
    etas = np.array([max(edof(result.coeffs[i], int(result.af[i]),
                              result.span_factor)[0], 1.0) for i in idx])
    return np.sqrt(2.0 / etas) * var
