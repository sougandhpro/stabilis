"""Averaging-factor selection.

This gets its own module because it is reportedly the single largest source
of "why doesn't this match Stable32": two tools computing identical
mathematics disagree because they chose different tau grids, or different
maximum averaging factors, and neither says so. Here the choice is explicit
and the estimators record it in provenance.

An averaging factor `m` is an integer number of samples; the corresponding
averaging time is `tau = m * tau0`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MODES", "averaging_factors", "taus_from_afs", "suggest",
           "DEFAULT_MAX_AF_DIVISOR", "MAX_TAUS", "DENSE_AF"]

MODES = ("all", "octave", "decade", "log")

# Default policy: m <= n/4. Beyond roughly this point an overlapping
# estimator has very few independent terms and the confidence interval is
# wider than the value is informative. Callers can override, and every
# estimator additionally clamps to its own hard mathematical limit.
DEFAULT_MAX_AF_DIVISOR = 4

# Hard ceiling on how many averaging factors any grid may produce.
#
# **This is a memory bound, not a taste one.** Each averaging factor keeps
# a coefficient array of order `n` for the confidence-interval path, so the
# cost is `n_taus * n`. An unbounded `all` grid on a 200 000-point record
# asks for 50 000 factors and 60 GB, and the process is killed rather than
# slowed; at a million points it is 1500 GB. Nothing warns, because running
# out of memory is not an error the code gets to report.
MAX_TAUS = 500

# Below this, `all` really is every integer. Above it the grid thins, which
# is what Stable32 does: its own all-tau export for a 4097-point record has
# 189 rows reaching AF 1006 -- consecutive at the start, then about 2 %
# geometric growth -- rather than 1006 rows.
DENSE_AF = 32


def averaging_factors(
    n: int,
    mode: str | np.ndarray | list | tuple = "octave",
    max_af: int | None = None,
    n_log: int = 30,
    max_taus: int = MAX_TAUS,
) -> np.ndarray:
    """Integer averaging factors for a record of `n` points.

    ``all``     every m up to max_af while they are few, thinning after
    ``octave``  1, 2, 4, 8, 16, ...
    ``decade``  1, 2, 4, 10, 20, 40, 100, ...  (the Stable32 convention)
    ``log``     `n_log` roughly log-spaced values, deduplicated

    `mode` may instead be an explicit sequence of averaging factors, for
    reproducing someone else's tau list — an instrument export, or a
    Stable32 run being diffed against. Explicit factors are validated and
    sorted but never thinned: an exact comparison is the one case where
    silently choosing different taus defeats the purpose.

    `max_af` defaults to ``n // DEFAULT_MAX_AF_DIVISOR``.
    """
    if n < 2:
        raise ValueError(f"need at least 2 points, got {n}")
    if not isinstance(mode, str):
        return _explicit(mode, n, max_taus)
    if mode not in MODES:
        raise ValueError(f"unknown tau mode {mode!r}; expected one of {MODES}")

    if max_af is None:
        max_af = n // DEFAULT_MAX_AF_DIVISOR
    max_af = int(min(max_af, n - 1))
    if max_af < 1:
        return np.empty(0, dtype=np.int64)

    if mode == "all":
        afs = _all(max_af, max_taus)
    elif mode == "octave":
        afs = _octave(max_af)
    elif mode == "decade":
        afs = _decade(max_af)
    else:
        afs = _log(max_af, n_log)

    if afs.size > max_taus:
        raise ValueError(
            f"tau mode {mode!r} produced {afs.size} averaging factors, over "
            f"the {max_taus} limit. Each one costs a coefficient array of "
            f"order n, so this is a memory bound rather than a preference. "
            f"Raise max_taus deliberately, or use a coarser grid."
        )
    return afs


def _explicit(afs, n: int, max_taus: int) -> np.ndarray:
    """Validate a caller-supplied list of averaging factors."""
    out = np.unique(np.asarray(afs, dtype=np.int64))
    if out.size == 0:
        raise ValueError("no averaging factors given")
    if out[0] < 1:
        raise ValueError(f"averaging factors must be >= 1, got {out[0]}")
    if out[-1] > n - 1:
        raise ValueError(
            f"averaging factor {out[-1]} needs more than {n} points")
    if out.size > max_taus:
        raise ValueError(
            f"{out.size} explicit averaging factors exceeds the {max_taus} "
            "limit; see MAX_TAUS")
    return out


def taus_from_afs(afs: np.ndarray, rate_hz: float) -> np.ndarray:
    """Averaging times in seconds: ``tau = m / rate_hz``."""
    return np.asarray(afs, dtype=np.float64) / rate_hz


def _all(max_af: int, max_taus: int) -> np.ndarray:
    """Every integer while they are few, then geometric. Bounded by build.

    Small records get exactly what "all" says -- every m from 1 to max_af.
    Large ones cannot: 250 000 factors for a million-point record is 1500 GB
    of coefficient arrays. So the grid stays consecutive to `DENSE_AF`, then
    grows geometrically at whatever ratio fits the remaining budget.

    That is the shape Stable32 produces, and its own export is the evidence:
    189 rows for a 4097-point record, consecutive at the start and about
    2 % apart at the end. This reproduces it closely at the default budget,
    which is why it is fast where a truly exhaustive grid is not.
    """
    if max_af <= max_taus:
        return np.arange(1, max_af + 1, dtype=np.int64)

    dense = min(max_af, DENSE_AF)
    head = np.arange(1, dense + 1, dtype=np.int64)
    budget = max_taus - dense
    if budget <= 0 or max_af <= dense:
        return head
    # geomspace over exactly `budget` points, so the total is bounded by
    # construction rather than by a loop that has to be checked afterwards
    # -- a walk with a `max(af + 1, ...)` floor can overshoot the budget
    # when the ratio is small, and did.
    tail = np.round(np.geomspace(dense + 1, max_af, budget)).astype(np.int64)
    return np.unique(np.concatenate([head, tail]))


def _octave(max_af: int) -> np.ndarray:
    """1, 2, 4, 8, ... up to max_af."""
    k = int(np.floor(np.log2(max_af)))
    return (1 << np.arange(k + 1, dtype=np.int64)).astype(np.int64)


def _decade(max_af: int) -> np.ndarray:
    """1, 2, 4 per decade -- what Stable32 calls decade spacing."""
    out: list[int] = []
    decade = 1
    while decade <= max_af:
        for mult in (1, 2, 4):
            af = mult * decade
            if af <= max_af:
                out.append(af)
        decade *= 10
    return np.asarray(out, dtype=np.int64)


def _log(max_af: int, n_log: int) -> np.ndarray:
    """`n_log` log-spaced factors; duplicates collapse at small max_af."""
    if n_log < 1:
        raise ValueError(f"n_log must be >= 1, got {n_log}")
    raw = np.logspace(0.0, np.log10(max_af), n_log)
    return np.unique(np.round(raw).astype(np.int64))


def suggest(n: int, rate_hz: float = 1.0) -> dict:
    """A sensible tau grid for a record of `n` points, and why.

    Returned rather than applied: the caller shows it, the user overrides
    it. A suggestion that cannot be overridden is a constraint, and the
    right grid depends on what the record is *for* -- a spec check wants
    the mask's taus, not ours.

    `octave` for anything large, because it is the fewest points that still
    resolves a power law; `all` while the record is short enough that every
    integer is affordable and the extra detail is free.
    """
    if n < 2:
        raise ValueError(f"need at least 2 points, got {n}")
    max_af = max(1, n // DEFAULT_MAX_AF_DIVISOR)
    if max_af <= MAX_TAUS:
        mode = "all"
        why = (f"every averaging factor to {max_af} = N/{DEFAULT_MAX_AF_DIVISOR}; "
               f"only {max_af} of them, so the detail is free")
    else:
        mode = "octave"
        why = (f"octave to {max_af} = N/{DEFAULT_MAX_AF_DIVISOR}; 'all' here "
               f"would thin to {MAX_TAUS} anyway")
    afs = averaging_factors(n, mode, max_af=max_af)
    return {
        "mode": mode,
        "max_af": int(max_af),
        "n_taus": int(afs.size),
        "tau_min_s": float(afs[0] / rate_hz),
        "tau_max_s": float(afs[-1] / rate_hz),
        "why": why,
    }
