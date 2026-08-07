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

__all__ = ["MODES", "averaging_factors", "taus_from_afs", "DEFAULT_MAX_AF_DIVISOR"]

MODES = ("all", "octave", "decade", "log")

# Default policy: m <= n/4. Beyond roughly this point an overlapping
# estimator has very few independent terms and the confidence interval is
# wider than the value is informative. Callers can override, and every
# estimator additionally clamps to its own hard mathematical limit.
DEFAULT_MAX_AF_DIVISOR = 4


def averaging_factors(
    n: int,
    mode: str = "octave",
    max_af: int | None = None,
    n_log: int = 30,
) -> np.ndarray:
    """Integer averaging factors for a record of `n` points.

    ``all``     every m from 1 to max_af
    ``octave``  1, 2, 4, 8, 16, ...
    ``decade``  1, 2, 4, 10, 20, 40, 100, ...  (the Stable32 convention)
    ``log``     `n_log` roughly log-spaced values, deduplicated

    `max_af` defaults to ``n // DEFAULT_MAX_AF_DIVISOR``.
    """
    if mode not in MODES:
        raise ValueError(f"unknown tau mode {mode!r}; expected one of {MODES}")
    if n < 2:
        raise ValueError(f"need at least 2 points, got {n}")

    if max_af is None:
        max_af = n // DEFAULT_MAX_AF_DIVISOR
    max_af = int(min(max_af, n - 1))
    if max_af < 1:
        return np.empty(0, dtype=np.int64)

    if mode == "all":
        afs = np.arange(1, max_af + 1, dtype=np.int64)
    elif mode == "octave":
        afs = _octave(max_af)
    elif mode == "decade":
        afs = _decade(max_af)
    else:
        afs = _log(max_af, n_log)
    return afs


def taus_from_afs(afs: np.ndarray, rate_hz: float) -> np.ndarray:
    """Averaging times in seconds: ``tau = m / rate_hz``."""
    return np.asarray(afs, dtype=np.float64) / rate_hz


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
