"""Reader for Stable32's exported statistics tables.

Lives in `validate/` rather than `io/` because it reads *results*, not
data: its whole purpose is to let this library's output be diffed against
the incumbent's. Nothing in `stats/` may import it.

The export is a concatenation of blocks, one per (sigma type, tau mode)
run, each shaped like

     STATISTICS FOR FILE: drift.dat
     Phase Data Points 1 thru 4097 of 4097
     Maximum = 8.388608e-07
     ...
     Sigma Type: Overlapping Allan
     Confidence Factor = 0.683
     Deadtime T/Tau = 1.000000

      AF        Tau        #    Alpha  Min Sigma      Sigma         Max Sigma
        1   1.0000e+00     4095     2   6.9652e-14    7.0711e-14     7.1820e-14

The sigma column is named after the statistic -- `Sigma`, `Mod Sigma`,
`Hadamard Sig`, `Time Sigma` -- so rows are parsed by position and arity,
never by header text. Seven fields or it is not a data row.

**In "all tau" mode Stable32 writes `0.0000e+00` for both bounds** rather
than omitting them. Those are absences, not measurements of zero, and are
parsed as NaN so nothing can accidentally assert against them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Stable32Table", "SIGMA_TYPES", "parse", "load_dir"]

# Stable32's "Sigma Type" against this library's estimator name. Stable32
# offers types this library does not implement yet; those parse fine and
# are simply not scored, which is how a partial export stays useful.
SIGMA_TYPES = {
    "Allan": "adev",
    "Overlapping Allan": "oadev",
    "Modified Allan": "mdev",
    "Time": "tdev",
    "Overlapping Hadamard": "ohdev",
}

_FILE = re.compile(r"STATISTICS FOR FILE:\s*(\S+)")
_POINTS = re.compile(r"Data Points\s+(\d+)\s+thru\s+(\d+)\s+of\s+(\d+)")
_SIGMA_TYPE = re.compile(r"Sigma Type:\s*(.+?)\s*$")
_CONFIDENCE = re.compile(r"Confidence Factor\s*=\s*([\d.]+)")
_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_ROW = re.compile(
    rf"^\s*(\d+)\s+({_NUMBER})\s+(\d+)\s+({_NUMBER})"
    rf"\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*$"
)


@dataclass(frozen=True)
class Stable32Table:
    """One (sigma type, tau mode) run, exactly as Stable32 printed it."""

    source: str
    input_file: str
    sigma_type: str
    estimator: str | None
    confidence: float
    n_points: int
    af: np.ndarray
    tau_s: np.ndarray
    n_terms: np.ndarray
    alpha: np.ndarray
    sigma_lo: np.ndarray
    sigma: np.ndarray
    sigma_hi: np.ndarray

    @property
    def tau_mode(self) -> str:
        """Which grid this is, inferred from the averaging factors.

        Stable32 does not record the mode in the export, so it is read back
        off the sequence. `all` begins 1, 2, 3 and then *thins
        logarithmically* rather than staying consecutive -- 189 rows reach
        AF 1006, not 189 -- so it is identified by its head, not by being
        an unbroken range. `octave` is powers of two; `decade` is
        1-2-4-10-20-40.
        """
        af = self.af
        if af.size >= 3 and af[0] == 1 and af[1] == 2 and af[2] == 3:
            return "all"
        if np.all(af == 2 ** np.arange(af.size)):
            return "octave"
        return "decade"

    @property
    def has_bounds(self) -> bool:
        """False where Stable32 declined to compute intervals."""
        return bool(np.isfinite(self.sigma_lo).any())


def parse(text: str, source: str = "<string>") -> list[Stable32Table]:
    """Every table in one exported file, in the order Stable32 wrote them."""
    tables: list[Stable32Table] = []
    header = {"input_file": "", "sigma_type": "", "confidence": float("nan"),
              "n_points": 0}
    rows: list[tuple] = []

    def flush() -> None:
        if not rows:
            return
        cols = np.array(rows, dtype=np.float64)
        lo, hi = cols[:, 4].copy(), cols[:, 6].copy()
        # Stable32 writes 0.0 for "no interval computed". A bound of exactly
        # zero is never a real measurement here, so it becomes NaN rather
        # than a number something might later compare against.
        lo[lo == 0.0] = np.nan
        hi[hi == 0.0] = np.nan
        tables.append(Stable32Table(
            source=source,
            input_file=header["input_file"],
            sigma_type=header["sigma_type"],
            estimator=SIGMA_TYPES.get(header["sigma_type"]),
            confidence=header["confidence"],
            n_points=header["n_points"],
            af=cols[:, 0].astype(int),
            tau_s=cols[:, 1],
            n_terms=cols[:, 2].astype(int),
            alpha=cols[:, 3].astype(int),
            sigma_lo=lo,
            sigma=cols[:, 5],
            sigma_hi=hi,
        ))
        rows.clear()

    for line in text.splitlines():
        row = _ROW.match(line)
        if row:
            rows.append(tuple(float(g) for g in row.groups()))
            continue
        # Any non-row line after a run of rows ends the block. The header
        # fields that follow belong to the next one.
        flush()
        for pattern, key, cast in (
            (_FILE, "input_file", str),
            (_SIGMA_TYPE, "sigma_type", str),
            (_CONFIDENCE, "confidence", float),
        ):
            m = pattern.search(line)
            if m:
                header[key] = cast(m.group(1))
        m = _POINTS.search(line)
        if m:
            header["n_points"] = int(m.group(3))
    flush()
    return tables


def load_dir(path) -> list[Stable32Table]:
    """Every table in every `.txt` under `path`. Empty if there are none.

    Returning empty rather than raising is deliberate: the reference
    session is collected incrementally, and the checks that consume this
    report how much they found instead of failing when they find nothing.
    """
    out: list[Stable32Table] = []
    directory = Path(path)
    if not directory.is_dir():
        return out
    for file in sorted(directory.glob("*.txt")):
        out.extend(parse(file.read_text(encoding="utf-8", errors="replace"),
                         source=file.name))
    return out
