"""Writing the block Stable32 prints, to Stable32's own bytes.

The counterpart of `validate/stable32.py`, which reads this format. They
are deliberately not one module and neither imports the other: importing
`stabilis.validate` executes its `__init__`, which pulls the whole
evidence-report machinery, and an export has no business dragging that in.
`tests/test_export_stable32.py` asserts the two agree on the estimator
names instead, which is the check that would have caught a drift anyway.

**The layout is transcribed, not derived.** Every one of the 6461 data rows
in `tests/data/stable32/outputs/` is reproduced character for character by
`ROW`, and the header lines are hand-tuned strings that do *not* align to
the columns beneath them -- so they are stored verbatim rather than
generated from a width. Line endings are CRLF, because that is what
Stable32 writes and byte-comparison against its own output is the only
interop evidence available here.

This is an interchange format, not an archive format: sigma is printed to
four decimals, so a curve written here has lost eleven digits it had. Use
`export.table` to keep them.
"""

from __future__ import annotations

import numpy as np

from ..preprocess import basic_stats
from ..types import PhaseSeries

__all__ = ["SIGMA_LABELS", "ROW", "HEADERS", "ABSENT", "NEWLINE",
           "to_stable32"]

# Inverted from `validate.stable32.SIGMA_TYPES`, transcribed rather than
# imported. A test compares the two.
SIGMA_LABELS = {
    "adev": "Allan",
    "oadev": "Overlapping Allan",
    "mdev": "Modified Allan",
    "ohdev": "Overlapping Hadamard",
    "tdev": "Time",
}

# 75 characters, contiguous, no separators. Field ends at 5, 18, 27, 33,
# 46, 60, 75.
ROW = "{af:5d}{tau:13.4e}{n:9d}{alpha:6d}{lo:13.4e}{sig:14.4e}{hi:15.4e}"

# Verbatim. The middle column is named after the statistic, which is why
# `validate/stable32.py` matches rows by arity and never by header text.
_ALLAN_HEADER = (
    "  AF        Tau        #    Alpha  Min Sigma      Sigma         Max Sigma")
HEADERS = {
    "Allan": _ALLAN_HEADER,
    "Overlapping Allan": _ALLAN_HEADER,
    "Modified Allan": (
        "  AF        Tau         #    Alpha  Min Sigma     Mod Sigma"
        "      Max Sigma"),
    "Overlapping Hadamard": (
        "  AF        Tau         #    Alpha  Min Sigma   Hadamard Sig"
        "     Max Sigma"),
    "Time": (
        "  AF        Tau         #    Alpha  Min Sigma    Time Sigma"
        "      Max Sigma"),
}

# What Stable32 prints for a bound it did not compute. `validate/stable32.py`
# reads it back as NaN, deliberately: it is an absence, not a measurement of
# zero. This module is the other half of that convention.
ABSENT = 0.0

NEWLINE = "\r\n"


def to_stable32(series, result, ci=None, alpha=None, source: str = "",
                deadtime: float = 1.0, newline: str = NEWLINE) -> str:
    """One statistics block, in the format Stable32 exports.

        STATISTICS FOR FILE: <source>
        Phase Data Points 1 thru N of N
        Maximum / Minimum / Average
        Sigma Type / Confidence Factor / Deadtime T/Tau

        AF   Tau   #   Alpha   Min Sigma   Sigma   Max Sigma

    Four things it refuses rather than invents, each because the format has
    no honest encoding for the case; see `theory/export.md`.
    """
    label = SIGMA_LABELS.get(result.kind)
    if label is None:
        raise ValueError(
            f"Stable32 has no column for {result.kind!r}. It names five "
            f"statistics: {sorted(SIGMA_LABELS)}. Write "
            "`export.table.to_csv` instead, which carries any of them."
        )
    if not isinstance(series, PhaseSeries):
        raise ValueError(
            "every committed Stable32 export says 'Phase Data Points', and "
            "no export of a frequency file has been captured, so the "
            "preamble for one would have to be guessed. Convert with "
            "`to_phase` and write that."
        )
    if series.n_gaps:
        raise ValueError(
            f"{series.n_gaps} gaps: the preamble states a contiguous point "
            "range and this record has none. Select a gap-free span with "
            "`edit.select_range` and export that."
        )

    alphas = _alphas(result, ci, alpha)
    lo, hi = _bounds(result, ci)
    stats = basic_stats(series.x)
    level = 0.0 if ci is None else ci.level

    lines = [
        f" STATISTICS FOR FILE: {source or series.provenance.source}",
        f" Phase Data Points 1 thru {series.n} of {series.n}",
        " Maximum = %.6e" % stats["max"],
        " Minimum = %.6e" % stats["min"],
        " Average = %.6e" % stats["mean"],
        f" Sigma Type: {label}",
        " Confidence Factor = %.3f" % level,
        " Deadtime T/Tau = %.6f" % deadtime,
        "",
        HEADERS[label],
    ]
    dev = result.dev
    for i in range(result.af.size):
        lines.append(ROW.format(
            af=int(result.af[i]), tau=float(result.tau_s[i]),
            n=int(result.n_terms[i]), alpha=int(alphas[i]),
            lo=float(lo[i]), sig=float(dev[i]), hi=float(hi[i])))
    lines.append("")
    return newline.join(lines) + newline


def _alphas(result, ci, alpha) -> np.ndarray:
    """The noise-type column, refused rather than defaulted.

    Stable32 always prints an integer here, and there is no encoding for
    "not identified". Writing 0 would be a claim of white FM, which is a
    fabricated measurement -- and the data-driven EDOF methods leave
    `ci.alpha` as None precisely because not assuming a noise type is the
    point of them.
    """
    if alpha is not None:
        return np.full(result.af.size, int(alpha), dtype=np.int64)
    if ci is not None and ci.alpha is not None:
        return np.asarray(ci.alpha, dtype=np.int64)
    raise ValueError(
        "this format has an Alpha column and no way to say 'unknown', and "
        "writing 0 there would assert white FM. Supply an interval built "
        "with edof_method='greenhall', or pass an explicit alpha=."
    )


def _bounds(result, ci):
    """The interval columns, or the zeros Stable32 uses to mean 'absent'."""
    if ci is None:
        zeros = np.full(result.af.size, ABSENT)
        return zeros, zeros
    return ci.dev_lo, ci.dev_hi
