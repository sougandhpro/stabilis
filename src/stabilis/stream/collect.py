"""Putting arriving samples back on the grid they were sampled on.

Everything in `stats/` assumes a uniform timebase: `x[i]` is the phase at
`epoch + i/rate_hz`, and every estimator's second difference is a
statement about three samples a fixed number of slots apart. A stream
arrives as `(t, value)` pairs and has no such guarantee, so the whole job
of this module is one rule:

**A sample that never arrived becomes a NaN in its own slot. It never
becomes the disappearance of a slot.**

Getting that wrong is the worst available failure mode, because it is
invisible. Append arriving samples to a list and a counter that dropped
fifty readings hands you a record that is fifty samples shorter and
*perfectly smooth*: every later sample has slid earlier by fifty
intervals, the timebase is wrong from the outage onward, and σ_y(τ) comes
back looking better than the oscillator is. Placing samples by timestamp
instead makes the outage a gap, which the estimators already handle
honestly (`theory/gaps.md`) and which shows up in `n_gaps` and in
provenance.

The three things `on_grid` refuses, rather than repairing:

* **A timestamp that is not near a grid slot.** Off by more than a
  quarter of a sample and the declared rate is not the rate the data was
  taken at. Interpolating onto the declared grid would be inventing data;
  the correct response is to say so and let the caller fix the rate.
* **A timestamp that goes backwards, or repeats one already placed.**
  Sorting it into position would mean taking the record's order from the
  network's scheduling rather than from the instrument.
* **A gap longer than `max_gap` slots.** An hour's outage at 1 MHz is
  three billion NaNs, and materializing them is not a service to anybody.

`parse_lines` is the text half: it turns lines into `(t, value)` pairs and
carries the one honest warning of the module -- a file with no timestamp
column *cannot* tell you a sample went missing, so its gaps are
undetectable by construction, and that fact goes into provenance rather
than being left for a reader to work out.
"""

from __future__ import annotations

from itertools import chain

import numpy as np

from ..log import record, warn
from ..provenance import new_provenance
from ..types import FreqSeries, PhaseSeries
from .online import OnlineAvar, online_push

__all__ = [
    "parse_lines",
    "grid_slot",
    "on_grid",
    "collect",
    "feed",
    "GRID_TOLERANCE",
    "MAX_GAP_SAMPLES",
]

# How far from a grid slot a timestamp may land, in samples. A quarter of
# an interval is well past any plausible logging jitter and well short of
# the half-sample at which the assignment becomes ambiguous.
GRID_TOLERANCE = 0.25

# Longest run of missing samples that will be materialized as NaN.
MAX_GAP_SAMPLES = 1 << 20

_COMMENT_CHARS = "#%"


def parse_lines(
    lines,
    column: int = 0,
    time_column: int | None = None,
    rate_hz: float | None = None,
    epoch: float = 0.0,
    delimiter: str | None = None,
    gap_value: float | None = None,
):
    """Turn text lines into `(t, value)` pairs, skipping comments and blanks.

    With `time_column` set, `t` comes from the file. Without it, `t` is
    synthesized as `epoch + i/rate_hz` from the arrival order -- which is
    the common case for a logger that just prints numbers, and which means
    **a dropped sample is undetectable**: the line that would have carried
    it is simply not there, and nothing in the remaining text says so. That
    is a property of the data, not a limitation here, but it is the reason
    `collect` records which of the two it used.

    A line that does not parse is skipped rather than fatal; a live feed
    that dies on one malformed line from a counter's startup banner is
    worse than one that ignores it.
    """
    if time_column is None and (rate_hz is None or rate_hz <= 0):
        raise ValueError(
            "parse_lines needs either time_column, or a positive rate_hz to "
            "synthesize timestamps from arrival order"
        )
    i = 0
    for line in lines:
        text = line.strip()
        if not text or text[0] in _COMMENT_CHARS:
            continue
        fields = text.split(delimiter) if delimiter else text.split()
        try:
            value = float(fields[column])
            t = float(fields[time_column]) if time_column is not None else None
        except (IndexError, ValueError):
            continue
        if gap_value is not None and value == gap_value:
            value = float("nan")
        yield (epoch + i / rate_hz if t is None else t), value
        i += 1


def grid_slot(t: float, epoch: float, rate_hz: float) -> tuple[int, float]:
    """Which grid slot a timestamp belongs in, and how far off it landed.

        slot = (t - epoch) * rate_hz
        index = round(slot);  offset = slot - index      [samples]

    Returned as a pair because the offset is the only diagnostic that
    cannot be recovered once the sample is placed: a record whose samples
    all sit 0.2 of an interval late was taken at a different rate from the
    one declared, and after placement it looks perfect.
    """
    slot = (t - epoch) * rate_hz
    index = int(np.rint(slot))
    return index, slot - index


def on_grid(
    pairs,
    rate_hz: float,
    epoch: float,
    tolerance: float = GRID_TOLERANCE,
    max_gap: int = MAX_GAP_SAMPLES,
    start_index: int = 0,
):
    """Place `(t, value)` pairs on the grid, filling missing slots with NaN.

    Yields `(index, value, offset)` for every slot from `start_index`
    onward, in order and with no slot skipped. `offset` is the arriving
    sample's fractional grid error in samples, and NaN for a slot that was
    filled rather than received.

    `start_index` is for resuming a feed: an accumulator that has already
    absorbed `k` samples must be handed slot `k` next, and a stream whose
    first timestamp lands earlier than that is a caller error rather than
    something to reconcile.
    """
    last = start_index - 1
    for t, value in pairs:
        index, offset = grid_slot(t, epoch, rate_hz)
        if abs(offset) > tolerance:
            raise ValueError(
                f"timestamp {t!r} lands {offset:+.3f} samples from grid slot "
                f"{index} at rate {rate_hz} Hz, past the {tolerance} tolerance. "
                "The declared rate is not the rate this data was taken at; "
                "placing it anyway would be interpolation."
            )
        if index <= last:
            raise ValueError(
                f"timestamp {t!r} maps to slot {index}, at or before slot "
                f"{last} which is already placed. Samples must arrive in "
                "order; reordering them here would take the record's "
                "timebase from the transport rather than the instrument."
            )
        missing = index - last - 1
        if missing > max_gap:
            raise ValueError(
                f"{missing} consecutive samples missing between slots {last} "
                f"and {index} ({missing / rate_hz:.3g} s). Past the "
                f"{max_gap}-sample limit this is an outage rather than a gap: "
                "analyse the segments either side as separate records, or "
                "raise max_gap deliberately."
            )
        for j in range(last + 1, index):
            yield j, float("nan"), float("nan")
        yield index, value, offset
        last = index


def collect(
    pairs,
    rate_hz: float,
    kind: str = "phase",
    epoch: float | None = None,
    source: str = "stream",
    tolerance: float = GRID_TOLERANCE,
    max_gap: int = MAX_GAP_SAMPLES,
    timestamps: bool = True,
):
    """Drain a stream into a `PhaseSeries` or `FreqSeries`.

    `epoch` defaults to the first timestamp seen, which is what makes the
    result's phase offsets small (see `theory/precision.md`). Pass one
    explicitly to place a stream on a grid that started earlier -- the
    leading slots come back as gaps, correctly.

    `timestamps=False` records that the timestamps were synthesized from
    arrival order rather than measured, which is what `parse_lines` does
    for a file with no time column. The samples land in the same places
    either way; what changes is the claim the record may make, so it goes
    into provenance and the gap count is reported as *detected* rather than
    complete.
    """
    if kind not in ("phase", "frequency"):
        raise ValueError(f"kind must be 'phase' or 'frequency', got {kind!r}")
    it = iter(pairs)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError(f"{source}: the stream yielded no samples") from None
    if epoch is None:
        epoch = float(first[0])

    values, offsets = [], []
    for _, value, offset in on_grid(
        chain([first], it), rate_hz, epoch, tolerance, max_gap
    ):
        values.append(value)
        offsets.append(offset)

    arr = np.asarray(values, dtype=np.float64)
    off = np.asarray(offsets, dtype=np.float64)
    n_gaps = int(np.count_nonzero(~np.isfinite(arr)))
    worst = float(np.nanmax(np.abs(off))) if np.isfinite(off).any() else 0.0

    note = record if timestamps else warn
    prov = note(
        new_provenance(source),
        "stream.collect",
        "collect",
        {
            "rate_hz": rate_hz,
            "epoch": epoch,
            "kind": kind,
            "n_slots": int(arr.size),
            "n_received": int(arr.size) - n_gaps,
            "gaps_filled": n_gaps,
            "worst_grid_offset_samples": round(worst, 6),
            "timestamps": "measured" if timestamps else "synthesized",
        },
        f"{arr.size} slots, {n_gaps} gaps"
        if timestamps
        else f"{arr.size} slots from arrival order; a dropped sample here "
        f"leaves no trace, so {n_gaps} is a floor rather than a count",
    )
    if kind == "phase":
        return PhaseSeries(x=arr, rate_hz=rate_hz, epoch=epoch, provenance=prov)
    return FreqSeries(y=arr, rate_hz=rate_hz, epoch=epoch, provenance=prov)


def feed(
    acc: OnlineAvar,
    pairs,
    epoch: float | None = None,
    tolerance: float = GRID_TOLERANCE,
    max_gap: int = MAX_GAP_SAMPLES,
) -> OnlineAvar:
    """Push a stream through the grid and into an online accumulator.

    The same placement `collect` does, without ever holding the record:
    memory stays at the accumulator's `2*max(m)+1` samples however long the
    measurement runs. Resumable -- feeding a second stream continues from
    the slot after the last one absorbed, so the two together give exactly
    what one uninterrupted feed would.

    `epoch` defaults to the accumulator's, which is the right default when
    resuming and the right one on the first call if `online_avar` was told
    the epoch. Falling back to the first timestamp would put a resumed feed
    on a fresh grid and silently discard the outage between them.
    """
    if epoch is None:
        epoch = acc.epoch
    for _, value, _ in on_grid(
        pairs, acc.rate_hz, epoch, tolerance, max_gap, start_index=acc.n_seen
    ):
        online_push(acc, value)
    return acc
