"""Measurements that are still running, and records that are not stationary.

Everything else in this library takes a finished record and asks what it
is worth. This package covers the two questions that arrangement cannot
answer.

**"What is it doing right now?"** -- `source.py` and `collect.py` bring
samples in from a file being appended to, a socket, or an array being
replayed, and `online.py` maintains the overlapping Allan variance in
O(max τ) memory as they arrive. The acceptance test is an identity, not a
comparison: replay a record through the streaming path and it must give
back what the batch estimator gives, which is why this package was built
after the batch core was validated rather than before. Measured, the two
agree to 2 ulp -- and the residual is the batch path's pairwise summation
error, since the accumulator's sum is compensated and the coefficients are
bit-identical.

**"Was it doing the same thing all along?"** -- `davar.py` evaluates any
registered estimator over a sliding window, giving σ(τ, t) as a surface.
A warm-up, a bump, an air conditioner cycling: all of them are invisible
in a single σ(τ) curve, which averages them into the answer, and obvious
as a ridge in the surface. `davar(x, window=len(x))` returns bit for bit
what the batch call returns, because it *is* the batch call.

The rule shared by both halves, and the reason `collect.py` is longer than
the arithmetic it contains: **a sample that never arrived becomes a NaN in
its own slot, never the disappearance of a slot.** Appending arriving
samples to a list is the natural implementation and it is wrong in the
worst possible way -- a dropped reading shifts the whole timebase after it
and makes the record look *better*, with no symptom anywhere.

Nothing here is re-exported at the top level of `stabilis`. Realtime work
is a different mode of use from analysing a file, and `from stabilis.stream
import davar` says which mode you are in.
"""

from __future__ import annotations

from .collect import (
    GRID_TOLERANCE,
    MAX_GAP_SAMPLES,
    collect,
    feed,
    grid_slot,
    on_grid,
    parse_lines,
)
from .davar import MAX_WINDOWS, STEP_DIVISOR, DavarSurface, davar
from .online import (
    MAX_RING_BYTES,
    OnlineAvar,
    online_avar,
    online_push,
    online_push_many,
    online_result,
)
from .source import replay, tail_file, tcp_lines, udp_lines

__all__ = [
    # sources
    "replay",
    "tail_file",
    "tcp_lines",
    "udp_lines",
    # ingest
    "parse_lines",
    "grid_slot",
    "on_grid",
    "collect",
    "feed",
    "GRID_TOLERANCE",
    "MAX_GAP_SAMPLES",
    # online estimation
    "OnlineAvar",
    "online_avar",
    "online_push",
    "online_push_many",
    "online_result",
    "MAX_RING_BYTES",
    # dynamic stability
    "DavarSurface",
    "davar",
    "MAX_WINDOWS",
    "STEP_DIVISOR",
]
