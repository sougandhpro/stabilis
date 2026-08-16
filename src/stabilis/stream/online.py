"""The overlapping Allan variance, maintained one sample at a time.

A live measurement has no length, so nothing here may touch the whole
record. What it keeps instead is one running sum of squared coefficients
per averaging factor, plus the last `2*max(m)+1` phase samples, and that
is enough: every term the batch estimator forms is formed here exactly
once, when its last sample arrives.

    d[i] = x[i+2m] - 2*x[i+m] + x[i]

is a function of three samples, so at sample index `k` the term ending
there is `d = (x[k] - 2*x[k-m]) + x[k-2m]` -- written in that association
order deliberately, because it is the order `numpy` uses to evaluate the
batch expression in `stats/classical.py`, and matching it makes every
coefficient bit-identical rather than merely close.

**What is therefore *not* bit-identical is the sum**, and that is the only
difference between the two paths. `np.mean` sums pairwise; an accumulator
cannot, because it never sees more than one term at a time. So the sum
here is Neumaier-compensated, which makes it the *more* accurate of the
two -- the residual disagreement is the batch estimator's pairwise error,
not ours. `theory/stream.md` carries the measured figure.

**This is the one mutable container in the library.** Returning a fresh
frozen dataclass per sample would allocate once per sample, which at a
counter's output rate is the whole cost of the thing. It has no methods,
and every function that touches it is a plain function below, so the
"frozen dataclasses for data, functions for behaviour" rule still holds in
the only way that matters: nothing here is polymorphic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..log import record
from ..provenance import Provenance, new_provenance
from ..stats.classical import SCALE_ALLAN, SHAPE_OAVAR
from ..stats.core import StabilityResult
from ..tau import taus_from_afs

__all__ = [
    "OnlineAvar",
    "online_avar",
    "online_push",
    "online_push_many",
    "online_result",
    "MAX_RING_BYTES",
]

# Ceiling on the phase history one accumulator may hold. The ring is
# `2*max(af)+1` float64, so this is a direct bound on the largest averaging
# factor a stream may track -- about 16 million at the default. It is a
# memory bound rather than a preference, and it is stated because the
# alternative is a process killed by the allocator, which is not an error
# the code gets to report.
MAX_RING_BYTES = 1 << 28


@dataclass
class OnlineAvar:
    """Running state for the overlapping Allan variance of one channel.

    Mutable, and mutated in place by `online_push`. See the module
    docstring for why. Nothing outside this module should assign to a
    field; read them freely.
    """

    rate_hz: float
    tau0: float
    af: np.ndarray
    epoch: float
    provenance: Provenance
    ring: np.ndarray
    # Neumaier running sum of c**2 per averaging factor, and its
    # compensation term. The variance uses `total + comp`; neither half is
    # meaningful alone.
    total: np.ndarray
    comp: np.ndarray
    count: np.ndarray
    n_seen: int = 0
    n_gaps: int = 0


def online_avar(
    rate_hz: float,
    af,
    epoch: float = 0.0,
    source: str = "stream",
) -> OnlineAvar:
    """Start an accumulator for the averaging factors `af`.

    The averaging factors are **explicit and required**. A batch estimator
    derives its tau grid from the record length; a stream has no length, so
    deriving one would mean guessing how long the measurement will run and
    then either wasting memory or silently capping the longest tau. Naming
    them is the honest form of that decision, and `tau.suggest(n)` will
    name a grid for an expected `n` if the caller wants one.
    """
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    afs = np.unique(np.asarray(af, dtype=np.int64))
    if afs.size == 0:
        raise ValueError("no averaging factors given")
    if afs[0] < 1:
        raise ValueError(f"averaging factors must be >= 1, got {afs[0]}")

    length = 2 * int(afs[-1]) + 1
    if length * 8 > MAX_RING_BYTES:
        raise MemoryError(
            f"averaging factor {afs[-1]} needs a phase history of {length} "
            f"samples ({length * 8 / 1e9:.1f} GB), past the "
            f"{MAX_RING_BYTES / 1e9:.1f} GB limit. A stream can only hold "
            "2*m+1 samples of history, so this is the ceiling on the "
            "longest tau a live measurement can report."
        )

    prov = record(
        new_provenance(source),
        "stream.online",
        "start",
        {
            "rate_hz": rate_hz,
            "af": afs.tolist(),
            "history_samples": length,
            "epoch": epoch,
        },
        f"{afs.size} averaging factors, af {afs[0]}..{afs[-1]}",
    )
    return OnlineAvar(
        rate_hz=float(rate_hz),
        tau0=1.0 / float(rate_hz),
        af=afs,
        epoch=float(epoch),
        provenance=prov,
        ring=np.zeros(length, dtype=np.float64),
        total=np.zeros(afs.size, dtype=np.float64),
        comp=np.zeros(afs.size, dtype=np.float64),
        count=np.zeros(afs.size, dtype=np.int64),
    )


def online_push(acc: OnlineAvar, value: float) -> OnlineAvar:
    """Absorb one phase sample. Returns the same accumulator, for chaining.

        k = n_seen
        for each m with k >= 2m:
            d = (x[k] - 2*x[k-m]) + x[k-2m]
            c = d / (m * tau0)
            total[m] += c**2  (compensated);  count[m] += 1

    A NaN sample is stored like any other, so it spoils exactly the terms
    whose window contains it and no others -- the same arithmetic the batch
    `gap_policy="mask"` path does, reached by the same route rather than by
    a second rule.
    """
    ring = acc.ring
    length = ring.size
    k = acc.n_seen
    ring[k % length] = value
    acc.n_seen = k + 1
    if not np.isfinite(value):
        acc.n_gaps += 1

    live = np.flatnonzero(acc.af <= k // 2)
    if live.size == 0:
        return acc
    m = acc.af[live]
    d = (value - 2.0 * ring[(k - m) % length]) + ring[(k - 2 * m) % length]
    sq = (d / (m * acc.tau0)) ** 2

    keep = np.isfinite(sq)
    if not keep.all():
        live, sq = live[keep], sq[keep]
    _neumaier_add(acc.total, acc.comp, live, sq)
    acc.count[live] += 1
    return acc


def online_push_many(acc: OnlineAvar, values) -> OnlineAvar:
    """Absorb a chunk of samples, in order. Identical to pushing one by one.

    Chunking is an ingest convenience -- a socket read returns whatever
    arrived -- and must never change the answer, so this really is the loop
    and not a second implementation of it.
    """
    for v in np.asarray(values, dtype=np.float64).reshape(-1):
        online_push(acc, float(v))
    return acc


def _neumaier_add(total: np.ndarray, comp: np.ndarray, idx: np.ndarray, value) -> None:
    """Neumaier-compensated `total[idx] += value`, vectorized over `idx`.

        s = t + v
        comp += (t - s) + v   if |t| >= |v|  else  (v - s) + t

    The branch is what makes this Neumaier rather than Kahan: it recovers
    the lost bits when the *increment* is the larger of the two, which is
    exactly the case Kahan drops. Each index appears at most once per call,
    so there is no aliasing to worry about in the fancy-index write.
    """
    t = total[idx]
    s = t + value
    comp[idx] += np.where(np.abs(t) >= np.abs(value), (t - s) + value, (value - s) + t)
    total[idx] = s


def online_result(acc: OnlineAvar) -> StabilityResult:
    """The overlapping Allan deviation implied by the stream so far.

        var[m] = scale * ( (total[m] + comp[m]) / count[m] ),  scale = 1/2

    Cheap and repeatable: call it after every sample if you want a live
    trace. Averaging factors that have not yet seen a complete term are
    absent from the result rather than present and zero.

    `coeffs` is empty, and permanently so -- retaining the coefficient
    arrays is the one thing an O(m) accumulator cannot do. `confidence()`
    therefore refuses this result, which is correct: the data-driven EDOF
    it would want is a function of the coefficients it does not have. Use
    `davar` over the collected record, or collect and re-run in batch, when
    an interval is needed.
    """
    kept = np.flatnonzero(acc.count > 0)
    if kept.size == 0:
        raise ValueError(
            f"no complete term yet: {acc.n_seen} samples seen, and the "
            f"smallest averaging factor {int(acc.af[0])} needs "
            f"{2 * int(acc.af[0]) + 1}"
        )
    af = acc.af[kept]
    count = acc.count[kept]
    # Written as scale * (sum / n), not (scale * sum) / n, to match the
    # batch `scale * float(np.mean(c**2))` rounding for rounding.
    var = SCALE_ALLAN * ((acc.total[kept] + acc.comp[kept]) / count)

    nominal = np.maximum(acc.n_seen - 2 * af, 0)
    prov = record(
        acc.provenance,
        "stream.online",
        "result",
        {
            "n_seen": acc.n_seen,
            "gaps": acc.n_gaps,
            "af": af.tolist(),
            "summation": "neumaier",
        },
        f"{af.size} points, af {int(af[0])}..{int(af[-1])}",
    )
    return StabilityResult(
        kind="oadev",
        af=af,
        tau_s=taus_from_afs(af, acc.rate_hz),
        var=var,
        n_terms=count,
        rate_hz=acc.rate_hz,
        provenance=prov,
        scale=SCALE_ALLAN,
        coeffs=(),
        span_factor=2,
        shape=SHAPE_OAVAR,
        run_lengths=(acc.n_seen,),
        run_terms=tuple(np.asarray([c], dtype=np.int64) for c in count),
        dropped_terms=tuple(int(n - c) for n, c in zip(nominal, count)),
    )
