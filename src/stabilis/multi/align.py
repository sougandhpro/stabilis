"""Putting two records on one grid, or refusing to.

A cross-variance is a mean of products of terms that are supposed to be
*simultaneous*. Get the alignment wrong by half a sample and nothing
raises: the products still average to something, the curve still has the
right shape, and the number is wrong. That failure has no symptom, so this
module is deliberately the strict part of the package.

**Nothing is interpolated onto a common grid.** The whole library refuses
to invent samples -- `bench.solve` will not interpolate two counters onto
shared taus, gapped data is segmented rather than filled -- and resampling
here would be the same sin at a lower level, where it would be invisible.
So alignment means one thing only: *crop to the window the records share*,
by an integer number of samples, and refuse when that is not enough.

Three refusals, each for a failure that would otherwise be silent:

* **Different rates.** Not compared for equality, which float64 timestamps
  rarely satisfy -- compared by the *slip they accumulate over the overlap*.
  Two counters nominally at 1 Hz whose rates differ by 2.4e-7 (which is
  what an MJD timebase infers to; see `theory/bench.md`) drift a quarter of
  a sample apart in a million points and two and a half in ten million. The
  question is never "are the rates equal" but "do these two arrays index
  the same instants", and only the record length answers it.
* **Epochs off the grid.** If the offset between two epochs is not a whole
  number of samples, no crop aligns them and only interpolation would.
* **No overlap at all**, which usually means someone passed the wrong pair
  of files.

`common_support` is the second half. Two records cropped to one window
still disagree about *which* samples are missing, and an estimator drops
the terms its own gaps spoil -- so two coefficient arrays from records with
different dropouts have different lengths, or worse, the same length with
different terms in it. Making the gaps common first is what lets the
products be formed index by index.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..log import record
from ..types import FreqSeries, PhaseSeries

__all__ = ["Alignment", "DataSet", "overlap", "common_support", "dataset",
           "MAX_SLIP_SAMPLES", "MAX_GRID_OFFSET_SAMPLES"]

# How far the records may drift apart over the cropped window, in samples,
# before the rates count as different. Half a sample is the point at which
# a term in one record is nearer a *neighbouring* term in the other than to
# its own partner, so the pairing itself has become wrong rather than
# merely imprecise.
MAX_SLIP_SAMPLES = 0.5

# How far an epoch may sit off the common grid, in samples. This is a
# float64 slop allowance and nothing more: an MJD-scale epoch resolves to
# about 6e-7 s, so exact equality is unattainable while 1e-3 of a sample is
# unreachable by any real misalignment. Anything larger is a genuine
# sub-sample offset, which no crop can fix.
MAX_GRID_OFFSET_SAMPLES = 1e-3


@dataclass(frozen=True)
class Alignment:
    """Records cropped to one window, and what that cost."""

    series: tuple
    epoch: float
    rate_hz: float
    n: int
    # Samples removed from the front of each input, in the input's order.
    # Zero for whichever record started latest.
    offsets: tuple[int, ...]
    # Samples removed from the end of each input.
    trimmed: tuple[int, ...]
    notes: tuple[str, ...]


def _values(series):
    """The data array of either domain, without caring which."""
    return series.x if isinstance(series, PhaseSeries) else series.y


def _with_values(series, values):
    return (replace(series, x=values) if isinstance(series, PhaseSeries)
            else replace(series, y=values))


def overlap(series) -> Alignment:
    """Crop records to the window they share. Interpolates nothing.

    Sample `j` of record `i` is at absolute time `epoch_i + j / rate`, so
    the shared window runs from `max_i epoch_i` to `min_i (epoch_i + n_i /
    rate)` and each record loses `k_i = (start - epoch_i) * rate` samples
    from the front. That number has to be an integer, and the rates have to
    agree well enough that it stays an integer to the end of the window --
    which is what `MAX_SLIP_SAMPLES` tests.

    Passing one record is legal and returns it unchanged; passing records
    that already agree is a no-op, which is why `cross_variance` can call
    this unconditionally.
    """
    items = list(series)
    if not items:
        raise ValueError("overlap needs at least one series")
    domains = {type(s).__name__ for s in items}
    if len(domains) > 1:
        raise TypeError(
            f"cannot align across domains: got {sorted(domains)}. Phase and "
            "frequency records index different things -- x[i] is an instant, "
            "y[i] is the interval after it -- and their gap handling differs "
            "too. Convert with `to_phase` first and align the results."
        )
    if not isinstance(items[0], (PhaseSeries, FreqSeries)):
        raise TypeError(f"expected PhaseSeries or FreqSeries, got {domains.pop()}")

    rate = items[0].rate_hz
    tau0 = 1.0 / rate
    start = max(s.epoch for s in items)
    end = min(s.epoch + s.n * tau0 for s in items)
    n = int(np.floor((end - start) * rate + 0.5))
    if n <= 0:
        span = ", ".join(f"[{s.epoch:.6g}, {s.epoch + s.n * tau0:.6g}) s"
                         for s in items)
        raise ValueError(
            f"these records do not overlap in time: {span}. Nothing can be "
            "cross-correlated across a gap between recordings."
        )

    notes = []
    for i, s in enumerate(items):
        slip = abs(s.rate_hz - rate) / rate * n
        if slip > MAX_SLIP_SAMPLES:
            raise ValueError(
                f"series {i} runs at {s.rate_hz!r} Hz against {rate!r} Hz for "
                f"series 0. Over the {n}-sample overlap that is a drift of "
                f"{slip:.2f} samples, past the {MAX_SLIP_SAMPLES} that keeps "
                "each term paired with its own partner rather than its "
                "neighbour. Resample deliberately, or record the rates "
                "correctly -- nothing here will interpolate for you."
            )
        if s.rate_hz != rate:
            notes.append(
                f"series {i} rate {s.rate_hz!r} Hz treated as {rate!r} Hz; "
                f"accumulated slip {slip:.3f} samples over the overlap")

    offsets, trimmed, cropped = [], [], []
    for i, s in enumerate(items):
        exact = (start - s.epoch) * rate
        k = int(np.floor(exact + 0.5))
        if abs(exact - k) > MAX_GRID_OFFSET_SAMPLES:
            raise ValueError(
                f"series {i} sits {exact - k:+.4f} samples off the common "
                f"grid (epoch {s.epoch!r} s against {start!r} s at {rate!r} "
                "Hz). A sub-sample offset cannot be removed by cropping, and "
                "this library does not interpolate. Give the series their "
                "true epochs, or resample one of them yourself and say so."
            )
        offsets.append(k)
        trimmed.append(int(s.n - k - n))
        prov = record(
            s.provenance, "multi.align", "overlap",
            {"rate_hz": rate, "epoch_s": start, "n": n,
             "dropped_front": k, "dropped_back": int(s.n - k - n)},
            f"{n} of {s.n} samples lie in the common window",
        )
        cropped.append(replace(
            _with_values(s, np.asarray(_values(s)[k:k + n])),
            epoch=start, provenance=prov))

    if any(offsets) or any(trimmed):
        notes.append(f"cropped to the common {n * tau0:.6g} s window")
    return Alignment(
        series=tuple(cropped), epoch=start, rate_hz=rate, n=n,
        offsets=tuple(offsets), trimmed=tuple(trimmed), notes=tuple(notes),
    )


def common_support(series) -> tuple:
    """The same records with a shared gap pattern: NaN wherever any is NaN.

    Two records of equal length still produce coefficient arrays of
    *unequal* length if their dropouts differ, because each estimator drops
    the terms its own gaps spoiled. Even at equal length the arrays would
    pair term `k` of one window with term `k` of a different one. So the
    union of the gaps is imposed on both before either is estimated, and
    from then on the two coefficient arrays correspond index for index --
    under `mask` and under `segment` alike, since identical gaps give
    identical runs and hence identical pooling order.

    The cost is real and is not hidden: a sample either record lost is a
    sample both lose. It is also the only honest option, since the product
    of a term with a term that was never measured is not an observation.

    Inputs must already be aligned -- call `overlap` first.
    """
    items = list(series)
    lengths = {s.n for s in items}
    if len(lengths) > 1:
        raise ValueError(
            f"common_support needs equal-length records, got {sorted(lengths)}. "
            "Call `overlap` first."
        )
    bad = ~np.all([np.isfinite(_values(s)) for s in items], axis=0)
    if not bad.any():
        return tuple(items)

    out = []
    for s in items:
        values = np.array(_values(s), dtype=float)
        values[bad] = np.nan
        added = int(bad.sum()) - int(np.count_nonzero(~np.isfinite(_values(s))))
        prov = record(
            s.provenance, "multi.align", "common_support",
            {"n": s.n, "gaps_after": int(bad.sum()), "gaps_added": added},
            f"{added} samples dropped to match the other records' gaps",
        )
        out.append(replace(_with_values(s, values), provenance=prov))
    return tuple(out)


@dataclass(frozen=True)
class DataSet:
    """Named records of one experiment, on one grid with one gap pattern.

    Data, not behaviour: it is what `dataset()` produced and nothing more.
    The point of it is that every member is *comparable to every other
    member term by term*, which is the precondition for a cross-variance
    and is tedious and error-prone to maintain by hand across six files
    from six counters.

    `alignment` is carried rather than flattened because what was cropped
    is evidence. A user who wonders why a six-channel solve used 40 % of
    the record they logged should be able to read the answer off the
    object, not re-derive it.
    """

    names: tuple[str, ...]
    series: tuple
    alignment: Alignment

    @property
    def n(self) -> int:
        return self.alignment.n

    @property
    def rate_hz(self) -> float:
        return self.alignment.rate_hz

    @property
    def epoch(self) -> float:
        return self.alignment.epoch

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str):
        """The series under `name`, or a KeyError that lists what there is."""
        try:
            return self.series[self.names.index(name)]
        except ValueError:
            raise KeyError(
                f"no series named {name!r}; this dataset holds "
                f"{list(self.names)}"
            ) from None


def dataset(named_series) -> DataSet:
    """Crop a mapping of named records to one window and one gap pattern.

    `overlap` then `common_support`, in that order, over all of them at
    once. The second step is the one worth pausing on: the gaps are unioned
    across **every** member, not pairwise, so a channel with heavy dropouts
    costs every other channel too.

    That is deliberate and it is the price of a *consistent* matrix. Doing
    it pairwise would give each entry the most data it could have, and the
    resulting table would not be an estimate of any one covariance matrix:
    entry (1,2) and entry (1,3) would be averages over different sets of
    terms, so the algebra that combines them -- the noise rejection below,
    or any solve -- would be adding numbers that do not belong to the same
    experiment. A cheaper matrix that means something beats a fuller one
    that does not.

    Insertion order is preserved, because the matrix's rows and columns are
    reported by name and a reader comparing two runs should not have to
    check whether the axes moved.
    """
    items = dict(named_series)
    if len(items) < 2:
        raise ValueError(
            f"a dataset needs at least two records, got {len(items)}. One "
            "record is a `PhaseSeries`; there is nothing here it would gain."
        )
    aligned = overlap(list(items.values()))
    kept = common_support(aligned.series)
    return DataSet(names=tuple(items), series=tuple(kept), alignment=aligned)
