"""Several records of the same clocks, and what their product tells you.

Everything in `stats/` asks what one record is worth. This package asks
what two of them share, which is a different question with a much better
answer: **the cross-variance rejects instrument noise the auto-variance
cannot.** Measure one oscillator through two independent channels,

    x_a = x_clock + n_a
    x_b = x_clock + n_b

and the auto-variance of either is `sigma^2_clock + sigma^2_n`, floored by
the instrument for good. The cross term is not:

    E[d_a * d_b] = E[d_clock^2]

because `n_a` and `n_b` are independent, so their contribution has
expectation zero and averages down with the number of terms. That is what
a cross-correlation phase noise test set does in hardware; done in
software it applies to any multi-channel capture, and neither `allantools`
nor Stable32 offers it.

Three prices, all of them stated rather than hidden:

* **The estimate is not positive.** It is a mean of products, not of
  squares, and at long tau it goes negative routinely. There is no
  deviation to report there, and no chi-square interval to draw -- so the
  signed variance is what is returned, `sem` is a symmetric standard error
  from the scatter of the products themselves, and nothing here clips or
  takes an absolute value.
* **Convergence is slow.** The instrument noise does not vanish, only its
  *estimate* averages down, as `1/sqrt(M)` in variance and hence `M^(1/4)`
  in amplitude. `rejection_db` reports how deep the measurement actually
  reached, measured rather than assumed.
* **Correlated instrument noise breaks it silently.** A shared reference,
  a shared supply, one thermal environment -- any leaves a term that does
  not cancel, and the answer is simply wrong with no symptom in the data.
  See `theory/cross.md`; the assumption is recorded in provenance because
  it cannot be checked from the records.

More than two channels is `matrix.py`: every pair at once, and then the
thing the extra channels were for -- **average the off-diagonal terms**,
since each is an unbiased estimate of the same clock carrying an
independent draw of its own two channels' noise. The pairs share channels
and are therefore not independent, which the error bar accounts for rather
than assuming away. Two channels is one pair, so `reject` on two returns
bit for bit what `cross_variance` does; they are one implementation.

Alignment is the hard part, not the arithmetic. `align.py` crops records
to their common window and refuses anything it would have to interpolate
onto a shared grid, because a silent half-sample slip yields a plausible
cross-variance that is wrong. Its `dataset()` is the M-channel form, and
unions the gaps over every member so that every entry of the matrix is an
average over the same terms -- without which the entries do not estimate
one covariance matrix and may not be combined at all. `cross.py` is the
estimator and owns the Isserlis machinery both paths share.

The three-cornered hat is *not* here. It needs to know which oscillator
and which instrument each record saw, which is metadata a bare series does
not carry -- that is `bench/`.
"""

from __future__ import annotations

from .align import Alignment, DataSet, common_support, dataset, overlap
from .cross import CrossResult, cross_sem, cross_variance
from .matrix import CrossMatrix, Rejection, cross_matrix, reject

__all__ = ["Alignment", "overlap", "common_support", "DataSet", "dataset",
           "CrossResult", "cross_variance", "cross_sem",
           "CrossMatrix", "cross_matrix", "Rejection", "reject"]
