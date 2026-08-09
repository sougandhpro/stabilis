"""The N x N of auto and cross variances, and what averaging it buys.

`cross.py` answers "what do these two channels share". This module asks it
of every pair at once, and then does the thing the extra channels were for:
**average the off-diagonal terms.** Each is an unbiased estimate of the
same clock variance, each carries an independent draw of its own two
channels' noise, so the average rejects deeper than any single pair.

    reject = (2 / (M(M-1))) * sum_{i<j} cross_ij

How much deeper is not a rule of thumb here. It is `cross.isserlis_var`
with every pair passed in at once rather than one, which costs this module
no new mathematics at all: the covariance of two pair estimates is

    Cov(cross_ij, cross_kl) = (1/M) sum_k (1 - |k|/M)
                              [R_ik(k) R_jl(k) + R_il(k) R_jk(k)]

and the variance of the average is that summed over every ordered pair of
pairs and divided by P^2. Two consequences worth knowing before reading
the number:

* **The pairs are not independent.** (1,2) and (1,3) share channel 1, and
  the formula counts that: the shared channel's own covariance appears in
  the cross term. Treating the P = M(M-1)/2 pairs as P independent
  estimates claims a rejection that is not there -- measured on four
  channels of a clock a factor of three down, the naive `5 log10 P` gain
  of 3.9 dB is really 3.0 dB, because the shared-channel terms are of
  order `A_ii A_ij` and do not vanish until the clock is far below the
  instrument.
* **In the limit the method is for** -- instruments dominating, so
  `A_ij << A_ii` off the diagonal -- those terms do vanish and what is
  left is `A^2 / (P eta)`, giving `5 log10(P eta)` dB. That limit is the
  argument for the extra hardware; the exact sum is what you are actually
  told.

**Why a plain average and not the best linear unbiased one.** The BLUE
would weight the pairs by the inverse of that same covariance matrix, and
it is genuinely more efficient when the covariance is known. It is not
known -- it is `A`, estimated from the very data being weighted -- and
inverting a noisy estimate to weight the sample it came from is a
well-known way to get an answer that is both wrong and confident, since
the weights correlate with the noise they are supposed to suppress. The
unweighted mean has no such feedback, and its variance is computed exactly
above rather than assumed. When the channels differ enough for weighting
to matter, the honest move is to say so, which `notes` does.

See theory/cross.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..log import record, warn
from ..provenance import Provenance, new_provenance
from ..registry import get
from ..stats.edof import DEFAULT_LAG_LIMIT, eta1, eta3
from .align import DataSet, dataset
from .cross import cross_sem, implied_eta, isserlis_var, lagged_covariance

__all__ = ["CrossMatrix", "Rejection", "cross_matrix", "reject"]


@dataclass(frozen=True)
class CrossMatrix:
    """Auto and cross variances of every pair, over a tau grid.

    `var[k]` is the symmetric `M x M` matrix at `tau_s[k]`: the diagonal is
    each channel's own variance and is exact, since `mean(c*c)` on one
    channel is `mean(c**2)`. Off-diagonal entries are signed.

    Every entry is an average over **the same terms**, because the dataset
    imposed one gap pattern on all members. That is what makes the matrix
    an estimate of a single covariance matrix rather than a table of
    unrelated numbers, and it is why `reject` may combine the entries at
    all.
    """

    kind: str
    names: tuple[str, ...]
    af: np.ndarray
    tau_s: np.ndarray
    var: np.ndarray
    eta: np.ndarray
    n_terms: np.ndarray
    rate_hz: float
    scale: float
    provenance: Provenance
    # One tuple per channel, each holding that channel's coefficient array
    # at every averaging factor -- the same arrays the estimators returned,
    # referenced rather than copied. `reject` needs them: its error bar is
    # a lag sum over the coefficients, not a function of the matrix alone.
    coeffs: tuple[tuple[np.ndarray, ...], ...] = ()
    edof_method: tuple[str, ...] = ()
    span_factor: int = 2

    @property
    def auto(self) -> np.ndarray:
        """Each channel's own variance, `(n_tau, M)`. The matrix diagonal."""
        return np.diagonal(self.var, axis1=1, axis2=2)

    @property
    def correlation(self) -> np.ndarray:
        """`A_ij / sqrt(A_ii A_jj)`, the shared fraction of every pair.

        Deliberately not clipped to [-1, 1]: an excursion past one says the
        estimate is noisy or the channels' noise is correlated, which is
        the diagnostic, not an error to be tidied away.
        """
        d = np.sqrt(self.auto)
        return self.var / (d[:, :, None] * d[:, None, :])

    def pair(self, a: str, b: str) -> np.ndarray:
        """The signed cross-variance of two named channels, over tau."""
        return self.var[:, self._index(a), self._index(b)]

    def _index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"no channel named {name!r}; this matrix holds "
                f"{list(self.names)}"
            ) from None


@dataclass(frozen=True)
class Rejection:
    """One clock variance, from every pair of channels that watched it."""

    kind: str
    names: tuple[str, ...]
    af: np.ndarray
    tau_s: np.ndarray
    # Signed, for the same reason `CrossResult.var` is: this is a mean of
    # products and at long tau it goes below zero.
    var: np.ndarray
    sem: np.ndarray
    auto: np.ndarray
    n_pairs: int
    n_terms: np.ndarray
    rate_hz: float
    provenance: Provenance
    # Degrees of freedom the error bar implies, by `cross.cross_sem`
    # inverted -- a number comparable with every other eta in the library
    # rather than an internal of the lag sum that produced it.
    eta: np.ndarray = None
    notes: tuple[str, ...] = ()
    edof_method: tuple[str, ...] = ()

    @property
    def negative(self) -> np.ndarray:
        return self.var < 0.0

    @property
    def dev(self) -> np.ndarray:
        """Deviation where one exists, NaN where the variance is negative.

        Same rule as `CrossResult.dev`, for the same reason: a negative
        estimate is a statement about resolution, and `sqrt(|var|)` on a
        log axis would draw a floor that is not there.
        """
        return np.sqrt(np.where(self.var > 0.0, self.var, np.nan))

    @property
    def floor(self) -> np.ndarray:
        """Geometric mean of the channels' own variances, `(n_tau,)`.

        What one channel would have reported, near enough -- the thing the
        rejection is measured against. Geometric because that is what the
        Isserlis covariance pairs up, and because it reduces to
        `sqrt(auto_a * auto_b)` for two channels.
        """
        return np.exp(np.mean(np.log(self.auto), axis=1))

    @property
    def correlation(self) -> np.ndarray:
        return self.var / self.floor

    @property
    def rejection_db(self) -> np.ndarray:
        """`10 log10(floor / sem)`: how far under one channel this reaches.

        For instrument-dominated channels it tends to `5 log10(P eta)`,
        which is the `5 log10 M` law of cross-spectrum measurement with the
        pair count in it. Reported from the measured covariance, so it
        degrades honestly when the pairs share channels.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return 10.0 * np.log10(self.floor / self.sem)


def cross_matrix(data, estimator: str = "oadev", taus="octave",
                 max_af: int | None = None, gap_policy: str = "auto",
                 edof_method: str = "auto",
                 lag_limit: int = DEFAULT_LAG_LIMIT) -> CrossMatrix:
    """Every pair's cross-variance, and every channel's own, on one grid.

    `data` is a `DataSet` or anything `dataset()` accepts -- a mapping of
    name to series, which is aligned and given a common gap pattern first.

    Cost is `M` estimator runs and `M(M+1)/2` array products, not `M**2`
    estimator runs: the coefficients are computed once per channel and
    reused for every pair they appear in. That is the same property that
    made the pairwise cross free, one level up.
    """
    ds = data if isinstance(data, DataSet) else dataset(data)
    fn = get("estimator", estimator)
    # `keep_coeffs` is not passed, for the reason given in `cross.py`.
    results = [fn(s, taus=taus, max_af=max_af, gap_policy=gap_policy)
               for s in ds.series]
    _check_matrixable(results, ds.names, estimator)

    ref = results[0]
    m_ch, n_tau = len(results), int(ref.af.size)
    var = np.empty((n_tau, m_ch, m_ch))
    eta = np.empty((n_tau, m_ch))
    methods = []
    for k, m in enumerate(ref.af):
        used = set()
        for i, ri in enumerate(results):
            eta[k, i], how = _channel_edof(ri, k, int(m), edof_method, lag_limit)
            used.add(how)
            for j, rj in enumerate(results[: i + 1]):
                v = ref.scale * float((ri.coeffs[k] * rj.coeffs[k]).mean())
                var[k, i, j] = var[k, j, i] = v
        methods.append("eta1" if used == {"eta1"} else "eta3")

    prov = record(
        new_provenance(f"multi.cross_matrix({estimator})"),
        "multi.matrix", "cross_matrix",
        {"estimator": estimator, "channels": list(ds.names), "n": ds.n,
         "rate_hz": ds.rate_hz, "points": n_tau, "edof_method": edof_method},
        f"{m_ch}x{m_ch} over {n_tau} taus",
    )
    return CrossMatrix(
        kind=f"cross_{estimator}", names=ds.names, af=ref.af, tau_s=ref.tau_s,
        var=var, eta=eta, n_terms=ref.n_terms, rate_hz=ds.rate_hz,
        scale=ref.scale, provenance=prov,
        coeffs=tuple(r.coeffs for r in results),
        edof_method=tuple(methods), span_factor=ref.span_factor,
    )


def reject(matrix: CrossMatrix) -> Rejection:
    """Average every pair's cross-variance, and say how deep that reached.

    Every channel is assumed to be watching **the same clock**, which is
    what makes the off-diagonal entries M(M-1)/2 estimates of one quantity
    rather than a table of different ones. Channels watching different
    sources belong in `bench/`, which knows what each one saw.

    The error bar is the exact variance of the average under the Isserlis
    covariance -- see the module docstring -- so it accounts for pairs
    sharing a channel instead of pretending they are independent.
    """
    m_ch = len(matrix.names)
    if m_ch < 2:
        raise ValueError(
            f"rejection needs at least two channels, got {m_ch}")
    if not matrix.coeffs:
        raise ValueError(
            "this matrix carries no coefficient arrays, so its error bar "
            "cannot be computed. It was not built by `cross_matrix`.")
    idx_i, idx_j = np.triu_indices(m_ch, k=1)
    pairs = list(zip(idx_i.tolist(), idx_j.tolist()))
    n_pairs = len(pairs)

    n_tau = int(matrix.af.size)
    var = np.empty(n_tau)
    sem = np.empty(n_tau)
    methods = []
    for k, m in enumerate(matrix.af):
        a = matrix.var[k]
        var[k] = float(a[idx_i, idx_j].mean())
        sem[k], used = _pair_average_sem(matrix, k, int(m), var[k], pairs)
        methods.append(used)
    auto = matrix.auto
    notes = _rejection_notes(matrix, auto, n_pairs,
                             methods.count("eta3"))
    prov = record(
        matrix.provenance, "multi.matrix", "reject",
        {"channels": list(matrix.names), "pairs": n_pairs,
         "points": n_tau},
        f"{n_pairs} pairs averaged",
    )
    # Louder than the pairwise warning, because M channels of one clock is
    # usually M channels of one instrument, which is precisely the
    # arrangement in which the noise does not cancel.
    prov = warn(
        prov, "multi.matrix", "independence_assumed",
        {"channels": len(matrix.names), "pairs": n_pairs},
        "every pair is assumed to have independent channel noise. M "
        "channels of one clock are often M channels of one instrument, "
        "which share a timebase, an LO, a supply and a thermal "
        "environment -- and that common part does not cancel and does not "
        "average down. See bench.common_mode_warnings")
    for text in notes:
        prov = record(prov, "multi.matrix", "note", {}, text)
    return Rejection(
        kind=matrix.kind, names=matrix.names, af=matrix.af, tau_s=matrix.tau_s,
        var=var, sem=sem, auto=auto, n_pairs=n_pairs,
        n_terms=matrix.n_terms, rate_hz=matrix.rate_hz, provenance=prov,
        notes=notes, edof_method=tuple(methods),
        eta=np.array([implied_eta(f, f, v, e)
                      for f, v, e in zip(np.exp(np.mean(np.log(auto), axis=1)),
                                         var, sem)]),
    )


def _pair_average_sem(matrix, k: int, m: int, cross: float,
                      pairs) -> tuple[float, str]:
    """Error bar on the pair average, exactly, with the same fallback.

    `cross.isserlis_var` over all P pairs at once. The dispatch is the one
    `cross.pair_sem` documents -- too few terms, or a truncated lag sum
    that comes out negative, and it takes the conservative equal-shape
    route on `eta3` instead of reporting a confident number it does not
    have.
    """
    from ..stats.edof import MIN_TERMS_FOR_ETA1

    per_channel = [ch[k] for ch in matrix.coeffs]
    n_terms = int(per_channel[0].size)
    if n_terms >= MIN_TERMS_FOR_ETA1:
        kmax = int(min(DEFAULT_LAG_LIMIT * m, n_terms - 1))
        r = lagged_covariance(np.stack(per_channel), kmax)
        total = isserlis_var(r, n_terms, pairs)
        if total > 0.0:
            return matrix.scale * float(np.sqrt(total)), "isserlis"
    floor = float(np.exp(np.mean(np.log(matrix.auto[k]))))
    eta = eta3(n_terms, m, matrix.span_factor) * len(pairs)
    return cross_sem(floor, floor, cross, eta), "eta3"


def _rejection_notes(matrix, auto, n_pairs, fallbacks) -> tuple[str, ...]:
    """What the caller should know that the numbers alone do not say."""
    notes = []
    spread = float(np.max(auto.max(axis=1) / auto.min(axis=1)))
    if spread > 10.0:
        loud = matrix.names[int(np.argmax(auto[0]))]
        notes.append(
            f"the channels' own variances differ by up to {spread:.0f}x "
            f"({loud} is the loudest). The pairs are averaged unweighted, so "
            "the noisiest channel contributes as much as the quietest. "
            "Weighting them would need the covariance this average is "
            "estimating, taken from the same data -- see multi/matrix.py for "
            "why that is refused. Dropping the loud channel and comparing is "
            "the honest check.")
    if fallbacks:
        notes.append(
            f"at {fallbacks} of {matrix.af.size} taus the exact lag sum was "
            "not usable -- too few terms, or a truncated sum that came out "
            "negative -- and the error bar fell back to the conservative "
            "equal-shape form on eta3. Those points have a wider bar than "
            "the data may deserve, which is the right direction to be wrong.")
    if n_pairs == 1:
        notes.append(
            "two channels is one pair, so this is exactly cross_variance on "
            "them and there is no averaging gain. The gain is 5*log10(P) dB "
            "over one pair, which needs a third channel to be non-zero.")
    return tuple(notes)


def _channel_edof(result, k: int, m: int, method: str,
                  lag_limit: int) -> tuple[float, str]:
    """One channel's EDOF at one averaging factor, falling back as `edof` does."""
    from ..stats.edof import DATA_DRIVEN_METHODS, MIN_TERMS_FOR_ETA1

    if method not in DATA_DRIVEN_METHODS:
        raise ValueError(
            f"cross_matrix: edof_method must be one of {DATA_DRIVEN_METHODS}, "
            f"got {method!r}. The noise-type-based estimators are derived for "
            "a sum of squares and do not carry over to a sum of products."
        )
    n_terms = int(result.coeffs[k].size)
    if method == "eta3" or (method == "auto" and n_terms < MIN_TERMS_FOR_ETA1):
        return eta3(n_terms, m, result.span_factor), "eta3"
    return eta1(result.coeffs[k], m, lag_limit), "eta1"


def _check_matrixable(results, names, estimator: str) -> None:
    """Everything that must hold before the coefficient arrays may be mixed."""
    ref = results[0]
    if not ref.is_variance:
        raise ValueError(
            f"{estimator!r} does not return a variance, so it has no cross "
            "form. See multi.cross_variance for why."
        )
    for name, r in zip(names, results):
        if not r.coeffs:
            raise ValueError(
                f"{estimator!r} returned no coefficient arrays for {name!r}, "
                "so there is nothing to multiply."
            )
        if not np.array_equal(r.af, ref.af):
            raise ValueError(
                f"channel {name!r} produced averaging factors {list(r.af)} "
                f"against {list(ref.af)} for {names[0]!r}, despite sharing a "
                "dataset. This should be impossible after `dataset()`; report it."
            )
        bad = [int(m) for i, m in enumerate(ref.af)
               if r.coeffs[i].size != ref.coeffs[i].size]
        if bad:
            raise ValueError(
                f"channel {name!r} has different term counts at averaging "
                f"factors {bad}, so its terms are not the same terms. "
                "Alignment should have prevented this; report it."
            )
