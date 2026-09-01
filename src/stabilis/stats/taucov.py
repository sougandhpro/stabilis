"""The covariance *between* taus of one stability curve.

`stats/ci.py` gives each point of a sigma-tau curve its own error bar and
stops there. That is enough to draw the curve and wrong for anything that
fits a model to it: the points of an overlapping estimator are computed
from the same samples, so they move together, and a fit that treats them as
independent reports parameter uncertainties that are too narrow. The bias
is not in the parameters, it is in the confidence in them -- which is
exactly the number an ISO-17025 report prints.

Nothing new is needed to measure it. Every estimator returns the
coefficient array it averaged, and for jointly Gaussian coefficients
Isserlis' theorem gives the covariance of two means of squares in closed
form:

    Cov(x_i, x_j) = scale_i scale_j (2 / (M_i M_j)) sum_k n_ij(k) R_ij(k)^2

where `R_ij(k)` is the cross-covariance of the two coefficient arrays at
lag `k` and `n_ij(k)` is the number of index pairs that lag admits. For
`i = j` that is `isserlis_var`'s Bartlett weight `1 - |k|/M` written out,
and the whole expression collapses to `2 sigma^4 / eta1` -- so the
confidence interval already shipped is the diagonal of this matrix, not a
parallel construction. A test asserts that to machine precision.

**Which estimators this can be measured for.** The construction needs the
two coefficient arrays to be indexed by the same clock: coefficient `t` of
averaging factor `m` and coefficient `t` of averaging factor `m'` must
start at the same phase sample, so that `t - s` is a lag in samples. That
is true of every *overlapping* estimator here -- oadev, mdev, tdev, ohdev
and totdev all step their window one sample at a time from a common origin
(totdev's origin is offset by one sample, the same offset at every `m`, so
differences still align). It is false for a decimating estimator such as
`adev`, whose index advances by `2m` samples, and unknown for an estimator
that declares no `Shape`. Those decline to the independent diagonal and say
so, rather than returning a correlation computed against the wrong clock.

See theory/taucov.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..log import record, warn
from ..provenance import Provenance
from .core import StabilityResult, require_ordered_coeffs
from .edof import DEFAULT_LAG_LIMIT, MIN_TERMS_FOR_ETA1, edof

__all__ = ["TauCovariance", "tau_covariance", "TAUCOV_METHODS"]

# "isserlis"    -- measured from the coefficient arrays, off-diagonals and all.
# "independent" -- the diagonal only, from `stats.edof`. Not a modelling
#                  choice but a refusal: it is what comes back when the
#                  correlation cannot be measured, and `reason` says why.
TAUCOV_METHODS = ("isserlis", "independent")


@dataclass(frozen=True)
class TauCovariance:
    """Covariance of the measured variances across a curve's tau grid."""

    af: np.ndarray
    tau_s: np.ndarray
    cov: np.ndarray
    # Equivalent degrees of freedom per tau, the same numbers `confidence`
    # would use. The diagonal of `cov` is `2*var**2/edof` by construction --
    # see `tau_covariance` for why that is imposed rather than merely
    # expected.
    edof: np.ndarray
    method: str
    # Why the correlation was not measured, empty where it was. A report
    # has to be able to answer "why is this one different"; "it silently
    # fell back" is not an answer.
    reason: str
    provenance: Provenance

    @property
    def corr(self) -> np.ndarray:
        """Correlation matrix. Unit diagonal, symmetric, entries in [-1, 1]."""
        d = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(d, d)

    @property
    def condition(self) -> float:
        """Condition number of the correlation matrix.

        Large means the curve carries fewer independent numbers than it has
        points, which is the honest reading of a heavily overlapped
        estimator and is worth printing beside a fit.
        """
        return float(np.linalg.cond(self.corr))


def _overlap_counts(m_i: int, m_j: int, lags: np.ndarray) -> np.ndarray:
    """`n_ij(k)`: index pairs `(t, s)` with `t - s = k`, `t < M_i`, `s < M_j`.

        n_ij(k) = max(0, min(M_j, M_i - k) - max(0, -k))

    The unequal-length generalization of the Bartlett factor. With
    `M_i = M_j = M` it is `M - |k|`, so `n(k)/M` is the familiar
    `1 - |k|/M` and this module's formula reduces to `isserlis_var`'s.
    """
    hi = np.minimum(m_j, m_i - lags)
    lo = np.maximum(0, -lags)
    return np.maximum(0, hi - lo)


def _isserlis_matrix(coeffs, afs, scale: float, lag_limit: int) -> np.ndarray:
    """The full cross-tau covariance from the coefficient arrays.

        Cov(x_i, x_j) = (2 scale^2 / (M_i M_j)^2) sum_k n_ij(k) S_ij(k)^2
        S_ij(k)       = sum_t c_i[t + k] c_j[t]

    `S_ij` is the raw (unnormalized) cross-correlation, so this is the
    biased estimator `R_ij = S_ij / sqrt(M_i M_j)` substituted into the
    module docstring's identity. The biased normalization is the same
    choice `stats.edof._autocorrelation` makes and for the same reason: it
    shrinks the noisiest high-lag terms towards zero instead of dividing
    them by a small count and amplifying them.

    Both signs of the lag are needed and they are not redundant --
    `S_ij(-k) = S_ji(k)`, and the two arrays have different lengths -- so
    the negative half is read off the far end of the circular correlation
    rather than mirrored.

    Costs one real FFT per tau plus one inverse per pair. The pair loop is
    quadratic in the number of taus, which is bounded by the tau grid and
    is why this is not computed unless a fit asks for it.
    """
    n = len(coeffs)
    lengths = np.array([c.size for c in coeffs], dtype=np.int64)
    nfft = 1 << int(np.ceil(np.log2(2 * int(lengths.max()))))
    spec = np.array([np.fft.rfft(c, nfft) for c in coeffs])

    cov = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            kmax = min(lag_limit * max(int(afs[i]), int(afs[j])),
                       int(max(lengths[i], lengths[j])) - 1)
            full = np.fft.irfft(spec[i] * np.conj(spec[j]), nfft)
            lags = np.arange(-kmax, kmax + 1)
            s = np.concatenate((full[nfft - kmax:], full[: kmax + 1]))
            counts = _overlap_counts(int(lengths[i]), int(lengths[j]), lags)
            total = float(np.sum(counts * s**2))
            cov[i, j] = cov[j, i] = (
                2.0 * scale**2 * total / float(lengths[i] * lengths[j]) ** 2
            )
    return cov


def _can_measure(result: StabilityResult) -> str:
    """Empty if the Isserlis route applies, else why it does not."""
    require_ordered_coeffs(result, "the cross-tau covariance")
    if not result.coeffs:
        return "the estimator was called with keep_coeffs=False"
    if len(result.coeffs) != result.af.size:
        return "the result carries fewer coefficient arrays than taus"
    if result.shape is None:
        return (f"{result.kind!r} declares no Shape, so whether its "
                "coefficients are indexed by phase sample is not knowable")
    if not result.shape.overlapped:
        return (f"{result.kind!r} decimates, so its coefficient index "
                "advances by the averaging factor and lags between two taus "
                "are not lags in samples")
    if len(result.run_lengths) > 1:
        return (f"the record is in {len(result.run_lengths)} runs and the "
                "coefficients are pooled across them, so index t of one tau "
                "and index t of another are not the same instant")
    if any(result.dropped_terms):
        return (f"{sum(result.dropped_terms)} terms were dropped at a gap, so "
                "the surviving coefficients are no longer contiguous in the "
                "phase index")
    return ""


def tau_covariance(
    result: StabilityResult,
    lag_limit: int = DEFAULT_LAG_LIMIT,
    edof_method: str = "auto",
) -> TauCovariance:
    """Covariance of `result.var` across its tau grid.

    The diagonal is **imposed** to be `2*var**2/eta` with `eta` from
    `stats.edof.edof` -- the same degrees of freedom `confidence` puts on
    the plotted error bars -- and the measured Isserlis matrix supplies only
    the correlations:

        cov = D^(1/2) corr(isserlis) D^(1/2),   D = diag(2*var**2/eta)

    Where `eta1` was used and unclipped this is an exact no-op, because the
    two quantities are the same sum. Where `edof` fell back to the
    conservative `eta3` -- too few terms for a sample autocorrelation to
    mean anything -- the Isserlis diagonal is untrustworthy in exactly the
    same way, so the conservative number wins and the measured correlation
    is kept. Either way a fit's weights and its covariance cannot disagree
    about how much information a point carries, which they otherwise would
    at the long-tau end and nowhere else.

    Returns the independent diagonal, with `reason` set, whenever the
    correlation cannot be measured. It never guesses one.
    """
    if not result.is_variance:
        raise ValueError(
            f"tau_covariance: {result.kind!r} is not a variance, so "
            "'the covariance of the variances' is not a statement about it. "
            "An extreme value like MTIE has no such sampling algebra"
        )
    if lag_limit < 1:
        raise ValueError(f"lag_limit must be at least 1, got {lag_limit}")

    etas, used = [], []
    for i, m in enumerate(result.af.tolist()):
        if result.coeffs:
            eta, how = edof(result.coeffs[i], int(m), result.span_factor,
                            edof_method)
        else:
            from .edof import eta3
            eta, how = eta3(int(result.n_terms[i]), int(m),
                            result.span_factor), "eta3"
        etas.append(eta)
        used.append(how)
    eta = np.asarray(etas, dtype=np.float64)
    diag = 2.0 * result.var**2 / eta

    reason = _can_measure(result)
    if not reason:
        raw = _isserlis_matrix(result.coeffs, result.af, result.scale, lag_limit)
        if np.all(np.diag(raw) > 0.0) and np.all(np.isfinite(raw)):
            d = np.sqrt(np.diag(raw))
            corr = np.clip(raw / np.outer(d, d), -1.0, 1.0)
            cov = corr * np.outer(np.sqrt(diag), np.sqrt(diag))
        else:
            reason = ("the truncated lag sum gave a non-positive variance "
                      "for at least one tau, which means the covariance "
                      "estimate failed rather than that it is zero")
    if reason:
        cov = np.diag(diag)

    method = "independent" if reason else "isserlis"
    prov = record(
        result.provenance,
        "stats.taucov",
        "tau_covariance",
        {
            "method": method,
            "lag_limit": lag_limit,
            "edof_methods": sorted(set(used)),
            "n_taus": int(result.af.size),
            "few_terms": int(np.sum(result.n_terms < MIN_TERMS_FOR_ETA1)),
        },
        f"{result.af.size}x{result.af.size} covariance",
    )
    if reason:
        prov = warn(
            prov, "stats.taucov", "independent", {"reason": reason},
            "the taus are treated as independent, which understates the "
            "uncertainty of anything fitted to this curve",
        )
    return TauCovariance(
        af=result.af,
        tau_s=result.tau_s,
        cov=cov,
        edof=eta,
        method=method,
        reason=reason,
        provenance=prov,
    )
