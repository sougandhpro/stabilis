"""Fourth moments of Gaussian coefficient arrays, exactly.

Every estimator returns the array of coefficients it averaged, so every
question about the *scatter* of an estimate is a question about the fourth
moment of that array. For jointly Gaussian coefficients Isserlis' theorem
answers it in closed form,

    Cov(c_i[t] c_j[t], c_p[s] c_q[s]) = R_ip(k) R_jq(k) + R_iq(k) R_jp(k)

with `k = t - s`, so nothing here is an approximation once the
coefficients are Gaussian -- and a second or third difference of phase is
Gaussian whenever the underlying noise is.

These two functions started life inside `multi/cross.py`, where the
cross-variance needed them. They are general machinery rather than
multi-channel machinery: the confidence interval on an auto-variance is the
`[(0, 0)]` case, and the cross-tau covariance in `taucov.py` is the case
where the two arrays come from different averaging factors of one record.
`stats/` may not import `multi/`, so they live here and `multi.cross`
re-exports them; every existing caller is unchanged.

See theory/cross.md and theory/taucov.md.
"""

from __future__ import annotations

import numpy as np

__all__ = ["lagged_covariance", "isserlis_var"]


def lagged_covariance(coeffs, kmax: int) -> np.ndarray:
    """`R[i, j, k]`, the biased sample cross-covariance at lags 0..kmax.

        R_ij(k) = (1/M) sum_t c_i[t + k] c_j[t]

    Zero-padded past `2M` so the circular correlation cannot wrap, and
    divided by `M` rather than `M - k` -- the biased normalization, for the
    reasons `stats.edof._autocorrelation` gives: it shrinks the noisiest
    high-lag estimates towards zero and keeps the sequence positive
    semi-definite. The diagonal `R[i, i]` is exactly that function's input,
    unnormalized.

    Not demeaned, and that is deliberate rather than an oversight: a second
    or third difference of phase has zero mean by construction, so
    subtracting a sample mean would remove signal and add variance.

    Computed one row at a time. The spectra of `M` channels over a long
    record are already large, and forming the full `M x M` outer product in
    the frequency domain at once turns "large" into "several times the
    record". One row is `M` spectra, which is what was already held.
    """
    c = np.atleast_2d(np.asarray(coeffs, dtype=np.float64))
    n_ch, n_terms = c.shape
    nfft = 1 << int(np.ceil(np.log2(2 * n_terms)))
    spec = np.fft.rfft(c, nfft, axis=1)
    out = np.empty((n_ch, n_ch, kmax + 1))
    for i in range(n_ch):
        out[i] = np.fft.irfft(spec[i] * np.conj(spec), nfft,
                              axis=1)[:, : kmax + 1] / n_terms
    return out


def isserlis_var(r: np.ndarray, n_terms: int, pairs) -> float:
    """Variance of the mean of the pair-averaged products, exactly.

    Given `R = lagged_covariance(...)` and a list of channel pairs, returns

        Var(mean_p mean_t c_ip[t] c_jp[t])
            = (1/(P^2 M)) sum_{p,q} sum_k (1 - |k|/M)
              [R_{ip iq}(k) R_{jp jq}(k) + R_{ip jq}(k) R_{jp iq}(k)]

    which is Isserlis' theorem with nothing approximated. Both signs of `k`
    are covered by the factor of two on `k >= 1`: the summand at `-k` for
    the ordered pair `(p, q)` equals the summand at `+k` for `(q, p)`, and
    every ordered pair is visited.

    Multiply by `scale**2` to get the variance of the *variance*; this
    function works in the coefficients' own units so it does not need to
    know the estimator.

    One pair `[(0, 1)]` is the two-channel cross-variance; `[(0, 0)]` is the
    auto case and returns `2 sigma^4 / eta1` exactly. A negative total is
    possible once the lag sum is truncated, and is returned as such -- the
    caller decides what to do with it, because "the covariance estimate
    failed" and "the covariance is zero" are different statements.
    """
    kmax = r.shape[2] - 1
    weights = np.empty(kmax + 1)
    weights[0] = 1.0
    weights[1:] = 2.0 * (1.0 - np.arange(1, kmax + 1) / n_terms)
    i = np.fromiter((p[0] for p in pairs), dtype=int, count=len(pairs))
    j = np.fromiter((p[1] for p in pairs), dtype=int, count=len(pairs))

    total = 0.0
    for q in range(i.size):
        summand = (r[i, i[q]] * r[j, j[q]] + r[i, j[q]] * r[j, i[q]])
        total += float((summand @ weights).sum())
    return total / (n_terms * i.size * i.size)
