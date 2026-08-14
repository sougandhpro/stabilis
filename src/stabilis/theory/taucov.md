# The covariance between taus

## What is missing from a per-point error bar

`stats/ci.py` puts a χ² interval on every point of a σ(τ) curve and stops
there. Drawing the curve, that is enough. Fitting anything to it, it is
not: the points of an overlapping estimator are computed from the same
phase samples, so they move together, and a fit that treats them as
independent gets the parameters roughly right and their **uncertainties
too narrow**.

That is the wrong error to make here. A stability curve exists so that
somebody can quote a number with an uncertainty beside it, and understating
the uncertainty is the one failure mode that survives review — the fitted
value looks fine, the residuals look fine, and the interval is a fiction.

## Isserlis, and why no new machinery is needed

Every estimator returns the coefficient array it averaged, with

    variance = scale · mean(c²)

so the covariance of two variances is a **fourth** moment of jointly
Gaussian coefficients, and Isserlis' theorem gives it in closed form:

    Cov(c_i[t] c_i[t], c_j[s] c_j[s]) = 2 R_ij(t − s)²

Summing over the index pairs,

    Cov(x_i, x_j) = scale_i scale_j (2 / (M_i M_j)) Σ_k n_ij(k) R_ij(k)²

where `R_ij(k)` is the cross-covariance of the two coefficient arrays at
lag `k`, and `n_ij(k)` counts the index pairs that lag admits:

    n_ij(k) = max(0, min(M_j, M_i − k) − max(0, −k))

With `M_i = M_j = M` that is `M − |k|`, so `n(k)/M` is the familiar
Bartlett factor `1 − |k|/M` and the expression collapses onto the one
`multi/cross.py` already uses for the cross-variance error bar. The
unequal-length count is the only new algebra on the page.

Set `i = j` and it collapses further, to `2σ⁴/η₁`. **The confidence
interval already shipped is the diagonal of this matrix**, not a parallel
construction — `tests/test_taucov.py` asserts the two code paths agree to
machine precision, which is a real cross-check because one goes through
`stats/edof.py`'s autocorrelation and the other through a pair FFT.

Both signs of the lag are needed and they are not redundant:
`S_ij(−k) = S_ji(k)`, and the two arrays have different lengths. The
negative half is read off the far end of the circular correlation rather
than mirrored.

## Truncation, and the biased normalization

Lags are cut at `4·max(m_i, m_j)`, and the raw correlation sums are divided
by `√(M_i M_j)` rather than by `n_ij(k)`. Both choices are inherited from
`eta1` and both are argued there: summing `ρ̂(k)²` over every lag adds a
spurious contribution of order 1 even when the true correlation is zero,
because each `ρ̂(k)` has variance ~1/M and there are M of them; and
dividing a high-lag sum by its own small count amplifies exactly the terms
that are mostly noise. The biased normalization shrinks them towards zero
instead.

## Which estimators this can be measured for

The construction needs coefficient `t` of averaging factor `m` and
coefficient `t` of averaging factor `m′` to **start at the same phase
sample**, so that `t − s` is a lag in samples. That holds for every
overlapping estimator here — `oadev`, `mdev`, `tdev`, `ohdev`, `totdev` all
advance their window one sample at a time from a common origin (`totdev`'s
origin is offset by one sample, the same offset at every `m`, so
differences still align).

It fails for a decimating estimator such as `adev`, whose index advances by
`2m` samples, and it is simply unknown for an estimator that declares no
`Shape`. It also fails when the coefficients were pooled across the runs of
a gapped record, or when terms were dropped at a gap, because index `t` of
one τ and index `t` of another are then not the same instant.

In every one of those cases `tau_covariance` returns the **independent
diagonal with a stated reason**, and the reason reaches provenance as a
warning. It does not guess a correlation, and it does not quietly omit one.

## The diagonal is imposed, not measured

    cov = D^(1/2) · corr(Isserlis) · D^(1/2),   D = diag(2·var²/η)

with `η` from `stats.edof.edof` — the same degrees of freedom `confidence`
puts on the plotted error bars. Where `η₁` was used and unclipped this is
an exact no-op, because the two are the same sum. Where `edof` fell back to
the conservative `η₃`, because there were too few terms for a sample
autocorrelation to mean anything, the Isserlis diagonal is untrustworthy in
exactly the same way and for exactly the same reason — so the conservative
number wins, and only the measured *correlation* is kept.

The point of doing it this way is that a fit's weights and a fit's
covariance then cannot disagree about how much information a point carries.
They otherwise would, at the long-τ end and nowhere else, which is the end
that matters.

## Validation status

**Measured against Monte-Carlo truth**, in `tests/test_taucov.py`.

Over 800 realizations of N=1024 the estimated correlation between adjacent
octave τ agrees with the sample correlation of the estimates themselves:
about 0.52 for white FM and about 0.88 for random-walk FM, matched to
within a few hundredths across the grid. The diagonal runs 0–20 % high,
which is `eta1`'s own documented accuracy showing through rather than
anything this page adds.

**The diagonal identity is exact**, to 1e-15: `Cov(x_i, x_i)` from the pair
FFT equals `isserlis_var` on the same array at the same lag truncation, and
equals `2·var²/η₁`.

Not yet validated: the correlation for `totdev`, whose reflected extension
makes the coefficients neither independent nor stationary in the way the
Isserlis derivation assumes. It is computed, because the index alignment
holds and refusing would be arbitrary, and it is named in
`VALIDATION-DEBT.md` rather than presented as measured.

## References

* Isserlis, "On a formula for the product-moment coefficient…",
  *Biometrika* 12, 1918 — the fourth-moment identity everything here rests
  on.
* Percival, "On estimation of the wavelet variance", *Biometrika* 82, 1995
  — the same Bartlett-weighted sum for the wavelet variance, and the
  argument for truncating it.
* `theory/edof.md` and `theory/cross.md` — the diagonal and the
  two-channel case of the same expression.
