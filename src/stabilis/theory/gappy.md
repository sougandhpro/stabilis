# The wavelet variance of a gappy record

`stats/gappy.py`: `gappy_wavelet_variance`, `stabilis wdev --gappy`.

D. Mondal and D. B. Percival, "Wavelet variance analysis for gappy time
series," *Ann. Inst. Stat. Math.* **62**, 943–966 (2010).

## Three ways to handle a gap

The library now has three, and they answer different questions.

* **Mask** (`wavelet_variance`): drop every MODWT coefficient whose filter
  touches a gap. Unbiased, and nothing is invented, but a level-`j` filter
  is `L_j = (2^j - 1)(L - 1) + 1` taps wide, so one gap removes `L_j`
  coefficients. Deep levels go first.
* **Fill** (`impute.fill_gaps`): Howe & Schlossberger's Total Imputer.
  Keeps every scale, but the filled samples are invented, and the error
  bar has to be told not to count them (see `impute`).
* **Pair** (this page): expand `W_t**2` into products of pairs of samples
  and average each pair over the times it was actually observed. Nothing
  is invented and nothing a gap did not touch is thrown away.

## Estimators

`y` is fractional frequency (phase input is differenced first), `d_t = 1`
where `y_t` was observed, `h_l = h_{j,l}` is the level-`j` MODWT wavelet
filter, `M_j = N - L_j + 1`, and every sum over `t` runs `L_j - 1 .. N-1`.

    beta_hat_{l,l'} = M_j / sum_t d_{t-l} d_{t-l'}

    u_j =        sum_{l,l'} h_l h_l' beta_hat_{l,l'} (1/M_j) sum_t y_{t-l} y_{t-l'} d d          (8)
    v_j = -(1/2) sum_{l,l'} h_l h_l' beta_hat_{l,l'} (1/M_j) sum_t (y_{t-l} - y_{t-l'})**2 d d  (9)

Each `beta_hat * (1/M_j) * sum_t` is a mean over the times the pair was
observed together, which is how the code reads. With no gaps every mean is
over all `M_j` times and both reduce to `(1/M_j) sum_t W_t**2`, because
`sum_l h_l = 0` makes `-(1/2) sum h h' (a - b)**2 = (sum h a)**2`.
Conditional on the gap pattern both are exactly unbiased (the paper's
Section 3); unlike the gap-free estimator, either can be negative. A
negative value is reported as it came: clipping it would bias an
estimator whose one guaranteed property is that it is unbiased.

**`v` is the default.** It is a semivariogram form, so it needs only
stationary increments of `y` -- `L > 2d` in the paper's notation, which
for random-walk FM (`d = 1`) means D4 or longer, not Haar. That is the
clock case. `u` needs `y` itself stationary and is not invariant to an
offset, so it is centred on its observed mean first (Remark 1). The paper's
Fig. 1 shows neither is uniformly more efficient: `u` wins on weakly
correlated data, `v` as the process approaches a random walk.

**A level is dropped when some pair is never co-observed** inside the
window. `beta_hat` is then undefined; the paper assumes it away, and here
the level is removed and the drop recorded. If every level goes, the call
refuses. Every-other-sample-missing is the simplest such record: no
lag-1 pair exists, and every Haar level needs one.

## Computation

The pair `(l, l + k)` at time `t` is the product `g_k[s]` at `s = t - l`,
where `g_k[s] = (y_s - y_{s-k})**2` (or `y_s y_{s-k}`). For one lag `k`
the sequence `g_k` does not depend on `l`, so one cumulative sum of `g_k`
and one of `d_s d_{s-k}` give every window sum and every count for that
lag in O(1). Folding the symmetric double sum onto `l < l'` turns the
`-1/2` of eq. 9 into `-1` (its diagonal is zero) and gives `u` a factor 2
off the diagonal. A level costs O(`L_j N`), the cost of its convolution,
and a call budgeted past `MAX_PAIR_OPS` refuses and asks for `levels=`.

The filter is read off `modwt` itself, by transforming a unit impulse, so
it is by construction the filter `wavelet_variance` uses.

## What it buys, measured

Ten percent of samples missing at random, `N = 1024`, 200 records, ensemble
mean over the exact expectation of `synth.power_law_freq` (seed 9):

| noise, filter | level | 1     | 2     | 3     | 4     | 5     |
|---------------|-------|-------|-------|-------|-------|-------|
| WFM, Haar     | pair  | 0.997 | 0.997 | 0.998 | 0.997 | 1.012 |
| RWFM, D4      | pair  | 1.000 | 0.994 | 0.993 | 0.995 | 1.006 |
| RWFM, D4      | mask: records with the level at all | 200 | 200 | 200 | 97 | 1 |

Under masking the D4 filter at level 5 is 94 taps wide, and at a 10 % gap
rate almost no 94-sample window is clean: the scale exists in one record
in 200. The pair estimator returns it in all 200.

## Error bars

None yet, and `confidence` refuses the result. Theorem 3 gives the large
sample variance as `S_v(0)/M_j`, the zero-frequency spectrum of the
per-time terms `Z_t`, which the paper estimates by a K = 5 multitaper.
That needs Slepian tapers, which numpy cannot compute without a routine
of its own, and `Z_t`, which costs O(`L_j**2`) per sample because
`beta_hat` depends on the pair and not only on its lag. Item 13 of
`VALIDATION-DEBT.md`.

## Validation status

**Identities, a published table, and measured unbiasedness.**
`tests/test_gappy.py`:

* with no gaps, both estimators equal `wavelet_variance` to 1e-11 at every
  level, for Haar, D4 and D6;
* `level_filter` equals the independent cascade
  `validate.expected.wavelet_kernel` to 1e-15, with squared norm `2^-j`;
* INTEROP: the paper's Table 1 truth row for AR(1), `phi = 0.9`,
  `0.0500 0.0689 0.1079 0.1585 0.1907 0.1710`, is reproduced from eq. 6
  and these filters to its four printed decimals;
* the paper's Section 5.1 experiment (AR(1), `p = 0.9`, Haar, levels
  1–6) at 300 realizations, and white FM (Haar) and random-walk FM (D4)
  against the generator's exact expectation at 200: every level's mean
  lies within 4 standard errors of the truth -- both estimators on the
  AR(1), `v` on the clock noise;
* offset invariance, the too-gappy refusal and the cost budget.

No row in `stabilis validate`, per the functional validation posture.
