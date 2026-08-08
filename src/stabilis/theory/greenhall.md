# Greenhall EDOF

## What this is for

Stable32 prints seven columns, and three of them — `Alpha`, `Min Sigma`,
`Max Sigma` — come from this algorithm. It computes the equivalent degrees
of freedom of a stability variance estimator from a *model*: you tell it
the power law, the differencing order, the averaging factor and the record
length, and it returns `edf` with no reference to the data at all.

That is the opposite of η₁ (see `edof`), which reads the degrees of freedom
out of the estimator's own coefficient autocovariance and never asks what
the noise is. **η₁ stays the default**, because not needing a noise type is
the point of this library. Greenhall is here so that a lab replacing
Stable32 can diff its old reports against its new ones line for line, and
so the validation report can show, measured, where the assumption costs
something.

`auto` never selects Greenhall. Assuming a power law has to be an explicit
act, recorded in provenance along with which α was assumed and how
confident the identification was.

## Source

Greenhall & Riley, *Uncertainty of Stability Variances Based on Finite
Differences*, 35th PTTI, 2003 — the preprint dated 3 October 2003, read in
full. Equation numbers below are the paper's. Sections 5 (Examples) and 6
(Conclusions) of that preprint are empty headings, so the paper supplies no
worked values to check against; see **Validation status**.

## The general form

Every variance in the family is the mean square of a `d`-th difference of a
(possibly averaged) phase. Writing `w(t)` for a continuous power-law
process and `Δ` for the backward difference,

    x(t) = Δ_{τ0} w(t)                                             (2)
    z(t) = Δ_ε^d w(t),     ε = τ0  (unmodified)  or  τ  (modified) (3)
    σ² = E z²(t),   V = (1/M) Σ_{n=1}^{M} z²(nδ)                   (4)

with stride `δ = τ0` for the overlapped estimator and `δ = τ` for the
nonoverlapped one. From `cov[z²(t), z²(u)] = 2 s_z²(u−t)`,

    var V = (2/M²) Σ_{n1,n2} s_z²((n2−n1)δ)                        (5)

and since `edf = 2σ⁴ / var V`, that collapses to

    1/edf = (1/M) · [ 1 + (2/s_z²(0)) · Σ_{j=1}^{M−1} (1 − j/M)·s_z²(jδ) ]
                                                                   (6)

Everything in the algorithm is a way of evaluating (6) accurately and
cheaply. Because the result is invariant to time scaling, the algorithm
sets `τ = 1`, `τ0 = 1/m`.

## Inputs (§4.1)

| paper | code | meaning |
|---|---|---|
| α | `alpha` | frequency noise exponent, `S_y(f) ~ f^α`, α ∈ {2,…,−4} |
| d | `d` | order of phase difference: 1 first-difference, 2 Allan, 3 Hadamard |
| m | `m` | averaging factor τ/τ0 |
| F | from `modified` | filter factor: `F = 1` modified, `F = m` unmodified |
| S | from `overlapped` | stride factor: `S = 1` nonoverlapped, `S = m` overlapped |
| N | `n` | number of phase samples at spacing τ0 |

**α is the frequency exponent**, so it is the same convention as
`noiseid.ALPHA_NAMES` and no conversion happens anywhere. α = 2 is white
PM, α = 0 is white FM, α = −2 is random-walk FM.

Restriction: **α + 2d > 1**. Below that the `d`-th differences are not
stationary and (6) does not exist. The implementation raises rather than
clamping, because a returned number would be meaningless.

## Generalized autocovariance (eq. 7)

| α | 2 | 1 | 0 | −1 | −2 | −3 | −4 |
|---|---|---|---|---|---|---|---|
| `s_w(t, α)` | −\|t\| | t² ln\|t\| | \|t\|³ | −t⁴ ln\|t\| | −\|t\|⁵ | t⁶ ln\|t\| | \|t\|⁷ |

The `ln|t|` entries evaluate to **0 at t = 0** (the polynomial factor wins).

## The two remaining kernels

    s_x(t, F, α) = F² Δ_{1/F} Δ_{−1/F} s_w(t, α)
                 = F² [ 2 s_w(t,α) − s_w(t−1/F, α) − s_w(t+1/F, α) ]   (8)
    s_x(t, ∞, α) = s_w(t, α+2),        −4 ≤ α ≤ 0

    s_z(t, F, α, d) = (Δ_1 Δ_{−1})^d s_x(t, F, α)                      (9)

expanded, with `F` and `α` suppressed:

    d=1:  2 s_x(t) − s_x(t−1) − s_x(t+1)
    d=2:  6 s_x(t) − 4 s_x(t±1) + s_x(t±2)
    d=3:  20 s_x(t) − 15 s_x(t±1) + 6 s_x(t±2) − s_x(t±3)

The `F → ∞` limit is `−s_w''(t)`: it treats `x(t)` as the mean-square
derivative of `w(t)`, and it exists only for α ≤ 0. It is what keeps the
unmodified large-`m` case from evaluating a difference of large `s_w`
values that cancel to something small — the roundoff hazard the paper warns
about in Appendix A.

## The truncated sum (eq. 10)

    BasicSum(J, M, S, F, α, d)
        = s_z²(0)
        + (1 − J/M)·s_z²(J/S)
        + 2 Σ_{j=1}^{J−1} (1 − j/M)·s_z²(j/S)

The isolated `j = J` term is not a typo and not a boundary fudge: it makes
the sum a **trapezoidal** approximation to the integral the large-`r` cases
evaluate in closed form, so the two branches agree where they meet.

## Term count and truncation point (eqs. 11–12)

    L = m·(1/F + d)
    M = 1 + floor( S·(N − L) / m ),      requires N ≥ L
    J = min( M, (d+1)·S )

`M` is the number of terms the estimator actually forms, and it agrees
exactly with the coefficient arrays this library builds — `N−2m` for
`oadev`, `N−3m+1` for `mdev`, `N−3m` for `ohdev`,
`1+floor((N−1−2m)/m)` for `adev`. There is a test asserting that for every
estimator and every averaging factor, which is a sharp check that the
`(d, modified, overlapped)` triple carried on each result is the right one.

`s_z(t)` decays fast beyond `t = d`, so lags past `(d+1)S` contribute
nothing but roundoff. Truncating there is the *simplified* algorithm:

    1/edf = BasicSum(J, M, S, F, α, d) / ( s_z²(0, F, α, d) · M )    (13)

## The full algorithm (§4.6)

`J_max = 100` (the paper's suggestion), `r = M/S`. Four cases, and the same
three-branch shape inside each: sum directly when the sum is short, use a
precomputed integral when the record is long, and otherwise re-sum with the
stride rescaled so the term count lands on `J_max`.

**Case 1 — modified variances, `F = 1`, any α.** Also covers unmodified
when `m = 1`, where the two coincide.

    J ≤ J_max               1/edf = BasicSum(J, M, S, 1, α, d) / (s_z²(0,1,α,d)·M)
    J > J_max, r ≥ d+1      1/edf = (a0 − a1/r)/r,        Table 1
    otherwise, m' = J_max/r 1/edf = BasicSum(J_max, J_max, m', 1, α, d)
                                    / (s_z²(0,1,α,d)·J_max)

**Case 2 — unmodified, α ≤ 0.** Identical in shape, but the kernel switches
to the `F → ∞` limit when the direct evaluation would lose precision:

    J ≤ J_max               F' = m if m(d+1) ≤ J_max else ∞
                            1/edf = BasicSum(J, M, S, F', α, d) / (s_z²(0,F',α,d)·M)
    J > J_max, r ≥ d+1      1/edf = (a0 − a1/r)/r,        Table 2
    otherwise, m' = J_max/r 1/edf = BasicSum(J_max, J_max, m', ∞, α, d)
                                    / (s_z²(0,∞,α,d)·J_max)

**Case 3 — unmodified, α = 1 (flicker PM).** As `S = m → ∞` the kernel
develops logarithmic singularities, so `s_z(0)` is replaced by its
asymptotic form `b0 + b1·ln m` in the two long-record branches:

    J ≤ J_max               1/edf = BasicSum(J, M, S, m, 1, d) / (s_z²(0,m,1,d)·M)
    J > J_max, r ≥ d+1      1/edf = (a0 − a1/r) / ((b0 + b1 ln m)²·r)
    otherwise, m' = J_max/r 1/edf = BasicSum(J_max, J_max, m', m', 1, d)
                                    / ((b0 + b1 ln m)²·J_max)

The paper notes `m` must stay below ~10⁶ in the first branch or roundoff
takes over, and declines to add a further subcase on the grounds that
flicker PM is not the dominant noise at large `m`. That judgement is
inherited, not re-litigated — but the implementation warns rather than
staying quiet about it.

**Case 4 — unmodified, α = 2 (white PM).** Exact and closed-form, because
the phase samples are genuinely uncorrelated when `w(t)` is Wiener. With
`K = ceil(r)`:

    K ≤ d:   1/edf = (1/M)·[ 1 + 2·C(2d,d)^{−2} Σ_{k=1}^{K−1} (1−k/r)·C(2d,d−k)² ]
    K > d:   1/edf = (a0 − a1/r)/M,   a0 = C(4d,2d)/C(2d,d)²,  a1 = d/2

Note the divisor is `M`, not `r` — Case 4 is the one case where those
differ, and swapping them produces a plausible-looking answer that is wrong
by the stride factor.

## Tables 1–3

Large-`r` coefficients, from `a0 = 2∫₀^{d+1} s_z²(t)dt` and
`a1 = 2∫₀^{d+1} s_z²(t)·t dt`, evaluated in advance by the authors.
Transcribed verbatim; blanks are blank in the source, and a blank means the
combination violates `α + 2d > 1`.

**Table 1 — modified variances**

| α | d=1 a0 | d=1 a1 | d=2 a0 | d=2 a1 | d=3 a0 | d=3 a1 |
|---|---|---|---|---|---|---|
| 2 | 2/3 | 1/3 | 7/9 | 1/2 | 22/25 | 2/3 |
| 1 | 0.840 | 0.345 | 0.997 | 0.616 | 1.141 | 0.843 |
| 0 | 1.079 | 0.368 | 1.033 | 0.607 | 1.184 | 0.848 |
| −1 | | | 1.048 | 0.534 | 1.180 | 0.816 |
| −2 | | | 1.302 | 0.535 | 1.175 | 0.777 |
| −3 | | | | | 1.194 | 0.703 |
| −4 | | | | | 1.489 | 0.702 |

**Table 2 — unmodified variances**

| α | d=1 a0 | d=1 a1 | d=2 a0 | d=2 a1 | d=3 a0 | d=3 a1 |
|---|---|---|---|---|---|---|
| 2 | 3/2 | 1/2 | 35/18 | 1 | 231/100 | 3/2 |
| 1 | 78.6 | 25.2 | 790. | 410. | 9950. | 6520. |
| 0 | 2/3 | 1/6 | 2/3 | 1/3 | 7/9 | 1/2 |
| −1 | | | 0.852 | 0.375 | 0.997 | 0.617 |
| −2 | | | 1.079 | 0.368 | 1.033 | 0.607 |
| −3 | | | | | 1.053 | 0.553 |
| −4 | | | | | 1.302 | 0.535 |

The α = 2 row is exactly the Case 4 closed form: `C(4d,2d)/C(2d,d)²` gives
6/4, 70/36, 924/400 = 3/2, 35/18, 231/100, and `d/2` gives 1/2, 1, 3/2.
A test asserts that identity, which validates the transcription of the
whole row against arithmetic rather than against eyesight.

**Table 3 — logarithmic denominator, unmodified, α = 1**

| | d=1 | d=2 | d=3 |
|---|---|---|---|
| b0 | 6 | 15.23 | 47.8 |
| b1 | 4 | 12 | 40 |

The caption in the source reads "FLFM (α = 1)", but §4.6.3 and the input
table both name α = 1 as **FLPM**. Recorded, not corrected — the numbers
are used as printed.

## Where α comes from

Two routes, and the difference matters.

**Given.** `--alpha`, or `alpha=` on `confidence`, as one value or one per
tau. **To reproduce a Stable32 run exactly, paste in its `Alpha` column.**
That is the parity path and it assumes nothing beyond what the user asked
for.

**Identified.** Pass the series instead and the noise is identified per
averaging factor by `noiseid`, which is what Stable32's `Alpha` column
itself is. Noise type is a function of tau, not of a record, so this runs
per averaging factor rather than once.

The identified route **declines where Stable32 answers.** Past
`MIN_POINTS_FOR_NOISE_ID = 30` averaged frequency points the lag-1
autocorrelation has a standard error of about `1/√n ≈ 0.18`, already
comparable to the 0.25 threshold `noiseid` uses to decide whether to
difference again — so the answer is close to a coin toss, and a coin toss
fed to this algorithm produces a confident error bar out of nothing. The
symptom is not hypothetical: before the guard, white FM at N = 1000 and
m = 128 was identified as white PM and the EDOF jumped from 43 to 420,
which is an error bar roughly three times too narrow at the longest tau on
the plot.

Declined points fall back to η₁, and the reason is carried per tau on
`Confidence.declined`, printed under the table, and recorded in
provenance. The same happens when the identified law is too steep for the
estimator to have a stationary difference of — α = −3 with `oadev` — where
the reason names the restriction so the user knows to reach for `ohdev`.

## Records with gaps

The algorithm assumes one contiguous record of `N` samples. `phase_runs`
may hand back several, and the estimator pools their coefficients (see
`gaps`). Two policies, both available:

* **`pool`** (default). Evaluate the algorithm on each run at that run's own
  `N_j`, then combine by Satterthwaite,

      η = (Σ M_j)² / Σ (M_j² / η_j)

  which is the standard degrees-of-freedom of a weighted mean of
  independent chi-square terms, with the same `M_j` weights the estimator
  itself used to pool. It reduces to `η_1` exactly when there is one run.

* **`fallback`**. Refuse on a segmented record and let the caller drop to
  η₁, which needs no such assumption. Strictly faithful to the paper.

Neither is free: pooling assumes the runs are independent, which is an
approximation for a slowly-varying process observed either side of a short
gap. The choice is recorded in provenance.

Under `gap_policy="mask"` there is one run but interior gaps have removed
terms, so the surviving count is below eq. (11)'s `M`. The implementation
uses the **surviving** count and records the shortfall; using the nominal
`M` would quote error bars for data that was not there.

## What the model costs at m = 1

Worth knowing before quoting these intervals. The paper's process model is
`x = Δ_τ0 w` for a `w` with an exact power law at *all* frequencies, so its
"white FM" is flat at low frequency but rolls off by a factor of six
approaching Nyquist. At `m = 1` the estimator sits right on that corner and
the difference from ideal power-law noise is large. Both directions are
measured and tested:

| noise at m = 1 | exact edf for ideal noise | Greenhall | effect |
|---|---|---|---|
| white FM | `M/1.5` (second differences are MA(1), ρ₁ = −½) | **17 % high** | error bars too narrow |
| random-walk FM | `M` (second differences are iid) | **24 % low** | error bars too wide |

By `m = 4` it is gone. It is the model showing through, not a
transcription error — η₁ gets both cases right from the data, without
being told what the noise is, which is the argument for the default.

## Validation status

**Validated by construction and by measurement, not by a reference table.**
The paper prints no worked values, and Stable32's `Min Sigma`/`Max Sigma`
columns need a reference session that has not happened yet. So, in
`tests/test_greenhall.py`:

* **Term count.** Eq. (11) reproduces the exact coefficient-array length of
  every estimator at every averaging factor on an `all` grid — five
  estimators, 250 averaging factors, zero mismatches. That is the sharpest
  check that each estimator's `Shape(d, modified, overlapped)` is right.
* **Kernels.** `s_x(t,1,0)` and `s_z(t,1,α,2)` for α = 0 and α = −2 are
  asserted against sequences worked out by hand from eq. (7) — exact
  integers `12, −4, −2, 0` and `132, 52, 2, 0`, asserted at equality.
* **Table 2, α = 2** matches the Case 4 closed form exactly, and every
  blank cell in Tables 1 and 2 is exactly a combination violating
  `α + 2d > 1`. Both transcriptions checked by arithmetic.
* **Closed form for white FM.** For α = 0, d = 2, overlapped, the
  long-standing published approximation
  `edf = [3(N−1)/(2m) − 2(N−2)/N]·4m²/(4m²+5)` agrees within a few percent
  for m ≥ 2. Labelled INTEROP.
* **Monte Carlo.** `2σ⁴/Var(σ̂²)` measured over 400 realizations at each
  power law: within ±30 % for m ≥ 2, against a truth whose own relative
  standard error is 7 %. The same measurement the report runs for η₁, so
  the two are directly comparable.
* **Differential.** Against `allantools.edf_greenhall`: **18 627
  parameter combinations agree to 1 part in 10⁹**, over four cases, three
  tables and every power law. Two independent implementations of one
  published algorithm; its source was not read and nothing was ported.

  Four combinations disagree, all at `r = d+1` exactly — the boundary
  between the summed branch and the tabulated-integral branch. §4.6 writes
  the condition as `r ≥ d+1` and Appendix A repeats it as `M ≥ (d+1)S`, so
  this implementation takes the tabulated branch. The two branches differ
  by under 0.1 % for α ≤ 0 and by 2.4 % for flicker PM, where the
  `b0 + b1 ln m` asymptotic is weakest. That is a discontinuity in the
  published algorithm, not in this transcription of it.

  That implementation also leaves two branches of the paper unwritten —
  the `K ≤ d` case of Case 4, and Table 2 below α = −3, so random-run FM
  at d = 3 is unreachable there. Both are implemented here and covered by
  the structural tests.

## Against Stable32 itself

The reference session has begun, and `tests/data/stable32/outputs/` now
carries real Stable32 1.62 exports. What they show:

* **The `#` (term count) column agrees exactly**, as integers, on every
  row of every table. No tolerance is involved, and it is where an
  off-by-one in the algebraic limit or the overlap stride would surface
  first.
* **The sigma columns agree to the last digit Stable32 prints** — worst
  relative difference `4.9e-5` over 3229 points, against the `5e-5`
  half-ulp of its fifth significant figure. That covers `adev`, `oadev`,
  `mdev`, `tdev` and `ohdev` across the `all`, `octave` and `decade` grids.
* **`Min Sigma` and `Max Sigma` agree to 5.5e-4** when Stable32's own
  `Alpha` column is fed into this library's Greenhall path — which is the
  comparison that isolates the EDOF algorithm from the noise
  identification.

### The EDOF behind those bounds differs, and by a definite amount

`sigma_hi/sigma` depends on nothing but the degrees of freedom and the
confidence level, so Stable32's *implied* EDOF inverts out of its own
printed columns. Doing that:

| averaging factor | their EDOF / ours (median) |
|---|---|
| 1 – 32 | **1.012** |
| 40 – 1024 | **1.001** |

A consistent ~1.2 % excess at small `m`, falling to ~0.1 % by `m ≈ 40`.

**This is a difference in Stable32, not a defect here, as far as two
independent implementations can establish.** On the same inputs
`allantools.edf_greenhall` returns *our* value to every printed digit —
3204.987 against Stable32's 3245.622 at α = 0, d = 2, m = 1, N = 4097 —
and the simplified procedure of eq. (13) does not explain the gap either:
it coincides with the full procedure exactly over the whole range where
the 1.2 % excess appears, and diverges the *other* way above it.

So two separate readings of the paper agree with each other and disagree
with the program. What Stable32 does differently is not known, and is
recorded rather than guessed at.

**What it costs in practice: under 0.06 % on a 68.27 % deviation bound.**
The chi-square quantile ratio flattens at large degrees of freedom, so a
percent of EDOF is worth well under a tenth of a percent of error bar. A
lab diffing an old report against a new one will see the bounds move in
the fourth decimal place, and this is the reason.

**Still owed:** the session covers synthetic records only so far. Real
oscillator data and the public NBS suites are still to come — see the run
sheet at `tests/data/stable32/README.md`.

## References

* Greenhall & Riley, "Uncertainty of Stability Variances Based on Finite
  Differences", 35th PTTI, 2003 — the algorithm, transcribed above.
* Riley, NIST SP 1065, §5 — the white-FM closed form used as a check.
* Satterthwaite, *Biometrics Bulletin* 2, 1946 — the pooling rule.
