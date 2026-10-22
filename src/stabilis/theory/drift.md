# Drift removal and decimation

`preprocess.py`: `fit_drift`, `remove_drift`, `decimate`.

## Drift is a choice, so it is recorded

A linear frequency drift `D` adds `D**2 * tau**2 / 2` to the Allan
variance (see `avar`). Subtracting a fitted trend removes that term, but
the answer then depends on which trend was fitted, so the model and the
fitted `D` go into provenance on every call. The alternative that needs no
choice is the Hadamard variance (see `hvar`), which annihilates a phase
quadratic exactly; a test holds `ohdev` unchanged by `remove_drift` to
1e-8 relative, which is that statement made executable.

## Models

`D` is always the fractional-frequency drift in 1/s; `drift_per_day` is
`D * 86400`. Time is `t[i] = i * tau0` from the first sample.

Phase:

    quadratic   x ~ a + b*t + (D/2)*t**2                  least squares
    diff2       D = mean(x[i+2] - 2*x[i+1] + x[i]) / tau0**2
                trend = (D/2)*t**2
    linear      x ~ a + b*t                               D = 0

Frequency:

    linear      y ~ a + D*t                               least squares
    bisection   D = (ybar2 - ybar1) / (tbar2 - tbar1)     two halves
                trend = ybar1 + D*(t - tbar1)
    mean        y ~ a                                     D = 0

`diff2` and `bisection` are the Stable32 alternatives. The mean of second
differences telescopes to `(y_last - y_first) / ((N-2) * tau0)`: it uses
the two end frequencies only, so it is unbiased but inefficient on white
noise. Bisection uses all the data with a coarser lever arm.

The least-squares fits are done in scaled time `u = (i - c)/h`, with
`c = (N-1)/2` and `h = max(c, 1)`. A Vandermonde matrix in raw sample index
has a condition number growing like `N**degree`; in `u` it is O(1), and
`D = 2*p2/(h*tau0)**2` (phase) or `p1/(h*tau0)` (frequency) converts back.

**Gaps.** NaN samples are left out of the fit and of every mean, and each
finite sample keeps its true time. The record is never closed up, so a gap
cannot re-time the trend. The subtracted trend has no gaps; the output has
exactly the input's.

**No standard error on D.** Least squares would supply one, but it assumes
white residuals, and a clock's residuals are not white. Under random-walk
FM the white-noise formula understates the scatter of `D` by a factor that
grows with the record length. A drift uncertainty belongs to a fit of the
variance curve (`fitting/`, the `drift` model), not to this regression.

## What detrending costs

A fitted trend absorbs noise as well as drift: the quadratic has three free
parameters, and its lowest-frequency content looks like part of a random
walk. The variance at long tau then comes out low. Measured: mean `oavar`
after quadratic removal over mean `oavar` before, `N = 1024`, 400
realisations of `synth.power_law_phase`, seed 0:

| m      | 1     | 16    | 64    | 128   | 256   |
|--------|-------|-------|-------|-------|-------|
| WPM    | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WFM    | 1.000 | 1.000 | 1.001 | 0.994 | 0.81  |
| FFM    | 1.000 | 1.000 | 1.000 | 0.966 | 0.70  |
| RWFM   | 0.999 | 0.982 | 0.907 | 0.772 | 0.40  |

So the bias is confined to `m > N/8` except under random-walk FM, where it
reaches into the middle of the curve. Removing drift and then quoting an
error bar at `m = N/4` from a chi-square that knows nothing of the fit is
wrong by more than the error bar. Prefer `ohdev` when drift is present and
the long-tau points matter; remove drift when an Allan curve is required
and say which points are inside the biased region.

## Decimation

    phase:      x_out[k] = x[k*m]
    frequency:  y_out[k] = mean(y[k*m : (k+1)*m])

These are one operation. The phase of the block-averaged frequency is the
decimated phase, because `sum(y[km:(k+1)m]) * tau0 = x[(k+1)m] - x[km]`;
the test holds the two routes equal to 1e-12 of the peak phase. And since
the non-overlapping Allan variance is computed from `x[::m]`, `adev` of
the decimated record at averaging factor `k` is the same set of terms as
`adev` of the original at `k*m` -- tested to 1e-12 relative.

The overlapping estimators lose information under decimation: `oadev` at
`m*tau0` on the original uses every starting point, on the decimated record
only every m-th. Decimate for speed or to match another tool's `tau0`, not
before computing an overlapping statistic you could have had in full.

Three edge rules, each because the alternative is a sample at a different
tau than the record claims:

* a trailing partial block is dropped, never averaged short;
* a frequency block containing a gap becomes a gap;
* a phase gap survives only if it falls on a kept sample -- phase at the
  kept instants is all the decimated record claims to know.

No anti-alias filter is applied. The Allan family is defined on the phase
at sample instants and needs none; a spectrum of decimated phase does, and
white PM above the new Nyquist frequency folds back into it.

## Validation status

**Identities and exact trends only.** `tests/test_preprocess.py`:

* each fit recovers `D` from a noiseless trend to 1e-9 relative, with and
  without gaps;
* `ohdev` is unchanged by quadratic removal, to 1e-8 relative;
* least-squares `D` on white FM lands inside 5 standard errors of the truth;
* frequency averaging equals phase decimation, and `adev` of a decimated
  record reproduces `adev` of the original at `k*m`.

Not validated against Stable32's drift functions; its `diff2` and
`bisection` are implemented from the definitions above, not from its
output. The bias table above is a measurement of this library, not a
reference value. No row in `stabilis validate`, per the functional
validation posture.
