"""The Total Imputer: filling gaps with real data instead of invented data.

D. A. Howe and N. Schlossberger, "Characterizing Frequency Stability
Measurements Having Multiple Data Gaps," *IEEE Trans. UFFC* **69**(2),
468-472, Feb. 2022, doi:10.1109/TUFFC.2021.3137425. Open copy at
`tf.nist.gov/general/pdf/3150.pdf`.

**Why this is separate from `edit.py`.** That module *removes* data; this
one *manufactures* it. They are the two directions of the same
metrological hazard and they should not share a page: an auditor reading
`edit.py` is asking what was thrown away, and an auditor reading this is
asking what was never measured. This is the only module in the library that
invents samples, and it is named for that.

**The idea.** Interpolation, zero-padding and adjoining live segments all
fail the same way: they insert an interval containing *no stochastic
noise*, so ADEV improves precisely because data was invented, and the
methods destabilize above roughly 10% dead time. Howe and Schlossberger
fill a gap with real measured data, **reflected and inverted** -- the same
extension that defines TOTVAR, applied to an interior gap rather than to
the record ends. The filled samples are drawn from the true noise process
and carry its correlation structure, so the method has no upper dead-time
limit in principle.

**Licence position: clean-room from the paper.** The authors' reference
implementation (`github.com/nkschlos/time-series-imputation`, also Zenodo
5594587) carries **no licence file at all**, so it is all-rights-reserved
and must not be vendored, ported or copied. Nothing here was read from it.
The published description is sufficient, and where it is ambiguous the
reading taken is argued in the docstring rather than guessed -- see
`_lowpass` and `theory/impute.md`.

**Frequency samples, not phase -- and that was settled by measurement.**
The paper is loose about which series it fills: Section II motivates the
extension through TOTVAR, which is a phase construction, while its own
eq. 1 defines AVAR over `y` data and calls the record "the number of y
data". Implemented on phase, the reversal flips the sign of the frequency
at the seam, and for random-walk FM -- where `y` wanders slowly and a sign
flip is a large excursion -- `AVAR(tau0)` came out **74x high** and the
paper's own overlap criterion fell to 3 trials in 100. Implemented on the
frequency samples it is 100 in 100, and the published ranking of Fig. 4
falls out in order: random-walk FM best, then flicker FM, then white FM,
with white PM the failing case at long tau. Reversing `y` preserves a
random walk's continuity where reversing `x` destroys it. See
`theory/impute.md` for the numbers.

A gapped `PhaseSeries` is refused rather than converted, because the phase
*across* a gap is not recoverable even once the frequency samples are: a
counter that stopped and resumed carries an unknown phase step, and
integrating imputed frequencies through the gap would assert it was zero.
That is exactly the assumption `preprocess.segments` exists to avoid making
silently.

**What the paper asks of anyone using it, quoted.** "The imputation
algorithm in the link above is not meant to be used as a substitute for a
measurement deficiency. Analysts should take seriously their measurement
methodologies to eliminate gaps if at all feasible. If that is not possible
or practical, the presence of gaps and the reason for them should be
clearly stated along with the analysis results." `fill_gaps` therefore
requires a `reason`, records the imputed fraction in provenance, and
returns the mask so that the confidence interval can refuse to count
invented samples as degrees of freedom (`live_terms`).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .log import record, warn
from .preprocess import longest_segment, segments
from .types import FreqSeries, PhaseSeries

__all__ = ["FILTER_EFOLD", "fill_gaps", "live_terms"]

# The `2**3` of the paper's eq. 3, as an e-folding in cycles per sample.
# See `_lowpass` for why it is a pure number here and not a function of
# tau0.
FILTER_EFOLD = 8.0


def _lowpass(block: np.ndarray) -> np.ndarray:
    """The paper's eq. 3 endpoint-matching filter, reflection-padded.

        x_filtered = ifft( fft(x) * exp(-2**3 * |f| * tau0**2 / T) )

    **The one ambiguity in the paper, resolved deliberately.** As printed,
    `f` is "the Fourier-frequency variable" and `T` "the total length of the
    imputed data". Read with `f` a physical frequency `k/(N*tau0)` and `T` a
    duration, the exponent collapses to `-8k/N**2`, which at the Nyquist bin
    is `-4/N` -- no filtering whatsoever, and flatter the longer the record.
    Read with `f` the FFT **bin index** and `T` the **number of points**
    (equivalently tau0 = 1 sample), the exponent is `-8k/T`: a real low-pass
    with an e-folding at one eighth of the sample rate, which is what Fig. 6
    of the paper plots. The index reading is also the only dimensionally
    sane one -- an endpoint-matching filter whose behaviour changed when the
    user relabelled tau0 from seconds to milliseconds would be absurd -- and
    `tests/test_impute.py` asserts that invariance directly.

    `T` is taken as the length of the array actually transformed, so the
    cutoff stays at one eighth of the sample rate whatever the padding
    factor. Taking it as the unpadded length would make the filter three
    times narrower here than the same formula applied without padding.

    **Why the padding.** The FFT treats the block as periodic, so an
    unpadded transform manufactures a step between the first and last point
    and the filtered curve acquires a ramp that is an artefact of that step.
    Padding with a flipped copy on each side removes it. The reflection is
    half-point -- `block[::-1]` duplicates the edge sample -- which is what
    makes the padded signal continuous at the join; the match points are
    then read at the two reflection points, which are the block's own ends.
    """
    n = block.size
    padded = np.concatenate((block[::-1], block, block[::-1]))
    m = padded.size
    weight = np.exp(-FILTER_EFOLD * np.arange(m // 2 + 1) / m)
    return np.fft.irfft(np.fft.rfft(padded) * weight, m)[n:2 * n]


def _endpoint_rise(y: np.ndarray, start: int, stop: int, width: int) -> float:
    """The rise across the gap that the *smoothed* record implies.

    The difference of the two filtered points either side of the gap -- the
    red dots of the paper's Fig. 7. Matching the raw endpoints instead
    produces the artificial sawtooth of their Fig. 5: a single endpoint that
    happens to sit far from the local mean sets the slope for the whole
    inserted block, and under white PM that is most of them.

    `width` samples are filtered on each side, or as many as are live. The
    left flank is guaranteed live by the ordering in `fill_gaps`; the right
    flank is *not*, because a later gap may sit just past this one, so it is
    clipped at the next non-finite sample. One NaN inside the transform
    would come back out across the whole filtered block, silently.
    """
    left = y[start - width:start]
    right = y[stop:stop + width]
    live = np.flatnonzero(~np.isfinite(right))
    if live.size:
        right = right[:live[0]]
    return float(_lowpass(right)[0] - _lowpass(left)[-1])


def _impute_one(y: np.ndarray, start: int, stop: int) -> np.ndarray:
    """One gap's worth of filled samples, by reflect, invert and match.

        r = y[start-n : start] reversed,   b[k] = r[k] + k*s
        s = (D - (r[n-1] - r[0])) / (n - 1)

    with `D` the filtered endpoint difference. The reversal is the paper's
    "reverse the temporal indices"; the slope is its "invert", and it is
    what decorrelates the extension from its source instead of merely
    repeating it. `r[0]` is the last live sample before the gap, so the
    block joins the left flank exactly and only the slope is free -- which
    is why the paper adds a slope and not an offset.

    `n == 1` is the one case the paper's constraint does not determine: a
    single point has no endpoint difference. It is placed on the smoothed
    line, at half the rise. That is interpolation, for exactly one sample,
    and it is recorded as such.
    """
    n = stop - start
    r = y[start - n:start][::-1].copy()
    rise = _endpoint_rise(y, start, stop, n)
    if n == 1:
        return r + 0.5 * rise
    slope = (rise - (r[-1] - r[0])) / (n - 1)
    return r + slope * np.arange(n, dtype=np.float64)


def _fill_rightward(y: np.ndarray, from_index: int) -> np.ndarray:
    """Fill every gap starting at or after `from_index`, left to right.

    Order is the whole point: each gap is imputed from data that is already
    live, so the source block is never itself invented at one remove, and
    the flanks `_endpoint_rise` filters are always real measurements.
    """
    work = y.copy()
    while True:
        holes = _gaps(work)
        pending = [(a, b) for a, b in holes if a >= from_index]
        if not pending:
            return work
        start, stop = pending[0]
        n = stop - start
        if stop == work.size:
            raise ValueError(
                f"fill_gaps: a gap {n} samples wide reaches an end of the "
                "record, so it has live data on one side only and the "
                "paper's endpoint match does not apply. A gap at an end "
                "bridges nothing and carries no information about what "
                "happened there -- trim it with edit.select_range, which "
                "loses no live data and keeps the edit on the record. "
                "(Index runs from the far end when the leading gap is the "
                "one at fault: this pass works on the reversed record.)")
        if start < n:
            raise ValueError(
                f"fill_gaps: the gap at samples {start}..{stop} is {n} long "
                f"but only {start} live samples precede it. The Total "
                "Imputer extends live data into a gap, so it needs at least "
                "as much live data before the gap as the gap is wide; the "
                "paper's own caveat is that the largest gap must be small "
                "compared with the initial run. Analyse the segments "
                "separately instead (gap_policy='segment'), or select a "
                "range that starts inside a longer run. Both flanks have "
                "already been tried: the gaps are filled outward from the "
                "longest run, so a gap with too little data on one side is "
                "reached from the other, and only a gap wider than either "
                "flank gets here. This pass may be running on the reversed "
                "record, in which case the index counts from the far end.")
        work[start:stop] = _impute_one(work, start, stop)


def _gaps(a: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges of the non-finite runs -- the complement of
    `preprocess.segments`, which returns the finite ones."""
    finite = np.isfinite(a)
    if finite.all():
        return []
    return segments(np.where(finite, np.nan, 0.0))


def fill_gaps(series, reason: str) -> tuple[FreqSeries, np.ndarray]:
    """Fill every gap in a frequency record, and say which samples were made up.

    Returns `(filled, imputed)` -- the record with no gaps left, and the
    boolean mask of the samples this function invented. **Keep the mask.**
    It is what `live_terms` needs to stop the confidence interval counting
    imputed samples as degrees of freedom, and the paper measures that
    inflation at up to 6x for white noise.

    The multi-gap order is the paper's, and it is not arbitrary -- it always
    imputes outward from the most trustworthy data. Find the longest
    contiguous run; fill the gaps to its right, left to right, to the end of
    the record; then reverse the record and do the same, which reaches the
    beginning.

    `reason` is required and goes into provenance, because the paper asks
    that "the presence of gaps and the reason for them should be clearly
    stated along with the analysis results" and a default would let that
    slide.
    """
    if isinstance(series, PhaseSeries):
        raise TypeError(
            "fill_gaps: the Total Imputer fills frequency samples, and a "
            "phase record cannot simply be converted -- the phase *across* "
            "a gap is not in the data. A counter that stopped and resumed "
            "carries an unknown phase step, and integrating imputed "
            "frequencies through the gap would assert that step was zero. "
            "If the timebase is known to be continuous, take to_frequency "
            "yourself and fill that; the result is a frequency record, "
            "which is what the estimators want anyway.")
    if not isinstance(series, FreqSeries):
        raise TypeError(f"expected a FreqSeries, got {type(series).__name__}")
    if not reason:
        raise ValueError(
            "fill_gaps: a reason is required. Imputed samples are not "
            "measurements, and six months later the only defensible record "
            "of why they exist is the one written down now.")

    y = np.asarray(series.y, dtype=np.float64)
    gap_count = len(_gaps(y))
    if gap_count == 0:
        return replace(series, provenance=record(
            series.provenance, "impute", "fill_gaps",
            {"reason": reason, "gaps": 0, "imputed": 0},
            "no gaps; record returned unchanged")), np.zeros(y.size, bool)

    start, stop = longest_segment(y)
    work = _fill_rightward(y, stop)
    work = _fill_rightward(work[::-1], y.size - start)[::-1]

    imputed = ~np.isfinite(y)
    fraction = float(imputed.mean())
    prov = warn(
        series.provenance, "impute", "fill_gaps",
        {"reason": reason, "method": "howe_schlossberger_total_imputer",
         "gaps": gap_count, "imputed": int(imputed.sum()),
         "imputed_fraction": fraction,
         "longest_run": [int(start), int(stop)]},
        f"{imputed.sum()} of {y.size} samples ({fraction:.1%}) are imputed, "
        "not measured -- state the gaps and their cause with any result "
        "derived from this record, and pass the mask to live_terms so the "
        "confidence interval does not count them")
    return replace(series, y=work, provenance=prov), imputed


def live_terms(result, imputed: np.ndarray) -> np.ndarray:
    """How many of each tau's terms were computed from live data alone.

    A term at offset `t` spans `span_factor * m` phase samples, so it is
    live exactly when `imputed[t : t + span_factor*m]` holds nothing. The
    count is what `confidence(..., live_terms=...)` deflates the degrees of
    freedom by.

    **Why zero and not a discount.** An imputed sample is real measured
    data, but it is real measured data that is *already in the record* --
    the extension is a reflection of live samples the estimator has
    counted once. It carries no independent information, so a term that
    touches one contributes no degrees of freedom. The paper reports EDF
    rising by up to 6x under extension and treats that as a benefit; for a
    confidence interval it is the opposite, and an interval computed from
    the filled term count is too narrow in the optimistic direction.

    Requires the estimator to have run on the *filled* record, where the
    terms are contiguous. On a gapped record they are not -- `n_terms`
    falling short is indistinguishable from a shorter record -- and this
    would silently mis-align.
    """
    imputed = np.asarray(imputed, dtype=bool)
    # Cumulative count of imputed samples, so each window is one subtraction.
    cumulative = np.concatenate(([0], np.cumsum(imputed)))
    out = []
    for m, n in zip(result.af, result.n_terms):
        span = int(result.span_factor) * int(m)
        starts = np.arange(int(n))
        ends = np.minimum(starts + span, imputed.size)
        out.append(int(np.count_nonzero(
            cumulative[ends] - cumulative[starts] == 0)))
    return np.asarray(out, dtype=np.int64)
