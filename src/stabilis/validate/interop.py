"""The interop tier: agreement with published tables and with Stable32.

INTEROP, not correctness -- every published ADEV table traces back to
Stable32, so matching one proves interoperability and nothing more.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._specfun import chi2_ppf
from ..reference import (
    EXPECTED_NBS14_1000POINT,
    INTEROP_RTOL,
    nbs14_1000point,
)
from ..stats import confidence
from ..types import FreqSeries
from .check import ESTIMATORS, Check

__all__ = ["STABLE32_OUTPUTS", "STABLE32_PRINT_RTOL"]


def _interop() -> list[Check]:
    """Tier 2. Proves agreement with the incumbent. Not a correctness claim."""
    out = []
    fs = FreqSeries(nbs14_1000point(), rate_hz=1.0)
    for name, fn, _ in ESTIMATORS:
        expected = EXPECTED_NBS14_1000POINT.get(name)
        if not expected:
            continue
        r = fn(fs, taus="all", max_af=100, keep_coeffs=False)
        worst = max(abs(r.dev_at(af) / want - 1.0) for af, want in expected.items())
        out.append(Check(
            "interop", f"{name} vs the published NBS14 1000-point table",
            "INTEROP, not correctness. These values were produced by "
            "Stable32 and reproduced in NIST SP 1065, so agreement shows "
            "this library matches the incumbent. Correctness is the "
            "certified tier above.",
            f"{INTEROP_RTOL:.0e} relative -- the tightest the seven "
            "published significant figures support. Asserting harder would "
            "test our transcription, not our arithmetic.",
            f"worst rel {worst:.1e}",
            worst < INTEROP_RTOL,
        ))
    out += _stable32_parity()
    return out


# Where the reference session's exports live, relative to this file. Absent
# from an installed wheel, which is fine -- the checks that read it say how
# many tables they found and pass vacuously on none, so a partially
# collected session is still worth running.
STABLE32_OUTPUTS = (
    Path(__file__).resolve().parents[3] / "tests" / "data" / "stable32"
)


# Stable32 exports five significant figures, so a value we compute at full
# precision can differ from the printed one by up to half an ulp of the
# fifth digit -- 5e-5 when the mantissa is near 1. Anything inside this is
# agreement to the last digit the export actually carries; asserting
# tighter would be asserting against digits Stable32 never wrote.
STABLE32_PRINT_RTOL = 1e-4


def _stable32_parity() -> list[Check]:
    """Against Stable32's own exported tables, where a session has produced them.

    This is the claim the whole Greenhall path exists to support, and until
    a session happens there is nothing here to check. Both checks report
    the number of tables scored, so "passed on nothing" is visible rather
    than silently reassuring.
    """
    from ..io import load_phase
    from ..registry import get
    from .stable32 import load_dir

    tables = [t for t in load_dir(STABLE32_OUTPUTS / "outputs")
              if t.estimator is not None]
    if not tables:
        return [Check(
            "interop", "Stable32 output parity",
            "Direct comparison against Stable32's own exported tables -- "
            "the strongest interoperability claim available, and the one an "
            "accredited lab replacing Stable32 will check first.",
            "not applicable: no reference outputs are present.",
            "NOT RUN -- no exports in tests/data/stable32/outputs. See the "
            "run sheet there; this is item 1 of VALIDATION-DEBT.md.",
            True,
        )]

    worst_sigma, worst_bound, worst_edof = 0.0, 0.0, 0.0
    n_sigma, n_bound, files = 0, 0, set()
    # Five estimators over three tau grids means the same few curves are
    # asked for repeatedly; computing each once keeps the report's runtime
    # proportional to the data rather than to the number of tables.
    cache: dict = {}
    # One curve per (record, estimator), computed to the largest averaging
    # factor any of that pair's tables needs. The three tau grids are
    # subsets of the same `all` grid, so keying on each grid's own maximum
    # would compute the same curve three times.
    needed: dict = {}
    for t in tables:
        key = (t.input_file, t.estimator)
        needed[key] = sorted(set(needed.get(key, [])) | set(t.af.tolist()))
    for t in tables:
        files.add(t.input_file)
        key = (t.input_file, t.estimator)
        if key not in cache:
            series = load_phase(STABLE32_OUTPUTS / "inputs" / t.input_file,
                                rate_hz=t.af[0] / t.tau_s[0])
            cache[key] = get("estimator", t.estimator)(
                series, taus=needed[key])
        r = cache[key].select_afs(t.af)

        # Hadamard of a pure ramp is ~1e-23 against the Allan family's
        # 1e-13: both programs are differencing away the entire signal, so
        # what is left is arithmetic on the file's own rounding. Compared
        # anyway -- they read the same bytes -- but it is not a claim about
        # the estimator, so it cannot dominate the reported worst case.
        scale = float(np.max(cache[key].dev))
        keep = (t.sigma > 0.0) & (t.sigma >= 1e-6 * scale)
        if keep.any():
            worst_sigma = max(worst_sigma, float(np.max(
                np.abs(r.dev[keep] / t.sigma[keep] - 1.0))))
            n_sigma += int(keep.sum())

        if not t.has_bounds:
            continue
        # Their Alpha column feeds our Greenhall path, so this compares the
        # EDOF algorithm alone rather than confounding it with noise ID.
        c = confidence(r, level=t.confidence, method="greenhall",
                       alpha=t.alpha.astype(int))
        usable = np.array([
            bool(np.isfinite(t.sigma_lo[k]) and t.sigma[k] >= 1e-6 * scale)
            for k in range(t.af.size)
        ])
        if usable.any():
            worst_bound = max(
                worst_bound,
                float(np.max(np.abs(c.dev_lo[usable] / t.sigma_lo[usable] - 1.0))),
                float(np.max(np.abs(c.dev_hi[usable] / t.sigma_hi[usable] - 1.0))),
            )
            implied = _implied_edof(t.sigma_hi[usable] / t.sigma[usable],
                                    t.confidence)
            worst_edof = max(worst_edof,
                             float(np.max(np.abs(implied / c.edof[usable] - 1.0))))
            n_bound += int(usable.sum())

    where = f"{len(tables)} tables over {len(files)} record(s)"
    return [
        Check(
            "interop", "sigma columns against Stable32's own export",
            "Every estimator this library implements, against the same "
            "statistic computed by Stable32 1.62 on byte-identical input. "
            "INTEROP, not correctness -- but it is the comparison a lab "
            "migrating off Stable32 actually performs.",
            f"{STABLE32_PRINT_RTOL:.0e} relative, which is the half-ulp of "
            "the fifth significant figure Stable32 prints. Agreement inside "
            "this means agreement to the last digit the export carries.",
            f"worst rel {worst_sigma:.2e} over {n_sigma} points, {where}",
            worst_sigma < STABLE32_PRINT_RTOL,
        ),
        Check(
            "interop", "Min/Max Sigma against Stable32, and the EDOF behind them",
            "INTEROP, not correctness. Stable32's error bars, reproduced by "
            "feeding its own Alpha column into this library's Greenhall "
            "path -- so this measures the EDOF algorithm alone, not the "
            "noise identification. Inverting the printed ratio also "
            "recovers the EDOF Stable32 used, which is the quantity "
            "actually in dispute: it runs about 1.2% above ours for m up to "
            "32, where an independent implementation of the same paper "
            "returns our value to every printed digit. See "
            "theory/greenhall.md.",
            "1% on the bounds. The bounds are far less sensitive than the "
            "EDOF behind them: a 1.3% EDOF difference moves a 68.27% "
            "deviation bound by under 0.06%, because the chi-square "
            "quantile ratio flattens at large degrees of freedom.",
            f"bounds worst {worst_bound:.2e} over {n_bound} points; implied "
            f"EDOF differs by up to {worst_edof:.1%} ({where})",
            worst_bound < 0.01,
        ),
    ]


def _implied_edof(ratio_hi, level: float):
    """The EDOF Stable32 must have used, from its own printed columns.

    `sigma_hi/sigma` depends on nothing but the degrees of freedom and the
    confidence level, so it inverts. Bisection, because the ratio is
    monotone in `eta` and nothing here needs a derivative.

    **Vectorized over the whole column deliberately.** `chi2_ppf` is itself
    an iterative inversion, so a scalar bisection per row costs one root
    find per step per row -- for one report that was 25 000 of them and
    thirty seconds. Bisecting every row together makes it 32 array calls.

    32 steps takes the bracket from 2e4 down to 5e-6, far finer than the
    0.1% this is reported to.
    """
    ratio = np.asarray(ratio_hi, dtype=np.float64)
    p = (1.0 - level) / 2.0
    lo = np.full(ratio.shape, 1.0)
    hi = np.full(ratio.shape, 2.0e4)
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        too_wide = np.sqrt(mid / chi2_ppf(p, mid)) > ratio
        lo = np.where(too_wide, mid, lo)
        hi = np.where(too_wide, hi, mid)
    return 0.5 * (lo + hi)

