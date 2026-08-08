"""The Stable32 export reader, and parity against whatever has been collected.

The reference session is collected by hand and incrementally, so these
tests are written to score what is present rather than to require a
particular file. When `outputs/` is empty they skip and say so; adding an
export makes them assert more without any edit here.
"""

import numpy as np
import pytest

from stabilis.io import load_phase
from stabilis.registry import get
from stabilis.stats import confidence
from stabilis.validate.report import STABLE32_OUTPUTS, STABLE32_PRINT_RTOL
from stabilis.validate.stable32 import SIGMA_TYPES, parse, load_dir

SAMPLE = """ STATISTICS FOR FILE: drift.dat
 Phase Data Points 1 thru 4097 of 4097
 Maximum = 8.388608e-07
 Sigma Type: Overlapping Allan
 Confidence Factor = 0.683
 Deadtime T/Tau = 1.000000

  AF        Tau        #    Alpha  Min Sigma      Sigma         Max Sigma
    1   1.0000e+00     4095     2   6.9652e-14    7.0711e-14     7.1820e-14
    2   2.0000e+00     4093     2   1.3930e-13    1.4142e-13     1.4364e-13

 STATISTICS FOR FILE: drift.dat
 Phase Data Points 1 thru 4097 of 4097
 Sigma Type: Modified Allan
 Confidence Factor = 0.683

  AF        Tau         #    Alpha  Min Sigma     Mod Sigma      Max Sigma
    1   1.0000e+00     4095     2   0.0000e+00    7.0711e-14     0.0000e+00
    2   2.0000e+00     4092     2   0.0000e+00    1.4142e-13     0.0000e+00
    3   3.0000e+00     4089     2   0.0000e+00    2.1213e-13     0.0000e+00
"""


def test_parses_two_blocks_with_their_headers():
    tables = parse(SAMPLE, source="sample.txt")
    assert len(tables) == 2
    a, b = tables
    assert a.input_file == "drift.dat"
    assert a.sigma_type == "Overlapping Allan"
    assert a.estimator == "oadev"
    assert a.confidence == 0.683
    assert a.n_points == 4097
    assert a.af.tolist() == [1, 2]
    assert a.n_terms.tolist() == [4095, 4093]
    assert a.alpha.tolist() == [2, 2]
    assert a.sigma[0] == pytest.approx(7.0711e-14)
    assert b.estimator == "mdev"


def test_zero_bounds_become_nan_not_zero():
    """Stable32 writes 0.0 for "no interval computed".

    Those are absences. Parsed as zero they would silently become the
    tightest possible bound and any comparison against them would pass or
    fail for the wrong reason, so they are NaN and `has_bounds` is False.
    """
    a, b = parse(SAMPLE)
    assert a.has_bounds
    assert np.isfinite(a.sigma_lo).all()
    assert not b.has_bounds
    assert np.isnan(b.sigma_lo).all()
    # The sigma column itself is a real measurement in both.
    assert np.isfinite(b.sigma).all()


def test_tau_mode_is_read_back_off_the_averaging_factors():
    a, b = parse(SAMPLE)
    assert a.tau_mode == "octave"      # 1, 2
    assert b.tau_mode == "all"         # 1, 2, 3
    assert SIGMA_TYPES["Overlapping Hadamard"] == "ohdev"


def test_a_missing_directory_is_empty_not_an_error():
    """The session is collected incrementally; nothing found is a state."""
    assert load_dir(STABLE32_OUTPUTS / "no-such-directory") == []


# ---------------------------------------------------------------------------
# Parity against the collected session
# ---------------------------------------------------------------------------


def collected():
    return [t for t in load_dir(STABLE32_OUTPUTS / "outputs")
            if t.estimator is not None]


_CACHE: dict = {}


def _ours(table):
    """Our own result for a table's record, aligned to its averaging factors.

    Memoized on (record, estimator, max af): fifteen tables per export are
    five estimators over three tau grids, so without this the same handful
    of curves is recomputed dozens of times and the suite triples in
    length for no extra coverage.
    """
    key = (table.input_file, table.estimator, int(table.af.max()))
    if key not in _CACHE:
        series = load_phase(STABLE32_OUTPUTS / "inputs" / table.input_file,
                            rate_hz=table.af[0] / table.tau_s[0])
        _CACHE[key] = get("estimator", table.estimator)(
            series, taus="all", max_af=key[2])
    return _CACHE[key].select_afs(table.af)


@pytest.mark.skipif(not collected(), reason="no Stable32 exports collected yet")
def test_every_sigma_column_matches_to_the_last_printed_digit():
    """INTEROP. Our estimators against Stable32 1.62 on identical input.

    The degenerate case is excluded and worth naming: the Hadamard variance
    of a pure frequency ramp is zero by construction, so on `drift.dat`
    both programs are left differencing the file's own decimal rounding,
    around 1e-23 against the Allan family's 1e-13. They agree there too --
    they read the same bytes -- but it is arithmetic on quantization noise,
    not evidence about an estimator.
    """
    worst, n = 0.0, 0
    for t in collected():
        r = _ours(t)
        keep = (t.sigma > 0.0) & (t.sigma >= 1e-6 * float(np.max(t.sigma)))
        if not keep.any():
            continue
        worst = max(worst, float(np.max(np.abs(r.dev[keep] / t.sigma[keep] - 1.0))))
        n += int(keep.sum())
    assert n > 0
    assert worst < STABLE32_PRINT_RTOL, (
        f"worst relative difference {worst:.2e} over {n} points exceeds the "
        f"half-ulp of Stable32's fifth printed significant figure"
    )


@pytest.mark.skipif(not collected(), reason="no Stable32 exports collected yet")
def test_the_term_count_column_agrees_exactly():
    """Stable32's `#` column against our `n_terms`, as integers.

    Cheap and unusually diagnostic: the number of terms is where an
    off-by-one in the algebraic limit or the overlap stride would show up
    first, and unlike the sigma columns it admits no tolerance at all.
    """
    checked = 0
    for t in collected():
        r = _ours(t)
        assert r.n_terms.tolist() == t.n_terms.tolist(), (
            f"{t.input_file} {t.sigma_type}: term counts differ"
        )
        checked += t.af.size
    assert checked > 0


@pytest.mark.skipif(not collected(), reason="no Stable32 exports collected yet")
def test_greenhall_bounds_agree_and_the_edof_gap_is_bounded():
    """Min/Max Sigma, with Stable32's own Alpha fed into our Greenhall path.

    Using their alpha isolates the EDOF algorithm from the noise
    identification; the two are separate claims and confounding them would
    make a failure in either unattributable.

    The bounds agree far better than the EDOF behind them does -- see
    `theory/greenhall.md` for the measured gap and what is known about it.
    The chi-square quantile ratio flattens at large degrees of freedom, so
    a percent of EDOF is worth well under a tenth of a percent of bound.
    """
    worst, n = 0.0, 0
    for t in collected():
        if not t.has_bounds:
            continue
        r = _ours(t)
        c = confidence(r, level=t.confidence, method="greenhall",
                       alpha=t.alpha.astype(int))
        scale = 1e-6 * float(np.max(t.sigma))
        for k in range(t.af.size):
            if not np.isfinite(t.sigma_lo[k]) or t.sigma[k] < scale:
                continue
            worst = max(worst, abs(c.dev_lo[k] / t.sigma_lo[k] - 1.0),
                        abs(c.dev_hi[k] / t.sigma_hi[k] - 1.0))
            n += 1
    assert n > 0
    assert worst < 0.01, f"worst bound difference {worst:.2e} over {n} points"
