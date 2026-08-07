"""Published reference datasets, as generators rather than data files.

The NBS14 1000-point set is defined by a published integer recursion, so it
ships as four lines of arithmetic instead of a file: no transcription loss,
and no question about whose copy of the numbers we are using or under what
licence.

The expected deviations below were produced by Stable32 and reproduced in
NIST SP 1065 Table 31. They are **interop** assertions, not correctness
assertions -- matching them proves this library agrees with the incumbent,
which is commercially necessary and epistemically weak. Correctness rests
on the analytic identities in the test suite. See `references.md`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "NBS14_9POINT",
    "NBS14_LEHMER_SEED",
    "NBS14_LEHMER_MULTIPLIER",
    "NBS14_LEHMER_MODULUS",
    "NBS14_1000_CHECK",
    "nbs14_9point",
    "nbs14_1000point",
    "EXPECTED_NBS14_9POINT",
    "EXPECTED_NBS14_1000POINT",
]

# The classic nine three-digit values from NBS Monograph 140, Annex 8.E.
# Verified 2026-08-07 against http://www.wriley.com/nbs.dat
NBS14_9POINT = (892.0, 809.0, 823.0, 798.0, 671.0, 644.0, 883.0, 903.0, 677.0)

# Lehmer / MINSTD parameters for the 1000-point test suite.
NBS14_LEHMER_SEED = 1234567890
NBS14_LEHMER_MULTIPLIER = 16807
NBS14_LEHMER_MODULUS = 2147483647

# Check values. The first two entries of http://www.wriley.com/tst_suit.dat
# are 5.748904731939040e-01 and 1.841829699390490e-01, which are exactly
# seed/modulus and (multiplier*seed mod modulus)/modulus -- so the recursion
# below is confirmed against the published file, not merely asserted.
NBS14_1000_CHECK = {0: NBS14_LEHMER_SEED, 1: 395529916}


def nbs14_9point() -> np.ndarray:
    """The nine-point NBS frequency test set, tau0 = 1, dimensionless."""
    return np.asarray(NBS14_9POINT, dtype=np.float64)


def nbs14_1000point(n: int = 1000) -> np.ndarray:
    """`n` fractional-frequency points from the published Lehmer recursion.

        k[0]   = 1234567890
        k[i+1] = 16807 * k[i]  mod  2147483647
        y[i]   = k[i] / 2147483647

    Pure Python integers, so the recursion is exact regardless of platform.
    """
    modulus = NBS14_LEHMER_MODULUS
    k = NBS14_LEHMER_SEED
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = k / modulus
        k = (NBS14_LEHMER_MULTIPLIER * k) % modulus
    return out


def nbs14_1000point_integers(n: int = 1000) -> list[int]:
    """The raw integer sequence, for verifying the recursion itself."""
    k = NBS14_LEHMER_SEED
    out = []
    for _ in range(n):
        out.append(k)
        k = (NBS14_LEHMER_MULTIPLIER * k) % NBS14_LEHMER_MODULUS
    return out


# --- Expected values (Stable32-produced; INTEROP assertions only) ----------
#
# Seven significant figures as published, so assert at rtol=1e-6, no tighter.
# Consumed by the M1 estimator tests and by the validation report.
#
# !! UNVERIFIED TRANSCRIPTION -- DO NOT TRUST UNTIL CHECKED !!
# These were relayed from a secondary source, not read off the original
# NIST SP 1065 Tables 29/30/31. A separate automated extraction of the same
# tables was reported to have dropped minus signs and mangled exponents,
# which is precisely the failure mode that matters here: a wrong "certified
# value" is worse than no certified value, because it makes a correct
# implementation look broken and can be "fixed" into a wrong one.
#
# Before any of this is used as a pass criterion:
#   1. transcribe from the SP 1065 PDF directly,
#   2. cross-check against a Stable32 run we perform ourselves,
#   3. reconcile the internal identities that must hold exactly --
#      adev == oadev == mdev == totdev at AF 1, and hdev == ohdev at AF 1.
# Step 3 is free and catches most transcription slips on its own.

EXPECTED_NBS14_9POINT = {
    # estimator: {averaging_factor: deviation}
    "adev": {1: 91.22945, 2: 115.8082},
    "oadev": {1: 91.22945, 2: 85.95287},
    "mdev": {1: 91.22945, 2: 74.78849},
    "totdev": {1: 91.22945, 2: 93.90379},
    # CORRECTED 2026-08-07. The relayed table gave hdev(1) = 70.80608, which
    # contradicts its own stated identity hdev == ohdev at AF 1 (there is no
    # overlap to exploit at m=1, so the two estimators coincide). Our exact
    # rational oracle certifies 70.806073185850378, i.e. 70.80607 to seven
    # figures. The relayed value was a transcription error -- exactly the
    # failure mode flagged in the header note above.
    "hdev": {1: 70.80607, 2: 116.7980},
    "ohdev": {1: 70.80607, 2: 85.61487},
    "tdev": {1: 52.67135, 2: 86.35831},
}

EXPECTED_NBS14_1000POINT = {
    "adev": {1: 2.922319e-01, 10: 9.965736e-02, 100: 3.897804e-02},
    "oadev": {1: 2.922319e-01, 10: 9.159953e-02, 100: 3.241343e-02},
    "mdev": {1: 2.922319e-01, 10: 6.172376e-02, 100: 2.170921e-02},
    "totdev": {1: 2.922319e-01, 10: 9.134743e-02, 100: 3.406530e-02},
    "hdev": {1: 2.943883e-01, 10: 1.052754e-01, 100: 3.910860e-02},
    "ohdev": {1: 2.943883e-01, 10: 9.581083e-02, 100: 3.237638e-02},
    "tdev": {1: 1.687202e-01, 10: 3.563623e-01, 100: 1.253382e00},
}
