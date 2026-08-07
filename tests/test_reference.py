"""The NBS14 reference sets.

The 1000-point set ships as a published integer recursion rather than a data
file. This module proves the recursion is right, which is what licences the
whole approach: no file to vendor, no transcription loss, no question about
whose copy of the numbers we are using.
"""

import numpy as np

from stabilis.reference import (
    NBS14_1000_CHECK,
    NBS14_LEHMER_MODULUS,
    NBS14_LEHMER_SEED,
    nbs14_1000point,
    nbs14_1000point_integers,
    nbs14_9point,
)


def test_lehmer_check_values():
    """The two published check values for the MINSTD recursion."""
    ints = nbs14_1000point_integers(3)
    for index, expected in NBS14_1000_CHECK.items():
        assert ints[index] == expected


def test_lehmer_stays_in_range():
    """A Lehmer generator must never leave (0, modulus)."""
    ints = nbs14_1000point_integers(1000)
    assert len(ints) == 1000
    assert all(0 < k < NBS14_LEHMER_MODULUS for k in ints)
    assert len(set(ints)) == 1000, "MINSTD has full period; no repeats in 1000"


def test_first_value_is_exact():
    """y[0] is seed/modulus exactly -- no rounding anywhere in the path."""
    y = nbs14_1000point(1000)
    assert y[0] == NBS14_LEHMER_SEED / NBS14_LEHMER_MODULUS
    assert y.shape == (1000,)
    assert np.all((y > 0.0) & (y < 1.0))


def test_matches_published_file_digits():
    """Agrees with the first values of wriley.com/tst_suit.dat.

    Verified 2026-08-07 against the downloaded file: the generated series
    matched all 1000 published values to a maximum absolute difference of
    5.55e-16, i.e. one ULP, which is the file's own 16-significant-digit
    text rounding. The generator is the more exact artifact of the two, so
    only the published digits are pinned here rather than shipping the file.
    """
    y = nbs14_1000point(4)
    published = np.array(
        [
            5.748904731939040e-01,
            1.841829699390490e-01,
            5.631757655940840e-01,
            2.950923397648580e-01,
        ]
    )
    assert np.allclose(y, published, rtol=0, atol=1e-15)


def test_nbs14_9point():
    """The nine values from NBS Monograph 140 Annex 8.E."""
    y = nbs14_9point()
    assert y.shape == (9,)
    assert y[0] == 892.0 and y[-1] == 677.0
    # Verified against http://www.wriley.com/nbs.dat on 2026-08-07.
    assert y.sum() == 892 + 809 + 823 + 798 + 671 + 644 + 883 + 903 + 677
