"""Windows, and the noise-equivalent bandwidth that makes a PSD a density.

A windowed periodogram is only a spectral *density* if the window's power
is divided out. Get that wrong and every level is off by a constant that no
later step can detect -- which is the same failure mode as an unrecorded
carrier frequency, and is why the bandwidth is computed from the window
rather than taken from a table.

Only windows whose coefficients are exact are here. Hann is
`0.5 - 0.5*cos`, written out; the rectangular window is ones. Flat-top
windows for amplitude-accurate spur measurement have transcribed
coefficients, and transcribing digits from memory is exactly what
`CLAUDE.md` forbids -- they arrive with a fetched source, beside the spur
detector that needs them.
"""

from __future__ import annotations

import numpy as np

__all__ = ["WINDOWS", "window", "enbw_bins", "coherent_gain"]

WINDOWS = ("rectangular", "hann")


def window(name: str, n: int) -> np.ndarray:
    """`n` window coefficients, periodic rather than symmetric.

        hann[k] = 0.5 - 0.5*cos(2*pi*k/n),   k = 0 .. n-1

    Periodic (dividing by `n`, not `n-1`) because these are used inside a
    DFT, where the periodic form is the one that makes the window's own
    transform land exactly on the frequency bins. The symmetric form is for
    filter design and leaks in this application.
    """
    if n < 2:
        raise ValueError(f"a window needs at least 2 points, got {n}")
    if name == "rectangular":
        return np.ones(n, dtype=np.float64)
    if name == "hann":
        k = np.arange(n, dtype=np.float64)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)
    raise ValueError(f"unknown window {name!r}; expected one of {WINDOWS}")


def enbw_bins(w: np.ndarray) -> float:
    """Noise-equivalent bandwidth in DFT bins.

        enbw = n * sum(w**2) / sum(w)**2

    The width of the ideal rectangular filter that would pass the same
    white-noise power as this window's main lobe. Exactly 1 for the
    rectangular window and exactly 1.5 for Hann, both of which are tests.

    It is what a *spur* amplitude must be divided by to become a density,
    and what a density must be multiplied by to become the power in a line
    -- so mixing the two up turns a discrete tone into a noise level.
    """
    w = np.asarray(w, dtype=np.float64)
    return float(w.size * np.sum(w * w) / np.sum(w) ** 2)


def coherent_gain(w: np.ndarray) -> float:
    """`sum(w) / n` -- the amplitude a sinusoid keeps after windowing.

    Separate from `enbw_bins` because they answer different questions:
    coherent gain is about a tone, noise-equivalent bandwidth is about
    noise, and a flat-top window trades one against the other.
    """
    w = np.asarray(w, dtype=np.float64)
    return float(np.sum(w) / w.size)
