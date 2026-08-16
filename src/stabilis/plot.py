"""Plots. matplotlib is optional and imported inside the functions.

**Nothing here is imported at package import time**, and `plot.py` itself
imports matplotlib only inside function bodies, so `import stabilis` stays
numpy-only. `tests/test_core_deps.py` asserts that in a subprocess.

Every function takes an optional `ax` and returns it, so plots compose by
calling them in turn on one axis rather than by any option-passing scheme:

    ax = plot.sigma_tau(a, label="A vs B")
    plot.sigma_tau(b, label="B vs C", ax=ax)
    plot.slope_guides(ax)

No styling opinions beyond what the data needs -- log-log axes, error bars
where an interval was supplied, and a legend when anything is labelled.
A caller who wants their own house style has a real matplotlib `Axes` and
can do as they like with it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SLOPES",
    "SPEC_STYLE",
    "sigma_tau",
    "slope_guides",
    "spec_points",
    "phase",
    "frequency",
    "spectrum",
    "script_l",
    "bench_solution",
    "davar",
    "save",
]

# sigma_y(tau) ~ tau**exponent for each power law. The deviation slope is
# half the variance slope: AVAR goes as tau**(-alpha-1) for -2 <= alpha <= 0
# and saturates at tau**-2 for both PM noises, so the deviation slopes are
# these. White PM and flicker PM share tau**-1 -- that they are
# indistinguishable here is the same degeneracy `powerlaw.py` stacks
# modified rows to break, and drawing them as one line says so honestly.
SLOPES = {
    "W/F PM": -1.0,
    "WFM": -0.5,
    "FFM": 0.0,
    "RWFM": 0.5,
}


# How a *claim* is drawn, as opposed to a measurement: open markers, dashed,
# never the solid-filled style of real data. Named rather than inlined because
# the measurement report draws spec points too, and the one thing that must
# not drift between the two is the visual distinction itself.
SPEC_STYLE = {"marker": "s", "markerfacecolor": "none", "linestyle": "--",
              "linewidth": 1.0, "alpha": 0.8}


def _axes(ax):
    """The caller's axis, or a new figure. matplotlib imported here, not above."""
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "plotting needs matplotlib, which is an optional dependency of "
            "stabilis: `pip install matplotlib`. The analysis core is "
            "deliberately numpy-only, so it is not pulled in by default."
        ) from exc
    return plt.subplots(figsize=(7.0, 4.5))[1]


def sigma_tau(result, confidence=None, label: str | None = None, ax=None,
              marker: str = "o", **kwargs):
    """A sigma-tau curve, log-log, with error bars if an interval is given.

    `confidence` is a `stats.Confidence` for the same result. Its bounds
    are asymmetric -- the chi-square interval on a variance is -- so they
    are drawn as such rather than as a symmetric +/- that would misstate
    the upper bound at small degrees of freedom.
    """
    ax = _axes(ax)
    tau, dev = np.asarray(result.tau_s), np.asarray(result.dev)
    if confidence is None:
        ax.plot(tau, dev, marker=marker, label=label, **kwargs)
    else:
        lo = dev - np.asarray(confidence.dev_lo)
        hi = np.asarray(confidence.dev_hi) - dev
        ax.errorbar(tau, dev, yerr=np.vstack([lo, hi]), marker=marker,
                    label=label, capsize=3, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("averaging time tau (s)")
    ax.set_ylabel(f"{result.kind}")
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()
    return ax


def slope_guides(ax, at=None, which=None, **kwargs):
    """Power-law reference slopes, anchored to the data already plotted.

    Anchored rather than absolute: a guide line is for reading the *slope*
    off a curve, and one drawn at some fixed level would sit off-screen for
    any real oscillator. The anchor defaults to the first plotted point.
    """
    lines = ax.get_lines()
    if at is None:
        if not lines:
            raise ValueError(
                "slope_guides: nothing plotted yet, so there is no data to "
                "anchor to. Plot a curve first, or pass at=(tau, sigma)."
            )
        xs, ys = lines[0].get_xdata(), lines[0].get_ydata()
        at = (float(xs[0]), float(ys[0]))
    tau0, sigma0 = at
    span = np.array(ax.get_xlim())
    # Both limits are captured first and restored after. A tau^-1 guide
    # drawn across three decades of tau spans three decades of sigma, so
    # letting it autoscale squashes the data it was meant to help read --
    # the guides should pass through the plot, not define it.
    keep_x, keep_y = ax.get_xlim(), ax.get_ylim()
    for name, exponent in SLOPES.items():
        if which and name not in which:
            continue
        ax.plot(span, sigma0 * (span / tau0) ** exponent,
                linestyle=":", linewidth=1.0, alpha=0.55,
                label=f"{name} (tau^{exponent:g})", **kwargs)
    ax.set_xlim(*keep_x)
    ax.set_ylim(*keep_y)
    ax.legend(fontsize="small")
    return ax


def spec_points(source, ax=None, **kwargs):
    """A vendor's datasheet sigma(tau) points, drawn as what they are.

    Open markers and a dashed line, never the same style as measured data,
    and the legend says "spec". These are a **claim**, not a measurement --
    the same rule that governs published reference tables applies to them,
    and a plot that renders them identically to your own data invites
    exactly the confusion the rule exists to prevent.
    """
    if not source.spec_adev:
        raise ValueError(f"source {source.id!r} carries no spec_adev points")
    ax = _axes(ax)
    tau = np.array([p[0] for p in source.spec_adev])
    dev = np.array([p[1] for p in source.spec_adev])
    style = dict(SPEC_STYLE)
    style.update(kwargs)
    ax.plot(tau, dev, label=f"{source.id} (spec, not measured)", **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    return ax


def phase(series, ax=None, label: str | None = None, **kwargs):
    """Phase against time, in seconds. Gaps stay gaps -- NaN breaks the line."""
    ax = _axes(ax)
    t = np.arange(series.n, dtype=np.float64) * series.tau0_s
    ax.plot(t, series.x, label=label, **kwargs)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("phase x(t) (s)")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def frequency(series, ax=None, label: str | None = None, **kwargs):
    """Fractional frequency against time."""
    ax = _axes(ax)
    t = np.arange(series.n, dtype=np.float64) * series.tau0_s
    ax.plot(t, series.y, label=label, **kwargs)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fractional frequency y(t)")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def spectrum(spec, label: str | None = None, ax=None, **kwargs):
    """A power spectral density, log-log, DC dropped.

    DC is dropped because the axis is logarithmic and because a detrended
    record's DC bin holds the residue of the detrend rather than a
    measurement -- plotting it would draw attention to the one bin that
    means least.
    """
    ax = _axes(ax)
    f, psd = np.asarray(spec.f), np.asarray(spec.psd)
    keep = f > 0.0
    ax.plot(f[keep], psd[keep], label=label, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fourier frequency f (Hz)")
    ax.set_ylabel(f"{spec.kind} ({spec.units})")
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()
    return ax


def script_l(spec, nu0_hz=None, label: str | None = None, ax=None, **kwargs):
    """`L(f)` in dBc/Hz, drawn only where the small-angle result holds.

    The refused bins are left as a gap rather than joined across, because a
    line drawn through them asserts a near-carrier phase-noise figure that
    the measurement does not support -- which is the one part of this plot
    a datasheet gets written from.
    """
    from .stats.psd import script_l as compute

    ax = _axes(ax)
    f, dbc, valid = compute(spec, nu0_hz)
    keep = f > 0.0
    ax.plot(f[keep], dbc[keep], label=label, **kwargs)   # NaN breaks the line
    ax.set_xscale("log")
    ax.set_xlabel("offset frequency f (Hz)")
    ax.set_ylabel("L(f) (dBc/Hz)")
    ax.grid(True, which="both", alpha=0.3)
    if not valid.all():
        ax.set_title(f"{int((~valid).sum())} bins withheld: "
                     "phase excursion too large for L = S_phi/2",
                     fontsize="small")
    if label:
        ax.legend()
    return ax


def bench_solution(solution, ax=None, channels: bool = True, **kwargs):
    """Every source the hat separated, on one axis. The payoff plot.

    Points the solve pinned at exactly zero are dropped rather than
    plotted at the bottom of the axis: on a log scale zero is not a small
    number, it is absent, and drawing it as a floor would invent a
    measurement. Where that happens the source's own noise was below what
    the bench could resolve -- read `signed_var` to see how far below.
    """
    ax = _axes(ax)
    for i, name in enumerate(solution.sources):
        var = solution.source_var[:, i]
        good = var > 0.0
        ax.plot(solution.tau_s[good], np.sqrt(var[good]), marker="o",
                label=name, **kwargs)
    if channels:
        for i, name in enumerate(solution.channels):
            var = solution.channel_var[:, i]
            good = var > 0.0
            ax.plot(solution.tau_s[good], np.sqrt(var[good]), marker="^",
                    linestyle="--", alpha=0.7, label=f"{name} (floor)", **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("averaging time tau (s)")
    ax.set_ylabel(f"{solution.estimator} of each source")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small")
    return ax


def davar(surface, ax=None, colorbar: bool = True, cmap: str = "viridis",
          vmin: float | None = None, vmax: float | None = None, **kwargs):
    """The dynamic stability surface as a waterfall: time across, tau up.

    Colour is `log10(dev)`, because the deviation spans decades in tau and
    a linear map would show only the shortest one. NaN cells -- a tau that
    window could not estimate -- are left unpainted rather than filled, so
    a hole in the surface reads as a hole.

    Drawn with `pcolormesh` on cell *edges* rather than `imshow`, because
    the tau axis is logarithmic and geometrically spaced: an image would
    stretch every cell to the same height and misplace every ridge.
    """
    ax = _axes(ax)
    z = np.log10(np.sqrt(np.where(surface.var > 0.0, surface.var, np.nan))).T
    mesh = ax.pcolormesh(
        _edges(surface.t_s), _edges(surface.tau_s, log=True), z,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", **kwargs,
    )
    ax.set_yscale("log")
    ax.set_xlabel("window centre t (s)")
    ax.set_ylabel("averaging time tau (s)")
    ax.set_title(
        f"{surface.kind} over {surface.window}-sample windows, step {surface.step}"
    )
    if colorbar:
        ax.get_figure().colorbar(mesh, ax=ax, label=f"log10 {surface.kind}")
    return ax


def _edges(centres, log: bool = False):
    """Cell boundaries around `centres`, for pcolormesh's flat shading.

    Midpoints inside, and the spacing of the first and last pair reflected
    outward at the ends. Geometric midpoints on a log axis, since the
    arithmetic mean of 1 and 2 is not where the boundary between the two
    decade-spaced cells belongs.
    """
    c = np.log(np.asarray(centres, dtype=np.float64)) if log \
        else np.asarray(centres, dtype=np.float64)
    if c.size == 1:
        half = 0.5 if log else 0.5 * max(abs(c[0]), 1.0)
        out = np.array([c[0] - half, c[0] + half])
    else:
        mid = 0.5 * (c[:-1] + c[1:])
        out = np.concatenate(([2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]))
    return np.exp(out) if log else out


def save(ax, path, dpi: int = 150):
    """Write the axis's figure. Format comes from the suffix."""
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    return path
