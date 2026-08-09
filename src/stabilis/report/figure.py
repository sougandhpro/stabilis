"""The report's one figure, as a self-contained data URI.

Separate from `render.py` so that rendering a report never imports
matplotlib: the document is legible without the plot, and a laboratory
machine that cannot install matplotlib must still be able to produce the
deliverable. `figure_uri` returns an empty string when matplotlib is
absent, and the renderer simply omits the figure.

Embedded as a `data:` URI rather than written alongside as a PNG because
the report is going to be emailed, and a document whose evidence lives in
a sibling file arrives with the evidence missing.
"""

from __future__ import annotations

import base64
import io

import numpy as np

__all__ = ["figure_uri"]


def figure_uri(result, ci=None, spec=(), spec_label: str = "",
               dpi: int = 130) -> str:
    """A sigma-tau PNG as `data:image/png;base64,...`, or "" if unavailable.

    Specification points are drawn in `plot.SPEC_STYLE` -- open markers,
    never the solid style of measured data -- because they are a
    manufacturer's claim and must not be mistaken on the page for something
    this measurement established. The connecting line is dropped: the
    conformity table refuses to interpolate a limit between stated taus, so
    a figure that draws one would contradict it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - depends on environment
        return ""

    from ..plot import SPEC_STYLE, sigma_tau

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    try:
        sigma_tau(result, confidence=ci, label=result.kind, ax=ax)
        if spec:
            tau = np.array([p[0] for p in spec], dtype=np.float64)
            dev = np.array([p[1] for p in spec], dtype=np.float64)
            order = np.argsort(tau)
            style = dict(SPEC_STYLE, linestyle="none")
            ax.plot(tau[order], dev[order],
                    label=f"{spec_label or 'spec'} (claim, not measured)",
                    **style)
            ax.legend(fontsize="small")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=dpi)
    finally:
        plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
