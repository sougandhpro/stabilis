"""Logging that doubles as the audit trail.

The human-readable log and the machine-readable provenance record carry the
same information, so they come from one call. `record()` writes to both: a
line on the logger for whoever is watching, and an immutable `Step` on the
returned `Provenance` for whoever has to defend the number later.

No classes -- a step log is just a `Provenance` being threaded through the
call chain.
"""

from __future__ import annotations

import logging

from .provenance import Provenance, add_step

__all__ = ["get_logger", "record", "warn", "configure"]

_ROOT = "stabilis"


def get_logger(name: str | None = None) -> logging.Logger:
    """Logger for a submodule. `get_logger("stats.classical")`."""
    return logging.getLogger(_ROOT if name is None else f"{_ROOT}.{name}")


def configure(level: str = "warning") -> None:
    """Attach a stderr handler at `level`. Called by the CLI, not on import.

    Libraries must not configure logging on import, so this is opt-in.
    """
    logger = logging.getLogger(_ROOT)
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)


def record(
    prov: Provenance,
    module: str,
    action: str,
    params: dict,
    outcome: str = "ok",
) -> Provenance:
    """Log one step and return the extended provenance.

    Use this for every decision the library takes on the caller's behalf --
    tau grid, gap policy, EDOF method actually used, assumed bandwidth. The
    default-valued decisions matter most, because those are the ones the
    caller does not know were made.
    """
    get_logger(module).info("%s %s -> %s", action, _brief(params), outcome)
    return add_step(prov, module, action, params, outcome)


def warn(
    prov: Provenance,
    module: str,
    action: str,
    params: dict,
    outcome: str,
) -> Provenance:
    """Record a step that the user needs to see. Same record, louder."""
    get_logger(module).warning("%s %s -> %s", action, _brief(params), outcome)
    return add_step(prov, module, action, params, outcome)


def _brief(params: dict) -> str:
    """Compact one-line rendering of params for the human log."""
    return "{" + ", ".join(f"{k}={v!r}" for k, v in params.items()) + "}"
