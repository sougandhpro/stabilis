"""Provenance: what the tool did, and what it decided on your behalf.

The commercial layer prints signed uncertainty statements, so every result
has to carry a traceable account of its own construction -- library and
numpy versions, the input and its hash, and an ordered list of every
decision including the ones taken by default. That makes provenance a core
concern rather than a reporting concern.

Records are immutable. `add_step` returns a new `Provenance` rather than
mutating, so a result object can never have its history edited after the
fact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__

__all__ = [
    "Step",
    "Provenance",
    "new_provenance",
    "add_step",
    "file_sha256",
    "to_dict",
    "to_json",
    "utc_now",
]


def utc_now() -> str:
    """ISO-8601 UTC timestamp, second resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Step:
    """One recorded action.

    `params` records the inputs to the decision and `outcome` records what
    was chosen -- including "fell back to X because Y", which is the part an
    auditor actually reads.
    """

    t_utc: str
    module: str
    action: str
    params: dict
    outcome: str


@dataclass(frozen=True)
class Provenance:
    """The full construction history of a result."""

    stabilis_version: str
    numpy_version: str
    created_utc: str
    source: str
    source_sha256: str | None = None
    steps: tuple[Step, ...] = ()


def new_provenance(source: str, source_sha256: str | None = None) -> Provenance:
    """Start a provenance record for data drawn from `source`."""
    return Provenance(
        stabilis_version=__version__,
        numpy_version=np.__version__,
        created_utc=utc_now(),
        source=source,
        source_sha256=source_sha256,
    )


def add_step(
    prov: Provenance,
    module: str,
    action: str,
    params: dict,
    outcome: str = "ok",
) -> Provenance:
    """Return a copy of `prov` with one more step appended."""
    step = Step(
        t_utc=utc_now(),
        module=module,
        action=action,
        params=dict(params),
        outcome=outcome,
    )
    return replace(prov, steps=prov.steps + (step,))


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in chunks so large records are fine."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_dict(prov: Provenance) -> dict:
    """Plain-Python view, safe for json.dump."""
    return asdict(prov)


def to_json(prov: Provenance, indent: int = 2) -> str:
    """Serialize to JSON. Matches the 2-space convention used elsewhere."""
    return json.dumps(to_dict(prov), indent=indent, default=str)
