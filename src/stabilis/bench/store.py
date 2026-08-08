"""`bench.json`: one file, schema-versioned, referencing data rather than holding it.

Plain JSON and human-editable on purpose. A bench is a description of an
experiment, and the person who ran the experiment has to be able to read it,
correct it, and diff it in a pull request without this library in the loop.

**Data is referenced by path and SHA-256, never contained.** `load` can then
verify that every file still hashes to what the bench says, and refuse to
analyse a bench whose data has moved underneath it -- a measurement whose
bytes changed is a different measurement, and silently analysing it would
put an unearned signature on the result. Verification is opt-in at load and
strict when asked, because a bench is often written before the data exists.

No pickle, ever, for the reason in `CLAUDE.md`: a bench file arriving from a
customer's rig is untrusted input, and JSON cannot execute.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from ..provenance import file_sha256, utc_now
from .model import SCHEMA_VERSION, Bench, Channel, Device, Measurement, Source

__all__ = ["to_dict", "from_dict", "save", "load", "verify", "HashMismatch"]


class HashMismatch(ValueError):
    """A referenced data file no longer hashes to what the bench recorded."""


def to_dict(bench: Bench) -> dict:
    """JSON-ready, with tuples as lists and no numpy anywhere.

    Writes `created` exactly as the bench carries it, including empty. A
    serializer that invents a timestamp is inventing data, and this one is
    also used for hashing and diffing, where a value that changes on every
    call is actively harmful. `save` does the stamping, visibly.
    """
    return {
        "schema": bench.schema,
        "created": bench.created,
        "notes": bench.notes,
        "sources": [
            {**asdict(s), "spec_adev": [list(p) for p in s.spec_adev]}
            for s in bench.sources
        ],
        "devices": [asdict(d) for d in bench.devices],
        "measurements": [asdict(m) for m in bench.measurements],
    }


def from_dict(data: dict, path: str = "") -> Bench:
    """Rebuild a bench, refusing a schema this reader does not know.

    Refusing an unknown version rather than best-guessing it is the whole
    point of having the field: a reader that guesses will one day guess
    wrong on a customer's file and produce a plausible answer.
    """
    schema = data.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"bench schema {schema!r} is not readable by this version, which "
            f"understands {SCHEMA_VERSION}. Refusing rather than guessing at "
            "the difference."
        )
    sources = tuple(
        Source(
            id=s["id"],
            name=s.get("name", ""),
            kind=s.get("kind", ""),
            nominal_hz=s.get("nominal_hz"),
            spec_adev=tuple(tuple(p) for p in s.get("spec_adev", ())),
            spec_floor=s.get("spec_floor"),
            notes=s.get("notes", ""),
        )
        for s in data.get("sources", [])
    )
    devices = tuple(
        Device(
            id=d["id"],
            model=d.get("model", ""),
            serial=d.get("serial", ""),
            channels=tuple(
                Channel(id=c["id"], label=c.get("label", ""),
                        floor=c.get("floor"), notes=c.get("notes", ""))
                for c in d.get("channels", [])
            ),
            notes=d.get("notes", ""),
        )
        for d in data.get("devices", [])
    )
    measurements = tuple(
        Measurement(
            id=m["id"],
            channel_id=m["channel_id"],
            source_a=m["source_a"],
            source_b=m["source_b"],
            path=m.get("path", ""),
            sha256=m.get("sha256", ""),
            kind=m.get("kind", "phase"),
            tau0_s=m.get("tau0_s"),
            epoch=m.get("epoch", ""),
            reason=m.get("reason", ""),
        )
        for m in data.get("measurements", [])
    )
    return Bench(
        sources=sources,
        devices=devices,
        measurements=measurements,
        schema=schema,
        created=data.get("created", ""),
        notes=data.get("notes", ""),
        path=path,
    )


def save(bench: Bench, path, stamp_created: bool = True) -> Bench:
    """Write `bench.json` and return the bench as written.

    Returns the bench rather than the path because saving can change it:
    an unstamped bench gets a `created` time here, and the caller needs the
    object that matches the file rather than the one it passed in.

    `stamp_created=False` for a byte-reproducible write -- regenerating a
    fixture, or hashing a bench for provenance, where a timestamp that
    changes on every call would make the output differ from itself.
    """
    if stamp_created and not bench.created:
        bench = replace(bench, created=utc_now())
    Path(path).write_text(
        json.dumps(to_dict(bench), indent=2) + "\n", encoding="utf-8")
    return replace(bench, path=str(Path(path)))


def load(path, verify_hashes: bool = False) -> Bench:
    """Read a bench. `verify_hashes=True` refuses if any file has changed.

    Off by default because benches are routinely written *before* the data
    exists -- that is the point of `bench check`. On when a solve is about
    to happen, where analysing the wrong bytes is the failure that matters.
    """
    p = Path(path)
    bench = from_dict(json.loads(p.read_text(encoding="utf-8")), path=str(p))
    if verify_hashes:
        problems = verify(bench)
        if problems:
            raise HashMismatch(
                "bench data does not match the recorded hashes:\n  "
                + "\n  ".join(problems)
                + "\nA measurement whose bytes changed is a different "
                "measurement. Re-record the hash deliberately if the change "
                "was intended."
            )
    return bench


def verify(bench: Bench) -> list[str]:
    """Every mismatch between a recorded hash and the file on disk.

    Returns a list rather than raising so a caller can report all of them
    at once; chasing these one exception at a time is miserable.

    A measurement with no `sha256` is skipped and *said to be* skipped by
    the caller -- absence is not agreement, and treating an unhashed row as
    verified would quietly defeat the whole mechanism.
    """
    problems = []
    base = Path(bench.path).parent if bench.path else Path.cwd()
    for m in bench.measurements:
        if not m.path or not m.sha256:
            continue
        target = Path(m.path)
        if not target.is_absolute():
            target = base / target
        if not target.exists():
            problems.append(f"{m.id}: {target} does not exist")
            continue
        actual = file_sha256(target)
        if actual != m.sha256:
            problems.append(
                f"{m.id}: {target} hashes {actual[:12]}..., "
                f"bench records {m.sha256[:12]}..."
            )
    return problems
