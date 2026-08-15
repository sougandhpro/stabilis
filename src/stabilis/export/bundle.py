"""A directory that carries its own evidence.

One measurement, everything needed to check it, and a manifest that says
what everything hashes to. The point is that it survives being emailed:
`report/figure.py` already makes this argument for embedding its PNG rather
than writing it alongside -- "a document whose evidence lives in a sibling
file arrives with the evidence missing" -- and a bundle is that argument
applied to the whole analysis instead of one image.

**The source record is copied byte for byte**, not re-serialized. That
keeps the copy's SHA-256 equal to `provenance.source_sha256`, so the chain
from the bytes that arrived to the number that was reported is unbroken. A
re-serialized copy would hash differently and quietly sever it.

**The digest covers the measurement, not the bytes that carry it.** Hashing
the files would have been the obvious construction and it is wrong:
`curve.json` embeds the provenance, provenance records *when* things
happened, and a digest that changes every run says nothing. So the digest
is taken over the curve, the interval and the hash of the data they came
from -- the same choice `report.build` makes, for the same reason.

The bundle's *bytes* are therefore not reproducible and are not claimed to
be. Provenance is a record of when work was done; a bundle that could be
made byte-identical on another day would be one that had thrown that away.
What is reproducible is the sixteen characters two people compare.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from ..io.write import WrittenFile
from ..log import record, warn
from ..provenance import Provenance, file_sha256, to_json as prov_to_json, utc_now
from .stable32 import SIGMA_LABELS
from .table import to_csv, to_json

__all__ = ["Bundle", "SCHEMA", "MANIFEST", "CHECKSUMS", "build",
           "manifest_text", "verify"]

SCHEMA = "stabilis-bundle-1"
MANIFEST = "MANIFEST.json"
CHECKSUMS = "SHA256SUMS"


@dataclass(frozen=True)
class Bundle:
    """A written bundle: where it is, what is in it, and what it hashes to."""

    path: str
    schema: str
    created: str
    files: tuple[WrittenFile, ...]
    digest: str
    provenance: Provenance

    def by_role(self, role: str) -> tuple[WrittenFile, ...]:
        return tuple(f for f in self.files if f.format.startswith(role))


def build(outdir, series, result, ci=None, report=None, report_html: str = "",
          source_path=None, include_data: bool = True,
          include_figure: bool = True, stamp: bool = True,
          overwrite: bool = False) -> Bundle:
    """Assemble the directory. Computes nothing the caller could pass in.

    `report` and `report_html` are optional and are the caller's: building
    one here would make the bundle a second place where analysis options
    get chosen, which is exactly what `report.build` refuses to be.
    """
    outdir = Path(outdir)
    if outdir.joinpath(MANIFEST).exists() and not overwrite:
        raise ValueError(
            f"{outdir} already holds a bundle. Refusing to write over it: "
            "the manifest it carries is somebody's evidence. Pass "
            "overwrite=True, or choose another directory."
        )
    outdir.mkdir(parents=True, exist_ok=True)

    prov = (ci or result).provenance
    files: list[WrittenFile] = []

    files.append(_write(outdir / "curve.csv", to_csv(result, ci),
                        "table/csv", prov))
    files.append(_write(outdir / "curve.json", to_json(result, ci) + "\n",
                        "table/json", prov))
    if result.kind in SIGMA_LABELS and ci is not None and ci.alpha is not None:
        from .stable32 import to_stable32
        files.append(_write(outdir / "curve.s32.txt",
                            to_stable32(series, result, ci,
                                        source=Path(source_path).name
                                        if source_path else ""),
                            "table/stable32", prov, newline=""))

    if include_data:
        written, prov = _copy_data(outdir, series, source_path, result, prov)
        if written is not None:
            files.append(written)

    files.append(_write(outdir / "provenance.json", prov_to_json(prov) + "\n",
                        "provenance/json", prov))
    if report_html:
        files.append(_write(outdir / "report.html", report_html,
                            "report/html", prov))
    if include_figure:
        written, prov = _figure(outdir, result, ci, prov)
        if written is not None:
            files.append(written)

    digest = _digest(result, ci, _source_sha(series, files))
    bundle = Bundle(
        path=str(outdir),
        schema=SCHEMA,
        created=utc_now() if stamp else "",
        files=tuple(files),
        digest=digest,
        provenance=record(
            prov, "export.bundle", "build",
            {"files": len(files) + 2, "include_data": include_data,
             "stamped": stamp},
            f"digest {digest[:16]} over the curve, the interval and the data",
        ),
    )
    outdir.joinpath(MANIFEST).write_text(manifest_text(bundle),
                                         encoding="utf-8", newline="\n")
    outdir.joinpath(CHECKSUMS).write_text(
        "".join(f"{f.sha256}  {_rel(outdir, f)}\n" for f in files),
        encoding="utf-8", newline="\n")
    return bundle


def manifest_text(bundle: Bundle, indent: int = 2) -> str:
    """The manifest, which lists every file but cannot hash itself."""
    payload = {
        "schema": bundle.schema,
        "created": bundle.created,
        "digest": bundle.digest,
        "digest_covers": ["source_sha256", "kind", "af", "tau_s", "var",
                          "n_terms", "confidence"],
        "digest_note": (
            "Taken over the measurement, not over the files: the curve, the "
            "interval, and the SHA-256 of the data they came from. The same "
            "analysis therefore gives the same digest on another day and "
            "another machine, while the files do not -- provenance records "
            "when work was done, and that is the point of it. The per-file "
            f"hashes below and in {CHECKSUMS} are for detecting tampering "
            f"with this copy. {MANIFEST} and {CHECKSUMS} cannot list "
            "themselves."
        ),
        "stabilis_version": bundle.provenance.stabilis_version,
        "numpy_version": bundle.provenance.numpy_version,
        "files": [
            {"name": _rel(Path(bundle.path), f), "role": f.format,
             "n_bytes": f.n_bytes, "sha256": f.sha256}
            for f in bundle.files
        ],
    }
    return json.dumps(payload, indent=indent, default=str) + "\n"


def verify(path) -> list[str]:
    """Every file whose bytes no longer match the manifest.

    A list rather than an exception, following `bench.store.verify`:
    chasing these one at a time is miserable, and a customer's metrologist
    wants the whole picture in one go.
    """
    path = Path(path)
    manifest = json.loads(path.joinpath(MANIFEST).read_text(encoding="utf-8"))
    problems = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"schema {manifest.get('schema')!r}, expected {SCHEMA!r}")
    for entry in manifest.get("files", ()):
        target = path / entry["name"]
        if not target.exists():
            problems.append(f"{entry['name']}: missing")
            continue
        actual = file_sha256(target)
        if actual != entry["sha256"]:
            problems.append(
                f"{entry['name']}: hashes {actual[:12]}..., "
                f"manifest records {entry['sha256'][:12]}...")
    return problems


# --------------------------------------------------------------------------


def _write(path: Path, text: str, role: str, prov, newline: str = "\n"):
    """One file, its hash, and the role that decides if the digest sees it."""
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)
    return WrittenFile(path=str(path), sha256=file_sha256(path),
                       n_bytes=path.stat().st_size, format=role,
                       provenance=prov)


def _copy_data(outdir: Path, series, source_path, result, prov):
    """The source record, byte for byte, so its hash still means something.

    When the series was built in memory there is nothing to copy and no
    hash to preserve, so it is serialized instead and the manifest says
    which of the two happened.
    """
    src = Path(source_path) if source_path else None
    if src is None or not src.exists():
        from ..io.write import save_frequency, save_phase

        outdir.joinpath("data").mkdir(exist_ok=True)
        target = outdir / "data" / "record.dat"
        save = save_phase if hasattr(series, "x") else save_frequency
        written = save(series, target)
        return replace(written, format="data/serialized"), warn(
            prov, "export.bundle", "copy_data", {"source": str(source_path)},
            "no source file to copy, so the record was re-serialized; its "
            "hash is of our bytes, not of the bytes that arrived",
        )

    outdir.joinpath("data").mkdir(exist_ok=True)
    target = outdir / "data" / src.name
    shutil.copyfile(src, target)
    sha = file_sha256(target)
    if series.provenance.source_sha256 and sha != series.provenance.source_sha256:
        prov = warn(prov, "export.bundle", "copy_data", {"file": src.name},
                    "the copied bytes do not hash to what provenance "
                    "recorded; the source changed after it was read")
    return WrittenFile(path=str(target), sha256=sha,
                       n_bytes=target.stat().st_size, format="data/original",
                       provenance=prov), prov


def _figure(outdir: Path, result, ci, prov):
    """The curve as a PNG, or nothing and a note saying why.

    A bundle must not require matplotlib -- the core is numpy-only and that
    is a selling point -- so the figure is the one optional member.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, warn(prov, "export.bundle", "figure", {},
                          "matplotlib is not installed, so the bundle "
                          "carries no figure; every number in it is present")

    from ..plot import save, sigma_tau

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    try:
        sigma_tau(result, confidence=ci, label=result.kind, ax=ax)
        fig.tight_layout()
        save(ax, outdir / "sigma_tau.png")
    finally:
        plt.close(fig)
    target = outdir / "sigma_tau.png"
    return WrittenFile(path=str(target), sha256=file_sha256(target),
                       n_bytes=target.stat().st_size, format="figure/png",
                       provenance=prov), prov


def _source_sha(series, files) -> str:
    """What the analysis was run on, whether or not it was copied in.

    Provenance's hash comes first: it is of the bytes that were *read*,
    which is the claim being anchored. The copied file's hash is the
    fallback for a series built in memory.
    """
    if series.provenance.source_sha256:
        return series.provenance.source_sha256
    data = [f for f in files if f.format.startswith("data/")]
    return data[0].sha256 if data else ""


def _digest(result, ci, source_sha: str) -> str:
    """SHA-256 over what was measured, so two copies compare in sixteen chars.

    Deliberately not over the files. `curve.json` embeds the provenance and
    provenance carries timestamps, so a file-hash digest would differ on
    every run and mean nothing. `report.build` reached the same conclusion
    about its own digest, and this is that rule applied one level up.
    """
    payload = {
        "source_sha256": source_sha,
        "kind": result.kind,
        "af": result.af.tolist(),
        "tau_s": result.tau_s.tolist(),
        "var": result.var.tolist(),
        "n_terms": result.n_terms.tolist(),
    }
    if ci is not None:
        payload["confidence"] = {
            "level": ci.level,
            "edof": ci.edof.tolist(),
            "var_lo": ci.var_lo.tolist(),
            "var_hi": ci.var_hi.tolist(),
            "method": list(ci.method),
            "alpha": None if ci.alpha is None else ci.alpha.tolist(),
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _rel(outdir: Path, written: WrittenFile) -> str:
    return Path(written.path).relative_to(outdir).as_posix()
