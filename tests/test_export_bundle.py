"""The report bundle: a directory that carries its own evidence.

Self-consistency throughout. What a bundle establishes about a measurement
is only as good as what produced it; these tests check that nothing is lost
or misreported between the analysis and the directory.
"""

import json

import numpy as np
import pytest

from stabilis import confidence, load_phase, oadev
from stabilis.export import MANIFEST, SCHEMA, verify
from stabilis.export.bundle import CHECKSUMS, build
from stabilis.provenance import file_sha256

INPUT = "tests/data/stable32/inputs/alphap0.dat"


def _analysis(path=INPUT, estimator=oadev):
    series = load_phase(path, rate_hz=1.0)
    result = estimator(series, taus="octave", keep_coeffs=True)
    return series, result, confidence(result, method="greenhall", alpha=0,
                                      series=series)


def _bundle(outdir, **kw):
    series, result, ci = _analysis()
    kw.setdefault("source_path", INPUT)
    kw.setdefault("include_figure", False)
    return build(outdir, series, result, ci, **kw)


# --------------------------------------------------------------------------
# The digest
# --------------------------------------------------------------------------


def test_the_same_analysis_gives_the_same_digest(tmp_path):
    """Two people compare bundles by sixteen characters, or not at all."""
    a = _bundle(tmp_path / "a")
    b = _bundle(tmp_path / "b")
    assert a.digest == b.digest


def test_the_digest_ignores_the_clock(tmp_path):
    """Hashing the files would not survive this.

    `curve.json` embeds the provenance and provenance carries a timestamp
    on every step, so a file-hash digest changes on every run and says
    nothing. The bytes of a bundle are deliberately *not* reproducible --
    what is reproducible is what was measured.
    """
    bundle = _bundle(tmp_path / "a")
    payload = json.loads((tmp_path / "a" / "curve.json").read_text())

    # The fact that makes file-hashing the wrong construction: an evidence
    # file carries a wall-clock time, so its bytes differ between two runs
    # of the same analysis.
    assert payload["provenance"]["created_utc"]
    assert all(step["t_utc"] for step in payload["provenance"]["steps"])

    # And the digest does not move when it does.
    series, result, ci = _analysis()
    later = build(tmp_path / "b", series, result, ci, source_path=INPUT,
                  include_figure=False)
    assert later.digest == bundle.digest


def test_a_different_analysis_gives_a_different_digest(tmp_path):
    """A digest that never changes is not a digest."""
    from stabilis import mdev

    series, result, ci = _analysis(estimator=mdev)
    other = build(tmp_path / "b", series, result, ci, source_path=INPUT,
                  include_figure=False)
    assert _bundle(tmp_path / "a").digest != other.digest


def test_the_digest_is_anchored_to_the_bytes_that_were_read(tmp_path):
    """Without the source hash, two records with the same curve collide."""
    bundle = _bundle(tmp_path / "a")
    manifest = json.loads((tmp_path / "a" / MANIFEST).read_text())
    assert "source_sha256" in manifest["digest_covers"]
    assert bundle.digest


# --------------------------------------------------------------------------
# Chain of custody
# --------------------------------------------------------------------------


def test_the_copied_record_is_the_original_bytes(tmp_path):
    """Byte-for-byte, so its hash still equals what provenance recorded.

    A re-serialized copy would hash differently and quietly sever the chain
    from the bytes that arrived to the number that was reported.
    """
    series, _, _ = _analysis()
    _bundle(tmp_path / "a")
    copied = tmp_path / "a" / "data" / "alphap0.dat"
    assert copied.read_bytes() == open(INPUT, "rb").read()
    assert file_sha256(copied) == series.provenance.source_sha256


def test_the_data_can_be_left_out(tmp_path):
    """Large or licence-encumbered records stay outside; the hash does not."""
    bundle = _bundle(tmp_path / "a", include_data=False)
    assert not (tmp_path / "a" / "data").exists()
    names = [f["name"] for f in
             json.loads((tmp_path / "a" / MANIFEST).read_text())["files"]]
    assert not any(n.startswith("data/") for n in names)
    # Still anchored: the digest uses provenance's hash of the source.
    assert bundle.digest == _bundle(tmp_path / "b").digest


def test_a_record_with_no_source_file_is_serialized_and_said_to_be(tmp_path):
    """An in-memory series has no original bytes, and the manifest says so."""
    from stabilis import PhaseSeries

    series = PhaseSeries(x=np.random.default_rng(1).normal(0, 1e-12, 256),
                         rate_hz=1.0)
    result = oadev(series, taus="octave", keep_coeffs=True)
    bundle = build(tmp_path / "a", series, result,
                   confidence(result, series=series), include_figure=False)
    roles = [f.format for f in bundle.files]
    assert "data/serialized" in roles
    assert any(s.action == "copy_data" for s in bundle.provenance.steps)


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


def test_the_manifest_hashes_every_file_it_lists(tmp_path):
    _bundle(tmp_path / "a")
    manifest = json.loads((tmp_path / "a" / MANIFEST).read_text())
    assert manifest["schema"] == SCHEMA
    for entry in manifest["files"]:
        target = tmp_path / "a" / entry["name"]
        assert file_sha256(target) == entry["sha256"], entry["name"]
        assert target.stat().st_size == entry["n_bytes"]


def test_the_manifest_lists_everything_except_itself(tmp_path):
    """It cannot hash itself, and the file says so rather than leaving a hole."""
    _bundle(tmp_path / "a")
    listed = {f["name"] for f in
              json.loads((tmp_path / "a" / MANIFEST).read_text())["files"]}
    on_disk = {str(p.relative_to(tmp_path / "a").as_posix())
               for p in (tmp_path / "a").rglob("*") if p.is_file()}
    assert on_disk - listed == {MANIFEST, CHECKSUMS}


def test_the_checksums_file_is_what_sha256sum_expects(tmp_path):
    """A metrologist verifies a bundle with a command they already know."""
    _bundle(tmp_path / "a")
    for line in (tmp_path / "a" / CHECKSUMS).read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert len(digest) == 64
        assert file_sha256(tmp_path / "a" / name) == digest


def test_verify_names_every_file_that_changed(tmp_path):
    """A list, not the first exception -- chasing these one at a time is
    miserable, and a customer's metrologist wants the whole picture."""
    _bundle(tmp_path / "a")
    assert verify(tmp_path / "a") == []

    with open(tmp_path / "a" / "curve.csv", "a") as fh:
        fh.write("1,1,1,1\n")
    with open(tmp_path / "a" / "data" / "alphap0.dat", "a") as fh:
        fh.write("0\n")
    problems = verify(tmp_path / "a")
    assert len(problems) == 2
    assert any("curve.csv" in p for p in problems)
    assert any("alphap0.dat" in p for p in problems)


def test_verify_reports_a_missing_file_rather_than_raising(tmp_path):
    _bundle(tmp_path / "a")
    (tmp_path / "a" / "curve.csv").unlink()
    assert any("missing" in p for p in verify(tmp_path / "a"))


# --------------------------------------------------------------------------
# Contents and refusals
# --------------------------------------------------------------------------


def test_a_bundle_is_complete_without_matplotlib(tmp_path, monkeypatch):
    """The core is numpy-only and that must hold at the bundle level too.

    Every number is present; only the picture is missing, and provenance
    says which.
    """
    import builtins

    real_import = builtins.__import__

    def no_matplotlib(name, *a, **kw):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    bundle = _bundle(tmp_path / "a", include_figure=True)
    assert not (tmp_path / "a" / "sigma_tau.png").exists()
    assert (tmp_path / "a" / "curve.csv").exists()
    assert any(s.action == "figure" for s in bundle.provenance.steps)


def test_a_bundle_refuses_to_overwrite_a_bundle(tmp_path):
    """The manifest in there is somebody's evidence."""
    _bundle(tmp_path / "a")
    with pytest.raises(ValueError, match="already holds a bundle"):
        _bundle(tmp_path / "a")
    _bundle(tmp_path / "a", overwrite=True)      # asked for, so allowed


def test_the_stable32_table_is_omitted_when_it_cannot_be_written(tmp_path):
    """No alpha, no Alpha column -- and the bundle carries the rest anyway."""
    series, result, _ = _analysis()
    driven = confidence(result, method="eta1", series=series)
    build(tmp_path / "a", series, result, driven, source_path=INPUT,
          include_figure=False)
    assert not (tmp_path / "a" / "curve.s32.txt").exists()
    assert (tmp_path / "a" / "curve.json").exists()


def test_cli_export_bundle_writes_the_directory_and_prints_the_digest(
        tmp_path, capsys):
    from stabilis.cli import main

    out = tmp_path / "run"
    assert main(["export", "bundle", INPUT, "--rate", "1",
                 "--edof-method", "greenhall", "--alpha", "0",
                 "--spec", "1:1e-11", "--no-figure", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "digest" in printed and "verdict" in printed
    assert verify(out) == []
    assert (out / "report.html").exists()
