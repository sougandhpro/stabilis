"""The generated documentation, and the help text it is captured from.

`docs/api/` and `docs/cli-reference.md` are derived from the source, so the
only failure mode worth testing is drift: a signature changes, the page
does not, and the reference starts describing a function that no longer
exists. Both generators answer `--check` for exactly that.

The `--help` test is here for a bug that actually shipped: argparse runs
every help string through `%`-formatting, and the estimator subcommands
take their help from the estimator docstring. "reaching 75 % of the record
length" reads as the conversion spec `% o`, and `stabilis --help` died with
a TypeError before printing anything at all. Nothing rendered help in the
suite, so nothing caught it.
"""

import argparse
import sys
from pathlib import Path

import pytest

from stabilis.cli import build_parser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs"))

import make_api  # noqa: E402
import make_cli  # noqa: E402


def _every_parser(parser, path=()):
    """The parser tree, flattened: (command path, parser)."""
    found = [(path, parser)]
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                found += _every_parser(child, path + (name,))
    return found


@pytest.mark.parametrize(
    "path,parser",
    _every_parser(build_parser()),
    ids=lambda x: " ".join(x) if isinstance(x, tuple) else "",
)
def test_every_command_can_print_its_own_help(path, parser):
    """`--help` renders for every command, not just the ones with tests."""
    text = parser.format_help()
    assert "usage:" in text


def test_api_reference_is_current():
    """docs/api/ matches the source it was generated from."""
    assert make_api.main(["--check"]) == 0, (
        "docs/api is stale; run `python3 docs/make_api.py`"
    )


def test_cli_reference_is_current():
    """docs/cli-reference.md matches the parser it was captured from."""
    assert make_cli.main(["--check"]) == 0, (
        "docs/cli-reference.md is stale; run `python3 docs/make_cli.py`"
    )


def test_every_public_name_has_a_docstring():
    """`__all__` is the promise; an undocumented entry in it is a gap."""
    missing = [
        f"stabilis.{module.dotted}.{entry.name}"
        for modules in make_api.collect().values()
        for module in modules
        for entry in module.entries
        if not entry.doc.strip() and entry.kind in ("function", "class")
    ]
    assert not missing, f"public names with no docstring: {missing}"


def test_the_reference_covers_the_promised_top_level_api():
    """Every name in `stabilis.__all__` is documented under its own module."""
    import stabilis

    documented = {
        entry.name
        for modules in make_api.collect().values()
        for module in modules
        for entry in module.entries
    }
    promised = set(stabilis.__all__) - {"__version__"}
    assert promised <= documented, f"undocumented: {sorted(promised - documented)}"


def test_generation_does_not_import_the_package():
    """The API generator parses source, so optional deps cannot change it.

    Importing would make the reference depend on whether matplotlib and
    scipy are installed -- `stabilis.plot` and `stabilis.report` need
    matplotlib -- and the freshness test above would then pass or fail by
    environment rather than by content.
    """
    source = (Path(make_api.__file__)).read_text(encoding="utf-8")
    assert "import stabilis" not in source
    assert "ast.parse" in source
