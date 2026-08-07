"""Access to the theory pages shipped inside the package.

The pages live in `stabilis/theory/*.md` rather than a repo-level `docs/`
so they survive a pip install and can be read from the CLI on a machine
that never saw the source tree.
"""

from __future__ import annotations

from importlib.resources import files

__all__ = ["list_topics", "read_topic", "topic_summary"]

_PACKAGE_DIR = "theory"


def list_topics() -> list[str]:
    """Topic names, without the .md suffix, sorted."""
    root = files("stabilis").joinpath(_PACKAGE_DIR)
    return sorted(
        p.name.removesuffix(".md") for p in root.iterdir() if p.name.endswith(".md")
    )


def read_topic(topic: str) -> str:
    """Markdown source of one theory page."""
    path = files("stabilis").joinpath(_PACKAGE_DIR, f"{topic}.md")
    if not path.is_file():
        raise KeyError(f"no theory page {topic!r}; available: {list_topics()}")
    return path.read_text(encoding="utf-8")


def topic_summary(topic: str) -> str:
    """The page's first heading, for the --list listing."""
    for line in read_topic(topic).splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return topic
