"""The plugin registry: one dict, one decorator, no framework.

Built-in estimators and readers register through exactly the decorator that
third-party packages use. If a built-in cannot be expressed as a plugin,
the plugin API is wrong -- that is the whole design rule here.

A separate pip package extends stabilis without forking it by declaring
entry points:

    [project.entry-points."stabilis.estimator"]
    myvar = "mypkg.variances:myvar"
"""

from __future__ import annotations

from importlib.metadata import entry_points

__all__ = ["KINDS", "register", "get", "available", "load_plugins"]

# "model" holds `fit.models.Model` records rather than bare callables. A
# model is five things -- the function, its parameter names, a starting
# guess, a description and which parameters are physically positive -- and
# a lookup that returned only the function would leave a plugin author
# unable to supply the other four. The registry is a name-to-thing dict; it
# does not care which thing.
KINDS = ("estimator", "reader", "noiseid", "model", "report")

_REGISTRIES: dict[str, dict] = {kind: {} for kind in KINDS}
_plugins_loaded = False


def register(kind: str, name: str):
    """Decorator registering a callable -- or a record -- under `name`.

    Built-ins use this too. Estimators, readers and noise identifiers
    register functions; models register a `Model` dataclass, since a model
    is more than its function. Nothing here inspects what it stores.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")

    def deco(fn):
        _REGISTRIES[kind][name] = fn
        return fn

    return deco


def get(kind: str, name: str):
    """Look up a registered callable, loading third-party plugins first."""
    load_plugins()
    try:
        return _REGISTRIES[kind][name]
    except KeyError:
        raise KeyError(
            f"no {kind} named {name!r}; available: {sorted(_REGISTRIES[kind])}"
        ) from None


def available(kind: str) -> list[str]:
    """Sorted names registered for `kind`, including third-party plugins."""
    load_plugins()
    return sorted(_REGISTRIES[kind])


def load_plugins() -> None:
    """Discover `stabilis.<kind>` entry points. Idempotent, called on demand.

    A broken third-party plugin must not take the library down, so failures
    are reported and skipped rather than raised.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True  # set first: a failure here must not retry forever
    from .log import get_logger

    for kind in KINDS:
        for ep in entry_points(group=f"stabilis.{kind}"):
            try:
                _REGISTRIES[kind][ep.name] = ep.load()
            except Exception as exc:  # noqa: BLE001 - third-party code
                get_logger("registry").warning(
                    "skipping %s plugin %r: %s", kind, ep.name, exc
                )
