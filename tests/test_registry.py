"""The plugin registry.

Built-ins register through exactly the decorator third parties use, so these
tests exercise the public extension path.
"""

import pytest

from stabilis import registry


def test_builtin_reader_is_registered():
    """io.text registers itself -- the built-in is a plugin."""
    import stabilis.io  # noqa: F401  (import triggers registration)

    assert "text" in registry.available("reader")
    assert callable(registry.get("reader", "text"))


def test_register_and_get():
    @registry.register("estimator", "_test_dummy")
    def dummy(x):
        return x * 2

    assert registry.get("estimator", "_test_dummy")(3) == 6
    assert "_test_dummy" in registry.available("estimator")


def test_decorator_returns_the_function_unchanged():
    """Registration must not wrap; a plugin author's function stays theirs."""

    def original(x):
        return x

    decorated = registry.register("estimator", "_test_identity")(original)
    assert decorated is original


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown plugin kind"):
        registry.register("nonsense", "x")


def test_unknown_name_lists_alternatives():
    """The error has to tell you what you could have asked for."""
    with pytest.raises(KeyError, match="available"):
        registry.get("estimator", "_definitely_not_registered")


def test_all_kinds_have_a_registry():
    for kind in registry.KINDS:
        assert isinstance(registry.available(kind), list)


def test_load_plugins_is_idempotent():
    registry.load_plugins()
    registry.load_plugins()
    assert "text" in registry.available("reader")
