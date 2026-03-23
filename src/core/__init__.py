"""Core package helpers."""

from importlib import import_module


def __getattr__(name: str):
    """Provide lazy submodule loading for tests that patch dotted paths."""
    if name == "pipeline":
        return import_module("src.core.pipeline")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
