"""Frozen benchmark dataset and harness (issues #120, #121).

Legacy mcSCRAM-era scripts in this directory (``bench_scram.py``,
``benchmark_compiler.py``, …) are kept for history and are not used by the
harness.

The loader is imported lazily so that ``benchmarks.reference`` (#179) loads
nothing else from this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .loader import Manifest, Model, load_manifest, repo_root, validate_manifest

__all__ = [
    "Manifest",
    "Model",
    "load_manifest",
    "repo_root",
    "validate_manifest",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
