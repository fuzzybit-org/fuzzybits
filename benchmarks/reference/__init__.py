"""Independent reference estimators for FTP projects (#179).

Reference values must come from methods other than the tool being scored
(#159), so this package imports nothing from ``fuzzybits``, ``src/``,
``python/`` or the benchmark harness; it needs only NumPy. It contains its own
minimal FTP parser (reference-only, not the product reader), exact Boolean and
monotone-hull evaluators for small models, a bit-parallel Boolean Monte Carlo
estimator and a monotone-hull estimator.
"""

from __future__ import annotations

from .cli import run
from .exact import exact_probability, exact_table
from .ftp import Diagnostic, ModelError, Project, parse_project
from .hull import LaneChecker, estimate_hull
from .montecarlo import estimate_boolean
from .structure import structure_counts

__all__ = [
    "Diagnostic",
    "LaneChecker",
    "ModelError",
    "Project",
    "estimate_boolean",
    "estimate_hull",
    "exact_probability",
    "exact_table",
    "parse_project",
    "run",
    "structure_counts",
]
