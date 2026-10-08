"""
simulation/__init__.py
-----------------------
Package initializer for the `simulation` sub-package.

Architecture position:
    The simulation layer is the top-level orchestrator.  It sits above every
    other module and coordinates all layers: data → environment → FL → RL →
    baselines.  Nothing imports FROM simulation/ (it is the entrypoint).

Exports:
    MetricsLogger, ExperimentRunner
"""
from .metrics import MetricsLogger
from .runner import ExperimentRunner

__all__ = ["MetricsLogger", "ExperimentRunner"]
