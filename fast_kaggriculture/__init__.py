"""Convenience import for the bundled C++ training simulator.

The upstream build keeps its Python package under ``python/``.  Re-exporting it
here makes ``from fast_kaggriculture import FastEnv`` work from the project root
without modifying ``PYTHONPATH``.
"""

from .python.fast_kaggriculture import (
    Config,
    FastBatchEnv,
    FastEnv,
    Item,
    NativeTeammateExecutor,
    NativeReplayOpponent,
    native_threshold_variants,
    native_tree_predict,
    audit_raw_tapes,
    raw_tape_audit_metric_names,
    audit_raw_tapes_detailed,
    raw_tape_first_failure_names,
    Op,
)

__all__ = [
    "Config", "FastEnv", "FastBatchEnv", "NativeTeammateExecutor",
    "NativeReplayOpponent",
    "native_threshold_variants", "native_tree_predict",
    "audit_raw_tapes", "raw_tape_audit_metric_names", "Op", "Item",
    "audit_raw_tapes_detailed", "raw_tape_first_failure_names",
]
