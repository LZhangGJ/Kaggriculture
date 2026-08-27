"""Typed C++ Kaggriculture simulator (Apache-2.0)."""

# Conda-forge's MinGW extension build keeps the GCC/OpenMP runtimes in
# separate package-cache directories on Windows.  Register those directories
# before importing the extension so native experiments also work from a fresh
# terminal, not only from the shell that performed the build.  A release wheel
# may vendor these DLLs beside the extension; that directory is checked first.
import os as _os
from pathlib import Path as _Path
import sys as _sys

_DLL_HANDLES = []
if _os.name == "nt" and hasattr(_os, "add_dll_directory"):
    _prefixes = dict.fromkeys(map(_Path, (_sys.prefix, _sys.base_prefix)))
    _candidates = [_Path(__file__).resolve().parent]
    for _prefix in _prefixes:
        _candidates.append(_prefix / "Library" / "bin")
        for _package in ("libgcc", "libgomp", "libstdcxx", "libwinpthread"):
            _candidates.extend(sorted(
                (_prefix / "pkgs").glob(f"{_package}-*/Library/bin"), reverse=True
            ))
    for _directory in dict.fromkeys(_candidates):
        if _directory.is_dir():
            try:
                _DLL_HANDLES.append(_os.add_dll_directory(str(_directory)))
            except OSError:
                pass

from ._fast_kaggriculture import (
    Config,
    FastEnv,
    FastBatchEnv,
    NativeTeammateExecutor,
    native_threshold_variants,
    native_tree_predict,
    audit_raw_tapes,
    raw_tape_audit_metric_names,
    audit_raw_tapes_detailed,
    raw_tape_first_failure_names,
    Op,
    Item,
)

__all__ = [
    "Config", "FastEnv", "FastBatchEnv", "NativeTeammateExecutor",
    "native_threshold_variants", "native_tree_predict",
    "audit_raw_tapes", "raw_tape_audit_metric_names", "Op", "Item",
    "audit_raw_tapes_detailed", "raw_tape_first_failure_names",
]
