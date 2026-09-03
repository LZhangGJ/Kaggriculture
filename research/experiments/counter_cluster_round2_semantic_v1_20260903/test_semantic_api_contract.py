#!/usr/bin/env python3
"""Pure-import contract check for the isolated Round2 native binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_FAST_PYTHON = HERE / "native" / "fast_kaggriculture" / "python"
DEFAULT_DLL_DIR = (
    Path(r"D:\Kaggriculture\route_clustering_top40_20260831")
    / ".conda_toolchain" / "Library" / "bin"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-python", type=Path, default=DEFAULT_FAST_PYTHON)
    parser.add_argument("--dll-dir", type=Path, default=DEFAULT_DLL_DIR)
    args = parser.parse_args()

    dll_handle = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dll_handle = os.add_dll_directory(str(args.dll_dir.resolve()))
    sys.path.insert(0, str(args.fast_python.resolve()))
    import fast_kaggriculture._fast_kaggriculture as native

    native_path = Path(native.__file__).resolve()
    if not native_path.is_relative_to(args.fast_python.resolve()):
        raise RuntimeError(f"wrong native extension loaded: {native_path}")
    method = native.NativeAdaptiveExecutor.candidate8_committed_sequence
    doc = method.__doc__ or ""
    required = (
        "decision_days",
        "selected_ranks",
        "selected_deltas",
        "capture_trace",
    )
    missing = [name for name in required if name not in doc]
    if missing:
        raise RuntimeError(f"semantic binding contract missing: {missing[0]}")
    payload = {
        "passed": True,
        "native_extension": str(native_path),
        "native_extension_sha256": sha256_file(native_path),
        "required_arguments": list(required),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    del dll_handle
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
