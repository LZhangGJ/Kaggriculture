#!/usr/bin/env python3
"""Freeze the exact Round2 executable runtime before any simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
NATIVE_ROOT = HERE / "native" / "fast_kaggriculture"
SOURCE = (
    Path(r"D:\Kaggriculture\route_clustering_top40_20260831")
    / "output-v3" / "agent-dynamic" / "multifile" / "teammate_base.py"
)
OUTPUT = HERE / "runtime.lock.json"
FAST_PYTHON = NATIVE_ROOT / "python"
DLL_DIR = (
    Path(r"D:\Kaggriculture\route_clustering_top40_20260831")
    / ".conda_toolchain" / "Library" / "bin"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_row(role: str, path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite runtime lock: {args.output}")
    pyds = list((NATIVE_ROOT / "python" / "fast_kaggriculture").glob("_fast_kaggriculture*.pyd"))
    if len(pyds) != 1:
        raise RuntimeError(f"expected exactly one Round2 native extension, got {len(pyds)}")
    native_record = file_row("round2_native_extension", pyds[0])
    contract_record = file_row(
        "api_contract_test", HERE / "test_semantic_api_contract.py"
    )
    dll_handle = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dll_handle = os.add_dll_directory(str(DLL_DIR.resolve()))
    sys.path.insert(0, str(FAST_PYTHON.resolve()))
    import fast_kaggriculture._fast_kaggriculture as native
    if Path(native.__file__).resolve() != pyds[0].resolve():
        raise RuntimeError(f"wrong native extension loaded: {native.__file__}")
    contract_doc = native.NativeAdaptiveExecutor.candidate8_committed_sequence.__doc__ or ""
    required_arguments = (
        "decision_days", "selected_ranks", "selected_deltas", "capture_trace"
    )
    missing = [name for name in required_arguments if name not in contract_doc]
    if missing:
        raise RuntimeError(f"semantic API contract missing: {missing[0]}")
    contract_record["passed"] = True
    contract_record["required_arguments"] = list(required_arguments)
    files = [
        file_row("v3_teammate_source", SOURCE),
        native_record,
        file_row("semantic_binding", NATIVE_ROOT / "src" / "bindings.cpp"),
        file_row("semantic_executor", NATIVE_ROOT / "src" / "native_adaptive.cpp"),
        file_row("semantic_executor_header", NATIVE_ROOT / "src" / "native_adaptive.hpp"),
        contract_record,
        file_row("materializer", HERE / "materialize_plan_sequences.py"),
        file_row("evaluator", HERE / "evaluate_semantic.py"),
        file_row("frozen_config", HERE / "config.json"),
        file_row("input_lock", HERE / "inputs.lock.json"),
    ]
    payload = {
        "schema": "kaggriculture.counter-cluster-round2-runtime-lock.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_started": False,
        "source_sha256": sha256_file(SOURCE),
        "native_extension": native_record,
        "api_contract_test": contract_record,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    del dll_handle
    print(json.dumps({"runtime_lock": str(args.output.resolve()), "files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
