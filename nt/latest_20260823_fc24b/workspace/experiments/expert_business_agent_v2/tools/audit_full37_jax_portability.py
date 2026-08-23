#!/usr/bin/env python3
"""Audit the frozen 37-opponent pool before an exact JAX port.

This is deliberately a read-only host-side audit.  It parses every source file
and imports it in an isolated module namespace to inventory the action streams,
route metadata, mutable controller state and procedural code that must be
ported.  The result is not a parity claim; it is the migration work ledger.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_module(path: Path, index: int):
    spec = importlib.util.spec_from_file_location(f"jax_portability_audit_{index}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assignment_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def function_rows(tree: ast.Module) -> list[dict[str, Any]]:
    rows = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = Counter()
        globals_written: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    calls[func.id] += 1
                elif isinstance(func, ast.Attribute):
                    calls[func.attr] += 1
            elif isinstance(child, ast.Global):
                globals_written.update(child.names)
        rows.append(
            {
                "name": node.name,
                "line_start": node.lineno,
                "line_end": getattr(node, "end_lineno", node.lineno),
                "line_count": getattr(node, "end_lineno", node.lineno) - node.lineno + 1,
                "calls": [name for name, _ in calls.most_common()],
                "globals_written": sorted(globals_written),
            }
        )
    return rows


def list_streams(module: Any) -> list[dict[str, Any]]:
    streams: list[tuple[str, Any]] = []
    cgr = getattr(module, "_CGR_STREAMS", None)
    if isinstance(cgr, dict):
        streams.extend((f"_CGR_STREAMS/{name}", value) for name, value in cgr.items())
    for name in ("_ACTIONS", "_LEGACY_ACTIONS", "_REBALANCE_ACTIONS", "_ROUTE"):
        value = getattr(module, name, None)
        if isinstance(value, list):
            streams.append((name, value))
    unique: dict[str, dict[str, Any]] = {}
    for name, stream in streams:
        canonical = json.dumps(
            stream,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest().upper()
        unique.setdefault(
            digest,
            {"name": name, "steps": len(stream), "action_sha256": digest},
        )
    return list(unique.values())


def classify(
    functions: list[dict[str, Any]],
    streams: list[dict[str, Any]],
    mutable_state: list[str],
    assignments: set[str],
) -> tuple[str, list[str]]:
    agent = next((row for row in functions if row["name"] in {"agent", "submission_agent"}), None)
    procedural = [
        row
        for row in functions
        if row["name"] not in {"agent", "submission_agent"}
        and not row["name"].startswith("__")
    ]
    route_metadata = any(
        name in assignments
        for name in (
            "_CGR_STREAMS",
            "_CGR_CENTROIDS",
            "_TREE",
            "_E279_LOW_ACTIONS",
            "_E279_HIGH_ACTIONS",
        )
    )
    reasons: list[str] = []
    if not streams:
        reasons.append("no directly extractable 719-step stream")
        return "manual_dynamic_port", reasons
    if route_metadata:
        reasons.append("explicit finite route bank/router metadata")
        if mutable_state:
            reasons.append("mutable per-seat state must be functionalized")
        return "route_state_machine_port", reasons
    if len(procedural) <= 8 and not mutable_state and agent and agent["line_count"] <= 24:
        reasons.append("single trace with a small stateless wrapper")
        return "trace_plus_small_wrapper", reasons
    reasons.append(f"{len(procedural)} helper/controller functions")
    if mutable_state:
        reasons.append("mutable per-seat/global state")
    return "procedural_controller_port", reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        type=Path,
        default=ROOT
        / "experiments"
        / "expert_business_agent_v2"
        / "configs"
        / "local_representative_pool_v1.json",
    )
    parser.add_argument(
        "--current-jax-receipt",
        type=Path,
        default=ROOT
        / "experiments"
        / "expert_business_agent_v2"
        / "receipts"
        / "jax_route46_dynamic_broad23_bank_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "expert_business_agent_v2"
        / "receipts"
        / "full37_jax_portability_audit_v1.json",
    )
    args = parser.parse_args()

    pool_path = resolve(args.pool)
    current_path = resolve(args.current_jax_receipt)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    current_sources = {
        Path(row["source"]).resolve(): row
        for row in current["opponents"]
    }

    rows: list[dict[str, Any]] = []
    for index, opponent in enumerate(pool["opponents"]):
        path = resolve(opponent["path"]).resolve()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        functions = function_rows(tree)
        assignments = assignment_names(tree)
        module = load_module(path, index)
        streams = list_streams(module)
        mutable_state = sorted(
            name
            for name, value in vars(module).items()
            if name.startswith("_")
            and isinstance(value, (dict, set))
            and any(
                token in name
                for token in ("STATE", "PENDING", "ACTIVE", "LAST", "CACHE")
            )
        )
        port_class, reasons = classify(
            functions, streams, mutable_state, assignments
        )
        current_spec = current_sources.get(path)
        rows.append(
            {
                "opponent_id": index,
                "name": opponent["name"],
                "class": opponent["class"],
                "family": opponent["family"],
                "path": str(path),
                "source_sha256": sha256(path),
                "source_lines": len(source.splitlines()),
                "stream_count": len(streams),
                "streams": streams,
                "function_count": len(functions),
                "functions": functions,
                "mutable_state": mutable_state,
                "port_class": port_class,
                "port_reasons": reasons,
                "current_jax_covered": current_spec is not None,
                "current_jax_kind": current_spec.get("kind") if current_spec else None,
                "current_jax_status": (
                    "COMPILED_NOT_PARITY_ACCEPTED" if current_spec else "MISSING"
                ),
            }
        )

    counts = Counter(row["port_class"] for row in rows)
    covered = sum(row["current_jax_covered"] for row in rows)
    result = {
        "schema": "kaggriculture-full37-jax-portability-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUDIT_ONLY_NOT_PARITY_EVIDENCE",
        "pool": str(pool_path),
        "pool_sha256": sha256(pool_path),
        "current_jax_receipt": str(current_path),
        "current_jax_receipt_sha256": sha256(current_path),
        "opponent_count": len(rows),
        "current_jax_covered": covered,
        "current_jax_missing": len(rows) - covered,
        "port_class_counts": dict(sorted(counts.items())),
        "opponents": rows,
        "acceptance_boundary": (
            "Every opponent still requires same-seed, same-seat, stepwise action "
            "and public-state differential testing against official Python 1.32.7."
        ),
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "opponent_count": result["opponent_count"],
                "current_jax_covered": result["current_jax_covered"],
                "current_jax_missing": result["current_jax_missing"],
                "port_class_counts": result["port_class_counts"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
