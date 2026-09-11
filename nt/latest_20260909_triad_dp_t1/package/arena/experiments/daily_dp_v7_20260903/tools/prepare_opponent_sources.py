"""Freeze requested opponents and inspect wrappers without executing their code.

Downloads must already exist under opponents/<id>/{source,output}.
Only literal Base85/zlib wrappers are decoded; this is not an eval/import runner.
"""
from __future__ import annotations

import ast
import argparse
import base64
import hashlib
import io
import json
import tarfile
import zlib
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
POOL = EXP / "opponents"
LIMIT = 128 * 1024 * 1024
PUBLIC = {
    "boatlee_v29": "boatlee/v29-r1-adaptive-market-hysteresis",
    "kaito_v58": "kaitofukami/238-238-known-streams-v58-minimax-closed-loop",
    "lynn_v5": "lynnsakurai/farming-score-v5-timing-optimized",
    "ecobot_v7": "premaananda108/economics-driven-rule-agent-ecobot-v7-arena",
    "yhay81_six_day": "yhay81/six-day-public-state-fieldbook",
    "yhay81_three_day": "yhay81/three-day-shop-router",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite frozen artifact: {path}")
    else:
        path.write_bytes(data)


def literal_unwrap(data: bytes) -> bytes | None:
    tree = ast.parse(data.decode("utf-8-sig"))
    # The accepted wrapper consists only of import statements and one exec.
    statements = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    if len(statements) != 1 or not isinstance(statements[0], ast.Expr):
        return None
    call = statements[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "exec":
        return None
    constants = [n.value for n in ast.walk(call) if isinstance(n, ast.Constant) and isinstance(n.value, bytes)]
    attrs = {n.attr for n in ast.walk(call) if isinstance(n, ast.Attribute)}
    if len(constants) != 1 or not {"decompress", "b85decode"} <= attrs:
        return None
    raw = base64.b85decode(constants[0])
    decoder = zlib.decompressobj()
    unpacked = decoder.decompress(raw, LIMIT + 1)
    if len(unpacked) > LIMIT or not decoder.eof:
        raise ValueError("Wrapper exceeds bounded decode size or is incomplete")
    return unpacked


def source_summary(data: bytes) -> dict:
    text = data.decode("utf-8-sig")
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return {"parse_error": str(error), "bytes": len(data)}
    return {
        "bytes": len(data), "lines": len(text.splitlines()), "sha256": sha(data),
        "imports": sorted({ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))}),
        "definitions": [{"name": n.name, "line": n.lineno, "kind": type(n).__name__}
                        for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
        "dynamic_calls": [{"line": n.lineno, "name": n.func.id}
                          for n in ast.walk(tree) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name) and n.func.id in ("exec", "eval", "compile", "__import__")],
    }


def inspect_python(path: Path, data: bytes) -> dict:
    entry = {"path": str(path.relative_to(EXP)), "source": source_summary(data)}
    for depth in range(4):
        unpacked = literal_unwrap(data)
        if unpacked is None:
            break
        decoded = path.with_name(path.stem + f".decoded{depth + 1}.py")
        put(decoded, unpacked)
        entry.setdefault("decoded", []).append({"path": str(decoded.relative_to(EXP)), **source_summary(unpacked)})
        data = unpacked
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-name", default="source_inventory_v2.json")
    args = parser.parse_args()
    assert Path(args.inventory_name).name == args.inventory_name
    g003 = ROOT / "submission/main_g003.py"
    raw = g003.read_bytes()
    expected = "ff32e2e47431dae2d98fa70a78f85425f8d5929d5910c5ff482e74750c3fa1ef"
    assert sha(raw) == expected, "G003 source changed"
    frozen = POOL / "g003/source/main.py"
    put(frozen, raw)
    all_records = [{
        "id": "g003", "submission_id": 55918053,
        "origin": str(g003), "provenance": "User supplied source; Kaggle submission identity checked separately; remote bytes not independently compared",
        "files": [{"path": str(frozen.relative_to(EXP)), "bytes": len(raw), "sha256": sha(raw)}],
        "python": [inspect_python(frozen, raw)], "native_status": "NOT_PORTED",
    }]
    for key, ref in PUBLIC.items():
        folder = POOL / key
        record = {"id": key, "url": "https://www.kaggle.com/code/" + ref,
                  "files": [], "python": [], "notebooks": [], "native_status": "NOT_PORTED"}
        for directory in (folder / "source", folder / "output"):
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or ".decoded" in path.name:
                    continue
                data = path.read_bytes()
                record["files"].append({"path": str(path.relative_to(EXP)), "bytes": len(data), "sha256": sha(data)})
                if path.suffix == ".py":
                    record["python"].append(inspect_python(path, data))
                elif path.suffix == ".ipynb":
                    nb = json.loads(data)
                    cells = []
                    for i, cell in enumerate(nb.get("cells", [])):
                        src = "".join(cell.get("source", []))
                        cells.append({"index": i, "type": cell.get("cell_type"), "chars": len(src),
                                      "preview": src[:350]})
                        if cell.get("cell_type") == "code":
                            put(folder / "inspection" / f"cell_{i:03d}.txt", src.encode("utf-8"))
                    record["notebooks"].append({"path": str(path.relative_to(EXP)), "cells": cells})
                elif path.name.endswith((".tar.gz", ".tgz", ".tar")):
                    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                        for member in archive.getmembers():
                            rel = Path(member.name)
                            if rel.is_absolute() or ".." in rel.parts or "\\" in member.name:
                                raise ValueError(f"Unsafe archive member: {member.name}")
                            if not member.isfile() or rel.suffix != ".py":
                                continue
                            if member.size > LIMIT:
                                raise ValueError("Oversized Python member")
                            out = folder / "inspection" / path.name.replace(".", "_") / rel
                            src = archive.extractfile(member).read()
                            put(out, src)
                            record["python"].append(inspect_python(out, src))
        all_records.append(record)
    doc = {"created_date": "2026-09-03", "purpose": "Frozen opponents for multi-opponent v7 development; static acquisition only",
           "no_source_executed": True, "opponents": all_records}
    put(POOL / args.inventory_name, json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"))
    print(json.dumps({"inventory": str(POOL / args.inventory_name), "opponents": [
        {"id": r["id"], "files": len(r["files"]), "python_artifacts": len(r["python"]),
         "bytes": sum(f["bytes"] for f in r["files"])} for r in all_records]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
