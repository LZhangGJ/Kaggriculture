from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import shutil
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentSpec:
    slug: str
    folder: str
    extractor: str


SPECS = (
    AgentSpec(
        "deniz_v111_8c4s_latest",
        "denizeryilmaz__v111-8c4s-economic-core-premium-lead",
        "writefile_main",
    ),
    AgentSpec(
        "boatlee_v20_latest",
        "boatlee__v20-adaptive-r1-multi-route-agent",
        "packed_source",
    ),
    AgentSpec(
        "kunal_2026_v1_latest",
        "kunaldesale2408__kaggriculture-2026-v1",
        "packed_source",
    ),
    AgentSpec(
        "rayk_rank_agent_latest",
        "raykkretzschmar__kaggriculture-rank-your-agent",
        "ray_submission",
    ),
    AgentSpec(
        "kaito_v36_latest",
        "kaitofukami__106-130-multi-generation-v36-robust-hybrid",
        "kaito_payload",
    ),
    AgentSpec(
        "x562_latest",
        "stevenleehans__kaggriculture-x544-nah-i-d-win",
        "x562_source",
    ),
    AgentSpec(
        "tetsutani_adaptive_latest",
        "tetsutani__adaptive-farming-strategy-for-kaggriculture",
        "tetsutani_files",
    ),
    AgentSpec(
        "flex_multi_route_latest",
        "flexonafft__kaggriculture-multi-route-farming-agent",
        "flex_encoded",
    ),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def notebook_cells(path: Path) -> list[dict[str, Any]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return list(notebook.get("cells", []))


def code_source(cell: dict[str, Any]) -> str:
    return "".join(cell.get("source", []))


def assignment_nodes(source: str) -> dict[str, ast.expr]:
    tree = ast.parse(source)
    result: dict[str, ast.expr] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                result[target.id] = value
    return result


def safe_static_value(node: ast.expr) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
    ):
        separator = ast.literal_eval(node.func.value)
        parts = ast.literal_eval(node.args[0])
        if isinstance(separator, str) and isinstance(parts, (tuple, list)):
            return separator.join(parts)
    raise ValueError(f"unsupported static expression: {ast.dump(node)[:240]}")


def find_assignments(
    cells: list[dict[str, Any]], required: set[str]
) -> tuple[int, dict[str, ast.expr]]:
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = code_source(cell)
        try:
            assignments = assignment_nodes(source)
        except SyntaxError:
            continue
        if required.issubset(assignments):
            return index, assignments
    raise RuntimeError(f"missing static assignments: {sorted(required)}")


def extract_writefile_main(cells: list[dict[str, Any]]) -> tuple[bytes, int, str | None]:
    expected: str | None = None
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = code_source(cell)
        first, separator, rest = source.partition("\n")
        if first.strip().lower() == "%%writefile main.py" and separator:
            data = rest.encode("utf-8")
            for check_cell in cells[index + 1 :]:
                if check_cell.get("cell_type") != "code":
                    continue
                try:
                    assignments = assignment_nodes(code_source(check_cell))
                except SyntaxError:
                    continue
                if "expected_sha256" in assignments:
                    expected = str(safe_static_value(assignments["expected_sha256"]))
                    break
            return data, index, expected
    raise RuntimeError("could not locate %%writefile main.py cell")


def extract_packed_source(cells: list[dict[str, Any]]) -> tuple[bytes, int, str]:
    index, assignments = find_assignments(
        cells, {"_PACKED_SOURCE", "EXPECTED_SOURCE_SHA256"}
    )
    payload = safe_static_value(assignments["_PACKED_SOURCE"])
    expected = str(safe_static_value(assignments["EXPECTED_SOURCE_SHA256"]))
    return zlib.decompress(base64.b85decode(payload)), index, expected


def extract_ray_submission(cells: list[dict[str, Any]]) -> tuple[bytes, int, str]:
    index, assignments = find_assignments(
        cells, {"SUBMISSION_B85", "SUBMISSION_SHA256"}
    )
    payload = safe_static_value(assignments["SUBMISSION_B85"])
    expected = str(safe_static_value(assignments["SUBMISSION_SHA256"]))
    return zlib.decompress(base64.b85decode(payload)), index, expected


def extract_kaito_payload(cells: list[dict[str, Any]]) -> tuple[bytes, int, str]:
    index, assignments = find_assignments(cells, {"payload"})
    payload = safe_static_value(assignments["payload"])
    data = zlib.decompress(base64.b85decode(payload))
    return data, index, sha256_bytes(data)


def extract_x562_source(cells: list[dict[str, Any]]) -> tuple[bytes, int, str]:
    index, assignments = find_assignments(cells, {"SOURCE_B85", "EXPECTED_SHA256"})
    payload = safe_static_value(assignments["SOURCE_B85"])
    expected = str(safe_static_value(assignments["EXPECTED_SHA256"]))
    return base64.b85decode(payload.encode("ascii")), index, expected


def extract_tetsutani_files(cells: list[dict[str, Any]]) -> tuple[bytes, int, str | None]:
    index, assignments = find_assignments(cells, {"TOP_AGENT_FILES"})
    files = safe_static_value(assignments["TOP_AGENT_FILES"])
    if set(files) != {"main.py"} or not isinstance(files["main.py"], str):
        raise RuntimeError("TOP_AGENT_FILES must contain exactly a text main.py")
    data = files["main.py"].encode("utf-8")
    return data, index, None


def extract_flex_encoded(cells: list[dict[str, Any]]) -> tuple[bytes, int, str]:
    index, assignments = find_assignments(cells, {"encoded", "expected_sha256"})
    payload = safe_static_value(assignments["encoded"])
    expected = str(safe_static_value(assignments["expected_sha256"]))
    return zlib.decompress(base64.b85decode(payload)), index, expected


EXTRACTORS = {
    "writefile_main": extract_writefile_main,
    "packed_source": extract_packed_source,
    "ray_submission": extract_ray_submission,
    "kaito_payload": extract_kaito_payload,
    "x562_source": extract_x562_source,
    "tetsutani_files": extract_tetsutani_files,
    "flex_encoded": extract_flex_encoded,
}


def freeze_one(input_root: Path, output_root: Path, spec: AgentSpec) -> dict[str, Any]:
    source_dir = input_root / spec.folder
    metadata_path = source_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    notebook_path = source_dir / metadata["code_file"]
    notebook_bytes = notebook_path.read_bytes()
    cells = notebook_cells(notebook_path)
    main_bytes, source_cell, expected_sha = EXTRACTORS[spec.extractor](cells)
    main_sha = sha256_bytes(main_bytes)
    if expected_sha is not None and main_sha != expected_sha:
        raise RuntimeError(
            f"{spec.slug}: extracted hash {main_sha} does not match {expected_sha}"
        )
    compile(main_bytes, f"{spec.slug}/main.py", "exec")

    destination = output_root / spec.slug
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "main.py").write_bytes(main_bytes)
    shutil.copy2(notebook_path, destination / notebook_path.name)
    shutil.copy2(metadata_path, destination / metadata_path.name)

    return {
        "slug": spec.slug,
        "kaggle_ref": metadata["id"],
        "title": metadata["title"],
        "extractor": spec.extractor,
        "source_cell_index": source_cell,
        "notebook_file": notebook_path.name,
        "notebook_bytes": len(notebook_bytes),
        "notebook_sha256": sha256_bytes(notebook_bytes),
        "main_bytes": len(main_bytes),
        "main_sha256": main_sha,
        "expected_main_sha256": expected_sha,
        "python_compile_ok": True,
        "notebook_code_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statically freeze the eight latest public Kaggriculture agents."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    agents = [freeze_one(input_root, output_root, spec) for spec in SPECS]
    manifest = {
        "schema": "kaggriculture.latest_public8.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "rules": {
            "static_extraction_only": True,
            "embedded_hash_enforced_when_present": True,
            "python_compile_required": True,
        },
        "agents": agents,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
