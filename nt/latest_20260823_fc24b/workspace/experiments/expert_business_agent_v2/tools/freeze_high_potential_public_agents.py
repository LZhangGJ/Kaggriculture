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
        slug="boatlee_v20_multi_route",
        folder="boatlee__v20-adaptive-r1-multi-route-agent",
        extractor="boatlee_packed_source",
    ),
    AgentSpec(
        slug="rayk_k320_adaptive_rank1",
        folder="raykkretzschmar__kaggriculture-rank-your-agent",
        extractor="rayk_submission_b85",
    ),
    AgentSpec(
        slug="kaito_v27_midgame_reset",
        folder="kaitofukami__25-27-strict-future-v27-midgame-meta-reset",
        extractor="kaito_agent_parts",
    ),
    AgentSpec(
        slug="tetsutani_adaptive_premium_queue",
        folder="tetsutani__adaptive-farming-strategy-for-kaggriculture",
        extractor="tetsutani_top_agent_files",
    ),
    AgentSpec(
        slug="flexonafft_v59_multi_route",
        folder="flexonafft__kaggriculture-multi-route-farming-agent",
        extractor="flex_agent_payload",
    ),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assignment_map(tree: ast.Module) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = node.value
    return result


def parse_code_cells(notebook_path: Path) -> list[tuple[int, dict[str, ast.expr]]]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells: list[tuple[int, dict[str, ast.expr]]] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Kaggle notebooks can contain IPython magics such as %%writefile.
            # No notebook code is executed by this freezer.
            continue
        cells.append((index, assignment_map(tree)))
    return cells


def find_assignments(
    cells: list[tuple[int, dict[str, ast.expr]]], required: set[str]
) -> tuple[int, dict[str, ast.expr]]:
    for cell_index, assignments in cells:
        if required.issubset(assignments):
            return cell_index, assignments
    missing = ", ".join(sorted(required))
    raise RuntimeError(f"could not locate a code cell containing: {missing}")


def literal(assignments: dict[str, ast.expr], name: str) -> Any:
    return ast.literal_eval(assignments[name])


def extract_boatlee(cells: list[tuple[int, dict[str, ast.expr]]]) -> tuple[bytes, int, str]:
    cell_index, assignments = find_assignments(
        cells, {"EXPECTED_SOURCE_SHA256", "_PACKED_SOURCE"}
    )
    call = assignments["_PACKED_SOURCE"]
    if not isinstance(call, ast.Call) or not call.args:
        raise RuntimeError("Boatlee _PACKED_SOURCE is not the expected b85decode call")
    payload = ast.literal_eval(call.args[0])
    if isinstance(payload, (tuple, list)):
        payload = "".join(payload)
    if not isinstance(payload, str):
        raise RuntimeError("Boatlee packed payload is not a string")
    data = zlib.decompress(base64.b85decode(payload))
    return data, cell_index, str(literal(assignments, "EXPECTED_SOURCE_SHA256"))


def extract_rayk(cells: list[tuple[int, dict[str, ast.expr]]]) -> tuple[bytes, int, str]:
    cell_index, assignments = find_assignments(
        cells, {"SUBMISSION_B85", "SUBMISSION_SHA256"}
    )
    payload = literal(assignments, "SUBMISSION_B85")
    data = zlib.decompress(base64.b85decode(payload))
    return data, cell_index, str(literal(assignments, "SUBMISSION_SHA256"))


def extract_kaito(cells: list[tuple[int, dict[str, ast.expr]]]) -> tuple[bytes, int, str]:
    cell_index, assignments = find_assignments(
        cells, {"_AGENT_B85_PARTS", "EXPECTED_MAIN_SHA256"}
    )
    payload = "".join(literal(assignments, "_AGENT_B85_PARTS"))
    data = zlib.decompress(base64.b85decode(payload))
    return data, cell_index, str(literal(assignments, "EXPECTED_MAIN_SHA256"))


def extract_tetsutani(
    cells: list[tuple[int, dict[str, ast.expr]]]
) -> tuple[bytes, int, None]:
    cell_index, assignments = find_assignments(cells, {"TOP_AGENT_FILES"})
    files = literal(assignments, "TOP_AGENT_FILES")
    if set(files) != {"main.py"} or not isinstance(files["main.py"], str):
        raise RuntimeError("Tetsutani TOP_AGENT_FILES does not contain exactly main.py")
    return files["main.py"].encode("utf-8"), cell_index, None


def extract_flex(cells: list[tuple[int, dict[str, ast.expr]]]) -> tuple[bytes, int, str]:
    cell_index, assignments = find_assignments(
        cells, {"AGENT_PAYLOAD", "EXPECTED_AGENT_SHA256"}
    )
    payload = literal(assignments, "AGENT_PAYLOAD")
    data = zlib.decompress(base64.b85decode(payload))
    return data, cell_index, str(literal(assignments, "EXPECTED_AGENT_SHA256"))


EXTRACTORS = {
    "boatlee_packed_source": extract_boatlee,
    "rayk_submission_b85": extract_rayk,
    "kaito_agent_parts": extract_kaito,
    "tetsutani_top_agent_files": extract_tetsutani,
    "flex_agent_payload": extract_flex,
}


def freeze_one(audit_root: Path, output_root: Path, spec: AgentSpec) -> dict[str, Any]:
    source_dir = audit_root / spec.folder
    metadata_path = source_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    notebook_path = source_dir / metadata["code_file"]
    notebook_bytes = notebook_path.read_bytes()
    cells = parse_code_cells(notebook_path)
    main_bytes, source_cell, expected_main_sha = EXTRACTORS[spec.extractor](cells)
    main_sha = sha256_bytes(main_bytes)
    if expected_main_sha is not None and main_sha != expected_main_sha:
        raise RuntimeError(
            f"{spec.slug}: extracted main.py SHA256 {main_sha} != expected {expected_main_sha}"
        )
    try:
        compile(main_bytes, f"{spec.slug}/main.py", "exec")
    except SyntaxError as error:
        raise RuntimeError(f"{spec.slug}: extracted main.py does not compile: {error}") from error

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
        "expected_main_sha256": expected_main_sha,
        "expected_hash_present": expected_main_sha is not None,
        "static_extraction_only": True,
        "python_compile_ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statically freeze the high-potential public Kaggriculture agents."
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    audit_root = args.audit_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    agents = [freeze_one(audit_root, output_root, spec) for spec in SPECS]
    manifest = {
        "schema": "kaggriculture.high_potential_public_agents.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_root": str(audit_root),
        "rules": {
            "notebook_code_executed": False,
            "embedded_expected_hash_required_when_present": True,
            "python_syntax_compile_required": True,
        },
        "agents": agents,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Keep the console receipt compatible with native Windows code pages; the
    # persisted manifest above remains human-readable UTF-8.
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
