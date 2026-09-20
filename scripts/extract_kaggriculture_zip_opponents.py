#!/usr/bin/env python3
"""Extract the strong runnable public agents referenced by Kaggriculture-main.zip."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import shutil
import zlib
from pathlib import Path


def _cell(notebook: Path, index: int) -> str:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    return "".join(payload["cells"][index].get("source", ()))


def _literal_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _write(output: Path, slug: str, data: bytes, provenance: str) -> dict[str, object]:
    compile(data, str(output / f"{slug}.py"), "exec")
    path = output / f"{slug}.py"
    path.write_bytes(data)
    return {
        "slug": slug,
        "file": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("external/Kaggriculture-main"))
    parser.add_argument("--output", type=Path, default=Path("external/kaggriculture-public-opponents"))
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    notebooks = repository / "research/notebooks"
    rows: list[dict[str, object]] = []

    kaito_nb = notebooks / (
        "93-wr-vs-kaito-s-v21-1-local-tuning-experiment/"
        "93-wr-vs-kaito-s-v21-1-local-tuning-experiment.ipynb"
    )
    kaito_source = _literal_assignment(_cell(kaito_nb, 11), "AGENT_CODE").encode()
    rows.append(_write(output, "kaito_v21_1_enhanced", kaito_source, f"{kaito_nb}:cell11 AGENT_CODE"))

    rank_nb = notebooks / (
        "kaggriculture-rank-your-agent/kaggriculture-rank-your-agent.ipynb"
    )
    rank_source = _cell(rank_nb, 20)
    c70 = zlib.decompress(base64.b85decode(
        _literal_assignment(rank_source, "SUBMISSION_B85").encode("ascii")
    ))
    expected_c70 = _literal_assignment(rank_source, "SUBMISSION_SHA256")
    if hashlib.sha256(c70).hexdigest() != expected_c70:
        raise ValueError("C70 payload hash mismatch")
    rows.append(_write(output, "c70_impact_first", c70, f"{rank_nb}:cell20 SUBMISSION_B85"))

    top_nb = notebooks / (
        "kaggriculture-findings-from-zero-to-top-meta/"
        "kaggriculture-findings-from-zero-to-top-meta.ipynb"
    )
    top_source = _cell(top_nb, 46)
    c92_encoded = "".join(_literal_assignment(top_source, "_AGENT_B64_PARTS"))
    c92 = zlib.decompress(base64.b64decode(c92_encoded.encode("ascii")))
    expected_c92 = _literal_assignment(top_source, "EXPECTED_MAIN_SHA256")
    if hashlib.sha256(c92).hexdigest() != expected_c92:
        raise ValueError("C92 payload hash mismatch")
    rows.append(_write(output, "c92_weed_route_repair", c92, f"{top_nb}:cell46 _AGENT_B64_PARTS"))

    economic_nb = notebooks / (
        "kaggriculture-structured-economic-policy/"
        "kaggriculture-structured-economic-policy.ipynb"
    )
    economic_source = _cell(economic_nb, 16)
    if economic_source.startswith("%%agentfile"):
        economic_source = "\n".join(economic_source.splitlines()[1:]) + "\n"
    rows.append(_write(
        output, "structured_economic_policy", economic_source.encode(),
        f"{economic_nb}:cell16 agentfile",
    ))

    reference = repository / "reference-agents"
    for slug in ("broker_bea", "ledger_lena", "slotter_silas", "closer_cleo"):
        source = reference / f"{slug}.py"
        data = source.read_bytes()
        rows.append(_write(
            output, slug, data,
            "raykkretzschmar/kaggriculture-reference-agents strong meta_line tier",
        ))

    hashes: dict[str, str] = {}
    for row in rows:
        digest = str(row["sha256"])
        if digest in hashes:
            raise ValueError(f"duplicate agents: {hashes[digest]} and {row['slug']}")
        hashes[digest] = str(row["slug"])
    manifest = {
        "schema_version": 1,
        "source_archive": "Kaggriculture-main.zip commit 6ac7ebc0397d8bf74e95552b8c1c0fb6a030f4fe",
        "selection": "strong/reference meta-line agents; tutorial tiers 0-5 excluded",
        "agents": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
