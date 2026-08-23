from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LATEST = ROOT / "references" / "public_latest8_20260820"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def top_level_symbols(source: str) -> dict[str, list[str]]:
    tree = ast.parse(source)
    return {
        "functions": [node.name for node in tree.body if isinstance(node, ast.FunctionDef)],
        "classes": [node.name for node in tree.body if isinstance(node, ast.ClassDef)],
        "assignments": [
            target.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        ],
    }


def decoded_x540() -> tuple[str, dict[str, object]]:
    path = LATEST / "x562_latest" / "main.py"
    source = path.read_text(encoding="utf-8")
    match = re.search(
        r'_X540_SOURCE_B85\s*=\s*"""(?P<payload>.*?)"""', source, re.DOTALL
    )
    if not match:
        raise RuntimeError("X562 embedded X540 payload not found")
    decoded = base64.b85decode(match.group("payload").encode("ascii")).decode("utf-8")
    return decoded, {
        "source": str(path),
        "decoded_bytes": len(decoded.encode("utf-8")),
        "decoded_sha256": sha256_bytes(decoded.encode("utf-8")),
        **top_level_symbols(decoded),
    }


def decoded_v36() -> tuple[dict[str, str], list[dict[str, object]], object]:
    path = LATEST / "kaito_v36_latest" / "main.py"
    source = path.read_text(encoding="utf-8")
    modules_match = re.search(
        r"_V36_MODULES\s*=\s*json\.loads\(zlib\.decompress\(base64\.b85decode\(\s*\(\s*(?P<chunks>.*?)\s*\)\s*\)\)\.decode\(\"utf-8\"\)\)",
        source,
        re.DOTALL,
    )
    route_match = re.search(
        r"_V36_ROUTE\s*=\s*json\.loads\(zlib\.decompress\(base64\.b85decode\(\s*\(\s*(?P<chunks>.*?)\s*\)\s*\)\)\.decode\(\"utf-8\"\)\)",
        source,
        re.DOTALL,
    )
    if not modules_match or not route_match:
        raise RuntimeError("V36 embedded module or route payload not found")

    def join_literals(fragment: str) -> str:
        expression = ast.parse("(" + fragment + ")", mode="eval")
        value = ast.literal_eval(expression)
        if not isinstance(value, str):
            raise TypeError("expected concatenated string literal")
        return value

    modules = json.loads(
        zlib.decompress(base64.b85decode(join_literals(modules_match.group("chunks"))))
    )
    route = json.loads(
        zlib.decompress(base64.b85decode(join_literals(route_match.group("chunks"))))
    )
    inventory = []
    for name, module_source in modules.items():
        inventory.append(
            {
                "name": name,
                "bytes": len(module_source.encode("utf-8")),
                "sha256": sha256_bytes(module_source.encode("utf-8")),
                **top_level_symbols(module_source),
            }
        )
    return modules, inventory, route


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-decoded-x540", type=Path)
    parser.add_argument("--write-v36-root", type=Path)
    args = parser.parse_args()
    decoded, x540 = decoded_x540()
    if args.write_decoded_x540:
        output = args.write_decoded_x540
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(decoded, encoding="utf-8")
        x540["decoded_output"] = str(output)
    modules, v36_inventory, route = decoded_v36()
    if args.write_v36_root:
        output_root = args.write_v36_root
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        for name, module_source in modules.items():
            output = output_root / Path(*name.split(".")).with_suffix(".py")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(module_source, encoding="utf-8")
        route_output = output_root / "route.json"
        route_output.write_text(
            json.dumps(route, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    rows: dict[str, object] = {
        "x540": x540,
        "v36_modules": v36_inventory,
        "v36_route_type": type(route).__name__,
        "v36_route_length": len(route) if hasattr(route, "__len__") else None,
    }
    for slug in (
        "deniz_v111_8c4s_latest",
        "kaito_v36_latest",
        "x562_latest",
        "tetsutani_adaptive_latest",
        "flex_multi_route_latest",
    ):
        path = LATEST / slug / "main.py"
        source = path.read_text(encoding="utf-8")
        rows[slug] = {
            "source": str(path),
            "bytes": len(source.encode("utf-8")),
            "sha256": sha256_bytes(source.encode("utf-8")),
            **top_level_symbols(source),
        }
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
