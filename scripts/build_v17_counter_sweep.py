"""Build immutable sparse counter-policy variants from the frozen V17 agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


DEFAULT_VARIANTS = (
    {"name": "r5_f000", "r5_fraction": 0.00},
    {"name": "r5_f025", "r5_fraction": 0.25},
    {"name": "r5_f050", "r5_fraction": 0.50},
    {"name": "r5_f075", "r5_fraction": 0.75},
    {"name": "r5_f100", "r5_fraction": 1.00},
    {"name": "r5_f125", "r5_fraction": 1.25},
    {"name": "r5_f150", "r5_fraction": 1.50},
    {"name": "r5_f200", "r5_fraction": 2.00},
)


def _replace_constant(source: str, name: str, value: str) -> str:
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*.*$"
    rendered, count = re.subn(pattern, f"{name} = {value}", source, count=1)
    if count != 1:
        raise ValueError(f"expected exactly one {name} assignment, found {count}")
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"counter sweep already exists: {args.output_dir}")

    source = args.source.read_text(encoding="utf-8")
    entries = []
    for variant in DEFAULT_VARIANTS:
        variant = {
            "fraction": 1.0,
            "max_batch": 30,
            "min_quantity": 1,
            "start": 24,
            "clone_distance": 0,
            "price_ratio": 0.0,
            "md_fraction": 2.0,
            "r5_fraction": 1.0,
            **variant,
        }
        rendered = _replace_constant(source, "_PREEMPT_ENABLED", "True")
        rendered = _replace_constant(rendered, "_PREEMPT_FRACTION", repr(variant["fraction"]))
        rendered = _replace_constant(rendered, "_PREEMPT_MAX_BATCH", str(variant["max_batch"]))
        rendered = _replace_constant(
            rendered, "_PREEMPT_MAX_CLONE_DISTANCE", str(variant["clone_distance"])
        )
        rendered = _replace_constant(
            rendered, "_PREEMPT_MIN_PRICE_RATIO", repr(variant["price_ratio"])
        )
        rendered = _replace_constant(
            rendered, "_PREEMPT_MIN_FUTURE_QUANTITY", str(variant["min_quantity"])
        )
        rendered = _replace_constant(rendered, "_PREEMPT_START", str(variant["start"]))
        rendered = _replace_constant(rendered, "_PREEMPT_STOP", "716")
        rendered = _replace_constant(
            rendered, "_V17_MD_FRACTION", repr(variant["md_fraction"])
        )
        rendered = _replace_constant(
            rendered, "_V17_R5_FRACTION", repr(variant["r5_fraction"])
        )
        rendered = rendered.replace(
            "__version__ = 'BL-V17-R1-RC2'",
            f"__version__ = 'BL-V17-COUNTER-{variant['name']}'",
            1,
        )
        candidate_dir = args.output_dir / variant["name"]
        candidate_dir.mkdir(parents=True, exist_ok=False)
        path = candidate_dir / "main.py"
        path.write_text(rendered, encoding="utf-8")
        compile(rendered, str(path), "exec")
        entries.append(
            {
                **variant,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    manifest = {
        "kind": "v17_sparse_counter_sweep",
        "source": str(args.source.resolve()),
        "variants": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "variants": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
