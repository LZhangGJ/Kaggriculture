#!/usr/bin/env python3
"""Build Thomas' Python-free tape/router blob via the existing MetaV4 packer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "opponents/thomas_2945/main.py"
PACKER = HERE.parent / "metav4_2965/generate_assets.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "thomas_2945.assets.bin")
    parser.add_argument("--manifest", type=Path, default=HERE / "assets.manifest.json")
    args = parser.parse_args()

    packer = load(PACKER, "thomas_asset_packer")
    source = load(SOURCE, "thomas_asset_source")
    source_bytes = SOURCE.read_bytes()
    blob, manifest = packer.build_blob(source, hashlib.sha256(source_bytes).digest())
    manifest.update(
        schema="thomas-2945-native-assets-v1",
        source=str(SOURCE.relative_to(ROOT)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
