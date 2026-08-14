"""Build a hash-bound manifest for released local PyTorch checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for path in sorted(args.directory.glob("*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", {}) if isinstance(checkpoint, dict) else {}
        parameters = sum(int(tensor.numel()) for tensor in state.values())
        metadata = {
            key: _json_safe(value)
            for key, value in checkpoint.items()
            if key != "model"
        } if isinstance(checkpoint, dict) else {}
        entries.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "parameters": parameters,
                "metadata": metadata,
            }
        )
    manifest = {
        "format": "kaggriculture-local-checkpoints-v1",
        "count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "checkpoints": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(entries), "output": str(args.output), "total_bytes": manifest["total_bytes"]}))


if __name__ == "__main__":
    main()
