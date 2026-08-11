"""Generate deterministic, full-state traces from the official simulator."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from kaggle_environments import make


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
TRACE_DIR = PROJECT / "reference" / "traces"
V16_PATH = WORKSPACE / "references" / "boatlee_v16_rc2" / "main.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def load_v16_agent(tag: str) -> Callable[[Any], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(f"boatlee_v16_trace_{tag}", V16_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V16_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def scenario_agents(name: str) -> list[Any]:
    if name == "pass_pass":
        return ["pass", "pass"]
    if name == "starter_starter":
        return ["starter", "starter"]
    if name == "v16_v16":
        return [load_v16_agent("p0"), load_v16_agent("p1")]
    raise ValueError(f"Unknown scenario: {name}")


def canonical_frame(frame_index: int, states: list[Any]) -> dict[str, Any]:
    obs0 = states[0].observation
    frame = {
        "frame": frame_index,
        "step": int(obs0.step),
        "day": int(obs0.day),
        "hour": int(obs0.hour),
        "status": [str(state.status) for state in states],
        "reward": [float(state.reward) for state in states],
        "actions": [json_safe(state.action) for state in states],
        "farms": json_safe(obs0.farms),
        "private": [json_safe(state.observation.private) for state in states],
        "market": json_safe(obs0.market),
        "town": json_safe(obs0.town),
    }
    payload = json.dumps(frame, sort_keys=True, separators=(",", ":"))
    frame["state_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return frame


def write_deterministic_gzip_jsonl(
    path: Path, records: list[dict[str, Any]]
) -> str:
    semantic_digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
                    encoded = (line + "\n").encode("utf-8")
                    semantic_digest.update(encoded)
                    text.write(line + "\n")
    return semantic_digest.hexdigest()


def generate(name: str, seed: int) -> dict[str, Any]:
    configuration = {"episodeSteps": 720, "seed": seed}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        env = make("kaggriculture", configuration=configuration, debug=True)
        env.run(scenario_agents(name))

    frames = [canonical_frame(i, states) for i, states in enumerate(env.steps)]
    if len(frames) != 720:
        raise RuntimeError(f"{name}: expected 720 frames, got {len(frames)}")
    if frames[-1]["status"] != ["DONE", "DONE"]:
        raise RuntimeError(f"{name}: terminal status is {frames[-1]['status']}")

    header = {
        "record_type": "header",
        "schema": "kaggriculture_official_trace_v1",
        "package": "kaggle-environments",
        "package_version": importlib.metadata.version("kaggle-environments"),
        "scenario": name,
        "configuration": configuration,
        "frame_count": len(frames),
        "v16_main_sha256": sha256_file(V16_PATH) if name == "v16_v16" else None,
    }
    records = [header, *({"record_type": "frame", **frame} for frame in frames)]
    output = TRACE_DIR / f"{name}_seed{seed}.jsonl.gz"
    semantic_sha = write_deterministic_gzip_jsonl(output, records)
    return {
        "scenario": name,
        "seed": seed,
        "path": str(output.relative_to(PROJECT)),
        "frames": len(frames),
        "terminal_status": frames[-1]["status"],
        "terminal_reward": frames[-1]["reward"],
        "semantic_sha256": semantic_sha,
        "file_sha256": sha256_file(output),
        "bytes": output.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["all", "pass_pass", "starter_starter", "v16_v16"],
        default="all",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=PROJECT / "receipts" / "reference_traces.json",
    )
    args = parser.parse_args()

    seeds = {"pass_pass": 0, "starter_starter": 1, "v16_v16": 0}
    names = list(seeds) if args.scenario == "all" else [args.scenario]
    rows = [generate(name, seeds[name]) for name in names]
    receipt = {
        "schema": "kaggriculture_reference_trace_receipt_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "traces": rows,
        "ok": all(row["frames"] == 720 for row in rows),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
