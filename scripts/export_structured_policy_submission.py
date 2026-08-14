"""Export a route-prior PolicyV2 checkpoint as a dependency-free Kaggle agent."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import tarfile
import zlib
from pathlib import Path

import torch

from kaggriculture_lab.gpu_policy import UNIT_ACTIONS
from kaggriculture_lab.policy_v2 import MARKET_TOKENS


def _action_tables(checkpoint: dict) -> list[list[dict]]:
    state = checkpoint["model"]
    required = (
        "unit_route_logits",
        "unit_quantity_route_logits",
        "market_route_logits",
        "market_quantity_route_logits",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"checkpoint has no fitted route prior: {missing}")

    unit = state["unit_route_logits"].argmax(-1).cpu()
    unit_quantity = state["unit_quantity_route_logits"].argmax(-1).cpu()
    market = state["market_route_logits"].argmax(-1).cpu()
    market_quantity = state["market_quantity_route_logits"].argmax(-1).cpu()
    tables: list[list[dict]] = [[], []]
    for seat in range(2):
        for step in range(720):
            unit_actions = []
            for slot in range(unit.shape[2]):
                op, item = UNIT_ACTIONS[int(unit[step, seat, slot])]
                action = [op] if item is None else [op, item]
                if op in ("PICKUP", "PLACE"):
                    action.append(max(1, int(unit_quantity[step, seat, slot])))
                unit_actions.append(action)
            market_actions = []
            for slot in range(market.shape[2]):
                op, item = MARKET_TOKENS[int(market[step, seat, slot])]
                if op == "NONE":
                    continue
                action = [op] if item is None else [
                    op,
                    item,
                    max(1, int(market_quantity[step, seat, slot])),
                ]
                market_actions.append(action)
            tables[seat].append(
                {
                    "farmer": unit_actions[0],
                    "hands": unit_actions[1:],
                    "market": market_actions,
                }
            )
    return tables


def _main_source(tables: list[list[dict]], checkpoint_sha256: str) -> str:
    raw = json.dumps(tables, separators=(",", ":")).encode("utf-8")
    payload = base64.b85encode(zlib.compress(raw, level=9)).decode("ascii")
    return f'''"""BC route-s40 submission exported from checkpoint {checkpoint_sha256}."""
import base64
import copy
import json
import zlib

_ROUTES = json.loads(zlib.decompress(base64.b85decode({payload!r})).decode("utf-8"))
CHECKPOINT_SHA256 = {checkpoint_sha256!r}


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def agent(obs, configuration=None):
    seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    step = min(719, max(0, int(_get(obs, "step", 0) or 0)))
    action = copy.deepcopy(_ROUTES[seat][step])
    farms = list(_get(obs, "farms", []) or [])
    farm = farms[seat] if seat < len(farms) else {{}}
    hands = len(_get(farm, "hands", []) or [])
    action["hands"] = action["hands"][:hands]
    if len(action["hands"]) < hands:
        action["hands"].extend([["PASS"] for _ in range(hands - len(action["hands"]))])
    return action
'''


def _full_main_source(tables: list[list[dict]], checkpoint_sha256: str) -> str:
    raw = json.dumps(tables, separators=(",", ":")).encode("utf-8")
    payload = base64.b85encode(zlib.compress(raw, level=9)).decode("ascii")
    return f'''"""Full PolicyV2 BC with asynchronous load and learned-route fallback."""
import base64
import copy
import gc
import json
from pathlib import Path
import threading
import zlib

_ROUTES = json.loads(zlib.decompress(base64.b85decode({payload!r})).decode("utf-8"))
CHECKPOINT_SHA256 = {checkpoint_sha256!r}
_MODEL = None
_TORCH = None
_POLICY_BATCH = None
_LOAD_ERROR = None


def _load_model():
    global _MODEL, _TORCH, _POLICY_BATCH, _LOAD_ERROR
    try:
        import torch
        from kaggriculture_lab.policy_v2 import policy_from_checkpoint, structured_policy_batch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        device = torch.device("cpu")
        try:
            checkpoint = torch.load(Path("model.pt"), map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(Path("model.pt"), map_location=device)
        model = policy_from_checkpoint(checkpoint, device).eval()
        del checkpoint
        gc.collect()
        _TORCH = torch
        _POLICY_BATCH = structured_policy_batch
        _MODEL = model
    except BaseException as error:
        _LOAD_ERROR = repr(error)


threading.Thread(target=_load_model, daemon=True).start()


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _route_action(obs):
    seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    step = min(719, max(0, int(_get(obs, "step", 0) or 0)))
    action = copy.deepcopy(_ROUTES[seat][step])
    farms = list(_get(obs, "farms", []) or [])
    farm = farms[seat] if seat < len(farms) else {{}}
    hands = len(_get(farm, "hands", []) or [])
    action["hands"] = action["hands"][:hands]
    if len(action["hands"]) < hands:
        action["hands"].extend([["PASS"] for _ in range(hands - len(action["hands"]))])
    return action


def agent(obs, configuration=None):
    model = _MODEL
    if model is None:
        return _route_action(obs)
    with _TORCH.inference_mode():
        return _POLICY_BATCH(
            model,
            [obs],
            "cpu",
            deterministic=True,
            mask_unit_actions=False,
        ).actions[0]
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("route", "full"), default="route")
    args = parser.parse_args()

    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.mode == "route":
        tables = _action_tables(checkpoint)
        source = _main_source(tables, checkpoint_sha256)
    else:
        tables = _action_tables(checkpoint)
        source = _full_main_source(tables, checkpoint_sha256)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_path = args.output_dir / "main.py"
    archive_path = args.output_dir / "submission.tar.gz"
    main_path.write_text(source, encoding="utf-8", newline="\n")
    compile(source, str(main_path), "exec")
    archive_files = [(main_path, "main.py")]
    if args.mode == "full":
        model_path = args.output_dir / "model.pt"
        package_dir = args.output_dir / "kaggriculture_lab"
        package_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.checkpoint, model_path)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        source_root = Path(__file__).resolve().parents[1] / "src" / "kaggriculture_lab"
        for name in ("gpu_policy.py", "policy_v2.py"):
            shutil.copy2(source_root / name, package_dir / name)
        archive_files.extend(
            [
                (model_path, "model.pt"),
                (package_dir / "__init__.py", "kaggriculture_lab/__init__.py"),
                (package_dir / "gpu_policy.py", "kaggriculture_lab/gpu_policy.py"),
                (package_dir / "policy_v2.py", "kaggriculture_lab/policy_v2.py"),
            ]
        )
    with tarfile.open(archive_path, "w:gz") as archive:
        for path, arcname in archive_files:
            archive.add(path, arcname=arcname)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getnames()
    expected_members = [arcname for _, arcname in archive_files]
    if members != expected_members:
        raise RuntimeError(f"unexpected archive members: {members}")

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "mode": args.mode,
                "main": str(main_path),
                "main_bytes": main_path.stat().st_size,
                "main_sha256": hashlib.sha256(main_path.read_bytes()).hexdigest(),
                "archive": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "archive_members": members,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
