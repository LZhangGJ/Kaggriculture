#!/usr/bin/env python3
"""Freeze replay-opening actions for the native prefix runner."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path

from experiments.student_action_event_agent import production


ROOT = Path(__file__).resolve().parents[2]
MAGIC = b"KGPFXC1\0"
VERSION = 1
STEPS = 288
OPPONENTS = {
    "thomas": {
        "code": 1,
        "module": "thomas_2945_cpp_native",
        "directory": ROOT / "experiments/native_opponents/thomas_2945_cpp",
        "asset": "thomas_2945.assets.bin",
    },
    "meta": {
        "code": 2,
        "module": "metav4_2965_native",
        "directory": ROOT / "experiments/native_opponents/metav4_2965",
        "asset": "metav4_2965.assets.bin",
    },
}


def sha(path: Path) -> bytes:
    return hashlib.sha256(path.read_bytes()).digest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_artifacts(name: str):
    spec = OPPONENTS[name]
    builds = sorted((spec["directory"] / "build").glob(
        f'{spec["module"]}*.so'))
    if len(builds) != 1:
        raise RuntimeError(f"{name}: expected one native module, found {builds}")
    asset = spec["directory"] / spec["asset"]
    return spec, builds[0], asset


def encode_action(action: dict) -> bytes:
    def atom(value):
        if not value or value[0] not in production.policy._OPS:
            return 0, -1, 1
        op = production.policy._OPS.index(value[0])
        item = (production.policy._IDS.get(value[1], -1)
                if len(value) > 1 and isinstance(value[1], str) else -1)
        quantity = (value[2] if item >= 0 and len(value) > 2 else
                    value[1] if item < 0 and len(value) > 1 else 1)
        return op, item, int(quantity)

    units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    market = list(action.get("market", []))
    if len(units) > 16 or len(market) > 10:
        raise ValueError("action exceeds engine actor/order bounds")
    output = bytearray(struct.pack("<HH", len(units), len(market)))
    for value in (*units, *market):
        op, item, quantity = atom(value)
        output.extend(struct.pack("<bbi", op, item, quantity))
    return bytes(output)


def raw_doubles(values) -> bytes:
    values = list(values)
    return struct.pack(f"<{len(values)}d", *values)


def build_one(name: str, seat: int, seed: int, binary: Path,
              output: Path) -> dict:
    from fast_kaggriculture import Config, FastEnv

    spec, module_path, asset = native_artifacts(name)
    module = load_module(spec["module"], module_path)
    opponent = module.Opponent(str(asset.resolve()))
    deployment = production.replay_deployment()
    route = production.create_replay_agent(
        deployment, f"prefix_cache_{name}_{seed}_{seat}")
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    r1 = production.policy.Agent(config=config, binary_path=binary)
    r1.lib.td_student_pre_context_observation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    r1.lib.td_student_pre_context_observation.restype = ctypes.c_int
    env = FastEnv(Config(), seed)
    env.reset_raw(seed)
    records = bytearray()
    try:
        for step in range(STEPS):
            if env.step_count != step:
                raise RuntimeError("FastEnv prefix clock drift")
            current = production.policy.normalize_observation(
                env.observation(seat))
            own = route(current, {})
            own = production.sell_before_unfunded_land(current, own, {})
            rival = opponent.action(env, 1 - seat)
            records.extend(encode_action(own))
            records.extend(encode_action(rival))
            r1.observe_external(current, own)
            actions = [None, None]
            actions[seat] = own
            actions[1 - seat] = rival
            env.step_raw(actions)
        current = production.policy.normalize_observation(env.observation(seat))
        packed = production.policy._pack(current)
        if int(packed[0]) != STEPS:
            raise RuntimeError("prefix did not end at step 288")
        if r1.lib.td_activate_external(r1.handle, packed, len(packed)):
            raise RuntimeError(r1.lib.td_debug(r1.handle).decode())
        r1.external = False
        context = (ctypes.c_double * 2233)()
        if r1.lib.td_student_pre_context_observation(
                r1.handle, packed, len(packed), context, len(context)) != 2233:
            raise RuntimeError(r1.lib.td_debug(r1.handle).decode())
        settings = [float(r1.config[key]) for key in
                    production.policy._ORDER[:r1.settings_count]]
        route_id = int(opponent.route(1 - seat))
        fast_builds = sorted((ROOT / "fast_kaggriculture/python/fast_kaggriculture").glob(
            "_fast_kaggriculture*.so"))
        if len(fast_builds) != 1:
            raise RuntimeError("expected exactly one FastEnv extension")
        hashes = (
            sha(binary),
            sha(ROOT / "agent/replay_deployment.json"),
            sha(module_path),
            sha(asset),
            sha(fast_builds[0]),
            hashlib.sha256(raw_doubles(packed)).digest(),
            hashlib.sha256(raw_doubles(context)).digest(),
        )
        header = bytearray(MAGIC)
        header.extend(struct.pack(
            "<IQBBHiIII", VERSION, seed, seat, spec["code"], 0,
            route_id, STEPS, len(settings), len(packed)))
        for value in hashes:
            header.extend(value)
        header.extend(raw_doubles(settings))
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as sink:
            temporary = Path(sink.name)
            sink.write(header)
            sink.write(records)
        temporary.replace(output)
        return {
            "opponent": name,
            "opponent_code": spec["code"],
            "route": route_id,
            "seed": seed,
            "seat": seat,
            "steps": STEPS,
            "settings_count": len(settings),
            "packed_count": len(packed),
            "packed_sha256": hashes[5].hex(),
            "context_sha256": hashes[6].hex(),
            "cache": str(output.resolve()),
            "cache_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    finally:
        r1.close()
        close = getattr(route, "close", None)
        if close:
            close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=(
        ROOT / "work/agent-student-actor-owned-v3.so"))
    parser.add_argument("--seed", type=int, default=2630900000)
    parser.add_argument("--opponents", nargs="+", choices=tuple(OPPONENTS),
                        default=tuple(OPPONENTS))
    parser.add_argument("--seats", nargs="+", type=int, choices=(0, 1),
                        default=(0, 1))
    parser.add_argument("--output-dir", type=Path, default=(
        ROOT / "work/native-student-rollout/prefix-cache-v1"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    rows = []
    for offset, opponent in enumerate(args.opponents):
        seed = args.seed + offset
        for seat in args.seats:
            output = args.output_dir / f"{opponent}-seed{seed}-seat{seat}.kgpfx"
            rows.append(build_one(
                opponent, seat, seed, args.binary.resolve(), output))
    manifest = {
        "schema": "native-prefix-cache-v1",
        "binary": str(args.binary.resolve()),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "cases": rows,
    }
    target = args.manifest or args.output_dir / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
