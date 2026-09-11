#!/usr/bin/env python3
"""Compile the six official dynamic opponents into a JAX route-router bank.

The large Python agents are mostly frozen choreographies selected at one or two
public-state decision points.  This host-side compiler extracts every reachable
stream together with the frozen routing metadata.  Runtime weed repair, sell
slot ordering and terminal liquidation remain GPU-native in ``trace_core``.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "gold_adaptive_rule_v2" / "tools"))

from build_current_gold_gpu_trace_bank import FIELDS, _encode_stream  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def ast_sha256(path: Path) -> str:
    """Hash Python semantics while ignoring comments and blank lines."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    payload = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def load_module(path: Path, label: str):
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Compiler:
    def __init__(self) -> None:
        self.arrays: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
        self.skeletons: list[dict[str, Any]] = []

    def add(self, opponent: str, route: str, stream: list[dict], source: Path) -> int:
        normalized = list(stream[:719])
        if len(normalized) != 719:
            raise ValueError(f"{opponent}/{route}: expected >=719 actions, got {len(stream)}")
        encoded = _encode_stream(normalized)
        skeleton_id = len(self.skeletons)
        for field in FIELDS:
            self.arrays[field].append(encoded[field])
        canonical = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.skeletons.append({
            "skeleton_id": skeleton_id,
            "opponent": opponent,
            "route": route,
            "source": str(source),
            "source_sha256": sha256(source),
            "action_sha256": hashlib.sha256(canonical).hexdigest().upper(),
        })
        return skeleton_id

    def add_centroid_router(
        self,
        opponent: str,
        namespace: Any,
        source: Path,
        *,
        label_prefix: str = "",
    ) -> dict[str, Any]:
        streams = namespace["_CGR_STREAMS"] if isinstance(namespace, dict) else namespace._CGR_STREAMS
        centroids = namespace["_CGR_CENTROIDS"] if isinstance(namespace, dict) else namespace._CGR_CENTROIDS
        feature_mean = namespace["_CGR_FEATURE_MEAN"] if isinstance(namespace, dict) else namespace._CGR_FEATURE_MEAN
        feature_scale = namespace["_CGR_FEATURE_SCALE"] if isinstance(namespace, dict) else namespace._CGR_FEATURE_SCALE
        decision_step = namespace["_CGR_DECISION_STEP"] if isinstance(namespace, dict) else namespace._CGR_DECISION_STEP
        routes = list(centroids)
        ids = [self.add(opponent, f"{label_prefix}{route}", streams[route], source) for route in routes]
        values = [[float(value) for value in centroids[route]] for route in routes]
        if not values or any(len(row) != 48 for row in values):
            raise ValueError(f"{opponent}: centroid feature count is not 48")
        return {
            "kind": "centroid",
            "decision_step": int(decision_step),
            "route_names": routes,
            "skeleton_ids": ids,
            "centroids": values,
            "feature_mean": [float(value) for value in feature_mean],
            "feature_scale": [float(value) for value in feature_scale],
            "default_skeleton_id": ids[0],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opponents",
        type=Path,
        default=ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "configs" / "dynamic_opponents_feasibility_v1.json",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "configs" / "fixed_routes_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "artifacts" / "jax_dynamic_opponent_bank_v1.npz",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "receipts" / "jax_dynamic_opponent_bank_v1.json",
    )
    args = parser.parse_args()
    opponents = json.loads(resolve(args.opponents).read_text(encoding="utf-8"))["opponents"]
    candidates = json.loads(resolve(args.candidates).read_text(encoding="utf-8"))["candidates"]
    compiler = Compiler()

    candidate_ids = []
    for index, row in enumerate(candidates):
        path = resolve(Path(row["path"]))
        module = load_module(path, f"jax_dynamic_candidate_{index}")
        # Generated fixed-route wrappers keep the selected route in
        # ``_FCGR_ROUTE`` while the embedded parent router intentionally leaves
        # ``_CGR_FIXED_ROUTE`` as None.  Prefer the wrapper selector.
        fixed_route = getattr(
            module,
            "_FCGR_ROUTE",
            getattr(module, "_CGR_FIXED_ROUTE", None),
        )
        streams = getattr(module, "_CGR_STREAMS", None)
        stream = streams.get(fixed_route) if isinstance(streams, dict) and fixed_route in streams else None
        # Standalone fixed-route agents expose their already-selected stream as
        # ``_ACTIONS`` instead of the multi-route ``_CGR_STREAMS`` interface.
        # Accepting that representation lets the same compiler build large
        # route-screening banks without changing agent semantics.
        if not isinstance(stream, list):
            stream = getattr(module, "_ACTIONS", None)
        if not isinstance(stream, list):
            raise ValueError(f"candidate has no fixed action stream: {path}")
        candidate_ids.append(compiler.add("candidate", str(row["name"]), stream, path))

    specs: list[dict[str, Any]] = []
    for opponent_id, row in enumerate(opponents):
        name = str(row["name"])
        path = resolve(Path(row["path"]))
        module = load_module(path, f"jax_dynamic_opponent_{opponent_id}")
        if name == "public_g01_rc2":
            reference = ROOT / "references" / "boatlee_v16_rc2" / "main.py"
            if ast_sha256(path) != ast_sha256(reference):
                raise ValueError(
                    "public_g01_rc2 differs semantically from the exact Boatlee "
                    "GPU port's frozen source"
                )
            # The exact policy uses its dedicated trace and five controller
            # state machines.  ``default_skeleton_id`` is an unused fixed-shape
            # placeholder for the generic router carry.
            spec = {
                "kind": "boatlee_v16_exact",
                "default_skeleton_id": candidate_ids[0],
                "reference_source": str(reference),
                "reference_source_sha256": sha256(reference),
                "source_ast_sha256": ast_sha256(path),
                "exact_trace": str(
                    ROOT
                    / "experiments"
                    / "strategic_v5"
                    / "artifacts"
                    / "boatlee_v16"
                    / "boatlee_v16_rc2_trace_v1.npz"
                ),
                "parity_receipt": str(
                    ROOT
                    / "experiments"
                    / "strategic_v5"
                    / "receipts"
                    / "h1c_boatlee_v16_gpu_parity_v1.json"
                ),
            }
        elif name == "public04_read_market":
            low = compiler.add(name, "low", module._E279_LOW_ACTIONS, path)
            high = compiler.add(name, "high", module._E279_HIGH_ACTIONS, path)
            spec = {
                "kind": "shop_dominance",
                "decision_step": int(module._E279_DECISION_STEP),
                "skeleton_ids": [low, high],
                "default_skeleton_id": low,
            }
        elif name == "public06_soil_rain":
            stream = module._MODAL_NS.get("_ACTIONS") or module._MODAL_NS.get("_ROUTE")
            if not isinstance(stream, list):
                raise ValueError("public06 embedded modal parent has no _ACTIONS")
            fixed = compiler.add(name, "modal", stream, path)
            spec = {
                "kind": "public_g04_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_g06_v25":
            if module._REBALANCE_ACTIONS is not module._LEGACY_ACTIONS:
                raise ValueError("public G06 V25 requires one shared legacy/rebalance tape")
            fixed = compiler.add(name, "v25", module._LEGACY_ACTIONS, path)
            spec = {
                "kind": "public_v25_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_g08_v14":
            fixed = compiler.add(name, "v14_h6", module._ACTIONS, path)
            if tuple(module._PREEMPT_HORIZONS) != (6, 5, 4, 3, 2, 1):
                raise ValueError("public G08 V14 exact port requires horizon ladder 6..1")
            spec = {
                "kind": "public_v14_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_g11_v21":
            fixed = compiler.add(name, "v21_1", module._ACTIONS, path)
            product_names = (
                "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER",
            )
            signatures = []
            sales = []
            for prototype in module._PROTOTYPES:
                packed_rows = []
                sales_rows = []
                for signature, predicted in zip(
                    prototype["signatures"], prototype["sales"], strict=True
                ):
                    packed = [
                        int(signature["workers"]),
                        int(signature["unlocks"]),
                        *[int(value) for value in signature["positions"]],
                        *[int(value) for value in signature["counts"]],
                        *[int(value) for value in signature["yields"]],
                    ]
                    if len(packed) != 47:
                        raise ValueError(f"public G11 V21 signature width {len(packed)} != 47")
                    packed_rows.append(packed)
                    sales_rows.append([str(item) in predicted for item in product_names])
                if len(packed_rows) != 719 or len(sales_rows) != 719:
                    raise ValueError("public G11 V21 prototypes require 719 rows")
                signatures.append(packed_rows)
                sales.append(sales_rows)
            spec = {
                "kind": "public_v21_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "prototype_signature": signatures,
                "prototype_sales": sales,
                "memory_max_distance": float(module._MEMORY_MAX_DISTANCE),
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_g12_v13_r3":
            fixed = compiler.add(name, "v13_r3", module._ACTIONS, path)
            premium = tuple(module._PREMIUM)
            hazard_enabled = np.zeros((719, len(premium)), dtype=np.bool_)
            hazard_cap = np.zeros((719, len(premium)), dtype=np.int16)
            for step_text, rows in module._GOLD_HAZARD.items():
                step = int(step_text)
                if not 0 <= step < 719:
                    continue
                for item, probability, expected, _count in rows:
                    if item not in premium or float(probability) < float(module._PREEMPT_THRESHOLD):
                        continue
                    ordinal = premium.index(item)
                    hazard_enabled[step, ordinal] = True
                    hazard_cap[step, ordinal] = max(
                        1, int(round(float(expected) * float(module._PREEMPT_FRACTION)))
                    )
            spec = {
                "kind": "public_v13_r3_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "hazard_enabled": hazard_enabled.tolist(),
                "hazard_cap": hazard_cap.tolist(),
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_g09_c68_thunder":
            fixed = compiler.add(name, "c68_thunder", module._ACTIONS, path)
            if int(module._ADAPT_MAX_OPP_HORIZON) != 6:
                raise ValueError("public G09 C68 exact port requires max learned opponent horizon 6")
            spec = {
                "kind": "public_c68_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_g10_four_hire":
            parent = module._V8._V7.V6.PARENT
            kaito = parent._KAITO
            parent_ray = parent._RAY
            outer_ray = module._V8._RAY
            if kaito._REBALANCE_ACTIONS is not kaito._LEGACY_ACTIONS:
                raise ValueError("public G10 embedded V25 tapes are not shared")
            if parent_ray._ACTIONS != outer_ray._ACTIONS:
                raise ValueError("public G10 embedded C68 tapes differ")
            if int(parent_ray._ADAPT_MAX_OPP_HORIZON) != 6:
                raise ValueError("public G10 embedded C68 requires max learned horizon 6")
            kaito_id = compiler.add(name, "embedded_v25", kaito._LEGACY_ACTIONS, path)
            ray_id = compiler.add(name, "embedded_c68", parent_ray._ACTIONS, path)
            spec = {
                "kind": "public_four_hire_exact",
                "default_skeleton_id": kaito_id,
                "skeleton_ids": [kaito_id, ray_id],
                "kaito_skeleton_id": kaito_id,
                "ray_skeleton_id": ray_id,
                "kaito_action_sha256": compiler.skeletons[kaito_id]["action_sha256"],
                "ray_action_sha256": compiler.skeletons[ray_id]["action_sha256"],
            }
        elif name == "public_g16_tran_cashflow":
            fixed = compiler.add(name, "tran_cashflow", module.TRACE_ACTIONS, path)
            if str(module._TRAN_CASHFLOW_SELL_MODE) != "notional":
                raise ValueError("public G16 exact port requires notional sell sorting")
            spec = {
                "kind": "public_tran_cashflow_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_g13_bruce_route1":
            fixed = compiler.add(name, "bruce_route1", module.TRACE_ACTIONS, path)
            if int(module.TERMINAL_START) != 716:
                raise ValueError("public G13 exact port requires terminal start 716")
            spec = {
                "kind": "public_bruce_route1_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_g14_v19_control":
            runtime = module._V19_RUNTIME
            if float(runtime["distance_strength"]) != 0.0:
                raise ValueError("public G14 V19 exact port requires zero distance strength")
            if float(runtime["stay_bonus"]) != 0.0:
                raise ValueError("public G14 V19 exact port requires zero stay bonus")
            expert_name = "himanshu_kumar"
            if any(
                runtime["board_by_seat"][str(seat)] != expert_name
                for seat in (0, 1)
            ):
                raise ValueError("public G14 V19 board expert changed")
            for seat in (0, 1):
                bias = runtime["market_bias_by_seat"][str(seat)]
                selected = max((float(score), name_) for name_, score in bias.items())[1]
                if selected != expert_name:
                    raise ValueError("public G14 V19 market expert changed")
            fixed = compiler.add(
                name, expert_name, runtime["experts"][expert_name]["actions"], path
            )
            spec = {
                "kind": "public_v19_control_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "expert_name": expert_name,
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_g15_v18_closed_loop":
            runtime = module._V18_RUNTIME
            if float(runtime.get("board_distance_strength", 0.0)) != 0.0:
                raise ValueError("public G15 V18 exact port requires fixed board route")
            if bool(module.STRATEGY.get("fixed_board_adaptation")):
                raise ValueError("public G15 V18 exact port requires disabled board adaptation")
            expert_names = list(runtime["experts"])
            expert_ids = [
                compiler.add(
                    name,
                    expert_name,
                    runtime["experts"][expert_name]["actions"],
                    path,
                )
                for expert_name in expert_names
            ]
            board_names = [runtime["board_by_seat"][str(seat)] for seat in (0, 1)]
            if board_names[0] != board_names[1]:
                raise ValueError("public G15 V18 requires one shared board expert")
            board_id = expert_ids[expert_names.index(board_names[0])]
            prototypes = [
                runtime["experts"][expert_name]["prototypes_by_day"]
                for expert_name in expert_names
            ]
            spec = {
                "kind": "public_v18_closed_loop_exact",
                "default_skeleton_id": board_id,
                "skeleton_ids": expert_ids,
                "expert_names": expert_names,
                "board_skeleton_id": board_id,
                "v18_feature_scale": runtime["feature_standardization"]["scale"],
                "market_bias_by_seat": [
                    [
                        float(runtime["market_bias_by_seat"][str(seat)][expert_name])
                        for expert_name in expert_names
                    ]
                    for seat in (0, 1)
                ],
                "prototypes_by_day": prototypes,
                "distance_strength": float(runtime["distance_strength"]),
                "stay_bonus": float(runtime["stay_bonus"]),
            }
        elif name == "public_g05_structured_econ":
            seat_streams = getattr(module, "_G05_SEAT_STREAMS", None)
            if not isinstance(seat_streams, dict) or set(seat_streams) != {"0", "1"}:
                raise ValueError(
                    "public G05 JAX coverage requires the explicit two-seat "
                    "trace-medoid proxy artifact"
                )
            seat_ids = [
                compiler.add(name, f"seat{seat}", seat_streams[str(seat)], path)
                for seat in (0, 1)
            ]
            spec = {
                "kind": "public_g05_trace_proxy",
                "default_skeleton_id": seat_ids[0],
                "skeleton_ids": seat_ids,
                "seat_skeleton_ids": seat_ids,
                "acceptance_boundary": getattr(
                    module,
                    "_G05_PROXY_BOUNDARY",
                    "APPROXIMATE_NOT_STEPWISE_EXACT",
                ),
            }
        elif name == "public14_rank_agent_v17":
            fixed = compiler.add(name, "v17", module._ACTIONS, path)
            if not bool(module._V17_MARKET) or int(module._V17_HORIZON) != 4:
                raise ValueError("public G02 exact port requires V17 market horizon 4")
            if bool(module._V17_TERMINAL):
                raise ValueError("public G02 exact port does not implement disabled V17 terminal override")
            premium_names = ("MELON", "MILK", "STRAWBERRY", "WOOL")
            rival_schedule = np.zeros((719, len(premium_names)), dtype=np.int16)
            for step, rows in module._V17_RIVAL_SCHEDULE.items():
                if not 0 <= int(step) < rival_schedule.shape[0]:
                    raise ValueError(f"public G02 rival schedule step out of range: {step}")
                for item, quantity in rows:
                    rival_schedule[int(step), premium_names.index(str(item))] = int(quantity)
            spec = {
                "kind": "public_g02_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "rival_schedule": rival_schedule.tolist(),
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif name == "public_opening_router_v8":
            prt = compiler.add_centroid_router(name, module._PRT, path, label_prefix="prt/")
            lgbm = compiler.add_centroid_router(name, module._LGBM, path, label_prefix="rank1/")
            spec = {
                "kind": "nested_tree",
                "decision_step": int(module._DECISION_STEP),
                "tree": module._TREE,
                "alternate_label": str(module._ALTERNATE_LABEL),
                "prt": prt,
                "alternate": lgbm,
                "default_skeleton_id": prt["default_skeleton_id"],
            }
        elif name == "local_prt_v6":
            route_names = list(module._PRT_V6_ROUTES)
            route_ids = [
                compiler.add(name, route, module._CGR_STREAMS[route], path)
                for route in route_names
            ]
            spec = {
                "kind": "prt_tree",
                "decision_step": int(module._CGR_DECISION_STEP),
                "route_names": route_names,
                "skeleton_ids": route_ids,
                "tree": module._PRT_V6_TREE,
                "default_skeleton_id": route_ids[0],
            }
        elif name == "public_g07_c95":
            # C95 is one frozen 719-step choreography wrapped by public-state
            # controllers (clone front-running, score-bank fallback, weed
            # recovery and one-turn sale splitting).  Keep the raw tape in the
            # shared bank; the controllers are reproduced GPU-native by the
            # dedicated C95 runtime.
            fixed = compiler.add(name, "c95", module._TRACE, path)
            spec = {
                "kind": "c95_exact",
                "default_skeleton_id": fixed,
                "skeleton_ids": [fixed],
                "action_trace_sha256": compiler.skeletons[fixed]["action_sha256"],
            }
        elif hasattr(module, "_CGR_STREAMS") and len(module._CGR_STREAMS) == 1:
            # Some Replay-derived GOLD agents have only one reachable route.
            # Their generated module intentionally carries an empty centroid
            # table because no state-conditioned selection is required.
            route, stream = next(iter(module._CGR_STREAMS.items()))
            fixed = compiler.add(name, str(route), stream, path)
            spec = {"kind": "fixed", "default_skeleton_id": fixed}
        elif hasattr(module, "_CGR_STREAMS") and hasattr(module, "_CGR_CENTROIDS"):
            spec = compiler.add_centroid_router(name, module, path)
        else:
            raise ValueError(f"unsupported dynamic opponent {name}")
        spec.update({"opponent_id": opponent_id, "name": name, "source": str(path), "source_sha256": sha256(path)})
        specs.append(spec)

    arrays = {field: np.stack(values, axis=0) for field, values in compiler.arrays.items()}
    output = resolve(args.output)
    receipt_path = resolve(args.receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    receipt = {
        "schema": "kawashigi-jax-dynamic-opponent-bank-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPILED_NOT_YET_PARITY_ACCEPTED",
        "candidate_ids": candidate_ids,
        "candidate_names": [str(row["name"]) for row in candidates],
        "opponents": specs,
        "skeletons": compiler.skeletons,
        "shapes": {field: list(value.shape) for field, value in arrays.items()},
        "output": str(output),
    }
    receipt["output_sha256"] = sha256(output)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "candidate_ids": candidate_ids,
        "opponent_count": len(specs),
        "skeleton_count": len(compiler.skeletons),
        "output": str(output),
        "receipt": str(receipt_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
