#!/usr/bin/env python3
"""Compare B0_KEEP and donor-target tails after the same routed 96:216 block."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (Path(__file__).resolve().parent, CODE_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_block_mvp_routed_intent_v2 as v2


DEFAULT_OUTPUT_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827\block_mvp_target_tail_oracle"
)
TAIL_B0 = "B0_KEEP"
TAIL_TARGET = "DONOR_TARGET"


def _state_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["opponent"]), int(row["seed"]), int(row["seat"])


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _tail_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = [int(row["first_post216_failure_step"]) for row in rows]
    first_unit = [int(row["first_post216_unit_failure_step"]) for row in rows]
    first_market = [int(row["first_post216_market_failure_step"]) for row in rows]
    observed = [value for value in first if value >= v2.v1.STOP]
    return {
        "games": len(rows),
        "all_finite_719": all(
            bool(row.get("completed")) and bool(row.get("finite_rewards"))
            and int(row.get("turns", -1)) == v2.v1.HORIZON
            for row in rows
        ),
        "raw_win_rate": _mean([int(row["outcome"] == 2) for row in rows]),
        "score_rate": .5 * _mean([int(row["outcome"]) for row in rows]),
        "mean_margin": _mean([float(row["margin"]) for row in rows]),
        "mean_positional_unit_failures": _mean([
            float(row["macro_unit_failures"]) for row in rows
        ]),
        "mean_market_failures": _mean([
            float(row["macro_market_failures"]) for row in rows
        ]),
        "post216_failure_state_rate": _mean([
            int(value >= v2.v1.STOP) for value in first
        ]),
        "post216_unit_failure_state_rate": _mean([
            int(value >= v2.v1.STOP) for value in first_unit
        ]),
        "post216_market_failure_state_rate": _mean([
            int(value >= v2.v1.STOP) for value in first_market
        ]),
        "mean_first_post216_failure_step_conditional": (
            _mean(observed) if observed else None
        ),
    }


def pair_tail_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate exact pairing and emit one deterministic two-choice oracle row."""

    paired: list[dict[str, Any]] = []
    for target_id in v2.TARGET_IDS:
        b0 = {
            _state_key(row): row for row in rows
            if row["target"] == target_id and row["tail"] == TAIL_B0
        }
        target = {
            _state_key(row): row for row in rows
            if row["target"] == target_id and row["tail"] == TAIL_TARGET
        }
        if not b0 or set(b0) != set(target):
            raise ValueError(f"unpaired tail rows for {target_id}")
        for key in sorted(b0):
            left, right = b0[key], target[key]
            if left["pre_tail_joint_trace_sha256"] != right["pre_tail_joint_trace_sha256"]:
                raise AssertionError(
                    f"tail variants diverged before step 216 for {target_id} {key}"
                )
            left_rank = int(left["outcome"]), float(left["margin"])
            right_rank = int(right["outcome"]), float(right["margin"])
            choice = TAIL_TARGET if right_rank > left_rank else TAIL_B0
            chosen = right if choice == TAIL_TARGET else left
            paired.append({
                "target": target_id,
                "opponent": key[0],
                "seed": key[1],
                "seat": key[2],
                "pre_tail_joint_trace_sha256": left["pre_tail_joint_trace_sha256"],
                "b0_outcome": int(left["outcome"]),
                "target_outcome": int(right["outcome"]),
                "b0_margin": float(left["margin"]),
                "target_margin": float(right["margin"]),
                "b0_positional_unit_failures": float(left["macro_unit_failures"]),
                "target_positional_unit_failures": float(right["macro_unit_failures"]),
                "b0_market_failures": float(left["macro_market_failures"]),
                "target_market_failures": float(right["macro_market_failures"]),
                "b0_first_post216_failure_step": int(
                    left["first_post216_failure_step"]
                ),
                "target_first_post216_failure_step": int(
                    right["first_post216_failure_step"]
                ),
                "oracle_choice": choice,
                "oracle_outcome": int(chosen["outcome"]),
                "oracle_margin": float(chosen["margin"]),
                "exact_outcome_margin_tie": left_rank == right_rank,
            })
    return paired


def summarize_tail_results(
    rows: Sequence[Mapping[str, Any]], phase: str,
) -> dict[str, Any]:
    pairs = pair_tail_rows(rows)
    targets: dict[str, Any] = {}
    for target_id in v2.TARGET_IDS:
        b0_rows = [
            row for row in rows
            if row["target"] == target_id and row["tail"] == TAIL_B0
        ]
        target_rows = [
            row for row in rows
            if row["target"] == target_id and row["tail"] == TAIL_TARGET
        ]
        target_pairs = [row for row in pairs if row["target"] == target_id]
        b0_rates = _tail_rates(b0_rows)
        target_rates = _tail_rates(target_rows)
        both_fail_delays = [
            row["target_first_post216_failure_step"]
            - row["b0_first_post216_failure_step"]
            for row in target_pairs
            if row["b0_first_post216_failure_step"] >= v2.v1.STOP
            and row["target_first_post216_failure_step"] >= v2.v1.STOP
        ]
        oracle_win = _mean([
            int(row["oracle_outcome"] == 2) for row in target_pairs
        ])
        oracle_score = .5 * _mean([
            int(row["oracle_outcome"]) for row in target_pairs
        ])
        oracle_margin = _mean([
            float(row["oracle_margin"]) for row in target_pairs
        ])
        b0_losses = sum(row["b0_outcome"] == 0 for row in target_pairs)
        repaired = sum(
            row["b0_outcome"] == 0 and row["target_outcome"] > 0
            for row in target_pairs
        )
        targets[target_id] = {
            "b0_tail": b0_rates,
            "target_tail": target_rates,
            "paired_target_minus_b0": {
                "raw_win_rate_pp": 100.0 * (
                    target_rates["raw_win_rate"] - b0_rates["raw_win_rate"]
                ),
                "score_rate_pp": 100.0 * (
                    target_rates["score_rate"] - b0_rates["score_rate"]
                ),
                "mean_margin": (
                    target_rates["mean_margin"] - b0_rates["mean_margin"]
                ),
                "mean_positional_unit_failures": (
                    target_rates["mean_positional_unit_failures"]
                    - b0_rates["mean_positional_unit_failures"]
                ),
                "mean_market_failures": (
                    target_rates["mean_market_failures"]
                    - b0_rates["mean_market_failures"]
                ),
                "post216_failure_repaired_states": sum(
                    row["b0_first_post216_failure_step"] >= v2.v1.STOP
                    and row["target_first_post216_failure_step"] < 0
                    for row in target_pairs
                ),
                "post216_failure_introduced_states": sum(
                    row["b0_first_post216_failure_step"] < 0
                    and row["target_first_post216_failure_step"] >= v2.v1.STOP
                    for row in target_pairs
                ),
                "mean_first_post216_failure_delay_when_both_fail": (
                    _mean(both_fail_delays) if both_fail_delays else None
                ),
            },
            "two_choice_oracle": {
                "raw_win_rate": oracle_win,
                "score_rate": oracle_score,
                "mean_margin": oracle_margin,
                "raw_win_gain_vs_b0_pp": 100.0 * (
                    oracle_win - b0_rates["raw_win_rate"]
                ),
                "score_gain_vs_b0_pp": 100.0 * (
                    oracle_score - b0_rates["score_rate"]
                ),
                "mean_margin_gain_vs_b0": oracle_margin - b0_rates["mean_margin"],
                "target_tail_selected_states": sum(
                    row["oracle_choice"] == TAIL_TARGET for row in target_pairs
                ),
                "target_tail_selection_rate": _mean([
                    int(row["oracle_choice"] == TAIL_TARGET) for row in target_pairs
                ]),
                "exact_tie_rate": _mean([
                    int(row["exact_outcome_margin_tie"]) for row in target_pairs
                ]),
                "b0_loss_repair_fraction": repaired / max(1, b0_losses),
            },
        }
    return {
        "schema": "block-mvp-target-tail-oracle-v1",
        "phase": phase,
        "status": "diagnostic_only",
        "states": len(pairs) // len(v2.TARGET_IDS),
        "target_state_pairs": len(pairs),
        "games": len(rows),
        "all_finite_719": all(
            bool(row.get("completed")) and bool(row.get("finite_rewards"))
            and int(row.get("turns", -1)) == v2.v1.HORIZON
            for row in rows
        ),
        "targets": targets,
        "selector_trained": False,
        "causal_contrast": (
            "identical routed prefix through step 215; B0_KEEP versus donor target "
            "full-tape tail from step 216"
        ),
        "scope_limitations": [
            "This is a per-state two-choice tail oracle diagnostic, not a trained selector.",
            "The donor target tail includes its complete unit and market tape from step 216.",
            "Whole-episode audit counts include failures before and after the tail switch; first_post216 fields isolate the first observed tail-period failure.",
            "first_post216 fields are a non-gating live Python pre-step/post-fill approximation; exact whole-episode totals come from the C++ trace-replay audit.",
        ],
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# BLOCK-MVP TARGET_TAIL oracle",
        "",
        f"- phase: `{result['phase']}`",
        f"- status: `{result['status']}`",
        f"- panel: {result['states']} states / {result['games']} games",
        f"- all finite / 719 steps: `{result['all_finite_719']}`",
        "- selector trained: `False`",
        "",
        "Both variants share an identical joint trace through step 215. Only the >=216 tail differs.",
        "",
        "| target | B0 win | target win | target-B0 win pp | margin delta | positional unit failure delta | market failure delta | post216 fail B0 / target | oracle win gain pp | oracle margin gain | select target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target_id in v2.TARGET_IDS:
        row = result["targets"][target_id]
        b0 = row["b0_tail"]
        target = row["target_tail"]
        delta = row["paired_target_minus_b0"]
        oracle = row["two_choice_oracle"]
        lines.append(
            f"| {target_id} | {100*b0['raw_win_rate']:.1f}% | "
            f"{100*target['raw_win_rate']:.1f}% | {delta['raw_win_rate_pp']:.2f} | "
            f"{delta['mean_margin']:.1f} | "
            f"{delta['mean_positional_unit_failures']:.2f} | "
            f"{delta['mean_market_failures']:.2f} | "
            f"{100*b0['post216_failure_state_rate']:.1f}% / "
            f"{100*target['post216_failure_state_rate']:.1f}% | "
            f"{oracle['raw_win_gain_vs_b0_pp']:.2f} | "
            f"{oracle['mean_margin_gain_vs_b0']:.1f} | "
            f"{100*oracle['target_tail_selection_rate']:.1f}% |"
        )
    lines.extend(("", "## Scope limitations", ""))
    lines.extend(f"- {value}" for value in result["scope_limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    baseline, blocks, v1_args, manifest = v2.load_formal_targets(args.v1_root)
    targets = {
        block_id: v2.build_routed_target(
            baseline, blocks[block_id],
            lookahead=args.lookahead, grace_steps=args.grace_steps,
        )
        for block_id in v2.TARGET_IDS
    }
    donor_routes: dict[str, tuple[Mapping[str, Any], ...]] = {}
    donor_sources: dict[str, dict[str, Any]] = {}
    for block_id in v2.TARGET_IDS:
        block = blocks[block_id]
        source = block.source
        carrier = v2.CarrierRoute.from_replay(
            Path(str(source["replay_path"])), int(source["player_index"]),
            horizon=v2.v1.HORIZON,
        )
        donor_id = f"DONOR_FULL_{block_id}"
        donor_routes[donor_id] = carrier.actions
        if list(carrier.actions[v2.v1.START:v2.v1.STOP]) != list(
            block.tape[v2.v1.START:v2.v1.STOP]
        ):
            raise AssertionError(f"formal block window disagrees with donor replay: {block_id}")
        donor_sources[block_id] = {
            "route_id": donor_id,
            "replay_path": str(carrier.replay_path),
            "player_index": carrier.player_index,
        }
    splits, split_lineages = v2.v1._opponent_splits(
        v1_args.base_metadata, v1_args.random_seed
    )
    states, panel = v2.build_panel(
        args.phase, splits, v1_args.base_metadata, args.seed_base
    )
    if args.limit_states:
        states = states[:args.limit_states]
        panel["smoke_limit_states"] = args.limit_states
    panel.update({
        "states": len(states),
        "games": len(states) * len(v2.TARGET_IDS) * 2,
        "tail_variants": [TAIL_B0, TAIL_TARGET],
    })
    additional = {baseline.block_id: baseline.tape}
    additional.update({block_id: blocks[block_id].tape for block_id in v2.TARGET_IDS})
    additional.update({target.routed_id: target.masked_tape for target in targets.values()})
    additional.update(donor_routes)
    opponents = list(dict.fromkeys(str(row["opponent"]) for row in states))
    bundle = v2.v1.NativeTeammateBundle(
        v1_args.source, v1_args.base_actions, v1_args.base_metadata,
        additional_routes=additional,
        included_families=tuple((*opponents, *additional)),
    )
    baseline_index = bundle.index(baseline.block_id)
    rows: list[dict[str, Any]] = []
    for state in states:
        opponent_index = bundle.index(str(state["opponent"]))
        common = {
            "opponent": str(state["opponent"]),
            "lineage": str(state["lineage"]),
        }
        for block_id in v2.TARGET_IDS:
            target = targets[block_id]
            raw_index = bundle.index(block_id)
            routed_index = bundle.index(target.routed_id)
            donor_tail_index = bundle.index(f"DONOR_FULL_{block_id}")
            for tail, tail_index in (
                (TAIL_B0, baseline_index), (TAIL_TARGET, donor_tail_index),
            ):
                episode = v2.run_episode(
                    bundle, baseline_index, opponent_index, int(state["seed"]),
                    int(state["seat"]), "ROUTED", target,
                    raw_index, routed_index, tail_index=tail_index,
                )
                rows.append({**episode, **common, "tail": tail})
        if len(rows) % 60 == 0:
            print(f"completed {len(rows)}/{panel['games']} games", flush=True)
    paired = pair_tail_rows(rows)
    result = summarize_tail_results(rows, args.phase)
    result.update({
        "elapsed_seconds": time.perf_counter() - started,
        "panel": panel,
        "v1_root": str(args.v1_root.resolve()),
        "v1_candidate_actions_sha256": str(manifest["actions_sha256"]),
        "opponent_split_lineages": split_lineages,
        "donor_tail_sources": donor_sources,
    })
    output = args.output_root / args.phase
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("episodes.jsonl", rows), ("paired_states.jsonl", paired)):
        with (output / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "FINAL_REPORT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output / "REPORT.md", result)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, default=v2.DEFAULT_V1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase", choices=tuple(v2.PHASE_SPECS), default="dev")
    parser.add_argument("--seed-base", type=int, default=2026082800)
    parser.add_argument("--lookahead", type=int, default=24)
    parser.add_argument("--grace-steps", type=int, default=24)
    parser.add_argument("--limit-states", type=int, default=0)
    args = parser.parse_args()
    args.v1_root = args.v1_root.resolve()
    args.output_root = args.output_root.resolve()
    if not (args.v1_root / "candidate_manifest.json").is_file():
        parser.error(f"formal v1 artifacts are missing: {args.v1_root}")
    if args.lookahead < 0 or args.grace_steps < 0 or args.limit_states < 0:
        parser.error("lookahead, grace-steps, and limit-states must be non-negative")
    run_experiment(args)


if __name__ == "__main__":
    main()
