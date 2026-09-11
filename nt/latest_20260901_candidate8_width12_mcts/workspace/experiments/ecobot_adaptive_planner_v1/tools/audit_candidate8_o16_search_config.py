#!/usr/bin/env python3
"""Re-run the 45 frozen O1.6 losses with one wider search configuration."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome, sha256
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


DECISION_DAYS = [6, 12, 18]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument(
        "--filter-receipt",
        type=Path,
        nargs="+",
        help=(
            "Optional earlier search receipts. Only cases still non-winning "
            "in every receipt are re-run, keyed by opponent/seed/seat."
        ),
    )
    parser.add_argument("--beam-width", required=True, type=int)
    parser.add_argument("--per-node-arms", required=True, type=int)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    losses = [
        row for row in baseline_payload["rows"]["depth3"]
        if float(row["margin"]) <= 0.0
    ]
    if len(losses) != 45:
        raise RuntimeError(f"expected 45 frozen losses, observed {len(losses)}")
    filtered_from = None
    if args.filter_receipt is not None:
        keep_keys = None
        for filter_path in args.filter_receipt:
            filter_payload = json.loads(filter_path.read_text(encoding="utf-8"))
            current_keys = {
                (str(row["opponent"]), int(row["seed"]), int(row["seat"]))
                for row in filter_payload["rows"]
                if float(row["margin"]) <= 0.0
            }
            keep_keys = (
                current_keys if keep_keys is None else keep_keys & current_keys
            )
        assert keep_keys is not None
        losses = [
            row for row in losses
            if (str(row["opponent"]), int(row["seed"]), int(row["seat"]))
            in keep_keys
        ]
        filtered_from = [str(path) for path in args.filter_receipt]
        if not losses:
            raise RuntimeError("filter receipt left no non-winning cases")

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    rows: list[dict] = []
    started = time.perf_counter()
    for case_index, baseline in enumerate(losses, start=1):
        opponent_name = str(baseline["opponent"])
        opponent_index = bundle.index(opponent_name)
        seed = int(baseline["seed"])
        seat = int(baseline["seat"])
        result = bundle.adaptive_executor.candidate8_sequence_oracle(
            genome,
            opponent_index,
            seed,
            DECISION_DAYS,
            seat,
            args.beam_width,
            args.per_node_arms,
            True,
            True,
        )
        selected_ranks = [int(value) for value in result["selected_rank"]]
        committed = bundle.adaptive_executor.candidate8_committed_sequence(
            genome,
            opponent_index,
            seed,
            DECISION_DAYS,
            selected_ranks,
            seat,
            True,
        )
        own = float(result["rewards"][seat])
        rival = float(result["rewards"][1 - seat])
        committed_own = float(committed["rewards"][seat])
        committed_rival = float(committed["rewards"][1 - seat])
        reward_exact = own == committed_own and rival == committed_rival
        if not reward_exact:
            raise RuntimeError(
                "committed widened sequence did not reproduce Oracle result: "
                f"{opponent_name}/{seed}/seat{seat}"
            )
        baseline_margin = float(baseline["margin"])
        final_candidate = np.asarray(
            result["final_path_candidate_rewards"], dtype=np.float64
        )
        final_rival = np.asarray(
            result["final_path_opponent_rewards"], dtype=np.float64
        )
        rows.append({
            "opponent": opponent_name,
            "seed": seed,
            "seat": seat,
            "baseline_margin": baseline_margin,
            "own_cash": own,
            "opponent_cash": rival,
            "margin": own - rival,
            "margin_gain": own - rival - baseline_margin,
            "converted_to_win": own > rival,
            "selected_ranks": selected_ranks,
            "selected_families": [
                FAMILY_NAMES[int(value)] if int(value) >= 0 else "NONE"
                for value in result["selected_family"]
            ],
            "selected_signatures": [
                int(value) for value in result["selected_signature"]
            ],
            "reported_feasible_counts": [
                int(value) for value in result["feasible_count"]
            ],
            "complete_continuations": int(result["complete_continuations"]),
            "expanded_nodes": int(result["expanded_nodes"]),
            "maximum_live_beam": int(result["maximum_live_beam"]),
            "final_complete_paths": int(len(final_candidate)),
            "final_winning_paths": int(np.sum(final_candidate > final_rival)),
            "end_overflow": int(committed["end_overflow"]),
            "avoidable_crop_losses": int(committed["avoidable_crop_losses"]),
            "avoidable_animal_losses": int(committed["avoidable_animal_losses"]),
            "hard_execution_issue": bool(
                int(committed["end_overflow"]) > 0
                or int(committed["avoidable_crop_losses"]) > 0
                or int(committed["avoidable_animal_losses"]) > 0
            ),
            "reward_exact": reward_exact,
        })
        if args.progress_every > 0 and (
            case_index % args.progress_every == 0 or case_index == len(losses)
        ):
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "label": args.label,
                "progress": case_index,
                "total": len(losses),
                "wins": sum(row["converted_to_win"] for row in rows),
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (
                    elapsed * (len(losses) - case_index) / case_index
                ),
            }), flush=True)

    margins = np.asarray([row["margin"] for row in rows], dtype=np.float64)
    gains = np.asarray([row["margin_gain"] for row in rows], dtype=np.float64)
    payload = {
        "schema": "kaggriculture.candidate8-o16-search-config.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUDIT_COMPLETE",
        "boundary": (
            "Exact actual future is used only to measure search and candidate "
            "headroom on the 45 losses frozen before O1.6."
        ),
        "parameters": {
            "label": args.label,
            "beam_width": args.beam_width,
            "per_node_arms": args.per_node_arms,
            "decision_days": DECISION_DAYS,
            "use_feasible_pool": True,
            "competitive_objective": True,
            "filtered_from": filtered_from,
        },
        "summary": {
            "cases": len(rows),
            "converted_to_win": int(np.sum(margins > 0.0)),
            "conversion_rate": float(np.mean(margins > 0.0)),
            "mean_margin": float(np.mean(margins)),
            "minimum_margin": float(np.min(margins)),
            "mean_margin_gain": float(np.mean(gains)),
            "median_margin_gain": float(np.median(gains)),
            "improved_cases": int(np.sum(gains > 0.0)),
            "improved_rate": float(np.mean(gains > 0.0)),
            "hard_execution_issue_cases": int(sum(
                row["hard_execution_issue"] for row in rows
            )),
            "all_committed_rewards_exact": all(row["reward_exact"] for row in rows),
            "complete_continuations": int(sum(
                row["complete_continuations"] for row in rows
            )),
        },
        "rows": rows,
        "simulation_seconds": time.perf_counter() - started,
        "inputs": {
            "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline)},
            "filter_receipt": None if args.filter_receipt is None else [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.filter_receipt
            ],
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(json.dumps({"simulation_seconds": payload["simulation_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
