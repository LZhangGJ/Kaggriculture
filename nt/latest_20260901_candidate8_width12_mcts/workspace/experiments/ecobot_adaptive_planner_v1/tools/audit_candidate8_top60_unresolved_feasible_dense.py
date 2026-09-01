#!/usr/bin/env python3
"""Audit the nine Top-60 misses with the full feasible pool and denser days."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


STAGES = (
    ("FEASIBLE_8D_W12", [0, 1, 3, 6, 9, 12, 18, 24]),
    ("FEASIBLE_15D_W12", [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24, 27]),
    ("FEASIBLE_30D_W12", list(range(30))),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def names(values: list[int]) -> list[str]:
    return ["NONE" if int(value) < 0 else FAMILY_NAMES[int(value)] for value in values]


def key(row: dict[str, Any]) -> tuple[float, float, float]:
    margin = float(row["margin"])
    return (
        1.0 if margin > 0 else (0.5 if margin == 0 else 0.0),
        margin,
        float(row["own_cash"]),
    )


def search(
    bundle: NativeTeammateBundle,
    genome,
    case: dict[str, Any],
    stage_name: str,
    decision_days: list[int],
    beam_width: int,
    maximum_arms: int,
) -> dict[str, Any]:
    opponent = bundle.index(case["family"])
    seat = int(case["candidate_seat"])
    seed = int(case["official_seed"])
    started = time.perf_counter()
    raw = bundle.adaptive_executor.candidate8_sequence_oracle(
        genome,
        opponent,
        seed,
        decision_days,
        seat,
        beam_width,
        maximum_arms,
        True,
        True,
    )
    elapsed = time.perf_counter() - started
    own = float(raw["rewards"][seat])
    rival = float(raw["rewards"][1 - seat])
    row = {
        "route_id": case["route_id"],
        "rank": int(case["rank"]),
        "team_name": case["team_name"],
        "submission_id": int(case["submission_id"]),
        "near_family_id": case["near_family_id"],
        "member_count": int(case["member_count"]),
        "representative_episode_id": int(case["representative_episode_id"]),
        "stage": stage_name,
        "decision_days": [int(value) for value in raw["decision_day"]],
        "selected_ranks": [int(value) for value in raw["selected_rank"]],
        "selected_families": names(raw["selected_family"]),
        "selected_path_available_counts": [int(value) for value in raw["feasible_count"]],
        "maximum_selected_path_available_count": max(
            (int(value) for value in raw["feasible_count"]), default=0
        ),
        "beam_width": beam_width,
        "maximum_arms": maximum_arms,
        "candidate_pool": "feasible",
        "own_cash": own,
        "opponent_cash": rival,
        "margin": own - rival,
        "win": own > rival,
        "seconds": elapsed,
        "expanded_nodes": int(raw["expanded_nodes"]),
        "complete_continuations": int(raw["complete_continuations"]),
        "maximum_live_beam": int(raw["maximum_live_beam"]),
    }
    committed = bundle.adaptive_executor.candidate8_committed_sequence(
        genome,
        opponent,
        seed,
        row["decision_days"],
        row["selected_ranks"],
        seat,
        True,
    )
    row["committed_rewards"] = [float(value) for value in committed["rewards"]]
    row["replay_exact"] = tuple(row["committed_rewards"]) == tuple(
        float(value) for value in raw["rewards"]
    )
    if not row["replay_exact"]:
        raise RuntimeError(f"committed replay mismatch: {case['route_id']} {stage_name}")
    return row


def load_stage_checkpoint(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                output[(row["route_id"], row["stage"])] = row
    return output


def baseline_row(row: dict[str, Any]) -> dict[str, Any]:
    selected = dict(row["selected"])
    selected.update(
        {
            "stage": f"SHORTLIST_{row['selected_width']}D8",
            "candidate_pool": "shortlist",
            "maximum_arms": 64,
            "replay_exact": True,
        }
    )
    return selected


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Candidate8 九条未解路线：完整 feasible 与密节点上限审计",
        "",
        f"生成时间：{payload['created_at_utc']}",
        "",
        "## 结论",
        "",
        f"- 原未解：{summary['input_unresolved_cases']} 条。",
        f"- 新救回：**{summary['newly_resolved_cases']}** 条。",
        f"- 最终仍未解：**{summary['remaining_unresolved_cases']}** 条。",
        f"- 9条内命中率：**{summary['resolved_rate_within_nine']:.2%}**。",
        f"- 与上一轮合并后的总代表路线覆盖：**{summary['combined_family_win_rate']:.2%}**。",
        f"- 按原始家族成员数投影后的总覆盖：**{summary['combined_implied_weighted_win_rate']:.2%}**。",
        "",
        "> 本实验仍是知道真实 seed 与冻结对手动作带的离线 exact-future Oracle，不是线上选择器。",
        "",
        "## 分路线结果",
        "",
        "|Rank|选手|Family|原分差|最佳阶段|新分差|结果|最大可行候选数|",
        "|---:|---|---|---:|---|---:|---|---:|",
    ]
    for row in payload["rows"]:
        best = row["best"]
        lines.append(
            f"|{row['rank']}|{row['team_name']}|{row['near_family_id']}|"
            f"{row['baseline']['margin']:.0f}|{best['stage']}|{best['margin']:.0f}|"
            f"{'WIN' if best['win'] else 'LOSS'}|"
            f"{best.get('maximum_selected_path_available_count', 64)}|"
        )
    lines.extend(
        [
            "",
            "## 阶段说明",
            "",
            "- FEASIBLE_8D_W12：原8个决策日，完整可行候选池。",
            "- FEASIBLE_15D_W12：增加第2、4、5、8、10、15、21、27天等节点。",
            "- FEASIBLE_30D_W12：每天都允许修改宏观计划。",
            "- 某一级找到胜路后停止继续加密；未获胜才进入下一级。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--baseline-receipt", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--maximum-arms", type=int, default=4096)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_root / "feasible_dense_stage_results.jsonl"
    checkpoint = load_stage_checkpoint(checkpoint_path)
    all_cases = {
        row["route_id"]: row
        for row in json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    }
    baseline_payload = json.loads(args.baseline_receipt.read_text(encoding="utf-8"))
    baseline_unresolved = [
        row for row in baseline_payload["rows"] if not row["found_winning_path"]
    ]
    if len(baseline_unresolved) != 9:
        raise RuntimeError(f"expected nine baseline misses, got {len(baseline_unresolved)}")

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    genome = load_genome(args.genomes, args.genome_index)
    rows = []
    for case_index, old in enumerate(baseline_unresolved, start=1):
        case = all_cases[old["route_id"]]
        best = baseline_row(old)
        stage_rows = []
        for stage_name, decision_days in STAGES:
            if best["win"]:
                break
            checkpoint_key = (case["route_id"], stage_name)
            row = checkpoint.get(checkpoint_key)
            if row is None:
                row = search(
                    bundle,
                    genome,
                    case,
                    stage_name,
                    decision_days,
                    args.beam_width,
                    args.maximum_arms,
                )
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                checkpoint[checkpoint_key] = row
            stage_rows.append(row)
            best = max((best, row), key=key)
            print(
                f"case={case_index}/9 route={case['route_id']} stage={stage_name} "
                f"margin={row['margin']:.0f} best={best['margin']:.0f} "
                f"available_max={row['maximum_selected_path_available_count']} "
                f"seconds={row['seconds']:.1f}",
                flush=True,
            )
        rows.append(
            {
                "route_id": case["route_id"],
                "rank": int(case["rank"]),
                "team_name": case["team_name"],
                "submission_id": int(case["submission_id"]),
                "near_family_id": case["near_family_id"],
                "member_count": int(case["member_count"]),
                "representative_episode_id": int(case["representative_episode_id"]),
                "baseline": baseline_row(old),
                "stages": stage_rows,
                "best": best,
                "newly_resolved": bool(best["win"]),
            }
        )

    rescued = [row for row in rows if row["newly_resolved"]]
    rescued_members = sum(int(row["member_count"]) for row in rescued)
    previous_wins = int(baseline_payload["summary"]["winning_family_count"])
    previous_weighted = int(
        baseline_payload["summary"]["winning_represented_source_trajectory_count"]
    )
    total_families = int(baseline_payload["summary"]["representative_family_count"])
    represented_total = int(
        baseline_payload["summary"]["represented_source_trajectory_count"]
    )
    all_stage_rows = [stage for row in rows for stage in row["stages"]]
    summary = {
        "input_unresolved_cases": len(rows),
        "newly_resolved_cases": len(rescued),
        "remaining_unresolved_cases": len(rows) - len(rescued),
        "resolved_rate_within_nine": len(rescued) / len(rows),
        "combined_winning_family_count": previous_wins + len(rescued),
        "combined_family_count": total_families,
        "combined_family_win_rate": (previous_wins + len(rescued)) / total_families,
        "newly_resolved_represented_trajectory_count": rescued_members,
        "combined_winning_represented_trajectory_count": previous_weighted + rescued_members,
        "combined_represented_trajectory_count": represented_total,
        "combined_implied_weighted_win_rate": (previous_weighted + rescued_members) / represented_total,
        "stages_executed": len(all_stage_rows),
        "total_complete_continuations": sum(
            int(row["complete_continuations"]) for row in all_stage_rows
        ),
        "total_search_seconds": sum(float(row["seconds"]) for row in all_stage_rows),
        "maximum_selected_path_available_count": max(
            int(row["maximum_selected_path_available_count"]) for row in all_stage_rows
        ),
        "feasible_pool_not_truncated_at_4096": all(
            int(row["maximum_selected_path_available_count"]) < args.maximum_arms
            for row in all_stage_rows
        ),
        "all_committed_replays_exact": all(
            bool(row["replay_exact"]) for row in all_stage_rows
        ),
    }
    payload = {
        "schema": "kaggriculture-candidate8-top60-unresolved-feasible-dense-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS"
        if summary["feasible_pool_not_truncated_at_4096"]
        and summary["all_committed_replays_exact"]
        else "FAIL",
        "boundary": (
            "Exact-future offline upper-bound audit over frozen opponent tapes and "
            "actual replay seeds; not a deployable online selector."
        ),
        "config": {
            "beam_width": args.beam_width,
            "maximum_arms": args.maximum_arms,
            "candidate_pool": "feasible",
            "stage_decision_days": {name: days for name, days in STAGES},
        },
        "summary": summary,
        "rows": rows,
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "source": args.source,
                "actions": args.actions,
                "metadata": args.metadata,
                "cases": args.cases,
                "baseline_receipt": args.baseline_receipt,
                "genomes": args.genomes,
            }.items()
        },
    }
    receipt_path = args.output_root / "CANDIDATE8_UNRESOLVED_FEASIBLE_DENSE_RECEIPT.json"
    report_path = args.output_root / "CANDIDATE8_UNRESOLVED_FEASIBLE_DENSE_REPORT_ZH.md"
    receipt_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(report_path, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"receipt={receipt_path}", flush=True)
    print(f"report={report_path}", flush=True)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
