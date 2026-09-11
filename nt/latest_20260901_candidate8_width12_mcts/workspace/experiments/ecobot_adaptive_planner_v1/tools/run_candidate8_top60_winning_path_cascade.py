#!/usr/bin/env python3
"""Run Beam-4 then Beam-12 Candidate8 exact-future search on Top-60 routes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def family_names(values: list[int]) -> list[str]:
    return ["NONE" if int(value) < 0 else FAMILY_NAMES[int(value)] for value in values]


def competitive_key(row: dict[str, Any]) -> tuple[float, float, float]:
    margin = float(row["margin"])
    return (1.0 if margin > 0 else (0.5 if margin == 0 else 0.0), margin, float(row["own_cash"]))


def result_row(width: int, elapsed: float, result: dict[str, Any], seat: int) -> dict[str, Any]:
    own = float(result["rewards"][seat])
    rival = float(result["rewards"][1 - seat])
    return {
        "beam_width": width,
        "own_cash": own,
        "opponent_cash": rival,
        "margin": own - rival,
        "win": own > rival,
        "tie": own == rival,
        "seconds": elapsed,
        "complete_continuations": int(result["complete_continuations"]),
        "continuations_per_second": (
            int(result["complete_continuations"]) / elapsed if elapsed else 0.0
        ),
        "expanded_nodes": int(result["expanded_nodes"]),
        "maximum_live_beam": int(result["maximum_live_beam"]),
        "decision_days": [int(value) for value in result["decision_day"]],
        "selected_ranks": [int(value) for value in result["selected_rank"]],
        "selected_families": family_names(result["selected_family"]),
    }


def run_one(
    bundle: NativeTeammateBundle,
    genome: np.ndarray,
    case: dict[str, Any],
    decision_days: list[int],
    primary_width: int,
    fallback_width: int,
    per_node_arms: int,
    use_feasible_pool: bool,
) -> dict[str, Any]:
    opponent = bundle.index(case["family"])
    seat = int(case["candidate_seat"])
    seed = int(case["official_seed"])

    def search(width: int) -> dict[str, Any]:
        started = time.perf_counter()
        raw = bundle.adaptive_executor.candidate8_sequence_oracle(
            genome,
            opponent,
            seed,
            decision_days,
            seat,
            width,
            per_node_arms,
            use_feasible_pool,
            True,
        )
        row = result_row(width, time.perf_counter() - started, raw, seat)
        committed = bundle.adaptive_executor.candidate8_committed_sequence(
            genome,
            opponent,
            seed,
            row["decision_days"],
            row["selected_ranks"],
            seat,
            use_feasible_pool,
        )
        row["committed_rewards"] = [float(value) for value in committed["rewards"]]
        row["replay_exact"] = tuple(row["committed_rewards"]) == tuple(
            float(value) for value in raw["rewards"]
        )
        if not row["replay_exact"]:
            raise RuntimeError(
                f"selected sequence did not replay exactly: {case['route_id']} width={width}"
            )
        return row

    primary = search(primary_width)
    fallback = None if primary["win"] else search(fallback_width)
    selected = primary if fallback is None else max((primary, fallback), key=competitive_key)
    return {
        "route_id": case["route_id"],
        "rank": int(case["rank"]),
        "team_id": int(case["team_id"]),
        "team_name": case["team_name"],
        "submission_id": int(case["submission_id"]),
        "near_family_id": case["near_family_id"],
        "member_count": int(case["member_count"]),
        "representative_episode_id": int(case["representative_episode_id"]),
        "official_seed": seed,
        "opponent_seat": int(case["opponent_seat"]),
        "candidate_seat": seat,
        "primary": primary,
        "fallback": fallback,
        "selected_width": int(selected["beam_width"]),
        "selected": selected,
        "found_winning_path": bool(selected["win"]),
        "beam12_rescue": bool(not primary["win"] and fallback and fallback["win"]),
        "status": "PASS",
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["route_id"]] = row
    return rows


def make_summary(rows: list[dict[str, Any]], represented_total: int) -> dict[str, Any]:
    wins = [row for row in rows if row["found_winning_path"]]
    primary_wins = [row for row in rows if row["primary"]["win"]]
    rescues = [row for row in rows if row["beam12_rescue"]]
    unresolved = [row for row in rows if not row["found_winning_path"]]
    weighted_wins = sum(int(row["member_count"]) for row in wins)
    fallback_rows = [row for row in rows if row["fallback"] is not None]
    all_search_rows = [row["primary"] for row in rows] + [
        row["fallback"] for row in fallback_rows
    ]
    return {
        "representative_family_count": len(rows),
        "beam4_winning_families": len(primary_wins),
        "beam12_attempted_families": len(fallback_rows),
        "beam12_rescued_families": len(rescues),
        "unresolved_families": len(unresolved),
        "winning_family_count": len(wins),
        "representative_family_win_rate": len(wins) / len(rows) if rows else 0.0,
        "represented_source_trajectory_count": represented_total,
        "winning_represented_source_trajectory_count": weighted_wins,
        "representative_implied_weighted_win_rate": (
            weighted_wins / represented_total if represented_total else 0.0
        ),
        "mean_selected_margin": float(np.mean([row["selected"]["margin"] for row in rows])),
        "median_selected_margin": float(np.median([row["selected"]["margin"] for row in rows])),
        "minimum_selected_margin": float(min(row["selected"]["margin"] for row in rows)),
        "mean_selected_own_cash": float(np.mean([row["selected"]["own_cash"] for row in rows])),
        "total_search_seconds_sum": float(sum(row["seconds"] for row in all_search_rows)),
        "total_complete_continuations": int(
            sum(row["complete_continuations"] for row in all_search_rows)
        ),
        "all_committed_replays_exact": all(
            row["replay_exact"] for row in all_search_rows
        ),
    }


def make_player_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["rank"]), int(row["submission_id"]))].append(row)
    output = []
    for (rank, submission_id), values in sorted(grouped.items()):
        represented = sum(int(row["member_count"]) for row in values)
        wins = [row for row in values if row["found_winning_path"]]
        weighted_wins = sum(int(row["member_count"]) for row in wins)
        output.append(
            {
                "rank": rank,
                "team_name": values[0]["team_name"],
                "submission_id": submission_id,
                "family_count": len(values),
                "winning_family_count": len(wins),
                "family_win_rate": len(wins) / len(values),
                "represented_trajectory_count": represented,
                "winning_represented_trajectory_count": weighted_wins,
                "representative_implied_weighted_win_rate": weighted_wins / represented,
                "beam12_rescues": sum(bool(row["beam12_rescue"]) for row in values),
                "minimum_selected_margin": min(row["selected"]["margin"] for row in values),
                "mean_selected_margin": float(
                    np.mean([row["selected"]["margin"] for row in values])
                ),
            }
        )
    return output


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Candidate8 对 Top60 非动态轨迹必胜路径审计",
        "",
        f"生成时间：{payload['created_at_utc']}",
        "",
        "## 结论",
        "",
        f"- 代表路线：{summary['winning_family_count']}/{summary['representative_family_count']}，命中率 **{summary['representative_family_win_rate']:.2%}**。",
        f"- Beam-4 直接找到胜路：{summary['beam4_winning_families']} 条。",
        f"- Beam-12 补跑：{summary['beam12_attempted_families']} 条，其中救回 {summary['beam12_rescued_families']} 条。",
        f"- 仍未找到胜路：{summary['unresolved_families']} 条。",
        f"- 代表路线按原始家族成员数加权：{summary['winning_represented_source_trajectory_count']}/{summary['represented_source_trajectory_count']}，投影覆盖率 **{summary['representative_implied_weighted_win_rate']:.2%}**。",
        "",
        "> 注意：这是知道官方 seed 和冻结对手完整动作带后的离线 exact-future Oracle。它衡量 Candidate8 的表达/搜索上限，不是线上可实现胜率。加权覆盖率是假设同一家族代表能代表其成员的投影，不等同于逐局精确复验。",
        "",
        "## 固定口径",
        "",
        "- 排除四名动态选手：Rank 1、4、9、49。",
        "- 使用其余 56 个 submission 的 family@0.90 代表轨迹。",
        "- 决策日：0、1、3、6、9、12、18、24；每节点 64 个候选；shortlist 候选池。",
        "- 先 Beam-4；只有未获胜的代表轨迹才补 Beam-12；终局现金差严格大于 0 才算胜。",
        "- 每条选中路径均用 committed-sequence 完整重放并要求终局奖励逐值一致。",
        "",
        "## 分选手结果",
        "",
        "|Rank|选手|Submission|代表胜路|代表率|加权覆盖|Beam12救回|最差分差|",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["player_summary"]:
        lines.append(
            f"|{row['rank']}|{row['team_name']}|{row['submission_id']}|"
            f"{row['winning_family_count']}/{row['family_count']}|{row['family_win_rate']:.1%}|"
            f"{row['representative_implied_weighted_win_rate']:.1%}|{row['beam12_rescues']}|"
            f"{row['minimum_selected_margin']:.0f}|"
        )
    unresolved = [row for row in payload["rows"] if not row["found_winning_path"]]
    lines.extend(["", "## 未解代表轨迹", ""])
    if unresolved:
        lines.extend(
            [
                "|Rank|选手|Family|Episode|成员数|最佳宽度|我方|对手|分差|",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in unresolved:
            selected = row["selected"]
            lines.append(
                f"|{row['rank']}|{row['team_name']}|{row['near_family_id']}|"
                f"{row['representative_episode_id']}|{row['member_count']}|{row['selected_width']}|"
                f"{selected['own_cash']:.0f}|{selected['opponent_cash']:.0f}|{selected['margin']:.0f}|"
            )
    else:
        lines.append("无。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--decision-days", type=parse_ints, default=parse_ints("0,1,3,6,9,12,18,24"))
    parser.add_argument("--primary-width", type=int, default=4)
    parser.add_argument("--fallback-width", type=int, default=12)
    parser.add_argument("--per-node-arms", type=int, default=64)
    parser.add_argument("--candidate-pool", choices=("shortlist", "feasible"), default="shortlist")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_root / "case_results.jsonl"
    cases_payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = list(cases_payload["cases"])
    represented_total = int(cases_payload["represented_source_trajectory_count"])
    completed = load_checkpoint(checkpoint_path)

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata, args.backbone)
    genome = load_genome(args.genomes, args.genome_index)
    pending = [case for case in cases if case["route_id"] not in completed]
    print(
        json.dumps(
            {
                "cases": len(cases),
                "completed": len(completed),
                "pending": len(pending),
                "workers": args.workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    write_lock = threading.Lock()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                bundle,
                genome,
                case,
                args.decision_days,
                args.primary_width,
                args.fallback_width,
                args.per_node_arms,
                args.candidate_pool == "feasible",
            ): case
            for case in pending
        }
        for done_index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            with write_lock:
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
            completed[row["route_id"]] = row
            if done_index == 1 or done_index % 10 == 0 or done_index == len(pending):
                elapsed = time.perf_counter() - started
                wins = sum(value["found_winning_path"] for value in completed.values())
                rescues = sum(value["beam12_rescue"] for value in completed.values())
                print(
                    f"progress={len(completed)}/{len(cases)} wins={wins} "
                    f"beam12_rescues={rescues} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    rows = [completed[case["route_id"]] for case in cases]
    summary = make_summary(rows, represented_total)
    player_summary = make_player_summary(rows)
    payload = {
        "schema": "kaggriculture-candidate8-top60-winning-path-cascade-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if summary["all_committed_replays_exact"] else "FAIL",
        "boundary": (
            "Exact-future offline Oracle over frozen opponent action tapes and actual "
            "replay seeds. It measures Candidate8 expression/search headroom, not a "
            "deployable online selector."
        ),
        "config": {
            "excluded_dynamic_ranks": [1, 4, 9, 49],
            "decision_days": args.decision_days,
            "primary_width": args.primary_width,
            "fallback_width": args.fallback_width,
            "per_node_arms": args.per_node_arms,
            "candidate_pool": args.candidate_pool,
            "workers": args.workers,
            "win_definition": "candidate terminal cash > opponent terminal cash",
        },
        "summary": summary,
        "player_summary": player_summary,
        "rows": rows,
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "source": args.source,
                "actions": args.actions,
                "metadata": args.metadata,
                "cases": args.cases,
                "genomes": args.genomes,
            }.items()
        },
    }
    if args.backbone:
        payload["inputs"]["backbone"] = {
            "path": str(args.backbone),
            "sha256": sha256(args.backbone),
        }
    receipt_path = args.output_root / "CANDIDATE8_TOP60_WINNING_PATH_RECEIPT.json"
    report_path = args.output_root / "CANDIDATE8_TOP60_WINNING_PATH_REPORT_ZH.md"
    receipt_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report_path, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"receipt={receipt_path}", flush=True)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
