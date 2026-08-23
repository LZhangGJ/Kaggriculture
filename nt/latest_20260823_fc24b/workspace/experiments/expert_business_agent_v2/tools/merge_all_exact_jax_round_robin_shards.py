#!/usr/bin/env python3
"""Merge the eight independently checkpointed strict-JAX Arena shards.

The merger is deliberately CPU-only.  It validates pair coverage, event-array
identity, seat/game counts, terminal completion and simulator diagnostics before
publishing a PASS receipt.  Rankings are only produced from the complete graph.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments" / "expert_business_agent_v2"
DEFAULT_SHARDS = tuple(BASE / "receipts" / f"all_exact_rr_s8_{i}_v1.json" for i in range(8))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def semantic_event_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as payload:
        for key in sorted(payload.files):
            value = np.ascontiguousarray(payload[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
    return digest.hexdigest().lower()


def fit_bradley_terry(pair_rows: list[dict], count: int) -> np.ndarray:
    theta = np.zeros((count,), dtype=np.float64)
    regularization = 0.05
    for _ in range(100):
        gradient = -regularization * theta
        hessian = np.eye(count, dtype=np.float64) * regularization
        for row in pair_rows:
            left, right = int(row["agent_a_id"]), int(row["agent_b_id"])
            games = int(row["games"])
            observed = float(row["a_wins"]) + 0.5 * float(row["ties"])
            delta = float(np.clip(theta[left] - theta[right], -30.0, 30.0))
            probability = 1.0 / (1.0 + np.exp(-delta))
            residual = observed - games * probability
            weight = games * probability * (1.0 - probability)
            gradient[left] += residual
            gradient[right] -= residual
            hessian[left, left] += weight
            hessian[right, right] += weight
            hessian[left, right] -= weight
            hessian[right, left] -= weight
        update = np.linalg.solve(hessian + np.eye(count) * 1e-9, gradient)
        theta += update
        theta -= np.mean(theta)
        if float(np.max(np.abs(update))) < 1e-9:
            break
    return theta * (400.0 / np.log(10.0))


def build_ranking(pair_rows: list[dict], roster: list[dict]) -> list[dict]:
    ratings = fit_bradley_terry(pair_rows, len(roster))
    totals = [
        {"wins": 0, "ties": 0, "losses": 0, "cash": [], "margin": [], "pairs": []}
        for _ in roster
    ]
    for row in pair_rows:
        left, right = int(row["agent_a_id"]), int(row["agent_b_id"])
        totals[left]["wins"] += int(row["a_wins"])
        totals[left]["ties"] += int(row["ties"])
        totals[left]["losses"] += int(row["a_losses"])
        totals[right]["wins"] += int(row["a_losses"])
        totals[right]["ties"] += int(row["ties"])
        totals[right]["losses"] += int(row["a_wins"])
        totals[left]["cash"].append(float(row["a_mean_cash"]))
        totals[right]["cash"].append(float(row["b_mean_cash"]))
        totals[left]["margin"].append(float(row["a_mean_margin"]))
        totals[right]["margin"].append(-float(row["a_mean_margin"]))
        totals[left]["pairs"].append((row["agent_b"], float(row["a_score_rate"])))
        totals[right]["pairs"].append((row["agent_a"], 1.0 - float(row["a_score_rate"])))

    ranking = []
    for rank, agent_id in enumerate(np.argsort(-ratings).tolist(), 1):
        total = totals[agent_id]
        games = total["wins"] + total["ties"] + total["losses"]
        worst_name, worst_rate = min(total["pairs"], key=lambda item: item[1])
        ranking.append(
            {
                "rank": rank,
                "agent_id": agent_id,
                "agent": roster[agent_id]["name"],
                "bt_elo": float(ratings[agent_id]),
                "games": games,
                "wins": total["wins"],
                "ties": total["ties"],
                "losses": total["losses"],
                "score_rate": (total["wins"] + 0.5 * total["ties"]) / games,
                "mean_cash": float(np.mean(total["cash"])),
                "mean_margin": float(np.mean(total["margin"])),
                "worst_opponent": worst_name,
                "worst_score_rate": float(worst_rate),
            }
        )
    return ranking


def render_report(payload: dict) -> str:
    lines = [
        "# 全部严格验收 JAX Agent 两两 100 局结果",
        "",
        f"生成时间：{payload['generated_at_utc']}",
        f"结论：**{payload['status']}**",
        "",
        "## 验收口径",
        "",
        f"- 参赛：{payload['roster_count']} 个去重后的完整 JAX Agent。",
        f"- 对局：{payload['pair_count']} 组 × 100 局 = {payload['games_total']:,} 局；每局 719 步。",
        "- 每组使用相同的 50 个独立事件种子，并以 50/50 交换座位。",
        "- 八个 GPU 分片并行运行；最终排名只在完整 378 组无缺失、无重复后计算。",
        "- 仅纳入官方 1.32.7 逐步一致性验收通过的完整 JAX 策略。",
        "- Bradley-Terry 分数只表示本次冻结本地对手池内的相对强度。",
        "",
        "## 综合排名",
        "",
        "| 排名 | Agent | BT/Elo | 得分率 | 胜-平-负 | 平均现金 | 平均分差 | 最差对手（得分率） |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["ranking"]:
        lines.append(
            f"| {row['rank']} | {row['agent']} | {row['bt_elo']:+.1f} | "
            f"{100 * row['score_rate']:.1f}% | {row['wins']}-{row['ties']}-{row['losses']} | "
            f"{row['mean_cash']:.0f} | {row['mean_margin']:+.0f} | "
            f"{row['worst_opponent']} ({100 * row['worst_score_rate']:.1f}%) |"
        )
    lines += [
        "",
        "## 运行与完整性",
        "",
        f"- GPU：{payload['device']}",
        f"- 并行墙钟时间：{payload['parallel_wall_seconds']:,.1f} 秒",
        f"- 有效总吞吐：{payload['transitions_per_second']:,.0f} transitions/s",
        f"- 全部终局完成：{payload['all_done']}",
        f"- hand cap / market loop cap / price LUT 越界：{payload['hand_cap_hits_total']} / {payload['market_loop_cap_hits_total']} / {payload['price_lut_oob_total']}",
        f"- 运行中外部观测峰值：GPU {payload['observed_peak_gpu_utilization_percent']}%，显存 {payload['observed_peak_gpu_memory_mib']} MiB。",
        "",
        "## 去重别名",
        "",
    ]
    for alias in payload["aliases"]:
        lines.append(f"- `{alias['alias']}` → `{alias['canonical']}`（{alias['reason']}）")
    lines += [
        "",
        "## 近重复审计",
        "",
        "- `boatlee_v20_multi_route` 与 `tetsutani_adaptive_latest` 在本次冻结事件库的汇总战绩相同，但不按别名合并。",
        "- 两者官方验收源文件 SHA 不同，JAX 分别使用 `MODE_BOATLEE` 与 `MODE_TETSUTANI_LATEST`；直接对战为 18 胜 / 64 平 / 18 负（按 Boatlee 视角）。",
        "- 本轮只说明聚合表现相同，不把它扩大解释成全状态动作完全一致。",
        "",
        "## 排除范围与解释边界",
        "",
        "- 46 个路线骨架、未严格对齐代理、实验性 BC/PPO 模型未纳入。",
        "- `gold_proxy_*` 是通过本地逐步一致性验收的模仿 Agent，不是原金牌选手源码。",
        "- 本结果是固定事件库下的本地 JAX 对手池排名，不等于当前 Public 榜分。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="*", type=Path, default=list(DEFAULT_SHARDS))
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=BASE / "receipts" / "all_exact_jax_round_robin_n28_seed530001_n50x2_v1.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=BASE / "reports" / "ALL_EXACT_JAX_ROUND_ROBIN_100_GAMES_20260820_ZH.md",
    )
    parser.add_argument(
        "--pair-csv-output",
        type=Path,
        default=BASE / "artifacts" / "all_exact_jax_round_robin_pair_results_v1.csv",
    )
    parser.add_argument(
        "--ranking-csv-output",
        type=Path,
        default=BASE / "artifacts" / "all_exact_jax_round_robin_ranking_v1.csv",
    )
    parser.add_argument("--observed-peak-gpu-memory-mib", type=int, default=0)
    parser.add_argument("--observed-peak-gpu-utilization-percent", type=int, default=0)
    args = parser.parse_args()

    shard_paths = [path.resolve() for path in args.shards]
    if len(shard_paths) != 8 or any(not path.is_file() for path in shard_paths):
        raise FileNotFoundError("all eight final shard receipts are required")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    roster = shards[0]["roster"]
    expected_pairs = set(itertools.combinations(range(len(roster)), 2))

    problems = []
    pair_rows = []
    event_hashes = []
    for index, (path, shard) in enumerate(zip(shard_paths, shards, strict=True)):
        if shard.get("status") != "PARTIAL_PASS" or not shard.get("integrity_pass"):
            problems.append(f"shard {index} not PARTIAL_PASS")
        if shard.get("roster") != roster:
            problems.append(f"shard {index} roster mismatch")
        if shard.get("official_package_version") != "1.32.7":
            problems.append(f"shard {index} package mismatch")
        if shard.get("steps") != 719 or shard.get("games_per_pair") != 100:
            problems.append(f"shard {index} protocol mismatch")
        if shard.get("event_seeds") != list(range(530001, 530051)):
            problems.append(f"shard {index} seed mismatch")
        event_path = Path(shard["event_bank"])
        if not event_path.is_file():
            problems.append(f"shard {index} missing event bank")
        else:
            event_hashes.append(semantic_event_hash(event_path))
        pair_rows.extend(shard["pair_results"])

    observed_pairs = [(int(row["agent_a_id"]), int(row["agent_b_id"])) for row in pair_rows]
    if len(observed_pairs) != len(set(observed_pairs)):
        problems.append("duplicate unordered pair")
    if set(observed_pairs) != expected_pairs:
        problems.append("pair graph is not complete")
    if any(int(row["games"]) != 100 for row in pair_rows):
        problems.append("not every pair has 100 games")
    if len(set(event_hashes)) != 1:
        problems.append("semantic event arrays differ across shards")

    hand_hits = sum(int(shard["hand_cap_hits_total"]) for shard in shards)
    market_hits = sum(int(shard["market_loop_cap_hits_total"]) for shard in shards)
    price_hits = sum(int(shard["price_lut_oob_total"]) for shard in shards)
    all_done = all(bool(shard["all_done"]) for shard in shards)
    if not all_done or hand_hits or market_hits or price_hits:
        problems.append("terminal/integrity diagnostic failed")

    pair_rows.sort(key=lambda row: (int(row["agent_a_id"]), int(row["agent_b_id"])))
    ranking = build_ranking(pair_rows, roster) if not problems else []
    ends = [datetime.fromisoformat(shard["generated_at_utc"]) for shard in shards]
    starts = [end - timedelta(seconds=float(shard["wall_seconds"])) for end, shard in zip(ends, shards, strict=True)]
    parallel_wall = (max(ends) - min(starts)).total_seconds()
    games_total = len(pair_rows) * 100
    transitions_total = games_total * 719
    aliases = [
        {"alias": alias, "canonical": row["name"], "reason": "source/action semantics identical"}
        for row in roster
        for alias in row.get("aliases", [])
    ]
    boatlee_tets_pair = next(
        row
        for row in pair_rows
        if int(row["agent_a_id"]) == 19 and int(row["agent_b_id"]) == 27
    )
    boatlee_source = ROOT / "references" / "public_high_potential_20260819" / "boatlee_v20_multi_route" / "main.py"
    tets_latest_source = ROOT / "references" / "public_latest8_20260820" / "tetsutani_adaptive_latest" / "main.py"
    payload = {
        "schema": "kaggriculture.all_exact_jax_round_robin.merged.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "official_package_version": "1.32.7",
        "backend": "gpu",
        "device": shards[0]["device"],
        "execution": "eight_parallel_batched_anchor_shards_v1",
        "roster_count": len(roster),
        "roster": roster,
        "aliases": aliases,
        "excluded": shards[0]["excluded"],
        "seed_start": 530001,
        "seed_count": 50,
        "event_seeds": list(range(530001, 530051)),
        "semantic_event_bank_sha256": event_hashes[0] if event_hashes else None,
        "steps": 719,
        "seat_protocol": "50 seeds as A-seat0/B-seat1 plus same 50 seeds swapped",
        "games_per_pair": 100,
        "pair_count": len(pair_rows),
        "games_total": games_total,
        "transitions_total": transitions_total,
        "compile_seconds_sum": sum(float(shard["compile_seconds"]) for shard in shards),
        "arena_worker_seconds_sum": sum(float(shard["arena_seconds"]) for shard in shards),
        "parallel_wall_seconds": parallel_wall,
        "transitions_per_second": transitions_total / max(parallel_wall, 1e-9),
        "all_done": all_done,
        "hand_cap_hits_total": hand_hits,
        "market_loop_cap_hits_total": market_hits,
        "price_lut_oob_total": price_hits,
        "integrity_pass": not problems,
        "observed_peak_gpu_memory_mib": args.observed_peak_gpu_memory_mib,
        "observed_peak_gpu_utilization_percent": args.observed_peak_gpu_utilization_percent,
        "shards": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "anchor_min": shard["anchor_min"],
                "anchor_max": shard["anchor_max"],
                "pairs": shard["pair_count"],
                "wall_seconds": shard["wall_seconds"],
            }
            for path, shard in zip(shard_paths, shards, strict=True)
        ],
        "near_duplicate_audit": {
            "agents": ["boatlee_v20_multi_route", "tetsutani_adaptive_latest"],
            "aggregate_ranking_metrics_equal_on_frozen_bank": True,
            "boatlee_source_sha256": sha256(boatlee_source),
            "tetsutani_latest_source_sha256": sha256(tets_latest_source),
            "source_hashes_equal": sha256(boatlee_source) == sha256(tets_latest_source),
            "jax_static_modes": ["MODE_BOATLEE", "MODE_TETSUTANI_LATEST"],
            "head_to_head": boatlee_tets_pair,
            "verdict": "retain_as_distinct; aggregate tie is not proof of action identity on all states",
        },
        "pair_results": pair_rows,
        "ranking": ranking,
        "truth_boundary": (
            "Gold proxy entries are strict-parity local imitation agents, not original gold source. "
            "This fixed-event local JAX ranking is not a current Public leaderboard score."
        ),
    }

    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_report(payload), encoding="utf-8")
    if pair_rows:
        args.pair_csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.pair_csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
            writer.writeheader()
            writer.writerows(pair_rows)
    if ranking:
        args.ranking_csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.ranking_csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
            writer.writeheader()
            writer.writerows(ranking)
    print(json.dumps({"status": payload["status"], "pairs": len(pair_rows), "games": games_total, "receipt": str(args.receipt_output), "report": str(args.report_output)}, ensure_ascii=False))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
