#!/usr/bin/env python3
"""Aggregate disjoint FC10 Kobe commitment holdout shards with hard gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _score(game: dict) -> float:
    if game["candidate_cash"] > game["opponent_cash"]:
        return 1.0
    if game["candidate_cash"] == game["opponent_cash"]:
        return 0.5
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--candidate-step", type=int, default=192)
    parser.add_argument("--control-step", type=int, default=719)
    parser.add_argument("--min-score", type=float, default=0.90)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not payloads or any(item.get("status") != "PASS" for item in payloads):
        raise ValueError("all input shards must be PASS receipts")
    frozen_fields = (
        "schema",
        "backend",
        "official_package_version",
        "goose_bank_sha256",
        "switch_steps",
        "trigger_mode",
        "opponent_melon_min",
        "third_yarn_policy",
        "feature_step",
        "seat_protocol",
    )
    reference = payloads[0]
    for payload in payloads[1:]:
        for field in frozen_fields:
            if payload.get(field) != reference.get(field):
                raise ValueError(f"holdout shard mismatch for {field}")

    all_seeds: set[int] = set()
    games_by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    opponents: set[str] = set()
    for payload in payloads:
        seeds = {int(value) for value in payload["seed_values"]}
        if all_seeds & seeds:
            raise ValueError("holdout shards contain overlapping seeds")
        all_seeds |= seeds
        for row in payload["rows"]:
            opponent = str(row["opponent"])
            opponents.add(opponent)
            games_by_key[(opponent, int(row["switch_step"]))].extend(row["per_game"])

    expected_games = 2 * len(all_seeds)
    summaries = []
    gates = []
    for opponent in sorted(opponents):
        by_step = {}
        for step in (args.candidate_step, args.control_step):
            games = games_by_key[(opponent, step)]
            identities = {(int(g["seed"]), int(g["candidate_seat"])) for g in games}
            if len(games) != expected_games or len(identities) != expected_games:
                raise ValueError(f"incomplete or duplicate games for {opponent} step {step}")
            scores = [_score(game) for game in games]
            margins = [int(game["candidate_cash"]) - int(game["opponent_cash"]) for game in games]
            by_step[step] = {
                "games": len(games),
                "wins": sum(value == 1.0 for value in scores),
                "ties": sum(value == 0.5 for value in scores),
                "losses": sum(value == 0.0 for value in scores),
                "score_rate": sum(scores) / len(scores),
                "mean_margin": sum(margins) / len(margins),
                "seat0_score_rate": sum(
                    _score(game) for game in games if int(game["candidate_seat"]) == 0
                ) / len(all_seeds),
                "seat1_score_rate": sum(
                    _score(game) for game in games if int(game["candidate_seat"]) == 1
                ) / len(all_seeds),
            }
        candidate = by_step[args.candidate_step]
        control = by_step[args.control_step]
        gate = candidate["score_rate"] >= args.min_score
        gates.append(gate)
        summaries.append(
            {
                "opponent": opponent,
                "candidate": candidate,
                "fc2b_control": control,
                "score_rate_delta": candidate["score_rate"] - control["score_rate"],
                "mean_margin_delta": candidate["mean_margin"] - control["mean_margin"],
                "candidate_min_score_gate": gate,
            }
        )

    output = {
        "schema": "kaggriculture.fusion-champion.fc10h-holdout-aggregate.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates) else "FAIL",
        "decision": "PROMOTE_TO_INTEGRATION" if all(gates) else "REJECT_OR_REVISE",
        "truth_boundary": (
            "Independent event holdout for one public-state project commitment rule. "
            "This proves the core3 gate only; it does not prove the full frozen-pool goal."
        ),
        "official_package_version": reference["official_package_version"],
        "backend": reference["backend"],
        "candidate_step": args.candidate_step,
        "control_step": args.control_step,
        "min_score_gate": args.min_score,
        "holdout_seed_count": len(all_seeds),
        "games_per_opponent": expected_games,
        "seed_values": sorted(all_seeds),
        "rule": {
            "decision_step": 192,
            "first_two_shops_exclude": [
                "YARN_STORE",
                "PIZZA_SHOP",
                "ICE_CREAM_SHOP",
                "SMOOTHIE_SHOP",
            ],
            "opponent_public_melon_min": reference["opponent_melon_min"],
            "project_commitment": "continue after activation, including third-shop YARN",
            "future_information": False,
            "opponent_identity": False,
        },
        "source_receipts": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in args.inputs
        ],
        "rows": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# FC10H：Kobe 鹅项目承诺规则独立 Holdout 验收",
        "",
        f"结论：**{output['status']} / {output['decision']}**。",
        "",
        f"- 未见事件种子：{len(all_seeds)}；",
        f"- 每个对手：{expected_games} 局（同事件双方换座）；",
        f"- 硬门：候选对每个 core3 对手得分率均不低于 {args.min_score:.0%}；",
        "- 决策只读取公开商店、公开对手瓜田和当前步，不读取身份或未来事件。",
        "",
        "| 对手 | FC2B基线 | Kobe承诺分支 | 增量 | 基线分差 | 候选分差 | 分差增量 | 硬门 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {opponent} | {base:.2%} | {candidate:.2%} | {delta:+.2%} | "
            "{base_margin:,.1f} | {candidate_margin:,.1f} | {margin_delta:+,.1f} | {gate} |".format(
                opponent=row["opponent"],
                base=row["fc2b_control"]["score_rate"],
                candidate=row["candidate"]["score_rate"],
                delta=row["score_rate_delta"],
                base_margin=row["fc2b_control"]["mean_margin"],
                candidate_margin=row["candidate"]["mean_margin"],
                margin_delta=row["mean_margin_delta"],
                gate="PASS" if row["candidate_min_score_gate"] else "FAIL",
            )
        )
    lines += [
        "",
        "## 边界",
        "",
        "该结果只允许把规则晋级到正式融合源码和全池评测。它不代表已经完成全部 Agent 的 90% 目标，也不能替代官方 Python 复验。",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output_json.resolve())}))
    return 0 if all(gates) else 2


if __name__ == "__main__":
    raise SystemExit(main())
