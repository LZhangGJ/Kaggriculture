#!/usr/bin/env python3
"""Build the UG0 live-corpus manifest, behavior families, and leakage-safe splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unseen_generalization_v1.inventory import build_asset_manifest  # noqa: E402
from unseen_generalization_v1.replay_features import iter_player_rows, summarize_replay_file  # noqa: E402
from unseen_generalization_v1.splits import build_group_splits  # noqa: E402


def _parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--root must use LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("root label cannot be empty")
    return label, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(r"D:\Kaggriculture"))
    parser.add_argument("--root", action="append", type=_parse_root, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--family-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--max-json-parse-mib", type=int, default=128)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _report(
    *,
    manifest: dict[str, Any],
    replay_summaries: list[Any],
    split_manifest: dict[str, Any],
    parse_failures: list[dict[str, str]],
) -> str:
    kind_counts = manifest.get("kind_counts", {})
    duplicate_groups = manifest.get("duplicate_content_groups", {})
    family_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    for summary in replay_summaries:
        family_counts.update(player.coarse_family for player in summary.players)
        warning_counts.update(summary.parse_warnings)

    lines = [
        "# UG0 实时语料冻结与未见族划分基线",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 结论",
        "",
        "本阶段只建立可审计的数据入口、Replay 行为签名和 group-level split；尚未训练路线选择器，也没有把任何排行榜结果当作因果标签。",
        "",
        "## 资产清单",
        "",
        f"- 已索引文件：{manifest.get('record_count', 0)}",
        f"- Replay：{kind_counts.get('replay', 0)}",
        f"- Notebook：{kind_counts.get('notebook', 0)}",
        f"- Python：{kind_counts.get('python', 0)}",
        f"- Submission archive：{kind_counts.get('submission_archive', 0)}",
        f"- 内容重复组：{len(duplicate_groups)}",
        f"- Replay 解析失败：{len(parse_failures)}",
        "",
        "## Replay 路线族",
        "",
    ]
    if family_counts:
        lines.extend(f"- `{family}`：{count}" for family, count in sorted(family_counts.items()))
    else:
        lines.append("- 未发现可解析 Replay。")

    lines.extend(
        [
            "",
            "## 数据划分",
            "",
            f"- Train：{split_manifest.get('split_counts', {}).get('train', 0)}",
            f"- Validation：{split_manifest.get('split_counts', {}).get('validation', 0)}",
            f"- Family holdout：{split_manifest.get('split_counts', {}).get('family_holdout', 0)}",
            f"- 完整保留的未见粗路线族：{', '.join(split_manifest.get('holdout_families', [])) or '无'}",
            f"- 行为签名跨 split 泄漏：{split_manifest.get('leakage_checks', {}).get('behavior_signature_cross_split', 'unknown')}",
            f"- Submission ID 跨 split 泄漏：{split_manifest.get('leakage_checks', {}).get('submission_id_cross_split', 'unknown')}",
            "",
            "## Replay 重建警告",
            "",
        ]
    )
    if warning_counts:
        lines.extend(f"- `{warning}`：{count}" for warning, count in sorted(warning_counts.items()))
    else:
        lines.append("- 无。")

    lines.extend(
        [
            "",
            "## 下一硬门",
            "",
            "1. 对最新 submission replay 做官方 Python 与 JAX 逐步复现。",
            "2. 保存 State、Events、双方 carry、市场债务和公开历史，形成 CheckpointBankV1。",
            "3. 对当前路线库建立 checkpoint × continuation route 反事实矩阵。",
            "4. 按 route gap / selector gap / bridge gap / opening gap 分类，而不是立即启动 GA。",
            "",
            "## 真实性边界",
            "",
            "- 粗路线族来自公开动作和公共盘面统计，只用于分组，不等价于真实策略语义。",
            "- Replay 中的实际动作不是其它路线的价值标签；后续标签必须由本地反事实续跑生成。",
            "- 本报告没有访问本机之外的数据，也没有宣称 Kaggle 下载或 GPU 实验已成功。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    roots = dict(args.root) if args.root else {"workspace": workspace}
    roots = {label: path.expanduser().resolve() for label, path in roots.items()}
    output = (args.output or workspace / "live_assets" / "ug0_corpus_v1").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = build_asset_manifest(
        roots,
        max_parse_bytes=args.max_json_parse_mib * 1024 * 1024,
        exclude_paths=(output,),
    )
    _write_json(output / "source_manifest_v1.json", manifest)

    replay_summaries = []
    parse_failures: list[dict[str, str]] = []
    root_lookup = {label: Path(path) for label, path in roots.items()}
    for record in manifest["records"]:
        if record["kind"] != "replay":
            continue
        path = root_lookup[record["root_label"]] / record["relative_path"]
        try:
            replay_summaries.append(summarize_replay_file(path))
        except Exception as exc:  # Preserve corpus progress and report exact file.
            parse_failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    with (output / "replay_summaries_v1.jsonl").open("w", encoding="utf-8") as handle:
        for summary in replay_summaries:
            handle.write(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    player_rows = [row for summary in replay_summaries for row in iter_player_rows(summary)]
    fieldnames = [
        "source_path",
        "replay_sha256",
        "episode_id",
        "episode_seed",
        "player_index",
        "behavior_signature",
        "coarse_family",
        "final_reward",
        "final_status",
        "submission_ids",
    ]
    with (output / "replay_players_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in player_rows:
            row = dict(row)
            row["submission_ids"] = ";".join(row["submission_ids"])
            writer.writerow(row)

    split_manifest = build_group_splits(
        replay_summaries,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        family_holdout_fraction=args.family_holdout_fraction,
    )
    _write_json(output / "split_manifest_v1.json", split_manifest)
    _write_json(output / "parse_failures_v1.json", parse_failures)

    report = _report(
        manifest=manifest,
        replay_summaries=replay_summaries,
        split_manifest=split_manifest,
        parse_failures=parse_failures,
    )
    (output / "UG0_CORPUS_BASELINE_V1_ZH.md").write_text(report, encoding="utf-8")
    print(f"UG0 manifest: {output / 'source_manifest_v1.json'}")
    print(f"Replay summaries: {len(replay_summaries)}; parse failures: {len(parse_failures)}")
    print(f"Split counts: {split_manifest['split_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
