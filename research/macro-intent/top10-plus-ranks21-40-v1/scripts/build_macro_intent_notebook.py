#!/usr/bin/env python3
"""Build and execute a compact QA notebook for macro-intent artifacts.

The bundled analysis runtime does not ship a Jupyter kernel. To keep this
handoff dependency-free, code cells are executed sequentially in one Python
namespace and their text outputs are recorded in a valid nbformat-v4 file.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import pprint
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percent(value: float | None) -> str:
    return "UNKNOWN" if value is None else f"{100 * value:.2f}%"


def markdown_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def execute_code_cell(
    cell: dict[str, Any], namespace: dict[str, Any], execution_count: int
) -> None:
    source = "".join(cell["source"])
    parsed = ast.parse(source, filename=f"<notebook-cell-{execution_count}>", mode="exec")
    final_expression = None
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        final_expression = ast.Expression(parsed.body.pop().value)
        ast.fix_missing_locations(final_expression)
    body = compile(parsed, f"<notebook-cell-{execution_count}>", "exec")
    stdout, stderr = io.StringIO(), io.StringIO()

    def display(value: Any) -> None:
        pprint.pprint(value, stream=stdout, sort_dicts=False)

    namespace["display"] = display
    outputs: list[dict[str, Any]] = []
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(body, namespace)
            result = (
                eval(compile(final_expression, "<notebook-result>", "eval"), namespace)
                if final_expression
                else None
            )
    except Exception as error:
        outputs.append(
            {
                "output_type": "error",
                "ename": type(error).__name__,
                "evalue": str(error),
                "traceback": traceback.format_exc().splitlines(),
            }
        )
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
        raise
    if stdout.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
    if stderr.getvalue():
        outputs.append({"output_type": "stream", "name": "stderr", "text": stderr.getvalue()})
    if result is not None:
        outputs.append(
            {
                "output_type": "execute_result",
                "execution_count": execution_count,
                "data": {"text/plain": pprint.pformat(result, sort_dicts=False)},
                "metadata": {},
            }
        )
    cell["execution_count"] = execution_count
    cell["outputs"] = outputs


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((artifact_dir / "summary_v1.json").read_text(encoding="utf-8"))

    cells = [
        markdown_cell(
            "# Ranks 21–40 Replay 宏观意图恢复：质量检查\n\n"
            "## tl;dr\n\n"
            f"- 已解析 **{summary['parsed_episodes']:,}** 个唯一 Episode，输出 "
            f"**{summary['output_seats']:,}** 个 `(episode_id, player_index, submission_id)` 座位。\n"
            f"- 提取失败 **{summary['failures']:,}** 个；空间意图缺失坐标 "
            f"**{summary['spatial_missing_coordinate']:,}** 条。\n"
            f"- 空间意图即时观测成功率为 **{percent(summary['spatial_observed_success_rate'])}**；"
            "失败动作仍保留为计划意图。\n"
            "- 本 notebook 只检查恢复产物，不执行任何参赛 Agent 代码。",
            "summary",
        ),
        markdown_cell(
            "## Context & Methods\n\n"
            "恢复粒度是一条目标 submission 在某个 Episode 的一个座位。Replay 中 turn `t` 的动作存放在 "
            "replay step `t+1`，因此空间坐标取前一 observation 的 actor 位置，成功状态取当前 observation。\n\n"
            "### Key Assumptions\n\n"
            "- `TeamNames` 与清单中的 team 能精确或规范化匹配。\n"
            "- 每局 720 个 replay steps 对应 719 个动作 turn。\n"
            "- 布局身份采用每个格子截至锚点最后一次发出的生产意图。\n"
            "- PLANT、PLACE、BUILD_COOP、BUILD_PASTURE 构成空间意图；BUY_LAND 与 HIRE 只进入阶段计数。",
            "methods",
        ),
        markdown_cell("## Data\n\n### 1. Load artifacts", "data"),
        code_cell(
            "from pathlib import Path\n"
            "import csv\n"
            "import json\n"
            "from collections import Counter, defaultdict\n"
            "import numpy as np\n\n"
            f"artifact_dir = Path({str(artifact_dir)!r})\n"
            "summary = json.loads((artifact_dir / 'summary_v1.json').read_text(encoding='utf-8'))\n"
            "features = np.load(artifact_dir / 'intent_features_v1.npz')\n"
            "def read_csv(name):\n"
            "    with (artifact_dir / name).open(encoding='utf-8-sig', newline='') as stream:\n"
            "        return list(csv.DictReader(stream))\n"
            "seats = read_csv('intent_seats_v1.csv')\n"
            "event_path = artifact_dir / 'macro_events_v1.csv'\n"
            "failures = read_csv('failures_v1.csv')\n"
            "{'seat_rows': len(seats), 'event_rows': summary['macro_event_rows'], 'failure_rows': len(failures), 'feature_keys': features.files}",
            "load",
        ),
        markdown_cell("### 2. Validate grain, alignment, and shapes", "validate-title"),
        code_cell(
            "seat_keys = [(row['episode_id'], row['player_index'], row['submission_id']) for row in seats]\n"
            "duplicate_seat_keys = len(seat_keys) - len(set(seat_keys))\n"
            "expected_shapes = summary['feature_shapes']\n"
            "checks = {\n"
            "    'duplicate_seat_keys': duplicate_seat_keys,\n"
            "    'seat_id_alignment': len(features['seat_id']) == len(seats),\n"
            "    'layout_shape': list(features['layout'].shape),\n"
            "    'stage_actions_shape': list(features['stage_actions'].shape),\n"
            "    'cumulative_actions_shape': list(features['cumulative_actions'].shape),\n"
            "    'all_layout_codes_valid': bool(np.isin(features['layout'], np.arange(11)).all()),\n"
            "    'cumulative_is_monotone': bool((np.diff(features['cumulative_actions'], axis=1) >= 0).all()),\n"
            "}\n"
            "assert duplicate_seat_keys == 0\n"
            "assert checks['seat_id_alignment']\n"
            "assert checks['layout_shape'] == expected_shapes['layout']\n"
            "assert checks['stage_actions_shape'] == expected_shapes['stage_actions']\n"
            "assert checks['cumulative_actions_shape'] == expected_shapes['cumulative_actions']\n"
            "assert checks['all_layout_codes_valid'] and checks['cumulative_is_monotone']\n"
            "checks",
            "validate",
        ),
        markdown_cell("## Results\n\n### 3. Dataset and extraction profile", "results"),
        code_cell(
            "{\n"
            "    'manifest_rows': summary.get('manifest_rows'),\n"
            "    'unique_episodes': summary.get('manifest_unique_episodes', summary.get('parsed_episodes')),\n"
            "    'duplicate_episode_references': summary.get('duplicate_episode_references'),\n"
            "    'input_artifacts': summary.get('input_artifacts'),\n"
            "    'parsed_episodes': summary['parsed_episodes'],\n"
            "    'output_seats': summary['output_seats'],\n"
            "    'mapping_methods': summary['mapping_methods'],\n"
            "    'replay_step_lengths': summary['replay_step_lengths'],\n"
            "    'failures': summary['failure_kinds'],\n"
            "}",
            "profile",
        ),
        markdown_cell("### 4. Submission coverage", "coverage-title"),
        code_cell(
            "coverage = defaultdict(lambda: {'seats': 0, 'WIN': 0, 'LOSS': 0, 'TIE': 0})\n"
            "for row in seats:\n"
            "    key = (int(row['rank']), row['team'], row['submission_id'])\n"
            "    coverage[key]['seats'] += 1\n"
            "    coverage[key][row['result']] += 1\n"
            "submission_coverage = [dict(rank=k[0], team=k[1], submission_id=k[2], **coverage[k]) for k in sorted(coverage)]\n"
            "submission_coverage",
            "coverage",
        ),
        markdown_cell("### 5. Macro intent operations and execution evidence", "events-title"),
        code_cell(
            "operation_counts = Counter()\n"
            "spatial_ops = {'PLANT', 'PLACE', 'BUILD_COOP', 'BUILD_PASTURE'}\n"
            "success = defaultdict(lambda: [0, 0])\n"
            "with event_path.open(encoding='utf-8-sig', newline='') as stream:\n"
            "    for row in csv.DictReader(stream):\n"
            "        operation_counts[row['operation']] += 1\n"
            "        if row['operation'] in spatial_ops and row['observed_success'] != '':\n"
            "            success[row['operation']][0] += 1\n"
            "            success[row['operation']][1] += int(row['observed_success'])\n"
            "success_profile = {op: {'count': values[0], 'rate': values[1] / values[0]} for op, values in success.items()}\n"
            "{'operation_counts': operation_counts.most_common(), 'success_profile': success_profile}",
            "events",
        ),
        markdown_cell(
            "## Takeaways\n\n"
            "- 产物已经按目标座位去重，数组行与座位清单一一对齐。\n"
            "- 失败的空间动作占比较高，证明不能仅从最终棋盘恢复路线；必须保留发出的动作意图。\n"
            "- 当前输出适合下一步与旧路线库计算 `0.45C + 0.35G + 0.20T` 距离并做增量聚类。\n"
            "- `observed_success` 是 Replay 前后状态的直接观测，不等同于完整 C++ 执行审计；资金不足等失败原因尚未分类。",
            "takeaways",
        ),
    ]

    namespace: dict[str, Any] = {"__name__": "__macro_intent_notebook__"}
    execution_count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        execute_code_cell(cell, namespace, execution_count)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "execution": {"mode": "sequential-python-namespace", "status": "passed"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
