#!/usr/bin/env python3
"""Build and execute the macro-intent clustering QA notebook."""

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

import sys

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
from scipy.cluster.hierarchy import cophenet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--cluster-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def execute_code_cell(cell: dict[str, Any], namespace: dict[str, Any], count: int) -> None:
    source = "".join(cell["source"])
    parsed = ast.parse(source, filename=f"<notebook-cell-{count}>", mode="exec")
    final_expression = None
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        final_expression = ast.Expression(parsed.body.pop().value)
        ast.fix_missing_locations(final_expression)
    body = compile(parsed, f"<notebook-cell-{count}>", "exec")
    stdout, stderr = io.StringIO(), io.StringIO()
    namespace["display"] = lambda value: pprint.pprint(value, stream=stdout, sort_dicts=False)
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
        cell["execution_count"] = count
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
                "execution_count": count,
                "data": {"text/plain": pprint.pformat(result, sort_dicts=False)},
                "metadata": {},
            }
        )
    cell["execution_count"] = count
    cell["outputs"] = outputs


def main() -> None:
    args = parse_args()
    feature_dir = args.feature_dir.resolve()
    cluster_dir = args.cluster_dir.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((cluster_dir / "cluster_summary_v1.json").read_text(encoding="utf-8"))
    distance = np.load(cluster_dir / "macro_intent_distance_v1.npy", mmap_mode="r")
    linkage_matrix = np.load(cluster_dir / "average_linkage_v1.npy")
    cophenetic_corr = float(cophenet(linkage_matrix, distance)[0])
    main_cut = summary["main_cut"]
    zero_pair_share = float(np.count_nonzero(distance == 0) / distance.size)

    cells = [
        markdown_cell(
            "# Top 10 + Ranks 21–40 宏观意图聚类\n\n"
            "## tl;dr\n\n"
            f"- 按论文距离与 average linkage，在主切点 `D ≤ 0.12` 得到 **{main_cut['family_count']:,}** 个临时路线族；"
            f"其中 **{main_cut['singleton_count']:,}** 个为单例。\n"
            f"- 最大族有 **{main_cut['largest_family']:,}** 个座位（{main_cut['largest_family']/summary['seat_count']:.1%}）；"
            f"前 10 族覆盖 **{main_cut['top10_coverage']:.1%}**。\n"
            f"- 树对原始成对距离的 cophenetic correlation 为 **{cophenetic_corr:.4f}**；"
            f"完全相同的特征对占 **{zero_pair_share:.1%}**。\n"
            "- 编号使用 `N001…`，只是当前新池内部的临时编号；尚未与历史 `G001…G275` 对齐。",
            "summary",
        ),
        markdown_cell(
            "## Context & Methods\n\n"
            "本次只聚类已经恢复出的 3,397 个座位，不执行参赛 Agent，也不把旧族标签预先灌入模型。"
            "距离为 `D = 0.45C + 0.35G + 0.20T`，其中 C 比较五个锚点的生产构成，G 比较格位类别，"
            "T 比较十个阶段的累计宏观动作；随后使用 average linkage。\n\n"
            "### Key Assumptions\n\n"
            "- 锚点为 168、288、432、576、719；动作阶段宽度为 72。\n"
            "- 主切点 0.12 沿用文章口径，0.04–0.20 只用于敏感性检查。\n"
            "- `N###` 按族规模降序、再按历史奖励中位数降序排序，不代表历史 `G###`。\n"
            "- medoid 是族内平均距离最小的真实座位；共识阶段表是逐单元中位数四舍五入后的描述性原型。",
            "methods",
        ),
        markdown_cell("## Data\n\n### 1. Load clustering artifacts", "data"),
        code_cell(
            "from pathlib import Path\n"
            "import csv, json, sys\n"
            "from collections import Counter\n"
            f"feature_dir = Path({str(feature_dir)!r})\n"
            f"cluster_dir = Path({str(cluster_dir)!r})\n"
            "local_deps = cluster_dir.parents[1] / '.python_deps'\n"
            "if local_deps.exists() and str(local_deps) not in sys.path: sys.path.insert(0, str(local_deps))\n"
            "import numpy as np\n"
            "def read_csv(name):\n"
            "    with (cluster_dir / name).open(encoding='utf-8-sig', newline='') as stream:\n"
            "        return list(csv.DictReader(stream))\n"
            "summary = json.loads((cluster_dir / 'cluster_summary_v1.json').read_text(encoding='utf-8'))\n"
            "features = np.load(feature_dir / 'intent_features_v1.npz')\n"
            "distance = np.load(cluster_dir / 'macro_intent_distance_v1.npy', mmap_mode='r')\n"
            "linkage_matrix = np.load(cluster_dir / 'average_linkage_v1.npy')\n"
            "assignments = read_csv('cluster_assignments_v1.csv')\n"
            "profiles = read_csv('cluster_profiles_v1.csv')\n"
            "submission_summary = read_csv('submission_cluster_summary_v1.csv')\n"
            "sensitivity = read_csv('threshold_sensitivity_v1.csv')\n"
            "{'seats': len(assignments), 'families': len(profiles), 'pairs': len(distance), 'submissions': len(submission_summary)}",
            "load",
        ),
        markdown_cell("### 2. Validate grain, alignment, hierarchy, and cuts", "validate-title"),
        code_cell(
            "n = len(assignments)\n"
            "family_sizes = Counter(row['provisional_family_id'] for row in assignments)\n"
            "profile_sizes = {row['family_id']: int(row['size']) for row in profiles}\n"
            "family_counts = [int(row['family_count']) for row in sensitivity]\n"
            "singleton_counts = [int(row['singleton_count']) for row in sensitivity]\n"
            "checks = {\n"
            "    'seat_ids_unique': len({(r['episode_id'], r['player_index'], r['submission_id']) for r in assignments}) == n,\n"
            "    'seat_feature_alignment': [r['seat_index'] for r in assignments] == [str(i) for i in range(n)],\n"
            "    'condensed_pair_count': len(distance) == n * (n - 1) // 2,\n"
            "    'distance_finite_nonnegative': bool(np.isfinite(distance).all() and (distance >= 0).all()),\n"
            "    'linkage_shape': linkage_matrix.shape == (n - 1, 4),\n"
            "    'linkage_monotone': bool((np.diff(linkage_matrix[:, 2]) >= -1e-12).all()),\n"
            "    'family_sizes_reconcile': family_sizes == profile_sizes,\n"
            "    'one_medoid_per_family': sum(int(r['is_medoid']) for r in assignments) == len(profiles),\n"
            "    'sensitivity_monotone': all(a >= b for a, b in zip(family_counts, family_counts[1:])) and all(a >= b for a, b in zip(singleton_counts, singleton_counts[1:])),\n"
            "}\n"
            "assert all(checks.values()), checks\n"
            "checks",
            "validate",
        ),
        markdown_cell("### 3. Independently recompute sampled pair distances", "spot-title"),
        code_cell(
            "layout = features['layout']\n"
            "cumulative = features['cumulative_actions'].astype(np.int32)\n"
            "def condensed_index(n, i, j):\n"
            "    if i > j: i, j = j, i\n"
            "    return n * i - i * (i + 1) // 2 + (j - i - 1)\n"
            "def direct_distance(i, j):\n"
            "    c_parts, g_parts, t_parts = [], [], []\n"
            "    for anchor in range(5):\n"
            "        ci = np.bincount(layout[i, anchor], minlength=11)[1:11]\n"
            "        cj = np.bincount(layout[j, anchor], minlength=11)[1:11]\n"
            "        c_parts.append(np.abs(ci - cj).sum() / max(8, ci.sum(), cj.sum()))\n"
            "        union = (layout[i, anchor] != 0) | (layout[j, anchor] != 0)\n"
            "        g_parts.append((((layout[i, anchor] != layout[j, anchor]) & union).sum() / union.sum()) if union.any() else 0.0)\n"
            "    for stage in range(10):\n"
            "        qi, qj = cumulative[i, stage], cumulative[j, stage]\n"
            "        t_parts.append(np.abs(qi - qj).sum() / max(6, qi.sum(), qj.sum()))\n"
            "    return 0.45 * np.mean(c_parts) + 0.35 * np.mean(g_parts) + 0.20 * np.mean(t_parts)\n"
            "rng = np.random.default_rng(20260825)\n"
            "pairs = rng.integers(0, n, size=(200, 2))\n"
            "pairs = [(int(i), int(j)) for i, j in pairs if i != j]\n"
            "errors = [abs(direct_distance(i, j) - float(distance[condensed_index(n, i, j)])) for i, j in pairs]\n"
            "spot_check = {'pairs_checked': len(errors), 'max_abs_error': max(errors), 'mean_abs_error': float(np.mean(errors))}\n"
            "assert spot_check['max_abs_error'] < 2e-7\n"
            "spot_check",
            "spot",
        ),
        markdown_cell("## Results\n\n### 4. Main cut and sensitivity", "results"),
        code_cell(
            "{\n"
            "    'main_cut': summary['main_cut'],\n"
            "    'distance_summary': summary['distance'],\n"
            "    'family_size_summary': summary['family_size'],\n"
            "    'cophenetic_correlation': float(__import__('scipy.cluster.hierarchy', fromlist=['cophenet']).cophenet(linkage_matrix, distance)[0]),\n"
            "    'zero_distance_pair_share': float(np.count_nonzero(distance == 0) / len(distance)),\n"
            "}",
            "main-results",
        ),
        markdown_cell("### 5. Visual overview", "visual-title"),
        code_cell(
            "from PIL import Image, ImageDraw, ImageFont\n"
            "canvas = Image.new('RGB', (1400, 760), 'white')\n"
            "draw = ImageDraw.Draw(canvas)\n"
            "font_path = r'C:\\Windows\\Fonts\\arial.ttf'\n"
            "bold_path = r'C:\\Windows\\Fonts\\arialbd.ttf'\n"
            "font = ImageFont.truetype(font_path, 22)\n"
            "small = ImageFont.truetype(font_path, 18)\n"
            "title = ImageFont.truetype(bold_path, 28)\n"
            "ink, grid, blue, blue_open, gold = '#263238', '#DDE3E8', '#1769AA', '#80B7D8', '#D99A2B'\n"
            "draw.text((60, 30), 'Macro-intent clustering overview', font=title, fill=ink)\n"
            "draw.text((60, 70), '3,397 seats; average linkage; provisional N families', font=small, fill='#60727D')\n"
            "# Panel A: sensitivity line chart\n"
            "x0, y0, x1, y1 = 80, 150, 720, 620\n"
            "draw.text((x0, 105), 'Family counts across distance thresholds', font=font, fill=ink)\n"
            "thresholds = [float(r['threshold']) for r in sensitivity]\n"
            "families = [int(r['family_count']) for r in sensitivity]\n"
            "singletons = [int(r['singleton_count']) for r in sensitivity]\n"
            "ymax = 1000\n"
            "for tick in range(0, ymax + 1, 200):\n"
            "    y = y1 - (y1-y0) * tick / ymax\n"
            "    draw.line((x0, y, x1, y), fill=grid, width=1)\n"
            "    draw.text((x0-55, y-10), str(tick), font=small, fill='#60727D')\n"
            "xs = [x0 + (x1-x0) * i / (len(thresholds)-1) for i in range(len(thresholds))]\n"
            "for x, value in zip(xs, thresholds):\n"
            "    draw.text((x-18, y1+15), f'{value:.2f}', font=small, fill='#60727D')\n"
            "def ycoord(value): return y1 - (y1-y0) * value / ymax\n"
            "for values, color, width in [(families, blue, 4), (singletons, blue_open, 4)]:\n"
            "    points = [(x, ycoord(v)) for x, v in zip(xs, values)]\n"
            "    draw.line(points, fill=color, width=width)\n"
            "    for x, y in points: draw.ellipse((x-5, y-5, x+5, y+5), fill='white', outline=color, width=3)\n"
            "draw.line((x0, y0, x0, y1), fill=ink, width=2); draw.line((x0, y1, x1, y1), fill=ink, width=2)\n"
            "draw.line((x0+20, 660, x0+70, 660), fill=blue, width=4); draw.text((x0+80, 648), 'Families', font=small, fill=ink)\n"
            "draw.line((x0+220, 660, x0+270, 660), fill=blue_open, width=4); draw.text((x0+280, 648), 'Singletons', font=small, fill=ink)\n"
            "# Panel B: part-to-whole concentration at 0.12\n"
            "bx0, bx1 = 810, 1330\n"
            "draw.text((bx0, 105), 'Seat share at the main cut (D <= 0.12)', font=font, fill=ink)\n"
            "sizes = sorted((int(r['size']) for r in profiles), reverse=True)\n"
            "parts = [('N001', sizes[0], blue), ('N002-N010', sum(sizes[1:10]), gold), ('Long tail', sum(sizes[10:]), '#B0BEC5')]\n"
            "bar_y, bar_h, current = 230, 90, bx0\n"
            "for label, value, color in parts:\n"
            "    width = (bx1-bx0) * value / n\n"
            "    draw.rectangle((current, bar_y, current+width, bar_y+bar_h), fill=color, outline='white', width=2)\n"
            "    current += width\n"
            "legend_y = 390\n"
            "for label, value, color in parts:\n"
            "    draw.rectangle((bx0, legend_y, bx0+24, legend_y+24), fill=color)\n"
            "    draw.text((bx0+40, legend_y-3), f'{label}: {value:,} ({value/n:.1%})', font=font, fill=ink)\n"
            "    legend_y += 58\n"
            "draw.text((bx0, 580), 'The long tail contains 667 families.', font=small, fill='#60727D')\n"
            "draw.text((bx0, 615), 'Source: reconstructed replay macro intents.', font=small, fill='#60727D')\n"
            "chart_path = cluster_dir / 'cluster_overview_v1.png'\n"
            "canvas.save(chart_path)\n"
            "{'chart': str(chart_path), 'size_pixels': canvas.size}",
            "visual",
        ),
        markdown_cell("![Cluster overview](cluster_overview_v1.png)", "visual-image"),
        markdown_cell("### 6. Largest families and submission-level tail", "detail-title"),
        code_cell(
            "largest_families = [{k: row[k] for k in ['family_id','size','share','median_reward','win_rate','unique_submissions','medoid_team','medoid_seat_id']} for row in profiles[:10]]\n"
            "long_tail_by_submission = sorted(\n"
            "    [{k: row[k] for k in ['rank','team','seat_count','family_count','singleton_seats','singleton_share','dominant_family_id','dominant_family_share']} for row in submission_summary],\n"
            "    key=lambda row: int(row['singleton_seats']), reverse=True\n"
            ")\n"
            "{'largest_families': largest_families, 'submissions_by_singletons': long_tail_by_submission}",
            "details",
        ),
        markdown_cell(
            "## Takeaways\n\n"
            "- 当前池并不是均匀分成数百族，而是一个跨 25 个 submission 的主流宏观骨架，加上少量中型族和很长的个性化尾部。\n"
            "- 566 个单例全部来自 6 个 submission，主要集中在 rank 3、rank 1、rank 2 与 rank 39；"
            "因此高族数主要反映这些 Agent 的逐局变化，不是所有队伍都同样碎片化。\n"
            "- 把切点从 0.12 放宽到 0.20，族数仍为 377，说明长尾对切点具有持续性；下一步不应只靠放宽阈值压缩族数。\n"
            "- 这批 `N###` 适合作为新样本内部的结构地图。正式恢复历史宏观意图时，应把它们与旧 275 族联合重聚类或做 medoid 对齐后再授予 `G###`。",
            "takeaways",
        ),
    ]

    namespace: dict[str, Any] = {"__name__": "__cluster_notebook__"}
    execution_count = 0
    for cell in cells:
        if cell["cell_type"] == "code":
            execution_count += 1
            execute_code_cell(cell, namespace, execution_count)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0] if 'sys' in globals() else "3"},
            "execution": {"mode": "sequential-python-namespace", "status": "passed"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
