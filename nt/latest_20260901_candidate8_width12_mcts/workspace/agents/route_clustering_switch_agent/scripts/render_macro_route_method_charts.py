#!/usr/bin/env python3
"""Render self-contained SVG charts for the macro-route methodology note."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/macro-route-unified-20260825"
OUT = ROOT / "docs/macro_route_charts"

INK = "#172033"
MUTED = "#65708a"
BLUE = "#356ae6"
CYAN = "#23a4a8"
ORANGE = "#e58a32"
GREEN = "#319b64"
RED = "#d85b68"
GRID = "#dfe5ef"
PANEL = "#f6f8fc"


def load(name: str) -> dict:
    return json.loads((EXP / name).read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value))


def text(x: float, y: float, value: object, size: int = 20, *,
         fill: str = INK, anchor: str = "start", weight: int = 400) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Noto Sans CJK SC, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{esc(value)}</text>'
    )


def svg(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def rect(x: float, y: float, w: float, h: float, fill: str = PANEL,
         stroke: str = GRID, radius: int = 12, sw: int = 2) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def render_pipeline() -> None:
    stages = [
        ("统一回放", "3,563 个座位", "旧库 1,165 + 新库 2,398"),
        ("意图聚类", "275 个路线族", "平均连接；切割阈值 0.12"),
        ("历史筛选", "175 个候选族", "删 100 个历史 0 胜族"),
        ("粗反事实搜索", "175 → 11 条目标路线", "1,372 万局；8 种子×双座位"),
        ("鲁棒树与序列搜索", "3 个切换节点", "144 / 168 / 216；一次切换锁定"),
        ("独立留出", "+8.44 个百分点", "256 新种子；95% 下界 +7.64pp"),
    ]
    width, height = 1740, 420
    body = [text(50, 48, "宏观路线构建与搜索漏斗", 30, weight=700)]
    box_w, gap, x0, y = 245, 42, 45, 100
    for i, (title, main, note) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        color = [BLUE, CYAN, ORANGE, BLUE, CYAN, GREEN][i]
        body.append(rect(x, y, box_w, 205, "white", color, 14, 3))
        body.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="12" rx="6" fill="{color}"/>')
        body.append(text(x + box_w / 2, y + 55, title, 21, anchor="middle", weight=700))
        body.append(text(x + box_w / 2, y + 105, main, 24, fill=color, anchor="middle", weight=700))
        words = note.split("；")
        for line, word in enumerate(words):
            body.append(text(x + box_w / 2, y + 151 + line * 29, word, 16, fill=MUTED, anchor="middle"))
        if i < len(stages) - 1:
            ax = x + box_w + 7
            body.append(f'<line x1="{ax}" y1="{y+102}" x2="{ax+27}" y2="{y+102}" stroke="{MUTED}" stroke-width="3"/>')
            body.append(f'<path d="M {ax+27} {y+102} l -9 -7 v 14 z" fill="{MUTED}"/>')
    body.append(text(50, 365, "计算主体：C++ 仿真与反事实评测；Python 仅负责聚类、决策树拟合和导出。", 18, fill=MUTED))
    (OUT / "pipeline.svg").write_text(svg(width, height, body), encoding="utf-8")


def render_sensitivity() -> None:
    data = load("intent-clusters-summary-v1.json")["sensitivity"]
    xs = [float(k) for k in data]
    families = [data[str(x) if str(x) in data else f"{x:g}"]["families"] for x in xs]
    singletons = [data[str(x) if str(x) in data else f"{x:g}"]["singletons"] for x in xs]
    coverage = [data[str(x) if str(x) in data else f"{x:g}"]["top56_coverage"] * 100 for x in xs]
    width, height = 1200, 660
    left, right, top, bottom = 105, 1100, 95, 545
    body = [text(50, 48, "聚类阈值敏感性", 30, weight=700),
            text(50, 76, "阈值越大，路线族越少；头部 56 族覆盖率提高，但异质计划更易被合并。", 17, fill=MUTED)]
    for v in range(0, 501, 100):
        y = bottom - v / 500 * (bottom - top)
        body.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
        body.append(text(left - 15, y + 6, v, 15, fill=MUTED, anchor="end"))
    for i, xval in enumerate(xs):
        x = left + i / (len(xs) - 1) * (right - left)
        body.append(text(x, bottom + 32, f"{xval:.2f}", 15, fill=MUTED, anchor="middle"))
    def path(values: list[float], color: str) -> None:
        points = []
        for i, value in enumerate(values):
            x = left + i / (len(values) - 1) * (right - left)
            y = bottom - value / 500 * (bottom - top)
            points.append(f"{x},{y}")
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="4"/>')
        for point in points:
            x, y = point.split(",")
            body.append(f'<circle cx="{x}" cy="{y}" r="5" fill="white" stroke="{color}" stroke-width="3"/>')
    path(families, BLUE)
    path(singletons, ORANGE)
    selected = xs.index(0.12)
    sx = left + selected / (len(xs) - 1) * (right - left)
    body.append(f'<line x1="{sx}" y1="{top}" x2="{sx}" y2="{bottom}" stroke="{GREEN}" stroke-width="2" stroke-dasharray="7 7"/>')
    body.append(text(sx + 12, top + 24, "当前切点 0.12", 17, fill=GREEN, weight=700))
    body.append(text(sx + 12, top + 50, "275 族 / 211 单例", 16, fill=GREEN))
    body.append(text(sx + 12, top + 75, f"头部56族覆盖 {coverage[selected]:.2f}%", 16, fill=GREEN))
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>')
    body.append(text(left, 610, "— 路线族数量", 17, fill=BLUE, weight=700))
    body.append(text(left + 190, 610, "— 单例族数量", 17, fill=ORANGE, weight=700))
    body.append(text(right, 610, "横轴：平均连接距离切割阈值", 16, fill=MUTED, anchor="end"))
    (OUT / "cluster_sensitivity.svg").write_text(svg(width, height, body), encoding="utf-8")


def render_distribution() -> None:
    routes = load("macro-intent-routes-v2.json")["routes"][:12]
    total = load("intent-clusters-summary-v1.json")["rows"]
    width, height = 1300, 680
    left, right, top, bottom = 90, 1210, 100, 555
    maxv = max(r["support"] for r in routes)
    body = [text(45, 48, "头部路线族规模与累计覆盖", 30, weight=700),
            text(45, 76, "前 2 族占全部样本 50.21%，前 8 族占 84.51%；长尾数量多但样本少。", 17, fill=MUTED)]
    bw = (right - left) / len(routes) * 0.62
    cumulative = 0
    linepoints = []
    for i, route in enumerate(routes):
        center = left + (i + 0.5) / len(routes) * (right - left)
        h = route["support"] / maxv * (bottom - top)
        body.append(f'<rect x="{center-bw/2}" y="{bottom-h}" width="{bw}" height="{h}" rx="5" fill="{BLUE}" opacity="0.88"/>')
        body.append(text(center, bottom - h - 10, route["support"], 14, fill=BLUE, anchor="middle", weight=700))
        body.append(text(center, bottom + 28, route["family"], 15, fill=MUTED, anchor="middle"))
        cumulative += route["support"]
        cy = bottom - (cumulative / total) * (bottom - top)
        linepoints.append(f"{center},{cy}")
    body.append(f'<polyline points="{" ".join(linepoints)}" fill="none" stroke="{ORANGE}" stroke-width="4"/>')
    for point in linepoints:
        x, y = point.split(",")
        body.append(f'<circle cx="{x}" cy="{y}" r="5" fill="white" stroke="{ORANGE}" stroke-width="3"/>')
    body.append(text(right, 127, f"前12族覆盖 {cumulative/total*100:.2f}%", 17, fill=ORANGE, anchor="end", weight=700))
    body.append(text(left, 625, "蓝柱：族内样本数", 17, fill=BLUE, weight=700))
    body.append(text(left + 220, 625, "橙线：累计占 3,563 个回放座位的比例", 17, fill=ORANGE, weight=700))
    (OUT / "family_distribution.svg").write_text(svg(width, height, body), encoding="utf-8")


def render_validation() -> None:
    selection = load("intent-11-switch-sequence-search-128seeds-v1.json")["best"]
    holdout = load("intent-11-switch-final-holdout-256seeds-v1.json")["best"]
    groups = [("序列选择集\n128 种子", selection), ("独立留出集\n256 新种子", holdout)]
    width, height = 1120, 650
    left, right, top, bottom = 120, 1010, 110, 535
    ymin, ymax = 0.75, 0.93
    body = [text(45, 48, "固定 G001 与动态切换策略的配对得分", 30, weight=700),
            text(45, 76, "得分按胜=1、平=0.5、负=0；误差结论使用按种子聚合的单侧 95% 下界。", 17, fill=MUTED)]
    for v in (0.75, 0.80, 0.85, 0.90):
        y = bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
        body.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="{GRID}"/>')
        body.append(text(left - 15, y + 6, f"{v*100:.0f}%", 15, fill=MUTED, anchor="end"))
    for gi, (label, row) in enumerate(groups):
        cx = 340 + gi * 440
        for j, (key, color, name) in enumerate((("baseline_score", MUTED, "固定 G001"), ("dynamic_score", GREEN, "动态切换"))):
            value = row[key]
            x = cx - 90 + j * 180
            y = bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
            body.append(f'<rect x="{x-55}" y="{y}" width="110" height="{bottom-y}" rx="7" fill="{color}" opacity="0.9"/>')
            body.append(text(x, y - 12, f"{value*100:.2f}%", 18, fill=color, anchor="middle", weight=700))
            body.append(text(x, bottom + 30, name, 16, fill=MUTED, anchor="middle"))
        for line_i, line in enumerate(label.split("\n")):
            body.append(text(cx, bottom + 72 + line_i * 24, line, 17, anchor="middle", weight=700 if line_i == 0 else 400))
        body.append(text(cx, top + 20, f"提升 +{row['improvement']*100:.2f}pp", 19, fill=GREEN, anchor="middle", weight=700))
        body.append(text(cx, top + 48, f"95% 下界 +{row['one_sided_95pct_lower']*100:.2f}pp", 16, fill=GREEN, anchor="middle"))
    (OUT / "validation.svg").write_text(svg(width, height, body), encoding="utf-8")


def render_switch_tree() -> None:
    width, height = 1500, 900
    body = [text(45, 48, "最终在线切换树（仅 G001 开局分支）", 30, weight=700),
            text(45, 77, "按 144 → 168 → 216 检查；首次预测非 G001 即切换并锁定。G136 开局没有通过鲁棒门槛的切换节点。", 17, fill=MUTED)]
    panels = [
        (120, "step 144", [
            ("yarn_store 未解锁", "G001"),
            ("yarn_store 已解锁 且 milk_price ≤ 192", "G006"),
            ("yarn_store 已解锁 且 milk_price > 192", "G036"),
        ]),
        (375, "step 168（仍为 G001 时）", [
            ("未解锁 yarn 且 tomato_inventory ≤ 9966", "G169"),
            ("未解锁 yarn；milk_inventory > 9990；strawberry_inventory > 9990", "G006"),
            ("未解锁 yarn 的其余状态", "G001"),
            ("已解锁 yarn 且 milk_price ≤ 187", "G006"),
            ("已解锁 yarn 且 milk_price > 187", "G036"),
        ]),
        (700, "step 216（仍为 G001 时）", [
            ("yarn_store 未解锁", "G001"),
            ("yarn_store 已解锁", "G036"),
        ]),
    ]
    for y, title, rules in panels:
        h = 185 if len(rules) <= 3 else 255
        body.append(rect(55, y, 1390, h, "white", GRID, 14, 2))
        body.append(text(85, y + 40, title, 22, fill=BLUE, weight=700))
        cols = 3 if len(rules) <= 3 else 3
        for i, (condition, target) in enumerate(rules):
            row, col = divmod(i, cols)
            x = 85 + col * 445
            yy = y + 78 + row * 82
            body.append(f'<circle cx="{x+8}" cy="{yy-6}" r="7" fill="{CYAN}"/>')
            body.append(text(x + 25, yy, condition, 16, fill=INK))
            body.append(text(x + 25, yy + 27, f"→ {target}", 19, fill=GREEN if target != "G001" else MUTED, weight=700))
    (OUT / "switch_tree.svg").write_text(svg(width, height, body), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render_pipeline()
    render_sensitivity()
    render_distribution()
    render_validation()
    render_switch_tree()
    print(f"wrote 5 charts to {OUT}")


if __name__ == "__main__":
    main()
