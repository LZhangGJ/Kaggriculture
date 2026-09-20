#!/usr/bin/env python3
"""Render the macro-route Markdown report as a compact, self-contained PDF."""

from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

import markdown
from matplotlib.mathtext import math_to_image
from weasyprint import CSS, HTML


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs/宏观路线去重聚类与切换搜参.md"
DEFAULT_OUTPUT = ROOT / "docs/宏观路线去重聚类与切换搜参.pdf"


CSS_TEXT = r"""
@page {
  size: A4;
  margin: 13mm 14mm 15mm;
  @bottom-left {
    content: "宏观路线去重、聚类与切换搜参";
    color: #738096;
    font-size: 7.5pt;
  }
  @bottom-right {
    content: counter(page) " / " counter(pages);
    color: #738096;
    font-size: 7.5pt;
  }
}
html { font-family: "Noto Sans CJK SC", "Noto Sans SC", sans-serif; color: #172033; }
body { font-size: 9.2pt; line-height: 1.47; }
h1, h2, h3 { color: #172033; page-break-after: avoid; }
h1 { font-size: 22pt; line-height: 1.18; margin: 0 0 8mm; border-bottom: 2.2pt solid #356ae6; padding-bottom: 4mm; }
h2 { font-size: 14.5pt; margin: 6mm 0 2.4mm; border-left: 3.5pt solid #356ae6; padding-left: 2.5mm; }
h3 { font-size: 11.2pt; margin: 4mm 0 1.8mm; color: #2455b8; }
p { margin: 1.6mm 0 2.2mm; orphans: 3; widows: 3; }
blockquote { margin: 3mm 0; padding: 2.5mm 4mm; background: #f2f6ff; border-left: 3pt solid #356ae6; color: #38445a; }
blockquote p { margin: 0; }
ul, ol { margin: 1.6mm 0 2.5mm 5.5mm; padding-left: 4mm; }
li { margin: .7mm 0; }
strong { color: #0d347f; }
code { font-family: "Noto Sans Mono CJK SC", monospace; font-size: 8.3pt; background: #f2f4f8; padding: .15mm .6mm; border-radius: 1mm; }
pre { white-space: pre-wrap; background: #f2f4f8; border: .5pt solid #dfe5ef; padding: 2.5mm; font-size: 8pt; }
table { width: 100%; border-collapse: collapse; margin: 2.5mm 0 4mm; font-size: 8.1pt; page-break-inside: avoid; }
thead { display: table-header-group; background: #eaf0fd; }
th { color: #163d91; font-weight: 700; }
th, td { border: .55pt solid #cad3e3; padding: 1.35mm 1.7mm; vertical-align: top; }
tbody tr:nth-child(even) { background: #f8f9fc; }
img { display: block; max-width: 100%; height: auto; margin: 3mm auto 4mm; page-break-inside: avoid; }
img.formula { max-height: 12mm; margin: 2.2mm auto 3mm; }
a { color: #245bc7; text-decoration: none; }
hr { border: 0; border-top: .6pt solid #dfe5ef; }
"""


def formula_image(match: re.Match[str]) -> str:
    formula = " ".join(match.group(1).strip().splitlines())
    formula = formula.replace(r"\Delta_{threshold\ worst}", r"\Delta_{\mathrm{threshold\ worst}}")
    buffer = io.BytesIO()
    try:
        math_to_image(f"${formula}$", buffer, format="svg", dpi=180, color="#172033")
    except Exception:
        # Keep the equation readable even if a future formula exceeds mathtext's subset.
        return f'<pre class="formula">{formula}</pre>'
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="formula" alt="formula" src="data:image/svg+xml;base64,{encoded}">'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    raw = source.read_text(encoding="utf-8")
    raw = re.sub(r"\\\[\s*(.*?)\s*\\\]", formula_image, raw, flags=re.DOTALL)
    content = markdown.markdown(
        raw,
        extensions=("tables", "fenced_code", "sane_lists"),
        output_format="html5",
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>宏观路线去重、聚类与切换搜参</title></head>
<body>{content}</body>
</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=document, base_url=str(source.parent)).write_pdf(
        output,
        stylesheets=[CSS(string=CSS_TEXT)],
        presentational_hints=True,
        custom_metadata=True,
    )
    print(output)


if __name__ == "__main__":
    main()
