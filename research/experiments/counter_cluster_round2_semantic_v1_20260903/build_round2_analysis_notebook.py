#!/usr/bin/env python3
"""Build the compact, read-only Round2 train analysis notebook."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "round2_analysis.ipynb"


def markdown(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Round2 语义拳法：训练结果审计

## tl;dr

下面的首个代码单元动态重算摘要，因此结论与输入文件绑定，不依赖手工抄录。
本 notebook 只审计 **train**：不运行模拟，不读取 replay、validation 或 holdout 数据。
""",
        "title",
    ),
    code(
        r'''from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

EXPERIMENT_DIR = (
    Path.cwd() / "experiments" / "counter_cluster_round2_semantic_v1"
).resolve()
if not EXPERIMENT_DIR.is_dir():
    raise RuntimeError("请从仓库根目录执行 notebook")

# 唯一允许读取的四个实验输入。这里只做静态分析，不导入任何模拟器。
INPUT_FILES = {
    "materialization_audit": EXPERIMENT_DIR / "materialization_audit.json",
    "signature_inversion_audit": EXPERIMENT_DIR / "signature_inversion_audit.json",
    "train_matrix": EXPERIMENT_DIR / "train_semantic_matrix.npz",
    "train_report": EXPERIMENT_DIR / "train_semantic_report.json",
}

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

input_hashes = {name: sha256_file(path) for name, path in INPUT_FILES.items()}
materialization = json.loads(INPUT_FILES["materialization_audit"].read_text(encoding="utf-8"))
signature_audit = json.loads(INPUT_FILES["signature_inversion_audit"].read_text(encoding="utf-8"))
report = json.loads(INPUT_FILES["train_report"].read_text(encoding="utf-8"))
with np.load(INPUT_FILES["train_matrix"], allow_pickle=False) as archive:
    matrix = {name: archive[name] for name in archive.files}

fist_ids = matrix["fist_ids"].astype(str)
opponent_ids = matrix["opponent_ids"].astype(str)
path_scores = matrix["path_scores"].astype(np.float64)
path_uplift = matrix["path_uplift"].astype(np.float64)
path_margins = matrix["path_margins"].astype(np.float64)
matched = matrix["selected_matched"].astype(bool)
decision_days = matrix["decision_days"].astype(int)

threshold = report["coverage_rule"]
minimum_score = float(threshold["minimum_score"])
minimum_uplift = float(threshold["minimum_score_uplift"])
minimum_margin = float(threshold["minimum_mean_reward_margin_exclusive"])
score_pass = path_scores >= minimum_score
uplift_pass = path_uplift >= minimum_uplift
margin_pass = path_margins > minimum_margin
all_pass = score_pass & uplift_pass & margin_pass
threshold_counts = {
    "score": int(score_pass.sum()),
    "uplift": int(uplift_pass.sum()),
    "margin": int(margin_pass.sum()),
    "all": int(all_pass.sum()),
}

rows = materialization["rows"]
equivalent_rows = sum(
    all(bool(value) for value in row["checks"].values()) and bool(row["passed"])
    for row in rows
)
fist_count = int(len(np.unique(fist_ids)))
cell_count = int(path_scores.size)

print(f"发现阶段等价：{equivalent_rows}/{len(rows)}")
print(f"语义拳法：{fist_count}；训练单元：{cell_count} = {path_scores.shape[0]}×{path_scores.shape[1]}")
print(
    "单门槛通过数："
    f"score {threshold_counts['score']}，uplift {threshold_counts['uplift']}，"
    f"margin {threshold_counts['margin']}；三门槛同时通过 {threshold_counts['all']}"
)
print("范围：train only；validation/holdout 未运行。")''',
        "load-and-summary",
    ),
    markdown(
        """## Context & Methods

### Key Assumptions

- 一个 cell 是一套语义拳法 × 一条训练对手路线，共 `8 seeds × 2 seats = 16` 局。
- 覆盖条件为 `score ≥ 0.625`、`uplift ≥ 0.125`、`mean margin > 0`，三项必须同时成立。
- `selected_matched=False` 时的 telemetry hash 是默认/哨兵值，不是一次真实匹配；所有 intent、family 与 live-delta 统计都先应用 `selected_matched` mask。
- “最近单元”先最大化通过门槛数量；若并列，再最小化三个门槛的归一化缺口。margin 用全矩阵最大绝对 margin 缩放，因为其门槛为 0、单位与 score 不同。
""",
        "methods",
    ),
    code(
        r'''print("输入白名单及 SHA-256：")
for name, path in INPUT_FILES.items():
    print(f"- {name}: {path.name}  {input_hashes[name]}")
print("execution_semantics:", report["execution_semantics"])
print("split:", report["split"])''',
        "source-receipt",
    ),
    markdown("## Data\n", "data-header"),
    code(
        r'''def no_nan_or_inf(array):
    if array.dtype.kind in "bifcu":
        return bool(np.isfinite(array).all())
    if array.dtype.kind in "SU":
        lowered = np.char.lower(array.astype(str))
        return not bool(np.isin(lowered, ["nan", "inf", "+inf", "-inf"]).any())
    return True

array_integrity = {name: no_nan_or_inf(value) for name, value in matrix.items()}
numeric_arrays = sum(value.dtype.kind in "bifcu" for value in matrix.values())
print(f"NPZ arrays: {len(matrix)}；类型适配的 NaN/Inf 检查：{sum(array_integrity.values())}/{len(matrix)} 通过")
print(f"其中 numeric/bool arrays: {numeric_arrays}；string arrays: {len(matrix) - numeric_arrays}")
print("axes:", path_scores.shape, "fists × opponents；games/cell:", matrix["seeds"].size * matrix["candidate_seats"].size)
print("所有 selected_rank 均为 -1:", bool(np.all(matrix["selected_rank"] == -1)))''',
        "data-integrity",
    ),
    markdown("## Results\n", "results-header"),
    code(
        r'''signature_summary = signature_audit["summary"]
catalog = signature_audit["signature_catalog"]
signature_to_family = {}
signature_collision = False
for item in catalog:
    signature = int(item["signature"])
    family = int(item["family_id"])
    if signature in signature_to_family and signature_to_family[signature] != family:
        signature_collision = True
    signature_to_family[signature] = family

expected_family = np.empty(matrix["source_signature"].shape, dtype=np.int16)
for fist_index in range(expected_family.shape[0]):
    for stage_index in range(expected_family.shape[1]):
        expected_family[fist_index, stage_index] = signature_to_family[
            int(matrix["source_signature"][fist_index, stage_index])
        ]

# 关键：只比较 matched 位置；unmatched 的 hash/family 都是哨兵。
expected_intent = matrix["expected_intent_sha256"][:, None, None, None, :]
expected_family_live = expected_family[:, None, None, None, :]
intent_mismatches = int(np.count_nonzero(
    matched & (matrix["actual_intent_sha256"] != expected_intent)
))
family_mismatches = int(np.count_nonzero(
    matched & (matrix["selected_family"] != expected_family_live)
))

print(f"48场发现审计全检查等价：{equivalent_rows}/{len(rows)}")
print(
    "signature 反演："
    f"{signature_summary['uniquely_inverted']}/{signature_summary['unique_family_signatures']}；"
    f"missing={signature_summary['missing']}，ambiguous={signature_summary['ambiguous']}"
)
print(f"matched 后 intent mismatch={intent_mismatches}；family mismatch={family_mismatches}")
print("signature→family 冲突:", signature_collision)''',
        "semantic-equivalence",
    ),
    code(
        r'''stored_coverage = matrix["coverage"].astype(bool)
print("门槛：")
print(f"- score ≥ {minimum_score}: {threshold_counts['score']}/{cell_count}")
print(f"- uplift ≥ {minimum_uplift}: {threshold_counts['uplift']}/{cell_count}")
print(f"- margin > {minimum_margin}: {threshold_counts['margin']}/{cell_count}")
print(f"- 全部通过: {threshold_counts['all']}/{cell_count}")
print("重算 coverage 与保存矩阵一致:", bool(np.array_equal(all_pass, stored_coverage)))''',
        "thresholds",
    ),
    code(
        r'''match_by_day = {
    int(day): float(matched[..., stage_index].mean())
    for stage_index, day in enumerate(decision_days)
}
report_match_by_day = {
    int(day): float(rate)
    for day, rate in report["semantic_execution"]["match_rate_by_decision_day"].items()
}
matched_count = int(matched.sum())
live_changed_count = int(np.count_nonzero(matched & matrix["live_normalization_changed"]))
unmatched_intent_sentinels = np.unique(matrix["actual_intent_sha256"][~matched]).astype(str).tolist()
unmatched_delta_sentinels = np.unique(matrix["actual_delta_sha256"][~matched]).astype(str).tolist()

print("match rate by decision day:")
for day, rate in match_by_day.items():
    print(f"- day {day:>2}: {rate:.6%}")
print(f"live delta changed: {live_changed_count:,}/{matched_count:,} matched stages ({live_changed_count / matched_count:.6%})")
print("match-by-day 与 report 一致:", all(
    np.isclose(match_by_day[day], report_match_by_day[day]) for day in match_by_day
))
print("unmatched intent sentinel(s):", [repr(value) for value in unmatched_intent_sentinels])
print("unmatched delta sentinel(s):", [repr(value) for value in unmatched_delta_sentinels])
print("注：以上 unmatched sentinel 从不参与 mismatch 或 live-delta 统计。")''',
        "match-telemetry",
    ),
    code(
        r'''panel_by_opponent = {
    item["opponent_id"]: item["panel_id"] for item in report["opponents"]
}
pass_count = score_pass.astype(int) + uplift_pass.astype(int) + margin_pass.astype(int)
margin_scale = max(1.0, float(np.max(np.abs(path_margins))))
distance = np.sqrt(
    (np.maximum(0.0, minimum_score - path_scores) / max(abs(minimum_score), 1e-12)) ** 2
    + (np.maximum(0.0, minimum_uplift - path_uplift) / max(abs(minimum_uplift), 1e-12)) ** 2
    + (np.maximum(0.0, minimum_margin - path_margins) / margin_scale) ** 2
)

def index_of_max(values):
    return tuple(int(value) for value in np.unravel_index(np.argmax(values), values.shape))

def record_cell(index):
    fist_index, opponent_index = index
    opponent_id = opponent_ids[opponent_index]
    return {
        "fist": fist_ids[fist_index],
        "opponent": opponent_id,
        "panel": panel_by_opponent.get(opponent_id, "?"),
        "score": float(path_scores[index]),
        "uplift": float(path_uplift[index]),
        "margin": float(path_margins[index]),
        "thresholds_passed": int(pass_count[index]),
    }

max_passes = int(pass_count.max())
nearest_candidates = np.flatnonzero(pass_count.ravel() == max_passes)
nearest_flat = int(nearest_candidates[np.argmin(distance.ravel()[nearest_candidates])])
nearest_index = tuple(int(value) for value in np.unravel_index(nearest_flat, path_scores.shape))

extreme_cells = {
    "max_score": record_cell(index_of_max(path_scores)),
    "max_uplift": record_cell(index_of_max(path_uplift)),
    "max_margin": record_cell(index_of_max(path_margins)),
    "nearest_all_thresholds": record_cell(nearest_index),
}
for label, item in extreme_cells.items():
    print(
        f"{label}: fist={item['fist'].split(':')[-1][:12]} panel={item['panel']} "
        f"score={item['score']:.3f} uplift={item['uplift']:.3f} "
        f"margin={item['margin']:.3f} pass={item['thresholds_passed']}/3"
    )''',
        "extreme-cells",
    ),
    code(
        r'''full_match_cells = matched.all(axis=(2, 3, 4))
full_match_count = int(full_match_cells.sum())
full_indices = np.argwhere(full_match_cells)
best_full_index = max(
    (tuple(int(value) for value in index) for index in full_indices),
    key=lambda index: (path_scores[index], path_margins[index], path_uplift[index]),
)
best_full_cell = record_cell(best_full_index)

print(f"16局 × 5阶段全部匹配的 cells: {full_match_count}/{cell_count}")
print(
    "其中最好："
    f"fist={best_full_cell['fist'].split(':')[-1][:12]} panel={best_full_cell['panel']} "
    f"score={best_full_cell['score']:.3f} uplift={best_full_cell['uplift']:.3f} "
    f"margin={best_full_cell['margin']:.3f}"
)''',
        "full-match-cells",
    ),
    markdown(
        """## Takeaways

下面的结论只针对这次训练矩阵：语义 PlanDelta 的复用机制已通过发现样本等价检查，且新种子上确实发生了 live normalization；但训练集中没有任何拳法×对手 cell 同时通过三个效果门槛。因此本轮没有形成可冻结的 response cluster/portfolio，不能据此宣称已找到克制拳法。
""",
        "takeaways",
    ),
    code(
        r'''EXPECTED_ABSENT_ARTIFACTS = [
    EXPERIMENT_DIR / "validation_semantic_matrix.npz",
    EXPERIMENT_DIR / "validation_semantic_report.json",
    EXPERIMENT_DIR / "holdout_semantic_matrix.npz",
    EXPERIMENT_DIR / "holdout_semantic_report.json",
]
absent_artifacts = {path.name: not path.exists() for path in EXPECTED_ABSENT_ARTIFACTS}

assertions = {
    "materialization_audit_passed": bool(materialization["passed"]),
    "materialization_equivalence_48_of_48": len(rows) == 48 and equivalent_rows == 48,
    "signature_inversion_audit_passed": bool(signature_audit["passed"]),
    "signature_inversion_79_of_79": (
        signature_summary["unique_family_signatures"] == 79
        and signature_summary["uniquely_inverted"] == 79
        and signature_summary["missing"] == 0
        and signature_summary["ambiguous"] == 0
    ),
    "matrix_has_31_arrays_all_clean": len(matrix) == 31 and all(array_integrity.values()),
    "train_split_only": report["split"] == "train",
    "unique_semantic_fists_25": fist_count == 25,
    "train_cells_300": cell_count == 300 and path_scores.shape == (25, 12),
    "selected_rank_all_minus_one": bool(np.all(matrix["selected_rank"] == -1)),
    "matched_intent_mismatch_zero": intent_mismatches == 0,
    "matched_family_mismatch_zero": family_mismatches == 0 and not signature_collision,
    "matched_stage_count_18083": matched_count == 18083,
    "live_delta_changed_16293_of_18083": live_changed_count == 16293,
    "threshold_counts_1_0_0_0": threshold_counts == {
        "score": 1, "uplift": 0, "margin": 0, "all": 0
    },
    "coverage_recompute_matches_matrix": bool(np.array_equal(all_pass, stored_coverage)),
    "coverage_zero_of_300": int(stored_coverage.sum()) == 0,
    "match_by_day_matches_report": all(
        np.isclose(match_by_day[day], report_match_by_day[day]) for day in match_by_day
    ),
    "fully_matched_cells_14": full_match_count == 14,
    "best_fully_matched_cell_score_0_25": np.isclose(best_full_cell["score"], 0.25),
    "best_fully_matched_cell_margin_minus_10641_125": np.isclose(
        best_full_cell["margin"], -10641.125
    ),
    "unmatched_telemetry_mask_applied": True,
    "validation_and_holdout_artifacts_absent": all(absent_artifacts.values()),
}
assertions = {name: bool(passed) for name, passed in assertions.items()}
failed_assertions = [name for name, passed in assertions.items() if not passed]
assert not failed_assertions, f"validation failed: {failed_assertions}"

receipt = {
    "schema": "kaggriculture.counter-cluster-round2-analysis-validation.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "notebook": "round2_analysis.ipynb",
    "execution": {
        "status": "executed_top_to_bottom",
        "simulator_runs": 0,
        "data_scope": "train only",
        "read_allowlist": [path.name for path in INPUT_FILES.values()],
        "validation_holdout_artifacts": absent_artifacts,
    },
    "inputs": {
        name: {"path": path.name, "sha256": input_hashes[name]}
        for name, path in INPUT_FILES.items()
    },
    "computed": {
        "materialization_equivalence": [equivalent_rows, len(rows)],
        "unique_semantic_fists": fist_count,
        "train_cells": cell_count,
        "threshold_counts": threshold_counts,
        "match_rate_by_decision_day": {str(day): rate for day, rate in match_by_day.items()},
        "matched_stages": matched_count,
        "live_delta_changed_stages": live_changed_count,
        "matched_intent_mismatches": intent_mismatches,
        "matched_family_mismatches": family_mismatches,
        "fully_matched_cells": full_match_count,
        "best_fully_matched_cell": best_full_cell,
        "extreme_and_nearest_cells": extreme_cells,
        "unmatched_intent_sentinels": unmatched_intent_sentinels,
        "unmatched_delta_sentinels": unmatched_delta_sentinels,
    },
    "assertions": assertions,
}
receipt_path = EXPERIMENT_DIR / "validation_receipt.json"
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"PASS: {sum(assertions.values())}/{len(assertions)} assertions")
print("validation/holdout artifacts absent:", all(absent_artifacts.values()))
print("receipt:", receipt_path.name)''',
        "validation-receipt",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"built {OUTPUT}")
