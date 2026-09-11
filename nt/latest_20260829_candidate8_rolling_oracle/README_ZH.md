# Candidate8 八类候选与滚动 Oracle 交接包

日期：2026-08-29
状态：**候选生成与离线 Oracle 验收完成；线上选择器未通过，禁止直接替换当前冠军。**

## 1. 本包包含什么

这次把规划器的局部调整能力扩展为八类统一的 `当前计划 + ΔPlan`：

1. 同一经营计划的调度/布局变化；
2. 八产业 `±1/2/3/4/6/8` 连续规模调整；
3. 单边扩产、缩产和 KEEP；
4. 两/三产业与人员、土地的联合调整；
5. 立即、延期和停止追加；
6. 先出售融资，再购买、雇工或扩地的有序事务；
7. 中期与终局后缀重选；
8. 杂草、部分成交、人员错位和任务断链的局部恢复。

这些候选均经过真实 719 步执行验证，没有复制 Rank1 的固定日期、坐标或目标数量。

## 2. 当前实验结论

### Candidate8 工程层

- C++ 完整测试：`20 passed`；
- 1,024 局执行硬错误/容量溢出：0；
- 八类候选均真实改变完整动作轨迹；
- 当前线上候选排序模型未通过留出验收，默认 R6 行为未改变；
- 完整续跑速度约 400–450 continuations/s。

### 静止对手的现金上限

8 个独立实际种子、双座位，共 16 局：

| 方法 | 平均现金 | 最低 | 最高 | 15 万达成率 |
|---|---:|---:|---:|---:|
| R6 基线 | 138,522 | 110,981 | 165,401 | 37.5% |
| 32 独立未来滚动选择 | 141,566 | 105,436 | 162,609 | 37.5% |
| 知道真实未来的滚动 Oracle | **167,747** | **153,830** | **182,838** | **100%** |

结论：静止对手下，候选空间和执行器具备稳定超过 15 万的能力；短板是根据当前可见状态选对候选。

### 强对手 G001

- R6 平均现金：65,005；
- 知道真实未来的滚动 Oracle：121,338；
- 平均提高 56,332，但平均分差仍为 `-36,051`；
- 当前只优化我方现金，不等于优化胜负。

因此，静止对手赚钱上限不能直接当作强对手对战上限。

## 3. 目录索引

- `workspace/agents/route_clustering_switch_agent/`
  - 可构建的 C++ 模拟器、动态规划器、Candidate8 生成器和 Python binding；
  - 未包含 `.venv-bench`、`build/`、`.so` 或缓存文件。
- `workspace/experiments/ecobot_adaptive_planner_v1/configs/`
  - 冻结 R6 genome。
- `workspace/experiments/ecobot_adaptive_planner_v1/artifacts/oracle/`
  - 复现实验所需的冻结动作库和元数据。
- `workspace/experiments/ecobot_adaptive_planner_v1/tools/`
  - 单阶段、多未来、行为效果、融资、配额、速度和滚动 Oracle 工具。
- `workspace/experiments/ecobot_adaptive_planner_v1/reports/`
  - W8、W9、W10 中文审计报告。
- `workspace/experiments/ecobot_adaptive_planner_v1/receipts/`
  - 最终基线、强对手、多未来和滚动 Oracle 机器收据。
- `MANIFEST_SHA256.tsv`
  - 本包所有交付文件的 SHA256。

## 4. 关键报告阅读顺序

1. `reports/W8_RANK1_THREE_VERSION_CANDIDATE_CAPABILITY_EVOLUTION_20260829_ZH.md`
2. `reports/W9_RANK1_CANDIDATE8_FINAL_ACCEPTANCE_20260829_ZH.md`
3. `reports/W9C_CANDIDATE8_COUNTERFACTUAL_SELECTION_ACCEPTANCE_20260829_ZH.md`
4. `reports/W10_CANDIDATE8_ROLLING_ORACLE_CEILING_ACCEPTANCE_20260829_ZH.md`

## 5. WSL 复现

在本包的 `workspace` 目录执行。建议 Python 3.12，并使用 16 个 CPU 线程：

```bash
cd agents/route_clustering_switch_agent/fast_kaggriculture
python3 -m venv .venv-bench
.venv-bench/bin/pip install -e . pytest
CMAKE_BUILD_PARALLEL_LEVEL=16 MAX_JOBS=16 \
  .venv-bench/bin/python setup.py build_ext --inplace
PYTHONPATH=python .venv-bench/bin/python -m pytest -q
```

随后回到 `workspace`，设置：

```bash
export PYTHONPATH="$PWD/agents/route_clustering_switch_agent/fast_kaggriculture/python:$PWD/agents/route_clustering_switch_agent/src"
PYTHON=agents/route_clustering_switch_agent/fast_kaggriculture/.venv-bench/bin/python
```

滚动 Oracle 工具入口：

```bash
$PYTHON experiments/ecobot_adaptive_planner_v1/tools/audit_candidate8_rolling_oracle.py --help
```

正式参数与输入哈希保存在对应 JSON 收据中。

## 6. 下一步边界

目前不要继续无目的增加候选，也不要将真实未来 Oracle 接入线上。下一工作包应当：

```text
多对手、前中后期状态
→ 完整可行候选
→ 独立筛选未来库
→ 独立确认未来库
→ 胜负 + 分差 + 双方现金 + 硬失败标签
→ 留对手/留来源训练候选排序器
→ 通过后才启用在线选择和宏观 Self-play
```

真实未来 Oracle 是诊断上限，只存在于未进入 genome schema 的离线接口中，不能被提交模型意外开启。
