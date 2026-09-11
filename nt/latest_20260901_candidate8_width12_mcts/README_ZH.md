# Candidate8 Width=12 / MCTS 最新搜索套件

日期：2026-09-01

这是 2026-08-29 Candidate8 交接包之后的增量完整快照。为了避免入口继续埋在旧目录里，本包使用独立顶层目录，包含当前可编译的 C++ 搜索内核、Candidate8 工具、必要路线资产以及验收材料。

本包不包含 Hasegawa 511 路线分析、交叉回放或集合覆盖产物。

## 1. 从哪里开始

- C++ 规则和搜索内核：`workspace/agents/route_clustering_switch_agent/fast_kaggriculture/`
- Candidate8 实验工具：`workspace/experiments/ecobot_adaptive_planner_v1/tools/`
- 一键复现实例：`scripts/run_width12_mcts.sh`
- Width 2/4/8/12 结论：`workspace/experiments/ecobot_adaptive_planner_v1/reports/CANDIDATE8_BEAM_WIDTH_2_4_8_12_SWEEP_20260901_ZH.md`
- MCTS 对照结论：`workspace/experiments/ecobot_adaptive_planner_v1/reports/CANDIDATE8_MCTS_VS_BEAM_EQUAL_BUDGET_ACCEPTANCE_20260901_ZH.md`
- 文件校验表：`MANIFEST_SHA256.tsv`

## 2. 相比 2026-08-29 包新增什么

1. Candidate8 八阶段完整路线搜索，决策日默认为 `0,1,3,6,9,12,18,24`。
2. Beam width 支持至少 `2/4/8/12`，并完成同场景、同候选池的正式对照。
3. C++ 原生 Candidate8 MCTS：UCB、渐进展开、启发式 rollout 和16棵独立树并行。
4. 已选候选序列的确定性重放与终局一致性检查。
5. R8 的24步任务级计划、4至8步多人联合调度及滚动更新代码。
6. O1系列候选排序、后果特征、反事实数据和失败归因工具源码。
7. Top60通用拳法搜索工具；没有携带任何 Hasegawa 专用分析。

## 3. 已验证结论

### Beam width

在 `G001/G004/G013 × 2 seeds × 2 seats` 的12个场景上：

- `width=12`：12/12不败，四档联合最佳命中10/12；
- `width=2 + width=12`：覆盖12/12场联合最佳；
- 单次快速搜索推荐 `width=8`；
- 离线追求路线质量推荐分别跑 `width=2` 和 `width=12` 后按真实终局取优。

### MCTS

- 小预算下，MCTS更容易发现Beam误剪掉的不同胜路；
- 中等预算下，Beam平均现金和分差仍明显更高；
- 当前裁定：Beam是主搜索器，MCTS是多样性和失败反例补充，不应直接替换Beam。

上述比较是可查看真实终局的离线 Oracle 搜索，不等于线上一秒内可以部署的决策器。

## 4. WSL/Linux构建

要求 Python 3.12、C++20、OpenMP、`pybind11`、`numpy`。

```bash
cd nt/latest_20260901_candidate8_width12_mcts/workspace/agents/route_clustering_switch_agent/fast_kaggriculture
python -m pip install pybind11 numpy
python setup.py build_ext --inplace
PYTHONPATH=python python -m pytest -q tests
```

## 5. 一键运行

从仓库根目录执行：

```bash
bash nt/latest_20260901_candidate8_width12_mcts/scripts/run_width12_mcts.sh width
bash nt/latest_20260901_candidate8_width12_mcts/scripts/run_width12_mcts.sh mcts
```

- `width`：运行 `2/4/8/12` Beam 对照，不运行MCTS；
- `mcts`：运行等续跑预算的 Beam/MCTS 对照；
- 输出写到本包的 `outputs/`，该目录不进入Git。

线程数可通过环境变量覆盖：

```bash
OMP_NUM_THREADS=16 MCTS_WORKERS=16 bash nt/latest_20260901_candidate8_width12_mcts/scripts/run_width12_mcts.sh mcts
```

## 6. 关键接口

Python绑定位于 `fast_kaggriculture/src/bindings.cpp`：

- `candidate8_sequence_oracle(...)`：固定宽度的多阶段Beam；
- `candidate8_mcts_oracle(...)`：固定续跑预算的MCTS；
- `candidate8_committed_sequence(...)`：重新播放选中的阶段候选，检查是否与搜索结果一致。

搜索入口位于：

- `tools/compare_candidate8_mcts_beam.py`
- `tools/export_candidate8_search_comparison_traces.py`
- `tools/audit_candidate8_sequence_pool.py`
- `tools/audit_candidate8_sequence_oracle.py`

## 7. 边界与省略项

- 官方 `kaggle-environments==1.32.7` 仍是最终裁判。
- Git包含运行 width/MCTS 所需的小型路线资产和验收回执。
- 大型训练数据、模型、海量逐局输出和Replay没有复制进本包。
- 搜索结果只证明当前候选语言和冻结对手池中的事后上限，不能视为线上策略强度。

