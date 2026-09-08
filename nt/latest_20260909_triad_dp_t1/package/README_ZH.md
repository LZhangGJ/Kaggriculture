# Kaggriculture · Triad-DP T1

默认入口为 `policy/agent.py::agent`，默认参数为 `policy/config.json`。源码为 C++20，Python 只负责 observation 编码及调用。先看 `REPORT_ZH.md` 的独立 100-seed 验收结果；它比开发面板、历史材料和模型训练指标优先。

## 本地运行

使用 Linux x86-64 / WSL2。需要 Python、g++、Python 开发头文件以及 pip 依赖。本轮实测环境记录在 `ENVIRONMENT.json`；附带的二进制是在该环境编译的，跨 Python/系统版本应重建。

```bash
python3 -m pip install -r requirements.txt
python3 build_all.py --jobs 2
python3 run.py --tag local_smoke --count 4 --threads 4
```

`--count 4` 是 4 seed × 7 对手 × 双座位，共 56 局。输出目录必须尚不存在，不会覆盖旧证据。只测试一个对手可以使用 `--opponents g001`；对手始终每步读取实际比赛状态，不是对手 action tape。

复跑独立验收面板：

```bash
python3 run.py --tag local_reproduction --count 100 --start 260909000 --threads 4
python3 run.py --tag baseline_reproduction --baseline j7 --count 100 --start 260909000 --threads 4
```

这两条命令现在是重现既有测试，不应再次被称作“全新未见测试”。继续修改代码后，应另选新 seed 做新的独立验收。

仅编译我方策略，不加载对手资料：

```bash
python3 build_all.py --policy-only
```

可直接在自己的官方 runner 中加载 `policy/agent.py`，或实例化 `Agent(config=...)`。每个 Agent 实例只用于一个座位；模块级 `agent` 自动区分两个座位。异常会明确抛出，不会偷偷恢复 J7、播放历史动作或将错误隐藏成 PASS。

## 验证

```bash
python3 tests/run_tests.py
python3 run.py --tag fresh_trace --count 1 --trace
python3 tests/official_parity.py fresh_trace --out tests/results/fresh_trace --limit 14
```

最后一步用随包冻结的官方 Python 规则逐步复放双方新动作，并在每个 observation 上重新调用导出的我方 Agent。它检查规则状态一致与我方动作一致，不等于重新证明所有对手移植在所有状态下一致。

## 文件导航

| 路径 | 内容 |
|---|---|
| `REPORT_ZH.md` / `ACCEPTANCE.json` | 独立验收结果、目标是否达到、配对置信区间 |
| `policy/` | 新规划器、状态反馈执行器、条件生产日历、Python 导出入口 |
| `DESIGN_AND_SCOPE_ZH.md` | 已实现机制、来源和未完成边界 |
| `RELEASE_FREEZE.json` | 独立测试之前冻结的代码、二进制和参数 |
| `runs/release_holdout100/` | 最终 1,400 局逐局记录及运行协议 |
| `runs/baseline_j7_holdout100/` | 同 seed、同座位、同对手的冻结 C3＋J7 对照 |
| `holdout_games.csv` | 便于外部分析的最终逐局表 |
| `tests/results/parity_release/` | 最终版本官方逐步对齐记录 |
| `labels/` / `models/` | 反事实估值实验的数据与实际训练模型；不是默认线上选择器 |
| `arena/` / `agent/` | 原七对手底座及旧 C3 来源；原始文档仅作历史材料 |
| `snapshots/` | 开发过程中按 SHA256 保留的策略版本及源码快照 |

交付目录已重新解压、无缓存从源码编译，并对新版和旧基线各 14 局核对动作哈希与现金，结果一致；证据见 `tests/clean_package_rebuild.json`。

## 修改与复现实验

修改 `policy/` 后运行 `build_policy.py`，再通过 `run.py` 指定新配置文件。构建前后检查源码哈希；编译期间源文件发生变化则拒绝输出。编译产物及每次对战均保存源码和二进制指纹。

`rotation` 控制未来轮作机会价值项，不等于是否允许实际换种；`tour_dp` 是可选的单路线精确顺序 DP。当前默认没有启用这两项，也没有启用重复种植预测项 `repeat`，因为较复杂的估值/最短路线并未在相同开发面板中提高总体胜率。`layout` 是保留字段，当前没有作用。`portfolio_passes` 当前表示候选方向数量，历史名称保留以兼容日志，不是组合迭代次数。

默认 `scenario=1` 表示公共情景前瞻一天。`scenario=0` 是无外层前瞻的原生对照；`scenario=-1` 是监督估值实验。训练模型的分支没有胜过默认版本，不要仅因为看到权重文件就认为线上使用了模型。

需要重现估值实验时另外安装 `requirements-training.txt`，参考 `TRAINING_BOUNDARY_ZH.md`、`collect_labels.py`、`train_value.py`、`train_ranker.py`。训练会覆盖实验估值头，必须重新构建及独立评估；不要把训练阶段的事后最佳或节点指标报为完整对战胜率。

本包没有验证 Kaggle 提交沙箱、线上天梯分数或当前最新版对手。所有强度陈述限定在报告明确列出的冻结对手、规则和测试种子上。
