# 构建、训练与评测

以下命令均从工程根目录运行：

```bash
cd /root/kaggriculture-replay-switch-clean-v1
export PYTHONPATH=$PWD
export PY=/root/miniforge3/envs/torch-npu/bin/python
```

## 快速检查与构建

```bash
$PY verify_project.py
$PY experiments/smoke_official.py

# 重建无开局模板 R1 接管库；单一重模板翻译单元，通常不能有效吃满 192 核
policy/r1/build.sh

# 重建高速仿真器；独立 C++ 源文件可并行
(cd fast_kaggriculture && $PY setup.py build_ext --inplace -j 192)
```

## 复用既有 rollout 训练树

浅树训练不需要重跑 2446 万局：

```bash
$PY scripts/train_robust_search_route_trees.py \
  --search data/artifacts/switch-fine-26x128.npz \
  --output work/trees-shallow.json \
  --depths 2,3,4,5,6 \
  --min-leaves 16,32,64 \
  --node-workers 192
```

现有 `trees-shallow-d2-6.json` 使用 seed 分组、对手家族分组、阈值扰动和单侧 95% 下界。不要仅按训练准确率选择深树。

## 重跑高速路线互打

下面是小范围示例；扩大 seed 区间会直接使用原生核：

```bash
$PY scripts/run_native_intent_round_robin.py \
  --source agent/teammate_base.py \
  --actions agent/route_actions.json.zlib \
  --metadata agent/route_library.json \
  --seeds 2609500000:2609500004 \
  --output work/round-robin.npz \
  --summary work/round-robin-summary.json

$PY scripts/run_native_intent_switch_search.py \
  --source agent/teammate_base.py \
  --actions agent/route_actions.json.zlib \
  --metadata agent/route_library.json \
  --openings G001,G210,G275,G379,G411 \
  --targets G001,G210,G275,G379,G411,G009 \
  --checkpoints 144,168,216 \
  --seeds 2609500000:2609500004 \
  --output work/switch.npz \
  --summary work/switch-summary.json
```

完整既有规模分别是 64 和 128 seeds。大规模仿真已经完成，只有路线库、检查点、特征或模拟器语义发生变化时才应重跑。

## 官方环境强对手 A/B

```bash
$PY experiments/run_strong_ab.py \
  --baseline policy/r1/agent.py \
  --candidate agent/main.py \
  --output work/r1-vs-hybrid-16seed.json \
  --seeds 16 --start 2609600000 --workers 192
```

每个策略运行 `7 × seeds × 2` 局，总任务数再乘两个 A/B 分支。工具按游戏分进程，192 workers 用于游戏级并行。正式汇报必须给总胜局、平均分差、逐 bot 结果、错误数、seed 范围和双座信息。

## 实验开关

```bash
# 固定另一个已有 opening；只有树中有该 opening 才能称为完整路线树实验
REPLAY_FORCED_OPENING=G210 $PY experiments/smoke_official.py

# 修改接管步；例如 day13
REPLAY_HANDOFF_STEP=312 $PY experiments/smoke_official.py

# 禁用 replay 部署，直接运行纯 R1
REPLAY_DISABLE_DEPLOYMENT=1 $PY experiments/smoke_official.py

# 仅离线实验：单地块 / 双地块联合作物动物局部搜索；默认均关闭
R1_CONFIG_OVERRIDES='{"portfolio_swaps":1}' $PY experiments/smoke_official.py
R1_CONFIG_OVERRIDES='{"portfolio_swaps":2}' $PY experiments/smoke_official.py
```

`REPLAY_HANDOFF_STEP` 是 step，不是 day。day `d` 的起点为 `24*d`。旧结果中 day0 使用了带 opening-template 的另一个 Triad 分支，不能作为纯 R1。

局部搜索的配对 A/B 使用同一个入口、同 seed、双座；例如：

```bash
$PY experiments/run_strong_ab.py \
  --baseline agent/main.py --candidate agent/main.py \
  --baseline-r1-config '{"portfolio_swaps":0}' \
  --candidate-r1-config '{"portfolio_swaps":2}' \
  --start 2610400100 --seeds 8 --workers 192 \
  --output work/warm-portfolio-pairs2-public7-8seed-2610400100.json
```

该 pilot 已为负，不应扩大到 64 seeds；命令用于复核代码路径，不代表推荐配置。
