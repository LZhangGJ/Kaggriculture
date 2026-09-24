# 中盘 BC 数据契约

`scripts/extract_midgame_bc.py` 把 `run_strong_ab.py` 保存的
`kaggriculture-bc-v1` 轨迹流式转换成 `kaggriculture-midgame-bc-v1`。它只处理
step 288 之后的数据，不改 agent，也不把动态脚本宣称为最优专家。

每条 `day` 记录包含：

- `state`：当天第一帧的完整线上 observation，即公共状态、`player` 和**本方**
  `private`；没有对手身份和对手 private。
- `decisions`：当天各小时的同口径 observation 及本方 action。
- `behavior_target.hourly_actions`：上述本方动作序列；它是 BC 行为标签，不是已恢复的
  R1 portfolio/bundle。
- `provenance.policy_source`、`policy_label` 和 `diagnostic`：训练时必须保留来源。
- `split_only`：`group_id / seed / opponent_group`，只用于分组切分和审计，禁止进入模型输入。
  `group_id=seed:opponent_group`，故同 seed 双座与不同策略臂天然绑定到同一 split。
- `replicate`：座位和 future seed，只标记同组内重复观测，不参与随机切分。
- `complete_day`：通常为 24 帧；终局 day29 只有 23 帧，因此为 false。

对手 private 默认完全丢弃。只有显式传入
`--include-opponent-private-target` 才会写入 `decisions[].auxiliary_target`，它只能用于
训练期 belief 辅助损失，线上输入和主策略特征都不得读取。

`--policy-source` 与 `--data-kind behavior|diagnostic` 是必填项，避免 forced-candidate
轨迹被无标记地混入专家数据。训练/验证必须按 `split_only.seed` 与
`split_only.opponent_group` 分组；同 seed 双座、baseline/candidate 和 future replicas
必须在同一 split。

```bash
/root/miniforge3/envs/torch-npu/bin/python scripts/extract_midgame_bc.py \
  work/.../*.jsonl.gz \
  --policy-source deployed-r1-<hash> \
  --data-kind behavior \
  --output data/bc/midgame.jsonl.gz

/root/miniforge3/envs/torch-npu/bin/python scripts/extract_midgame_bc.py --self-check
```

抽取器以一天为最大内存窗口，输出先写临时文件；任一输入损坏时不会发布半成品。

## 逐空格 student 契约

原始 `hourly_actions` 不能无歧义恢复 R1 当天选择的 portfolio：动作还混有移动、收获、
repair，且一天内新释放的地块在日初可能并非空地。因此它只能做行为 prior，**不得直接转成
逐格 expert label**。逐格训练数据必须由离线 teacher 在选中完整 bundle 时额外导出
`kaggriculture-autofill-v1`。

稳定类别顺序为：

`SKIP, WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, GOOSE, COW, SHEEP`。

teacher 先确定本轮可分配 slot，再沿用 R1 的
`(到四个 depot 的最短 Manhattan 距离, cell=y*10+x)` 顺序输出。每个 slot 必须带：

- 九类 `legal_mask` 和一个合法 `label`；
- 该步前的现金、种子/动物库存、土地/棚容量、饲料/肥料约束和剩余 slot；
- 从当天到 day29 的 `flow[30,9] / labor[30] / fixed[30]` 日历及有效日 mask；只保留未来三日
  会把短期相同、终局兑现能力不同的方案混成同一状态；
- 当前 slot 特征；此前 label 通过虚拟填充形成 `selected_portfolio` 和九类计数。

`SKIP` 只表示“不为该 slot 新增 successor”，不能表示挖掉当前资产。mask、剩余资源和 label
必须来自同一个 C++ feasibility/teacher 状态，不能由成交前标价静态相减。完整 bundle 最终仍交给
现有 executor/repair。

`experiments/midgame_autofill_student.py` 目前只提供 49,497 参数的 decoder 结构自检。其中
`192/32` 是待实现 encoder 的 latent 宽度，11 个 `remaining_resources` 名称也是占位，不是完整
raw feature schema；正式 teacher 导出完整状态、日历、mask 和 label 前禁止开训。逐字段缺口见
`docs/MIDGAME_STUDENT_FEATURE_AUDIT_ZH.md`。

## 训练存储格式

`jsonl.gz` 只作可追溯原始档案，不能在每个 epoch 重新解析。最终 feature schema 固定后只转换一次：

- C++ rollout/teacher 直接写定长数值数组；历史 replay 则按文件并行流式解析一次；
- 数值按用途保存为 `float16/float32`、`uint8/uint16` 和 `int32/int64` 的分片平面数组，变长维度使用
  offset 表；Python 通过 `numpy.memmap` / `torch.from_numpy` 读取；
- manifest 必须保存 schema 版本与 hash、dtype/shape、样本和 group offset、来源文件 hash、策略来源和
  split 字段；训练按 shard shuffle，绝不从文件名或对手身份生成输入；
- 不用压缩 NPZ 作为热训练路径，因为它仍需整块解压。JSON 下载保持单线程；下载完成后的独立文件解析
  才可使用多进程吃满 CPU。

原始 JSON 在二进制 shard 通过计数、hash 和抽样 parity 前不得删除；即使验证通过，也只有用户明确同意
后才清理。Python 只负责 DataLoader 和 PyTorch 优化，C++ 负责仿真、精确特征/标签和线上推理。
