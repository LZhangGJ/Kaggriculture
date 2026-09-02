# 对手前缀信念与 Counterbank 数据契约 v0

## 1. 目的

本文固定 `BAYESIAN-PREFIX-COUNTERBANK-v0` 的离线文件和线上接口，使 replay 提取、
聚类、C++ 对局、贝叶斯模型和 selector 可以独立开发并通过哈希连接。v0 只定义契约，
不预先固定最终特征维度或类别数。

所有 schema 必须带版本。发现字段语义变化时升级 schema，不允许在同名 schema 下静默
改变归一化、checkpoint 或特征顺序。

## 2. ID 和因果边界

### 2.1 可用于离线分组、不可进入线上特征

- `team_id` / `team_name`；
- `submission_id`；
- `episode_id`；
- `opponent_route_id`；
- `seed`；
- 最终奖励、胜负和未来动作。

这些字段只用于去重、切分、溯源和评测。训练导出器必须显式证明它们没有进入向量。

### 2.2 可用于线上 belief

- 当前及历史的对手公开棋盘和公开资源；
- 已经发生的公开动作或可从状态差分推断的事件；
- 当前及历史公开市场、商店和解锁状态；
- 我方已经发生的公开开局行为；
- 当前时间和座位。

### 2.3 只用于我方候选可执行性

- 我方私有库存、未来资本需求和 farmer/hand 状态；
- 当前路线、已切换标记和 continuation contract。

这些字段进入候选过滤和 switch cost，不进入对手身份似然。

## 3. 前缀索引

文件：`prefix_index.jsonl.gz`

每行对应一个 episode、座位和 checkpoint：

```json
{
  "schema": "opponent-prefix-index-v0",
  "sample_id": "sha256:...",
  "episode_id": "offline-only",
  "submission_id": "offline-only",
  "team_name": "offline-only",
  "opponent_route_id": "RG2_...",
  "split": "train",
  "seat": 0,
  "seed": 1234,
  "checkpoint_step": 96,
  "checkpoint_day": 4,
  "feature_row": 17,
  "context_row": 17,
  "source_path_hash": "sha256:...",
  "complete": true
}
```

`sample_id` 由 episode、观察座位、checkpoint 和 schema 生成。原始绝对路径不进入可发布
产物，只保存路径哈希和数据快照标识。

## 4. 前缀向量

文件：`prefix_features.npz`

必需数组：

| 数组 | 形状 | 含义 |
|---|---|---|
| `identity_features` | `[N, D_identity]` | 仅对手公开前缀 |
| `identity_mask` | `[N, D_identity]` | 1 表示该维可观测 |
| `context_features` | `[N, D_context]` | 市场、商店、我方公开开局 |
| `context_mask` | `[N, D_context]` | 上下文可观测 mask |
| `checkpoint_steps` | `[N]` | checkpoint step |
| `seats` | `[N]` | 观察座位 |

必须同时保存 `feature_schema.json`，其中按顺序列出每个维度：

```json
{
  "schema": "opponent-prefix-feature-schema-v0",
  "checkpoints": [48, 96, 144, 192, 288],
  "identity_features": [
    {"name": "opp.crop.wheat.count", "kind": "snapshot", "scale": "robust"},
    {"name": "opp.event.buy_land.delta_24", "kind": "delta", "scale": "none"}
  ],
  "context_features": [
    {"name": "market.wheat.price", "kind": "snapshot", "scale": "robust"}
  ],
  "forbidden_fields": [
    "team_id", "submission_id", "opponent_route_id", "seed", "final_reward"
  ]
}
```

缩放参数只允许从 train 拟合，并保存在同一 schema。dev/holdout 只能读取，不得重新拟合。

## 5. 推荐特征组

### 5.1 Identity

- 对手现金或其他公开经济量的快照与日增量；
- 土地、farmer、建筑、动物、作物数量；
- 布局分区计数、占用率和离散类别；
- 建造、买地、雇工、放置动物、播种、收获和出售事件；
- 事件首次出现日、最近出现距今天数、累计次数；
- 最近 24/48/96 回合的活动计数；
- 对手未来资本需求的公开可推断代理量。

### 5.2 Context

- 市场库存和价格；
- 商店解锁和公开城镇状态；
- 我方已发生的公开动作摘要；
- 当前 day/hour、座位；
- 当前共同开局 ID 仅作为我方策略上下文，不作为对手标签。

### 5.3 不进入 v0

- 原始 team/submission/opponent ID embedding；
- 最终奖励或未来窗口统计；
- 文件路径、读取顺序和 replay 更新时间；
- 只有 holdout 才出现的全局归一化统计。

## 6. Payoff 矩阵

文件：`payoff_matrix.npz`

必需数组：

| 数组 | 形状 | 含义 |
|---|---|---|
| `wins` | `[A, R, S, 2]` | 候选、对手、seed、座位的原始胜负 |
| `scores` | `[A, R, S, 2]` | 得分率或离散结果 |
| `margins` | `[A, R, S, 2]` | 最终奖励差 |
| `completed` | `[A, R, S, 2]` | 是否正常完成 |
| `candidate_ids` | `[A]` | 我方候选路线 ID |
| `route_ids` | `[R]` | 对手路线 ID |
| `seeds` | `[S]` | 面板 seed |

配套 `payoff_index.json` 保存：

- 引擎和入口 SHA-256；
- 719 回合完成定义；
- 胜、平、负和 score 的计算方式；
- seed 集和切分；
- 每条路线动作带 SHA-256；
- 缺失或执行失败单元格。

执行失败不得按普通输局静默填入 `wins=0`。先设 `completed=false`，评测报告单独处理。

## 7. 反制类别映射

文件：`response_class_map.json`

```json
{
  "schema": "response-class-map-v0",
  "payoff_matrix_sha256": "...",
  "fit_split": "train",
  "classes": [
    {
      "class_id": "RC000",
      "member_route_ids": ["RG2_..."],
      "medoid_route_id": "RG2_...",
      "member_count": 12,
      "within_class_regret": 0.03,
      "observable_at_steps": [144, 192]
    }
  ],
  "route_to_class": {"RG2_...": "RC000"}
}
```

`observable_at_steps` 是诊断信息，不保证线上一定正确分类。类别建立只能使用 train payoff；
dev 只用于选择超参数，holdout 不得参与重聚类。

## 8. Belief 模型

文件：`belief_model.npz` 和 `belief_model.json`

JSON 至少包含：

```json
{
  "schema": "bayesian-opponent-belief-v0",
  "feature_schema_sha256": "...",
  "response_class_map_sha256": "...",
  "model_type": "diagonal-student-t",
  "class_ids": ["RC000", "RC001", "UNKNOWN"],
  "checkpoint_steps": [48, 96, 144, 192, 288],
  "prior_smoothing": 1.0,
  "temperature": 1.0,
  "ood_rule": {
    "metric": "max_class_log_likelihood",
    "threshold_source": "dev-known-vs-unknown"
  }
}
```

NPZ 保存每个 checkpoint 和 class 的似然参数、先验、缩放参数及 OOD 参数。所有概率在
log 空间归一化。运行时必须对非有限值回退到前一日后验或全局先验。

## 9. Counterbank

文件：`counterbank.json`

```json
{
  "schema": "opponent-counterbank-v0",
  "candidate_library_sha256": "...",
  "payoff_matrix_sha256": "...",
  "global_fallback_route_id": "NR295",
  "shared_candidate_ids": ["NR295", "D103_0003"],
  "classes": {
    "RC000": {
      "candidate_ids": ["NR295", "D103_0003"],
      "utility_lcb": [0.82, 0.86]
    }
  },
  "unknown": {
    "candidate_ids": ["NR295"]
  }
}
```

同一候选只在 `shared_candidate_ids` 物化一次。类别中只保存引用和校准效用，避免动作带
重复。v0 每类最多三个候选，但总共享候选预算由 run manifest 决定。

## 10. 线上接口

计划模块：`opponent_belief.py`

```python
class OpponentBelief:
    def reset(self, *, seat: int, opening_id: str) -> None: ...

    def update(self, observation: object) -> "BeliefSnapshot": ...
```

`BeliefSnapshot`：

```python
@dataclass(frozen=True)
class BeliefSnapshot:
    checkpoint_step: int
    class_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    entropy: float
    max_probability: float
    ood_score: float
    is_unknown: bool
```

计划模块：`counterbank_policy.py`

```python
class CounterbankPolicy:
    def choose(
        self,
        *,
        observation: object,
        belief: BeliefSnapshot,
        current_route_id: str,
        switched: bool,
    ) -> "CounterbankDecision": ...
```

`CounterbankDecision` 至少返回：

- `route_id`；
- `decision`：`KEEP` 或 `SWITCH`；
- `expected_lcb`；
- `keep_lcb`；
- `switch_cost`；
- `continuation_compatible`；
- 仅用于离线审计的 top posterior classes。

线上日志不得写入未来标签、最终奖励或真实 `opponent_route_id`。

## 11. 运行清单

文件：`RUN_MANIFEST.json`

必须在封闭测试前写入并只读：

- Git commit 和 dirty status；
- 数据快照、引擎、agent 和候选库哈希；
- train/dev/holdout 的 seed、团队、submission 和时间边界；
- checkpoint、模型类型和超参数搜索空间；
- Go/No-Go 门槛；
- 每类路线数和总共享路线预算；
- CPU、服务器和并行限制；
- 预期输出文件列表。

若运行中修改任何预注册项，生成新的 `run-id`，不得覆盖旧报告。

## 12. 校验要求

每次正式运行至少执行：

1. schema 和数组形状校验；
2. sample ID、episode 和 SHA-256 去重；
3. forbidden field 泄漏测试；
4. train-only scaler 测试；
5. split 隔离测试；
6. payoff 单元格完成率检查；
7. posterior 和为 1、非有限值回退测试；
8. UNKNOWN 回退测试；
9. counterbank 引用和动作带哈希检查；
10. 完整 agent 719 回合和双座位执行测试。

## 13. 与现有格式的关系

- 147 维 `semantic_route_switch_v1` 继续作为我方状态和 continuation 可行性的基线；
  对手 prefix schema 单独版本化，不直接覆盖它。
- RouteGenome v2 继续作为 replay 路线身份和宏观意图来源。
- 54 维动作意图、14 维事件、100 维布局和 unlock mask 可复用于候选/block 表示，
  但不得被误当成已经训练好的路线 embedding。
- 当前 `fingerprint_distance` 可以作为 kNN likelihood 的一个基线距离；正式模型需要
  将身份特征和环境上下文拆开，并改为序列后验。
