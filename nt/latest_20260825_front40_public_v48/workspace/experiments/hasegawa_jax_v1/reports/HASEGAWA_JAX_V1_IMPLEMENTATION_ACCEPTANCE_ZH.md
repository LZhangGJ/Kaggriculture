# Hasegawa JAX V1 实现与验收报告

日期：2026-08-20  
规则版本：官方 1.32.7  
状态：**工程可运行 / 行为复现 V1 / 尚未宣称精确复刻**

## 1. 实现边界

本版本依据 Ryo Hasegawa 当前提交 55614463 的 111 局 Public Replay：

- 102 局胜局编译为高层经营日历；
- 9 局败局不进入教师路线库；
- 不保存、不播放 Replay 原始单位动作；
- 不保存、不播放 Replay 市场动作；
- 不保存 Replay 地块坐标；
- 路由时不读取未来商店或未来随机事件。

Kaggle discussion 736219 的实际内容是天梯评分、提交频率和离线评测建议，
不是 Hasegawa Agent 的公开源码。因此当前实现只能称为基于 Replay 证据的
行为复现，不能称为源代码精确移植。

## 2. Agent 结构

```text
当前 JAX State
  ├─ 已解锁商店需求
  ├─ 当前市场价格
  ├─ 当前现金
  ├─ 自家作物/动物规模
  ├─ 当前雇工数
  └─ 已解锁土地
          ↓
每天一次 GPU 最近路线选择（102 条胜局日历）
          ↓
只取所选路线“当天”的经营义务
          ↓
通用 Fulfillment / Task Scheduler
          ↓
原子 Action → JAX 官方语义模拟器
```

首步经营事务按 111 局一致证据固定为：

1. BUILD_PASTURE；
2. HIRE × 5；
3. BUY_SHEEP 2；
4. BUY_COW 2；
5. BUY_MELON_SEED 11；
6. BUY_WHEAT_SEED 6；
7. BUY_PRODUCT_WHEAT 4。

首个商店出现后，YARN_STORE 与 FARMERS_MARKET 会选择不同路线；后续每天
可根据新增商店、价格和实际经营偏差重新选择路线。切换路线时只接管当天
尚未执行的义务，不补做新路线过去的历史动作。

## 3. 验收结果

### 3.1 单元与首步事务

- pytest：3/3 PASS；
- 路线库：102/102 胜局完整编译；
- 首步 Action 顺序：PASS；
- 首步结算：现金 142、雇工 5、小麦入仓 4、麦种 6、瓜种 11；
- Yarn / Farmers Market 分叉：PASS。

### 3.2 GPU 完整局 smoke

batch=8、双座位、被动对手：

| 指标 | 座位0 | 座位1 |
|---|---:|---:|
| 完整719步 | PASS | PASS |
| 硬错误 | 0 | 0 |
| 平均现金 | 100,394 | 93,949.6 |
| 吞吐 | 413.4 trans/s | 427.0 trans/s |

batch=256、座位0、被动对手：

| 指标 | 结果 |
|---|---:|
| 完整719步 | PASS |
| 硬错误 | 0 |
| 平均现金 | 85,830.8 |
| 胜率 | 100% |
| 稳态吞吐 | 12,854.9 trans/s |
| 最终使用的不同路线 | 87 |

## 4. 结论与未通过项

已经完成：

- 可 JIT、可 GPU、可双座位运行的 Hasegawa 行为复现 Agent；
- 102 条胜局高层路线库；
- 状态驱动的商店/价格分叉；
- 719 步无硬错误；
- 与本地 Exact5 共用 Action 接口的 Arena 入口。

尚未完成：

- 未证明逐步动作与 Hasegawa 原 Agent 一致；
- 未达到 Exact5 专用 Agent 的 18.4万～25.8万 trans/s；
- 当前约 1.29万 trans/s 的主要瓶颈是通用任务池和调度器；
- 尚未运行对本地五个 Agent 的正式双座位对战；
- batch=256 被动场均 85,831，说明行为覆盖可用，但仍不是第一名策略的完整兑现。

因此当前正确标签是：

> **Hasegawa JAX V1 可执行行为复现基线，可进入 Exact5 Arena；不可宣传为精确复刻。**

## 5. 可复现实物

- `src/hasegawa_jax_v1/agent.py`：JAX Agent 与每日动态路由；
- `src/hasegawa_jax_v1/plan_bank.py`：路线库结构和加载；
- `artifacts/hasegawa_plan_bank_v1.npz`：102 条高层胜局路线；
- `receipts/hasegawa_plan_bank_v1.json`：来源与哈希；
- `receipts/passive_gpu_smoke_v1.json`：双座位小批量 GPU 验收；
- `receipts/passive_gpu_b256_v1.json`：batch=256 GPU 基准；
- `tools/run_hasegawa_jax_arena.py`：passive / Exact5 双座位 Arena。

下一步正式实验应使用 `--opponents all5`，同一批 seeds、双座位比较每个
对手的胜率和现金差；不能只看合并平均值。
