# 工作入口

先读 `README.md`、`HANDOFF_ZH.md`，以及本轮的分析与诊断记录 `docs/ECONOMIC_MODEL_ZH.md`。
本工程唯一主线是：高手 replay 离线提取/聚类 → 路线互打 → 147 维状态上的浅树切换 replay 路线 →
中期接管无开局模板的 JointAFS R1。

## 硬约束

- 不得把 `experiments/results` 中带原生 opening-template 的旧 Triad 扫描当作 R1 结论。
- 不使用 Nash 选开局。
- 正式公开对手验证只用 `opponents/` 的 7 个强对手，同 seed 双座、多 seed。
- replay 只离线使用。线上 `agent/main.py` 只读取导出的路线、浅树和当前公开观察。
- 先复用 `scripts/` 的成熟聚类、原生对打和鲁棒树训练，不另写简化替代品。

## 当前部署语义（2026-09-21）

- 开局固定 `G275`；浅树在 step 144 / 168 切换，最多一次。
- **接管延迟 2 天**，即固定到 step 288（实测 288 为单峰最优：264 差 52 胜、312 差 29 胜）。
- `target_fallbacks` = `{G114: G275, G019: G195}`。G019 是按叶归因找出的坏叶（条件胜率 69.7%）。
- G275 的 **cp144 节点已换成用真实对手重训的 depth-3 树**（896 个「状态 / 5 候选目标 / 结局」单元）；
  原树是在自对弈矩阵（对手 = 245 条库内路线）上训练的，标签域与线上不符。

改动明细与配对 A/B 证据见 `README.md`「当前部署语义」。

## 验收口径（容易踩的坑）

`experiments/run_strong_ab.py` 的 `both_seats` 几乎不提供独立信息：本对局对称且双方确定，
**约 87% 的 (对手, seed) 配对中 seat0 与 seat1 的 margin 逐元相同**。所有「N/896」的有效样本
约为名义值的 57%。历史上一批 8–16 seed 块的 accept/reject 决策都建立在这之上。

- 用 `experiments/eval_seed_paired.py`：按 (对手, seed) 配对 + Wilson 区间。
- **seed 段互斥**，不要重复使用同一段。
- 判定「每个对手 ≥80%」需要约 683–1537 seed（现在只有 512）。`engine fast` 跑 64 seed × 7 手 × 双座约 5 分钟。

## 性能研究的边界（已实测封死，别再走）

只评估真实 warm 链（replay 浅树切换 → delay=2 温接管 R1）。以下方向已逐一实测，收益为 0 或负：

| 方向 | 结果 |
|---|---|
| R1 配置杆（21+ 项，含叠加） | 均分差最高 +830，胜率 0（翻转一局需 +5,017） |
| 换 opening（7 条） | −66 ~ −367 |
| 编辑 tape（作物 / 预算 / 无耦合） | −454 ~ −566 |
| 叶回退（G024 / G275 / G316 方向） | −10 ~ −289 |
| 二次切换、更早切换点 cp72 | 胜率 0 / 明显更差 |
| 候选路线前瞻特征（48 步、120 步逐日） | +1.2pp / −3.9pp（后者更差） |
| 树重训扩样本（896 → 2,240 单元）、换模型类别 | 饱和 / 全部差于 depth-2 |
| 移植对手的 sale-lead 抢卖层 | +2 胜（≈0） |

**oracle（逐状态最优目标）比可达高 6–8pp，但用现有 147 维特征 + 任何模型类别都取不到 —— 瓶颈是特征信息量，不是数据量或模型容量。**

## Kaggle 提交

```bash
PYTHONPATH=. python scripts/pack_kaggle_submission.py \
  --so <x86_64 且 Jammy 兼容的 policy/r1/agent.so> \
  --output build/kaggriculture_submission.py
```

两个必须注意的点（否则线上直接得零分）：

1. **架构**：本地 `policy/r1/agent.so` 是 aarch64，Kaggle 是 x86_64。
2. **glibc**：用 Noble 的 `x86_64-linux-gnu-g++` 编出的产物要求 GLIBC 2.36/2.38，Jammy 只有 2.35。
   必须用 Jammy sysroot 工具链（`kaggriculture-t2-ideas-v1/submissions/toolchain-jammy`，GCC 11.2），
   加 `-static-libstdc++ -static-libgcc -march=x86-64 -ffp-contract=off -Wl,-Bsymbolic`，
   产物最高需求 GLIBC 2.34、无 libstdc++ 依赖。

改完编译器/宏后**必须做跨架构 parity**：同一批真实对局观测分别喂 aarch64（本地）与 x86_64（qemu + Jammy sysroot），
逐步比对 `td_observe` 输出。本会话已验证 300 步 0 mismatch。
