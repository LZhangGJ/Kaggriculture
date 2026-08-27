# 全池联合路线进化与 selector 实验（2026-08-27）

## 结论

当前冻结 agent 满足本轮验收标准：对 652 对手池总体原始胜率不低于 90%，每个对手
不低于 80%，并且对上一版最强动态 agent 不低于 80%。运行时从 183 条路线压缩为
7 条，未发生路线库数据爆炸。

| 验证面板 | seed | 双座位对局 | 原始胜率 | 最低单对手 | 完成率 |
|---|---:|---:|---:|---:|---:|
| 652 路线对手池 | 128 | 166,912 | 95.51% | 82.81% | 100% |
| 上一版最强 G001 动态树 | 512 | 1,024 | 81.84% | 81.84% | 100% |

对旧 agent 的两个座位胜率均为 81.84%；对手池双座位同时获胜率为 95.29%，配对
seed 原始胜率单侧 95% 下界为 94.66%。C++ 完成检查得到有限终局奖励，另抽查 8 条
完整轨迹，长度全部为 719。

## 最终 agent

- 固定开局：`NR295`。
- 公共决策点：48、96 回合；在线最多切换一次。
- 部署路线：`NR295`、`SP0027`、`SP0049`、`SP0053`、`SP0055`、`SP0059`、
  `D103_0003`。
- 新路线 `D103_0003`：严格保留 NR295 的前 96 回合，后缀取自新 replay 池筛出的
  `NR103`。
- selector：精确匹配 147 维 `semantic_route_switch_v1` 公共状态，未见状态安全回退
  NR295；不使用对手 ID、seed、未来市场或 oracle。
- 多文件入口：`runtime/main.py`；单文件入口：`main.py`。

运行时动作库压缩后为 63,100 bytes，selector NPZ 嵌入 JSON 后的原始模型为
14,341 bytes。相比把 1,530 条实验路线全部打包，部署只保留实际会被选择的 7 条。

## replay 与初代池

watcher 同时监控：

- `D:\Kaggriculture\top40`
- `D:\Kaggriculture\our_latest_two\55766855_Fixed_G001___searched_175-route_robust_switch_tree__official_local_paired_eval\replays`

最近快照发现 12,833 个文件、8,537 个唯一 episode、4,296 个重复副本；已写入 9,222
条基因记录和 6,811 个唯一基因，扫描错误为 0。watcher 每批增量处理 8 个 episode，
新 replay 无需重启进化池。

## 实验路线

1. 从实时 RouteGenome 池物化 1,530 条候选路线，使用 C++ 引擎筛选。
2. 以 NR295 为共同前缀，在 48/96 等检查点生成严格前缀相等的 full/market 后缀。
3. 先训练两阶段公共状态 selector；全池独立 holdout 达到 95.44% / 最低 82.81%。
4. 发现其对上一版动态 G001 仅 71.29%，将旧 agent 作为动态对手而非固定 G001 路线。
5. 1,530 条现成 replay 路线对旧 agent 的最佳固定胜率只有 70.31%，证明必须生成新
   路线而非继续手工筛 replay。
6. 第一版 GA 错误地奖励单路线胜率；修正为新增路线对状态 selector 的边际覆盖。
7. 64-seed GA 训练可到 79.69%，但独立 holdout 只有 66.60%，被判定为未来市场过拟合。
8. 扩大到 256/512 seed 并做分折验证；96/144/168 单点实验表明增加节点本身不能解决
   donor 缺失。
9. 将全量筛选中最强的 `NR103` 加入此前遗漏的 donor 集合。`NR295@96→NR103` 分支使
   当前 agent 对旧最强的第三段 512-seed holdout 达到 85.64%。
10. 最后以原 652 模型为基线做 653 对手联合约束，只允许状态改写在不破坏池最低胜率
    时保留，得到最终 95.51% / 82.81% / 81.84% 模型。

## 关键方法修正

旧适应度偏向“一个变异自己赢多少”，会淘汰单独较弱但能覆盖路线库失败状态的候选。
新适应度首先计算候选加入集合后，公共状态 selector 的交叉验证胜率增量，再使用座位、
seed 分折、单对手最低胜率和 margin 排序。多 farmer 路径变异只按完整同步 worker
阶段替换，且要求活动 hand 数一致，避免把单个 farmer 路径嫁接到不存在的工人上。

## 复现产物

本地实验根目录（Git 忽略的大矩阵与 replay 派生产物）：

`artifacts/cpp_route_experiments_20260827/causal_tree_v5/live_panel352_v2/`

关键报告：

- `shared_nr103_v1/nt652_holdout128_joint_v4_report.json`
- `shared_nr103_v1/current_strongest_holdout512_joint_v4_report.json`
- `shared_nr103_v1/nt653_d103_constrained_joint_v4_report.json`
- `shared_nr103_v1/submission_joint_v4/agent.py`

所有大规模对局均使用编译后的 C++ `NativeTeammateExecutor`，没有使用 JAX；本轮未调用
doraemon 服务器。
