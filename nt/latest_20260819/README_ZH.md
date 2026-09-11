# Kaggriculture NT 最新开发快照（2026-08-19）

这个目录是在旧版 `nt/gpu_sim`、EBA26v2 和 GPT 路线包之后追加的最新开发快照。旧目录作为历史证据保留不动；本快照使用官方 `kaggle-environments==1.32.7` 口径。

## 包含内容

- `gpu_sim/`：当前 JAX 规则内核、静态表、官方 1.32.7 冻结源码、测试和工具。
- `experiments/project_route_search_v2/`：M3.9 三层经营控制器源码、测试、工具、配置和关键验收凭证。
- `experiments/strategic_v5/`：当前策略与五个高潜力公开方案的 JAX 控制器。
- `experiments/expert_business_agent_v2/`：Exact5 路线库、动态运行表、验收工具和严格镜像一致性凭证。
- `references/public_high_potential_20260819/`：Boatlee V20、Ray K320、Tetsutani、Kaito V27 和 Flex V59 的冻结公开源码。
- `docs/`：五个公开方案能力如何抽象到通用规划器的设计复审。该文档是 `DESIGN_ONLY / NOT_RUN`，不能当作实验结果。

为了控制仓库体积，本快照不包含虚拟环境、Python 缓存、海量官方对局、M3.x 中间调试轨迹和 Exact5 的 20 条压缩逐帧轨迹。关键结果由 JSON/XML/Markdown 收据保存；若要重新执行严格逐帧验收，需要在主工作区补回对应官方轨迹。

## 当前真实结论

### M3.9 规划器

- 路线卡工程能力通过：支持作物闭环、动物闭环、跨域混合路线、资源预留、失效取消与重新规划。
- 11/11 定向测试通过；当前轨迹与官方 Python 1.32.7 的 720 帧状态一致。
- RTX 3090、batch 2048 的完整控制器约为 87k–89k transitions/s。
- 但高收益表达能力没有通过：V20 官方 Replay 为 155,450，原始动作送入 JAX 仍为 155,450，而当前日历和路线卡只能重新规划到 106,333（68.4%）。
- 因此当前版本不能宣称已经覆盖 15.5 万或 17 万路线。下一步应补完整生产义务日历、多工人并行作物项目、空间/产能评估和基于 rollout 的项目比较，而不是继续给动物追加特调。

详见 `experiments/project_route_search_v2/reports/M3.9_CROP_MIXED_ROUTE_CARD_ACCEPTANCE_REPORT_V1_ZH.md`。

### Exact5 GPU 冻结对手

- 五个方案在 5 × 2 seeds × 2 seats 的 20 个镜像上下文中完成 14,380 个决策步验收。
- 单位动作、市场订单、逐帧状态和终局奖励均为 100% 一致。
- RTX 3090、batch 2048 稳态吞吐：Boatlee 184,283、Ray 184,726、Tetsutani 191,780、Kaito 258,362、Flex 251,514 transitions/s。
- 该证据只证明选定镜像上下文的一致性，不是对所有可能状态的穷举证明。

详见 `experiments/expert_business_agent_v2/reports/HIGH_POTENTIAL_EXACT5_JAX_COMPLETION_20260819_ZH.md`。

## 环境和快速检查

推荐在 WSL2 的 CUDA JAX 环境中运行。Windows 侧只用于源码检查和官方 Python 裁判。

```powershell
python -m compileall -q nt/latest_20260819/gpu_sim/src nt/latest_20260819/experiments
```

在 `nt/latest_20260819/experiments/project_route_search_v2` 下运行规划器测试：

```powershell
python -m pytest tests/test_m38_route_cards.py -q
```

在具备 CUDA JAX 的环境中运行 Exact5 吞吐探针：

```powershell
python experiments/expert_business_agent_v2/tools/benchmark_high_potential_exact5_gpu.py `
  --bank experiments/expert_business_agent_v2/artifacts/high_potential_route_bank_v1.npz `
  --runtime experiments/expert_business_agent_v2/artifacts/high_potential_runtime_tables_v1.npz `
  --batch 2048 `
  --output experiments/expert_business_agent_v2/reports/exact5_gpu_benchmark_local.json
```

## 架构边界

- `gpu_sim` 是规则层，除官方版本迁移外不应被策略改写。
- Exact5 是冻结对手和行为 oracle，不允许正式通用规划器按作者名、Replay ID 或来源路由。
- 当前 M3.x 有成熟的执行基础设施，但项目选择与经济规划仍处于半特调状态。
- 下一版应收敛为单一热路径：`RouteGenome/State → ProjectProposal → SpatialCapacityPlan → TaskGraph → UnifiedScheduler → TransactionCompiler → AtomicExecutor`，偏差发生后只做局部项目编辑与重规划。
