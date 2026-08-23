# FC24B：34-Agent 90%门、官方/JAX一致性与完整开发链快照

冻结日期：2026-08-23  
官方规则：`kaggle-environments==1.32.7`  
最终本地候选：`FC24B`  
Kaggle提交：`55708153`（冻结本快照时仍为 `PENDING`）

## 1. 先看结论

FC24B 在冻结协议下完成了以下验收：

- 2026-08-22 最新6个公开强Agent全部完成源码冻结、官方1.32.7轨迹验收和精确JAX迁移；
- 最新6个与旧28个本地Agent均执行128个事件种子、双方换座，即每个对手256局；
- 34个对手、8,704局，FC24B对每个对手的得分率均至少90%；
- 最新6个最低为 Soil V26-H：235/256，91.80%；
- 旧28个最低为 Local PRT V6：231/256，90.23%；
- 相对FC22：4局败转胜、0局胜转败、1,313局现金增加、0局现金下降；
- FC24B CPU版与JAX版在36个完整官方上下文中，719步动作、每帧状态和终局现金全部严格一致；
- CPU动作23,008次全部复现，平均0.523 ms，P99 1.350 ms，最大5.004 ms；
- 同一Python模块连续正序、倒序复跑64局，共46,016次动作，无跨局状态污染。

机器可读总门：

`workspace/experiments/fusion_champion_v1/receipts/fc24b_frozen_acceptance_manifest_v1.json`

状态：`PASS`，14/14硬检查通过。

## 2. 推荐阅读顺序

1. `docs/DEVELOPMENT_TIMELINE_FC0_TO_FC24B_ZH.md`：先理解版本演进、失败实验和最终能力栈。
2. `docs/development_chain/FC24B_FINAL_OFFICIAL_JAX_ACCEPTANCE_20260823_ZH.md`：最终验收总报告。
3. `docs/development_chain/FC24B_FULL_LOCAL_NON_REGRESSION_GATE_20260823_ZH.md`：34-Agent逐项90%门。
4. `docs/development_chain/FC22_FULL_LOCAL_REGRESSION_GATE_20260823_ZH.md`：FC24B的直接基座。
5. `docs/development_chain/FC15_EARLY3_EXACT_COUNTERFACTUAL_FIX_REVIEW_20260823_ZH.md`：Public失败如何暴露本地覆盖不足。
6. `docs/development_chain/FC2A_PUBLIC_STALL_ROOT_CAUSE_20260821_ZH.md`：为什么状态同步错误会使Python版中途卡死。
7. `docs/ALL_EXACT_JAX_ROUND_ROBIN_100_GAMES_20260820_ZH.md`：历史28-Agent Arena背景。

`docs/development_chain/` 内保留了30篇原始阶段报告，没有删除失败、无效或被拒绝的实验。

## 3. FC24B实际策略栈

FC24B不是按对手名字切换的Meta列表。正式策略只读取比赛中可见的当前或历史公开状态：

1. FC15基座：K320公共盘面路由、PRT/X562杂草与雇工同步、显式羊压力分支；
2. Moon公共市场观察器：估计异常供给，提前出售后用债务环在原销售日扣回，避免凭空制造库存；
3. 开局小麦种子数量8；
4. 早期批量胡萝卜种子订单减1，保护后续小麦与喂养现金；
5. 第10天后按公开的产品价值与小麦替代成本决定是否容忍首次漏喂，并维护小麦信用账本；
6. 终局只在最后可行时刻补收成熟作物，且作物价值必须至少是随身货物价值的2倍。

禁止在线使用：对手作者名、Replay ID、事件seed、固定地图坐标、未来隐藏商店或对手私有库存。

## 4. 目录结构

```text
latest_20260823_fc24b/
├─ README_ZH.md
├─ docs/
│  ├─ DEVELOPMENT_TIMELINE_FC0_TO_FC24B_ZH.md
│  ├─ ALL_EXACT_JAX_ROUND_ROBIN_100_GAMES_20260820_ZH.md
│  ├─ LATEST_PUBLIC6_SCAN_20260822_ZH.md
│  └─ development_chain/                 FC0到FC24B的30篇原始报告
├─ workspace/
│  ├─ experiments/
│  │  ├─ fusion_champion_v1/             FC24B源码、配置、工具、报告、关键回执和官方轨迹
│  │  ├─ strategic_v5/                   JAX公共策略依赖
│  │  ├─ expert_business_agent_v2/       最新6迁移工具、路线库和验收回执
│  │  ├─ route_playbook_v1/              轨迹执行与事件依赖
│  │  └─ kawashigi_counterfactual_ranker_v2/  路线库加载依赖
│  ├─ gpu_sim/                            官方1.32.7 JAX规则核与冻结规则
│  ├─ public_notebooks/                   选定最新6个原始Notebook
│  ├─ references/public_latest6_20260822/ 静态解包源码和来源清单
│  └─ submission/55708153_*/              实际上传包、main.py和上传回执
└─ MANIFEST_SHA256.tsv                    全文件大小与SHA-256
```

## 5. 最重要的冻结文件

- 最终参数：`workspace/experiments/fusion_champion_v1/configs/fc24b_frozen_best_v1.json`
- 当前最佳指针：`workspace/experiments/fusion_champion_v1/configs/CURRENT_BEST_RULE_CONFIG.json`
- JAX策略：`workspace/experiments/fusion_champion_v1/src/fusion_champion_v1/policy_gpu.py`
- 自包含CPU Agent：`workspace/experiments/fusion_champion_v1/artifacts/fc24b_cpu_v1/main.py`
- 实际上传包：`workspace/submission/55708153_fc24b_34agent_90pct_strict_jax_parity/submission.tar.gz`
- 最新6 JAX控制器：`workspace/experiments/strategic_v5/src/strategic_v5/latest_public6_20260822_gpu.py`

CPU Agent SHA-256：

`E7ABC6C215CF5A2FB2CF8FEDE83157568123DDBA0E086571ECD82A24751D9E6B`

上传包 SHA-256：

`9871648E899E7498C0E7913F454B658F99D5EC4953FF5745D289AFE78E069A59`

## 6. 快照内静态验收

在本目录执行：

```powershell
python workspace\experiments\fusion_champion_v1\tools\freeze_fc24b_acceptance.py
```

该命令会重新读取来源Notebook、静态解包源码、路线库、竞技场回执、官方轨迹、JAX一致性、CPU时限、跨局重入、最终配置和源码哈希。所有硬门满足时返回0并输出 `PASS`。

## 7. RTX3090逐步一致性复跑

从 `workspace` 目录运行，并把Python路径替换为本机CUDA JAX环境：

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/ai_coding/kaggle/kaggriculture/.publish/LZhangGJ_Kaggriculture/nt/latest_20260823_fc24b/workspace && /mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim/.venv-wsl/bin/python experiments/fusion_champion_v1/tools/accept_fc12g_jax_stepwise_parity.py --trace-receipt experiments/fusion_champion_v1/receipts/fc24b_cpu_v1_official_stepwise_traces_seed594001_n8x2_v1.json --output /tmp/fc24b_snapshot_parity.json --policy fc24b"
```

历史完整34-Agent测试已有8,704局，不建议只为了验证文件复制而重复运行；应先执行静态清单，再按需要运行少量GPU烟测。

## 8. 真实性边界

- “精确JAX”指冻结官方轨迹语料逐动作、逐状态、终局严格一致，不是有限seed对全部可达状态的数学证明。
- 90%门只代表冻结的34-Agent池、指定事件库和双座位协议，不保证未来新增Agent。
- Public分数与本地Arena不是同一分布；提交 `55708153` 的线上结果应单独记录。
- FC24初版因伤害牛奶/羊毛终局出售而被拒绝；只有加入2倍机会成本门的FC24B被提升。
