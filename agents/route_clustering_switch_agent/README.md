# Route-clustering switch agent

这是 2026-08-25 路线聚类与反事实搜索 router 的精简工程。当前 agent **不使用
PPO、神经网络或 PyTorch 权重**。2026-08-27 的当前提交由 7 条动作带、两个公共
状态检查点和精确状态回退表组成，在线最多切换一次。

## 入口与目录

- `main.py`：当前 NR295 + 遗传 donor selector 的 Kaggle 单文件提交，可直接上传。
- `runtime/`：同一策略的可读多文件运行时，只部署 7 条路线和 48/96 两个切换节点。
- `src/meta_agent/`：在线特征、切换控制器和动作带执行器。
- `scripts/`：当前聚类、反事实搜索、鲁棒树选择和提交导出流水线。
- `scripts/run_pipeline.sh`：从队友自己的 Replay 清单开始的一键复现入口。
- `configs/manifest.example.json`：Replay 清单的最小字段契约。
- `fast_kaggriculture/`：C++20/pybind11 仿真源码；已删除平台相关构建产物。
- `docs/宏观路线去重聚类与切换搜参.md`：完整方法、公式、局数与局限。
- `OMITTED_ARTIFACTS.md`：未上传内容、原因、大小和关键哈希。

## 方法摘要

1. 从 3,563 个 replay 座位恢复生产构成、布局和扩张时序。
2. 用预计算距离的 average-linkage 凝聚聚类得到 275 族，筛为 175 个有效族。
3. 175 条代表路线全互打 1,948,800 局，求 G001/G136 经验纳什开局。
4. 粗搜 13,720,000 局，将 175 个候选目标压缩到 11 个。
5. 精搜 6,899,200 局，生成同状态、同 seed、同座位下的反事实切换标签。
6. 使用浅 `DecisionTreeClassifier`，结合 seed/对手族分组交叉验证、阈值扰动、
   bootstrap 和配对收益置信下界选择 router。
7. 额外 2,867,200 局选择 checkpoint 组合，最终在 144、168、216 检查，在线最多
   切换一次。

最终独立留出包含 175 个路线对手、256 个新 seed 和双座位，共 179,200 局：固定
G001 得分率 81.18%，动态切换 89.61%，提升 8.44 个百分点；单侧 95% 下界为
7.64 个百分点。这个结论只针对当前 175 个代表对手，不等同于真实榜单总体胜率。

## 2026-08-27 全池联合 selector

实时 watcher 当前从 top40 与自有提交目录发现 12,833 个 replay 文件、8,537 个唯一
episode，持续聚合到 6,811 个唯一 RouteGenome。C++ 引擎将新 replay 路线、共享前缀
重组和后缀遗传变异统一放入反事实矩阵；适应度按“加入路线后 selector 的边际覆盖”
计算，而不是只看单路线胜率。

最终开局为 `NR295`，48 回合允许切到 `SP0027`，若延迟则在 96 回合从 `NR295`、
`SP0049`、`SP0053`、`SP0055`、`SP0059` 和新 donor 分支 `D103_0003` 中选择。
`D103_0003` 保留 NR295 的前 96 回合，后缀来自新 replay 筛出的 `NR103`。selector
只读取 147 维公开状态；未见状态回退 NR295，不读取 seed、对手 ID 或 oracle 标签。

冻结模型的独立结果：对 652 路线对手、128 个新 seed、双座位共 166,912 局，原始
总体胜率 95.51%，最低单对手 82.81%，全部 652 个对手达到 80%；对旧最强动态
G001 agent、512 个另一组新 seed、双座位共 1,024 局，胜率 81.84%，两个座位均为
81.84%。全部对局完成，抽查轨迹均为 719 回合。详细实验与失败消融见
`docs/JOINT_ROUTE_EVOLUTION_SELECTOR_20260827.md`。

## 2026-08-26 遗传路线库扩展

当前提交已由 175 条 replay 路线扩展为 183 条，其中 8 条 `EV001`–`EV008` 是多父本
遗传变异，不对应任何一条原始 replay。生成适应度使用路线对当前 11 目标组合的边际
轨迹覆盖；上线采用保守残差策略改进，完整保留旧切换树，仅在跨 seed、跨对手和完整
在线序列验证都改善时替换旧叶节点。最终接入的是 168 回合的 `EV003` 分支和 216
回合的两个 `EV001` 分支，仍由实际市场、商店、双方资产与历史状态决定路线。

独立原生留出中，旧线上策略得分率为 89.59%，扩展策略为 90.63%。官方 Python
引擎中新策略对旧策略运行 64 个新 seed、双座位共 128 局，配对得分率 77.34%，
平均收益差 +4,665.35，128/128 完成。对 `D:\Kaggriculture\public_agents` 当前实际
32 个顶层 agent 使用另一组 5 个 seed、双座位逐一验证，共 320/320 完成，每个
对手均为 10/10 胜，最低实际胜率 100%。完整方法、种子隔离和产物索引见
`docs/GENETIC_ROUTE_LIBRARY_EXPERIMENT_20260826.md`。

## 用自己的 Replay 从头复现

本目录现在包含完整源码，不再依赖 Git 中现有的 NPZ、矩阵或树结果。准备符合
`configs/manifest.example.json` 的清单，并保证清单中的 `replay` 路径相对清单文件
可解析；Replay 文件名应为 `episode-<id>-replay.json`。

```bash
python -m pip install -r requirements.txt
cp configs/pipeline.env.example /path/to/pipeline.env
# 修改 REPLAY_MANIFEST、REPLAY_ROOT 和 OUTPUT_ROOT
bash scripts/run_pipeline.sh /path/to/pipeline.env
```

所有特征缓存、距离矩阵、互打结果和搜索结果只写入 `OUTPUT_ROOT`，不写入源码目录。
如需合并两批 Replay，可在配置中同时设置 `EXTRA_REPLAY_MANIFEST` 与
`EXTRA_REPLAY_ROOT`。入口会依次完成 Replay→275 类路线族（数量取决于新数据）→
零胜场过滤→互打/Nash→粗细两阶段反事实搜索→鲁棒树→63 个 checkpoint 子集→
holdout→最终 Agent 导出。

“275 族、175 路线、11 个候选、63 个子集”是原始 2026-08-25 Replay 集合的结果；
换用队友数据后，族数、过滤后路线数、Nash 支持和候选数允许自然变化。若要严格复现
原实验开局，可在配置中指定 `OPENING_FAMILIES=G001,G136` 和
`FINAL_OPENING=G001`；未指定时由新互打矩阵的 Nash 支持自动决定。

搜索阶段必须使用 `runtime/teammate_base.py`：它含路线干预接口、动作对齐、除草修复
和市场覆盖。新压缩包里没有干预接口的普通 base 文件没有采用。

## RouteGenome v2 初代种子库

`src/meta_agent/src/route_genome.py` 定义了与 719 步动作带解耦的宏观路线基因。
每个 replay 座位被压缩为 5 个意图布局目标、10 个阶段宏观动作计数、建造/动物/
买地/雇工事件、市场价格行为摘要和现金画像。`genome_id` 只对遗传内容计算哈希：
市场买卖方向、商品、次数和数量参与身份计算，replay 中观察到的成交价格只作为环境
上下文保留。队名、奖励、现金和文件路径也不参与哈希，因此不同环境下的相同计划
会聚合到同一基因。

批量入口支持多个目录、按 episode 去重、双座位提取、gzip JSONL、断点续跑和多进程：

```bash
python scripts/extract_route_genomes.py \
  --source our_latest=/data/our-latest/replays \
  --source top40=/data/top40 \
  --output /data/route-genomes/route-genome-v2.jsonl.gz \
  --workers 4
```

已有 v1 执行库可以直接重新计算 v2 身份并生成一基因一行的聚合库，无需再次解析
数千个大型 replay：

```bash
python scripts/aggregate_route_genomes.py \
  --input /data/route-genomes/route-genome-v1.jsonl.gz \
  --output-records /data/route-genomes/route-genome-v2.records.jsonl.gz \
  --output-groups /data/route-genomes/route-genome-v2.groups.jsonl.gz
```

执行库保留每场比赛的完整市场价格、现金和结果；聚合库以 `genome_id` 合并相同计划，
同时给出样本数、来源、队伍、胜负、奖励、现金和市场环境分布。v2 读取器仍可校验和
迁移旧 v1 记录。

基于原仓库意图距离的族谱实验以每个唯一基因一票运行，保留全部基因并单独输出
`genome_id → family_id` 映射：

```bash
python scripts/cluster_route_genome_families.py \
  --groups /data/route-genome-v2.groups.jsonl.gz \
  --output-root /data/route-genome-families \
  --threshold 0.12
```

首轮生成实验先选取跨族父代，再运行可复现的计划层单点与复合变异：

```bash
python scripts/select_route_genome_parents.py \
  --groups /data/route-genome-v2.groups.jsonl.gz \
  --records /data/route-genome-v2.records.jsonl.gz \
  --family-map /data/route-genome-families/genome-family-map-v1.jsonl.gz \
  --families /data/route-genome-families/route-families-v1.json \
  --output /data/route-genome-families/initial-parent-panel-v1.json

python scripts/generate_route_genome_mutations.py \
  --groups /data/route-genome-v2.groups.jsonl.gz \
  --family-map /data/route-genome-families/genome-family-map-v1.jsonl.gz \
  --parents /data/route-genome-families/initial-parent-panel-v1.json \
  --output /data/route-genome-mutations/plan-mutations-v1.jsonl.gz
```

变异输出首先是尚未执行的计划基因。第一版可执行编译器位于
`src/meta_agent/src/route_plan.py` 和 `src/meta_agent/src/route_compiler.py`：它保留父代
动作带中的成熟微观行为，静态改写市场数量和同坐标作物，并为事件移时、相邻布局搬迁
生成带前置资源检查、寻路、确认和重试的状态感知任务。运行
`scripts/evaluate_route_genome_compiler.py` 可先验证父路线无损重建，再对分层变异样本统计
锚点完成度、宏观动作失败率和奖励变化。该版本是从 replay 载体到完全生成式规划器的
过渡层，编译报告会明确标出部分支持和风险项，不能把“成功生成 JSON”当作适应度。

公开 replay 在这里仅作为进化初代和溯源；后续变异、宏观路线编译与适应度搜索不再
要求生成路线对应某条原始 replay。

## 完全生成式路线闭环

`src/meta_agent/src/generative_route.py` 是独立于动作带的第二条执行路径。v2 基因保存
三阶段作物/动物容量、阶段日期、每日劳力、最早扩张日、现金储备、销售储备、布局种子、
购种前视期和饲料缓冲。`GenerativeRouteAgent` 每回合从实际地图、库存和现金重新生成
除草、播种、浇水、结构建造、动物购买/搬运/放置、喂养、照料、收获、肥料收集、移动、
雇工、买地和出售动作；它不读取父 replay 的任何逐回合动作。纯作物 v1 checkpoint
仍可读取，且零动物策略保持原行为 ID。

`src/meta_agent/src/route_evolution.py` 和 `scripts/evolve_generative_routes.py` 构成可断点续跑
的质量多样性搜索：

- MAP-Elites 在作物主导类型、田地规模、劳力和土地规模生态位中各保留一个精英；
- 变异、交叉、随机移民和 1–3 步复合变异都能创建从未出现在 replay 中的新基因；
- UCB 风格算子池根据子代是否超过父代自动更新生成概率；
- 无跨地目标时自动排除无表型作用的土地时序变异；
- 评估缓存绑定对手、种子与座位面板，换面板时必须重新验证；
- 新 replay 可只按宏观锚点转换为初代，并与已有搜索档案合并，不需要重启实验。

最小搜索、接入 replay 初代和断点续跑示例：

```bash
python scripts/evolve_generative_routes.py \
  --output /data/search/initial.json \
  --generations 2 --offspring 8 --seeds 17 --one-seat \
  --replay-groups /data/route-genome-v2.groups.jsonl.gz \
  --replay-bootstrap-limit 8

python scripts/evolve_generative_routes.py \
  --resume /data/search/initial.json \
  --merge-checkpoint /data/search/new-replay-archive.json \
  --output /data/search/continued.json \
  --generations 4 --offspring 16 --seeds 17 --one-seat \
  --max-mutation-depth 3
```

便宜的单种子筛选不能直接作为最终结论。用独立种子和双座位晋级验证：

```bash
python scripts/validate_generative_route_archive.py \
  --checkpoint /data/search/continued.json \
  --output /data/search/continued-validation.json \
  --top-k 3 --seeds 17 29 43
```

2026-08-26 本地试验中，完全生成式作物路线从人工高价值先验的 24,023 提升到
29,151；加入动物任务链后，结构先验达到 40,869，进化产生的 3 COW + 4 SHEEP
路线达到 49,440。后者在 4 个种子、双座位的八场面板上平均 44,860.125、最低
36,651。作物阶段见 `docs/GENERATIVE_ROUTE_EXPERIMENT_20260826.md`，动物阶段见
`docs/GENERATIVE_ROUTE_ANIMAL_EXPERIMENT_20260826.md`。

## 快速检查

单文件仅做初始化检查：

```bash
python -I main.py
```

可读运行时也可以独立初始化：

```bash
python -I runtime/main.py
```

运行搜索脚本时，再按脚本需要设置 `PYTHONPATH=src`。

重建 C++ 扩展可运行 `scripts/build_native.sh /path/to/output-root`。服务器来源与无法
提供上游 Git commit 的原因见 `SOURCE_PROVENANCE.md`。
