# Agent maintenance guide

本文约束在 `agents/route_clustering_switch_agent/` 内工作的代码 Agent。目标是让修改
保持可复现、可提交，并且不误伤 Kaggriculture 仓库的其他目录。

## 工作边界

- 默认只读写本目录及其子目录；不要修改父目录、兄弟 Agent 或仓库级配置。
- 若任务确实需要跨目录变更，先说明具体文件、原因和影响，获得用户明确授权后再做。
- 工作区可能已有用户修改。先检查本目录的状态，保留无关改动，不覆盖或回滚它们。
- Replay、生成矩阵、搜索缓存和本机构建产物必须写到源码树外的 `OUTPUT_ROOT`。

## 先识别修改层级

- `main.py`：生成的单文件提交，不作为日常手工编辑入口。
- `runtime/`：已导出的可读提交快照，包含最终路线库、动作带和策略。
- `src/meta_agent/`：特征、策略控制器和执行器的开发源码。
- `scripts/`：离线数据处理、搜索、评估和导出逻辑。
- `fast_kaggriculture/`：C++20/pybind11 仿真器；其独立说明见本目录内 README。
- `docs/`：方法说明与图表，不是运行时依赖。

修改行为逻辑时，优先修改 `src/`、`scripts/` 或原生源码，再用
`export_teammate_meta_submission.py` 和 `export_single_file_submission.py` 生成成品。
不要直接修改压缩后的 `main.py`。`runtime/` 是导出快照，和 `src/` 的同名文件不一定
完全相同；不要进行未经验证的双向覆盖。

## 关键不变量

- 当前 Agent 是基于路线搜索的浅决策树策略，不依赖 PPO、PyTorch 或 `.pt` 权重。
- 当前在线策略从 G001 开局，在 144、168、216 checkpoint 判断，最多切换一次。
- 搜索和评估必须使用 `runtime/teammate_base.py` 的路线干预接口。
- 特征 schema 为 `semantic_route_switch_v1`；改变 schema 时必须同步训练、导出和运行时。
- 原实验的 275/175/11/63 等数量来自特定数据，不是换数据后仍成立的常量。
- `OUTPUT_ROOT` 应位于源码树外；不要把大体积生成物加入普通 Git 历史。

如有意改变这些不变量，应同时更新 README、运行时 manifest、相关测试和方法文档，
并明确说明兼容性影响。

## 第一阶段聚类质检门

完整命令成功退出只说明流程能运行，不说明路线库可用。进入路线互打、Nash 或反事实
搜索前，必须完成下面的质检；否则后续高成本计算应暂停。

### 1. 检查族规模分布

- 先区分两层聚类：`analyze_macro_route_library.py` 的按队伍诊断聚类，与
  `cluster_intended_macro_routes.py` 的全数据统一 `G###` 聚类。不得混用两层标签或占比。
- 对实际输出计算每个族的 `support / replay_sides`，列出至少前 10 族、累计覆盖率、
  单例族数量和总族数。
- 第一层按队伍聚类的经验轮廓通常头重脚轻：第一族约含 40% 的局，随后头部族约为
  二十几、十几个百分点，再进入长尾。
- 对前排队伍最近约 20 个有效 Replay 单独做一次窗口检查。分成 3–4 条路线通常合理；
  达到 6 条及以上通常不正常，应检查 Replay 选择、意图恢复、距离和阈值。
- 若新结果近似平均、最大族明显过小、单例族异常多，优先检查 Replay/座位选择、意图
  恢复、布局与时序对齐、距离分量是否退化，以及聚类阈值。
- 若单族吞掉绝大多数 Replay，同样停止后续搜索，检查特征是否变成常量或阈值过大。

40%/20%/10% 和“最近 20 局得到 3–4 条、6 条以上需排查”都是经验启发式，不是最佳
标准、统计置信区间或自动拒绝规则。策略真实换代、Replay 数量不足或样本偏置都可能使
结果偏离；偏离意味着要解释和审计，不等于直接判错。

### 2. 检查中心样本，而不只看 medoid 距离

当前 `scripts/export_macro_intent_route_specs.py` 使用纯几何 medoid：执行失败、胜负和
现金会被输出，但不会自动影响中心样本选择。不得因为脚本生成了 `intent_medoid_source`
就默认该动作带质量合格。

对每个待评测族，尤其是覆盖率高的头部族，至少记录：

| 字段 | 检查目的 |
|---|---|
| `family`、`support`、占比 | 判断错误代表会影响多少 Replay |
| medoid route 与平均族内距离排名 | 保证候选仍处于路线中心附近 |
| 未成功生产任务数 | 重点检查种植、建造、放置动物失败 |
| `completion_rate`、奖励重放一致性 | 排除执行器或 Replay 语义异常 |
| 族内 W/T/L、中心样本自身赛果 | 避免选中明显失败载体 |
| 最终现金/奖励、最低现金、资本缺口 | 检查资金链和路线可执行性 |

### 3. 在中心性约束内选质量更好的 carrier

1. 按平均族内距离先取 medoid 附近的一小批真实 Replay，而不是遍历全族后直接挑最高分。
2. 优先选择未成功生产任务更少、完成率更高且奖励重放一致的样本。
3. 其余条件接近时，再参考胜负表现、最终现金和最低现金。
4. 若质量较好的样本离几何中心很远，不要用高分离群点强行替换；应回查聚类、拆族，
   或为该宏观族保留多个候选 carrier。
5. 胜率和现金仅用于聚类后的代表质量控制，不得加入聚类距离。

建议采用字典序：`生产硬失败少 → 赛果合理 → 最终现金高 → 平均族内距离小`。最终
选择必须形成一份位于 `OUTPUT_ROOT` 的质检 JSON/CSV，包含采用或拒绝每个中心样本的
原因。该文件是实验凭证，不提交普通 Git。

### 4. 停止条件与低效果排查顺序

出现以下任一情况时，默认暂停并人工复核后再决定是否进入全互打：族规模轮廓显著异常；
最近 20 局被拆成 6 条及以上且没有策略换代证据；头部族 medoid 有大量生产失败；奖励
无法精确重放；头部 carrier 普遍资金链断裂；质检表缺失。

如果最终 Agent 效果不佳，按以下顺序排查：聚类分布 → 头部族 carrier 质量 →
C++/JAX 语义一致性 → 互打矩阵/Nash → 反事实标签 → 决策树。不要先调树深或切换
阈值；上游动作带不合格时，后续搜索只会更准确地拟合错误路线。

## 推荐工作流程

1. 阅读 `README.md`，再按任务查看方法文档、脚本或原生模块。
2. 用 `git status --short -- agents/route_clustering_switch_agent` 检查已有改动。
3. 只修改完成任务所需的最小文件集；不要顺手清理无关文件。
4. 生成路线族后先执行“第一阶段聚类质检门”，通过后才运行互打和搜索。
5. 先运行与改动最相关的测试，再执行下面的基础验证。
6. 若重新导出提交，对比 `runtime/MANIFEST.json`、生成日志和最终文件，并记录输入来源。
7. 交付时列出修改文件、质检表、验证命令以及未执行的高成本步骤。

## 验证清单

文档或轻量 Python 修改至少运行：

```bash
python -I main.py
python -I runtime/main.py
PYTHONPATH=src pytest -q tests
```

涉及原生仿真器时，另在源码树外构建并运行其测试：

```bash
bash scripts/build_native.sh /path/to/output-root
pytest -q fast_kaggriculture/tests
```

涉及训练、路线数量、checkpoint 或收益结论时，应使用独立 `OUTPUT_ROOT` 运行对应流水线
阶段，并以新生成的摘要为准。完整流水线成本很高；未运行时必须明确写明，不得把静态
检查描述成实验复现。

## 数据与文档约定

- Replay 清单遵循 `configs/manifest.example.json`，Replay 路径相对清单文件解析。
- 配置文件遵循 `configs/pipeline.env.example`；本地 `configs/pipeline.env` 不提交。
- 来源、恢复过程或可信边界变化时更新 `SOURCE_PROVENANCE.md`。
- 删除或外置大型产物时更新 `OMITTED_ARTIFACTS.md`，保留原因、大小和可用哈希。
- README 面向首次使用者，优先保持入口、验证、复现步骤和结果边界准确；深入公式留在
  `docs/`，原生实现细节留在 `fast_kaggriculture/README.md`。

## 2026-09-03：NT 语义克制拳法 Round2 重测

本轮实验验证“固定少数开局后，在 day 6/9/12/18/24 根据实时状态切换语义克制拳法”是否可行。完整实验代码、锁文件、结果和逐步记录已复制到独立目录：

`D:\Kaggriculture\counter_cluster_round2_semantic_v1_20260903`

Git 仓库内对应目录：`research/experiments/counter_cluster_round2_semantic_v1_20260903`。

详细过程见该目录的 `agent.md`，实验协议见 `PROTOCOL.md`，摘要见 `RESULT.md`，可复核分析见 `round2_analysis.ipynb`。

执行要点：固定当前 v3 G003（`103928643:1`）至 step 143；将以前 48 个 W12/A64 搜索选择物化为 day 6/9/12/18/24 的语义 `PlanDelta`，禁止固定动作带和 rank 重放；通过 48/48 discovery 等价门并去重为 25 套拳法；随后在 12 条训练 Replay、8 个新 seed、双座位上运行 4,800 个 treatment 对局和 192 个 Opening A baseline 对局；最后从原始 reward 独立复算并执行 notebook 的 22 项断言。训练门失败后没有运行 validation/holdout，也没有在本轮重新运行昂贵的 NT 搜索。

初步结果：24,000 个 rank 均为 `-1`；PlanDelta 阶段匹配率 `75.3458%`；已匹配阶段中 `90.1012%` 的完整 delta 会随实时状态改变，证明自适应执行生效。但 300 个拳法–对手单元中，仅 1 个通过 score 门，0 个通过相对 Opening A 的 uplift 门，0 个通过正 mean-margin 门，最终 coverage 为 `0/300`，无法形成有效 response cluster。即使只看 14 个五阶段全匹配单元，最好 score 也只有 `0.25`。

当前主要问题：旧候选由单 seed/单座位 discovery 搜索产生，明显存在搜索过拟合；25 套候选不足以否定更大 NT 空间；day 9/day 12 意图匹配率约为 62%，但完全匹配单元同样失败，因此匹配不是首要瓶颈；目前仅测试 G003 和 12 条训练路线；历史 Replay 对手不会在线响应我方变招；归档仍依赖原工作区的 Round1 输入、Candidate8 genome/meta-agent 源码、原始 Replay 和 CPython 3.13/MinGW 运行时，适合审阅但不是搬目录即运行的发行包。

下一步只应对 12 条 residual 路线运行多 seed、双座位的稳健 NT 搜索，直接优化 score、uplift 和 reward margin。训练集得到非空拳法 portfolio 后，才继续 response clustering、前缀后验分类和 validation。
