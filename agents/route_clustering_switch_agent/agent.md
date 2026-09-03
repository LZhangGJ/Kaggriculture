# Route Clustering Switch Agent

## 一句话说明

从 Replay 恢复并聚类宏观路线，筛选代表路线和开局，再用实时特征选择是否切换路线。

## 状态

`可运行`。仓库中的现有成品是历史浅层切换策略；“固定少数开局后按对手后验切换语义
克制拳法”的目标仍处于实验阶段，尚未生成有效 response cluster。

不要把两者混为同一功能：当前 `main.py` 没有实现新的多阶段贝叶斯后验克制树。

## 代码地图

| 路径 | 用途 | 修改规则 |
|---|---|---|
| `main.py` | 生成的单文件提交 | 不手改；由导出脚本生成 |
| `runtime/` | 可读的已导出运行时、路线库和策略 | 与 `main.py` 一起验证 |
| `src/meta_agent/` | 特征、控制器和路线执行开发源码 | 行为修改的首选入口 |
| `scripts/` | Replay 恢复、聚类、互打、Nash、搜索、评估和导出 | 实验脚本入口 |
| `fast_kaggriculture/` | C++20/pybind11 快速仿真器 | 修改后跑差分测试 |
| `tests/` | Python 策略测试 | 与相关代码同步更新 |
| `docs/` | 深入方法和历史图表 | 不作为运行时依赖 |
| `README.md` | 使用、重建和历史方法说明 | 面向首次使用者 |

关键运行时约束：现有成品从 G001 开局，在 step 144、168、216 检查，最多切换一次；
特征 schema 为 `semantic_route_switch_v1`。搜索和评估必须使用
`runtime/teammate_base.py` 的路线干预接口。改变这些约束时同步更新源码、运行时
manifest、测试、README 和本文件。

## 输入与输出

输入：Replay manifest、Replay 文件、流水线配置和 seed。清单格式见
`configs/manifest.example.json`，环境配置见 `configs/pipeline.env.example`。

输出：路线族、执行质检、互打矩阵、Nash 支持、切换策略、`runtime/` 和单文件
`main.py`。大型 Replay、缓存、矩阵及本机构建产物必须写到源码树外的 `OUTPUT_ROOT`；
仓库只保存小型摘要、配置、哈希和复现入口。

## 验证

轻量修改至少执行：

```bash
python -I main.py
python -I runtime/main.py
PYTHONPATH=src pytest -q tests
python ../../scripts/check_agent_docs.py
```

原生仿真器修改还需在源码树外构建并运行：

```bash
bash scripts/build_native.sh /path/to/output-root
pytest -q fast_kaggriculture/tests
```

完整流水线成本高。没有运行时必须写“未运行”，不能把导入、静态检查或历史 JSON
描述成实验复现。

进入高成本互打前先检查：族规模是否异常、头部 carrier 是否有生产硬失败、完成率和
奖励能否重放、现金链是否可执行。胜率和现金只用于同族代表的质量控制，不进入路线
相似度。质检失败时先修上游，不继续调 Nash 或决策树。

## 结果

- `历史记录，本分支未重跑`：README 记录 175 个代表对手、256 个 seed、双座位共
  179,200 局；固定 G001 得分率 81.18%，动态切换 89.61%，提升 8.44 个百分点。
- `已执行`：2026-09-03 Round2 把旧 NT 选择转换成实时语义 PlanDelta 后，在训练门
  得到 0/300 coverage，不能形成 response cluster。详情见相关实验目录。

这些结果针对不同协议，不能相互替代，也不等同于真实榜单胜率。

## 已知问题

- 当前路线聚类主要表达行为相似性，不等于“能被同一拳法稳定击败”的 response cluster。
- Round2 候选来自单 discovery seed/座位搜索，换 seed 和座位后失败。
- 历史 Replay 对手按已记录动作执行，不会模拟对方对我方变招后的完整在线响应。
- `runtime/` 是导出快照，不保证和 `src/` 同名文件逐字节一致。
- 原数据中的 275/175/11/63 等数量不是换数据后仍成立的常量。

## 相关实验

| 实验 | 状态 | 位置 | 结论 |
|---|---|---|---|
| NT 语义克制拳法 Round2 | 实验失败 | [`实验说明`](../../research/experiments/counter_cluster_round2_semantic_v1_20260903/agent.md) | 旧候选训练覆盖为 0；应重新做多 seed、双座位 NT 搜索 |

新增或重跑实验时，按仓库根 `agent.md` 规则创建/更新实验记录，并在本表增加链接。
