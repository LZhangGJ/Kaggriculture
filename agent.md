# Kaggriculture 协作索引与上传规则

这是仓库的统一入口。任何人或代码 Agent 应先看这里，再进入对应模块或实验目录。
目标是用一分钟回答四个问题：代码在哪里、实现了什么、用了什么数据、结果是否有效。

最后更新：2026-09-03。

## 历史资产入口

完整 Git 分支、D/E 盘代码、生成 Agent、Replay、模型和旧实验盘点见
[`research/HISTORICAL_ASSETS.md`](research/HISTORICAL_ASSETS.md)。该文件是唯一历史资产
总表；本页只保留当前推荐入口。

## 仓库地图

| 目录 | 内容 | 是否应直接修改 |
|---|---|---|
| `agents/<module>/` | 可提交 Agent、模块源码及模块说明 | 是；先读模块 `agent.md` |
| `research/experiments/<experiment>/` | 一次实验的代码、配置、结果和实验说明 | 是；每次实验独立一目录 |
| `research/notebooks/` | 外部或探索性 Notebook | 一般只读；结论必须进入实验目录 |
| `research/episodes-index/` | Replay 索引，不是 Replay 本体 | 谨慎更新 |
| `src/`、`tests/`、`benchmarks/` | 通用模拟与训练基础设施 | 是；同步测试和模块索引 |
| `official/`、`vendor/` | 官方规则和固定版本依赖 | 默认只读 |

## 模块索引

| 模块 | 状态 | 功能 | 代码入口 | 模块说明 | 主要结论 |
|---|---|---|---|---|---|
| 通用本地模拟实验室 | 可运行；本分支未重跑 | CPU/GPU 环境、批量对局和训练基础设施 | [`src/kaggriculture_lab/`](src/kaggriculture_lab/) | [`README.md`](README.md) | README 记录了历史吞吐量；不是克制拳法模块 |
| Route Clustering Switch Agent | 可运行；克制聚类仍在研究 | Replay 路线聚类、路线库、Nash 开局和在线切换 | [`agents/route_clustering_switch_agent/`](agents/route_clustering_switch_agent/) | [`agent.md`](agents/route_clustering_switch_agent/agent.md) | 历史切换策略有效；Round2 语义克制拳法训练覆盖为 0 |

新增模块时必须在本表增加一行。状态只允许使用：`提案`、`开发中`、`可运行`、
`实验失败`、`已验证`、`已废弃`；需要补充说明时放在分号后。

## 实验索引

| 实验 | 状态 | 要验证的问题 | 代码与说明 | 数据 | 结论 |
|---|---|---|---|---|---|
| NT 语义克制拳法 Round2（2026-09-03） | 实验失败；保留负结果 | 旧 NT 选择能否在新 seed、双座位下稳定克制 Replay 路线 | [`实验说明`](research/experiments/counter_cluster_round2_semantic_v1_20260903/agent.md) | 12 条训练路线、8 seeds、双座位；锁文件见实验目录 | 0/300 单元同时过门，无法形成 response cluster |

实验失败也必须登记，防止其他人重复消耗算力。新增实验时必须在本表增加一行；同一实验
的重跑放在原目录中并记录新 run，改变目标或数据切分时新建实验目录。

## 数据索引

| 数据 | 位置 | 内容与范围 | 可用性 |
|---|---|---|---|
| Top episode 路由索引 | [`research/episodes-index/manifest.csv`](research/episodes-index/manifest.csv) | Replay 下载索引，不包含完整 Replay | Git 内可用 |
| Round2 小型结果 | [`research/experiments/counter_cluster_round2_semantic_v1_20260903/`](research/experiments/counter_cluster_round2_semantic_v1_20260903/) | 配置、锁文件、汇总 JSON、NPZ、Notebook | Git 内可审计 |
| Round2 完整本地归档 | `D:\Kaggriculture\counter_cluster_round2_semantic_v1_20260903` | Windows 运行时及本轮完整归档 | 仅该本机；详见实验 `agent.md` |
| 原始 Replay 与 Round1/Candidate8 输入 | 见 Round2 `inputs.lock.json` | 精确复跑依赖的外部输入 | 不在 Git；必须先检查路径和哈希 |
| 历史资产总表 | [`research/HISTORICAL_ASSETS.md`](research/HISTORICAL_ASSETS.md) | Git 分支、D/E 盘代码、模型、Replay 和旧实验 | 2026-09-03 已盘点 |

不要只写“使用 top40”或“结果在 D 盘”。每个外部数据项必须说明：用途、选择范围、绝对
路径或可下载地址、文件数量/大小（能取得时）以及哈希或版本。禁止提交密钥和账户信息。

## 上传规则

### 1. 代码按模块归档

- 已有功能放回已有模块，不创建 `final_v2_new_latest` 一类平行目录。
- 新模块使用 `agents/<module_slug>/`，入口代码、配置和测试放在该目录或明确列出的通用目录。
- 模块 `agent.md` 必须包含：一句话说明、状态、代码地图、输入与输出、验证、结果、
  已知问题、相关实验。
- 生成的单文件 Agent、压缩包或二进制必须写明生成源和生成命令，不能成为唯一源码。

### 2. 每次实验单独归档

- 目录名使用 `<主题>_<YYYYMMDD>`；同一协议的重跑可在目录下使用 `runs/<run_id>/`。
- 实验目录至少保留：`agent.md`、实际执行代码、配置/seed、输入锁文件、机器可读摘要。
- `agent.md` 必须包含：一句话结论、状态、问题、代码位置、数据位置、方法、结果、复现、
  局限、下一步。
- 原始日志和大型矩阵可外置，但必须留下摘要、外部路径、大小、哈希和生成命令。

### 3. 结果必须可区分可信度

- `已执行`：本次确实运行，写明命令、环境、seed、双座位与否、样本数和退出状态。
- `历史记录`：来自已有文件或他人报告，本次未重跑，必须明确标注。
- `计划/假设`：尚未运行，不能写成结果。
- 负结果、失败门和未运行的 validation/holdout 都必须保留。
- 对比实验必须使用相同数据切分、seed 和座位设置；否则说明不可直接比较。

### 4. 每次提交都更新索引并校验

```bash
python scripts/check_agent_docs.py
git status --short
```

代码、实验结果、对应 `agent.md` 和根 `agent.md` 索引应在同一提交中。GitHub Actions
会运行同一校验；仓库管理员应把 `agent-docs` 设置为受保护分支的必需检查。

新增、移动或废弃仓库外代码、数据、模型和生成 Agent 时，还必须同步更新
`research/HISTORICAL_ASSETS.md`。

## 文档模板

模块 `agent.md`：

```markdown
# <模块名>
## 一句话说明
## 状态
## 代码地图
## 输入与输出
## 验证
## 结果
## 已知问题
## 相关实验
```

实验 `agent.md`：

```markdown
# <实验名>
## 一句话结论
## 状态
## 问题
## 代码位置
## 数据位置
## 方法
## 结果
## 复现
## 局限
## 下一步
```

每个标题下面必须给出实际内容；未知就写“未知/未运行”并说明原因，不能删除标题。
