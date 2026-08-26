# Route Clustering Switch Agent

基于 Replay 路线聚类和反事实搜索的 Kaggriculture 提交 Agent。本目录同时包含可直接
提交的成品、可读运行时，以及从 Replay 重建路线库和切换策略的完整流水线。

当前策略不依赖 PPO、神经网络、PyTorch 或模型权重。它由 175 条代表路线、动作带、
147 维可解释状态特征和 3 个浅层决策树切换节点组成，在线最多切换一次。

## 从这里开始

按目的选择入口：

| 目的 | 入口 | 说明 |
|---|---|---|
| 直接提交 | `main.py` | zlib + Base85 打包的单文件 Agent，固定从 G001 开局 |
| 阅读或调试运行时 | `runtime/main.py` | 与单文件提交对应的多文件版本和运行资产 |
| 修改搜索/训练逻辑 | `src/meta_agent/`、`scripts/` | 特征、控制器、执行器和离线流水线 |
| 用自己的 Replay 重建 | `scripts/run_pipeline.sh` | 从清单开始生成路线库、策略和新提交 |
| 理解方法与实验 | `docs/宏观路线去重聚类与切换搜参.md` | 公式、实验规模、验证结果与局限 |

只想确认仓库中的成品能加载，可直接运行：

```bash
python -I main.py
python -I runtime/main.py
```

这两条命令只检查初始化，不会启动完整训练或搜索。

## 目录结构

```text
.
├── main.py                         # 可直接提交的单文件 Agent
├── runtime/                        # 可读的最终运行时与策略资产
├── src/meta_agent/                 # 离线开发使用的 Agent 源码
├── scripts/                        # 聚类、搜索、评估和导出脚本
├── configs/                        # Replay 清单与流水线配置示例
├── fast_kaggriculture/             # C++20 / pybind11 原生仿真器
├── tests/                          # Python 策略单元测试
├── docs/                           # 方法文档、PDF 和图表
├── OMITTED_ARTIFACTS.md            # 未纳入 Git 的大型/历史产物
└── SOURCE_PROVENANCE.md            # 源码恢复位置与可信边界
```

`runtime/` 是已经导出的提交快照，不是 `src/` 的机械镜像。修改开发源码后应通过导出
脚本重新生成运行时和 `main.py`，不要假设同名文件必须逐字节相同。

## 环境与快速验证

建议使用支持 C++20 和 OpenMP 的 Linux 环境。Python 依赖为 NumPy、SciPy、
scikit-learn、orjson、pybind11 和 pytest：

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src pytest -q tests
```

原生扩展和距离工具可构建到指定的工作目录：

```bash
bash scripts/build_native.sh /path/to/output-root
```

构建脚本会安装 `fast_kaggriculture` 的可编辑扩展，并把 `intent_distance` 写入
`/path/to/output-root/bin/`。子模块的 API 和差分测试说明见
[`fast_kaggriculture/README.md`](fast_kaggriculture/README.md)。

## 用自己的 Replay 完整重建

完整流水线计算量很大，会执行路线聚类、全互打、两阶段反事实搜索、鲁棒树选择、
checkpoint 组合搜索和独立留出评估。先准备 Replay 清单；最小字段格式见
[`configs/manifest.example.json`](configs/manifest.example.json)。清单中的 `replay`
路径相对清单文件解析，文件名应为 `episode-<id>-replay.json`。

然后复制并修改配置：

```bash
cp configs/pipeline.env.example /path/to/pipeline.env
# 设置 REPLAY_MANIFEST、REPLAY_ROOT、OUTPUT_ROOT 和 worker 数量
bash scripts/run_pipeline.sh /path/to/pipeline.env
```

关键配置：

| 变量 | 是否必需 | 含义 |
|---|---:|---|
| `REPLAY_MANIFEST` | 是 | 主 Replay 清单路径 |
| `REPLAY_ROOT` | 是 | 主 Replay 文件根目录 |
| `OUTPUT_ROOT` | 是 | 所有缓存、矩阵、策略和导出 Agent 的输出目录 |
| `EXTRA_REPLAY_MANIFEST` / `EXTRA_REPLAY_ROOT` | 否 | 第二批 Replay；必须同时设置 |
| `PYTHON_BIN` | 否 | Python 解释器，默认 `python` |
| `FEATURE_WORKERS` / `AUDIT_WORKERS` / `TREE_WORKERS` | 否 | 各阶段并行度 |
| `OPENING_FAMILIES` / `FINAL_OPENING` | 否 | 覆盖自动求得的 Nash 开局支持 |

生成物全部写入 `OUTPUT_ROOT`，不会写入源码目录。成功后最终单文件位于
`$OUTPUT_ROOT/agent/main.py`。如需严格沿用原实验开局，可配置：

```bash
OPENING_FAMILIES=G001,G136
FINAL_OPENING=G001
```

搜索阶段必须使用 `runtime/teammate_base.py`；其中包含路线干预接口、动作对齐、除草
修复和市场覆盖。不要替换成不含干预接口的普通 base 文件。

### 流水线阶段

1. 从 Replay 恢复生产、布局和扩张时序，并合并去重特征缓存。
2. 审计执行质量，以 intended macro actions 计算距离并聚类路线。
3. 导出代表路线及动作带，过滤零胜场路线。
4. 全互打并求经验 Nash 开局支持。
5. 粗搜候选切换路线，再以更多 seed 精搜反事实标签。
6. 训练并筛选鲁棒浅树，搜索最多一次切换的 checkpoint 组合。
7. 在新 seed 上做 holdout，导出多文件运行时和单文件提交。

“275 个聚类族、175 条有效路线、11 个精搜候选、63 个 checkpoint 子集”是原始
2026-08-25 数据集的结果。更换 Replay 后，族数、路线数、Nash 支持和候选数会自然
变化，不应把这些数量写成流水线不变量。

## 原实验结果

原始数据包含 3,563 个 Replay 座位。最终独立留出覆盖 175 个路线对手、256 个新
seed 和双座位，共 179,200 局：

| 策略 | 得分率 |
|---|---:|
| 固定 G001 | 81.18% |
| 动态切换 | 89.61% |
| 提升 | 8.44 个百分点 |
| 提升的单侧 95% 下界 | 7.64 个百分点 |

最终策略在第 144、168、216 步检查，在线最多切换一次。以上结果只针对当前 175 个
代表对手，不等同于真实榜单总体胜率。完整方法和防过拟合边界见方法文档。

## 文档索引

- [`docs/宏观路线去重聚类与切换搜参.md`](docs/宏观路线去重聚类与切换搜参.md)：方法、公式、实验和局限。
- [`docs/宏观路线去重聚类与切换搜参.pdf`](docs/宏观路线去重聚类与切换搜参.pdf)：同主题 PDF 版本。
- [`OMITTED_ARTIFACTS.md`](OMITTED_ARTIFACTS.md)：未提交的 Replay、缓存、旧 PPO 资产和哈希。
- [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md)：服务器源码来源及无法提供上游 commit 的原因。
- [`fast_kaggriculture/README.md`](fast_kaggriculture/README.md)：原生仿真器构建、API 和一致性测试。
- [`agent.md`](agent.md)：后续维护本目录时的边界、约定和验证清单。

## 维护注意事项

- 不要把 Replay、NPZ、距离矩阵、编译产物或模型权重提交到本目录；`.gitignore` 已
  覆盖常见格式，详细原因见 `OMITTED_ARTIFACTS.md`。
- `main.py` 是生成文件。功能修改应落在源码或导出脚本中，再重新导出并验证。
- 不要凭旧实验文件名推断动态数量；读取实际生成的 JSON/NPZ 元数据。
- 运行完整流水线前确认 `OUTPUT_ROOT` 位于源码树之外，并按机器资源降低 worker 数量。
