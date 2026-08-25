# Route-clustering switch agent

这是 2026-08-25 路线聚类与反事实搜索 router 的精简工程。当前 agent **不使用
PPO、神经网络或 PyTorch 权重**。最终提交是路线动作带、可解释状态特征和浅决策
树组成的单次切换策略。

## 入口与目录

- `main.py`：fixed-G001 的 Kaggle 单文件提交，可直接上传。
- `runtime/`：同一策略的可读多文件运行时，包括 175 条路线、动作带和 3 个切换
  节点。
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

重建 C++ 扩展可运行 `scripts/build_native.sh /path/to/output-root`。本次按要求只做了
源码和调用关系的静态整理，没有执行构建、训练、搜索或测试。服务器来源与无法提供
上游 Git commit 的原因见 `SOURCE_PROVENANCE.md`。
