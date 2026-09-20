# 数据、产物与来源

## 609 replay

`data/replays/raw/` 含选定的 609 份完整 Kaggle replay，选择清单为 `data/artifacts/replay-manifest-609.json`。清单记录 episode、目标队伍、分数、玩家座位、字节数和观察/动作对齐语义。本批次覆盖 60 个排行榜目标、56 个实际队伍侧，提取出 752 个目标参赛方路线。

为避免在仅剩约 161GB 的磁盘上重复占用约 20GB，raw replay 和已有源文件使用同文件系统硬链接。它们在本工程中是普通完整文件；删除原目录不会删除这里的文件。复制到其他磁盘或打包时会自然实体化。不要用保留硬链接的增量工具误以为它们是符号链接。

旧日级 receipt 提取器只成功处理其中 177 局，432 局因动作不是非空字符串数组而隔离。完整状态记录在 `data/replays/corpus-selection-609.json`。这不等于宏观路线流水线只用了 177 局：609 路线资产来自直接 replay 提取链。后续若统一 receipt 提取，必须先修输入正规化，再重建日级质量标签。

## 核心中间产物

| 文件 | 含义 |
|---|---|
| `features-609.npz` | replay 宏观特征 |
| `execution-audit-609.npz` | 路线执行质量审计 |
| `intent-distance-*.{bin,f32}` | 聚类距离输入与矩阵 |
| `intent-clusters-609.npz` | 聚类结果 |
| `macro-intent-routes-609.json` | 经营意图路线描述 |
| `route-actions-609.json.zlib` | 完整逐步动作载体 |
| `route-library-609.json` | 411 家族及 245 个选中代表 |
| `round-robin-245x64.npz` | 245 路线、64 seeds、双座互打 |
| `switch-coarse-245x8.npz` | 粗筛切换候选 |
| `switch-fine-26x128.npz` | 精细反事实切换数据，约 675MB |
| `trees-shallow-d2-6.json` | 浅树候选，部署资产来源 |
| `trees-deep-d8-16.json` | 更深树实验，未准入部署 |

产物中的旧绝对路径只是生成时溯源字段，不参与运行。在线资产复制在 `agent/`，`agent/replay_deployment.json` 用 SHA-256 锁定四个文件。

## 强公开对手

`opponents/` 固定七个近期强公开脚本：Thomas、Melon、Demand Preserving、Ahmed v47、Pipe8、Herd Safe 2700、Salemali7 2900。`experiments/run_strong_ab.py` 只读取本地相对路径，对每个 seed 跑双座并以 spawn 进程隔离全局状态。部分公开脚本没有明确许可证；它们只用于本地评测，不应打包进对外发布物。Herd 目录保留原 LICENSE/NOTICE。

## 二进制与环境

- 架构：当前 `.so` 为 Linux AArch64、CPython 3.11。
- Python：`/root/miniforge3/envs/torch-npu/bin/python`。
- 官方环境：该环境中的 `kaggle_environments` / `kaggriculture`。
- 编译器：G++，C++20；高速仿真器用 OpenMP，R1 为单个重模板翻译单元。
- `SOURCE_SHA256.txt` 记录在线资产和 R1 接管文件哈希。

高速仿真器的已编译扩展位于 `fast_kaggriculture/python/fast_kaggriculture/`；其最小重建依赖源码收在 `native_deps/`。R1 的已编译库位于 `policy/r1/agent.so`。
