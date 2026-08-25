# 未纳入 Git 的产物

原始归档 `route_clustering_switch_agent_bundle_20260825.tar.gz` 为 271 MiB，
SHA-256：

```text
4d07d76a435f5560403de30180b6869a8a02bc8f58e14242beba9ef1811bdbe8
```

仓库保留可运行 agent、当前方法源码、最终策略和关键评测摘要。以下内容已主动删除
或没有从归档复制到精简目录。

## PPO 遗留物

当前 router 不加载这些文件，因此全部删除：

| 原始路径 | 大小 | SHA-256 / 说明 |
|---|---:|---|
| `recurrent-route-best.pt` | 4.43 MiB | `ec7946e9c8c5997285713717ac1c09c0f90597d03ec424d6c5a6c6e7f74fd315` |
| `recurrent-route-v8.pt` | 4.43 MiB | `99699293c749f6dc0aa33c09de86e717ae31f1e1f853c5086f6c555ed985aab0` |
| `training_runs/route-ppo-v10-hard-focus/snapshot-008.pt` | 4.43 MiB | 与 `recurrent-route-best.pt` 逐字节相同 |
| `training_runs/` | 4.5 MiB | PPO 配置、统计和重复 checkpoint |
| `evaluation/` | 9.3 MiB | 旧 PPO checkpoint 评测与逐局记录 |
| `branch-descriptors-v8.pkl`、`opening-mixture-v8.json` | 4.5 MiB | 旧 recurrent route selector 资产 |
| `route-runtime-v8/` | 183 MiB | 旧 PPO 路线运行库；当前 searched router 使用独立动作带 |

## 当前 router 的大型可再生成缓存

这些文件用于精确重训或逐样本审计，但不参与最终在线推理。Git 中保留了同名
`*-summary-*.json`、最终树、选择结果和留出评测。

| 原始路径（位于 `experiments/macro-route-unified-20260825/`） | 大小 | SHA-256 |
|---|---:|---|
| `intent-11-switch-fine-128seeds-v1.npz` | 384.6 MiB | `bcf82ba68cf2a566b8408ba0e944f437a05170f3ed253b1694ad79438d219947` |
| `intent-175-switch-coarse-8seeds-v1.npz` | 109.4 MiB | `a42494009b8083befa78afacf1c8e708e32e64042a4c8701709277c829144dc3` |
| `macro-distance-v1.f32` | 48.4 MiB | `df1e9c1bb6b275e1071dff506004bf3adc88475ceccd82b2c293902440c57c45` |
| `intent-distance-v1.f32` | 48.4 MiB | `694021c33e6cf91af43da795a6ac9b79043f166a796d86f4bc37b70e9408096a` |
| `macro-distance-input.bin` | 5.0 MiB | `7a15693318fbe39f070e05b3dc9e6f296fe2b0ccbbbed74ddd0a72fc7a4874a5` |
| `intent-distance-input-v1.bin` | 2.5 MiB | `e5f736efd60b824b7c217f4785789ac6f8dff1e24a3d8dc8a6ee21385d07df3e` |
| `bin/` | 144 KiB | Linux/aarch64 编译产物；由保留的 C++ 源码重建 |

## 旧实验、外部数据和重复发布物

| 原始路径/类别 | 约大小 | 删除原因 |
|---|---:|---|
| `experiments/global-route-search/` | 25 MiB | 被统一路线聚类流水线取代 |
| `experiments/macro-route-audit/` | 5.1 MiB | 旧 898 replay 审计 |
| `experiments/macro-route-audit-top60-20260824/` | 83 MiB | 当前 3,563 replay 统一实验的前身 |
| `experiments/simple-route-trees/` | 3.0 MiB | 早期 router 探索 |
| `experiments/teammate-route-meta/` | 67 MiB | 中间搜索和重复 carrier；最终 carrier 已在 `runtime/` |
| `data/` | 17 MiB | Kaggle/社区 notebook、日志和重复 submission，可从原来源重新获取 |
| `submission_variants/`、多个 `.zip`/`.tar.gz` | 约 15 MiB | 最终 `main.py` 与 `runtime/` 的重复发布形式 |
| `share/`、route-atlas ZIP | 约 1 MiB | 可由保留的数据和脚本重新导出的可视化 |
| `fast_kaggriculture/build/`、预编译 `.so` | 约 2.5 MiB | CPython/aarch64 平台相关构建产物 |

如果未来需要完整重训，应从上述 SHA-256 对应的原始归档恢复大矩阵，并放在同名
路径下；不要把它们直接加入普通 Git 历史。需要长期托管时应使用数据集存储、Release
附件或 Git LFS。
