# Artifacts omitted from Git

The original `route_clustering_switch_agent_bundle_20260825.tar.gz` archive was 271 MiB, with SHA-256:

```text
4d07d76a435f5560403de30180b6869a8a02bc8f58e14242beba9ef1811bdbe8
```

The package retains a runnable agent, reconstruction source, configuration templates and method documentation. Raw replays, calculation caches, intermediate matrices and full search results are not shipped. The following original artifacts were removed or not copied into the source package.

## Legacy PPO artifacts

The active router does not load any of these artifacts.

| Original path | Size | SHA-256 or explanation |
|---|---:|---|
| `recurrent-route-best.pt` | 4.43 MiB | `ec7946e9c8c5997285713717ac1c09c0f90597d03ec424d6c5a6c6e7f74fd315` |
| `recurrent-route-v8.pt` | 4.43 MiB | `99699293c749f6dc0aa33c09de86e717ae31f1e1f853c5086f6c555ed985aab0` |
| `training_runs/route-ppo-v10-hard-focus/snapshot-008.pt` | 4.43 MiB | Byte-identical to `recurrent-route-best.pt` |
| `training_runs/` | 4.5 MiB | PPO configuration, statistics and duplicate checkpoints |
| `evaluation/` | 9.3 MiB | Legacy PPO evaluations and per-game records |
| `branch-descriptors-v8.pkl`, `opening-mixture-v8.json` | 4.5 MiB | Legacy recurrent selector assets |
| `route-runtime-v8/` | 183 MiB | Legacy PPO route runtime; the active searched router has separate action tapes |

## Regenerable search caches

These support sample-level auditing but are not required for online inference. Reconstruct them from replay inputs using the retained pipeline, writing to an external `OUTPUT_ROOT`.

| Original path under `experiments/macro-route-unified-20260825/` | Size | SHA-256 |
|---|---:|---|
| `intent-11-switch-fine-128seeds-v1.npz` | 384.6 MiB | `bcf82ba68cf2a566b8408ba0e944f437a05170f3ed253b1694ad79438d219947` |
| `intent-175-switch-coarse-8seeds-v1.npz` | 109.4 MiB | `a42494009b8083befa78afacf1c8e708e32e64042a4c8701709277c829144dc3` |
| `macro-distance-v1.f32` | 48.4 MiB | `df1e9c1bb6b275e1071dff506004bf3adc88475ceccd82b2c293902440c57c45` |
| `intent-distance-v1.f32` | 48.4 MiB | `694021c33e6cf91af43da795a6ac9b79043f166a796d86f4bc37b70e9408096a` |
| `macro-distance-input.bin` | 5.0 MiB | `7a15693318fbe39f070e05b3dc9e6f296fe2b0ccbbbed74ddd0a72fc7a4874a5` |
| `intent-distance-input-v1.bin` | 2.5 MiB | `e5f736efd60b824b7c217f4785789ac6f8dff1e24a3d8dc8a6ee21385d07df3e` |
| `bin/` | 144 KiB | Platform-specific Linux/aarch64 output, rebuilt from retained C++ source |

## Earlier experiments, external data and duplicate distributions

| Original path or category | Approximate size | Reason for omission |
|---|---:|---|
| `experiments/global-route-search/` | 25 MiB | Superseded by the unified route-clustering pipeline |
| `experiments/macro-route-audit/` | 5.1 MiB | Older 898-replay audit |
| `experiments/macro-route-audit-top60-20260824/` | 83 MiB | Predecessor of the unified 3,563-seat experiment |
| `experiments/simple-route-trees/` | 3.0 MiB | Earlier router exploration |
| `experiments/teammate-route-meta/` | 67 MiB | Intermediate search and duplicate carriers; final carrier assets are in `runtime/` |
| `data/` | 17 MiB | Kaggle/community notebooks, logs and duplicate submissions |
| `submission_variants/` and multiple archives | About 15 MiB | Duplicate distributions of `main.py` and `runtime/` |
| `share/` and route-atlas ZIP | About 1 MiB | Visualizations regenerable with source data and scripts |
| `fast_kaggriculture/build/` and precompiled `.so` files | About 2.5 MiB | CPython/aarch64-specific builds |

New training does not require restoring old matrices: provide replay manifests and regenerate them. Reproducing the exact historical numerical result does require the corresponding original inputs and environment. Put shared large outputs in dataset storage, release attachments or Git LFS rather than ordinary Git history.

## English publication changes on 2026-09-11

This branch replaces the source package's Chinese method Markdown with [docs/METHOD.md](docs/METHOD.md), and translates the README, maintenance guide and this omissions record. It omits the duplicate Chinese method PDF from this new branch. Both original documents remain available in the source commit `eec409775bc538cf65b3676172ee3ce741ca2469` on `main`.

All existing explanatory SVG charts are retained. Frozen runtime assets, the single-file agent and their trained policy are retained byte-for-byte. The [publication record](../../docs/agent_approaches/PUBLICATION.md) identifies the two small offline packaging corrections separately.
