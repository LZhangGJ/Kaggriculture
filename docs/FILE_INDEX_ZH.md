# 文件索引

`FILE_MANIFEST.json` 给出除 609 个 raw replay 和 177 个日级目录外每个工程文件的相对路径、字节数和 SHA-256；raw replay 的逐局来源与校验在 `data/replays/corpus-selection-609.json`，选择语义在 `data/artifacts/replay-manifest-609.json`。这样既能审计所有文件，又不重复生成一个包含 20GB 文件哈希的大清单。

## 根目录

- `README.md`：工程入口、目录和当前结论。
- `HANDOFF_ZH.md`：最新断点、可信结果、疑似 bug 和下一步。
- `AGENTS.md`：后续 coding agent 必须遵守的主线边界。
- `PROJECT_MANIFEST.json`：机器可读的规模、默认部署和验证状态。
- `SOURCE_SHA256.txt`：在线路线资产及 R1 接管源码/二进制哈希。
- `docs/SESSION_20260920_ZH.md`：本轮（2026-09-20）会话记录：测量纪律修复、opening 与接管重标定、
  更正过的错误清单与下一步优先级。
- `verify_project.py`：轻量结构、自包含依赖、浅树身份和结果检查。

## 在线 agent

- `agent/main.py`：唯一混合入口；构造成熟 replay 路线控制器并在接管前持续喂给 R1 公开历史。
- `agent/replay_deployment.json`：固定 G275、三块地后延迟 1 天、step288 截止及在线资产哈希。
- `agent/teammate_base.py`：成熟路线执行底座与运行期修补逻辑。
- `agent/route_actions.json.zlib`：245 个代表所需的压缩动作载体。
- `agent/route_library.json`：family→route 映射及路线元数据。
- `agent/route_policy.json`：与 `trees-shallow-d2-6.json` 等价的部署浅树。

`policy/r1/agent.py` 负责观察编码、ctypes 调用和 NOOP/PASS 正规化；`bridge.cpp` 暴露原生 R1、外部观察与接管函数；`build.sh` 固化 C++20 和生产宏；`config.json` 是 R1 参数。`triad.hpp` 内含默认关闭的 portfolio 单换/双换实验；其余 `.hpp` 与 `executor/` 是无 opening-template R1 规划、市场、维护、执行修补和 simulator 代码。

## 成熟路线代码

`meta_agent/src/teammate_expanded_routes.py` 执行和修补 replay 路线，`search_route_policy.py` 负责树节点与路线切换，`route_switch_features.py` 定义 147 维特征，`native_teammate_executor.py` 封装全 C++ 互打核。其余模块保留旧路线库、belief、市场管理、树模型与导出工具所需依赖。

`scripts/` 按用途分为：

- 数据与聚类：`analyze_macro_route_library.py`、`audit_macro_execution_quality.py`、`audit_macro_thresholds.py`、`cluster_intended_macro_routes.py`、`merge_macro_route_caches.py`、`build_complete_core_route_families.py`。
- 导出：`export_intent_route_carriers.py`、`export_macro_intent_route_specs.py`、`export_teammate_route_library.py`、`export_single_file_submission.py`、`export_teammate_meta_submission.py`。
- 原生仿真：`run_native_intent_round_robin.py`、`run_native_intent_switch_search.py`、`evaluate_native_route_trees_holdout.py`。
- 树训练与序列：`train_robust_search_route_trees.py`、`train_search_route_trees.py`、`train_direct_route_tree.py`、`search_native_route_tree_sequences.py`、`analyze_compact_switch_search.py`。
- 其余脚本是成熟工程中的路线筛选、历史评测、图表和打包工具；当前主线不调用 Nash 或弱公开对手评测脚本。

## 仿真、对手与实验

`fast_kaggriculture/src/` 是游戏状态转移、市场和原生路线 executor；`python/fast_kaggriculture/` 含 CPython 3.11 扩展；`native_deps/` 是从旧巨型树抽出的最小重建依赖。`fast_kaggriculture/tests/test_rng.py` 是当前已跑的最小确定性检查。

`opponents/*/main.py` 是七个强公开对手入口，Herd 的辅助资产和许可证保留在其目录。`opponents/manifest.json` 记录来源和本地相对入口。

`experiments/run_strong_ab.py` 是通用 192-worker 同 seed 双座 A/B；`run_handoff_scan.py` 扫 R1 接管日；`run_route_opening_scan.py` 扫 opening；`replay_stable_cold_handoff12.py` 是冷接管对照；`smoke_official.py` 是一局完整官方环境烟测。`experiments/results/` 保存原始逐局输出，交接文档指出哪些结果来自错误动态分支。
