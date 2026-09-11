# GPT 路线搜索工作包（1.32.7 V2）

这里归档的是 2026-08-15 为“只有 4 核 CPU、没有 GPU 的 GPT 审查者”整理的已验收工作包。

## 为什么 Git 中只放 ZIP

ZIP 已包含 557 个 manifest 输入文件、官方 1.32.7 裁判、JAX 模拟器、路线执行器、Rank13 trace、event bank、正式候选、逐局收据和运行入口。解压后约 194 MB；不在 Git 中同时展开，避免重复存储同一份证据。

## 完整性

```text
文件：Kaggriculture_Simulator_Route_Bundle_1327_20260815_v2.zip
大小：95,290,690 bytes
SHA256：BE63CFE92E6E695D5E49621A1BB6708AE2F2F5A34EDB336F6D719967A5A4D4FB
ZIP 条目：700
解压总量：194,477,824 bytes
```

本次发布前重新执行：

- `scripts/verify_bundle.py`：557/557 manifest 文件匹配，`PASS`；
- ZIP 全条目读取/CRC：700/700，`PASS`。

## 使用

解压后，无需 JAX 的 4 核 CPU 收据检查：

```bash
bash RUN_CPU_FINAL_RECEIPT_CHECK.sh
```

需要了解完整目录、正式搜索入口和结论边界，请先阅读 `BUNDLE_README_ZH.md` 和 `PACKAGE_ACCEPTANCE_ZH.md`。

## 结论边界

该包证明的是：在冻结路线域中完成 GPU 筛选，最终候选在官方 1.32.7 上逐局复验且 JAX/官方终局现金零误差。对手为 NullAgent，所以它不证明强手胜率、BC、RL 或线上排名。
