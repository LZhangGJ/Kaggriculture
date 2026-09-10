# 两份 P16 路线候选的交付证据

开发过程与结论以 [根目录 agent.md §12](../../agent.md#r2p16-route-agents) 为入口。这里保存本轮需要审阅和重新计数的证据，未重跑强度测试或修改策略。

| 内容 | 文件 |
|---|---|
| 两份真实提交的身份、下载时间、原包/成员哈希 | [DOWNLOADS.json](DOWNLOADS.json) |
| 13 对手与 25 种子两种口径、逐对手结果及胜负翻转 | [RESULTS.json](RESULTS.json) |
| 新增 200 局的原始逐局记录 | [latest25/rows.json.gz](latest25/rows.json.gz) |
| 新增比赛事前协议、来源和程序哈希 | [latest25/PROTOCOL.json](latest25/PROTOCOL.json) |
| 新增 200 局的逐帧核验记录 | [latest25/QA.json](latest25/QA.json) |
| 100 组新旧动作变化检查 | [latest25/CAUSAL_AUDIT.json](latest25/CAUSAL_AUDIT.json) |
| 原 11 对手的 220 局复用记录 | [previous11/rows.json.gz](previous11/rows.json.gz)、[QA.json](previous11/QA.json) |
| 采购不足及普通路径回归的 32 局 | [recovery_regression/SUMMARY.json](recovery_regression/SUMMARY.json)、[CAUSAL_AUDIT.json](recovery_regression/CAUSAL_AUDIT.json) |
| 工作链版此前 30 种子完整对照 | [workflow_expanded/SUMMARY.json](workflow_expanded/SUMMARY.json)、[VALIDATED_ROWS.json.gz](workflow_expanded/VALIDATED_ROWS.json.gz) |
| 路线经济阶段对 P16 的历史 880 局摘要 | [economics_stage/POOL_VALIDATION.json](economics_stage/POOL_VALIDATION.json) |
| 从 Git 暂存内容导出的独立副本验收 | [工作链版](HANDOFF_workflow.json)、[采购恢复版](HANDOFF_recovery.json) |

本次交付验收在导出副本内重新编译单元检查程序，工作链版 20+58 项、恢复版 20+58+19 项全部通过；实际交付的策略 `.so` 未重编译。分别使用导出的 `main.py` 和原始 `.so` 重放一个已有完整对局的 719 次决策，全部动作一致，恢复版轨迹含一次真实采购恢复事件。这是搬迁/装载检查，不计入新强度样本。

全部复制文件保留原始字节；超过 256 KB 的结果 JSON 以无损 gzip 入库。`PACKAGE_MANIFEST.json` 记录压缩前后 SHA-256 和本机来源路径。来源路径是历史定位信息，不是克隆后需要存在的路径。新的生成脚本和本说明由 manifest 单独登记。

最新面板以**本地工作链/恢复 agent 为受测方**，两份 Kaggle 原提交作为实时对手。13 对手面板每版本 130 局，25 种子两提交面板每版本 100 局；两面板重叠 40 局，总计 420 局不同比赛。旧阶段的 880、1,380 和回归 32 局另行保留，不能合并成这个 420 局面板的胜率。

完整 replay 和逐日动作审计留在原工作区 `experiments/r2p16_latest_submissions_20260910/runs/latest25/matches` 及各旧实验目录；逐局记录与 QA 文件保留它们的 SHA-256。本次未上传全量 replay、浏览器缓存、Kaggle 凭据、下载包或额外对手二进制。`DOWNLOADS.json` 的包路径因此是本机归档路径。

从仓库根目录运行：

```bash
python3 -B evidence/r2p16_route_agents_20260911/verify.py
```

该命令验证两份 agent 的交付清单、原构建收据对应的全部源文件，以及已归档面板的逐局胜负和汇总计数。它不运行比赛，不重做缺少原 replay 的逐帧核验，也不宣称是 Kaggle 时限认证。
