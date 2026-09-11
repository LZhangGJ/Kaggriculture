# FC2A 已知严重问题

状态：**禁止复用、禁止晋级**。

Public 提交 `55662157` 的两个败局（Episode `95784781`、`95787063`）均在第 289 帧开始连续输出全 PASS，直到第 719 帧结束，共 431 帧。

原因不是官方环境或对手造成的普通路线失败，而是 FC2A CPU 融合器在第 288 步首次切换到有内部状态的 PRT 子策略。PRT 之前没有随真实对局影子运行，因此接管时缺少路线状态，未能完成下一日的雇工事务，随后永久空转。

原提交文件和压缩包保留不变，用作故障证据。修复版位于：

`submission/_staging_fc2a_shadow_prt_fix_20260821/main.py`

修复版尚未提交 Public。已知失败种子 `545102` 的官方 1.32.7 双座位复验：

- 原版：13,313，最长连续全 PASS 431 帧；
- 修复版：70,382，最长连续全 PASS 10 帧；
- 修复版第 289 帧恢复卖牛奶、雇 5 人和取小麦事务。

证据回执：

- `experiments/fusion_champion_v1/receipts/fc2a_public_replay_stall_audit_20260821_v1.json`
- `experiments/fusion_champion_v1/receipts/fc2a_shadow_prt_fix_seed545102_20260821_v1.json`
