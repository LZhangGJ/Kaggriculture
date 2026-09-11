# P16 JointAFS R1 Public 提交与取证

本目录用于用户于 2026-09-10 明确授权的 Kaggriculture Public 提交。

发布门：

- 冻结源码及开发二进制哈希一致；
- GCC 11 通用 x86-64、静态 libstdc++/libgcc 重新构建；
- 对 11 个冻结对手、双座位共 22 局，与开发二进制逐动作及终局完全一致；
- 解压后的提交文件使用官方 1.32.7 环境跑满两局 719 步；
- 动作均未超过官方单步 1 秒上限；
- 只上传一次，状态不确定时仅查询，禁止盲目重试。

AFS 的本地强度结论是：在同一批 11 对手、50 个新种子、双座位的 1,100 局中，
原始胜率为 77.82%；相对 T3R1 TakeoverMerged 救回 125 局，同时损失 66 个旧胜局。
该结果用于说明 AFS 是有价值但有取舍的分支，不代表线上强度或 90% 胜率保证。

## 提交回执

- Submission：[56146577](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56146577)
- 描述：`NT P16 JointAFS R1 - joint animal feed successor planning`
- 上传次数：1
- 上传结果：`COMPLETE`，Public 分数 **1430.3**，错误字段为空
- 发布版与冻结 AFS：22 局、176 项逐动作/终局字段完全一致
- 官方文件验收：2 局均完成 719 步，最慢单步 323.335 ms
- 提交包 SHA256：`c421a416b538df4eb4ad2da2d7598504c52dedd691537ab6fc246c3dec6c8d8d`
