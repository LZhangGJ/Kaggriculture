# T2R2 BatchedDelivery Public提交归档

- 官方提交：[56114767](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114767)。
- 2026-09-09 14:46:12日本时间确认：**COMPLETE**，错误信息为空。
- [官方Validation 107026853](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114767&episodeId=107026853)：720帧、双方DONE，1438个动作与上传包完全一致。
- 上传仅1次，策略/参数没有修改。为兼容平台重编译后，完整复现七对手1400局＋710的200局，无动作哈希/现金/结果差异。
- 本地官方文件入口6局通过；最大动作调用80.182ms，无超过默认1秒。本地速度不等于官方硬件保证。
- 包SHA256：`b3fe2e770710182d14e66006ed5d27ae10d396ac59497829afc124e13ac870ed`。
- 原测试binary：`877f196113692722edc8a5b3e30d2d3dd1d4700b0772c13f1e3f2ad7a9ac0ef2`。
- 提交binary：`13388459309550c43cdadc1f274e50b274e8d61e62d5f4e799c86bb51943c663`。

`submission.tar.gz`为实际上传原件；`submission_record.json`和`upload_receipt.json`为上传回执；`latest_status.json`为官方状态；`RELEASE_ACCEPTANCE.json`为汇总验收；`CLOUD_ACTION_IDENTITY.json`为云端动作验同记录。

完整源码、构建/回归程序和官方Replay位于`../t2_r1_r2_public_20260909_v1/R2`。本快照不重复复制大录像，录像路径和SHA256见`OFFICIAL_VALIDATION_REPLAY_RECEIPT.json`。

**勿再次上传本包。** 当前初始600.0分不代表天梯稳定实力。
