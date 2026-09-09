# T2R1 ServiceAligned Public提交归档

- 官方提交：[56114689](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114689)。
- 2026-09-09 14:46:12日本时间确认：**COMPLETE**，错误信息为空。
- [官方Validation 107025900](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114689&episodeId=107025900)：720帧、双方DONE，1438个动作与上传包完全一致。
- 上传仅1次，策略/参数没有修改。为兼容平台重编译后，完整复现七对手1400局＋710的200局，无动作哈希/现金/结果差异。
- 本地官方文件入口6局通过；最大动作调用82.959ms，无超过默认1秒。本地速度不等于官方硬件保证。
- 这是T2-R1，不是之前的T1修复版。
- 包SHA256：`5c689499e360df9ec61e944457a4414b7199665846aa4fe07e22c6f6c39525bf`。
- 原测试binary：`ead6b00d982e798fa570756859304ad40d77aa9a6a278e0ca2d3c1a9140f9b44`。
- 提交binary：`aa7f397c4f04d20014732b5f7279281491f7bc57c96797d12e29c565ad99bb2e`。

`submission.tar.gz`为实际上传原件；`submission_record.json`和`upload_receipt.json`为上传回执；`latest_status.json`为官方状态；`RELEASE_ACCEPTANCE.json`为汇总验收；`CLOUD_ACTION_IDENTITY.json`为云端动作验同记录。

完整源码、构建/回归程序和官方Replay位于`../t2_r1_r2_public_20260909_v1/R1`。本快照不重复复制大录像，录像路径和SHA256见`OFFICIAL_VALIDATION_REPLAY_RECEIPT.json`。

**勿再次上传本包。** 当前初始600.0分不代表天梯稳定实力。
