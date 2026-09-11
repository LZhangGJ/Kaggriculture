# T2R1、T2R2 Public提交与官方取证

两版均完成提交、通过官方Validation；不是只收到上传成功消息。状态确认时间：2026-09-09 14:46:12日本时间。

| 版本 | Submission ID | 官方状态 | Validation Episode | 官方录像 | 与上传包动作验同 |
|---|---|---|---|---|---|
| T2R1 ServiceAligned | [56114689](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114689) | COMPLETE、无错误 | [107025900](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114689&episodeId=107025900) | 720帧、双方DONE | 1438/1438 |
| T2R2 BatchedDelivery | [56114767](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114767) | COMPLETE、无错误 | [107026853](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114767&episodeId=107026853) | 720帧、双方DONE | 1438/1438 |

两版各上传1次；当前600.0为初始分，尚不代表天梯实战强度。未设置最终参赛选择，也未推送Git。

## 没有提交错版本的证据

- 原始源码、包装器、配置哈希与刚完成本地比较的R1/R2一致。R1为用户明确指定的ServiceAligned目录。
- 仅采用之前T2成功提交的通用x86-64/GCC11兼容构建方式，静态链接C++运行库；策略、参数、编译相关安全修复均保留。
- R1先通过5条完整冻结轨迹，R2通过8条。
- 兼容构建后，每版1600局全部复现：七对手1400局＋710的200局。逐局现金、胜负、双方动作哈希、操作计数、overflow均无差异。
- 解压实际上传包后，各6局官方1.32.7文件入口验收通过，无报错，无超过默认1秒动作；本地最大调用分别82.959ms/80.182ms。
- 官方Validation录像下载后，使用其中的真实当前观察再次调用上传包，双方1438个动作全部一致。该核验是提交身份及执行一致性证据，不是新增独立强度样本。

## 原始证据与归档

- [T2R1归档](E:/ai_coding/kaggle/kaggriculture/submission/56114689_t2r1_20260909/README_ZH.md)
- [T2R2归档](E:/ai_coding/kaggle/kaggriculture/submission/56114767_t2r2_20260909/README_ZH.md)
- [构建、验收与取证程序说明](E:/ai_coding/kaggle/kaggriculture/submission/t2_r1_r2_public_20260909_v1/README_ZH.md)

各归档含实际上传`submission.tar.gz`、完整可执行入口、官方上传回执、状态、各项验收JSON和包哈希。完整官方Replay保存在原始取证目录的`R1/official_evidence`、`R2/official_evidence`；不重复复制大录像。

| 版本 | 上传包字节 | 上传包SHA256 |
|---|---:|---|
| R1 | 901170 | `5c689499e360df9ec61e944457a4414b7199665846aa4fe07e22c6f6c39525bf` |
| R2 | 901648 | `b3fe2e770710182d14e66006ed5d27ae10d396ac59497829afc124e13ac870ed` |

包内无Replay、真实未来随机事件、账户凭据、对手身份表、运行时编译或GPU依赖。当前证据不能保证未来所有局面均无Bug，更不能保证金牌，但已证实官方跑到终局且执行的是本地验收版本。
