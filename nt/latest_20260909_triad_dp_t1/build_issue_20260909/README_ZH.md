# T1 重编译分歧：GCC 13.3 原因调查包

## 结论

同一源码、配置和首步观测，本机 GCC 13.3 `-O3` 把新动物的收益预测编译错：算好的后续喂养决策被当作 0，预计产品产量降为 0，导致 Agent 放弃牛羊、改种大量瓜。

独立 C++ 小程序可以复现。**仅在构建时追加 `-fno-ipa-modref`，不改策略源码和参数，已恢复原版 400 局、287,600 步的全部动作，以及 G001 两座位已知终局现金。**

这是恢复发布版行为，不是提升策略强度。原发布 Agent 及其 manifest 位于 [../package](../package)，本次没有覆盖它们，也没有修改正式构建脚本。

## 入口

| 文件 | 用途 |
|---|---|
| [原因报告](ROOT_CAUSE_REPORT_ZH.md) | 排除项、具体错误、独立复现、编译矩阵、400 局验收与边界 |
| [独立 C++ 复现](evidence/diagnosis/ipa_modref_repro.cpp) | 不依赖游戏、Python、对手；正常应输出鹅／牛／羊喂养次数 29/26/28，受影响构建输出 1/1/1 |
| [一键复现程序](reproduce.py) | 临时目录中构建、运行独立例子及真实 T1 首步比较；不覆盖原 Agent |
| [文件校验程序](verify_issue.py) | SHA256、ZIP CRC 与验收记录的完整性检查 |
| [交接命令实际复跑](PORTABLE_REPRO_ACCEPTANCE.json) | 从该目录一键重新编译验证成功，用时约 18.24 秒，新库 SHA256 与已验收诊断库一致 |
| [最初的问题复现 ZIP](first_step_reproducer.zip) | 冻结原 `.so`、错误 `.so`、源码配置、官方首步 fixture、包装器、原始问题证据 |
| [验收通过的诊断库](binaries/agent_gcc13_nomodref.so) | GCC 13.3，原参数追加 `-fno-ipa-modref`；Linux x86-64，不是 Windows DLL |
| [无源码改动的 400 局验收](evidence/diagnosis/o3_nomodref/FULL_TRACE_REGRESSION.json) | 400/400 局一致，287,600 步一致，现金一致 |
| [局部函数对照验收](evidence/diagnosis/animal_noalias/FULL_TRACE_REGRESSION.json) | 只对动物预测函数禁用相关优化，也得到相同 400 局结果 |
| [G001 实时两座位验收](evidence/diagnosis/o3_nomodref/G001_ANCHOR.json) | 均为 113,939 : 99,080 |
| [构建矩阵](evidence/diagnosis/EVIDENCE_INDEX.json) | 编译器版本、原文件哈希、多个诊断变体 |
| [来源清单](COPY_PROVENANCE.json) | 文件来源、哈希；报告只重新定位相对链接，原文保留在 evidence 中 |

**ZIP 内的旧报告来自尚未查明原因的阶段；当前结论以本目录原因报告为准。** 原始机器绝对路径作为历史证据保留，并非运行依赖。

## 队友如何运行

先在仓库根目录执行文件校验，Windows／Linux 的 Python 3 都可以：

```text
python nt/latest_20260909_triad_dp_t1/build_issue_20260909/verify_issue.py
```

独立复现和真实首步比较需要 Linux x86-64（或 WSL2）、`g++` C++20、Python 3；不需要 GPU、pip 安装或下载 Replay。从仓库根目录运行：

```text
python3 nt/latest_20260909_triad_dp_t1/build_issue_20260909/reproduce.py --mode all
```

该程序会：

1. 在新临时目录编译独立小程序，比较 `-O0`、`-O3`、`-O3 -fno-ipa-modref`。
2. 解压冻结首步测试包，核查路径和所有成员哈希。
3. 从冻结 T1 源码重新编译一个 `-O3 -fno-ipa-modref` 诊断库。
4. 让原发布库、旧错误库、新诊断库在各自子进程接收相同官方首步观测和配置，比较完整动作和 debug。
5. 打印结果及临时目录；需要留 JSON 时加 `--out /tmp/t1_issue_result_新的名字.json`，已有文件拒绝覆盖。

只跑独立例子用 `--mode minimal`；只跑真实首步用 `--mode step0`。默认编译器为 `g++`，可通过 `--compiler /path/to/g++` 指定。

在不受影响的编译器上，独立例子的 `-O3` 可能已经正常，这是好事；程序不会要求新编译器也产生错误。ZIP 中的旧错误二进制仍用于展示已经冻结的行为分歧。新编译结果与原版若不同，程序报错，不能据此直接推广使用。

## 在 T1 工程中采用绕过方式

策略构建保持原命令，只追加下列选项：

```text
-fno-ipa-modref
```

完整诊断命令由 `reproduce.py` 输出，构建日志也保存。原 T1 构建脚本没有读取 `CXXFLAGS` 的保证，**不能只设置环境变量就假设已经生效**；正式调整时须检查实际编译命令和 receipt，再做行为验收。

也可通过原包装器显式加载附带诊断库：`Agent(config=完整冻结配置, binary_path=诊断库绝对路径)`。不要仅覆盖 `agent.so` 却继续沿用原 manifest／构建 receipt。随包独立复现的成功也不能代替新环境中的完整 Agent 验收。

## 验收范围

- 400 局来自 T1 与原版 94、710 的 100 个 seed × 双座位 × 两对手；逐步重建当时观测，比较诊断库与原库已经保存的有序动作。
- G001 另有 2 局双方实时决策的已知种子现金复验。
- 附的是逐局结果，不是所有 400 局大轨迹和本地 C++ arena 缓存；一键程序复现独立例子与真实首步，不宣称重新执行了这 400 局。
- 原版对 94 的 38.5%、对 710 的 40% 仍是原版强度，不因恢复编译正确性而改称高胜率。
- 未向 GCC 上游提交 bug，未锁定已公开 bug 编号；没有证明所有 GCC 13 版本或所有 T1 路径均受影响。
- 本次包不含训练缓存、不提交 Kaggle、不替换原 Agent。
