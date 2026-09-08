# T1 本地重编译行为分歧：原因调查与证据

日期：2026-09-09。工作目录：`E:\ai_coding\kaggle\kaggriculture`。

## 1. 结论

本次“同样源码、同样参数，重编译后开局完全改变”的直接原因已经定位：**本机 GCC 13.3.0 在 `-O3` 下，对动物收益预测函数进行跨函数内存分析／常量传播优化时生成了错误代码。**

错误发生在 `triad::Controller::animal_path` 的“新购买动物、没有现有 Tile”分支。编译器为这个分支生成了 `.constprop.0` 专用版本：动物维护 DP 已经算出后续喂养／照顾决策，但该版本使用了相当于初始零值的决策，忽略计算结果。

所以不是“真正执行时忘了喂动物”，而是更早一步：**规划器错误预判所有新动物都会很快断粮、没有产品收入，于是根本不愿投资动物。** 第 0 步就从牛羊混合开局变成大量买瓜种子，后续强度自然不同。

这不是仅凭更换编译参数的猜测：已抽出不包含游戏、Python、ctypes、对手、随机数的独立 C++ 复现程序，仍出现相同错误；并通过完整策略逐动作回归确认绕过方法有效。

## 2. 先排除了什么

- 原包与本地测试副本的 30 个策略源码／配置文件按原 manifest 校验一致，调查结束后仍一致。
- 首步官方 observation 经同一个 Python 包装器打包，字节哈希相同。
- 同一份配置文件、同一组参数；额外核查了 C++ 对象中的 `base` 和 `live.s`，实际数值也相同。
- 原二进制能在本机 WSL 正常加载，不是 GLIBC 导致无法运行。
- 原版已知 G001 两座位结果可在本机重现；原始动作在官方 1.32.7 与本地 C++ 裁判中逐步一致。
- 差异从第 0 步出现，不是几百步后随机事件或执行误差累积。
- 首步 ASan＋UBSan 构建未报内存／未定义行为错误。注意：这不等于证明整个 T1 的所有路径都不存在其他错误。

## 3. 具体错在哪里

相关源码：

- [动物收益预测](../package/policy/triad.hpp)：`animal_path`。
- [动物维护 DP](../package/policy/animal_service_dp.hpp)：`solve` 填写 `choices[day][hunger][care_bonus]`。

源码语义是：

```cpp
AnimalServiceDP dp;
dp.solve(...);                       // 计算并写入各天的维护选择
auto c = dp.choices[d][hunger][bonus]; // 必须读取刚刚算出的结果
```

GCC 13.3 生成的错误专用函数仍调用了 `solve`，但后续预测循环没有正常使用它写回的喂养选择。对于新购买动物，源码强制第一天喂养／照顾，之后错误地按不喂养推进，预测在 day index 2 的结算后退出，返回 `end=3`。

同一组固定价格输入，直接调用原二进制与错误二进制的这个专用函数，得到：

| 预测量 | 原发布二进制 | 本机错误重编译 |
|---|---:|---:|
| 一只鹅预测期间消耗小麦 | 29 | 1 |
| 一头牛预测期间消耗小麦 | 26 | 1 |
| 一只羊预测期间消耗小麦 | 28 | 1 |
| 鹅的预计产品总量 | 54 | 0 |
| 牛的预计产品总量 | 36 | 0 |
| 羊的预计产品总量 | 34 | 0 |

**此表是固定价格条件下的内部预测函数测试，不是比赛真实终局产量，也不是对维护 DP 的经济最优性证明。**

尤其重要：同一错误二进制中的通用 `animal_path` 入口返回正常结果，只有编译器生成的“新动物”专用入口出错。这解释了为什么仅单独调用通用函数，或者只看源码，很容易漏掉问题。

证据：[专用函数比较](evidence/diagnosis/ANIMAL_CLONE_COMPARISON.json)、[通用函数比较](evidence/diagnosis/ANIMAL_SYMBOL_COMPARISON.json)、[错误专用函数反汇编](evidence/diagnosis/rebuilt_animal_constprop.asm)。

## 4. 独立小程序证明

[ipa_modref_repro.cpp](evidence/diagnosis/ipa_modref_repro.cpp) 只保留：固定价格数组、动物维护 DP、读取 DP 后的喂养次数统计。

它没有游戏状态、对手、Python 接口、类型转换或外部库。

| 编译选项 | 鹅／牛／羊的喂养次数 |
|---|---|
| GCC 13.3 `-O0` | 29 / 26 / 28 |
| GCC 13.3 `-O3` | **1 / 1 / 1** |
| GCC 13.3 `-O3 -fno-ipa-modref` | 29 / 26 / 28 |
| GCC 13.3 `-O3 -fno-strict-aliasing` | 29 / 26 / 28 |

记录：[REDUCED_COMPARISON.json](evidence/diagnosis/REDUCED_COMPARISON.json)。

复现命令（在 WSL 中，以 diagnosis 为当前目录；输出文件使用新的名称，避免覆盖已有证据）：

```bash
g++ -std=c++20 -O3 -ffp-contract=off ipa_modref_repro.cpp -o /tmp/t1_repro_bad
/tmp/t1_repro_bad 0
g++ -std=c++20 -O3 -fno-ipa-modref -ffp-contract=off ipa_modref_repro.cpp -o /tmp/t1_repro_good
/tmp/t1_repro_good 0
```

`ipa-modref` 负责跨函数分析哪些内存被读取／修改；`ipa-strict-aliasing` 控制跨函数应用类型别名规则。两者的作用以 [GCC 官方选项文档](https://gcc.gnu.org/onlinedocs/gcc-13.4.0/gcc/Optimize-Options.html#index-fipa-modref) 为准。本次根因判断来自本地复现和反汇编，不是拿一个网上的相似 bug 套用；**尚未对应到已公开的 GCC Bugzilla 编号，也未向上游提交问题。**

## 5. 定位过程的对照

全部保持同一首步输入、同一策略参数。

| 诊断变体 | 恢复原版首步？ |
|---|---|
| 原 GCC 13.3 `-O3` | 否 |
| `-O0` | 是 |
| `-O1`＋ASan／UBSan | 是，首步无报错 |
| 全局关闭 strict-aliasing | 是 |
| 只对 bridge 关闭 | 否 |
| 只对 `plan` 关闭 | 否 |
| 只对 planner.hpp 关闭 | 否 |
| 只对构造／configure 关闭 | 否 |
| 只对 crop 关闭 | 否 |
| **只对 animal_path 关闭** | **是** |
| 关闭 SLP 向量化 | 否 |
| 关闭循环向量化 | 否 |
| 关闭跨函数 `ipa-modref` | 是 |
| 关闭跨函数 `ipa-strict-aliasing` | 是 |

详见 [EVIDENCE_INDEX.json](evidence/diagnosis/EVIDENCE_INDEX.json)，各变体目录保留构建命令、日志、二进制哈希及首步结果。

## 6. 不只是修首步：400 局完整回归

选取之前原发布 T1 对原版 94、710 的完整实时对战存档：

- 100 个 seed：260909100–260909199。
- 双座位 × 两个对手，共 400 局。
- 每局 719 步，共 287,600 个 T1 决策。
- 诊断版每一步只接收当时的 observation；未来存档只供验证器比较，不传入策略。
- 每次比较完整有序动作，再用存档中原版双方动作推进裁判，核查终局现金。

| 绕过方式 | 匹配局数 | 匹配步数 | 不同动作 |
|---|---:|---:|---:|
| 仅 animal_path 禁用相关别名优化 | 400/400 | 287,600/287,600 | 0 |
| **源码不改，构建追加 `-fno-ipa-modref`** | **400/400** | **287,600/287,600** | **0** |

两次回归分别约 42.05 秒和 42.33 秒；16 个验证进程。此时间不是严格性能对照，也不是 Kaggle 沙箱时限认证。

报告：[局部函数版](evidence/diagnosis/animal_noalias/FULL_TRACE_REGRESSION.json)、[仅编译选项版](evidence/diagnosis/o3_nomodref/FULL_TRACE_REGRESSION.json)。

另用仅编译选项版与实时 G001，重跑已知 seed 260909000 两个座位，均恢复 **113,939 : 99,080**，与原包记录完全相同。[两座位结果](evidence/diagnosis/o3_nomodref/G001_ANCHOR.json)

这证明绕过方法恢复了已验收样本中的原版行为；**不声称覆盖所有可能局面，也不把同一批轨迹的两次复验当成 800 个独立种子。**

## 7. 处理建议与当前边界

最小、已验证的本机构建方案：保留 `-O3`、原有编译参数和全部策略源码，只追加 `-fno-ipa-modref`。不必通过修改投资阈值、删除动物分支或重写规划器来补救这个问题。

诊断构建产物：

`experiments/triad_t1_vs_94_710_20260909_v1/diagnosis/o3_nomodref/agent.so`

SHA256：`9bc7b5cc2ae2fb90cfc0df26bb06c716edd2e384cbf0a49bf510cbd380187f4e`。

当前只完成原因调查和隔离验证：

- 未覆盖原发布 `.so` 或原错误 `.so`。
- 未修改正式 `build_policy.py`、经营策略或参数。
- 未改原交接 ZIP，未推送 Git，未提交 Kaggle。
- 后续正式采用时，应把这一独立复现和逐动作一致性检查加入构建验收；更换编译器也要重跑，不能只检查编译成功。

原版对 94 的 38.5%、对 710 的 40% 胜率结论仍然成立。本次是**恢复原版行为**，不是让 T1 获得了新的策略强度。
