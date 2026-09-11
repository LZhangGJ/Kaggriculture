# P16：冻结策略、开发经过与独立复验交接

这是 **2026-09-10 的 R2P16 全量开局供给先验版**，不是旧 R2，不是半量 P16，也不是已经达到90%的发布冠军。它是规则自主经营＋动态C++执行，不依赖高手日历，不读取对手程序身份或真实未来。**本次只推送Git，没有提交Kaggle。**

## 1. 先看结果

11个冻结公开原程序、相同种子、双座位、完整719步实时对战。平均胜率是逐对手等权平均；平局单列，不计半胜。

| 同批比较范围 | 原R2 | P12 | P16 |
|---|---:|---:|---:|
| 100开发种子，每版2200局 | 1501 / 68.23% | 1723 / 78.32% | 1790 / 81.36% |
| 新32种子第一批，每版704局 | 558 / 79.26% | 547 / 77.70% | 548 / 77.84% |
| 新32种子第二批，每版704局 | 491 / 69.74% | 501 / 71.16% | 567 / 80.54% |
| **共同64个新种子，每版1408局** | **1049 / 74.50%** | **1048 / 74.43%** | **1115 / 79.19%** |

P16共同64种子为1115胜、291负、2平、0运行异常。相对原版救回161场、丢掉95场原胜局，净增66胜，胜率 **+4.69个百分点**；按seed整块重采样的95%区间为 **[-2.27, +11.65]个百分点**，仍包含0。**正向信号，但还不能保证稳定提升，更不能称为逐对手90%。** 对手里有很多近亲公开路线，也不能把1408局当成1408个完全独立样本。

最重要的报告：[共同64种子公平比较](evidence/new32_block2/COMMON64_ZH.md)，逐对手/现金/配对统计在同目录 [COMMON64.json](evidence/new32_block2/COMMON64.json)。[开发全量报告](development/candidates/candidate_r2p16/FULL_DEVELOPMENT_ZH.md)不能替代独立复验。

P12另外有更早的32新种子，只测了P12/原版，没有P16。因此其96种子累计成绩单独列出，不能混进三版公平对照。最终保留的 `2609130000..2609130099` 在本次交接时未使用。

## 2. 文件索引

| 内容 | 入口 |
|---|---|
| **可调用P16入口** | [main.py](main.py)：`agent(obs, configuration)` / 每局 `create_agent()` |
| 冻结完整策略源码 | [policy/](policy/)；包括C++底层执行器和其内嵌模拟器 |
| **实际P16二进制** | [policy/startupsupply2.so](policy/startupsupply2.so) |
| 完整38项运行配置 | [policy/config.json](policy/config.json) |
| 开局供给先验 | [startup_supply.hpp](policy/startup_supply.hpp) |
| 每日估值、计划与条件前瞻 | [triad.hpp](policy/triad.hpp)、[search.hpp](policy/search.hpp) |
| P9/P10/P11功能 | [短卖出时点](policy/local_sale_timing.hpp)、[有限作物施肥](policy/finite_fertilizer.hpp)、[公开作物时钟](policy/observed_crop_clock.hpp) |
| **单命令重建及单测** | [build.py](build.py)、[tests/test_startup_supply.cpp](tests/test_startup_supply.cpp) |
| **可搬迁实时评测** | [run.py](run.py)、[POOL.json](POOL.json)、[对手来源索引](OPPONENTS_ZH.md) |
| 原版/P12对照二进制 | [baselines/](baselines/)；使用相同wrapper和完整配置 |
| 官方冻结裁判 | [referee/](referee/)：官方1.32.7源码/JSON/seed工具＋本地host |
| 开发经过、淘汰理由、下一步 | [DEVELOPMENT_ZH.md](DEVELOPMENT_ZH.md) |
| **8组代表胜负转换复盘** | [evidence/review/](evidence/review/)；16份配对轨迹和现金账分析 |
| 全部三版逐局记录 | [evidence/](evidence/) 下的 `rows.json.gz`，无损压缩，包含失败局 |
| 原始开发脚本、计划和验收 | [development/](development/)；是历史档案，不是独立运行接口 |
| 本次搬迁/重编译验收 | [HANDOFF_ACCEPTANCE.json](HANDOFF_ACCEPTANCE.json) |
| 文件完整性/来源 | [MANIFEST_SHA256.json](MANIFEST_SHA256.json)、[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)、[verify_package.py](verify_package.py) |

**不要直接调用 `policy/agent.py` 的默认入口**：它是原封不动的原始wrapper，默认找 `agent.so`。本包故意不提供这个易混淆名字。`main.py`明确指定P16库和完整配置，不能用一小部分config覆盖后意外退回wrapper的旧默认参数。

## 3. 队友直接使用

运行环境：**Linux x86-64 / WSL2**；交接验收用Ubuntu24.04、Python3.12、g++13。不需要GPU、不需要Kaggle凭据；评测仅用Python标准库。原生Windows Python不能加载Linux `.so`。

从Git根目录进入：

```bash
cd nt/latest_20260910_r2p16
python3 verify_package.py
python3 run.py --reference --repeat-reset --seeds 3 --workers 16 --tag p16_reference
```

第二条运行3个历史种子×双座位×11对手=66局，比较完整联合动作哈希、双方现金、胜负与原始记录；复用同进程的重复检查另列，**不冒充新增独立强度样本**。本次搬迁已实际通过同样66局及重复检查，0差异。

评测原版和P12：

```bash
python3 run.py --version original --reference --workers 16 --tag original_reference
python3 run.py --version p12 --reference --workers 16 --tag p12_reference
```

默认每版1个历史种子、双座位、11对手=22局。`--reference`只允许第二个32种子块中已保存的种子。以下重跑整个第二块（不是新数据）：

```bash
python3 run.py --version p16 --reference --seed-start 2609125000 --seeds 32 --workers 16 --tag p16_block2
```

新种子测试：自行冻结未使用的种子范围，去掉 `--reference`，三版使用相同 `--seed-start/--seeds`，分别运行并比较。**不要直接用最终保留100种子反复调参。** 本包不会自动启动大批量、联网、下载或提交。每个tag必须新建，程序拒绝覆盖已有结果。可用 `--names ahmed_v27,thomas_955_v2` 缩小面板。

加载到自己的arena：

```python
import importlib.util
from pathlib import Path
p = Path("/absolute/path/to/nt/latest_20260910_r2p16/main.py")
spec = importlib.util.spec_from_file_location("p16_entry", p)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
player = entry.create_agent()  # 每局/每个座位独立实例
action = player(observation)  # 只传官方当前可见observation
# 完整局结束后：
player.close()
```

`entry.agent(obs, config)`也支持按座位隔离；step0会重新初始化。环境控制方不能把真实seed、对手私有仓库或未来动作塞进观测。`main.py`不读取评测记录、对手名称或路线文件。

## 4. 从源码重编译

```bash
python3 build.py --unit
python3 run.py --binary build/p16_rebuilt.so --reference --seeds 3 --workers 16 --tag rebuilt_reference
```

默认使用g++13，编译写入 `build/`，**不覆盖冻结库**。保留 `animal_path` 的 `__attribute__((noipa))` 修复：此前GCC13 `-O3` 的常量传播优化曾使动物预测行为错误，不能为了简洁删除。

完整宏开关由 [startupsupply2.BUILD.json](development/candidates/candidate_r2p16/startupsupply2.BUILD.json) 读取，主要开启 `STARTUP_SUPPLY_MODE=2`、`LOCAL_SALE_TIMING=1`、`FINITE_FERTILIZER=1`、`CROP_CLOCK_MODE=1`、`OBSERVE_PUBLIC_TRADES=1`；`MARKET_INTEGRAL`和旧`SALE_CLOCK_MODE`关闭。**不要把所有研究开关一起打开当作P16。**

本次搬迁目录重编译所得SHA256与原冻结库逐字节一致：

`4d7ef55bb79e4b312445a67628dd15609421e8c9afff21e8192a9b1b055f875b`

单测验证首步边界、私有信息隔离、下一天清除、前瞻复制后先验字段不丢失且真正影响估值。重新编译只证明工具链和源码还原；换编译器或修改源码后仍必须做动作级回归。不要重新生成发布manifest掩盖源码变化。

## 5. P16到底改了什么

P16在P12上增加 **开局竞争者可能投资产能的先验**：仅step0用自己的自主规划器，结合对手公开起始农场/现金及规则确定为空的起始私有库存，估计一个假想竞争者的未来供给。它不是对手真实未来的恢复，也没有训练的ML模型。下一天清除该先验，使用真实公开农场和已观察的历史。

P12本身组合了短卖出时点、有限作物施肥、对手公开有限作物收获时钟。详见[开发经过](DEVELOPMENT_ZH.md)。P16不是从零重写规划器。

已复盘代表局中，P16与P12的首日采购规模相同，最早行为差异在step10的牛羊放置次序/布局。这随后可能改变其他经营选择。**不能说已证明它学会了更聪明的宏观开局投资。** 官方杂草和后续商店使用相关随机状态，同seed、不同动作可能造成后续商店不同；必须以多seed整体结果判断，不能把一局救回全部归因于某笔交易。

## 6. 解释边界与省略内容

- 这轮11对手不同于旧7对手；R2旧7对手91.79%不能与本11对手68.23%直接比较。
- 本包评测是**官方Python裁判＋我方C++策略＋对手原Python程序**，不是GPU，也不是全C++对手池。开发2200局约489.6秒、4.49完整局/s（16进程，本机）。
- 开发面板P16最慢单次约0.151秒，0次超过1秒；这不是Kaggle线上硬件/容器时间、ABI或完整schema认证。**本包不是已完成线上兼容验证的submission包。**
- `development/`和历史结果中的原绝对路径、运行会话及“尚未运行/未推送”等文字保留历史事实；不应据此恢复已结束实验。当前状态以本README、COMMON64和本次验收为准。
- [代表复盘](evidence/review/REVIEW_ZH.md)里“原R2”表头实际指当时传入的父版 **P12**，不是最初发布R2。原报告不篡改，按此口径阅读。
- 没有上传工具链、venv、全量大Replay缓存、编译对象或每个淘汰候选二进制。保留小型历史诊断、所有主要结果及16份代表配对轨迹。五份相同的公开对手 `actions.json` 是其必要运行资产，不是P16需要的回放策略。
- 原R2发布资产、旧P11/P12均未覆盖；P16作为独立开发候选交接。下一步先复验与针对失败机制做可回滚消融，不能用留出种子特调。
