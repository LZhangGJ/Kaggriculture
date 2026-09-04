# 2026-09-04 动态策略生成器、300条路线和完整实验结果

这是本次 **Daily DP V7 / all_intraday_insert** 的真实执行代码与结果，不是旧版 Candidate8 的重复入口。

**先看结论：**生成器能在多个训练 seed 上共同搜索经营调整序列，再用真实盘面动态执行。300条计划交叉测试共28,800局，最均衡的是 **Driz Lo来源第027条（DP27）**，三个对手分别23/32、27/32、26/32。没有单条计划对三者全部达到90%。逐局事后任选赢家的100%覆盖，不等于线上有100%胜率。

## 1. 从哪里开始

| 需求 | 入口 |
| --- | --- |
| 运行、继续搜索、复验 | [tools/run.py](tools/run.py)，命令见下文 |
| 候选如何生成 | [investment_candidates.hpp](engine/native/investment_candidates.hpp) |
| 经营与工人动态执行底座 | [policy.hpp](engine/native/policy.hpp) |
| 多seed Beam搜索、顺序执行 | [search.cpp](engine/search8_multiseed/search.cpp) |
| C++官方规则实现 | [simulator.cpp](engine/native/vendor/simulator.cpp) |
| 对手路线及状态修复 | [native_teammate.cpp](engine/native/vendor/native_teammate.cpp) |
| 300条完整宏观计划、排名与各对手统计 | [plans300.json](data/plans300.json) |
| 全实验索引、结果边界 | [EXPERIMENT_INDEX_ZH.md](EXPERIMENT_INDEX_ZH.md) |
| 最新交叉结果 | [RESULT_ZH.md](reports/search29_cross300/RESULT_ZH.md)、[CROSS300.csv](reports/search29_cross300/CROSS300.csv) |
| 已提交DP27 | [submission.tar.gz](submission/dp27/submission.tar.gz)、[main.py](submission/dp27/main.py) |
| 原始证据分卷索引 | [evidence/index.json](evidence/index.json) |
| 包的验收与校验和 | [HANDOFF_ACCEPTANCE.json](HANDOFF_ACCEPTANCE.json)、[MANIFEST.json](MANIFEST.json) |

不需要D盘Replay目录，不需要原本E盘工作区，不需要GPU，也不依赖旧的已编译`.so`。
不上传venv、编译对象缓存、下载工具链或重复源码快照。保留程序、必要输入、完整执行结果和报告。
原始结果无损ZIP分卷，含成功及失败证据；每个成员有SHA256并已实际解压校验。

## 2. 安装与第一条运行命令

在 **Linux / WSL2** 中进入本目录。已验证Ubuntu24.04、Python3.12、GCC13.3、pybind11 2.13.6。该环境是本地搜索环境；DP27提交用的独立二进制另见第6节。

```bash
# 仅缺少编译器/开发头文件时安装；已有环境可以略过。
sudo apt-get install g++ python3-dev python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/run.py build --jobs 2
.venv/bin/python tools/run.py smoke --threads 16 --out runs/smoke
```

`smoke`不是只测试能导入：它重新执行DP27全部96局，比较逐局完整状态/动作哈希及终盘现金；重新生成完整首日68候选前沿；再用冻结官方1.32.7逐步复核三对手双座位6局。
所有原始生产源码保持不变，仅新增相对路径打包和调用入口。`build`约47秒是本机实测，不承诺其他机器相同。

已经存在的输出目录会被拒绝覆盖。重跑改用新`--out`；重编译改用新`--build-dir`，后续所有命令传同一个目录。

## 3. 生成新拳法

```bash
# 原来的8节点（0基日），默认4个训练seed、双座位。
.venv/bin/python tools/run.py search --target "Driz Lo" --days 8 --width 4 --count 20 --out runs/driz8

# 最新Day0–28，共29节点、Beam12，尝试选100条多样化训练全胜计划。
.venv/bin/python tools/run.py search --target "Driz Lo" --days 29 --width 12 --count 100 --out runs/driz29
```

`--target`接受`OceanMix`、`Driz Lo`、`QQ Farming`，或者 [frozen_inputs.json](data/frozen_inputs.json) 中的完整route_id。
包内保留153条冻结对手输入（其中152条竞争性路线、1条空夹具），但**本次搜索与交叉结果只覆盖上述三个代表**，不能宣称已经测遍153条。
若目标不能找到足量训练全胜/不同行为计划，实际数量会少于`--count`，不会伪造获胜路线凑数。

候选来自真实盘面，没有固定“每个节点321条”上限：

- `KEEP`：沿用底座当前规划；不是全场停止。
- `DEFER_NEW`：撤回尚未投入地块的新项目。
- `REBUILD_NEW`：在尚未投入地块重建新的产业组合。
- `REPLACE_NEW`：替换部分拟新增项目。
- `EXPAND_NEW`：利用计划内空闲地块增加项目。

8个产业为5种作物和3种动物。候选保护已有活资产、执行可行性检查，不是任意拆掉现有牛羊。完整语法和数量范围以源码为准。8训练世界合并语义候选后，共同评估同一条计划，不能每个seed各选答案。
每层对“此前前缀＋本次候选＋后续底座”续跑到终局，按训练胜场、分差等排序，再继续扩展保留的前缀。有限Beam不是全局最优保证。
输出保存全部阶段前沿、终局Beam、训练获胜目录、按原语义距离筛选且实际行为去重的计划。

## 4. 换seed和交叉测试

```bash
# 重现此次300×3×16seed×双座位（全部重算，无原机缓存依赖）。
.venv/bin/python tools/run.py cross --plans data/plans300.json --target all --seed-start 20264501 --seed-count 16 --out runs/cross300

# 对刚生成的计划使用未参与原实验的新seed。
.venv/bin/python tools/run.py cross --plans runs/driz29/selected.json --target all --seed-start 20265001 --seed-count 16 --out runs/driz29_newseeds

# 生成一局DP27完整动作/现金轨迹。
.venv/bin/python tools/run.py replay --plan DP27 --target OceanMix --seed 20264501 --seat 0 --out runs/dp27_ocean
```

此次原交叉测试28,800局中复用9,600局，新增19,200局耗时555秒；上述重算命令不复用，所以不能直接承诺555秒。
默认20264501–20264516已经被用于比较300条及选择DP27，**不是DP27选定后的独立认证库**。今后选版须另外冻结新seed。
训练seed默认20264001–20264004。命令会拒绝训练与评测seed重叠。
所有搜索/批量运行最多16线程；更少CPU用`--threads 4`等。没有Kaggle调用，不会自动提交。

## 5. 原始结果与校验

```bash
# 校验Git包本身；--deep还会解压校验全部原始结果成员，不落盘。
.venv/bin/python tools/run.py verify --deep

# 需要逐stage、逐局分析时再解压，约5.2GiB，不必日常全量解压。
.venv/bin/python tools/run.py unpack --out unpacked
```

原六组脚本在证据中原样保留，是历史审计资料，包含原工作区绝对路径；**直接运行用新tools/run.py**。
`unpacked/`保持原六组目录层级，原报告的相对链接在这里可查。
`tools/package_original_evidence.py`和`trim_original_evidence.py`仅记录打包过程，不是队友运行入口。
`runs/`、`build/`、`unpacked/`均不入Git。

## 6. DP27：固定宏观干预，动态执行

DP27是**我们从Driz Lo来源候选池生成的第27条**，不是Driz Lo作者原Agent。
固定的是Day0–28宏观干预序列；日常经营规划、单位分配和每步原子动作仍由当前状态驱动。某天该干预不再可行时显式回退KEEP，并记录`missing_recipe`。
它不在线Beam搜索、不读取真实未来seed、不识别对手身份、不读取对手私有库存，也不是回放719条固定动作。

可提交压缩包只有根目录`main.py`与`agent.so`：Python处理观察输入，C++执行原动态Controller。C ABI不依赖固定CPython版本；x86-64通用指令、静态C++运行库、GLIBC最低2.34。本地搜索扩展需自行编译，不能拿提交用`.so`替代搜索模块。

提交前验收：96局全部719动作一致；6局官方完整状态一致；另2局无`__file__`加载及跨局重置通过。
本地单进程官方复核的最大单步约28.8ms，只代表本地机器。
Kaggle Submission **56011202** 已`COMPLETE`；714.9只是2026-09-04上传初期查询快照，不是稳定天梯实力估计。

## 7. 结果不能越界解读

1. 对手是历史冻结路线加已有修复，不等同于作者最新完整动态Agent。
2. 32局是16个seed双座位；不同计划共享同批seed，样本相关。
3. 训练全胜、单条验证胜率、事后任意赢家覆盖是三种不同指标。
4. DP27均衡结果76/96，不是全对手90%，更不是天梯保证。
5. 本包不训练在线路由器，不恢复此前暂停目标，也没有把单seed最优答案写进部署策略。
6. 编译对象可重建；原源码、配置、结果、失败证据和原校验口径不修改。各外部Agent和官方源码保持其原归属，本包不重新授予第三方代码许可证。
