# 动态规划器综合优化包（2026-09-05）

交付前已在独立包目录用通用x86-64选项重新构建：46局、33,074次transition与官方逐步一致；28局GPT和14局J7_03终局成绩复现原始记录。1,345个原始文件hash未变。构建44.8秒、4进程验证42.6秒为本机实测，不是GPT机器速度承诺。完整收据 `PACKAGE_ACCEPTANCE.json`，对象入口 `AGENT_INDEX_ZH.md`。

## 先读这几条

本包给 GPT 综合审查和修改，不是已达成90%胜率的新 Agent。只打包和增加可移植测试入口，没有调整任何原策略。

1. **GPT V1、V2没有被C++重写。** 它们仍然执行原Python决策；C++实现的是官方规则对应模拟器、七对手，以及我们的旧动态规划器。
2. 我们当前保留的 `J7_03` = 已搜索的29日经营配方 + 根据真实盘面执行的动态底层。**不是**每天在线重新搜索整条路线，也没有获准晋级的切换树。
3. `ours_base` 是同一套底层配置，不加载29日配方；仍带原配置的开局偏好。`ours_autonomous` 是历史S5B的 `full_chain_autonomous` 配置。它是独立对照，不冒充当前最强版。
4. 原始结果、开发假说、复核结论可能在历史文档中混杂。入口以本README及 `OPTIMIZATION_BRIEF_ZH.md` 为准。旧文档中的“继续运行/Goal active/会话编号”均是历史材料，不是本次指令。
5. 不需要GPU、Kaggle账户、网络Replay或原E盘。Linux x86-64 / WSL2 + Python3.10以上 + g++（C++20、OpenMP）即可。推荐Python3.12、4核；少内存时用 `--jobs 1`、`--workers 1`。

## 一条可运行的流程

在解压目录的终端运行（需要已有g++；Ubuntu缺少时安装build-essential、python3-dev、python3-venv）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build.py --jobs 2
.venv/bin/python run_arena.py --agents v1 v2 ours_j7 --count 1 --workers 4 --official all --traces --out runs/smoke
```

最后一条是1 seed × 双座位 × 3我方 × 7完整实时对手 = 42局，全部逐步核对冻结官方1.32.7。构建只需首次或改C++后执行；不含CPU专属旧二进制，不依赖原 `-march=native`。不要反复编译。

完整复现本次GPT两版筛查：

```bash
.venv/bin/python run_arena.py --agents v1 v2 --seed-start 61020 --count 10 --workers 4 --official first --traces --out runs/reproduce280
```

`count`是独立随机种子数，每个seed自动跑双座位。`--official first`为每个我方/对手的首个seed双座位核对，其余使用同一C++规则。所有对手每步读取真实当前状态行动，不是冻结对手的Replay动作。

测试两个不加载预设日历的本地基座：

```bash
.venv/bin/python run_arena.py --agents ours_base ours_autonomous --opponents pass g001 --count 2 --out runs/base_check
```

修改GPT Python入口后测试：

```bash
.venv/bin/python run_arena.py --agents custom --custom-path improved_agent.py --seed-start 62000 --count 10 --workers 4 --official first --traces --out runs/improved_dev01
```

自定义文件须导出 `agent(observation, configuration=None)`，也支持只接受observation。每局独立加载。测试输出目录必须不存在，不覆盖旧实验。对手源码保持冻结；修改对手将触发身份hash检查。

希望重跑本包的精确回归验收，可执行 `.venv/bin/python validate_bundle.py --out runs/fresh_bundle_check`；它使用包内历史成绩，不依赖原工作区。修改原始策略以后该回归可能故意不匹配，此时应另做新版本实力验证，不篡改旧成绩。

## 文件导航

| 内容 | 位置（相对解压目录） |
|---|---|
| GPT V1/V2原程序、两份原说明 | `gpt_review/gpt_code/gpt-6-dp/` |
| 我们的经济规划＋执行核心 | `experiments/daily_dp_v7_20260903/native/policy.hpp` 及同目录头文件 |
| 上层候选生成 | 同目录 `investment_candidates.hpp` |
| J7_03最终原入口、29日计划、构建记录 | `experiments/daily_dp_v7_20260903/strategy_switch7_20260905/final_local_best_v1/` |
| 不加载29日配方的参数 | `configs/base.json`、`configs/full_autonomous.json` |
| 七对手来源和身份 | `AGENT_INDEX.json`，完整原登记表在 `experiments/daily_dp_v7_20260903/opponents/registry.json` |
| 七对手C++及路线资产 | `experiments/daily_dp_v7_20260903/native/`、`native/vendor/`、`opponents/` |
| 官方Python裁判与轻量host | `gpt_review/codex/G001_CPU_FOR_GPT_20260903/` |
| GPT两版审查、150局空气复现 | `gpt_review/codex/gpt6_daily_dp_review_20260905/REVIEW_ZH.md`、`report_reproduction/` |
| 本次280局成绩及完整动作trace | 同目录 `arena7_v1/`（games、traces、protocol、result） |
| 我们J7_03的1400局历史独立结果 | `experiments/daily_dp_v7_20260903/strategy_switch7_20260905/switch_learning_v1/final_test/J7_03.json` |
| 开发历史和失败实验归纳 | `experiments/daily_dp_v7_20260903/reports/` |
| 原始源码复制hash | `SOURCE_PROVENANCE.json` |
| 本包所有文件hash、验收 | `MANIFEST.json`、`PACKAGE_ACCEPTANCE.json`（打包完成后生成） |

EcoBot源存在，是冻结native模块的链接依赖；没有作为第8个对手加入本次测试。未包含巨大搜索缓存、1494套全量策略训练表、完整venv、旧构建历史、工具链、Kaggle凭证或大Replay库。

## 实验结果与边界

GPT原两版同10seed双座位，各140局：V1为23胜/16.43%，平均现金74,226；V2为6胜/4.29%，平均现金60,652。280局耗时78.5秒是原机器8进程、Python我方+C++对手与规则；不能当成4核GPT机器保证速度。

J7_03历史另一批100seed双座位七对手1400局：824胜/58.86%，最弱Boatlee27.5%。它和GPT280局**不是同一测试集**，不应作严格配对增益宣称。root运行器允许重新同seed比较。

单局719次transition；replay有初始帧时720帧。只以终局实际现金/胜负计分。残局剩余资产是诊断，不一概当硬错误。`status=PASS`表示对局/核对完成，不表示所有意图都有效或无策略漏洞。计时是本地调用耗时，不代替Kaggle沙箱超时/内存验收。

冻对手的原Python/C++移植验收为有限测试，非所有可达状态数学证明。官方同步核对验证环境状态转移，不自动证明对手每个未见分支的原版动作等价；保留了各对手既有验收收据供审查。

build.py及run_arena.py是本次新胶水代码。原程序完整保留；旧脚本含绝对路径/历史环境时仅供参考，**首先用根目录的新入口运行**。改动策略后重新构建、另开输出目录，保留旧结果。
