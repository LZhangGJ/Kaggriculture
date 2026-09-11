# S9 本地运行与复核

本目录包含经营计划、浅决策树及动态执行入口。不是独立 Kaggle 提交包；不要把本机耗时直接当作官方设备时限结论。只有 `ACCEPTANCE.json` 和 `deployment_v1/official.json` 实际完成后才可称验收通过。

## 运行环境

Windows 项目目录：`E:\ai_coding\kaggle\kaggriculture`。
使用 WSL Ubuntu-24.04；Python 为：

```text
/mnt/e/ai_coding/kaggle/kaggriculture/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/.venv-bench/bin/python
```

在 WSL 中将本目录加入 `sys.path`，然后：

```python
from local_agent import Agent
player = Agent()
player.reset()  # 开始新局时调用
action = player.agent(observation, configuration=None)
```

`observation` 为官方当步观察，不是完整私有模拟器状态。每个进程只创建一个 Agent 实例；实例内部可区分座位，但不同实例共用动态库静态状态，不应在同一进程交叉运行多个实例。两边都使用本入口时可在同一实例中按各自座位调用，或者每边使用独立进程。

首次作出不同后缀选择后锁定，之后不再次切换。宏观计划按当天真实状态安装，不替换农场、不移植理想前缀，不播放工人坐标动作带。当前无法生成的宏观动作显式回退 KEEP；这是兼容性限制，不等于承诺全部兑现。

## 依赖及文件

- `local_agent.py`：观察输入入口；复用项目内冻结 DP27 观察打包和队友可见状态历史特征。
- `deployment_v1/agent.so`：Linux x86-64 C++ 动态执行器和树。
- `deployment_v1/compact_policy.json`：实际使用的计划、树、原池索引。
- `selected_policy.json`：选型冻结文件及完整来源计划，不能用最终测试重新挑选后覆盖。
- `RULES_ZH.md` / `FEATURE_SCHEMA.json`：生成后的中文树规则及146维输入定义。
- `RESULT_ZH.md`：最终真实胜率，不是事后 Oracle。
- `switch_learning_v1/final_rescue_harm.json`：相同种子/座位下实际救回与伤害。
- `deployment_v1/build.json`：源代码和二进制 SHA；`official.json`：官方逐步一致性与入口时延。

依赖仍位于原工作区；迁移机器时须连同源码、规则引擎、观察打包及特征文件迁移并重建，不能只复制 `local_agent.py`。不需要 GPU、JAX、PyTorch 或在线搜索。

## 实验复核

训练与选型均已按输入 SHA 冻结。先读 `README_ZH.md` 的最新状态，不重复启动完成的阶段，不在现有输出目录覆盖构建。

```text
python test_analysis.py
python switch_learning.py collect_select
python switch_learning.py select
python switch_learning.py final_test
python verify.py build
python verify.py official
python verify.py report
```

这些是阶段命令，不是建议全部重跑。`build` 仅允许首次创建部署目录；采样阶段可凭完整 SHA 收据续跑。C++ 最多16线程、只运行一个重型实验。最终100个seed × 双座位 × 七实时对手，每个Agent1,400局；不能把两个座位当两个独立随机seed。

本轮没有 Kaggle 提交、Git 推送、替换原最佳，亦不恢复旧搜索目标。
