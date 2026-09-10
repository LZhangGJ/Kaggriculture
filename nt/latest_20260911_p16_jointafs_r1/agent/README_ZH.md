# P16 JointAFS R1 — 动物—供料—后继投产

**已实现的经济候选分支；不是90%冠军包。**

基底：P16–T3R1–TakeoverMerged R1；38项运行配置不变。新增结构化候选、供料与一次后继状态承诺、共同事件窗口执行比较。

| 范围 | 父版 | 新版 |
|---|---:|---:|
| 3开发种子，66局/版 | 46 | 49 |
| 冻结后5确认种子，110局/版 | 79 | 101 |
| 合计8种子，176局/版 | 125（71.02%） | 150（85.23%） |

救回26场、丢掉1场原胜；确认集22场新增胜利中18场集中于一个seed块。不能用确认集91.82%替代总面板。原型及负向结果全部保留，详见[完整报告](REPORT_ZH.md)。

## 运行

```bash
python3 verify_package.py
python3 check_binary.py
python3 evaluate.py --out runs/local --seeds 3 --workers 3
python3 evaluate.py --out runs/local --seeds 8 --workers 3
```

第二次只补5种子；最终每版11对手各16场，不启动1400局。主入口`main.py::agent`；自己的arena每局调用`main.py::create_agent()`，终局`close()`。

```python
import main
player = main.create_agent()
action = player(observation)  # 只能传当时官方可见observation
player.close()
```

重编译：

```bash
python3 build.py --cxx g++ --out build/local.so
python3 evaluate.py --out runs/local_rebuilt --binary build/local.so --seeds 3 --workers 3
```

构建检查通过才生成新文件，不覆盖发布库；Python标准库，无需GPU。Linux x86-64 / WSL2动态库，原生Windows不可直接加载。本轮GCC14.2编译验证；换工具链要跑检查。

**耗时警告：**并发面板最慢单次2.984秒，最慢案例单进程重复0.450秒且动作哈希不变。没有完成Kaggle线上时间/提交认证，不能保证线上不超时。不要只替换`.so`或用旧wrapper的默认入口。

## 文件索引

- `policy/`：最终源码＋`joint.so`，`main.py`明确选择该库。
- `baselines/merged.so`：实际配对父版；`joint_disabled.so`：关闭新功能的回归对照。
- `opponents/`、`referee/`：11个完整原对手及官方冻结规则。
- `REPORT_ZH.md`、`evidence/RESULTS.json`、`evidence/paired176.csv`：口径、正负转换和逐局数据。
- `evidence/PROTOCOL.json`、`evidence/FREEZE.json`：种子分块、选择版本和冻结记录。
- `evidence/panels/`：各组真实对战结果和半数座位完整轨迹；旧绝对路径只是历史记录。
- `evidence/COMPLETION_TRACE88.json`：88局联合生命周期统计，不冒充全部176局审计。
- `evidence/cash/`：固定动作会计复核；`tools/replay_cash.py`可重算。
- `tests/`、`golden/`、`evidence/build/`：合约与实际重编译、动作回归证据。
- `research/v1`、`research/v3`：淘汰原型，默认不加载。
- `JointAFS_R1.patch`、`SOURCE_PROVENANCE.json`、`RELEASE_BUILD.json`：源码差异与版本身份。

本轮未重跑我方12甜瓜接管、90场旧天梯录像；不要把对11对手的默认开局结果外推成陌生盘面接管证明。
