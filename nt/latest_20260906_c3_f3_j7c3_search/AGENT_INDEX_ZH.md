# Agent 和源码索引

日期：2026-09-06。七对手顺序见 [OPPONENTS.json](OPPONENTS.json)，源码与资产 hashes 见 [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)。

## 我方三个入口

| Agent | 实际运行封装 | 经济规划和执行源码 | 配置 |
|---|---|---|---|
| C3 完整自主 `c3auto` | [c3_policy.cpp](src/c3_policy.cpp)，编译时 `RL_C3_AUTO` | [hybrid.hpp](src/stage/c3auto/hybrid.hpp)、[planner.hpp](src/stage/c3auto/planner.hpp)、[executor/policy.hpp](src/stage/c3auto/executor/policy.hpp) | [c3auto_config.hpp](src/c3auto_config.hpp)、[config.json](src/stage/c3auto/config.json) |
| F3 完整自主 `f3` | [f3_policy.cpp](src/f3_policy.cpp) | [agent.cpp](src/stage/f3/agent.cpp)、[ledger_core.cpp](src/stage/f3/ledger_core.cpp)、[修复账本](src/stage/f3/project_forecast_ledger.hpp) | [f3_config.json](src/f3_config.json) |
| J7 经营参考＋C3 `c3j7` | [c3_policy.cpp](src/c3_policy.cpp)，不设 `RL_C3_AUTO` | [hybrid.hpp](src/stage/c3j7/hybrid.hpp)、[J7 参考数据](src/stage/c3j7/local_reference.hpp)、[executor/policy.hpp](src/stage/c3j7/executor/policy.hpp) | [c3j7_config.hpp](src/c3j7_config.hpp)、[config.json](src/stage/c3j7/config.json) |

`*_plan.so` 是对应底座加 29 节点选择覆盖；本体 `.so` 是 KEEP 基准或历史权重诊断接口。不是六种新 Agent。

## 七个实时对手

| 索引 | 名称 | C++ 入口 | 必需策略资产 / 原始来源 |
|---:|---|---|---|
| 0 | G001 | [runner_module.cpp](src/runner_module.cpp) 中 `bridge::G001`；[native_teammate.cpp](src/stage/arena/vendor/native_teammate.cpp) | [冻结资产](assets/g001.json.zlib)、[原 Python](provenance/opponents/g001.py) |
| 1 | G003 | 同一 G001 通用路由实现，独立资产与局内状态 | [冻结资产](assets/g003.json.zlib)、[原 Python](provenance/opponents/g003.py) |
| 2 | Boatlee V29-R1 | [boatlee_v29.cpp](src/stage/arena/boatlee_v29.cpp) | [冻结资产](assets/boatlee_v29.json.zlib)、[原 Python](provenance/opponents/boatlee_v29.py) |
| 3 | Kaito V58 | [kaito_v58.cpp](src/stage/arena/kaito_v58.cpp) | [冻结资产](assets/kaito_v58.json.zlib)、[原 Python](provenance/opponents/kaito_v58.py) |
| 4 | Lynn V5 | [lynn_v5.cpp](src/stage/arena/lynn_v5.cpp) | [冻结资产](assets/lynn_v5.json.zlib)、[原 Python](provenance/opponents/lynn_v5.py) |
| 5 | yhay81 Six-Day Public-State Fieldbook | [fieldbook_adapter.cpp](src/stage/arena/fieldbook_adapter.cpp)、[原 C++ policy](src/stage/opponents/yhay81_six_day/output/sixday_r4_source/policy.cpp) | 原生源码内置；[Python 外壳](provenance/opponents/yhay81_six_day.py) |
| 6 | yhay81 Three-Day Shop Router | [three_day_adapter.cpp](src/stage/arena/three_day_adapter.cpp)、[原 C++ policy](src/stage/opponents/yhay81_three_day/output/source/policy.cpp) | [内置 tape](src/stage/opponents/yhay81_three_day/output/source/tape.inc)、[Python 外壳](provenance/opponents/yhay81_three_day.py) |

G001/G003 的 C++ 原生实现不是只截取首选路线，必须带上各自整个资产。虽然运行过程是 C++，Python 仍负责启动时解压 JSON 资产并传入池；每一步没有 Python Agent 调用。

共享历史桥接模块还包含 EcoBot 的实现与绑定，因此构建时保留其源文件依赖。**它不在本次七对手列表，也不参与本包的搜索评分或验收胜率。** 不要见到 `ecobot_v7.cpp` 就报告成八对手。

不要把这里冻结的 V29/V58/V5 描述成“当前 Kaggle 最新版本”；本次未访问 Kaggle 更新源码。
