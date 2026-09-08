# 本包可执行对象索引

根目录 `run_arena.py` 调用以下对象；所有输出都是真实动作，每一步对手重新响应。

| CLI名称 | 对象 | 源码与实际执行 |
|---|---|---|
| v1 | GPT-6 DP V1 | `gpt_review/gpt_code/gpt-6-dp/gpt-6-kaggriculture_daily_dp_agent.py`，Python原版 |
| v2 | GPT-6 DP V2 | 同目录 `gpt-6-kaggriculture_daily_dp_agent-v2.py`，Python原版 |
| ours_j7 | 当前保留J7_03 | `experiments/daily_dp_v7_20260903/strategy_switch7_20260905/final_local_best_v1/local_agent.py`，观测入口＋C++ |
| ours_base | 同配置不加载29日日历 | `configs/base.json`＋`native/policy.hpp`；仍有开局偏好 |
| ours_autonomous | S5B全自主历史配置 | `configs/full_autonomous.json`＋同一C++控制器；不冒充最强版 |
| custom | 待优化的新Python Agent | `--custom-path`指定文件；每局重新加载 |

下表路径默认相对于 `experiments/daily_dp_v7_20260903/`。JSON索引另给出原始程序hash和URL。

| 对手名称 | 原程序 | 本地C++实现与资产 |
|---|---|---|
| g001 | 包根 `research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/main.py` | `native/module.cpp` G001类、`native/vendor/native_teammate.cpp`、`native/g001_frozen.json.zlib` |
| g003 | `opponents/g003/source/main.py` | 同一执行框架＋`native/g003_frozen.json.zlib`，保留356路线 |
| boatlee_v29 | `opponents/boatlee_v29/output/main.py` | `native/boatlee_v29.cpp`＋同名frozen资产 |
| kaito_v58 | `opponents/kaito_v58/output/main.py` | `native/kaito_v58.cpp`＋同名frozen资产 |
| lynn_v5 | `opponents/lynn_v5/output/generated_submission/` | `native/lynn_v5.cpp`＋同名frozen资产 |
| yhay81_six_day | `opponents/yhay81_six_day/output/` | 原C++ `sixday_r4_source/policy.cpp`＋`native/fieldbook_adapter.cpp` |
| yhay81_three_day | `opponents/yhay81_three_day/output/` | 原C++ `source/policy.cpp`＋`native/three_day_adapter.cpp` |

我们的经济规划与执行核心并不只在一个文件中：`policy.hpp`是总入口；`investment_candidates.hpp`生成经营调整；`resource_exchange.hpp`、`idle_task_handoff.hpp`、`crop_delivery.hpp`、`service_recovery.hpp`等是能力开关实现。请以配置中实际开启情况和动作效果测试为准。

原 `agents/agent_v7.py` 只是旧Python参考，不包含之后所有C++改进；不能误选它来代表当前J7_03。
