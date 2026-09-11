# 11个冻结对手索引

这些是这轮实际受测的原Python程序，不是重新写的近似C++版本。日期固定2026-09-10，未来Notebook更新不会自动替换本包。必要的多文件、动作表、许可证原样保留；P16自身不使用它们作决策。

| ID / 可调用源码 | 来源 |
|---|---|
| [soil_v219g](opponents/soil_v219g/main.py) | [prvsiyan / Soil](https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-soil-remembers-rain)，Notebook30 |
| [moon_v215](opponents/moon_v215/main.py) | [prvsiyan / Moon](https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-moon-counts-melons)，Notebook33 |
| [flexon_v5](opponents/flexon_v5/main.py) | [flexonafft](https://www.kaggle.com/code/flexonafft/kaggriculture-most-powerfull-route)，Notebook5，完整多文件 |
| [market_smart_v8](opponents/market_smart_v8/main.py) | [tetsutani](https://www.kaggle.com/code/tetsutani/market-smart-farming-kaggriculture)，Notebook8，完整多文件 |
| [nagatakengo_v70](opponents/nagatakengo_v70/main.py) | [nagatakengo](https://www.kaggle.com/code/nagatakengo/kaggriculture)，Notebook70 |
| [aurax_reactive_v1](opponents/aurax_reactive_v1/main.py) | [aurax7 Reactive Router](https://www.kaggle.com/code/aurax7/kaggriculture-reactive-router)，Notebook1 |
| [thomas_955_v2](opponents/thomas_955_v2/main.py) | [thomastschinkel](https://www.kaggle.com/code/thomastschinkel/kaggriculture-95-5-win-rate-via-replay-routing)，Notebook2 |
| [shop0909](opponents/shop0909/main.py) | [yhay81 Shop0909](https://www.kaggle.com/code/yhay81/shop-router-0909) |
| [aurax_shop_v2](opponents/aurax_shop_v2/main.py) | [aurax7 Shop Reactive](https://www.kaggle.com/code/aurax7/kaggriculture-shop-router-reactive-v2)，不同于上面的Reactive Router |
| [seven_turn](opponents/seven_turn/main.py) | [dmitriigluzdov](https://www.kaggle.com/code/dmitriigluzdov/kaggriculture-seven-turn-rescue-best-lb-2800) |
| [ahmed_v27](opponents/ahmed_v27/main.py) | [ahmedberatozer](https://www.kaggle.com/code/ahmedberatozer/notebookfea7d71a35) |

新7个的原下载/静态提取清单见 [原始manifest](development/opponent_freeze/MANIFEST.json)。其“尚未实战”等表述是冻结子任务当时状态，实战请看本包 evidence。

最后4个元数据保存在各自目录的 `kernel-metadata.json`。全部入口都使用 `agent(observation, configuration)`，Nagatakengo第二参数不可省略。

`run.py`对每局清理Agent目录所属模块、重新导入并独立创建我方上下文；不能把多个叫policy/main的模块混用。对手只拿自身当前官方可见观测，双方先取同一pre-step状态再同时结算，不会看到对面当步未执行订单。
