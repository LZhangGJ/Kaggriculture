# S4W：隔离测试的经济前提复审

2026-09-04。v2原失败、源码和日志保留；正式Agent未改。

诊断程序 `native/diagnose_replacement_fixture.cpp` 复用失败夹具，其实际输出：

```text
selected=0 evaluated=2 eligible=1 feasible=1
KEEP known=1 score=10031 trade=31 wages=0 funding_gap=0
candidate known=1 score=10100 trade=100 wages=0 funding_gap=0
CARE alive=0 fert_available=0
FEED alive=1 fert_available=1
last_day_wool=0 fertilizer=1 feed=0
```

这证明原断言“未成熟、终局前无羊毛，所以不值得维护”漏算了喂养产生的肥料。原夹具下保留小麦预计卖31，喂养后肥料预计卖100，多69。选择FEED不等于强制救动物，更不能据此判定调度器有错。

v3测试改为两项：原场景应考虑有利副产物；提高当前小麦的机会售价，使其高于肥料收益时，应保留原计划。这里的价格设定仅在合成机制夹具中，不进入运行策略，不是调参。

这是测试前提修正，不是提高胜率的策略改动。只有机制通过，才能继续原4个真实漏喂局的只读可行性/估值审计。
