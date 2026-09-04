# 本次实验索引

所有实验采用同一冻结C++规则及动态执行底座。按下面顺序阅读，不要混淆不同评测seed。

| 阶段 | 做了什么 | 原报告 |
| --- | --- | --- |
| search8_multiseed | 8节点、Beam4、4训练seed双座位；每对手先选冠军 | [结果](reports/search8_multiseed/RESULT_ZH.md) |
| search8_top20_multiseed | 从获胜前沿选20条不同计划，重新做多seed验证 | [结果](reports/search8_top20_multiseed/RESULT_ZH.md) |
| search29_multiseed | Day0–28共29节点、Beam4；与8节点同库对照 | [结果](reports/search29_multiseed/RESULT_ZH.md) |
| search29_top20_multiseed | 逐日搜索各取20条，与8节点20条同库比较 | [结果](reports/search29_top20_multiseed/RESULT_ZH.md) |
| search29_beam12_top100 | Beam12、每来源取100条；与Beam4前20同库比较 | [结果](reports/search29_beam12_top100/RESULT_ZH.md) |
| search29_cross300 | 全部300条×三个对手×32局，保留逐局结果 | [结果](reports/search29_cross300/RESULT_ZH.md)、[CSV](reports/search29_cross300/CROSS300.csv) |

## 核心结果

- 逐日Beam12三目标搜索总计1,332.6秒，249,560次后缀续跑、122,949,160次环境转移。
- 每目标选择100条训练全胜、训练实际轨迹不同的计划；并非100个独立产业家族。
- 完整候选未裁成321上限；Beam12单层候选合计峰值960。
- 300×3交叉测试共28,800局：复用9,600，新算19,200；新增计算及写入555秒。
- 12场跨来源官方逐步复核通过，**不是28,800局全部经官方Python复核**。
- 同时对三对手≥90%的路线数：**0/300**。

| 均衡排名 | 来源/编号 | OceanMix | Driz Lo | QQ Farming | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Driz Lo #027 | 23/32 | 27/32 | 26/32 | 76/96 |
| 2 | Driz Lo #072 | 22/32 | 21/32 | 21/32 | 64/96 |
| 3 | OceanMix #046 | 22/32 | 20/32 | 29/32 | 71/96 |

排名先看最弱对手胜场，再看总胜场及分差，故第二名总胜场可能比第三名少。

逐局允许事后任选赢家，三个对手都能覆盖32/32；这只说明池中有答案，不表示已获得能在线挑对答案的策略。

## 完整证据如何找

[evidence/index.json](evidence/index.json)逐文件标出所在分卷、原相对路径、字节数和SHA256。
解压后重点入口：

```text
search29_beam12_top100/receipts_v1/<route>/stage*.json.gz
search29_beam12_top100/receipts_v1/<route>/selected100.json
search29_beam12_top100/receipts_v1/<route>/w12_???_validation.json
search29_cross300/receipts_v1/games/<source>/<001..100>/<opponent>.json
search29_cross300/receipts_v1/official_by_source/
search29_cross300/receipts_v1/acceptance.json
```

其中`games`共900个单元，每个含32局，完整保留现金、对手现金、分差、seed、座位、动作/状态指纹、回退KEEP数量和宏观计划。
原报告与收据中的绝对路径是来源证明，不要求队友本机有这些路径。

后续建议：先用本包smoke确认环境，再冻结新seed复验DP27及互补候选；不要把这批选择用seed反复当独立证据。不在本次推送中追加新实验或策略修改。
