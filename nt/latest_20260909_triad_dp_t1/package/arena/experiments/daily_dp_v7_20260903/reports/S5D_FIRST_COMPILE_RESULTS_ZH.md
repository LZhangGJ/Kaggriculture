# S5D 首次编排 × 可延期补种：完整实战复盘

原目标保持八个实时对手逐一90%以上，未以单局、平均胜率或PASS收益替代。

## 问题与实现

S5C发现维护义务生成并预留后在首次编排丢失；现有候选保护旧工作导致可延期补种挤占喂养。A将公开条件日内后果＋余值用于首次编排；B允许旧收获保留、下一茬补种延期一天。共享、完整自主、完整自主＋换种时机三个背景各KEEP/A/B/A+B。

原递归预测时延不合格，失败版本完整保留。最终本轮采用独立有界预测开关：内部不反复展开明天的完整投资规划，真实逐步协调不关闭；它是近似预测，非无损声明。增量插入只替换成本计算，10,000随机排班完全等价。

## 对战结果

2160完整C++局，10旧开发seed×双座位，每对手20局；540旧控制逐局完全一致，1620新配置账本重复完全一致。8对手为真实闭环，不是冻结动作流。

|配置|PASS现金|g001|g003|boatlee_v29|kaito_v58|lynn_v5|yhay81_six_day|yhay81_three_day|ecobot_v7|八对手平均|综合局/s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|all_intraday_insert|189,894|5/20|0/20|6/20|4/20|11/20|6/20|7/20|20/20|36.875%|111.76|
|all_intraday_insert_first_value|183,728|4/20|0/20|5/20|2/20|6/20|4/20|4/20|20/20|28.125%|13.71|
|all_intraday_insert_replant|187,277|8/20|2/20|8/20|2/20|11/20|7/20|7/20|20/20|40.625%|17.62|
|all_intraday_insert_first_value_replant|178,336|9/20|0/20|6/20|2/20|6/20|9/20|4/20|20/20|35.000%|10.02|
|full_chain_autonomous|163,072|8/20|4/20|2/20|6/20|4/20|2/20|10/20|20/20|35.000%|5.23|
|full_chain_autonomous_first_value|163,443|6/20|4/20|0/20|6/20|4/20|2/20|11/20|20/20|33.125%|2.13|
|full_chain_autonomous_replant|161,533|8/20|4/20|2/20|8/20|4/20|3/20|13/20|20/20|38.750%|1.49|
|full_chain_autonomous_first_value_replant|166,537|6/20|4/20|0/20|6/20|4/20|2/20|13/20|20/20|34.375%|1.77|
|full_chain_autonomous_timing|179,930|2/20|2/20|4/20|4/20|2/20|0/20|1/20|19/20|21.250%|2.65|
|full_chain_autonomous_timing_first_value|179,620|6/20|2/20|4/20|6/20|2/20|0/20|3/20|19/20|26.250%|1.10|
|full_chain_autonomous_timing_replant|181,933|4/20|0/20|4/20|4/20|6/20|3/20|1/20|20/20|26.250%|1.73|
|full_chain_autonomous_timing_first_value_replant|179,457|6/20|2/20|4/20|6/20|2/20|0/20|3/20|20/20|26.875%|1.04|

## 独立效应与长链组合

完整配对胜率、现金、分差及交互 `AB-A-B+KEEP` 见stage acceptance.json；每个seed先合并双座位，再算10seed近似t区间，未校正多重比较。旧开发seed已反复用于选型；不代表独立泛化。

八对手汇总仍以seed为独立单位，不能把同一seed的16局当16个独立样本。

|改动|八对手胜率差pp及近似95%区间|平均现金差|平均分差变化|
|---|---|---:|---:|
|all_intraday_insert_first_value|-8.750 [-19.94, +2.44]|-1455|-3905|
|all_intraday_insert_replant|+3.750 [-2.64, +10.14]|+8|-406|
|all_intraday_insert_first_value_replant|-1.875 [-15.21, +11.46]|+1239|-197|
|full_chain_autonomous_first_value|-1.875 [-6.12, +2.37]|+1000|-207|
|full_chain_autonomous_replant|+3.750 [-0.57, +8.07]|+1532|+2252|
|full_chain_autonomous_first_value_replant|-0.625 [-5.98, +4.73]|+2103|+1599|
|full_chain_autonomous_timing_first_value|+5.000 [-1.60, +11.60]|+1606|-217|
|full_chain_autonomous_timing_replant|+5.000 [+0.89, +9.11]|+866|+2748|
|full_chain_autonomous_timing_first_value_replant|+5.625 [-1.51, +12.76]|+2224|+316|

|改动|对手|胜率差pp|现金差|分差变化及近似95%区间|
|---|---|---:|---:|---|
|all_intraday_insert_first_value|g001|-5.0|+2600|-2902 [-11292, 5489]|
|all_intraday_insert_replant|g001|+15.0|-2665|+417 [-2602, 3437]|
|all_intraday_insert_first_value_replant|g001|+20.0|+5895|+102 [-8078, 8282]|
|all_intraday_insert_first_value|g003|+0.0|-4763|-3784 [-8973, 1405]|
|all_intraday_insert_replant|g003|+10.0|+1591|+435 [-1775, 2644]|
|all_intraday_insert_first_value_replant|g003|+0.0|-5707|+414 [-5518, 6345]|
|all_intraday_insert_first_value|boatlee_v29|-5.0|+3445|-3415 [-13908, 7078]|
|all_intraday_insert_replant|boatlee_v29|+10.0|-118|-116 [-4110, 3879]|
|all_intraday_insert_first_value_replant|boatlee_v29|+0.0|+2687|-2347 [-13735, 9041]|
|all_intraday_insert_first_value|kaito_v58|-10.0|+1787|-7878 [-13757, -1998]|
|all_intraday_insert_replant|kaito_v58|-10.0|+1693|-492 [-1883, 898]|
|all_intraday_insert_first_value_replant|kaito_v58|-10.0|+3991|-4445 [-10669, 1778]|
|all_intraday_insert_first_value|lynn_v5|-25.0|-12913|-5866 [-19578, 7847]|
|all_intraday_insert_replant|lynn_v5|+0.0|-680|-763 [-1847, 321]|
|all_intraday_insert_first_value_replant|lynn_v5|-25.0|-7914|-2572 [-16936, 11792]|
|all_intraday_insert_first_value|yhay81_six_day|-10.0|+1624|+3581 [-7578, 14739]|
|all_intraday_insert_replant|yhay81_six_day|+5.0|+7305|+2250 [-6419, 10919]|
|all_intraday_insert_first_value_replant|yhay81_six_day|+15.0|+5067|+7749 [-4585, 20084]|
|all_intraday_insert_first_value|yhay81_three_day|-15.0|+7269|-6164 [-22257, 9928]|
|all_intraday_insert_replant|yhay81_three_day|+0.0|-856|-312 [-1935, 1310]|
|all_intraday_insert_first_value_replant|yhay81_three_day|-15.0|+12873|+2353 [-12538, 17244]|
|all_intraday_insert_first_value|ecobot_v7|+0.0|-10690|-4812 [-17221, 7597]|
|all_intraday_insert_replant|ecobot_v7|+0.0|-6210|-4665 [-13503, 4173]|
|all_intraday_insert_first_value_replant|ecobot_v7|+0.0|-6983|-2833 [-17384, 11718]|
|full_chain_autonomous_first_value|g001|-10.0|+1467|-889 [-5229, 3452]|
|full_chain_autonomous_replant|g001|+0.0|+389|+584 [-2983, 4150]|
|full_chain_autonomous_first_value_replant|g001|-10.0|+2363|+451 [-4415, 5317]|
|full_chain_autonomous_first_value|g003|+0.0|+823|+358 [-1059, 1774]|
|full_chain_autonomous_replant|g003|+0.0|+55|+1555 [-567, 3676]|
|full_chain_autonomous_first_value_replant|g003|+0.0|+1479|+2221 [-955, 5396]|
|full_chain_autonomous_first_value|boatlee_v29|-10.0|-346|-970 [-3895, 1955]|
|full_chain_autonomous_replant|boatlee_v29|+0.0|-1020|-282 [-2449, 1884]|
|full_chain_autonomous_first_value_replant|boatlee_v29|-10.0|+800|+224 [-3031, 3480]|
|full_chain_autonomous_first_value|kaito_v58|+0.0|+1017|+854 [-895, 2602]|
|full_chain_autonomous_replant|kaito_v58|+10.0|-580|+706 [-2362, 3775]|
|full_chain_autonomous_first_value_replant|kaito_v58|+0.0|+2359|+2761 [-746, 6268]|
|full_chain_autonomous_first_value|lynn_v5|+0.0|+525|+580 [-1310, 2470]|
|full_chain_autonomous_replant|lynn_v5|+0.0|+160|+890 [-749, 2529]|
|full_chain_autonomous_first_value_replant|lynn_v5|+0.0|+1491|+1722 [-451, 3896]|
|full_chain_autonomous_first_value|yhay81_six_day|+0.0|+4380|-109 [-2717, 2499]|
|full_chain_autonomous_replant|yhay81_six_day|+5.0|+10422|+7521 [-1284, 16326]|
|full_chain_autonomous_first_value_replant|yhay81_six_day|+0.0|+5476|+1485 [-2888, 5858]|
|full_chain_autonomous_first_value|yhay81_three_day|+5.0|+74|-806 [-3541, 1928]|
|full_chain_autonomous_replant|yhay81_three_day|+15.0|+790|+4514 [236, 8791]|
|full_chain_autonomous_first_value_replant|yhay81_three_day|+15.0|+2581|+3992 [843, 7141]|
|full_chain_autonomous_first_value|ecobot_v7|+0.0|+60|-671 [-3931, 2589]|
|full_chain_autonomous_replant|ecobot_v7|+0.0|+2040|+2527 [-903, 5957]|
|full_chain_autonomous_first_value_replant|ecobot_v7|+0.0|+277|-62 [-5646, 5521]|
|full_chain_autonomous_timing_first_value|g001|+20.0|+2966|+150 [-3981, 4280]|
|full_chain_autonomous_timing_replant|g001|+10.0|+888|+1378 [-234, 2990]|
|full_chain_autonomous_timing_first_value_replant|g001|+20.0|+3155|+561 [-3612, 4733]|
|full_chain_autonomous_timing_first_value|g003|+0.0|+250|-866 [-8898, 7166]|
|full_chain_autonomous_timing_replant|g003|-10.0|+55|+1574 [-4515, 7664]|
|full_chain_autonomous_timing_first_value_replant|g003|+0.0|+447|-583 [-8605, 7439]|
|full_chain_autonomous_timing_first_value|boatlee_v29|+0.0|+4811|+1640 [-988, 4268]|
|full_chain_autonomous_timing_replant|boatlee_v29|+0.0|+1173|+4817 [-776, 10411]|
|full_chain_autonomous_timing_first_value_replant|boatlee_v29|+0.0|+5040|+1885 [-851, 4620]|
|full_chain_autonomous_timing_first_value|kaito_v58|+10.0|+1060|+1880 [-4262, 8021]|
|full_chain_autonomous_timing_replant|kaito_v58|+0.0|-2621|+1007 [-3505, 5520]|
|full_chain_autonomous_timing_first_value_replant|kaito_v58|+10.0|+1453|+1935 [-4379, 8248]|
|full_chain_autonomous_timing_first_value|lynn_v5|+0.0|+1210|-3780 [-7105, -456]|
|full_chain_autonomous_timing_replant|lynn_v5|+20.0|+7987|+5449 [-1395, 12292]|
|full_chain_autonomous_timing_first_value_replant|lynn_v5|+0.0|+1518|-3289 [-6948, 370]|
|full_chain_autonomous_timing_first_value|yhay81_six_day|+0.0|+1935|+1394 [-2233, 5022]|
|full_chain_autonomous_timing_replant|yhay81_six_day|+15.0|-100|+5301 [1613, 8989]|
|full_chain_autonomous_timing_first_value_replant|yhay81_six_day|+0.0|+2298|+1761 [-1797, 5320]|
|full_chain_autonomous_timing_first_value|yhay81_three_day|+10.0|+714|-1731 [-5144, 1682]|
|full_chain_autonomous_timing_replant|yhay81_three_day|+0.0|-2440|-1300 [-5313, 2712]|
|full_chain_autonomous_timing_first_value_replant|yhay81_three_day|+10.0|+3105|-772 [-4240, 2696]|
|full_chain_autonomous_timing_first_value|ecobot_v7|+0.0|-99|-425 [-3907, 3057]|
|full_chain_autonomous_timing_replant|ecobot_v7|+5.0|+1986|+3756 [-1653, 9165]|
|full_chain_autonomous_timing_first_value_replant|ecobot_v7|+5.0|+777|+1032 [-3323, 5386]|

## 异常与验收边界

|配置|对手|平均无效单位动作|平均逃跑|
|---|---|---:|---:|
|all_intraday_insert_first_value|pass|0.00|0.00|
|all_intraday_insert_first_value|g001|0.00|0.00|
|all_intraday_insert_first_value|g003|0.00|0.00|
|all_intraday_insert_first_value|boatlee_v29|0.00|0.00|
|all_intraday_insert_first_value|kaito_v58|0.00|0.00|
|all_intraday_insert_first_value|lynn_v5|0.00|0.00|
|all_intraday_insert_first_value|yhay81_six_day|0.00|0.00|
|all_intraday_insert_first_value|yhay81_three_day|0.00|0.00|
|all_intraday_insert_first_value|ecobot_v7|0.00|0.00|
|all_intraday_insert_replant|pass|0.00|0.00|
|all_intraday_insert_replant|g001|0.00|0.00|
|all_intraday_insert_replant|g003|0.00|0.00|
|all_intraday_insert_replant|boatlee_v29|0.00|0.00|
|all_intraday_insert_replant|kaito_v58|0.00|0.00|
|all_intraday_insert_replant|lynn_v5|0.00|0.00|
|all_intraday_insert_replant|yhay81_six_day|0.00|0.00|
|all_intraday_insert_replant|yhay81_three_day|0.00|0.00|
|all_intraday_insert_replant|ecobot_v7|0.00|0.00|
|all_intraday_insert_first_value_replant|pass|0.00|0.00|
|all_intraday_insert_first_value_replant|g001|0.00|0.00|
|all_intraday_insert_first_value_replant|g003|0.00|0.00|
|all_intraday_insert_first_value_replant|boatlee_v29|0.00|0.00|
|all_intraday_insert_first_value_replant|kaito_v58|0.00|0.00|
|all_intraday_insert_first_value_replant|lynn_v5|0.00|0.00|
|all_intraday_insert_first_value_replant|yhay81_six_day|0.00|0.00|
|all_intraday_insert_first_value_replant|yhay81_three_day|0.00|0.00|
|all_intraday_insert_first_value_replant|ecobot_v7|0.00|0.00|
|full_chain_autonomous_first_value|pass|0.00|0.00|
|full_chain_autonomous_first_value|g001|0.00|0.00|
|full_chain_autonomous_first_value|g003|0.00|0.00|
|full_chain_autonomous_first_value|boatlee_v29|0.00|0.00|
|full_chain_autonomous_first_value|kaito_v58|0.00|0.00|
|full_chain_autonomous_first_value|lynn_v5|0.00|0.00|
|full_chain_autonomous_first_value|yhay81_six_day|0.00|0.15|
|full_chain_autonomous_first_value|yhay81_three_day|0.00|0.00|
|full_chain_autonomous_first_value|ecobot_v7|0.00|0.00|
|full_chain_autonomous_replant|pass|0.00|0.00|
|full_chain_autonomous_replant|g001|0.00|0.00|
|full_chain_autonomous_replant|g003|0.00|0.00|
|full_chain_autonomous_replant|boatlee_v29|0.00|0.00|
|full_chain_autonomous_replant|kaito_v58|0.00|0.00|
|full_chain_autonomous_replant|lynn_v5|0.00|0.00|
|full_chain_autonomous_replant|yhay81_six_day|0.00|0.00|
|full_chain_autonomous_replant|yhay81_three_day|0.00|0.00|
|full_chain_autonomous_replant|ecobot_v7|0.00|0.00|
|full_chain_autonomous_first_value_replant|pass|0.00|0.00|
|full_chain_autonomous_first_value_replant|g001|0.00|0.00|
|full_chain_autonomous_first_value_replant|g003|0.00|0.00|
|full_chain_autonomous_first_value_replant|boatlee_v29|0.00|0.00|
|full_chain_autonomous_first_value_replant|kaito_v58|0.00|0.20|
|full_chain_autonomous_first_value_replant|lynn_v5|0.00|0.00|
|full_chain_autonomous_first_value_replant|yhay81_six_day|0.00|0.10|
|full_chain_autonomous_first_value_replant|yhay81_three_day|0.00|0.00|
|full_chain_autonomous_first_value_replant|ecobot_v7|0.00|0.00|
|full_chain_autonomous_timing_first_value|pass|0.00|0.00|
|full_chain_autonomous_timing_first_value|g001|0.00|0.00|
|full_chain_autonomous_timing_first_value|g003|0.00|0.00|
|full_chain_autonomous_timing_first_value|boatlee_v29|0.00|0.00|
|full_chain_autonomous_timing_first_value|kaito_v58|0.00|0.00|
|full_chain_autonomous_timing_first_value|lynn_v5|0.00|0.00|
|full_chain_autonomous_timing_first_value|yhay81_six_day|0.00|0.00|
|full_chain_autonomous_timing_first_value|yhay81_three_day|0.00|0.20|
|full_chain_autonomous_timing_first_value|ecobot_v7|0.00|0.00|
|full_chain_autonomous_timing_replant|pass|0.00|0.00|
|full_chain_autonomous_timing_replant|g001|0.00|0.00|
|full_chain_autonomous_timing_replant|g003|0.00|0.00|
|full_chain_autonomous_timing_replant|boatlee_v29|0.00|0.00|
|full_chain_autonomous_timing_replant|kaito_v58|0.00|0.00|
|full_chain_autonomous_timing_replant|lynn_v5|0.00|0.00|
|full_chain_autonomous_timing_replant|yhay81_six_day|0.00|0.00|
|full_chain_autonomous_timing_replant|yhay81_three_day|0.00|0.20|
|full_chain_autonomous_timing_replant|ecobot_v7|0.00|0.00|
|full_chain_autonomous_timing_first_value_replant|pass|0.00|0.00|
|full_chain_autonomous_timing_first_value_replant|g001|0.00|0.00|
|full_chain_autonomous_timing_first_value_replant|g003|0.00|0.00|
|full_chain_autonomous_timing_first_value_replant|boatlee_v29|0.00|0.00|
|full_chain_autonomous_timing_first_value_replant|kaito_v58|0.00|0.00|
|full_chain_autonomous_timing_first_value_replant|lynn_v5|0.00|0.00|
|full_chain_autonomous_timing_first_value_replant|yhay81_six_day|0.00|0.00|
|full_chain_autonomous_timing_first_value_replant|yhay81_three_day|0.00|0.20|
|full_chain_autonomous_timing_first_value_replant|ecobot_v7|0.00|0.00|

32抽样官方完整局与6隔离检查通过；原生24局时延探针最大0.325秒。不是最终线上Python验收。

## 是否晋级与下一步

不晋级：未见seed、最终线上时延和全八90%目标均未完成。即使动作合法，未维护资产和少投资问题仍须按真实现金分渠道复盘。下一轮应核对首次选择实际改变了哪些任务、漏喂是否修复、是否挤掉更值钱的生产；再决定是否补投资/用工融资联合候选，不能直接加固定雇工或作物数量。P未用，所有原配置、失败、源码及build hash保留。

## 本轮后续审计已完成

- B共享背景48场原样真实对战逐局复现；73次实际改动，71次包含新补种延期、2次仅顺序/分组变化。证明能力被调用，不证明71次都赚钱，也未隔离延期与重新排班各自的收益。
- 复用108份既有逐日账本，完成81组人员—产销—现金成对分解；不是新模拟或独立样本。见 `S5D_LABOUR_SALES_REVIEW_ZH.md`。
- B共享/G001的平均现金反而少2665，但对手少3082，平均分差改善417；B共享/G003多赚1591，对手多赚1156，分差仅改善435。不能只看我方现金。
- A共享/G003多收获13.25次、多播种14.75次，但少赚4763；A共享/Kaito多赚1787，对手却多赚9664。说明动作数量、我方收入和胜率不是同一目标，不能给多干活固定奖励。
- B三个背景的平均胜率变化为+3.75、+3.75、+5个百分点。前两个seed聚类区间含0；第三个未校正多重比较的区间约[+0.89,+9.11]pp，但最终仍只有26.25%，不能作为全八强度突破。
- 1620场新配置账本中记录到的我方无效单位动作均0，但六个配置—对手单元仍有动物逃跑，合计21只；没有显式计划退出，不能将其改称主动放弃。共享背景A/B/AB本批均无逃跑，不外推到全部状态。
- 140项隔离断言确认延期/换种计划在存量剩余估值中的表达缺口；这是S4Q/S4R已知问题的新入口核对，不是新发现。S4R跨日延续大测此前无整体改善，不能机械重复“多看一天”。

下一步：先验证B静态评分候选的无效计算能否精确剪枝（保留候选顺序、1e-6平局规则和所有可能胜出的候选，不缩小策略空间）；再结合真实延期时点与下游产销分析哪些收益被抵消，而非继续堆余值权重。原始完整实战仍为本轮唯一强度证据，全部新候选不晋级。
