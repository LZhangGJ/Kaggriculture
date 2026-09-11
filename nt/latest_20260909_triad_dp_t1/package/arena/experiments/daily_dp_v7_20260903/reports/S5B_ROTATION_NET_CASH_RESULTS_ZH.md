# S5B 净收益换种与时机：实战筛查结果

## 最终裁定与失败定位补充

不晋级、不替换保留基线。新配置提高部分静止对手现金，不等于提高实时对战胜率；完整自主背景的退步尤其明显。

- 机制12562+26+17检查、32官方完整局和6隔离通过；只证明对应规则/机制覆盖，不能证明策略正确。
- 720新配置账本重复、720旧控制复现完成。`s5b_escape_triage_v1`确认**6局、12只动物非计划逃跑**，原对照同局均0，不是允许的计划性退出。
- 净换种/Kaito的失败窗口日24–25现金约3.5–3.9万；净换种/Three-Day日16–17现金约0.7–1.1万，小麦亦有余量，仍漏喂。不能归因为一律缺现金。
- 时机/Three-Day另一失败窗口日11–12现金9→0，仅雇1→0人、同时拥有大量动物，属于另一类流动性—维护风险，须逐步追因。
- `s5b_wait_execution_v1`的12场原样实战全部现金匹配；观察到同地块连续空置等待8–11天。部分日期现金极紧，尚不能证明延期本身是错误；不设置武断的最大等待天数。
- 当前条件续跑关闭`day_consequence_compare`/`day_value_replan_next_day`，与完整自主真实执行决策并非完全一致；余值只追踪实际资产及显式延期项目，不包括所有未来自主投资。它是近似估值，不是保证可兑现的终局现金。
- 本轮换种只覆盖有限作物到新作物及延期/停止新种，**没有实现减少动物维护或退出释放资源的完整跨产业候选**，不能称已覆盖高手全部转产能力。

下一轮先以只读探针定位维护义务在哪一层丢失，并区分资源不足、任务被删、编排放不下、计划执行漂移；修复须为可证伪的通用能力。所有失败分支、源码快照和配置保留，未见P没有使用，目标仍未完成。

1440完整C++实时局，720旧控制逐局等于S4Z，720新配置账本重复一致。每格10个旧开发seed、双座位20局；不是独立最终验收，不晋级。

|配置|PASS现金|g001|g003|boatlee_v29|kaito_v58|lynn_v5|yhay81_six_day|yhay81_three_day|ecobot_v7|八对手平均|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|all_intraday_insert|189,894|5/20|0/20|6/20|4/20|11/20|6/20|7/20|20/20|36.9%|
|all_intraday_insert_rotation|192,916|4/20|4/20|8/20|2/20|4/20|6/20|2/20|20/20|31.2%|
|all_intraday_insert_net|193,820|9/20|2/20|2/20|4/20|6/20|8/20|8/20|20/20|36.9%|
|all_intraday_insert_timing|193,007|7/20|4/20|4/20|2/20|0/20|11/20|4/20|20/20|32.5%|
|full_chain_autonomous|163,072|8/20|4/20|2/20|6/20|4/20|2/20|10/20|20/20|35.0%|
|full_chain_autonomous_rotation|185,757|2/20|0/20|0/20|0/20|2/20|2/20|2/20|20/20|17.5%|
|full_chain_autonomous_net|173,850|4/20|0/20|0/20|4/20|2/20|2/20|10/20|20/20|26.2%|
|full_chain_autonomous_timing|179,930|2/20|2/20|4/20|4/20|2/20|0/20|1/20|19/20|21.2%|

## 配对差异（相对不开轮作的原背景）

按seed合并两座位计算，t区间近似且没有多重比较校正；零经验方差不意味着确定性。

|新配置|对手|胜率变化pp|分差变化|分差近似95%区间|
|---|---|---:|---:|---|
|all_intraday_insert_net|g001|+20.0|+2,630|[-7502, 12761]|
|all_intraday_insert_net|g003|+10.0|+499|[-11365, 12363]|
|all_intraday_insert_net|boatlee_v29|-20.0|-6,879|[-16195, 2438]|
|all_intraday_insert_net|kaito_v58|+0.0|-6,793|[-16903, 3316]|
|all_intraday_insert_net|lynn_v5|-25.0|+864|[-13286, 15013]|
|all_intraday_insert_net|yhay81_six_day|+10.0|+4,181|[-10103, 18465]|
|all_intraday_insert_net|yhay81_three_day|+5.0|+2,176|[-15397, 19750]|
|all_intraday_insert_net|ecobot_v7|+0.0|+1,482|[-9536, 12500]|
|all_intraday_insert_timing|g001|+10.0|+1,843|[-4636, 8321]|
|all_intraday_insert_timing|g003|+20.0|-2,661|[-19205, 13883]|
|all_intraday_insert_timing|boatlee_v29|-10.0|-11,804|[-29241, 5632]|
|all_intraday_insert_timing|kaito_v58|-10.0|-10,019|[-17482, -2557]|
|all_intraday_insert_timing|lynn_v5|-55.0|-17,627|[-29415, -5838]|
|all_intraday_insert_timing|yhay81_six_day|+25.0|+399|[-16171, 16969]|
|all_intraday_insert_timing|yhay81_three_day|-15.0|-1,774|[-15006, 11458]|
|all_intraday_insert_timing|ecobot_v7|+0.0|-5,373|[-19882, 9136]|
|full_chain_autonomous_net|g001|-20.0|-5,229|[-18938, 8481]|
|full_chain_autonomous_net|g003|-20.0|-6,714|[-22501, 9072]|
|full_chain_autonomous_net|boatlee_v29|-10.0|-8,157|[-20385, 4071]|
|full_chain_autonomous_net|kaito_v58|-10.0|-6,864|[-17967, 4239]|
|full_chain_autonomous_net|lynn_v5|-10.0|-4,935|[-15500, 5630]|
|full_chain_autonomous_net|yhay81_six_day|+0.0|-3,652|[-17350, 10046]|
|full_chain_autonomous_net|yhay81_three_day|+0.0|+2,210|[-10254, 14675]|
|full_chain_autonomous_net|ecobot_v7|+0.0|-3,296|[-18949, 12357]|
|full_chain_autonomous_timing|g001|-30.0|-3,894|[-22597, 14809]|
|full_chain_autonomous_timing|g003|-10.0|-10,149|[-28041, 7744]|
|full_chain_autonomous_timing|boatlee_v29|+10.0|-1,223|[-17429, 14983]|
|full_chain_autonomous_timing|kaito_v58|-10.0|-6,200|[-22449, 10049]|
|full_chain_autonomous_timing|lynn_v5|-10.0|-5,171|[-19678, 9337]|
|full_chain_autonomous_timing|yhay81_six_day|-10.0|-5,649|[-14864, 3565]|
|full_chain_autonomous_timing|yhay81_three_day|-45.0|-7,612|[-16603, 1378]|
|full_chain_autonomous_timing|ecobot_v7|-5.0|+6,058|[-11140, 23256]|

## 真实现金变化渠道

均为整套策略变动的账目差，不是独立可叠加因果。

|配置|对手|现金Δ|草莓收入Δ|奶收入Δ|羊毛收入Δ|小麦采购Δ|肥料采购Δ|工资Δ|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|all_intraday_insert_net|pass|+3,926|+19,200|-8,622|-911|+6,047|+0|+104|
|all_intraday_insert_timing|pass|+3,112|+14,717|-4,206|+2,307|+5,494|+0|-41|
|all_intraday_insert_net|g001|-5,134|+6,463|-4,154|-7,942|+2,204|+10|-680|
|all_intraday_insert_timing|g001|+1,743|+8,928|-10,826|+8,670|+3,864|-7|-566|
|all_intraday_insert_net|g003|-21,417|-9,152|-14,517|+2,026|+4,858|-6|+1,389|
|all_intraday_insert_timing|g003|-11,359|-7,827|-2,348|-3,087|+2,615|+24|+305|
|all_intraday_insert_net|boatlee_v29|-15,305|+2,714|-10,061|-7,781|+7,407|+22|+593|
|all_intraday_insert_timing|boatlee_v29|-13,919|-2,946|-6,540|-2,960|+6,155|+13|+416|
|all_intraday_insert_net|kaito_v58|-13,770|-186|-7,072|-5,748|+3,350|+28|+101|
|all_intraday_insert_timing|kaito_v58|-6,292|-4,388|-4,737|+5,256|+4,488|+10|-218|
|all_intraday_insert_net|lynn_v5|-20,938|-12,784|-16,582|+5,489|+3,650|+7|+1,254|
|all_intraday_insert_timing|lynn_v5|-17,549|-1,959|-19,260|+5,569|+3,460|+17|+538|
|all_intraday_insert_net|yhay81_six_day|-1,433|+10,874|-6,673|-4,984|+3,608|+34|-75|
|all_intraday_insert_timing|yhay81_six_day|-1,271|+1,832|-3,010|+2,525|+2,415|+13|-216|
|all_intraday_insert_net|yhay81_three_day|-1,203|+7,425|-2,797|-5,366|+3,402|+27|-391|
|all_intraday_insert_timing|yhay81_three_day|+4,354|+12,086|-990|-2,189|+4,492|+0|-1,472|
|all_intraday_insert_net|ecobot_v7|-921|+4,204|-2,139|-4,737|+2,520|-46|+98|
|all_intraday_insert_timing|ecobot_v7|-57|+9,556|-2,837|-1,169|+3,163|-16|-1,198|
|full_chain_autonomous_net|pass|+10,778|+40,407|-10,942|-7,180|+5,488|+121|-1,906|
|full_chain_autonomous_timing|pass|+16,857|+35,732|+4,034|-8,184|+14,036|+32|-2,541|
|full_chain_autonomous_net|g001|-924|+9,919|+1,615|-6,398|+4,406|+26|-1,125|
|full_chain_autonomous_timing|g001|+3,110|+14,504|+2,113|+584|+8,219|+11|-2,368|
|full_chain_autonomous_net|g003|-11,896|+4,449|-6,133|-6,878|+2,560|+85|-35|
|full_chain_autonomous_timing|g003|-11,721|+118|-3,671|+4,763|+8,916|-26|-824|
|full_chain_autonomous_net|boatlee_v29|-865|+5,812|-1,757|+5,209|+5,693|-0|+518|
|full_chain_autonomous_timing|boatlee_v29|+7,558|+2,429|+64|+12,082|+7,406|-6|-842|
|full_chain_autonomous_net|kaito_v58|-7,403|+5,334|-8,274|-2,480|+3,295|+91|-478|
|full_chain_autonomous_timing|kaito_v58|-8,157|+4,322|-5,708|+6,685|+9,006|-21|-1,570|
|full_chain_autonomous_net|lynn_v5|-5,815|+9,136|-2,887|-11,742|+2,857|+137|-808|
|full_chain_autonomous_timing|lynn_v5|-8,184|+3,270|+3,298|-2,386|+7,678|+12|-1,507|
|full_chain_autonomous_net|yhay81_six_day|+7,754|+13,803|-2,808|+2,637|+4,044|+36|-399|
|full_chain_autonomous_timing|yhay81_six_day|+10,872|+9,318|-3,629|+13,646|+6,394|+3|-1,262|
|full_chain_autonomous_net|yhay81_three_day|-678|+16,195|-1,055|-14,654|+3,267|+86|-1,463|
|full_chain_autonomous_timing|yhay81_three_day|-220|+15,505|+6,225|-6,400|+8,167|+18|-1,662|
|full_chain_autonomous_net|ecobot_v7|+3,750|+14,480|+10,791|-15,614|+3,674|+67|-404|
|full_chain_autonomous_timing|ecobot_v7|+10,663|+20,796|+4,787|-264|+10,267|+19|-1,742|

## 异常与边界

|配置|对手|平均无效单位动作|平均动物逃跑|
|---|---|---:|---:|
|all_intraday_insert_net|pass|0.00|0.00|
|all_intraday_insert_net|g001|0.00|0.00|
|all_intraday_insert_net|g003|0.00|0.00|
|all_intraday_insert_net|boatlee_v29|0.00|0.00|
|all_intraday_insert_net|kaito_v58|0.00|0.00|
|all_intraday_insert_net|lynn_v5|0.00|0.00|
|all_intraday_insert_net|yhay81_six_day|0.00|0.00|
|all_intraday_insert_net|yhay81_three_day|0.00|0.00|
|all_intraday_insert_net|ecobot_v7|0.00|0.00|
|all_intraday_insert_timing|pass|0.00|0.00|
|all_intraday_insert_timing|g001|0.00|0.00|
|all_intraday_insert_timing|g003|0.00|0.00|
|all_intraday_insert_timing|boatlee_v29|0.00|0.00|
|all_intraday_insert_timing|kaito_v58|0.00|0.00|
|all_intraday_insert_timing|lynn_v5|0.00|0.00|
|all_intraday_insert_timing|yhay81_six_day|0.00|0.00|
|all_intraday_insert_timing|yhay81_three_day|0.00|0.00|
|all_intraday_insert_timing|ecobot_v7|0.00|0.00|
|full_chain_autonomous_net|pass|0.00|0.00|
|full_chain_autonomous_net|g001|0.00|0.00|
|full_chain_autonomous_net|g003|0.00|0.00|
|full_chain_autonomous_net|boatlee_v29|0.00|0.00|
|full_chain_autonomous_net|kaito_v58|0.00|0.10|
|full_chain_autonomous_net|lynn_v5|0.00|0.00|
|full_chain_autonomous_net|yhay81_six_day|0.00|0.00|
|full_chain_autonomous_net|yhay81_three_day|0.00|0.30|
|full_chain_autonomous_net|ecobot_v7|0.00|0.00|
|full_chain_autonomous_timing|pass|0.00|0.00|
|full_chain_autonomous_timing|g001|0.00|0.00|
|full_chain_autonomous_timing|g003|0.00|0.00|
|full_chain_autonomous_timing|boatlee_v29|0.00|0.00|
|full_chain_autonomous_timing|kaito_v58|0.00|0.00|
|full_chain_autonomous_timing|lynn_v5|0.00|0.00|
|full_chain_autonomous_timing|yhay81_six_day|0.00|0.00|
|full_chain_autonomous_timing|yhay81_three_day|0.00|0.20|
|full_chain_autonomous_timing|ecobot_v7|0.00|0.00|

32局抽样官方一致性和6项隔离检查通过；原生12局探针最大动作耗时0.402秒，不是最终Python提交时延保证。首次step144接入错误及修复记录保留。

四项执行背景与换种模块同开并不等于已达90%。后续方向必须由真实数据决定；本轮没有用未见P，也没有将预测最佳当实际终局。
