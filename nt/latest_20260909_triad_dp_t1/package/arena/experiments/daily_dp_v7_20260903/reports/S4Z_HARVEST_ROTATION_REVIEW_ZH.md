# S4Z 成熟作物轮作小面板复盘

## 判定与下一步

本轮两个新配置均不晋级。完整自主轮作对PASS从163,072升到185,757，但对八对手平均胜率从35%降至17.5%；共享背景从36.875%降至31.25%。这是同10个开发seed的配对小面板结果，不能与此前50seed均值混比，也不是总体胜率的精确估计。

增加草莓确实能形成实际产出，说明轮作不是dead field；但高价值单品收入不等于整场净增收。完整自主对G003草莓收入增加7,741.6，牛奶减少7,207.3、羊毛减少3,924，商品采购支出增加8,106，终盘现金减少11,464.2。对G001/Kaito也有类似“草莓更多、净现金更少”的完整政策结果。不能把这些差额视为独立可叠加因果。

下一项不是直接关闭所有动态调整，也不是继续扫草莓偏好。源码发现当前`rotate`调用的是`values()`，两实测配置`cashflow_value_mode=0`；完整自主的空地投资却走`compare_portfolios`的净现金/联合未来用工。因此新开启的轮作没有自动纳入那套更完整的组合比较。此处存在明确的估值口径分离，不代表修正后一定加分。

下一轮应做统一的轮作净后果原型：同一观察下，旧作物本次收获必须在KEEP与换种两边保留；对后续同类补种与新作物分别生成现金/投入/义务日历，合并到整个已有农场后再比较。尤其检查自产小麦替代饲料采购、肥料自用与出售的机会成本、新增用工与首次到账时间。不要将真实未来用于选择。

实现前需要两项机制防错：混合作物过渡日不能把旧小麦收获记成新草莓收入；替换已有地块后不能在成本/产能日历中同时保留旧项目并再加新项目。先在隔离原型验证数值与执行对应，再接独立开关、小面板和完整链组合；预算限制属于计算预算而非固定产业阈值。

本轮没有修改生产源码、没有新增对手特调、没有隐藏信息或事后Oracle。原配置保留，Goal仍active。

原生17项机制通过，720完整实时局、360新配置重复账本通过；360旧控制逐局与S4V相同。每格10旧开发seed×双座位20局。未做新配置官方逐步与线上时延验收，不晋级。

|配置|PASS现金|g001|g003|boatlee_v29|kaito_v58|lynn_v5|yhay81_six_day|yhay81_three_day|ecobot_v7|平均胜率|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|all_intraday_insert|189,894|5/20|0/20|6/20|4/20|11/20|6/20|7/20|20/20|36.9%|
|all_intraday_insert_rotation|192,916|4/20|4/20|8/20|2/20|4/20|6/20|2/20|20/20|31.2%|
|full_chain_autonomous|163,072|8/20|4/20|2/20|6/20|4/20|2/20|10/20|20/20|35.0%|
|full_chain_autonomous_rotation|185,757|2/20|0/20|0/20|0/20|2/20|2/20|2/20|20/20|17.5%|

## 配对变化（同seed两个座位合并）

|背景|对手|胜率变化pp|资金差变化|资金差近似95%区间|
|---|---|---:|---:|---|
|all_intraday_insert|g001|-5.0|-2,098|[-14914, 10717]|
|all_intraday_insert|g003|+20.0|+646|[-10647, 11939]|
|all_intraday_insert|boatlee_v29|+10.0|+2,791|[-8402, 13984]|
|all_intraday_insert|kaito_v58|-10.0|-2,859|[-12737, 7019]|
|all_intraday_insert|lynn_v5|-35.0|-1,924|[-13245, 9396]|
|all_intraday_insert|yhay81_six_day|+0.0|+1,995|[-7394, 11383]|
|all_intraday_insert|yhay81_three_day|-25.0|-2,672|[-15785, 10441]|
|all_intraday_insert|ecobot_v7|+0.0|+5,781|[-5949, 17511]|
|full_chain_autonomous|g001|-30.0|-10,111|[-25702, 5481]|
|full_chain_autonomous|g003|-20.0|-6,156|[-17831, 5520]|
|full_chain_autonomous|boatlee_v29|-10.0|-4,140|[-12482, 4203]|
|full_chain_autonomous|kaito_v58|-30.0|-9,684|[-18213, -1155]|
|full_chain_autonomous|lynn_v5|-10.0|-5,981|[-16646, 4684]|
|full_chain_autonomous|yhay81_six_day|+0.0|+87|[-11647, 11821]|
|full_chain_autonomous|yhay81_three_day|-40.0|-4,741|[-16476, 6995]|
|full_chain_autonomous|ecobot_v7|+0.0|+1,074|[-15395, 17544]|

## 异常与解释边界

以下是每局平均次数，不能当独立失败概率。开启更多重选机会并不保证原有评分正确；不能把本轮失败归为所有轮作都无效，也不能将某个对手20局提高当稳定泛化。

|配置|对手|平均无效单位动作|平均动物逃跑|
|---|---|---:|---:|
|all_intraday_insert_rotation|pass|0.00|0.00|
|all_intraday_insert_rotation|g001|0.00|0.00|
|all_intraday_insert_rotation|g003|0.00|0.00|
|all_intraday_insert_rotation|boatlee_v29|0.00|0.00|
|all_intraday_insert_rotation|kaito_v58|0.00|0.00|
|all_intraday_insert_rotation|lynn_v5|0.00|0.00|
|all_intraday_insert_rotation|yhay81_six_day|0.00|0.00|
|all_intraday_insert_rotation|yhay81_three_day|0.00|0.00|
|all_intraday_insert_rotation|ecobot_v7|0.00|0.00|
|full_chain_autonomous_rotation|pass|0.00|0.00|
|full_chain_autonomous_rotation|g001|0.00|0.00|
|full_chain_autonomous_rotation|g003|0.00|0.00|
|full_chain_autonomous_rotation|boatlee_v29|0.00|0.00|
|full_chain_autonomous_rotation|kaito_v58|0.00|0.00|
|full_chain_autonomous_rotation|lynn_v5|0.00|0.00|
|full_chain_autonomous_rotation|yhay81_six_day|0.00|0.00|
|full_chain_autonomous_rotation|yhay81_three_day|0.00|0.00|
|full_chain_autonomous_rotation|ecobot_v7|0.00|0.00|

## 同seed实际现金/生产变化

以下是完整政策干预的账目差异，不是独立可叠加收益；所有现金收支守恒检查通过。

|背景|对手|现金Δ|草莓收入Δ|番茄收入Δ|小麦收入Δ|瓜收入Δ|工资支出Δ|草莓种植Δ|番茄种植Δ|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|all_intraday_insert|pass|+3,022.0|+12,780.4|+3,437.5|-1,943.8|+53.6|+461.7|+11.6|+4.1|
|all_intraday_insert|g001|-2,988.9|+2,881.2|+2,279.8|-177.6|-468.6|-667.0|+8.0|+4.4|
|all_intraday_insert|g003|-12,925.0|-13,353.2|+3,743.6|+200.9|-383.5|+1,407.3|-2.9|+8.1|
|all_intraday_insert|boatlee_v29|-1,623.5|+3,188.2|+5,124.2|+496.6|-655.8|+505.7|+5.1|+8.5|
|all_intraday_insert|kaito_v58|-3,353.6|-4,326.9|+3,341.4|-311.6|-319.9|+720.8|+3.2|+7.8|
|all_intraday_insert|lynn_v5|-7,683.6|-6,753.8|+2,168.8|+3,188.8|-655.3|+1,333.1|-2.3|+5.5|
|all_intraday_insert|yhay81_six_day|+7,186.8|+7,825.1|-226.3|+22.6|-484.9|+251.3|+9.1|+2.9|
|all_intraday_insert|yhay81_three_day|-1,361.5|+10,396.0|+337.1|+216.8|-380.5|-1,013.4|+14.4|+2.9|
|all_intraday_insert|ecobot_v7|+5,063.9|+16,982.8|+1,363.0|-678.5|+177.1|-930.2|+11.8|+1.6|
|full_chain_autonomous|pass|+22,685.0|+37,382.3|+950.0|-4,466.6|+2,873.1|-678.5|+28.4|+1.9|
|full_chain_autonomous|g001|-6,556.2|+7,521.6|+1,426.8|-2,996.3|+590.8|-1,077.3|+16.2|+5.5|
|full_chain_autonomous|g003|-11,464.2|+7,741.6|+503.6|-379.7|+829.0|-131.4|+9.7|+8.3|
|full_chain_autonomous|boatlee_v29|-2,902.1|+6,355.3|+4,460.8|-1,918.4|+1,327.5|-752.2|+10.5|+13.9|
|full_chain_autonomous|kaito_v58|-10,442.6|+9,593.8|-373.2|-472.0|+558.1|-489.4|+10.2|+7.1|
|full_chain_autonomous|lynn_v5|-7,574.2|+8,593.3|+91.5|-1,889.4|+170.2|-750.2|+10.5|+5.4|
|full_chain_autonomous|yhay81_six_day|+2,003.3|+7,701.9|+5,088.6|-1,568.8|+1,566.2|-680.4|+10.1|+12.0|
|full_chain_autonomous|yhay81_three_day|-9,411.4|+10,148.5|+573.6|-1,034.2|+525.9|-726.1|+17.1|+1.9|
|full_chain_autonomous|ecobot_v7|+4,667.7|+14,309.8|+5,461.0|-2,502.6|+67.7|-1,143.8|+15.1|+9.6|
