# 自主v7：冻结多对手清单与接入状态

更新：2026-09-03。这里的源码是**对手**，不得作为我方规划器的身份特攻分支。公开EcoBot v7与我们正在开发的自主v7不同。

## 当前范围

| 对手 | 冻结入口/来源 | 本地状态 |
|---|---|---|
| G001 | `../../../research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/main.py` | 完整C++对手，已有官方复验 |
| G003 / 55918053 | [用户提供的原版](g003/source/main.py) | 356条路线、144/216步切换树已接入；8局官方逐步一致，其中4局发生切换；另有100局开发对照 |
| Boatlee V29-R1 | [公开Notebook](https://www.kaggle.com/code/boatlee/v29-r1-adaptive-market-hysteresis)、[源码](boatlee_v29/output/main.py) | 完整C++接入；14局官方逐步对照含杂草/镜像，1,000局线程隔离；A/B各100局开发测试完成 |
| Kaito v58 | [公开Notebook](https://www.kaggle.com/code/kaitofukami/238-238-known-streams-v58-minimax-closed-loop)、[源码](kaito_v58/output/main.py) | 十套控制器完整C++接入；当前构建6局官方逐步对照、976项路由夹具、1,000局并行隔离通过；A/B各100局开发对战完成 |
| Lynn V5 | [公开Notebook](https://www.kaggle.com/code/lynnsakurai/farming-score-v5-timing-optimized)、[完整包](lynn_v5/output/submission.tar.gz) | 完整C++接入；10局官方逐步对照、291组临界夹具、1,000局并行隔离通过，A/B各100局完成 |
| EcoBot v7 | [公开Notebook](https://www.kaggle.com/code/premaananda108/economics-driven-rule-agent-ecobot-v7-arena)、[源码](ecobot_v7/output/main.py) | 完整C++接入，8局官方逐步动作/记忆/状态一致，368调度和730经济调用差分通过；1,000局隔离和200局开发面板完成 |
| yhay81 Six-Day Public-State Fieldbook | [公开Notebook](https://www.kaggle.com/code/yhay81/six-day-public-state-fieldbook)、[C++源码](yhay81_six_day/output/sixday_r4_source/policy.cpp)、[提交包](yhay81_six_day/output/sixday-publicstate-agent.tar.gz) | 已接入完整C++策略；10局官方逐步一致，1,000局并行复跑隔离通过，A/B两批各100局对照完成 |
| yhay81 Three-Day Shop Router | [公开Notebook](https://www.kaggle.com/code/yhay81/three-day-shop-router)、[C++源码](yhay81_three_day/output/source/policy.cpp)、[提交包](yhay81_three_day/output/shopstate-router-agent.tar.gz) | 完整C++接入，10局原版二进制/官方逐步一致、1,000局隔离通过；A/B各100局完成 |

共8个真实对手；PASS另作纯赚钱基准。Three-Day不替代Six-Day。Lynn V4的运行时间更新不代表应替换用户指定的V5。下文历史里程碑的分母保留当时范围。

机器索引：[registry.json](registry.json)。冻结资产清单：[source_inventory_v3.json](source_inventory_v3.json)。这些来源哈希标识本次下载的实际字节，不把标题或作者历史分数当作版本一致性依据。各条历史验收的构建与当前构建可能不同；当前回归收据以registry指针为准，不把历史分支覆盖冒充全部在新构建重跑。

## 必须保留的原版语义

- G003：不是一条G003动作带；使用全部356条候选路线和原版切换树。其底层执行器文本与G001不同，但完整AST相同；在核对后复用已有C++执行内核，不能复用G001的路线/模型数据替代G003。
- Boatlee：保留完整生产路线及有记忆的市场调节，而非仅取719步动作表。
- Kaito：保留全部专家、初始化/同步和公开状态切换；Notebook的“238/238”是作者已知冻结轨迹实验，不是本项目的新对战成绩。
- Lynn：保留完整模块依赖、路线数据和出售/融资/牧场覆盖规则。仅入口文件无法独立运行。
- EcoBot：保留原版经济估值、逐步决策和调度，包括原版弱点，不能把我们的修复版冒充公开原版。
- Fieldbook：它是带公开状态决策的分段路线方案，不是从零在线生成完整农场路线。原Python调用的C++提交桥含全局 `Session`，不能直接让16线程共享该会话；应使用每局独立policy context，并对照原提交入口验证。不能把它附带的模拟器替换本项目冻结官方规则引擎。

## G003首轮接入结果

同一开发集20261401–20261450，50个seed双座位；每个配置/对手100局。重复执行第二遍仅验证状态隔离，不计为新增独立样本。

| 我方配置 | PASS平均现金 | 对G001 | 对G003 | 对G003平均现金 / 对手现金 |
|---|---:|---:|---:|---:|
| L3_base（S3C03） | 177,100.73 | 36/100 | 7/100 | 95,221.38 / 114,679.03 |
| L3_return（暂定物流修复） | 179,319.03 | 40/100 | 9/100 | 95,858.72 / 114,367.64 |

这是开发证据，不是独立终验。仅凭4或2场胜场差不能宣布稳定增益；G003明显暴露额外弱点，继续只围着G001选参数不够。当时另5个对手尚未接入；现已补Fieldbook，其余4个未完成前，不晋级所谓通用冠军。

原始数据：[pool_g001_g003_A50_v2](../receipts/pool_g001_g003_A50_v2/results.json)。报告按seed聚类处理双座位相关性；零样本方差时不输出虚假的[100%,100%]正态区间，另记录有独立seed假设的保守界。v1收据保留，现金/胜负与v2完全相同，v2仅修正不确定性报告。

官方逐步复验：[4局未切换](../receipts/g003_original_official_parity_v1/acceptance.json)、[4局发生切换](../receipts/g003_switched_official_parity_v2/acceptance.json)。每局719步，对手动作及双方官方状态一致；保存完整轨迹。未声称覆盖356条路线的每个分支。

切换复验曾在146步报告差异：`PICKUP WHEAT`与`PICKUP WHEAT 1`。官方1.32.7明确缺省数量为1，因此修正比较器的规范化，再重跑完整对局通过；没有修改对手策略或模拟器，也没有删除失败收据。明确数量2仍然与1不同。

## 后续接入与评测

1. 先按源代码审查调用链；确保运行完整Agent，不降级为Replay。
2. 原版Python与C++逐步对照，覆盖切换/维护/市场条件，并做少量官方1.32.7完整局复验。
3. 全C++16线程运行同seed、双座位面板，检查每局719步、异常、复跑状态污染。
4. 逐对手报告胜率、现金、分差；保留PASS。各机制独立消融，改善一个对手但伤害另一个须明确标识。
5. 所有用于选型的seed归开发集；原最终50局PASS与100局G001的预留集仍未使用。用户未额外要求新增5个公开方案都90%，不擅自增加验收目标。

目前 `run_opponent_panel.py` 允许已接入的 `pass,g001,g003,yhay81_six_day,boatlee_v29,kaito_v58,lynn_v5,yhay81_three_day,ecobot_v7`。真实对手完成8/8；请求未接入名字或过期构建验收会报错，不会静默换成PASS或旧版。最新全池证据见 [S3I报告](../reports/S3I_ECOBOT_V7_NATIVE_AND_PANEL_ZH.md)。

WSL运行示例（仓库路径按本机）：

```text
/mnt/e/ai_coding/kaggle/kaggriculture/research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/.venv-bench/bin/python /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/tools/run_opponent_panel.py --opponents pass,g001,g003,yhay81_six_day --configs /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/profiles/s3c/hauling_configs.json --labels L3_base,L3_return --seed 20261401 --count 50 --repeat 2 --threads 16 --out /mnt/e/ai_coding/kaggle/kaggriculture/experiments/daily_dp_v7_20260903/receipts/NEW_UNIQUE_PANEL_DIR
```

`--out`必须是新目录。两个最终目标尚未达标，goal保持开发中；本轮无训练、无Kaggle提交、无Git推送。

## Fieldbook接入后的新证据

完整说明和运行命令：[S3D_FIELDBOOK_NATIVE_AND_PANEL_ZH.md](../reports/S3D_FIELDBOOK_NATIVE_AND_PANEL_ZH.md)。

| 我方配置 | 对Fieldbook开发A，100局 | 开发B，100局 | 合计，200局 |
|---|---:|---:|---:|
| L3_base（S3C03） | 29胜 | 20胜 | 49胜，24.5% |
| L3_return（仍暂定） | 28胜 | 22胜 | 50胜，25.0% |

每批50个seed双座位。重复跑第二遍只校验确定性，不把样本量翻倍。新增对手而未改我方策略；原PASS/G001/G003开发A的1,200条执行结果均保持一致。暂定物流修复并未在Fieldbook上表现出明确稳定增益，不能据此晋级。

Fieldbook是每6天选分段路线，加阶段资金/库存保护；不是每日从零规划。原版策略源码未经修改编译，适配层只转换合法可见输入与原始动作，并为每局建立独立context。10局官方对照覆盖末阶段路线0/1/2/4，未声称穷尽所有决策树叶子。

## Boatlee V29接入后的新证据

详见[S3E_BOATLEE_V29_NATIVE_AND_PANEL_ZH.md](../reports/S3E_BOATLEE_V29_NATIVE_AND_PANEL_ZH.md)。开发A/B每配置各100局，S3C03对Boatlee为13/100、28/100，暂定物流修复为15/100、28/100；两批总计分别20.5%与21.5%。旧4项面板（含PASS）A/B共3,200条复跑记录保持一致。

原版动作表有720条，其中末条不在正常719步决策中调用，不能去掉首帧或平移索引。静态`_FR_ITEMS`为空，旧front-run分支始终不生效；保留后期自适应出售、市场压力记忆、额外销量预算、289步近镜像锁定以及杂草恢复。没有把对手原本关闭的功能打开。

Fieldbook在新构建下补4局官方回归和1,000局线程隔离；其旧报告二进制哈希是历史快照，当前可用收据以registry为准。全部原策略、数据、旧收据与旧二进制保留，不宣称整体goal达标。

## Kaito V58接入后的新证据

详见[S3F_KAITO_V58_NATIVE_AND_PANEL_ZH.md](../reports/S3F_KAITO_V58_NATIVE_AND_PANEL_ZH.md)。基础版开发A/B为20/100、16/100，暂定物流修复20/100、17/100，合计18.0%与18.5%。两批旧5项（含PASS）共4,000条执行记录不变。迁移完成不代表我方规划能力增强。

## Lynn V5接入后的新证据

详见[S3G_LYNN_V5_NATIVE_AND_PANEL_ZH.md](../reports/S3G_LYNN_V5_NATIVE_AND_PANEL_ZH.md)。对Lynn基础版A/B17/100、19/100，暂定物流修正19/100、26/100，合计18%/22.5%。原六项面板（含PASS）4,800条记录不变。目标仍未通过，正式验收seed未使用。

十套子控制器都保留独立供给/近镜像/欠售/杂草状态；输出未选中的路线也须与原版逐步一致。原公开策略内的盘面指纹仅留在对手模块，禁止迁入我方。初次导入发现字符串数量误读问题，修复后完整重测，失败记录保留。

当前构建有6场官方逐步对照（包含2场同版本对照），976个合成路由条件测试覆盖十个输出；不声称自然对局穷尽全部分支。Fieldbook与Boatlee亦补当前构建的官方/隔离回归；当前收据以registry为准，历史构建仍保存。
