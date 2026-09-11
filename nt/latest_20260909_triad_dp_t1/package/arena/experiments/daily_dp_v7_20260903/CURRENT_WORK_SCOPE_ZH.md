# 当前工作边界与恢复入口

更新：2026-09-04，原八对手Goal已paused；用户另行授权8节点、多种子共同计划搜索试跑。S5D/S5E保持原样，不恢复旧开发、不改生产。

## 当前恢复入口（优先于下面旧阶段说明）

**2026-09-05 S9冻结实验结束但总Goal仍active：** 127切换组合无可靠增益，正式保留不切换；最终S9 38.86%/最弱21.5%，J7_03 58.86%/最弱27.5%，没有形成更强Agent，禁止以工程完成关闭总Goal。14局官方全状态/动作一致，观察入口最大30.27ms，0实际切换，不能称部署树已验收。代码、报告和收据完整；`GOAL_STATUS_ZH.md`明确覆盖ACCEPTANCE/COMPLETION_AUDIT的工程完成含义。当前继续`policy_tree_probe.py`开发诊断：相同特征、开局和五后缀，直接以全部候选胜负/分差训练浅策略树，旧训练/选型数据只作开发，不读旧最终、不晋级，后续有效必须新seed冻结验证。仍最多16线程，不改旧生产、不Git/Kaggle、不恢复旧Goal。

**2026-09-05 S9最新恢复点：** P0–P2已验收，粗筛12,544续跑、训练15,680续跑和7棵候选日浅树已完成；训练固定186/448、有限5后缀事后398/448，不是在线胜率。当前唯一重型会话47756为`strategy_switch7_20260905/switch_learning.py collect_select`，另32全新seed正在生成选型标签；之后select→final_test→verify build/official/report。83395/17287/41134已结束，勿重启。入口源码与模型训练输入SHA冻结，不改原策略；verify.py尚未构建冻结，增加了最终实际救回/伤害统计，10项纯数据测试通过。继续最弱对手胜率优先，不切换也保留；不根据最终集重选。未Git/Kaggle，Goal仍active。

**2026-09-05 S9进度更新，覆盖下方旧S9状态：** `strategy_switch7_20260905`已完成P0/P1/P2。1,494套×56世界共83,664局，约30分钟；经营行为聚类29族＋3基线，新16seed双座位七对手7,168局；另外896局全单位动作效果与cash/hash审计全部通过。所有32代表保留（含11单例家族），最弱对手优先的新开局索引38，验证最弱6/32、总83/224，不是强度达标。P2验收报告和JSON均已落地。当前唯一重型会话**83395**=`switch_learning.py coarse`，固定38，在7检查日接32代表后缀、4发现seed双座位七对手，共12,544续跑；勿重复启动。44271/58693/83776/20532/48227均exit0。下一步按真实coarse结果压缩互补后缀→训练32seed精标签→有稳定增益再浅树→选型32seed→最终100seed/每对手200局→部署/官方复验。`switch_learning.py`已冻结SHA，不能边采样边改。所有旧Goal/Oracle/生产版本不恢复，未Git/Kaggle；新Goal保持active。

**2026-09-05最新：S9行为聚类＋单次盘面切换Goal已创建并active。** 用户明确要求整理1,494套、代表质检、单次切换Oracle、浅树、最终7对手胜率。唯一当前入口 `strategy_switch7_20260905/README_ZH.md` 与 `PLAN_ZH.md`。有效 `receipts_v2`；v1只有Result名称冲突编译失败日志。P0已通过旧DP27七对手14局hash、诊断开关、串并行、reset逆序、S8执行和第1/12/24天分支拼接一致性；224局44.9177局/s，首阶段83,664局估算31分钟。正在开始 `run.py collect`，新发现seed20270001..04双座位×7，每策略56局，16策略一块。随后 `cluster.py cluster`（已编写待实测）和`validate`；发现硬生产失败须质检，不跳过。P3/P4尚未执行，不声称最强Agent或90%。参考队友main eec4097的质检文档，但不照搬40%等经验比例。旧S8/旧所有实验保持冻结。`pool_node_oracle_20260905`只编译过，未启动Oracle，当前不要启动。新Goal替代此前计划，禁止创建重复Goal或恢复旧Goal，未Kaggle/Git。

**Beam12×8先去重再全池测试已完成（2026-09-05，最新）**：85919已exit0；`search29_three_day_b12_w8_20260905/receipts_v2_pool1000/ACCEPTANCE.json`为COMPLETE_THREE_DAY_BEAM12_WORLD8_POOL1000_NOT_ROUTER，源码未改，2官方PASS。10,631候选/85,048终局续跑，搜索436.94秒；2,370套8/8参数全量训练复演去重后181套，距1,000缺819，不凑数。18,968训练完整复演（含最佳8局），全池5,792＋7参照224＝6,016未见局；未见20269401..16双座位。新训练首选12/32、旧B4×8训练首选12/32、旧测试最佳12/32、B4×32训练最佳10/32、DP27/J7_03各16/32、KEEP8/32；新池单套事后最高007为16/32，平均35.32%，90%策略0，事后任选覆盖28/32非在线。新181个训练行为hash全部在旧181池中出现（参数相同仅68，Day0各1、前3日参数旧6新13），增加宽度未增加训练行为。见`CONCLUSION_POOL1000_ZH.md`、`selected1000.json`及逐策略报告，未在测试集重选/晋级。总经过约23.5分钟。当前无活跃本轮模拟，不重启78605/67005/85919，不启动B12×32草稿、不继续新搜索/ML/RL、不Git/Kaggle、不恢复旧Goal。

**Beam12×8改1000套：搜索已保存、提取验证中（2026-09-05，最新）**：用户要求先取1000不同经营策略、最后测试。原78605已被守护67005在29层search_done完整后SIGTERM结束（exit1是有意切换，不是搜索失败），67005 exit0；`receipts_v1/POOL_AMENDMENT_STOP.json`确认29前沿hash与未见文件0。原搜索436.94秒、10,631候选/85,048续跑/43,629,320transitions，最佳8/8。当前85919=`search29_three_day_b12_w8_20260905/run1000.py all`，输出`receipts_v2_pool1000`，只将提取上限200→1000，复用相同前沿不重跑；`SEARCH_REUSE.json`已PASS。目标训练8/8参数与实际行为双去重1000套，不足如实；全池冻结后再跑20269401..16双座位每套32局，加7冻结参照和2官方核对。新脚本/PLAN_POOL1000/inputs已冻结，不能边跑改。先查85919及v2 progress，勿启动原run.py，不改底层、不Git/Kaggle、不恢复旧Goal。run1000.py/新计划为当前入口，旧200和B12×32入口均非当前授权流程。

**Three-Day Beam12×8已启动（2026-09-05，最新授权）**：用户在Beam4×32完成后要求Beam12、8局。入口`search29_three_day_b12_w8_20260905/run.py`；check39310已exit0（双座位Arena、复制恢复、1/16线程、真实8世界Beam12及与最初Beam4×8首层全候选一致全部PASS），当前78605=`run.py all`。训练仍20269001..04双座位，Day0–28；新验证20269401..16双座位。训练8/8全胜策略按参数和实际行为去重，最多200不足不凑，另存训练最佳。池冻结后与原Beam4×8训练首选/旧测试最佳、Beam4×32训练最佳、DP27/J7_03/KEEP同库复验，再2官方完整核对。当前源码/PLAN/inputs冻结，不边跑改；不改底层/候选规则、未新编译。恢复先查78605/progress，不能启动旧B12×32草稿，不重跑已完成B4×32；不Kaggle/Git，不恢复旧Goal。

**Three-Day Beam4×32已完成（2026-09-05，优先于下方启动状态）**：99926已exit0，`search29_three_day_b4_w32_20260905/receipts_v1/ACCEPTANCE.json`为`COMPLETE_THREE_DAY_BEAM4_WORLD32_NOT_ROUTER`，源码未改、2官方PASS。Day0–28共4,927候选/157,664终局续跑/75,369,760transitions，搜索740.2秒，总经过约14.1分钟。最佳训练30/32，无32/32计划，合格池0/200，不降门槛；`selected200.json.training_best`保存B4W32_TRAIN_BEST完整29日经济参数。新20269301..16双座位：新最佳10/32，旧训练首选5/32，旧测试最佳4/32，DP27 18/32，J7_03 16/32，KEEP8/32；未见新策略平均资金95,881、对手107,987、差−12,106。6策略共192完整未见局；未在测试集重选或晋级。见`CONCLUSION_ZH.md`/`COMPARISON_ZH.md`，不可将空池覆盖0当全搜索上限。程序无活跃运行；不启动Beam12草稿，不做新搜索/ML/RL，不恢复旧Goal，不Kaggle/Git。

**Three-Day Beam4×32重跑已启动（2026-09-05，最新授权）**：用户暂停Beam12后最终要求Beam4、搜索共同评分8→32局。入口`search29_three_day_b4_w32_20260905/run.py`；check会话7891已exit0（Arena双座位、复制恢复、1/16线程一致、真实32世界Beam4全部PASS），当前99926=`run.py all`。搜索Day0–28，训练原20269001..4＋新20269201..12共16seed双座位；未见20269301..16。只Three-Day，最多200套训练32/32且行为不同策略，不足不凑；训练最佳另存诊断，所有池策略和旧Beam4训练首选/旧测试最佳等6参照同库新seed复验，再2官方逐步核对。冻结脚本/PLAN/inputs，不边跑改。无新编译，不改底层/候选，不恢复S8/旧Goal，不Kaggle/Git。恢复先查99926/progress，勿重启。此前`search29_three_day_b12_w32_20260905`只有未启动草稿，禁止启动。

**七对手分别搜索已完成（2026-09-05，优先于下方旧状态）**：handle50155已exit0，`search29_separate7_20260904/receipts_v1/ACCEPTANCE.json`为`COMPLETE_SEPARATE7_POOL_NOT_ROUTER`，冻结源码不变。7组Day0–28/Beam4，24,005候选、192,040终局续跑；保存1,381套训练8/8且训练行为不同的经济策略（前6组各200，Three-Day181，少19如实报告）。全部16新seed双座位复验44,192局、基准672局；14官方完整核对PASS。各组单套事后最高新seed胜场G001/G003/B29/K58/L5/六日/三日=20/32、32/32、32/32、32/32、20/32、18/32、11/32；≥90%策略数0/198/197/200/0/0/0。这不是通用Agent90%，也没有对测试最佳重新晋级。最终`audit_pools.py`已exit0核对全部hash/结果；见`CONCLUSION_ZH.md`、`STRATEGY_INDEX_ZH.md`、`strategy_library.json`。当前实验无活跃模拟，不重启；未做跨7交叉、ML/RL或新搜索，未改生产，未Kaggle/Git。旧paused Goal仍未完成，不假标完成。

**分别200策略已启动（2026-09-04）**：`search29_separate7_20260904/run.py check` handle25596 exit0，7对手原Arena14局、clone14局、各首层串并行全部PASS。当前handle50155=`run.py all`，依次单对手搜索→训练8/8经营参数池去重最多200→16未见seed双座位全部复验→2官方局→报告。脚本/PLAN/inputs已冻结，不能边跑改。用户特意强调是200套经营策略，不是工人路径；底层自动。最终4训练seed为20269001..4，未见20269101..16。无新编译，旧生产/S8不改。恢复先查50155及progress，切勿重跑。

**最新授权：每个对手200套获胜经营策略（2026-09-04）**：暂停S8（已选型，最终集未执行），转`search29_separate7_20260904`。用户确认Day0–28/Beam4，明确4搜索seed双座位、每对手200套能赢的不同经营策略、底层执行器自动。仅抽取8/8训练全胜经营参数，去参数重复和无行为改变空变体；不足200如实报告。冻结后16新seed双座位测试全部策略，再每对手2官方局。7对手排除Eco；复用已验收C++，不改生产、不重编译、不做S8切换。当前计划/代码待check/all启动。旧Goal占槽不假标完成，不Git/Kaggle。以本条为准，禁止恢复下方过期会话。

**S8正在选型（2026-09-04最新）**：初始6144完成；10601扩池train已exit0（两开局各6144续跑，合12288），62964浅树训练已exit0。4目标J7开局训练事后227/256=88.7%、最弱20/32，非上线。当前handle8990=`learn.py collect_select`，DP27一半12288续跑184.89秒已完成，J7一半运行中；完成后learn.py select→final_test→verify.py build/official/report。学习脚本与catalog已冻结，不改。verify.py、local_agent.py、online_bridge.cpp为尚未冻结构建的观察输入部署验收入口；真正Python每步动作、特征与C++一致必须验证。最终50新seed未用。旧Goal占位，禁止假完成，不恢复旧任务。

**S8运行中（优先于下方初始条目）**：`switch8_20260904/receipts_v1` build70957/check71432/collect32305均exit0，6144配对续跑完成。DP27固定50.8%→事后77.7%，J7_03固定62.5%→事后84.0%（均256训练世界，非上线）；后者最弱25%→59.4%。`learning_v1/catalog.json`已冻结4计划：原两条＋从旧300计划96世界按补救输局选CROSS_01(QQ041)、CROSS_02(Ocean046)，不是按新最终数据选。84203catalog已结束；正在learn.py collect_train扩到4候选（12288续跑），之后train→collect_select→select→final_test。选型20268101..32，最终20268201..50；尚未动最终集。online_bridge.cpp/local_agent.py为待选型后构建的观察专用入口，未验收。所有新实验独立，不改旧生产，Goal仍旧paused占位新建失败。

**最新授权：S8八对手经营计划拼接（2026-09-04）**：用户已授权开始。目录`switch8_20260904`，复用冻结DP27/J7_03、原控制器与8个完整实时对手；Day1/3/6/9/12/18，训练20268001..16×双座位，两开局两后缀共6144次终局续跑；拟训练小切换树、整局最多一次切换，选型20268101..32、最终20268201..50。先build/check/collect；不得重启下面旧实验。生产和历史收据不改。旧Goal仍paused/unfinished，create_goal已再次失败，不能假完成旧目标，按用户本轮授权落地执行。本轮Goal未注册成功、没有Kaggle/Git。

**七对手联合搜索和独立验收全部完成（2026-09-04，优先于下文）**：`search29_joint7_20260904/receipts_v2/ACCEPTANCE.json`为`COMPLETE_JOINT7_BEAM4_SELECTED_ONE_INDEPENDENT_TEST`。56训练世界、Day0..28、Beam4，4818候选269808续跑128954448transitions，搜索1328.92秒；最终四条训练52/56，选型20267101..50选J7_03（402/700、最弱30%）。冻结计划SHA `921e4dc4cb661fc206135e9c75769d4856580479477d0bf7387ce16c292d8063`。最终20267201..50双座位700局，J7_03各对手G001/G003/B29/K58/L5/六日/三日胜场61/100/22/100/47/36/33，399/700=57%；同库DP27为51/24/96/38/60/46/42，357/700=51%。最弱22%vs24%，未实现更稳健共同策略，不晋级替换DP27。四条最终计划仅第23–25天不同，前23天相同；全部前沿独立评分复核、14官方完整状态及hash复验通过。保存`selected_plan.json`、`RESULT_ZH.md`、`INTERPRETATION_ZH.md`。99005选型、71740final_test+official和close均已结束，不重跑。旧Goal仍paused/unfinished占位，工具新建失败；不假完成旧Goal，实验按用户授权完成。Top40聚类仍暂停，未Kaggle/Git。

**七对手联合搜索已完成，正在新seed选型（2026-09-04）**：有效目录 `search29_joint7_20260904/receipts_v2`；初始v1仅编译失败保留。11696 build、18179 check、4754 search均exit0。29层Beam4搜索1328.92秒、4818候选、269808续跑、128954448transitions；最终4条，最佳训练52/56、最弱6/8；从头训练重放全部一致。audit_scores.py独立重算全部269808结果及排序PASS。当前handle99005=`run.py select`，20267101..50×双座位×7，四条2800局＋DP27基线700。随后 `final_test`（20267201..50只跑选中一条和DP27，共1400）→`official`（七对手首测试seed双座位14局）→`close`。不修改冻结代码/参数，不重新搜索，不在最终测试集重选。旧Goal仍paused/unfinished创建失败，不得假完成旧Goal；本轮按用户授权执行。Top40聚类暂停，不重启。

**最新授权：七实时对手联合共同计划搜索（2026-09-04）**。目录 `search29_joint7_20260904`。排除EcoBot，G001/G003/BoatleeV29/KaitoV58/LynnV5/六日/三日；Day0..28、Beam4，训练20267001..04×双座位×7=56世界，共用宏观语义计划、对手每步实时响应。评分最弱对手胜率→总体胜率→资金差。最终Beam4冻结，20267101..50选型、20267201..50独立测试冠军与DP27，预定14局官方完整状态复验。旧Goal仍paused/unfinished，create_goal因占槽失败，禁止假完成旧目标；本轮按用户授权执行并落地计划。Top40聚类暂停，其输出仅freeze/smoke未开始批量；注意下载器更新过current manifest，已冻结快照与先前口头2210条统计不同，恢复需先确认快照时间。旧生产不改，不Kaggle/Git。当前新程序待编译验收，勿把下方已结束任务重启。

**DP27八对手正式对战已完成（2026-09-04，最新）**：用户要求DP27对本地8个Agent，不改配置、不重新选拳。目录`dp27_arena8_20260904`；`receipts_v1/results.json`状态`COMPLETE_DP27_FROZEN_ARENA8_800_GAMES`。未参与原DP27训练及选版的20266001–20266050，50seed×双座位，每对手100局。胜场：G001 65、G003 33、Boatlee V29 95、Kaito V58 44、Lynn V5 43、EcoBot V7 100、yhay81六日43、三日42，共465/800=58.125%，无平局；只有2/8对手样本胜率达到90%。C++完整对局循环16线程，800局约16.50秒，48.5局/秒，不含33.1秒编译和官方Python预检。16局旧底座控制现金一致；8对手各首seed双座位16局DP27提交入口719动作与C++一致、冻结官方1.32.7完整状态一致。所有进程已结束（5601 build、62881 check、25799 arena）。完整报告`dp27_arena8_20260904/RESULT_ZH.md`。本轮未Git推送、未再提交、未恢复旧Goal；不能宣称DP27已全对手90%。

**本轮Git交接已完成（2026-09-04，覆盖下方旧状态）**：已推送队友仓库`LZhangGJ/Kaggriculture`原分支`agent/add-nt-simulator-orbit-migrations`，commit `30488a38b6c28e3f808f2542db203ab603e9d3be`；11:19:03 UTC之前已用ls-remote确认远端HEAD一致，本地工作树干净。入口`nt/latest_20260904_dynamic_generator_cross300/README_ZH.md`，nt/README已添加索引。包含当前动态执行底座与完整C++搜索源码、153对手必要输入、300参数计划、六组实验1832份原始结果（无损ZIP约120MiB）、报告、可移植命令和DP27提交包。按用户要求不推工具链、编译对象缓存或重复源码快照。包内程序独立编译通过，96局全状态/动作指纹与现金重现、6局官方逐步一致、QQ完整8阶段所有前沿与原结果相同；349个暂存包文件逐blob哈希通过。包共约127MiB。提交DP27 `56011202`已COMPLETE；714.9仅上传初期快照，不能视为稳定分数。没有重新提交、没有恢复旧Goal。后续若继续工作先看本条，不重跑下面已结束的会话。

**DP27已上传Public（最新）**：Submission `56011202`，2026-09-04T10:54:35.833000 UTC，描述`NT DP27 fixed macro dynamic C++ executor - exact 96 game parity`；第一次查询PENDING。包439,921字节、SHA256 `ce5175e916507e86257960e5fb4d9fbf4c7616c83e170d690e4a51f4d1d6c76a`，位于`submission/pending_dp27_fixed_macro_dynamic_20260904/submission.tar.gz`（保留目录名以免破坏验证路径）；上传收据已落地。compat_v1完整96局所有动作/现金一致，官方6局全状态一致，无__file__源码加载及连续跨局重置2局通过；本机单进程最大28.8ms、8进程测试最大43.1ms。62729 parity等编译验证进程均结束。还需检查官方COMPLETE/ERROR；不要重复提交。策略固定的是Day0–28宏观干预，底座每日规划、逐步动态执行。原Git请求被用户插队暂停，尚未push：队友本地分支已ff到c242cba（原远端领先1提交），没有本轮Git提交或包；后续恢复时需重新确认状态。原八对手Goal仍paused。

**用户插队要求先提交DP27，Git推送暂停（最新）**：目标为Driz Lo来源第27条，plan_id `a882fe77385f5e59`，不是重新搜索或固定原子动作回放。已在`submission/pending_dp27_fixed_macro_dynamic_20260904`实现Python ctypes入口＋观察状态C ABI＋原样日候选安装与动态act；完整源代码和计划配置冻结。最初系统GCC13构建通过6场官方逐步一致性，单步最大约29ms，但依赖GLIBC2.38；已保留历史，改用本地解包Ubuntu22.04 GCC11工具链（不改系统）构建成功，最低依赖降为GLIBC2.34，不需Python扩展ABI、libstdc++动态库或AVX。当前正在运行`validate.py smoke --version compat_v1`；通过后需`validate.py parity --version compat_v1`全96局动作/现金与交叉记录严格一致，再打包main.py+agent.so并用Kaggle CLI提交一次、保存收据检查状态。尚未上传，不能说已提交。

**300×3交叉测试已完成（最新，全部进程结束）**：47132 run、4723 official、48359 close均exit0。`search29_cross300/receipts_v1/acceptance.json`状态`COMPLETE_CROSS300_NOT_PROMOTION`；28,800局（复用9,600、新增19,200），新增运行和结果写入555秒，12跨来源官方完整719步一致性PASS。三个对手都≥90%的路线0/300。最均衡Driz Lo #027分别Ocean23/32、Driz27/32、QQ26/32，合计76/96；全部300对各对手单独最佳23/29/32胜。300参数计划在96世界形成299行为签名。逐局事后任取路线覆盖三者均32/32，不等于在线选得对。全部300行CSV、参数JSON、来源平均3×3表及完整报告在`search29_cross300`。用户询问速度与动态性，已确认保存的是固定日宏观干预而非719步动作表：底座每日规划、每步动态act；还存在每日完整generate/preview后只取一个的重复开销，见`PERFORMANCE_NOTE_ZH.md`；没有做成本分解或修改冻结实现。原八对手Goal仍paused，无生产晋级、无线上路由、无新搜索任务。

**已启动300条×三对手交叉复验（最新）**：用户明确授权。`search29_cross300/run_cross.py freeze --version v1` handle63426已exit0，原300条、配置、脚本和复用9600局哈希冻结，6重复局状态/动作一致。当前handle47132运行`run_cross.py run --version v1`，最多16线程，补19,200局，保持原20264501–16双座位；不重新搜索，不改参数。完成后依次official（每个有向跨来源组合第100条、首seed双座位，共12场）→close生成300行CSV/完整计划JSON/报告。未接线上路由、未晋级生产，原八对手Goal仍paused。

**Beam12每对手100条实验已完成（最新）**：16463 validate、88657 official、1845 close均exit0；11,616局新seed复验完成，6场官方719步逐步一致性PASS。验收为`COMPLETE_BEAM12_DAILY29_100_ROUTES_PER_TARGET_NOT_PROMOTION`。报告`search29_beam12_top100/RESULT_ZH.md`。同一新库W4_20/W12_20/W12_100最高胜场分别：OceanMix18/18/22，Driz Lo24/24/27，QQ Farming32/32/32，均分母32；W12百条中≥90%分别0/0/99，全胜0/0/95。QQ训练100条行为不同，新库收敛成94种，不能声称100独立家族。三组百条的事后逐局任选覆盖均32/32，不等于在线胜率。用户问是否300条全部交叉：已说明不是，每个对手自己的100条仅对来源对手测试（9600局，另有对照2016局）；全交叉尚未做，需300×3×32=28800局。没有启动交叉测试、没有生产晋级，原八对手Goal仍paused。所有本轮计算已结束，不要重跑旧handle。

**Beam12三百路线已冻结，正在新seed复验（最新）**：6380 select exit0，各100条训练8/8且训练hash不同；QQ尝试127条、剔除27行为重复。当前handle16463=`search29_beam12_top100/run_w12.py validate --version v1`，全新20264501–16双座位9600新路线＋1920旧Beam4＋96KEEP；先查会话，不重新搜索。随后official（每组第100条6完整局）→close→make_report.py；所有冻结脚本不得改。61950搜索/58776预检均已terminal。报告要分W4_20/W12_20/W12_100，不能混用候选数量与搜索宽度效果。生产不改、原Goal仍paused。

**Beam12搜索已完成，正在取100条（覆盖下方运行快照）**：61950搜索exit0，总1332.6秒，Ocean/Driz/QQ各76400/80040/93120次续跑，三个29层和完整前沿均保存。当前handle6380=`search29_beam12_top100/run_w12.py select --version v1`，每组100条训练8/8且行为不同；先查会话不要重启。后续validate新20264501–16双座位9600＋旧Beam4池1920＋KEEP96，共11616局，然后official第100条六完整局→close→make_report.py。程序和输入已冻结，严禁改run_w12.py/helper/生产源码。make_report.py已准备，须验收完成后运行。原Goal仍paused。

**Beam12＋Day0–28＋每对手100条正在执行（最新，2026-09-04）**：用户要求“321候选＋1–28＋Beam12，取100路线”，随后“继续”。已明确321是旧层峰值，不固定裁剪；保持完整原候选语法和4训练seed双座位，宽度12。独立目录`search29_beam12_top100`，`PLAN_ZH.md`/`run_w12.py`，输出`receipts_v1`。预检58776 exit0：三组首日68候选与旧Beam4全部前沿和前4结果相同，保留数12。当前正式搜索handle61950，先核对progress/会话，勿重复启动或改冻结脚本。完成3组29层后select各100条训练8/8且行为签名不同，共300条；validate全新20264501–16双座位9600局，同时旧29/Beam4冻结20条1920局＋KEEP96局；报告W4_20/W12_20/W12_100分离。随后official第100条×双座位6完整局、close、报告。下一步命令`run_w12.py select/validate/official/close --version v1`，WSL既有venv；不改生产不恢复旧Goal，不Git/Kaggle。全部失败/前沿保留。

**29节点多路线公平复验完成（最新，2026-09-04）**：`search29_top20_multiseed/RESULT_ZH.md`、`receipts_v1/acceptance.json`。用户要求用1–28结果试，同3对手各取20条29节点训练全胜且训练行为不同方案；旧8节点已冻结20条也重跑同一新20264401–16双座位。3936局对照＋6官方完整一致通过。92447选择/95574验证/82347官方全exit0，close和报告完成，无活跃本轮计算。新库各池最高8→29：Ocean15→14/32、Driz14→20/32、QQ32→32/32；池平均26.25→35.47%、38.44→52.97%、97.81→98.75%。QQ全胜17→19条；前两组均未达90%。29节点20条在新库各有20个不同轨迹签名，但开局仍集中（选中Day0+1组合2/2/4）。全部选择预冻结，表中本库最高仍是筛选结果，不能宣称独立确认或通杀。选择23.1秒，3936局验证82.6秒，二进制复用无编译。生产未改、原Goal仍paused、无Git/Kaggle操作；旧报告不覆盖、不重启历史会话。

**8节点多路线新seed复验完成（最新，2026-09-04）**：用户要求至少20条，已按原3个对手各20条，共60条；入口`search8_top20_multiseed/RESULT_ZH.md`和`receipts_v1/acceptance.json`。91398选择、7804验证、42545官方均exit0，close成功，无活跃本轮计算。训练全胜且每组20条训练行为hash不同，60条预冻结；新20264301–16双座位1920局＋96KEEP，6官方完整一致。新库原冠军→池内最好：Ocean8→11/32、Driz12→18/32、QQ32→32/32；QQ17/20条全胜、19/20条≥90%，其他两组0条≥90%。源全胜池493/353/1010条的开局组合只有1/2/224种；抽取20条的开局组合1/2/9，不能把参数变体当20个家族。QQ两条在新库行为收敛，训练库各自不同，报告已注明。新库best是筛选结果，不是额外独立认证。全部参数、逐局/前沿保存，生产未改，原Goal仍paused，不重启旧运行快照。

**Day0＋Day1–28试跑已结束（最新，覆盖下方运行快照）**：`search29_multiseed/RESULT_ZH.md`及`receipts_v1/acceptance.json`完成。63204搜索、52007验证、41220官方均exit0，close成功，无本轮活跃计算。29节点Beam4搜索446.6秒，同新16seed双座位旧8/新29冠军胜率OceanMix7→12/32、Driz16→24/32、QQ32→27/32。480新种子完整局、24训练重放、6官方完整一致通过，生产不改、原Goal仍paused。关键：每个目标仅独立复验冠军，不是所有获胜分支！前沿按参数序列去重复表示后训练8/8方案分别2026/2023/3436条；尚未按实际轨迹去重或全库新seed验证。用户正在追问多条获胜路线，不能以冠军泛化失败推断整个库失败。下一轮若授权，应保留多样代表并与开发/最终验证分离，不重启旧运行快照。

**最新授权：Day1–28宽时间范围搜索，保留Day0**。入口`search29_multiseed/daily.py`，计划`search29_multiseed/PLAN_ZH.md`，输出`search29_multiseed/receipts_v1`。同三条固定路线、同4训练seed双座位、同共享底座、同Beam4，仅决策日变为0..28共29节点。复用search8冻结二进制，不改生产，不恢复暂停原Goal。12世界全KEEP与8计划补KEEP到29节点逐步hash等价通过。当前搜索handle63204（预检69391已exit0），先检查progress/会话，不重复启动。下一步29层搜索→旧8/新29/KEEP同一新16seed（20264201..16）双座位验证→官方六完整轨迹→报告。旧8的全结果保留，不逐seed拼两方案最佳。用户此轮已授权，不受下方“下一步需另行安排”的旧快照限制。

**最新授权已完成：search8_multiseed首次试跑**。见`search8_multiseed/RESULT_ZH.md`。3条代表、8节点、Beam4、4训练seed双座位共同计划搜索144.8秒；训练均8/8，新16seed双座位OceanMix5/32、Driz Lo7/32、QQ Farming32/32。前两条退化，QQ拳只对目标强，不是通杀。384完整验证/交叉局、24训练重放、12全KEEP、串并行隔离及6官方完整轨迹通过。79465/96733/96532/60042全exit0，无活跃旧模拟，不重启。结果`search8_multiseed/receipts_v1`，完整政策/前沿/失败都保留，生产b47源码/registry未改，原Goal仍paused。没有全153搜索、不恢复S5E、不提交。下一步需用户另行安排；旧P不用于本轮，新验证seed20264101..16今后若用于改动即归开发集。

**S5D关闭、S5E隔离剪枝全部通过（2026-09-04，最终恢复入口）**：39854已exit0，不再有本轮活跃模拟。48原样B共享局1392个真实首次compile，完整计划/意图/当前单位和市场动作等价；预演8854→174，仅比较时间17.885→3.571秒。不是正式整局加速、不是新胜率。见`reports/S5E_STATIC_PRUNE_REVIEW_ZH.md`。生产仍b47dc58且源码/registry未变；`compile_static_prune_prototype.hpp`不在正式include链。下一步先冻结生产，再以默认关独立性能开关接入，保留原候选顺序与1e-6规则，A不得静态剪枝；做三个B背景的真实逐动作/状态等价及整局吞吐。S5D完整总结`S5D_ROUND_CONCLUSION_ZH.md`，最好开发B40.625%仍未晋级；P未用。S4Q/S4R空地延续问题早已测试，不重新包装成新发现。原Goal仍全八逐一90%，未完成。69252/92005/67095/81906/39854全部terminal，不重启。

**S5E剪枝原型通过，真实状态等价在跑（2026-09-04，最新）**：81906已exit0，48构造场景194断言，原228次预演→12，选择/意图相同；未接生产，不是整局吞吐。当前handle39854=`audit_s5e_static_prune.py`，隔离probe与b47 ABI匹配；48原样B共享对局、每个真实首次compile检查原/剪枝选中的完整计划及当前单位/市场动作，要求最终双方现金复现。先核对handle，不重启。其结果在`receipts/s5e_static_prune_live_v1`。生产源码仍未改，正式build/registry b47；之前69252/92005/67095/81906均terminal。S5D完整总结已落地`reports/S5D_ROUND_CONCLUSION_ZH.md`，不晋级、P未用、原Goal未完成。

**S5D正式结束（2026-09-04，最新）**：handle69252已exit0，2160完整实时/540旧控制/1620新账本全部相符；`close_s5d_round.py`已完成，阶段不晋级。48场B行为审计handle92005 exit0，73次改动=71延期+2纯重排；81组现金渠道审计handle67095 exit0，108原账本全部核对。报告 `S5D_FIRST_COMPILE_RESULTS_ZH.md` / `S5D_LABOUR_SALES_REVIEW_ZH.md`。140项S5E估值覆盖隔离测试通过；空地价值遗漏是S4Q/S4R已知缺口，不要再次包装成新发现或重复7200场跨日测试。生产build仍b47dc58，生产源码未改，P未用。当前只有handle81906在跑 `test_s5e_static_prune.py`（隔离原型，不是正式策略）：先实查结果；目标是B按原顺序跳过静态评分不可能胜出的候选预演，不能套给A。通过后仍需真实状态/逐动作等价与速度验证再接生产开关。所有更早handle均已terminal。Goal全八逐一90%未完成。

**S5D面板进行中（2026-09-04，最新）**：handle69252 / WSL367父进程与578面板进程已实查活跃，不是旧锁文件。32官方完整＋6隔离全通过，registry已绑定b47dc58。已读1480/2160完整局；共享KEEP/A/B/AB为59/45/65/56胜（各160），完整自主56/53/62/55胜；B平均+3.75pp但两个seed聚类区间均含0，不晋级。第三背景仍在跑，之后1620新账本；不要改源码/重跑。详见 `reports/S5D_PROGRESS_ZH.md`。结束后 `close_s5d_round.py`；再 `audit_s5d_choice_effects.py`（48原样C++对局，分类B实际是否延期，不改策略，要求与面板现金一致）。所有当前ABI外的旧probe勿加载。P未用，Goal未完成。

**S5D正在官方/完整面板验收（2026-09-04）**：生产build `b47dc58b60ffcc8261c53fc99fa0b7a637b9c44b5ab3deeda031e001fc80c2ca`，冻结 `profiles/s5d_bounded_v2`。runtime v4共24完整串行局通过，最大约0.325秒；旧机制、共享插入、重复领料、交接均通过。handle69252 / WSL367=`run_s5d_round.py`运行中，官方检查逐个进行，之后2160实时及1620新增账本；先核对该handle/进程，严禁重复启动或改运行源码。540旧控制必须与S5B完全相同。结束后执行`close_s5d_round.py`。还没有S5D完整胜率，不能晋级；P未用，Goal未完成。下面91539/81021均已成功退出，runtime v1–v3为保留的失败分支。

**S5D时延修复，正在重建（2026-09-04）**：mechanism v6 247断言通过；增量插入10,000排班/170,336断言完全等价。runtime v1/v2/v3因218/242/360步时延终止，不得重启旧handle。只读cost v5确认360步约91%成本在换种嵌套首次编排。新增显式有界推演（预测副本禁嵌套明日规划/首次排班，不删真实功能）及首次compiled tick继续位置修正。源码已变，handle91539为当前build，先核对；随后冻结`profiles/s5d_bounded_v2`，runtime v4完成前不得启动`run_s5d_round.py`。旧build/分支全部留档，registry仍S5B；尚无S5D胜率结论，P未用，Goal未完成。

**S5D机制排错中（2026-09-04）**：`compile_consequence` / `compile_replant_choices` 默认关已接入源码，匹配旧生产的完整源码已冻结 `profiles/s5d/before`。新 Params/Controller 布局与旧 build 不兼容，重建前严禁加载旧 ABI 探针。机制v1已终止FAIL_PRESERVED，报 duplicate shared physical effect；先定位是否同地块旧作物WATER→HARVEST→PLANT→新作物WATER被错误合并，不能为过测试删掉合法动作。当前无活跃旧模拟/编译进程；正式build仍ee733，registry未变；尚无S5D对战结论，P未用，Goal未完成。S5C以下内容为此前快照。

**S5C完成，下一步首次编排×可延期补种（2026-09-04）**：4945/60237/45573/16400均exit0，不重跑。6场逐步只读轨迹和6场条件探针重复均与原面板完全一致；10真实时点68候选，只是条件预测。具体见 `reports/S5C_COMPILE_OVERLOAD_REVIEW_ZH.md`。远端维护在任务生成/资源预留中存在，却在首次compile丢掉；8个非现金崩塌时点存在不漏喂替代编排，但现有余值评分没有可靠选出，不能认为已提升胜率。首次`value_schedule`仍是静态任务分，不等同于后续recoordinate的日后果比较；含PLANT的收获/补种链仍合并。下一步按报告第5节独立A（首次后果比较）、B（可延期补种）及A×B做通用机制和真实对战验证；另查现金崩塌的投资/用工融资。不准硬编码6个失败场景或强制所有动物永不退出。生产仍ee733、源码未改、新功能未晋级、P未用、Goal未完成。

**S5B全部结束；S5C只读因果定位（2026-09-04）**：S5B实时1440、新增720账本、32官方和6隔离均完成；`s5b_stage_acceptance_v1`拒绝晋级，所有下方S5B运行handle已terminal，不得重启。当前正式build仍`ee73398ec418b973695c16c255f2eab78ecc613f6fddfb5995704d48420e1608`，新换种开关默认关。12局延期追踪逐场与面板相同，观察到同地块连续空置延期8–11天，但并未证明所有等待亏损。`s5b_escape_triage_v1`确认6局共12只非计划逃跑，原对照这些局均为0：既有现金耗尽型，也有现金/小麦充足而漏排维护型。下一步S5C只读逐步审计任务生成、资源预留、编排和实际执行，不能一律追加现金buffer，也不能硬编码最大延期天数。未使用P，不Git/Kaggle，Goal仍active。

**S5B已完成1440实时、正在账本（优先2026-09-04）**：修复后的build `ee73398ec418b973695c16c255f2eab78ecc613f6fddfb5995704d48420e1608`；mechanisms v3通过，runtime v2原报错场景及12局均完成（原生最大0.402秒）。32官方完整+6隔离通过，registry已同build更新。`run_s5b_round.py --version v2` handle36936仍在720新增配置账本，最近30/36对手单元完成；不可重启。1440实战已完：共享基线36.875%，net36.875%，timing32.5%；完整自主35%，旧轮作17.5%，net26.25%，timing21.25%。均不晋级、P未用。账本完成后运行 `close_s5b_round.py`，再运行只读 `audit_s5b_wait_execution.py`（12原样局），不要与16线程批测同时跑。只读已有意图审计 `s5b_wait_intent_audit_v1` 发现一个开发seed两背景分别36/31次同地块同作物继续延期，尚需实际空地核对，不能直接说都是坏等待。旧failure与首次f7067f构建全部保留；90761/94120已exit0。Goal仍active，未达全八90%。

**S5B正在接入（2026-09-04）**：两个默认关开关 `portfolio_rotation` / `rotation_timing` 已实现；原型统一为 `rotation_calendar.hpp`，旧prototype仅兼容include，避免重复所有者。首次构建f7067f在完整自主时机/G003/20262701/seat0/step144报错，已完整复现并保存 `s5b_runtime_failure_v1`；原因和修复见 `S5B_FAILURE_AND_REPAIR_ZH.md`。机制v3 12562+26+17通过；正在重新编译handle90761，之后须冻结after_v2并用 `probe_s5b_runtime.py --version v2` 跑同原场景。原runtime78782和execution v1均失败终止，不可当完成。生产源码已修改，旧S4V保留在build_history及profiles/s5b/before，正式对手registry尚为S4V，需同新build官方验收后才更新。不要加载旧ABI探针；新Goal目标未完成，P未用。

**S5A隔离原型完成（2026-09-04）**：用户补充高手转产关键在时机，研究范围必须包含现在/延期/保持/停止新投入，不得将低羊毛价硬编码为转草莓。新增 `native/rotation_portfolio_prototype.hpp`，未接生产源码。混合旧收获/新种子按商品记账、唯一地块替换、全农场资源净额、延期空地投入等机制v3共12,562断言通过；v1编译错误与v2/v3快照保留。详见 `reports/S5A_ROTATION_TIMING_PROTOTYPE_ZH.md`。本轮没有新胜率、没有上线，生产仍S4V，所有本轮进程已结束。下一步先真实盘面估值审计，再统一尚未成交的新项目、当前准备/人员可行性与延期计划状态；不能把隔离预测器直接当完整转产策略。Goal仍active，P未用。

**S4Z完成，不重启（2026-09-04）**：handle73863 exit0，17机制、720实时、360新配置重复账本完成，360旧控制逐场与S4V一致。`s4z_stage_acceptance_v1`已关闭，不晋级；新配置无效动作/动物逃跑均0，但未做新配置官方/最终线上验收。10开发seed小面板：共享轮作PASS189894→192916、全八平均36.875→31.25%；完整自主PASS163072→185757、平均35→17.5%。见 `S4Z_HARVEST_ROTATION_REVIEW_ZH.md`。草莓增收被采购和其他产业损失抵消，下一项应统一轮作与空地投资的完整净现金/义务口径，而不是继续扫偏好。生产正式构建仍S4V、源码未改。所有本轮进程已完成，Goal仍active，P未用。

**S4Y完成、S4Z正在筛查（2026-09-04）**：S4Y64原样实时局通过，见 `S4Y_INVESTMENT_CHOICE_REVIEW_ZH.md`。完整自主中期320个时点只有15个有空地，两个背景原`rotate_finite`默认false，成熟短周期作物无法从该开关重选下一产业。本次不改生产源码，只冻结 `profiles/s4z` 四配置（两背景×轮作开关）；专用机制已PASS，handle73863=`run_s4z_screen.py`正在720局小面板，之后360新配置账本。先检查此handle/实际进程，勿重复启动。结果不是独立晋级，P未用，未保证胜率。旧S4Y25083已exit0，无需重跑。正式构建仍S4V。

**最新研究结论（覆盖下方S4X下一轮顺序）**：`s4x_production_gap_v2` 复用1600局进一步拆产出/销售并通过商品守恒。共享对G003草莓种植27.94 vs36.08、产出207.70 vs259.24、单次种植产出7.43 vs7.19；我方损失+终局残余仅5.71。七个难对手均有草莓种植量缺口。因此优先核查现有投资/容量估值为何少启动高价值项目，不能把纯抢卖或物流小补丁当主突破，也不能固定多种8棵。完整研究第8节/下一轮计划已更新。S4W v3 handle35116 exit0：27机制通过，4原失败局原样复现；80观察点可行候选0（含首动作前编译预览），未修复真实漏喂，不接生产、不大面板，详见 `S4W_REPLACEMENT_PROTOTYPE_REVIEW_ZH.md`。本轮所有编译/模拟已结束，无待恢复实验。Goal仍active，所有90%门均未完成。

**S4X已完成，不重复启动**：handle44954 exit0，64完整局约73.8秒，全部双方现金等于冻结S4V面板，`receipts/s4x_cash_window_audit_v1/acceptance.json` 及 `s4x_cash_window_summary_v1` 已落地。共享草莓单位价格预测MAE9.68，对照现价不变9.10；自主草莓17.98 vs13.89、牛奶20.80 vs17.20。只有两个旧开发seed，不是强度验收或因果利润。详见更新后的研究报告第6–7节。运行策略未改。下一轮优先市场窗口×收获物流×部分出售/融资的共同后果，不再把只插入SELL或只省工时当主突破。

**最新：S4V 已关闭，S4W 原型未通过，S4X 只读研究开始（2026-09-04）**。S4V 的 10,800 对局、8,100 新配置账本、旧控制复现及官方/隔离验收已完成；`s4v_stage_acceptance_v1` 拒绝晋级，详见 `reports/S4V_FULL_REVIEW_ZH.md`。原 handle12098 exit0，等待器98456 exit1（S4W 机制失败），不要恢复下方历史 PID。S4W v1/v2 失败各自保留；v2 失败断言为临终未成熟动物不应被选中维护，需核实副产物与评分，不可直接认定实现错误。两个隔离探针 v2 已编译完成。S4X 将核对64局成熟产品未排程及原出售模型的预测误差，不改变动作，不是 Oracle。研究主线见 `reports/WINRATE_IMPROVEMENT_RESEARCH_20260904_ZH.md`。正式源码/构建未改，Goal 仍 active，未达到任何全池90%目标。以下旧进程叙述全部是历史快照。

**最新恢复入口（S4V面板完成、S4W隔离原型已编译）**：2026-09-04已确认handle12098仍活着，WSL父291=`run_s4v_round.py`、子691=`run_pool_audit.py`，不是历史S4T的同号进程。10,800强度局与配对分析已完成；旧2,700控制逐局一致；8,100新配置账本还在运行，最近已完成共享背景，进入完整自主背景。原/共享/完整自主三个背景四组合结果见 `receipts/s4v_comparison_N50_v1/TABLES_ZH.md`：最佳新共享补喂39.5%平均（原39.25%），G00316→18胜；完整自主市场双开33.375%（原33%），G00136→38、Kaito20→18。不稳定、不晋级、未用P，Goal仍active。

**S4W不要重复启动**：隔离文件 `native/service_replacement_prototype.hpp`、`native/test_service_replacement.cpp`、`native/service_replacement_probe.cpp` 已编译，`native/service_replacement_probe_build_v1/build_receipt.json`存在，compile handle47233已exit0。没有改生产源码/ABI，原型尚未执行验收。候选允许撤回同一漏喂动物的CARE再插入FEED，按条件净现金比较，不强制救动物。`tools/run_s4w_after_s4v.py` handle98456 / WSL739已启动并核对真实S4V进程等待，不占额外模拟线程；S4V自然结束后依次 `close_s4v_stage.py` 写完整复盘和拒绝晋级收据，再 `probe_service_replacement.py` 做机制与4原失败局只读诊断。先读同handle、`receipts/s4w_execution_v1`和实际进程。若失败，保留failure，查具体约束，不重复覆盖out/build目录。不得混用旧ABI探针。下面7600条目是旧快照。

**S4V当前（2026-09-04，优先于所有下方旧恢复文字）**：S4U流水线handle7227已exit0，4500实时/2700账本/32官方/6隔离完成，完整自主组合仍退步，不晋级；出现两个seed双座位共4次羊逃跑。96场只读市场审计完成，1107次建议漏执行多数在日初订单队列；不是已证明可获利机会。4个逃跑局逐步原样复现：养羊目标仍在，有小麦，CARE存在但FEED缺失，归非计划漏维护。

本次最新检查：S4V22新/既有机制、48时延、32官方及6隔离全部完成，registry已按同build推进。handle12098仍在panel，已读7600/10800局；共享背景FEED插入对G00316→18/100，其余已完成胜率无变化，不晋级。原4逃跑局仍失败；只读排除探针63268已exit0，证明大多数插入超时，少量缺随身材料，不能称已修好。详细 `reports/S4V_PROGRESS_ZH.md`。下一轮需要任务替换/可撤销的低价值工作候选，不只增加无损插入点。旧S4U ABI探针不要混用S4V；本轮新`feed_insertion_probe`绑定S4V。不得提前编辑生产源码破坏正在运行的冻结批测。

已新增两个默认关、独立开关`continuous_market_execution`（不挤占订单/不乱融资顺序的当前出售建议落实）与`insert_missing_feed`（当前资源足够、保留原任务顺序和期限的跨工人任意边界插入）；22机制检查通过。build `074c98bd83cf8aa47cd227f050033dbb05d4617dac6b137bb3dbb9c7ae40debd`，冻结 `profiles/s4v`。`tools/run_s4v_round.py` handle12098正在进行既有机制→原4败局→48场时延→32官方/6隔离→10800实时（3背景×4组合×9对手×100）→8100新配置账本。**先检查同handle/进程，勿重复启动；registry只在同build官方验收后自动推进。**当前尚无新强度结论，Goal仍active，P未用，不Git/Kaggle。新增探针`plan_execution_probe`及旧`market_execution_probe`绑定S4U旧ABI，禁止与S4V新模块混用。

下一项等实际结果判断，不按“只要补漏一定加钱”晋级。详细预注册 `reports/S4V_MARKET_AND_FEED_INSERTION_PRE_REGISTER_ZH.md`；S4U完整结果 `reports/S4U_FULL_CHAIN_AND_EXACT_SPEED_REVIEW_ZH.md`。下方S4T暂停/S4U待验文字是历史，不得据此恢复旧进程。

**S4U最新恢复入口（2026-09-04）**：S4T父291/子563已经为了纯性能升级主动终止，3600局保留；不是继续暂停，不要SIGCONT或重启旧S4T。原handle4775已终止，build handle49900已消失，正式新build收据存在（28.12秒）。已接入默认关 `exact_schedule_cache` 与 `incremental_regret_cost`，新Controller ABI禁止加载旧 `_dp7_packmemo` 原型模块。输入冻结 `profiles/s4u/input.json`，旧源码与3600局已备份。原型12完整对照每步一致，按天缓存+增量排程约减少58%～60%计算，仅性能证据不是胜率。现在先运行 `validate_s4u_production.py`，验证新正式模块112组完整对照＋重置，再冻结后构建源码、32官方＋6隔离、时延、4500实时与2700重复账本，核对旧3600逐局结果。完整自主上层＋下层尚未获得全池胜率，不能宣布成功。后续下面S4T暂挂文字仅历史；Goal仍active。

**最新调度：S4T昂贵批测已暂挂，不是终止/失败**。用户指出效率太差，优先修计算膨胀。已核对并SIGSTOP WSL父291、面板563，二者STAT T，保留内存与现有结果；handle4775。恢复前必须检查进程身份，若仍是对应脚本用SIGCONT，绝不盲重启。暂时不跑后续2700账本。正式源码/build仍冻结S4S；新准备的 `build_pack_cost_probe.py` 在隔离复制树 `native/s4u_cost_probe_v1` 插计时和精确重复输入计数，不修改正式运行源码。编译handle76169需核实，之后运行 `profile_s4t_pack.py`（4真实局，24个已知决策点，比较原动作/终局不变），据真实重复率再做保行为缓存。不要把暂挂误认为实验完结或永久放弃全开验收。

S4T补充性能诊断已完成：`reports/S4T_LATENCY_DIAGNOSIS_ZH.md`，8场外部计时不改动作，进程10221已exit0。最慢step211跨日估值两个座位重复1.85/1.86秒，CPU实算；源码可见两方案→下一日组合→逐日工资→逐个人数→regret排程嵌套。没有发现无限自递归，不应称模拟器一步1.87秒。正式S4T流水线handle4775仍需检查；不要重复启动 `run_s4t_round.py` 或覆盖既有收据。配套渠道汇总工具 `summarize_s4t_channels.py` 已写，需等2700账本完成再运行。

**S4T当前任务：整链共同开启**（2026-09-04用户明确要求，优先于下方旧下一步）。用户认为只开几个可能没激活完整链条，并要求包括细节多人协作、经营计划与底层规划器同步进化。冻结五配置 `profiles/s4t`，主实验先比较全开工人25项、完整关联链、完整自主上层＋下层，不能先以旧单项失败排除能力。源码仍为S4S，无重编译。20场原生探针已完成，最完整版本单步最大约1.87秒，时限不通过；仅继续离线研究、不晋级。计划32官方＋6隔离→4500实时→2700账本；状态看 `receipts/s4t_execution_v1`，入口 `tools/run_s4t_round.py`。补充边界 `reports/S4T_JOINT_EVOLUTION_ADDENDUM_ZH.md`。Goal仍active。

**S4S已完成，不重跑**（2026-09-04）：`reports/S4S_IDLE_TASK_HANDOFF_REVIEW_ZH.md`。新增默认关`idle_task_handoff`，21机制＋旧机制、32官方、6隔离通过；build `87584dd12bb4d12d435b07103d9ee79498ae2f53c8d8c41dc5ba9e3f0485b43c`，registry已更新。3600实时、1800旧对照复现、1800新账本、128配对日界重复完成，进程60847/20012/48352均exit0。八对手胜率开关前后全部未变（36.75%/39.25%），不晋级、不用P。87次接手分布79局—日：79个日末地块全相同，52个完整可见物理终点也相同。说明本轮多数只是提前已会完成工作，不是新增生产能力。下一步应查多seed中有价值项目未兑现、资金/材料/联合排程约束及变现—再投资堵点，不再单纯提高动作数；Goal仍active。

**S4R已完成，不重跑**（2026-09-04）：`reports/S4R_CROSS_DAY_CONTINUATION_REVIEW_ZH.md`。默认关`day_value_replan_next_day`，38新机制、全部旧回归、32官方及6隔离通过，registry已更新。build `f13dfd56faa87bd684cf0fc1ff9a7ae03fbefbcf1c82a4435793a10ac98d9b73`。7200实时、3600旧控制复现、3600重复账本完成，进程92025已确认exit0。阶段收据 `s4r_stage_acceptance_v1`。最佳新配置平均38%，未超过共享39.25%；不晋级，P未用。32原生时延最大19.447ms，不是最终线上验证。下一项S4S检查真实多人同回合前置接力：先调查频率并核对官方Python，不能从一个构造例推定主要败因或直接写策略。原Goal仍active。下面S4Q已完成，不重跑。

**S4Q完成，不重跑**（2026-09-04）：`reports/S4Q_DAY_RESIDUAL_VALUE_REVIEW_ZH.md`。新增默认关`day_consequence_compare`和`day_value_public_supply`，比较真实观察下保留/重排的日内PASS情景及剩余现有项目条件价值；未用真实未来或Oracle。build `2ce7cd81e4741c16ada9fefd4fca00b43d661a215842e93c07f1fc36a55c1f0a`，冻结`profiles/s4q`。32官方、6隔离通过，registry已指同build；EcoBot首次因独立probe过期头文件哈希失败，已重建补验，失败不删除。`s4q_prerequisites_v2`为正式补完入口。

5400实时、1800旧控制逐场一致、3600重复完整账本、64逐步选择审计均完成；动作无效/动物逃跑0。六配置胜率见报告：基线36.75%→估值37.125%；共享39.25%→估值37.625%，Six-Day退化，不晋级。公开产能改变部分现金但胜负无变化。24原生动作探针最大9.69ms，不是线上部署验收。全部本轮进程43402/44938/24655/62261/64186均已成功终止；初次94256失败已补验，勿重启旧脚本覆盖收据。`s4q_stage_acceptance_v1`已落地；Goal仍active，P未用。

64场八对手原样实时重跑中1994次真实选择、554次否决、418次日末规模不同；选中安排预测日末规模仅21次不符。因此不能继续笼统归因“当天执行差”。**下一项待验证结构缺口**：当前估值把在田项目按同作物续种到终局计价，却不给暂空地块、现金和人力的后续替代项目计价，也没有下一日完整经济重规划；可能把短暂规模差当永久收益差。先核对真实下一日目标/任务/资金，受控验证收获后继续种、暂缓再种与真正停止的区别，再修跨日条件延续。它只是优先假设，未证明全部败因；禁止直接加固定作物/日期/金额阈值，也不恢复事后Oracle。下面S4P及更早内容为历史。

**S4O/S4P已完成，不重跑**（2026-09-04）：`reports/S4P_DAY_CONSEQUENCE_REVIEW_ZH.md`。S4O直接仓边融资机会仅128局中1局，不追加该补丁。S4P新增 `observed_day_scenario.hpp` 和独立 `day_schedule_probe.cpp`，仅由View重建当日条件情景，对手PASS、未知新杂草/商店不预测，禁止下一天决策。原规则函数未改；正式二进制仍 `09391dd4dcda373d29736038cf4dbfad17dd3a5da1242b160f9e065ce5a551eb`，未接入在线选择、未晋级。

机制v3：1,920双座位窗口/28,736转移/3,641,616字段断言通过；64实时重复与原记录双方现金相同。5,568观察点中718次排程不同，521次日末状态和现金相同；197次实体状态不同，其中119次生产规模不同。没有同实体终点只增加现金的样本；6次现金与现金加库存报价方向相反。情景均值0.20–0.26ms、观察最大2.52ms，不是线上时延验收。本轮进程均正常结束。冻结 `profiles/s4p`、`receipts/s4p_stage_acceptance_v1`。O已读、P未用；本轮未新增强度选型或未见集晋级。

下一步不能直接按日末现金/清仓报价择优：需要给日内后果的库存、在田生产、资源义务继续计价，并给销售变化增加只依赖对手公开产能的条件市场压力。之后接独立选择开关、做八对手实时独立/组合消融；禁止恢复真实未来Oracle。当前Agent强度没有因为诊断模块完成而增加。下面段落为旧阶段证据。

**S4N/S4N1已完成，不重跑**：`reports/S4N1_MARGINAL_WORKFORCE_REVIEW_ZH.md`。前置128逐步实时、128完整账本重复、3840日界匹配；64组中62组首次市场分歧为日内小额买种子。新增独立默认关`intraday_future_workforce`，用同地块日期任务集差分工资替代粗略单项目动作费；保留交易/调度原样。build `09391dd4dcda373d29736038cf4dbfad17dd3a5da1242b160f9e065ce5a551eb`，冻结`profiles/s4n1`。37新、64共享、4490旧、27恢复、42领料机制和32官方完整、6隔离均通过。registry已推进有凭据的新build。

本轮所有进程已终止成功：N3600、O1800实时；N1800＋O1800完整账本重复；16原生时延最大9.51ms；动作无效/逃跑记录0。`s4n1_stage_acceptance_v1`及复盘已落地，原基线不替换。N共享组合G00143→44、Boat45→47、Kaito31→33，O只有G00142→44，其余不变；O静止现金172572→170539。O已读取，后续调参则归开发，**P未用**。O亏损账：奶收入-2498，其中成交均价部分-2121（算术拆分不是因果）；麦采购+532，工资+12。下一方向是入库/销售/融资的可核对短期后果比较，不继续调劳动权重；不要重复只防溢仓的S4C。必须先有多seed瓶颈证据，不恢复Candidate Oracle。下面S4M1为历史。

**S4M1完成，不重跑**：`shared_service_insertions` 默认关；冻结 `profiles/s4m1`，build `0c274aa2e653ce180d2f705e77a481f3a1581214605886dc06161f73567741a4`，registry已指新官方/隔离收据。64+4490+27+42机制、32官方完整、7200实时、4500账本、32原生时延均完成；2700旧控制逐场复现S4L。新动作无效/动物逃跑均0，原生最大41.43ms，不是最终Python部署保证。无运行中的本轮测试。

最新完整复盘 `reports/S4M1_SHARED_INSERTION_REVIEW_ZH.md`，验收 `receipts/s4m1_stage_acceptance_v1`。单独插入基本不加分；叠加日内生产后G00145→43%、G00314→16%、Boatlee30→45%、Kaito26→31%、Lynn22→33%、Six-Day28→20%、Three-Day29→26%、Eco100→100%。不晋级、不启用O/P，保留全部独立/组合分支。调度×生产交互及36组现金/工人账见 `s4m1_interaction_N50_v1` 和 `s4m1_cash_labour_N50_v1`。G001少40.8MOVE、少4HARVEST、销售多6270；Six-Day多0.5项目、多1.5HARVEST、工资多519但销售少3189，必须合并分析，不能优化动作计数。

下一步：同一任务集的经济打分不区分入库/销售/融资时机，插入仍以峰值/总步数比较。先从已发生的正负组合中定位真实时序及资源约束差异，验证通用经济后果缺项，再独立开关改进；不凭固定“提前奖励”、对手身份或单seed最优顺序加规则。四项大能力尚未完整解决，Goal仍active。以下S4L和更旧段落只是历史证据。

**S4L优先**：`incremental_pickup_repair` 默认关；build e08850a5575293d365537230efb4da6d199e8c254c723b541de25e136a26dde0，registry已指向S4L。42+27+4490机制、7200实时、3600审计、32官方、32时延全部完成；3600组开关对照终局现金/溢仓逐局相同，不晋级。无运行中的测试。最新复盘 `reports/S4L_PICKUP_AND_JOINT_LABOUR_REVIEW_ZH.md`。新增36组联合工人/产销/现金审计，以及16场全部719步真实轨迹，均与原面板匹配；下一步重点分析工人调度与产销的联合缺口，不能只省移动、强制补喂或只怪卖价。入口 `receipts/s4l_labor_cash_chain_v1`、`s4l_competitive_cash_gap_v1`、`s4l_service_trace_v1`。不重跑已经结束的S4L。以下S4K2为历史证据。

S4K新增缺料维护重接/采购/融资三级开关，但误删终局DROP→SELL。S4K2只修保护范围，已完成相同八配置的7200实时、5400重复账本、32官方样本、27新机制及4490旧机制。现金恢复，但新增能力未突破：保留对照G00145%/G00314%，恢复采购组合43%/14%；自主组合43%/4%。不晋级，不用O/P。

**下一步优先修已复现的硬伤**：`repair_plan` / `semantic_needed(PICKUP)` 会因当前携带1份小麦，删掉“先喂一头→再领1份→喂另一头”的后续PICKUP。专项原计划喂2、重排喂1。收据 `receipts/s4k2_repeated_pickup_reproduction_v1/acceptance.json`，测试 `native/test_repeated_input_pickup.cpp`。先修顺序资源因果，再机制回归与多seed实时复验，不能跳回Oracle。

当前build为bb0cd92f54570bae2fd106f5a274f8917a42a60d9f85cd6f4e8143288e9dec62，registry已是S4K2。无运行中测试。运行源码最后改动只有service资源保护范围；新增领料失败测试未修改运行代码。

- S4K失败复盘：`reports/S4K_SERVICE_REENTRY_FAILURE_ZH.md`
- S4K2预注册：`reports/S4K2_TRANSACTION_PROTECTION_FIX_PRE_REGISTER_ZH.md`
- 最新完成复盘：`reports/S4K2_SERVICE_REENTRY_REVIEW_ZH.md`
- 旧保留对照还是S4J2/all_intraday。S4K开关不能晋级。

## 唯一当前目标

从真实败局改进自主动态规划器和执行器；四项执行能力独立开关及组合检验；C++最多16线程、多seed、双座位、实时原始对手。一个冻结Agent分别超过 G001、G003、BoatleeV29、KaitoV58、LynnV5、EcoBotV7、Six-Day、Three-Day 90%胜率，且完成官方规则和线上时延验收。

禁止恢复旧 Candidate 事后 Oracle、真实未来后缀选择、单seed最佳路线、对手身份特调。不能因为压缩上下文而切回那些旧任务。不得提交Kaggle或推Git，除非另获用户明确指令。

## 先前保留对照

S4J2新旧作物条件日历统一：120受控整局/127,475检查、4,490既有机制、32官方对照、48时延探针通过；10,800实时面板，3,600重复真实账本全部闭合。新日历帮助失败的组合规划，但没有全面超过保留对照；不晋级。

同四项执行背景，对G001：保留对照45%，新固定开局组合42%，新自主开局组合43%；G003对应14%、12%、10%。每格N50双座位，不是未见集。其余逐对手结果见最新复盘，不能按三项平均概括全面提升。

- 最新复盘：`reports/S4J2_UNIFIED_CROP_CALENDAR_REVIEW_ZH.md`
- 冻结源码/配置：`profiles/s4j2/`
- 验收：`receipts/s4j2_stage_acceptance_v1/acceptance.json`
- 当前编译SHA：`7c51e45fb87cee89863e74ba9be96ca894a21373080bdf781d00c8d5775ca8ac`
- 对手registry已指向该build的官方与隔离收据，旧指针保留。
- 本轮开发N=20262701–20262750。O=20262801–20262850、P=20262901–20262950尚未使用。
- 旧对照 `intraday_funded`、`all_intraday` 保留，不能为了新方案失败而删除；新 `joint_investment_portfolio` / `portfolio_crop_calendar` 默认均关。

## 后续方向，不是已经证明的结论

检查条件预测与实际生产/现金兑现的差距，再据真实多局证据修正通用能力。不得重复把改现金权重、加续种总量、扩大beam、固定高手日历当成新突破；S3K、S3O、S3P等已留有失败实测。

此文件是项目恢复说明，不是Memory更新或新的自动化任务。
