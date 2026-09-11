# KEEP=2 四模型1000轮：结论、验收与交付索引

## 结论

四组均从300轮权重及Adam状态连续训练到1000轮。没有因中途退步停训、换回最佳模型继续训练，或更改网络、候选、特征、奖励、KEEP强度、执行器及对手池。

最终独立复验共40,600局：100个未见环境seed × 7个实时对手 × 双座位 × 29个模型/决策配置。训练和开发选模在终测之前结束；本报告不按终测重新选择参数或部署Agent。

|模型|1000轮确定性胜率|1000轮采样胜率|确定性平均现金|开发选定轮数|开发选定模型确定性胜率|
|---|---:|---:|---:|---:|---:|
|KEEP基座|62.93%|不适用|103,524|不适用|不适用|
|普通PPO重复1|61.79%|62.71%|102,658|200|65.36%|
|普通PPO重复2|64.21%|63.57%|103,817|180|62.29%|
|辅助PPO重复1|71.57%|68.64%|105,927|640|71.71%|
|辅助PPO重复2|64.64%|64.07%|104,081|780|65.64%|

实事求是的解释：

- 辅助第一组在未见seed上仍明显强于KEEP：1000轮确定性胜率增加8.64个百分点，单项、未作多重比较校正的95%配对seed区间为[+4.14,+13.21]。但辅助第二组相对KEEP的区间包含零，不能说两次训练都已稳定变强。
- 训练延长至1000轮并非持续增长：辅助第一组300轮模型在同批独立seed上已有70.14%，1000轮为71.57%，差值区间[-2.36,+5.21]个百分点。1000相对900的八项单组比较区间全部包含零。
- 普通两组1000轮确定性的描述性平均为63.00%，接近KEEP的62.93%。辅助两组为68.11%，但只有两次训练重复，不足以精确估计跨训练随机性的稳定性。
- 胜率和平均现金不是同一个指标。辅助640轮模型胜率71.71%、平均现金103,111；其1000轮模型胜率71.57%、现金105,927。不能单凭现金高低替代胜负评估。
- 最强一组仍有对手短板：1000轮确定性对G003为54.00%、Six-Day为50.50%。没有证明全对手高胜率，更没有线上强度或公开提交验收。

两次重复的平均值不是组合Agent，也没有逐局挑选模型。配对区间以环境seed为分组单位，保留同seed不同对手/座位的相关性；本次有多个比较，单项区间不等于全族置信保证。

## 原目标逐项验收

|原要求|实际证据与范围|
|---|---|
|四模型300→1000连续训练|七阶段各四组100轮，每轮224局；各阶段 `training/*/COMPLETE.json` 和 `FINAL_AUDIT.json` 逐轮检查输入模型、输出模型及对局记录|
|恢复模型及Adam而非重开|各阶段 `G0_ACCEPTANCE.json` 检查完整恢复及下一次更新一致；Adam从16,800连续到56,000次更新；从上一阶段末轮而非开发最佳继续|
|固定网络、候选、特征、奖励、KEEP等|`PLAN_ZH.md`、各阶段 `PROTOCOL.json` 冻结源码/二进制哈希；`SCHEDULE_ACCEPTANCE.json` 对照300轮调用语法树；阶段审计核查奖励、模型维度及实际输入输出|
|C++16线程、GPU更新|冻结 `run100_common.py` 的16线程采样入口；续训脚本以CUDA模型及 `update_aux(...,'cuda',...)` 更新；每轮保存采样和GPU更新时间|
|七个实时对手、完整对局|全部训练/监测/终测检查seed×对手×座位网格，每局719步、30次规划、719次执行、引用路线调用0；异常计数和候选合法mask检查通过|
|严格隔离数据|训练两重复使用69600000/69700000起各16,000seed；开发69800000起32seed；固定监测70300000起100seed；独立终测71100000起100seed；阶段审计也检查旧评测和预检seed不重叠|
|400…1000每100轮报告|每阶段 `SUMMARY_ZH.md`、`ACCEPTANCE_ZH.md`、`FINAL_RESULTS.json`、`FINAL_AUDIT.json`、`MANIFEST.json`；七次汇报记录见 `USER_MILESTONE_REPORTS_ZH.md`|
|一致监测、对比上一百轮和KEEP|各阶段21配置共29,400局，含上一百轮和当前轮确定性/采样以及开发选定模型；包含逐对手、现金、分差、救回/损失、配对区间|
|监测协议实际可复现|七份 `MONITOR_REPRODUCTION_XXXX.json`；每阶段基准加旧模型共12,600局的游戏与决策数组完全复现上一阶段|
|C++/Torch一致性|每阶段每组首/中/末轮各抽取1,024条均匀覆盖的实际记录，对输入检查点重新计算概率、value、logp及ratio，误差均小于2e-5；这是分层抽样检查，不是逐条浮点恒等证明|
|保存所有权重/Adam/对局和旧最佳|原100/200/300轮及七个新阶段清单逐文件复核长度和SHA-256；每轮 `.bin`、`.pt`、`rollout_*` 保留，开发历史和选定旧检查点均保留|
|最终未见seed验证|`independent1000/SELECTION_FROZEN.json` 在终测前锁定29配置与模型哈希；`ACCEPTANCE.json` 验收40,600局；`RESULTS.json` 保存结果和配对区间|
|完整最终审计|`diagnostics/COMPLETION_AUDIT.json`：复核七阶段、独立集、旧文件与清单；主控制进程正常退出0，状态 `COMPLETE/all_accepted`|
|不自动提交/推送|本目标仅本地训练、评测、验收与报告；没有执行Kaggle提交或Git推送|

累计训练896,000局；300→1000本次新增627,200局；七阶段固定监测205,800局；最终独立复验40,600局。开发集与预检对局另计。这里的“局”包含同seed对不同对手、座位及方法的对战，不等于同数量的独立随机seed。

最后一个百轮阶段实测，四组“模拟+GPU更新”的训练吞吐约69.7–72.7局/秒，不含开发评测、磁盘保存和审计；完整阶段约32.1分钟。这不是裸模拟器吞吐，也不是线上推理延迟。

## 从哪里看

所有下面的路径均位于当前实验目录，链接使用本机E盘绝对地址。

- [完整总报告：固定监测趋势、最终独立结果和逐对手胜率](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/FINAL_SUMMARY_ZH.md)
- [1000轮固定监测详细报告](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round1000/ACCEPTANCE_ZH.md)
- [独立复验完整数据与配对区间](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/independent1000/RESULTS.json)
- [最终计算证据审计](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/diagnostics/COMPLETION_AUDIT.json)
- [全部百轮用户汇报记录](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/USER_MILESTONE_REPORTS_ZH.md)
- [分层文件清单与哈希索引](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/MANIFEST_INDEX.json)

逐阶段入口：

|阶段|摘要|完整报告、模型、对局与清单目录|
|---|---|---|
|400|[摘要](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0400/SUMMARY_ZH.md)|[round0400](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0400)|
|500|[摘要](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0500/SUMMARY_ZH.md)|[round0500](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0500)|
|600|[摘要](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0600/SUMMARY_ZH.md)|[round0600](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0600)|
|700|[摘要](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0700/SUMMARY_ZH.md)|[round0700](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0700)|
|800|[摘要](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0800/SUMMARY_ZH.md)|[round0800](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0800)|
|900|[摘要](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0900/SUMMARY_ZH.md)|[round0900](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0900)|
|1000|[摘要](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round1000/SUMMARY_ZH.md)|[round1000](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round1000)|

## 模型位置与使用边界

末轮四个模型均在 `stages/round1000/training/{control_r0,control_r1,aux_r0,aux_r1}/`：

- `step1000.bin`：70,402参数的actor/critic导出权重，供冻结C++策略推理；不是可单独提交的Agent。
- `step1000.pt`：包括完整模型、训练辅助头、Adam状态、轮数与bin哈希，可用于以后经授权的连续训练。
- `selection.json`、`best.json`：仅按开发集确定的历史选择；`rollout_901`至`rollout_1000`保存本阶段每轮真实对局和决策。
- 推理还需要原 [KEEP=2运行库](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_100_20260906/build/keep2.so)、对应F3执行器和C++对手数据；将bin直接接到不含KEEP=2偏置的推理器，不等价于本次评测。

开发选定的旧模型没有覆盖或删除：

|运行|bin权重（同目录同名前缀pt包含Adam）|
|---|---|
|普通1 / 200轮|[step200.bin](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_200_20260906/training/control_r0/step200.bin)|
|普通2 / 180轮|[step180.bin](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_200_20260906/training/control_r1/step180.bin)|
|辅助1 / 640轮|[step640.bin](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0700/training/aux_r0/step640.bin)|
|辅助2 / 780轮|[step780.bin](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/stages/round0800/training/aux_r1/step780.bin)|

机器可读模型路径和SHA-256见 [终测前冻结选择](E:/ai_coding/kaggle/kaggriculture/experiments/economic_rl_keep2_1000_20260906/independent1000/SELECTION_FROZEN.json)。该文件使用实际执行环境WSL路径，`/mnt/e/`对应Windows的`E:/`。

`MANIFEST_INDEX.json`索引根运行程序和七阶段/终测清单，各清单再覆盖其全部模型及数据。此补充解释文件、用户汇报记录和额外诊断另由 `HANDOFF_MANIFEST.json` 列出，不修改已冻结清单。

本目标验收完成的是按既定协议训练与验证，不是保证PPO持续变强、所有对手90%或可直接上线。本次没有新增完整官方Python逐步复验；如果以后要公开提交，仍须另做导出一致性及线上时限验证。
