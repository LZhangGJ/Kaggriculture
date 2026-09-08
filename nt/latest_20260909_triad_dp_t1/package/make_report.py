from pathlib import Path
import json,csv,hashlib,math,sys
import numpy as np
R=Path(__file__).resolve().parent
new=json.loads((R/'runs/release_holdout100/rows.json').read_text());old=json.loads((R/'runs/baseline_j7_holdout100/rows.json').read_text())
new=sorted(new,key=lambda r:(r['seed'],r['opponent'],r['seat']));old=sorted(old,key=lambda r:(r['seed'],r['opponent'],r['seat']))
assert [(r['seed'],r['opponent'],r['seat']) for r in new]==[(r['seed'],r['opponent'],r['seat']) for r in old]
assert len(new)==1400 and len({r['seed'] for r in new})==100 and all(r['steps']==719 or r['error'] for r in new)
seeds=sorted({r['seed'] for r in new});NAMES=['G001','G003','Boatlee V29','Kaito V58','Lynn V5','yhay81 Six-Day','yhay81 Three-Day']
a=np.array([float(r['win']) for r in new]).reshape(100,7,2);b=np.array([float(r['win']) for r in old]).reshape(100,7,2)
rng=np.random.default_rng(9082026);ix=rng.integers(0,100,(10000,100))
ci=lambda x:[float(v) for v in np.quantile(x,[.025,.975])]
ac=a.mean((1,2));bc=b.mean((1,2));overall_ci=ci(ac[ix].mean(1));paired_ci=ci((ac-bc)[ix].mean(1))
summary=json.loads((R/'runs/release_holdout100/summary.json').read_text());baseline=json.loads((R/'runs/baseline_j7_holdout100/summary.json').read_text());frozen=json.loads((R/'RELEASE_FREEZE.json').read_text())
per=[]
for i,name in enumerate(NAMES):
 rs=[r for r in new if r['opponent']==i];vs=a[:,i,:].mean(1)
 per.append(dict(opponent=name,games=len(rs),wins=sum(r['win'] for r in rs),rate=float(vs.mean()),ci=ci(vs[ix].mean(1)),baseline_rate=float(b[:,i,:].mean()),mean_cash=float(np.mean([r['cash'] for r in rs])),mean_margin=float(np.mean([r['margin'] for r in rs]))))
res=dict(overall=summary['overall'],baseline=baseline['overall'],seed_cluster_bootstrap95=overall_ci,paired_difference95=paired_ci,overall_target_met=bool(a.mean()>=.9),each_target_met=bool((a.mean((0,2))>=.9).all()),per_opponent=per,freeze=frozen,bootstrap='10000 resamples of the 100 whole seed clusters, retaining 7 opponents and both seats; opponents treated as fixed benchmark')
(R/'ACCEPTANCE.json').write_text(json.dumps(res,indent=2));wins=int(a.sum());rate=float(a.mean());oldrate=float(b.mean());delta=rate-oldrate
with (R/'holdout_games.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['seed','opponent_name','seat','cash','opponent_cash','win','margin','steps','error','overflow','max_action_ms','policy_seconds','action_hash_fnv64']);w.writeheader()
 for r in new:w.writerow({k:r[k] for k in w.fieldnames})
# Audited holdout trace subset and official wrapper parity.
qa=json.loads((R/'tests/results/parity_release/acceptance.json').read_text());tr=json.loads((R/'runs/release_official14/rows.json').read_text());assert qa['status']=='PASS'
noeffects=np.zeros(24,dtype=int);attempts=np.zeros(24,dtype=int);drop=0
for r in tr:
 for d in r['production_audit']:noeffects+=d['no_effect'];attempts+=d['attempts'];drop+=sum(d['drop_loss'])
quality=dict(official_games=len(qa['rows']),official_transitions=sum(r['transitions'] for r in qa['rows']),official_state_mismatches=0,python_action_mismatches=0,unit_attempts=int(attempts.sum()),unit_no_effects=int(noeffects.sum()),no_effect_by_op=noeffects.tolist(),drop_discard_in_audited_subset=drop,eod_discard_in_holdout=sum(r['overflow'] for r in new),scope='Unit effect audit and official step parity cover the same 14 fresh holdout trajectories, not every holdout match or every possible state')
(R/'QUALITY_ACCEPTANCE.json').write_text(json.dumps(quality,indent=2))
status='总体达到 90%，但逐对手条件尚未全部满足' if res['overall_target_met'] and not res['each_target_met'] else '总体和逐对手均达到 90% 点估计目标' if res['each_target_met'] else '没有达到 90% 胜率目标'
fmt=lambda x:f'{100*x:.2f}%'
lines=[f'# Triad-DP T1 实施与独立验收报告','',f'日期：2026-09-08。**{status}。**', '',f'独立面板：100 个预先冻结 seed × 7 个冻结实时 C++ 对手 × 双座位 = **1,400 局**。T1 获得 **{wins}/1400 胜，{fmt(rate)}**。平局不计胜，异常计负，没有删去任何弱对手、失败 seed 或座位。', '', '## 1. 结果','', '| 对手 | T1 胜局 / 200 | T1 胜率 | 同种子冻结 C3＋J7 胜率 | T1 平均现金差 |','|---|---:|---:|---:|---:|']
for p in per:lines.append(f'| {p["opponent"]} | {p["wins"]} / 200 | {fmt(p["rate"])} | {fmt(p["baseline_rate"])} | {p["mean_margin"]:,.0f} |')
lines+= [f'| **七对手等权平均** | **{wins} / 1400** | **{fmt(rate)}** | **{fmt(oldrate)}** | **{summary["overall"]["margin"]:,.0f}** |','',f'相对上述同种子基线，胜率差为 **{delta*100:+.2f} 个百分点**。按完整 seed 聚类的配对 bootstrap 95% 区间为 [{paired_ci[0]*100:+.2f}, {paired_ci[1]*100:+.2f}] 个百分点。T1 总体胜率相同口径区间为 [{fmt(overall_ci[0])}, {fmt(overall_ci[1])}]。相同 seed 下的两个座位和七个对手没有被假装成彼此独立样本。该区间也不覆盖未见对手的泛化不确定性。','',f'平均我方现金 **{summary["overall"]["cash"]:,.2f}**，平均对手现金差 **{summary["overall"]["margin"]:,.2f}**；异常 **{summary["overall"]["errors"]}** 局，平局 **{summary["overall"]["ties"]}** 局。本机记录的最大单次 C++ act 耗时 **{summary["overall"]["max_ms"]:.2f} ms**，平均每局 C++ policy 耗时 **{summary["overall"]["mean_policy_sec"]:.3f} s**。这些是当前容器实测，不是 Kaggle 沙箱耗时保证。','', '总体 90% 和每个对手 90% 分开验收，见 `ACCEPTANCE.json`。不把达到某个对手 90% 写成七个对手都达标。','', '## 2. 真正交付的实现','', 'T1 复用本地 9 月 5 日 C3 的底层执行与经济工具，新增独立的经营项目规划、条件生产日历、未完成项目/轮作后继承诺、当日可行性预览、公共信息整日前瞻、库存保护及即时现金兑现。新策略不读取 J7 宏观配方或七对手资产。','', '每日候选由当前真实盘面重新生成；公共情景执行到次日边界，比较实际安排下的现金和后续估值，再安装当天经营计划。实际比赛仍每步读取 observation，不播放预先模拟出的原子动作序列。','', '保留 C3 的日内有效路线与安全修复；跨日保存经营承诺、重新分配实际存在的工人。已买的动物、已买种子的后继作物不会仅因日切或旧作物被收掉而丢失。执行前的资源与时间预览可以拒绝尚未投入的新项目。','', '完整实现说明与未完成项见 `DESIGN_AND_SCOPE_ZH.md`。特别注意：全局项目组合仍是启发式；未来轮作收益项和额外局部路线 DP 在默认配置中关闭。不能把本版称为全局最优的 720 步 DP。','', '## 3. 开发实验与冻结方式','', '最终配置选择之前，固定同一开发面板 32 seed × 7 对手 × 双座位。原生 T1（无每日候选前瞻）为 318/448 = 70.98%；默认公共情景前瞻为 374/448 = 83.48%。更宽候选和额外局部最短路线未继续提高总体胜率，因此没有因为名字更复杂就强行启用。','', '另采集了 **7,049 个反事实分支、2,464 个决策节点、356 维公开/我方特征**，训练并导出回归与排序两种小型估值模型。回归模型开发胜率 318/448 = 70.98%，排序模型 326/448 = 72.77%，均未超过公共情景前瞻。默认晋级配置不调用这些模型；保留数据和训练代码，不声称训练成功提高最终版本胜率。','', '`SELECTION_BEFORE_HOLDOUT.json` 在独立面板之前记录选择；`RELEASE_FREEZE.json` 在首次 holdout 之前固定源码、二进制和配置。holdout seed 为 **260909000–260909099**，不参与训练、调参或版本选择。本次看到 holdout 结果后没有回调参数或替换胜者。','', '## 4. 规则、执行及导出验收','', f'同一最终版本另跑七对手首个 holdout seed 的双座位，共 **{quality["official_games"]} 局、{quality["official_transitions"]:,} 次转换**，逐步重新送入随包冻结的官方 Python 1.32.7 规则。官方与 C++ 观察字段差异 **0**；通过 Python 导出入口重新调用候选策略，动作差异 **0**。','',f'该 14 局单位动作效果审计记录 **{quality["unit_attempts"]:,} 次单位动作尝试**，无效果动作 **{quality["unit_no_effects"]}**。所有 1,400 局共记录日末库存溢出丢弃 **{quality["eod_discard_in_holdout"]:,} 单位**。没有无效动作不等于没有经济损失，也不等于已经满足全部经营承诺。','', '附带机制测试涵盖可控作物/动物生产日历、PLANT→WATER、同回合入库后销售、维护资源预留、跨日动物部署、旧作物存在/消失两种轮作接续、局部路线 DP 对照穷举、公开特征确定性。反事实分支正序/倒序及 KEEP 共同恢复检查通过。测试是有限案例，不是所有状态的形式证明。','', '从交付目录重新解压后，另外完成无缓存源码构建及机制测试；新版和旧基线各 14 局复现的双方动作哈希、现金、胜负与溢出记录均相同。见 `tests/clean_package_rebuild.json`。这是复现检查，不计入独立胜率样本。','', '## 5. 还没有解决什么','', '对手未来销售目前由公开产能和条件到货窗口近似；候选执行只前瞻至短期边界，后续生产、劳动力和价格依然依赖近似估值。SELL/HOLD、跨日融资、仓储、再投资没有组成一个精确的统一长期 DP。当前候选也不覆盖所有联合经营调整。','', '因此，这版交付的是可运行、可复现、比纯结构说明更进一步的动态规划系统，但是否达到 90% 必须服从上表。本轮没有把剩余输局逐项做完因果归因，不把所有差距笼统归咎于执行器或学习算法。','', '## 6. 来源与重现边界','', '本地基线是 2026-09-05 重建包里的 C3＋J7 参考配置，不是 9 月 6 日最新 C3/F3，不能拿跨版本、跨 seed 的历史百分比直接比较。本轮未重新构建完整 F3。七个对手是随包冻结的实时 C++ 实现；不是冻结对手的动作，不宣称是今天 Kaggle 最新版本。','', '官方逐步测试核对环境规则及我方 Python 导出入口；它没有重新穷尽核对七个对手 Python↔C++ 在所有反事实状态下的行为。部署到 Kaggle 的打包、沙箱限制和线上天梯成绩未验证。','', '源码：`policy/`；默认配置：`policy/config.json`；逐局结果：`runs/release_holdout100/rows.json`、`holdout_games.csv`；官方对齐：`tests/results/parity_release/acceptance.json`；完整冻结信息：`RELEASE_FREEZE.json`。','',f'最终二进制 SHA256：`{frozen["binary_hash"]}`。',f'最终源码集合 SHA256：`{frozen["source_hash"]}`。','']
diagnosis=R/'POSTHOLDOUT_LOSS_DIAGNOSIS.json'
if diagnosis.exists():
 j=json.loads(diagnosis.read_text())
 days={r['day']:r for r in j['snapshots']}
 d12=days.get(12,{})
 insert=['', '### 冻结后只读败局诊断', '', f'为避免用测试集继续调参，只在冻结验收后复放了 G003 最弱现金差的一局：seed {j["seed"]}，座位 {j["seat"]}。终局 T1 {j["cash"]:,.0f}，对手 {j["opponent_cash"]:,.0f}，差额 {j["margin"]:,.0f}。第 12 天双方现金分别为 {d12.get("cash",0):,.0f} 与 {d12.get("opponent_cash",0):,.0f}，生产布局和扩张节奏已明显不同。逐日盘面与价格保存在 `POSTHOLDOUT_LOSS_DIAGNOSIS.json`。这只是观察证据，不是逐因素因果消融；没有据此修改最终配置。', '']
 at=lines.index('## 6. 来源与重现边界');lines[at:at]=insert
(R/'REPORT_ZH.md').write_text('\n'.join(lines));print(json.dumps(res['overall']),flush=True)
