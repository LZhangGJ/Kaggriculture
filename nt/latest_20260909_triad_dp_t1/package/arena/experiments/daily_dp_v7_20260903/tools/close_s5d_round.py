from pathlib import Path
import json,math,statistics as st
E=Path(__file__).resolve().parents[1];read=lambda p:json.loads((E/p).read_text())
panel=read('receipts/s5d_twelveway_N10_v1/results.json');ledger=read('receipts/s5d_pool_audit_N10_v1/summary.json');run=read('receipts/s5d_execution_v1/acceptance.json');runtime=read('receipts/s5d_runtime_v4/acceptance.json')
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and ledger['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
assert run['status']=='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE' and run['exact_control_games']==540
assert len(panel['rows'])==2160 and ledger['unchanged_result_games']==1620
assert runtime['status']=='COMPLETE_NATIVE_PROBE' and runtime['build']==panel['build']
rows={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
opponents=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
bases=['all_intraday_insert','full_chain_autonomous','full_chain_autonomous_timing']
def ci(values):
 mean=st.fmean(values);se=st.stdev(values)/math.sqrt(len(values))
 return dict(mean=mean,seed_se=se,approximate_t95=[mean-2.262157*se,mean+2.262157*se] if se else None,seeds=len(values))
effects=[];interactions=[];pool_effects=[]
for base in bases:
 for suffix in ('_first_value','_replant','_first_value_replant'):
  after=base+suffix
  metrics={field:ci([st.fmean(float(rows[after,o,s,z][field])-float(rows[base,o,s,z][field]) for o in opponents for z in (0,1)) for s in range(20262701,20262711)]) for field in ('win','cash','margin')}
  pool_effects.append(dict(before=base,after=after,metrics=metrics))
 for opponent in panel['identities']:
  for suffix in ('_first_value','_replant','_first_value_replant'):
   after=base+suffix;metrics={field:ci([st.fmean(float(rows[after,opponent,s,z][field])-float(rows[base,opponent,s,z][field]) for z in (0,1)) for s in range(20262701,20262711)]) for field in ('win','cash','margin')}
   effects.append(dict(before=base,after=after,opponent=opponent,metrics=metrics))
  metrics={field:ci([st.fmean(float(rows[base+'_first_value_replant',opponent,s,z][field])-float(rows[base+'_first_value',opponent,s,z][field])-float(rows[base+'_replant',opponent,s,z][field])+float(rows[base,opponent,s,z][field]) for z in (0,1)) for s in range(20262701,20262711)]) for field in ('win','cash','margin')}
  interactions.append(dict(base=base,opponent=opponent,metrics=metrics))
anomalies=[dict(label=r['label'],opponent=r['opponent'],games=r['games'],mean_no_effect=sum(r['own_production']['no_effect']),mean_escaped=sum(r['own_production']['escaped'])) for r in ledger['summary']]
qualifying=[label for label,s in panel['summary'].items() if all(s[o]['win_rate']>.9 for o in opponents)]
out=E/'receipts/s5d_stage_acceptance_v1';out.mkdir(exist_ok=False)
receipt=dict(status='COMPLETE_DEVELOPMENT_SCREEN_NOT_PROMOTED',build=panel['build'],real_games=2160,controls_verified=540,new_repeat_ledgers=1620,effects=effects,interactions=interactions,pool_effects=pool_effects,anomalies=anomalies,development_threshold_candidates=qualifying,official_full_games=32,isolation_checks=6,native_runtime_max=max(r['max_act_seconds'] for r in runtime['rows']),native_runtime_games=len(runtime['rows']),online_python_validated=False,holdout_used=False,promoted=False,final_goal_acceptance=False)
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2))
lines=['# S5D 首次编排 × 可延期补种：完整实战复盘','','原目标保持八个实时对手逐一90%以上，未以单局、平均胜率或PASS收益替代。','','## 问题与实现','','S5C发现维护义务生成并预留后在首次编排丢失；现有候选保护旧工作导致可延期补种挤占喂养。A将公开条件日内后果＋余值用于首次编排；B允许旧收获保留、下一茬补种延期一天。共享、完整自主、完整自主＋换种时机三个背景各KEEP/A/B/A+B。','','原递归预测时延不合格，失败版本完整保留。最终本轮采用独立有界预测开关：内部不反复展开明天的完整投资规划，真实逐步协调不关闭；它是近似预测，非无损声明。增量插入只替换成本计算，10,000随机排班完全等价。','','## 对战结果','','2160完整C++局，10旧开发seed×双座位，每对手20局；540旧控制逐局完全一致，1620新配置账本重复完全一致。8对手为真实闭环，不是冻结动作流。','','|配置|PASS现金|'+'|'.join(opponents)+'|八对手平均|综合局/s|','|---|---:|'+'---:|'*10]
for label,s in panel['summary'].items():
 rate=st.fmean(s[o]['win_rate'] for o in opponents);speed=sum(s[o]['games'] for o in opponents)/sum(s[o]['wall_seconds'] for o in opponents)
 lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(f"{s[o]['wins']}/20" for o in opponents)+f'|{rate:.3%}|{speed:.2f}|')
lines+=['','## 独立效应与长链组合','','完整配对胜率、现金、分差及交互 `AB-A-B+KEEP` 见stage acceptance.json；每个seed先合并双座位，再算10seed近似t区间，未校正多重比较。旧开发seed已反复用于选型；不代表独立泛化。','','八对手汇总仍以seed为独立单位，不能把同一seed的16局当16个独立样本。','','|改动|八对手胜率差pp及近似95%区间|平均现金差|平均分差变化|','|---|---|---:|---:|']
for r in pool_effects:
 m=r['metrics'];b=m['win']['approximate_t95'];bounds='经验方差为零' if b is None else f'[{100*b[0]:+.2f}, {100*b[1]:+.2f}]'
 lines.append(f"|{r['after']}|{100*m['win']['mean']:+.3f} {bounds}|{m['cash']['mean']:+.0f}|{m['margin']['mean']:+.0f}|")
lines+=['','|改动|对手|胜率差pp|现金差|分差变化及近似95%区间|','|---|---|---:|---:|---|']
for r in effects:
 if r['opponent']=='pass':continue
 m=r['metrics'];b=m['margin']['approximate_t95'];bounds='零经验方差，不给确定性区间' if b is None else f'[{b[0]:.0f}, {b[1]:.0f}]'
 lines.append(f"|{r['after']}|{r['opponent']}|{100*m['win']['mean']:+.1f}|{m['cash']['mean']:+.0f}|{m['margin']['mean']:+.0f} {bounds}|")
lines+=['','## 异常与验收边界','','|配置|对手|平均无效单位动作|平均逃跑|','|---|---|---:|---:|']
for r in anomalies:lines.append(f"|{r['label']}|{r['opponent']}|{r['mean_no_effect']:.2f}|{r['mean_escaped']:.2f}|")
lines+=['',f"32抽样官方完整局与6隔离检查通过；原生{receipt['native_runtime_games']}局时延探针最大{receipt['native_runtime_max']:.3f}秒。不是最终线上Python验收。",'', '## 是否晋级与下一步','','不晋级：未见seed、最终线上时延和全八90%目标均未完成。即使动作合法，未维护资产和少投资问题仍须按真实现金分渠道复盘。下一轮应核对首次选择实际改变了哪些任务、漏喂是否修复、是否挤掉更值钱的生产；再决定是否补投资/用工融资联合候选，不能直接加固定雇工或作物数量。P未用，所有原配置、失败、源码及build hash保留。']
(E/'reports/S5D_FIRST_COMPILE_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines[:28]),flush=True)
