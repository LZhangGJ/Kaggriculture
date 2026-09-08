from pathlib import Path
import hashlib,json,statistics as st,math,gzip
E=Path(__file__).resolve().parents[1]
def read(rel):return json.loads((E/rel).read_text())
panel=read('receipts/s4z_fourway_N10_v1/results.json');previous=read('receipts/s4v_twelveway_N50_v1/results.json')
stage=read('receipts/s4z_execution_v1/acceptance.json');ledger=read('receipts/s4z_pool_audit_N10_v1/summary.json')
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and stage['status']=='COMPLETE_SCREEN_NOT_GOAL_ACCEPTANCE'
assert ledger['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and ledger['unchanged_result_games']==360
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in panel['rows']};old={key(r):r for r in previous['rows']};checked=0
for k,r in rows.items():
 if not k[0].endswith('_rotation'):
  assert all(r[f]==old[k][f] for f in ('cash','opponent_cash','margin','win','overflow')),k;checked+=1
assert checked==360 and len(rows)==720
def ci(xs):
 mean=st.fmean(xs);se=st.stdev(xs)/math.sqrt(len(xs))
 return dict(mean=mean,seed_se=se,approximate_t95=[mean-2.262157*se,mean+2.262157*se] if se else None,n=len(xs),caveat='10 reused development seeds; t interval approximate. Zero empirical variance is not certainty; no multiplicity correction.')
effects=[]
for base in ('all_intraday_insert','full_chain_autonomous'):
 for opp in panel['identities']:
  after=base+'_rotation';effects.append(dict(before=base,after=after,opponent=opp,metrics={k:ci([st.fmean(float(rows[after,opp,s,z][k])-float(rows[base,opp,s,z][k]) for z in (0,1)) for s in range(20262701,20262711)]) for k in ('cash','margin','win')}))
anomalies=[]
for r in ledger['summary']:
 x=r['own_production'];anomalies.append(dict(label=r['label'],opponent=r['opponent'],mean_no_effect=sum(x['no_effect']),mean_escaped=sum(x['escaped'])))
channels=[]
for base,folder in [('all_intraday_insert','s4m1_pool_audit_N50_v1'),('full_chain_autonomous','s4u_pool_audit_N50_v1')]:
 for opp in panel['identities']:
  a=json.load(gzip.open(E/f'receipts/{folder}/{base}_{opp}.json.gz','rt'))['rows'];a=[r for r in a if 20262701<=r['seed']<=20262710]
  b=json.load(gzip.open(E/f'receipts/s4z_pool_audit_N10_v1/{base}_rotation_{opp}.json.gz','rt'))['rows'];assert len(a)==len(b)==20
  def metrics(rs):
   result={};checks=[]
   for r in rs:
    k=(r['seed'],r['seat']);side=r['seat'];cash=[d[side] for d in r['cash_ledger']];prod=[d[side] for d in r['production']]
    d={name:sum(sum(day[name]) if isinstance(day[name],list) else day[name] for day in cash) for name in ('sales','products','seeds','animals','hired','land')}
    d['cash']=r['money'][side];assert abs(d['cash']-3000-d['sales']+sum(d[x] for x in ('products','seeds','animals','hired','land')))<1e-6
    for i in range(9):d['sales_'+str(i)]=sum(day['sales'][i] for day in cash);d['generated_'+str(i)]=sum(day['generated'][i] for day in prod)
    for i in range(5):d['planted_'+str(i)]=sum(day['planted'][i] for day in prod)
    result[k]=d
   return result
  aa,bb=metrics(a),metrics(b);assert aa.keys()==bb.keys();channels.append(dict(base=base,opponent=opp,delta={k:st.fmean(bb[idx][k]-aa[idx][k] for idx in aa) for k in next(iter(aa.values()))}))
out=E/'receipts/s4z_stage_acceptance_v1';out.mkdir(exist_ok=False)
result=dict(status='COMPLETE_DEVELOPMENT_SCREEN_NOT_PROMOTED',unchanged_controls=checked,repeated_audits=360,effects=effects,channels=channels,anomalies=anomalies,holdout_used=False,promoted=False,online_latency_tested=False,new_configuration_official_parity_tested=False,final_goal_acceptance=False)
(out/'acceptance.json').write_text(json.dumps(result,indent=2))
opps=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4Z 成熟作物轮作小面板复盘','','原生17项机制通过，720完整实时局、360新配置重复账本通过；360旧控制逐局与S4V相同。每格10旧开发seed×双座位20局。未做新配置官方逐步与线上时延验收，不晋级。','','|配置|PASS现金|'+'|'.join(opps)+'|平均胜率|','|---|---:|'+'---:|'*9]
for label,s in panel['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(f"{s[o]['wins']}/20" for o in opps)+f"|{st.fmean(s[o]['win_rate'] for o in opps):.1%}|")
lines+=['','## 配对变化（同seed两个座位合并）','','|背景|对手|胜率变化pp|资金差变化|资金差近似95%区间|','|---|---|---:|---:|---|']
for r in effects:
 if r['opponent']=='pass':continue
 m=r['metrics']['margin'];bounds=m['approximate_t95'];interval='无经验方差，不作确定性区间' if bounds is None else f"[{bounds[0]:.0f}, {bounds[1]:.0f}]"
 lines.append(f"|{r['before']}|{r['opponent']}|{r['metrics']['win']['mean']*100:+.1f}|{m['mean']:+,.0f}|{interval}|")
lines+=['','## 异常与解释边界','','以下是每局平均次数，不能当独立失败概率。开启更多重选机会并不保证原有评分正确；不能把本轮失败归为所有轮作都无效，也不能将某个对手20局提高当稳定泛化。','','|配置|对手|平均无效单位动作|平均动物逃跑|','|---|---|---:|---:|']
for r in anomalies:lines.append(f"|{r['label']}|{r['opponent']}|{r['mean_no_effect']:.2f}|{r['mean_escaped']:.2f}|")
lines+=['','## 同seed实际现金/生产变化','','以下是完整政策干预的账目差异，不是独立可叠加收益；所有现金收支守恒检查通过。','','|背景|对手|现金Δ|草莓收入Δ|番茄收入Δ|小麦收入Δ|瓜收入Δ|工资支出Δ|草莓种植Δ|番茄种植Δ|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in channels:
 d=r['delta'];lines.append('|'+r['base']+'|'+r['opponent']+'|'+'|'.join(f"{d[k]:+,.1f}" for k in ('cash','sales_3','sales_2','sales_0','sales_4','hired','planted_3','planted_2'))+'|')
(E/'reports/S4Z_HARVEST_ROTATION_REVIEW_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines))
