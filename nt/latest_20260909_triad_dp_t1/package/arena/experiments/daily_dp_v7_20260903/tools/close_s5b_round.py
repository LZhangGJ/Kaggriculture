from pathlib import Path
import gzip,hashlib,json,math,statistics as st
E=Path(__file__).resolve().parents[1]
def read(rel):return json.loads((E/rel).read_text())
panel=read('receipts/s5b_eightway_N10_v1/results.json');old=read('receipts/s4z_fourway_N10_v1/results.json')
ledger=read('receipts/s5b_pool_audit_N10_v1/summary.json');run=read('receipts/s5b_execution_v2/acceptance.json')
runtime=read('receipts/s5b_runtime_v2/acceptance.json')
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and ledger['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
assert run['status']=='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE' and runtime['status']=='COMPLETE_NATIVE_PROBE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in panel['rows']};refs={key(r):r for r in old['rows']};verified=0
for k,r in rows.items():
 if k[0] in old['configurations']:
  assert all(r[f]==refs[k][f] for f in ('cash','opponent_cash','margin','win','overflow')),(k,r,refs[k]);verified+=1
assert verified==720 and len(rows)==1440 and ledger['unchanged_result_games']==720
def ci(xs):
 mean=st.fmean(xs);se=st.stdev(xs)/math.sqrt(len(xs));return dict(mean=mean,seed_se=se,t95=[mean-2.262157*se,mean+2.262157*se] if se else None,n=len(xs))
effects=[]
for base in ('all_intraday_insert','full_chain_autonomous'):
 for after in (base+'_net',base+'_timing'):
  for before in (base,base+'_rotation'):
   for opponent in panel['identities']:
    metrics={f:ci([st.fmean(float(rows[after,opponent,s,z][f])-float(rows[before,opponent,s,z][f]) for z in (0,1)) for s in range(20262701,20262711)]) for f in ('cash','margin','win')}
    effects.append(dict(before=before,after=after,opponent=opponent,metrics=metrics))
anomalies=[dict(label=r['label'],opponent=r['opponent'],no_effect=sum(r['own_production']['no_effect']),escaped=sum(r['own_production']['escaped'])) for r in ledger['summary']]
def money_metrics(raw):
 result={}
 for r in raw:
  seat=r['seat'];cash=[d[seat] for d in r['cash_ledger']];prod=[d[seat] for d in r['production']]
  d={name:sum(sum(day[name]) if isinstance(day[name],list) else day[name] for day in cash) for name in ('sales','products','seeds','animals','hired','land')}
  d['cash']=r['money'][seat];assert abs(d['cash']-3000-d['sales']+sum(d[k] for k in ('products','seeds','animals','hired','land')))<1e-6
  for i in range(9):
   d['sales_'+str(i)]=sum(day['sales'][i] for day in cash);d['products_'+str(i)]=sum(day['products'][i] for day in cash);d['generated_'+str(i)]=sum(day['generated'][i] for day in prod)
  for i in range(5):d['planted_'+str(i)]=sum(day['planted'][i] for day in prod)
  result[r['seed'],r['seat']]=d
 return result
channels=[]
for base,folder in [('all_intraday_insert','s4m1_pool_audit_N50_v1'),('full_chain_autonomous','s4u_pool_audit_N50_v1')]:
 for opponent in panel['identities']:
  previous=json.load(gzip.open(E/f'receipts/{folder}/{base}_{opponent}.json.gz','rt'))['rows'];a=money_metrics([r for r in previous if 20262701<=r['seed']<=20262710])
  for after in (base+'_net',base+'_timing'):
   b=money_metrics(json.load(gzip.open(E/f'receipts/s5b_pool_audit_N10_v1/{after}_{opponent}.json.gz','rt'))['rows']);assert a.keys()==b.keys()
   channels.append(dict(before=base,after=after,opponent=opponent,delta={k:st.fmean(b[i][k]-a[i][k] for i in a) for k in next(iter(a.values()))}))
out=E/'receipts/s5b_stage_acceptance_v1';out.mkdir(exist_ok=False)
receipt=dict(status='COMPLETE_DEVELOPMENT_SCREEN_NOT_PROMOTED',build=panel['build'],controls_verified=verified,new_repeat_ledgers=720,effects=effects,channels=channels,anomalies=anomalies,
 official_full_games=32,isolation_checks=6,native_runtime_max=max(r['max_act_seconds'] for r in runtime['rows']),native_runtime_games=len(runtime['rows']),
 online_python_validated=False,holdout_used=False,promoted=False,final_goal_acceptance=False)
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2))
opponents=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S5B 净收益换种与时机：实战筛查结果','','1440完整C++实时局，720旧控制逐局等于S4Z，720新配置账本重复一致。每格10个旧开发seed、双座位20局；不是独立最终验收，不晋级。','','|配置|PASS现金|'+'|'.join(opponents)+'|八对手平均|','|---|---:|'+'---:|'*9]
for label,s in panel['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(f"{s[o]['wins']}/20" for o in opponents)+f"|{st.fmean(s[o]['win_rate'] for o in opponents):.1%}|")
lines+=['','## 配对差异（相对不开轮作的原背景）','','按seed合并两座位计算，t区间近似且没有多重比较校正；零经验方差不意味着确定性。','','|新配置|对手|胜率变化pp|分差变化|分差近似95%区间|','|---|---|---:|---:|---|']
for r in effects:
 if r['before'].endswith('_rotation') or r['opponent']=='pass':continue
 x=r['metrics'];bounds=x['margin']['t95'];interval='不作确定性区间' if bounds is None else f'[{bounds[0]:.0f}, {bounds[1]:.0f}]'
 lines.append(f"|{r['after']}|{r['opponent']}|{100*x['win']['mean']:+.1f}|{x['margin']['mean']:+,.0f}|{interval}|")
lines+=['','## 真实现金变化渠道','','均为整套策略变动的账目差，不是独立可叠加因果。','','|配置|对手|现金Δ|草莓收入Δ|奶收入Δ|羊毛收入Δ|小麦采购Δ|肥料采购Δ|工资Δ|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in channels:
 d=r['delta'];lines.append('|'+r['after']+'|'+r['opponent']+'|'+'|'.join(f'{d[k]:+,.0f}' for k in ('cash','sales_3','sales_6','sales_7','products_0','products_8','hired'))+'|')
lines+=['','## 异常与边界','','|配置|对手|平均无效单位动作|平均动物逃跑|','|---|---|---:|---:|']
for r in anomalies:lines.append(f"|{r['label']}|{r['opponent']}|{r['no_effect']:.2f}|{r['escaped']:.2f}|")
lines+=['',f"32局抽样官方一致性和6项隔离检查通过；原生12局探针最大动作耗时{receipt['native_runtime_max']:.3f}秒，不是最终Python提交时延保证。首次step144接入错误及修复记录保留。",'','四项执行背景与换种模块同开并不等于已达90%。后续方向必须由真实数据决定；本轮没有用未见P，也没有将预测最佳当实际终局。']
(E/'reports/S5B_ROTATION_NET_CASH_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines[:14]),flush=True)
