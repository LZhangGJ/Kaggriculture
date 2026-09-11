"""Reuse exact paired ledgers; no simulation, search or policy mutation."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
E=Path(__file__).resolve().parents[1]
read=lambda p:json.loads((E/p).read_text())
panel=read('receipts/s5d_twelveway_N10_v1/results.json')
done=read('receipts/s5d_execution_v1/acceptance.json')
assert done['status']=='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE'
assert read('receipts/s5d_pool_audit_N10_v1/summary.json')['unchanged_result_games']==1620
refs={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
out=E/'receipts/s5d_channels_v1';out.mkdir(exist_ok=False)
hashes={}

def load(folder,label,opp):
 path=E/f'receipts/{folder}/{label}_{opp}.json.gz'
 hashes[str(path.relative_to(E))]=hashlib.sha256(path.read_bytes()).hexdigest()
 with gzip.open(path,'rt') as f:raw=json.load(f)['rows']
 result={}
 for r in raw:
  if not 20262701<=r['seed']<=20262710:continue
  seat=r['seat'];ref=refs[label,opp,r['seed'],seat]
  assert (r['money'][seat],r['money'][1-seat])==(ref['cash'],ref['opponent_cash'])
  assert len(r['cash_ledger'])==len(r['production'])==30
  sides=[]
  for who in (seat,1-seat):
   money=[d[who] for d in r['cash_ledger']];prod=[d[who] for d in r['production']]
   d={name:sum(sum(day[name]) if isinstance(day[name],list) else day[name] for day in money) for name in ('sales','products','seeds','animals','hired','land')}
   d['cash']=r['money'][who]
   assert abs(d['cash']-3000-d['sales']+sum(d[k] for k in ('products','seeds','animals','hired','land')))<1e-6
   for i in range(9):
    for k in ('sales','sold'):d[f'{k}_{i}']=sum(day[k][i] for day in money)
   for k in ('planted','fed','cared','escaped','no_effect','eod_loss','environment_loss'):
    d[k]=sum(sum(day[k]) for day in prod)
   d['moves']=sum(sum(day['attempts'][1:5]) for day in prod)
   d['harvests']=sum(day['attempts'][10]-day['no_effect'][10] for day in prod)
   sides.append(d)
  result[r['seed'],seat]=sides
 assert len(result)==20
 return result

channels=[]
for base,folder in [('all_intraday_insert','s4m1_pool_audit_N50_v1'),('full_chain_autonomous','s4u_pool_audit_N50_v1'),('full_chain_autonomous_timing','s5b_pool_audit_N10_v1')]:
 for opp in panel['identities']:
  a=load(folder,base,opp)
  for suffix in ('_first_value','_replant','_first_value_replant'):
   label=base+suffix;b=load('s5d_pool_audit_N10_v1',label,opp);assert a.keys()==b.keys()
   sides=[]
   for who in (0,1):
    delta={k:st.fmean(b[key][who][k]-a[key][who][k] for key in a) for k in next(iter(a.values()))[who]}
    assert abs(delta['cash']-delta['sales']+sum(delta[k] for k in ('products','seeds','animals','hired','land')))<1e-6
    attribution=[]
    for i in range(9):
     volume=[];price=[]
     for key in a:
      ra,rb=a[key][who][f'sales_{i}'],b[key][who][f'sales_{i}'];qa,qb=a[key][who][f'sold_{i}'],b[key][who][f'sold_{i}']
      if qa and qb:
       pa,pb=ra/qa,rb/qb;v=(qb-qa)*(pa+pb)/2;p=(pb-pa)*(qa+qb)/2
      else:v=rb-ra;p=0
      assert abs(v+p-(rb-ra))<1e-6
      volume.append(v);price.append(p)
     attribution.append(dict(item=i,volume=st.fmean(volume),average_realized_price=st.fmean(price)))
    sides.append(dict(delta=delta,sales_decomposition=attribution))
   channels.append(dict(base=base,after=label,opponent=opp,own=sides[0],rival=sides[1]))

receipt=dict(status='PASS_PAIRED_LEDGER_DECOMPOSITION',source_hashes=hashes,comparisons=len(channels),rows=channels,new_games=0,caveat='Accounting identities, not isolated causal effects. Price terms include changed timing and both agents responses; zero-volume endpoints are assigned to volume.')
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2))
lines=['# S5D 人员—产销—现金联合复盘','','只读复用原始逐日账本；所有旧/新配置、双方现金与S5D面板逐局一致。没有新增独立样本。差额是整套政策变化后的会计分解，不能当成每个因素单独的因果收益。','','|配置|对手|现金Δ|对手现金Δ|移动Δ|收获Δ|播种Δ|工资Δ|销售Δ|总采购Δ|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in channels:
 d=r['own']['delta'];q=r['rival']['delta'];purchase=sum(d[k] for k in ('products','seeds','animals','land'))
 lines.append(f"|{r['after']}|{r['opponent']}|{d['cash']:+.0f}|{q['cash']:+.0f}|{d['moves']:+.1f}|{d['harvests']:+.1f}|{d['planted']:+.1f}|{d['hired']:+.0f}|{d['sales']:+.0f}|{purchase:+.0f}|")
lines+=['','每种商品另保存成交量与平均成交价的对称分解，恒等于销售收入差。平均成交价不等于单独的抢卖效果，可能同时受数量、时间和对手响应影响。详见 `receipts/s5d_channels_v1/acceptance.json`。','','是否晋级以完整阶段报告为准；不得根据对手列挑不同最优开关拼接成身份路由。']
(E/'reports/S5D_LABOUR_SALES_REVIEW_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=receipt['status'],comparisons=len(channels),source_files=len(hashes))),flush=True)
