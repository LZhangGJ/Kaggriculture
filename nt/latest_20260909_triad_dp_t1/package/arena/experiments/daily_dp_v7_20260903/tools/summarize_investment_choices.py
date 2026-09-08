from pathlib import Path
from collections import Counter,defaultdict
import gzip,hashlib,json,statistics as st
E=Path(__file__).resolve().parents[1];root=E/'receipts/s4y_investment_choice_audit_v1'
receipt=json.loads((root/'acceptance.json').read_text());assert receipt['status']=='PASS_READ_ONLY_INVESTMENT_AUDIT'
out=E/'receipts/s4y_investment_choice_summary_v1';out.mkdir(exist_ok=False)
names={0:'小麦',1:'胡萝卜',2:'番茄',3:'草莓',4:'瓜',9:'鹅',10:'牛',11:'羊'};totals=defaultdict(Counter);days=defaultdict(Counter);examples=[];hashes={};proposals=defaultdict(list)
for path in sorted(root.glob('*.json.gz')):
 hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();data=json.load(gzip.open(path,'rt'));g=data['summary'];label=g['label']
 for r in data['rows']:
  if r['prescribed_opening'] or r['day']>=20:continue
  phase='early_0_9' if r['day']<10 else 'middle_10_19'
  if not r['rounds']:days[label,phase]['no_free_plot']+=1;continue
  days[label,phase]['with_candidates']+=1
  for i,name in names.items():
   option=next(x for x in r['rounds'][0]['options'] if x['kind']==i)
   status=option['reason'] or ('chosen_first' if r['rounds'][0]['chosen']==i else 'eligible_lower_rank')
   totals[label,phase,i][status]+=1
   if option['reason']=='capital_budget' and (r['seed_stock'][i] if i<5 else r['shed'][i])>0:
    totals[label,phase,i]['capital_block_with_owned_input']+=1
    if len(examples)<80:examples.append(dict(**g,day=r['day'],kind=i,cash=r['cash'],budget=r['budget'],stock=r['seed_stock'][i] if i<5 else r['shed'][i],option=option))
  proposals[label,phase].append(dict(raw=r['raw_proposals'],intended=r['intended'],started=r['projected_started'],free=r['free'],cash=r['cash'],budget=r['budget'],hands=r['prepared_hands']))
result=dict(totals=[dict(label=l,phase=p,item=names[i],counts=dict(v)) for (l,p,i),v in totals.items()],days=[dict(label=l,phase=p,counts=dict(v)) for (l,p),v in days.items()],examples=examples,hashes=hashes)
(out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
lines=['# S4Y 投资首轮候选过滤与选取','','64局原样复跑；日初候选只读诊断，不是新策略强度。下面是每个时点首轮候选的过滤/排序；完整自主背景仍有后续组合选择，不能把贪心首轮当成最终路线。','','|背景|阶段|产业|预算不足|非正估值|规模/时间限制|可行但排序靠后|首个选中|预算不足但持有该投入品|','|---|---|---|---:|---:|---:|---:|---:|---:|']
for (l,p,i),v in sorted(totals.items()):
 if i in (0,3,4,10,11):lines.append(f"|{l}|{p}|{names[i]}|{v['capital_budget']}|{v['nonpositive_value']}|{v['scale_limit']+v['animal_limit_or_time']}|{v['eligible_lower_rank']}|{v['chosen_first']}|{v['capital_block_with_owned_input']}|")
lines+=['','## 条件启动预测','','只预测新增目标的当前启动（不是未来收入）。真实同日采购可能部分成交，且原生产同步执行会改变准备状态；未将它标为已验证成交。','','|背景|阶段|诊断时点|草莓原贪心数量|最终意向|条件可启动|小麦原贪心数量|最终意向|条件可启动|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for (l,p),rs in sorted(proposals.items()):
 values=[sum(r[k][i] for r in rs) for i in (3,0) for k in ('raw','intended','started')]
 lines.append(f"|{l}|{p}|{len(rs)}|"+'|'.join(map(str,values))+'|')
lines+=['','“持有投入品”尚未扣除当前其他项目预留，不能直接认定可以免费扩产；下一步需要查净未承诺库存。所有时点来自两个旧seed，过滤次数不是独立样本或可获利机会。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines))
