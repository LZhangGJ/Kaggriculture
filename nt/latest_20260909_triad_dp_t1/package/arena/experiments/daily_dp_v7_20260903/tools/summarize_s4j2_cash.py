"""Actual accounting; an old receipt is reused only after exact live replay checks."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics as st
from summarize_startup_cash import summarize
EXP=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--round',default='s4j2');a=p.parse_args();r=a.round
panel_path=EXP/f'receipts/{r}_eightway_N50_v1/results.json';panel=json.loads(panel_path.read_text())
old_panel_path=EXP/'receipts/s4i2_eightway_N50_v1/results.json';old_panel=json.loads(old_panel_path.read_text())
newroot=EXP/f'receipts/{r}_pool_audit_N50_v1';meta=json.loads((newroot/'summary.json').read_text())
assert meta['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
out=EXP/f'receipts/{r}_cash_chain_N50_v1';out.mkdir(exist_ok=False);pairs=[];hashes={};cache={}
for path in (panel_path,old_panel_path,newroot/'summary.json'):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
key=lambda x:(x['variant'],x['opponent'],x['seed'],x['seat'])
now={key(x):x for x in panel['rows']};old={key(x):x for x in old_panel['rows']}
verified=0
def load(root,label,opp):
    path=root/f'{label}_{opp}.json.gz'
    if path not in cache:
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path,'rt',encoding='utf8') as f:payload=json.load(f)
        rows=payload['rows'];stats=summarize(rows);stats['production']=payload['summary']['own_production']
        stats['rival_production']=payload['summary']['rival_production']
        stats['rival_cash']=payload['summary']['rival_cash']
        stats['planning']={field:[st.fmean(x['planning'][d][field] for x in rows) for d in range(30)] for field in ('compile_drop','degraded','max_hands','land')}
        stats['counters']={field:st.fmean(x['planning'][-1].get(field,0) for x in rows) for field in ('portfolio_generated','portfolio_evaluated','portfolio_switches','portfolio_rejected')}
        cache[path]=stats
    return cache[path]
for e in meta['summary']:
    label=e['label'];opp=e['opponent'];context='all_intraday' if label.startswith('all_intraday') else 'intraday_funded'
    comparisons=[('versus_retained_control',EXP/'receipts/s4e1_pool_audit_N50_v1',context)]
    if '_auto_' in label:
        before_label=label.removesuffix('_calendar')
        for seed in range(20262701,20262751):
            for seat in (0,1):
                x=now[before_label,opp,seed,seat];y=old[before_label,opp,seed,seat]
                assert all(x[k]==y[k] for k in ('cash','opponent_cash','win','margin','overflow'));verified+=1
        comparisons.append(('calendar_only_at_autonomous_start',EXP/'receipts/s4i2_pool_audit_N50_v1',before_label))
    after=load(newroot,label,opp)
    for kind,root,before_label in comparisons:
        before=load(root,before_label,opp)
        # Controls have been independently replayed by the paired panel script.
        assert before['terminal_cash']==panel['summary'][before_label][opp]['mean_cash']
        delta={d:{k:[y-x for x,y in zip(before['snapshots'][d][k],v)] if isinstance(v,list) else v-before['snapshots'][d][k]
                  for k,v in snap.items()} for d,snap in after['snapshots'].items()}
        pairs.append(dict(comparison=kind,label=label,baseline=before_label,opponent=opp,before=before,after=after,delta=delta))
data=dict(status='COMPLETE_ACTUAL_PAIRED_ACCOUNTING',historical_auto_controls_reverified=verified,
          rows=pairs,input_hashes=hashes,holdout_used=False,
          caveat='Amounts reconcile actual live trajectories. A whole-policy contrast does not isolate the causal contribution of any individual crop or sale. No future data enters the agent.')
(out/'summary.json').write_text(json.dumps(data,indent=2))
lines=['# S4J2 实际现金链','','|比较|配置|对手|现金变化|销售变化|供料支出变化|种子支出变化|动物支出变化|工资变化|土地变化|',
       '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in pairs:
    if x['opponent'] not in ('pass','g001','g003'):continue
    d=x['delta'][29];lines.append('|'+x['comparison']+'|'+x['label']+'|'+x['opponent']+'|'+'|'.join(f'{d[k]:,.0f}' for k in ('cash','sales','supplies','seed_cost','animal_cost','wages','land'))+'|')
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
