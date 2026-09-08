"""Read only completed live ledgers; compare mechanism/long-chain outcomes."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics as st
from summarize_startup_cash import summarize
p=argparse.ArgumentParser();p.add_argument('--round',default='s4k');a=p.parse_args();r=a.round;assert r in ('s4k','s4k2','s4l')
EXP=Path(__file__).resolve().parents[1];newroot=EXP/f'receipts/{r}_pool_audit_N50_v1'
panel=json.loads((EXP/f'receipts/{r}_eightway_N50_v1/results.json').read_text())
meta=json.loads((newroot/'summary.json').read_text());assert meta['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
out=EXP/f'receipts/{r}_cash_chain_N50_v1';out.mkdir(exist_ok=False);hashes={};cache={};pairs=[]
keys={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
counter_names=('service_checks','service_recovered_feed','service_recovered_fertilize','service_buys','service_financed','service_deposits')
def load(label,opp):
    root=(EXP/'receipts/s4e1_pool_audit_N50_v1' if label=='all_intraday' else
          EXP/'receipts/s4j2_pool_audit_N50_v1' if label=='all_intraday_auto_portfolio_calendar' else newroot)
    if r=='s4l' and not label.endswith('_causal_pickup') and label.endswith('_procure'):
        root=EXP/'receipts/s4k2_pool_audit_N50_v1'
    path=root/f'{label}_{opp}.json.gz'
    if path not in cache:
        hashes[str(path.relative_to(EXP))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path,'rt',encoding='utf8') as f:p=json.load(f)
        rows=p['rows']
        for x in rows:
            ref=keys[label,opp,x['seed'],x['seat']]
            assert x['money'][x['seat']]==ref['cash'] and x['money'][1-x['seat']]==ref['opponent_cash']
            assert x['overflow']==ref['overflow']
        v=summarize(rows);v['counters']={k:st.fmean(x['planning'][-1].get(k,0) for x in rows) for k in counter_names}
        v['counter_increments_by_day']=[{k:st.fmean(x['planning'][d].get(k,0)-(x['planning'][d-1].get(k,0) if d else 0) for x in rows) for k in counter_names} for d in range(30)]
        v['daily_planning']=[dict(cash=st.fmean(x['cash_ledger'][d][x['seat']]['end'] for x in rows),
            hands=st.fmean(x['planning'][d]['max_hands'] for x in rows),
            unassigned=st.fmean(x['planning'][d]['compile_drop'] for x in rows),
            target=[st.fmean(x['planning'][d]['target'][i] for x in rows) for i in range(12)],
            live=[st.fmean(x['planning'][d]['live'][i] for x in rows) for i in range(12)]) for d in range(30)]
        v['unfed_by_day']=[st.fmean(x['planning'][d]['unfed'] for x in rows) for d in range(30)]
        v['production']=p['summary']['own_production'];v['cash_ledger']=p['summary']['own_cash']
        cache[path]=v
    return cache[path]
bases=('all_intraday','all_intraday_auto_portfolio_calendar')
if r=='s4l':bases=('all_intraday','all_intraday_procure','all_intraday_auto_portfolio_calendar','all_intraday_auto_portfolio_calendar_procure')
for base in bases:
    for opp in panel['identities']:
        comparisons=((base,base+'_causal_pickup'),) if r=='s4l' else ((base,base+'_recover'),(base+'_recover',base+'_procure'),(base+'_procure',base+'_finance'),(base,base+'_finance'))
        for before,after in comparisons:
            b=load(before,opp);v=load(after,opp)
            deltas={d:{k:[y-x for x,y in zip(b['snapshots'][d][k],values)] if isinstance(values,list) else values-b['snapshots'][d][k]
                       for k,values in snap.items()} for d,snap in v['snapshots'].items()}
            pairs.append(dict(before=before,after=after,opponent=opp,baseline=b,result=v,delta=deltas))
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_ACTUAL_CASH_AND_SERVICE_ACCOUNTING',rows=pairs,input_hashes=hashes,holdout_used=False,
    caveat='Full-policy paired effects; production deltas alone are not isolated marginal causal values.'),indent=2))
lines=[f'# {r} 日内恢复与真实现金链','','|版本|对手|喂养重接|施肥重接|采购|融资|定量入库|现金变化|销售变化|供料支出变化|',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in pairs:
    if x['before'] not in bases or x['opponent'] not in ('pass','g001','g003'):continue
    d=x['delta'][29];c=x['result']['counters'];lines.append('|'+x['after']+'|'+x['opponent']+'|'+'|'.join(f'{c[k]:.2f}' for k in counter_names[1:])+'|'+'|'.join(f'{d[k]:,.0f}' for k in ('cash','sales','supplies'))+'|')
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
