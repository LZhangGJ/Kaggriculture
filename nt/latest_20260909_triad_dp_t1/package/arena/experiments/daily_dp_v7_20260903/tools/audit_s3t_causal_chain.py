"""Paired daily diagnostics, not counterfactual claims from accounting alone."""
from pathlib import Path
import argparse, gzip, hashlib, json, statistics as st

def load(path):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    return {(r['seed'], r['seat']): r for r in data['rows']}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--baseline',default='old')
    p.add_argument('--labels',default='compile_only,hire_only,compile_hire')
    a=p.parse_args()
    src=Path(a.audit);out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    receipt=json.loads((src/'summary.json').read_text())
    assert receipt['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    opponents=sorted({r['opponent'] for r in receipt['summary']})
    results=[];hashes={}
    for opponent in opponents:
        oldpath=src/f'{a.baseline}_{opponent}.json.gz'
        old=load(oldpath);hashes[str(oldpath)]=hashlib.sha256(oldpath.read_bytes()).hexdigest()
        for label in a.labels.split(','):
            path=src/f'{label}_{opponent}.json.gz';rows=load(path)
            assert rows.keys()==old.keys()
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            deltas=[];days=[[] for _ in range(30)]
            first={k:[] for k in ('hands','target_counts','live_counts','sales','cash','compile_drop','intraday_activated')}
            for key,r in rows.items():
                b=old[key];seat=r['seat'];total={k:0. for k in ('cash','sales','wages','products','seeds','animals','land','labor_days','dropped_jobs')}
                touched={k:None for k in first}
                for day in range(30):
                    cur=r['cash_ledger'][day][seat];ref=b['cash_ledger'][day][seat]
                    cp=r['planning'][day];bp=b['planning'][day]
                    d=dict(day=day,cash=cur['end']-ref['end'],sales=sum(cur['sales'])-sum(ref['sales']),
                        wages=cur['hired']-ref['hired'],products=sum(cur['products'])-sum(ref['products']),
                        seeds=sum(cur['seeds'])-sum(ref['seeds']),animals=sum(cur['animals'])-sum(ref['animals']),
                        land=cur['land']-ref['land'],labor_days=cp['max_hands']-bp['max_hands'],
                        dropped_jobs=cp['compile_drop']-bp['compile_drop'])
                    for k in range(12):
                        d[f'live_{k}']=cp['live'][k]-bp['live'][k]
                        d[f'added_{k}']=cp.get('intraday_activated_by_kind',[0]*12)[k]-(r['planning'][day-1].get('intraday_activated_by_kind',[0]*12)[k] if day else 0)
                    d['intraday_activated']=cp.get('intraday_activated',0)-(r['planning'][day-1].get('intraday_activated',0) if day else 0)
                    days[day].append(d)
                    for metric in total:
                        if metric!='cash':total[metric]+=d[metric]
                    total['cash']=d['cash']
                    changed=dict(hands=cp['max_hands']!=bp['max_hands'],target_counts=cp['target']!=bp['target'],
                        live_counts=cp['live']!=bp['live'],sales=cur['sold']!=ref['sold'] or cur['sales']!=ref['sales'],
                        cash=cur['end']!=ref['end'],compile_drop=cp['compile_drop']!=bp['compile_drop'],
                        intraday_activated=cp.get('intraday_activated',0)>0)
                    for metric,change in changed.items():
                        if change and touched[metric] is None:touched[metric]=day
                # Cash differences must be explained by the same actual cash ledger.
                explained=total['sales']-sum(total[k] for k in ('wages','products','seeds','animals','land'))
                assert abs(explained-total['cash'])<1e-6,(label,opponent,key,explained,total['cash'])
                deltas.append(total)
                for metric,value in touched.items():
                    if value is not None:first[metric].append(value)
            results.append(dict(label=label,opponent=opponent,games=len(rows),
                paired_mean={k:st.fmean(d[k] for d in deltas) for k in deltas[0]},
                daily_mean=[{k:st.fmean(d[k] for d in ds) for k in ds[0]} for ds in days],
                first_observed_daily_difference={k:dict(changed_games=len(v),median_day=st.median(v) if v else None,
                    histogram={str(day):v.count(day) for day in sorted(set(v))}) for k,v in first.items()}))
    caveat=('Daily target/live comparisons are counts, not complete layouts or first action divergence. '
            'An execution-only intervention keeps the hiring rule, not the entire future hiring calendar. '
            'Ledger decomposition is exact accounting; it does not identify a unique mediator of the policy effect.')
    (out/'summary.json').write_text(json.dumps(dict(status='PASS_PAIRED_DAILY_ACCOUNTING',build=receipt['build'],
        input_hashes=hashes,results=results,caveat=caveat),indent=2))
    lines=[f'# 对 {a.baseline} 的现金链差额','','每行是同 seed、同座位整局差额的平均；不是各项可独立收回的收益。',
        '','| 配置 | 对手 | 现金差 | 销售额差 | 工资差 | 雇工人日差 | 未排入任务差 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        d=r['paired_mean'];lines.append(f"| {r['label']} | {r['opponent']} | {d['cash']:+,.0f} | {d['sales']:+,.0f} | {d['wages']:+,.0f} | {d['labor_days']:+.1f} | {d['dropped_jobs']:+.2f} |")
    lines+=['',caveat]
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines))

if __name__=='__main__':main()
