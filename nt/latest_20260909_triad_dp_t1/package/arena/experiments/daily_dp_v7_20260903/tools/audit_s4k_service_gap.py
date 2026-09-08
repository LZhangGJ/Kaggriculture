"""Existing live ledgers only; no simulator branches or hindsight selection."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
EXP=Path(__file__).resolve().parents[1]
out=EXP/'receipts/s4k_service_gap_v1';out.mkdir(exist_ok=False)
groups=[];hashes={}
for label,root in [('all_intraday','s4e1'),('all_intraday_portfolio_calendar','s4j2'),('all_intraday_auto_portfolio_calendar','s4j2')]:
    for opp in ('g001','g003','boatlee_v29','kaito_v58','lynn_v5','ecobot_v7','yhay81_six_day','yhay81_three_day'):
        path=EXP/f'receipts/{root}_pool_audit_N50_v1/{label}_{opp}.json.gz'
        hashes[str(path.relative_to(EXP))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path,'rt') as f:data=json.load(f)
        rows=[]
        for r in data['rows']:
            ads=[a for a in r['admissions'] if a['captured'] and a['day']<29]
            feed=[a for a in ads if a['feed_shortage']>0];fert=[a for a in ads if a['fertilizer_shortage']>0]
            rows.append(dict(seed=r['seed'],seat=r['seat'],win=r['money'][r['seat']]>r['money'][1-r['seat']],
                shortage_days=len({a['day'] for a in feed+fert}),feed_short=sum(a['feed_shortage'] for a in feed),
                fert_short=sum(a['fertilizer_shortage'] for a in fert),resource_dropped=sum(a['resource_dropped_jobs'] for a in ads),
                unassigned=sum(p['compile_drop'] for p in r['planning']),
                first_shortage_day=min((a['day'] for a in feed+fert),default=-1),
                short_with_harvest=sum(a['feed_shortage']+a['fertilizer_shortage'] for a in feed+fert
                    if sum(r['production'][a['day']][r['seat']]['acquired'])>0)))
        groups.append(dict(label=label,opponent=opp,games=len(rows),
            affected_games=sum(x['shortage_days']>0 for x in rows),
            averages={k:st.fmean(x[k] for x in rows) for k in ('shortage_days','feed_short','fert_short','resource_dropped','unassigned','short_with_harvest')},rows=rows))
result=dict(status='COMPLETED_REAL_SERVICE_GAP_AUDIT',groups=groups,input_hashes=hashes,
    caveat='Admission shortage is recorded after preparation, not necessarily a whole-day failure. Same-day harvest does not establish usable input or income timing. Counts diagnose a coverage gap, not recoverable profit.')
(out/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# S4K 存量维护缺料证据','','只统计既有实时账本；不是反事实上限。每格50seed双座位。','','|配置|对手|遇到缺料的局数|每局缺料天数|缺小麦数量|缺肥料数量|未排入工作数|','|---|---|---:|---:|---:|---:|---:|']
for g in groups:
    x=g['averages'];lines.append(f"|{g['label']}|{g['opponent']}|{g['affected_games']}|{x['shortage_days']:.2f}|{x['feed_short']:.2f}|{x['fert_short']:.2f}|{x['unassigned']:.2f}|")
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
