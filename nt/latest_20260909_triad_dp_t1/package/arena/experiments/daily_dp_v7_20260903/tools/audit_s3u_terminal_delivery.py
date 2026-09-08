"""Verify that the intervention changes only the terminal day and trace its cash."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics as st

def read(path):
    with gzip.open(path,'rt') as f:return {(r['seed'],r['seat']):r for r in json.load(f)['rows']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    src=Path(a.audit);out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    receipt=json.loads((src/'summary.json').read_text());assert receipt['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    results=[];hashes={}
    for opponent in sorted({s['opponent'] for s in receipt['summary']}):
        for label,baseline in [('terminal_only','old'),('compile_terminal','compile_only')]:
            paths=[src/f'{x}_{opponent}.json.gz' for x in (label,baseline)];rows,refs=map(read,paths)
            for path in paths:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            assert rows.keys()==refs.keys();paired=[]
            for key,r in rows.items():
                b=refs[key];seat=r['seat']
                for field in ('production','cash_ledger','planning','admissions'):
                    assert r[field][:29]==b[field][:29],('nonterminal changed',opponent,label,key,field)
                end=r['production'][-1][seat];old=b['production'][-1][seat]
                ledger=r['cash_ledger'][-1][seat];prev=b['cash_ledger'][-1][seat]
                gain=r['money'][seat]-b['money'][seat];plan=r['planning'][-1]
                paired.append(dict(seed=key[0],seat=seat,cash_gain=gain,
                    margin_gain=(r['money'][seat]-r['money'][1-seat])-(b['money'][seat]-b['money'][1-seat]),
                    win_gain=int(r['money'][seat]>r['money'][1-seat])-int(b['money'][seat]>b['money'][1-seat]),
                    final_private=sum(end['end_private']),baseline_final_private=sum(old['end_private']),
                    final_field=sum(end['end_field']),baseline_final_field=sum(old['end_field']),
                    acquired=sum(end['acquired']),baseline_acquired=sum(old['acquired']),
                    sold=sum(ledger['sold']),baseline_sold=sum(prev['sold']),
                    actual_dropped_jobs=plan['compile_drop'],baseline_dropped_jobs=b['planning'][-1]['compile_drop'],
                    evaluations=plan['terminal_schedule_evaluations'],switches=plan['terminal_schedule_switches'],
                    projected_gain=plan['terminal_expected_cash_gain'],
                    prediction_error=gain-plan['terminal_expected_cash_gain']))
            means={k:st.fmean(r[k] for r in paired) for k in paired[0] if k not in ('seed','seat')}
            results.append(dict(label=label,baseline=baseline,opponent=opponent,games=len(rows),
                nonterminal_exact_games=len(rows),mean=means,
                increased_cash=sum(r['cash_gain']>0 for r in paired),decreased_cash=sum(r['cash_gain']<0 for r in paired),
                equal_cash=sum(r['cash_gain']==0 for r in paired),
                largest_gains=sorted(paired,key=lambda x:x['cash_gain'],reverse=True)[:3],
                largest_losses=sorted(paired,key=lambda x:x['cash_gain'])[:3]))
    (out/'summary.json').write_text(json.dumps(dict(status='PASS_TERMINAL_ONLY_INTERVENTION',build=receipt['build'],input_hashes=hashes,results=results,
        caveat='Identical first 29 days verified. Cash forecast ignores unknown rival orders and future town changes; actual gains may differ. '
        'Inventory/field totals mix product units and are diagnostic, not monetary targets.'),indent=2))
    lines=['# 终局波次：实际影响与前29天不变核对','','| 配置 | 对手 | 现金增益 | 提升/持平/下降局数 | 原终局私人存货 | 新存货 | 预测增益 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        d=r['mean'];lines.append(f"| {r['label']} | {r['opponent']} | {d['cash_gain']:+,.0f} | {r['increased_cash']}/{r['equal_cash']}/{r['decreased_cash']} | {d['baseline_final_private']:.1f} | {d['final_private']:.1f} | {d['projected_gain']:+,.0f} |")
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
if __name__=='__main__':main()
