"""Post-treatment accounting; new investment changes remain explicit."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
EXP=Path(__file__).resolve().parents[1]
def read(p):
    with gzip.open(p,'rt') as f:return {(r['seed'],r['seat']):r for r in json.load(f)['rows']}
def main():
    src=EXP/'receipts/s3v_idle_audit_A50_v1';out=EXP/'receipts/s3v_delivery_diagnostics_v1';out.mkdir(exist_ok=False)
    meta=json.loads((src/'summary.json').read_text());assert meta['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    summaries={(r['label'],r['opponent']):r for r in meta['summary']};results=[];hashes={}
    for opponent in sorted({r['opponent'] for r in meta['summary']}):
        for label,base in [('idle_only','old'),('compile_idle','compile_only')]:
            paths=[src/f'{x}_{opponent}.json.gz' for x in (label,base)];rows,refs=map(read,paths)
            assert rows.keys()==refs.keys()
            for p in paths:hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
            paired=[]
            for k,r in rows.items():
                b=refs[k];s=r['seat'];pl=r['planning'][-1]
                paired.append(dict(seed=k[0],seat=k[1],cash_gain=r['money'][s]-b['money'][s],
                    margin_gain=(r['money'][s]-r['money'][1-s])-(b['money'][s]-b['money'][1-s]),
                    checks=pl['overflow_dispatch_checks'],dispatches=pl['overflow_dispatch_units'],quantity=pl['overflow_dispatch_quantity']))
            new=summaries[label,opponent];old=summaries[base,opponent]
            def facts(row):
                x=row['own_production'];cash=row['own_cash'];return dict(cash=cash['cash'],sold=sum(x['sold']),acquired=sum(x['acquired']),
                    eod_loss=sum(x['eod_loss']),private=sum(x['end_private']),field=sum(x['end_field']),
                    no_effect=sum(x['no_effect']),escapes=sum(x['escaped']),wages=cash['hired'],products=sum(cash['products']),
                    unassigned=row['planner_mean_unassigned_tasks'])
            results.append(dict(label=label,baseline=base,opponent=opponent,games=len(rows),old=facts(old),new=facts(new),
                mean={key:st.fmean(r[key] for r in paired) for key in ('cash_gain','margin_gain','checks','dispatches','quantity')},
                increased=sum(r['cash_gain']>0 for r in paired),equal=sum(r['cash_gain']==0 for r in paired),decreased=sum(r['cash_gain']<0 for r in paired),
                largest_gains=sorted(paired,key=lambda r:r['cash_gain'],reverse=True)[:3],largest_losses=sorted(paired,key=lambda r:r['cash_gain'])[:3]))
    (out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_OFFLINE_ATTRIBUTION_NOT_CAUSAL_GUARANTEE',build=meta['build'],input_hashes=hashes,results=results,
        caveat='Dispatch does not overwrite outstanding plans. Altered cash/market can change later investment and workers; no global claim of identical production. Counts mix products, not currency.'),indent=2))
    lines=['# 空闲运力的现金/生产影响','','| 配置 | 对手 | 现金变化 | 提升/相同/下降 | 追加回仓/局 | 丢货旧→新 | 工资变化 | 未排任务旧→新 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        n,o=r['new'],r['old'];lines.append(f"| {r['label']} | {r['opponent']} | {r['mean']['cash_gain']:+,.0f} | {r['increased']}/{r['equal']}/{r['decreased']} | {r['mean']['dispatches']:.2f} | {o['eod_loss']:.1f}→{n['eod_loss']:.1f} | {n['wages']-o['wages']:+.0f} | {o['unassigned']:.2f}→{n['unassigned']:.2f} |")
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
if __name__=='__main__':main()
