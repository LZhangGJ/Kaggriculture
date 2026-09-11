"""Paired factorial interaction, clustering the two seats by seed."""
from pathlib import Path
import argparse,hashlib,json,statistics as st,math
def interval(xs):
    mean=st.fmean(xs);radius=1.96*st.stdev(xs)/math.sqrt(len(xs)) if len(xs)>1 else 0
    return dict(mean=mean,approx95=[mean-radius,mean+radius])
def main():
    p=argparse.ArgumentParser();p.add_argument('--panels',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--baseline',default='old');p.add_argument('--factor-a',default='split_only')
    p.add_argument('--factor-b',default='regret_only');p.add_argument('--both',default='split_regret');a=p.parse_args()
    rows={};build=None;hashes={}
    for name in a.panels:
        path=Path(name);panel=json.loads(path.read_text());assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
        if build is None:build=panel['build']
        assert build['binary_sha256']==panel['build']['binary_sha256']
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for r in panel['rows']:
            key=r['variant'],r['opponent'],r['seed'],r['seat']
            if key in rows:assert rows[key]==r
            rows[key]=r
    results=[]
    for opponent in sorted({k[1] for k in rows}):
        seeds=sorted({k[2] for k in rows if k[1]==opponent});metrics={k:[] for k in ('win','cash','margin')}
        for seed in seeds:
            for metric in metrics:
                def v(label,seat):return float(rows[label,opponent,seed,seat][metric])
                metrics[metric].append(st.fmean(v(a.both,s)-v(a.factor_a,s)-v(a.factor_b,s)+v(a.baseline,s) for s in (0,1)))
        results.append(dict(opponent=opponent,seeds=len(seeds),interaction={k:interval(v) for k,v in metrics.items()}))
    out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    (out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_PAIRED_INTERACTION',build=build,input_hashes=hashes,results=results,
        formula=f'{a.both} - {a.factor_a} - {a.factor_b} + {a.baseline}',
        caveat='Development seed clusters. Positive interaction does not imply that the combined policy beats baseline or other agents.'),indent=2))
    lines=['# 两开关交互项','',f'{a.both} − {a.factor_a} − {a.factor_b} ＋ {a.baseline}。正数仅表示超出两个单项的相加效果，不等于组合已足够强。','','| 对手 | 胜率交互 | 现金交互 | 分差交互及近似95%区间 |','|---|---:|---:|---:|']
    for r in results:
        d=r['interaction'];lo,hi=d['margin']['approx95'];lines.append(f"| {r['opponent']} | {100*d['win']['mean']:+.1f} pp | {d['cash']['mean']:+,.0f} | {d['margin']['mean']:+,.0f} [{lo:,.0f}, {hi:,.0f}] |")
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
if __name__=='__main__':main()
