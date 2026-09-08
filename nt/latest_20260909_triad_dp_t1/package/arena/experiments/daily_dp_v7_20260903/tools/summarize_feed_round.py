"""Aggregate every tested feed variant; paired deltas retain seed clusters."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(rows):
    groups={}
    for r in rows:groups.setdefault(r['seed'],[]).append(r)
    scores=[statistics.fmean(float(r['win']) for r in group) for group in groups.values()]
    rate=statistics.fmean(scores);se=statistics.stdev(scores)/math.sqrt(len(scores))
    return dict(games=len(rows),independent_seeds=len(groups),win_rate=rate,wins=sum(r['win'] for r in rows),
        mean_cash=statistics.fmean(r['cash'] for r in rows),mean_margin=statistics.fmean(r['margin'] for r in rows),
        approximate_seed_cluster_95pct_interval=[max(0,rate-1.96*se),min(1,rate+1.96*se)] if se>0 else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    panels={k:json.loads((EXP/f'receipts/s3j_feed_{k}50_v1/results.json').read_text()) for k in 'ABC'}
    for panel in panels.values():assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and not panel['pending_required_opponents']
    aggregate={}
    for group,parts in [('AB','AB'),('C','C')]:
        selected=[panels[x] for x in parts];labels=selected[0]['configurations'];aggregate[group]={}
        for label in labels:
            aggregate[group][label]={};base='log1_own0_new0' if label.startswith('log1') else 'log0_own0_new0'
            for opponent in selected[0]['summary'][label]:
                rows=[r for panel in selected for r in panel['rows'] if r['variant']==label and r['opponent']==opponent]
                refs={(r['seed'],r['seat']):r for panel in selected for r in panel['rows'] if r['variant']==base and r['opponent']==opponent}
                assert len(refs)==len(rows)
                gains={}
                for r in rows:
                    ref=refs[r['seed'],r['seat']]
                    gains.setdefault(r['seed'],[]).append((int(r['win'])-int(ref['win']),r['cash']-ref['cash'],r['margin']-ref['margin']))
                delta=[tuple(statistics.fmean(v[i] for v in vs) for i in range(3)) for vs in gains.values()]
                s=stats(rows);s['paired_baseline']=base
                for i,k in enumerate(['win_rate','cash','margin']):
                    values=[v[i] for v in delta];m=statistics.fmean(values);se=statistics.stdev(values)/math.sqrt(len(values))
                    s['paired_'+k+'_delta']=m;s['paired_'+k+'_approx95']=[m-1.96*se,m+1.96*se] if se>0 else None
                aggregate[group][label][opponent]=s
    result=dict(status='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE',aggregate=aggregate,
        inputs={k:sha(EXP/f'receipts/s3j_feed_{k}50_v1/results.json') for k in 'ABC'},
        final_holdout_used=False,promotion=False,caveat='All eight opponent results retained. New C is development confirmation, not final holdout. No post-selection population guarantee.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# S3J：饲料需求估值消融全量结果','', 'AB为100seed双座位200局；C为新增50seed双座位100局。各对手单独统计，重复不计。', '']
    order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
    for group in aggregate:
        lines+=['## '+group,'','| 配置 | PASS现金 | '+' | '.join(order)+' |','|---|---:|'+'---:|'*8]
        for label,opps in aggregate[group].items():lines.append('| '+label+' | '+f"{opps['pass']['mean_cash']:,.0f}"+' | '+' | '.join(f"{opps[k]['win_rate']:.1%}" for k in order)+' |')
        lines.append('')
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':main()
