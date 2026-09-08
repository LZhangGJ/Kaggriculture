"""All-opponent, same-seed paired development summaries. Never selects a winner.

Repeat executions do not become independent samples; two seats share a seed
cluster. No micro-opponent averaging can replace the per-opponent hard gates.
"""
from pathlib import Path
import argparse,hashlib,json,math,statistics

def interval(values):
    m=statistics.fmean(values)
    se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None
    return dict(mean=m,approx95=[m-1.96*se,m+1.96*se] if se and se>0 else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--panels',nargs='+',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--out',required=True);p.add_argument('--prior-baseline-panels',nargs='*',default=[])
    p.add_argument('--prior-baseline-label',default='log1_own0_new0');a=p.parse_args()
    paths=[Path(x) for x in a.panels];panels=[json.loads(x.read_text()) for x in paths]
    for panel in panels:
        assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and not panel['pending_required_opponents']
        assert panel['configurations']==panels[0]['configurations']
        assert panel['build']['binary_sha256']==panels[0]['build']['binary_sha256']
    rows={}
    for panel in panels:
        for r in panel['rows']:
            k=r['variant'],r['opponent'],r['seed'],r['seat']
            if k in rows:
                assert tuple(r[x] for x in ('cash','opponent_cash','margin','win','overflow'))==tuple(rows[k][x] for x in ('cash','opponent_cash','margin','win','overflow'))
            rows[k]=r
    compared=0
    for path in a.prior_baseline_panels:
        old=json.loads(Path(path).read_text());seen=set()
        for r in old['rows']:
            if r['variant']!=a.prior_baseline_label:continue
            k=a.baseline,r['opponent'],r['seed'],r['seat']
            if k in seen:continue
            seen.add(k);ref=rows[k]
            assert tuple(r[x] for x in ('cash','opponent_cash','margin','win','overflow'))==tuple(ref[x] for x in ('cash','opponent_cash','margin','win','overflow')),('baseline changed',k)
            compared+=1
    aggregate={}
    for label in panels[0]['configurations']:
        aggregate[label]={}
        for opponent in panels[0]['identities']:
            own=[r for k,r in rows.items() if k[:2]==(label,opponent)]
            groups={}
            for r in own:groups.setdefault(r['seed'],[]).append(r)
            assert all(sorted(r['seat'] for r in rs)==[0,1] for rs in groups.values())
            scores=[statistics.fmean(int(r['win']) for r in rs) for rs in groups.values()]
            deltas=[tuple(statistics.fmean((int(r['win'])-int(rows[a.baseline,opponent,r['seed'],r['seat']]['win']),
                r['cash']-rows[a.baseline,opponent,r['seed'],r['seat']]['cash'],
                r['margin']-rows[a.baseline,opponent,r['seed'],r['seat']]['margin'])[i] for r in rs) for i in range(3)) for rs in groups.values()]
            wr=statistics.fmean(scores);radius=math.sqrt(math.log(40)/(2*len(scores)))
            s=dict(games=len(own),seeds=len(groups),wins=sum(r['win'] for r in own),ties=sum(r['margin']==0 for r in own),
                win_rate=wr,win_rate_normal_cluster=interval(scores),win_rate_hoeffding95=[max(0,wr-radius),min(1,wr+radius)],
                cash=statistics.fmean(r['cash'] for r in own),opponent_cash=statistics.fmean(r['opponent_cash'] for r in own),
                margin=statistics.fmean(r['margin'] for r in own),paired={key:interval([x[i] for x in deltas]) for i,key in enumerate(('win_rate','cash','margin'))})
            aggregate[label][opponent]=s
    out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    result=dict(status='COMPLETE_DEVELOPMENT_SUMMARY_NOT_GOAL_ACCEPTANCE',aggregate=aggregate,
        input_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
        build=panels[0]['build'],baseline_regression_games=compared,unique_match_records=len(rows),
        final_holdout_used=False,promotion_decision='NOT_MADE_BY_SUMMARIZER',
        caveat='Intervals describe seeded development sampling, not post-selection guarantees or real ladder population.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
    lines=['# 同seed双座位完整结果','','| 配置 | PASS现金 | '+' | '.join(order)+' |','|---|---:|'+'---:|'*8]
    for label,opps in aggregate.items():lines.append('| '+label+' | '+f"{opps['pass']['cash']:,.0f}"+' | '+' | '.join(f"{opps[k]['win_rate']:.1%}" for k in order)+' |')
    lines+=['',f'唯一对局记录：{len(rows)}；关闭开关与旧版逐局核对：{compared}。','全部是开发集；最终验收未执行。']
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':main()
