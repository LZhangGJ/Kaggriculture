from pathlib import Path
import hashlib,json,statistics as st
from summarize_declared_value_trial import ci
EXP=Path(__file__).resolve().parents[1]
paths=[EXP/'receipts/s4m1_eightway_N50_v1/results.json',EXP/'receipts/s4l_eightway_N50_v1/results.json']
panel,prior=[json.loads(p.read_text()) for p in paths]
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in panel['rows']};olds={key(r):r for r in prior['rows']}
freeze=json.loads((EXP/'profiles/s4m1/freeze.json').read_text());controls=0;effects=[]
for base,oldlabel in freeze['prior_mapping'].items():
    for opp in panel['identities']:
        for seed in range(20262701,20262751):
            for seat in (0,1):
                now=rows[base,opp,seed,seat];old=olds[oldlabel,opp,seed,seat]
                assert all(now[k]==old[k] for k in ('cash','opponent_cash','margin','win','overflow')),(base,opp,seed,seat)
                controls+=1
bases=('no_intraday','all_intraday','auto_portfolio','auto_portfolio_procure')
for base in bases:
    after=base+'_insert'
    for opp in panel['identities']:
        delta={metric:[] for metric in ('cash','opponent_cash','margin','win')};flips={'loss_to_win':0,'win_to_loss':0}
        for seed in range(20262701,20262751):
            for metric in delta:
                delta[metric].append(st.fmean(float(rows[after,opp,seed,s][metric])-float(rows[base,opp,seed,s][metric]) for s in (0,1)))
            for s in (0,1):
                a,b=rows[after,opp,seed,s]['win'],rows[base,opp,seed,s]['win']
                flips['loss_to_win']+=a and not b;flips['win_to_loss']+=b and not a
        effects.append(dict(context=base,opponent=opp,estimates={k:ci(v) for k,v in delta.items()},flips=flips))
interactions=[]
for opp in panel['identities']:
    ds={k:[] for k in ('cash','margin','win')}
    for seed in range(20262701,20262751):
        for k in ds:
            ds[k].append(st.fmean(float(rows['all_intraday_insert',opp,seed,s][k])-float(rows['all_intraday',opp,seed,s][k])
                -float(rows['no_intraday_insert',opp,seed,s][k])+float(rows['no_intraday',opp,seed,s][k]) for s in (0,1)))
    interactions.append(dict(opponent=opp,estimates={k:ci(v) for k,v in ds.items()}))
out=EXP/'receipts/s4m1_interaction_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_PAIRED_DEVELOPMENT_COMPARISON',unchanged_controls=controls,effects=effects,
    insertion_x_intraday=interactions,input_hashes={str(x.relative_to(EXP)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},holdout_used=False),indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4M1 实时对照','','|配置|PASS现金|'+'|'.join(order)+'|','|---|---:|'+'---:|'*8]
for label,s in panel['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+'|')
lines+=['',f'{len(rows)}场；{controls}旧对照逐场复现S4L。每格N50双座位100局；50 seed聚类成对区间见summary.json；未用O/P。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
