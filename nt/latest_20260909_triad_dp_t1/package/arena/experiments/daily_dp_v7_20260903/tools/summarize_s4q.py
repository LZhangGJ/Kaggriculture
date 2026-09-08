from pathlib import Path
import hashlib,json,statistics as st
from summarize_declared_value_trial import ci
E=Path(__file__).resolve().parents[1];path=E/'receipts/s4q_sixway_N50_v1/results.json';priorpath=E/'receipts/s4n1_eightway_N50_v1/results.json'
panel=json.loads(path.read_text());prior=json.loads(priorpath.read_text());assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat']);rows={key(r):r for r in panel['rows']};old={key(r):r for r in prior['rows']};controls=0;effects=[]
for base in ('all_intraday','all_intraday_insert'):
    for opp in panel['identities']:
        for seed in range(20262701,20262751):
            for seat in (0,1):
                a,b=rows[base,opp,seed,seat],old[base,opp,seed,seat]
                assert all(a[k]==b[k] for k in ('cash','opponent_cash','margin','win','overflow'));controls+=1
        for before,after in ((base,base+'_value'),(base+'_value',base+'_value_public'),(base,base+'_value_public')):
            ds={k:[] for k in ('cash','opponent_cash','margin','win')}
            for seed in range(20262701,20262751):
                for k in ds:ds[k].append(st.fmean(float(rows[after,opp,seed,s][k])-float(rows[before,opp,seed,s][k]) for s in (0,1)))
            flips={name:sum(condition(bool(rows[before,opp,seed,s]['win']),bool(rows[after,opp,seed,s]['win'])) for seed in range(20262701,20262751) for s in (0,1)) for name,condition in (('loss_to_win',lambda a,b:not a and b),('win_to_loss',lambda a,b:a and not b))}
            effects.append(dict(before=before,after=after,opponent=opp,estimates={k:ci(v) for k,v in ds.items()},flips=flips))
interactions=[]
for suffix in ('_value','_value_public'):
    for opp in panel['identities']:
        ds={k:[] for k in ('cash','margin','win')}
        for seed in range(20262701,20262751):
            for k in ds:ds[k].append(st.fmean(float(rows['all_intraday_insert'+suffix,opp,seed,s][k])-float(rows['all_intraday_insert',opp,seed,s][k])-float(rows['all_intraday'+suffix,opp,seed,s][k])+float(rows['all_intraday',opp,seed,s][k]) for s in (0,1)))
        interactions.append(dict(suffix=suffix,opponent=opp,estimates={k:ci(v) for k,v in ds.items()}))
out=E/'receipts/s4q_interaction_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_PAIRED_DEVELOPMENT_COMPARISON',unchanged_controls=controls,effects=effects,interactions=interactions,input_hashes={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,priorpath)},holdout_used=False),indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4Q 实时对照','','|配置|PASS现金|'+'|'.join(order)+'|','|---|---:|'+'---:|'*8]
for label,s in panel['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+'|')
lines+=['',f'{len(rows)}场；{controls}旧对照逐场复现。每格50 seed×双座位100局。区间按seed聚类；不使用新未见集选型。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines),flush=True)
