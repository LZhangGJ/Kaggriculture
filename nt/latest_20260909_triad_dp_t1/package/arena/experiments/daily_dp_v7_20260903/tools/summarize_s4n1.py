from pathlib import Path
import hashlib,json,statistics as st
from summarize_declared_value_trial import ci
E=Path(__file__).resolve().parents[1];paths=[E/'receipts/s4n1_eightway_N50_v1/results.json',E/'receipts/s4m1_eightway_N50_v1/results.json']
panel,prior=[json.loads(p.read_text()) for p in paths];assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in panel['rows']};old={key(r):r for r in prior['rows']};controls=0;effects=[];interactions=[]
for base in ('all_intraday','all_intraday_insert'):
    for opp in panel['identities']:
        delta={k:[] for k in ('cash','opponent_cash','margin','win')};flips=dict(loss_to_win=0,win_to_loss=0)
        for seed in range(20262701,20262751):
            for seat in (0,1):
                now=rows[base,opp,seed,seat];was=old[base,opp,seed,seat]
                assert all(now[k]==was[k] for k in ('cash','opponent_cash','margin','win','overflow'));controls+=1
                a,b=rows[base+'_workforce',opp,seed,seat]['win'],now['win'];flips['loss_to_win']+=a and not b;flips['win_to_loss']+=b and not a
            for k in delta:delta[k].append(st.fmean(float(rows[base+'_workforce',opp,seed,s][k])-float(rows[base,opp,seed,s][k]) for s in (0,1)))
        effects.append(dict(context=base,opponent=opp,estimates={k:ci(v) for k,v in delta.items()},flips=flips))
for opp in panel['identities']:
    ds={k:[] for k in ('cash','margin','win')}
    for seed in range(20262701,20262751):
        for k in ds:ds[k].append(st.fmean(float(rows['all_intraday_insert_workforce',opp,seed,s][k])-float(rows['all_intraday_insert',opp,seed,s][k])-float(rows['all_intraday_workforce',opp,seed,s][k])+float(rows['all_intraday',opp,seed,s][k]) for s in (0,1)))
    interactions.append(dict(opponent=opp,estimates={k:ci(v) for k,v in ds.items()}))
out=E/'receipts/s4n1_interaction_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_PAIRED_DEVELOPMENT_COMPARISON',unchanged_controls=controls,effects=effects,workforce_x_insertion=interactions,input_hashes={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},holdout_used=False),indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4N1 实时对照','','|配置|PASS现金|'+'|'.join(order)+'|','|---|---:|'+'---:|'*8]
for label,s in panel['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+'|')
lines+=['',f'{len(rows)}场；{controls}旧对照逐场复现S4M1。每格50 seed双座位100局；成对区间按seed聚类；O/P未使用。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
