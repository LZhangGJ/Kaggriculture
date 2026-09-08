from pathlib import Path
import hashlib,json,statistics as st
from summarize_declared_value_trial import ci
E=Path(__file__).resolve().parents[1];path=E/'receipts/s4s_fourway_N50_v1/results.json';priorpath=E/'receipts/s4r_eightway_N50_v1/results.json'
p=json.loads(path.read_text());old=json.loads(priorpath.read_text());assert p['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat']);rows={key(r):r for r in p['rows']};prior={key(r):r for r in old['rows']}
count=0;effects=[];interactions=[]
for opp in p['identities']:
    for base in ('all_intraday','all_intraday_insert'):
        rr=[r for r in p['rows'] if r['variant']==base and r['opponent']==opp]
        for r in rr:assert all(r[k]==prior[key(r)][k] for k in ('cash','opponent_cash','margin','win','overflow'));count+=1
        estimates={k:ci([st.fmean(float(rows[base+'_handoff',opp,seed,s][k])-float(rows[base,opp,seed,s][k]) for s in (0,1)) for seed in range(20262701,20262751)]) for k in ('cash','opponent_cash','margin','win')}
        effects.append(dict(before=base,after=base+'_handoff',opponent=opp,estimates=estimates,loss_to_win=sum(not r['win'] and rows[base+'_handoff',opp,r['seed'],r['seat']]['win'] for r in rr),win_to_loss=sum(r['win'] and not rows[base+'_handoff',opp,r['seed'],r['seat']]['win'] for r in rr)))
    interactions.append(dict(opponent=opp,estimates={k:ci([st.fmean(float(rows['all_intraday_insert_handoff',opp,seed,s][k])-float(rows['all_intraday_insert',opp,seed,s][k])-float(rows['all_intraday_handoff',opp,seed,s][k])+float(rows['all_intraday',opp,seed,s][k]) for s in (0,1)) for seed in range(20262701,20262751)]) for k in ('cash','margin','win')}))
out=E/'receipts/s4s_interaction_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_PAIRED_DEVELOPMENT_COMPARISON',unchanged_controls=count,effects=effects,interactions=interactions,input_hashes={str(f.relative_to(E)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (path,priorpath)},holdout_used=False),indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4S 实时对照','','|配置|PASS现金|'+'|'.join(order)+'|均胜率|','|---|---:|'+'---:|'*9]
for label,s in p['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+f"|{st.fmean(s[k]['win_rate'] for k in order)*100:.3f}%|")
lines+=['',f'{len(rows)}场；{count}旧对照逐场复现。每格50 seed×双座位100局；seed聚类，不作为独立泛化证明。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines),flush=True)
