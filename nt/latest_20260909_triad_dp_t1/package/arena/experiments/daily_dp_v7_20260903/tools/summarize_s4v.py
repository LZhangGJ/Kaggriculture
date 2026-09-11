from pathlib import Path
import hashlib,json,statistics as st
from summarize_declared_value_trial import ci
E=Path(__file__).resolve().parents[1];path=E/'receipts/s4v_twelveway_N50_v1/results.json';oldpath=E/'receipts/s4u_fiveway_N50_v1/results.json'
p=json.loads(path.read_text());old=json.loads(oldpath.read_text());assert p['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in p['rows']};previous={key(r):r for r in old['rows']};bases=('all_intraday','all_intraday_insert','full_chain_autonomous')
checked=0
for k,r in previous.items():
    if k[0] in bases:assert all(r[f]==rows[k][f] for f in ('cash','opponent_cash','margin','win','overflow')),k;checked+=1
assert checked==2700 and len(rows)==10800
effects=[]
for base in bases:
    for suffix in ('market','feed','both'):
        after=base+'_'+suffix
        for opp in p['identities']:
            estimates={k:ci([st.fmean(float(rows[after,opp,seed,seat][k])-float(rows[base,opp,seed,seat][k]) for seat in (0,1)) for seed in range(20262701,20262751)]) for k in ('cash','opponent_cash','margin','win')}
            effects.append(dict(before=base,after=after,opponent=opp,estimates=estimates))
interaction=[]
for base in bases:
    for opp in p['identities']:
        estimates={k:ci([st.fmean(float(rows[base+'_both',opp,seed,seat][k])-float(rows[base+'_market',opp,seed,seat][k])-float(rows[base+'_feed',opp,seed,seat][k])+float(rows[base,opp,seed,seat][k]) for seat in (0,1)) for seed in range(20262701,20262751)]) for k in ('cash','margin','win')}
        interaction.append(dict(base=base,opponent=opp,estimates=estimates))
out=E/'receipts/s4v_comparison_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_PAIRED_DEVELOPMENT_COMPARISON',unchanged_controls=checked,effects=effects,interactions=interaction,input_hashes={str(f.relative_to(E)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (path,oldpath)},holdout_used=False,final_goal_acceptance=False),indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4V 既有建议兑现与漏喂插入','','|配置|PASS现金|'+'|'.join(order)+'|均胜率|','|---|---:|'+'---:|'*9]
for label,s in p['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+f"|{st.fmean(s[k]['win_rate'] for k in order)*100:.3f}%|")
lines+=['','每格50开发seed×双座位100局；旧2700控制逐局一致。配对区间及双开交互见summary.json。不是未见集验收，不自动晋级。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines),flush=True)
