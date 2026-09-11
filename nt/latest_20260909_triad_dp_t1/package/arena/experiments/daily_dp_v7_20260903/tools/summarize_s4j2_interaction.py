"""Paired, seed-clustered calendar effects within both startup/executor contexts."""
from pathlib import Path
import argparse,hashlib,json,statistics as st
from summarize_declared_value_trial import ci
EXP=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--round',default='s4j2');a=p.parse_args()
panel_path=EXP/f'receipts/{a.round}_eightway_N50_v1/results.json'
prior_path=EXP/'receipts/s4g1_eightway_N50_v1/results.json'
panel=json.loads(panel_path.read_text());prior=json.loads(prior_path.read_text())
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in panel['rows']};olds={key(r):r for r in prior['rows']};controls=0;effects=[]
for context in ('intraday_funded','all_intraday'):
    for opp in panel['identities']:
        for label in (context,context+'_auto'):
            for seed in range(20262701,20262751):
                for seat in (0,1):
                    now=rows[label,opp,seed,seat];old=olds[label,opp,seed,seat]
                    assert all(now[k]==old[k] for k in ('cash','opponent_cash','margin','win','overflow'));controls+=1
        for prefix in ('','_auto'):
            old=context+prefix+'_portfolio';new=old+'_calendar';baseline=context+prefix
            for name,b in (('calendar_only',old),('portfolio_and_calendar',baseline)):
                delta={metric:[] for metric in ('cash','opponent_cash','margin','win')}
                for seed in range(20262701,20262751):
                    for metric in delta:
                        delta[metric].append(st.fmean(float(rows[new,opp,seed,s][metric])-float(rows[b,opp,seed,s][metric]) for s in (0,1)))
                effects.append(dict(context=context,startup=prefix or 'fixed',opponent=opp,effect=name,
                                    before=b,after=new,estimates={k:ci(v) for k,v in delta.items()}))
out=EXP/f'receipts/{a.round}_interaction_N50_v1';out.mkdir(exist_ok=False)
report=dict(status='COMPLETE_PAIRED_DEVELOPMENT_COMPARISON',unchanged_controls=controls,effects=effects,
            input_hashes={str(x.relative_to(EXP)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (panel_path,prior_path)},holdout_used=False)
(out/'summary.json').write_text(json.dumps(report,indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=[f'# {a.round}实时对照','','|配置|PASS现金|'+'|'.join(order)+'|','|---|---:|'+'---:|'*8]
for label,s in panel['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+'|')
lines+=['',f'{len(rows)}条配置对局；{controls}旧对照逐局复现S4G。每格50个开发seed、双座位100局；不是100个独立seed，不是最终验收。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
