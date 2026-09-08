"""Paired development results. No per-game policy or hindsight selection."""
from pathlib import Path
import hashlib,json,statistics as st
from summarize_declared_value_trial import ci
EXP=Path(__file__).resolve().parents[1]
paths=[EXP/'receipts/s4l_eightway_N50_v1/results.json',EXP/'receipts/s4k2_eightway_N50_v1/results.json']
panel,prior=[json.loads(p.read_text()) for p in paths]
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in panel['rows']};olds={key(r):r for r in prior['rows']}
bases=('all_intraday','all_intraday_procure','all_intraday_auto_portfolio_calendar','all_intraday_auto_portfolio_calendar_procure')
controls=0;effects=[]
for base in bases:
    after=base+'_causal_pickup'
    for opp in panel['identities']:
        delta={metric:[] for metric in ('cash','opponent_cash','margin','win')}
        for seed in range(20262701,20262751):
            for seat in (0,1):
                now=rows[base,opp,seed,seat];old=olds[base,opp,seed,seat]
                assert all(now[k]==old[k] for k in ('cash','opponent_cash','margin','win','overflow'));controls+=1
            for metric in delta:
                delta[metric].append(st.fmean(float(rows[after,opp,seed,s][metric])-float(rows[base,opp,seed,s][metric]) for s in (0,1)))
        effects.append(dict(context=base,opponent=opp,before=base,after=after,estimates={k:ci(v) for k,v in delta.items()}))
out=EXP/'receipts/s4l_interaction_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_PAIRED_DEVELOPMENT_COMPARISON',unchanged_controls=controls,effects=effects,
    input_hashes={str(x.relative_to(EXP)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},holdout_used=False),indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4L 实时对照','','|配置|PASS现金|'+'|'.join(order)+'|','|---|---:|'+'---:|'*8]
for label,s in panel['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+'|')
lines+=['',f'{len(rows)}场；{controls}旧对照逐场复现S4K2。每格N50双座位100局，未用O/P；不是最终验收。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
