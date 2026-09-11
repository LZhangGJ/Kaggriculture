"""Whole-live-game 2x2 effects, with seed-clustered uncertainty."""
from pathlib import Path
import argparse,hashlib,json,statistics as st
from summarize_declared_value_trial import ci
EXP=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
panel_path=EXP/'receipts/s4h1_eightway_N50_v1/results.json';prior_path=EXP/'receipts/s4g1_eightway_N50_v1/results.json'
panel=json.loads(panel_path.read_text());prior=json.loads(prior_path.read_text())
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in panel['rows']};olds={key(r):r for r in prior['rows']};controls=0;effects=[]
for context in ('intraday_funded','all_intraday'):
    l00=context+'_fixed_cal';l01=context+'_fixed_cal_feed';l10=context+'_auto_calendar_funded';l11=context+'_auto_cal_feed'
    for opp in panel['identities']:
        groups={};reference=[]
        for seed in range(20262701,20262751):
            for seat in (0,1):
                r00,r01,r10,r11=[rows[k,opp,seed,seat] for k in (l00,l01,l10,l11)]
                prev=olds[l10,opp,seed,seat]
                assert all(r10[k]==prev[k] for k in ('cash','opponent_cash','margin','win','overflow'));controls+=1
                original=olds[context,opp,seed,seat]
                vectors={k:[float(r[k]) for r in (r00,r01,r10,r11,original)] for k in ('cash','margin','win')}
                for k,(x00,x01,x10,x11,xold) in vectors.items():
                    vals=dict(auto_at_feed0=x10-x00,feed_at_fixed=x01-x00,feed_at_auto=x11-x10,
                              interaction=x11-x10-x01+x00,combined_vs_retained=x11-xold)
                    for label,v in vals.items():groups.setdefault((label,k),{}).setdefault(seed,[]).append(v)
        effects.append(dict(context=context,opponent=opp,estimates={label:{k:ci([st.fmean(v) for v in groups[label,k].values()]) for k in ('cash','margin','win')} for label in ('auto_at_feed0','feed_at_fixed','feed_at_auto','interaction','combined_vs_retained')}))
out=EXP/a.out;out.mkdir(parents=True,exist_ok=False)
report=dict(status='COMPLETE_2X2_DEVELOPMENT_COMPARISON',unchanged_controls=controls,effects=effects,
            input_hashes={str(x.relative_to(EXP)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (panel_path,prior_path)},holdout_used=False)
(out/'summary.json').write_text(json.dumps(report,indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4H完整实时对照','','|配置|PASS现金|'+'|'.join(order)+'|','|---|---:|'+'---:|'*8]
for label,s in panel['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+'|')
lines+=['',f'7200条配置对局；{controls}旧控制逐局复现S4G。每格50个开发seed、双座位100局。不是最终验证。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
