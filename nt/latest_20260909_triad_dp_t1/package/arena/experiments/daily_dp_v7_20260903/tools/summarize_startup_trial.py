"""Paired whole-policy tests; no per-seed policy selection or future Oracle."""
from pathlib import Path
import argparse,json,hashlib
from summarize_declared_value_trial import ci
import statistics as st
p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--prior-panel',required=True);p.add_argument('--out',required=True);a=p.parse_args()
data=json.loads(Path(a.panel).read_text());prior=json.loads(Path(a.prior_panel).read_text())
assert data['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in data['rows']};older={key(r):r for r in prior['rows']};controls=0;pairs=[]
for context in ('intraday_funded','all_intraday'):
    for r in data['rows']:
        if r['variant']!=context:continue
        old=older[key(r)]
        assert all(old[k]==r[k] for k in ('cash','opponent_cash','margin','win','overflow'));controls+=1
    comparisons=[(context,context+'_auto'),(context+'_auto',context+'_auto_funded'),
                 (context+'_auto_funded',context+'_auto_calendar_funded'),(context,context+'_auto_calendar_funded')]
    for left,right in comparisons:
        for opponent in data['identities']:
            groups={};won=lost=changed=0
            for r in data['rows']:
                if r['variant']!=left or r['opponent']!=opponent:continue
                t=rows[right,opponent,r['seed'],r['seat']]
                delta=(t['cash']-r['cash'],t['margin']-r['margin'],int(t['win'])-int(r['win']))
                groups.setdefault(r['seed'],[]).append(delta);won+=delta[2]>0;lost+=delta[2]<0
                changed+=delta[0]!=0 or t['opponent_cash']!=r['opponent_cash']
            pairs.append(dict(baseline=left,trial=right,opponent=opponent,changed_games=changed,loss_to_win=won,win_to_loss=lost,
                **{k:ci([st.fmean(d[i] for d in ds) for ds in groups.values()]) for i,k in enumerate(('cash','margin','win'))}))
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
result=dict(status='COMPLETE_PAIRED_DEVELOPMENT_NOT_GOAL_ACCEPTANCE',unchanged_controls=controls,pairs=pairs,
    input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(a.panel),Path(a.prior_panel))},holdout_used=False)
(out/'summary.json').write_text(json.dumps(result,indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4G同批开局对照','','|配置|PASS现金|'+'|'.join(order)+'|','|---|---:|'+'---:|'*8]
for label,s in data['summary'].items():lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+'|')
lines+=['',f'{len(data["rows"])}条配置对局；{controls}旧控制逐局相同。每格50个开发seed、双座位，共100局。组合变化不是单项效应。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
