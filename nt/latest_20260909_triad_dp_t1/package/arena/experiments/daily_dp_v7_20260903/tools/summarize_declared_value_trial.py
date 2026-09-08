"""Paired live policy changes, not independent per-state counterfactual labels."""
from pathlib import Path
import argparse,json,math,statistics,hashlib


def ci(xs):
    m=statistics.fmean(xs);w=1.96*statistics.stdev(xs)/math.sqrt(len(xs)) if len(xs)>1 else 0
    return dict(mean=m,approx95=[m-w,m+w])


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--prior-panel',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    data=json.loads(Path(a.panel).read_text());prior=json.loads(Path(a.prior_panel).read_text())
    assert data['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
    keys=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
    rows={keys(r):r for r in data['rows']};olds={keys(r):r for r in prior['rows']};regression=0;pairs=[]
    for baseline in ('intraday_stock','intraday_funded','all_intraday'):
        for opponent in data['identities']:
            rs=[r for r in data['rows'] if r['variant']==baseline and r['opponent']==opponent];groups={};changed=won=lost=0
            for r in rs:
                old=olds[keys(r)];assert all(r[k]==old[k] for k in ('cash','opponent_cash','margin','win','overflow'));regression+=1
                t=rows[baseline+'_declared',opponent,r['seed'],r['seat']]
                d=(t['cash']-r['cash'],t['margin']-r['margin'],int(t['win'])-int(r['win']))
                groups.setdefault(r['seed'],[]).append(d);changed+=d[0]!=0 or t['opponent_cash']!=r['opponent_cash'];won+=d[2]>0;lost+=d[2]<0
            pairs.append(dict(baseline=baseline,trial=baseline+'_declared',opponent=opponent,games=len(rs),
                changed_terminal_games=changed,loss_to_win=won,win_to_loss=lost,
                **{k:ci([statistics.fmean(d[i] for d in ds) for ds in groups.values()]) for i,k in enumerate(('cash','margin','win'))}))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='COMPLETE_PAIRED_DEVELOPMENT_NO_PROMOTION',unchanged_controls=regression,pairs=pairs,
                input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(a.panel),Path(a.prior_panel))},holdout_used=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
    lines=['# S4F完整局对照','','| 配置 | PASS现金 | '+' | '.join(order)+' |','|---|---:|'+'---:|'*8]
    for label,s in data['summary'].items(): lines.append('| '+label+f" | {s['pass']['mean_cash']:,.0f} | "+' | '.join(f"{s[k]['wins']}%" for k in order)+' |')
    lines+=['',f'6配置5400局；3旧控制2700局与S4E逐局相同。每格100局=50 seed×2座位，均为开发集。']
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))


if __name__=='__main__': main()
