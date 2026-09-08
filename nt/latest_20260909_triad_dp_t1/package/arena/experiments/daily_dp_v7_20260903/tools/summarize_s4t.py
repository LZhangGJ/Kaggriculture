from pathlib import Path
import hashlib,json,statistics as st
from summarize_declared_value_trial import ci

E=Path(__file__).resolve().parents[1]
path=E/'receipts/s4t_fiveway_N50_v1/results.json';oldpath=E/'receipts/s4s_fourway_N50_v1/results.json'
p=json.loads(path.read_text());old=json.loads(oldpath.read_text())
assert p['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE'
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat'])
rows={key(r):r for r in p['rows']};previous={key(r):r for r in old['rows']}
assert len(rows)==4500
controls=0
for r in p['rows']:
    if r['variant'] in ('all_intraday','all_intraday_insert'):
        assert all(r[k]==previous[key(r)][k] for k in ('cash','opponent_cash','margin','win','overflow'))
        controls+=1
assert controls==1800
pairs=[(base,after) for base in ('all_intraday','all_intraday_insert') for after in ('full_workers25','full_chain','full_chain_autonomous')]
pairs += [('full_workers25','full_chain'),('full_chain','full_chain_autonomous')]
effects=[]
for before,after in pairs:
    for opp in p['identities']:
        estimates={k:ci([st.fmean(float(rows[after,opp,seed,seat][k])-float(rows[before,opp,seed,seat][k]) for seat in (0,1))
             for seed in range(20262701,20262751)]) for k in ('cash','opponent_cash','margin','win')}
        effects.append(dict(before=before,after=after,opponent=opp,estimates=estimates,
            loss_to_win=sum(not rows[before,opp,seed,seat]['win'] and rows[after,opp,seed,seat]['win'] for seed in range(20262701,20262751) for seat in (0,1)),
            win_to_loss=sum(rows[before,opp,seed,seat]['win'] and not rows[after,opp,seed,seat]['win'] for seed in range(20262701,20262751) for seat in (0,1))))
out=E/'receipts/s4t_comparison_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_PAIRED_DEVELOPMENT_COMPARISON',unchanged_controls=controls,
    effects=effects,input_hashes={str(f.relative_to(E)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (path,oldpath)},
    holdout_used=False,final_goal_acceptance=False),indent=2))
order=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
lines=['# S4T 整链全开对照','','|配置|PASS现金|'+'|'.join(order)+'|均胜率|','|---|---:|'+'---:|'*9]
for label,s in p['summary'].items():
    lines.append('|'+label+f"|{s['pass']['mean_cash']:,.0f}|"+'|'.join(str(s[k]['wins']) for k in order)+f"|{st.fmean(s[k]['win_rate'] for k in order)*100:.3f}%|")
lines+=['',f'{len(rows)}场；{controls}旧对照逐场复现。每格50 seed×双座位100局；同批开发集，不是独立泛化证明。',
    '整套共同开启不等于已实现完美协作；原生时延探针有超过1秒的记录，不具备晋级/线上验收资格。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines),flush=True)
