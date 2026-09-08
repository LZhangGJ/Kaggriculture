from pathlib import Path
import hashlib,json,statistics as st
from summarize_declared_value_trial import ci
E=Path(__file__).resolve().parents[1];path=E/'receipts/s4n1_two_way_O50_v1/results.json';p=json.loads(path.read_text())
assert p['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE';assert p['args']['seed']==20262801 and p['args']['count']==50
out=E/'receipts/s4n1_O50_confirmation_v1';out.mkdir(exist_ok=False)
rows={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in p['rows']};base='all_intraday_insert';after=base+'_workforce';effects=[]
for opp in p['identities']:
    delta={k:[] for k in ('win','cash','margin')};flips=dict(loss_to_win=0,win_to_loss=0)
    for seed in range(20262801,20262851):
        for k in delta:delta[k].append(st.fmean(float(rows[after,opp,seed,s][k])-float(rows[base,opp,seed,s][k]) for s in (0,1)))
        for s in (0,1):
            a,b=rows[after,opp,seed,s]['win'],rows[base,opp,seed,s]['win'];flips['loss_to_win']+=a and not b;flips['win_to_loss']+=b and not a
    effects.append(dict(opponent=opp,before=p['summary'][base][opp],after=p['summary'][after][opp],estimates={k:ci(v) for k,v in delta.items()},flips=flips))
doc=E/'reports/S4N1_O_CONFIRMATION_PRE_REGISTER_ZH.md'
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_UNSEEN_SEED_FIXED_POLICY_COMPARISON',build=p['build']['binary_sha256'],games=len(rows),effects=effects,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pre_register_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),N_pooled=False,P_used=False,final_goal_complete=False),indent=2))
for e in effects:print(json.dumps(dict(opponent=e['opponent'],before=e['before']['wins'],after=e['after']['wins'],win_delta=e['estimates']['win'],cash_delta=e['estimates']['cash'])))
