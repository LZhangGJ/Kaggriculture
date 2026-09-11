from pathlib import Path
import gzip,hashlib,json,statistics as st
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4r_next_day_audit_v1';out.mkdir(exist_ok=False)
path=E/'receipts/s4q_value_choices_v1/acceptance.json';source=json.loads(path.read_text())
assert source['status']=='PASS_READONLY_LIVE_SELECTOR_AUDIT_NOT_COUNTERFACTUAL_LABELS'
hashes={str(path.relative_to(E)):hashlib.sha256(path.read_bytes()).hexdigest()};games={};first={};rows=[]
for r in source['rows']:
    k=(r['opponent'],r['seed'],r['seat'],r['day'])
    if k not in first or r['step']<first[k]['step']:first[k]=r
for opp in sorted({r['opponent'] for r in source['games']}):
    p=E/f"receipts/s4q_pool_audit_N50_v1/{source['label']}_{opp}.json.gz"
    hashes[str(p.relative_to(E))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with gzip.open(p,'rt',encoding='utf8') as f:raw=json.load(f)
    for g in raw['rows']:games[opp,g['seed'],g['seat']]=g
for ref in source['games']:
    g=games[ref['opponent'],ref['seed'],ref['seat']];s=ref['seat']
    assert g['money'][s]==ref['cash'] and g['money'][1-s]==ref['opponent_cash']
for key,r in sorted(first.items()):
    opp,seed,seat,day=key;g=games[opp,seed,seat];now=g['planning'][day];nxt=g['planning'][day+1]
    scale=[r['actual_endpoint']['scale'][i] for i in range(8)];ids=[0,1,2,3,4,9,10,11]
    assert scale==[now['live'][i] for i in ids]
    desired=[nxt['target'][i] for i in ids];actual=[nxt['live'][i] for i in ids]
    rows.append(dict(opponent=opp,seed=seed,seat=seat,day=day,first_step=r['step'],rejected=r['rejected'],end_scale=scale,next_targets=desired,next_end_scale=actual,target_delta=[b-a for a,b in zip(scale,desired)],next_cash=g['cash_ledger'][day+1][seat]['end']))
groups=[]
for opp in sorted({r['opponent'] for r in rows}):
    rr=[r for r in rows if r['opponent']==opp]
    groups.append(dict(opponent=opp,game_days=len(rr),target_changed_days=sum(any(r['target_delta']) for r in rr),target_expand_days=sum(any(x>0 for x in r['target_delta']) for r in rr),target_reduce_days=sum(any(x<0 for x in r['target_delta']) for r in rr),mean_l1_target_change=st.fmean(sum(abs(x) for x in r['target_delta']) for r in rr)))
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_ACTUAL_NEXT_DAY_AUDIT_NOT_CAUSAL_ABLATION',groups=groups,rows=rows,source_games=64,input_hashes=hashes,caveat='First observed selector check per game-day. Next-day proposed and realized scale are different fields. Changes alone do not prove the old decision was wrong, and no alternate true future was simulated.'),indent=2))
print(json.dumps(groups,indent=2),flush=True)
