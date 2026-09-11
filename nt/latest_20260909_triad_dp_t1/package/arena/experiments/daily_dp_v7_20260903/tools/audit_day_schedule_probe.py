"""Read-only policy scenarios embedded in unchanged responsive live matches."""
from pathlib import Path
import argparse,hashlib,json,statistics as st,sys,time,zlib
p=argparse.ArgumentParser();p.add_argument('--probe',required=True);p.add_argument('--out',required=True);a=p.parse_args()
E=Path(__file__).resolve().parents[1];out=E/a.out;out.mkdir(exist_ok=False)
primary=json.loads((E/'native/build/build_receipt.json').read_text());probe=json.loads((E/a.probe/'acceptance.json').read_text())
for b in (primary,probe):
    for rel,h in b['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
    assert hashlib.sha256(Path(b['binary']).read_bytes()).hexdigest()==b['binary_sha256']
sys.path[:0]=[str(E/'native/build'),str(E/a.probe)]
import _dp7_native as native
import _dp7_dayprobe as diagnostic
panel=json.loads((E/'receipts/s4n1_eightway_N50_v1/results.json').read_text());cfg=panel['configurations']['all_intraday']
expected={(r['opponent'],r['seed'],r['seat']):r for r in panel['rows'] if r['variant']=='all_intraday'}
rows=[];games=[];start=time.perf_counter()
for opponent in ('g001','g003','boatlee_v29','yhay81_six_day'):
    identity=panel['identities'][opponent];agent=None
    if opponent!='yhay81_six_day':
        asset=E/identity['asset'];assert hashlib.sha256(asset.read_bytes()).hexdigest()==identity['asset_sha256']
        data=json.loads(zlib.decompress(asset.read_bytes()));agent=native.BoatleeV29(data) if opponent=='boatlee_v29' else native.G001(data)
    for seed in range(20262701,20262709):
        for seat in (0,1):
            env=native.Env(seed);c=native.Controller(cfg)
            state=native.Fieldbook() if opponent=='yhay81_six_day' else native.BoatleeState() if opponent=='boatlee_v29' else native.G001State()
            # Three distinct observation times/day, with overlapping forecast
            # windows. Never add their hypothetical bank deltas.
            while not env.done:
                if env.step_count%24 in (6,12,18):
                    before=env.observation(seat);debug=c.debug();r=diagnostic.inspect(c,env,seat)
                    assert env.observation(seat)==before and c.debug()==debug
                    if r['eligible']:rows.append(dict(opponent=opponent,seed=seed,seat=seat,**r))
                own=c.act(env,seat);other=state.act(env,1-seat) if opponent=='yhay81_six_day' else agent.act(env,1-seat,state)
                env.step([own,other] if seat==0 else [other,own])
            final=env.observation(seat);ref=expected[opponent,seed,seat]
            assert env.step_count==719 and final['farms'][seat]['money']==ref['cash'] and final['farms'][1-seat]['money']==ref['opponent_cash']
            games.append(dict(opponent=opponent,seed=seed,seat=seat,cash=ref['cash'],opponent_cash=ref['opponent_cash'],win=ref['win']))
        print(json.dumps(dict(games=len(games),probes=len(rows),changed=sum(r['changed'] for r in rows))),flush=True)
        (out/'progress.json').write_text(json.dumps(dict(games=len(games),probes=len(rows))))
groups=[]
for opponent in ('g001','g003','boatlee_v29','yhay81_six_day'):
    allrows=[r for r in rows if r['opponent']==opponent];changed=[r for r in allrows if r['changed']]
    times=[r[k]['milliseconds'] for r in changed for k in ('base','alternate')]
    groups.append(dict(opponent=opponent,observations=len(allrows),changed=len(changed),
        bank_higher=sum(r['bank_delta']>0 for r in changed),bank_lower=sum(r['bank_delta']<0 for r in changed),
        quote_higher=sum(r['cash_quote_delta_not_profit']>0 for r in changed),quote_lower=sum(r['cash_quote_delta_not_profit']<0 for r in changed),
        strict_cash_improvement=sum(r['strict_cash_improvement_in_PASS_scenario'] for r in changed),strict_cash_loss=sum(r['strict_cash_loss_in_PASS_scenario'] for r in changed),
        physical_endpoint_changed=sum(not r['same_endpoint_except_cash'] for r in changed),
        mean_preview_ms=st.fmean(times) if times else 0,max_preview_ms=max(times,default=0)))
receipt=dict(status='PASS_READONLY_DAY_SCENARIOS_NOT_STRENGTH',build=primary,probe=probe,games=games,observations=rows,groups=groups,seconds=time.perf_counter()-start,
    caveat='Current-public-state PASS-opponent scenarios, not true rival future, not Oracle, not final profit labels. Repeated observation differences cannot be added. Existing joint reassignment disabled within both previews to preserve the compared intervention; all other live production/market/recovery policies retained.')
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(groups,indent=2),flush=True)
