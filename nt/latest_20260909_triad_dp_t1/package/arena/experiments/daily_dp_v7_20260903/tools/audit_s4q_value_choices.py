"""Replay live policy unchanged; inspect only public-information day forecasts."""
from pathlib import Path
import hashlib,json,statistics as st,sys,time,zlib
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4q_value_choices_v1';out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text())
probe=json.loads((E/'native/s4q_value_probe_v1/acceptance.json').read_text());assert probe['status']=='PASS_BUILD'
for b in (build,probe):
    for rel,h in b['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
    assert hashlib.sha256(Path(b['binary']).read_bytes()).hexdigest()==b['binary_sha256']
sys.path[:0]=[str(E/'native/build'),str(E/'native/s4q_value_probe_v1')]
import _dp7_native as native
import _dp7_valueprobe as diagnostic
panel=json.loads((E/'receipts/s4q_sixway_N50_v1/results.json').read_text())
label='all_intraday_insert_value';cfg=panel['configurations'][label]
expected={(r['opponent'],r['seed'],r['seat']):r for r in panel['rows'] if r['variant']==label}
constructors={'g001':(native.G001,native.G001State),'g003':(native.G001,native.G001State),'boatlee_v29':(native.BoatleeV29,native.BoatleeState),'kaito_v58':(native.KaitoV58,native.KaitoState),'lynn_v5':(native.LynnV5,native.LynnState)}
direct={'yhay81_six_day':native.Fieldbook,'yhay81_three_day':native.ThreeDay,'ecobot_v7':native.EcoBotV7}
rows=[];games=[];tic=time.perf_counter()
for opp in panel['identities']:
    if opp=='pass':continue
    agent=None
    if opp in constructors:
        entry=panel['identities'][opp];asset=E/entry['asset'];assert hashlib.sha256(asset.read_bytes()).hexdigest()==entry['asset_sha256']
        agent=constructors[opp][0](json.loads(zlib.decompress(asset.read_bytes())))
    for seed in range(20262701,20262705):
        for seat in (0,1):
            env=native.Env(seed);c=native.Controller(cfg);rival=constructors[opp][1]() if agent is not None else direct[opp]()
            pending=[];local=[]
            while not env.done:
                debug=c.debug();before=native.day_value_stats(c);obs=env.observation(seat)
                inspected=diagnostic.inspect(c,env,seat)
                assert c.debug()==debug and env.observation(seat)==obs and native.day_value_stats(c)==before
                own=c.act(env,seat);after=native.day_value_stats(c)
                if inspected['checked']:
                    assert after['checks']==before['checks']+1
                    assert after['rejected']==before['rejected']+int(inspected['rejected'])
                    r=dict(opponent=opp,seed=seed,seat=seat,**inspected);local.append(r);pending.append(r)
                elif after['checks']>before['checks']:
                    raise AssertionError(('diagnostic missed selector',opp,seed,seat,env.step_count))
                other=agent.act(env,1-seat,rival) if agent is not None else rival.act(env,1-seat)
                env.step([own,other] if seat==0 else [other,own])
                if env.step_count%24==0:
                    actual=diagnostic.assets(env,seat)
                    for r in pending:
                        predicted=r['keep_endpoint'] if r['rejected'] else r['proposal_endpoint']
                        assert predicted['step']==env.step_count
                        r['actual_endpoint']=actual
                        r['forecast_error']={k:actual[k]-predicted[k] if k=='cash' else [a-b for a,b in zip(actual[k],predicted[k])] for k in ('cash','scale','field','shed','seeds','market')}
                    pending=[]
            assert not pending
            ref=expected[opp,seed,seat];final=env.observation(seat)['farms']
            assert env.step_count==719 and final[seat]['money']==ref['cash'] and final[1-seat]['money']==ref['opponent_cash']
            games.append(dict(opponent=opp,seed=seed,seat=seat,cash=ref['cash'],opponent_cash=ref['opponent_cash'],win=ref['win'],stats=native.day_value_stats(c)))
            rows+=local
    print(json.dumps(dict(opponent=opp,games=len(games),checks=len(rows))),flush=True)
    (out/'progress.json').write_text(json.dumps(dict(games=len(games),checks=len(rows))))
groups=[]
for opp in panel['identities']:
    if opp=='pass':continue
    rr=[r for r in rows if r['opponent']==opp];known=[r for r in rr if not r['unknown']];rejected=[r for r in known if r['rejected']]
    groups.append(dict(opponent=opp,checks=len(rr),unknown=sum(r['unknown'] for r in rr),rejected=len(rejected),
      mean_rejected_proposal_deltas={k:st.fmean(r['proposal'][k]-r['keep'][k] for r in rejected) if rejected else 0 for k in ('score','trade','wages','seeds','calendars')},
      different_scale_forecasts=sum(r['keep_endpoint']['scale']!=r['proposal_endpoint']['scale'] for r in rr),
      scale_forecast_mismatch=sum(any(r['forecast_error']['scale']) for r in rr),
      mean_abs_endpoint_cash_error=st.fmean(abs(r['forecast_error']['cash']) for r in rr) if rr else 0,
      mean_endpoint_cash_error=st.fmean(r['forecast_error']['cash'] for r in rr) if rr else 0))
receipt=dict(status='PASS_READONLY_LIVE_SELECTOR_AUDIT_NOT_COUNTERFACTUAL_LABELS',build=build,probe=probe,label=label,games=games,rows=rows,groups=groups,seconds=time.perf_counter()-tic,caveat='Eight opponents, first four development seeds, both seats; outcomes exactly repeat the panel. Forecasts are same-day PASS-opponent scenarios with further reassignment disabled, not real futures. Endpoint errors include real opponent response, future replanning and unknown events; overlapping checks are not independent samples or additive gains. No alternative true terminal continuation or hindsight choice.')
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(groups,indent=2),flush=True)
