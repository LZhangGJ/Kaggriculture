"""Unmodified live opponents and own policy; inspect current unit phase only."""
from pathlib import Path
import hashlib,json,sys,time,zlib,collections
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4s_cooperation_live_v1';out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text());probe=json.loads((E/'native/s4s_cooperation_probe_v1/acceptance.json').read_text())
assert probe['status']=='PASS_BUILD'
for b in (build,probe):
    for rel,h in b['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
    assert hashlib.sha256(Path(b['binary']).read_bytes()).hexdigest()==b['binary_sha256']
sys.path[:0]=[str(E/'native/build'),str(E/'native/s4s_cooperation_probe_v1')]
import _dp7_native as native
import _dp7_coopprobe as diagnostic
panel=json.loads((E/'receipts/s4r_eightway_N50_v1/results.json').read_text())
constructors={'g001':(native.G001,native.G001State),'g003':(native.G001,native.G001State),'boatlee_v29':(native.BoatleeV29,native.BoatleeState),'kaito_v58':(native.KaitoV58,native.KaitoState),'lynn_v5':(native.LynnV5,native.LynnState)}
direct={'yhay81_six_day':native.Fieldbook,'yhay81_three_day':native.ThreeDay,'ecobot_v7':native.EcoBotV7}
rows=[];games=[];tic=time.perf_counter()
for label in ('all_intraday','all_intraday_insert'):
    cfg=panel['configurations'][label];expected={(r['opponent'],r['seed'],r['seat']):r for r in panel['rows'] if r['variant']==label}
    for opp,entry in panel['identities'].items():
        if opp=='pass':continue
        agent=None
        if opp in constructors:
            asset=E/entry['asset'];assert hashlib.sha256(asset.read_bytes()).hexdigest()==entry['asset_sha256']
            agent=constructors[opp][0](json.loads(zlib.decompress(asset.read_bytes())))
        for seed in range(20262701,20262705):
            for seat in (0,1):
                env=native.Env(seed);c=native.Controller(cfg);rival=constructors[opp][1]() if agent is not None else direct[opp]()
                local=[];locked=0
                while not env.done:
                    own=c.act(env,seat);before=c.debug();step=env.step_count
                    inspected=diagnostic.inspect(c,env,seat,own);assert c.debug()==before and env.step_count==step
                    locked+=inspected.get('locked_crop_jobs',0)
                    for r in inspected['opportunities']:local.append(dict(label=label,opponent=opp,seed=seed,seat=seat,step=step,day=step//24,hour=step%24,**r))
                    other=agent.act(env,1-seat,rival) if agent is not None else rival.act(env,1-seat)
                    env.step([own,other] if seat==0 else [other,own])
                ref=expected[opp,seed,seat];farm=env.observation(seat)['farms'];assert env.step_count==719 and farm[seat]['money']==ref['cash'] and farm[1-seat]['money']==ref['opponent_cash']
                games.append(dict(label=label,opponent=opp,seed=seed,seat=seat,cash=ref['cash'],opponent_cash=ref['opponent_cash'],win=ref['win'],opportunities=len(local),prefix_enabled=sum(r['prefix_enabled'] for r in local),locked_job_observations=locked));rows+=local
        print(json.dumps(dict(label=label,opponent=opp,games=len(games),opportunities=len(rows))),flush=True)
groups=[]
for label in ('all_intraday','all_intraday_insert'):
    for opp in panel['identities']:
        if opp=='pass':continue
        gs=[g for g in games if g['label']==label and g['opponent']==opp];rr=[r for r in rows if r['label']==label and r['opponent']==opp]
        groups.append(dict(label=label,opponent=opp,games=len(gs),games_with_opportunities=sum(g['opportunities']>0 for g in gs),games_with_prefix=sum(g['prefix_enabled']>0 for g in gs),opportunities=len(rr),prefix_enabled=sum(r['prefix_enabled'] for r in rr),by_operation=dict(collections.Counter(r['operation'] for r in rr)),planned=sum(r['owner']>=0 for r in rr),precedence_risk=sum(r['earlier_planned_same_plot'] for r in rr)))
d=dict(status='PASS_LIVE_CAPABILITY_CENSUS_NOT_PROFIT_CLAIM',build=build,probe=probe,games=games,rows=rows,groups=groups,seconds=time.perf_counter()-tic,caveat='128 unchanged complete live games; same-day unit prefixes only, no future continuations. Opportunities overlap and are not cumulative savings or profit. Passing effect check alone does not preserve fertilizer ordering, deadlines, original output delivery or economic preference. No policy modification.')
(out/'acceptance.json').write_text(json.dumps(d,indent=2));print(json.dumps(groups,indent=2),flush=True)
