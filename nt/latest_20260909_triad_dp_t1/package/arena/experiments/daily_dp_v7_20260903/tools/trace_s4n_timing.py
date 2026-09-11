"""Paired complete live trajectories of frozen policies. No suffix search."""
from pathlib import Path
import gzip,hashlib,json,sys,time,zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
panel=json.loads((EXP/'receipts/s4m1_eightway_N50_v1/results.json').read_text());assert panel['build']['binary_sha256']==build['binary_sha256']
configs=panel['configurations'];refs={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
out=EXP/'receipts/s4n_timing_trace_v1';out.mkdir(exist_ok=False);results=[];tic=time.perf_counter()
for opponent in ('g001','g003','boatlee_v29','yhay81_six_day'):
    entry=panel['identities'][opponent]
    if opponent=='yhay81_six_day':agent=None
    else:
        asset=EXP/entry['asset'];assert hashlib.sha256(asset.read_bytes()).hexdigest()==entry['asset_sha256']
        payload=json.loads(zlib.decompress(asset.read_bytes()));agent=native.BoatleeV29(payload) if opponent=='boatlee_v29' else native.G001(payload)
    for seed in range(20262701,20262709):
        for seat in (0,1):
            for label in ('all_intraday','all_intraday_insert'):
                env=native.Env(seed);ctl=native.Controller(configs[label]);trace=[];overflow=[0,0]
                state=(native.Fieldbook() if opponent=='yhay81_six_day' else native.BoatleeState() if opponent=='boatlee_v29' else native.G001State())
                while not env.done:
                    before=env.observation(seat);before_stats=native.shared_insertion_stats(ctl)
                    own=ctl.act(env,seat)
                    other=state.act(env,1-seat) if opponent=='yhay81_six_day' else agent.act(env,1-seat,state)
                    actions=[own,other] if seat==0 else [other,own]
                    debug=ctl.debug();after_stats=native.shared_insertion_stats(ctl)
                    env.step(actions)
                    trace.append(dict(before=before,actions=actions,debug=debug,insertion_before=before_stats,insertion_after=after_stats))
                final=env.observation(seat);ref=refs[label,opponent,seed,seat]
                assert env.step_count==719 and final['farms'][seat]['money']==ref['cash'] and final['farms'][1-seat]['money']==ref['opponent_cash']
                path=out/f'{label}_{opponent}_{seed}_seat{seat}.json.gz'
                with gzip.open(path,'wt',encoding='utf8',compresslevel=1) as f:json.dump(dict(label=label,opponent=opponent,seed=seed,seat=seat,trace=trace,final=final),f,separators=(',',':'))
                results.append(dict(label=label,opponent=opponent,seed=seed,seat=seat,cash=ref['cash'],opponent_cash=ref['opponent_cash'],margin=ref['margin'],win=ref['win'],path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(json.dumps(dict(opponent=opponent,seed=seed,completed_games=len(results))),flush=True)
        (out/'progress.json').write_text(json.dumps(results,indent=2))
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_FROZEN_LIVE_TRACE_NOT_STRENGTH',build=build,matched_games=len(results),rows=results,
    seconds=time.perf_counter()-tic,pre_register_sha256=hashlib.sha256((EXP/'reports/S4N_TIMING_CAUSAL_AUDIT_PRE_REGISTER_ZH.md').read_bytes()).hexdigest(),
    caveat='128 repeated development games, all full original live policies. No unseen validation, no extra independent sample, no suffix simulation.'),indent=2))
print(json.dumps(dict(status='PASS',matched_games=len(results),seconds=time.perf_counter()-tic)))
