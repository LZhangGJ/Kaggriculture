"""Replay current live policies diagnostically, not opponent tapes or suffixes.

First two preregistered development seeds, both seats. Keep real observations
and actions all season so worker deployment can be followed into actual money.
Full-game final cash must match the frozen panel.
"""
from pathlib import Path
import gzip,hashlib,json,sys,zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
panel=json.loads((EXP/'receipts/s4l_eightway_N50_v1/results.json').read_text())
assert panel['build']['binary_sha256']==build['binary_sha256']
cfg=panel['configurations'];refs={(r['variant'],r['seed'],r['seat']):r for r in panel['rows'] if r['opponent']=='g003'}
asset=EXP/'native/g003_frozen.json.zlib';rival=native.G001(json.loads(zlib.decompress(asset.read_bytes())))
out=EXP/'receipts/s4l_service_trace_v1';out.mkdir(exist_ok=False);results=[]
base='all_intraday_auto_portfolio_calendar'
for label in (base,base+'_causal_pickup',base+'_procure',base+'_procure_causal_pickup'):
    for seed in (20262701,20262702):
        for seat in (0,1):
            env=native.Env(seed);ctl=native.Controller(cfg[label]);rs=native.G001State();trace=[]
            while not env.done:
                record=True
                if record:before=env.observation(seat);stats_before=native.service_stats(ctl)
                own=ctl.act(env,seat);other=rival.act(env,1-seat,rs)
                actions=[own,other] if seat==0 else [other,own]
                if record:debug=ctl.debug();stats_after=native.service_stats(ctl)
                env.step(actions)
                if record:trace.append(dict(before=before,actions=actions,after=env.observation(seat),debug=debug,
                    service_before=stats_before,service_after=stats_after))
            final=env.observation(seat);cash=final['farms'][seat]['money'];opponent_cash=final['farms'][1-seat]['money']
            ref=refs[label,seed,seat]
            assert cash==ref['cash'] and opponent_cash==ref['opponent_cash'] and env.step_count==719
            path=out/f'{label}_{seed}_seat{seat}.json.gz'
            with gzip.open(path,'wt',encoding='utf8') as f:json.dump(dict(label=label,seed=seed,seat=seat,trace=trace,final=final),f)
            results.append(dict(label=label,seed=seed,seat=seat,cash=cash,opponent_cash=opponent_cash,
                trace=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_MATCHED_LIVE_TRACE_NOT_STRENGTH',build=build,
    games=len(results),rows=results,policy_source_unchanged=True,
    caveat='All 719 real steps saved to connect labour, production and finance. Diagnostic repeats, not extra independent evaluation games.'),indent=2))
print(json.dumps(dict(status='PASS',matched_live_games=len(results))))
