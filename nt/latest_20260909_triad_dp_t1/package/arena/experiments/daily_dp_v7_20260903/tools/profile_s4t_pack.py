"""Function-level pack/reuse diagnostic at preselected real slow decisions."""
from pathlib import Path
import hashlib,json,sys,time,zlib
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E/'native/build'));sys.path.insert(0,str(E/'native/s4u_cost_probe_v1'))
import _dp7_native as native
import _dp7_packcost as probe
b=json.loads((E/'native/build/build_receipt.json').read_text())
pb=json.loads((E/'native/s4u_cost_probe_v1/acceptance.json').read_text())
assert pb['status']=='PASS_BUILD' and pb['base_build']['binary_sha256']==b['binary_sha256']
for rel,h in b['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
cfg=json.loads((E/'profiles/s4t/configs.json').read_text())['full_chain_autonomous']
g=native.G001(json.loads(zlib.decompress((E/'native/g003_frozen.json.zlib').read_bytes())))
out=E/'receipts/s4t_pack_cost_v1';out.mkdir(exist_ok=False);rows=[];games=[]
points={0,192,211,216,240,432}
for seed in (20262701,20262702):
    for seat in (0,1):
        env=native.Env(seed);c=native.Controller(cfg);state=native.G001State()
        while not env.done:
            step=env.step_count;measured=probe.inspect(c,env,seat) if step in points else None
            tic=time.perf_counter();own=c.act(env,seat);native_time=time.perf_counter()-tic
            if measured is not None:
                assert own==measured.pop('action'),('profiling changed action',seed,seat,step,own,measured)
                row=dict(seed=seed,seat=seat,step=step,native_seconds=native_time,**measured);rows.append(row)
                print(json.dumps(row),flush=True)
                (out/'progress.json').write_text(json.dumps(rows,indent=2))
            other=g.act(env,1-seat,state);env.step([own,other] if seat==0 else [other,own])
        assert env.step_count==719
        games.append(dict(seed=seed,seat=seat,money=[x['money'] for x in env.observation(seat)['farms']]))
ref=json.loads((E/'receipts/s4t_action_cost_trace_v1/acceptance.json').read_text())
for game in games:
    old=next(x for x in ref['games'] if x['label']=='full_chain_autonomous' and (x['seed'],x['seat'])==(game['seed'],game['seat']))
    assert game['money']==old['money']
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_READ_ONLY_PACK_PROFILE',build=b,probe_build=pb,
    points=sorted(points),games=games,rows=rows,action_checks=len(rows),unchanged_money_games=len(games),
    caveat='Instrumentation tracks exact ordered pack inputs, not approximate board similarity. Profile overhead is excluded from individual pack bodies but affects overall call timing. No cache or policy modification yet.'),indent=2))
