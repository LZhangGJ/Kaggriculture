"""Offline witnesses selected by the batch diagnostic, not tuning cases."""
from pathlib import Path
import gzip,hashlib,json,sys,zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native
out=EXP/'receipts/s3l_admission_witnesses_v1';out.mkdir(exist_ok=False)
config=json.loads((EXP/'profiles/s3l/configs.json').read_text())['hire_base']
data=json.loads(zlib.decompress((EXP/'native/g001_frozen.json.zlib').read_bytes()));rival=native.G001(data)
records=[]
for seed,day in ((20261434,12),(20261429,17),(20261433,17)):
    env=native.Env(seed);ctl=native.Controller(config);st=native.G001State();trace=[]
    for step in range(719):
        before=env.observation(0);own=ctl.act(env,0);op=rival.act(env,1,st)
        if day*24<=step<day*24+6:trace.append(dict(step=step,observation=before,action=own,controller=ctl.debug()))
        env.step([own,op])
    changed=[];first=trace[0]['observation'];initial=[t for row in first['farms'][0]['tiles'] for t in row]
    compiled=next(r for r in trace if r['controller']['phase']==3);later=[t for row in compiled['observation']['farms'][0]['tiles'] for t in row]
    for pos,(old,new) in enumerate(zip(initial,later)):
        if old!=new:changed.append(dict(position=[pos%10,pos//10],before=old,at_compile=new))
    row=dict(seed=seed,day=day,compile_hour=compiled['step']%24,cash=compiled['observation']['farms'][0]['money'],
        seed_stock=compiled['observation']['private']['seeds'],changed_tiles=changed,
        market_actions=[dict(step=r['step'],orders=r['action']['market']) for r in trace],
        result=env.observation(0)['farms'][0]['money'])
    records.append(row)
    with gzip.open(out/f'{seed}_day{day}.json.gz','wt') as f:json.dump(dict(summary=row,trace=trace),f)
    print(json.dumps(row),flush=True)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_WITNESSES_NOT_GENERALIZATION_TEST',config=config,records=records,
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())),indent=2))
