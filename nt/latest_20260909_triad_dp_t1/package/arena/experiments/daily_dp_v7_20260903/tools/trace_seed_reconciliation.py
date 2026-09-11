"""Compare pre-registered diagnostic cases; full-game evidence, no policy inputs."""
from pathlib import Path
import gzip,hashlib,json,sys,zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native
out=EXP/'receipts/s3m_seed_witnesses_v1';out.mkdir(exist_ok=False)
configs=json.loads((EXP/'profiles/s3m/configs.json').read_text())
data=json.loads(zlib.decompress((EXP/'native/g001_frozen.json.zlib').read_bytes()));rival=native.G001(data)
records=[]
for seed,day in ((20261434,12),(20261429,17),(20261433,17)):
    for label,config in configs.items():
        env=native.Env(seed);ctl=native.Controller(config);st=native.G001State();trace=[];repair_events=[];end_snapshot=None
        for step in range(719):
            before=env.observation(0);debug0=ctl.debug();own=ctl.act(env,0);debug=ctl.debug();op=rival.act(env,1,st)
            if debug['seed_reconciliations']>debug0['seed_reconciliations']:
                repair_events.append(dict(step=step,orders=own['market'],cash=before['farms'][0]['money']))
            if day*24<=step<(day+1)*24:trace.append(dict(step=step,observation=before,action=own,controller=debug))
            env.step([own,op])
            if step==(day+1)*24-1:end_snapshot=env.observation(0)
        compiled=next(r for r in trace if r['controller']['phase']==3)
        row=dict(seed=seed,day=day,label=label,compile_hour=compiled['step']%24,
            cash_at_compile=compiled['observation']['farms'][0]['money'],
            seed_stock=compiled['observation']['private']['seeds'],repair_events=repair_events,
            final_cash=env.observation(0)['farms'][0]['money'],opponent_cash=env.observation(0)['farms'][1]['money'],
            day_end_farm=end_snapshot['farms'][0],controller=ctl.debug())
        records.append(row)
        with gzip.open(out/f'{seed}_day{day}_{label}.json.gz','wt') as f:json.dump(dict(summary=row,trace=trace),f)
        print(json.dumps({k:row[k] for k in ('seed','day','label','compile_hour','cash_at_compile','seed_stock','repair_events','final_cash','opponent_cash')}),flush=True)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_DIAGNOSTIC_CASES_NOT_GENERALIZATION',configs=configs,records=records,
    build=json.loads((EXP/'native/build/build_receipt.json').read_text()),
    tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
