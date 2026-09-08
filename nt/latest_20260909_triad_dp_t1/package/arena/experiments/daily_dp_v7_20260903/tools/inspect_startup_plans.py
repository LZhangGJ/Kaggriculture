"""Read proposals before the first settlement. No future rollout or selection."""
from pathlib import Path
import argparse,collections,hashlib,json,sys
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native
p=argparse.ArgumentParser();p.add_argument('--configs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
cfg=json.loads(Path(a.configs).read_text());build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
rows=[]
for label,params in cfg.items():
    reference=None
    for seed in range(20262701,20262751):
        for seat in (0,1):
            env=native.Env(seed);ctl=native.Controller(params);action=ctl.act(env,seat);debug=ctl.debug()
            counts=collections.Counter(k for pos,k in debug['target'] if k>=0)
            record=dict(target=debug['target'],counts=dict(counts),first_action=action)
            if reference is None:reference=record
            else:assert record==reference,('initial public-state noninvariance',label,seed,seat)
    rows.append(dict(label=label,initial_states=100,**reference));print(json.dumps(dict(label=label,counts=reference['counts'])))
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='PASS_INITIAL_STATE_INVARIANCE_NOT_STRENGTH',
    build=build,config_sha256=hashlib.sha256(Path(a.configs).read_bytes()).hexdigest(),rows=rows,
    caveat='Targets are conditional proposals, not filled purchases or successfully planted assets. Each state has an identical observable opening.'),indent=2))
