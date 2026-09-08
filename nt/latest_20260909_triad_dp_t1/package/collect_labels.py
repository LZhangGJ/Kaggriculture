"""Offline counterfactual data. Hidden suffixes supply labels, NEVER features."""
from experiment import *
import numpy as np
import argparse
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--start',type=int,default=260908132);p.add_argument('--count',type=int,default=16);p.add_argument('--opponents',default='0,1,2,3,4,5,6');p.add_argument('--days',default='0,1,3,6,9,12,15,18,21,24,27');p.add_argument('--reverse',action='store_true');p.add_argument('--threads',type=int,default=4);x=p.parse_args()
cfg=DEFAULTS.copy();cfg.update(repeat=0,max_land=3,feed_cover=1,competition=2,labor_hours=10,work_price=4,rotation=0,portfolio_passes=9,scenario=0)
out=R/'labels'/x.tag;out.mkdir(parents=True,exist_ok=False)
lib=R/'policy/agent.so';h=digest(lib);frozen=R/'snapshots'/(h+'.so');shutil.copy2(lib,frozen) if not frozen.exists() else None
protocol=dict(config=cfg,source_build=json.loads(lib.with_suffix('.build.json').read_text()),seeds=list(range(x.start,x.start+x.count)),opponents=[int(o) for o in x.opponents.split(',')],days=[int(d) for d in x.days.split(',')],feature_boundary='dp7::View only; frozen before label rollout',intervention='candidate plan today only; common native continuation thereafter',reverse=x.reverse)
write(out/'protocol.json',protocol)
tic=time.perf_counter();rows=pool().collect(str(frozen),list(cfg.values()),protocol['seeds'],protocol['opponents'],protocol['days'],x.threads,x.reverse)
features=np.array([r.pop('features') for r in rows],dtype=np.float64)
np.savez_compressed(out/'features.npz',X=features)
write(out/'labels.json',rows)
groups={}
for r in rows:groups.setdefault((r['seed'],r['opponent'],r['seat'],r['day']),[]).append(r)
changed=[g for g in groups.values() if len(g)>1]
summary=dict(seconds=time.perf_counter()-tic,rows=len(rows),features=features.shape[1],groups=len(groups),nontrivial_groups=len(changed),keep_checks='all exact',finite=bool(np.isfinite(features).all()),oracle_better_groups=sum(max(r['margin'] for r in g)>g[0]['margin'] for g in changed),oracle_win_groups=sum(max(r['margin'] for r in g)>0 for g in changed),keep_win_groups=sum(g[0]['margin']>0 for g in changed))
write(out/'summary.json',summary);print(json.dumps(summary),flush=True)
