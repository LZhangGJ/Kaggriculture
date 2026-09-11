"""Read-only post-holdout diagnosis. Never used for parameter selection."""
from experiment import *
import gzip,statistics
freeze=json.loads((R/'RELEASE_FREEZE.json').read_text());assert digest(R/'policy/agent.so')==freeze['binary_hash']
rs=json.loads((R/'runs/release_holdout100/rows.json').read_text())
losses=[r for r in rs if not r['win'] and not r['error']]
if not losses:raise SystemExit('No loss to diagnose')
# Representative largest deficit within the weakest opponent, not new training.
by={i:[r for r in rs if r['opponent']==i] for i in range(7)}
weak=min(by,key=lambda i:sum(r['win'] for r in by[i])/len(by[i]));bad=min(by[weak],key=lambda r:r['margin'])
tag='postholdout_loss_audit';run(pool(),tag,freeze['config'],count=1,start=bad['seed'],opponents=[weak],trace=True)
trace=json.loads(gzip.decompress((R/'runs'/tag/f'trace_{bad["seed"]}_{weak}_{bad["seat"]}.json.gz').read_bytes()))
prices=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER'];days=[]
def assets(farm):
 count={}
 for row in farm['tiles']:
  for t in row:
   if isinstance(t,dict):
    k=t.get('animal') or t.get('crop')
    if k:count[k]=count.get(k,0)+1
 return count
for o in trace['days']:
 p=bad['seat'];day=o['day']
 if day%3 and o['step']!=719:continue
 days.append(dict(day=day,step=o['step'],cash=o['farms'][p]['money'],opponent_cash=o['farms'][1-p]['money'],own_assets=assets(o['farms'][p]),opponent_assets=assets(o['farms'][1-p]),shops=o['town']['unlocked_shops'],prices={k:o['market']['prices'][k] for k in prices},own_shed=o['private']['shed']))
write(R/'POSTHOLDOUT_LOSS_DIAGNOSIS.json',dict(scope='read-only, after frozen evaluation, not used to tune or choose version',seed=bad['seed'],opponent=NAMES[weak],seat=bad['seat'],cash=bad['cash'],opponent_cash=bad['opponent_cash'],margin=bad['margin'],snapshots=days))
print(json.dumps(dict(seed=bad['seed'],opponent=NAMES[weak],margin=bad['margin']),indent=2))
