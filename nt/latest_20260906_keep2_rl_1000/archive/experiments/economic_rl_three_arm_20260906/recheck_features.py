"""New candidate estimates must not alter baseline/random semantics."""
from runtime import *
import gzip
def main():
 pool=make_pool();rows=[]
 for arm in ('c3auto','f3','c3j7'):
  for mode in (0,3):
   result=rollout(pool,arm,mode=mode,start=64000000,count=1,sample=17,trace=True)
   with gzip.open(P/f'checks_v3/{arm}_{mode}/traces.json.gz','rt')as f:assert json.load(f)==result['traces']
   assert summary(result)['status']=='PASS'
   assert np.isfinite(result['candidates']).all()
   effects=dict(changed=0,investment=0,realized_same_day=0,defer=0,defer_no_investment_same_day=0)
   if mode==3:
    for game,row in enumerate(result['rows']):
     env=native.Env(row['seed']);events={}
     for j in np.where((result['game']==game)&(result['changed']>0))[0]:events[int(result['day'][j])]=dict(pos=int(result['pos'][j]),kind=int(result['kind'][j]),realized=False,any_investment=False)
     for step,pair in enumerate(result['traces'][game]):
      env.step(pair);event=events.get(step//24)
      if event:
       p=event['pos'];tile=env.observation(row['seat'])['farms'][row['seat']]['tiles'][p//10][p%10]
       if isinstance(tile,dict):
        name=tile.get('animal')or tile.get('crop');items=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP']
        kind=items.index(name)if name in items else -1
        event['any_investment']|=kind>=0;event['realized']|=kind==event['kind']and kind>=0
     for event in events.values():
      effects['changed']+=1
      if event['kind']>=0:effects['investment']+=1;effects['realized_same_day']+=event['realized']
      else:effects['defer']+=1;effects['defer_no_investment_same_day']+=not event['any_investment']
   rows.append(dict(arm=arm,mode=mode,steps=14*719,summary=summary(result),effects=effects))
 receipt=read(P/'G1_RECEIPT.json');save(P/'G1_PRE_EFFECT_RECEIPT.json',receipt)
 assert all(r['effects']['defer']==r['effects']['defer_no_investment_same_day'] for r in rows)
 receipt.update(feature_extension_parity=rows,hashes={str(f):digest(f)for f in (P/'build').glob('*.so')});save(P/'G1_RECEIPT.json',receipt)
 print('Feature-only 60396 action parity PASS',flush=True)
if __name__=='__main__':main()
