#!/usr/bin/env python3
"""Independent frozen-official market oracle, not full games or win-rate evidence."""
from pathlib import Path
import argparse, copy, hashlib, itertools, json, subprocess, sys, time
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
def main():
 p=argparse.ArgumentParser();p.add_argument('--referee',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert not a.out.exists();sys.path.insert(0,str(a.referee));from cpu_runtime import LocalGame,load_engine
 engine=load_engine();base=LocalGame(7,engine);cases=[]
 for item in range(1,8):
  name=ITEMS[item]
  floor=next((s for s in range(10000,20001) if engine.market_price(name,s,base.state[0].observation.market.get('params'))<=1),None)
  stocks=[floor-7,floor-2,floor-1,floor,floor+1] if floor else [9900,10000,15000,19998]
  for stock,own,rival,order,seat in itertools.product(stocks,[0,1,3,24,100],[0,1,2,7,100],range(3),range(2)):
   if stock+max(1,own)+rival>20000:continue
   cases.append((item,stock,own,rival,order,seat))
 inp=''.join('%d %d %d %d %d\n'%c[:5] for c in cases)
 started=time.perf_counter();proc=subprocess.run([str(a.probe)],input=inp,text=True,capture_output=True,check=True,timeout=60)
 lines=proc.stdout.splitlines();assert len(lines)==len(cases)
 rows=[]
 for case,line in zip(cases,lines):
  item,stock,own,rival,order,seat=case;other=1-seat;name=ITEMS[item]
  env=LocalGame(7,engine);env.state=copy.deepcopy(base.state)
  market=env.state[0].observation.market;market['inventory'][name]=stock
  for s in (0,1):
   env.state[s].observation.market=market;env.state[s].observation.farms=env.state[0].observation.farms
   env.state[s].observation.private['shed']={name:own if s==seat else rival}
   env.state[s].observation.farms[s]['money']=1000
  ownslot,otherslot=((1,0) if order==0 else (0,0) if order==1 else (0,1))
  env.state[seat].action={'market':[['PASS']]*ownslot+([['SELL',name,own]] if own else [])}
  env.state[other].action={'market':[['PASS']]*otherslot+([['SELL',name,rival]] if rival else [])}
  engine._process_market(env.state,env)
  actual=[env.state[0].observation.farms[seat]['money']-1000,env.state[0].observation.farms[other]['money']-1000,market['inventory'][name]]
  expected=[int(float(x)) for x in line.split()];assert actual==expected,(case,actual,expected)
  assert env.state[seat].observation.private['shed'][name]==0
  assert env.state[other].observation.private['shed'][name]==0
  rows.append({'case':list(case),'official':actual,'native':expected})
 result={'scope':'synthetic single-market transitions against the unmodified official engine; not matches','passed':True,'cases':len(rows),'both_seats':True,'queue_orderings':3,'seconds':time.perf_counter()-started,'official_sha256':hashlib.sha256((a.referee/'official/kaggriculture.py').read_bytes()).hexdigest(),'probe_sha256':hashlib.sha256(a.probe.read_bytes()).hexdigest(),'rows':rows}
 a.out.write_text(json.dumps(result,separators=(',',':')));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
