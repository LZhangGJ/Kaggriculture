"""Recalculate realized cash from a stored action pair using frozen official rules.
This is a deterministic replay audit, NOT new realtime-opponent strength data.
"""
import sys,json,gzip,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'referee'))
from cpu_runtime import LocalGame,load_engine

def audit(path):
 data=json.loads(gzip.decompress(Path(path).read_bytes()));seed=data['result']['seed'];seat=data['result']['opponent_seat'];own=1-seat
 env=LocalGame(seed,load_engine());transactions=[]
 commit=env.engine._commit_unit
 def hooked(op,item,price,farm,private,market,shed_capacity=100):
  before=float(farm['money']);side=next(i for i in (0,1)if farm is env.state[0].observation.farms[i])
  result=commit(op,item,price,farm,private,market,shed_capacity);delta=float(farm['money'])-before
  if delta:transactions.append([env.t,side,op,item,delta])
  return result
 env.engine._commit_unit=hooked
 def hook_atomic(fn,op):
  def wrapped(farm,*a,**k):
   before=float(farm['money']);side=next(i for i in(0,1)if farm is env.state[0].observation.farms[i]);result=fn(farm,*a,**k);delta=float(farm['money'])-before
   if delta:transactions.append([env.t,side,op,op,delta])
   return result
  return wrapped
 env.engine._do_hire=hook_atomic(env.engine._do_hire,'HIRE');env.engine._do_buy_land=hook_atomic(env.engine._do_buy_land,'BUY_LAND')
 days=[]
 for actions in data['actions']:
  env.advance(actions)
  if env.t%24==0 or env.done:
   obs=env.observation(own);money=[f['money']for f in obs['farms']]
   days.append({'step':env.t,'own':money[own],'rival':money[seat],'margin':money[own]-money[seat],'shops':obs['town']['unlocked_shops']})
 cash=[f['money']for f in env.observation(own)['farms']]
 for s in (0,1):
  if abs(3000+sum(t[4]for t in transactions if t[1]==s)-cash[s])>1e-6:raise RuntimeError('Cash identity failed')
 if cash[own]!=data['result']['own_cash']or cash[seat]!=data['result']['opponent_cash']:raise RuntimeError('Archived final cash mismatch')
 expenses={};revenues={}
 for step,s,op,item,delta in transactions:
  if s!=own:continue
  target=expenses if delta<0 else revenues;key=op+':'+str(item);target[key]=target.get(key,0)+abs(delta)
 return{'source':str(path),'status':'PASS_CASH_IDENTITIES_AND_ARCHIVE','result':data['result'],'expenses':expenses,'revenues':revenues,'days':days,'transactions':transactions}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trace',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.exists():raise SystemExit('Refusing overwrite')
 result=audit(a.trace);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['days','transactions']}))
