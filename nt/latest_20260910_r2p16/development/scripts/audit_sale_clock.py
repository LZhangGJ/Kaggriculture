"""Probe-only official replay conformance, own/public trades and causal clocks."""
from pathlib import Path
from collections import defaultdict
import argparse,ctypes,gzip,json,concurrent.futures as futures,multiprocessing,time
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'sale_clock_audit'
PROBE=HERE/'candidate_r2p8/policy/clock_probe_market0.so'
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
def load(p):return json.loads(gzip.decompress(p.read_bytes()))
def expected(hist):
 p=[0.]*24;p[1]=p[17]=.5
 rows=list(hist.values())[-3:]
 if len(rows)<2:return p
 for r in rows:
  n=sum(r)
  for h in range(24):p[h]+=r[h]/n
 return [x/(1+len(rows)) for x in p]
def one(row):
 key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}";path=OUT/(key+'.json.gz')
 if path.exists():
  old=load(path);assert old['status']=='PASS';return old['summary']
 trace=load(ROOT/row['trace']);audit=load(HERE/'baseline_audit'/key/'AUDIT.json.gz');own=1-row['opponent_seat']
 truth=defaultdict(int)
 for tx in audit['transactions']:
  if tx['item'] in ITEMS and tx['op'] in ('SELL','BUY_PRODUCT'):
   truth[(tx['step'],tx['seat'],tx['item'])]+=(1 if tx['op']=='SELL' else -1)*tx['quantity']
 agent=panel.R2_MODULE.Agent(binary_path=str(PROBE));agent.lib.td_clock_json.argtypes=[ctypes.c_void_p];agent.lib.td_clock_json.restype=ctypes.c_char_p
 env=panel.old.LocalGame(row['seed'],panel.ENGINE);hist=defaultdict(lambda:defaultdict(lambda:[0]*24));sale_checks=profile_checks=0;daily=[]
 saved={d['step']:d['observations'] for d in trace['days']}
 try:
  for step,joint in enumerate(trace['actions']):
   obs=env.observation(own)
   if step in saved:assert obs==saved[step][own]
   action=agent(obs);assert action==joint[own],dict(key=key,step=step,original=joint[own],probe=action)
   f=json.loads(agent.lib.td_clock_json(agent.handle));assert f['source_step']==step-1
   for side,seat in [('own',own),('rival',1-own)]:
    for i in range(1,8):
     if f[side+'_valid'][i]:
      q=truth[(step-1,seat,ITEMS[i])];sale_checks+=1
      assert f[side+'_sales'][i]==q,(key,step,side,ITEMS[i],f[side+'_sales'][i],q)
      if q>0:hist[(side,i)][(step-1)//24][(step-1)%24]+=q
     if step%24==0:
      target=expected(hist[(side,i)]);got=f[side][i]['density'];profile_checks+=24
      assert max(abs(a-b) for a,b in zip(got,target))<1e-6,(key,step,side,i,got,target)
      assert f[side][i]['days']==min(3,len(hist[(side,i)]))
   if step%24==0:daily.append(f)
   assert not f['invalid_market']
   env.advance(joint)
  assert env.done and env.observation(own)==saved[719][own]
  summary=dict(key=key,status='PASS',actions=719,sale_checks=sale_checks,profile_checks=profile_checks)
  path.parent.mkdir(exist_ok=True);path.write_bytes(gzip.compress(json.dumps(dict(status='PASS',summary=summary,daily=daily),separators=(',',':')).encode(),compresslevel=1))
  return summary
 finally:agent.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=22);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
 allrows=json.loads((HERE/'baseline_audit/SELECTION.json').read_text())['rows']
 rows=[allrows[round(i*(len(allrows)-1)/max(1,args.limit-1))] for i in range(args.limit)] if args.limit else allrows
 OUT.mkdir(exist_ok=True);protocol=dict(binary_sha256=panel.sha(PROBE),engine_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),boundary='Native observes normal seat data only. Past true trades are Python audit labels after action; profiles must match causal history.')
 if (OUT/'PROTOCOL.json').exists():assert json.loads((OUT/'PROTOCOL.json').read_text())==protocol
 else:panel.save(OUT/'PROTOCOL.json',protocol)
 start=time.perf_counter()
 with futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=panel.init_worker) as pool:
  results=list(pool.map(one,rows))
 receipt=dict(status='PASS',cases=len(results),actions=sum(r['actions'] for r in results),sale_checks=sum(r['sale_checks'] for r in results),profile_checks=sum(r['profile_checks'] for r in results),seconds=time.perf_counter()-start,rows=results)
 panel.save(OUT/('PILOT.json' if args.limit else 'RESULTS.json'),receipt)
 print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
