"""Compare the compiled PRODUCTION Planner::trade with the frozen Python engine."""
from pathlib import Path
import subprocess,sys,json,hashlib
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'input/referee'))
from cpu_runtime import load_engine
eng=load_engine();exe=R/'candidate/build/test_tri_market_floor';proc=subprocess.run([str(exe),'--rows'],capture_output=True,text=True,check=True)
(R/'logs/floor_oracle.raw_rows.txt').write_text(proc.stdout)
count=0;units=0
for line in proc.stdout.splitlines():
 i,stock,q,cash,end=map(float,line.split());item=list(eng.PRODUCTS)[int(i)];stock=int(stock);q=int(q)
 market={'inventory':{item:stock}};farm={'money':0};private={'shed':{item:q}}
 for _ in range(q):
  p=eng.market_price(item,market['inventory'][item]);assert eng._commit_unit('SELL',item,p,farm,private,market);units+=1
 assert farm['money']==cash and market['inventory'][item]==end,(item,stock,q,cash,end,farm,market)
 count+=1
result={'scope':'Compiled production Planner::trade, integer floor-crossing one-sided batches vs frozen official Python _commit_unit. Not games or recoverable-profit estimates.','cases':count,'successful_official_units':units,'all_passed':True,'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'rows_sha256':hashlib.sha256(proc.stdout.encode()).hexdigest()}
(R/'diagnostics/production_floor_oracle.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
