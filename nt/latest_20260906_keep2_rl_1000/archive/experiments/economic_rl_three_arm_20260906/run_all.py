"""One experiment at a time, at most16 simulator workers; never auto-resume errors."""
from pathlib import Path
import subprocess,sys,time,json
P=Path(__file__).resolve().parent
logs=P/'logs';logs.mkdir(exist_ok=True)
def run(name,args):
 log=logs/(name+'.log');start=time.perf_counter()
 with log.open('x')as f:
  p=subprocess.Popen([sys.executable,*map(str,args)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in p.stdout:f.write(line);f.flush();print(name,line.rstrip(),flush=True)
  code=p.wait()
 (logs/(name+'.receipt.json')).write_text(json.dumps(dict(returncode=code,seconds=time.perf_counter()-start,command=[sys.executable,*map(str,args)]),indent=2))
 if code:raise SystemExit(code)
def main():
 for gate in ('G1_RECEIPT.json','G2_RECEIPT.json'):
  assert json.loads((P/gate).read_text())['status']=='PASS'
 for rep in (0,1):
  for arm in ('c3auto','f3','c3j7'):
   run(f'train_{arm}_{rep}',[P/'train.py','--arm',arm,'--run',rep])
 run('baselines',[P/'baselines.py'])
 run('report',[P/'report.py'])
if __name__=='__main__':main()
