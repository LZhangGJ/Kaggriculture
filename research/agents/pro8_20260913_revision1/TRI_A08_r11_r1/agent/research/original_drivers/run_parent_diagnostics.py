from pathlib import Path
import subprocess,concurrent.futures,json,datetime,time
R=Path(__file__).resolve().parent
end=datetime.datetime.fromisoformat('2026-09-13T04:15:47.575+00:00').timestamp()
out=R/'logs/parent_history';out.mkdir(exist_ok=True)
files=sorted((R/'input/replays').glob('*.gz'))
def run(p):
 if time.time()>end-900: return {'error':'deadline budget guard','case':p.name}
 name=p.name.removesuffix('.json.gz')
 cmd=['/usr/bin/time','-v','-o',str(out/(name+'.time')),'timeout','-k','3s','120s','python','-B',str(R/'candidate/tests/historical_inputs.py'),'--root',str(R/'input/agent'),'--binary',str(R/'input/agent/policy/a08_r11.so'),'--replay',str(p),'--limit','719','--out',str(out/(name+'.json'))]
 with open(out/(name+'.stdout'),'w') as a,open(out/(name+'.stderr'),'w') as b:
  result=subprocess.run(cmd,stdout=a,stderr=b)
 s={'case':name,'returncode':result.returncode};print(json.dumps(s),flush=True);return s
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 rows=list(pool.map(run,files))
(out/'run.json').write_text(json.dumps(rows,indent=2))
