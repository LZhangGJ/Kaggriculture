from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, subprocess, sys, time
w=Path('/mnt/data/TRI_A08_work');out=w/'logs/anchored_branches';out.mkdir(exist_ok=True)
tasks=[]
for casefile in sorted((w/'logs/anchored_cases').glob('*.json')):
 case=json.loads(casefile.read_text())
 for version in ['parent','candidate']:
  root=w/('input/agent' if version=='parent' else 'candidate');binary=root/('policy/a08_r11.so' if version=='parent' else 'policy/tri_a08_r11_r1.so')
  stem=out/(case['id']+'_'+version)
  cmd=['/usr/bin/time','-v','-o',str(stem)+'.time',sys.executable,'-B',str(w/'candidate/tests/short_sale_branches.py'),'--root',str(root),'--binary',str(binary),'--fixture',str(w/'input/replays'/case['fixture']),'--case',str(casefile),'--referee',str(w/'input/referee'),'--out',str(stem)+'.json.gz']
  tasks.append((str(stem),cmd))
def run(task):
 stem,cmd=task;start=time.time()
 try:
  with open(stem+'.stdout','w') as o,open(stem+'.stderr','w') as e:r=subprocess.run(cmd,stdout=o,stderr=e,timeout=100)
  x={'stem':stem,'command':cmd,'returncode':r.returncode,'seconds':time.time()-start}
 except Exception as e:x={'stem':stem,'command':cmd,'error':repr(e),'seconds':time.time()-start}
 Path(stem+'.run.json').write_text(json.dumps(x,indent=2));print(json.dumps(x),flush=True);return x
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,tasks))
assert all(x.get('returncode')==0 for x in results)
