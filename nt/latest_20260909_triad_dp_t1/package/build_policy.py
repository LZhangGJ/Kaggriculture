"""Atomic source-checked C++20 build; receipts refer to files actually compiled."""
from pathlib import Path
import subprocess,argparse,time,hashlib,json,zipfile,shutil,os
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--baseline',choices=['auto','j7']);a=p.parse_args()
src=R/('src/baseline_bridge.cpp' if a.baseline else 'policy/bridge.cpp')
out=R/(f'baseline_{a.baseline}.so' if a.baseline else 'policy/agent.so')
root=R/('agent' if a.baseline else 'policy')
files=[f for f in root.rglob('*') if f.is_file() and f.suffix in ('.hpp','.cpp','.py','.json') and not f.name.endswith('.build.json')]
if a.baseline:files.extend([src,R/'policy/executor/vendor/simulator.cpp'])
files=sorted(set(files))
def hashes():return {str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
before=hashes();source_hash=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest();store=R/'snapshots';store.mkdir(exist_ok=True)
with zipfile.ZipFile(store/(source_hash+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,f.relative_to(R))
start=time.perf_counter();tmp=out.with_name(out.name+f'.{os.getpid()}.tmp')
cmd=['g++','-std=c++20','-O3','-DNDEBUG','-ffp-contract=off','-fPIC','-shared','-Wl,-Bsymbolic']
if a.baseline=='auto':cmd+=['-DBASE_AUTO']
cmd +=[str(src),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(tmp)]
try:
 subprocess.run(cmd,check=True)
 if hashes()!=before:raise RuntimeError('Source changed during build; output rejected')
 tmp.replace(out)
except Exception:
 tmp.unlink(missing_ok=True);raise
binary_hash=hashlib.sha256(out.read_bytes()).hexdigest();shutil.copy2(out,store/(binary_hash+'.so'))
receipt=dict(binary_hash=binary_hash,source_hash=source_hash,source_files=before,command=cmd,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],source_stability_check='PASS')
out.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2));print(out.name,time.perf_counter()-start,flush=True)
