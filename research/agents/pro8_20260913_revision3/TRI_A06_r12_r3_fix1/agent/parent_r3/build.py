"""Offline C++20 rebuild; hashes every used source and records the actual compiler."""
from pathlib import Path
import argparse,subprocess,hashlib,json,time,shutil,sys
from datetime import datetime, timezone
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--cxx',default=shutil.which('g++'));p.add_argument('--out',type=Path);p.add_argument('--unit',action='store_true');a=p.parse_args()
if not a.cxx:raise SystemExit('C++20 compiler required')
out=(a.out or R/'policy/a06.so').resolve();out.parent.mkdir(parents=True,exist_ok=True)
flags=json.loads((R/'COMPILER_FLAGS.json').read_text())
cmd=[a.cxx,*flags,str(R/'policy/bridge.cpp'),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(out)]
t=time.perf_counter();result=subprocess.run(cmd,cwd=R,capture_output=True,text=True)
log=R/'build';log.mkdir(exist_ok=True);(log/'compile.stdout').write_text(result.stdout);(log/'compile.stderr').write_text(result.stderr)
if result.returncode:raise SystemExit(result.returncode)
src={str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((R/'policy').rglob('*')) if x.is_file() and x.suffix in ('.cpp','.hpp','.inc','.py','.json') and not x.name.endswith('.BUILD.json')}
build_inputs=dict(src)
for name in ('main.py','build.py','COMPILER_FLAGS.json'):
 build_inputs[name]=hashlib.sha256((R/name).read_bytes()).hexdigest()
receipt={'built_at_utc':datetime.now(timezone.utc).isoformat(),'all_build_inputs':build_inputs,'command':cmd,'compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],'seconds':time.perf_counter()-t,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'sources':src,'config':json.loads((R/'policy/config.json').read_text())}
out.with_suffix('.BUILD.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='sources'},indent=2))
if a.unit:subprocess.run([sys.executable,str(R/'tests/run_units.py'),'--cxx',a.cxx],check=True)
