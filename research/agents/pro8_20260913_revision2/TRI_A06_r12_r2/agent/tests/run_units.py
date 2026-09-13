"""Focused source unit checks; no match runner or online dependencies."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,time
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--cxx',default=shutil.which('g++'));a=p.parse_args()
if not a.cxx:raise SystemExit('g++ required')
d=R/'build/tests';d.mkdir(parents=True,exist_ok=True)
flags=[x for x in json.loads((R/'COMPILER_FLAGS.json').read_text()) if x not in ('-O3','-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic')]+['-O1']
cmd=[a.cxx,*flags,str(R/'tests/unit_ongoing_labor.cpp'),'-o',str(d/'unit_ongoing_labor')]
t=time.monotonic();c=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
(d/'unit_compile.stdout').write_text(c.stdout);(d/'unit_compile.stderr').write_text(c.stderr)
if c.returncode:raise SystemExit(c.returncode)
r=subprocess.run([str(d/'unit_ongoing_labor')],capture_output=True,text=True,timeout=10)
(d/'unit.stdout').write_text(r.stdout);(d/'unit.stderr').write_text(r.stderr)
(d/'receipt.json').write_text(json.dumps({'command':cmd,'compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],'returncode':r.returncode,'elapsed_seconds':time.monotonic()-t,'executable_sha256':hashlib.sha256((d/'unit_ongoing_labor').read_bytes()).hexdigest()},indent=2))
print(r.stdout,end='');raise SystemExit(r.returncode)
