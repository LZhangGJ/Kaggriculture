"""Offline positive/negative source tests; no match or online dependency."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,time
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--cxx',default=shutil.which('g++'));a=p.parse_args()
if not a.cxx:raise SystemExit('g++ required')
d=R/'build/tests';d.mkdir(parents=True,exist_ok=True)
flags=[x for x in json.loads((R/'COMPILER_FLAGS.json').read_text()) if x not in ('-O3','-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic')]+['-O1']
receipts=[]
for name in ('unit_ongoing_labor','unit_ongoing_supply','unit_ongoing_admission','unit_terminal_supply','unit_terminal_rolling','unit_r14_tail'):
 cmd=[a.cxx,*flags,str(R/'tests'/f'{name}.cpp')]
 if name in ('unit_ongoing_admission','unit_terminal_supply','unit_terminal_rolling','unit_r14_tail'):cmd += [str(R/'policy/executor/vendor/simulator.cpp')]
 cmd += ['-o',str(d/name)]
 t=time.monotonic();c=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 (d/f'{name}.compile.stdout').write_text(c.stdout);(d/f'{name}.compile.stderr').write_text(c.stderr)
 if c.returncode:raise SystemExit(c.returncode)
 r=subprocess.run([str(d/name)],capture_output=True,text=True,timeout=20)
 (d/f'{name}.stdout').write_text(r.stdout);(d/f'{name}.stderr').write_text(r.stderr)
 receipt={'name':name,'command':cmd,'compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],'returncode':r.returncode,'elapsed_seconds':time.monotonic()-t,'executable_sha256':hashlib.sha256((d/name).read_bytes()).hexdigest()}
 receipts.append(receipt);(d/'receipt.json').write_text(json.dumps(receipts,indent=2));print(r.stdout,end='',flush=True)
 if r.returncode:print(r.stderr,end='');raise SystemExit(r.returncode)
