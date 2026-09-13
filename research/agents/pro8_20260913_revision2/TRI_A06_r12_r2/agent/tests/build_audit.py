"""Build the test-only frozen-parent context / unchanged-executor adapter."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,time
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,default=R/'provenance/parent_agent');p.add_argument('--out',type=Path,default=R/'build/tests/fixed_core_audit.so');p.add_argument('--cxx',default=shutil.which('g++'));a=p.parse_args()
if not a.cxx:raise SystemExit('C++20 compiler required')
parent=a.parent.resolve();out=a.out.resolve();out.parent.mkdir(parents=True,exist_ok=True)
flags=json.loads((R/'COMPILER_FLAGS.json').read_text())
cmd=[a.cxx,*flags,'-DAUDIT_PARENT_BRIDGE="'+str(parent/'policy/bridge.cpp')+'"',str(R/'tests/fixed_core_audit.cpp'),str(parent/'policy/executor/vendor/simulator.cpp'),'-o',str(out)]
t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
out.with_suffix('.stdout').write_text(r.stdout);out.with_suffix('.stderr').write_text(r.stderr)
receipt={'command':cmd,'compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],'seconds':time.monotonic()-t,'returncode':r.returncode,'test_only':True,'source_parent':'TRI_A06_r12_r1','native_sha256':hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() and not r.returncode else None}
out.with_suffix('.BUILD.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));raise SystemExit(r.returncode)
