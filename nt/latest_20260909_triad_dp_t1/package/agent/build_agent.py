"""Build the portable x86-64 native policy. No Python development headers needed."""
from pathlib import Path
import argparse,os,subprocess,time,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--cxx',default=os.environ.get('CXX','g++'));a=p.parse_args()
root=Path(__file__).resolve().parent;start=time.perf_counter()
cmd=[a.cxx,'-std=c++20','-O3','-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic','-ffp-contract=off',str(root/'hybrid_bridge.cpp'),str(root/'executor/vendor/simulator.cpp'),'-o',str(root/'agent.so')]
subprocess.run(cmd,check=True)
receipt={'command':cmd,'seconds':time.perf_counter()-start,'sha256':hashlib.sha256((root/'agent.so').read_bytes()).hexdigest(),'compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0]}
(root/'build_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
