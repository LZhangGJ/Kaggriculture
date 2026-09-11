from pathlib import Path
import argparse, hashlib, json, subprocess
EXP=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
out=EXP/a.out;out.mkdir(exist_ok=False)
source=EXP/'native/test_crop_delivery.cpp';binary=EXP/'native/build/test_crop_delivery'
cmd=['g++','-std=c++20','-O2','-fopenmp','-I'+str(EXP/'native'),str(source),str(EXP/'native/build/simulator.o'),'-o',str(binary)]
compile=subprocess.run(cmd,text=True,capture_output=True)
(out/'compile.log').write_text(compile.stdout+compile.stderr)
if compile.returncode:raise SystemExit(compile.returncode)
r=subprocess.run([str(binary)],text=True,capture_output=True)
(out/'run.log').write_text(r.stdout+r.stderr)
data=json.loads(r.stdout) if not r.returncode else dict(status='FAIL',error=r.stderr)
data.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),build=json.loads((EXP/'native/build/build_receipt.json').read_text()))
(out/'acceptance.json').write_text(json.dumps(data,indent=2));print(json.dumps({k:v for k,v in data.items() if k!='build'}))
raise SystemExit(r.returncode)
