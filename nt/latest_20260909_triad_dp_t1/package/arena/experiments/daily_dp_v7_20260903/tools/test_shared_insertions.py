from pathlib import Path
import argparse,hashlib,json,subprocess
parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
EXP=Path(__file__).resolve().parents[1];out=EXP/args.out;out.mkdir(exist_ok=False)
src=EXP/'native/test_shared_insertions.cpp';binary=EXP/'native/build/test_shared_insertions'
cmd=['g++','-std=c++20','-O2','-fopenmp','-I'+str(EXP/'native'),str(src),str(EXP/'native/build/simulator.o'),'-o',str(binary)]
r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if r.returncode:raise SystemExit(r.returncode)
r=subprocess.run([str(binary)],capture_output=True,text=True);(out/'run.log').write_text(r.stdout+r.stderr)
data=json.loads(r.stdout) if not r.returncode else dict(status='FAIL',error=r.stderr)
data.update(command=cmd,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),build=json.loads((EXP/'native/build/build_receipt.json').read_text()))
(out/'acceptance.json').write_text(json.dumps(data,indent=2));print(json.dumps({k:v for k,v in data.items() if k!='build'}));raise SystemExit(r.returncode)
