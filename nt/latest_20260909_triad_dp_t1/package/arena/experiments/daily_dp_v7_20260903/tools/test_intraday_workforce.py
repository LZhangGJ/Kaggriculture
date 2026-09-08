from pathlib import Path
import argparse,hashlib,json,subprocess
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
E=Path(__file__).resolve().parents[1];out=E/a.out;out.mkdir(exist_ok=False)
src=E/'native/test_intraday_workforce.cpp';binary=E/'native/build/test_intraday_workforce'
cmd=['g++','-std=c++20','-O2','-fopenmp','-I'+str(E/'native'),str(src),str(E/'native/build/simulator.o'),'-o',str(binary)]
r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if r.returncode:raise SystemExit(r.returncode)
r=subprocess.run([str(binary)],capture_output=True,text=True);(out/'run.log').write_text(r.stdout+r.stderr)
d=json.loads(r.stdout) if not r.returncode else dict(status='FAIL',error=r.stderr)
d.update(command=cmd,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),build=json.loads((E/'native/build/build_receipt.json').read_text()))
(out/'acceptance.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='build'}));raise SystemExit(r.returncode)
