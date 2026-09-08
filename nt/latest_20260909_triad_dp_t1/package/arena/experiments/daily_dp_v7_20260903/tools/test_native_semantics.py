from pathlib import Path
import argparse,hashlib,json,subprocess,time
EXP=Path(__file__).resolve().parents[1];native=EXP/'native';build=native/'build';source=native/'test_policy.cpp'
binary=build/'test_policy';cmd=['g++','-std=c++20','-O2','-fopenmp','-I'+str(native),str(source),str(build/'simulator.o'),'-o',str(binary)]
subprocess.run(cmd,check=True);r=subprocess.run([str(binary)],check=True,text=True,capture_output=True)
data=json.loads(r.stdout);data.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),build=json.loads((build/'build_receipt.json').read_text()))
p=argparse.ArgumentParser();p.add_argument('--out',default='receipts/s3_native_mechanisms');a=p.parse_args()
out=EXP/a.out;out.mkdir(exist_ok=False)
(out/'acceptance.json').write_text(json.dumps(data,indent=2),encoding='utf8');print(r.stdout)
