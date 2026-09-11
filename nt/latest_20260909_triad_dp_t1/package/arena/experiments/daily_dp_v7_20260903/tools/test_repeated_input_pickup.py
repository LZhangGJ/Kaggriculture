from pathlib import Path
import hashlib,json,subprocess
EXP=Path(__file__).resolve().parents[1];out=EXP/'receipts/s4k2_repeated_pickup_reproduction_v1';out.mkdir(exist_ok=False)
src=EXP/'native/test_repeated_input_pickup.cpp';binary=EXP/'native/build/test_repeated_input_pickup'
cmd=['g++','-std=c++20','-O2','-fopenmp','-I'+str(EXP/'native'),str(src),str(EXP/'native/build/simulator.o'),'-o',str(binary)]
subprocess.run(cmd,check=True);r=subprocess.run([str(binary)],capture_output=True,text=True)
(out/'run.log').write_text(r.stdout+r.stderr);result=json.loads(r.stdout);result.update(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    build=json.loads((EXP/'native/build/build_receipt.json').read_text()),scope='Synthetic legal repeated pickup, no live policy search or future events.')
(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(r.stdout)
raise SystemExit(r.returncode)
