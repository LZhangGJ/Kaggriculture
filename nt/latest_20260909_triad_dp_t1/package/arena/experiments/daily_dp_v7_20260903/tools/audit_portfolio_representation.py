from pathlib import Path
import hashlib,json,subprocess,time
EXP=Path(__file__).resolve().parents[1];build=EXP/'native/build';source=EXP/'native/test_portfolio_representation.cpp'
receipt=json.loads((build/'build_receipt.json').read_text())
for rel,h in receipt['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
out=EXP/'receipts/s4j1_representation_audit_v1';out.mkdir(exist_ok=False)
command=['g++','-std=c++20','-O2','-fopenmp','-I'+str(EXP/'native'),str(source),str(build/'simulator.o'),'-o',str(build/'test_portfolio_representation')]
subprocess.run(command,check=True);r=subprocess.run([str(build/'test_portfolio_representation')],capture_output=True,text=True,check=True)
data=json.loads(r.stdout);data.update(build=receipt,command=command,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    interpretation='Same maintained/replanted business assumption, different representations. This is a forecast consistency audit, not official output equality or a chosen winning future.')
(out/'summary.json').write_text(json.dumps(data,indent=2));print(r.stdout)
