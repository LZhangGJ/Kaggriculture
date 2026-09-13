from pathlib import Path
import json,subprocess,hashlib,time
W=Path(__file__).resolve().parents[1];C=W/'candidate';P=W/'parent'
flags=json.loads((C/'COMPILER_FLAGS.json').read_text());out=C/'build/tests/fixed_core_audit.so'
cmd=['/usr/bin/g++',*flags,'-DAUDIT_PARENT_BRIDGE="'+str(P/'policy/bridge.cpp')+'"',str(C/'tests/fixed_core_audit.cpp'),str(P/'policy/executor/vendor/simulator.cpp'),'-o',str(out)]
t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True);(W/'logs/fixed_audit_compile.raw.stdout').write_text(r.stdout);(W/'logs/fixed_audit_compile.raw.stderr').write_text(r.stderr)
receipt={'command':cmd,'seconds':time.monotonic()-t,'returncode':r.returncode,'test_only':True,'source_parent':'TRI_A06_r12_r1','native_sha256':hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else None}
(W/'logs/fixed_audit_compile_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));raise SystemExit(r.returncode)
