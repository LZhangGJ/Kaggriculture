from pathlib import Path
import hashlib,json,shutil,subprocess,time
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4u_pack_mechanisms_v1';out.mkdir(exist_ok=False)
probe=E/'native/s4u_memo_probe_v2';receipt=json.loads((probe/'acceptance.json').read_text());assert receipt['status']=='PASS_BUILD'
source=(E/'native/policy.hpp').read_text();start=source.index('std::pair<std::vector<Route>,int>regret_pack(');end=source.index('std::pair<std::vector<Route>,int>pack(',start)
(out/'reference_regret.inc').write_text(source[start:end].replace('>regret_pack(', '>regret_pack_reference('))
binary=out/'test_pack_memo';src=E/'native/test_pack_memo_equivalence.cpp'
test_src=out/src.name;shutil.copy2(src,test_src)
cmd=['g++','-std=c++20','-O3','-march=native','-DNDEBUG','-I'+str(probe),'-I'+str(out),str(test_src),str(probe/'vendor/simulator.cpp'),'-o',str(binary)]
tic=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
assert r.returncode==0,r.stderr[-2500:]
run=subprocess.run([str(binary)],capture_output=True,text=True);(out/'run.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stdout+run.stderr
result=json.loads(run.stdout);result.update(seconds=time.perf_counter()-tic,probe_build=receipt,
    source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256((out/'reference_regret.inc').read_bytes()).hexdigest())
(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(run.stdout,flush=True)
