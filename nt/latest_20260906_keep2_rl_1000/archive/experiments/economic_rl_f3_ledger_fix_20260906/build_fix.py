"""Build isolated F3 fix; leave all original source, results and SO files intact."""
from pathlib import Path
import hashlib,json,subprocess,time
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parent;OLD=P.parent/'economic_rl_three_arm_20260906';ROOT=P.parents[1]
B=P/'build';B.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen=json.loads((OLD/'SOURCE_FREEZE.json').read_text())
for f,h in frozen.items():assert sha(f)==h,f
preserve=[OLD/'f3_policy.cpp',OLD/'stage/f3/agent.cpp',OLD/'build/f3.so',OLD/'FINAL_RESULTS.json',OLD/'G1_RECEIPT.json']
before={str(p):sha(p)for p in preserve}
boost=ROOT/'gpt_review/gpt_code/gpt-6-dp/Kaggriculture_CPP_Daily_DP_20260905_v1_improve_c++/Kaggriculture_Daily_DP_CPP_20260905/vendor'
common=['g++','-std=c++20','-O3','-ffp-contract=off','-pthread','-I'+str(boost)]
jobs=[('f3',common+['-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic',str(P/'f3_policy.cpp'),'-o',str(B/'f3.so')]),
      ('test',common+[str(P/'test_ledger.cpp'),str(OLD/'stage/arena/vendor/simulator.cpp'),'-o',str(B/'test_ledger')])]
def run(job):
    name,command=job;t=time.perf_counter();r=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (B/f'{name}.log').write_text(r.stdout)
    row=dict(name=name,command=command,seconds=time.perf_counter()-t,exit_code=r.returncode)
    print(name,r.returncode,round(row['seconds'],1),r.stdout[-3500:],flush=True);return row
with ThreadPoolExecutor(max_workers=2)as pool:rows=list(pool.map(run,jobs))
assert all(sha(p)==h for p,h in before.items())
receipt=dict(builds=rows,preserved_original=before,files={str(p.relative_to(P)):sha(p)for p in P.rglob('*')if p.is_file() and p.suffix in ('.cpp','.hpp','.so')})
(P/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
if any(r['exit_code']for r in rows):raise SystemExit(1)
r=subprocess.run([str(B/'test_ledger')],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(P/'UNIT_TEST.log').write_text(r.stdout)
(P/'UNIT_TEST_RECEIPT.json').write_text(json.dumps(dict(status='PASS'if r.returncode==0 else'FAIL',exit_code=r.returncode,output=r.stdout,binary_sha256=sha(B/'test_ledger')),indent=2))
print(r.stdout,flush=True)
raise SystemExit(r.returncode)
