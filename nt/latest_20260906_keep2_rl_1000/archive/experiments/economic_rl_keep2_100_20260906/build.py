from pathlib import Path
import subprocess,json,hashlib,time
P=Path(__file__).resolve().parent;ROOT=P.parents[1];PRIOR=P.parent/'economic_rl_keep_bias_20260906'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
receipt=json.loads((PRIOR/'BUILD_RECEIPT.json').read_text());assert receipt['status']=='PASS'
hashes=dict(receipt['preserved']);hashes.update(receipt['generated'])
for f,h in hashes.items():assert sha(f)==h,f
B=P/'build';B.mkdir(exist_ok=False)
boost=ROOT/'gpt_review/gpt_code/gpt-6-dp/Kaggriculture_CPP_Daily_DP_20260905_v1_improve_c++/Kaggriculture_Daily_DP_CPP_20260905/vendor'
command=['g++','-std=c++20','-O3','-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic','-ffp-contract=off','-pthread','-I'+str(boost),str(P/'policy.cpp'),'-o',str(B/'keep2.so')]
tic=time.perf_counter();r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(B/'compiler.log').write_text(r.stdout)
for f,h in hashes.items():assert sha(f)==h,f
result=dict(status='PASS'if r.returncode==0 else'FAIL',command=command,seconds=time.perf_counter()-tic,
            preserved=hashes,built={str(B/'keep2.so'):sha(B/'keep2.so')}if r.returncode==0 else {})
(P/'BUILD_RECEIPT.json').write_text(json.dumps(result,indent=2));print(result['status'],result['seconds'],r.stdout[-1500:],flush=True)
if r.returncode:raise SystemExit(r.returncode)
