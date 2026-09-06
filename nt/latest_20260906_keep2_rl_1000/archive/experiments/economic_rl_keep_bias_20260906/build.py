"""Isolated mechanical source generation, with prior files frozen by hash."""
from pathlib import Path
import json, subprocess, hashlib, time
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
OLD=P.parent/'economic_rl_three_arm_20260906';FIX=P.parent/'economic_rl_f3_ledger_fix_20260906'
B=P/'build';B.mkdir(exist_ok=False)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
fixed=json.loads((FIX/'ACCEPTANCE.json').read_text())
hashes=dict(fixed['fixed_hashes']);hashes.update(fixed['preserved_original'])
hashes.update(json.loads((OLD/'SOURCE_FREEZE.json').read_text()))
for f,h in hashes.items():assert digest(f)==h,f
header=(OLD/'rl_common.hpp').read_text()
assert header.count('j==0?3.58351893846f:0.f')==1
header=header.replace('namespace econrl {','namespace econrl {\ninline thread_local float inference_keep_bonus=3.58351893846f;',1)
header=header.replace('j==0?3.58351893846f:0.f','j==0?inference_keep_bonus:0.f')
(B/'rl_common.hpp').write_text(header)
source=(FIX/'f3_policy.cpp').read_text()
replacements={'"stage/f3/agent.cpp"':str(FIX/'stage/f3/agent.cpp'),
 '"../economic_rl_three_arm_20260906/stage/arena/vendor/simulator.hpp"':str(OLD/'stage/arena/vendor/simulator.hpp'),
 '"../economic_rl_three_arm_20260906/rl_common.hpp"':str(B/'rl_common.hpp'),
 '"../economic_rl_three_arm_20260906/f3_config.hpp"':str(OLD/'f3_config.hpp')}
for before,after in replacements.items():
    assert source.count(before)==1,before
    source=source.replace(before,'"'+after+'"')
(B/'f3_policy.cpp').write_text(source)
boost=ROOT/'gpt_review/gpt_code/gpt-6-dp/Kaggriculture_CPP_Daily_DP_20260905_v1_improve_c++/Kaggriculture_Daily_DP_CPP_20260905/vendor'
command=['g++','-std=c++20','-O3','-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic','-ffp-contract=off','-pthread','-I'+str(boost),str(P/'policy.cpp'),'-o',str(B/'keep.so')]
tic=time.perf_counter();r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(B/'compiler.log').write_text(r.stdout)
for f,h in hashes.items():assert digest(f)==h,f
receipt=dict(status='PASS'if r.returncode==0 else'FAIL',exit_code=r.returncode,seconds=time.perf_counter()-tic,command=command,preserved=hashes,
             generated={str(f):digest(f)for f in B.iterdir()if f.suffix in ('.hpp','.cpp','.so')})
(P/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
print('BUILD',receipt['status'],receipt['seconds'],r.stdout[-1800:],flush=True)
if r.returncode:raise SystemExit(r.returncode)
