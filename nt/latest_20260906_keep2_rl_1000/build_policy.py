"""Rebuild KEEP=2 policy using the previously delivered F3/opponent package."""
from pathlib import Path
import subprocess,sys,json,hashlib,time
P=Path(__file__).resolve().parent
BASE=P.parent/'latest_20260906_c3_f3_j7c3_search'

def main():
    if sys.platform!='linux':raise SystemExit('Use Linux/WSL, g++ and Python 3.12.')
    if not (BASE/'src/stage/f3/agent.cpp').is_file():raise SystemExit('Fetch the sibling C3/F3 C++ package first; see README.')
    out=P/'build';out.mkdir(exist_ok=True)
    cmd=['g++','-std=c++20','-O3','-DNDEBUG','-fPIC','-ffp-contract=off','-pthread','-shared','-Wl,-Bsymbolic',
         '-I'+str(BASE/'vendor'),str(P/'src/policy.cpp'),'-o',str(out/'keep2.so')]
    t=time.perf_counter();subprocess.run(cmd,check=True)
    evidence=dict(command=cmd,seconds=time.perf_counter()-t,compiler=subprocess.check_output(['g++','--version'],text=True),
        policy_sha256=hashlib.sha256((out/'keep2.so').read_bytes()).hexdigest())
    (out/'BUILD_RECEIPT.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
    print('KEEP2_BUILD_PASS',round(evidence['seconds'],2),flush=True)
if __name__=='__main__':main()
