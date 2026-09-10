from pathlib import Path
import argparse,subprocess,hashlib,json,time
HERE=Path(__file__).resolve().parent/'candidate_r2p16';POLICY=HERE/'policy'
p=argparse.ArgumentParser();p.add_argument('--mode',type=int,choices=[0,1,2],required=True);a=p.parse_args()
name=f'startupsupply{a.mode}';target=POLICY/(name+'.so');assert not target.exists(),target
sources={str(p.relative_to(POLICY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in POLICY.rglob('*') if p.is_file() and p.suffix in {'.cpp','.hpp','.h','.inc'}}
cmd=['g++-13','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-ffp-contract=off',f'-DR2_STARTUP_SUPPLY_MODE={a.mode}','-DR2_LOCAL_SALE_TIMING=1','-DR2_FINITE_FERTILIZER=1','-DR2_CROP_CLOCK_MODE=1','-DR2_OBSERVE_PUBLIC_TRADES=1','-DR2_MARKET_INTEGRAL=0','-DR2_SALE_CLOCK_MODE=0','-fPIC','-shared','-Wl,-Bsymbolic',str(POLICY/'bridge.cpp'),str(POLICY/'executor/vendor/simulator.cpp'),'-o',str(target)]
started=time.perf_counter()
with (HERE/(name+'.compile.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(status='BUILT_NEEDS_VALIDATION' if r.returncode==0 else 'FAILED',mode=a.mode,command=cmd,sources=sources,seconds=time.perf_counter()-started)
if target.exists():receipt['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
(HERE/(name+'.BUILD.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='sources'}),flush=True)
raise SystemExit(r.returncode)
