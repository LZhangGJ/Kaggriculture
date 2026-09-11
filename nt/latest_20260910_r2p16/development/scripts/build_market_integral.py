from pathlib import Path
import subprocess,hashlib,json,time,argparse
HERE=Path(__file__).resolve().parent/'candidate_r2p7';POLICY=HERE/'policy'
p=argparse.ArgumentParser();p.add_argument('--mode',type=int,choices=[0,1,2],required=True);args=p.parse_args()
target=POLICY/f'market_{args.mode}.so';assert not target.exists(),target
sources={str(p.relative_to(POLICY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in POLICY.rglob('*') if p.is_file() and p.suffix in {'.cpp','.hpp','.h','.inc'}}
cmd=['g++-13','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-ffp-contract=off',f'-DR2_MARKET_INTEGRAL={args.mode}',
     '-fPIC','-shared','-Wl,-Bsymbolic',str(POLICY/'bridge.cpp'),str(POLICY/'executor/vendor/simulator.cpp'),'-o',str(target)]
start=time.perf_counter()
with (HERE/f'compile_{args.mode}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(status='BUILT_NEEDS_VALIDATION' if r.returncode==0 else 'FAILED',mode=args.mode,command=cmd,sources=sources,seconds=time.perf_counter()-start)
if target.exists():receipt['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
(HERE/f'BUILD_{args.mode}.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='sources'}),flush=True)
raise SystemExit(r.returncode)
