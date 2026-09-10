from pathlib import Path
import subprocess,hashlib,json,time,argparse
HERE=Path(__file__).resolve().parent/'candidate_r2p5';POLICY=HERE/'policy'
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['probe','half','one'],default='probe');p.add_argument('--revision',default='v2');args=p.parse_args()
weight={'probe':0,'half':.5,'one':1}[args.variant];target=POLICY/f'ledger_{args.variant}_{args.revision}.so'
assert not target.exists(),target
sources={str(p.relative_to(POLICY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in POLICY.rglob('*')
         if p.is_file() and p.suffix in {'.cpp','.hpp','.h','.inc'}}
cmd=['g++-13','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-ffp-contract=off','-DR2_OBSERVE_PUBLIC_TRADES=1',f'-DR2_PENDING_STOCK_WEIGHT={weight}',
     '-fPIC','-shared','-Wl,-Bsymbolic',str(POLICY/'bridge.cpp'),str(POLICY/'executor/vendor/simulator.cpp'),'-o',str(target)]
start=time.perf_counter()
with (HERE/f'compile_{args.variant}_{args.revision}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(status='BUILT_NEEDS_VALIDATION' if r.returncode==0 else 'FAILED',variant=args.variant,weight=weight,command=cmd,sources=sources,seconds=time.perf_counter()-start)
if target.exists():receipt['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
(HERE/f'BUILD_{args.variant}_{args.revision}.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='sources'}),flush=True)
raise SystemExit(r.returncode)
