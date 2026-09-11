from pathlib import Path
import subprocess,hashlib,json,time,argparse
HERE=Path(__file__).resolve().parent/'candidate_r2p8';POLICY=HERE/'policy'
p=argparse.ArgumentParser();p.add_argument('--name',choices=['off','probe','mpc','coupled'],required=True);p.add_argument('--market',type=int,choices=[0,1,2],default=0);args=p.parse_args()
mode={'off':0,'probe':0,'mpc':1,'coupled':2}[args.name];observe=int(args.name!='off')
target=POLICY/f'clock_{args.name}_market{args.market}.so';assert not target.exists(),target
sources={str(p.relative_to(POLICY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in POLICY.rglob('*') if p.is_file() and p.suffix in {'.cpp','.hpp','.h','.inc'}}
cmd=['g++-13','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-ffp-contract=off',f'-DR2_MARKET_INTEGRAL={args.market}',f'-DR2_SALE_CLOCK_MODE={mode}',f'-DR2_OBSERVE_PUBLIC_TRADES={observe}',
     '-fPIC','-shared','-Wl,-Bsymbolic',str(POLICY/'bridge.cpp'),str(POLICY/'executor/vendor/simulator.cpp'),'-o',str(target)]
start=time.perf_counter()
with (HERE/f'compile_{args.name}_market{args.market}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(status='BUILT_NEEDS_VALIDATION' if r.returncode==0 else 'FAILED',name=args.name,mode=mode,observe=observe,market=args.market,command=cmd,sources=sources,seconds=time.perf_counter()-start)
if target.exists():receipt['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
(HERE/f'BUILD_{args.name}_market{args.market}.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='sources'}),flush=True)
raise SystemExit(r.returncode)
