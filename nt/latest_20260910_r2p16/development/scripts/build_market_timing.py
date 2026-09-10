from pathlib import Path
import subprocess,hashlib,json,time
HERE=Path(__file__).resolve().parent/'candidate_r2p4';POLICY=HERE/'policy'
sources={str(p.relative_to(POLICY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in POLICY.rglob('*')
         if p.is_file() and p.suffix in {'.cpp','.hpp','.h','.inc'}}
for label,early,lag in [('off',.5,0),('quarter',.25,0),('quarter_half',.25,.5),('quarter_one',.25,1)]:
    target=POLICY/f'timing_{label}.so';assert not target.exists()
    cmd=['g++-13','-std=c++20','-O3','-march=x86-64','-ffp-contract=off',f'-DR2_RIVAL_EARLY_SHARE={early}',f'-DR2_PRODUCT_HANDOFF_DELAY={lag}',
         '-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic',str(POLICY/'bridge.cpp'),str(POLICY/'executor/vendor/simulator.cpp'),'-o',str(target)]
    start=time.perf_counter()
    with (HERE/f'compile_{label}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(status='BUILT_NEEDS_TRAJECTORY_CHECK' if r.returncode==0 else 'FAILED',early=early,lag=lag,command=cmd,sources=sources,seconds=time.perf_counter()-start)
    if target.exists():receipt['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (HERE/f'BUILD_{label}.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='sources'}),flush=True)
    if r.returncode:raise SystemExit(r.returncode)
