from pathlib import Path
import subprocess,hashlib,json,time
HERE=Path(__file__).resolve().parent/'candidate_r2p3';POLICY=HERE/'policy'
sources={str(p.relative_to(POLICY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in POLICY.rglob('*')
         if p.is_file() and p.suffix in {'.cpp','.hpp','.h','.inc'}}
for label,lag in [('off',0),('half',.5),('one',1)]:
    target=POLICY/f'shipment_{label}.so';assert not target.exists()
    common=['g++-13','-std=c++20','-O3','-march=x86-64','-ffp-contract=off',f'-DR2_PRODUCT_HANDOFF_DELAY={lag}']
    test=HERE/f'test_{label}'
    subprocess.run(common+[str(HERE/'test_shipment.cpp'),'-o',str(test)],check=True)
    subprocess.run([str(test)],check=True)
    cmd=common+['-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic',str(POLICY/'bridge.cpp'),str(POLICY/'executor/vendor/simulator.cpp'),'-o',str(target)]
    start=time.perf_counter()
    with (HERE/f'compile_{label}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(status='BUILT_NEEDS_TRAJECTORY_CHECK' if r.returncode==0 else 'FAILED',lag=lag,unit_checks=7,command=cmd,sources=sources,seconds=time.perf_counter()-start)
    if target.exists():receipt['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (HERE/f'BUILD_{label}.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='sources'}),flush=True)
    if r.returncode:raise SystemExit(r.returncode)
