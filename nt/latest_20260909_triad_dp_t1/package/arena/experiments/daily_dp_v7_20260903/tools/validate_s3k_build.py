"""Sequential official/parity rechecks for the candidate-only S3K build.

Does not edit the opponent registry. All receipts must pass before its pointers
can be advanced. Historical failures and certificates are retained.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
cli=argparse.ArgumentParser();cli.add_argument('--attempt',type=int,default=1);cli.add_argument('--round',default='s3k')
cli.add_argument('--configs',default=str(EXP/'profiles/s3k/configs.json'));cli.add_argument('--label',default='replant_1');a=cli.parse_args()
assert a.round.replace('_','').isalnum()
tools=EXP/'tools';out=EXP/f'receipts/{a.round}_build_validation_v{a.attempt}';out.mkdir(exist_ok=False)
jobs=[]
for name in ('boatlee','kaito','lynn','fieldbook','three_day','ecobot'):
    args=['--configs',a.configs,'--label',a.label,
          '--seeds','20261401,20261404','--seats','0,1','--out',str(EXP/f'receipts/{name}_{a.round}_official_v1')]
    if name=='ecobot':args+=['--arena-adapter']
    jobs.append((f'{name}_official',f'check_{name}_native.py',args))
    jobs.append((f'{name}_isolation',f'test_{name}_isolation.py',['--out',str(EXP/f'receipts/{name}_{a.round}_isolation_v1')]))
results=[];started=time.perf_counter()
for name,tool,args in jobs:
    cmd=[sys.executable,str(tools/tool),*args];print('START '+name,flush=True)
    tic=time.perf_counter()
    receipt=Path(args[args.index('--out')+1])/'acceptance.json'
    if receipt.exists():
        check=json.loads(receipt.read_text());build=json.loads((EXP/'native/build/build_receipt.json').read_text())
        assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
        results.append(dict(name=name,command=cmd,returncode=0,reused_verified_receipt=str(receipt)))
        print('REUSE_PASSED '+name,flush=True);continue
    with (out/f'{name}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    item=dict(name=name,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-tic);results.append(item)
    (out/'progress.json').write_text(json.dumps(results,indent=2));print(json.dumps(item),flush=True)
    if r.returncode:raise RuntimeError(f'{name} failed: inspect preserved log')
    check=json.loads(receipt.read_text());assert check['status']=='PASS'
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS',build=build,results=results,seconds=time.perf_counter()-started,
    caveat='24 sampled official full games and isolation checks; not exhaustive equivalence or final Kaggle timeout acceptance'),indent=2))
print('ALL_PASS',flush=True)
