"""Resume the verified terminal S4Q attempt; never overwrite its failure."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
E=Path(__file__).resolve().parents[1]
out=E/'receipts/s4q_prerequisites_v2';out.mkdir(exist_ok=False)
steps=[]
def run(label,tool,args):
    cmd=[sys.executable,str(E/'tools'/tool),*map(str,args)]
    print('START '+label,flush=True);tic=time.perf_counter()
    with (out/(label+'.log')).open('w') as log:
        r=subprocess.run(cmd,cwd=E,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-tic))
    (out/'progress.json').write_text(json.dumps(steps,indent=2))
    print(json.dumps(steps[-1]),flush=True)
    if r.returncode:raise RuntimeError(label+' failed; inspect preserved log')
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
run('rebuild_stale_ecobot_probe','build_ecobot_probe.py',[])
cfg='profiles/s4q/configs.json';label='all_intraday_insert_value_public'
run('ecobot_official','check_ecobot_native.py',['--configs',cfg,'--label',label,'--seeds','20261401,20261404','--seats','0,1','--arena-adapter','--out','receipts/ecobot_s4q_official_v2'])
run('ecobot_isolation','test_ecobot_isolation.py',['--out','receipts/ecobot_s4q_isolation_v1'])
for opponent in ('g001','g003'):
    run(opponent+'_official','check_official_native.py',['--configs',cfg,'--label',label,'--seed',20261401,'--count',2,'--seats','0,1','--opponent',opponent,'--out',f'receipts/{opponent}_s4q_official_v1'])
verified={}
for name in ('boatlee','kaito','lynn','fieldbook','three_day','ecobot','g001','g003'):
    kinds=('official','isolation') if name not in ('g001','g003') else ('official',)
    for kind in kinds:
        version=2 if name=='ecobot' and kind=='official' else 1
        path=E/f'receipts/{name}_s4q_{kind}_v{version}/acceptance.json'
        receipt=json.loads(path.read_text());assert receipt['status']=='PASS'
        assert receipt['build']['binary_sha256']==build['binary_sha256']
        verified[str(path.relative_to(E))]=hashlib.sha256(path.read_bytes()).hexdigest()
prior=json.loads((E/'receipts/s4q_prerequisites_v1/progress.json').read_text())
assert len(prior)==9 and all(x['returncode']==0 for x in prior[:8])
assert prior[-1]['returncode']==1
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_PREREQUISITES_NOT_STRENGTH',build=build,steps=steps,verified_receipts=verified,prior_passed_stages=prior[:8],preserved_failure='receipts/s4q_prerequisites_v1',failure_cause='Stale independent EcoBot probe source hash after simulator.hpp friend declaration; no failed game was reached.'),indent=2))
print('PREREQUISITES_PASS',flush=True)
