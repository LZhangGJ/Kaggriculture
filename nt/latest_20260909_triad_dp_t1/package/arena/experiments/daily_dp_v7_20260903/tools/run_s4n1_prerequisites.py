from pathlib import Path
import json,subprocess,sys,time
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4n1_prerequisites_v1';out.mkdir(exist_ok=False);steps=[]
def run(label,tool,args):
    cmd=[sys.executable,str(E/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+label,flush=True)
    with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=E,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-tic))
    (out/'progress.json').write_text(json.dumps(steps,indent=2));print(json.dumps(steps[-1]),flush=True)
    if r.returncode:raise RuntimeError(label+' failed: preserved log')
for label,tool in [('workforce','test_intraday_workforce.py'),('shared','test_shared_insertions.py'),('old','test_native_semantics.py'),('recovery','test_service_recovery.py'),('pickup','test_incremental_pickup.py')]:
    run(label,tool,['--out',f'receipts/s4n1_{label}_mechanisms_v1'])
run('freeze','prepare_s4n1_round.py',[])
cfg='profiles/s4n1/configs.json';label='all_intraday_insert_workforce'
run('official_six','validate_s3k_build.py',['--round','s4n1','--configs',cfg,'--label',label])
for opp in ('g001','g003'):
    run(opp+'_official','check_official_native.py',['--configs',cfg,'--label',label,'--seed',20261401,'--count',2,'--seats','0,1','--opponent',opp,'--out',f'receipts/{opp}_s4n1_official_v1'])
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_PREREQUISITES_NOT_STRENGTH',build=json.loads((E/'native/build/build_receipt.json').read_text()),steps=steps),indent=2))
print('PREREQUISITES_PASS',flush=True)
