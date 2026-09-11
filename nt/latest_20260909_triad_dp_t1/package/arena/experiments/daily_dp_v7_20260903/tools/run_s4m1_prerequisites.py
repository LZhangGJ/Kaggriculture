from pathlib import Path
import json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1];out=EXP/'receipts/s4m1_prerequisites_v1';out.mkdir(exist_ok=False);steps=[]
def run(label,tool,args):
    cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+label,flush=True)
    with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=EXP,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-tic))
    (out/'progress.json').write_text(json.dumps(steps,indent=2));print(json.dumps(steps[-1]),flush=True)
    if r.returncode:raise RuntimeError(label+' failed: preserved log')
for label,tool in [('shared','test_shared_insertions.py'),('old','test_native_semantics.py'),('recovery','test_service_recovery.py'),('pickup','test_incremental_pickup.py')]:
    run(label,tool,['--out',f'receipts/s4m1_{label}_mechanisms_v1'])
run('freeze','prepare_s4m1_round.py',[])
cfg='profiles/s4m1/configs.json';label='auto_portfolio_procure_insert'
run('official_six','validate_s3k_build.py',['--round','s4m1','--configs',cfg,'--label',label])
for opp in ('g001','g003'):
    run(opp+'_official','check_official_native.py',['--configs',cfg,'--label',label,'--seed',20261401,'--count',2,'--seats','0,1',
         '--opponent',opp,'--out',f'receipts/{opp}_s4m1_official_v1'])
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_PREREQUISITES_NOT_STRENGTH',build=json.loads((EXP/'native/build/build_receipt.json').read_text()),steps=steps),indent=2))
print('PREREQUISITES_PASS',flush=True)
