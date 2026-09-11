from pathlib import Path
import hashlib,json,subprocess,sys,time
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4r_prerequisites_v1';out.mkdir(exist_ok=False);steps=[]
def run(label,tool,args):
    cmd=[sys.executable,str(E/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+label,flush=True)
    with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=E,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-tic));(out/'progress.json').write_text(json.dumps(steps,indent=2));print(json.dumps(steps[-1]),flush=True)
    if r.returncode:raise RuntimeError(label+' failed; inspect preserved log')
tested=E/'receipts/s4r_next_day_mechanisms_v1/acceptance.json';r=json.loads(tested.read_text());assert r['status']=='PASS'
for rel,h in r['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
run('build','build_native.py',[])
for label,tool in [('day_value','test_day_consequence.py'),('day_scenario','test_observed_day_scenario.py'),('workforce','test_intraday_workforce.py'),('shared','test_shared_insertions.py'),('old','test_native_semantics.py'),('recovery','test_service_recovery.py'),('pickup','test_incremental_pickup.py')]:
    run(label,tool,['--out',f'receipts/s4r_{label}_mechanisms_v1'])
run('freeze','prepare_s4r_round.py',[])
cfg='profiles/s4r/configs.json';label='all_intraday_insert_value_next_public'
run('official_six','validate_s3k_build.py',['--round','s4r','--configs',cfg,'--label',label])
for opponent in ('g001','g003'):
    run(opponent+'_official','check_official_native.py',['--configs',cfg,'--label',label,'--seed',20261401,'--count',2,'--seats','0,1','--opponent',opponent,'--out',f'receipts/{opponent}_s4r_official_v1'])
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_PREREQUISITES_NOT_STRENGTH',build=json.loads((E/'native/build/build_receipt.json').read_text()),steps=steps,new_mechanism_sha256=hashlib.sha256(tested.read_bytes()).hexdigest()),indent=2));print('PREREQUISITES_PASS',flush=True)
