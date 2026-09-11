from pathlib import Path
import json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1];out=EXP/'receipts/s4m1_execution_v1';out.mkdir(exist_ok=False);steps=[]
check=json.loads((EXP/'receipts/s4m1_prerequisites_v1/acceptance.json').read_text())
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
assert check['status']=='PASS_PREREQUISITES_NOT_STRENGTH' and check['build']['binary_sha256']==build['binary_sha256']
cfgpath=EXP/'profiles/s4m1/configs.json';cfg=json.loads(cfgpath.read_text())
def run(label,tool,args):
    cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+label,flush=True)
    with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=EXP,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-tic))
    (out/'progress.json').write_text(json.dumps(steps,indent=2));print(json.dumps(steps[-1]),flush=True)
    if r.returncode:raise RuntimeError(label+' failed: preserved log')
run('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
    '--configs',cfgpath,'--labels',','.join(cfg),'--seed',20262701,'--count',50,'--threads',16,'--out','receipts/s4m1_eightway_N50_v1'])
run('interaction','summarize_s4m1_interaction.py',[])
labels=['no_intraday']+[k for k in cfg if cfg[k]['shared_service_insertions']]
run('audit','run_pool_audit.py',['--panel','receipts/s4m1_eightway_N50_v1/results.json','--labels',','.join(labels),'--out','receipts/s4m1_pool_audit_N50_v1'])
run('runtime','probe_s4k_runtime.py',['--round','s4m1'])
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE',build=build,steps=steps),indent=2))
print('EXECUTION_COMPLETE_NOT_GOAL_ACCEPTANCE',flush=True)
