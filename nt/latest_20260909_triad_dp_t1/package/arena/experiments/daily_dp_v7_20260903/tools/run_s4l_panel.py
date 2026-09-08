"""Frozen live tests and actual-effect audit; never hypothetical continuations."""
from pathlib import Path
import json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
check=json.loads((EXP/'receipts/s4l_build_validation_v1/acceptance.json').read_text())
assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
cfgpath=EXP/'profiles/s4l/configs.json';cfg=json.loads(cfgpath.read_text())
out=EXP/'receipts/s4l_execution_v1';out.mkdir(exist_ok=False);steps=[]
def run(label,tool,args):
    cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+label,flush=True)
    with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=EXP,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-tic))
    (out/'progress.json').write_text(json.dumps(steps,indent=2));print(json.dumps(steps[-1]),flush=True)
    if r.returncode:raise RuntimeError(label+' failed: see preserved log')
for opp in ('g001','g003'):
    run(opp+'_official','check_official_native.py',['--configs',cfgpath,'--label','all_intraday_auto_portfolio_calendar_procure_causal_pickup',
        '--seed',20261401,'--count',2,'--seats','0,1','--opponent',opp,'--out',f'receipts/{opp}_s4l_official_v1'])
run('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
    '--configs',cfgpath,'--labels',','.join(cfg),'--seed',20262701,'--count',50,'--threads',16,'--out','receipts/s4l_eightway_N50_v1'])
run('interaction','summarize_s4l_interaction.py',[])
changed=[k for k in cfg if cfg[k]['incremental_pickup_repair']]
run('audit','run_pool_audit.py',['--panel','receipts/s4l_eightway_N50_v1/results.json','--labels',','.join(changed),'--out','receipts/s4l_pool_audit_N50_v1'])
run('runtime','probe_s4k_runtime.py',['--round','s4l'])
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE',build=build,steps=steps),indent=2))
print('EXECUTION_COMPLETE_NOT_GOAL_ACCEPTANCE',flush=True)
