"""Frozen live panel, then repeated actual-effect audit; no hypothetical games."""
from pathlib import Path
import argparse,json,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--round',default='s4k');a=p.parse_args();r=a.round;assert r in ('s4k','s4k2')
EXP=Path(__file__).resolve().parents[1]
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
check=json.loads((EXP/f'receipts/{r}_build_validation_v1/acceptance.json').read_text())
assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
cfgpath=EXP/f'profiles/{r}/configs.json';cfg=json.loads(cfgpath.read_text())
out=EXP/f'receipts/{r}_execution_v1';out.mkdir(exist_ok=False);steps=[]
def run(label,tool,args):
    cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+label,flush=True)
    with (out/(label+'.log')).open('w') as log:result=subprocess.run(cmd,cwd=EXP,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,command=cmd,returncode=result.returncode,seconds=time.perf_counter()-tic))
    (out/'progress.json').write_text(json.dumps(steps,indent=2));print(json.dumps(steps[-1]),flush=True)
    if result.returncode:raise RuntimeError(label+' failed: see preserved log')
for opponent in ('g001','g003'):
    run(opponent+'_official','check_official_native.py',['--configs',cfgpath,'--label','all_intraday_auto_portfolio_calendar_finance',
        '--seed',20261401,'--count',2,'--seats','0,1','--opponent',opponent,'--out',f'receipts/{opponent}_{r}_official_v1'])
run('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
    '--configs',cfgpath,'--labels',','.join(cfg),'--seed',20262701,'--count',50,'--threads',16,'--out',f'receipts/{r}_eightway_N50_v1'])
run('interaction','summarize_s4k_interaction.py',['--round',r])
changed=[k for k in cfg if cfg[k]['recover_service_inputs']]
run('audit','run_pool_audit.py',['--panel',f'receipts/{r}_eightway_N50_v1/results.json','--labels',','.join(changed),'--out',f'receipts/{r}_pool_audit_N50_v1'])
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE',build=build,steps=steps),indent=2))
print('EXECUTION_COMPLETE_NOT_GOAL_ACCEPTANCE',flush=True)
