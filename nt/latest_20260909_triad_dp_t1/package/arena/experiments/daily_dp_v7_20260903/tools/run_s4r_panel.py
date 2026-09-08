from pathlib import Path
import json,subprocess,sys,time
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4r_execution_v1';out.mkdir(exist_ok=False);steps=[]
check=json.loads((E/'receipts/s4r_prerequisites_v1/acceptance.json').read_text());build=json.loads((E/'native/build/build_receipt.json').read_text())
assert check['status']=='PASS_PREREQUISITES_NOT_STRENGTH' and check['build']['binary_sha256']==build['binary_sha256']
cfg=E/'profiles/s4r/configs.json';labels=list(json.loads(cfg.read_text()))
def run(label,tool,args):
    cmd=[sys.executable,str(E/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+label,flush=True)
    with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=E,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-tic));(out/'progress.json').write_text(json.dumps(steps,indent=2));print(json.dumps(steps[-1]),flush=True)
    if r.returncode:raise RuntimeError(label+' failed: inspect log')
run('runtime','probe_s4k_runtime.py',['--round','s4r'])
probe=json.loads((E/'receipts/s4r_runtime_probe_v1/acceptance.json').read_text());assert probe['hard_max_action_seconds']<1.0
run('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7','--configs',cfg,'--labels',','.join(labels),'--seed',20262701,'--count',50,'--threads',16,'--out',E/'receipts/s4r_eightway_N50_v1'])
run('summary','summarize_s4r.py',[])
run('full_ledger','run_pool_audit.py',['--panel',E/'receipts/s4r_eightway_N50_v1/results.json','--labels',','.join(x for x in labels if '_next' in x),'--out',E/'receipts/s4r_pool_audit_N50_v1'])
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE',build=build,steps=steps),indent=2));print('EXECUTION_COMPLETE_NOT_GOAL_ACCEPTANCE',flush=True)
