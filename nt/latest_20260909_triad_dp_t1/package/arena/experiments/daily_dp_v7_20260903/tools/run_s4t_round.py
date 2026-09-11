"""Frozen all-chain experiment; sequential jobs, <=16 simulator threads.

The initial native timing probe exceeds 1 s. Continue OFFLINE research only;
this receipt cannot promote a policy or claim online timeout acceptance.
"""
from pathlib import Path
import hashlib,json,subprocess,sys,time

E=Path(__file__).resolve().parents[1]
out=E/'receipts/s4t_execution_v1';out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text())
freeze=json.loads((E/'profiles/s4t/freeze.json').read_text())
cfg=E/'profiles/s4t/configs.json'
assert freeze['build']['binary_sha256']==build['binary_sha256']
assert hashlib.sha256(cfg.read_bytes()).hexdigest()==freeze['configs_sha256']
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
runtime=json.loads((E/'receipts/s4t_runtime_probe_v1/acceptance.json').read_text())
assert runtime['games']==20 and all(r['steps']==719 for r in runtime['rows'])
steps=[];start=time.perf_counter()
state=dict(status='RUNNING_OFFLINE_FULL_CHAIN',build=build,offline_only=True,
    native_probe_under_one_second=runtime['hard_max_action_seconds']<1,
    native_probe_max_seconds=runtime['hard_max_action_seconds'],holdout_used=False,
    final_goal_acceptance=False,steps=steps)

def run(label,tool,args):
    cmd=[sys.executable,str(E/'tools'/tool),*map(str,args)]
    print('START '+label,flush=True);state['active_step']=label
    (out/'progress.json').write_text(json.dumps(state,indent=2))
    tick=time.perf_counter()
    with (out/(label+'.log')).open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=E)
    row=dict(label=label,command=cmd,returncode=result.returncode,seconds=time.perf_counter()-tick)
    steps.append(row);(out/'progress.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(row),flush=True)
    if result.returncode:
        state['status']='FAILED_PRESERVED';(out/'failure.json').write_text(json.dumps(state,indent=2))
        raise RuntimeError(label+' failed; inspect preserved log')

run('official_six','validate_s3k_build.py',['--round','s4t','--configs',cfg,'--label','full_chain_autonomous'])
for opp in ('g001','g003'):
    run(opp+'_official','check_official_native.py',['--configs',cfg,'--label','full_chain_autonomous',
        '--seed',20261401,'--count',2,'--seats','0,1','--opponent',opp,'--out',E/f'receipts/{opp}_s4t_official_v1'])
labels=list(json.loads(cfg.read_text()))
run('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
    '--configs',cfg,'--labels',','.join(labels),'--seed',20262701,'--count',50,'--threads',16,
    '--out',E/'receipts/s4t_fiveway_N50_v1'])
run('summary','summarize_s4t.py',[])
run('full_ledger','run_pool_audit.py',['--panel',E/'receipts/s4t_fiveway_N50_v1/results.json',
    '--labels','full_workers25,full_chain,full_chain_autonomous','--out',E/'receipts/s4t_pool_audit_N50_v1'])
state.update(status='COMPLETE_OFFLINE_FULL_CHAIN_NOT_PROMOTION',active_step=None,seconds=time.perf_counter()-start)
(out/'acceptance.json').write_text(json.dumps(state,indent=2));print(state['status'],flush=True)
