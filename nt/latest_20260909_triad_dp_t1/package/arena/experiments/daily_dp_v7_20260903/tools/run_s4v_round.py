"""Independent market/feed switches and their full-chain interaction, live only."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
E=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=E/'receipts/s4v_execution_v1';out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
profile=E/'profiles/s4v';profile.mkdir(exist_ok=False)
before=json.loads((E/'profiles/s4u/configs.json').read_text());configs={}
for base in ('all_intraday','all_intraday_insert','full_chain_autonomous'):
    for suffix,market,feed in (('',False,False),('_market',True,False),('_feed',False,True),('_both',True,True)):
        configs[base+suffix]=dict(before[base],continuous_market_execution=market,insert_missing_feed=feed)
cfg=profile/'configs.json';cfg.write_text(json.dumps(configs,indent=2))
for rel in build['source_hashes']:
    dest=profile/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/rel,dest)
(profile/'freeze.json').write_text(json.dumps(dict(build=build,configs_sha256=sha(cfg),pre_register_sha256=sha(E/'reports/S4V_MARKET_AND_FEED_INSERTION_PRE_REGISTER_ZH.md'),development=[20262701,20262750],holdout_used=False),indent=2))
state=dict(status='RUNNING',build=build,steps=[],holdout_used=False,final_goal_acceptance=False);tic=time.perf_counter()
def run(label,tool,args):
    command=[sys.executable,str(E/'tools'/tool),*map(str,args)];state['active_step']=label
    (out/'progress.json').write_text(json.dumps(state,indent=2));print('START '+label,flush=True);start=time.perf_counter()
    with (out/(label+'.log')).open('w') as log:r=subprocess.run(command,cwd=E,stdout=log,stderr=subprocess.STDOUT)
    row=dict(label=label,returncode=r.returncode,seconds=time.perf_counter()-start,command=command);state['steps'].append(row);print(json.dumps(row),flush=True)
    (out/'progress.json').write_text(json.dumps(state,indent=2))
    if r.returncode:
        state['status']='FAILED_PRESERVED';(out/'failure.json').write_text(json.dumps(state,indent=2));raise RuntimeError(label)
for label,tool in (('old_mechanisms','test_native_semantics.py'),('shared_mechanisms','test_shared_insertions.py'),('pickup_mechanisms','test_incremental_pickup.py'),('handoff_mechanisms','test_idle_handoff.py')):
    run(label,tool,['--out',f'receipts/s4v_{label}_v1'])
run('escape_regression','check_s4v_escapes.py',[])
run('runtime','probe_s4k_runtime.py',['--round','s4v'])
probe=json.loads((E/'native/eco7_probe_build/build_receipt.json').read_text())
if any(sha(E/rel)!=h for rel,h in probe['source_hashes'].items()):run('eco_probe','build_ecobot_probe.py',[])
run('official_six','validate_s3k_build.py',['--round','s4v','--configs',cfg,'--label','full_chain_autonomous_both'])
for opp in ('g001','g003'):
    run(opp+'_official','check_official_native.py',['--configs',cfg,'--label','full_chain_autonomous_both','--seed',20261401,'--count',2,'--seats','0,1','--opponent',opp,'--out',E/f'receipts/{opp}_s4v_official_v1'])
registry_path=E/'opponents/registry.json';shutil.copy2(registry_path,out/'registry_before.json');registry=json.loads(registry_path.read_text())
for key,prefix in {'boatlee_v29':'boatlee','kaito_v58':'kaito','lynn_v5':'lynn','yhay81_six_day':'fieldbook','yhay81_three_day':'three_day','ecobot_v7':'ecobot'}.items():
    entry=registry['opponents'][key]
    for field,suffix in (('initial_parity_receipt','official'),('isolation_receipt','isolation')):
        rel=f'receipts/{prefix}_s4v_{suffix}_v1/acceptance.json';r=json.loads((E/rel).read_text());assert r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256']
        entry['pre_s4v_'+field]=entry[field];entry[field]=rel
registry_path.write_text(json.dumps(registry,indent=2,ensure_ascii=False)+'\n');shutil.copy2(registry_path,out/'registry_after.json')
run('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7','--configs',cfg,'--labels',','.join(configs),'--seed',20262701,'--count',50,'--threads',16,'--out',E/'receipts/s4v_twelveway_N50_v1'])
run('comparison','summarize_s4v.py',[])
changed=[label for label in configs if label not in before]
run('ledger','run_pool_audit.py',['--panel',E/'receipts/s4v_twelveway_N50_v1/results.json','--labels',','.join(changed),'--out',E/'receipts/s4v_pool_audit_N50_v1'])
state.update(status='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE',active_step=None,seconds=time.perf_counter()-tic)
(out/'acceptance.json').write_text(json.dumps(state,indent=2));print(state['status'],flush=True)
