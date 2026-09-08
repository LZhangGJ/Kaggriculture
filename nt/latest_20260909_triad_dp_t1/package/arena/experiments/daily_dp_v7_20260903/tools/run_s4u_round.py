"""Close the interrupted full-chain experiment with exact optimized execution."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
E=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=E/'receipts/s4u_execution_v1';out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text());cfg=E/'profiles/s4u/configs.json'
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
proof=E/'receipts/s4u_production_equivalence_v1/acceptance.json'
p=json.loads(proof.read_text());assert p['status']=='PASS_EXACT_PRODUCTION_EQUIVALENCE' and p['build']==build and p['full_game_pairs']==112
for rel in build['source_hashes']:
    dest=E/'profiles/s4u/source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/rel,dest)
freeze=E/'profiles/s4u/freeze.json';assert not freeze.exists()
freeze.write_text(json.dumps(dict(status='FROZEN_BEFORE_FULL_POOL',build=build,configs_sha256=sha(cfg),
    equivalence_sha256=sha(proof),pre_register_sha256=sha(E/'reports/S4U_EXACT_SCHEDULE_REUSE_PRE_REGISTER_ZH.md'),
    development=[20262701,20262750],holdout_used=False),indent=2))
steps=[];state=dict(status='RUNNING',steps=steps,build=build,holdout_used=False,final_goal_acceptance=False);tic=time.perf_counter()
def run(label,tool,args):
    command=[sys.executable,str(E/'tools'/tool),*map(str,args)];state['active_step']=label
    (out/'progress.json').write_text(json.dumps(state,indent=2));print('START '+label,flush=True);t=time.perf_counter()
    with (out/(label+'.log')).open('w') as f:r=subprocess.run(command,cwd=E,stdout=f,stderr=subprocess.STDOUT)
    row=dict(label=label,returncode=r.returncode,seconds=time.perf_counter()-t,command=command);steps.append(row)
    (out/'progress.json').write_text(json.dumps(state,indent=2));print(json.dumps(row),flush=True)
    if r.returncode:
        state['status']='FAILED_PRESERVED';(out/'failure.json').write_text(json.dumps(state,indent=2));raise RuntimeError(label)
probe=json.loads((E/'native/eco7_probe_build/build_receipt.json').read_text())
if any(sha(E/rel)!=h for rel,h in probe['source_hashes'].items()):run('eco_probe','build_ecobot_probe.py',[])
run('official_six','validate_s3k_build.py',['--round','s4u','--configs',cfg,'--label','full_chain_autonomous'])
for opp in ('g001','g003'):
    run(opp+'_official','check_official_native.py',['--configs',cfg,'--label','full_chain_autonomous','--seed',20261401,'--count',2,'--seats','0,1','--opponent',opp,'--out',E/f'receipts/{opp}_s4u_official_v1'])
# Mechanical manifest advancement, gated by current-build parity proofs.
registry_path=E/'opponents/registry.json';shutil.copy2(registry_path,out/'registry_before.json')
registry=json.loads(registry_path.read_text())
for key,prefix in {'boatlee_v29':'boatlee','kaito_v58':'kaito','lynn_v5':'lynn','yhay81_six_day':'fieldbook','yhay81_three_day':'three_day','ecobot_v7':'ecobot'}.items():
    entry=registry['opponents'][key]
    for field,suffix in (('initial_parity_receipt','official'),('isolation_receipt','isolation')):
        rel=f'receipts/{prefix}_s4u_{suffix}_v1/acceptance.json';r=json.loads((E/rel).read_text())
        assert r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256']
        entry['pre_s4u_'+field]=entry[field];entry[field]=rel
registry_path.write_text(json.dumps(registry,indent=2,ensure_ascii=False)+'\n',encoding='utf8');shutil.copy2(registry_path,out/'registry_after.json')
run('runtime','probe_s4k_runtime.py',['--round','s4u'])
run('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7','--configs',cfg,'--labels',','.join(json.loads(cfg.read_text())),'--seed',20262701,'--count',50,'--threads',16,'--out',E/'receipts/s4u_fiveway_N50_v1'])
run('comparison','summarize_s4u.py',[])
run('full_ledger','run_pool_audit.py',['--panel',E/'receipts/s4u_fiveway_N50_v1/results.json','--labels','full_workers25,full_chain,full_chain_autonomous','--out',E/'receipts/s4u_pool_audit_N50_v1'])
run('channels','summarize_s4t_channels.py',['--round','s4u'])
state.update(status='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE',active_step=None,seconds=time.perf_counter()-tic)
(out/'acceptance.json').write_text(json.dumps(state,indent=2));print(state['status'],flush=True)
