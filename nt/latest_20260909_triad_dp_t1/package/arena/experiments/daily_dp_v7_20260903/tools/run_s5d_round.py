"""Mechanisms -> official parity -> paired live panel -> repeat cash ledger."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, sys, time
E=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--runtime',default='v4');p.add_argument('--configs',default=str(E/'profiles/s5d_bounded_v2/configs.json'));a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
runtime=json.loads((E/f'receipts/s5d_runtime_{a.runtime}/acceptance.json').read_text())
assert runtime['status']=='COMPLETE_NATIVE_PROBE' and runtime['build']==build
assert max(r['max_act_seconds'] for r in runtime['rows'])<=1
out=E/'receipts/s5d_execution_v1';out.mkdir(exist_ok=False)
cfg=Path(a.configs);configs=json.loads(cfg.read_text())
state=dict(status='RUNNING',build=build,steps=[],holdout_used=False);started=time.perf_counter()
def run(label,tool,args):
 command=[sys.executable,str(E/'tools'/tool),*map(str,args)];state['active_step']=label
 (out/'progress.json').write_text(json.dumps(state,indent=2));print('START '+label,flush=True);tic=time.perf_counter()
 with (out/(label+'.log')).open('w') as log:r=subprocess.run(command,cwd=E,stdout=log,stderr=subprocess.STDOUT)
 row=dict(label=label,returncode=r.returncode,seconds=time.perf_counter()-tic,command=command);state['steps'].append(row);print(json.dumps(row),flush=True)
 if r.returncode:
  state['status']='FAILED_PRESERVED';(out/'failure.json').write_text(json.dumps(state,indent=2));raise RuntimeError(label)
for label,tool in [('old_mechanisms','test_native_semantics.py'),('shared_mechanisms','test_shared_insertions.py'),('pickup_mechanisms','test_incremental_pickup.py'),('handoff_mechanisms','test_idle_handoff.py')]:
 run(label,tool,['--out',f'receipts/s5d_{label}_v1'])
probe=json.loads((E/'native/eco7_probe_build/build_receipt.json').read_text())
if any(sha(E/rel)!=h for rel,h in probe['source_hashes'].items()):run('eco_probe','build_ecobot_probe.py',[])
label='full_chain_autonomous_timing_first_value_replant'
run('official_six','validate_s3k_build.py',['--round','s5d','--configs',cfg,'--label',label])
for opp in ('g001','g003'):
 run(opp+'_official','check_official_native.py',['--configs',cfg,'--label',label,'--seed',20261401,'--count',2,'--seats','0,1','--opponent',opp,'--out',E/f'receipts/{opp}_s5d_official_v1'])
rp=E/'opponents/registry.json';shutil.copy2(rp,out/'registry_before.json');registry=json.loads(rp.read_text())
for key,prefix in {'boatlee_v29':'boatlee','kaito_v58':'kaito','lynn_v5':'lynn','yhay81_six_day':'fieldbook','yhay81_three_day':'three_day','ecobot_v7':'ecobot'}.items():
 for field,suffix in [('initial_parity_receipt','official'),('isolation_receipt','isolation')]:
  rel=f'receipts/{prefix}_s5d_{suffix}_v1/acceptance.json';r=json.loads((E/rel).read_text());assert r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256']
  entry=registry['opponents'][key];entry['pre_s5d_'+field]=entry[field];entry[field]=rel
rp.write_text(json.dumps(registry,indent=2,ensure_ascii=False)+'\n');shutil.copy2(rp,out/'registry_after.json')
panel=E/'receipts/s5d_twelveway_N10_v1'
run('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7','--configs',cfg,'--labels',','.join(configs),'--seed',20262701,'--count',10,'--threads',16,'--out',panel])
# The optimization must not alter any of the 540 unchanged control games.
old=json.loads((E/'receipts/s5b_eightway_N10_v1/results.json').read_text());now=json.loads((panel/'results.json').read_text())
key=lambda r:(r['variant'],r['opponent'],r['seed'],r['seat']);refs={key(r):r for r in old['rows']};matched=0
for r in now['rows']:
 if r['variant'] not in old['configurations']:continue
 assert all(r[f]==refs[key(r)][f] for f in ('cash','opponent_cash','margin','win','overflow')),(key(r),r,refs[key(r)])
 matched+=1
assert matched==540
state['exact_control_games']=matched;(out/'progress.json').write_text(json.dumps(state,indent=2))
new=[x for x in configs if x not in old['configurations']]
run('ledger','run_pool_audit.py',['--panel',panel/'results.json','--labels',','.join(new),'--out',E/'receipts/s5d_pool_audit_N10_v1'])
state.update(status='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE',active_step=None,seconds=time.perf_counter()-started)
(out/'acceptance.json').write_text(json.dumps(state,indent=2));print(state['status'],flush=True)
