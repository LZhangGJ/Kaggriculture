from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,time
cli=argparse.ArgumentParser();cli.add_argument('--version',default='v2');args=cli.parse_args()
E=Path(__file__).resolve().parents[1];out=E/('receipts/s5b_execution_'+args.version)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
runtime=json.loads((E/('receipts/s5b_runtime_'+args.version+'/acceptance.json')).read_text());assert runtime['status']=='COMPLETE_NATIVE_PROBE' and runtime['build']==build
out.mkdir(exist_ok=False)
cfg=E/'profiles/s5b/configs.json';configs=json.loads(cfg.read_text())
state=dict(status='RUNNING',build=build,steps=[],holdout_used=False);tic=time.perf_counter()
def run(label,tool,args):
 command=[sys.executable,str(E/'tools'/tool),*map(str,args)];state['active_step']=label
 (out/'progress.json').write_text(json.dumps(state,indent=2));print('START '+label,flush=True);start=time.perf_counter()
 with (out/(label+'.log')).open('w') as log:r=subprocess.run(command,cwd=E,stdout=log,stderr=subprocess.STDOUT)
 row=dict(label=label,returncode=r.returncode,seconds=time.perf_counter()-start,command=command);state['steps'].append(row);print(json.dumps(row),flush=True)
 if r.returncode:
  state['status']='FAILED_PRESERVED';(out/'failure.json').write_text(json.dumps(state,indent=2));raise RuntimeError(label)
for label,tool in [('old_mechanisms','test_native_semantics.py'),('shared_mechanisms','test_shared_insertions.py'),('pickup_mechanisms','test_incremental_pickup.py'),('handoff_mechanisms','test_idle_handoff.py')]:
 run(label,tool,['--out',f'receipts/s5b_{label}_v1'])
probe=json.loads((E/'native/eco7_probe_build/build_receipt.json').read_text())
if any(sha(E/rel)!=h for rel,h in probe['source_hashes'].items()):run('eco_probe','build_ecobot_probe.py',[])
run('official_six','validate_s3k_build.py',['--round','s5b','--configs',cfg,'--label','full_chain_autonomous_timing'])
for opp in ('g001','g003'):
 run(opp+'_official','check_official_native.py',['--configs',cfg,'--label','full_chain_autonomous_timing','--seed',20261401,'--count',2,'--seats','0,1','--opponent',opp,'--out',E/f'receipts/{opp}_s5b_official_v1'])
rp=E/'opponents/registry.json';shutil.copy2(rp,out/'registry_before.json');registry=json.loads(rp.read_text())
for key,prefix in {'boatlee_v29':'boatlee','kaito_v58':'kaito','lynn_v5':'lynn','yhay81_six_day':'fieldbook','yhay81_three_day':'three_day','ecobot_v7':'ecobot'}.items():
 for field,suffix in [('initial_parity_receipt','official'),('isolation_receipt','isolation')]:
  rel=f'receipts/{prefix}_s5b_{suffix}_v1/acceptance.json';r=json.loads((E/rel).read_text());assert r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256']
  entry=registry['opponents'][key];entry['pre_s5b_'+field]=entry[field];entry[field]=rel
rp.write_text(json.dumps(registry,indent=2,ensure_ascii=False)+'\n');shutil.copy2(rp,out/'registry_after.json')
run('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7','--configs',cfg,'--labels',','.join(configs),'--seed',20262701,'--count',10,'--threads',16,'--out',E/'receipts/s5b_eightway_N10_v1'])
run('ledger','run_pool_audit.py',['--panel',E/'receipts/s5b_eightway_N10_v1/results.json','--labels',','.join(x for x in configs if x.endswith(('_net','_timing'))),'--out',E/'receipts/s5b_pool_audit_N10_v1'])
state.update(status='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE',active_step=None,seconds=time.perf_counter()-tic)
(out/'acceptance.json').write_text(json.dumps(state,indent=2));print(state['status'],flush=True)
