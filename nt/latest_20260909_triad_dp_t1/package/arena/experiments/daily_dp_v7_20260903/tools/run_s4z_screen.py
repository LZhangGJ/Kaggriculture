"""Existing rotation capability: mechanisms then 2x2 live panel and ledgers."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,shutil
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4z_execution_v1';out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
profile=E/'profiles/s4z';profile.mkdir(exist_ok=False)
old=json.loads((E/'profiles/s4v/configs.json').read_text());configs={}
for base in ('all_intraday_insert','full_chain_autonomous'):
 configs[base]=old[base];configs[base+'_rotation']=dict(old[base],rotate_finite=True)
 assert {k for k in configs[base+'_rotation'] if configs[base+'_rotation'][k]!=configs[base].get(k,False)}=={'rotate_finite'}
cfg=profile/'configs.json';cfg.write_text(json.dumps(configs,indent=2))
(profile/'freeze.json').write_text(json.dumps(dict(build=build,configs_sha256=sha(cfg),test_sha256=sha(E/'native/test_harvest_rotation.cpp'),pre_register_sha256=sha(E/'reports/S4Z_HARVEST_ROTATION_PRE_REGISTER_ZH.md'),development=[20262701,20262710],holdout_used=False),indent=2))
state=dict(status='RUNNING',steps=[],build=build);tic=time.perf_counter()
def run(label,cmd):
 state['active_step']=label;(out/'progress.json').write_text(json.dumps(state,indent=2));print('START '+label,flush=True);start=time.perf_counter()
 with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=E,stdout=log,stderr=subprocess.STDOUT)
 record=dict(label=label,returncode=r.returncode,seconds=time.perf_counter()-start,command=cmd);state['steps'].append(record);print(json.dumps(record),flush=True)
 if r.returncode:
  state['status']='FAILED_PRESERVED';(out/'failure.json').write_text(json.dumps(state,indent=2));raise RuntimeError(label)
def tool(label,file,args):run(label,[sys.executable,str(E/'tools'/file),*map(str,args)])
run('compile_mechanisms',['g++','-std=c++20','-O2','-I'+str(E/'native'),str(E/'native/test_harvest_rotation.cpp'),str(E/'native/vendor/simulator.cpp'),'-o',str(out/'mechanisms')])
run('mechanisms',[str(out/'mechanisms')])
tool('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7','--configs',cfg,'--labels',','.join(configs),'--seed',20262701,'--count',10,'--threads',16,'--out',E/'receipts/s4z_fourway_N10_v1'])
tool('ledger','run_pool_audit.py',['--panel',E/'receipts/s4z_fourway_N10_v1/results.json','--labels','all_intraday_insert_rotation,full_chain_autonomous_rotation','--out',E/'receipts/s4z_pool_audit_N10_v1'])
state.update(status='COMPLETE_SCREEN_NOT_GOAL_ACCEPTANCE',active_step=None,seconds=time.perf_counter()-tic)
(out/'acceptance.json').write_text(json.dumps(state,indent=2));print(state['status'],flush=True)
