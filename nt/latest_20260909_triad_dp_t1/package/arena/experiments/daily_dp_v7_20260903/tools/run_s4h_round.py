"""Execute a frozen interaction experiment; no source change or Oracle."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
build=read(EXP/'native/build/build_receipt.json')
for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
assert sha(next((EXP/'native/build').glob('_dp7_native*.so')))==build['binary_sha256']
assert read(EXP/'receipts/s4g1_stage_acceptance_v1/acceptance.json')['goal_acceptance'] is False
out=EXP/'profiles/s4h1';out.mkdir(exist_ok=False);prior=read(EXP/'profiles/s4g1/configs.json');cfg={}
for context in ('intraday_funded','all_intraday'):
    auto=dict(prior[context+'_auto_calendar_funded']);assert auto['own_feed_demand_weight']==0
    cfg[context+'_fixed_cal']=dict(auto,autonomous_start=False)
    cfg[context+'_fixed_cal_feed']=dict(auto,autonomous_start=False,own_feed_demand_weight=1)
    cfg[context+'_auto_calendar_funded']=auto
    cfg[context+'_auto_cal_feed']=dict(auto,own_feed_demand_weight=1)
configs=out/'configs.json';configs.write_text(json.dumps(cfg,indent=2))
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',source_ref='profiles/s4g1/source',
    source_hashes=build['source_hashes'],binary_sha256=build['binary_sha256'],configs_sha256=sha(configs),
    pre_register_sha256=sha(EXP/'reports/S4H_STARTUP_FEED_INTERACTION_PRE_REGISTER_ZH.md'),
    development_N=[20262701,20262750],holdout_used=False,oracle_used=False),indent=2))
run=EXP/'receipts/s4h1_execution_v1';run.mkdir(exist_ok=False);steps=[]
def command(label,tool,args):
    cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];print('START '+label,flush=True);t=time.perf_counter()
    with (run/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=EXP,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-t))
    (run/'progress.json').write_text(json.dumps(steps,indent=2));print(json.dumps(steps[-1]),flush=True)
    if r.returncode:raise RuntimeError(label+' failed: preserved log')
command('official','check_official_native.py',['--configs',configs,'--label','all_intraday_auto_cal_feed',
    '--seed',20261401,'--count',2,'--seats','0,1','--opponent','g003','--out','receipts/g003_s4h1_official_v1'])
command('panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
    '--configs',configs,'--labels',','.join(cfg),'--seed',20262701,'--count',50,'--threads',16,'--out','receipts/s4h1_eightway_N50_v1'])
command('audit','run_pool_audit.py',['--panel','receipts/s4h1_eightway_N50_v1/results.json','--labels',','.join(cfg),'--out','receipts/s4h1_pool_audit_N50_v1'])
(run/'acceptance.json').write_text(json.dumps(dict(status='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE',steps=steps,build=build),indent=2))
print('EXECUTION_COMPLETE_NOT_GOAL_ACCEPTANCE',flush=True)
