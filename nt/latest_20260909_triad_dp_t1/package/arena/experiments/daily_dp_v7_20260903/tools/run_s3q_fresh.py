"""Use the previously registered G seeds after inspecting complete A/B only."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
out=EXP/'receipts/s3q_fresh_execution_v1';out.mkdir(exist_ok=False)
done=json.loads((EXP/'receipts/s3q_execution_v1/acceptance.json').read_text())
assert done['status']=='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE'
configs=EXP/'profiles/s3q/configs.json'
assert hashlib.sha256(configs.read_bytes()).hexdigest()==done['config_sha256']
reason='A/B show gains on Boatlee/Kaito with small mixed changes elsewhere; investigate replication, not promotion. PASS mean is lower. Both frozen variants kept; no parameters changed.'
(out/'decision_before_G.json').write_text(json.dumps(dict(reason=reason,seed=[20262001,20262050],
    configs_sha256=done['config_sha256'],final_holdout=False),indent=2))
dest=EXP/'receipts/s3q_bundle_G50_v1';progress=[]
jobs=[('panel','run_opponent_panel.py',[
    '--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
    '--configs',configs,'--labels','old,funded_repair,funded_all','--seed',20262001,'--count',50,'--threads',16,'--out',dest]),
    ('fresh_summary','summarize_paired_panels.py',['--panels',dest/'results.json','--baseline','old','--out',EXP/'receipts/s3q_G_summary_v1']),
    ('combined_summary','summarize_paired_panels.py',['--panels',EXP/'receipts/s3q_bundle_A50_v1/results.json',
     EXP/'receipts/s3q_bundle_B50_v1/results.json',dest/'results.json','--baseline','old','--out',EXP/'receipts/s3q_ABG_summary_v1',
     '--prior-baseline-panels',EXP/'receipts/s3p_cash_A50_v1/results.json',EXP/'receipts/s3p_cash_B50_v1/results.json','--prior-baseline-label','old']),
    ('diagnostics','summarize_bundle_diagnostics.py',['--audit',EXP/'receipts/s3q_bundle_audit_A50_v1','--out',EXP/'receipts/s3q_bundle_diagnostics_v1'])]
for name,tool,args in jobs:
    cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+name,flush=True)
    with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    progress.append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-tic))
    (out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress[-1]),flush=True)
    if r.returncode:raise RuntimeError(name+' failed; check retained log')
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_FRESH_DEVELOPMENT_NOT_GOAL_ACCEPTANCE',
    build=done['build'],steps=progress,reason=reason,final_holdout_used=False),indent=2))
print((EXP/'receipts/s3q_G_summary_v1/TABLES_ZH.md').read_text(),flush=True)
