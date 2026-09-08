"""Unchanged four-arm fresh-seed check after the frozen A/B results."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    prior=json.loads((EXP/'receipts/s3s_execution_v1/acceptance.json').read_text())
    assert prior['status']=='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE'
    out=EXP/'receipts/s3s_fresh_execution_v1';out.mkdir(exist_ok=False)
    build=json.loads((EXP/'native/build/build_receipt.json').read_text());assert build['binary_sha256']==prior['build']['binary_sha256']
    cfg=EXP/'profiles/s3s/configs.json';assert sha(cfg)==prior['config_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    decision=dict(reason='Split-only had small G001/Boatlee/Fieldbook development gains with nearly unchanged PASS cash; combined policy regressed. Check unchanged four-way outcomes and interaction, not tune to I.',
        reviewed_AB_summary_sha256=sha(EXP/'receipts/s3s_AB_summary_v1/summary.json'),
        reviewed_AB_interaction_sha256=sha(EXP/'receipts/s3s_AB_interaction_v1/summary.json'),
        configs_sha256=sha(cfg),I=[20262201,20262250],final_holdout_used=False,refit=False,
        registration_date_JST='2026-09-04',promotion='NONE_PENDING_INDEPENDENT_EVIDENCE')
    (out/'decision_before_I.json').write_text(json.dumps(decision,indent=2))
    progress=[]
    def run(name,tool,args):
        cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+name,flush=True)
        with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        progress.append(dict(name=name,command=cmd,seconds=time.perf_counter()-tic,returncode=r.returncode))
        (out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress[-1]),flush=True)
        if r.returncode:raise RuntimeError(name+' failed; logs retained')
    panel=EXP/'receipts/s3s_shared_I50_v1/results.json'
    run('I_panel','run_opponent_panel.py',[
        '--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
        '--configs',cfg,'--labels','old,split_only,regret_only,split_regret',
        '--seed',20262201,'--count',50,'--threads',16,'--out',panel.parent])
    for name,panels in [('I',[panel]),('ABI',[EXP/'receipts/s3s_shared_A50_v1/results.json',EXP/'receipts/s3s_shared_B50_v1/results.json',panel])]:
        run(name+'_summary','summarize_paired_panels.py',[
            '--panels',*panels,'--baseline','old','--out',EXP/f'receipts/s3s_{name}_summary_v1'])
        run(name+'_interaction','summarize_s3s_interaction.py',[
            '--panels',*panels,'--out',EXP/f'receipts/s3s_{name}_interaction_v1'])
    (out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_UNCHANGED_FRESH_DEVELOPMENT',build=build,
        decision=decision,steps=progress,final_holdout_used=False,promotion='NOT_AUTOMATIC'),indent=2))
    print((EXP/'receipts/s3s_I_summary_v1/TABLES_ZH.md').read_text(),flush=True)
if __name__=='__main__':main()
