"""Unchanged fresh development L after reviewing complete A/B evidence."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    accepted=json.loads((EXP/'receipts/s3v_execution_v1/acceptance.json').read_text());assert accepted['status']=='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE'
    audit=json.loads((EXP/'receipts/s3v_delivery_diagnostics_v1/summary.json').read_text());assert audit['status']=='COMPLETE_OFFLINE_ATTRIBUTION_NOT_CAUSAL_GUARANTEE'
    out=EXP/'receipts/s3v_fresh_execution_v1';out.mkdir(exist_ok=False)
    build=json.loads((EXP/'native/build/build_receipt.json').read_text());assert build['binary_sha256']==accepted['build']['binary_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    cfg=EXP/'profiles/s3v/configs.json';assert sha(cfg)==accepted['config_sha256']
    decision=dict(date_JST='2026-09-04',reason='Idle-only has tiny positive cash differences across all nine A/B conditions, mostly unchanged wins. Validate unchanged before retention; combined arm not assumed superior.',
        reviewed_AB_sha256=sha(EXP/'receipts/s3v_AB_summary_v1/summary.json'),reviewed_audit_sha256=sha(EXP/'receipts/s3v_delivery_diagnostics_v1/summary.json'),
        config_sha256=sha(cfg),L=[20262501,20262550],refit=False,final_holdout_used=False)
    (out/'decision_before_L.json').write_text(json.dumps(decision,indent=2));steps=[]
    def run(name,tool,args):
        cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];start=time.perf_counter();print('START '+name,flush=True)
        with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        steps.append(dict(name=name,command=cmd,seconds=time.perf_counter()-start,returncode=r.returncode));(out/'progress.json').write_text(json.dumps(steps,indent=2))
        if r.returncode:raise RuntimeError(name)
    panel=EXP/'receipts/s3v_idle_L50_v1'
    run('L_panel','run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
        '--configs',cfg,'--labels','old,compile_only,idle_only,compile_idle','--seed',20262501,'--count',50,'--threads',16,'--out',panel])
    for name,panels in [('L',[panel/'results.json']),('ABL',[EXP/f'receipts/s3v_idle_{batch}50_v1/results.json' for batch in ('A','B','L')])]:
        run(name+'_summary','summarize_paired_panels.py',['--panels',*panels,'--baseline','old','--out',EXP/f'receipts/s3v_{name}_summary_v1'])
        run(name+'_interaction','summarize_s3s_interaction.py',['--panels',*panels,'--baseline','old','--factor-a','compile_only','--factor-b','idle_only','--both','compile_idle','--out',EXP/f'receipts/s3v_{name}_interaction_v1'])
    (out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_UNCHANGED_FRESH_DEVELOPMENT',build=build,decision=decision,steps=steps,final_holdout_used=False,promotion='NOT_AUTOMATIC'),indent=2))
    print((EXP/'receipts/s3v_L_summary_v1/TABLES_ZH.md').read_text(),flush=True)
    print((EXP/'receipts/s3v_ABL_summary_v1/TABLES_ZH.md').read_text(),flush=True)
if __name__=='__main__':main()
