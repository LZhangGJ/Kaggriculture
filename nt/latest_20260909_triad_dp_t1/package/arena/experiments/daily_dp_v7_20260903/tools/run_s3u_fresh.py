"""Unchanged K-seed test after A/B terminal capacity results; no fitting."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    prior=json.loads((EXP/'receipts/s3u_execution_v1/acceptance.json').read_text());assert prior['status']=='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE'
    checked=json.loads((EXP/'receipts/s3u_terminal_delivery_v1/summary.json').read_text());assert checked['status']=='PASS_TERMINAL_ONLY_INTERVENTION'
    build=json.loads((EXP/'native/build/build_receipt.json').read_text());assert build['binary_sha256']==prior['build']['binary_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    cfg=EXP/'profiles/s3u/configs.json';assert sha(cfg)==prior['config_sha256']
    out=EXP/'receipts/s3u_fresh_execution_v1';out.mkdir(exist_ok=False)
    decision=dict(registration_date_JST='2026-09-04',
        reason='A/B terminal-only improves PASS and several opponents without win-rate regression; combined arm recovers much of scheduling loss. Validate unchanged four arms, not just a selected positive batch.',
        reviewed_AB_summary_sha256=sha(EXP/'receipts/s3u_AB_summary_v1/summary.json'),
        reviewed_delivery_sha256=sha(EXP/'receipts/s3u_terminal_delivery_v1/summary.json'),
        configurations_sha256=sha(cfg),K=[20262401,20262450],refit=False,final_holdout_used=False)
    (out/'decision_before_K.json').write_text(json.dumps(decision,indent=2))
    progress=[]
    def run(name,tool,args):
        cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+name,flush=True)
        with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        progress.append(dict(name=name,command=cmd,seconds=time.perf_counter()-tic,returncode=r.returncode));(out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress[-1]),flush=True)
        if r.returncode:raise RuntimeError(name+' failed; logs retained')
    panel=EXP/'receipts/s3u_terminal_K50_v1/results.json'
    run('K_panel','run_opponent_panel.py',[
        '--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
        '--configs',cfg,'--labels','old,compile_only,terminal_only,compile_terminal',
        '--seed',20262401,'--count',50,'--threads',16,'--out',panel.parent])
    for name,panels in [('K',[panel]),('ABK',[EXP/'receipts/s3u_terminal_A50_v1/results.json',EXP/'receipts/s3u_terminal_B50_v1/results.json',panel])]:
        run(name+'_summary','summarize_paired_panels.py',['--panels',*panels,'--baseline','old','--out',EXP/f'receipts/s3u_{name}_summary_v1'])
        run(name+'_interaction','summarize_s3s_interaction.py',['--panels',*panels,'--baseline','old','--factor-a','compile_only','--factor-b','terminal_only','--both','compile_terminal','--out',EXP/f'receipts/s3u_{name}_interaction_v1'])
    (out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_UNCHANGED_FRESH_DEVELOPMENT',build=build,decision=decision,steps=progress,final_holdout_used=False,promotion='NOT_AUTOMATIC'),indent=2))
    print((EXP/'receipts/s3u_K_summary_v1/TABLES_ZH.md').read_text(),flush=True)
if __name__=='__main__':main()
