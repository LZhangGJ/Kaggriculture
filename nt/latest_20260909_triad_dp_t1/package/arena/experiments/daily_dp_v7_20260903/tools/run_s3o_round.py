"""Run fixed S3O development comparisons after native/parity validation.

Sequential jobs; each batch uses at most 16 native threads. Existing receipts
are reused only after a build/config check. No Kaggle, Git or final holdout.
"""
from pathlib import Path
import hashlib,json,subprocess,sys,time

EXP=Path(__file__).resolve().parents[1]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=EXP/'receipts/s3o_execution_v1';out.mkdir(exist_ok=False)
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    validated=json.loads((EXP/'receipts/s3o_build_validation_v1/acceptance.json').read_text())
    assert validated['status']=='PASS' and validated['build']['binary_sha256']==build['binary_sha256']
    for rel,h in build['source_hashes'].items(): assert sha(EXP/rel)==h
    configs=EXP/'profiles/s3o/configs.json';progress=[]
    def run(name,tool,args):
        started=time.perf_counter();cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)]
        print('START '+name,flush=True)
        with (out/(name+'.log')).open('w') as f:
            completed=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        progress.append(dict(name=name,command=cmd,seconds=time.perf_counter()-started,returncode=completed.returncode))
        (out/'progress.json').write_text(json.dumps(progress,indent=2))
        print(json.dumps(progress[-1]),flush=True)
        if completed.returncode: raise RuntimeError(name+' failed; inspect retained log')
    for opponent in ('g001','g003'):
        run(opponent+'_official','check_official_native.py',[
            '--configs',configs,'--label','portfolio_cash','--opponent',opponent,
            '--seed',20261401,'--count',2,'--seats','0,1',
            '--out',EXP/f'receipts/{opponent}_s3o_official_v1'])
    run('local_cash_official','check_official_native.py',[
        '--configs',configs,'--label','local_cash','--opponent','g001',
        '--seed',20261401,'--count',1,'--seats','0,1',
        '--out',EXP/'receipts/local_cash_s3o_official_v1'])
    registry=EXP/'opponents/registry.json'
    (out/'registry_before.json').write_bytes(registry.read_bytes())
    entries=json.loads(registry.read_text())
    for name,key in [('boatlee','boatlee_v29'),('kaito','kaito_v58'),('lynn','lynn_v5'),
                     ('fieldbook','yhay81_six_day'),('three_day','yhay81_three_day'),('ecobot','ecobot_v7')]:
        entry=entries['opponents'][key]
        for kind,field in [('official','initial_parity_receipt'),('isolation','isolation_receipt')]:
            receipt=f'receipts/{name}_s3o_{kind}_v1/acceptance.json'
            check=json.loads((EXP/receipt).read_text())
            assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
            entry['pre_s3o_'+field]=entry[field];entry[field]=receipt
    # Mechanical receipt-pointer advancement, not opponent policy modification.
    registry.write_text(json.dumps(entries,indent=2),encoding='utf8')
    panels=[]
    for label,seed in [('A',20261401),('B',20261501),('E',20261801)]:
        dest=EXP/f'receipts/s3o_cash_{label}50_v1'
        run('panel_'+label,'run_opponent_panel.py',[
            '--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
            '--configs',configs,'--labels','old,local_cash,portfolio_cash',
            '--seed',seed,'--count',50,'--threads',16,'--out',dest])
        panels.append(dest/'results.json')
    run('summary','summarize_paired_panels.py',[
        '--panels',*panels,'--baseline','old','--out',EXP/'receipts/s3o_cash_summary_v1',
        '--prior-baseline-panels',EXP/'receipts/s3n_expiry_A50_v1/results.json',
        EXP/'receipts/s3n_expiry_B50_v1/results.json','--prior-baseline-label','expiry_plan'])
    run('fresh_summary','summarize_paired_panels.py',[
        '--panels',panels[-1],'--baseline','old','--out',EXP/'receipts/s3o_fresh_E_summary_v1'])
    # Reuse matched A conditions, no new scenario count is claimed by auditing.
    run('cash_production_audit','run_pool_audit.py',[
        '--panel',panels[0],'--labels','old,local_cash,portfolio_cash',
        '--out',EXP/'receipts/s3o_cash_audit_A50_v1'])
    (out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE',
        build=build,steps=progress,config_sha256=sha(configs),final_holdout_used=False,
        promotion='NOT_AUTOMATIC'),indent=2))
    print((EXP/'receipts/s3o_cash_summary_v1/TABLES_ZH.md').read_text(),flush=True)
    print((EXP/'receipts/s3o_fresh_E_summary_v1/TABLES_ZH.md').read_text(),flush=True)


if __name__=='__main__': main()
