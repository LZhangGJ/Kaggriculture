"""Run pre-registered A/B funded-bundle development tests, with provenance."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=EXP/'receipts/s3q_execution_v1';out.mkdir(exist_ok=False)
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    checked=json.loads((EXP/'receipts/s3q_build_validation_v1/acceptance.json').read_text())
    assert checked['status']=='PASS' and checked['build']['binary_sha256']==build['binary_sha256']
    for rel,h in build['source_hashes'].items(): assert sha(EXP/rel)==h
    configs=EXP/'profiles/s3q/configs.json';progress=[]
    def run(name,tool,args):
        cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter()
        print('START '+name,flush=True)
        with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        progress.append(dict(name=name,command=cmd,seconds=time.perf_counter()-tic,returncode=r.returncode))
        (out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress[-1]),flush=True)
        if r.returncode:raise RuntimeError(name+' failed; inspect retained log')
    for opponent in ('g001','g003'):
        run(opponent+'_official','check_official_native.py',[
            '--configs',configs,'--label','funded_all','--opponent',opponent,
            '--seed',20261401,'--count',2,'--seats','0,1',
            '--out',EXP/f'receipts/{opponent}_s3q_official_v1','--save-replays'])
    run('funded_repair_official','check_official_native.py',[
        '--configs',configs,'--label','funded_repair','--opponent','g001',
        '--seed',20261401,'--count',1,'--seats','0,1',
        '--out',EXP/'receipts/repair_s3q_official_v1'])
    registry=EXP/'opponents/registry.json';(out/'registry_before.json').write_bytes(registry.read_bytes())
    entries=json.loads(registry.read_text())
    for name,key in [('boatlee','boatlee_v29'),('kaito','kaito_v58'),('lynn','lynn_v5'),
                     ('fieldbook','yhay81_six_day'),('three_day','yhay81_three_day'),('ecobot','ecobot_v7')]:
        for kind,field in [('official','initial_parity_receipt'),('isolation','isolation_receipt')]:
            receipt=f'receipts/{name}_s3q_{kind}_v1/acceptance.json';check=json.loads((EXP/receipt).read_text())
            assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
            entry=entries['opponents'][key];entry['pre_s3q_'+field]=entry[field];entry[field]=receipt
    registry.write_text(json.dumps(entries,indent=2),encoding='utf8')
    panels=[]
    for batch,seed in [('A',20261401),('B',20261501)]:
        dest=EXP/f'receipts/s3q_bundle_{batch}50_v1'
        run('panel_'+batch,'run_opponent_panel.py',[
            '--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
            '--configs',configs,'--labels','old,funded_repair,funded_all',
            '--seed',seed,'--count',50,'--threads',16,'--out',dest])
        panels.append(dest/'results.json')
    run('summary','summarize_paired_panels.py',[
        '--panels',*panels,'--baseline','old','--out',EXP/'receipts/s3q_bundle_summary_v1',
        '--prior-baseline-panels',EXP/'receipts/s3p_cash_A50_v1/results.json',
        EXP/'receipts/s3p_cash_B50_v1/results.json','--prior-baseline-label','old'])
    run('cash_production_audit','run_pool_audit.py',[
        '--panel',panels[0],'--labels','old,funded_repair,funded_all',
        '--out',EXP/'receipts/s3q_bundle_audit_A50_v1'])
    (out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE',
        build=build,steps=progress,config_sha256=sha(configs),final_holdout_used=False,
        new_G_used=False,promotion='NOT_AUTOMATIC'),indent=2))
    print((EXP/'receipts/s3q_bundle_summary_v1/TABLES_ZH.md').read_text(),flush=True)

if __name__=='__main__':main()
