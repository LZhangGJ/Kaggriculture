"""Paired full-game S3R evaluation. Failures and old certificates are retained."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=EXP/'receipts/s3r_execution_v1';out.mkdir(exist_ok=False)
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    checked=json.loads((EXP/'receipts/s3r_build_validation_v1/acceptance.json').read_text())
    assert checked['status']=='PASS' and checked['build']['binary_sha256']==build['binary_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    configs=EXP/'profiles/s3r/configs.json';progress=[]
    def run(name,tool,args):
        cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter()
        print('START '+name,flush=True)
        with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        progress.append(dict(name=name,command=cmd,seconds=time.perf_counter()-tic,returncode=r.returncode))
        (out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress[-1]),flush=True)
        if r.returncode:raise RuntimeError(name+' failed; inspect retained log')
    for opponent in ('g001','g003'):
        run(opponent+'_official','check_official_native.py',[
            '--configs',configs,'--label','bundle_prep','--opponent',opponent,
            '--seed',20261401,'--count',2,'--seats','0,1',
            '--out',EXP/f'receipts/{opponent}_s3r_official_v1','--save-replays'])
    run('prep_only_official','check_official_native.py',[
        '--configs',configs,'--label','prep_only','--opponent','g001',
        '--seed',20261401,'--count',1,'--seats','0,1',
        '--out',EXP/'receipts/prep_only_s3r_official_v1'])
    registry=EXP/'opponents/registry.json';(out/'registry_before.json').write_bytes(registry.read_bytes())
    entries=json.loads(registry.read_text())
    for name,key in [('boatlee','boatlee_v29'),('kaito','kaito_v58'),('lynn','lynn_v5'),
                     ('fieldbook','yhay81_six_day'),('three_day','yhay81_three_day'),('ecobot','ecobot_v7')]:
        for kind,field in [('official','initial_parity_receipt'),('isolation','isolation_receipt')]:
            receipt=f'receipts/{name}_s3r_{kind}_v1/acceptance.json';check=json.loads((EXP/receipt).read_text())
            assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
            entry=entries['opponents'][key];entry['pre_s3r_'+field]=entry[field];entry[field]=receipt
    registry.write_text(json.dumps(entries,indent=2),encoding='utf8')
    panels=[]
    for batch,seed in [('A',20261401),('B',20261501)]:
        dest=EXP/f'receipts/s3r_prep_{batch}50_v1'
        run('panel_'+batch,'run_opponent_panel.py',[
            '--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
            '--configs',configs,'--labels','old,prep_only,bundle_only,bundle_prep',
            '--seed',seed,'--count',50,'--threads',16,'--out',dest])
        panels.append(dest/'results.json')
    for baseline,prior in [('old','old'),('bundle_only','funded_all')]:
        run('summary_'+baseline,'summarize_paired_panels.py',[
            '--panels',*panels,'--baseline',baseline,'--out',EXP/f'receipts/s3r_{baseline}_summary_v1',
            '--prior-baseline-panels',EXP/'receipts/s3q_bundle_A50_v1/results.json',
            EXP/'receipts/s3q_bundle_B50_v1/results.json','--prior-baseline-label',prior])
    run('cash_production_audit','run_pool_audit.py',[
        '--panel',panels[0],'--labels','old,prep_only,bundle_only,bundle_prep',
        '--out',EXP/'receipts/s3r_prep_audit_A50_v1'])
    (out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE',
        build=build,steps=progress,config_sha256=sha(configs),final_holdout_used=False,
        new_H_used=False,promotion='NOT_AUTOMATIC'),indent=2))
    print((EXP/'receipts/s3r_old_summary_v1/TABLES_ZH.md').read_text(),flush=True)

if __name__=='__main__':main()
