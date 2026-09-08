"""Complete frozen factorial round. Audit/validation receipts are not wins."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=EXP/'receipts/s3w_execution_v1';out.mkdir(exist_ok=False)
    build=json.loads((EXP/'native/build/build_receipt.json').read_text());cfg=EXP/'profiles/s3w/configs.json'
    freeze=json.loads((EXP/'profiles/s3w/freeze.json').read_text());assert sha(cfg)==freeze['config_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    for rel,h in freeze['source_hashes'].items():assert sha(EXP/rel)==h
    progress=[]
    def run(name,tool,args):
        cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];start=time.perf_counter();print('START '+name,flush=True)
        with (out/f'{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        progress.append(dict(name=name,command=cmd,seconds=time.perf_counter()-start,returncode=r.returncode));(out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress[-1]),flush=True)
        if r.returncode:raise RuntimeError(name+' failed; inspect log, no restart without diagnosis')
    run('mechanisms','test_native_semantics.py',['--out','receipts/s3w_mechanisms_v1'])
    run('six_ports','validate_s3k_build.py',['--round','s3w','--configs',cfg,'--label','calendar_turnover'])
    for label in ('turnover','calendar_turnover'):
        for opponent in ('g001','g003'):
            run(label+'_'+opponent,'check_official_native.py',['--configs',cfg,'--label',label,'--opponent',opponent,
                '--seed',20261401,'--count',1,'--seats','0,1','--out',EXP/f'receipts/{label}_{opponent}_s3w_official_v1','--save-replays'])
    registry=EXP/'opponents/registry.json';(out/'registry_before.json').write_bytes(registry.read_bytes());entries=json.loads(registry.read_text())
    for name,key in [('boatlee','boatlee_v29'),('kaito','kaito_v58'),('lynn','lynn_v5'),('fieldbook','yhay81_six_day'),('three_day','yhay81_three_day'),('ecobot','ecobot_v7')]:
        for kind,field in [('official','initial_parity_receipt'),('isolation','isolation_receipt')]:
            path=f'receipts/{name}_s3w_{kind}_v1/acceptance.json';c=json.loads((EXP/path).read_text());assert c['status']=='PASS' and c['build']['binary_sha256']==build['binary_sha256']
            e=entries['opponents'][key];e['pre_s3w_'+field]=e[field];e[field]=path
    registry.write_text(json.dumps(entries,indent=2),encoding='utf8')
    panels=[];compat=0
    for batch,seed in [('A',20261401),('B',20261501),('M',20262601)]:
        dest=EXP/f'receipts/s3w_capital_{batch}50_v1'
        run('panel_'+batch,'run_opponent_panel.py',['--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
            '--configs',cfg,'--labels','old,calendar,turnover,calendar_turnover','--seed',seed,'--count',50,'--threads',16,'--out',dest]);panels.append(dest/'results.json')
        if batch in ('A','B'):
            prior=json.loads((EXP/f'receipts/s3v_idle_{batch}50_v1/results.json').read_text());refs={(r['opponent'],r['seed'],r['seat']):r for r in prior['rows'] if r['variant']=='old'}
            for r in json.loads(panels[-1].read_text())['rows']:
                if r['variant']!='old':continue
                old=refs[r['opponent'],r['seed'],r['seat']];assert all(r[k]==old[k] for k in ('cash','opponent_cash','win','margin','overflow'));compat+=1
        (out/'compatibility.json').write_text(json.dumps(dict(status='PASS',unchanged_old_games=compat),indent=2))
    run('summary','summarize_paired_panels.py',['--panels',*panels,'--baseline','old','--out',EXP/'receipts/s3w_ABM_summary_v1'])
    run('fresh_summary','summarize_paired_panels.py',['--panels',panels[-1],'--baseline','old','--out',EXP/'receipts/s3w_M_summary_v1'])
    run('interaction','summarize_s3s_interaction.py',['--panels',*panels,'--baseline','old','--factor-a','calendar','--factor-b','turnover','--both','calendar_turnover','--out',EXP/'receipts/s3w_ABM_interaction_v1'])
    run('audit','run_pool_audit.py',['--panel',panels[0],'--labels','old,calendar,turnover,calendar_turnover','--out',EXP/'receipts/s3w_factorial_audit_A50_v1'])
    (out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE',build=build,
        config_sha256=sha(cfg),steps=progress,unchanged_old_games=compat,final_holdout_used=False,new_M_development=True),indent=2))
    print((EXP/'receipts/s3w_ABM_summary_v1/TABLES_ZH.md').read_text(),flush=True)
if __name__=='__main__':main()
