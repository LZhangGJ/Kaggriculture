"""Frozen 2x2 idle-capacity experiment; no opponent identity enters policy."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=EXP/'receipts/s3v_execution_v1';out.mkdir(exist_ok=False)
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    for rel in ('receipts/s3v_build_validation_v1/acceptance.json','receipts/s3v_mechanisms_v1/acceptance.json'):
        c=json.loads((EXP/rel).read_text());assert c['status']=='PASS' and c['build']['binary_sha256']==build['binary_sha256']
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    cfg=EXP/'profiles/s3v/configs.json';assert sha(cfg)==json.loads((EXP/'profiles/s3v/freeze.json').read_text())['config_sha256']
    progress=[]
    def run(name,tool,args):
        cmd=[sys.executable,str(EXP/'tools'/tool),*map(str,args)];tic=time.perf_counter();print('START '+name,flush=True)
        with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        progress.append(dict(name=name,command=cmd,seconds=time.perf_counter()-tic,returncode=r.returncode));(out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress[-1]),flush=True)
        if r.returncode:raise RuntimeError(name+' failed; logs retained')
    for label in ('idle_only','compile_idle'):
        for opponent in ('g001','g003'):
            run(label+'_'+opponent+'_official','check_official_native.py',[
                '--configs',cfg,'--label',label,'--opponent',opponent,'--seed',20261401,'--count',1,'--seats','0,1',
                '--out',EXP/f'receipts/{label}_{opponent}_s3v_official_v1','--save-replays'])
    registry=EXP/'opponents/registry.json';(out/'registry_before.json').write_bytes(registry.read_bytes());entries=json.loads(registry.read_text())
    for name,key in [('boatlee','boatlee_v29'),('kaito','kaito_v58'),('lynn','lynn_v5'),('fieldbook','yhay81_six_day'),('three_day','yhay81_three_day'),('ecobot','ecobot_v7')]:
        for kind,field in [('official','initial_parity_receipt'),('isolation','isolation_receipt')]:
            path=f'receipts/{name}_s3v_{kind}_v1/acceptance.json';check=json.loads((EXP/path).read_text());assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
            e=entries['opponents'][key];e['pre_s3v_'+field]=e[field];e[field]=path
    registry.write_text(json.dumps(entries,indent=2),encoding='utf8')
    panels=[]
    for batch,seed in [('A',20261401),('B',20261501)]:
        dest=EXP/f'receipts/s3v_idle_{batch}50_v1'
        run('panel_'+batch,'run_opponent_panel.py',[
            '--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
            '--configs',cfg,'--labels','old,compile_only,idle_only,compile_idle','--seed',seed,'--count',50,'--threads',16,'--out',dest]);panels.append(dest/'results.json')
    counts={}
    for label,prior in [('old','terminal_only'),('compile_only','compile_terminal')]:
        count=0
        for batch,path in zip(('A','B'),panels):
            prev=json.loads((EXP/f'receipts/s3u_terminal_{batch}50_v1/results.json').read_text());refs={(r['opponent'],r['seed'],r['seat']):r for r in prev['rows'] if r['variant']==prior}
            for r in json.loads(path.read_text())['rows']:
                if r['variant']!=label:continue
                k=r['opponent'],r['seed'],r['seat'];assert all(r[x]==refs[k][x] for x in ('cash','opponent_cash','margin','win','overflow')),(label,k);count+=1
        counts[label]=dict(prior_label=prior,games=count)
    (out/'compatibility.json').write_text(json.dumps(dict(status='PASS',exact_game_counts=counts),indent=2))
    run('summary','summarize_paired_panels.py',['--panels',*panels,'--baseline','old','--out',EXP/'receipts/s3v_AB_summary_v1'])
    run('interaction','summarize_s3s_interaction.py',['--panels',*panels,'--baseline','old','--factor-a','compile_only','--factor-b','idle_only','--both','compile_idle','--out',EXP/'receipts/s3v_AB_interaction_v1'])
    run('cash_production_audit','run_pool_audit.py',['--panel',panels[0],'--labels','old,compile_only,idle_only,compile_idle','--out',EXP/'receipts/s3v_idle_audit_A50_v1'])
    (out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE',build=build,
        config_sha256=sha(cfg),steps=progress,compatibility=counts,final_holdout_used=False,new_L_used=False,promotion='NOT_AUTOMATIC'),indent=2))
    print((EXP/'receipts/s3v_AB_summary_v1/TABLES_ZH.md').read_text(),flush=True)
if __name__=='__main__':main()
