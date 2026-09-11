"""Run frozen workforce-vs-execution 2x2; retain all diagnostics and failures."""
from pathlib import Path
import hashlib, json, subprocess, sys, time

EXP = Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out = EXP / 'receipts/s3t_execution_v1'
    out.mkdir(exist_ok=False)
    build = json.loads((EXP / 'native/build/build_receipt.json').read_text())
    for path in ('receipts/s3t_build_validation_v1/acceptance.json', 'receipts/s3t_mechanisms_v1/acceptance.json'):
        checked = json.loads((EXP / path).read_text())
        assert checked['status'] == 'PASS' and checked['build']['binary_sha256'] == build['binary_sha256']
    for rel, digest in build['source_hashes'].items():
        assert sha(EXP / rel) == digest
    configs = EXP / 'profiles/s3t/configs.json'
    frozen = json.loads((EXP / 'profiles/s3t/freeze.json').read_text())
    assert sha(configs) == frozen['config_sha256']
    progress = []
    def run(name, tool, args):
        cmd = [sys.executable, str(EXP / 'tools' / tool), *map(str, args)]
        print('START ' + name, flush=True)
        tic = time.perf_counter()
        with (out / (name + '.log')).open('w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        progress.append(dict(name=name, command=cmd, seconds=time.perf_counter()-tic, returncode=r.returncode))
        (out / 'progress.json').write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress[-1]), flush=True)
        if r.returncode:
            raise RuntimeError(name + ' failed; inspect retained log')

    for label in ('compile_only', 'hire_only'):
        for opponent in ('g001', 'g003'):
            run(label+'_'+opponent+'_official', 'check_official_native.py', [
                '--configs', configs, '--label', label, '--opponent', opponent,
                '--seed', 20261401, '--count', 1, '--seats', '0,1',
                '--out', EXP/f'receipts/{label}_{opponent}_s3t_official_v1', '--save-replays'])

    registry = EXP / 'opponents/registry.json'
    (out / 'registry_before.json').write_bytes(registry.read_bytes())
    entries = json.loads(registry.read_text())
    for name,key in [('boatlee','boatlee_v29'),('kaito','kaito_v58'),('lynn','lynn_v5'),
                     ('fieldbook','yhay81_six_day'),('three_day','yhay81_three_day'),('ecobot','ecobot_v7')]:
        for kind,field in [('official','initial_parity_receipt'),('isolation','isolation_receipt')]:
            receipt = f'receipts/{name}_s3t_{kind}_v1/acceptance.json'
            check = json.loads((EXP / receipt).read_text())
            assert check['status']=='PASS' and check['build']['binary_sha256']==build['binary_sha256']
            entry=entries['opponents'][key]
            entry['pre_s3t_'+field]=entry[field]
            entry[field]=receipt
    registry.write_text(json.dumps(entries,indent=2),encoding='utf8')

    panels=[]
    for batch,seed in [('A',20261401),('B',20261501)]:
        dest=EXP/f'receipts/s3t_workforce_{batch}50_v1'
        run('panel_'+batch,'run_opponent_panel.py',[
            '--opponents','pass,g001,g003,boatlee_v29,kaito_v58,lynn_v5,yhay81_six_day,yhay81_three_day,ecobot_v7',
            '--configs',configs,'--labels','old,compile_only,hire_only,compile_hire',
            '--seed',seed,'--count',50,'--threads',16,'--out',dest])
        panels.append(dest/'results.json')

    # Two endpoints must exactly reproduce frozen S3S, not merely similar means.
    keys=('cash','opponent_cash','margin','win','overflow')
    counts={}
    for current,previous in [('old','old'),('compile_hire','regret_only')]:
        compared=0
        for batch,panel_path in zip(('A','B'),panels):
            now=json.loads(panel_path.read_text())
            old=json.loads((EXP/f'receipts/s3s_shared_{batch}50_v1/results.json').read_text())
            refs={(r['opponent'],r['seed'],r['seat']):r for r in old['rows'] if r['variant']==previous}
            for r in now['rows']:
                if r['variant']!=current:continue
                key=(r['opponent'],r['seed'],r['seat'])
                assert tuple(r[x] for x in keys)==tuple(refs[key][x] for x in keys),(current,key)
                compared+=1
        counts[current+'='+previous]=compared
    (out/'compatibility.json').write_text(json.dumps(dict(status='PASS',exact_game_counts=counts),indent=2))

    run('summary','summarize_paired_panels.py',[
        '--panels',*panels,'--baseline','old','--out',EXP/'receipts/s3t_AB_summary_v1',
        '--prior-baseline-panels',EXP/'receipts/s3s_shared_A50_v1/results.json',
        EXP/'receipts/s3s_shared_B50_v1/results.json','--prior-baseline-label','old'])
    run('interaction','summarize_s3s_interaction.py',[
        '--panels',*panels,'--baseline','old','--factor-a','compile_only','--factor-b','hire_only',
        '--both','compile_hire','--out',EXP/'receipts/s3t_AB_interaction_v1'])
    run('cash_production_audit','run_pool_audit.py',[
        '--panel',panels[0],'--labels','old,compile_only,hire_only,compile_hire',
        '--out',EXP/'receipts/s3t_workforce_audit_A50_v1'])
    run('diagnostics','audit_s3s_shared_services.py',[
        '--audit',EXP/'receipts/s3t_workforce_audit_A50_v1',
        '--out',EXP/'receipts/s3t_workforce_diagnostics_v1'])
    (out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE',
        build=build,steps=progress,config_sha256=sha(configs),compatibility=counts,
        final_holdout_used=False,new_J_used=False,promotion='NOT_AUTOMATIC'),indent=2))
    print((EXP/'receipts/s3t_AB_summary_v1/TABLES_ZH.md').read_text(),flush=True)

if __name__=='__main__':
    main()
