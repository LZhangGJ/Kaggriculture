"""Independent, pre-frozen confirmation of both P9 mechanisms."""
from pathlib import Path
import json,statistics,subprocess,sys
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'local_sale_validation32'
HASHES={1:'51a8e08c675eead688520d444f2cd1a1f7e50e342bfd5a8f4689b2d38472ba0c',2:'ec844b27e4d8faa79c23ddd52b1c9bb1541964c04281d157e63779e1d89f4c4c'}
def main():
    assert panel.sha(panel.old.R2/'agent.so')=='877f196113692722edc8a5b3e30d2d3dd1d4700b0772c13f1e3f2ad7a9ac0ef2'
    for i,h in HASHES.items():assert panel.sha(HERE/f'candidate_r2p9/policy/local_sale_{i}.so')==h
    # Require full development results; this prevents accidental concurrent panels.
    for i in HASHES:assert (HERE/f'local_sale_screen100/local_sale_{i}/local_sale_{i}/RESULTS.json').exists()
    protocol=dict(names=['original','mode1','mode2'],seed_start=2609122000,seeds=32,seats=[0,1],candidate_sha256={str(k):v for k,v in HASHES.items()},pool_sha256=panel.sha(HERE/'pool_all11.json'),rule_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),original_config_sha256=panel.sha(panel.old.R2/'config.json'),boundary='Frozen before confirmation outcomes; live original opponents; all candidates reported; final 100-seed holdout untouched')
    file=OUT/'PROTOCOL.json'
    if file.exists():assert json.loads(file.read_text())==protocol
    else:panel.save(file,protocol)
    for name in protocol['names']:
        folder=OUT/name
        if (folder/'RESULTS.json').exists():continue
        cmd=[sys.executable,str(HERE/'run_panel.py'),'--pool',str(HERE/'pool_all11.json'),'--output',str(folder),'--seed-start',str(protocol['seed_start']),'--seeds','32','--workers','16']
        if name!='original':cmd+=['--binary',str(HERE/f'candidate_r2p9/policy/local_sale_{name[-1]}.so')]
        if not (folder/'PILOT.json').exists():subprocess.run(cmd+['--pilot'],check=True,cwd=ROOT)
        subprocess.run(cmd,check=True,cwd=ROOT)
    rows={name:json.loads((OUT/name/'rows.json').read_text()) for name in protocol['names']};old={(r['opponent'],r['seed'],r['opponent_seat']):r for r in rows['original']}
    result=dict(original=panel.summarize(rows['original']),final_holdout_used=False,seed_start=protocol['seed_start'])
    for name in ('mode1','mode2'):
        paired=[]
        for r in rows[name]:
            b=old[(r['opponent'],r['seed'],r['opponent_seat'])]
            paired.append(dict(opponent=r['opponent'],seed=r['seed'],opponent_seat=r['opponent_seat'],old_win=b['r2_win'],new_win=r['r2_win'],cash_delta=r['r2_cash']-b['r2_cash'],margin_delta=r['r2_margin']-b['r2_margin']))
        result[name]=dict(overall=panel.summarize(rows[name]),by_opponent={o:panel.summarize([r for r in rows[name] if r['opponent']==o]) for o in {r['opponent'] for r in rows[name]}},rescued=sum(not r['old_win'] and r['new_win'] for r in paired),lost_wins=sum(r['old_win'] and not r['new_win'] for r in paired),mean_cash_delta=statistics.mean(r['cash_delta'] for r in paired),mean_margin_delta=statistics.mean(r['margin_delta'] for r in paired))
        panel.save(OUT/f'PAIRED_{name}.json',paired)
    panel.save(OUT/'RESULTS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
