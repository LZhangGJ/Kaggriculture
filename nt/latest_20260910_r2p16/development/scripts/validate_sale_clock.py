"""Frozen second-panel check; do not use the final 90-percent holdout yet."""
from pathlib import Path
import hashlib,json,statistics,subprocess,sys
import run_panel as panel

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'clock_validation32'
BINARY=HERE/'candidate_r2p8/policy/clock_coupled_market0.so'
EXPECTED='f93c388cf6199ddbb33faa0c7a96d8d14b2999b0e7bd2880fa5da6d349a0b0fc'
def main():
    assert panel.sha(BINARY)==EXPECTED
    assert panel.sha(panel.old.R2/'agent.so')=='877f196113692722edc8a5b3e30d2d3dd1d4700b0772c13f1e3f2ad7a9ac0ef2'
    # This independent development confirmation is not the final reserved set.
    protocol=dict(names=['original','clock'],seed_start=2609121000,seeds=32,seats=[0,1],
                  candidate_sha256=EXPECTED,pool_sha256=panel.sha(HERE/'pool_all11.json'),
                  rule_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),
                  boundary='Frozen before results; 32 new seeds; 11 original live opponents; no tuning inside this comparison. Final 2609130000..99 untouched.')
    if (OUT/'PROTOCOL.json').exists():assert json.loads((OUT/'PROTOCOL.json').read_text())==protocol
    else:panel.save(OUT/'PROTOCOL.json',protocol)
    for name in protocol['names']:
        folder=OUT/name
        if (folder/'RESULTS.json').exists():continue
        cmd=[sys.executable,str(HERE/'run_panel.py'),'--pool',str(HERE/'pool_all11.json'),
             '--output',str(folder),'--seed-start',str(protocol['seed_start']),'--seeds','32','--workers','16']
        if name=='clock':cmd+=['--binary',str(BINARY)]
        if not (folder/'PILOT.json').exists():subprocess.run(cmd+['--pilot'],check=True,cwd=ROOT)
        subprocess.run(cmd,check=True,cwd=ROOT)
    rows={name:json.loads((OUT/name/'rows.json').read_text()) for name in protocol['names']}
    old={(r['opponent'],r['seed'],r['opponent_seat']):r for r in rows['original']}
    paired=[]
    for r in rows['clock']:
        b=old[(r['opponent'],r['seed'],r['opponent_seat'])]
        paired.append(dict(opponent=r['opponent'],seed=r['seed'],opponent_seat=r['opponent_seat'],
                           old_win=b['r2_win'],new_win=r['r2_win'],cash_delta=r['r2_cash']-b['r2_cash'],
                           margin_delta=r['r2_margin']-b['r2_margin']))
    by={o:{name:panel.summarize([r for r in rs if r['opponent']==o]) for name,rs in rows.items()} for o in {r['opponent'] for r in rows['clock']}}
    result=dict(original=panel.summarize(rows['original']),clock=panel.summarize(rows['clock']),
                rescued=sum(not r['old_win'] and r['new_win'] for r in paired),lost_wins=sum(r['old_win'] and not r['new_win'] for r in paired),
                mean_cash_delta=statistics.mean(r['cash_delta'] for r in paired),mean_margin_delta=statistics.mean(r['margin_delta'] for r in paired),
                by_opponent=by,final_holdout_used=False)
    panel.save(OUT/'PAIRED.json',paired);panel.save(OUT/'RESULTS.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
