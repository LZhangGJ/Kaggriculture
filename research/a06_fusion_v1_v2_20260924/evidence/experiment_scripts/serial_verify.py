"""Sequentially replay predetermined opponents plus the slowest parallel case."""
from pathlib import Path
import json
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/local_dynamic4_vs_public_top5_20260923'))
from run_matchups import play


def main():
    frozen=json.loads((HERE/'FINAL_FREEZE.json').read_text())
    out=HERE/'runs'/frozen.get('run','holdout_v1')
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    summary=json.loads((out/'SUMMARY.json').read_text());assert summary['games']==summary['complete_719']==2300 and summary['errors']==0
    candidate=frozen['candidate']
    rows=[json.loads(s) for s in (out/'games.jsonl').read_text().splitlines() if s]
    first=protocol['seeds'][0]
    names=('internal/r14_cashflow','external/n14_prvsiyan','external/n69_hosen42','external/d24_02_guruprasaathas111')
    selected=[r for r in rows if r['seed']==first and r['public'] in names]
    slowest=max(rows,key=lambda r:r['max_local_action_s'])
    if not any((r['public'],r['seed'],r['local_seat'])==(slowest['public'],slowest['seed'],slowest['local_seat']) for r in selected):selected.append(slowest)
    pool={o['id']:o for o in protocol['opponents']};checks=[]
    for r in selected:
        job=(candidate,str(HERE/'candidates'/candidate/'main.py'),r['public'],pool[r['public']]['entry'],r['seed'],r['local_seat'])
        replay=play(job)
        exact=all(replay.get(k)==r.get(k) for k in ('error','steps','local_cash','public_cash','local_win','tie'))
        check=dict(opponent=r['public'],seed=r['seed'],seat=r['local_seat'],exact=exact,
            parallel_max_action_s=r['max_local_action_s'],serial_max_action_s=replay.get('max_local_action_s'),
            local_cash=replay.get('local_cash'),public_cash=replay.get('public_cash'),error=replay.get('error'))
        checks.append(check);print(json.dumps(check),flush=True)
    result=dict(games=len(checks),exact_cash_and_result_matches=sum(r['exact'] for r in checks),
        max_serial_action_s=max(r['serial_max_action_s'] or 0 for r in checks),
        boundary='Single-process local checks; not Kaggle sandbox certification.',rows=checks)
    (HERE/'SERIAL_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['games']==result['exact_cash_and_result_matches']


if __name__=='__main__':main()
