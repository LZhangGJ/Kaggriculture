"""Audit every baseline loss plus matched deterministic winner controls."""
from pathlib import Path
import concurrent.futures as futures
import gzip
import json
import multiprocessing
import time

import audit_trace
import run_panel as panel

HERE=Path(__file__).resolve().parent
OUT=HERE/'baseline_audit'

def one(row):
    directory=OUT/f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}"
    result_path=directory/'COMPACT.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    result=audit_trace.run_case(row,directory)
    panel.save(result_path,result)
    return result

def main():
    rows=json.loads((HERE/'baseline_rows_all11.json').read_text())
    selection=[r for r in rows if not r['r2_win']]
    for name in sorted({r['opponent'] for r in rows}):
        winners=sorted([r for r in rows if r['opponent']==name and r['r2_win'] and r['opponent_seat']==0],key=lambda r:r['seed'])
        # Five wins spread through the seed range per opponent; not best-cash selection.
        indexes=sorted({round(i*(len(winners)-1)/4) for i in range(5)}) if winners else []
        selection.extend(winners[i] for i in indexes)
    panel.save(OUT/'SELECTION.json',dict(method='All 699 baseline losses + five seed-spread seat0 wins per opponent',rows=selection))
    started=time.perf_counter(); results=[]
    with futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as executor:
        for f in futures.as_completed([executor.submit(one,r) for r in selection]):
            results.append(f.result())
            if len(results)%20==0 or len(results)==len(selection):
                progress=dict(done=len(results),total=len(selection),seconds=time.perf_counter()-started)
                panel.save(OUT/'PROGRESS.json',progress)
                print(json.dumps(progress),flush=True)
    panel.save(OUT/'RESULTS.json',dict(status='PASS',cases=len(results),seconds=time.perf_counter()-started,rows=results))

if __name__=='__main__': main()
