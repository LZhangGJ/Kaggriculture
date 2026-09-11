from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing,json
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'candidate_r2p7/off_regression'
def main():
 panel.init_worker();pool=json.loads((HERE/'pool_all11.json').read_text());settings=json.loads((panel.old.R2/'config.json').read_text())
 baseline={(r['opponent'],r['seed'],r['opponent_seat']):r for r in json.loads((HERE/'baseline_rows_all11.json').read_text())}
 jobs=[(p['id'],str(ROOT/p['working']),2609110000+[0,14,28,42,57,71,85,99][n%8],seat,str(OUT),settings,str(HERE/'candidate_r2p7/policy/market_0.so')) for n,p in enumerate(pool) for seat in (0,1)]
 with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),initializer=panel.init_worker) as executor:rows=list(executor.map(panel.game,jobs))
 for r in rows:
  assert not r['runtime_error'],r
  old=baseline[(r['opponent'],r['seed'],r['opponent_seat'])]
  assert r['joint_action_sha256']==old['joint_action_sha256'],(r,old)
 panel.save(OUT/'RESULTS.json',dict(status='PASS',games=len(rows),actions=719*len(rows),rows=rows))
 print(json.dumps(dict(status='PASS',games=len(rows),actions=719*len(rows))),flush=True)
if __name__=='__main__':main()
