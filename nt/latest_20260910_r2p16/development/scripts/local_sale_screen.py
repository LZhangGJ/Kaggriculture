from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing,json,argparse
import run_panel as panel
import economic_screen as screen
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def verify_off():
    output=HERE/'candidate_r2p9/off_regression';binary=HERE/'candidate_r2p9/policy/local_sale_0.so'
    if (output/'RESULTS.json').exists():
        r=json.loads((output/'RESULTS.json').read_text());assert r['status']=='PASS' and r['binary_sha256']==panel.sha(binary);return
    pool=json.loads((HERE/'pool_all11.json').read_text());settings=json.loads((panel.old.R2/'config.json').read_text())
    old={(r['opponent'],r['seed'],r['opponent_seat']):r for r in json.loads((HERE/'baseline_rows_all11.json').read_text())}
    jobs=[(p['id'],str(ROOT/p['working']),2609110000+[0,14,28,42,57,71,85,99][n%8],seat,str(output),settings,str(binary)) for n,p in enumerate(pool) for seat in (0,1)]
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),initializer=panel.init_worker) as executor:rows=list(executor.map(panel.game,jobs))
    for r in rows:
        assert not r['runtime_error'],r
        assert r['joint_action_sha256']==old[(r['opponent'],r['seed'],r['opponent_seat'])]['joint_action_sha256'],r
    panel.save(output/'RESULTS.json',dict(status='PASS',games=len(rows),actions=719*len(rows),binary_sha256=panel.sha(binary),rows=rows))
    print('OFF parity PASS',len(rows),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=8);p.add_argument('--modes',default='1,2');p.add_argument('--off-only',action='store_true');args=p.parse_args()
    assert 1<=args.seeds<=100
    verify_off()
    if args.off_only:return
    assert json.loads((HERE/'candidate_r2p9/UNIT_TESTS.json').read_text())['status']=='PASS'
    for mode in map(int,args.modes.split(',')):
        assert mode in (1,2);name=f'local_sale_{mode}';screen.OPTIONS[name]={}
        screen.run([name],args.seeds,HERE/f'local_sale_screen{args.seeds}'/name,True,HERE/f'candidate_r2p9/policy/{name}.so')
if __name__=='__main__':main()
