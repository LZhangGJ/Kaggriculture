from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing,json,argparse
import run_panel as panel
import economic_screen as screen
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def verify_off():
    output=HERE/'candidate_r2p10/off_regression_v2';binary=HERE/'candidate_r2p10/policy/finite0_sale0_v2.so'
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
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=8);p.add_argument('--sales',default='0,1');p.add_argument('--off-only',action='store_true');args=p.parse_args()
    assert 1<=args.seeds<=100
    verify_off()
    if args.off_only:return
    unit=json.loads((HERE/'candidate_r2p10/UNIT_TESTS.json').read_text());assert unit['status']=='PASS'
    for p,h in unit['source_sha256'].items():assert panel.sha(HERE/'candidate_r2p10'/p)==h
    for sale in map(int,args.sales.split(',')):
        assert sale in (0,1);name=f'finite1_sale{sale}_v2';screen.OPTIONS[name]={}
        build=json.loads((HERE/'candidate_r2p10'/(name+'.BUILD.json')).read_text());binary=HERE/'candidate_r2p10/policy'/(name+'.so');assert panel.sha(binary)==build['binary_sha256']
        for p,h in build['sources'].items():assert panel.sha(HERE/'candidate_r2p10/policy'/p)==h
        screen.run([name],args.seeds,HERE/f'finite_crop_screen{args.seeds}'/name,True,binary)
if __name__=='__main__':main()
