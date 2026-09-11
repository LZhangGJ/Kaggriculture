"""Repeat the slowest seeds as isolated serial live games; exclude from win-rate samples."""
import argparse,json,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from run import ROOT,WORKSPACE,BINARIES,run,read,digest
def main():
    p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('--count',type=int,default=3)
    p.add_argument('--include-opponent-slowest',action='store_true');a=p.parse_args()
    out=ROOT/'runs'/a.tag;rows=read(out/'rows.json');selected=[]
    protocol=read(out/'PROTOCOL.json')
    for name,sha in protocol['hashes'].items():assert digest(WORKSPACE/name)==sha,name
    for arm in sorted({r['arm'] for r in rows}):
        selected+=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['max_seconds'],reverse=True)[:a.count]
    if a.include_opponent_slowest:
        slowest=max(rows,key=lambda r:r['opponent_max_seconds'])
        if slowest['name'] not in {r['name'] for r in selected}:selected.append(slowest)
    items=[((r['arm'],r['opponent'],r['seed'],r['seat']),a.tag+'_serial',None) for r in selected]
    results=[]
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
        for row in pool.map(run,items):
            assert not row['error'] and row['frames']==720
            results.append(row);print(json.dumps({k:row[k] for k in ['name','max_seconds','over_one_second','cash','margin']}),flush=True)
    for name,sha in protocol['hashes'].items():assert digest(WORKSPACE/name)==sha,name
    result=dict(method='Slowest seeds rerun as serial live games in isolated processes. Recorded full replays. Excluded from strength samples.',
                includes_opponent_slowest=a.include_opponent_slowest,
                hashes={arm:digest(BINARIES[arm]) for arm in {r['arm'] for r in results}},rows=results)
    (out/'SERIAL_TIMING.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
