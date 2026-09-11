"""Portable Python orchestration; all 719-step matches execute in C++."""
from pathlib import Path
import hashlib, json, os, sys, time, zlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
P = Path(__file__).resolve().parent
sys.path.insert(0,str(P/'build'))
import _economic_rl_native as native
NAMES = ['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day']
ARMS = ['c3auto','f3','c3j7']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make_pool():
    rows=json.loads((P/'OPPONENTS.json').read_text(encoding='utf-8-sig'))
    assert [r['name'] for r in rows]==NAMES
    data={}
    for row in rows:
        assert sha(P/row['source'])==row['source_sha256'],row['name']
        if row['asset']:
            data[row['name']]=json.loads(zlib.decompress((P/row['asset']).read_bytes()))
            assert data[row['name']]['source_sha256']==row['source_sha256']
    return native.RLPool(data)

def jobs(start,count,opponents=None):
    if count<1 or start<0:
        raise ValueError('positive seed count and non-negative seed start required')
    opponents=NAMES if opponents is None else opponents
    return [(s,seat,NAMES.index(name)) for s in range(start,start+count) for name in opponents for seat in (0,1)]

def batch(pool,arm,joblist,plan=None,threads=16,trace=False):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    if not joblist or not 1<=threads<=16:
        raise ValueError('empty batch or invalid threads')
    if plan is not None:
        values=[int(v) for v in Path(plan).read_text().split()]
        if len(values)!=29 or any(not 0<=v<10 for v in values):
            raise ValueError('plan must have 29 integer candidate IDs in 0..9')
    library=P/'build'/(arm+('_plan' if plan is not None else '')+'.so')
    start=time.perf_counter()
    result=pool.batch(str(library),str(Path(plan).resolve()) if plan is not None else '',0,
        [j[0] for j in joblist],[j[1] for j in joblist],[j[2] for j in joblist],17,threads,trace)
    result['call_seconds']=time.perf_counter()-start
    result['library_sha256']=sha(library)
    result['plan_sha256']=sha(plan) if plan is not None else None
    for row in result['rows']:
        row['opponent_name']=NAMES[row['opponent']]
    if plan is not None:
        requested=np.array([values[d] if d<29 else 0 for d in result['day']],dtype=np.int32)
        result['requested']=requested
        result['fallback']=requested!=result['choice']
    return result

def summary(result):
    rows=result['rows']
    def group(rr):
        return dict(games=len(rr),wins=sum(r['win'] for r in rr),ties=sum(r['tie'] for r in rr),
            win_rate=sum(r['win'] for r in rr)/len(rr),mean_cash=sum(r['cash'] for r in rr)/len(rr),
            mean_margin=sum(r['margin'] for r in rr)/len(rr))
    errors=[r for r in rows if r['error'] or r['steps']!=719]
    return dict(status='FAIL' if errors else 'PASS',overall=group(rows),
        per_opponent={n:group([r for r in rows if r['opponent_name']==n]) for n in NAMES if any(r['opponent_name']==n for r in rows)},
        wall_seconds=result['wall_seconds'],call_seconds=result['call_seconds'],
        games_per_second=len(rows)/result['call_seconds'],decisions=len(result['choice']),
        changed=int(np.sum(result['changed'])),fallbacks=int(np.sum(result.get('fallback',[]))),
        library_sha256=result['library_sha256'],plan_sha256=result['plan_sha256'],errors=errors)

def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def save_result(out,result):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    save_json(out/'summary.json',summary(result));save_json(out/'games.json',result['rows'])
    np.savez_compressed(out/'decisions.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})

def rank(result):
    s=summary(result)
    if s['status']!='PASS':
        raise RuntimeError(s['errors'])
    return (min(r['win_rate'] for r in s['per_opponent'].values()),s['overall']['win_rate'],s['overall']['mean_margin'])
