"""Portable build/search/replay entrypoint. Never submits to Kaggle."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, gzip, hashlib, importlib.util, json, os, subprocess, sys, sysconfig, time, zipfile

ROOT=Path(__file__).resolve().parents[1]
KEEP={'family':'KEEP','kind':-1,'amount':0}
DAYS8=[0,1,3,6,9,12,18,24]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def zipped(p,d):
    with gzip.open(p,'wt',encoding='utf8',compresslevel=6) as f:json.dump(d,f,separators=(',',':'))
def fresh(p):p=Path(p).resolve();p.mkdir(parents=True,exist_ok=False);return p
def build_dir(a):return Path(a.build_dir).resolve()
def get_plan(uid):
    plans=read(ROOT/'data/plans300.json')
    if uid.upper()=='DP27':return next(x for x in plans if x['source_team']=='Driz Lo' and x['index']==27)
    return next(x for x in plans if x['uid']==uid or x['plan_id']==uid)
def context(a):
    path=build_dir(a);receipt=read(path/'build.json');assert receipt['status']=='PASS'
    binary=path/receipt['binary'];assert sha(binary)==receipt['binary_sha256']
    for rel,h in receipt['source_hashes'].items():assert sha(ROOT/rel)==h,('Changed after build',rel)
    sys.path.insert(0,str(path));import _dp7_search8 as n
    with gzip.open(ROOT/'data/route_pool.json.gz','rt') as f:payload=json.load(f)
    payload['moon']={int(k):v for k,v in payload['moon'].items()}
    payload['moon_legacy']={int(k):v for k,v in payload['moon_legacy'].items()}
    return read(ROOT/'data/frozen_inputs.json'),n,n.RoutePool(payload)
def target_id(f,name):
    for c in f['selected']:
        if name.casefold() in (c['team_name'].casefold(),c['route_id'].casefold()):return c['route_id']
    if name in f['route_ids']:return name
    raise ValueError('Unknown target: '+name)
def stats(rows):
    return {'games':len(rows),'wins':sum(r['win'] for r in rows),
            'mean_cash':sum(r['cash'] for r in rows)/len(rows),
            'mean_opponent_cash':sum(r['opponent_cash'] for r in rows)/len(rows),
            'mean_margin':sum(r['margin'] for r in rows)/len(rows),
            'missing_recipe':sum(r['missing_recipe'] for r in rows)}
def evaluate(n,pool,f,rid,days,plan,seeds,threads):
    jobs=[(s,p) for s in seeds for p in (0,1)];route=f['route_ids'].index(rid)
    def one(job):
        r=n.replay8(pool,route,f['config'],*job,days,plan,False);assert r['steps']==719
        return r
    with ThreadPoolExecutor(max_workers=threads) as workers:return list(workers.map(one,jobs))

def build(a):
    import pybind11
    out=build_dir(a);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'build.json').exists(),'Use a new --build-dir; preserve previous builds'
    native=ROOT/'engine/native';base=['g++','-std=c++20','-O3','-DNDEBUG','-march=native','-fopenmp','-fPIC',
        '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-I'+str(native)]
    sources=['vendor/simulator.cpp','fieldbook_adapter.cpp','boatlee_v29.cpp','kaito_v58.cpp','lynn_v5.cpp',
             'three_day_adapter.cpp','ecobot_v7_core.cpp','ecobot_v7.cpp']
    commands=[];tic=time.perf_counter()
    def compile_one(rel):
        obj=out/(Path(rel).stem+'.o');cmd=list(base)
        if rel in ('boatlee_v29.cpp','kaito_v58.cpp','lynn_v5.cpp','ecobot_v7_core.cpp','ecobot_v7.cpp'):cmd+=['-ffp-contract=off']
        if rel=='three_day_adapter.cpp':cmd+=['-I'+str(ROOT/'engine/opponents/yhay81_three_day/output/source/include')]
        cmd+=['-c',str(native/rel),'-o',str(obj)]
        p=subprocess.run(cmd,capture_output=True,text=True);(out/(obj.stem+'.log')).write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(p.stderr[-6000:])
        print(json.dumps({'compiled':rel}),flush=True);return obj,cmd
    with ThreadPoolExecutor(max_workers=a.jobs) as workers:results=list(workers.map(compile_one,sources))
    binary=out/('_dp7_search8'+sysconfig.get_config_var('EXT_SUFFIX'))
    cmd=base+['-shared',str(ROOT/'engine/search8_multiseed/search.cpp'),*[str(o) for o,c in results],'-o',str(binary)]
    p=subprocess.run(cmd,capture_output=True,text=True);(out/'search.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise RuntimeError(p.stderr[-6000:])
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'engine').rglob('*') if p.is_file()}
    r=dict(status='PASS',seconds=time.perf_counter()-tic,commands=[c for o,c in results]+[cmd],
           binary=binary.name,binary_sha256=sha(binary),source_hashes=hashes,python=sys.version,
           pybind11=pybind11.__version__,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0])
    write(out/'build.json',r);print(json.dumps({'build':'PASS','seconds':r['seconds']}),flush=True)

def smoke(a):
    out=fresh(a.out);f,n,pool=context(a);refs=read(ROOT/'data/dp27_reference96.json');plan=get_plan('DP27')['plan'];rows=[];tic=time.perf_counter()
    # All 96 full dynamic trajectories must reproduce cash AND full state/action fingerprints.
    for rid,expected in refs.items():
        actual=evaluate(n,pool,f,rid,list(range(29)),plan,sorted({r['seed'] for r in expected}),a.threads)
        for x,y in zip(actual,expected):
            for key in ('seed','seat','cash','opponent_cash','state_action_hash','missing_recipe','changed_nodes','steps'):
                assert x[key]==y[key],('portable parity',rid,key,x[key],y[key])
        rows.append({'route':rid,**stats(actual)})
        print(json.dumps({'parity96':rid,**stats(actual)}),flush=True)
    # New build must also generate the original full Day0 beam frontier, not just play a saved plan.
    fixture=read(ROOT/'data/day0_frontier_reference.json');seeds=[s for s in f['train_seeds'] for _ in (0,1)];seats=[p for _ in f['train_seeds'] for p in (0,1)]
    eng=n.Search8(pool,f['route_ids'].index(fixture['route_id']),f['config'],seeds,seats,list(range(29)),12,a.threads)
    stage=eng.expand()
    for key in ('day','proposals','frontier','beam'):assert stage[key]==fixture[key],('search frontier',key)
    # Official frozen Python: 3 targets, both seats, complete public/private/market/town state.
    sys.path.insert(0,str(ROOT/'referee'));from cpu_runtime import LocalGame
    official=[]
    for rid,expected in refs.items():
        for seat in (0,1):
            seed=expected[0]['seed'];trace=n.replay8(pool,f['route_ids'].index(rid),f['config'],seed,seat,list(range(29)),plan,True)
            env=n.Env(seed);judge=LocalGame(seed)
            for step,pair in enumerate(trace['trace']):
                env.step(pair);judge.advance(pair)
                for p in (0,1):
                    x=env.observation(p);y=judge.observation(p)
                    for key in ('farms','private','market','town'):
                        assert json.loads(json.dumps(x[key]))==json.loads(json.dumps(y[key])),('official',rid,seat,step,key)
            assert env.done and judge.done and env.step_count==719
            official.append({'route':rid,'seed':seed,'seat':seat,'steps':719})
            print(json.dumps({'official':'PASS',**official[-1]}),flush=True)
    write(out/'acceptance.json',dict(status='PASS_PORTABLE96_SEARCH_FRONTIER_OFFICIAL6',rows=rows,official=official,
        frontier_candidates=stage['proposals'],seconds=time.perf_counter()-tic,binary_sha256=read(build_dir(a)/'build.json')['binary_sha256']))

def pkey(plan,days):return tuple((r['family'],r['kind'],r['amount']) for r in list(plan)+[KEEP]*(len(days)-len(plan)))
def pid(plan,days):return hashlib.sha256(json.dumps(pkey(plan,days),separators=(',',':')).encode()).hexdigest()[:16]
def distance(a,b):return sum(2*(x[0]!=y[0])+(x[1]!=y[1])+0.5*abs(x[2]-y[2])/max(1,abs(x[2]),abs(y[2])) for x,y in zip(a,b))
def search(a):
    out=fresh(a.out);f,n,pool=context(a);rid=target_id(f,a.target);route=f['route_ids'].index(rid)
    days=DAYS8 if a.days==8 else list(range(29));train=a.train_seeds;seeds=[s for s in train for _ in (0,1)];seats=[p for _ in train for p in (0,1)]
    write(out/'inputs.json',dict(target=rid,days=days,width=a.width,count=a.count,train=train,seats=[0,1],config=f['config'],
        binary_sha256=read(build_dir(a)/'build.json')['binary_sha256'],mode='OFFLINE_TRAIN_SEEDS_ONLY',candidate_limit=None))
    eng=n.Search8(pool,route,f['config'],seeds,seats,days,a.width,a.threads);catalog={};tic=time.perf_counter()
    for depth in range(len(days)):
        stage=eng.expand();zipped(out/f'stage{depth:02d}.json.gz',stage)
        for row in stage['frontier']:
            if row['wins']!=len(seeds):continue
            key=pkey(row['plan'],days);full=list(row['plan'])+[dict(KEEP)]*(len(days)-len(row['plan']))
            if key in catalog:
                for k in ('wins','mean_margin','mean_cash'):assert row[k]==catalog[key][k]
            else:catalog[key]=dict(row,plan=full,origin_day=days[depth],plan_id=pid(full,days))
        print(json.dumps({'day':days[depth],'proposals':stage['proposals'],'best_train_wins':stage['frontier'][0]['wins'],'seconds':stage['seconds']}),flush=True)
    write(out/'beam_final.json',eng.state());zipped(out/'winning_catalog.json.gz',list(catalog.values()))
    # Exact original diversity criterion; selection never reads validation futures.
    first=pkey(eng.state()['beam'][0]['plan'],days);remaining=dict(catalog);chosen=[];seen=set();attempts=[];mins={k:float('inf') for k in remaining}
    if first in remaining:
        while remaining and len(chosen)<a.count:
            key=first if not chosen else max(remaining,key=lambda k:(mins[k],remaining[k]['mean_margin'],remaining[k]['mean_cash'],k))
            item=remaining.pop(key);mins.pop(key);actual=evaluate(n,pool,f,rid,days,item['plan'],train,a.threads);summary=stats(actual)
            assert summary['wins']==len(seeds)
            for k in ('mean_margin','mean_cash'):assert summary[k]==item[k]
            signature=tuple(r['state_action_hash'] for r in actual);duplicate=signature in seen
            attempts.append(dict(plan_id=item['plan_id'],duplicate_behavior=duplicate))
            if duplicate:continue
            seen.add(signature);chosen.append(dict(item,selection_index=len(chosen)+1,training_rows=actual,training=summary))
            for k in remaining:mins[k]=min(mins[k],distance(k,key))
    write(out/'selected.json',dict(status='FROZEN_BEFORE_VALIDATION',target=rid,days=days,train=train,candidates=chosen,
        requested=a.count,actual=len(chosen),semantic_train_allwin=len(catalog),attempts=attempts,seconds=time.perf_counter()-tic,
        caveat='May yield fewer than requested all-win diverse plans; never manufacture winners. Offline search, not online win rate.'))
    print(json.dumps({'selected':len(chosen),'requested':a.count,'seconds':time.perf_counter()-tic}),flush=True)

def replay(a):
    out=fresh(a.out);f,n,pool=context(a);rid=target_id(f,a.target);plan=get_plan(a.plan)
    r=n.replay8(pool,f['route_ids'].index(rid),f['config'],a.seed,a.seat,list(range(29)),plan['plan'],True)
    zipped(out/'trace.json.gz',r);write(out/'summary.json',{'plan':plan['uid'],'target':rid,**{k:v for k,v in r.items() if k not in ('trace','cash_by_step','node_effects')}})
    print(json.dumps(read(out/'summary.json')),flush=True)

def cross(a):
    out=fresh(a.out);f,n,pool=context(a);raw=read(a.plans)
    plans=raw if isinstance(raw,list) else raw['candidates'];days=list(range(29)) if isinstance(raw,list) else raw['days']
    train=f['train_seeds'] if isinstance(raw,list) else raw['train'];seeds=list(range(a.seed_start,a.seed_start+a.seed_count))
    assert not set(seeds)&set(train),'Training and validation seeds must not overlap'
    targets=[c['route_id'] for c in f['selected']] if a.target=='all' else [target_id(f,a.target)]
    write(out/'inputs.json',dict(plans_sha256=sha(a.plans),seeds=seeds,seats=[0,1],targets=targets,days=days,config=f['config']))
    tic=time.perf_counter();summaries=[]
    for i,item in enumerate(plans,1):
        uid=item.get('uid',str(i));per={}
        for rid in targets:
            rows=evaluate(n,pool,f,rid,days,item['plan'],seeds,a.threads);s=stats(rows);per[rid]=s
            write(out/f'games/{i:03d}/{rid}.json',dict(uid=uid,plan=item['plan'],rows=rows,summary=s))
        summaries.append(dict(uid=uid,opponents=per));print(json.dumps({'plan':i,'total':len(plans),'wins':{k:v['wins'] for k,v in per.items()}}),flush=True)
    write(out/'summary.json',dict(status='COMPLETE_NOT_PROMOTION',plans=len(plans),games=len(plans)*len(targets)*len(seeds)*2,
        seconds=time.perf_counter()-tic,rows=summaries))

def verify(a):
    manifest=read(ROOT/'MANIFEST.json');checked=0
    for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel;checked+=1
    idx=read(ROOT/'evidence/index.json');members=0
    for part in idx['shards']:
        path=ROOT/'evidence'/part['path'];assert sha(path)==part['sha256']
        if a.deep:
            with zipfile.ZipFile(path) as z:
                for m in part['members']:
                    with z.open(m['path']) as f:
                        h=hashlib.sha256()
                        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
                    assert h.hexdigest()==m['sha256'];members+=1
    print(json.dumps({'status':'PASS','files':checked,'deep_evidence_members':members}),flush=True)

def unpack(a):
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);idx=read(ROOT/'evidence/index.json');count=0
    for part in idx['shards']:
        path=ROOT/'evidence'/part['path'];assert sha(path)==part['sha256']
        with zipfile.ZipFile(path) as z:
            for m in part['members']:
                dest=(out/m['path']).resolve();assert out in dest.parents,'Unsafe archive path'
                if dest.exists():assert sha(dest)==m['sha256'];continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(m['path']) as src,dest.open('xb') as dst:
                    for b in iter(lambda:src.read(1024*1024),b''):dst.write(b)
                assert sha(dest)==m['sha256'];count+=1
    print(json.dumps({'unpacked':count,'root':str(out)}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['build','smoke','search','replay','cross','verify','unpack'])
    p.add_argument('--build-dir',default=str(ROOT/'build'));p.add_argument('--out',default=str(ROOT/'runs/local'))
    p.add_argument('--jobs',type=int,default=2);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--target',default='Driz Lo');p.add_argument('--plan',default='DP27');p.add_argument('--seed',type=int,default=20264501);p.add_argument('--seat',type=int,choices=[0,1],default=0)
    p.add_argument('--days',type=int,choices=[8,29],default=29);p.add_argument('--width',type=int,default=12);p.add_argument('--count',type=int,default=100)
    p.add_argument('--train-seeds',nargs='+',type=int,default=list(range(20264001,20264005)))
    p.add_argument('--plans',default=str(ROOT/'data/plans300.json'));p.add_argument('--seed-start',type=int,default=20264501);p.add_argument('--seed-count',type=int,default=16);p.add_argument('--deep',action='store_true')
    a=p.parse_args();assert 1<=a.threads<=16 and 1<=a.jobs<=16 and a.width>=1 and a.count>=1 and a.seed_count>=1
    globals()[a.mode](a)
if __name__=='__main__':main()
