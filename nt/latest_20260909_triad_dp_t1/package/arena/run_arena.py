"""Original GPT / current local planner against seven complete live opponents."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, copy, gzip, hashlib, importlib.util, inspect, json, multiprocessing, statistics
import sys, time, traceback, zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'experiments/daily_dp_v7_20260903'
HOST=ROOT/'gpt_review/codex/G001_CPU_FOR_GPT_20260903'
GPT=ROOT/'gpt_review/gpt_code/gpt-6-dp'
NAMES=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day']
sys.path[:0]=[str(EXP/'native/build'),str(HOST)]
from cpu_runtime import LocalGame, load_agent, pass_agent
import _dp7_native as native

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def canon(obj): return json.loads(json.dumps(obj))
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@lru_cache(None)
def asset(name):
    info=read(EXP/'opponents/registry.json')['opponents'][name]
    payload=json.loads(zlib.decompress((EXP/info['asset']).read_bytes()))
    assert digest(EXP/info['source'])==info['source_sha256']==payload['source_sha256']
    cls={'g001':native.G001,'g003':native.G001,'boatlee_v29':native.BoatleeV29,
         'kaito_v58':native.KaitoV58,'lynn_v5':native.LynnV5}[name]
    return cls(payload)

def rival(name):
    if name=='pass': return None,None
    if name=='yhay81_six_day': return native.Fieldbook(),None
    if name=='yhay81_three_day': return native.ThreeDay(),None
    cls={'g001':native.G001State,'g003':native.G001State,'boatlee_v29':native.BoatleeState,
         'kaito_v58':native.KaitoState,'lynn_v5':native.LynnState}[name]
    return asset(name),cls()

def candidate(name,custom=None):
    if name in ('ours_base','ours_autonomous'):
        config=read(ROOT/'configs'/('base.json' if name=='ours_base' else 'full_autonomous.json'))
        return native.Controller(config),True
    if name=='ours_j7':
        path=EXP/'strategy_switch7_20260905/final_local_best_v1/local_agent.py'
        spec=importlib.util.spec_from_file_location('local_j7_entry',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        obj=module.Agent();obj.reset()
        return obj.agent,False
    path=Path(custom).resolve() if name=='custom' else GPT/(
        'gpt-6-kaggriculture_daily_dp_agent.py' if name=='v1' else 'gpt-6-kaggriculture_daily_dp_agent-v2.py')
    return load_agent(path),False

def run_one(job):
    name,opp,seed,seat,referee,out,save_trace,custom=job
    key=f'{name}_{opp}_{seed}_seat{seat}'
    row=dict(agent=name,opponent=opp,seed=seed,seat=seat,status='RUNNING',official_checked_steps=0)
    actions=[];days=[];latency=[];tic=time.perf_counter()
    try:
        own,is_native=candidate(name,custom)
        other,state=rival(opp)
        env=native.Env(seed)
        official=LocalGame(seed) if referee else None
        cfg={k:copy.deepcopy(v.get('default') if isinstance(v,dict) else v)
             for k,v in read(HOST/'official/kaggriculture.json')['configuration'].items()}
        cfg.update(seed=None,runTimeout=1200)
        if not is_native:
            sig=inspect.signature(own)
            takes_two=len(sig.parameters)>=2
        for step in range(719):
            obs=env.observation(seat)
            if step%24==0:
                days.append(canon(dict(step=step,farms=obs['farms'],private=obs['private'],market=obs['market'],town=obs['town'])))
            start=time.perf_counter()
            action=own.act(env,seat) if is_native else own(obs,copy.deepcopy(cfg)) if takes_two else own(obs)
            latency.append(time.perf_counter()-start)
            enemy=pass_agent(env.observation(1-seat)) if opp=='pass' else other.act(env,1-seat) if state is None else other.act(env,1-seat,state)
            pair=[None,None];pair[seat]=action;pair[1-seat]=enemy
            actions.append(canon(pair));env.step(pair)
            if official:
                official.advance(pair)
                for side in (0,1):
                    a,b=env.observation(side),official.observation(side)
                    for field in ('farms','private','market','town','day','hour','player'):
                        assert canon(a[field])==canon(b[field]),(step,side,field)
                row['official_checked_steps']+=1
        assert env.done and env.step_count==719
        farms=env.observation(seat)['farms'];cash=farms[seat]['money'];oc=farms[1-seat]['money']
        if official: assert official.done and official.state[seat].reward==cash and official.state[1-seat].reward==oc
        row.update(status='PASS',cash=cash,opponent_cash=oc,margin=cash-oc,win=cash>oc,tie=cash==oc,
                   steps=env.step_count,action_max_ms=max(latency)*1000,action_seconds=sum(latency),
                   action_over_1s=sum(x>1 for x in latency))
    except Exception:
        row.update(status='ERROR',error=traceback.format_exc(),steps=len(actions))
    row['wall_seconds']=time.perf_counter()-tic
    write(Path(out)/'games'/(key+'.json'),row)
    if save_trace:
        body=json.dumps(dict(metadata=row,days=days,actions=actions),separators=(',',':')).encode()
        (Path(out)/'traces'/(key+'.json.gz')).write_bytes(gzip.compress(body))
    return row

def summarize(rows):
    good=[r for r in rows if r['status']=='PASS']
    result=dict(games=len(rows),completed=len(good),errors=len(rows)-len(good))
    if good:
        result.update(wins=sum(r['win'] for r in good),ties=sum(r['tie'] for r in good),
                      win_rate=sum(r['win'] for r in good)/len(rows),
                      mean_cash=statistics.mean(r['cash'] for r in good),
                      mean_margin=statistics.mean(r['margin'] for r in good),
                      max_action_ms=max(r['action_max_ms'] for r in good),
                      official_steps=sum(r['official_checked_steps'] for r in good))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--agents',nargs='+',default=['v1','v2','ours_j7'],choices=['v1','v2','ours_j7','ours_base','ours_autonomous','custom'])
    parser.add_argument('--opponents',nargs='+',default=NAMES,choices=NAMES+['pass'])
    parser.add_argument('--seed-start',type=int,default=61020)
    parser.add_argument('--count',type=int,default=1)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--official',choices=['none','first','all'],default='first')
    parser.add_argument('--traces',action='store_true')
    parser.add_argument('--custom-path')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.count<1 or args.workers<1: parser.error('positive count/workers required')
    if 'custom' in args.agents and not args.custom_path: parser.error('--custom-path required')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'games').mkdir();(out/'traces').mkdir()
    files=[ROOT/'run_arena.py',*GPT.glob('*.py'),* (EXP/'native').glob('*.hpp'),*(EXP/'native').glob('*.cpp')]
    if args.custom_path: files.append(Path(args.custom_path).resolve())
    write(out/'protocol.json',dict(arguments=vars(args),source_hashes={str(p):digest(p) for p in files},
          note='Full live opponents; no true seed/future/opponent-private input to candidate. First seed checked against official. Timings are local, not Kaggle sandbox.'))
    jobs=[(a,o,s,seat,args.official=='all' or args.official=='first' and s==args.seed_start,str(out),args.traces,args.custom_path)
          for s in range(args.seed_start,args.seed_start+args.count) for o in args.opponents for a in args.agents for seat in (0,1)]
    rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for fut in as_completed([pool.submit(run_one,j) for j in jobs]):
            row=fut.result();rows.append(row)
            print(f'{len(rows)}/{len(jobs)} {row["agent"]} {row["opponent"]} {row["status"]} margin={row.get("margin")}',flush=True)
    summary=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',seconds=time.perf_counter()-started,
                 overall={a:summarize([r for r in rows if r['agent']==a]) for a in args.agents},
                 per_opponent={a:{o:summarize([r for r in rows if r['agent']==a and r['opponent']==o]) for o in args.opponents} for a in args.agents})
    write(out/'result.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
