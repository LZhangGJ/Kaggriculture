"""Live original opponents with the previously compiled, frozen C++ simulator."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import ctypes
import gzip
import json
import multiprocessing
import time
import traceback
from pathlib import Path
from pool_test import ROOT, WORKSPACE, PACKAGE, BASELINE, SOURCE, BINARY, SEEDS, ARMS, digest, module, read
from _pool_sim import FastEnv

OUT = ROOT / 'runs/native20'
OLD = ROOT / 'runs/pool20/smoke'
METRICS = ('wages','hires','moves','idle','pickups','drops','production','overflow_units','daytime_overflow_units')

def compact(obj):
    return json.dumps(obj,separators=(',',':'))

def load_replay(path):
    return json.loads(gzip.decompress(path.read_bytes()))

def normalize_obs(obs):
    # JSON recordings convert coordinate tuples to lists; change only that representation.
    for farm in obs['farms']:
        farm['farmer'] = list(farm['farmer'])
        farm['hands'] = [list(p) for p in farm['hands']]
    return obs

def check_obs(actual, expected, context):
    actual = normalize_obs(actual)
    assert actual.keys() == expected.keys(), (context,actual.keys(),expected.keys())
    for key in actual:
        assert actual[key] == expected[key], (context,key)

def calibrate_one(path):
    row = read(path)
    rp=OLD/(row['name']+'.json.gz');ap=OLD/(row['name']+'.audit.json.gz')
    assert digest(rp)==row['replay_sha256'] and digest(ap)==row['audit_sha256']
    replay=load_replay(rp);audit=load_replay(ap);env=FastEnv(row['seed'])
    totals=Counter();completed=[Counter() for _ in range(30)];invalid=[]
    for t,frame in enumerate(replay['steps']):
        for seat in (0,1):check_obs(env.observation(seat),frame[seat]['observation'],(row['name'],t,seat))
        if t==719:break
        actions=[s['action'] for s in replay['steps'][t+1]]
        step=env.advance(actions,row['seat'])
        for key in METRICS:totals[key]+=step[key]
        completed[t//24].update(tuple(x) for x in step['completed'])
        invalid.extend((t,u,x,y) for u,x,y in step['invalid'])
    missing=[]
    for d,day in enumerate(audit['daily']):
        missing.append([list(k)+[n] for k,n in (Counter(tuple(x) for x in day.get('required_tasks',[]))-completed[d]).items()])
        assert sorted(missing[-1]) == sorted(day.get('uncompleted_required_tasks',[])), (row['name'],d,'required tasks')
    for key in ('wages','hires','moves','overflow_units'):assert totals[key]==row[key],(row['name'],key,totals[key],row[key])
    assert len(invalid)==row['invalid_actions'],(row['name'],'invalid',invalid)
    assert env.done and env.step_count==719
    return dict(name=row['name'],frames=720,observations=1440,replay_sha256=digest(rp),status='PASS')

def complete_python_rows():
    return [p for p in sorted(OLD.glob('*.result.json')) if 'replay_sha256' in read(p)]

def calibration(workers):
    paths=complete_python_rows()
    assert paths
    (ROOT/'runs/pool20/INTERRUPTED.json').write_text(json.dumps(dict(
        reason='Stopped at user request: use previously compiled C++ match simulator instead of Python interpreter.',
        complete_receipts=len(paths),planned=880,used_as='calibration only; not mixed into native panel'),indent=2)+'\n')
    rows=[];tick=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(calibrate_one,p) for p in paths]):
            rows.append(f.result())
            if len(rows)%44==0:print(compact(dict(calibrated=len(rows),planned=len(paths))),flush=True)
    result=dict(status='PASS',games=len(rows),observations=sum(r['observations'] for r in rows),
                checks='Every step, both private observations, all public fields; wages, hires, moves, night overflow, no-effect work and missing commitments.',
                representation_normalization='Only coordinate tuples to lists. No fields or numeric differences ignored.',
                seconds=time.perf_counter()-tick,rows=rows)
    (ROOT/'NATIVE_CALIBRATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(compact({k:v for k,v in result.items() if k!='rows'}),flush=True)

def run(job):
    arm,opponent,seed,seat=job
    name=f'{arm}_{opponent}_{seed}_seat{seat}'
    runtime=module(BASELINE/'referee/cpu_runtime.py','cpu_runtime')
    loader=module(BASELINE/'run.py','native_public_loader')
    entry=module(PACKAGE/'main.py','native_candidate_entry')
    # Read defaults only; no Python interpreter is called in this game loop.
    config={k:copy.deepcopy(v.get('default') if isinstance(v,dict) else v)
            for k,v in runtime.load_engine().specification['configuration'].items()}
    config.update(seed=None,runTimeout=1200)
    configuration=runtime.AttrDict(config)
    env=FastEnv(seed);agent=entry.create_agent(BASELINE/'policy/startupsupply2.so' if arm=='P16' else BINARY)
    public=loader.FreshPublic(SOURCE/'opponents'/opponent)
    modified=arm!='P16'
    if modified:
        agent.lib.td_route_required.argtypes=[ctypes.c_void_p]
        agent.lib.td_route_required.restype=ctypes.c_char_p
    daily=[dict(day=d+1,**{k:0 for k in METRICS}) for d in range(30)]
    completed=[Counter() for _ in range(30)]
    traces=[];invalid=[];times=[[],[]];frames=[];last_changes=0;error=None
    started=time.perf_counter();engine_seconds=0
    def snapshot(actions):
        observations=[normalize_obs(env.observation(p)) for p in (0,1)]
        return [dict(observation=observations[p],action=copy.deepcopy(actions[p]),
                     status='DONE' if env.done else 'ACTIVE',
                     reward=observations[p]['farms'][p]['money'] if env.done else 0) for p in (0,1)]
    frames.append(snapshot([{},{}]))
    try:
        while not env.done:
            t=env.step_count;actions=[None,None]
            for p in (0,1):
                obs=runtime.AttrDict(normalize_obs(env.observation(p)))
                assert 'seed' not in obs and configuration.seed is None
                tick=time.perf_counter()
                actions[p]=(agent if p==seat else public.call)(obs,copy.deepcopy(configuration))
                times[p].append(time.perf_counter()-tick)
            assert all(isinstance(a,dict) and len(a.get('market',[]))<=10 for a in actions)
            assert len(actions[seat]['hands'])==len(env.observation(seat)['farms'][seat]['hands'])
            info=agent.debug() if modified else {}
            day=daily[t//24]
            changes=info.get('route_hire_changes',0)+info.get('route_changes',0)
            if changes>last_changes:
                day['required_tasks']=json.loads(agent.lib.td_route_required(agent.handle));last_changes=changes
            traces.append(dict(step=t,info=info))
            tick=time.perf_counter();audit=env.advance(actions,seat);engine_seconds+=time.perf_counter()-tick
            for key in METRICS:day[key]+=audit[key]
            completed[t//24].update(tuple(x) for x in audit['completed'])
            for u,x,y in audit['invalid']:
                action=actions[seat]['farmer'] if u==0 else actions[seat]['hands'][u-1]
                invalid.append(dict(step=t,unit=u,position=[x,y],action=action))
            frames.append(snapshot(actions))
            if env.step_count%24==0 or env.done:
                day['cash']=env.observation(seat)['farms'][seat]['money']
                day['route']=info
                day['uncompleted_required_tasks']=[list(k)+[n] for k,n in
                    (Counter(tuple(x) for x in day.get('required_tasks',[]))-completed[t//24]).items()]
        assert len(frames)==720 and env.step_count==719
    except Exception:
        error=traceback.format_exc()
    finally:
        agent.close();public.close()
    cash=[f['money'] for f in env.observation(seat)['farms']]
    own=times[seat];other=times[1-seat]
    row=dict(name=name,arm=arm,opponent=opponent,seed=seed,seat=seat,frames=len(frames),
             cash=cash[seat],opponent_cash=cash[1-seat],margin=cash[seat]-cash[1-seat],win=cash[seat]>cash[1-seat],
             status=[f['status'] for f in frames[-1]],error=error,
             **{k:sum(d[k] for d in daily) for k in METRICS},invalid_actions=len(invalid),
             route_commitment_days=sum(bool(d.get('required_tasks')) for d in daily),
             uncompleted_required_tasks=sum(sum(x[-1] for x in d.get('uncompleted_required_tasks',[])) for d in daily),
             max_seconds=max(own,default=0),p99_seconds=sorted(own)[int((len(own)-1)*.99)] if own else 0,
             over_one_second=sum(t>1 for t in own),opponent_decisions=len(other),
             opponent_max_seconds=max(other,default=0),opponent_over_one_second=sum(t>1 for t in other),
             route=traces[-1]['info'] if traces else {},seconds=time.perf_counter()-started,
             simulator='frozen compiled P16 fastkag::Simulator',engine_and_audit_seconds=engine_seconds)
    oldpath=OLD/(name+'.result.json')
    row['python_live_parity']=None
    if oldpath.exists() and 'replay_sha256' in read(oldpath) and not error:
        old=read(oldpath);replay=load_replay(OLD/(name+'.json.gz'))
        for t,(actual,expected) in enumerate(zip(frames,replay['steps'])):
            for p in (0,1):
                check_obs(copy.deepcopy(actual[p]['observation']),expected[p]['observation'],(name,t,p))
                assert actual[p]['action']==expected[p]['action'],(name,t,p,'live action parity')
        for k in ('cash','opponent_cash','wages','hires','invalid_actions','uncompleted_required_tasks','overflow_units'):
            assert row[k]==old[k],(name,k,row[k],old[k])
        row['python_live_parity']='PASS'
    replay=dict(name='kaggriculture',version='0.1.0',configuration=config,
                info=dict(seed=seed,TeamNames=[arm if p==seat else opponent for p in (0,1)]),steps=frames,
                rewards=[f['reward'] for f in frames[-1]],statuses=row['status'],schema_version=1)
    directory=OUT/'matches';directory.mkdir(parents=True,exist_ok=True)
    for suffix,data in [('.json.gz',replay),('.audit.json.gz',dict(result=row,daily=daily,route=traces,invalid=invalid,action_seconds=own))]:
        (directory/(name+suffix)).write_bytes(gzip.compress(compact(data).encode(),compresslevel=3))
    row['replay_sha256']=digest(directory/(name+'.json.gz'));row['audit_sha256']=digest(directory/(name+'.audit.json.gz'))
    path=directory/(name+'.result.json');temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(path)
    return row

def panel(workers, pilot=False):
    assert read(ROOT/'NATIVE_CALIBRATION.json')['status']=='PASS'
    old=read(ROOT/'runs/pool20/PROTOCOL.json')
    for path,sha in old['hashes'].items():assert digest(WORKSPACE/path)==sha,path
    build=read(ROOT/'NATIVE_BUILD.json')
    for path,sha in build['hashes'].items():assert digest(WORKSPACE/path)==sha,path
    jobs=old['jobs'];OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(jobs=jobs,workers=workers,source_protocol_sha256=digest(ROOT/'runs/pool20/PROTOCOL.json'),
                  native_build_sha256=digest(ROOT/'NATIVE_BUILD.json'),calibration_sha256=digest(ROOT/'NATIVE_CALIBRATION.json'),
                  runner_sha256=digest(Path(__file__)),engine='Existing compiled P16 simulator',
                  policies='Unchanged frozen P16 and revision5 route3',opponents='Original 11 Python programs, live feedback',
                  seed_not_given_to_agents=True)
    path=OUT/'PROTOCOL.json'
    if path.exists():assert read(path)==protocol,'Native protocol changed'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    if pilot:jobs=jobs[:44]+[j for j in jobs if j[0]=='economic' and j[1] in ('moon_v215','aurax_reactive_v1') and j[2]==2609168005]
    rows=[];pending=[]
    for job in jobs:
        arm,op,seed,seat=job;path=OUT/'matches'/f'{arm}_{op}_{seed}_seat{seat}.result.json'
        if path.exists():
            row=read(path)
            for suffix,key in [('.json.gz','replay_sha256'),('.audit.json.gz','audit_sha256')]:
                assert digest(path.parent/(row['name']+suffix))==row[key]
            rows.append(row)
        else:pending.append(job)
    tick=time.perf_counter();print(compact(dict(planned=len(jobs),completed=len(rows),pending=len(pending))),flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
        for f in as_completed([pool.submit(run,j) for j in pending]):
            row=f.result();rows.append(row)
            if len(rows)%22==0 or row['error'] or row['uncompleted_required_tasks']:
                print(compact(dict(completed=len(rows),seconds=time.perf_counter()-tick,
                      **{k:row[k] for k in ('name','cash','margin','wages','invalid_actions','uncompleted_required_tasks','error')})),flush=True)
    rows.sort(key=lambda r:(r['seed'],r['opponent'],r['seat'],r['arm']))
    (OUT/('pilot_rows.json' if pilot else 'rows.json')).write_text(json.dumps(rows,indent=2)+'\n')
    for path,sha in old['hashes'].items():assert digest(WORKSPACE/path)==sha,path
    for path,sha in build['hashes'].items():assert digest(WORKSPACE/path)==sha,path
    print(compact(dict(completed=len(rows),errors=sum(bool(r['error']) for r in rows),seconds=time.perf_counter()-tick)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--calibrate',action='store_true');parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    calibration(args.workers) if args.calibrate else panel(args.workers,args.pilot)
