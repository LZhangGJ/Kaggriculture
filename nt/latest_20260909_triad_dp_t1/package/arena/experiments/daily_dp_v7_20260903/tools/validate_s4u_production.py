"""Exact optimization validation on the production ABI, never old probe modules."""
from pathlib import Path
import concurrent.futures as cf
import hashlib,json,multiprocessing as mp,shutil,subprocess,sys,time,zlib

E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E/'native/build'))
import _dp7_native as native

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run_pair(task):
    label,opponent,seed,seat,mask=task
    cfg=json.loads((E/'profiles/s4t/configs.json').read_text())[label]
    modified=dict(cfg,exact_schedule_cache=bool(mask&1),incremental_regret_cost=bool(mask&2))
    classes={'g001':(native.G001,native.G001State),'g003':(native.G001,native.G001State),
        'boatlee_v29':(native.BoatleeV29,native.BoatleeState),'kaito_v58':(native.KaitoV58,native.KaitoState),'lynn_v5':(native.LynnV5,native.LynnState)}
    direct={'yhay81_six_day':native.Fieldbook,'yhay81_three_day':native.ThreeDay,'ecobot_v7':native.EcoBotV7}
    if opponent in classes:
        entry=json.loads((E/'profiles/s4u/s4t_partial_results.json').read_text())['identities'][opponent]
        asset=E/entry['asset'];assert sha(asset)==entry['asset_sha256']
        rival=classes[opponent][0](json.loads(zlib.decompress(asset.read_bytes())))
        state=classes[opponent][1]()
    else:rival=direct[opponent]();state=None
    env=native.Env(seed);a=native.Controller(cfg);b=native.Controller(modified)
    fns=[native.portfolio_stats,native.day_value_stats,native.intraday_workforce_stats,
         native.shared_insertion_stats,native.service_stats,native.idle_handoff_stats]
    def check():
        assert a.debug()==b.debug(),('debug mismatch',task,env.step_count)
        for fn in fns:assert fn(a)==fn(b),(fn.__name__,task,env.step_count)
        assert not native.schedule_cache_stats(b)['active_context']
    timings=[[],[]];digest=hashlib.sha256();last_day=-1;max_bytes=0;reset_checks=0
    for step in range(719):
        actions=[None,None]
        for i in (range(2) if step%2 else (1,0)):
            tic=time.perf_counter();actions[i]=(a if i==0 else b).act(env,seat);timings[i].append(time.perf_counter()-tic)
        assert actions[0]==actions[1],('action mismatch',task,step)
        check();cache=native.schedule_cache_stats(b)
        if mask&1:
            assert cache['day']==step//24 and cache['present'],(task,step,cache)
            assert cache['entries']<=8192 and cache['accounted_bytes']<=128*1024*1024
            max_bytes=max(max_bytes,cache['accounted_bytes'])
        else:assert not cache['present']
        other=rival.act(env,1-seat,state) if state is not None else rival.act(env,1-seat)
        env.step([actions[0],other] if seat==0 else [other,actions[0]])
        digest.update(json.dumps(actions[0],sort_keys=True).encode())
    assert env.done and env.step_count==719
    cash=[f['money'] for f in env.observation(seat)['farms']]
    # Reused completed Controller must reset like a brand-new one, with a
    # different seed. Deliberately interleave both on this thread.
    env=native.Env(seed+101);fresh=native.Controller(modified)
    for step in range(25):
        old=b.act(env,seat);before=native.schedule_cache_stats(b)
        new=fresh.act(env,seat)
        assert old==new and b.debug()==fresh.debug(),('new-game contamination',task,step)
        assert native.schedule_cache_stats(b)==before,('another controller mutated cache',task,step)
        assert before==native.schedule_cache_stats(fresh),('cache reset mismatch',task,step)
        env.step([old,{}] if seat==0 else [{},old]);reset_checks+=1
    return dict(label=label,opponent=opponent,seed=seed,seat=seat,mask=mask,action_checks=719,
        reset_checks=reset_checks,cash=cash,action_sha256=digest.hexdigest(),max_cache_bytes=max_bytes,
        old_seconds=sum(timings[0]),new_seconds=sum(timings[1]),old_max=max(timings[0]),new_max=max(timings[1]))

def main():
    out=E/'receipts/s4u_production_equivalence_v1';out.mkdir(exist_ok=False)
    build=json.loads((E/'native/build/build_receipt.json').read_text())
    for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
    source=(E/'profiles/s4u/before/native/policy.hpp').read_text()
    start=source.index('std::pair<std::vector<Route>,int>regret_pack(')
    end=source.index('std::pair<std::vector<Route>,int>pack(',start)
    (out/'reference_regret.inc').write_text(source[start:end].replace('>regret_pack(', '>regret_pack_reference('))
    src=E/'native/test_pack_memo_equivalence.cpp';shutil.copy2(src,out/src.name)
    cmd=['g++','-std=c++20','-O3','-march=native','-DNDEBUG','-I'+str(E/'native'),'-I'+str(out),str(out/src.name),str(E/'native/vendor/simulator.cpp'),'-o',str(out/'test_pack_memo')]
    r=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr);assert not r.returncode,r.stderr[-3000:]
    r=subprocess.run([str(out/'test_pack_memo')],capture_output=True,text=True);(out/'run.log').write_text(r.stdout+r.stderr);assert not r.returncode,r.stdout+r.stderr
    mechanism=json.loads(r.stdout);print(json.dumps(mechanism),flush=True)
    opponents=('g001','g003','boatlee_v29','kaito_v58','lynn_v5','ecobot_v7','yhay81_six_day','yhay81_three_day')
    tasks=[(label,opponent,seed,seat,3) for label in ('full_workers25','full_chain','full_chain_autonomous') for opponent in opponents for seed in (20262701,20262702) for seat in (0,1)]
    tasks += [('full_chain_autonomous',opp,seed,seat,mask) for opp in ('g001','g003') for seed in (20262701,20262702) for seat in (0,1) for mask in (1,2)]
    rows=[];tic=time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(run_pair,tasks):
            rows.append(row);(out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
    result=dict(status='PASS_EXACT_PRODUCTION_EQUIVALENCE',build=build,mechanism=mechanism,rows=rows,
        full_game_pairs=len(rows),action_checks=sum(r['action_checks'] for r in rows),reset_checks=sum(r['reset_checks'] for r in rows),
        seconds=time.perf_counter()-tic,source_sha256=sha(Path(__file__)),test_source_sha256=sha(src),
        caveat='Paired identical actions on shared authoritative live environments. Not extra independent strength samples. Concurrent timing is not online latency acceptance.')
    (out/'acceptance.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)
if __name__=='__main__':main()
