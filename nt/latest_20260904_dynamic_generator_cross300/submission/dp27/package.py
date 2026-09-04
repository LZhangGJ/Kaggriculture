"""Package only after exact deployment parity, and test the no-__file__ loader."""
from pathlib import Path
import hashlib,json,sys,tarfile,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];EXP=ROOT/'experiments/daily_dp_v7_20260903'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    build=read(HERE/'build_receipt.json');smoke=read(HERE/'validation_smoke_compat_v1.json');parity=read(HERE/'validation_parity_compat_v1.json')
    assert build['status']=='PASS'
    for r,games in ((smoke,6),(parity,96)):
        assert r['status']=='PASS_ACTION_PARITY' and r['games']==games
        assert r['binary_sha256']==sha(HERE/'agent.so') and r['entry_sha256']==sha(HERE/'main.py')
        assert r['max_cpu']<1 and r['max_wall']<1
    results={rid:sum(row['win'] for row in parity['rows'] if row['route']==rid) for rid in {r['route'] for r in parity['rows']}}
    assert sorted(results.values())==[23,26,27]
    target=HERE/'submission.tar.gz';assert not target.exists(),'Do not overwrite a submission archive'
    with tarfile.open(target,'w:gz') as tar:
        for name in ('main.py','agent.so'):tar.add(HERE/name,arcname=name)
    unpack=HERE/'package_check';unpack.mkdir(exist_ok=False)
    with tarfile.open(target,'r:gz') as tar:
        assert tar.getnames()==['main.py','agent.so']
        for member in tar.getmembers():
            assert member.isfile() and member.name in ('main.py','agent.so')
            data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==sha(HERE/member.name)
            (unpack/member.name).write_bytes(data)
    sys.path.insert(0,str(EXP/'search8_multiseed'));import run as old
    f,n,pool=old.runtime(EXP/'search8_multiseed/receipts_v1');chosen=read(HERE/'frozen_plan.json')
    path=unpack/'main.py';ns={'__name__':'__kaggle_agent__'};exec(compile(path.read_bytes(),str(path),'exec'),ns)
    assert '__file__' not in ns;fn=ns['agent'];checks=[]
    # No manual library reset: a new episode's step zero must clear old state.
    for seat in (0,1):
        rid='R03_F001_E104234696_P0';seed=20264501
        fixture=n.replay8(pool,f['route_ids'].index(rid),f['config'],seed,seat,list(range(29)),chosen['plan'],True);env=n.Env(seed)
        maxwall=0
        for pair in fixture['trace']:
            tic=time.perf_counter();action=fn(env.observation(seat),{'episodeSteps':720});maxwall=max(maxwall,time.perf_counter()-tic)
            assert action==pair[seat];env.step(pair)
        assert env.done
        checks.append(dict(seed=seed,seat=seat,steps=719,actions_equal=True,max_wall=maxwall))
    info=dict(status='READY_FOR_KAGGLE',label='NT DP27 fixed macro dynamic executor',plan_id=chosen['plan_id'],
        source='Driz Lo pool #027 (generated plan, not the original author agent)',
        fixed='Day0-28 macro intervention sequence',dynamic='Current-state daily planner, task scheduling and action execution',
        online_oracle=False,opponent_identity_input=False,replay_playback=False,
        local_results=results,parity_games=96,official_games=6,source_loader_checks=checks,
        local_single_process_max_wall=smoke['max_wall'],local_parallel_max_wall=parity['max_wall'],
        minimum_glibc='2.34',cpu_target='x86-64 baseline, no AVX required',python_binding='ctypes C ABI; no CPython-version-specific extension',
        archive_sha256=sha(target),archive_bytes=target.stat().st_size,
        files={name:sha(HERE/name) for name in ('main.py','agent.so','frozen_plan.json','frozen_config.json','build_receipt.json','validation_smoke_compat_v1.json','validation_parity_compat_v1.json')},
        caveat='Local tests do not guarantee Kaggle hardware timing or leaderboard strength. Submission status must be checked after upload.')
    (HERE/'PACKAGE_ACCEPTANCE.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()
