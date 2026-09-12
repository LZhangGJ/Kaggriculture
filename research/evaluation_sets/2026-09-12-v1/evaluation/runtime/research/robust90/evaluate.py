"""Bounded CPU development games using the release's official interpreter host."""
from __future__ import annotations
import argparse
import copy
import ctypes
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'nt/latest_20260911_p16_jointafs_r1/agent'
BASELINE = ROOT / 'nt/latest_20260911_afs_workflow_repair_r1_r2/r2'


def worker(job):
    opponent, seed, seat, binary, config, trace = job
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def expired(signum, frame):
        raise TimeoutError('Development game exceeded 180 seconds')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    sys.path.insert(0, str(PACKAGE))
    import arena
    try:
        if opponent.startswith(('native:', 'extra:', 'stress:')) or any(k in config for k in ('__switch','__selector','__engine','__prefix_selectors')):
            return extended_game(arena, opponent, seed, seat, binary, config, trace)
        return arena.game((opponent, seed, seat, binary, 'native', config, 719, trace))
    finally:
        signal.alarm(0)


def fork_profiles(profiles,callback):
    """Offline only: clone exact actor and simulator state, one child at a time."""
    results=[]
    for name,changes in profiles.items():
        readfd,writefd=os.pipe();pid=os.fork()
        if pid==0:
            os.close(readfd)
            signal.alarm(180)
            try:
                value=callback(name,changes)
            except BaseException:
                value=dict(candidate=name,runtime_error=traceback.format_exc())
            with os.fdopen(writefd,'w') as stream:json.dump(value,stream)
            os._exit(0)
        os.close(writefd)
        with os.fdopen(readfd) as stream:payload=stream.read()
        _,status=os.waitpid(pid,0)
        if status!=0 or not payload:
            raise RuntimeError(f'Branch child {name} failed: status {status}')
        result=json.loads(payload)
        if result.get('runtime_error'):raise RuntimeError(result['runtime_error'])
        results.append(result)
    return results


def extended_game(arena, opponent, seed, seat, binary, config, trace):
    start=time.monotonic(); own=rival=env=None
    row=dict(opponent=opponent,seed=seed,opponent_seat=seat,binary=binary,runtime_error=None)
    try:
        config=config.copy();switch=config.pop('__switch',None);selector_path=config.pop('__selector',None);engine_kind=config.pop('__engine','official')
        row['engine']=engine_kind
        packaged_path=config.pop('__package',None)
        branch_profiles=config.pop('__branch_profiles',None)
        capture_only=config.pop('__capture_only',False)
        capture_history=config.pop('__history_features',False)
        capture_profiles=config.pop('__plan_profiles',None)
        future_seed=config.pop('__future_seed',None)
        restore_base_day=config.pop('__restore_base_day',None)
        initial_pass_steps=config.pop('__initial_pass_steps',0)
        market_order_rule=config.pop('__market_order_rule',None)
        if market_order_rule is not None:
            if market_order_rule!='value':raise ValueError('Unknown market order rule')
            row.update(market_order_rule=market_order_rule,market_order_changes=0)
        suffix_paths=config.pop('__suffix_selectors',[])
        suffix_models=[json.loads(Path(p).read_text()) for p in suffix_paths]
        sparse_observations=config.pop('__sparse_observations',False)
        if sparse_observations and engine_kind not in ('direct','direct_future'):
            raise ValueError('Sparse observations require a direct engine; parity keeps every observation check')
        if future_seed is not None and (not switch or engine_kind not in ('direct_future','direct_future_parity','direct_common','direct_common_parity','direct_common_intraday','direct_common_intraday_parity')):
            raise ValueError('Future resampling requires a named offline engine and decision day')
        prefix_paths=config.pop('__prefix_selectors',[])
        prefix_models=[json.loads(Path(p).read_text()) for p in prefix_paths]
        row['engine']=engine_kind
        row['sparse_observations']=bool(sparse_observations)
        selector_model=json.loads(Path(selector_path).read_text()) if selector_path else None
        if selector_model:
            program=selector_model.get('policy_program')
            if program:
                if program.get('version')!=1 or program['switch_day']!=selector_model['day'] or initial_pass_steps:
                    raise ValueError('Unsupported selector program')
                actual_hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in suffix_paths]
                if restore_base_day!=program['restore_base_day'] or actual_hashes!=program['suffix_sha256']:
                    raise ValueError('Selector continuation schedule mismatch')
            if hashlib.sha256(Path(binary).read_bytes()).hexdigest()!=selector_model['binary_sha256']:
                raise ValueError('Selector binary mismatch')
            switch={'day':selector_model['day'],'config':{}}
            if selector_model.get('prefix_selector_sha256',[])!=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in prefix_paths]:
                raise ValueError('Selector prefix chain mismatch')
        if prefix_models:
            days_prefix=[m['day'] for m in prefix_models]
            if len(set(days_prefix))!=len(days_prefix) or days_prefix!=sorted(days_prefix) or not switch or days_prefix[-1]>=switch['day']:
                raise ValueError('Invalid ordered selector prefix')
            if any(m['binary_sha256']!=hashlib.sha256(Path(binary).read_bytes()).hexdigest() for m in prefix_models):raise ValueError('Prefix binary mismatch')
            prefix_hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in prefix_paths]
            for i,m in enumerate(prefix_models):
                if m.get('prefix_selector_sha256',[])!=prefix_hashes[:i]:raise ValueError('Earlier selector chain mismatch')
        if restore_base_day is not None:
            if not switch or not int(switch['day'])<restore_base_day<30:raise ValueError('Invalid temporary intervention')
            row['restore_base_day']=restore_base_day
        if suffix_models:
            # Explicit policy-program experiment: retain a frozen later actor.
            # Its original training lineage is preserved, not rewritten.
            ds=[m['day'] for m in suffix_models]
            if len(suffix_models)!=1 or not switch or ds[0]<=switch['day'] or restore_base_day is None or ds[0]<=restore_base_day:
                raise ValueError('Suffix requires one later selector after base restoration')
            for m in suffix_models:
                if m.get('prefix_selector_sha256') or m['base_config']!=config or m['binary_sha256']!=hashlib.sha256(Path(binary).read_bytes()).hexdigest():
                    raise ValueError('Unsupported transferred continuation')
            row['experimental_suffix_transfer']=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in suffix_paths]
        scheduled_models=prefix_models+suffix_models
        switch_step=int(switch.get('step',24*switch['day'])) if switch else None
        if switch and (switch_step<0 or switch_step//24!=switch['day']):raise ValueError('Invalid switch clock')
        if not isinstance(initial_pass_steps,int) or not 0<=initial_pass_steps<24 or initial_pass_steps and (switch_step is None or initial_pass_steps>switch_step):
            raise ValueError('Invalid observation delay')
        if initial_pass_steps:row['initial_pass_steps']=initial_pass_steps
        if packaged_path:
            if switch or selector_model or scheduled_models or restore_base_day is not None:raise ValueError('Packaged policy owns its selector chain')
            packaged=arena.module(Path(packaged_path)/'main.py','robust90_packaged_candidate')
            own=packaged.create_agent()
        else:
            own=arena.codec.Agent(config=config,binary_path=binary)
        def reconfigure(changes):
            # Each profile has the original settings as its reference. Keep the
            # accumulated farm and execution state, not leftover profile biases.
            own.config.update(config);own.config.update(changes)
            params=(ctypes.c_double*len(arena.codec._ORDER))(*(own.config[k] for k in arena.codec._ORDER))
            own.lib.td_reconfigure.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]
            own.lib.td_reconfigure.restype=ctypes.c_int
            if own.lib.td_reconfigure(own.handle,params,len(params))!=0:raise RuntimeError('reconfigure failed')
        if opponent.startswith('stress:'):
            from stress import PROFILES
            stress_config=json.loads((BASELINE/'policy/config.json').read_text())
            stress_config.update(PROFILES[opponent.split(':',1)[1]])
            rival=arena.codec.Agent(config=stress_config,binary_path=BASELINE/'policy/agent.so');call=rival
        elif opponent.startswith('native:'):
            name=opponent.split(':',1)[1]
            directory=BASELINE.parent/name
            rival=arena.codec.Agent(config=json.loads((directory/'policy/config.json').read_text()),binary_path=directory/'policy/agent.so')
            call=rival
        elif opponent.startswith('extra:'):
            directory=ROOT/'research/robust90/opponents'/opponent.split(':',1)[1]
            # One isolated module per game; source and native step-zero reset remain intact.
            mod=arena.module(directory/'main.py','robust90_extra_opponent')
            fn=mod.agent
            try:
                inspect.signature(fn).bind({}, {})
            except TypeError:
                inspect.signature(fn).bind({})
                call=lambda obs, cfg:fn(obs)
            else:
                call=fn
        else:
            rival=arena.FreshPublic(opponent);call=rival.call
        engine_class=arena.LocalGame
        if engine_kind!='official':
            from native_env import NativeGame,ParityGame,DirectGame,DirectParityGame,FutureDirectGame,FutureDirectParityGame
            if engine_kind.startswith('direct_common'):
                from common_env import CommonDirectGame,CommonDirectParityGame,IntradayCommonDirectGame,IntradayCommonDirectParityGame
                engine_class={'direct_common':CommonDirectGame,'direct_common_parity':CommonDirectParityGame,
                    'direct_common_intraday':IntradayCommonDirectGame,'direct_common_intraday_parity':IntradayCommonDirectParityGame}[engine_kind]
            else:
                engine_class={'native':NativeGame,'parity':ParityGame,'direct':DirectGame,'direct_parity':DirectParityGame,'direct_future':FutureDirectGame,'direct_future_parity':FutureDirectParityGame}[engine_kind]
        env=engine_class(seed,arena.load_engine()); actions=[]; days=[]; maximum=0.
        def play(branches=None):
            nonlocal maximum
            while not env.done and env.t<719:
                if restore_base_day is not None and env.t==restore_base_day*24:
                    reconfigure({});row['base_restored_step']=env.t
                decision_turn=bool(switch and env.t==switch_step) or any(env.t==p['day']*24 for p in scheduled_models)
                if not sparse_observations or decision_turn or market_order_rule or trace and env.t%24==0:
                    obs=[env.observation(s) for s in (0,1)]
                else:
                    # Native policies build their legal View inside agent_act.
                    # Only Python actors need serialized observations here.
                    obs=[None,None]
                    if packaged_path:obs[1-seat]=env.observation(1-seat)
                    if not opponent.startswith(('native:','stress:')):obs[seat]=env.observation(seat)
                policy_start=time.monotonic()
                for prefix in scheduled_models:
                    if env.t==prefix['day']*24:
                        from features import policy_features
                        from selector import choose
                        f=policy_features(obs[1-seat],own.config)
                        if any(k.startswith('history_') for k in prefix['features']):
                            from features import history_features
                            f.update(history_features(own))
                        if any(k.startswith('plan_') for k in prefix['features']):
                            from features import plan_features
                            f.update(plan_features(own,obs[1-seat],arena.codec,prefix['profiles'],prefix['base_config']))
                        name,changes,scores=choose(prefix,f)
                        row.setdefault('prefix_profiles',[]).append(dict(day=prefix['day'],name=name))
                        reconfigure(changes)
                if switch and env.t==switch_step:
                    from features import policy_features
                    row['decision_features']=policy_features(obs[1-seat],own.config)
                    if capture_history or selector_model and any(k.startswith('history_') for k in selector_model['features']):
                        from features import history_features
                        row['decision_features'].update(history_features(own))
                    if capture_profiles or selector_model and any(k.startswith('plan_') for k in selector_model['features']):
                        from features import plan_features
                        profiles=capture_profiles or selector_model['profiles']
                        row['decision_features'].update(plan_features(own,obs[1-seat],arena.codec,profiles,config))
                    row['decision_observation_hash']=arena.digest(obs[1-seat])
                    if capture_only:
                        row.update(prefix_only=True,steps=env.t,action_hash=arena.digest(actions),seconds=time.monotonic()-start)
                        return row
                    if branches:
                        def continue_choice(name,changes):
                            row['candidate']=name;switch['config']=changes
                            return play(None)
                        return fork_profiles(branches,continue_choice)
                    if future_seed is not None:
                        env.reseed_future(future_seed)
                        row.update(future_seed=future_seed,future_reseed_step=env.t,scope='offline resampled continuation; not a primary evaluation game')
                    if selector_model:
                        from selector import choose
                        name,changes,scores=choose(selector_model,row['decision_features'])
                        row['selected_profile']=name;row['selection_scores']=scores
                        switch['config']=changes
                    reconfigure(switch['config'])
                if env.t<initial_pass_steps:
                    a=dict(farmer=['PASS'],hands=[],market=[])
                else:
                    a=env.agent_act(own,1-seat) if hasattr(env,'agent_act') and not packaged_path else own(obs[1-seat])
                if market_order_rule:
                    from market_orders import order_sales
                    ordered=order_sales(a,obs[1-seat]['market']['prices'],market_order_rule)
                    row['market_order_changes']+=ordered is not a
                    a=ordered
                maximum=max(maximum,time.monotonic()-policy_start)
                b=env.agent_act(rival,seat) if hasattr(env,'agent_act') and opponent.startswith(('native:','stress:')) else call(obs[seat],copy.deepcopy(env.configuration))
                issued=[None,None];issued[1-seat]=a;issued[seat]=b
                if trace and env.t%24==0: days.append(dict(step=env.t,observations=obs,debug=own.debug()))
                actions.append(copy.deepcopy(issued));env.advance(issued)
            if env.t!=719 or not env.done: raise RuntimeError('Incomplete terminal game')
            obs=env.observation(0);cash=[f['money'] for f in obs['farms']];margin=cash[1-seat]-cash[seat]
            if engine_kind in ('parity','direct_parity','direct_future_parity','direct_common_parity','direct_common_intraday_parity'):
                env.observation(1);row['observations_checked']=env.observations_checked
            row.update(steps=env.t,own_cash=cash[1-seat],opponent_cash=cash[seat],margin=margin,win=margin>0,tie=margin==0,
                       action_hash=arena.digest(actions),latency_max=maximum,debug=own.debug(),terminal_shops=obs['town']['unlocked_shops'])
            if trace:
                import gzip
                Path(trace).write_bytes(gzip.compress(json.dumps(dict(result=row,days=days,actions=actions)).encode()))
            row['seconds']=time.monotonic()-start
            return row
        return play(branch_profiles)
    except Exception:
        row['runtime_error']=traceback.format_exc()
    finally:
        if own:own.close()
        if rival:rival.close()
        if env and hasattr(env,'close'):env.close()
    row['seconds']=time.monotonic()-start
    return row


def summarize(rows):
    good = [r for r in rows if not r.get('runtime_error') and r.get('steps') == 719]
    wins = sum(r['win'] for r in good)
    ties = sum(r['tie'] for r in good)
    return dict(games=len(rows), complete=len(good), wins=wins, ties=ties,
                failures=len(rows)-len(good), win_rate=wins/len(rows) if rows else None,
                mean_cash=sum(r['own_cash'] for r in good)/len(good) if good else None,
                mean_margin=sum(r['margin'] for r in good)/len(good) if good else None,
                max_action_seconds=max((r['latency_max'] for r in good), default=None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed-start', type=int, default=2612001000)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--binary', type=Path, default=BASELINE/'policy/agent.so')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--opponents', nargs='+')
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--sparse-observations',action='store_true')
    parser.add_argument('--engine',choices=['official','native','parity','direct','direct_parity'],default='official')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16: raise ValueError('worker limit is 1..16')
    args.out = args.out.resolve()
    args.binary = args.binary.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'rows.jsonl').exists(): raise FileExistsError('Use a new result directory')
    config = json.loads((BASELINE/'policy/config.json').read_text())
    if args.config: config.update(json.loads(args.config.read_text()))
    if args.engine!='official':config['__engine']=args.engine
    if args.sparse_observations:config['__sparse_observations']=True
    opponents = args.opponents or (json.loads(args.pool.read_text())['primary_opponents'] if args.pool else [x['id'] for x in json.loads((PACKAGE/'POOL.json').read_text())])
    jobs = [(opp, seed, seat, str(args.binary), config,
             str(args.out/f'trace-{opp.replace(":", "_")}-{seed}-{seat}.json.gz') if args.trace else None)
            for seed in range(args.seed_start, args.seed_start+args.seeds)
            for opp in opponents for seat in (0, 1)]
    manifest = dict(binary=str(args.binary), binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    config=config, opponents=opponents, seeds=list(range(args.seed_start,args.seed_start+args.seeds)),
                    seats=[0,1], workers=args.workers, purpose='development; not held-out confirmation',
                    scheduled=len(jobs), started=time.time())
    manifest['engine']=args.engine
    manifest['host_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('features.py')]}
    if args.engine!='official':manifest['native_engine_sha256']=hashlib.sha256(Path(__file__).with_name('native_direct_host.so' if args.engine.startswith('direct') else 'native_host.so').read_bytes()).hexdigest()
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    snapshot=args.out/'host_source_snapshot';snapshot.mkdir()
    for name in ('evaluate.py','features.py','selector.py','native_env.py','native_host.cpp','native_host.so','native_direct_host.cpp','native_direct_host.so','stress.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    if config.get('__market_order_rule'):
        shutil.copy2(Path(__file__).with_name('market_orders.py'),snapshot/'market_orders.py')
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(worker,j):j for j in jobs}
        with (args.out/'rows.jsonl').open('w') as out:
            for future in as_completed(pending):
                row=future.result(); rows.append(row)
                out.write(json.dumps(row)+'\n'); out.flush()
                print(json.dumps(dict(done=len(rows), scheduled=len(jobs), opponent=row['opponent'],
                                      seed=row['seed'], opponent_seat=row['opponent_seat'],
                                      margin=row.get('margin'), error=row.get('runtime_error'))), flush=True)
    result=dict(summary=summarize(rows), by_opponent={o:summarize([r for r in rows if r['opponent']==o]) for o in opponents},
                by_seat={str(s):summarize([r for r in rows if 1-r['opponent_seat']==s]) for s in (0,1)},
                elapsed=time.time()-manifest['started'])
    (args.out/'RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
