"""Pinned official games and CPU feature construction, owned by actor processes."""
import importlib.util
import multiprocessing as mp
import time
import traceback
import os
from types import SimpleNamespace
import numpy as np
from exact_actions import WORKER, needs_quantity, worker_features, worker_quantity_features, worker_wire
from exact_decoder import (ledger_vector, market_state, market_candidates, remember_requests)
from exact_features import encode_exact
from features_v2 import PublicHistory, quantity_features
from bc_runtime import MARKET
from worker_phase import resolve_worker_phase, engine
from ppo.transport import ArrayArena
from ppo.gpu_market import market_seed
from ppo.stream_env import step as stream_step, history_snapshot
from ppo.worker_state import pack as pack_worker_state
from ppo.fast_features import WorkerPrefix, own_farm, worker_stats, market_stats, worker_features_fast, encode_exact_fast


class Games:
    def __init__(self, assignments, wq, mq, stream=False):
        from kaggle_environments import make
        self.wq, self.mq = wq, mq
        self.stream = stream
        self.games, self.seats = {}, {}
        self.sandboxes = []
        self.seconds = dict(environment=0., features=0., script=0.)
        self.seconds_by_command = {}
        for a in assignments:
            env = make('kaggriculture', configuration={'seed':a['seed'], 'episodeSteps':720}, debug=False)
            env.reset(2)
            actual_seed = env.info.get('seed')
            if actual_seed is not None and actual_seed != a['seed']:
                raise ValueError('Official seed does not match requested seed')
            self.games[a['game']] = dict(env=env, assignment=a, scripts={})
            for seat in (0, 1):
                key = f"{a['game']}:{seat}"
                self.seats[key] = dict(history=PublicHistory(), slots=[], ledger=None, obs=None)
                policy = a['policies'][seat]
                if policy.startswith('script:'):
                    name = policy[7:]
                    if name.startswith('arena:'):
                        from ppo.arena_opponents import SandboxOpponent
                        opponent = SandboxOpponent(a)
                        self.sandboxes.append(opponent)
                        self.games[a['game']]['scripts'][seat] = lambda obs, p=opponent, conf=env.configuration: p(obs, conf)
                    elif name in engine().agents:
                        self.games[a['game']]['scripts'][seat] = engine().agents[name]
                    else:
                        # Each game/seat gets separate module globals.
                        spec = importlib.util.spec_from_file_location(f'ppo_script_{a["game"]}_{seat}', name)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        self.games[a['game']]['scripts'][seat] = lambda obs, mod=module, conf=env.configuration: mod.agent(obs, conf)

    def run(self, command, payload):
        start = time.perf_counter()
        result = getattr(self, command)(payload)
        elapsed = time.perf_counter()-start
        self.seconds['environment' if command == 'step' else 'features'] += elapsed
        self.seconds_by_command[command] = self.seconds_by_command.get(command,0.) + elapsed
        return result

    def start(self, keys):
        out = {}
        for key in keys:
            game, seat = map(int, key.split(':'))
            env = self.games[game]['env']
            if env.done: continue
            state = self.seats[key]
            obs = dict(env.state[seat].observation)
            obs['step'] = env.state[0].observation.step
            # GPU collection resolves the complete worker program once below.
            # Do not deep-copy and project an unused empty CPU prefix first.
            prefix = (SimpleNamespace(resolved=None) if os.environ.get('PPO_GPU_WORKERS') == '1'
                      else WorkerPrefix(obs))
            state.update(obs=obs, slots=[], ledger=None, prefix=prefix)
            state['history'].observe(obs)
            out[key] = dict(x=encode_exact_fast(obs, state['history']),
                            workers=1+len(obs['farms'][seat]['hands']), step=obs['step'])
            if os.environ.get('PPO_GPU_WORKERS') == '1':
                out[key]['worker_seed'] = pack_worker_state(obs)
        return out

    def worker(self, keys):
        out = {}
        for key in keys:
            s = self.seats[key]
            depth = len(s['slots'])
            provisional = s['prefix'].resolved
            compact = worker_features_fast(s['obs'], provisional, s['slots'], depth,s['prefix'].points())
            need = np.array([needs_quantity(provisional, depth, i) for i in range(len(WORKER))])
            out[key] = dict(compact=compact, need=need, qstats=worker_stats(provisional, depth))
        return out

    def worker_apply(self, choices):
        for key, (index, quantity, has_q) in choices.items():
            s = self.seats[key]
            wire = worker_wire(index, quantity if has_q else None)
            s['slots'].append(wire)
            s['prefix'].append(wire)
        return {}

    def worker_next(self, choices):
        self.worker_apply(choices)
        keys = [key for key in choices
                if len(self.seats[key]['slots']) < 1 + len(self.seats[key]['obs']['farms'][int(key.split(':')[1])]['hands'])]
        return self.worker(keys)

    def market_begin(self, keys):
        out = {}
        for key in keys:
            s = self.seats[key]
            resolved = s['prefix'].resolved
            s['ledger'] = market_state(resolved)
            out[key] = dict(post_ledger=ledger_vector(s['ledger']),
                            post_farm=own_farm(resolved),market_seed=market_seed(s["ledger"]))
        return out

    def worker_sequence(self, choices):
        for key, rows in choices.items():
            s = self.seats[key]
            s['slots'] = [worker_wire(index, quantity if need else None) for index, quantity, need in rows]
            s['prefix'].resolved = resolve_worker_phase(s['obs'], s['slots'])[0]
        return self.market_begin(list(choices))

    def market(self, keys):
        out = {}
        for key in keys:
            ledger = self.seats[key]['ledger']
            need = np.array([i>1 and item is not None for i, (_, item) in enumerate(MARKET)])
            out[key] = dict(candidates=market_candidates(ledger), ledger=ledger_vector(ledger),
                            need=need, qstats=market_stats(ledger))
        return out

    def market_apply(self, choices):
        out = {}
        for key, (index, quantity) in choices.items():
            ledger = self.seats[key]['ledger']
            before = ledger_vector(ledger)
            if index: ledger.add_order(index, quantity)
            out[key] = ledger_vector(ledger)-before
        return out

    def market_sequence(self, choices):
        for key,orders in choices.items():
            for index,quantity in orders:
                self.seats[key]["ledger"].add_order(index,quantity)
        return {}

    def finish(self, keys):
        for key in keys:
            s = self.seats[key]
            remember_requests(s['history'], s['ledger'].orders)
        return {}

    def step(self, games):
        out = {}
        for game in games:
            entry = self.games[game]
            env = entry['env']
            actions = []
            for seat in (0, 1):
                if seat in entry['scripts']:
                    obs = dict(env.state[seat].observation)
                    obs['step'] = env.state[0].observation.step
                    from kaggle_environments.utils import structify
                    tick = time.perf_counter()
                    action = entry['scripts'][seat](structify(obs))
                    self.seconds['script'] += time.perf_counter()-tick
                else:
                    s = self.seats[f'{game}:{seat}']
                    action = dict(farmer=s['slots'][0], hands=s['slots'][1:], market=s['ledger'].orders)
                actions.append(action)
            if self.stream:
                for seat in (0, 1):
                    history = self.seats[f'{game}:{seat}']['history']
                    if history.previous is not None:
                        history.previous = history_snapshot(history.previous)
                stream_step(env, actions)
            else:
                env.step(actions)
            statuses = [s.status for s in env.state]
            # Policy faults are distinct from infrastructure exceptions, which propagate.
            faults = [s in ('INVALID', 'ERROR', 'TIMEOUT') for s in statuses]
            done = env.done
            cash = [s.reward for s in env.state] if done else None
            if done and not any(faults):
                if statuses != ['DONE','DONE']:raise RuntimeError(f'Unclassified official terminal: {statuses}')
                if cash is None or len(cash)!=2 or not np.isfinite(cash).all():
                    raise FloatingPointError(f'Nonfinite official terminal cash: {cash}')
            farms = env.state[0].observation['farms']
            out[game] = dict(done=done, faults=faults, statuses=statuses, cash=cash,
                             money=[float(farms[i]['money']) for i in (0, 1)],  # shaped-reward-v6: own cash after this turn
                             turns=int(env.state[0].observation.step))
        return out

    def close(self):
        errors=[]
        for opponent in getattr(self,'sandboxes',[]):
            try:opponent.close()
            except BaseException as exc:errors.append(exc)
        self.sandboxes=[]
        if errors:raise RuntimeError('Opponent cleanup failed: '+repr(errors[0])) from errors[0]

    def timing(self, unused): return dict(**self.seconds, by_command=dict(self.seconds_by_command))


def actor_main(connection, name, assignments, wq, mq):
    import torch
    torch.set_num_threads(1)
    arena = ArrayArena(name=name)
    games = Games.__new__(Games)
    try:
        games.__init__(assignments, wq, mq, stream=True)
        connection.send(('ready', None))
        while True:
            command, payload = connection.recv()
            if command == 'close': break
            try:
                response = games.run(command, payload)
                spec, size = arena.write(response)
                connection.send(('ok', spec, size))
            except Exception:
                connection.send(('error', traceback.format_exc()))
                break
    except Exception:
        connection.send(('error', traceback.format_exc()))
    finally:
        for resource in (games,arena,connection):
            try:resource.close()
            except BaseException:traceback.print_exc()


class ActorPool:
    def __init__(self, assignments, wq, mq, workers=8, arena_mib=32, timeout=120):
        self.channels, self.owners = [], {}
        self.timeout = timeout
        self.arena_names = []
        import uuid
        for a in assignments:
            if a.get("arena_opponent"):
                a["arena_container_name"] = f"ppo-practice-{os.getpid()}-" + uuid.uuid4().hex
                self.arena_names.append(a["arena_container_name"])
        self.ipc_seconds, self.bytes = 0., 0
        self.bytes_by_command = {}; self.calls_by_command = {}; self.seconds_by_command = {}
        ctx = mp.get_context('spawn')
        groups = [assignments[i::min(workers,len(assignments))] for i in range(min(workers,len(assignments)))]
        try:
            for group in groups:
                arena = ArrayArena(size=arena_mib*1024*1024)
                parent, child = ctx.Pipe()
                process = ctx.Process(target=actor_main, args=(child, arena.shm.name, group, wq, mq))
                process.start()
                child.close()
                self.channels.append((parent, process, arena))
                for a in group: self.owners[a['game']] = len(self.channels)-1
            for c, _, _ in self.channels:
                if not c.poll(timeout): raise TimeoutError('CPU actor startup timed out')
                result = c.recv()
                if result[0] != 'ready': raise RuntimeError(result)
        except BaseException:
            try:self.close()
            except BaseException:traceback.print_exc()
            raise

    def call(self, command, payload):
        tick = time.perf_counter()
        groups = {}
        for key in payload:
            game = int(str(key).split(':')[0])
            owner = self.owners[game]
            if isinstance(payload, dict): groups.setdefault(owner, {})[key] = payload[key]
            else: groups.setdefault(owner, []).append(key)
        for i, group in groups.items(): self.channels[i][0].send((command, group))
        out = {}
        for i in groups:
            c, _, arena = self.channels[i]
            if not c.poll(self.timeout): raise TimeoutError(f'CPU actor timeout in {command}')
            result = c.recv()
            if result[0] != 'ok': raise RuntimeError(result[1])
            out.update(arena.read(result[1], copy=command not in ('start','worker','worker_next','market')))
            self.bytes_by_command[command] = self.bytes_by_command.get(command,0) + result[2]
            self.bytes += result[2]
        self.calls_by_command[command] = self.calls_by_command.get(command,0) + 1
        elapsed = time.perf_counter()-tick
        self.ipc_seconds += elapsed
        self.seconds_by_command[command] = self.seconds_by_command.get(command,0.) + elapsed
        return out

    def close(self):
        errors=[]
        try:
            for c,p,arena in self.channels:
                try:
                    if p.is_alive():
                        try:c.send(('close',None))
                        except (BrokenPipeError,EOFError,OSError):pass
                        p.join(timeout=3)
                        if p.is_alive():p.terminate();p.join(timeout=3)
                        if p.is_alive():p.kill();p.join(timeout=3)
                        if p.is_alive():raise RuntimeError('Actor survived kill/join')
                except BaseException as exc:errors.append(exc)
                finally:
                    for resource in (c,arena):
                        try:resource.close()
                        except BaseException as exc:errors.append(exc)
        finally:
            self.channels=[]
            if self.arena_names:
                import subprocess
                try:
                    result=subprocess.run(['docker','rm','-f',*self.arena_names],capture_output=True,timeout=60)
                    # Already removed containers return nonzero; verify actual absence.
                    remaining=subprocess.run(['docker','ps','-a','--format','{{.Names}}'],capture_output=True,text=True,check=True,timeout=15).stdout.splitlines()
                    if set(self.arena_names)&set(remaining):raise RuntimeError('Owned arena containers survived cleanup')
                    self.arena_names=[]
                except BaseException as exc:errors.append(exc)
        if errors:raise RuntimeError('Actor cleanup failed: '+repr(errors[0])) from errors[0]

    def timings(self):
        result=[]
        for c,_,_ in self.channels: c.send(('timing',None))
        for c,_,arena in self.channels:
            if not c.poll(self.timeout): raise TimeoutError('Actor timing response timed out')
            message=c.recv()
            if message[0]!='ok': raise RuntimeError(message)
            result.append(arena.read(message[1]))
        return result
