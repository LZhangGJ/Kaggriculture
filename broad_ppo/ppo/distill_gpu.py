"""distill-v1 (source-v36): teacher queries on learner seats inside the GPU collector.

For a sampled subset of learner seats (one per sampled game), each of K teacher programs "shadows" that seat for the
whole game: every turn it receives that seat's official observation (ppo.arena_obs via the pinned arena_mux_worker, in
arena sandboxes of its own pool, so arena-opponent timing is untouched) and answers with the action it would play; the
answer is never applied. Answers are converted to the policy's factorized wire rows (ppo.distill_labels.convert, exact,
validated against exact_decoder.prepare_turn) by host label workers (ppo.distill_worker) off the collector's critical
path, and stored next to the rollout as
  teacher/workers int16 [T, K, S, 33, 5], teacher/market int16 [T, K, S, 10, 5], teacher/status int8 [T, K, S] (0 = ok),
  teacher/slot long [2*games] (label slot of a seat, -1 if unlabelled).
"""
import json, os, pickle, queue, random, struct, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STATUS = ('ok', 'fault', 'not_object', 'worker_unsupported', 'worker_quantity_oov', 'worker_nonequivalent', 'market_too_many',
          'market_unsupported', 'market_quantity_oov', 'missing')
# Top internal-arena families (continuous-elo 2026-09-22), one program each; dev-panel families excluded
# (guruprasaathas111 master-engine-v3, alperen5252525, team). Checked again at runtime by select_teachers().
DEFAULT_TEACHERS = ('a93964afb01c',  # nathanjacob pipe18-six-layers, Elo 1948
                    'b1ea30ce8ca6',  # shiiin9 your-market-list-is-an-order-book, 1928
                    'e2dd3e9c6e5a',  # dmitriigluzdov one-more-wheat, 1887
                    '27dfff16b561',  # ahmedberatozer v56-smarter-seeds-and-fertilizer, 1880
                    '0b5eb9560f4e')  # haodou092 notebookdb6965aa8e, 1855


def select_teachers(pool, prefixes, panel_path):
    """Pool rows for the teacher id prefixes; refuses any dev-panel family."""
    panel = json.loads(Path(panel_path).read_text())
    banned = {o['family'].split(':')[-1] for o in panel['opponents']}
    rows = []
    for p in prefixes:
        match = [o for o in pool['opponents'] if o['id'].startswith(p)]
        if len(match) != 1:
            raise ValueError('Teacher %s not uniquely in the arena pool' % p)
        if match[0]['family'] in banned:
            raise ValueError('Teacher %s belongs to dev-panel family %s' % (p, match[0]['family']))
        rows.append(match[0])
    if len({r['family'] for r in rows}) != len(rows):
        raise ValueError('One teacher per family')
    return rows


def _write(stream, value):
    data = pickle.dumps(value, protocol=4)
    stream.write(struct.pack('<Q', len(data)) + data)
    stream.flush()


def _read(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    return pickle.loads(stream.read(struct.unpack('<Q', head)[0]))


class LabelPool:
    """Host processes converting teacher answers to wire rows; turns are dealt round-robin and never block the game."""
    def __init__(self, workers, wq, mq, shape):
        T, K, S = shape
        self.w = np.zeros((T, K, S, 33, 5), np.int16)
        self.m = np.zeros((T, K, S, 10, 5), np.int16)
        self.status = np.full((T, K, S), STATUS.index('missing'), np.int8)
        self.seconds = 0.
        self.procs, self.inbox, self.pending = [], [], 0
        self.lock = threading.Lock()
        self.done = threading.Condition(self.lock)
        env = dict(os.environ, PYTHONPATH=str(HERE.parent) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        for i in range(workers):
            p = subprocess.Popen([sys.executable, '-m', 'ppo.distill_worker'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, cwd=str(HERE.parent), env=env)
            _write(p.stdin, ('init', list(wq), list(mq)))
            q = queue.Queue()
            threading.Thread(target=self._sender, args=(p, q), daemon=True).start()
            threading.Thread(target=self._receiver, args=(p,), daemon=True).start()
            self.procs.append(p)
            self.inbox.append(q)
        self.next = 0

    def _sender(self, p, q):
        while True:
            item = q.get()
            if item is None:
                return
            try:
                _write(p.stdin, item)
            except (OSError, ValueError):
                return

    def _receiver(self, p):
        while True:
            msg = _read(p.stdout)
            if msg is None:
                return
            turn, res, dt = msg
            with self.lock:
                for slot, (w, m, st) in res.items():
                    self.w[turn, :, slot] = w
                    self.m[turn, :, slot] = m
                    self.status[turn, :, slot] = st
                self.seconds += dt
                self.pending -= 1
                self.done.notify_all()

    def submit(self, turn, payload):
        with self.lock:
            self.pending += 1
        self.inbox[self.next].put(('turn', turn, payload))
        self.next = (self.next + 1) % len(self.procs)

    def finish(self, timeout=600):
        deadline = time.time() + timeout
        with self.lock:
            while self.pending > 0 and time.time() < deadline:
                self.done.wait(timeout=5)
            lost = self.pending
        self.close()
        return lost

    def close(self):
        for q in self.inbox:
            q.put(None)
        for p in self.procs:
            try:
                p.kill()
                p.wait(10)
            except Exception:
                pass


class TeacherHook:
    """labeled: list of dict(game, seat (learner seat), job); teachers: pool rows."""
    def __init__(self, labeled, teachers, device, workers, label_workers, wq, mq):
        import torch
        from ppo.arena_gpu import BotMux
        self.labeled, self.teachers, self.device = labeled, teachers, device
        self.K, self.S = len(teachers), len(labeled)
        games = [dict(gid=j * self.K + k, row=t, seed=a['job']['seed'], seat=a['seat'])
                 for j, a in enumerate(labeled) for k, t in enumerate(teachers)]
        job = labeled[0]['job']
        config = job.get('arena_config')
        repo = job.get('arena_repo')
        if config is None:  # self-play jobs carry no arena fields; teachers use the pool's sandbox config
            raise ValueError('TeacherHook needs arena_config/arena_repo on the labelled job dicts')
        prefix = os.environ.get('PPO_ARENA_CONTAINER_PREFIX', 'ppo-practice-gpu-') + 'teacher-'
        t0 = time.perf_counter()
        self.bots = BotMux(games, config, repo, workers, prefix=prefix)
        self.start_seconds = time.perf_counter() - t0
        self.gids = [g['gid'] for g in games]
        self.game_idx = torch.tensor([a['game'] for a in labeled], device=device, dtype=torch.long)
        self.rep = np.repeat(np.arange(self.S), self.K)  # host row of each (slot, teacher) instance
        self.labels = LabelPool(label_workers, wq, mq, (719, self.K, self.S))
        self.seconds = dict(host=0., submit_labels=0.)
        self.host = None

    def before_act(self, turn, state):
        from ppo.arena_gpu import packed_host
        t = time.perf_counter()
        self.host = packed_host(state, self.game_idx)
        rep = {k: v[self.rep] for k, v in self.host.items()}
        self.seconds['host'] += time.perf_counter() - t
        self.bots._turn = turn
        self.bots.submit(rep, self.gids)

    def after_act(self, turn, action):
        res = self.bots.collect(turn)
        t = time.perf_counter()
        payload = {}
        for j, a in enumerate(self.labeled):
            answers = []
            for k in range(self.K):
                r = res.get(j * self.K + k)
                answers.append(None if r is None or r[3] else r[0])
            payload[j] = ({k: v[j:j + 1] for k, v in self.host.items()}, a['seat'], answers)
        self.labels.submit(turn, payload)
        self.seconds['submit_labels'] += time.perf_counter() - t
        return action

    def finish(self):
        import torch
        self.bots.close()
        t = time.perf_counter()
        lost = self.labels.finish()
        wait = time.perf_counter() - t
        L = self.labels
        counts = {STATUS[s]: int(c) for s, c in zip(*np.unique(L.status, return_counts=True))}
        per_teacher = {t['id'][:12]: dict(family=t['family'], ok_rate=round(float((L.status[:, k] == 0).mean()), 4))
                       for k, t in enumerate(self.teachers)}
        data = {'teacher/workers': torch.from_numpy(L.w).to(self.device), 'teacher/market': torch.from_numpy(L.m).to(self.device),
                'teacher/status': torch.from_numpy(L.status).to(self.device)}
        metrics = dict(slots=self.S, teachers=per_teacher, status=counts, lost_turns=lost, teacher_faults=len(self.bots.faults),
                       start_seconds=round(self.start_seconds, 3), bot_wait_seconds=round(self.bots.seconds['wait'], 3),
                       bot_submit_seconds=round(self.bots.seconds['submit'], 3), host_copy_seconds=round(self.seconds['host'], 3),
                       label_submit_seconds=round(self.seconds['submit_labels'], 3), label_cpu_seconds=round(L.seconds, 3),
                       label_tail_seconds=round(wait, 3), workers=len(self.bots.groups),
                       max_worker_turn_ms=round(max(self.bots.max_worker_ms or [0.]), 1),
                       labeled_games=[dict(game=a['game'], seat=a['seat'], family=a['job']['family']) for a in self.labeled])
        return data, metrics

    def close(self):
        self.bots.close()
        self.labels.close()


class HookChain:
    """Runs several per-turn hooks: all before_act first (so every bot pool starts), then all after_act."""
    def __init__(self, hooks):
        self.hooks = [h for h in hooks if h is not None]

    def before_act(self, turn, state):
        for h in self.hooks:
            h.before_act(turn, state)

    def after_act(self, turn, action):
        for h in self.hooks:
            action = h.after_act(turn, action)
        return action


def sample_labeled(jobs, games, seed, teacher_config):
    """Deterministic sample of `games` games, one learner seat each; arena config attached for the teacher sandboxes."""
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(len(jobs)), min(games, len(jobs))))
    out = []
    for g in picks:
        job = jobs[g]
        seat = job['learner_seats'][rng.randrange(len(job['learner_seats']))]
        out.append(dict(game=g, seat=seat, job=dict(job, arena_config=teacher_config['arena_config'], arena_repo=teacher_config['arena_repo'])))
    return out
