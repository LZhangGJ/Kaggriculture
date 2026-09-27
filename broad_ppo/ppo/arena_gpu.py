"""arena-in-gpu-v1 (source-v26): real arena programs as opponent seats inside the GPU collector.

gpu_sim steps every game; the learner acts through the normal GPU roles; an arena game's opponent seat is "external":
each turn the collector copies the arena games' state to the host in one packed transfer, the bot mux workers (arena
sandbox containers, several games each) rebuild the official observation (arena_obs, byte-identical to the official
engine), run the real bot and return its action as gpu_sim rows, which are written into the opponent seats before the
simulator step. Bots run while the GPU computes the learner's action.

Evidence (search-20260922/arena-speed): 96 official games x 719 turns observation-identical and cash-exact in gpu_sim;
live bots in the mux reproduce the official docker actions on every turn; hybrid games replayed through the official
engine give identical bot actions and cash.

Opponent faults (exception, non-object action, >turn timeout, worker crash/hang) end that bot's play (empty actions) and
the game's learner episode is discarded, as the official collector discards opponent-fault games.
A sampled official-engine replay of a few games per collection runs in a detached subprocess (ppo.arena_parity) and
appends a diagnostic line; it never raises into training.
"""
import gzip, hashlib, json, os, pickle, queue, random, struct, subprocess, sys, threading, time, uuid
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MUX_SHA256 = '6ea02ca7972b774b1274a80d77875a2d2b762e8ca86b3ee25b50d6f172c1a73b'  # source-v42 (v26: 02d32c64...)
OBS_SHA256 = 'ea414ae41a7ee5285c080aac6bfdb7ae1b43843a2c5daca3ccf709c09c9b4c73'
# gpu-bots-v1 (source-v72): ported bots (bit-exact batched ports of pool bots) served by host port workers
PORTS_DIR = HERE.parent / 'ppo_ports'
PORTS_SHA256 = 'e760603689989f32cc6da222453a594084e4f521b25844b212fff16e5f2b2b54'
PORT_WORKERS = int(os.environ.get('PPO_PORT_WORKERS', '32'))


def _ports_manifest():
    h = hashlib.sha256()
    for p in sorted(x for x in PORTS_DIR.rglob('*') if x.is_file() and '__pycache__' not in x.parts):
        h.update(str(p.relative_to(PORTS_DIR)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def ported_ids():
    if os.environ.get('PPO_ARENA_PORTS', '1') != '1' or not PORTS_DIR.exists():
        return {}
    if _ports_manifest() != PORTS_SHA256:
        raise ValueError('ppo_ports changed: %s' % _ports_manifest())
    return json.loads((PORTS_DIR / 'ported.json').read_text())   # bot12 -> group


def plan_port_workers(games, workers, groups):
    """source-v72: exactly ONE group (bot) per port worker, never merged: bots of different lineages publish the same
    shared-state keys in different formats (a merged d968 + E-lineage worker raised AttributeError in the u1644 trial).
    `workers` is ignored (kept for the signature)."""
    by = {}
    for g in games: by.setdefault(groups[g['row']['id'][:12]], []).append(g)
    return [gs for _, gs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
ROWS = ('unit_op', 'unit_item', 'unit_amount', 'unit_count', 'market_op', 'market_item', 'market_amount', 'market_count')
WIDTHS = (33, 33, 33, 1, 10, 10, 10, 1)
FIELDS = ('step', 'money', 'tile_kind', 'tile_crop', 'tile_animal', 'tile_origin_day', 'tile_yield', 'tile_neglect',
          'tile_max_lifespan', 'tile_fertilized_until', 'tile_pending_care', 'tile_flags', 'unit_pos', 'unit_active',
          'unit_inventory', 'unit_inventory_order', 'shed', 'seeds', 'hires_today', 'unlocked_count',
          'market_inventory', 'market_price', 'town_shops', 'town_count')
CONF = {'seed': None, 'episodeSteps': 720, 'actTimeout': 1, 'runTimeout': 1200, 'boardSize': 10, 'startingMoney': 3000,
        'maxMarketOrdersPerTurn': 10, 'turnsPerDay': 24, 'shedCapacity': 100, 'weedSpawnChance': 0.005,
        'townShopUnlockInterval': 3, 'townShopSellInterval': 4, 'townCenterSellInterval': 24, 'farmHandCostMult': 1, 'marketParams': {}}
# Measured sandbox CPython ms/turn per program (arena-speed sweep, 3 full games each); unknown programs get DEFAULT_MS.
BOT_MS = {'0af437c2cdce': 0.63, '0b5eb9560f4e': 1.11, '1906a69871a3': 5.09, '1f8c404afcc0': 0.68, '235bd66241b9': 2.75, '272c21f7b664': 2.91,
          '27dfff16b561': 1.02, '3aec904ff8db': 0.91, '3e5c42bb8ed6': 4.35, '41648797c0c0': 1.04, '5e6ae5bd5e4f': 0.85, '70ba5f58a74f': 0.6,
          '79d5c88570fd': 3.0, '92376f2c47a8': 8.82, '9a380bdc884a': 3.74, 'a93964afb01c': 1.21, 'ae20476d9a36': 1.14, 'b0517d490657': 1.14,
          'b1ea30ce8ca6': 1.18, 'b4155bce1845': 0.67, 'ba26025f5486': 0.67, 'c2d3f5b73506': 1.14, 'd5f2a7c17717': 20.61, 'd8f9229d205e': 9.29,
          'd96852632f37': 1.02, 'dab37c280437': 0.61, 'e0bb8ef97093': 2.06, 'e2dd3e9c6e5a': 1.04, 'ef07d0f404e0': 2.04, 'efd5c340f44c': 0.9,
          'f2867b37786c': 0.9, 'fdc5199330b1': 0.64}
DEFAULT_MS = 3.0
DEDICATED_FAMILIES = ('github-101642208',)
DEDICATED_MS = 5.0
GAME_MB = 160  # sandbox memory per hosted game, on top of 512 MiB for the worker itself
SUBMIT_THREAD = os.environ.get('PPO_ARENA_SUBMIT_THREAD', '1') == '1'  # source-v42: overlap bot submit with the GPU act
PLAN = os.environ.get('PPO_ARENA_PLAN', 'lpt-v2')  # source-v42: 'lpt-v2' (measured costs + per-game overhead) or 'v26'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pins():
    got = dict(mux=file_sha(HERE / 'arena_mux_worker.py'), obs=file_sha(HERE / 'arena_obs.py'))
    if got != dict(mux=MUX_SHA256, obs=OBS_SHA256):
        raise ValueError('arena-in-gpu worker files changed: %s' % got)
    return got


def cost(game):
    # source-v47: a pool row may carry its own measured 'bot_ms' (heavy teammate programs); otherwise the v46 table/default.
    return float(game['row'].get('bot_ms') or BOT_MS.get(game['row']['id'][:12], DEFAULT_MS))


def game_mb(game):
    # source-v47: a pool row may declare 'memory_mb' per hosted game (measured peak RSS); otherwise GAME_MB as in v46.
    return int(game['row'].get('memory_mb') or GAME_MB)


GAME_OVERHEAD_MS = 0.6  # measured per hosted game per turn besides the bot call: observation rebuild, (de)serialisation


def plan_workers(games, workers):
    """source-v42 'lpt-v2': one LPT over all games with cost = BOT_MS + per-game overhead. The per-turn critical path is the
    slowest worker; v26's dedicated github-101642208 workers held 1-3 games each while the other workers held 15-17, so
    the light workers (and the spiky non-dedicated programs among them) set the pace. Grouping has no semantic effect:
    every game keeps its own namespace, cwd and random state inside a worker (replay hashes are identical)."""
    if PLAN == 'v26':
        return plan_workers_v26(games, workers)
    workers = max(1, min(workers, len(games)))
    bins = [[0., i, []] for i in range(workers)]
    for g in sorted(games, key=lambda g: (-cost(g), g['gid'])):
        b = min(bins, key=lambda x: (x[0], x[1]))
        b[0] += cost(g) + GAME_OVERHEAD_MS
        b[2].append(g)
    return [b[2] for b in bins if b[2]]


def plan_workers_v26(games, workers):
    """Cost-balanced (LPT) assignment; slow github-101642208 programs get dedicated workers."""
    workers = max(1, min(workers, len(games)))
    heavy = [g for g in games if g['row']['family'] in DEDICATED_FAMILIES and cost(g) >= DEDICATED_MS]
    light = [g for g in games if g not in heavy]
    groups = []
    if heavy and light and workers > 1:
        share = sum(map(cost, heavy)) / sum(map(cost, games))
        n_heavy = min(len(heavy), workers - 1, max(1, round(workers * share)))
    else:
        n_heavy = workers if heavy and not light else 0
    for part, n in ((heavy, n_heavy), (light, workers - n_heavy)):
        if not part or n <= 0:
            continue
        bins = [[0., []] for _ in range(n)]
        for g in sorted(part, key=cost, reverse=True):
            b = min(bins, key=lambda x: x[0])
            b[0] += cost(g)
            b[1].append(g)
        groups.extend(b[1] for b in bins if b[1])
    return groups


def _write(stream, value):
    data = pickle.dumps(value, protocol=4)
    stream.write(struct.pack('<Q', len(data)) + data)
    stream.flush()


def _reader(stream, q):
    try:
        while True:
            head = stream.read(8)
            if len(head) < 8:
                break
            n = struct.unpack('<Q', head)[0]
            v = pickle.loads(stream.read(n))
            q.last_arrival = time.perf_counter()  # source-v52 bot-timing-v1: reply arrival (one reply per request)
            q.put(v)
    except Exception:
        pass
    q.put(EOFError)


class BotMux:
    """games: list of dict(gid, row (pool opponent row), seed, seat). One arena sandbox container per worker."""
    def __init__(self, games, config, arena_repo, workers, prefix=None, cpuset=None, turn_timeout=None, sandbox=True):
        verify_pins()
        for row in {g['row']['id']: g['row'] for g in games}.values():
            if file_sha(row['file']) != row['archive']:
                raise ValueError('Arena artifact changed: ' + row['id'])
        if arena_repo not in sys.path:
            sys.path.insert(0, arena_repo)
        from tools.arena.sandbox import docker_args
        prefix = prefix or os.environ.get('PPO_ARENA_CONTAINER_PREFIX', 'ppo-practice-gpu-')
        cpuset = cpuset or os.environ.get('PPO_ARENA_CPUSET')
        self.timeout = float(turn_timeout or config.get('turn_timeout_seconds', 5))
        _ported = ported_ids()  # gpu-bots-v1 (source-v68)
        _pg = [g for g in games if g['row']['id'][:12] in _ported]
        _dg = [g for g in games if g['row']['id'][:12] not in _ported]
        self.groups = plan_workers(_dg, workers) if _dg else []
        self.n_docker = len(self.groups)
        self.groups += plan_port_workers(_pg, PORT_WORKERS, _ported) if _pg else []
        self.names, self.procs, self.queues, self.owner, self.alive = [], [], [], {}, []
        self.faults, self.seconds = {}, dict(start=0., load=0., submit=0., wait=0.)
        self.max_worker_ms = [0.] * len(self.groups)
        self.turn_stats, self.game_dt = [], {}  # source-v52 bot-timing-v1
        t0 = time.perf_counter()
        try:
            self.jail = None  # chroot-v1 (source-v73)
            if sandbox and os.environ.get('PPO_ARENA_SANDBOX', 'docker') == 'chroot' and self.n_docker:
                from ppo.arena_chroot import Jail
                _vend = sorted({g['row']['vendor'] for grp in self.groups[:self.n_docker] for g in grp if g['row'].get('vendor')})
                if len(_vend) > 1: raise ValueError('one shared vendor runtime per jail: %s' % _vend)
                self.jail = Jail(_vend[0] if _vend else None)
            for w, group in enumerate(self.groups):
                if w >= self.n_docker:  # gpu-bots-v1: vectorized port worker (host process, our own code)
                    env = dict(os.environ, PPO_PORT_DEVICE='cpu', PPO_PORT_THREADS='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
                    p = subprocess.Popen([sys.executable, str(PORTS_DIR / 'gb' / 'port_worker.py')], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
                    q = queue.Queue()
                    threading.Thread(target=_reader, args=(p.stdout, q), daemon=True).start()
                    self.procs.append(p); self.queues.append(q); self.alive.append(True)
                    for g in group:
                        self.owner[g['gid']] = w
                    p._init = [dict(gid=g['gid'], bot=g['row']['id'][:12], seed=int(g['seed']), seat=int(g['seat'])) for g in group]
                    continue
                if sandbox and self.jail is not None:  # chroot-v1: same runtime/worker as docker, UID + seccomp isolation
                    cmd, _wd = self.jail.command(w)
                    zips = {g['gid']: self.jail.add_zip(g['gid'], g['row']['file']) for g in group}
                elif sandbox:
                    name = prefix + uuid.uuid4().hex[:16]
                    mounts = [(HERE / 'arena_mux_worker.py', '/opt/arena_gpu/arena_mux_worker.py'), (HERE / 'arena_obs.py', '/opt/arena_gpu/arena_obs.py')]
                    mounts += [(g['row']['file'], '/bots/%d.zip' % g['gid']) for g in group]
                    vendors = sorted({g['row']['vendor'] for g in group if g['row'].get('vendor')})  # source-v47: shared read-only runtime (e.g. torch)
                    if len(vendors) > 1:
                        raise ValueError('one shared vendor runtime per worker: %s' % vendors)
                    mounts += [(v, '/opt/arena_gpu/vendor') for v in vendors]
                    cfg = dict(config, memory_mb=max(int(config['memory_mb']), 512 + sum(game_mb(g) for g in group)))
                    cmd = docker_args(cfg, name, mounts, ['python', '/opt/arena_gpu/arena_mux_worker.py', '/work'], max(512, 64 * len(group)))
                    cmd.insert(2, '-i')
                    if cpuset:
                        cmd.insert(3, '--cpuset-cpus=' + cpuset)
                    self.names.append(name)
                    zips = {g['gid']: '/bots/%d.zip' % g['gid'] for g in group}
                else:  # tests only: same worker as a host process
                    import tempfile
                    root = tempfile.mkdtemp(prefix='arena-mux-')
                    cmd = [sys.executable, str(HERE / 'arena_mux_worker.py'), root]
                    zips = {g['gid']: g['row']['file'] for g in group}
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                q = queue.Queue()
                threading.Thread(target=_reader, args=(p.stdout, q), daemon=True).start()
                self.procs.append(p)
                self.queues.append(q)
                self.alive.append(True)
                for g in group:
                    self.owner[g['gid']] = w
                p._init = [dict(gid=g['gid'], zip=zips[g['gid']], seed=int(g['seed']), seat=int(g['seat'])) for g in group]
            for w in range(len(self.procs)):
                if self._get(w, 120) != 'ready':
                    self._fail(w, 'INFRA: sandbox start')
            self.seconds['start'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            for w, p in enumerate(self.procs):
                if self.alive[w]:
                    _write(p.stdin, ('init', p._init, CONF, self.timeout))
            self.load = {}
            for w in range(len(self.procs)):
                if not self.alive[w]:
                    continue
                res = self._get(w, 120 + 10 * len(self.groups[w]))
                if res is None:
                    self._fail(w, 'INFRA: bot load')
                    continue
                for gid, v in res.items():
                    if isinstance(v, str):
                        self.faults[gid] = (-1, v)
                    else:
                        self.load[gid] = v
            self.seconds['load'] = time.perf_counter() - t0
        except BaseException:
            self.close()
            raise

    def _get(self, w, timeout):
        try:
            v = self.queues[w].get(timeout=timeout)
        except queue.Empty:
            return None
        return None if v is EOFError else v

    def _fail(self, w, reason, turn=-1):
        self.alive[w] = False
        for g in self.groups[w]:
            self.faults.setdefault(g['gid'], (turn, reason))
        try:
            self.procs[w].kill()
        except Exception:
            pass

    def submit(self, host, gids, flat=None):
        """host: dict field -> numpy [len(gids), ...], rows aligned with gids.
        source-v42 (collect-speed): each game travels as the raw bytes of its int32 row of the packed_host layout (one
        bytes object per game instead of 24 pickled numpy slices); the worker rebuilds exactly the arrays packed_host
        returned (int32, or bool where the simulator field is bool)."""
        t = time.perf_counter()
        n = len(gids)
        if flat is None:
            flat = np.concatenate([np.ascontiguousarray(host[k]).reshape(n, -1).astype(np.int32) for k in FIELDS], 1)
        layout = [(k, tuple(host[k].shape[1:]), host[k].dtype == bool) for k in FIELDS]
        per = {}
        for i, gid in enumerate(gids):
            w = self.owner[gid]
            if self.alive[w]:
                per.setdefault(w, {})[gid] = flat[i].tobytes()
        self._pending = sorted(per)
        self._sent = time.perf_counter()
        for w in self._pending:
            try:
                _write(self.procs[w].stdin, ('stepp', per[w], layout))
            except (OSError, ValueError):
                self._fail(w, 'INFRA: worker pipe', self._turn)
        self.seconds['submit'] += time.perf_counter() - t

    def collect(self, turn):
        """{gid: (json_text, rows, dt, fault)} for live workers; faulted/dead games are absent (caller plays empty actions)."""
        t = time.perf_counter()
        out = {}
        _bmax = _rt = 0.  # source-v52 bot-timing-v1
        for w in self._pending:
            if not self.alive[w]:
                continue
            budget = 30 + (self.timeout + 1) * len(self.groups[w])
            res = self._get(w, max(1., budget - (time.perf_counter() - self._sent)))
            if res is None:
                self._fail(w, 'INFRA: worker hang or crash', turn)
                continue
            self.max_worker_ms[w] = max(self.max_worker_ms[w], 1000 * sum(r[2] for r in res.values()))
            _bmax = max(_bmax, 1000 * sum(r[2] for r in res.values()))
            _rt = max(_rt, 1000 * (getattr(self.queues[w], 'last_arrival', self._sent) - self._sent))
            for gid, r in res.items():
                a = self.game_dt.setdefault(gid, [0., 0., 0]); a[0] += r[2]; a[1] = max(a[1], r[2]); a[2] += 1
            for gid, r in res.items():
                if r[3] and gid not in self.faults:
                    self.faults[gid] = (turn, r[3])
                out[gid] = r
        self.seconds['wait'] += time.perf_counter() - t
        self.turn_stats.append([round(1000 * (t - self._sent)), round(1000 * (time.perf_counter() - t)), round(_rt), round(_bmax)])
        return out

    def stats(self):
        out = []
        for w, p in enumerate(self.procs):
            if self.alive[w]:
                try:
                    _write(p.stdin, ('stats',))
                    out.append(self._get(w, 30))
                except Exception:
                    out.append(None)
        return out

    def close(self):
        for p in self.procs:
            try:
                _write(p.stdin, ('close',))
            except Exception:
                pass
            try:
                p.kill()
                p.wait(10)
            except Exception:
                pass
            for stream in (p.stdin, p.stdout):
                try:
                    stream.close()
                except Exception:
                    pass
        if self.names:
            subprocess.run(['docker', 'rm', '-f', *self.names], capture_output=True, timeout=120)
            self.names = []
        if getattr(self, 'jail', None) is not None:
            self.jail.cleanup(); self.jail = None


def packed_host(state, gidx, return_flat=False):
    """One gather + one device->host copy of every observation field for the arena games."""
    import torch
    parts, meta = [], []
    for name in FIELDS:
        v = state[name][gidx]
        meta.append((name, tuple(v.shape[1:]), v.dtype))
        parts.append(v.reshape(len(gidx), -1).to(torch.int32))
    flat = torch.cat(parts, 1).cpu().numpy()
    out, o = {}, 0
    for (name, shape, dtype), p in zip(meta, parts):
        w = p.shape[1]
        arr = flat[:, o:o + w].reshape(len(gidx), *shape)
        out[name] = arr.astype(bool) if dtype == torch.bool else arr
        o += w
    if return_flat:
        return out, flat
    return out


UNIT_NAMES = ('PASS', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'DROP', 'PICKUP', 'PLACE', 'PLANT', 'WATER', 'HARVEST', 'FERTILIZE', 'DIG',
              'BUILD_COOP', 'BUILD_PASTURE', 'FEED', 'COLLECT_FERTILIZER', 'CARE')
MARKET_NAMES = ('NONE', 'HIRE', 'BUY_LAND', 'BUY_SEED', 'BUY_PRODUCT', 'BUY_ANIMAL', 'SELL')
ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP")


def decode(rows):
    """gpu_sim action rows of one seat -> an official action dict the official engine executes identically."""
    units = []
    for i in range(int(rows['unit_count'])):
        op = UNIT_NAMES[int(rows['unit_op'][i])] if 0 <= int(rows['unit_op'][i]) < len(UNIT_NAMES) else 'INVALID'
        item = int(rows['unit_item'][i])
        a = [op]
        if op == 'PLANT' and item >= 0:
            a.append(ITEMS[item])
        elif op in ('PICKUP', 'PLACE') and item >= 0:
            a += [ITEMS[item], int(rows['unit_amount'][i])]
        units.append(a)
    market = []
    for i in range(int(rows['market_count'])):
        op = MARKET_NAMES[int(rows['market_op'][i])]
        if op in ('HIRE', 'BUY_LAND'):
            market.append([op])
        elif op == 'NONE':
            market.append(['NONE'])
        else:
            market.append([op, ITEMS[int(rows['market_item'][i])], int(rows['market_amount'][i])])
    return dict(farmer=units[0] if units else ['PASS'], hands=units[1:], market=market)


class ArenaHook:
    """Per-turn hook for GPUCollector.collect. arena: list of dict(game, seat (bot seat), job)."""
    def __init__(self, arena, device, workers, parity_games=0, parity_seed=0, sandbox=True):
        import torch
        self.arena = arena
        self.device = device
        self.gids = [a['game'] for a in arena]
        job = arena[0]['job']
        games = [dict(gid=a['game'], row=a['job']['arena_opponent'], seed=a['job']['seed'], seat=a['seat']) for a in arena]
        self.bots = BotMux(games, job['arena_config'], job['arena_repo'], workers, sandbox=sandbox)
        self.gidx = torch.tensor(self.gids, device=device, dtype=torch.long)
        self.bot_seat = torch.tensor([a['seat'] for a in arena], device=device, dtype=torch.long)
        self.empty = None
        self.seconds = dict(host=0., inject=0., record=0.)
        rng = random.Random(parity_seed)
        self.record = {g: [] for g in rng.sample(self.gids, min(parity_games, len(self.gids)))}
        self.turn = 0
        # surrogate-capture-v1 (source-v67): real-bot action rows for the first N arena games (index into self.gids)
        self.capture_n = min(len(self.gids), int(os.environ.get('PPO_CAPTURE_BOTS', '0')))  # source-v68: off by default
        self.capture_rows = []; self.capture_json = []

    def before_act(self, turn, state):
        t = time.perf_counter()
        self.turn = turn
        self.bots._turn = turn
        host, flat = packed_host(state, self.gidx, return_flat=True)
        self.seconds['host'] += time.perf_counter() - t
        # source-v42 (collect-speed): the pipe writes to the bot workers run on a helper thread while this thread launches
        # the learner's GPU act; after_act joins it before collecting. Inputs, bot outputs and GPU work are unchanged.
        if SUBMIT_THREAD:
            self._submit_error = None
            def run():
                try:
                    self.bots.submit(host, self.gids, flat)
                except BaseException as exc:  # re-raised on the collector thread in after_act
                    self._submit_error = exc
            self._submit = threading.Thread(target=run, daemon=True)
            self._submit.start()
        else:
            self.bots.submit(host, self.gids, flat)

    def after_act(self, turn, action):
        import torch
        if getattr(self, '_submit', None) is not None:
            self._submit.join()
            self._submit = None
            if self._submit_error is not None:
                raise self._submit_error
        res = self.bots.collect(turn)
        t = time.perf_counter()
        if self.empty is None:
            from ppo.arena_mux_worker import encode
            self.empty = encode({})
        rows = [res[g][1] if g in res and not res[g][3] else self.empty for g in self.gids]
        packed = np.array([[*r['unit_op'], *r['unit_item'], *r['unit_amount'], r['unit_count'],
                            *r['market_op'], *r['market_item'], *r['market_amount'], r['market_count']] for r in rows], np.int32)
        if self.capture_n:  # surrogate-capture-v1
            self.capture_rows.append(packed[:self.capture_n].copy())
            self.capture_json.append([res[g][0] if g in res and not res[g][3] else '{}' for g in self.gids[:self.capture_n]])
        packed = torch.from_numpy(packed).to(self.device, non_blocking=True)
        o = 0
        for name, width in zip(ROWS, WIDTHS):
            part = packed[:, o:o + width].to(action[name].dtype)
            action[name][self.gidx, self.bot_seat] = part[:, 0] if width == 1 else part
            o += width
        self.seconds['inject'] += time.perf_counter() - t
        if self.record:
            t = time.perf_counter()
            ids = [self.gids.index(g) for g in self.record]
            gi = self.gidx[ids]
            ls = 1 - self.bot_seat[ids]
            learner = {k: action[k][gi, ls].cpu().numpy() for k in ROWS}
            for i, g in enumerate(self.record):
                la = decode({k: learner[k][i] for k in ROWS})
                ba = json.loads(res[g][0]) if g in res and not res[g][3] else {}
                seat = self.arena[self.gids.index(g)]['seat']
                self.record[g].append([ba, la] if seat == 0 else [la, ba])
            self.seconds['record'] += time.perf_counter() - t
        return action

    def close(self):
        if getattr(self, "_submit", None) is not None:
            self._submit.join(60)
        self.bots.close()

    def launch_parity(self, money, parity_dir, tag):
        """Write the sampled games' action traces and replay them in the official engine in a detached subprocess."""
        if not self.record or not parity_dir:
            return None
        os.makedirs(parity_dir, exist_ok=True)
        paths = []
        for g, turns in self.record.items():
            a = self.arena[self.gids.index(g)]
            job = a['job']
            path = os.path.join(parity_dir, 'trace-%s-%s-g%d.jsonl.gz' % (tag, job['seed'], g))
            with gzip.open(path, 'wt') as f:
                header = dict(header=True, tag=tag, game=g, seed=job['seed'], bot_seat=a['seat'], gpu_money=money[g],
                              faulted=g in self.bots.faults, arena_opponent=job['arena_opponent'], arena_config=job['arena_config'],
                              arena_repo=job['arena_repo'], arena_runtime_files=job['arena_runtime_files'])
                f.write(json.dumps(header) + '\n')
                for t, acts in enumerate(turns):
                    f.write(json.dumps(dict(step=t, actions=acts)) + '\n')
            paths.append(path)
        log = open(os.path.join(parity_dir, 'parity.log'), 'ab')
        proc = subprocess.Popen([sys.executable, '-m', 'ppo.arena_parity', '--out', os.path.join(parity_dir, 'parity.jsonl'), '--delete', *paths],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=str(HERE.parent), start_new_session=True)
        log.close()
        return dict(pid=proc.pid, traces=len(paths))
