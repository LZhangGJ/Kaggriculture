"""arena-in-gpu-v1 (source-v26) multiplexed arena-bot host. Hash-pinned in ppo/arena_gpu.py.

Runs inside the arena sandbox image (production docker flags, CPython 3.12 + numpy), one process per container, hosting
the real bots of several games. Per game it reproduces tools/arena _arena_bridge.py exactly: fresh runpy namespace of
main.py, `random` seeded with the game seed before load, Obj-wrapped observation/configuration, JSON round trip of the
action. Isolation between games of one process: own cwd, sys.path[0], private sys.modules entries and `random` state
swapped in around every call. Each observation is rebuilt here from raw gpu_sim state (arena_obs.observation, verified
byte-identical to the official engine) and the action is returned as JSON plus encoded gpu_sim rows.

Faults (the official AgentProcess rules): exception, non-object action, or a call longer than the turn timeout
(SIGALRM, default 5 s) -> the game is faulted and plays empty actions from then on.

Frames on stdin/stdout: 8-byte little-endian length + pickle.
  ('init', [dict(gid, zip, seed, seat)], configuration, timeout) -> {gid: load_seconds or 'ERROR: ...'}
  ('step', {gid: state_slice}) -> {gid: (json_text, rows, dt, fault)}
  ('stats',) -> dict(maxrss_kb, games)
  ('close',)
"""
import contextlib, inspect, json, os, pickle, random, resource, runpy, signal, struct, sys, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena_obs import observation

UNIT_OPS = {n: i for i, n in enumerate(('PASS', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'DROP', 'PICKUP', 'PLACE', 'PLANT', 'WATER', 'HARVEST',
                                        'FERTILIZE', 'DIG', 'BUILD_COOP', 'BUILD_PASTURE', 'FEED', 'COLLECT_FERTILIZER', 'CARE'))}
MARKET_OPS = {n: i for i, n in enumerate(('NONE', 'HIRE', 'BUY_LAND', 'BUY_SEED', 'BUY_PRODUCT', 'BUY_ANIMAL', 'SELL'))}
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
SHED_ITEM_IDS = {n: i for i, n in enumerate(PRODUCTS + ("GOOSE", "COW", "SHEEP"))}
PRODUCT_IDS = {n: i for i, n in enumerate(PRODUCTS)}
CROP_IDS = {n: i for i, n in enumerate(PRODUCTS[:5])}
ANIMAL_IDS = {n: 9 + i for i, n in enumerate(("GOOSE", "COW", "SHEEP"))}
MAX_UNITS, MAX_ORDERS = 33, 10


def _unit(action):  # identical rules to kaggriculture_jax.codec._unit
    if not isinstance(action, list) or not action:
        return 0, -1, 1
    op = UNIT_OPS.get(action[0], -1)
    item, amount = -1, 1
    if op == 8 and len(action) >= 2:
        item = CROP_IDS.get(action[1], -1)
    elif op in (6, 7) and len(action) >= 2:
        item = SHED_ITEM_IDS.get(action[1], -1)
        if len(action) >= 3:
            try:
                amount = int(action[2])
            except (TypeError, ValueError, OverflowError):
                op, amount = -1, 0
    return op, item, amount


def _market(order):  # identical rules to kaggriculture_jax.codec._market
    if not isinstance(order, list) or not order:
        return 0, -1, 0
    op = MARKET_OPS.get(order[0], 0)
    if op in (1, 2):
        return op, -1, 0
    if op not in (3, 4, 5, 6) or len(order) < 3:
        return 0, -1, 0
    item = CROP_IDS.get(order[1], -1) if op == 3 else ANIMAL_IDS.get(order[1], -1) if op == 5 else PRODUCT_IDS.get(order[1], -1)
    try:
        amount = int(order[2])
    except (TypeError, ValueError, OverflowError):
        return 0, -1, 0
    return op, item, amount


def encode(raw):
    """One seat's official action dict -> gpu_sim action rows (same rules as codec.encode_actions)."""
    raw = raw if isinstance(raw, dict) else {}
    hands = raw.get('hands', [])
    hands = hands if isinstance(hands, list) else []
    units = [raw.get('farmer', ['PASS']), *hands][:MAX_UNITS]
    u = [_unit(a) for a in units]
    orders = raw.get('market', [])
    orders = (orders if isinstance(orders, list) else [])[:MAX_ORDERS]
    m = [_market(o) for o in orders]
    pad, mp = MAX_UNITS - len(u), MAX_ORDERS - len(m)
    return dict(unit_op=[x[0] for x in u] + [0] * pad, unit_item=[x[1] for x in u] + [-1] * pad, unit_amount=[x[2] for x in u] + [1] * pad,
                unit_count=len(u), market_op=[x[0] for x in m] + [0] * mp, market_item=[x[1] for x in m] + [-1] * mp,
                market_amount=[x[2] for x in m] + [0] * mp, market_count=len(m))


class Obj(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def obj(x):
    if isinstance(x, dict):
        return Obj({k: obj(v) for k, v in x.items()})
    if isinstance(x, list):
        return [obj(v) for v in x]
    return x


class TurnTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise TurnTimeout()


class Game:
    def __init__(self, gid, zip_path, seed, seat, root, configuration, timeout):
        self.gid, self.seat, self.seed, self.timeout = gid, seat, seed, timeout
        self.dir = os.path.join(root, 'g%d' % gid)
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                target = os.path.realpath(os.path.join(self.dir, info.filename))
                if not target.startswith(os.path.realpath(self.dir) + os.sep) and target != os.path.realpath(self.dir):
                    raise ValueError('Unsafe member')
            z.extractall(self.dir)
        self.conf = configuration
        self.fault = None
        self.modules = {}
        before = set(sys.modules)
        with self.context(initial=True):
            t = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                scope = runpy.run_path('main.py', run_name='arena_policy')
            self.policy = scope.get('agent') or scope.get('my_agent')
            if not callable(self.policy):
                raise RuntimeError('No agent function in main.py')
            self.accepts_config = len(inspect.signature(self.policy).parameters) >= 2
            self.load = time.perf_counter() - t
        for name in set(sys.modules) - before:  # the bot's own modules stay with this game
            mod = sys.modules[name]
            if (getattr(mod, '__file__', None) or '').startswith(self.dir):
                self.modules[name] = sys.modules.pop(name)

    @contextlib.contextmanager
    def context(self, initial=False):
        os.chdir(self.dir)
        sys.path.insert(0, self.dir)
        os.environ['ARENA_AGENT_SEED'] = str(self.seed)
        if initial:
            random.seed(self.seed)
        else:
            random.setstate(self.random_state)
        sys.modules.update(self.modules)
        try:
            yield
        finally:
            self.random_state = random.getstate()
            for name in self.modules:
                sys.modules.pop(name, None)
            sys.path.remove(self.dir)

    def step(self, s):
        if self.fault:
            return '{}', encode({}), 0., self.fault
        obs = observation(s, 0, self.seat)
        t = time.perf_counter()
        try:
            # source-v42: no JSON round trip of the input line. observation() yields only dict/list/str/int/float/bool/
            # None (JSON-exact values, str keys, key order preserved) and obj() below rebuilds every container, so the
            # bot receives the same Obj tree the bridge's json.loads produced (tests: test_arena_collect_speed.py).
            request = {'observation': obs, 'configuration': self.conf}
            with self.context(), contextlib.redirect_stdout(sys.stderr):
                o = obj(request['observation'])
                signal.setitimer(signal.ITIMER_PROF, self.timeout)  # cpu-turn-timer-v1: the bot's CPU time, like a dedicated core
                try:
                    action = self.policy(o, obj(request['configuration'])) if self.accepts_config else self.policy(o)
                finally:
                    signal.setitimer(signal.ITIMER_PROF, 0)
            out = json.dumps(action, allow_nan=False)
            parsed = json.loads(out)
            if not isinstance(parsed, dict):
                raise ValueError('Action must be an object')
        except TurnTimeout:
            self.fault = 'TIMEOUT'
        except Exception as exc:
            self.fault = 'ERROR: ' + repr(exc)[:300]
        dt = time.perf_counter() - t
        if self.fault:
            return '{}', encode({}), dt, self.fault
        return out, encode(parsed), dt, None


def unpack(raw, layout):
    """One game's packed_host row -> the dict of [1, ...] arrays packed_host returned (int32; bool fields as bool)."""
    import numpy as np
    flat = np.frombuffer(raw, np.int32)
    s, o = {}, 0
    for name, shape, is_bool in layout:
        n = 1
        for d in shape:
            n *= d
        a = flat[o:o + n].reshape((1, *shape))
        s[name] = a.astype(bool) if is_bool else a
        o += n
    if o != flat.size:
        raise ValueError('packed row size mismatch')
    return s


def read(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    return pickle.loads(stream.read(struct.unpack('<Q', head)[0]))


def write(stream, value):
    data = pickle.dumps(value, protocol=4)
    stream.write(struct.pack('<Q', len(data)) + data)
    stream.flush()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '/work'
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # stray prints must never corrupt the frame stream
    signal.signal(signal.SIGPROF, _alarm)  # cpu-turn-timer-v1 (source-v76)
    games = {}
    write(out, 'ready')
    while True:
        msg = read(inp)
        if msg is None or msg[0] == 'close':
            break
        if msg[0] == 'init':
            res = {}
            for g in msg[1]:
                try:
                    games[g['gid']] = Game(g['gid'], g['zip'], g['seed'], g['seat'], root, msg[2], msg[3])
                    res[g['gid']] = games[g['gid']].load
                except Exception as exc:  # a bot that cannot load is a faulted game, as a failed sandbox start would be
                    res[g['gid']] = 'ERROR: load ' + repr(exc)[:300]
            write(out, res)
        elif msg[0] == 'step':
            write(out, {gid: games[gid].step(s) if gid in games else ('{}', encode({}), 0., 'ERROR: not loaded') for gid, s in msg[1].items()})
        elif msg[0] == 'stepp':  # source-v42: {gid: int32 row bytes} + layout [(field, shape, is_bool)]
            write(out, {gid: games[gid].step(unpack(raw, msg[2])) if gid in games else ('{}', encode({}), 0., 'ERROR: not loaded')
                        for gid, raw in msg[1].items()})
        elif msg[0] == 'stats':
            write(out, dict(maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, games=len(games)))


if __name__ == '__main__':
    main()
