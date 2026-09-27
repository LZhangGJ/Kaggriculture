"""start-state-v1 (source-v44): reverse-curriculum / backplay starts from recorded top-bot games (Salimans & Chen 2018).

A fraction F of each rank's SELF-PLAY games is replaced by a recorded official game (bank-v1): the game gets the recorded
seed, and on every turn t < S (the start turn) BOTH seats play the recorded actions (the same gpu_sim action rows the
arena-in-gpu hook injects for real bots, ppo.arena_mux_worker.encode), so the pinned simulator reaches exactly the
recorded state at turn S. During the prefix every role still runs its encoder on the real observations, so the learner's
GRU memory, the frozen reference's memory and the public-history features are built on the recorded prefix; the history
remembers the RECORDED market requests (exact_actions.market_target, as score_replay.py / remember_requests do), not the
sampled ones. The sampled prefix decisions are discarded: their stored rows are scrubbed (no decoder events, logp 0,
factors 0) and the learner mask is False, so prefix turns carry no policy, value, entropy or KL loss; the trainer also
drops them from the advantage moments (episode 'train_from') and skips windows that end at or before S. From turn S on
the game is an ordinary self-play game (both seats = learner, the learner sits in the recorded seats, i.e. the bots').

Why self-play only: the learner owns both seats, so a bot-vs-bot recording yields two learner episodes from top-bot
positions; the champion-slot games carry the league's matchup statistics (league_runtime.matchup_counts) and stay full
games; arena games stay full because the real bot programs cannot be restarted mid-game.

Bank (npz, lazily read per game): 'meta' (JSON), 'money' [G,720,2] recorded money before each turn,
'a{i}' [719,2,131] int32 forced action rows (ROWS/WIDTHS order of ppo.arena_gpu) for turn t (= official steps[t+1].action),
'o{i}' [719,2,10,3] int32 canonical market requests (MARKET index, quantity, valid) for the history channels.
"""
import hashlib, json, random
from pathlib import Path
import numpy as np

PROTOCOL = 'start-state-v1'
BANK_VERSION = 'start-state-bank-v1'
ROWS = ('unit_op', 'unit_item', 'unit_amount', 'unit_count', 'market_op', 'market_item', 'market_amount', 'market_count')
WIDTHS = (33, 33, 33, 1, 10, 10, 10, 1)
TURNS = 719


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parse_days(text):
    lo, hi = (int(x) for x in str(text).split(','))
    if not 1 <= lo <= hi or (hi + 1) * 24 > TURNS:
        raise ValueError('--start-state-days must be "lo,hi" with 1 <= lo <= hi and (hi+1)*24 <= 719')
    return lo, hi


def bank_meta(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
    if meta.get('version') != BANK_VERSION:
        raise ValueError('Start-state bank version mismatch')
    if not meta['games']:
        raise ValueError('Empty start-state bank')
    return meta


def identity(path, days):
    meta = bank_meta(path)
    return dict(protocol=PROTOCOL, bank_sha256=file_sha(path), bank_games=len(meta['games']), days=list(days),
                families=['selfplay'], start='uniform turn in [24*lo, 24*(hi+1)-1] (uniform day, uniform hour)',
                prefix='both seats forced to the recorded actions; roles/reference encoders run (memory + history warm-up); '
                       'prefix rows scrubbed and learner-masked; advantages moments and windows from the start turn')


def assign(rows, meta, bank_path, bank_sha, frac, days, seed, iteration, rank):
    """Mark round(frac * n) of this rank's self-play jobs (deterministic in seed/iteration/rank) as start-state games."""
    selfplay = sorted((r for r in rows if r['family'] == 'selfplay'), key=lambda r: r['game'])
    count = int(round(frac * len(selfplay)))
    if not count:
        return rows
    rng = random.Random(f'{PROTOCOL}:{seed}:{iteration}:{rank}')
    chosen = {r['game'] for r in rng.sample(selfplay, count)}
    lo, hi = days
    out = []
    for row in rows:
        if row['game'] in chosen:
            r2 = random.Random(f'{PROTOCOL}:{seed}:{iteration}:{row["game"]}')
            index = r2.randrange(len(meta['games']))
            turn = r2.randrange(24 * lo, 24 * (hi + 1))
            g = meta['games'][index]
            row = dict(row, seed=int(g['seed']), start_state=dict(bank=str(bank_path), bank_sha256=bank_sha, index=index, turn=turn,
                                                                   seed=int(g['seed']), original_seed=row['seed'], seats=g['seats']))
        out.append(row)
    return out


class Bank:
    """Lazy per-game reader; the sha is checked once per process."""
    def __init__(self, path, sha):
        if file_sha(path) != sha:
            raise ValueError('Start-state bank changed: ' + str(path))
        self.path, self.sha = str(path), sha
        self.z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(self.z['meta']))
        self.money = self.z['money']

    def game(self, index, turns):
        return self.z['a%d' % index][:turns], self.z['o%d' % index][:turns]


class Forcer:
    """Device-side forcing for GPUCollector. games: list of (batch game row, bank index, start turn)."""
    def __init__(self, bank, games, device):
        import torch
        self.device = device
        self.rows = [g for g, _, _ in games]
        self.index = [i for _, i, _ in games]
        self.start = [s for _, _, s in games]
        if min(self.start) < 1 or max(self.start) >= TURNS:
            raise ValueError('Start turn out of range')
        horizon = max(self.start)
        acts, orders = [], []
        for _, i, s in games:
            a, o = bank.game(i, horizon)
            acts.append(a); orders.append(o)
        self.actions = torch.from_numpy(np.stack(acts)).to(device)          # [G, H, 2, 131]
        self.orders = torch.from_numpy(np.stack(orders)).to(device).long()  # [G, H, 2, 10, 3]
        self.expected_money = np.stack([bank.money[i, s] for i, s in zip(self.index, self.start)])  # [G, 2] at turn S
        self.checked = []
        self._active = {}

    def active(self, turn):
        """(positions into self.rows, batch game rows, flat seat ids) of games still in their prefix at this turn."""
        live =tuple(k for k, s in enumerate(self.start) if turn < s)
        if live not in self._active:
            import torch
            pos = torch.tensor(live, device=self.device, dtype=torch.long)
            games = torch.tensor([self.rows[k] for k in live], device=self.device, dtype=torch.long)
            seats = torch.stack((2 * games, 2 * games + 1), 1).flatten()
            self._active[live] = (pos, games, seats)
        return self._active[live] if live else None

    def mask(self, turn, learner_mask):
        act = self.active(turn)
        if act is None:
            return learner_mask
        # collect-sync-v1 (source-v49): index_fill_ takes the scalar on device; `out[idx] = False` forced a host sync
        return learner_mask.clone().index_fill_(0, act[2], False)

    def history_orders(self, turn, orders):
        """orders [10, n, 5] (index, quantity, qindex, has_q, active) -> copy whose forced seats carry the recorded requests."""
        act = self.active(turn)
        if act is None:
            return orders
        pos, games, seats = act
        rec = self.orders[pos, turn]                      # [g, 2, 10, 3]
        rec = rec.permute(2, 0, 1, 3).reshape(10, -1, 3)  # [10, 2g, 3] in seat order 2g, 2g+1
        out = orders.clone()
        out[:, seats, 0] = rec[..., 0]
        out[:, seats, 1] = rec[..., 1]
        out[:, seats, 4] = rec[..., 2]
        return out

    def force(self, turn, action):
        act = self.active(turn)
        if act is None:
            return action
        pos, games, _ = act
        rows = self.actions[pos, turn]                    # [g, 2, 131]
        o = 0
        for name, width in zip(ROWS, WIDTHS):
            part = rows[..., o:o + width].to(action[name].dtype)
            action[name][games] = part[..., 0] if width == 1 else part
            o += width
        return action

    def scrub(self, turn, workers, market, logp, factors):
        act = self.active(turn)
        if act is None:
            return workers, market, logp, factors
        seats = act[2]
        # collect-sync-v1 (source-v49): same zeros via index_fill_; `t[idx] = 0` copied a CPU scalar with cudaStreamSynchronize
        return (workers.clone().index_fill_(1, seats, 0), market.clone().index_fill_(1, seats, 0),
                logp.clone().index_fill_(0, seats, 0), factors.clone().index_fill_(0, seats, 0))

    def observe(self, turn, state):
        for k, s in enumerate(self.start):
            if s == turn:
                g = self.rows[k]
                self.checked.append((k, state['money'][g].clone(), state['step'][g].clone()))

    def verify(self):
        """Recorded money at the start turn must equal the simulator's for every forced game (both seats); step == S."""
        bad = []
        for k, money, step in self.checked:
            got = money.double().cpu().numpy()
            if int(step) != self.start[k] or not np.array_equal(got, self.expected_money[k].astype(np.float64)):
                bad.append(dict(game=self.rows[k], bank_index=self.index[k], start=self.start[k], step=int(step),
                                simulator=got.tolist(), recorded=self.expected_money[k].tolist()))
        if len(self.checked) != len(self.start):
            raise RuntimeError('start-state-v1: not every forced game reached its start turn')
        if bad:
            raise RuntimeError('start-state-v1: simulator state differs from the recording at the start turn: %s' % bad[:3])
        return dict(games=len(self.start), money_checked=len(self.checked), start_turns=list(self.start), bank_index=list(self.index))
