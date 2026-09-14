"""Frozen daily game ownership across the coordinator and CPU executors."""
from collections import defaultdict
from pathlib import Path
from . import schedule
from .store import read, write


def assignments(root, manifest):
    return read(Path(root)/'private/daily-assignments'/f"{manifest['id']}.json", {})


def prepare(root, hosts):
    root = Path(root)
    enabled = {name: cfg for name, cfg in hosts if cfg.get('enabled')}
    if not enabled:
        return
    # Relative measured full-game throughput, configurable per host. Keep pairs
    # together and ownership immutable so no two machines execute the same game.
    weights = {'coordinator': read(root/'config.json').get('tournament_weight', 1800)}
    weights.update({name: cfg.get('tournament_weight', 1800) for name, cfg in enabled.items()})
    if any(not isinstance(w, (int, float)) or w <= 0 for w in weights.values()):
        raise ValueError('Tournament weights must be positive')
    for path in sorted((root/'runs').glob('daily-*/manifest.json')):
        m = read(path)
        if m['kind'] != 'daily' or schedule.complete(root, m):
            continue
        target = root/'private/daily-assignments'/f"{m['id']}.json"
        if target.exists():
            continue
        pairs = defaultdict(list)
        for g in m['games']:
            if not read(path.parent/'games'/f"{g['id']}.json", {}).get('resolved'):
                pairs[(g['seed'], tuple(sorted(g['agents'])))].append(g)
        owners, loads = {}, dict.fromkeys(weights, 0)
        for games in pairs.values():
            host = min(weights, key=lambda h: loads[h]/weights[h])
            for g in games:
                owners[g['id']] = host
            loads[host] += len(games)
        write(target, owners)


def shard(root, manifest, host):
    owners = assignments(root, manifest)
    games = [g for g in manifest['games'] if owners.get(g['id']) == host]
    if not games:
        return None
    return {**manifest, 'id': manifest['id']+'-'+host, 'games': games,
            'parent_daily': manifest['id'], 'executor': host,
            'seed_count': len({g['seed'] for g in games})}


def local_game(root, manifest, game):
    return assignments(root, manifest).get(game['id'], 'coordinator') == 'coordinator'
