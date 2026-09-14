"""Re-fit all eligible history between active versions, separately per contract."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .ratings import METHOD, summary
from .schedule import validate_result
from .store import digest, now, read, write


def reports(root):
    root = Path(root)
    active = sorted(r['agent'] for r in read(root / 'roster.json', []))
    # Immutable resolved results are indexed once; unfinished runs are revisited.
    with closing(sqlite3.connect(root / 'private/bt-history.sqlite')) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS finished (run TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, contract TEXT, a TEXT, b TEXT, payload TEXT)')
        for path in sorted((root / 'runs').glob('*/manifest.json')):
            if db.execute('SELECT 1 FROM finished WHERE run=?', (path.parent.name,)).fetchone():
                continue
            m = read(path)
            if m['kind'] not in ('daily', 'continuous'):
                continue
            done = True
            for g in m['games']:
                key = digest([m['contract_hash'], g['seed'], g['agents']])
                if db.execute('SELECT 1 FROM games WHERE id=?', (key,)).fetchone():
                    continue
                r = read(path.parent / 'games' / (g['id'] + '.json'))
                if not r or not r.get('resolved'):
                    done = False
                    continue
                validate_result(g, r)
                # Drop private diagnostics from the index, too.
                row = {k: r[k] for k in ('agents', 'outcome', 'cash')}
                row['seed'] = g['seed']
                db.execute('INSERT INTO games VALUES (?,?,?,?,?)',
                           (key, m['contract_hash'], *g['agents'], json.dumps(row)))
            if done:
                db.execute('INSERT INTO finished VALUES (?)', (m['id'],))
        results = []
        if not active:
            return results
        placeholders = ','.join('?' for _ in active)
        query = f'SELECT contract,payload FROM games WHERE a IN ({placeholders}) AND b IN ({placeholders}) ORDER BY id'
        groups = {}
        for contract, payload in db.execute(query, active + active):
            groups.setdefault(contract, []).append(json.loads(payload))
        for contract, rows in sorted(groups.items()):
            key = digest([METHOD, active, rows])
            cache = root / 'private' / ('bt-active-' + contract + '.json')
            old = read(cache)
            if old and old.get('fingerprint') == key:
                results.append(old)
                continue
            result = dict(run='cumulative-active-' + contract[:12], kind='cumulative',
                          created=now(), updated=now(), fingerprint=key, contract=contract,
                          completed=len(rows), planned=len(rows), complete=False,
                          scope='All resolved daily and continuous games; both versions active; no bootstrap intervals',
                          **summary(rows, active, bootstrap=0))
            write(cache, result)
            results.append(result)
    return results
