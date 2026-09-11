"""Verify shipped bytes, original build sources, and recount the paired evaluation."""
from pathlib import Path
import gzip, hashlib, json
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)

def main():
    checked = 0
    for arm, revision in [('workflow', 2), ('recovery', 3)]:
        package = REPO / f'nt/latest_20260910_r2p16_route_{arm}'
        manifest = read(package / 'PACKAGE_MANIFEST.json')
        for name, record in manifest['files'].items():
            data = (package / name).read_bytes()
            assert len(data) == record['bytes'] and sha(data) == record['sha256'], name
            checked += 1
        build = read(package / f'build/revision{revision}/route3.BUILD.json')
        assert sha((package / f'build/revision{revision}/route3.so').read_bytes()) == build['sha256']
        for name, digest in build['sources'].items():
            assert sha((package / name).read_bytes()) == digest, name
        assert manifest['binary_sha256'] == build['sha256']
    for name, record in read(ROOT / 'PACKAGE_MANIFEST.json')['files'].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == record['bytes'] and sha(data) == record['sha256'], name
        if 'uncompressed_sha256' in record:
            assert sha(gzip.decompress(data)) == record['uncompressed_sha256'], name
        checked += 1
    latest = read(ROOT / 'latest25/rows.json.gz')
    previous = read(ROOT / 'previous11/rows.json.gz')
    protocol = read(ROOT / 'latest25/PROTOCOL.json')
    expected = read(ROOT / 'RESULTS.json')
    assert len(latest) == 200 and len(previous) == 220
    assert {(r['arm'], r['opponent'], r['seed'], r['seat']) for r in latest} == {tuple(j) for j in protocol['jobs']}
    matched = previous + [r for r in latest if r['seed'] in protocol['matched_pool_seeds']]
    totals = {}
    for label, rows in [('latest_two25', latest), ('matched13', matched)]:
        assert all(not r['error'] and r['frames'] == 720 and r['margin'] == r['cash'] - r['opponent_cash'] for r in rows)
        assert len({(r['arm'], r['opponent'], r['seed'], r['seat']) for r in rows}) == len(rows)
        for arm in ('workflow', 'recovery'):
            own = [r for r in rows if r['arm'] == arm]
            for op in [None] + sorted({r['opponent'] for r in own}):
                subset = own if op is None else [r for r in own if r['opponent'] == op]
                s = expected[label][arm]['overall'] if op is None else expected[label][arm]['opponents'][op]
                assert len(subset) == s['games']
                assert sum(r['margin'] > 0 for r in subset) == s['wins']
                assert sum(r['margin'] == 0 for r in subset) == s['draws']
                assert sum(r['margin'] < 0 for r in subset) == s['losses']
                assert s['wins']/s['games'] == s['win_rate']
            totals[label + '/' + arm] = dict(games=len(own), wins=sum(r['margin'] > 0 for r in own))
    for folder, rows in [('latest25', latest), ('previous11', previous)]:
        qa = read(ROOT / folder / 'QA.json')
        assert qa['status'] == 'PASS'
        assert {r['name'] for r in qa['rows']} == {r['name'] for r in rows}
    print(json.dumps(dict(status='PASS',checked_files=checked,recount=totals,
        scope='Files, source/build identities and recorded results; does not rerun the full replays retained on the original machine'), indent=2))

if __name__ == '__main__':
    main()
