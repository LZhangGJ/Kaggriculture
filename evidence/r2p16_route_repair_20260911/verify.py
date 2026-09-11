"""Verify shipped bytes/builds and recount frozen outcomes; no replay rerun."""
from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PANELS = ('historical25', 'joint_r1_100', 'joint_r2_100')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def check_manifest(folder):
    manifest = read(folder / 'PACKAGE_MANIFEST.json')
    for name, record in manifest['files'].items():
        raw = (folder / name).read_bytes()
        assert len(raw) == record['bytes'] and sha(raw) == record['sha256'], name
        if 'uncompressed_sha256' in record:
            plain = gzip.decompress(raw)
            assert len(plain) == record['uncompressed_bytes'], name
            assert sha(plain) == record['uncompressed_sha256'], name
    return manifest


def check_group(rows, qa, expected):
    assert len(rows) == expected['games']
    assert sum(r['margin'] > 0 for r in rows) == expected['wins']
    assert sum(r['margin'] == 0 for r in rows) == expected['draws']
    assert math.isclose(100 * expected['wins'] / len(rows), expected['win_rate'])
    for key in ('cash', 'opponent_cash', 'margin', 'wages', 'hires', 'overflow_units'):
        assert math.isclose(statistics.fmean(r[key] for r in rows), expected['mean_' + key], abs_tol=1e-8), key
    for row_key, expected_key in [('invalid_actions', 'invalid_actions'),
                                  ('uncompleted_required_tasks', 'recorded_uncompleted')]:
        assert sum(r[row_key] for r in rows) == expected[expected_key]
    assert sum(qa[r['name']]['unexpected'] for r in rows) == expected['unexpected_uncompleted']
    assert sum(qa[r['name']]['explicit_deferrals'] for r in rows) == expected['explicit_deferrals']


def main():
    checked = len(check_manifest(ROOT)['files'])
    for folder, revision in [('latest_20260910_r2p16_route_economics', 5),
                             ('latest_20260910_r2p16_route_workflow', 2),
                             ('latest_20260910_r2p16_route_recovery', 3),
                             ('latest_20260911_r2p16_route_repair', 4)]:
        package = REPO / 'nt' / folder
        manifest = check_manifest(package)
        checked += len(manifest['files'])
        build = read(package / f'build/revision{revision}/route3.BUILD.json')
        assert sha((package / manifest['binary']).read_bytes()) == build['sha256'] == manifest['binary_sha256']
        for name, digest in build['sources'].items():
            assert sha((package / name).read_bytes()) == digest, (folder, name)

    for name, record in read(ROOT / 'SOURCE_FILES.json').items():
        raw = (ROOT / name).read_bytes()
        if record['lossless_gzip']:
            raw = gzip.decompress(raw)
        assert len(raw) == record['source_bytes'] and sha(raw) == record['source_sha256'], name

    acquisition = read(ROOT / 'downloads/56149565/USER_ACQUISITION.json')
    for name, digest in acquisition['member_hashes'].items():
        assert sha((ROOT / 'opponents/submission_56149565' / name).read_bytes()) == digest
    identities = read(ROOT / 'IDENTITY.json')
    r1 = identities['original_r1']
    assert sha((REPO / r1['repository_path']).read_bytes()) == r1['archive_sha256']
    assert identities['supplied_r2']['archive_sha256'] != r1['archive_sha256']
    assert not identities['separate_workflow_builds_in_local_100_seed_panel']
    for name, record in identities['separate_workflow_builds'].items():
        assert sha((REPO / name).read_bytes()) == record['sha256']

    independent = read(ROOT / 'INDEPENDENT_VERIFY.json')
    fresh = {int.from_bytes(hashlib.sha256(f'route-repair-20260911-holdout-{i}'.encode()).digest()[:4], 'big') for i in range(100)}
    assert len(fresh) == 100 and independent['seed_check']['overlap'] == []
    previous = set().union(*(set(v) for v in independent['seed_check']['prior_protocols'].values()))
    assert not fresh.intersection(previous)
    regression = read(ROOT / 'runs/regression_final/PROTOCOL.json')
    assert not fresh.intersection(j[2] for j in regression['jobs'])
    report = (ROOT / 'REPORT.md').read_text(encoding='utf-8')
    totals = {}
    observations = 0
    for tag in PANELS:
        folder = ROOT / 'runs' / tag
        rows = read(folder / 'rows.json.gz')
        protocol = read(folder / 'PROTOCOL.json')
        summary = read(folder / 'SUMMARY.json')
        qa_raw = read(folder / 'QA.json.gz')
        assert qa_raw['status'] == 'PASS'
        qa = {r['name']: r for r in qa_raw['checks']}
        expected = {tuple(j) for j in protocol['jobs']}
        actual = {(r['arm'], r['opponent'], r['seed'], r['seat']) for r in rows}
        assert actual == expected and len(rows) == len(actual) == summary['games']
        assert set(qa) == {r['name'] for r in rows}
        pairs = defaultdict(dict)
        for row in rows:
            assert not row['error'] and row['frames'] == 720
            assert row['margin'] == row['cash'] - row['opponent_cash']
            assert row['win'] == (row['margin'] > 0)
            assert row['seed'] in fresh
            check = qa[row['name']]
            assert check['status'] == 'PASS' and check['frames'] == 720
            assert check['replay_sha256'] == row['replay_sha256']
            observations += check['observations']
            pairs[row['opponent'], row['seed'], row['seat']][row['arm']] = row
        for arm, expected_arm in summary['arms'].items():
            own = [r for r in rows if r['arm'] == arm]
            check_group(own, qa, expected_arm)
            for opponent, agents in summary['opponents'].items():
                check_group([r for r in own if r['opponent'] == opponent], qa, agents[arm])
            totals[tag + '/' + arm] = dict(games=len(own), wins=expected_arm['wins'])
            saved = independent['panels'][tag]['aggregates'][arm]
            assert saved['games'] == len(own) and saved['wins'] == expected_arm['wins']
        for comparison in summary['comparisons']:
            old = comparison['baseline']
            assert len(pairs) == comparison['pairs']
            assert all(set(p) == {'workflow', 'recovery', 'fixed'} for p in pairs.values())
            for key in ('win', 'cash', 'margin', 'wages', 'hires'):
                values = [(float(p['fixed'][key]) - float(p[old][key])) * (100 if key == 'win' else 1) for p in pairs.values()]
                assert math.isclose(statistics.fmean(values), comparison['deltas'][key]['mean'], abs_tol=1e-8)
            assert sum(p['fixed']['win'] > p[old]['win'] for p in pairs.values()) == comparison['gained_wins']
            assert sum(p['fixed']['win'] < p[old]['win'] for p in pairs.values()) == comparison['lost_wins']
            label = {'historical25': 'Historical 13', 'joint_r1_100': 'Original R1', 'joint_r2_100': 'Original R2'}[tag]
            d = comparison['deltas']; lo, hi = d['win']['ci95']
            line = f"| {label} | {old.title()} | {d['win']['mean']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {d['cash']['mean']:+,.2f} |"
            assert line in report, line
    assert sum(r['games'] for r in totals.values()) == 3150
    assert observations == 4536000
    markdown = [REPO / 'agent.md', *ROOT.glob('*.md')]
    for package in ('latest_20260910_r2p16_route_economics', 'latest_20260911_r2p16_route_repair'):
        markdown.extend((REPO / 'nt' / package).glob('*.md'))
    for path in markdown:
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'[\u3400-\u9fff]', text), path
    print(json.dumps(dict(status='PASS', checked_files=checked, games=3150,
        archived_observations_verified=observations, recount=totals,
        scope='Packaged bytes, build/source identities and recorded results. Full replay transitions are not rerun.'), indent=2))


if __name__ == '__main__':
    main()
