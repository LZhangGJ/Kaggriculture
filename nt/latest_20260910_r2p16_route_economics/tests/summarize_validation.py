"""Check immutable inputs and report all paired games, including failures."""
from pathlib import Path
import hashlib
import gzip
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT.parents[2] / 'experiments/r2p16_route_economics_20260910'

def read(path):
    return json.loads(path.read_text())

def read_audit(path):
    return json.loads(gzip.decompress(path.read_bytes()))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    provenance = read(ROOT / 'SOURCE_PROVENANCE.json')
    for name, expected in provenance['baseline_sha256'].items():
        assert digest(EVIDENCE / 'baseline' / name) == expected, name
    builds = {}
    for mode in range(4):
        receipt = read(ROOT / f'build/revision5/route{mode}.BUILD.json')
        assert digest(ROOT / f'build/revision5/route{mode}.so') == receipt['sha256']
        for name, expected in receipt['sources'].items():
            assert digest(ROOT / name) == expected, name
        builds[str(mode)] = receipt['sha256']
    assert 'PASS 20 ' in read(ROOT / 'build/revision5/route3.BUILD.json')['unit']
    parity = read(EVIDENCE / 'parity.json')
    assert parity['status'] == 'PASS' and parity['observations'] == 719
    assert parity['rebuilt_sha256'] == builds['0']
    panels = {}
    for tag in ('fresh5_r5', 'regression_r5'):
        directory = EVIDENCE / 'runs' / tag
        rows = read(directory / 'rows.json')
        protocol = read(directory / 'PROTOCOL.json')
        expected = {tuple(j[:4]) for j in protocol['jobs']}
        observed = {(r['arm'], r['opponent'], r['seed'], r['seat']) for r in rows}
        assert len(rows) == len(expected) and observed == expected
        for path, sha in protocol['hashes'].items():
            assert digest(Path(path)) == sha, path
        # Audit per-step and per-day records as well as the result summaries.
        for row in rows:
            audit = read_audit(directory / 'smoke' / (row['name'] + '.audit.json.gz'))
            assert len(audit['action_seconds']) == 719 and len(audit['daily']) == 30
            assert len(audit['invalid']) == row['invalid_actions']
            assert sum(d['wages'] for d in audit['daily']) == row['wages']
            row['local_overage_seconds'] = sum(max(0, t-1) for t in audit['action_seconds'])
        panels[tag] = rows
    rows = panels['fresh5_r5']
    def summary(group):
        return dict(games=len(group), wins=sum(r['win'] for r in group),
                    mean_cash=statistics.mean(r['cash'] for r in group),
                    mean_margin=statistics.mean(r['margin'] for r in group),
                    mean_wages=statistics.mean(r['wages'] for r in group),
                    mean_moves=statistics.mean(r['moves'] for r in group),
                    accepted_route_changes=sum(r['route'].get('route_changes',0) for r in group),
                    accepted_hire_reductions=sum(r['route'].get('route_hire_changes',0) for r in group),
                    mean_overflow=statistics.mean(r['overflow_units'] for r in group),
                    invalid_actions=sum(r['invalid_actions'] for r in group),
                    missed_promises=sum(r['uncompleted_required_tasks'] for r in group),
                    max_seconds=max(r['max_seconds'] for r in group),
                    over_one_second=sum(r['over_one_second'] for r in group),
                    max_local_overage_seconds=max(r['local_overage_seconds'] for r in group))
    arms = {name:summary([r for r in rows if r['arm']==name]) for name in ('P16','guards','delivery','economic')}
    by_case = {(r['opponent'],r['seed'],r['seat']):r for r in rows if r['arm']=='P16'}
    for name in ('guards','delivery','economic'):
        group = [r for r in rows if r['arm']==name]
        arms[name]['mean_paired_cash_change'] = statistics.mean(r['cash']-by_case[(r['opponent'],r['seed'],r['seat'])]['cash'] for r in group)
        arms[name]['mean_paired_margin_change'] = statistics.mean(r['margin']-by_case[(r['opponent'],r['seed'],r['seat'])]['margin'] for r in group)
        arms[name]['cases_with_lower_wages'] = sum(r['wages']<by_case[(r['opponent'],r['seed'],r['seat'])]['wages'] for r in group)
        arms[name]['mean_paired_wage_change'] = statistics.mean(r['wages']-by_case[(r['opponent'],r['seed'],r['seat'])]['wages'] for r in group)
    all_rows = rows+panels['regression_r5']
    bad = [{k:r[k] for k in ('name','error','frames','invalid_actions','uncompleted_required_tasks','over_one_second')}
           for r in all_rows if r['error'] or r['frames']!=720 or r['invalid_actions'] or r['uncompleted_required_tasks'] or r['uninstrumented_replay_check']!='PASS']
    slow = [{k:r[k] for k in ('name','max_seconds','over_one_second','local_overage_seconds')} for r in all_rows if r['over_one_second']]
    timing_path = EVIDENCE / 'sequential_timing_fresh5_r5.json'
    isolated = read(timing_path) if timing_path.exists() else None
    if isolated:
        assert isolated['binary_sha256'] == builds['3']
        for game in isolated['games']:
            assert game['action_parity']=='PASS' and game['observations']==719
            for name in ('wall_seconds','cpu_seconds'):
                assert len(game[name])==719
            assert digest(EVIDENCE / 'runs/fresh5_r5/smoke' / (game['name']+'.json.gz')) == game['replay_sha256']
    result = dict(status='PASS' if not bad and not slow else 'FAIL',
                  functional_status='PASS' if not bad else 'FAIL', local_timing_status='PASS' if not slow else 'FAIL',
                  baseline_commit=provenance['base_commit'], frozen_baseline_files_checked=len(provenance['baseline_sha256']),
                  native_binaries=builds, unit_checks=20, disabled_mode_parity=parity,
                  fresh_games=len(rows), regression_games=len(panels['regression_r5']),
                  official_action_pairs_checked=len(all_rows)*719,
                  by_arm=arms, failures=bad, slow_games=slow,
                  sequential_timing=None if not isolated else [{k:v for k,v in g.items() if k not in {'wall_seconds','cpu_seconds'}} for g in isolated['games']],
                  regression_summary=summary(panels['regression_r5']),
                  per_opponent={op:{arm:summary([r for r in rows if r['arm']==arm and r['opponent']==op]) for arm in arms} for op in sorted({r['opponent'] for r in rows})},
                  gates=dict(functional='No interpreter error, ineffective work action, or audited promise miss; all full replays agree.',
                             local_timing='Conservative gate: no measured call exceeds one second. Overage allowance is reported separately; this is not a Kaggle timeout verdict.'),
                  limitations=['Five expanded environment seeds; both seats and all opponents are paired, not independent samples. The immature-harvest defect was discovered on this panel; it is not an untouched holdout.',
                               'Promise audit covers accepted, already declared work; it does not certify every baseline planning wish.',
                               'Future opponent orders and shops are conditional forecasts, not known outcomes.',
                               'Local official interpreter and WSL timing, not Kaggle sandbox certification.',
                               'No automatic replacement of the frozen competition baseline.'])
    (ROOT / 'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
