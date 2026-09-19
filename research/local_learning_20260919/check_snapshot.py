"""Check copied identities, syntax and reported evidence, without reading any replay."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read('SNAPSHOT_MANIFEST.json')
assert len(manifest['files']) == manifest['file_count']
for item in manifest['files']:
    path = ROOT / item['path']
    assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], item['path']
secret_pattern = re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-proj-[A-Za-z0-9_-]{30,}|AKIA[0-9A-Z]{16}')
hashes = {}
python_files = 0
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or path.name == 'PACKAGE_CHECK.json':
        continue
    assert path.suffix in ('.py', '.json', '.jsonl', '.md') or path.name in ('.gitignore', '.gitattributes'), path.name
    text = path.read_text(encoding='utf-8-sig')
    assert not secret_pattern.search(text), f'Credential-shaped content in {path.relative_to(ROOT)}'
    if path.suffix == '.py':
        ast.parse(text, filename=str(path))
        python_files += 1
    hashes[path.relative_to(ROOT).as_posix()] = digest(path)

readme = (ROOT / 'README.md').read_text(encoding='utf8')
results = []
for name, panel, games, wins in (
    ('joint_win65_20260918', 'fresh_final', 480, 358),
    ('learned_expand75_20260918', 'development', 240, 177),
    ('market_quantity75_20260918', 'development', 240, 169),
):
    summary = read(f'experiments/{name}/evaluation/{panel}/SUMMARY.json')
    assert summary['completed'] == summary['games'] == games and summary['wins'] == wins
    assert summary['status'] == 'COMPLETE' and summary['ties'] == summary['errors'] == 0
    assert summary['strict_win_rate'] == wins/games
    assert summary['checkpoint_sha256'] in readme
    results.append(dict(experiment=name, wins=wins, games=games, win_rate=summary['strict_win_rate']))

for name in ('learned_expand75_20260918', 'market_quantity75_20260918', 'retained_market75_20260919'):
    folder = f'experiments/{name}'
    plan = read(f'{folder}/PLAN.json')
    audit = read(f'{folder}/SEED_AUDIT.json')
    assert digest(ROOT / folder / 'SEED_AUDIT.json') == plan['seed_audit_sha256']
    seeds = [job[0] for panel in [plan['training_jobs'], *plan['development_panels'], plan['final_jobs']] for job in panel]
    assert len(seeds) == len(set(seeds)) == 1260
    assert not set(seeds) & set(audit['excluded_positive_integers'])
    for path, expected in read(f'{folder}/CODE_HASHES.json').items():
        assert digest(ROOT / folder / path) == expected, (name, path)
    for path, expected in plan['frozen_code'].items():
        assert digest(ROOT / folder / 'frozen_code' / path) == expected, (name, path)

probe = read('autoregressive_prototype/PROTOTYPE_CHECK.json')
assert probe['status'] == 'PASS' and probe['after']['exact_market_actions'] == probe['samples'] == 27
assert probe['new_match_seeds_used'] == 0 and probe['trained_probe_weights_discarded']
assert digest(ROOT / 'autoregressive_prototype/prototype.py') == probe['code_sha256']
report = dict(status='PASS', checked_utc=datetime.now(timezone.utc).isoformat(),
              copied_files_verified=manifest['file_count'], python_files_parsed=python_files,
              weight_data_binary_files=0, credential_pattern_scan='PASS',
              completed_results=results, expansion_round_seed_and_code_checks='PASS',
              goal_complete=False, results_source='Archived runner SUMMARY.json; no replay rescan',
              package_hashes=hashes)
(ROOT / 'PACKAGE_CHECK.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in report.items() if k != 'package_hashes'}))
