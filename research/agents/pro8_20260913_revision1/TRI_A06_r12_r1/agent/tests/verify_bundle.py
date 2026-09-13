"""Verify the frozen payload and its actual test/build identities, offline.

Run on a fresh extraction. Build/unit reruns intentionally mutate their logs.
No game is executed and no native policy is loaded by this verifier.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path: str) -> object:
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def verify(skip_manifest: bool) -> dict:
    freeze = load('SOURCE_FREEZE.json')
    build = load('BUILD.json')
    provenance = load('PROVENANCE.json')
    native = sha(ROOT / 'policy/a06.so')
    require(native == freeze['native_sha256'] == build['sha256'] == provenance['candidate_native_sha256'], 'Native hash mismatch')
    require(build == load('policy/a06.BUILD.json'), 'Root and native BUILD receipts differ')
    require(build['all_build_inputs'] == freeze['production_files'], 'Build inputs and frozen inputs differ')
    for rel, wanted in freeze['production_files'].items():
        require(sha(ROOT / rel) == wanted, 'Production input differs: ' + rel)
    parent = ROOT / 'tests/reference_parent'
    require(sha(parent / 'policy/a06.so') == provenance['parent_native_sha256'], 'Parent native mismatch')
    require(sha(ROOT / 'policy/config.json') == sha(parent / 'policy/config.json'), 'Configuration changed from parent')
    actual_changes = sorted(rel for rel, wanted in freeze['production_files'].items() if sha(parent / rel) != wanted)
    require(actual_changes == sorted(provenance['changed_production_files']), 'Changed-production-file list is incorrect')
    current_code = {rel: value for rel, value in freeze['production_files'].items() if Path(rel).suffix in ('.cpp', '.hpp', '.inc')}
    units = load('tests/UNIT_RESULTS.json')
    require(len(units) == 7 and len({row['test'] for row in units}) == 7, 'Expected seven distinct current unit suites')
    for row in units:
        require(row['compile_exit'] == row['exit'] == 0, 'Failed unit: ' + row['test'])
        require(row['production_source_sha256'] == current_code, 'Unit tested different production source: ' + row['test'])
        require(row['source_sha256'] == sha(ROOT / 'tests' / (row['test'] + '.cpp')), 'Unit source changed: ' + row['test'])
    summaries = sorted((ROOT / 'evidence/logs/closed_loop').glob('*.summary.json'))
    require(len(summaries) == 12, 'Final panel must contain exactly 12 games')
    candidate_results = []
    for p in summaries:
        row = json.loads(p.read_text())
        rawpath = p.with_name(p.name.removesuffix('.summary.json') + '.json.gz')
        require(sha(rawpath) == row['raw_sha256'], 'Game raw-log hash mismatch: ' + p.name)
        raw = json.loads(gzip.decompress(rawpath.read_bytes()))
        require(row['steps'] == 719 and row['player_days'] == 60 and row['exception_count'] == 0, 'Game integrity failure: ' + p.name)
        require(len(raw['steps']) == 719 and len(raw['player_days_cash']) == 60, 'Missing raw game records: ' + p.name)
        require([r['step'] for r in raw['steps']] == list(range(719)), 'Missing or duplicate game step: ' + p.name)
        require(row['statuses'] == ['DONE', 'DONE'], 'Incomplete game: ' + p.name)
        require(row['engine_sha256'] == sha(ROOT / 'tests/reference_bundle/referee/official/kaggriculture.py'), 'Game engine identity differs')
        require(row['native_sha256'][1 - row['tested_seat']] == provenance['parent_native_sha256'], 'Wrong opponent identity')
        if row['variant'] == 'candidate':
            require(row['native_sha256'][row['tested_seat']] == native, 'Wrong tested candidate identity')
            candidate_results.append(row)
        else:
            require(row['native_sha256'] == [provenance['parent_native_sha256']] * 2, 'Wrong control identities')
    require(len(candidate_results) == 8, 'Expected eight final candidate games')
    wins = sum(r['strict_win'] for r in candidate_results)
    draws = sum(r['draw'] for r in candidate_results)
    require(wins == 5 and draws == 0, 'Panel result summary differs from raw game receipts')
    prefixes = sorted((ROOT / 'evidence/logs/prefix_final').glob('*.summary.json'))
    require(len(prefixes) == 7, 'Expected seven final prefix probes')
    matched = compared = 0
    for p in prefixes:
        row = json.loads(p.read_text())
        require(row['native_sha256'] == native, 'Wrong prefix candidate identity')
        require(row['completed_candidate_games'] == 0 and row['first_difference'] is not None, 'Prefix incorrectly classified')
        require(row['matched'] == row['first_difference'] and row['compared'] == row['matched'] + 1, 'Prefix counts inconsistent')
        require(row['one_step']['no_saved_suffix_used'], 'Saved suffix used')
        matched += row['matched']
        compared += row['compared']
    require((matched, compared) == (4459, 4466), 'Prefix aggregate differs')
    payload_files = None
    if not skip_manifest:
        manifest = load('MANIFEST.json')
        wanted_files = manifest['files']
        actual_files = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p != ROOT / 'MANIFEST.json' and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        require(actual_files == set(wanted_files), 'Full payload file listing differs from manifest')
        for rel, record in wanted_files.items():
            p = ROOT / rel
            require(p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], 'Payload hash/size mismatch: ' + rel)
        payload_files = len(wanted_files)
    return {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS', 'root': str(ROOT), 'production_inputs': len(freeze['production_files']), 'native_sha256': native, 'unchanged_configuration': True, 'changed_production_files': actual_changes, 'unit_suites': len(units), 'final_panel_completed_games': len(summaries), 'final_candidate_games': len(candidate_results), 'candidate_wins': wins, 'candidate_draws': draws, 'candidate_losses': len(candidate_results) - wins - draws, 'prefix_matched_actions': matched, 'prefix_comparisons': compared, 'payload_files_verified': payload_files, 'scope': 'Frozen artifact, source/native identity and retained evidence verification; not a new game or acceptance run'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-manifest', action='store_true', help='Check production/test evidence before payload manifest is created')
    parser.add_argument('--out', type=Path, help='Optional JSON receipt; write outside this archive to preserve its manifest')
    args = parser.parse_args()
    try:
        result = verify(args.skip_manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('FAIL: ' + str(exc), file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
