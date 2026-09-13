#!/usr/bin/env python3
"""Small, explicit, serial validation commands; never runs the 1536-game panel.

Examples (output must be outside the release directory and must not exist):
  python -B tests/run_validation.py unit --with-sanitizers --out /tmp/tri-unit
  python -B tests/run_validation.py official-floor --out /tmp/tri-quotes
  python -B tests/run_validation.py prefixes --out /tmp/tri-prefixes
  python -B tests/run_validation.py branches --out /tmp/tri-branches
Full matches are opt-in, limited to the predeclared eight parent-opponent games.
"""
from __future__ import annotations
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = [sys.executable, '-B']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['unit', 'official-floor', 'prefixes', 'ablation-prefixes', 'branches', 'audit', 'parent-matches'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--input', type=Path, default=ROOT/'validation/source_input')
    p.add_argument('--binary', type=Path)
    p.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    p.add_argument('--with-sanitizers', action='store_true')
    p.add_argument('--allow-full-games', action='store_true')
    args = p.parse_args()
    out = args.out.resolve()
    source_input = args.input.resolve()
    if out.is_relative_to(ROOT):
        p.error('--out must be outside the frozen release directory')
    if out.exists():
        p.error('--out already exists; use a new directory')
    if args.mode == 'parent-matches' and not args.allow_full_games:
        p.error('Eight complete games require --allow-full-games; these are NOT R2/acceptance games')
    out.mkdir(parents=True)
    records: list[dict] = []
    binary = (args.binary or ROOT/'policy/tri_a08_r11_r1.so').resolve()

    def run(label: str, command: list[str], timeout: int = 120, env: dict | None = None) -> None:
        start = time.perf_counter()
        actual = command
        timer = Path('/usr/bin/time')
        if timer.exists():
            actual = [str(timer), '-v', '-o', str(out/(label+'.time')), *command]
        record = {'label': label, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'command': actual}
        try:
            with (out/(label+'.stdout')).open('w') as stdout, (out/(label+'.stderr')).open('w') as stderr:
                result = subprocess.run(actual, cwd=ROOT, stdout=stdout, stderr=stderr,
                                        timeout=timeout, env=env, check=False)
            record['returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError(f'{label}: exit {result.returncode}; see {label}.stderr')
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['wall_seconds'] = time.perf_counter()-start
            records.append(record)
            (out/(label+'.receipt.json')).write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(record), flush=True)

    def compile_test(name: str, sanitizer: bool = False) -> Path:
        if shutil.which(args.cxx) is None:
            raise RuntimeError(f'C++ compiler unavailable: {args.cxx}')
        label = name+('_asan_ubsan' if sanitizer else '')
        exe = out/label
        flags = (['-std=c++20', '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
                 if sanitizer else ['-std=c++20', '-O2', '-march=x86-64', '-ffp-contract=off'])
        run(label+'.compile', [args.cxx, *flags, str(ROOT/'tests'/(name+'.cpp')), '-o', str(exe)])
        return exe

    scope = 'No full games'
    try:
        if args.mode == 'unit':
            for name in ['sale_dp_tests', 'floor_sale_dp_tests', 'resume_boundary_tests', 'floor_observation_guard', 'floor_margin_regression']:
                exe = compile_test(name)
                run(name, [str(exe)], timeout=90)
                if args.with_sanitizers and name == 'floor_sale_dp_tests':
                    exe = compile_test(name, True)
                    run(name+'_asan_ubsan', [str(exe)], timeout=90,
                        env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                             'UBSAN_OPTIONS': 'halt_on_error=1:print_stacktrace=1'})
        elif args.mode == 'official-floor':
            exe = compile_test('floor_quote_probe')
            run('official_floor', [*PYTHON, str(ROOT/'tests/official_floor_check.py'),
                '--referee', str(source_input/'referee'), '--probe', str(exe), '--out', str(out/'official_floor.json')])
        elif args.mode in ('prefixes', 'ablation-prefixes'):
            ablation = args.mode == 'ablation-prefixes'
            if ablation and args.binary is None:
                binary = ROOT/'research/ablations/floor_off.so'
            rows = []
            for fixture in sorted((source_input/'replays').glob('*.json.gz')):
                name = fixture.name.removesuffix('.json.gz')
                output = out/(name+'.json.gz')
                command = [*PYTHON, str(ROOT/'tests/historical_inputs.py'), '--root', str(ROOT),
                           '--binary', str(binary), '--replay', str(fixture), '--limit', '719', '--out', str(output)]
                if not ablation:
                    command.append('--stop-first-difference')
                run(name, command)
                summary = json.loads(gzip.decompress(output.read_bytes()))['summary']
                if ablation:
                    assert summary['calls'] == summary['match_historical_actions'] == 719
                    assert summary['first_historical_difference'] is None
                rows.append(summary)
            assert len(rows) == 7
            (out/'PREFIX_SUMMARY.json').write_text(json.dumps({'new_games': 0, 'rows': rows}, indent=2)+'\n')
        elif args.mode == 'branches':
            for case_file in sorted((ROOT/'validation/cases').glob('*.json')):
                case = json.loads(case_file.read_text())
                for version in ['parent', 'candidate']:
                    root = source_input/'agent' if version == 'parent' else ROOT
                    native = root/'policy/a08_r11.so' if version == 'parent' else binary
                    label = case['id']+'_'+version
                    run(label, [*PYTHON, str(ROOT/'tests/short_sale_branches.py'), '--root', str(root),
                        '--binary', str(native), '--fixture', str(source_input/'replays'/case['fixture']),
                        '--case', str(case_file), '--referee', str(source_input/'referee'),
                        '--out', str(out/(label+'.json.gz'))])
        elif args.mode == 'audit':
            run('saved_actions_audit', [*PYTHON, str(ROOT/'tests/audit_saved_replays.py'),
                '--input', str(source_input), '--out', str(out/'audit')], timeout=180)
        elif args.mode == 'parent-matches':
            if args.binary is not None:
                p.error('parent-matches uses the frozen root default binary only')
            scope = 'Eight complete parent-opponent development games; NOT R2/public-pool/holdout acceptance'
            results = []
            for seed in [4145681829, 4145681832, 4145681843, 4145681844]:
                for seat in [0, 1]:
                    label = f'seed{seed}_seat{seat}'
                    run(label, [*PYTHON, str(ROOT/'tests/closed_loop_match.py'), '--candidate', str(ROOT),
                        '--parent', str(source_input/'agent'), '--referee', str(source_input/'referee'),
                        '--seed', str(seed), '--candidate-seat', str(seat), '--out', str(out/label)], timeout=120)
                    result = json.loads((out/label/'result.json').read_text())
                    assert result['complete'] and result['steps'] == 719 and not result['errors']
                    results.append(result)
            (out/'MATCH_SUMMARY.json').write_text(json.dumps({'scope': scope, 'games': len(results),
                'wins': sum(r['candidate_strict_win'] for r in results),
                'draws': sum(r['draw'] for r in results), 'results': results}, indent=2)+'\n')
    finally:
        (out/'COMMANDS.json').write_text(json.dumps({'mode': args.mode, 'scope': scope,
            'native_sha256': sha(binary) if binary.exists() else None, 'commands': records}, indent=2)+'\n')

if __name__ == '__main__':
    main()
