"""Run new evaluations without changing the frozen v1 campaign."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

from contracts import (DEFAULT_BUNDLE, create_freeze, digest, jobs, journal_rows, read,
                       require, require_complete, result_key, resume_manifest,
                       runtime_environment, sha, verify_freeze, write)


@lru_cache(maxsize=1)
def adapter(bundle):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    path = Path(bundle) / 'evaluation/panel_runner.py'
    spec = importlib.util.spec_from_file_location('frozen_v1_game_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(payload):
    bundle, candidate, panel, seed, opponent, seat, phase = payload
    return adapter(bundle).execute((candidate, panel, seed, opponent, seat, phase))


@contextmanager
def run_lock(output):
    lock = output / 'RUNNING.json'
    write(lock, dict(pid=os.getpid(), created_unix=time.time()), exclusive=True)
    try:
        yield
    finally:
        lock.unlink()


def preflight_receipt(directory, freeze_hash, expected):
    directory = Path(directory)
    status, manifest = read(directory / 'STATUS.json'), read(directory / 'MANIFEST.json')
    require_complete(status, len(expected))
    require(manifest['freeze_sha256'] == freeze_hash and manifest['phase'] == 'preflight',
            'Preflight belongs to another freeze or phase')
    require(manifest['jobs_sha256'] == digest(expected), 'Preflight job plan changed')
    rows = journal_rows(directory / 'rows.jsonl', expected, complete=True)
    for row in rows.values():
        require(not any(row['comparisons'].values()), 'Preflight parity mismatch')
        parity, direct, package = row['parity'], row['direct'], row.get('package')
        require(parity.get('observations_checked') == 1440, 'Preflight did not compare all observations')
        for game in (parity, direct, package):
            if game:
                require(game.get('steps') == 719 and not game.get('runtime_error'), 'Preflight game failed')
                for field in ('steps', 'own_cash', 'opponent_cash', 'margin', 'win', 'tie',
                              'action_hash', 'terminal_shops', 'economy'):
                    require(game[field] == direct[field], 'Preflight comparison receipt is inconsistent')
    receipt = {name: sha(directory / name) for name in ('MANIFEST.json', 'STATUS.json', 'rows.jsonl')}
    return receipt


def build_manifest(freeze_path, bundle, phase, workers, preflight=None, release=None, environment=None):
    _, plan, roster, opponents = verify_freeze(freeze_path, bundle)
    freeze_hash = sha(freeze_path)
    manifest = dict(schema='kaggriculture-run-v2', phase=phase, freeze_sha256=freeze_hash,
                    environment=runtime_environment() if environment is None else environment, workers=workers)
    held = None
    if phase != 'preflight':
        require(preflight is not None, 'A complete v2 preflight is required')
        expected = jobs(plan, roster, opponents, 'preflight')
        manifest['preflight'] = preflight_receipt(preflight, freeze_hash, expected)
        require(read(Path(preflight) / 'MANIFEST.json')['environment'] == manifest['environment'],
                'Execution environment differs from preflight')
    if phase == 'holdout':
        from holdout import verify_release
        require(release is not None, 'A new v2 holdout release is required; the original release is reserved')
        held = verify_release(release, freeze_path, bundle)
        manifest['release_sha256'] = sha(release)
    expected = jobs(plan, roster, opponents, phase, held)
    require(len(expected) == len(set(expected)), 'Duplicate planned result cells')
    manifest.update(jobs_sha256=digest(expected), scheduled=len(expected))
    return manifest, expected, roster


def run(freeze_path, bundle, output, phase, workers=4, preflight=None, release=None):
    require(os.name == 'posix', 'The pinned native runtime requires Linux or WSL')
    require(1 <= workers <= 16, 'Use 1 through 16 CPU workers')
    freeze_path, bundle, output = Path(freeze_path).resolve(), Path(bundle).resolve(), Path(output).resolve()
    manifest, expected, roster = build_manifest(freeze_path, bundle, phase, workers, preflight, release)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        resume_manifest(output / 'MANIFEST.json', manifest)
        seen = journal_rows(output / 'rows.jsonl', expected)
        # Never append to a failed panel or silently retry only its failed cells.
        require(all(row.get('valid') is True for row in seen.values()),
                'Failed rows retained. Document a repair and create a new versioned run')
        if phase == 'holdout':
            from holdout import claim_release
            claim_release(release, output, manifest)
        if len(seen) == len(expected):
            require_complete(read(output / 'STATUS.json'), len(expected))
            return read(output / 'STATUS.json')
        candidates = {c['id']: c for c in roster['candidates']}
        pending = [key for key in expected if key not in seen]
        started = time.time()
        initial = len(seen)

        def status(input_error=None):
            invalid = sum(r.get('valid') is not True for r in seen.values())
            elapsed = time.time() - started
            value = dict(scheduled=len(expected), completed=len(seen), invalid=invalid,
                         missing=len(expected) - len(seen),
                         complete=len(seen) == len(expected) and invalid == 0 and input_error is None,
                         phase=phase, freeze_sha256=sha(freeze_path),
                         games_per_second=(len(seen) - initial) / elapsed if elapsed else 0,
                         updated_unix=time.time(), elapsed_this_process=elapsed)
            if input_error:
                value['input_error'] = input_error
            write(output / 'STATUS.json', value)
            print(json.dumps(value), flush=True)
            return value

        # The adapter uses the already pinned v1 game functions, never its launcher.
        adapter(str(bundle))
        with (output / 'rows.jsonl').open('a', encoding='utf8', buffering=1) as stream, \
                ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
            futures, iterator = {}, iter(pending)

            def fill():
                while len(futures) < workers * 2:
                    key = next(iterator, None)
                    if key is None:
                        break
                    cid, panel, seed, opponent, seat = key
                    payload = (str(bundle), candidates[cid], panel, seed, opponent, seat, phase)
                    futures[pool.submit(execute, payload)] = key

            fill()
            last = 0
            while futures:
                done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    key = futures.pop(future)
                    try:
                        row = future.result()
                        require(result_key(row) == key, 'Worker returned the wrong result cell')
                    except BaseException:
                        cid, panel, seed, opponent, seat = key
                        row = dict(candidate_id=cid, panel=panel, seed=seed, opponent=opponent,
                                   opponent_seat=seat, valid=False, runtime_error=traceback.format_exc())
                    require(key not in seen, 'Duplicate result returned')
                    seen[key] = row
                    stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                fill()
                if time.time() - last >= 30:
                    status()
                    last = time.time()
        try:
            current, _, _ = build_manifest(freeze_path, bundle, phase, workers, preflight, release)
            require(current == manifest, 'Inputs changed during execution')
        except Exception as error:
            status(str(error))
            raise
        value = status()
        require_complete(value, len(expected))
        if phase == 'preflight':
            receipt = preflight_receipt(output, sha(freeze_path), expected)
            write(output / 'VERIFIED.json', dict(freeze_sha256=sha(freeze_path), files=receipt), exclusive=True)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('freeze')
    prepare.add_argument('--out', type=Path, required=True)
    prepare.add_argument('--roster', type=Path)
    prepare.add_argument('--candidate', action='append')
    start = sub.add_parser('run')
    start.add_argument('--freeze', type=Path, required=True)
    start.add_argument('--out', type=Path, required=True)
    start.add_argument('--phase', choices=('preflight', 'development', 'holdout'), required=True)
    start.add_argument('--workers', type=int, default=4)
    start.add_argument('--preflight', type=Path)
    start.add_argument('--release', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        record = create_freeze(args.bundle, args.out, args.roster, args.candidate)
        print(json.dumps(dict(freeze=str(args.out / 'FREEZE.json'), experiment_id=record['experiment_id'])))
    else:
        run(args.freeze, args.bundle, args.out, args.phase, args.workers, args.preflight, args.release)


if __name__ == '__main__':
    main()
