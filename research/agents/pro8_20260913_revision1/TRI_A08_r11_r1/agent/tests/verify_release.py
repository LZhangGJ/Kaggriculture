#!/usr/bin/env python3
"""Verify frozen source/native identity and the default root entry on legal prefixes.

This is not a complete game or a win-rate test. No explicit binary override is
passed to main.agent. The source tree can contain production files only; fixtures
are supplied independently. All checks use only Python's standard library.
"""
from __future__ import annotations
import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    fixtures = (args.fixtures or root/'validation/source_input/replays').resolve()
    if args.out.exists():
        raise FileExistsError(f'Refusing to replace a verification receipt: {args.out}')
    started = time.perf_counter()
    receipt = json.loads((root/'policy/tri_a08_r11_r1.BUILD.json').read_text())
    for filename, expected in receipt['sources'].items():
        assert digest(root/filename) == expected, f'Source mismatch: {filename}'
    native = root/'policy/tri_a08_r11_r1.so'
    assert digest(native) == receipt['binary_sha256'], 'Native mismatch'
    checked_manifest = 0
    manifest = root/'MANIFEST.sha256'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            expected, filename = line.split('  ', 1)
            assert digest(root/filename) == expected, f'Manifest mismatch: {filename}'
            checked_manifest += 1
    spec = importlib.util.spec_from_file_location('tri_release_default_entry', root/'main.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected: dict[int, tuple[Path, dict]] = {}
    for path in sorted(fixtures.glob('*.json.gz')):
        data = json.loads(gzip.decompress(path.read_bytes()))
        selected.setdefault(int(data['result']['seat']), (path, data))
    assert set(selected) == {0, 1}, 'Need one legal fixture for each seat'
    rows = []
    for seat, (path, data) in sorted(selected.items()):
        module.reset()
        maximum = 0.0
        for step in range(48):
            observation = data['steps'][step][seat]['observation']
            before = time.perf_counter()
            action = module.agent(observation, None)  # canonical root API, default native
            elapsed = time.perf_counter() - before
            maximum = max(maximum, elapsed)
            assert action == data['actions'][step][seat], (path.name, step, action)
        module.reset()
        assert not module._seats, 'reset() did not close seat instances'
        rows.append({'seat': seat, 'fixture': path.name, 'calls': 48,
                     'matched_saved_prefix': 48, 'max_call_seconds': maximum,
                     'fixture_sha256': digest(path)})
    result = {
        'scope': 'default root main.agent import and saved legal common-prefix smoke; not matches',
        'passed': True, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'root': str(root), 'cwd': str(Path.cwd()), 'explicit_binary_override': False,
        'native_sha256': digest(native), 'sources_checked': len(receipt['sources']),
        'manifest_files_checked': checked_manifest, 'calls': 96, 'matches': 96,
        'rows': rows, 'wall_seconds': time.perf_counter()-started,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'python': sys.version,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
