"""Verify the delivered bytes before rebuilding (build receipts change on rebuild)."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    manifest = ROOT / 'MANIFEST.sha256'
    failures = []
    count = 0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Unsafe manifest path: {name}')
        path = ROOT / rel
        count += 1
        if not path.is_file() or digest(path) != expected:
            failures.append(name)
    identity = json.loads((ROOT / 'IDENTITY.json').read_text())
    if digest(ROOT / 'policy/a06.so') != identity['native_sha256']:
        failures.append('identity:production native')
    if digest(ROOT / 'parent/policy/a06.so') != identity['parent_native_sha256']:
        failures.append('identity:parent native')
    if failures:
        print(json.dumps({'pass': False, 'failures': failures}, indent=2))
        sys.exit(1)
    print(json.dumps({'pass': True, 'files_checked': count,
                      'native_sha256': identity['native_sha256'],
                      'new_games': 0}, indent=2))

if __name__ == '__main__':
    main()
