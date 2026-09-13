"""Rebuild the frozen Pro8 workspace from the published inputs and journals."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

BASE = '86cf19cd528cb071a1fed019e3bc84e45a007e62'
BUNDLE = 'research/evaluation_sets/2026-09-12-v1'
EXT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--repo', type=Path, default=EXT, help='Git checkout containing the pinned baseline commit')
parser.add_argument('--inputs-only', action='store_true', help='Prepare a new run without copying completed journals')
args = parser.parse_args()
out = args.out.resolve()
assert not out.exists(), 'Choose a new output directory; existing evidence is preserved'
out.mkdir(parents=True)
repo = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=args.repo,
                           capture_output=True, text=True, check=True).stdout.strip())

with tempfile.TemporaryDirectory(dir=out) as temporary:
    archive = Path(temporary) / 'baseline.tar'
    subprocess.run(['git', 'archive', BASE + ':' + BUNDLE, '--output', str(archive)], cwd=repo, check=True)
    with tarfile.open(archive) as source:
        source.extractall(out / 'bundle', filter='data')
shutil.copytree(EXT / 'runtime/pro8_20260913', out / 'bundle/evaluation/runtime/pro8_20260913')
for name in ('inputs', 'tools'):
    shutil.copytree(EXT / name, out / name)
for name in ('combined_roster.json', 'run_host.py', 'check_cross_host.py', 'join_and_analyze.py',
             'render_master_leaderboard.py'):
    shutil.copyfile(EXT / name, out / name)

if not args.inputs_only:
    shutil.copyfile(EXT / 'CROSS_HOST_PREFLIGHT.json', out / 'CROSS_HOST_PREFLIGHT.json')
    for host in ('local', 'wrx90'):
        shutil.copyfile(EXT / ('HOST_STATUS_' + host + '.json'), out / ('HOST_STATUS_' + host + '.json'))
        for phase in ('preflight', 'benchmark'):
            name = phase + '-' + host
            source, destination = EXT / 'runs' / name, out / 'runs' / name
            destination.mkdir(parents=True)
            for file in source.glob('*.json'):
                shutil.copyfile(file, destination / file.name)
            with gzip.open(source / 'rows.jsonl.gz', 'rb') as stream, (destination / 'rows.jsonl').open('wb') as target:
                shutil.copyfileobj(stream, target)

# The frozen checker also verifies the exact runtime inventory and tool files.
import sys
sys.path.insert(0, str(out / 'tools'))
from contracts import verify_freeze
verify_freeze(out / 'inputs/FREEZE.json', out / 'bundle')
print(json.dumps({'status': 'PASS', 'workspace': str(out), 'copied_completed_journals': not args.inputs_only}))
