"""Run one frozen phase on the named host and preserve terminal status."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--host', choices=('local', 'wrx90'), required=True)
parser.add_argument('--phase', choices=('preflight', 'benchmark'), required=True)
args = parser.parse_args()
def dump(value):
    path = ROOT / ('HOST_STATUS_' + args.host + '.json')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf8')
    temporary.replace(path)

record = dict(host=args.host, phase=args.phase, stage='running', started_unix=time.time(),
              workers=16, freeze_sha256=hashlib.sha256((ROOT / 'inputs/FREEZE.json').read_bytes()).hexdigest())
dump(record)
command = [sys.executable, '-B', str(ROOT / 'tools/panel_runner.py'), '--bundle', str(ROOT / 'bundle'), 'run',
           '--freeze', str(ROOT / 'inputs/FREEZE.json'), '--phase', args.phase, '--workers', '16',
           '--out', str(ROOT / 'runs' / (args.phase + '-' + args.host))]
if args.phase == 'benchmark':
    cross = json.loads((ROOT / 'CROSS_HOST_PREFLIGHT.json').read_bytes())
    assert cross['status'] == 'PASS' and cross['freeze_sha256'] == record['freeze_sha256']
    command += ['--shard', args.host, '--preflight', str(ROOT / 'runs' / ('preflight-' + args.host))]
try:
    run = subprocess.run(command)
    record.update(stage='complete' if run.returncode == 0 else 'failed', exit_code=run.returncode, finished_unix=time.time())
    dump(record)
    raise SystemExit(run.returncode)
except Exception as error:
    record.update(stage='failed', error=repr(error), finished_unix=time.time())
    dump(record)
    raise
