"""Train, select with fresh development matches, freeze, then open distinct final seeds."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parent


def save(value):
    temp = ROOT / 'PIPELINE.tmp'
    temp.write_text(json.dumps(dict(heartbeat_unix=time.time(), **value), indent=2), encoding='utf8')
    temp.replace(ROOT / 'PIPELINE.json')


def run(script, *arguments):
    subprocess.run([sys.executable, '-u', str(ROOT / script), *map(str, arguments)], cwd=ROOT, check=True)


try:
    if not (ROOT / 'run/DONE.json').exists():
        save(dict(status='TRAINING'))
        run('train.py')
    else:
        done = json.loads((ROOT / 'run/DONE.json').read_text(encoding='utf8'))
        plan = json.loads((ROOT / 'PLAN.json').read_text(encoding='utf8'))
        assert done['status'] == 'COMPLETE' and done['steps'] == plan['training']['steps']
        assert all((ROOT / 'run' / f'step_{step}.pt').exists() for step in done['candidates'])
    metrics = [json.loads(line) for line in (ROOT / 'run/metrics.jsonl').read_text(encoding='utf8').splitlines()]
    # Only newly trained checkpoints are eligible; old scores are initialization evidence only.
    choices = sorted({m['step']: m for m in metrics}.values(), key=lambda m: (m['loss'], -m['full_action_exact']))
    selected = None
    for panel_index, candidate in enumerate(choices):
        path = ROOT / 'run' / f"step_{candidate['step']}.pt"
        name = f"development_step{candidate['step']}" + ('_ipcfix' if (ROOT / 'EVALUATION_REPAIR.json').exists() else '')
        save(dict(status='DEVELOPMENT', step=candidate['step'], checkpoint=str(path)))
        run('evaluate.py', '--checkpoint', path, '--name', name, '--panel', 'development', '--panel-index', panel_index)
        result = json.loads((ROOT / f'evaluation/{name}/SUMMARY.json').read_text(encoding='utf8'))
        if result['meets_65']:
            selected = path
            break
    if selected is None:
        save(dict(status='NEEDS_FURTHER_TRAINING', reason='No trained candidate passed full-pool development; fresh final seeds remain unused'))
        sys.exit(0)
    shutil.copy2(selected, ROOT / 'selected.pt')
    selected_hash = hashlib.sha256((ROOT / 'selected.pt').read_bytes()).hexdigest()
    save(dict(status='FINAL_EVALUATION', checkpoint=str(selected), selected_sha256=selected_hash))
    run('evaluate.py', '--checkpoint', ROOT / 'selected.pt', '--name', 'fresh_final', '--panel', 'final')
    result = json.loads((ROOT / 'evaluation/fresh_final/SUMMARY.json').read_text(encoding='utf8'))
    save(dict(status='TARGET_MET' if result['meets_65'] else 'NEEDS_FURTHER_TRAINING', final=result,
              selected_sha256=selected_hash, threshold=.65, metric='strict win rate'))
except BaseException as exc:
    if not isinstance(exc, SystemExit):
        save(dict(status='FAILED', error=repr(exc)))
    raise
