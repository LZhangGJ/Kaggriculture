"""Run trusted PPO evaluation work before another ordinary arena batch."""
import json
import subprocess
from pathlib import Path


def waiting(root):
    config=Path(root)/'priority-eval.json'
    if not config.exists():return False
    queue=Path(json.loads(config.read_text())['queue'])
    return any(not p.with_suffix('.done').exists() and not p.with_suffix('.failed').exists()
               for p in queue.glob('*.json'))


def work(root):
    config = Path(root) / 'priority-eval.json'
    if not config.exists():
        return False
    cfg = json.loads(config.read_text())
    pending = [p for p in Path(cfg['queue']).glob('*.json')
               if not p.with_suffix('.done').exists() and not p.with_suffix('.failed').exists()]
    if not pending:
        return False
    try:
        subprocess.run(cfg['command'], check=True, timeout=3600)
    except subprocess.SubprocessError as exc:
        # A broken evaluation must not hold the regular arena queue indefinitely.
        for request in pending:
            if request.with_suffix('.done').exists():continue
            error=dict(state='failed',error=repr(exc))
            output=Path(json.loads(request.read_text())['output']);output.mkdir(parents=True,exist_ok=True)
            temp=output/'status.json.tmp';temp.write_text(json.dumps(error));temp.replace(output/'status.json')
            request.with_suffix('.failed').write_text(json.dumps(error))
    return True
