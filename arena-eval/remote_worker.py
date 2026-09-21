"""Trusted CPU-only evaluation queue; regular arena owns the scheduling lock."""
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time


def atomic(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj))
    temp.replace(path)


def init(cfg):
    import ctypes
    import signal
    parent=os.getppid()
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError('Unable to bind worker lifetime to parent')
    if os.getppid()!=parent:raise SystemExit('Evaluation parent exited')
    sys.path[:0] = [cfg['source'], cfg['arena']]
    from ppo.league_eval import initialize
    initialize(cfg['arena'])
    if cfg.get('backend') == 'chroot':
        # Keep the exact referee and neural decoder; only sandbox startup differs.
        import chroot_adapter
        chroot_adapter.install(cfg['arena_config'])


def pair(job):
    from ppo.league_eval import seed_pair
    return seed_pair(job)


def run(cfg):
    init(cfg)
    from cache_identity import engine_identity
    for path in sorted(Path(cfg['queue']).glob('*.json')):
        if path.with_suffix('.done').exists() or path.with_suffix('.failed').exists():
            continue
        req = json.loads(path.read_text())
        out = Path(req['output']); out.mkdir(parents=True, exist_ok=True)
        try:
            if engine_identity() != req['engine']:
                raise ValueError('Remote engine identity differs')
            for name, digest in req['files'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Remote file identity differs: ' + name)
            todo = [(key, job) for key, job in req['jobs'] if not (out / (key+'.json')).exists()]
            atomic(out/'status.json', dict(state='running', at=time.time(), jobs=len(req['jobs'])))
            # Bound outstanding work to the number of worker processes.
            pool=mp.get_context('spawn').Pool(min(cfg['workers'],max(1,len(todo))),init,(cfg,))
            active=[]
            try:
                for start in range(0, len(todo), cfg['workers']):
                    if (out/'STOP').exists(): raise InterruptedError('Evaluation stopped')
                    active=todo[start:start+cfg['workers']]
                    futures=[(key,pool.apply_async(pair,(job,))) for key,job in active]
                    for key,future in futures:
                        atomic(out/(key+'.json'), future.get(timeout=1500))
                        atomic(out/'status.json', dict(state='running', at=time.time(), last_pair=key))
            finally:
                pool.terminate();pool.join()
                if cfg.get('backend')!='chroot':
                    from ppo.league_runtime import digest
                    from tools.arena.sandbox import cleanup
                    for key,job in active:
                        for seat in (0,1):cleanup('league-eval-'+digest([job[4],job[1]['name'],job[2],seat])[:16])
            atomic(out/'status.json', dict(state='complete', at=time.time(), jobs=len(req['jobs'])))
            atomic(path.with_suffix('.done'), dict(at=time.time()))
        except Exception as exc:
            atomic(out/'status.json', dict(state='failed', at=time.time(), error=repr(exc)))
            atomic(path.with_suffix('.failed'), dict(error=repr(exc)))
            # Leave evidence and return resources to normal arena work.


if __name__ == '__main__':
    import signal
    def terminate(signum,frame):raise KeyboardInterrupt('Evaluation stopping')
    signal.signal(signal.SIGTERM,terminate)
    run(json.loads(Path(sys.argv[1]).read_text()))
