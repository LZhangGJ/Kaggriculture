"""Run exactly the frozen twelve CPU trials; fail without retrying a trial."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from .protocol import verify_freeze
from .runtime import dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    verify_freeze(root/'inputs/FREEZE.json');cfg=json.loads((root/'inputs/PROTOCOL.json').read_text())
    trials=root/'trials';trials.mkdir(exist_ok=False);logs=root/'logs';logs.mkdir(exist_ok=False)
    start=time.perf_counter();completed=[]
    def run(job):
        name,module,extra=job
        cmd=[sys.executable,'-B','-m','research.independent_pilot.'+module,'--out',str(trials/name),
             '--seconds',str(cfg['cpu_seconds_per_trial']),'--seed-manifest',str(root/'inputs/seeds.json'),*extra]
        with (logs/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        if result.returncode:raise RuntimeError(f'{name} failed with exit {result.returncode}; see its preserved log')
        return name
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(method,entropy) in enumerate(zip(cfg['search_methods'],cfg['ppo_entropy'],strict=True)):
            for seed in cfg['training_seeds']:
                jobs=[(f'{method}-{seed}','search',['--method',method,'--seed',str(seed)]),
                      (f'ppo-{entropy}-{seed}','ppo',['--entropy',str(entropy),'--lr',str(cfg['ppo_lr']),
                                                    '--seed',str(seed),'--envs',str(cfg['ppo_envs'])])]
                names=[f.result() for f in [pool.submit(run,j) for j in jobs]]
                completed.extend(names);status={'complete':len(completed)==12,'trials_completed':completed,'wall_seconds':time.perf_counter()-start}
                dump(root/'TRAINING_STATUS.json',status);print(json.dumps(status),flush=True)
    verify_freeze(root/'inputs/FREEZE.json')


if __name__=='__main__':main()
