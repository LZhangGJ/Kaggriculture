"""Freeze source, seeds, evaluation rules, and the pilot compute allowance."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
from .runtime import HERE,REPO,BUNDLE,RUNTIME,sha,dump,public_seeds


def verify_freeze(path):
    path=Path(path);d=json.loads(path.read_text())
    if d['environment']!=environment():raise RuntimeError('pilot runtime environment changed')
    for rel,digest in d['files'].items():
        if sha(REPO/rel)!=digest:raise RuntimeError('frozen input changed: '+rel)
    for rel,digest in d['experiment_files'].items():
        if sha(path.parent/rel)!=digest:raise RuntimeError('frozen experiment input changed: '+rel)
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.suffix in ('.py','.cpp','.hpp','.so')}
    if actual!=set(d['source_paths']):raise RuntimeError('source inventory changed')
    return d


def environment():
    return {'python':sys.version,'platform':platform.platform(),
            'packages':{p:importlib.metadata.version(p) for p in ('numpy','torch','ortools')},
            'libstdcpp_sha256':sha('/usr/lib/x86_64-linux-gnu/libstdc++.so.6')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--reserved',type=Path,required=True)
    p.add_argument('--base-commit',required=True)
    p.add_argument('--seconds',type=int,default=180);args=p.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    checks=args.out.parent.parent
    preflight=json.loads((checks/'preflight-ubuntu24.json').read_text())
    learning=json.loads((checks/'learning-check-ubuntu24.json').read_text())
    assert preflight['passed'] and learning['passed']
    for name in ('native.cpp','planner.hpp','actor.hpp','policy.py'):
        assert preflight['sources'][name]==sha(HERE/name),'correctness receipt source mismatch'
    excluded=public_seeds()
    for relative in ('audit/exclusions.json','campaign_audit/exclusions.json'):
        excluded.update(json.loads((BUNDLE/relative).read_text())['seeds'])
    # Membership only. Reserved seed values never appear in logs, model features,
    # result rows or the public artifact. No holdout games are run.
    reserved=json.loads(args.reserved.read_text())['seeds'];excluded.update(reserved)
    excluded.update(range(174043800,174043900)) # correctness and throughput cases
    rng=random.Random('independent-e1-e2-20260912-v1');seeds=[]
    while len(seeds)<1040:
        s=rng.randrange(2**31)
        if s not in excluded and s not in seeds:seeds.append(s)
    seed_manifest={'train':seeds[:1024],'selection':seeds[1024:1032],'pilot_test':seeds[1032:],
                   'reserved_manifest_sha256':sha(args.reserved),'excluded_count':len(excluded),
                   'scope':'local pilot split; final confirmation requires a new custodian release'}
    dump(args.out/'seeds.json',seed_manifest)
    if len(args.base_commit)!=40 or any(c not in '0123456789abcdef' for c in args.base_commit):raise ValueError('expected verified Git commit hash')
    protocol={'name':'independent-e1-e2-cpu-pilot-v1','base_commit':args.base_commit,
      'cpu_seconds_per_trial':args.seconds,'training_seeds':[1101,2202,3303],
      'search_methods':['beam','lns'],'ppo_entropy':[.003,.01],'ppo_lr':.0003,'ppo_envs':4,
      'search_training_worlds_per_trial':4,'search_training_opponents':['own-wheat','own-melon','own-cow'],
      'ppo_training_mixture':'half current self-play; half the same three own-plan controls, rotated across updates',
      'budget_unit':'process CPU seconds, includes all native and Torch worker threads; finish an in-flight complete rollout/update',
      'max_concurrent_trials':2,'no_gpu':True,'primary_metric':'strict win rate','secondary_metric':'match score',
      'configuration_selection':'mean strict win rate over selection worlds and three training seeds; then match score; then mean margin',
      'pilot_opponents':['native:r1','native:r2','extra:old-champion','extra:hysteresis'],
      'opponent_scope':'four fixed benchmark opponents; limited pilot coverage, no claim of independent ancestry',
      'analysis':'seed-cluster paired bootstrap; retain all opponents/seats and average training seeds within world; also report each training seed',
      'bootstrap_resamples':4000,'promotion':False,
      'advance':'reproducible improvement over own control on fresh pilot worlds; positive paired lower bound plus no failures/runtime breaches',
      'limited_power':'eight fresh pilot worlds cannot certify leaderboard gains; no adaptive repeat on this split',
      'jax_backend':'source audit only; GPU rates in team docs are not current host measurements'}
    dump(args.out/'PROTOCOL.json',protocol)
    files=list(p for p in HERE.rglob('*') if p.suffix in ('.py','.cpp','.hpp','.so'))
    source_paths=[str(p.relative_to(HERE)) for p in files]
    files += [p for p in RUNTIME.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix not in ('.pyc','.pyo')]
    files += [BUNDLE/'manifests'/f'{n}.json' for n in ('representative','stress','stress_pool')]
    files += [BUNDLE/'opponents.json',BUNDLE/'campaign_audit/exclusions.json',BUNDLE/'audit/exclusions.json',
              HERE/'requirements-tested.txt',checks/'preflight-ubuntu24.json',checks/'learning-check-ubuntu24.json',checks/'eval-smoke/results.json']
    freeze={'files':{str(p.relative_to(REPO)):sha(p) for p in files},'source_paths':source_paths,
            'experiment_files':{name:sha(args.out/name) for name in ('seeds.json','PROTOCOL.json')},
            'environment':environment()}
    dump(args.out/'FREEZE.json',freeze);verify_freeze(args.out/'FREEZE.json')
    print(json.dumps({'frozen':True,'files':len(files),'train':1024,'selection':8,'pilot_test':8,'reserved_values_printed':False}))


if __name__=='__main__':main()
