"""One frozen learner collects GPU and official-arena games before PPO."""
from collections import Counter
import time
import math
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import shutil
import torch
from ppo.arena_opponents import assign
from ppo.rollout import collect


def collect_hybrid(jobs, pool, gpu_backend, models, reference, base, device, args, world, sampler):
    started = time.perf_counter()
    weights_path = Path(args.output)/'arena-family-weights.json'
    family_weights = None
    if weights_path.exists():
        family_weights = json.loads(weights_path.read_text()).get('weights') or None
    jobs = assign(jobs, pool, args.seed, args.update_number, family_weights)
    arena = [j for j in jobs if j['family'] == 'arena']
    neural = [j for j in jobs if j['family'] != 'arena']
    world=int(os.environ.get('WORLD_SIZE',1))
    if getattr(args,'arena_games',0):
        if len(arena)!=args.arena_games//world:raise ValueError('Hybrid arena allocation mismatch')
    elif len(arena)*4 != len(jobs) or len(neural)*4 != len(jobs)*3:
        raise ValueError('Hybrid game allocation is not 4/8 self-play, 2/8 older, 2/8 arena')
    # collection-overlap-v1: the official sampler draws from its own generator, seeded per update and
    # rank, so official trajectories are reproducible and independent of the GPU collector's RNG.
    rank = int(os.environ.get('RANK', 0))
    official_seed = (args.seed + 1000003*args.update_number + 7919*rank) % (2**63-1)
    if sampler.generator is not None:
        sampler.generator.manual_seed(official_seed)
    official_workers = getattr(args, 'official_workers', 0) or args.workers//world
    def run_official():
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        return collect(arena, models, reference,
            base['worker_quantities'], base['market_quantities'], device,
            official_workers, args.bf16, args.arena_mib, args.burn, args.sequence,
            sampler=sampler, gae_lambda=args.gae_lambda,
            reference_sha=gpu_backend.reference_sha, all_anchors=True,
            reward_beta=args.reward_beta, reward_sigma=args.reward_sigma,
            reward_cash_weight=args.reward_cash_weight, reward_cash_center=args.reward_cash_center,
            reward_shape=args.reward_shape, reward_dense=bool(args.reward_dense))
    if getattr(args, 'overlap_collection', 0):
        # Child process: own CUDA context, so its graph captures and the parent's simulator never share streams.
        if sampler.generator is None:
            raise ValueError('Overlapped collection requires the dedicated official generator')
        spool = Path(tempfile.mkdtemp(prefix=f'official-{rank}-', dir=os.environ.get('PPO_SPOOL_DIR') or '/dev/shm'))
        try:
            spec = dict(device=str(device), bc_reference=str(args.bc_reference), reference_sha=gpu_backend.reference_sha,
                learner_state={k: v.detach().cpu() for k, v in models['learner'].state_dict().items()}, jobs=arena,
                official_workers=official_workers, generator_seed=official_seed,
                args=dict(bf16=args.bf16, arena_mib=args.arena_mib, burn=args.burn, sequence=args.sequence, gae_lambda=args.gae_lambda, reward_beta=args.reward_beta, reward_sigma=args.reward_sigma, reward_cash_weight=args.reward_cash_weight, reward_cash_center=args.reward_cash_center, reward_shape=args.reward_shape, reward_dense=bool(args.reward_dense)))
            torch.save(spec, spool/'spec.pt')
            log = open(spool/'child.log', 'w')
            child = subprocess.Popen([sys.executable, '-u', '-m', 'ppo.official_worker', str(spool)], stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, RANK=str(rank)))
            try:
                gpu_episodes, gpu_metrics = gpu_backend.collect(neural)
            except BaseException:
                child.kill(); child.wait(); raise
            code = child.wait()
            log.close()
            if code != 0 or not (spool/'result.pt').exists():
                raise RuntimeError('Official collection child failed (rc=%s): %s' % (code, (spool/'child.log').read_text()[-2000:]))
            result = torch.load(spool/'result.pt', map_location='cpu', weights_only=False)
            official_episodes, official_metrics = result['episodes'], result['metrics']
            official_metrics['child_log_tail'] = (spool/'child.log').read_text()[-400:]
        finally:
            shutil.rmtree(spool, ignore_errors=True)
    else:
        gpu_episodes, gpu_metrics = gpu_backend.collect(neural)
        official_episodes, official_metrics = run_official()
    # The first-collection lambda comparison applies to GPU collection only.
    # Do not mislabel it as a comparison covering the external rollouts.
    for ep in gpu_episodes:
        ep.pop('comparison_advantages', None)
    if official_metrics['full_seasons'] != len(arena):
        raise RuntimeError('Real arena batch has invalid or incomplete games; no PPO update')
    episodes = gpu_episodes + official_episodes
    expected = len(jobs)*3//2*719
    if sum(len(e['turns']) for e in episodes) != expected:
        raise RuntimeError('Hybrid learner-turn coverage mismatch')
    if os.environ.get('PPO_SPIKE_DUMP'):  # isolated equality proof only; unset in production
        import sys as _sys
        dump=dict(overlap=int(getattr(args,'overlap_collection',0)),official_workers=official_workers,
            official=[dict(game=e['assignment']['game'],seat=e['seat'],outcome=e['outcome'],shaped_return=e['shaped_return'],old_logp=[float(t['old_logp']) for t in e['turns']],factor_count=[int(t['factor_count']) for t in e['turns']],value=[float(t['value']) for t in e['turns']]) for e in official_episodes],
            official_games=sorted([(g['game'],g['cash']) for g in official_metrics['games']]),
            gpu_games=sorted([(g['game'],g['cash']) for g in gpu_metrics['games']]),gpu_logp=gpu_backend.rollout.data['logp'].detach().cpu(),gpu_value=gpu_backend.rollout.data['value'].detach().cpu(),
            gpu_outcomes=[int(e['outcome']) for e in gpu_episodes],gpu_returns=[float(e['shaped_return']) for e in gpu_episodes],
            timing=dict(gpu=gpu_metrics['collection_seconds'],official=official_metrics['collection_seconds'],wall=time.perf_counter()-started,sampler=official_metrics['sampler_seconds'],ipc=official_metrics['ipc_seconds']))
        torch.save(dump,os.environ['PPO_SPIKE_DUMP'])
        print(json.dumps(dict(stage='spike_dump',path=os.environ['PPO_SPIKE_DUMP'],**dump['timing'],overlap=dump['overlap'],workers=official_workers)),flush=True)
        _sys.stdout.flush(); os._exit(0)
    seconds = time.perf_counter()-started
    games = gpu_metrics['games'] + official_metrics['games']
    if any(g['cash'] is None or len(g['cash'])!=2 or not all(math.isfinite(x) for x in g['cash']) for g in games):
        raise FloatingPointError('Nonfinite hybrid terminal cash; no PPO update')
    # Avoid repeating the immutable pool configuration in every metric row.
    games = [{k:v for k,v in g.items() if k not in
              ('arena_config','arena_repo','arena_runtime_files','arena_opponent','arena_container_name')}
             | ({'arena_agent_id':g['arena_opponent']['id']} if 'arena_opponent' in g else {})
             for g in games]
    return episodes, dict(games=games, complete_games=len(games), valid_games=len(games),
        full_seasons=len(games), learner_turns=expected, collection_seconds=seconds,
        games_per_second=len(games)/seconds, learner_turns_per_second=expected/seconds,
        families=dict(Counter(g['family'] for g in games)),
        arena_families=dict(Counter(g['arena_family'] for g in games if 'arena_family' in g)),
        gpu_collection_seconds=gpu_metrics['collection_seconds'],
        official_collection_seconds=official_metrics['collection_seconds'],
        official_actor_startup_seconds=official_metrics['actor_startup_seconds'],
        official_actor_timings=official_metrics['actor_timings'],
        gpu_peak_allocated_gib=gpu_metrics.get('peak_allocated_gib'),
        arena_family_weights=family_weights,
        protocol='real-arena-training-v3')
