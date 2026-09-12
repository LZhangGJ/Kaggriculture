"""Offline alternative-plan targets averaged over independent future draws."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
from evaluate import BASELINE,worker
from sweep import PROFILES,summary_row


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed-start',type=int,required=True);p.add_argument('--seeds',type=int,required=True)
    p.add_argument('--samples',type=int,default=4);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--day',type=int,default=9);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--profiles',nargs='+',default=['keep','higher_supply','defensive','crop_expansion','delayed_sales'])
    p.add_argument('--opponents',nargs='+',default=['soil_v219g','moon_v215','thomas_955_v2','native:r2','native:r1','stress:delayed_seller'])
    p.add_argument('--parity',action='store_true');p.add_argument('--original-first',action='store_true')
    p.add_argument('--prefix-selectors',type=Path,nargs='+',default=[])
    p.add_argument('--no-plan-features',action='store_true')
    p.add_argument('--common-randomness',action='store_true')
    p.add_argument('--restore-base-day',type=int)
    p.add_argument('--suffix-selectors',type=Path,nargs='+',default=[])
    p.add_argument('--profile-file',type=Path)
    p.add_argument('--step',type=int)
    p.add_argument('--initial-pass-steps',type=int,default=0)
    p.add_argument('--intraday-common',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=16 or not 1<=a.samples<=32 or not 0<=a.day<=28:raise ValueError('Invalid bounds')
    if a.common_randomness and a.original_first:raise ValueError('Common scenarios cannot claim original-seed controls')
    if a.intraday_common and not a.common_randomness:raise ValueError('Intraday host requires common randomness')
    if a.step is not None and (a.step<0 or a.step//24!=a.day):raise ValueError('Step and day mismatch')
    a.out.mkdir(parents=True,exist_ok=False);base=json.loads((BASELINE/'policy/config.json').read_text())
    profiles=json.loads(a.profile_file.read_text()) if a.profile_file else {n:PROFILES[n] for n in a.profiles}
    if any(not set(v)<=set(base) for v in profiles.values()):raise ValueError('Profiles must contain flat policy settings')
    seeds=list(range(a.seed_start,a.seed_start+a.seeds));jobs=[]
    engine='direct_future_parity' if a.parity else 'direct_future'
    if a.common_randomness:engine='direct_common_parity' if a.parity else 'direct_common'
    host='native_common_host' if a.common_randomness else 'native_future_host'
    if a.intraday_common:
        engine='direct_common_intraday_parity' if a.parity else 'direct_common_intraday'
        host='native_common_intraday_host'
    for seed in seeds:
        for opponent in a.opponents:
            for seat in (0,1):
                for sample in range(a.samples):
                    identity=f'robust90-future:{seed}:{opponent}:{seat}:{sample}'
                    future_seed=seed if a.original_first and sample==0 else int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4],'little')
                    config=dict(base,__engine=engine,__switch={'day':a.day,'config':{}},__branch_profiles=profiles,__future_seed=future_seed)
                    if a.step is not None:config['__switch']['step']=a.step
                    if a.initial_pass_steps:config['__initial_pass_steps']=a.initial_pass_steps
                    if not a.no_plan_features:config['__plan_profiles']=profiles
                    if a.prefix_selectors:config['__prefix_selectors']=[str(p.resolve()) for p in a.prefix_selectors]
                    if a.suffix_selectors:config['__suffix_selectors']=[str(p.resolve()) for p in a.suffix_selectors]
                    if a.restore_base_day is not None:config['__restore_base_day']=a.restore_base_day
                    jobs.append(((opponent,seed,seat,str(a.binary.resolve()),config,None),sample))
    manifest=dict(profiles=profiles,base=base,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),seeds=seeds,
                  opponents=a.opponents,workers=a.workers,switch_day=a.day,prefix_selector_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in a.prefix_selectors],future_samples=a.samples,
                  inspect_plans=not a.no_plan_features,
                  switch_step=a.step if a.step is not None else a.day*24,initial_pass_steps=a.initial_pass_steps,
                  restore_base_day=a.restore_base_day,suffix_selector_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in a.suffix_selectors],
                  original_first=a.original_first,engine=engine,scheduled=len(jobs)*len(profiles),
                  source_contexts=len(seeds)*len(a.opponents)*2,purpose='offline resampled continuations; never primary confirmation',
                  random_coupling='shared exogenous draws' if a.common_randomness else 'native action-dependent random draws',
                  native_engine_sha256=hashlib.sha256(Path(__file__).with_name(host+'.so').read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    snapshot=a.out/'host_source_snapshot';snapshot.mkdir()
    for name in ('future_sweep.py','evaluate.py','features.py','selector.py','native_env.py','native_host.cpp','native_direct_host.cpp','native_future_host.cpp','native_future_host.so'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    if a.common_randomness:
        for name in ('common_env.py','native_common_host.cpp','native_common_host.so','build_common_host.py'):
            shutil.copy2(Path(__file__).with_name(name),snapshot/name)
        shutil.copytree(Path(__file__).parent/'candidates/common-random-host',snapshot/'common-host')
    for i,path in enumerate(a.suffix_selectors):shutil.copy2(path,snapshot/f'suffix_selector_{i}.json')
    if a.intraday_common:
        shutil.copy2(Path(__file__).with_name('native_common_intraday_host.so'),snapshot/'native_common_intraday_host.so')
        shutil.copytree(Path(__file__).parent/'candidates/common-intraday-host',snapshot/'intraday-common-host')
    rows=[];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(worker,j):sample for j,sample in jobs}
        with (a.out/'rows.jsonl').open('w') as stream:
            for future in as_completed(pending):
                sample=pending.pop(future);batch=future.result()
                if not isinstance(batch,list) or len(batch)!=len(profiles):raise ValueError(batch)
                for row in batch:
                    if row.get('runtime_error') or row.get('steps')!=719:raise ValueError(row)
                    row['future_sample']=sample;stream.write(json.dumps(row)+'\n');rows.append(summary_row(row))
                stream.flush()
                if len(rows)%20==0:print(json.dumps(dict(done=len(rows),total=manifest['scheduled'],seconds=time.time()-start)),flush=True)
    result=dict(complete_counterfactual_games=len(rows),source_contexts=manifest['source_contexts'],seconds=time.time()-start,
                primary_evaluation_games=0,profiles={n:dict(games=sum(r['candidate']==n for r in rows),wins=sum(r['win'] for r in rows if r['candidate']==n)) for n in profiles})
    (a.out/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
