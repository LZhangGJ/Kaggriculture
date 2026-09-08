"""Whole live games, same inputs/action/state/debug for exact memo on/off."""
from pathlib import Path
import argparse,hashlib,json,sys,time,zlib
cli=argparse.ArgumentParser();cli.add_argument('--version',choices=['v1','v2','v3','v4'],default='v1');args=cli.parse_args()
binary_version='v3' if args.version=='v4' else args.version
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E/'native/build'));sys.path.insert(0,str(E/f'native/s4u_memo_probe_{binary_version}'))
import _dp7_native as native
import _dp7_packmemo as memo
b=json.loads((E/'native/build/build_receipt.json').read_text());m=json.loads((E/f'native/s4u_memo_probe_{binary_version}/acceptance.json').read_text())
assert m['status']=='PASS_BUILD' and m['base_build']['binary_sha256']==b['binary_sha256']
for rel,h in b['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
configs=json.loads((E/'profiles/s4t/configs.json').read_text())
g=native.G001(json.loads(zlib.decompress((E/'native/g003_frozen.json.zlib').read_bytes())))
out=E/f'receipts/s4u_memo_live_{args.version}';out.mkdir(exist_ok=False);games=[];steps=[]
stat_fns=[native.portfolio_stats,native.day_value_stats,native.intraday_workforce_stats,native.shared_insertion_stats,native.service_stats,native.idle_handoff_stats]
for label in ('full_workers25','full_chain','full_chain_autonomous'):
    for seed in (20262701,20262702):
        for seat in (0,1):
            env=native.Env(seed);a=native.Controller(configs[label]);c=native.Controller(configs[label]);rs=native.G001State()
            storage=[memo.Cache()] if args.version in ('v3','v4') else []
            local=[]
            while not env.done:
                step=env.step_count
                if args.version=='v4' and step>0 and step%24==0:storage=[memo.Cache()]
                # Alternate execution order to reduce systematic warm-cache bias.
                if step%2:
                    tic=time.perf_counter();original=a.act(env,seat);uncached=time.perf_counter()-tic
                    tic=time.perf_counter();r=memo.act(c,env,seat,True,*storage);cached=time.perf_counter()-tic
                else:
                    tic=time.perf_counter();r=memo.act(c,env,seat,True,*storage);cached=time.perf_counter()-tic
                    tic=time.perf_counter();original=a.act(env,seat);uncached=time.perf_counter()-tic
                assert original==r.pop('action'),('action mismatch',label,seed,seat,step)
                assert a.debug()==c.debug(),('persistent debug mismatch',label,seed,seat,step)
                for fn in stat_fns:assert fn(a)==fn(c),('counter mismatch',fn,label,seed,seat,step)
                local.append(dict(step=step,uncached_seconds=uncached,cached_seconds=cached,**r))
                other=g.act(env,1-seat,rs);env.step([original,other] if seat==0 else [other,original])
            assert len(local)==719
            row=dict(label=label,seed=seed,seat=seat,money=[x['money'] for x in env.observation(seat)['farms']],
                uncached_seconds=sum(x['uncached_seconds'] for x in local),cached_seconds=sum(x['cached_seconds'] for x in local),
                max_uncached=max(x['uncached_seconds'] for x in local),max_cached=max(x['cached_seconds'] for x in local),
                max_cache_accounted_bytes=max(x['accounted_bytes'] for x in local),hits=sum(x['hits'] for x in local),
                calls=sum(x['calls'] for x in local),not_stored=sum(x['not_stored'] for x in local))
            games.append(row);steps.append(dict(label=label,seed=seed,seat=seat,rows=local))
            (out/'progress.json').write_text(json.dumps(games,indent=2));print(json.dumps(row),flush=True)
result=dict(status='PASS_MEMO_ACTION_AND_DEBUG_EQUIVALENCE',build=b,memo_build=m,cache_scope='day' if args.version=='v4' else 'game' if args.version=='v3' else 'decision',games=games,steps=steps,
    full_game_pairs=len(games),action_checks=sum(len(x['rows']) for x in steps),production_source_changed=False,
    caveat='12 paired full games vs actual G003, not a strength or final timeout acceptance. Every issued action and exported persistent/debug counter matched; shared authoritative env advances once with the identical actions.')
(out/'acceptance.json').write_text(json.dumps(result,indent=2))
