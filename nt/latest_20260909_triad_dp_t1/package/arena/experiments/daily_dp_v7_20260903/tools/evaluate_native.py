"""Whole 719-step matches stay inside C++; Python loads assets and writes receipts."""
from pathlib import Path
import argparse, hashlib, json, shutil, statistics, sys, time, zlib
EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native

def main():
    p=argparse.ArgumentParser();p.add_argument('--opponent',choices=['pass','g001'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--seats',default='0,1');p.add_argument('--repeat',type=int,default=1);p.add_argument('--threads',type=int,default=16);p.add_argument('--variants',default='core');p.add_argument('--params',default='{}');p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    assert a.count>0 and a.repeat>0 and 1<=a.threads<=16
    data=json.loads(zlib.decompress((EXP/'native/g001_frozen.json.zlib').read_bytes()))
    opponent=native.G001(data) if a.opponent=='g001' else None
    tasks=[(seed,int(seat)) for _ in range(a.repeat) for seed in range(a.seed,a.seed+a.count) for seat in a.seats.split(',')]
    rows=[];summary={};configurations={}
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    snapshots=out/'source';snapshots.mkdir()
    for rel,digest in build['source_hashes'].items():
        source=EXP/rel;assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,(rel,'rebuild required')
        destination=snapshots/rel;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    for v in a.variants.split(','):
        assert v in ('core','none','resources','expiry','values','calendar')
        cfg={'fix_'+k:v=='core' or v==k for k in ('resources','expiry','values','calendar')};cfg.update(json.loads(a.params));configurations[v]=cfg
        start=time.perf_counter();r=native.batch([x[0] for x in tasks],[x[1] for x in tasks],cfg,opponent,a.threads);wall=time.perf_counter()-start
        assert len(r)==len(tasks) and all(x['steps']==719 for x in r)
        # Repeated runs are a throughput/reentrancy check, never extra independent evidence.
        by_key={}
        for x in r:
            key=(x['seed'],x['seat']);signature=(x['cash'],x['opponent_cash'],x['g001_switched'])
            assert key not in by_key or by_key[key]==signature,('cross-game contamination',key)
            by_key[key]=signature;x['variant']=v
        unique=r[:len(r)//a.repeat]
        summary[v]=dict(games=len(r),unique_seed_count=a.count,unique_seed_seat_games=len(unique),wins=sum(x['win'] for x in unique),mean_cash=statistics.fmean(x['cash'] for x in unique),mean_opponent_cash=statistics.fmean(x['opponent_cash'] for x in unique),mean_margin=statistics.fmean(x['margin'] for x in unique),min_cash=min(x['cash'] for x in unique),g001_switch_games=sum(x['g001_switched'] for x in unique),wall_seconds=wall,games_per_second=len(r)/wall,transitions_per_second=719*len(r)/wall,mean_game_single_worker_seconds=statistics.fmean(x['seconds'] for x in r))
        rows+=r;print(json.dumps({v:summary[v]}),flush=True)
    receipt=dict(args=vars(a),configurations=configurations,build=build,opponent_asset_sha256=hashlib.sha256((EXP/'native/g001_frozen.json.zlib').read_bytes()).hexdigest() if opponent else None,opponent_source_sha256=data['source_sha256'] if opponent else None,summary=summary,rows=rows)
    (out/'results.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')

if __name__=='__main__':main()
