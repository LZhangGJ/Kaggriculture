"""Resumable, hash-frozen official-rule dual-panel experiments."""
from pathlib import Path
import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/local_dynamic4_vs_public_top5_20260923'))
from run_matchups import play


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def source_hashes(folder):
    return {p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc' and 'build' not in p.relative_to(folder).parts}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('smoke','screen','development','holdout','comparison'))
    p.add_argument('--candidate',action='append'); p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=16); p.add_argument('--seed-count',type=int)
    p.add_argument('--opponent',action='append',help='Development/screen diagnostic subset only')
    p.add_argument('--group',choices=('internal','external'),help='Development/comparison subset only')
    p.add_argument('--seeds-file',default='SEEDS.json')
    p.add_argument('--freeze-file',default='FINAL_FREEZE.json')
    args=p.parse_args()
    assert 1<=args.workers<=16
    manifest=read(HERE/'CANDIDATES.json')
    ids=args.candidate or [r['id'] for r in manifest]
    assert len(set(ids))==len(ids) and set(ids)<={r['id'] for r in manifest}
    pool=read(HERE/'SOURCE_POOL.json'); rivals=pool['opponents']
    engine=ROOT/'dp/AFS_R2_DP_Fusion_R3_Experimental_Delivery_ver b/Kaggriculture_Fusion_R2_20260916/verification/referee/official/kaggriculture.py'
    assert sha(engine)==pool['engine_sha256']
    for r in rivals:
        for file,digest in r['files'].items(): assert sha(Path(r['entry']).parent/file)==digest,(r['id'],file)
    seedfile=read(HERE/args.seeds_file)
    if args.stage=='smoke':
        seeds=seedfile['development'][:1]; rivals=[r for r in rivals if r['id']=='external/n14_prvsiyan']
    elif args.stage=='screen':
        seeds=seedfile['development'][:args.seed_count or 4]
        chosen={'internal/r14_cashflow','internal/r14_asset_tail','internal/r13_nml','internal/r14_liquidity',
                'external/n14_prvsiyan','external/n69_hosen42','external/d24_02_guruprasaathas111','external/d24_20_haideptry'}
        rivals=[r for r in rivals if r['id'] in chosen]; assert len(rivals)==8
    elif args.stage=='development': seeds=seedfile['development'][:args.seed_count or 8]
    else:
        assert not args.seed_count,'Do not truncate final holdout'
        if args.stage=='holdout':assert not args.opponent and not args.group,'Do not truncate final holdout opponents'
        frozen=read(HERE/args.freeze_file)
        assert ids==[frozen['candidate']]
        assert source_hashes(HERE/'candidates'/ids[0])==frozen['files']
        seeds=seedfile['sealed_holdout']
    if args.opponent:
        assert set(args.opponent)<={r['id'] for r in rivals}
        rivals=[r for r in rivals if r['id'] in args.opponent]
    if args.group:
        rivals=[r for r in rivals if r['group']==args.group]
    candidates={ident:source_hashes(HERE/'candidates'/ident) for ident in ids}
    jobs=[(ident,str(HERE/'candidates'/ident/'main.py'),r['id'],r['entry'],seed,seat)
          for ident in ids for r in rivals for seed in seeds for seat in (0,1)]
    out=HERE/'runs'/args.out; out.mkdir(parents=True,exist_ok=True)
    protocol=dict(stage=args.stage,candidates=candidates,opponents=rivals,seeds=seeds,total_games=len(jobs),
                  workers=args.workers,engine_sha256=pool['engine_sha256'],strict_win_definition='terminal own cash > rival cash; draws are not wins')
    proto=out/'PROTOCOL.json'
    if proto.exists(): assert read(proto)==protocol,'Existing run protocol differs'
    else: save(proto,protocol)
    log=out/'games.jsonl'
    rows=[json.loads(s) for s in log.read_text().splitlines() if s] if log.exists() else []
    def key(r):return r['local'],r['public'],r['seed'],r['local_seat']
    completed={key(r) for r in rows}; assert len(completed)==len(rows)
    jobs=[j for j in jobs if (j[0],j[2],j[4],j[5]) not in completed]
    started=time.monotonic()
    print(json.dumps(dict(done=len(rows),pending=len(jobs),total=protocol['total_games'])),flush=True)
    with log.open('a',encoding='utf-8') as stream:
        for start in range(0,len(jobs),128):
            with cf.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as executor:
                futures=[executor.submit(play,j) for j in jobs[start:start+128]]
                for future in cf.as_completed(futures):
                    row=future.result(); rows.append(row); stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    if len(rows)%32==0 or row['error']:
                        print(json.dumps(dict(done=len(rows),total=protocol['total_games'],errors=sum(bool(r['error']) for r in rows),elapsed_s=round(time.monotonic()-started,1))),flush=True)
                        if row['error']: print(row['error'][-1200:],flush=True)
    summary=dict(games=len(rows),expected=protocol['total_games'],errors=sum(bool(r['error']) for r in rows),complete_719=sum(r.get('steps')==719 for r in rows))
    save(out/'SUMMARY.json',summary); print(json.dumps(summary),flush=True)
    assert summary['games']==summary['expected']==summary['complete_719'] and summary['errors']==0


if __name__=='__main__':main()
