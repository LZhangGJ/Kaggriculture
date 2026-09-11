"""Exact product/seed balance from native read-only unit-phase projections."""
from pathlib import Path
import argparse,hashlib,json,shutil,statistics,sys,time,zlib
EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native


def total(days,seat):
    d=[x[seat] for x in days]
    out={k:[sum(x[k][i] for x in d) for i in range(len(d[0][k]))]
         for k in d[0] if k not in ('end_private','end_field','end_seeds','checks')}
    out.update({k:d[-1][k] for k in ('end_private','end_field','end_seeds')})
    out['checks']=sum(x['checks'] for x in d)
    for i in range(12):
        assert out['acquired'][i]+out['bought'][i]-out['sold'][i]-out['used'][i]-out['drop_loss'][i]-out['eod_loss'][i]==out['end_private'][i],('inventory balance',seat,i,out)
    for i in range(9):
        assert out['generated'][i]-out['acquired'][i]-out['unit_discard'][i]-out['environment_loss'][i]==out['end_field'][i],('field balance',seat,i,out)
    for i in range(5):
        assert out['seed_bought'][i]-out['planted'][i]==out['end_seeds'][i],('seed balance',seat,i,out)
    assert out['checks']==719
    return out


def means(rows):
    return {k:([statistics.fmean(r[k][i] for r in rows) for i in range(len(v))] if isinstance(v,list) else statistics.fmean(r[k] for r in rows)) for k,v in rows[0].items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261401);p.add_argument('--count',type=int,default=50);p.add_argument('--opponents',default='pass,g001')
    a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False,parents=True);cfg=json.loads(Path(a.profile).read_text())
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    for rel,h in build['source_hashes'].items():
        src=EXP/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
        dst=out/'source'/rel;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dst)
    data=json.loads(zlib.decompress((EXP/'native/g001_frozen.json.zlib').read_bytes()));g=native.G001(data)
    tasks=[(seed,seat) for seed in range(a.seed,a.seed+a.count) for seat in (0,1)];summaries=[];start=time.perf_counter()
    for opponent in a.opponents.split(','):
        assert opponent in ('pass','g001')
        engine=g if opponent=='g001' else None
        rows=native.production_audit_batch([x[0] for x in tasks],[x[1] for x in tasks],cfg,engine,16)
        reference=native.batch([x[0] for x in tasks],[x[1] for x in tasks],cfg,engine,16)
        own=[];rival=[]
        for row,ref in zip(rows,reference):
            seat=row['seat'];assert row['money'][seat]==ref['cash'] and row['money'][1-seat]==ref['opponent_cash']
            own.append(total(row['days'],seat));rival.append(total(row['days'],1-seat))
        entry=dict(opponent=opponent,games=len(rows),own=means(own),rival=means(rival),cash=statistics.fmean(r['cash'] for r in reference),wins=sum(r['win'] for r in reference))
        summaries.append(entry)
        (out/f'{opponent}.json').write_text(json.dumps(dict(rows=rows,summary=entry)),encoding='utf-8')
        print(json.dumps({'opponent':opponent,'games':len(rows),'cash':entry['cash'],'wins':entry['wins'],'own_planted':entry['own']['planted'],'own_harvested':entry['own']['acquired'],'own_decay':entry['own']['environment_loss'],'own_no_effect':entry['own']['no_effect']}),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(config=cfg,summaries=summaries,all_quantity_balances_pass=True,build=build,seconds=time.perf_counter()-start),indent=2),encoding='utf-8')


if __name__=='__main__':main()
