"""Cash attribution only: native actions/engine/ledger, both seats, full G001."""
from pathlib import Path
import argparse, hashlib, json, shutil, statistics, sys, time, zlib
EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native


def totals(days,seat):
    records=[d[seat] for d in days]
    out={k:[sum(d[k][i] for d in records) for i in range(len(records[0][k]))]
         for k in ('sales','products','seeds','animals','sold','bought_products','bought_seeds','bought_animals')}
    out.update({k:sum(d[k] for d in records) for k in ('hired','land','hires','lands','checks')})
    out['cash']=records[-1]['end']
    assert records[0]['start']+sum(out['sales'])-sum(out['products'])-sum(out['seeds'])-sum(out['animals'])-out['hired']-out['land']==out['cash']
    assert out['checks']==719
    return out


def mean_totals(rows):
    out={}
    for key,value in rows[0].items():
        out[key]=[statistics.fmean(r[key][i] for r in rows) for i in range(len(value))] if isinstance(value,list) else statistics.fmean(r[key] for r in rows)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--configs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261401);p.add_argument('--count',type=int,default=50)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    configs=json.loads(Path(a.configs).read_text());build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    for rel,h in build['source_hashes'].items():
        source=EXP/rel;assert hashlib.sha256(source.read_bytes()).hexdigest()==h
        dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
    data=json.loads(zlib.decompress((EXP/'native/g001_frozen.json.zlib').read_bytes()));g=native.G001(data)
    tasks=[(seed,seat) for seed in range(a.seed,a.seed+a.count) for seat in (0,1)]
    summary=[];started=time.perf_counter()
    for label,cfg in configs.items():
        for opponent in ('pass','g001'):
            rows=native.cash_audit_batch([x[0] for x in tasks],[x[1] for x in tasks],cfg,g if opponent=='g001' else None,16)
            baseline=native.batch([x[0] for x in tasks],[x[1] for x in tasks],cfg,g if opponent=='g001' else None,16)
            own=[];rival=[]
            for r,b in zip(rows,baseline):
                x=totals(r['days'],r['seat']);y=totals(r['days'],1-r['seat'])
                assert x['cash']==b['cash'] and y['cash']==b['opponent_cash'], 'auditor must not change actions/results'
                own.append(x);rival.append(y)
            entry=dict(label=label,opponent=opponent,config=cfg,games=len(rows),own=mean_totals(own),rival=mean_totals(rival),wins=sum(x['cash']>y['cash'] for x,y in zip(own,rival)))
            summary.append(entry)
            (out/f'{label}_{opponent}.json').write_text(json.dumps(dict(rows=rows,summary=entry)),encoding='utf-8')
            print(json.dumps(entry),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(summaries=summary,build=build,seed=a.seed,count=a.count,seconds=time.perf_counter()-started,all_steps_cash_reconciled=True,opponent_sha256=data['source_sha256']),indent=2),encoding='utf-8')


if __name__=='__main__':main()
