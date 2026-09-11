"""Native live audit verifies trace day boundaries and overflow, no tape opponent."""
from pathlib import Path
import gzip,hashlib,json,sys,zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'));import _dp7_native as native
root=EXP/'receipts/s4n_timing_trace_v1';meta=json.loads((root/'acceptance.json').read_text())
panel=json.loads((EXP/'receipts/s4m1_eightway_N50_v1/results.json').read_text());build=json.loads((EXP/'native/build/build_receipt.json').read_text())
assert meta['build']['binary_sha256']==panel['build']['binary_sha256']==build['binary_sha256']
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
refs={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
names=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP'];idx={n:i for i,n in enumerate(names)}
out=EXP/'receipts/s4n_trace_ledger_validation_v1';out.mkdir(exist_ok=False);checked=days=0;outputs=[]
for opponent in ('g001','g003','boatlee_v29','yhay81_six_day'):
    entry=panel['identities'][opponent];agent=None;kind=5
    if opponent!='yhay81_six_day':
        path=EXP/entry['asset'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['asset_sha256'];payload=json.loads(zlib.decompress(path.read_bytes()))
        kind=2 if opponent=='boatlee_v29' else 1;agent=native.BoatleeV29(payload) if kind==2 else native.G001(payload)
    for label in ('all_intraday','all_intraday_insert'):
        traces=[r for r in meta['rows'] if r['opponent']==opponent and r['label']==label]
        rows=native.pool_audit_batch([r['seed'] for r in traces],[r['seat'] for r in traces],panel['configurations'][label],kind,agent,16)
        for raw,tr in zip(rows,traces):
            ref=refs[label,opponent,tr['seed'],tr['seat']];seat=tr['seat'];assert raw['overflow']==ref['overflow'] and raw['money'][seat]==tr['cash'] and raw['money'][1-seat]==tr['opponent_cash']
            path=root/tr['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==tr['sha256']
            with gzip.open(path,'rt',encoding='utf8') as f:data=json.load(f)
            for d in range(30):
                end=min((d+1)*24,719);o=data['final'] if end==719 else data['trace'][end]['before']
                for p in (0,1):assert raw['cash_ledger'][d][p]['end']==o['farms'][p]['money']
                held=[0]*12
                for inv in [o['private']['shed'],*o['private']['inventories']]:
                    for n,q in inv.items():held[idx[n]]+=q
                field=[0]*12
                for tiles in o['farms'][seat]['tiles']:
                    for t in tiles:
                        if not isinstance(t,dict):continue
                        if 'crop' in t:field[idx[t['crop']]]+=t['yield_units']
                        if 'animal' in t:
                            field[{'GOOSE':5,'COW':6,'SHEEP':7}[t['animal']]]+=t['yield_units'];field[8]+=int(t['fertilizer_available'])
                prod=raw['production'][d][seat]
                assert held==prod['end_private'] and field==prod['end_field'] and [o['private']['seeds'][n] for n in names[:5]]==prod['end_seeds'],(label,opponent,tr['seed'],seat,d)
                days+=1
            checked+=1
        dest=out/f'{label}_{opponent}.json.gz'
        with gzip.open(dest,'wt',encoding='utf8',compresslevel=1) as f:json.dump(rows,f,separators=(',',':'))
        outputs.append(dict(path=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));print(json.dumps(dict(games=checked,days=days)),flush=True)
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_NATIVE_LIVE_TRACE_LEDGER_AND_OVERFLOW',build=build,matched_games=checked,matched_day_boundaries=days,outputs=outputs,
    caveat='Python step tracing API has no overflow accessor. Separate original-agent native audit checks overflow against the frozen panel and matches every trace daily cash/private/field/seed balance. Not additional independent games.'),indent=2))
