"""All four observed failures, original control and independent/combined repairs."""
from pathlib import Path
import gzip,hashlib,json,sys,zlib
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'native/build'))
import _dp7_native as n
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
cfg=json.loads((E/'profiles/s4v/configs.json').read_text());old=json.load(gzip.open(E/'receipts/s4u_pool_audit_N50_v1/full_chain_autonomous_g001.json.gz','rt'))['rows']
refs={(r['seed'],r['seat']):r for r in old};keys=[(seed,seat) for seed in (20262703,20262736) for seat in (0,1)]
identity=json.loads((E/'receipts/s4u_fiveway_N50_v1/results.json').read_text())['identities']['g001']
asset=E/identity['asset'];assert sha(asset)==identity['asset_sha256']
rival=n.G001(json.loads(zlib.decompress(asset.read_bytes())))
out=E/'receipts/s4v_escape_regression_v1';out.mkdir(exist_ok=False);summary=[]
for label in ('full_chain_autonomous','full_chain_autonomous_market','full_chain_autonomous_feed','full_chain_autonomous_both'):
    rows=n.pool_audit_batch([k[0] for k in keys],[k[1] for k in keys],cfg[label],1,rival,4)
    (out/(label+'.json.gz')).write_bytes(gzip.compress(json.dumps(rows).encode()))
    if label=='full_chain_autonomous':
        for r in rows:assert r['money']==refs[r['seed'],r['seat']]['money']
    row=dict(label=label,games=len(rows),escaped=sum(sum(d[r['seat']]['escaped']) for r in rows for d in r['production']),
        no_effect=sum(sum(d[r['seat']]['no_effect']) for r in rows for d in r['production']),
        cash=[r['money'][r['seat']] for r in rows]);summary.append(row);print(json.dumps(row),flush=True)
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_TARGETED_FAILURE_REGRESSION_NOT_STRENGTH',build=build,summary=summary,script_sha256=sha(Path(__file__))),indent=2))
