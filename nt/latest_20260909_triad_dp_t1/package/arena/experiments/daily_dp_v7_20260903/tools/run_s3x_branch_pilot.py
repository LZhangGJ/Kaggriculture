"""Engineering pilot for snapshot fidelity and branch isolation."""
from pathlib import Path
import gzip,hashlib,json,sys,time,zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=EXP/'receipts/s3x_branch_pilot_v2';out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('pilot output contains prior evidence; choose a new directory')
    config=json.loads((EXP/'profiles/s3w/configs.json').read_text())['old'];registry=json.loads((EXP/'opponents/registry.json').read_text())['opponents']
    mapping={'pass':(0,None),'yhay81_six_day':(5,None),'yhay81_three_day':(6,None),'ecobot_v7':(7,None)}
    constructors={'g001':(1,native.G001),'g003':(1,native.G001),'boatlee_v29':(2,native.BoatleeV29),'kaito_v58':(3,native.KaitoV58),'lynn_v5':(4,native.LynnV5)}
    hashes={};opponents={}
    for name,(kind,constructor) in constructors.items():
        entry=registry[name];asset=EXP/entry['asset'];source=EXP/entry['source'];assert sha(source)==entry['source_sha256']
        if 'asset_sha256' in entry:assert sha(asset)==entry['asset_sha256']
        hashes[entry['asset']]=sha(asset);hashes[entry['source']]=sha(source);payload=json.loads(zlib.decompress(asset.read_bytes()))
        if name=='lynn_v5':
            for rel,h in payload['source_hashes'].items():assert sha(source.parent/rel)==h
        mapping[name]=(kind,constructor(payload))
    days=[0,1,3,6,9,12,18,24];summary=[];total=0;started=time.perf_counter()
    for name,(kind,agent) in mapping.items():
        tic=time.perf_counter();rows=[]
        # One order-independence witness plus the other three seed/seat cases.
        rows.extend(native.investment_branch_batch([20261401],[0],config,kind,agent,days,1,True))
        rows.extend(native.investment_branch_batch([20261401,20261402,20261402],[1,0,1],config,kind,agent,days,3,False))
        for row in rows:
            assert row['cash']==next(n for n in row['nodes'] if n['day']==0)['choices'][0]['cash']
            assert all(n['keep_before_after'] for n in row['nodes'])
            assert all(n['reverse_equal'] for n in rows[0]['nodes'])
        candidates=sum(len(n['choices']) for r in rows for n in r['nodes']);total+=candidates
        entry=dict(opponent=name,games=len(rows),nodes=sum(len(r['nodes']) for r in rows),candidate_suffixes=candidates,
            transitions=sum(r['transitions'] for r in rows),seconds=time.perf_counter()-tic,
            candidate_counts=sorted({len(n['choices']) for r in rows for n in r['nodes']}))
        summary.append(entry);print(json.dumps(entry),flush=True)
        with gzip.open(out/f'{name}.json.gz','wt',encoding='utf8',compresslevel=1) as f:json.dump(rows,f,separators=(',',':'))
    assert set(mapping)=={'pass','g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7'}
    build=json.loads((EXP/'native/build/build_receipt.json').read_text())
    receipt=dict(status='PASS_BRANCH_PILOT_NOT_STRENGTH_RESULT',days=days,seeds=[20261401,20261402],seats=[0,1],
        config=config,config_source='profiles/s3w/configs.json:old',build=build,input_hashes=hashes,summary=summary,
        total_candidate_suffixes=total,seconds=time.perf_counter()-started,full_goal_complete=False,
        supersedes='s3x_branch_pilot_v1, which omitted G003 because its loader key was absent',
        caveat='True future/opponent continuation is offline labeling only. Each branch changes one day then returns to baseline; this is not an online oracle or multi-stage ceiling.')
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(dict(status=receipt['status'],suffixes=total,seconds=receipt['seconds'])))
if __name__=='__main__':main()
