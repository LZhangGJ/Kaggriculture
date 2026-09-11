"""Read-only verification of the frozen handoff and paired result identities."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
    for rel,record in manifest.items():
        p=ROOT/rel
        assert p.is_file(),rel
        assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],rel
    receipt=json.loads((ROOT/'development/candidates/candidate_r2p16/startupsupply2.BUILD.json').read_text())
    for rel,h in receipt['sources'].items():
        assert sha(ROOT/'policy'/rel)==h,rel
    assert sha(ROOT/'policy/startupsupply2.so')==receipt['binary_sha256']=='4d7ef55bb79e4b312445a67628dd15609421e8c9afff21e8192a9b1b055f875b'
    assert sha(ROOT/'baselines/p12.so')=='f62c0a303133d28eb93380f52aaead97ff9e7b0ca2613668ac859b638f9adf63'
    assert sha(ROOT/'baselines/original.so')=='877f196113692722edc8a5b3e30d2d3dd1d4700b0772c13f1e3f2ad7a9ac0ef2'
    assert sha(ROOT/'referee/official/kaggriculture.py')=='bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
    assert json.loads((ROOT/'policy/config.json').read_text())==json.loads((ROOT/'evidence/development/p16/CONFIG.json').read_text())
    expected={'development':{'original':1501,'p12':1723,'p16':1790},
              'new32_block1':{'original':558,'p12':547,'p16':548},
              'new32_block2':{'original':491,'p12':501,'p16':567}}
    totals={}; common={v:[] for v in ('original','p12','p16')}
    for group,versions in expected.items():
        identities=None
        for version,wins in versions.items():
            rows=json.loads(gzip.decompress((ROOT/f'evidence/{group}/{version}/rows.json.gz').read_bytes()))
            keys={(r['opponent'],r['seed'],r['opponent_seat']) for r in rows}
            assert len(keys)==len(rows)==(2200 if group=='development' else 704)
            assert all(r['steps']==719 and r['runtime_error'] is None for r in rows)
            assert all(r['r2_margin']==r['r2_cash']-r['opponent_cash'] and r['r2_win']==(r['r2_margin']>0) and r['tie']==(r['r2_margin']==0) for r in rows)
            assert sum(r['r2_win'] for r in rows)==wins,(group,version)
            assert identities is None or identities==keys,'Unpaired panels'
            identities=keys
            totals[group+'/'+version]=dict(games=len(rows),wins=wins,ties=sum(r['tie'] for r in rows))
            if group!='development':common[version].extend(rows)
    for version,rows in common.items():
        assert len({(r['opponent'],r['seed'],r['opponent_seat']) for r in rows})==1408
        assert len({r['seed'] for r in rows})==64
        totals['common64/'+version]=dict(games=len(rows),wins=sum(r['r2_win'] for r in rows),ties=sum(r['tie'] for r in rows))
    assert totals['common64/p16']['wins']==1115 and totals['common64/p16']['ties']==2
    acceptance=json.loads((ROOT/'HANDOFF_ACCEPTANCE.json').read_text())
    assert acceptance['status']=='PASS'
    print(json.dumps(dict(status='PASS',manifest_files=len(manifest),source_files=len(receipt['sources']),
                          recount=totals),indent=2))
if __name__=='__main__':main()
