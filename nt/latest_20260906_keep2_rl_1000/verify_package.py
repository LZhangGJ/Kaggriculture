"""Verify only the light package; do not claim absent rollout arrays were audited again."""
from pathlib import Path
import gzip,hashlib,json,math
P=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    m=read(P/'PACKAGE_MANIFEST.json')
    for relative,entry in m['files'].items():
        p=(P/relative).resolve();assert p.is_relative_to(P.resolve())
        data=p.read_bytes();assert len(data)==entry['bytes'] and sha(data)==entry['sha256'],relative
    assert len(m['files'])==m['count']
    for e in read(P/'SOURCE_PROVENANCE.json'):
        b=(P/e['path']).read_bytes()
        if e['transform']=='gzip_lossless':b=gzip.decompress(b)
        assert sha(b)==e['sha256'],e['path']
    tested=games=0
    for f in P.rglob('games.json.gz'):
        with gzip.open(f,'rt',encoding='utf8')as s:rows=json.load(s)
        summary=read(f.parent/'summary.json')['overall']
        assert len(rows)==summary['games'] and sum(r['win']for r in rows)==summary['wins']
        for key in ('cash','margin'):
            assert math.isclose(sum(r[key]for r in rows)/len(rows),summary['mean_'+key],abs_tol=1e-6)
        assert all(not r['error'] and r['steps']==719 for r in rows)
        assert len({(r['seed'],r['seat'],r['opponent'])for r in rows})==len(rows)
        tested+=1;games+=len(rows)
    for e in read(P/'MODEL_INDEX.json'):
        data=(P/e['bin']).read_bytes();assert len(data)==70402*4 and sha(data)==e['bin_sha256']
        if e['pt']:assert (P/e['pt']).is_file()
    receipt=read(P/'PORTABLE_ACCEPTANCE.json');assert receipt['status']=='PASS' and receipt['games']==812
    print(json.dumps(dict(status='PASS',files=m['count'],game_configurations=tested,result_games_verified=games,models=len(read(P/'MODEL_INDEX.json'))),indent=2))
if __name__=='__main__':main()
