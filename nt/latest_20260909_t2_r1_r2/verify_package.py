"""Read-only byte integrity and result recount; no Kaggle credentials required."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-manifest',action='store_true');a=ap.parse_args()
    manifest=ROOT/'MANIFEST_SHA256.json'
    if a.freeze_manifest:
        assert not manifest.exists()
        files={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and not any(x in ('__pycache__','runs','build','.venv') for x in p.relative_to(ROOT).parts)}
        manifest.write_text(json.dumps(files,indent=2),encoding='utf-8')
    files=json.loads(manifest.read_text());bad=[p for p,h in files.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not bad,bad
    expected={'seven':{'T2':1217,'R1':1291,'R2':1285},'710':{'T2':103,'R1':145,'R2':148}};results={}
    for suite,versions in expected.items():
        results[suite]={}
        for version,wins in versions.items():
            rows=json.loads(gzip.decompress((ROOT/'evidence'/suite/(version+'_rows.json.gz')).read_bytes()))
            assert len(rows)==(1400 if suite=='seven' else 200)
            actual=sum(not r['error'] and r['steps']==719 and r['cash']>r['opponent_cash'] for r in rows);assert actual==wins
            results[suite][version]=dict(games=len(rows),wins=actual)
    for version in ('R1','R2'):
        folder=ROOT/'public'/version;p=json.loads((folder/'PACKAGE_INTEGRITY.json').read_text())
        assert sha(folder/'submission.tar.gz')==p['archive_sha256']
        assert all(sha(folder/n)==h for n,h in p['files'].items())
        assert json.loads((folder/'latest_status.json').read_text())['status']=='COMPLETE'
        assert json.loads((folder/'CLOUD_ACTION_IDENTITY.json').read_text())['status']=='PASS'
    print(json.dumps(dict(status='PASS',files=len(files),bytes=sum((ROOT/p).stat().st_size for p in files),results=results)))

if __name__=='__main__':main()
