"""Read-only file, ZIP, and recorded acceptance checks; no fresh matches."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def verify():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    total=0
    for rel,meta in manifest['files'].items():
        path=(ROOT/rel).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Unsafe path: '+rel)
        if path.stat().st_size!=meta['bytes'] or sha(path)!=meta['sha256']:raise ValueError('Hash/size mismatch: '+rel)
        total+=meta['bytes']
    with zipfile.ZipFile(ROOT/'first_step_reproducer.zip') as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
    for variant in ('o3_nomodref','animal_noalias'):
        data=json.loads((ROOT/f'evidence/diagnosis/{variant}/FULL_TRACE_REGRESSION.json').read_text())
        assert data['games']==data['matched']==len(data['rows'])==400
        assert data['checked_steps']==287600
        assert all(r['status']=='PASS' and r['steps']==719 for r in data['rows'])
    anchor=json.loads((ROOT/'evidence/diagnosis/o3_nomodref/G001_ANCHOR.json').read_text())
    assert anchor['all_match'] and len(anchor['rows'])==2
    assert anchor['sha256']==sha(ROOT/'binaries/agent_gcc13_nomodref.so')
    return dict(status='PASS',files=len(manifest['files']),bytes=total,zip_crc='PASS',recorded_regression='PASS',new_matches_executed=False)
if __name__=='__main__':print(json.dumps(verify(),indent=2))
