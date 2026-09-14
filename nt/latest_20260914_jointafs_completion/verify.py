"""Check the exact submitted artifact and source files without loading x86 code."""
from pathlib import Path
import hashlib,json,tarfile
H=Path(__file__).resolve().parent
if __name__=='__main__':
    m=json.loads((H/'MANIFEST.json').read_text())
    for name,d in m['source_hashes'].items():assert hashlib.sha256((H/'source/policy'/name).read_bytes()).hexdigest()==d,name
    archive=H/'submission.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['submission_sha256']
    with tarfile.open(archive) as t:
        assert len(t.getmembers())==len(m['package_files']) and set(t.getnames())==set(m['package_files'])
        for name,d in m['package_files'].items():
            assert t.getmember(name).isfile();data=t.extractfile(name).read();assert hashlib.sha256(data).hexdigest()==d,name
            if name.endswith('.so'):assert data[:6]==b'\x7fELF\x02\x01' and int.from_bytes(data[18:20],'little')==62
            if name=='main.py':assert data==(H/'main.py').read_bytes()
    print('PASS: 42 source files and exact submitted x86-64 package, submission 56225552')
