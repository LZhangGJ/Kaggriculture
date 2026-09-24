"""Read-only validation of the exported files and evaluated version identities."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
def verify():
    manifest=json.loads((HERE/'MANIFEST.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():
        path=(HERE/name).resolve()
        assert path.is_relative_to(HERE) and path.is_file(),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    versions=json.loads((HERE/'VERSIONS.json').read_text(encoding='utf-8'))
    for version,info in versions.items():
        for name,digest in info['files'].items():
            assert manifest[f'{version}/{name}']==digest,(version,name)
    return dict(verified_files=len(manifest),versions={v:len(x['files'])for v,x in versions.items()})
if __name__=='__main__':print(json.dumps(verify()))
