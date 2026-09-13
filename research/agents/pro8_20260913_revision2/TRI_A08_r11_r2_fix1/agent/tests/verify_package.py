"""Verify the release file inventory without invoking policy code."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1];manifest=root/'PACKAGE_SHA256SUMS.txt'
checks=0
for line in manifest.read_text().splitlines():
 expected,name=line.split('  ',1);path=root/name
 assert path.is_file(),name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 checks+=1
identity=json.loads((root/'IDENTITY.json').read_text())
for name,expected in identity['production_files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print(json.dumps({'status':'PASS','files_verified':checks,'production_files_verified':len(identity['production_files']),'native_sha256':identity['native_sha256'],'new_games':0}))
