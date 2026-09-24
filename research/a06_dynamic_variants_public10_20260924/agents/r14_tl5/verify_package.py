"""Verify the distributed files and the frozen candidate source hashes."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parent
manifest=json.loads((R/'FILES_SHA256.json').read_text())
errors=[]
for name,expected in manifest['files'].items():
 p=R/name
 if not p.is_file():errors.append({'path':name,'error':'missing'})
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append({'path':name,'error':'sha256 mismatch'})
freeze=R/'protocol/freeze.json'
if freeze.exists():
 for name,expected in json.loads(freeze.read_text())['files'].items():
  p=R/'agent'/name
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append({'path':'agent/'+name,'error':'differs from pre-heldout freeze'})
print(json.dumps({'passed':not errors,'checked_files':len(manifest['files']),'errors':errors},indent=2))
raise SystemExit(bool(errors))
