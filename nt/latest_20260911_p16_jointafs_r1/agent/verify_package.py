"""Verify immutable shipped files; generated build/runs/__pycache__ are ignored."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    expected=json.loads((ROOT/'MANIFEST_SHA256.json').read_text());bad=[]
    for name,info in expected.items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT)or not p.is_file():bad.append({'file':name,'reason':'missing_or_outside_package'});continue
        data=p.read_bytes()
        if len(data)!=info['bytes']or hashlib.sha256(data).hexdigest()!=info['sha256']:bad.append({'file':name,'reason':'checksum_mismatch'})
    result={'status':'PASS'if not bad else'FAIL','files':len(expected),'mismatches':bad}
    print(json.dumps(result,indent=2));return bool(bad)
if __name__=='__main__':raise SystemExit(main())
