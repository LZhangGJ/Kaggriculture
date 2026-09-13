"""Verify the byte-frozen deliverable without running an agent or a match."""
from pathlib import Path
import argparse,hashlib,json,sys,tarfile
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path);a=p.parse_args();R=a.root.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'DELIVERY_MANIFEST.json').read_text());expected=m['files'];actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p.name!='DELIVERY_MANIFEST.json'}
assert actual==set(expected),{'missing':sorted(set(expected)-actual),'extra':sorted(actual-set(expected))}
for name,entry in expected.items():
 path=R/name;assert path.is_file() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],name
ident=json.loads((R/'IDENTITY.json').read_text());freeze=json.loads((R/'SOURCE_FREEZE.json').read_text());build=json.loads((R/'BUILD.json').read_text())
assert ident['candidate']=='TRI_A06_r12_r2' and ident['exact_parent']=='TRI_A06_r12_r1'
for name,h in freeze['files'].items():assert sha(R/name)==h,name
assert freeze['files']==build['all_build_inputs']
assert sha(R/'policy/a06.so')==ident['native_sha256']==freeze['native_sha256']==build['sha256']
ph=json.loads((R/'provenance/PARENT_FILES_SHA256.json').read_text());base=R/'provenance/parent_agent'
assert {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}==set(ph)
for name,h in ph.items():assert sha(base/name)==h,name
archive=R/'provenance/TRI_A06_r12_r1_immutable_parent.tar.gz';assert sha(archive)==ident['parent_archive_sha256']
with tarfile.open(archive) as tar:
 assert {x.name for x in tar.getmembers()}==set(ph)
 for x in tar:
  assert x.isfile() and hashlib.sha256(tar.extractfile(x).read()).hexdigest()==ph[x.name]
assert sha(R/'policy/config.json')==sha(base/'policy/config.json')
assert (R/'COMPILER_FLAGS.json').read_bytes()==(base/'COMPILER_FLAGS.json').read_bytes()
result={'candidate':ident['candidate'],'manifest_files_verified':len(expected),'source_inputs_verified':len(freeze['files']),'parent_original_files_verified':len(ph),'parent_archive_members_verified':len(ph),'config_and_flags_unchanged':True,'native_sha256':ident['native_sha256'],'new_games':0,'status':'PASS'}
print(json.dumps(result))
if a.out:a.out.write_text(json.dumps(result,indent=2))
