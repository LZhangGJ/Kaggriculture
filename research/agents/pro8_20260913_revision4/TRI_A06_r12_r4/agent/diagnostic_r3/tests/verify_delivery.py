"""Verify an extracted delivery without rebuilding or altering its source."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();t=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
manifest=read(r/'MANIFEST.json'); checks=0
for name,row in manifest['files'].items():
 f=r/name;assert f.is_file() and f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],name;checks+=1
actual={str(p.relative_to(r)) for p in r.rglob('*') if p.is_file()}
assert actual==set(manifest['files'])|{'MANIFEST.json'},sorted(actual-set(manifest['files'])-{'MANIFEST.json'})
id=read(r/'IDENTITY.json');assert sha(r/'policy/a06.so')==id['native_sha256'];assert sha(r/'parent/policy/a06.so')==id['parent_native_sha256']
source=read(r/'provenance/MANIFEST.json')
for name,row in source.items():
 f=r/'parent'/name.removeprefix('agent/') if name.startswith('agent/') else r/'provenance'/name
 assert sha(f)==row['sha256'],name
feedback=read(r/'feedback/MANIFEST.json')
for name,row in feedback.items():assert sha(r/'feedback'/name)==row['sha256'],name
parent=read(r/'logs/parent_immutable_verification.json')
# Original source manifest independently binds every parent file, not only native.
assert len([p for p in (r/'parent').rglob('*') if p.is_file()])==id['parent_files']==51
assert sha(r/'policy/config.json')==sha(r/'parent/policy/config.json')
assert sha(r/'COMPILER_FLAGS.json')==sha(r/'parent/COMPILER_FLAGS.json')
build=read(r/'policy/a06.BUILD.json')
for name,h in build['all_build_inputs'].items():assert sha(r/name)==h,name
entry=a.out.with_name(a.out.stem+'_entry.json')
cmd=[sys.executable,'-B',str(r/'tests/verify_entry.py'),'--root',str(r),'--out',str(entry)]
x=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
a.out.with_suffix('.stdout').write_text(x.stdout);a.out.with_suffix('.stderr').write_text(x.stderr)
assert x.returncode==0,(x.returncode,x.stderr)
# Entry calls must not mutate the frozen delivery.
for name,row in manifest['files'].items():assert sha(r/name)==row['sha256'],name
result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'package_files_hashed':checks,'original_source_manifest_checks':len(source),'feedback_manifest_checks':len(feedback),'parent_files':51,'build_inputs':len(build['all_build_inputs']),'native_sha256':id['native_sha256'],'parent_native_sha256':id['parent_native_sha256'],'all_passed':True,'entry':read(entry),'elapsed_seconds':time.monotonic()-t,'entry_command':cmd}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
