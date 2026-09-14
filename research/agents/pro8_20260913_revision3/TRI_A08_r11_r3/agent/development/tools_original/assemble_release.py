from pathlib import Path
import shutil,json,hashlib,datetime,difflib
B=Path('/mnt/data/r3_work');R=Path('/mnt/data/TRI_A08_r11_r3_release');C=B/'candidate';R.mkdir(exist_ok=False)
ignore=shutil.ignore_patterns('__pycache__','*.pyc')
def copytree(src,dst):shutil.copytree(src,dst,ignore=ignore)
receipt=json.loads((C/'policy/tri_a08_r11_r3.BUILD.json').read_text())
# A single production entry, native and complete matching build input set.
for name in receipt['sources']:
 target=R/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(C/name,target)
for name in ['policy/tri_a08_r11_r3.so','policy/tri_a08_r11_r3.BUILD.json']:
 shutil.copy2(C/name,R/name)
copytree(C/'tests',R/'tests');copytree(C/'build_runs',R/'build_runs')
copytree(B/'source_input/agent',R/'parent')
copytree(B/'feedback',R/'evidence/feedback')
(R/'evidence/source_metadata').mkdir()
for p in (B/'source_input').iterdir():
 if p.is_file():shutil.copy2(p,R/'evidence/source_metadata'/p.name)
for name in ['INPUT_VERIFY.json','MANIFEST_VERIFY.json','RESOURCE_START.json']:
 (R/'validation').mkdir(exist_ok=True);shutil.copy2(B/name,R/'validation'/name)
copytree(B/'checkpoints',R/'development/checkpoints')
copytree(B/'tests',R/'development/tools_original')
copytree(B/'diagnostic',R/'development/parent_diagnostic')
for name in ['diagnostic.so','diagnostic.BUILD.json','parent_rebuilt.BUILD.json']:
 shutil.copy2(B/name,R/'development'/name)
(R/'development/final_variants').mkdir()
for name in ['clang.so','clang.BUILD.json','ablation.so','ablation.BUILD.json']:
 shutil.copy2(B/name,R/'development/final_variants'/name)
copytree(B/'clean_rebuild',R/'development/clean_rebuild')
copytree(B/'atomic_gate_negative',R/'development/atomic_gate_negative')
# Portable bounded own-state execution script; model and transition semantics
# are unchanged from the corrected tested work script. No extra game runner.
s=(B/'tests/own_branch_checks.py').read_text()
s=s.replace("B=Path('/mnt/data/r3_work')","B=Path(__file__).resolve().parents[1]\nOUT=B/'test_runs/own_branches'")
s=s.replace("B/'feedback", "B/'evidence/feedback")
s=s.replace("B/'candidate/main.py'", "B/'main.py'")
s=s.replace("B/'candidate/policy/tri_a08_r11_r3.so'", "B/'policy/tri_a08_r11_r3.so'")
s=s.replace("path=B/'logs'/", "path=OUT/")
s=s.replace("(B/'logs'/f'own_branch_summary", "(OUT/f'own_branch_summary")
s=s.replace("def main():\n ap=", "def main():\n global OUT\n ap=")
s=s.replace("a=ap.parse_args();rows=[]", "ap.add_argument('--out-dir',type=Path);a=ap.parse_args();OUT=(a.out_dir or OUT).resolve();OUT.mkdir(parents=True,exist_ok=True);rows=[]")
(R/'tests/own_branch_checks.py').write_text(s)
shutil.copy2(B/'feedback/probe_observations.py',R/'tests/probe_observations.py')
# Exact parent byte hash and readable diff, not an executable alternate agent.
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent_files={str(p.relative_to(R/'parent')):sha(p) for p in (R/'parent').rglob('*') if p.is_file()}
assert len(parent_files)==46
(R/'PARENT_SHA256SUMS.txt').write_text(''.join(f'{h}  parent/{name}\n' for name,h in sorted(parent_files.items())))
modified=[];same=[];added=[];diff=[]
for name in receipt['sources']:
 if name in parent_files:
  if sha(R/name)==parent_files[name]:same.append(name)
  else:modified.append(name);diff+=list(difflib.unified_diff((R/'parent'/name).read_text().splitlines(True),(R/name).read_text().splitlines(True),fromfile='parent/'+name,tofile='r3/'+name))
 else:
  added.append(name);diff+=list(difflib.unified_diff([], (R/name).read_text().splitlines(True),fromfile='/dev/null',tofile='r3/'+name))
(R/'SOURCE_DIFF.patch').write_text(''.join(diff))
(R/'SOURCE_DIFF.json').write_text(json.dumps({'modified_parent_production_files':modified,'unchanged_parent_production_files':same,'added_production_files':added,'required_native_gate_unchanged':sha(R/'tests/native_choice_gate.py')==parent_files['tests/native_choice_gate.py'],'parent_input_files':46},indent=2)+'\n')
assert len(modified)==5 and len(same)==39 and len(added)==2
for p in (R/'parent').rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
(R/'parent').chmod(0o555)
(R/'PACKAGING_START.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task_started_utc':'2026-09-13T19:00:45Z','deadline_utc':'2026-09-13T21:00:45Z','production_native_sha256':sha(R/'policy/tri_a08_r11_r3.so')},indent=2)+'\n')
print(R, 'assembled, native',sha(R/'policy/tri_a08_r11_r3.so'))
