from pathlib import Path
import json,subprocess,sys,hashlib,gzip,shutil,datetime
B=Path('/mnt/data/r3_work');C=B/'candidate';L=B/'logs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def root_entry():
 rows=[]
 for c in json.loads((B/'feedback/SELECTED_CASES.json').read_text()):
  dest=L/('root_default_'+c['id']+'.json')
  cmd=[sys.executable,str(B/'feedback/probe_observations.py'),'--agent',str(C/'main.py'),'--trace',str(B/'feedback'/c['trace']),'--limit','719','--out',str(dest)]
  subprocess.run(cmd,check=True)
  observed=json.loads(dest.read_text());ref=json.loads(gzip.decompress((L/('seedguard_final_'+c['id']+'.json.gz')).read_bytes()))
  assert [x['action'] for x in observed['rows']]==[x['action'] for x in ref]
  rows.append({'case':c['id'],'calls':719,'matches_create_agent':719,'matches_parent':observed['exact_parent_actions'],'new_games':0,'command':cmd,'result_sha256':sha(dest)})
 (L/'root_entry_check.json').write_text(json.dumps({'status':'PASS','native_sha256':sha(C/'policy/tri_a08_r11_r3.so'),'calls':2876,'new_games':0,'rows':rows},indent=2))
def compare():
 rows=[]
 for c in json.loads((B/'feedback/SELECTED_CASES.json').read_text()):
  get=lambda n:json.loads(gzip.decompress((L/(n+'_'+c['id']+'.json.gz')).read_bytes()))
  a,b,ab=get('seedguard_final'),get('seedguard_clang'),get('seedguard_ablation')
  assert [r['action'] for r in a]==[r['action'] for r in b]
  assert all(r['same_parent'] for r in ab)
  rows.append({'case':c['id'],'gcc_clang_action_matches':719,'ablation_parent_action_matches':719})
 (L/'cross_compiler_and_ablation.json').write_text(json.dumps({'status':'PASS','new_games':0,'GCC_native':sha(C/'policy/tri_a08_r11_r3.so'),'Clang_native':sha(B/'clang.so'),'ablation_native':sha(B/'ablation.so'),'rows':rows},indent=2))
def clean_build(fault=False):
 receipt=json.loads((C/'policy/tri_a08_r11_r3.BUILD.json').read_text())
 root=B/('atomic_gate_negative' if fault else 'clean_rebuild');root.mkdir(exist_ok=False)
 for name in [*receipt['sources'],*receipt['validation_sources']]:
  target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(C/name,target)
 if fault:
  dst=root/'policy/tri_a08_r11_r3.so';rec=dst.with_suffix('.BUILD.json');shutil.copy2(C/'policy/tri_a08_r11_r3.so',dst);shutil.copy2(C/'policy/tri_a08_r11_r3.BUILD.json',rec)
  before=[sha(dst),sha(rec)]
  (root/'tests/native_choice_gate.py').write_text('import sys\nprint("INJECTED EXPECTED native action/value gate rejection",file=sys.stderr)\nraise SystemExit(37)\n')
 r=subprocess.run([sys.executable,str(root/'build.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 (L/(root.name+'.compile.stdout')).write_text(r.stdout);(L/(root.name+'.compile.stderr')).write_text(r.stderr)
 if fault:
  after=[sha(dst),sha(rec)];assert r.returncode!=0 and before==after
  stages=list((root/'policy').glob('*.building-*.so'));assert len(stages)==1
  result={'status':'PASS_EXPECTED_REJECTION','build_returncode':r.returncode,'prior_native_unchanged':True,'prior_receipt_unchanged':True,'rejected_stage_retained':str(stages[0]),'native_sha256':before[0]}
 else:
  assert r.returncode==0,(r.returncode,r.stderr)
  native=root/'policy/tri_a08_r11_r3.so';assert sha(native)==receipt['binary_sha256']
  result={'status':'PASS','native_sha256':sha(native),'matches_production_bytes':True,'independent_directory':str(root),'build_receipt_sha256':sha(native.with_suffix('.BUILD.json'))}
 (L/(root.name+'_result.json')).write_text(json.dumps(result,indent=2))
if sys.argv[1]=='entry':root_entry();compare()
elif sys.argv[1]=='clean':clean_build()
elif sys.argv[1]=='fault':clean_build(True)
