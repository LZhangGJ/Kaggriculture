from pathlib import Path
import json,shutil,hashlib,datetime,gzip,difflib,os
W=Path(__file__).resolve().parent; C=W/'candidate'; P=W/'parent_r3'; S=Path('/mnt/data/TRI_A06_r12_r3_fix1')
assert not S.exists(), 'do not overwrite staging'
S.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):
 p=S/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def cp(src,dst):
 dst=S/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for n in ('main.py','build.py','COMPILER_FLAGS.json'):cp(C/n,n)
for n in ('policy','tests','build'):shutil.copytree(C/n,S/n,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for n in ('feedback','logs','research'):shutil.copytree(W/n,S/n,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(W/'diagnostic_r3',S/'diagnostic_r3',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
# Immutable immediate-parent production snapshot plus byte-exact original ZIP.
for n in ('main.py','build.py','COMPILER_FLAGS.json','BUILD.json','BUILD.md','IDENTITY.json','SOURCE_FREEZE.json','README.md','DEVELOPMENT_SEEDS.json','VALIDATION.json','VALIDATION.md'):
 cp(P/n,'parent_r3/'+n)
shutil.copytree(P/'policy',S/'parent_r3/policy',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
cp(Path('/mnt/data/TRI_A06_r12_r3.zip'),'provenance/TRI_A06_r12_r3.zip')
cp(W/'rebuild_check/policy/a06.BUILD.json','logs/independent_rebuild_receipt.json')
for n in ('make_diagnostic.py','run_diagnostic.py','check_fix1_prefixes.py','implement_fix1.py','package_fix1.py'):cp(W/n,'research/checkpoint_scripts/'+n)
build=json.loads((C/'policy/a06.BUILD.json').read_text());rh=sha(W/'rebuild_check/policy/a06.so');assert rh==build['sha256']==sha(C/'policy/a06.so')
for name,h in build['all_build_inputs'].items():assert sha(S/name)==h,name
assert sha(S/'policy/config.json')==sha(P/'policy/config.json')
changed=[];diff=[]
for name,h in build['all_build_inputs'].items():
 if sha(P/name)!=h:
  changed.append({'path':name,'parent_sha256':sha(P/name),'candidate_sha256':h})
  diff.extend(difflib.unified_diff((P/name).read_text().splitlines(True),(S/name).read_text().splitlines(True),fromfile='r3/'+name,tofile='r3_fix1/'+name))
assert sorted(x['path'] for x in changed)==['main.py','policy/ongoing_supply_admission.inc','policy/triad.hpp']
(S/'SOURCE_CHANGE.diff').write_text(''.join(diff))
write('SOURCE_FREEZE.json',{'name':'TRI_A06_r12_r3_fix1','all_build_inputs':build['all_build_inputs'],'changed_inputs':changed,'count':len(build['all_build_inputs']),'config_unchanged':True})
write('BUILD.json',build)
write('IDENTITY.json',{'name':'TRI_A06_r12_r3_fix1','native_sha256':build['sha256'],'native_bytes':(S/'policy/a06.so').stat().st_size,'parent_name':'TRI_A06_r12_r3','parent_zip_sha256':sha(Path('/mnt/data/TRI_A06_r12_r3.zip')),'parent_native_sha256':sha(P/'policy/a06.so'),'r2_ancestor_native_sha256':'638626de9398f0aed45e98394742ee22df875dd805f2647dee6d5666eb1c7755','excluded_alternate_r2_native':'bd0c7dd9170b2f70781a7e2bd211b367fc625aa66154a99f5855641f08aefd62','feedback_zip_sha256':'6560bda6b5ddd20ef5629539e1887be71e4e81f1a763810e990d63474b51694e','production_configuration_unchanged':True,'status':'bounded-tested candidate; no new full opponent-pool result'})
unit=json.loads((C/'build/tests/receipt.json').read_text());assert len(unit)==4 and all(x['returncode']==0 for x in unit)
prefix=json.loads((W/'logs/fix1_prefix_summary.json').read_text())
pairs=[]
for f in sorted((W/'logs').glob('terminal_*.json')):
 a=json.loads(f.read_text())
 if not isinstance(a,list) or len(a)!=2:continue
 p,q=a
 assert p['variant']=='parent' and q['variant']=='candidate' and p['terminal_reached'] and q['terminal_reached'] and p['cash_ledger_reconciled'] and q['cash_ledger_reconciled']
 raw1=json.loads(gzip.decompress((W/'logs/official_terminal'/f"{p['case']}_parent_{p['scenario']}.json.gz").read_bytes()))
 raw2=json.loads(gzip.decompress((W/'logs/official_terminal'/f"{p['case']}_candidate_{p['scenario']}.json.gz").read_bytes()))
 identical=all(x['own_action']==y['own_action'] for x,y in zip(raw1['actions'],raw2['actions']))
 pairs.append({'case':p['case'],'scenario':p['scenario'],'steps_each':p['steps'],'own_cash_parent':p['own_cash_end'],'own_cash_fix1':q['own_cash_end'],'own_gain':q['own_cash_end']-p['own_cash_end'],'raw_cash_margin_gain':q['own_cash_end']-q['rival_cash_end']-p['own_cash_end']+p['rival_cash_end'],'own_actions_identical':identical,'fertilizer_bought_parent':p['units'].get('BUY_PRODUCT:FERTILIZER',0),'fertilizer_bought_fix1':q['units'].get('BUY_PRODUCT:FERTILIZER',0),'fertilizer_paid_parent':-p['flow'].get('BUY_PRODUCT:FERTILIZER',0),'fertilizer_paid_fix1':-q['flow'].get('BUY_PRODUCT:FERTILIZER',0),'service_parent':p['fertilize_success'],'service_fix1':q['fertilize_success'],'production_bonus_parent':p['production_bonus_ticks'],'production_bonus_fix1':q['production_bonus_ticks'],'max_fix1_call_seconds':q['max_call_seconds'],'ledger_ok':True,'raw_receipt':'logs/'+f.name})
assert len(pairs)==5
write('VALIDATION.json',{'new_complete_matches':0,'unit_suites':4,'unit_assertions':23892,'unit_breakdown':[18403,3029,46,2414],'r3_research_identity_probe':{'matched':2876,'total':2876,'scope':'research-only diagnostic library against actual supplied r3 observations, not games'},'fix1_history_probe':[{'case':v['case'],'matched':v['matched'],'calls':v['calls'],'first_divergence':v['first_divergence'],'scope':'stop immediately at first divergence; not games'} for v in prefix],'official_conditional_trajectories':10,'official_conditional_steps':470,'conditional_pairs':pairs,'production_rebuild_byte_equal':True,'native_sha256':build['sha256'],'entry':json.loads((W/'logs/entry_result.json').read_text()),'unverified':['No actual R2 805057947 fix1 full rematch','No new 12-opponent pool or >85% claim','Earlier-than-terminal quantity/economics gap remains','Public scenario forecast and hidden responses can be wrong']})
rows=json.loads((W/'feedback/PARENT_FULL64_ROWS.json').read_text());assert len(rows)==1536
write('DEVELOPMENT_SEEDS.json',{'warning':'Public development feedback; not a sealed holdout. No production seed/opponent lookup. No new full match generated.','inherited_r3_development_seeds':json.loads((P/'DEVELOPMENT_SEEDS.json').read_text()),'feedback_r2_panel_seeds':sorted(set(r['seed'] for r in rows)),'current_case_seeds':[568422814,1827852203,31485687,805057947],'synthetic_conditional_seed':[2609134201],'unit_simulator_seed':[0],'conditional_scope':'Synthetic fork after current observation672; no saved future is consumed','full_parent_rows':'feedback/PARENT_FULL64_ROWS.json'})
res=json.loads((W/'logs/resume_resources.json').read_text());res['staged_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();res['memory_current_at_stage']=Path('/sys/fs/cgroup/memory.current').read_text().strip();res['main_fix_completed_before_utc']='2026-09-13T20:42:00Z';res['note']='Continuation received with less than 30 minutes left; original 90/100-minute milestones had already passed. New work was limited to this fix; original deadline was never reset.'
write('RESOURCE_AND_TIME.json',res)
(S/'BUILD.md').write_text('# Actual offline build\n\nCompiler: '+build['compiler']+'\n\nProduction command (original absolute working paths):\n\n```sh\n'+' '.join(build['command'])+'\n```\n\nRebuild from this archive root:\n\n```sh\npython3 build.py\npython3 build.py --unit\npython3 tests/verify_entry.py --root . --evidence feedback --out entry_check.json\n```\n\nNo network, pip, downloaded code, or opponent source is required. A C++20 compiler and Python standard library are required. The unchanged flags use portable Linux x86-64, not -march=native. The author's compiler is GCC14.2.0; GCC13.3 testing reported by central applied to parent r3, not to this fix. See BUILD.json, policy/a06.BUILD.json, logs/fix1_build.*, logs/rebuild.*, and logs/independent_rebuild_receipt.json for actual receipts. Unit-test assertions are enabled (-O1, no -DNDEBUG).\n')
# Ensure immediate-parent snapshot is immutable in the delivered archive.
for f in (S/'parent_r3').rglob('*'):
 if f.is_file():f.chmod(0o444)
(S/'provenance/TRI_A06_r12_r3.zip').chmod(0o444)
print(json.dumps({'stage':str(S),'native':build['sha256'],'changed':changed,'pairs':pairs},indent=2))
