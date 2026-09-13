from pathlib import Path
import datetime,hashlib,json,shutil,os,zipfile
w=Path('/mnt/data/a06_r6_r2_work');r=w/'delivery';r.mkdir(exist_ok=True)
shutil.copytree(w/'candidate',r,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','build'))
p=r/'provenance/parent';shutil.copytree(w/'parent_immutable',p,dirs_exist_ok=True)
shutil.copy2(w/'initial/parent_files_sha256.json',r/'provenance/PARENT_FILES_SHA256.json');shutil.copy2(w/'input/PARENT_IDENTITY.json',r/'provenance/PARENT_IDENTITY.json')
e=r/'evidence/own_visible';e.mkdir(parents=True,exist_ok=True)
for f in (w/'input').iterdir():
 if f.name=='agent':continue
 if f.is_dir():shutil.copytree(f,e/f.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 else:shutil.copy2(f,e/f.name)
for name in ('analysis','logs','tools','initial'):shutil.copytree(w/name,r/'validation'/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(w/'development',r/'development',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['parent_rebuild.so','parent_rebuild.BUILD.json','audit_parent.so']:shutil.copy2(w/name,r/'validation'/name)
(e/'SCOPE_NOTE.md').write_text('''# Evidence mapping\n\nThese files are exact copies from `TRI_A06_r6_r1_own_visible.zip`.\nThe original `MANIFEST.json` paths under `agent/` map to\n`../../provenance/parent/` relative to this directory; every other manifest path\nis relative to this directory. No opponent code, private state, complete\nreplay, previous larger ZIP, network retrieval, or new game is included.\nThe public official referee and its local CPU host are retained unchanged.\nThe local host is not the Kaggle timeout or schema validator.\n\nAll `FULL64_*` results refer to parent r1. They are not r2 validation scores.\n''')
(r/'development/README.md').write_text('''# Development checkpoint, not the selected agent\n\nThe root of this archive is the only r2 candidate.\n`broad_matched_deferral/` is the first implementation of the same land-wait idea,\nretained with its complete source and matching native. Its hash is recorded in\nREFINEMENT.json. It added alternatives even when some original candidate already\navoided land. Saved-observation probes diverged in four cases, including both\nnarrow-win fixtures. Those are not measured new losses.\n\nThe final intervention is narrower: alternatives are added only when ALL original\nprepared queues require BUY_LAND. This avoids widening an already two-sided\ndecision and reproduces both narrow-win own-action traces exactly. No new seed\nsweep, score threshold, opponent-name gate, or seed table was introduced.\n\nRaw broad-stage results are under validation/analysis/candidate_root* and\nchoice_units*. The final-stage results are explicitly named narrow_*.\nEvery supplied development seed and all 1536 outcomes remain in evidence.\n''')
shutil.copy2(w/'analysis/trace_land_evidence/FULL64_RECOUNT.json',r/'development/SUPPLIED_DEVELOPMENT_SEEDS_AND_COUNTS.json')
(r/'validation/README.md').write_text('''# Validation record index\n\nFinal code/native: root main.py + policy/a06.so.\n- analysis/narrow_root_probes: actual root entry, 7 x 719 saved observations.\n- analysis/narrow_preservation.json: original candidates before/at first divergence.\n- analysis/narrow_choice_units: actual production classes + official prefix checks.\n- analysis/trace_land_evidence: own unit-phase facts, 210 day cash reconciliations.\n- analysis/parent_audits and parent_immutable_root_probes: exact parent diagnostics.\n- analysis/source_changes.json and parent_integrity.json: immutable-parent checks.\n- logs/*.receipt.json: bounded command exit, UTC time and cgroup sampling.\n- initial/: first resource probe, input hash and original parent hash manifest.\n\nThe earlier broad checkpoint is retained for audit, NOT mixed into final metrics.\nOuter tool timeouts with complete child summaries but missing parent exit receipts\nare documented in logs/tooling_incidents.json; they were independently rerun.\nNo full game, opponent strategy or closed-loop win rate was measured this round.\nThe 24-tick exercises explicitly use the parent's declared public-flow scenario.\n''')
files={str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in r.rglob('*') if f.is_file() and (str(f.relative_to(r)) in json.load(open(w/'initial/parent_files_sha256.json')) or f.parent==r/'tests')}
(r/'PRODUCTION_AND_TEST_SHA256.json').write_text(json.dumps(files,indent=2,sort_keys=True)+'\n')
identity={'candidate':'TRI_A06_r6_r2','status':'source-correctness-and-observation-tested candidate; new closed-loop win rate unmeasured','selected_intervention':'matched no-new-land-today alternatives ONLY when all original candidates require BUY_LAND','parent':'TRI_A06_r6_r1','input_zip_sha256':'12caf9cf20aece50cadb4331cedd76be2e3dbc5b534f27665d82322b9232a9c4','parent_native_sha256':'174ba2ad749cb3213a72a03302ef50b08a37a579848ba922bde132a19b8b6c23','native_sha256':hashlib.sha256((r/'policy/a06.so').read_bytes()).hexdigest(),'main_sha256':files['main.py'],'changed_production_sources':['policy/search.hpp'],'config_sha256':files['policy/config.json'],'compiler_flags_sha256':files['COMPILER_FLAGS.json'],'new_games':0,'acceptance_target':{'games':1536,'minimum_strict_wins':1306,'measured':False},'freeze_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
# Verify all 69 original-manifest records through the explicit mapping.
manifest=json.load(open(e/'MANIFEST.json'));results=[]
for key,item in manifest.items():
 f=p/key[len('agent/'):] if key.startswith('agent/') else e/key
 assert f.stat().st_size==item['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],key
 results.append(key)
(r/'validation/INPUT_MAPPING_VERIFICATION.json').write_text(json.dumps({'records':len(results),'all_pass':True,'keys':results},indent=2)+'\n')
for f in p.rglob('*'):f.chmod(0o555 if f.is_dir() else 0o444)
p.chmod(0o555)
pre=w/'production_preflight.zip'
with zipfile.ZipFile(pre,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(r.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(r))
with zipfile.ZipFile(pre) as z:
 assert z.testzip() is None;z.extractall(w/'preflight_unpacked')
print('Preflight snapshot',pre.stat().st_size,'native',identity['native_sha256'],'input records',len(results))
