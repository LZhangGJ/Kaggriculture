from pathlib import Path
import json,hashlib,shutil,zipfile,datetime,re
W=Path('/mnt/data/fix1_work');C=W/'candidate';R=Path('/mnt/data/TRI_A08_r11_r2_fix1_release');Z=Path('/mnt/data/TRI_A08_r11_r2_fix1.zip');V=W/'unpacked_final'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in (W/'logs').glob('final_candidate_exact_checks.*'):shutil.copy2(p,C/'fix1/logs'/p.name)
for f in ('archive_release.py',):shutil.copy2(W/f,C/'fix1'/f)
p=C/'VALIDATION_REPORT.md';s=p.read_text().replace('- `rerun/20260913T164157736553Z/`', '- `rerun/20260913T165058311690Z/` — final default GCC14 exact `python tests/run_checks.py --sanitize`, all commands PASS; complete outer wall/RSS record in `fix1/logs/final_candidate_exact_checks.*`.\n- `rerun/20260913T164157736553Z/`');p.write_text(s)
results={}
for name,path in [('final_default_suite',C/'rerun/20260913T165058311690Z/RESULT.json'),('gcc14_vs_parent',W/'repro/paired_original_fix1/RESULT.json'),('gcc14_vs_clang17',W/'repro/paired_gcc14_clang17/RESULT.json'),('portable_refusal',W/'repro/portable_build_refusal/RESULT.json'),('native_gate',C/'rerun/20260913T165058311690Z/native_choice_gate.json'),('official_execution',C/'rerun/20260913T165058311690Z/official_execution.json')]:
 results[name]=json.loads(path.read_text());assert results[name]['status']=='PASS',name
results['status']='LOCAL_GCC14_CLANG17_PASS_CENTRAL_GCC13_RECHECK_PENDING';results['new_games']=0
(C/'fix1/RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
timing=json.loads((C/'RESOURCE_AND_TIMING.json').read_text());entry=json.loads((W/'logs/final_candidate_exact_checks.json').read_text());entry['peak_rss_kib']=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',(W/'logs/final_candidate_exact_checks.time').read_text())[1]);timing['measured_commands'].append(entry);timing['snapshot_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();timing['memory_current']=Path('/sys/fs/cgroup/memory.current').read_text().strip();(C/'RESOURCE_AND_TIMING.json').write_text(json.dumps(timing,indent=2)+'\n')
check=json.loads((C/'DELIVERY_CHECKPOINT.json').read_text());check['source_frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();check['status']='LOCAL_TESTS_PASS_FROZEN_FOR_ARCHIVE_GCC13_PENDING';(C/'DELIVERY_CHECKPOINT.json').write_text(json.dumps(check,indent=2)+'\n')
assert not R.exists();shutil.copytree(C,R)
manifest=R/'PACKAGE_SHA256SUMS.txt';files=sorted(p for p in R.rglob('*') if p.is_file() and p!=manifest);manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(R).as_posix()}\n' for p in files))
with zipfile.ZipFile(Z,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(R.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(R).as_posix())
with zipfile.ZipFile(Z) as z:
 assert z.testzip() is None
 assert 'main.py' in z.namelist()
 for n in z.namelist():assert not n.startswith('/') and '..' not in Path(n).parts
 z.extractall(V)
result={'task':'TRI_A08_r11_r2_fix1','zip_path':str(Z),'zip_sha256':sha(Z),'zip_bytes':Z.stat().st_size,'zip_crc':'PASS','files_in_manifest':len(files),'files_in_zip':len(files)+1,'native_sha256':sha(R/'policy/tri_a08_r11_r2_fix1.so'),'source_manifest_sha256':sha(R/'SOURCE_SHA256SUMS.txt'),'archive_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_start_utc':'2026-09-13T15:22:07.194Z','hard_deadline_utc':'2026-09-13T17:22:07.194Z','new_games':0,'gcc13_status':'UNAVAILABLE_LOCALLY_NOT_CLAIMED_PASS'}
(W/'archive_identity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
