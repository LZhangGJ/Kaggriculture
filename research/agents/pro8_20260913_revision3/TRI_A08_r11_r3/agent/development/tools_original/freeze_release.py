from pathlib import Path
import json,shutil,hashlib,datetime,os,platform,subprocess,re
B=Path('/mnt/data/r3_work');R=Path('/mnt/data/TRI_A08_r11_r3_release');L=B/'logs'
for name,n in [('atomic_final_queue.json',5),('portable_queue.json',3),('final_precheck_queue.json',3)]:
 q=json.loads((L/name).read_text());assert len(q)==n and all(x['returncode']==0 for x in q),name
# The portable copied branch code must reproduce every semantic summary value.
checks=[]
for dirname,basename in [('portable_branches','own_branch_summary_clear1_supply0.json'),('portable_branches_negative','own_branch_summary_clear0_supply0.json'),('portable_branches_stress','own_branch_summary_clear1_supply3.json')]:
 a=json.loads((L/basename).read_text());b=json.loads((L/dirname/basename).read_text());assert a==b
 checks.append({'portable_directory':dirname,'summary':basename,'identical':True})
(L/'portable_reproduction.json').write_text(json.dumps({'status':'PASS','checks':checks,'unique_conditions':3,'new_games':0},indent=2)+'\n')
# Final observed resource snapshot, distinct from host-visible memory.
now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat('2026-09-13T19:00:45+00:00')
res={'observed_utc':now.isoformat(),'task_start_utc':start.isoformat(),'hard_deadline_utc':'2026-09-13T21:00:45Z','elapsed_seconds':(now-start).total_seconds(),'affinity':sorted(os.sched_getaffinity(0)),'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'effective_cpu':4,'memory_limit_bytes':int(Path('/sys/fs/cgroup/memory.max').read_text()),'memory_current_bytes':int(Path('/sys/fs/cgroup/memory.current').read_text()),'memory_cgroup_lifetime_peak_bytes':int(Path('/sys/fs/cgroup/memory.peak').read_text()),'memory_cgroup_peak_at_first_probe_bytes':1087463424,'memory_events':Path('/sys/fs/cgroup/memory.events').read_text(),'cpu_stat':Path('/sys/fs/cgroup/cpu.stat').read_text(),'host_MemAvailable_not_allocation':next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),'disk_usage':dict(zip(['total','used','free'],shutil.disk_usage('/mnt/data'))),'resource_policy':'Compile queues serial, at most two bounded jobs briefly overlapping; no full match jobs. Peak cgroup memory remains below 70% of the 4-GiB limit; host memory never treated as allowance.'}
assert res['memory_cgroup_lifetime_peak_bytes']<.7*res['memory_limit_bytes'];assert now<datetime.datetime.fromisoformat('2026-09-13T21:00:45+00:00')
(R/'validation/RESOURCE_END.json').write_text(json.dumps(res,indent=2)+'\n')
idx=[]
for p in sorted(L.glob('*.time')):
 s=p.read_text();m=re.search(r'Maximum resident set size \(kbytes\): (\d+)',s);q=p.with_suffix('.result.json');row={'time_file':p.name,'peak_rss_kib':int(m[1]) if m else None,'complete_outer_result':q.is_file()}
 if q.is_file():row['run']=json.loads(q.read_text())
 idx.append(row)
(R/'validation/RESOURCE_LOG_INDEX.json').write_text(json.dumps(idx,indent=2)+'\n')
# Raw current logs and all successes/failures/checkpoints are preserved.
ignore=shutil.ignore_patterns('__pycache__','*.pyc')
shutil.copytree(L,R/'validation/raw',ignore=ignore)
shutil.rmtree(R/'development/tools_original');shutil.copytree(B/'tests',R/'development/tools_original',ignore=ignore)
# Original source checkpoint for the driver was copied before its last raw writes.
shutil.rmtree(R/'development/checkpoints');shutil.copytree(B/'checkpoints',R/'development/checkpoints',ignore=ignore)
# No bytecode is part of any frozen source tree.
for p in list(R.rglob('__pycache__')):shutil.rmtree(p)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rec=json.loads((R/'BUILD.json').read_text());sources={n:sha(R/n) for n in rec['sources']};assert sources==rec['sources']
inputs={**sources,**{n:sha(R/n) for n in rec['validation_sources']}}
assert all(inputs[n]==h for n,h in rec['validation_sources'].items())
assert sha(R/'policy/tri_a08_r11_r3.so')==rec['binary_sha256']
(R/'SOURCE_SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n,h in sorted(inputs.items())))
for line in (R/'PARENT_SHA256SUMS.txt').read_text().splitlines():
 h,n=line.split('  ',1);assert sha(R/n)==h,n
assert len([p for p in (R/'parent').rglob('*') if p.is_file()])==46
# Literal isolation check on production source; no runtime data-file paths or case IDs.
case_ids=[c['id'] for c in json.loads((B/'feedback/SELECTED_CASES.json').read_text())]
text='\n'.join((R/n).read_text() for n in sources)
assert not any(c in text for c in case_ids)
assert not any(re.search(r'\b'+c.split('_')[-2]+r'\b',text) for c in case_ids)
identity={'revision':'TRI_A08_r11_r3','status':'FROZEN_CANDIDATE_NOT_FORMALLY_EVALUATED','parent':'TRI_A08_r11_r2_fix1','parent_native_sha256':'91828b8a9f6882030ba5247a8e40f15d788a9219fe64667502ebd6a5d211beed','source_input_zip_sha256':'759b6a9992dfcb1d0dd9d7718561eb586fb04e57cafbc532e6ba7755584e58b2','feedback_zip_sha256':'eaafa450c1055d2adae5f7c02b32467e5ddd21a3e0da0dadb3110ffe3e4e40b6','main_sha256':sha(R/'main.py'),'native_path':'policy/tri_a08_r11_r3.so','native_sha256':rec['binary_sha256'],'config_sha256':sha(R/'policy/config.json'),'source_manifest_sha256':sha(R/'SOURCE_SHA256SUMS.txt'),'parent_manifest_sha256':sha(R/'PARENT_SHA256SUMS.txt'),'production_source_config_build_files':46,'build_input_count_including_required_native_gate':47,'parent_files':46,'source_compiler':rec['compiler'],'new_competitive_games':0,'formal_win_rate':None,'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'acceptance_target':{'strict_wins_min':1306,'total_games':1536}}
(R/'IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
(R/'validation/SOURCE_SCOPE_AUDIT.json').write_text(json.dumps({'status':'PASS_LITERAL_CHECK_ONLY_NOT_FORMAL_INFORMATION_FLOW_PROOF','case_identifiers_in_production':[],'case_seed_literals_in_production':[],'fixed_config_matches_parent':sha(R/'policy/config.json')==sha(R/'parent/policy/config.json'),'required_gate_matches_parent':sha(R/'tests/native_choice_gate.py')==sha(R/'parent/tests/native_choice_gate.py')},indent=2)+'\n')
(R/'validation/RAW_LOG_INDEX.md').write_text('''# Raw evidence index

Final native: `49c98f2c1930cf741bdd9e2eb7da9af398117bb64f903f9375c7da9350e8fa64`.
Current actual-native/official checks: `raw/atomic_checks_gcc/`,
`raw/atomic_checks_clang/`. Current bounded branches: root
`raw/branch_*.json.gz`, `raw/own_branch_summary_*.json`, `raw/atomic_branches*`.
Portable repetitions are in `raw/portable_branches*/`; summaries compare equal
in `raw/portable_reproduction.json`. These do not add new independent cases.

Current full own-observation probes: `raw/seedguard_final_*`,
`raw/seedguard_clang_*`, `raw/seedguard_ablation_*`; actual default-root
outputs: `raw/root_default_*` and `raw/root_entry_check.json`.

The earlier `candidate_v1`, `candidate_final`, non-seedguard `clang`/`ablation`
and numbered candidate-build files are development checkpoints, not current
identity. `seedguard_checks_*` official execution outputs and early branches
were superseded by the atomic-PLANT harness correction; do not count them as
independent validating observations. Their actual-native animal gate and C++
property results are unaffected by that Python harness defect.

Absolute paths in command logs are the actual original work paths. They are
not required by the root offline builder/tests. `development/tools_original`
retains original investigation scripts, including work-path assumptions.
Use root `tests/run_checks.py`, `tests/probe_observations.py` and
`tests/own_branch_checks.py` for portable reproduction. The final ZIP
post-unpack check is reported outside the immutable ZIP.
''')
# Do this only after every included report has been written.
allfiles=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
(R/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in allfiles))
print(json.dumps({'files_hashed':len(allfiles),'identity':identity,'resources':res},indent=2))
