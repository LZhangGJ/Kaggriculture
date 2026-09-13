from pathlib import Path
import datetime,hashlib,json,os,re,shutil,zipfile
w=Path('/mnt/data/a06_r6_r2_work');r=w/'delivery';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc)
for name in ('narrow_compile','narrow_root_probes','narrow_choice_units','unpacked_build_units','unpacked_root_probes','trace_land_evidence','narrow_preservation'):
 assert (w/'logs'/f'{name}.exit').read_text().strip()=='0',name
n=0;casecomp=[]
for f in (w/'analysis/narrow_root_probes').glob('*.json'):
 if f.name=='SUMMARY.json':continue
 a=json.load(open(f));b=json.load(open(w/'analysis/unpacked_root_probes'/f.name));assert [x['action'] for x in a['rows']]==[x['action'] for x in b['rows']],f.name
 n+=len(a['rows']);casecomp.append({'case':f.stem,'identical_actions':len(a['rows']),'same_first_divergence':a['first_divergence']==b['first_divergence']})
assert n==5033
original=json.load(open(w/'initial/parent_files_sha256.json'))
for k,old in original.items():
 assert sha(w/'parent_immutable'/k)==old and sha(r/'provenance/parent'/k)==old
 assert sha(r/k)==sha(w/'candidate'/k)==sha(w/'preflight_unpacked'/k),k
native=sha(r/'policy/a06.so');assert native==sha(w/'preflight_rebuilt.so')
for f in (r/'tests').iterdir():
 if f.is_file():assert sha(f)==sha(w/'preflight_unpacked/tests'/f.name)
for name in ('analysis','logs','tools','initial'):shutil.copytree(w/name,r/'validation'/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(w/'preflight_unpacked/build/units',r/'validation/analysis/unpacked_default_units',dirs_exist_ok=True)
shutil.copy2(w/'preflight_rebuilt.BUILD.json',r/'validation/preflight_rebuilt.BUILD.json')
# Production receipt hashes the exact policy files deployed at the root.
build=json.load(open(r/'policy/a06.BUILD.json'))
assert build['sha256']==native
for k,h in build['sources'].items():assert sha(r/k)==h,k
out={'production_source_files_compared_to_parent':len(original),'immutable_parent_all_original_files_match':True,'only_changed_inherited_files':['policy/search.hpp','policy/a06.so'],'input_manifest_records':69,'input_manifest_mapping_all_pass':True,'unpacked_default_build_and_unit_exit':0,'unpacked_rebuild_matches_deployed_native':True,'selected_native_sha256':native,'unpacked_actual_root_calls':n,'unpacked_actions_match_prepack_candidate':n,'cases':casecomp,'new_games':0,'archive_preflight_note':'The exact deployment source/config/native and tests were extracted from production_preflight.zip, rebuilt and exercised. Final additions are documentation and the resulting raw receipts; deployment bytes are unchanged.'}
(r/'validation/DELIVERY_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
# Record resource usage and elapsed time from the user-supplied original dispatch.
now=utc();start=datetime.datetime.fromisoformat('2026-09-13T15:20:22.689+00:00');deadline=datetime.datetime.fromisoformat('2026-09-13T17:20:22.689+00:00')
assert now<deadline
resource=json.load(open(w/'logs/final_resource_snapshot.json'));resource['packaging_utc']=now.isoformat();resource['memory_peak_at_packaging_bytes']=int(Path('/sys/fs/cgroup/memory.peak').read_text());resource['memory_current_at_packaging_bytes']=int(Path('/sys/fs/cgroup/memory.current').read_text());resource['elapsed_from_original_dispatch_seconds']=(now-start).total_seconds();resource['deadline_utc']=deadline.isoformat();resource['elapsed_within_120_minutes']=True
assert resource['memory_peak_at_packaging_bytes']<int(resource['memory_max'])*.7
(r/'validation/RESOURCE_AND_TIMING.json').write_text(json.dumps(resource,indent=2)+'\n')
(r/'RESOURCE_AND_TIMING.md').write_text(f'''# Actual resource and timing record

Original dispatch: 2026-09-13T15:20:22.689Z. Hard deadline: 17:20:22.689Z.
First container resource probe: 15:21:06.695681182Z (44.007 seconds after dispatch).
Packaging checkpoint: {now.isoformat()}, {(now-start).total_seconds()/60:.2f} minutes after the original dispatch.
All work, failures, rebuilds and packaging belong to that single unchanged budget.

## Allocation, not host capacity

- Affinity and effective cpuset: CPUs 0-4 (5 visible).
- Cgroup CPU quota: 400000 / 100000 = 4 effective CPU equivalents.
- Memory limit: 4294967296 bytes (4 GiB).
- Initial memory.current: 311156736 bytes.
- Initial host MemAvailable: 5130528 kB, explicitly NOT treated as allocation.
- Peak cgroup memory observed by packaging: {resource['memory_peak_at_packaging_bytes']} bytes ({resource['memory_peak_at_packaging_bytes']/2**30:.3f} GiB).
- Memory headroom exceeded the requested 30%; no OOM/max events were observed.
- Disk available at final snapshot: {resource['disk_bytes']['free']} bytes.
- Python 3.13.5; g++ (Debian 14.2.0-19) 14.2.0; Linux x86-64.

One compile/probe worker was used at a time. Initial timeout-bounded compilation
and 48-observation parent probe were measured before scaling to seven saved
histories and the focused tests. No full opponent games or new seed panel were
launched.

## Measured commands

| Work | Wall time | Peak process-tree RSS |
|---|---:|---:|
| Initial parent rebuild | 26.99 s | 570160 KiB |
| Selected final rebuild | 26.80 s | 571980 KiB |
| Selected root probes (5033 calls) | 26.42 s | 248544 KiB |
| Unpacked rebuild plus default --unit | 80.51 s | 589576 KiB |
| Unpacked root-entry recheck (5033 calls) | 27.02 s | 248448 KiB |

These are local container measurements, not Kaggle sandbox timing guarantees.
Raw GNU time output, stdout/stderr, command/exit receipts and cgroup samples are
in validation/logs. Initial outer-tool timeouts and their successful independent
reruns are kept, rather than silently relabeled as successful tool invocations.
''')
# Check documentation's identity references against the bytes in the archive.
identity=json.load(open(r/'IDENTITY.json'));assert identity['native_sha256']==native
identity.update(packaging_checkpoint_utc=now.isoformat(),elapsed_from_dispatch_seconds=(now-start).total_seconds())
(r/'IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
(r/'README.md').write_text((r/'README.md').read_text()+f'''\n## Frozen runtime identity\n\nNative SHA256: `{native}`\n\nRoot main SHA256: `{sha(r/'main.py')}`\n\nThe unpacked rebuild matched this native byte-for-byte and the unpacked root\nreproduced all 5,033 selected-candidate probe actions. Full delivery checks are\nin validation/DELIVERY_CHECKS.json. Resource/timing records are in\nRESOURCE_AND_TIMING.md and validation/RESOURCE_AND_TIMING.json.\n''')
# Source and test hashes exclude documentation, archives, diagnostics and build products.
prod={k:sha(r/k) for k in sorted(original)}
for f in sorted((r/'tests').iterdir()):
 if f.is_file():prod[str(f.relative_to(r))]=sha(f)
(r/'PRODUCTION_AND_TEST_SHA256.json').write_text(json.dumps(prod,indent=2,sort_keys=True)+'\n')
files=[f for f in sorted(r.rglob('*')) if f.is_file() and f.name!='SHA256SUMS.txt' and '__pycache__' not in f.parts]
manifest={str(f.relative_to(r)):sha(f) for f in files}
(r/'SHA256SUMS.txt').write_text(''.join(f'{h}  {k}\n' for k,h in manifest.items()))
zip_path=Path('/mnt/data/TRI_A06_r6_r2.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
 for f in [*files,r/'SHA256SUMS.txt']:z.write(f,f.relative_to(r))
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None
 for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
 assert hashlib.sha256(z.read('policy/a06.so')).hexdigest()==native
 assert 'main.py' in z.namelist() and len(z.namelist())==len(set(z.namelist()))
finished=utc();assert finished<deadline
receipt={'file':str(zip_path),'bytes':zip_path.stat().st_size,'zip_sha256':sha(zip_path),'native_sha256':native,'archive_members_verified':len(manifest)+1,'crc_pass':True,'all_sha256_pass':True,'finished_utc':finished.isoformat(),'elapsed_seconds_from_original_dispatch':(finished-start).total_seconds(),'new_games':0,'closed_loop_win_rate_unmeasured':True}
Path('/mnt/data/TRI_A06_r6_r2_delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
