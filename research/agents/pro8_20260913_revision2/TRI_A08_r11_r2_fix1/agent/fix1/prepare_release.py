from pathlib import Path
import json, hashlib, shutil, difflib, datetime, subprocess, zipfile, re
R=Path('/mnt/data/fix1_work'); C=R/'candidate'; F=C/'fix1'; F.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
b=json.loads((C/'policy/tri_a08_r11_r2_fix1.BUILD.json').read_text()); old=json.loads((C/'history/r2/IDENTITY.json').read_text())
# Preserve exact parent ZIP contents in the existing immutable workspace, and
# all production files in the downloadable reference tree. No nested old ZIP.
zip_path=Path('/mnt/data/TRI_A08_r11_r2.zip');errors=[]
with zipfile.ZipFile(zip_path) as z:
    for info in z.infolist():
        if not info.is_dir():
            p=R/'original_r2'/info.filename
            if not p.is_file() or p.read_bytes()!=z.read(info):errors.append(info.filename)
assert not errors,errors
parent_files={}
for k,v in old['production_files'].items():
    p=C/'reference/r2'/k;assert sha(p)==v,(k,sha(p),v);parent_files[k]=v
(C/'reference/r2/PARENT_IDENTITY.json').write_bytes((C/'history/r2/IDENTITY.json').read_bytes())
(C/'reference/r2/PARENT_SOURCE_SHA256SUMS.txt').write_bytes((C/'history/r2/SOURCE_SHA256SUMS.txt').read_bytes())
(C/'reference/r2/PARENT_MANIFEST_SHA256SUMS.txt').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(parent_files.items())))
for name in ('central_failure.json','resource_initial.json','source_review.txt'):
    shutil.copy2(R/name,F/name)
for name in ('logs','repro'):
    shutil.copytree(R/name,F/name,dirs_exist_ok=True)
shutil.copytree(R/'clean_rebuild',F/'clean_rebuild',dirs_exist_ok=True)
shutil.copytree(R/'guard_fixture',F/'initial_expected_failure_fixture',dirs_exist_ok=True)
for name in ('run_measured.py','apply_fix1.py','strengthen_cpp_checks.py','add_build_gate.py','prepare_repro.py','test_build_transaction.py','core_verify.sh','secondary_verify.sh','prepare_release.py'):
    shutil.copy2(R/name,F/name)
# Patch is relative to the exact r2 source; test changes are separate.
changed=[];patch=[]
for name,h in sorted(b['sources'].items()):
    assert sha(C/name)==h,name
    if old['production_files'].get(name)!=h:
        changed.append(name)
        patch.extend(difflib.unified_diff((R/'original_r2'/name).read_text().splitlines(True),(C/name).read_text().splitlines(True),fromfile='TRI_A08_r11_r2/'+name,tofile='TRI_A08_r11_r2_fix1/'+name))
(F/'PRODUCTION_DIFF.patch').write_text(''.join(patch))
test_patch=[]
for p in sorted((C/'tests').glob('*')):
    if not p.is_file() or p.suffix!='.py' and p.suffix!='.cpp':continue
    oldp=R/'original_r2/tests'/p.name
    oldtext=oldp.read_text().splitlines(True) if oldp.exists() else []
    test_patch.extend(difflib.unified_diff(oldtext,p.read_text().splitlines(True),fromfile='r2/tests/'+p.name,tofile='fix1/tests/'+p.name))
(F/'TEST_DIFF.patch').write_text(''.join(test_patch))
assert len(b['sources'])==44 and len(changed)==4,changed
assert (C/'policy/triad.hpp').read_bytes()==(C/'reference/r2/policy/triad.hpp').read_bytes()
# Exact source-set hash is defined by this sorted text manifest, not a vague tree hash.
(C/'SOURCE_SHA256SUMS.txt').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(b['sources'].items())))
manifest_sha=sha(C/'SOURCE_SHA256SUMS.txt'); native=b['binary_sha256'];assert native==sha(C/'policy/tri_a08_r11_r2_fix1.so')
prod=dict(b['sources']);prod['policy/tri_a08_r11_r2_fix1.so']=native
identity={'task':'TRI_A08_r11_r2_fix1','status':'SOURCE_AND_BUILD_FIX_CANDIDATE_GCC13_CENTRAL_RECHECK_REQUIRED','parent':'TRI_A08_r11_r2','parent_zip_sha256':sha(zip_path),'parent_native_sha256':old['native_sha256'],'native_sha256':native,'source_manifest_sha256':manifest_sha,'main_sha256':sha(C/'main.py'),'config_sha256':sha(C/'policy/config.json'),'production_files':prod,'production_source_count':44,'unchanged_source_count_vs_r2':40,'changed_source_files':changed,'compiler':b['compiler'],'flags':b['flags'],'new_closed_loop_games':0,'gcc13_local_validation':'UNAVAILABLE; no local reproduction or claim of established compiler defect','central_report_preserved':'fix1/central_failure.json','created_utc':now()}
(C/'IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
(F/'PARENT_VERIFICATION.json').write_text(json.dumps({'status':'PASS','original_zip_sha256':sha(zip_path),'original_workspace_matches_every_zip_member':True,'reference_r2_production_files_verified':len(parent_files),'parent_native_sha256':old['native_sha256'],'new_games':0},indent=2)+'\n')
# Plain-English docs are current; all inherited reports are explicitly historical.
readme=f'''# TRI_A08_r11_r2_fix1

**Status: source/build repair candidate, not a centrally accepted agent.**
The exact parent is TRI_A08_r11_r2 (ZIP `{sha(zip_path)}`). The complete
production parent, including its tested native, is retained in `reference/r2/`;
the earlier r1 ancestor is unchanged in `reference/parent/`.

## What is and is not established

Central Ubuntu 24.04 / GCC 13.3 reported an optimized r2 test with
`feed=0` but `value=65`, although the unique optimal action is `feed=1`.
This record is preserved verbatim in substance in `fix1/central_failure.json`.
GCC 13 is not installed in this runtime, and attempts to retrieve it failed.
**Neither the original GCC 13 failure nor its disappearance after this change
has been reproduced locally. Undefined behavior versus a compiler defect remains
unresolved.** The repair isolates code generation, makes the chosen action/value
structurally inseparable at selection, and adds a fail-closed test of the actual
linked native. It is not a claim that an identified GCC bug has been fixed.

Actual local compilers are GCC 14.2.0 and Clang 17.0.0. Both original and fixed
standalone reductions pass on these compilers; the unmodified original full
unit is retained as the central reproducer, not replaced with a passing reduction.
`fix1/repro/matrix_gcc13_unavailable/RESULT.json` records the missing compiler.

## Focused source repair

`policy/animal_service_dp.hpp` now chooses one immutable candidate index from the
three legal service actions, preserving the original enumeration order and
`1e-9` tie rule, and commits both `Choice` and `value` from that same winner.
The service reward formula and r2 animal-collection-labor improvement are unchanged.
GCC's `noipa` attribute places `solve` behind a genuine optimized call boundary;
Clang uses `noinline`. The body is still compiled at **`-O3`**. There is no global
`-O0`, no `optimize("O0")`, no removed assertion, and no disabled production feature.
This is a targeted code-generation workaround pending the GCC 13 recheck.

`policy/bridge.cpp` adds `td_service_dp_snapshot`, a read-only synthetic-input
validation seam calling that exact linked solver. It owns a fresh DP object and
cannot inspect or mutate a running agent. It is never called by strategy decisions.

`build.py` compiles to a unique staging file, checks the resulting native's action,
value, independent Bellman optimum, tie choice and full-policy realization, then
promotes it only after success. A failing check preserves the prior native and
receipt and retains the rejected staging file and all logs. `main.py` loads only
`policy/tri_a08_r11_r2_fix1.so`; there is no silent fallback to an ancestor.

Only four of 44 production source/configuration files differ from r2: the service
DP, bridge, entrypoint and build script. The other 40 are byte-identical, including
`triad.hpp`, fixed configuration, positive-return investment, rolling scheduling,
low-level execution and all existing sales-DP boundaries.
The exact patch is `fix1/PRODUCTION_DIFF.patch`.

## Offline build and checks

Linux x86-64, a C++20 compiler, Python standard library, GNU `time` and `timeout`
are required. There is no package installation or network dependency.

```sh
python build.py --cxx g++
python tests/run_checks.py --sanitize
python tests/verify_package.py  # Before rebuilding: verifies the delivered inventory.
```

A rebuild deliberately changes its receipt and appends logs, so run inventory
verification before rebuilding, or on a fresh unpack. A different compiler is
not expected to emit the same native bytes. It must pass the exact-native gate
and behavioral checks. To test the central compiler explicitly:

```sh
python build.py --cxx g++-13
python tests/run_checks.py --cxx g++-13 --sanitize
python tests/repro_matrix.py --cxx g++-13 --out /tmp/a08_fix1_gcc13_new_directory
```

The matrix includes original standalone, boundary-only, argmax-only, combined
repair and the original full unit, each with `-O3`, `-O0` and `-O1` ASan/UBSan.
Original failures are diagnostic; fixed-case failures return a failure status.
No pretense is made that a locally passing reduction reproduces a foreign compiler.

```sh
python tests/test_build_refusal.py --out /tmp/a08_fix1_expected_rejection_new_directory
python tests/compare_saved_entries.py \\
  --left "$PWD/reference/r2/main.py" --right "$PWD/main.py" \\
  --traces "$PWD/evidence/own_traces" \\
  --out /tmp/a08_fix1_paired_new_directory --require-exact
```

The refusal test deliberately mutates a Python checker copy only, outside the
production tree. It is an expected-failure safety test, not a GCC bug reproduction.
All test output directory arguments must be fresh directories.

## Actual validation

Both GCC 14.2 and Clang 17 optimized full builds passed the expanded C++ unit:
148,821 assertions, 5,120 brute-force cases, 4,608 path cases and 20,496 action-state
checks. ASan/UBSan versions passed as separate evidence, not substitutes for `-O3`.
The exact-native gate made 530 DP constructions and validated 29,456 states with
171,628 successful checks before the deliberate mutation. The rich control is
`value=Choice.value=65, feed=1, care=0`; the poor control retires the animal.
The injected `feed=0, value=65` mismatch is rejected.

Official local execution checks passed 4,608 single-farm lifecycles, 16,128 day
states and 90,914 checks. These do not prove global scheduler feasibility.
The default entry/receipt/config/ABI/isolation checks passed 183 assertions.

Actual paired calls on all seven saved own-observation traces: **5,033/5,033 actions
match original r2**; **5,033/5,033 match between GCC 14.2 and Clang 17 fix1 builds**.
The two narrow-win saved traces each match r2 for 719/719 calls. These are fixed
observation probes, not new matches, not counterfactual futures and not evidence
that either narrow win is retained in closed loop.

A clean-directory GCC 14.2 rebuild produced identical native bytes. A separate
expected-failure build test preserved the previous native and its receipt.
See `VALIDATION_REPORT.md` and raw results under `fix1/` and `rerun/`.

**New closed-loop games: 0. New official panel: 0. No new win rate is claimed.**
The centrally supplied r1 1,054/1,536 result remains historical. The acceptance
threshold remains strictly greater than 85% (at least 1,306/1,536 wins), to be
measured by the central fresh-seed panel after identity freezing.

## Identity and preserved evidence

- Production native SHA256: `{native}`
- Source manifest SHA256: `{manifest_sha}`
- Configuration SHA256: `{sha(C/'policy/config.json')}`
- Exact compiler commands, flags, native gate and all source hashes:
  `policy/tri_a08_r11_r2_fix1.BUILD.json` and `BUILD.md`.

`history/r2/` contains the original r2 top-level claims and manifests unchanged;
they are historical, not fix1 results. Existing `analysis/`, `logs/`, `probes/`,
`development/` and `checkpoints/` originated in the r2 ZIP and are not relabelled as
new work. New repair records are in `fix1/`, new time-stamped tests in `rerun/`,
and transactional build logs in `build_runs/`. Earlier failed/partial tool
wrappers, network retrieval failures, intermediate build receipts, and the central
failure are retained. No full match panel or new development game seeds were run.

The original round start/deadline remain 2026-09-13 15:22:07.194 / 17:22:07.194 UTC.
Resource/timing records distinguish these from the repair's 16:26:13 UTC start.
'''
(C/'README.md').write_text(readme)
# Build document uses the actual staging command, rather than a reconstruction.
import shlex
builddoc=f'''# BUILD — TRI_A08_r11_r2_fix1

## Exact deployed build

Compiler: `{b['compiler']}`. Python: 3.13.5. Linux x86-64.
Feature flags are unchanged from the r2 default build, including
`A08_ANIMAL_COLLECTION_LABOR=1`; optimization is `-O3`, with `-ffp-contract=off`.
No native-architecture-specific `-march=native` and no global deoptimization.

Actual command (absolute paths and staging filename preserved):

```sh
{shlex.join(b['command'])}
```

The staging library was checked by this actual command before atomic promotion:

```sh
{shlex.join(b['native_choice_gate_command'])}
```

The stage was renamed to `policy/tri_a08_r11_r2_fix1.so` only after the gate passed.
Source and native byte identities are recorded in `IDENTITY.json`,
`SOURCE_SHA256SUMS.txt` and the adjacent `.BUILD.json` receipt.

Production native SHA256: `{native}`.
Source manifest SHA256: `{manifest_sha}`.

## Independently rebuilt artifact

`fix1/clean_rebuild/` was populated with the 44 production source/configuration
files and the standalone native checker, without a native library. Running
`python clean_rebuild/build.py` compiled and gated an identical GCC 14.2 binary.
See `fix1/logs/clean_offline_rebuild.*`, `fix1/logs/clean_rebuild_hashes.txt`,
`fix1/clean_rebuild/policy/tri_a08_r11_r2_fix1.BUILD.json`.

Clang 17 full optimized native and its receipt are retained as
`fix1/repro/fix1_clang17.so` and `.BUILD.json`. It is not the default production
library. Its table digest and all 5,033 tested observation actions match GCC 14.2;
its binary hash intentionally differs. These compilers use the installed Linux
headers/standard library. This is not a claim of testing on Ubuntu 24.04 itself.
`fix1/logs/native_platform.txt` records ELF, dynamic dependencies and symbol versions.

## Exact-native fail-closed gate

The gate is `tests/native_choice_gate.py` (SHA256
`{sha(C/'tests/native_choice_gate.py')}`). It checks both stored score fields,
independent optimal action/tie order, one-step Bellman realization and full policy
rollout. Build failure, mismatch or gate failure prevents promotion and preserves
existing production bytes. Full logs and the staging artifact are retained.
It is mandatory; there is no skip flag.

The self-contained new build-refusal test confirms rejection with the exact
positive-return assertion, while preserving BOTH the original native and receipt.
The fault injection changes only a disposable Python checker copy. It must not be
mistaken for a GCC 13 reproduction or a failed production candidate.

## Compiler-specific boundary and attribution limit

The only new optimization boundary is `AnimalServiceDP::solve`: GCC `noipa`,
Clang `noinline`. GCC documents `noipa` as preventing interprocedural optimization
between a function and its callers; it is not an instruction to compile the body
at `-O0`:
https://gcc.gnu.org/onlinedocs/gcc-13.3.0/gcc/Common-Function-Attributes.html#index-noipa-function-attribute

Original GCC 14 unit symbols contain a `solve [clone .constprop.0]`; the final
GCC 14 symbols retain the un-cloned solver boundary. See the two symbol logs in
`fix1/repro/`. That demonstrates the intended boundary in the tested compiler;
it does NOT establish that GCC 13 constant propagation caused the central failure.
No GCC bug number, diagnosed UB, or GCC 13 pass is asserted.
'''
(C/'BUILD.md').write_text(builddoc)
(C/'BUILD.json').write_bytes((C/'policy/tri_a08_r11_r2_fix1.BUILD.json').read_bytes())
# Detailed evidence table with parseable paths and exact local denominators.
validation='''# Validation report — TRI_A08_r11_r2_fix1

## Verdict

The source/build repair is implemented, compiled at production `-O3`, and tested
with GCC 14.2 and Clang 17. The originally reported GCC 13.3 environment could not
be reproduced locally. Final root-cause classification (source UB, GCC optimizer,
or another compiler-specific issue) and the GCC 13.3 fixed-build result are open.
This archive is an honest candidate/checkpoint, not an assertion of central acceptance.

## Central failure, not rewritten as a local result

`fix1/central_failure.json` preserves the user's exact original ZIP/native hashes,
`unit_1` failure after 25,859 checks and the O3/O0/sanitized observations. The
original assertion remains in `history/r2/animal_collection_test.cpp`; the complete
parent production source and native are in `reference/r2/`. The current test still
asserts positive-return maintenance; neither the expectation nor its tolerance
has been relaxed. The original ZIP remains byte-identical in the working archive.

## Reduction and bounded source review

`fix1/repro/original_standalone.cpp` removes the game engine and imports only the
DP, numeric constants and rich/poor test. It is a reduction candidate, NOT an
independently reproduced GCC 13 failure. The matrix also runs the untouched full
original unit so that a reduction which loses the trigger cannot replace it.
Boundary-only, argmax-only and combined variants distinguish the two changes.

On each locally available compiler all 15 variants/modes compile and run: original
standalone, boundary-only, argmax-only, combined and original full unit, each in
O3, O0 and O1 ASan/UBSan. This consistency is useful regression evidence but cannot
identify a fault that these compilers never exhibit. The GCC 13 matrix reports
COMPILER_UNAVAILABLE rather than PASS. The attempted apt/HTTPS retrieval failures
are preserved under `fix1/logs/`.

Bounds review: kind 9..11 maps to species 0..2; hunger 0..1; pending bonus is
bounded by held-1 <= 5; only d <= 28 reads d+1; starvation is checked before
indexing; next bonus is capped; terminal tables are aggregate-zero-initialized.
No UB has been established by that bounded review. Sanitizer success does not
prove the absence of every possible UB. The inherited animal_path noipa boundary
is supporting context only, not proof of the cause of this new central symptom.

## Source repair invariant

One immutable candidate index supplies `(feed, care, value)` and the second value
table. All three legal service choices retain the same rewards, collection work,
care-cap restriction, action order and 1e-9 tie rule. A targeted noipa/noinline
boundary stops caller-specialized DP clones while leaving the body at O3.
No prices, strategy thresholds, investment logic, sales-DP branches or seeds were
changed. Production diff: `fix1/PRODUCTION_DIFF.patch`. Test diff: `fix1/TEST_DIFF.patch`.

## Executed checks

| Evidence | Actual result | Scope/limit |
|---|---|---|
| Expanded GCC14 O3 C++ unit | 148,821 checks; 5,120 brute-force cases; 4,608 path cases; 20,496 choice states; 0 repaired-model mismatches | Conditional service/path model, not a match |
| Expanded Clang17 O3 C++ unit | Same counts, PASS | Independent compiler, same platform libraries |
| GCC14 and Clang17 ASan/UBSan units | PASS, same 148,821 checks | Separate O1 evidence, not a substitute for O3 |
| Actual linked native gate, both compilers | 530 constructions; 29,456 validated states; 171,628 successful checks before mutation | Table value, Choice.value, independent argmax, Bellman edge, policy rollout |
| Rich control | value=65, Choice.value=65, feed=1, care=0 | Original positive expectation retained |
| Poor control | value=0, feed=0 | Non-profitable maintenance still stops |
| Deliberately mutated feed bit | Rejected by independent argmax check | Negative test, NOT a compiler reproduction |
| Official execution, both compilers | 4,608 local lifecycles; 16,128 day states; 26,389 effective actions; 90,914 checks | Single own farm; does not prove global scheduling feasibility |
| Root entry contract | 183 checks, PASS | Native/source receipts, config, ABI, reset, two-context isolation and rejected input |
| Original r2 native vs fix1 native | 5,033/5,033 action equality across 7 fixed own-observation traces | No new futures or matches |
| GCC14 fix1 vs Clang17 fix1 native | 5,033/5,033 action equality | Root factory using explicitly recorded alternate Clang native |
| Clean GCC14 build | Identical deployed native SHA256 | Same compiler/platform reproducibility only |
| Expected-failure build promotion | Rejected with positive assertion; previous native and receipt unchanged | Isolated checker mutation; raw failure retained |

The legacy `A08_ANIMAL_COLLECTION_LABOR=0` unit deliberately exhibits the known
r2 pre-fix economic mismatches (3,937 value, 5,236 labor-day, 3,521 stream-value
mismatches). Its PASS means the regression detector found the expected old defect
and action/value consistency held. It is NOT a production configuration and does
not mean those legacy economics are correct. The production feature remains 1.

New C++ checks validate the stored chosen action at every generated state, not
merely the optimal value. The native test is independent Python and reads the
exact deployed C++ table; it additionally rolls out the selected policy to its
terminal state. An action with the right score but wrong feed flag is no longer
allowed to pass. The native library and gate hashes are recorded by the builder.

## Trace and counterfactual limitations

The paired tests really execute both root agents on every one of the seven saved
own-visible observation sequences (719 calls per sequence), with fresh state per
case; full action rows are retained as gzip JSON. They are not a closed-loop game.
After any difference from a source trajectory, its saved future cannot be treated
as the candidate's true future. No recovered cash, new win or loss is inferred.

`market_smart_v8_915042141_seat0` and
`submission_56149565_915042141_seat1` each match original r2 719/719. This preserves
r2 behavior on those observations, not proof of preserving the source r1 +78/+324
wins in a new simulation. r2 was already different from r1; that historical risk
has not disappeared. The original parent panel's 1,054/1,536 is not a fix1 result.

New closed-loop matches = 0. Full new panel = 0. No winner-based acceptance claim.
Only central fresh-64-seed/12-opponent/two-seat play can measure the requested
strict >85% threshold (1,306/1,536 wins). No opponent code/private state was used,
no seed lookup or opponent-name branch was added, and no new game seeds were drawn.

## Raw result index

- `fix1/repro/matrix_gcc14/RESULT.json` and `matrix_clang17/RESULT.json` — 15-mode matrices and per-command output/time logs.
- `fix1/repro/matrix_gcc13_unavailable/RESULT.json` — unavailable compiler, not success.
- `fix1/repro/paired_original_fix1/RESULT.json` — actual original/fix paired calls and per-case action hashes.
- `fix1/repro/paired_gcc14_clang17/RESULT.json` — actual cross-compiler paired calls.
- `fix1/repro/portable_build_refusal/RESULT.json` — native AND receipt preservation under expected failure.
- `fix1/repro/build_transaction.json` — initial independent failure-injection run.
- `policy/tri_a08_r11_r2_fix1.BUILD.json` — final default native and gate identity.
- `fix1/clean_rebuild/policy/tri_a08_r11_r2_fix1.BUILD.json` — independently rebuilt same bytes.
- `fix1/logs/*.command.json`, `.stdout`, `.stderr`, `.time`, `.json` — actual commands, status, wall/RSS.
- `rerun/20260913T164157736553Z/` — full Clang17 suite, native gate and default root contract.
- `rerun/20260913T163743144751Z/` — expanded GCC14 suite and all saved observations; inner suite finished, outer timed wrapper was interrupted (see timing limitations).
- `history/r2/` — original claims/manifests, retained as historical evidence, superseded by the explicit limitations above.

## Timing and record limitations

Original start/deadline: 2026-09-13 15:22:07.194 / 17:22:07.194 UTC. Repair resources
were measured at 16:26:28 UTC; repair began at approximately 16:26:13 UTC. The main
source repair was saved at 16:32:10 UTC, before the original 90-minute target.
Budget was not reset. Effective quota: 4 CPUs, cgroup limit 4 GiB, one worker,
30% memory headroom policy. MemAvailable is recorded separately and not treated
as the cgroup allowance. Actual measured compilation peak was below 600 MiB.

Two outer wrapper records are incomplete: attempted apt retrieval and the early
long GCC14 suite. Their raw stdout/stderr/inner results are preserved, rather than
fabricating completion times or peak RSS. Later bounded clean build, compiler
checks, paired calls and final unpacked-package tests provide separate complete
records. Intermediate receipts identify the bytes actually compiled then; they
must not be confused with the final default native. No time estimate is substituted
for a completed experiment.
'''
(C/'VALIDATION_REPORT.md').write_text(validation)
# Summarize complete outer measured records. Preserve interrupted ones explicitly.
measured=[];maxrss=0
for path in sorted((R/'logs').glob('*.json')):
    try:x=json.loads(path.read_text())
    except:continue
    if isinstance(x,dict) and 'start_utc' in x and 'wall_seconds' in x:
        t=path.with_suffix('.time');rss=None
        if t.exists():
            match=re.search(r'Maximum resident set size \(kbytes\): (\d+)',t.read_text())
            if match:rss=int(match[1]);maxrss=max(maxrss,rss)
        measured.append(dict(x,peak_rss_kib=rss))
resources={'original_start_utc':'2026-09-13T15:22:07.194Z','hard_deadline_utc':'2026-09-13T17:22:07.194Z','repair_start_utc':'2026-09-13T16:26:13Z','initial':json.loads((R/'resource_initial.json').read_text()),'snapshot_utc':now(),'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'memory_current':Path('/sys/fs/cgroup/memory.current').read_text().strip(),'max_observed_command_peak_rss_kib':maxrss,'measured_commands':measured,'outer_wrapper_incomplete':['apt_update','fix1_gcc14_all_checks'],'new_games':0}
(C/'RESOURCE_AND_TIMING.json').write_text(json.dumps(resources,indent=2)+'\n')
print(json.dumps({'prepared':now(),'native_sha256':native,'source_manifest_sha256':manifest_sha,'changed':changed,'maxrss_kib':maxrss,'files_current':len([p for p in C.rglob('*') if p.is_file()])},indent=2))
