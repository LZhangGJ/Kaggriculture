from pathlib import Path
import datetime as dt, gzip, hashlib, json, os, shutil, subprocess
w=Path('/mnt/data/TRI_A08_work');r=Path('/mnt/data/TRI_A08_r11_r1_release')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=dt.datetime.now(dt.timezone.utc)
dispatch=dt.datetime.fromisoformat('2026-09-13T02:15:47.575+00:00')
b=json.loads((r/'BUILD.json').read_text())
source_diff=json.loads((w/'logs/final_source_comparison.json').read_text())
for name,h in b['sources'].items():assert sha(r/name)==h
assert sha(r/'policy/tri_a08_r11_r1.so')==b['binary_sha256']
# Rescan final legal prefixes without assigning wins to saved observations.
prefix=[]
for p in sorted((w/'logs/margin_prefixes').glob('*.json.gz')):
 prefix.append(json.loads(gzip.decompress(p.read_bytes()))['summary'])
assert len(prefix)==7
ps={'scope':'final candidate saved legal common prefixes, stop at first action divergence; no new games',
    'native_sha256':b['binary_sha256'],'calls':sum(s['calls'] for s in prefix),
    'matches':sum(s['match_historical_actions'] for s in prefix),'full_identical_719_paths':sum(s['calls']==s['match_historical_actions']==719 for s in prefix),
    'first_divergences':sum(s['first_historical_difference'] is not None for s in prefix),'rows':prefix}
(w/'logs/FINAL_PREFIX_SUMMARY.json').write_text(json.dumps(ps,indent=2)+'\n')
assert ps['calls']==4341 and ps['matches']==4338 and ps['full_identical_719_paths']==4
cg=lambda name:Path('/sys/fs/cgroup',name).read_text().strip()
resource={'utc':now.isoformat(),'dispatch_utc':dispatch.isoformat(),'hard_deadline_utc':'2026-09-13T04:15:47.575Z',
 'elapsed_minutes_from_original_dispatch_at_documentation':(now-dispatch).total_seconds()/60,
 'effective_cpus':4,'affinity':sorted(os.sched_getaffinity(0)),'cpu_max':cg('cpu.max'),
 'memory_limit_bytes':int(cg('memory.max')),'memory_current_bytes':int(cg('memory.current')),
 'cgroup_lifetime_peak_bytes':int(cg('memory.peak')),
 'cgroup_peak_caveat':'Lifetime peak includes infrastructure and earlier process activity, not solely the candidate.',
 'reserved_fraction':.30,'total_working_budget_bytes':3006477107,
 'host_memavailable_not_used_as_budget':[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0],
 'disk_free_bytes':shutil.disk_usage(w).free,
 'first_resource_capture_utc':'2026-09-13T02:19:40.648323160Z',
 'first_resource_capture_seconds_after_original_dispatch':233.07332316,
 'initial_workers':1,'subsequent_short_probe_workers_max':2,'max_concurrent_full_games':1,
 'full_game_policy_workers':2,'final_build_wall_seconds_gnu_time':27.52,'final_build_peak_rss_kib':594820,
 'clean_rebuild_wall_seconds_gnu_time':28.15,'clean_rebuild_peak_rss_kib':594728,
 'short_compile_wall_seconds':3.15,'short_compile_peak_rss_kib':327464,
 'initial_48_observation_probe_wall_seconds':1.02,'initial_48_observation_probe_peak_rss_kib':194648}
(w/'logs/RESOURCE_SUMMARY.json').write_text(json.dumps(resource,indent=2)+'\n')
freeze={'task':'TRI_A08_r11_r1','status':'FROZEN_CANDIDATE_FOR_CENTRAL_TESTING_NOT_ACCEPTED',
 'frozen_utc':now.isoformat(),'input_zip_sha256':'d2b4dad66c24be494421cfbcb48942a10162e12d8d81ea660eae5a7aeaab40c3',
 'parent':'A08_r11','parent_native_sha256':b['parent_native_sha256'],
 'native':'policy/tri_a08_r11_r1.so','native_sha256':b['binary_sha256'],
 'native_bytes':(r/'policy/tri_a08_r11_r1.so').stat().st_size,
 'production_sources':b['sources'],'changed_from_parent':source_diff['changed_files'],
 'unchanged_production_file_count':39,'config_sha256':sha(r/'policy/config.json'),
 'central_1536_games_run_in_this_round':False,'central_strict_wins_required':1306}
(r/'SOURCE_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
provenance={
 'task':'TRI_A08_r11_r1','author_lineage':'A08',
 'sole_parent':{'name':'A08_r11','input_archive':'A08_r11_source_and_losses.zip','input_zip_sha256':freeze['input_zip_sha256'],
 'original_parent_archive_sha256_recorded_in_input':'1e3ef9d4b960bcbc9fe47430a253877e68518bad18c9e4a31c6a7a5013dc8af5',
 'native_sha256':b['parent_native_sha256'],'source_files_verified':44,'all_manifest_source_hashes_match':True},
 'repository_provenance_from_user':{'repository':'github.com/LZhangGJ/Kaggriculture','branch':'research/pro8-a06-a08-20260913','commit':'d65325334067cc6c2b616f346bc43231b1c11190'},
 'verification_scope':'Attached identities and all 44 parent source hashes verified locally. Repository commit/tree read successfully. No claim of a separate remote hash audit of every source.',
 'no_external_policy_imported':True,'no_A06_or_A08_r12_code_used':True,
 'other_agent_names_in_original_SOURCE_IDENTITIES':'Metadata supplied by user, not additional production policies.',
 'candidate_native_sha256':b['binary_sha256'],
 'validation_restrictions':'Case IDs, seeds, saved replays, private referee states, and explicit hypothetical supply are validation-only. Production sees legal own observation and public state only.'}
(r/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
(r/'README.md').write_text('''# TRI_A08_r11_r1 — observed-stock price-floor sale-window DP

**Status: compiled, source/native matched, and frozen for central testing. This is not an accepted >85% agent.**

The only parent is the supplied **A08_r11**, not A08_r12 and not an A06 variant. Root `main.py` loads `policy/tri_a08_r11_r1.so`. This directory contains the complete production implementation, not a patch or a replacement policy. Production requires only `main.py`, `policy/`, and the matching shared library; the offline tests and research evidence are not runtime inputs.

## Change and reason

The parent's sale-window DP correctly avoided its cumulative-sales stock model when projected supply could reach the one-unit price floor. However, that rejection returned the decision to a local holding rule without rival-supply timing. Congestion therefore disabled precisely the competition-aware scheduling that could matter most.

The official engine still pays 1 for a unit sold at the price floor, but **does not add that unit to market inventory**. A final lockstep pair quoted above the floor can advance inventory two units and overshoot the serial saturation threshold by one. Consequently, `(tick, remaining quantity)` alone is not a sufficient DP state across the floor.

This revision adds an exact bounded `(tick, remaining quantity, actual market stock)` continuation **only for such floor-crossing windows**. It uses exact serial and lockstep stock transitions, retained integral quote tables, and the existing common-schedule scenario comparison. The original non-floor DP is unchanged. A current-observation anchor prevents the new branch from authorizing sales of later synthetic MPC inventory. Within this new floor branch only, the objective is actual own cash minus opponent cash; macro proposal weights greater than one are not treated as a cash exchange rate.

This is one sale-window repair, not a new supply predictor or a portfolio redesign. The same physical shed surplus after current unit execution, feed/material reservations, previously chosen same-day deadline, cash reserve, capacity guard, and action budget remain in force. No future harvest is credited as current cash. The new branch advances sales; it does not extend a holding deadline or independently alter worker routes. Later real feedback can still change subsequent plans.

The window remains at most five ticks (current tick plus four); batch size is at most 100 and reachable floor-stock span at most 1,024. Quote-domain or other eligibility failures use the parent behavior. Rival traffic is a hypothesis from completed observed sale-days, never enemy private inventory or saved future actions.

Exactly **5 of the 44 production text files** differ from the parent: `policy/sale_schedule_dp.hpp` (the mechanism), `policy/search.hpp` (current observation anchor), `policy/triad.hpp` (diagnostics), `main.py` (entry identity/default library), and `build.py` (build identity/feature switch). Configuration, Python observation codec, investment logic, rolling planner, executor, and the other 39 files retain parent hashes. See `SOURCE_DIFF.patch` and `SOURCE_FREEZE.json`.

## Entry and offline build

```python
from main import agent, reset
# action = agent(legal_observation, configuration)
# reset() between independent matches when reusing a Python process
```

The supplied library is Linux x86-64 ELF. Build without downloads using a C++20 compiler and Python's standard library:

```sh
python -B build.py
# Diagnostic ablation: disable only the added floor branch.
python -B build.py --sale-floor-dp 0 --out /tmp/tri_floor_off.so
```

`BUILD.json` and `policy/tri_a08_r11_r1.BUILD.json` are actual receipts for the delivered library, including every flag and all 44 source hashes. `BUILD.md` records the command, resource measurements, ABI inspection, and independent clean rebuild. The clean production-only rebuild produced the **same native SHA256**. Root default `main.agent` also passed two 48-observation common prefixes from a working directory outside the source tree, without specifying a binary override.

The library dynamically requires the inspected system C/C++ runtime; the highest recorded GLIBCXX and GLIBC symbol versions are GLIBCXX_3.4.31 and GLIBC_2.32. Rebuild on an incompatible host rather than assuming universal binary compatibility. This was not a Kaggle sandbox ABI or official agent-time-limit certification.

## Validation summary — scopes must not be mixed

| Check | Recorded outcome | What it establishes |
|---|---|---|
| Input and parent identity | Input archive SHA matched; all 44 parent source hashes matched | Correct lineage |
| All historical rows recounted | 393/480; public 368/440; original AFS R2 25/40 | Historical development results only |
| Saved-action official re-execution | 5,033 matching transitions; 420 matching player-days; cash residuals zero | Seven factual trajectories audited, not new wins |
| Original non-floor C++ tests | 420 problems; 1,680 exhaustive scenario optima; 17,108 transitions | Old DP regression coverage |
| New floor C++ tests | 702 problems; 2,592 independent exhaustive scenario optima; 6,174 primitives | Bounded floor-state correctness |
| ASan/UBSan | Same floor unit suite passed, leak detection enabled | Sanitizer coverage of unit solver, not whole-agent certification |
| Official market oracle | 4,974 cases, both seats and three orderings, all matched | Exact unit receipts and price-floor stock transitions |
| Additional guards | 31 observation/funding guard assertions; 9 resume boundary problems / 129 assertions; actual-margin negative regression passed | Authorization, capacity/reservations, and scoped objective protection |
| Floor-off ablation | All 5,033/5,033 supplied actions matched the parent | Removing the new feature recovers tested parent behavior |
| Final candidate legal prefixes | Four entire 719-step paths unchanged; three market-only first divergences | Localizes the first change without following saved future after divergence |
| Conditional official short branches | 32 branch runs / 16 paired comparisons | Conditional, explicitly hypothetical supply stress tests, not wins |
| Final full closed-loop games | **5 wins, 3 losses, 0 draws / 8**, against the supplied full A08_r11 | New parent-opponent development results, **not R2/public-pool acceptance** |

All eight final full games ran 719 steps with no runtime exceptions, zero cash-identity residuals, and no sell quantity beyond actual post-unit inventory. There were 10 unsuccessful `BUY_SEED` unit attempts per side across the panel (20 combined), equally present in parent and candidate; they are not silently counted as successful orders or recoverable revenue. Maximum observed policy-call time in this small panel was about 0.377 s. Neither this runtime sample nor these eight games establishes the central acceptance result.

Repeated oracle/unit runs are replications of the same cases, not additional independent sample sizes. The round contains 17 full parent-opponent games across three implementation checkpoints: one unanchored pilot, eight anchored/configured-weight games, and eight final games. Only the last eight evaluate the delivered native.

## Positive and negative evidence

In the original R2 worst-loss trajectory the parent held 24 wool units while the observed price went from 158 at step 530 to 1 at step 532. The original full-game deficit was 18,759. This is a mechanism clue, **not** a claim that its lost cash or victory can be recovered.

At the final candidate's first common-prefix divergence in that fixture (step 482), four-tick official branches with declared public-history-derived rival wool supply improved actual cash margin by +342, +2,222, or +2,222 depending on queue position; the no-additional-supply branch was unchanged. These are bounded conditional receipts, not a replayed R2 win.

The Thomas historical **+70** narrow-win path is unchanged for all 719 actions in the final candidate. A harmful intermediate price-floor choice that sacrificed 26 own cash to deny 13 rival cash was removed by the scoped true-cash-margin objective. Its negative branch now has zero difference in all four tested supply/order conditions.

**The R2 +2 narrow-win fixture is not certified safe.** Its first divergence is at step 434. A conditional early-rival branch improves actual margin by 291, but no additional rival supply or late queue positions cost 19 relative to the parent. The Aurax close-loss window also has an adverse -8 branch. These negatives are retained in `validation/logs/FINAL_BRANCH_COMPARISON.json`, not discarded or reclassified as wins.

## Limitations and acceptance

The original AFS R2 executable and the complete public-opponent set were not present in the attachment. No new genuine R2 or 12-opponent full matches were run here; the parent was used only as an explicitly named, non-PASS functional opponent. Saved replay futures after a different action were never used as a counterfactual trajectory. Short branches clear actual enemy private inventory, inject declared hypothetical supply, and obtain subsequent own observations from the official engine.

The trailing rival profile remains lagged and uncertain, and the scenario set does not model every multi-product queue interaction or long-term investment response. Exact DP values are conditional model values, not proof of realizable game-level benefit. Supply-estimation redesign and full opponent-pool validation remain unfinished. This candidate can regress despite the repaired floor-state transition.

The user's target is strict overall win rate >85%. The central mixed development panel is 64 seeds × 12 opponents × 2 seats = 1,536 games and requires **at least 1,306 wins**. It is not a sealed holdout. That acceptance panel was **not run in this chat**, and no result from it is claimed. The frozen candidate must be evaluated on the central updated-seed panel before promotion.

## Reproduce the small checks

Use fresh output directories **outside** this release to preserve frozen evidence. The runner is serial and never launches the central panel. It needs no third-party Python package.

```sh
python -B tests/verify_release.py --out /tmp/tri-release-check.json
python -B tests/run_validation.py unit --with-sanitizers --out /tmp/tri-unit
python -B tests/run_validation.py official-floor --out /tmp/tri-floor-oracle
python -B tests/run_validation.py prefixes --out /tmp/tri-prefixes
python -B tests/run_validation.py ablation-prefixes --out /tmp/tri-ablation
python -B tests/run_validation.py branches --out /tmp/tri-branches
python -B tests/run_validation.py audit --out /tmp/tri-saved-audit
# Optional: exactly eight original development-seed games against parent, not R2.
python -B tests/run_validation.py parent-matches --allow-full-games --out /tmp/tri-parent8
```

The portable unit and official-floor runner modes were actually exercised from the staged release. Raw phase drivers preserve their original absolute execution paths under `research/original_drivers`; they are audit records rather than the recommended portable entry point.

## Evidence layout

`validation/source_input/` is the unpacked, unmodified supplied bundle, including the complete parent, all 480 rows, selected-case replays, provided audits, and frozen official CPU referee. `validation/logs/` contains actual compilation, raw observations/actions, official unit cash logs, rejected-order accounting, timing, resource, ablation, prefix, and intermediate failed-attempt logs. `validation/cases/` declares conditional branch inputs. `research/checkpoints/` preserves the superseded implementations and matching libraries; **none is the root runtime**. `research/ablations/` holds the compiled floor-off variant. No original ZIP is nested and no old binary substitutes for the final one.

The source-input identity file also names A06 variants as supplied metadata; no A06 production policy was imported. An inherited test field named `historical_r8_action` is a stale label in the provided driver: its values are the attached **A08_r11** action trace. Some parent-history `.json` files contain gzip bytes; inspect the magic header or use `gzip.decompress`.

Resources were recorded 3 min 53 s after original dispatch: effective CPU quota 4 cores, cgroup limit 4 GiB, and 30% memory reserved. The host memory reading was recorded only as context, not used as an allocation limit. Build/probe measurements determined the small validation scale. Full matches were limited to one at a time. See `validation/logs/RESOURCE_SUMMARY.json` and `DELIVERY_TIMING.json` for actual timing and the unchanged hard deadline.
''',encoding='utf-8')
# Build documentation: exact generated command, no lossy abbreviation.
command=' '.join(b['command'])
(r/'BUILD.md').write_text(f'''# Actual build record — TRI_A08_r11_r1

The root `BUILD.json` is copied from the receipt emitted by the actual successful final build. It is not the parent build manifest. The identical receipt remains next to the library. All 44 production source hashes match.

## Actual command

Driver:
```sh
python -B /mnt/data/TRI_A08_work/candidate/build.py
```

Compiler: `{b['compiler']}`.

```sh
{command}
```

The absolute paths document where this build happened. The equivalent relocatable command is `python -B build.py` from the extracted release. C++20, `-O3`, `-DNDEBUG`, baseline x86-64, `-ffp-contract=off`, and every feature define above were used. No package download, runtime compilation, other policy binary, or network fetch is required for the production build.

## Actual outputs and measurements

- Library: `policy/tri_a08_r11_r1.so`, **{freeze['native_bytes']:,} bytes**.
- SHA256: `{b['binary_sha256']}`.
- Compiler-driver receipt duration: {b['seconds']:.6f} seconds.
- GNU time wall clock: 27.52 seconds; maximum RSS: 594,820 KiB; exit 0.
- Clean production-only rebuild: 28.15 seconds; maximum RSS 594,728 KiB; exit 0; native bytes identical.
- Root default entry smoke in that clean tree: two seats ×48 legal inputs, 96/96 saved-prefix actions match; no binary override.

Raw records: `validation/logs/margin_build.*`, `clean_rebuild.*`, `CLEAN_REBUILD_CHECK.json`, `CLEAN_DEFAULT_ENTRY.json`, `native_abi.txt`, and `parent_full_source_identity.json`. `CLEAN_REBUILD_CHECK.json` includes the independent clean build receipt. Whole-tree checksums are in `MANIFEST.sha256`; this file is not part of the 44 production-source compilation manifest.

## Runtime dependencies and bounds of verification

ELF64 Linux x86-64 shared object, dynamically linked to libstdc++, libm, libgcc_s, and libc. Inspected symbol requirements include GLIBCXX_3.4.31, CXXABI_1.3.9, and GLIBC_2.32. The recorded environment used Python 3.13.5 and Debian GCC 14.2.0. This confirms successful local loading, not compatibility with every Linux image or Kaggle resource-limit harness. Recompile offline with the target toolchain as needed; preserve the new source/native receipt rather than claiming a different-toolchain binary has this SHA.

## Ablation

`python -B build.py --sale-floor-dp 0 --out /tmp/tri_floor_off.so` disables only this added branch. The retained actual ablation binary is `research/ablations/floor_off.so` (SHA256 `dd3b27502f12ef5c770c3ed6171c5e87b425c1e435cf9decfad7fd3fe9ad249c`) with its matching receipt. On all seven supplied legal observation paths it reproduces 5,033/5,033 original A08_r11 actions. It is not the root production binary.
''',encoding='utf-8')
(r/'research/README.md').write_text('''# Research checkpoints, not production entry points

The only delivered production choice is the root `main.py` plus `policy/tri_a08_r11_r1.so` identified by `SOURCE_FREEZE.json`.

`checkpoints/unanchored_floor/` retains the first price-floor extension (native SHA256 `450223885af7296e9b60683a9b65d96d9829fcf02819baa205499cadd33c90fd`). It allowed floor decisions in later synthetic rolling-planner states. One complete parent-opponent pilot lost by 412. A Thomas short-branch first-action guard failed when route actions changed; that failure is preserved in the raw logs. This version is superseded.

`checkpoints/anchored_config_weight/` retains the observation-anchored version (native SHA256 `6e6c95e93ed02680957472d1f36f001c7f85a79a38dbc014a2051c9f9a9cb3eb`). Its eight parent-opponent development games were 5 wins / 3 losses. The configured macro opponent weight could worsen actual cash margin in a low-traffic Thomas window, motivating the final scoped weight of one. It is superseded even though those eight terminal outcomes happen to equal the final panel.

These historical tree snapshots can contain copied parent README/BUILD documents at their root. For each snapshot, its native-adjacent `.BUILD.json` and the corresponding raw build log are the authoritative source/native identity. Do not mistake its inherited root prose for current acceptance evidence.

`ablations/floor_off.so` was built from the final 44 production sources with `A08_SALE_FLOOR_DP=0`. Its source manifest and exact flags are in the adjacent receipt.

`original_drivers/` preserves the paths and commands used during development. Use root `tests/run_validation.py` for relocatable re-execution. The early `floor_original_units` failure exposed an out-of-range floor-threshold computation for a logarithmic item curve; the delivered version uses a bounded search and explicit no-floor-in-domain sentinel. The final unit/oracle suites and floor-unit sanitizers passed after that repair.

Earlier failed runs, aborted conditional branches, replays, and smoke checks are not extra full games. There were 17 complete games across the three candidate implementations, all against the supplied A08 parent, and only 8 belong to the frozen final candidate. No original R2 executable was replaced with the parent in any claim.
''',encoding='utf-8')
print('docs written', now.isoformat(), 'elapsed minutes', (now-dispatch).total_seconds()/60)
