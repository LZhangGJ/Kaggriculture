from pathlib import Path
import json,hashlib
w=Path('/mnt/data/TRI_A08_work');r=Path('/mnt/data/TRI_A08_r11_r1_release')
full=json.loads((w/'logs/margin_closed_loop/SUMMARY.json').read_text())
pref=json.loads((w/'logs/FINAL_PREFIX_SUMMARY.json').read_text())
branches=json.loads((w/'logs/FINAL_BRANCH_COMPARISON.json').read_text())
hist=json.loads((w/'logs/independent_saved_audit/HISTORICAL_480_RECOUNT.json').read_text())
units={}
for n in ['sale_dp_tests','floor_sale_dp_tests','floor_sale_dp_tests_asan_ubsan','resume_boundary_tests','floor_observation_guard','floor_margin_regression']:
 d=json.loads((w/'logs/final_units'/(n+'.json')).read_text());units[n]={k:v for k,v in d.items() if k not in ('rows','cases')}
lines=['# Validation report — TRI_A08_r11_r1', '',
'**Decision status: candidate checkpoint for updated-seed central evaluation; not accepted and not proven to improve the original R2/pool win rate.**', '',
'## Frozen execution identity', '',
'Native SHA256: `'+full['candidate_native_sha256']+'`.',
'Parent native SHA256: `3b95f2c633242e31329adffae2438440a76b5eb6b11ece4897fe69f5710bb9cb`.',
'Official engine SHA256: `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.', '',
'## New full games: final native against the full supplied A08_r11 parent', '',
'The four seeds were predeclared in `validation/logs/ANCHORED_PARENT_PANEL.json` and reused for the final version. These are attachment development seeds, not fresh or sealed test seeds. Policies ran in isolated processes and received only legal own observations. No PASS/full-air opponent was used. Every result below is from a newly executed complete official 719-step game.', '',
'| Referee seed | Candidate seat | Candidate cash | Parent cash | Actual margin | Strict outcome |',
'|---|---:|---:|---:|---:|---|']
for g in full['results']:
 s=g['candidate_seat'];cash=g['terminal_cash_both'];lines.append(f"| {g['seed_referee_only']} | {s} | {cash[s]:,.0f} | {cash[1-s]:,.0f} | {g['candidate_margin']:+,.0f} | {'WIN' if g['candidate_strict_win'] else 'DRAW' if g['draw'] else 'LOSS'} |")
lines+=['', '**5 wins / 8 games = 62.5% against this parent-only panel.** No inference that this equals the 12-opponent win rate is made. The final panel has zero runtime exceptions, zero cash-identity residuals, 60 player-day cash rows per game, and zero sell requests exceeding actual post-unit warehouse stock. Each side had 10 failed BUY_SEED unit attempts across the panel. No failed orders or leftover goods have been converted into hypothetical recovered cash.', '',
'Total final panel wall time inside the game drivers: '+f"{sum(g['wall_seconds'] for g in full['results']):.3f}"+' seconds (individual games 7.283–8.718 s). Maximum measured policy call: '+f"{max(max(g['max_policy_call_seconds_both']) for g in full['results']):.6f}"+' seconds. Raw `trajectory.jsonl.gz` files include legal inputs, both actions, post-unit shed, official cash events, resource receipts and action hashes. Evidence: `validation/logs/margin_closed_loop/` and `FINAL_MARKET_REJECTIONS.json`.', '',
'## Historical evidence, deliberately kept separate', '',
'Recounting all 480 rows gives 393 strict wins, 87 losses, zero ties; public opponents 368/440 and original AFS R2 25/40. The complete 20×12×2 Cartesian panel was checked. None of these is a result of the new candidate. The seven selected fixtures are not substituted for the denominator.', '',
'Independent factual re-execution of saved actions matched all 5,033 official transitions and 420 player-day cash points. Parent legal-observation reproduction and the floor-off ablation each matched all 5,033 supplied actions. These are factual/reproducibility checks, not new games. Evidence: `validation/logs/independent_saved_audit/`, `parent_history/`, and `ablation_prefixes/`.', '',
'## Final candidate common-prefix behavior', '',
'The test stops at the first different action; no subsequent saved future is fed as if it belonged to the candidate trajectory. Four full paths remain identical. Three first divergences only change market orders, not farmer/hand actions.', '',
'| Saved fixture | Calls | First divergence (zero-based step) | Result |',
'|---|---:|---:|---|']
for s in pref['rows']:
 st=s['first_historical_difference'];lines.append(f"| `{s['id']}` | {s['calls']} | {'None' if st is None else st} | {'All 719 actions unchanged' if st is None else 'Stop immediately after first market divergence'} |")
lines+=['', 'The Thomas +70 narrow-win saved path remains fully unchanged. The R2 +2 path does not. The common-prefix scope is 4,341 calls / 4,338 matches / 3 first differences, not 7 new wins or losses. Evidence: `validation/logs/FINAL_PREFIX_SUMMARY.json` and `margin_prefixes/`.', '',
'## Conditional official branches: include the adverse cases', '',
'Four windows × two versions × four declared supply/queue conditions = 32 short branch runs (16 paired comparisons). Rival private inventory was cleared and explicit hypothetical supplies were injected. Only the pre-branch common prefix was read from the saved observation stream. Every subsequent observation was produced by actual official transitions. This is a controlled mechanism test, not execution of the missing original R2 policy.', '',
'Table entries below are **(candidate own cash change − parent own cash change) − (candidate rival cash change − parent rival cash change)** over the same short branch. They include actual executed cash events, not DP scores.', '',
'| Window | No additional rival supply | Queue 0 | Queue 3 | Queue 9 |',
'|---|---:|---:|---:|---:|']
by={}
for x in branches['comparisons']:by.setdefault(x['case'],{})[x['scenario']]=x
for case,v in by.items():
 x=next(iter(v.values()));values=[v[k]['actual_cash_margin_delta_vs_parent'] for k in ['no_additional_sales','traffic_queue0','traffic_queue3','traffic_queue9']]
 lines.append('| `'+case+'` (step '+str(x['start'])+', '+str(x['ticks'])+' ticks) | '+' | '.join(f'{a:+,.0f}' for a in values)+' |')
lines+=['', '**Negative result:** the R2 +2 fixture loses 19 in three of these four explicit conditions, and the Aurax window loses 8 in three conditions. Therefore the floor repair does not certify narrow-win preservation. The R2 worst-loss wool window shows a beneficial structural response under the modeled competitive traffic, but it does not establish recovery of the historical 18,759 full-game deficit. Queue and supply assumptions remain material.', '',
'The corrected Thomas negative window is unchanged in all four conditions. The intermediate weighted objective chose an action that sacrificed 26 own cash to deny 13 rival cash; the final new floor branch uses weight one and passes the synthetic actual-margin regression. The non-floor/configured macro objective was not globally changed.', '',
'Evidence: `validation/logs/FINAL_BRANCH_COMPARISON.json`, `final_branches/`, `validation/cases/`, and `final_units/floor_margin_regression.json`.', '',
'## Solver, guard, oracle, and build checks', '',
'| Check | Result |', '|---|---|',
'| Original non-floor unit suite | 420 problems; 1,680 exhaustive scenario optima; 17,108 transitions; 26,285 assertions; PASS |',
'| New floor unit suite | 702 problems; 2,592 independent exhaustive optima; 6,174 primitive cases; 29,512 assertions; PASS |',
'| Floor ASan/UBSan suite | Same unit cases, address/undefined behavior checks and leak detection; PASS |',
'| Resume boundary suite | 9 problems; 36 exhaustive scenarios; 129 assertions; PASS |',
'| Physical/observation guard suite | 31 assertions; PASS |',
'| Official floor quote/stock oracle | 4,974 cases, both seats × all three queue orderings; PASS |',
'| Source and native binding | All 44 hashes match actual BUILD; 39 production files unchanged |',
'| Independent clean rebuild | Native byte-identical; 28.15 s wall, 594,728 KiB peak RSS |',
'| Default root entry in production-only directory | No binary override, both seat prefixes; 96/96 matching calls |', '',
'The portable unit and official-floor runner modes were re-executed from the staged release. Repeated cases are not counted as additional coverage. ASan/UBSan were applied to the floor solver tests, not to every full-game policy call. Exact per-scenario optima do not imply an exact global robust optimum over all possible opponent schedules: the inherited candidate-profile comparison remains finite.', '',
'## Iteration record and unresolved work', '',
'An initial out-of-range logarithmic floor threshold was detected by tests and replaced by a bounded search with a sentinel. An unanchored trial changed synthetic rolling-plan route choices and its branch guard failed. The current-observation anchor removed that early planning effect. The configured-weight floor variant then exposed an actual-cash-margin regression, leading to the scoped weight-one rule. Raw failures and matching intermediate sources/native libraries are retained; neither superseded library is loaded by root main.py.', '',
'There are 17 actual complete parent-opponent games across the three implementations: 1 + 8 + 8. Only 8 use the final native. No 1,536-game panel, full PASS benchmark, real original-R2 match, or updated-seed central result was generated here.', '',
'Outstanding: original R2/public-pool complete reactive opponents, updated-seed mixed-development evaluation, lagged supply-estimate robustness, cross-product queue interactions, downstream long-horizon cash effects, and target sandbox resource/ABI verification. The user acceptance remains at least 1,306 strict wins out of 1,536. This package supplies a frozen test candidate with explicit risks, not a promoted winner.', '',
'## Resource and evidence controls', '',
'Effective CPU quota 4 cores despite affinity spanning five CPUs; memory cap 4 GiB with a 30% reserve. Initial resource capture was within the first five minutes from the original dispatch. A short compile and 48-observation probe preceded scale selection. Full games were run one at a time, with two policy processes and a lightweight driver. Every reported experiment is bounded and recorded. `RESOURCE_SUMMARY.json` labels the cgroup peak as a lifetime reading rather than attributing it solely to this candidate.', '',
'All original and generated evidence files have checksums in `MANIFEST.sha256`. `BUILD.json` binds production source to native separately. `DELIVERY_TIMING.json` records release packaging relative to the original 02:15:47.575Z dispatch and unchanged 04:15:47.575Z deadline.', '']
(r/'VALIDATION_REPORT.md').write_text('\n'.join(lines))
compact={'task':'TRI_A08_r11_r1','status':'FROZEN_NOT_ACCEPTED','native_sha256':full['candidate_native_sha256'],
 'historical_only':{'wins':393,'games':480,'public_wins':368,'public_games':440,'R2_wins':25,'R2_games':40},
 'new_final_parent_only_games':{'games':8,'wins':5,'losses':3,'draws':0,'all719':True,'runtime_errors':0,'acceptance_panel':False,
 'seed_seat_margins':[{'seed':g['seed_referee_only'],'seat':g['candidate_seat'],'margin':g['candidate_margin']} for g in full['results']]},
 'units':units,'final_legal_prefixes':pref,'conditional_branch_comparisons':branches,
 'acceptance':{'games':1536,'strict_wins_required':1306,'run_here':False,'updated_seed_results_available':False},
 'critical_risks':['R2 narrow +2 saved fixture first diverges at step434; three explicit four-tick conditions lose19 versus parent.','No new original R2/pool full matches.','Lagged public supply profile and multi-product execution interactions remain uncertain.']}
(r/'VALIDATION.json').write_text(json.dumps(compact,indent=2)+'\n')
print('report and machine-readable validation written')
