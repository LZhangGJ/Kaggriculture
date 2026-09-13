# TRI_A08_r11_r2

**Status: compiled and frozen candidate for central testing; NOT accepted.**
The exact parent is `TRI_A08_r11_r1`. This package contains the complete production
agent, not a patch-only submission. No new closed-loop match was run in this
round, and no candidate win rate or recovered episode cash is asserted.

## Run and rebuild

The submission entry is `main.py:agent(observation, configuration=None)`.
It loads only `policy/tri_a08_r11_r2.so` and the fixed `policy/config.json`.
The included native is Linux x86-64, compiled with Debian g++ 14.2.0.
No network access or nonstandard Python packages are required.

```sh
export PYTHONDONTWRITEBYTECODE=1
python build.py
python tests/audit_inputs.py
python tests/run_checks.py --sanitize
```

The default builder reconstructs the candidate from the included production
sources. Its real command, feature switches, compiler, elapsed build time,
source hashes and native hash are in `policy/tri_a08_r11_r2.BUILD.json`.
`BUILD.md` points to the actual independent clean-build receipts. Test reruns
write to a fresh timestamped `rerun/` directory, not the delivered raw evidence.
`python tests/run_checks.py --saved-limit 719` additionally calls the actual root
entry on all seven saved own-observation streams; it still runs **zero games**.

## One structural change: charge the collection work of credited animal output

The selected issue is inconsistent animal collection-labor accounting between
`AnimalServiceDP` and `Controller::animal_path`, not a new market-price tweak.
The parent already models feed and care. The correction is not to treat all
wheat procurement as waste, and it does not suppress the investment module.

The parent credits current held animal yield without its required HARVEST work.
It books one manure-collection action on day d while crediting the manure made
at dawn d+1, losing the correct collection-day allocation at horizon boundaries.
Its service DP includes future product revenue but not the corresponding
HARVEST action, and clips manure net value at zero even though the actual jobs
collect available manure. Thus the chosen service plan can appear profitable
under a different labor convention from the executable lifecycle stream.

The correction charges one HARVEST per positive collection (not per unit), one
COLLECT_FERTILIZER when manure is collected, and places each on the actual
collection day, including current stock and day 29. The service DP deducts the
same collection actions from conditional future value. Existing feed/care and
production clocks, legal observations, startup purchase and placement, rolling
planning, crop allocation, labor-to-hire mapping and actual executor are kept.

`work_price` is the parent's existing labor opportunity-price parameter. It is
**not an additional fee charged by the official engine**. Actual wages and cash
still follow the existing execution model. Moving a work atom between days can
change hiring and project ranking; this does not establish a realizable profit
increase in a game. Route travel/drop overhead remains a pre-existing heuristic.
The conditional DP is not a proof of global schedule or market realization.

## Source and observed evidence

The seven supplied own ledgers were independently reconciled: all 210 own
player-days balance and their terminal cash matches the 720th observation.
All 1,536 supplied result rows were retained and reaggregated, rather than using
the seven selected examples as a win-rate denominator.

A concrete non-price-floor example is `flexon_v5_113649547_seat0`, step 144,
day 6, sheep at (4,2): held yield 6, manure available, not fed/cared. The parent's
asset books first-day labor 3.4; its actual jobs require HARVEST,
COLLECT_FERTILIZER, FEED and CARE plus the unchanged 0.4 travel proxy. The own
trace records CARE at step 149, FEED 150, HARVEST 159 and COLLECT_FERTILIZER 160 at
that tile. The revised accounting is four collection/service operations, not
three. At that observation wool is 217, fertilizer 90 and wheat 32. These are
historical facts, not a counterfactual cash recovery.
See `analysis/observed_labor_example.json` and the exact own trace/ledger.

An initial ongoing-crop water/fertilizer first-day disagreement hypothesis was
**not supported**: zero mismatches across 4,636 eligible crop-days in seven
parent-reproducing audits. No crop-service patch was made. The raw parent
diagnostics (compressed JSON) and hypothesis summary are preserved in
`analysis/`. The diagnostic native was independently rebuilt from the frozen
parent and reproduced byte-for-byte.

## Preserved capabilities and identity

40 of the 44 original production source/configuration files are byte-identical.
The only strategy-code changes are `policy/animal_service_dp.hpp` and
`policy/triad.hpp`; `main.py` and `build.py` change artifact identity/build wiring.
`policy/config.json`, the executor, crop/land allocation, paid continuation,
finite-sale-window DP and price-floor DP sources remain unchanged. All ancestor
feature switches remain enabled at their exact parent settings.

The only new compile switch is `A08_ANIMAL_COLLECTION_LABOR=1`. It is fixed ON
in this release. `--animal-collection-labor 0 --out /absolute/ablation.so` is a
research ablation only, not the submitted candidate. It restores all 5,033
parent actions on the seven saved streams. That isolates this source change;
it is not a win-rate experiment.

No opponent source/private inventory, historical seed dispatch, future script
or opponent-name lookup was introduced. Production consumes the same explicit
current-observation fields through the unchanged codec. Fixture IDs and seeds
exist only in evidence/tests and are never read by the production entry.

## Validation completed

| Check | Actual result | Scope |
|---|---|---|
| Input manifest and parent native | 69 input files verified; exact 45-file parent retained | Identity |
| Parent rebuild | Byte-identical original native | Offline build |
| Candidate default build + independent clean default rebuild | Byte-identical candidate native | Offline build |
| Conditional animal service DP | 5,120 independent exhaustive state cases agree | Conditional model, not global policy optimality |
| Lifecycle cost/date stream | 4,608 cases agree; positive service and negative-cost extension controls | Synthetic focused tests |
| ASan + UBSan | Same focused C++ suite passed; separate UBSan run also passed | No full-game sanitizer claim |
| Frozen official unit/dawn rules | 4,608 own-animal lifecycles; 16,128 day states; 26,389 effective actions; 90,914 checks | Local rule/physical accounting tests, not matches |
| Delivery/resource negative controls | Undeposited, off-depot, missing carried feed, limited-action and full-shed cases | No fabricated executable sale |
| Actual root contract | 183 checks; receipt/native/ABI, seat isolation, reset, metadata ignored, loud invalid-input errors | Not Kaggle's sandbox certification |
| Parent diagnostic/ablation saved entry | 5,033/5,033 exact parent actions each | Saved-observation regression |
| Candidate saved entry | 5,033 calls, no exception; 3,252 same and 1,781 changed parent actions | No candidate trajectory or win inference |

A focused negative-return maintenance example uses day 28, an existing goose
with one prior unfed day, wheat 25, next-day egg 30, manure 2 and work 4. The parent
chooses a +1 extension because it omits/clips collection work; the corrected
choice retires at value 0. Raising egg price to 100 makes continued feeding
positive (+65 under the same conditional shadow-cost model), and the corrected
DP keeps it. This verifies a cost boundary, not a realized game cash gain.

The negative-control legacy model intentionally exposes 3,937 DP-value,
5,236 labor-day and 3,521 stream-value disagreements. Its test reports PASS
because it successfully detects the old defect; those are not new candidate
failures. All raw outputs, exact commands and timing logs are included.

## Narrow-win protection: unresolved, explicitly not claimed

| Saved own trace | Historical parent margin | Candidate actions equal parent | First changed step | Ablation equal parent |
|---|---:|---:|---:|---:|
| `aurax_reactive_v1_1742537856_seat0` | -11567 | 433/719 | 192 | 719/719 |
| `flexon_v5_113649547_seat0` | -17185 | 519/719 | 144 | 719/719 |
| `market_smart_v8_915042141_seat0` | +78 | 417/719 | 96 | 719/719 |
| `soil_v219g_113649547_seat0` | -17178 | 519/719 | 144 | 719/719 |
| `submission_56149565_521596133_seat1` | -15877 | 559/719 | 264 | 719/719 |
| `submission_56149565_915042141_seat1` | +324 | 381/719 | 192 | 719/719 |
| `thomas_955_v2_616522013_seat1` | -163 | 424/719 | 144 | 719/719 |

The +78 market_smart and +324 original-R2 cases change at steps 96 and 192.
Their wins are **not proven preserved**. From the first divergence onward,
saved observations describe the parent's trajectory, not the candidate's
actual future; all later actions are only entry/robustness probes. No replayed
future prices, cash ledger or parent opponent response was used to claim
counterfactual proceeds. Central closed-loop testing must decide regressions.

## Formal evaluation status

The supplied parent panel is 64 seeds (48 representative, 16 stress), 12 opponents
and both seats: **1,054/1,536**, public 11 **969/1,408**, original AFS R2 **85/128**,
all 719 steps, no errors or ties. These are parent results only. Different rounds
use different seeds and cannot establish a causal benefit of the previous patch.

The candidate has **zero new closed-loop games** here. No full 1,536 batch and
no all-PASS opponent run was started. Updated central evaluation must freeze
this identity and use its new 64-seed panel; acceptance remains strictly >85%,
at least **1,306 wins out of 1,536**. The supplied development pool is not a sealed
holdout. `analysis/DEVELOPMENT_SEEDS.json` preserves all supplied seed identities,
and declares no new game seeds.

## Resource budget and interruptions

First local resource capture: 2026-09-13 15:22:51 UTC, within the first minute
of the user's 15:22:07.194 UTC start. Effective affinity has 5 logical CPUs but
cgroup quota is 4 CPUs; memory limit is 4 GiB. Work was serialized and kept a 30%
memory reserve rather than treating host totals as entitlement. The complete parent rebuild peaked at 595,104 KiB RSS (about 581 MiB);
the candidate build at 595,032 KiB and the clean rebuild at 594,712 KiB.
Saved-entry probes peaked below 170,000 KiB. Exact timing/RSS receipts are
preserved in `logs/` and summarized in `RESOURCE_AND_TIMING.json`.

Three outer-tool wrapper interruptions left incomplete GNU-time receipts: the
first parent-build batch, the last parent diagnostic in a multi-probe batch,
and the first clean-rebuild batch. Their artifacts/output had completed, but
the incomplete wrapper receipt is not counted as a measured pass. Each was
repeated with complete successful timing; original incomplete logs are retained.
There was no compiler error or failed correctness assertion in the selected
candidate checks. Observed narrow-win action changes and the unsupported crop
hypothesis are retained rather than hidden.

## Package map

`main.py`, `build.py`, `policy/`: complete production and matching library.
`reference/parent/`: exact read-only parent source/config/native.
`evidence/`: supplied manifest, identities, full results, selected cases, own
traces and ledgers, unchanged official public rules and provided probe script.
`tests/`: new standalone focused tests; no missing previous-round test dependency.
`analysis/`, `logs/`, `probes/`, `rerun/`, `checkpoints/`, `development/`: source
diff, rejected hypothesis, raw diagnostics, all ablations/intermediate builds,
resource records and original development checkpoints. `PACKAGE_SHA256SUMS.txt`
checks all packaged files other than itself. No external files are needed to
run the submission or the documented focused tests.
