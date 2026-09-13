# Evidence index and result-accounting boundaries

## Frozen identities

- Parent: `1400cdd7ef2677dc16b9c2b46a34a7ad41419d038c6e9bf5ebf60137707add3a`.
- Preliminary candidate: `832a59e5ef4456c6ab16adf12ed6e90f4af025d4cde82ef489ee7fdd755e3216`.
- Final candidate: `d642c5650c916365136bfc5ce5c47f82240c27dc34d9baf691a580376b56ec0b`.

The preliminary and final production strategy correction is the same. The final
build also synchronizes the pre-existing offline r12 enable control with the new
executor flag. That control is not called by the root agent. No outcome-based
parameter change was made. The final native was independently rebuilt byte-for-byte
from the full delivery directory.

## Read first

`../VALIDATION_SUMMARY.json`, `../PROVENANCE.json`, `../SOURCE_FREEZE.json`,
`../BUILD.json`, `logs/closed_loop_summary.json` and `../tests/UNIT_RESULTS.json`.
`logs/input_identity.json` verifies the input, parent source and native.
`logs/history_summary.json` aggregates all 480 rows, not the seven selected cases.

## Current production evidence

`logs/delivery_clean_build.*` is the final clean compilation.
`logs/candidate_final_rebuild.*` (where present) and `logs/candidate_rebuild.*`
are earlier compile checkpoints; read the native hash in each JSON receipt.
The per-unit logs and current `tests/UNIT_RESULTS.json` retain seven successful
suites. `logs/terminal_parent_*` records the old-source counterexample:
`--expect-legacy` successfully reproduces rejection of three productive paths;
without that flag the assertion fails intentionally (exit 134). That failure is
the expected negative control, not a failure of the delivered implementation.

`logs/prefix_final/` contains all seven final-native first-divergence probes.
Their `.summary.json` files state `completed_candidate_games: 0`. Every compared
historical frame is checked against the official engine before either candidate
call. At the first action difference there is at most one same-tick transition,
using the opponent's action at its still-unchanged observation. No saved opponent
future is followed. The historical +7 and +363 wins both diverge and therefore
have no verified candidate terminal win result.

`logs/closed_loop/PANEL_PLAN.json` declares the four-seed development panel and
notes the preceding first-seed pilot. There are 12 final-panel `.summary.json`
files: four real parent-parent controls and eight real candidate-parent games.
Each summary identifies both actual native hashes, engine hash, seed supplied
only to the driver, steps, player-day cash count, outcomes and the raw-log hash.
Every `.json.gz` includes all 719 emitted joint actions, observation/state hashes,
cash before/after and 60 player-day cash records. These are genuine policies
reacting to newly advanced official states, not fixed replay actions. Opponent
identity is the A06_r12 parent only. No public/R2 win-rate claim follows.

`logs/entrypoint.*` and `logs/delivery_entrypoint.*` test the root callable,
independent contexts and resets on short real-policy prefixes. They do not add
completed games to the denominator. Post-extraction checking writes outside the
extracted archive and is summarized in the delivery verification record.

## Preliminary, historical and failed harness work

`logs/preliminary/` contains one complete game with the preliminary native plus
early action-difference diagnostics. That one game is excluded from final-candidate
results but included in task accounting: 13 actual complete games total =
12 final-panel games + one preliminary candidate pilot. The parent-parent pilot
is already one of the four controls, not an extra fourteenth game.

An early all-case prefix invocation outlived the tool's synchronous response
window. Its bounded child completed, but the associated timing record was empty.
Those incomplete timings are not reported as measurements. Final probes were
rerun case by case with complete timing logs. Original diagnostics remain as raw
checkpoints, not final candidate outcomes.

`logs/harness_failures/prefix_seed_keyerror.stderr` records an initial offline
harness failure before a game comparison: the seed was nested under `row` in the
case manifest. The corrected driver verifies this against both replay metadata
fields before execution. The failed v0 script is retained for audit, not offered
as a runnable test. Production code was not implicated.

`logs/historical_reproduction.*` and `*_parent_trace.json.gz` are 5,033 exact
parent-action reproductions on seven supplied legal histories, not new matches.
The audits and all original 480 rows are in `../tests/reference_bundle/`.
`parent_metadata/` and `../tests/historical/` contain inherited artifacts, clearly
separate from this task's current receipts. Some inherited scripts reference the
original author's absolute experiment paths; use the documented current portable
drivers instead. No omitted large original ZIP is nested in this delivery.

## Unfinished, not silently passed

No 1,536-game central panel, no overall >85% acceptance, no completed reruns against
the original selected public/R2 policies, and no proof that both supplied narrow
wins remain wins. One paired final-panel margin regresses by 189 despite retaining
its win. The calendar mismatch is a proven local defect, not a proven explanation
for every historical loss or the entire r6-to-r12 difference. Current interpreter
checks do not certify Kaggle sandbox/schema/time-limit acceptance or successful
fills of all orders. Conditional output is never booked as realized profit in
these reporting claims.
