# C03: Round02 economic revision

Your unchanged original source averaged **198,392.78125 terminal cash** on qualification 04: 16 shared seeds, both seats, 32 complete games against legal PASS. It missed the 200,000 cash gate. C01 averaged **219,845.21875** on the same panel. The four disclosed controller panels are development evidence now; their matched PASS cash comparisons are not head-to-head win rates.

## Task and time limit

Produce one coherent improvement to economic choices or rolling scheduling, supported by the loss evidence. Keep the complete strategy working. Measure actual CPU quota, usable memory, build time and full-game speed first. Plan the work around those measurements. Use at most two workers and **90 minutes from the controller's actual dispatch**, reserving the final **20 minutes** for fixed validation and packaging. The controller will provide the start time and hard cutoff when it sends this task.

Within five minutes, save and link a complete source/runtime checkpoint. Link another checkpoint after each accepted source change. The final archive alone does not prove that an early backup was available. Return by the cutoff even if the candidate fails.

## Diagnose before changing the policy

The packet includes your exact frozen source in `strategy/`, the unchanged official referee in `referee/`, and C01's current source in `reference_C01/`. It also contains all four panels' results, settings, seeds and source identities. Older C01 versions may differ; use each panel's identity record when comparing results.

The critical cases combine your two lowest-cash cells and two largest deficits to C01, with duplicate cells removed. Each has both full replays. Find the **first material cash divergence**, then trace realized purchases, labor costs, harvests and sales. Explain how scheduling affected those transactions. Stored goods and predicted profit alone do not establish recoverable cash.

Use that diagnosis to choose one improvement and a small number of informative comparisons on disclosed development seeds, both seats. Avoid a blind parameter sweep or lengthy extra evidence without a clear finding. You may reuse a measured C01 component with explicit attribution; do not return an unchanged, renamed C01 strategy. Preserve your own source lineage and explain the complete final policy in English.

Aim for a robust margin above 200,000 across the disclosed panels, since results near the threshold have changed on fresh samples. This is a research aim, not an extra acceptance gate or a guarantee. Do not lower the threshold.

## Fixed validation and evidence

Declare 16 fresh author-validation seeds before observing their outcomes, excluding every ID in `KNOWN_USED_SEEDS.json` and every new seed used during this task. Freeze the selected source before running all 32 cells, both seats against legal PASS. Report every declared cell, failure or interruption. Keep incomplete games separate from terminal cash; do not silently reduce the denominator or report a completed-only mean as the full panel result. Do not add seeds until the mean passes, change the source after validation, or select a fallback from validation results.

Publish baseline and selected development means with the exact matched cells. Reserve time for a matched validation baseline where measured speed permits; if omitted, say so clearly. Always report the frozen candidate's complete 32-cell validation panel. Seeds belong only to the evaluator and must not enter policy observations or configuration.

Preserve every attempted game, including failures, interrupted prefixes and separate retries. Save the full actions and observations/states for each transition, settings, source and native hashes, timestamps, resource use and errors. Complete games need 719 transitions, both players DONE and terminal cash equal to rewards. Bind each replay to the exact source variant that produced it. Keep rejected and unbuilt variants clearly labeled.

## Delivery

Return a separate complete source/runtime ZIP, English report, all results, seed inventory, settings, resource measurements and SHA-256 manifest. Include source-only build instructions, entry/reset checks and runnable tests for the changed behavior. Package full evidence in ordinary independently readable ZIP parts **below 50 MB each**; do not use split binary fragments or embed old large archives. Verify the archives and every manifest member before linking them.

The controller will download and audit the delivery, then draw fresh shared evaluation seeds after revision. Economic qualification precedes local strategy matches. Competitive acceptance later requires a strict win rate above 85% against the fixed 12-opponent pool on 64 shared seeds, both seats: 1,536 games per candidate, with no competitive cash threshold. Do not publish externally or start unrelated work from this author task.
