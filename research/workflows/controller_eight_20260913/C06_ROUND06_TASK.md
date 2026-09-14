# C06 Round06: recover reliable economic performance

Your Round05 source averaged **197,472.0625 terminal cash** on qualification 05: 16 fresh shared seeds, both seats, 32 complete games against legal PASS. It was the only candidate below the 200,000 mean gate. C03 averaged **209,813.40625** and C01 **201,736.15625** on the same cells. These are matched economic results, not direct matches or competitive win rates.

Make one coherent improvement to economic choice, rolling scheduling or execution in your complete C06 strategy. The current configuration uses scenario 0, work_price 0, capital_power 0.7, discount 0.08 and an animal ceiling of 26. Those settings identify the starting source; they are not a request to sweep them.

## Time and resources

Use **at most 60 minutes from actual dispatch**, reserving the final **20 minutes for validation and packaging**. The controller will provide the exact start and cutoff. Your last measured environment had a four-core CPU quota and 4 GiB memory, a 29.86-second initial build and a 27.80-second clean build. Selected validation games averaged 1.683 seconds. Current controller games averaged 1.685 seconds, with a maximum of 1.733 seconds and observed child RSS of 23,704 KiB. These are separate environments and workloads; remeasure CPU quota, memory, build and full-game speed before planning the work. Use at most two simultaneous build, game or replay workers.

Save and link a complete source/runtime checkpoint within five minutes and after every accepted source change. Deliver by the cutoff even if the candidate fails.

## Diagnose and select before validation

`strategy/` is your exact verified Round05 runtime. `reference_C03/` and `reference_C01/` contain the current matched reference sources with their identities. All five controller panels include results, settings, seeds and exact source identities. Some earlier panels used different source versions: do not label old-version cash as the current baseline without actually running the current source on those cells.

The critical cases are the two largest paired deficits to each reference, with duplicate own cells removed. Both reference replays are included where needed, along with daily cash and the first cash difference. A first material difference uses an absolute cash gap of 1,000 as a diagnostic convention, not a new gate. Trace actual harvests, sales, purchases, hiring and delivery timing around that divergence. Explain the realized transactions before changing a rule. Remaining inventory and forecast profit do not prove recoverable cash or the cause of a loss.

Use that evidence to state one economic question and test a small number of informative variants. Declare the selection rule before seeing development outcomes. Compare against the current C06 baseline on matched disclosed cells, both seats, across the five panels where measured capacity permits. Report exact panel coverage, baseline and candidate means, pooled results and every regression. Do not select a change merely because it fixes the supplied critical cases. Avoid a blind parameter sweep or more large runs without a specific finding.

You may reuse a reference component after a measured comparison and clear attribution. Retain C06's source lineage and complete policy; do not return an unchanged renamed reference. Seek a robust margin above 200,000, without lowering the gate or claiming that a margin guarantees fresh-panel success.

## Fixed author validation

Before any validation outcomes, declare 16 fresh seeds excluding every ID in `KNOWN_USED_SEEDS.json` and all new seeds used in this task. Freeze the selected source before all 32 validation games, both seats against legal PASS. Do not extend the seed set until the mean passes, change source after validation or select a fallback from validation outcomes.

Include every declared cell. Preserve failed or interrupted attempts separately; a completed-only mean must not be labeled the complete panel. Report baseline and selected development means, and run a matched validation baseline if measured speed permits. If that baseline is omitted, say so. Keep evaluator seeds out of policy observations and configuration.

## Complete evidence and delivery

Preserve every attempt, partial prefix, failure and separate retry, with full actions and observations/states. Every completed game requires 719 transitions, both players DONE and terminal cash equal to rewards. Bind each replay to the exact source/native hashes, settings, timestamps and resource use. Preserve rejected and unbuilt source variants with clear labels.

Deliver a separate complete source/runtime ZIP, English strategy description, all results, seeds, settings, build instructions, entry/reset checks, tests for the changed behavior and SHA-256 member manifests. Use ordinary independently readable evidence ZIPs below 50 MB each. Do not embed old large archives. Verify every member and ZIP CRC before linking it.

C06 Round02's claimed 502 complete games remain missing historical evidence. This task must not recreate or relabel those claims as verified. Unknown lost seed histories remain unknown; the packet excludes all recoverable IDs.

The controller must qualify all eight strategies economically before local strategy matches and selection of two candidates. Competitive acceptance later requires **at least 1,306 strict wins in 1,536 games** against the fixed 12-opponent pool on 64 shared seeds, both seats. There is no competitive cash threshold. Do not draw controller evaluation seeds, dispatch work, publish externally or start unrelated work.
