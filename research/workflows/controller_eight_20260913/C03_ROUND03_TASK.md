# C03 Round03: improve the full-panel economic result

Round02 did not improve the full strategy. Its selected source averaged **195,097.75** on the 32 fresh validation cells, versus **195,448.09375** for the original baseline. Its pooled 128-cell development mean was about **203,037.98**, versus **203,090.36** for the original, with regressions on three of four panels. Improvements in a few critical cases did not support selecting it as a broad improvement.

Start from your verified original source in `strategy/`. The failed Round02 source is preserved in `failed_round02/` for diagnosis. Produce one justified change to production, investment choice or rolling scheduling while retaining your own complete strategy. C01's current source is included as an attributed reference; a measured component may be reused, but an unchanged renamed C01 is not an independent C03 revision.

## Sixty-minute task

The last author environment had a four-core CPU quota and 4 GiB of memory. A build took about 51 seconds, and selected full games averaged 7.739 seconds. Measure these limits and speeds again at the start and size the work accordingly. Use at most two workers. The maximum is **60 minutes from actual dispatch**, with the final **15 minutes reserved for fixed validation and packaging**. The controller will set the exact start and cutoff when sending the task.

Save and link a complete source/runtime checkpoint within five minutes and after every accepted source change. Return the final package by the cutoff even if the strategy fails.

## Diagnose the failed selection

The packet includes four distinct largest selected-minus-baseline validation losses with both full replays, daily cash comparisons and the first observed and first material cash differences. A material difference is defined here as an absolute cash gap of at least 1,000; that is a diagnostic convention, not a performance threshold. The packet also preserves the original author transaction records, labeled separately from controller-derived replay cash.

Trace realized production, sales, purchases and labor costs around those differences. Explain why the proposed change should help more than the selected bad cases. Use all four disclosed controller panels to test that explanation. Avoid a blind parameter sweep or more large runs without a specific economic question. Do not claim unused stock is recoverable profit without checking collection, travel, sale timing and costs.

Compare one coherent change against the original source on matched disclosed cells, both seats. Treat critical-case gains as diagnostic evidence, not the selection objective. Declare the selection rule before comparing outcomes. Report baseline and candidate panel means, the pooled result, each regression and the reason for selection. If no change earns selection, preserve that conclusion rather than claiming improvement.

Seek a robust margin above 200,000 because results near the gate have changed on new seed panels. The gate remains unchanged; a development margin does not guarantee fresh-panel success.

## Fixed validation

Before any validation outcomes, declare 16 fresh seeds excluding `KNOWN_USED_SEEDS.json` and every new seed used during this task. Freeze the selected source before all 32 candidate games, both seats against legal PASS. Never add seeds until the mean passes, change source after validation, or pick a fallback based on validation results. Include every declared cell and preserve failures or interrupted attempts separately; a completed-only mean must not be presented as the complete panel.

Report baseline and selected development means. Run a matched validation baseline if measured speed permits; otherwise state that it was omitted. Keep evaluation seeds out of policy observations and configuration.

## Evidence and delivery

Preserve every attempt, partial prefix, failure and separate retry, with complete action and observation/state traces. Complete games require 719 transitions, both players DONE and terminal cash equal to rewards. Bind each replay to its exact source, native library, settings, timestamps and resource measurements. Preserve rejected and unbuilt source variants with clear labels.

Deliver a complete source/runtime ZIP, English description, all results, seed inventory, settings, build instructions, entry/reset checks, changed-behavior tests, failures and SHA-256 member manifests. Split evidence into ordinary independently readable ZIPs below 50 MB each. Do not embed old large archives. Verify every archive member before linking the files.

The controller must first qualify all eight strategies at the economic gate, then run local strategy matches and select two. Competitive acceptance is at least **1,306 strict wins out of 1,536 games** against the fixed 12-opponent pool on 64 shared seeds, both seats. There is no competitive cash threshold. Do not create controller evaluation seeds, publish externally or start unrelated work.
