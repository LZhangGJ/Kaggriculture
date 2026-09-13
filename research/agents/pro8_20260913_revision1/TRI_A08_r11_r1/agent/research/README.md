# Research checkpoints, not production entry points

The only delivered production choice is the root `main.py` plus `policy/tri_a08_r11_r1.so` identified by `SOURCE_FREEZE.json`.

`checkpoints/unanchored_floor/` retains the first price-floor extension (native SHA256 `450223885af7296e9b60683a9b65d96d9829fcf02819baa205499cadd33c90fd`). It allowed floor decisions in later synthetic rolling-planner states. One complete parent-opponent pilot lost by 412. A Thomas short-branch first-action guard failed when route actions changed; that failure is preserved in the raw logs. This version is superseded.

`checkpoints/anchored_config_weight/` retains the observation-anchored version (native SHA256 `6e6c95e93ed02680957472d1f36f001c7f85a79a38dbc014a2051c9f9a9cb3eb`). Its eight parent-opponent development games were 5 wins / 3 losses. The configured macro opponent weight could worsen actual cash margin in a low-traffic Thomas window, motivating the final scoped weight of one. It is superseded even though those eight terminal outcomes happen to equal the final panel.

These historical tree snapshots can contain copied parent README/BUILD documents at their root. For each snapshot, its native-adjacent `.BUILD.json` and the corresponding raw build log are the authoritative source/native identity. Do not mistake its inherited root prose for current acceptance evidence.

`ablations/floor_off.so` was built from the final 44 production sources with `A08_SALE_FLOOR_DP=0`. Its source manifest and exact flags are in the adjacent receipt.

`original_drivers/` preserves the paths and commands used during development. Use root `tests/run_validation.py` for relocatable re-execution. The early `floor_original_units` failure exposed an out-of-range floor-threshold computation for a logarithmic item curve; the delivered version uses a bounded search and explicit no-floor-in-domain sentinel. The final unit/oracle suites and floor-unit sanitizers passed after that repair.

Earlier failed runs, aborted conditional branches, replays, and smoke checks are not extra full games. There were 17 complete games across the three candidate implementations, all against the supplied A08 parent, and only 8 belong to the frozen final candidate. No original R2 executable was replaced with the parent in any claim.
