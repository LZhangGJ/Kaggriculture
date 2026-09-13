# Economic revision round 2

C02 passed controller import verification. C01 also delivered before its cutoff and is under review. C04, C06 and C07 are still working; their cutoffs are around 19:54 UTC on 13 September 2026.

C02 reports mean terminal cash of 201,405.03125 across all 32 predeclared validation games, compared with 200,617.9375 for its supplied baseline. The difference is 787.09375, with 16 improved games and 16 worse games. This is a narrow author-panel result. Fresh controller qualification has not run.

The revision removes two rotation-forecast proposal families while retaining the no-fertilizer alternative. The controller checked all 231 new game records and replayed all 166,089 transitions through the official engine. Its GCC 11 rebuild passed 621 assertions and seven entry checks. Both used-seed games reproduced every action and terminal cash. Two non-game errors remain documented.

C01 reports a single policy change: reducing the planning horizon from four days to two. It reports 209,593.875 mean cash on 32 untouched validation games, 290 complete new replays and 620 native checks. The controller safely extracted all 1,978 files; source, replay and local build verification are pending.

| Archive | Bytes | SHA-256 | Status |
|---|---:|---|---|
| C02_round2_strategy.zip | 262,780,104 | 5729f5bd617b2a4439e9fd123d5f8d208bd077950a594ec91c16c75d2fa27a75 | Import verified; fresh qualification pending |
| C02_ROUND02_CONTROLLER_VERIFICATION.zip | 1,270,767 | 1ed573505021e925c92e101d51321337581be65fe8231f492199102664d5835f | Verification evidence |
| C01_round2_strategy.zip | 267,981,338 | 9c5fe72e41580c9ab9ce61019169ffd8c829ad7d98de3b13b8431b04dda644af | Hash and safe extraction checked; verification pending |

Original round-one archives and runtimes remain unchanged. Each new revision receives a separate verification directory and receipt. The next shared controller seed panel will be drawn only after all revised sources are verified and frozen.
