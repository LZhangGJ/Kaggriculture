# Round 1 author deliveries

All eight ChatGPT 6 Pro author rounds have ended. C01, C03, C05, repaired C06 and C08 passed source, record, build and development reproduction checks. C02 also passed import checks, including verification of its failed holdout. C07 also passed import checks. All eight surviving entries completed the shared controller panel. C04 recovered only the unchanged input archive; the controller rebuilt and smoke-tested that parent fallback. C04 has no surviving evidence of new author work.

| Author | Author economic panel | Mean terminal cash | Fresh controller qualification |
|---|---:|---:|---|
| C01 | 128 games, 64 seeds | 201,614.625 | 196,342.0625: Fail |
| C02 | 64 holdout games, 32 seeds | 191,950.71875 (below threshold) | 197,116.4375: Fail |
| C03 | 256 games, 128 seeds | 207,077.0234375 | 209,496.6562: Pass |
| C04 | No surviving current records | Unverified | 195,548.0000: Fail |
| C05 | 128 games, 64 seeds | 204,386.359375 | 201,762.3750: Pass |
| C06 | 64 recovery games, 32 seeds | 204,465.921875 | 195,548.0000: Fail |
| C07 | 16 recovery games, 8 seeds | 206,421.25 | 195,548.0000: Fail |
| C08 | 256 games, 128 seeds | 201,529.2265625 | 211,787.5625: Pass |

These author panels use different seeds and cannot rank strategies. The shared controller panel completed all eight surviving entries on 16 newly generated seeds in both seats: 256 games with four workers and full replays. All source hashes were frozen before the draw; it excludes 820 known seeds. The qualification mean must reach 200,000 across all 32 complete games per entry. C03, C05 and C08 passed economic qualification. The other five authors have started 45-minute revision rounds with full low-cash replays and matched parent comparisons. All 256 games and 184,064 recorded transitions passed the independent integrity audit. No competitive acceptance result exists.

C01 supplies 930 complete traces and 194 configuration errors. All complete traces reconcile; 620 checks and both-seat local reproduction pass. Its saved development mean differs from its stale final-chat figure. Pre-restart logs are missing, and inherited parent receipts remain separate from current evidence.

C02 reports a failed frozen holdout despite a 210,729.12 development mean. The parent scored 196,381.50 on the same holdout. The controller verified all 414 attempts (410 complete and four interrupted), replayed all 149 available traces (107,131 transitions), and passed 545 build tests. Used-seed development cash and all 30 daily frames reproduce in both seats. Full historical actions for that development seed were not stored, so no action-hash reproduction is claimed.

C03's 1,054 recorded attempts and 757,826 trace transitions reconcile. Local build, tests and both-seat development reproduction pass.

C04 had no downloadable checkpoint at its 18:45:26.806 UTC deadline. The controller stopped generation at 18:45:34.054 UTC, 7.248 seconds late. A separate retrieval-only turn found only the supplied archive. It made no source changes and ran no builds or experiments. All C04-specific source changes, seeds, attempts, failures, replays and timing receipts are missing. The controller verified all 71 manifest entries, rebuilt the parent source and ran two complete reused-seed smoke games. This is a parent fallback, not evidence of C04 improvement.

C05 supplies 1,897 complete records and eight configuration failures. The controller replayed all 107 supplied full traces (76,933 transitions). Only 16 of its 128 author qualification games have full traces; the other 112 have terminal records. Local development cash and complete actions reproduce. The parent scored higher on its author panel.

C06's first archive omitted its claimed research directory and remains preserved as a failed delivery. Its repaired archive withdraws that unsupported claim and contains 70 complete replays (50,330 transitions), including six duplicate checks, plus 16 failed launches. The mandatory 16-game recovery panel averaged 196,455.375; a predeclared 48-game expansion averaged 207,136.104167. The unchanged policy averaged 204,465.921875 across all 64 economic games. Source, replay, build, API/reset and both-seat reproduction checks pass. Earlier logs and seeds remain unknown.

C07 lost its first generation and recovered no new source or earlier records. It reports an unchanged parent rebuild, 18 full replays including two smoke games, and one interrupted attempt. All 18 full replays (12,942 transitions), one interrupted attempt, source identity, build, seven entry checks and both-seat full-action reproduction were verified. C04, C06 and C07 are parent fallbacks; eight chats did not produce eight distinct strategies.

C08's 2,862 attempts and 111 panels reconcile with zero game errors. All 540 supplied traces (388,260 transitions) replay exactly. Local build, tests and both-seat reproduction pass. Two compiler interruptions remain recorded separately.

Original archives preserve all supplied source, descriptions, settings and results. Verification archives preserve controller scripts, local runtimes and receipts. Missing records remain explicit. Compiler-dependent binary hashes are recorded separately from original author binaries. Known author seeds join the exclusion ledger before fresh evaluation; lost histories prevent a claim of complete historical seed coverage.

## Archive identities

| File | Bytes | SHA-256 |
|---|---:|---|
| C01_CONTROLLER_VERIFICATION.zip | 1,371,638 | 8ecc0d1f0f0a80ecb00fddd050a19e303fc0e50fbb93d7dbb3aa11152b17cd13 |
| C01_round1_strategy.zip | 20,987,379 | 44a97c11edf1cff755d484c5f45db727a1b3097a84067947aab0d35493872bc1 |
| C02_CONTROLLER_VERIFICATION.zip | 1,184,166 | caacc3fc478a5331d7e5d874ca860135bf20986fe742c5004ddf1b1ee75a92f9 |
| C02_round1_strategy.zip | 41,808,117 | 13062d1229dca0ab6aea83e0c6e1ab2b04725dd1b8eb6d26175a30254584a0b0 |
| C03_CONTROLLER_VERIFICATION.zip | 1,530,545 | ea751c49b4817b0821d1ccea1da669a6dd221d347a357e56a4461af99de603e1 |
| C03_round1_strategy.zip | 33,930,565 | c5e95c87c27085af1d3c2a8fab268583af0ad29b2cd05eda34f34674f3c3cc2f |
| C04_CONTROLLER_VERIFICATION.zip | 2,104,138 | 927ccb5f581466abadfada107d5fb29333dba8d11ee834629e9b5460e0c7f1b4 |
| C04_round1_surviving_inputs.zip | 1,445,404 | 75356640e384979f504fa7393796ab3c4e8dd22b051d5284599d7e4266b2e76d |
| C05_CONTROLLER_VERIFICATION.zip | 2,315,087 | e4a73d84bbde7b214a2baebd4143c0b8e9da9be1faff2f7304a603ecba2ccdaf |
| C05_round1_strategy.zip | 133,606,945 | 93397be0a56b7ca8f27c4a6d4b3a1f9b70b1c6115472bda97629b18b5a43f74f |
| C06_CONTROLLER_VERIFICATION.zip | 1,094,540 | 0bce28b3aab716a214f525575d76ef318d561a29e1799e7246fb656e5caa05f9 |
| C06_round1_repaired_evidence.zip | 71,527,816 | 762423049128ed5d7a6e1165b7fa99d70e75deb17e15d454a381011f48f0cc33 |
| C06_round1_strategy.zip | 2,583,667 | d30e9f251449dc84a7efbd53f564efec6c2c2340ef52e45348376fccfef35bdd |
| C07_CONTROLLER_VERIFICATION.zip | 1,090,795 | 95e52ac141d250f9ea91285f1967c78a7ae195683714447d0cd92907e6ab0626 |
| C07_round1_strategy.zip | 19,403,675 | e2690fe5ce1cb1dcf62ad2f1407daa8a28dfe45cceed0ecd0436b7657cee8105 |
| C08_CONTROLLER_VERIFICATION.zip | 2,370,131 | 2d8d68de9e38b1e76aaa0231b56722c113d404c5e9fd1025696524a6595aa674 |
| C08_round1_strategy.zip | 54,433,128 | 098299ff6df8c921116558bc8274615bc888bf48e53f4d417c69ade848bea896 |
