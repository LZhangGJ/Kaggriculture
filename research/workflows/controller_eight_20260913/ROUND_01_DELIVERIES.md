# Round 1 author deliveries

C01, C03 and C05 passed controller import, source, record, local build and development reproduction checks. C08 has delivered and is under review. C06's first archive failed its evidence check and has returned to its author for repair. C02, C04 and C07 are still running.

| Author | Games against PASS reported by author | Mean terminal cash verified from records | Controller fresh qualification |
|---|---:|---:|---|
| C01 | 128, on 64 seeds in both seats | 201,614.625 | Not run |
| C03 | 256, on 128 seeds in both seats | 207,077.0234375 | Not run |
| C05 | 128, on 64 seeds in both seats | 204,386.359375 | Not run |

These panels use different seeds and do not rank the strategies. The controller will test all eight on a fresh shared panel. No competitive acceptance result exists for either candidate.

C01's saved records contain 930 complete games and 194 settings-count errors. All complete traces reconcile, and its local rebuild passed 620 checks and both-seat reproduction. Its package development mean is 209,684.0833, which differs from the final chat's earlier figure. Pre-restart logs remain missing. The archive also retains inherited parent documents; those are not current C01 evaluation evidence.

C03's 1,054 recorded attempts and 757,826 trace transitions reconcile. Its GCC 11.4 rebuild passed the supplied tests and reproduced both complete development games, including all action hashes and terminal cash. The rebuilt binary has a different hash from the author's GCC 14.2 binary; both identities are recorded.

C05's archive contains 1,905 attempted games across its research history: 1,897 complete games and eight failed configuration attempts. Its frozen candidate's 128-game mean recomputes from the records. The unchanged parent scored higher on that same author panel; the cash result does not prove an improvement over the parent.

The controller checked all 1,897 successful C05 terminal records and replayed every supplied trace: 107 files and 76,933 transitions. Full traces cover 16 of its 128 qualification games; the remaining 112 have terminal records. Both local development games reproduced their cash and complete action sequences. The fresh controller panel will save complete replays for every game.

C06's report refers to a `research_snapshot` directory that is absent from its archive. Its experiment inventory is empty. The controller has not verified the claimed baseline cash result and has requested actual post-recovery records within the original two-hour deadline. The incomplete archive remains preserved.

Original archives preserve each author's source, build instructions, English descriptions, evaluation settings and complete supplied records. Controller verification packages preserve the audit scripts, local runtime, build receipts and reproduction evidence. Used author seeds join the shared exclusion record before controller evaluation.

## Archive identities

| File | Bytes | SHA-256 |
|---|---:|---|
| C01_round1_strategy.zip | 20,987,379 | 44a97c11edf1cff755d484c5f45db727a1b3097a84067947aab0d35493872bc1 |
| C03_round1_strategy.zip | 33,930,565 | c5e95c87c27085af1d3c2a8fab268583af0ad29b2cd05eda34f34674f3c3cc2f |
| C05_round1_strategy.zip | 133,606,945 | 93397be0a56b7ca8f27c4a6d4b3a1f9b70b1c6115472bda97629b18b5a43f74f |
| C06_round1_strategy.zip, incomplete | 2,583,667 | d30e9f251449dc84a7efbd53f564efec6c2c2340ef52e45348376fccfef35bdd |
| C08_round1_strategy.zip, under review | 54,433,128 | 098299ff6df8c921116558bc8274615bc888bf48e53f4d417c69ade848bea896 |
| C01_CONTROLLER_VERIFICATION.zip | 1,371,638 | 8ecc0d1f0f0a80ecb00fddd050a19e303fc0e50fbb93d7dbb3aa11152b17cd13 |
| C03_CONTROLLER_VERIFICATION.zip | 1,530,545 | ea751c49b4817b0821d1ccea1da669a6dd221d347a357e56a4461af99de603e1 |
| C05_CONTROLLER_VERIFICATION.zip | 2,315,087 | e4a73d84bbde7b214a2baebd4143c0b8e9da9be1faff2f7304a603ecba2ccdaf |
