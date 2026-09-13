# Round 1 author deliveries

C03 and C05 have delivered complete strategy archives. C03 passed controller import, source, trace and local reproduction checks. C05 passed archive, manifest, record and local build checks; its remaining trace and reproduction checks are in progress. Six authors are still running.

| Author | Games against PASS reported by author | Mean terminal cash verified from records | Controller fresh qualification |
|---|---:|---:|---|
| C03 | 256, on 128 seeds in both seats | 207,077.0234375 | Not run |
| C05 | 128, on 64 seeds in both seats | 204,386.359375 | Not run |

These panels use different seeds and do not rank the strategies. The controller will test all eight on a fresh shared panel. No competitive acceptance result exists for either candidate.

C03's 1,054 recorded attempts and 757,826 trace transitions reconcile. Its GCC 11.4 rebuild passed the supplied tests and reproduced both complete development games, including all action hashes and terminal cash. The rebuilt binary has a different hash from the author's GCC 14.2 binary; both identities are recorded.

C05's archive contains 1,905 attempted games across its research history: 1,897 complete games and eight failed configuration attempts. Its frozen candidate's 128-game mean recomputes from the records. The unchanged parent scored higher on that same author panel; the cash result does not prove an improvement over the parent.

Original archives preserve each author's source, build instructions, English descriptions, evaluation settings and complete supplied records. Controller verification packages preserve the audit scripts, local runtime, build receipts and reproduction evidence. Used author seeds join the shared exclusion record before controller evaluation.

## Archive identities

| File | Bytes | SHA-256 |
|---|---:|---|
| C03_round1_strategy.zip | 33,930,565 | c5e95c87c27085af1d3c2a8fab268583af0ad29b2cd05eda34f34674f3c3cc2f |
| C05_round1_strategy.zip | 133,606,945 | 93397be0a56b7ca8f27c4a6d4b3a1f9b70b1c6115472bda97629b18b5a43f74f |
| C03_CONTROLLER_VERIFICATION.zip | 1,530,545 | ea751c49b4817b0821d1ccea1da669a6dd221d347a357e56a4461af99de603e1 |
