# Eight new Kaggriculture authors

## Competitive round 01: verified result

All 3,072 planned cells are accounted for: 3072 valid games and 0 failures. The independent saved-record audit passed. Accepted: none.

| Candidate | Strict wins | Win rate | Errors | Accepted |
|---|---:|---:|---:|---|
| C05 | 843 / 1,536 | 54.882812% | 0 | NO |
| C03 | 743 / 1,536 | 48.372396% | 0 | NO |

Acceptance requires at least **1,306 strict wins out of all 1,536 planned games per candidate**, plus complete valid coverage and unchanged source identities. Draws and errors are not wins. **There is no competitive cash threshold.** The 64 shared seeds were drawn after source freeze, with both seats against the designated 11 public opponents plus original R2.

[Verified report](COMPETITIVE_01_REPORT.md) · [Settings](COMPETITIVE_01_SETTINGS.json) · [Source identities](COMPETITIVE_01_FREEZE.json) · [Full archive and prior evidence](https://github.com/LZhangGJ/Kaggriculture/releases/tag/controller-eight-round01-20260913)

The complete archive contains source and available opponent runtimes, full and partial replays, all results, errors, settings, seeds, exclusions, provenance and audit. Large analysis remains in the archive. Original R2's C++ source remains unavailable; its exact native runtime is preserved. Earlier missing author evidence and unknown historical seeds remain explicit gaps. Daily cash differences identify cases to inspect, not proved causes or promised improvements.

## Earlier campaign record

The following status record predates competitive round 01. Its then-current statements are historical. The verified result above is current.

Eight independent ChatGPT 6 Pro chats began on 13 September 2026. Each received the same verified source bundle and a distinct research focus. Their first task is a complete strategy that averages at least 200,000 terminal cash against an inactive opponent across multiple seeds.

The fourth shared cash panel completed all 256 games; six strategies passed. C03 Round02 then failed at 195,097.75 mean cash. The controller verified all 428 games and 307,732 official transitions. [Verified report](C03_ROUND02_VERIFIED_REPORT.md). C03 Round03 passed its author sample at 205,368, using the original policy retained before validation. The controller verified all 326 new games and 234,394 official transitions. This shows no policy improvement. [Verified report](C03_ROUND03_VERIFIED_REPORT.md). The fifth shared cash panel completed all 256 games and passed its independent record audit. Seven candidates cleared 200,000; C06 averaged 197,472.0625. [Results](QUALIFICATION_05_REPORT.md). C06 Round06 has delivered. All 15 archives match the final author receipt and are published. The final report claims a changed policy at 203,546.0625 mean cash; it conflicts with earlier live claims of an unchanged policy at 216,051.8125. Independent verification is complete: all 722 official replays passed, with 519,118 transitions. The changed policy and prevalidation selection are supported; earlier live claims remain unsupported. [Verified report](C06_ROUND06_VERIFIED_REPORT.md). Shared qualification 06 is complete: all eight passed, all 256 games were valid, and the independent record audit passed. [Results](QUALIFICATION_06_REPORT.md). The first round robin completed all 896 games without errors. Its independent audit verified 644,224 recorded transitions and selected C05 (176/224 strict wins) and C03 (143/224). [Tournament report](ROUNDROBIN_01_REPORT.md). Their fixed-pool evaluation is now running 1,536 games each on 64 fresh shared seeds, both seats, against the designated 11 public opponents plus original R2. Its ranking follows strict win rate, head-to-head results, worst-opponent rate, then ID for exact ties. [Delivery receipt](C06_ROUND06_delivery_receipt.json). C08 Round02 passed its author cash gate at 212,618.125. All 644 games and 463,036 official transitions passed controller verification. Its source is frozen pending shared qualification. [C08 verified report](C08_ROUND02_IMPORT_VERIFICATION_REPORT.md). No competitive acceptance results exist yet. [Qualification 4 results](QUALIFICATION_04_REPORT.md).

- [Current deliveries, evidence and archive hashes](ROUND_01_DELIVERIES.md)
- [C01 controller verification and recovery gaps](C01_IMPORT_VERIFICATION.json)
- [C03 controller verification](C03_IMPORT_VERIFICATION.json)
- [C05 controller verification](C05_IMPORT_VERIFICATION.json)
- [C06 incomplete delivery and repair request](C06_IMPORT_FAILURE.json)
- [C08 controller verification](C08_IMPORT_VERIFICATION.json)
- [Original strategy archives and controller verification packages](https://github.com/LZhangGJ/Kaggriculture/releases/tag/controller-eight-round01-20260913)

- [Round plan and acceptance rules](ROUND_01_PLAN.md)
- [Author roster and start times](AUTHOR_ROSTER.json)
- [Source bundle](AUTHOR_INPUTS.zip)
- [Imported historical record verification](IMPORT_VERIFICATION.json)
- [Local input build and runtime smoke](INPUT_SMOKE.json)
- [Evaluator source, fixed opponent runtimes and all preflight rows](EVALUATOR_PACKAGE.zip)
- [Four-worker runtime preflight](RESOURCE_PREFLIGHT.json)
- [Known seed exclusions](SEED_HISTORY.json)

The source bundle contains 71 files plus a hash manifest: buildable A06 r6 revision 1 source, the frozen official referee and historical results. Bundle SHA256: `75356640e384979f504fa7393796ab3c4e8dd22b051d5284599d7e4266b2e76d`.

The local smoke rebuilt the parent with GCC 11.4 in 14.81 seconds. Two complete games on one seed took 2.32 and 2.55 seconds, with terminal cash of 188,345 and 240,555. This checks that the input runs; one seed does not establish economic qualification. Author environments must be measured separately.

The controller independently checked the published baseline and revision report hashes and all 9,216 game rows, including exact seed/opponent/seat coverage and terminal cash comparisons. It checked CRC integrity for six original ZIPs. Full historical central replay bytes were absent, so this import check does not verify those replay hashes or reproduce the games.

Formal competitive acceptance requires at least 1,306 strict wins from 1,536 planned games per candidate. No cash threshold applies. Future reports will retain source identities, every game result and failures, both-seat coverage and fresh seed records.

The fixed-pool preflight completed all 24 games: one already-used seed, all 12 opponents and both seats. Four CPU workers took 24.73 seconds, with no errors and unchanged source hashes. This projects about 53 minutes for two 1,536-game panels at that measured throughput; revised strategies and other seeds may take longer. It does not establish competitive strength.

The evaluator package includes all 46 pinned opponent runtime files, preserves their licenses and records exact source hashes. Original R2 retains the required native hash. Its original C++ build source is unavailable; the runtime is preserved unchanged. The package contains a reproduction note for paths that came from the original host.

- [Shared qualification results and critical cases](QUALIFICATION_01_REPORT.md)
- [Independent full-frame audit](QUALIFICATION_01_ANALYSIS.json)
- [Revision round dispatches and deadlines](ROUND02_DISPATCHES.json)


[Round05 tasks and actual deadlines](ROUND05_DISPATCHES.json): C02, C04, C05 and C06 completed these revision tasks. Their later verified sources and results are recorded in qualification 4. [Qualification 3 complete results](QUALIFICATION_03_REPORT.md).


## C05 shared benchmark: final result

[Verified shared benchmark results](C05_SHARED_REPORT.md) rank C05 separately on the representative, stress and retired-holdout panels. All 20,480 cells and the 204,800 unchanged baseline records passed audit. These are reused public benchmark results, not fresh competitive acceptance.
