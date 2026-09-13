# Economic revision round 2

All five revision authors finished research before their 45-minute cutoffs. C01 and C02 passed controller source and replay verification. Fresh controller qualification has not run.

| Author | Author validation mean cash | Controller verification |
|---|---:|---|
| C01 | 209,593.875 | 290 full replays; 208,510 official transitions matched |
| C02 | 201,405.03125 | 231 full replays; 166,089 official transitions matched |
| C04 | 195,431.625 | Failed author panel; revised source and full archive unavailable |
| C06 | 204,759.1875 claimed | Runtime and used-seed smoke checks passed; full author evidence unavailable |
| C07 | 197,938.94 reported | Original archive identity unavailable; different surviving archive under audit |

C01 changes the planning horizon from four days to two. C02 removes two rotation-forecast proposal families while keeping the no-fertilizer alternative. Both local builds passed their native and entry checks; both used-seed games reproduced all actions and terminal cash.

C04 and C07 reported validation below 200,000. C04 recovered only its original inputs, report, validation CSV and inventory. Its revised source and full replays remain unavailable. C04 started a new 30-minute research round at 20:01:54.463 UTC, with a hard cutoff of 20:31:54.463 UTC, early source checkpoints and evidence archives below 50 MB.

C07 recovered a different 363,858,523-byte archive, whose source and native library differ from the original receipt. The controller verified its two parts and whole SHA-256, and is auditing the actual contents. Its original archive and claims remain distinct and unverified.

C06's runtime ZIP downloaded and matched SHA-256 c512a03db1ad718d7102e8b1956bce02360c7da1eecdfc5082c325b38afcbfe7. Its local build passed eight fertilizer test groups and seven entry/reset/privacy checks. The author reports that the full evidence ZIP no longer exists in its current environment. The claimed 502 completed games, two interruptions and validation mean remain unverified. Recovery finished without any round-two game records or replays. The controller verified 690 recovery-manifest hashes and matched 74 older replays to already preserved evidence. Those files do not verify the round-two claims. Two local used-seed smoke games completed 719 transitions each, with terminal cash 194,519 and 205,439.

All original archives, failed results and runtime versions remain preserved. C03, C05 and C08 retain their unchanged round-one strategies. The next shared controller panel will use newly drawn seeds after source verification and freezing, excluding all recoverable prior seeds. Unknown lost seed histories remain an explicit limit.

No round-robin or competitive acceptance games have run. Acceptance requires at least 1,306 strict wins in 1,536 games against the fixed twelve-opponent pool, with no competitive cash threshold.

C07 audit update: the surviving archive contains 232 complete new game replays, all verified through 166,808 official transitions. Its actual 32-game validation mean is 187,473.125, versus 187,673.875 for its matched parent. It failed the economic gate. Its local build, seven entry checks and both-seat action reproduction passed. This evidence belongs to surviving archive SHA-256 9d4f6aafb9f40ddf05916f275284ca6250d87b018994ba5e9052ab47c41f3310; it does not verify the different original delivery. Both seed sets remain excluded from future draws.
