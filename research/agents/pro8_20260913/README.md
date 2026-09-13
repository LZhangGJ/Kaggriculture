# Three frozen Kaggriculture agents

This release publishes the exact A08 r11, A06 r6 and A06 r12 harvest-calendar agents evaluated locally on 13 September 2026. Each folder contains the runnable entry, matching native library, production source, build metadata, an English introduction and the complete original delivery archive.

| Agent | Core mechanism | Latest12-opponent win rate |
|---|---|---:|
| [A08 r11](A08_r11/README.md) | Finite sale-window DP with public supply scenarios | 393/480 (81.88%) |
| [A06 r6](A06_r6/README.md) | Sale-timing DP with public opponent-supply risk | 374/480 (77.92%) |
| [A06 r12 calendar](A06_r12/README.md) | Whole-portfolio finite harvest-date DP | 349/480 (72.71%) |

The shared historical 20-seed comparison used 4145681825–4145681844, 11 frozen public entries plus original AFS R2, and both seats. All 1920 games across the four originally tested agents completed 719 transitions with zero errors. Three of those agents are published here; the historical CSV retains the fourth comparison arm for an auditable record.

Public opponents: soil_v219g, moon_v215, flexon_v5, market_smart_v8, nagatakengo_v70, aurax_reactive_v1, thomas_955_v2, shop0909, aurax_shop_v2, seven_turn, ahmed_v27. Original R2 is submission_56149565, native SHA256 1ead09a9bd48b20b512fb8fe57bbbbd87c86bb12fc9b1b42553e5c5b5bec121c.

Opponent entries are weighted equally. flexon_v5 and seven_turn share production files, so names are not independent strategy families. Wins mean strictly greater terminal cash; draws earn no win. Each policy has an isolated process per game; the hidden seed is withheld. The host watchdog is 120 seconds per response; this is not Kaggle timeout-forfeit certification.

The next user-requested shared panel contains 64 seeds sampled without replacement from the published representative 256 / stress 128 union: 35 representative and 29 stress. It schedules 1536 games per candidate, 4608 total. See [NEXT64_PANEL.json](evaluation/NEXT64_PANEL.json). Its mixed overall rate and both strata will be reported; it is development feedback, not sealed holdout. No results on that panel are claimed in this release.

The target for subsequent iterations is an overall win rate strictly above 85% for each agent against the specified 12 opponents. Every author round is limited to 120 minutes, uses measured CPU/memory capacity and preserves a buildable checkpoint. Full local panels are orchestrated separately.

Verification: published runtime bytes match the exact evaluated identities; production source checks match author build metadata; full original ZIP archives pass CRC validation. All payload SHA256 values are listed in SHA256SUMS.txt. No strategy code was modified for this publication.
