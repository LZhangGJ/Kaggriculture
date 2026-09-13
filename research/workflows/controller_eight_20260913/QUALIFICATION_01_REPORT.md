# Economic qualification 1

C03, C05 and C08 passed. C01, C02, C04, C06 and C07 need economic revisions before round-robin selection.

All 256 games completed: 16 shared fresh seeds, both seats, against an inactive PASS opponent. Each entry needed mean terminal cash of at least 200,000 across all 32 games. The independent audit checked 184,064 recorded transitions, terminal DONE states, cash/reward agreement, replay hashes and frozen source identities. It did not rerun the interpreter.

| Author | Mean terminal cash | Mean difference from parent | Result |
|---|---:|---:|---|
| C01 | 196,342.0625 | +794.0625 | Fail |
| C02 | 197,116.4375 | +1,568.4375 | Fail |
| C03 | 209,496.6562 | +13,948.6562 | Pass |
| C04 | 195,548.0000 | +0.0000 | Fail |
| C05 | 201,762.3750 | +6,214.3750 | Pass |
| C06 | 195,548.0000 | +0.0000 | Fail |
| C07 | 195,548.0000 | +0.0000 | Fail |
| C08 | 211,787.5625 | +16,239.5625 | Pass |

C04, C06 and C07 returned the unchanged parent and produced identical cash results. Eight independent chats have not produced eight distinct strategies.

## Critical economic cases

These are shortfalls against PASS and comparisons with the parent, not competitive losses. The JSON preserves daily cash and final-day resource snapshots. The complete archive contains each replay under `evaluation/results/games/`.

| Author | Seed | Seat | Cash | Difference from parent | Replay file |
|---|---:|---:|---:|---:|---|
| C01 | 2027896586 | 0 | 139,533 | -36,282 | game_000028.replay.jsonl.gz |
| C01 | 1856155813 | 0 | 142,327 | -42,857 | game_000006.replay.jsonl.gz |
| C02 | 2027896586 | 0 | 133,592 | -42,223 | game_000060.replay.jsonl.gz |
| C02 | 2027896586 | 1 | 146,694 | -41,807 | game_000061.replay.jsonl.gz |
| C03 | 2027896586 | 1 | 122,381 | -66,120 | game_000093.replay.jsonl.gz |
| C03 | 1681080428 | 1 | 139,329 | -58,799 | game_000067.replay.jsonl.gz |
| C04 | 1681080428 | 0 | 131,126 | +0 | game_000098.replay.jsonl.gz |
| C04 | 1658182076 | 0 | 155,189 | +0 | game_000126.replay.jsonl.gz |
| C05 | 1681080428 | 1 | 137,978 | -60,150 | game_000131.replay.jsonl.gz |
| C05 | 1856155813 | 0 | 152,588 | -32,596 | game_000134.replay.jsonl.gz |
| C06 | 1681080428 | 0 | 131,126 | +0 | game_000162.replay.jsonl.gz |
| C06 | 1658182076 | 0 | 155,189 | +0 | game_000190.replay.jsonl.gz |
| C07 | 1681080428 | 0 | 131,126 | +0 | game_000194.replay.jsonl.gz |
| C07 | 1658182076 | 0 | 155,189 | +0 | game_000222.replay.jsonl.gz |
| C08 | 337943030 | 1 | 155,191 | -66,612 | game_000241.replay.jsonl.gz |
| C08 | 1681080428 | 1 | 159,966 | -38,162 | game_000227.replay.jsonl.gz |

## Revision tasks

- C01: compare investment horizons and trace when spending pays back. Its worst cases lose more than 36,000 cash against the parent.
- C02: test crop changes with matched ablations and untouched validation seeds. Its two lowest-cash cells lose more than 41,000 against the parent.
- C04: preserve a tested labor and execution change; the first round retained no new work.
- C06: test resource and storage bottlenecks with feasible collection and sale schedules.
- C07: test rolling replanning after observed production or scheduling changes. Save source and records early.

Each failed author received its complete source, all 32 results, two full low-cash replays, matched parent replays, and a 45-minute task. Authors must measure current CPU, memory and speed, preserve every attempt, freeze before validation, and return a buildable checkpoint even on failure. The controller will draw new shared seeds after revisions.

Cash trajectories and leftover resources can guide tests; they do not prove the cause of a shortfall. No competitive acceptance claim follows from qualification. Competitive acceptance still requires more than 85% strict wins against the fixed 12-entry pool, with no cash threshold.
