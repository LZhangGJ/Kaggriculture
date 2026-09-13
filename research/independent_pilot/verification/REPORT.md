# Independent Kaggriculture search and PPO: pilot result

The bounded pilot finished. Neither approach qualified to advance under the frozen win-rate rule. Independent search built a profitable economy, but all candidates lost against the four benchmark opponents. PPO failed to learn a useful economy within this CPU budget. No champion changed and no agent was submitted.

## Fresh test results

The test used eight fresh worlds, four frozen opponents and both seats. Each learned family retained all three training seeds. The simple control ran once per case; the untrained control retained three random initializations. These 640 games are separate from the 1,024 configuration-selection games.

| Agent | Games | Wins / draws | Mean final cash | Mean cash margin |
|---|---:|---:|---:|---:|
| LNS economic plans | 192 | 0 / 0 | 41,626.43 | -98,402.09 |
| Recurrent PPO, entropy 0.01 | 192 | 0 / 0 | 0.00 | -172,081.56 |
| Independent wheat control | 64 | 0 / 0 | 9,324.25 | -169,952.55 |
| Untrained recurrent policy | 192 | 0 / 0 | 0.62 | -178,006.57 |

Strict win rate and match score were zero for every candidate. LNS earned about 4.46 times the control’s cash, but its mean deficit remained 98,402.09. This cash gain is a diagnostic result, not a promotion result. Opponent cash can change with our actions through shared markets and the game’s random stream.

The three LNS training seeds averaged 41,841.17, 43,464.59 and 39,573.53 cash. Each lost all 64 test games. All three selected PPO seeds averaged zero cash and lost all 64 games.

All configurations also had zero wins and draws on selection. The predeclared cash-margin tie-break selected LNS over beam search (-97,188.76 versus -98,144.76) and PPO entropy 0.01 over 0.003 (-165,693.88 versus -168,565.27). Thus selection did not establish a win-rate advantage for either chosen configuration.

The frozen analysis reports paired bootstrap win-rate differences of zero with intervals [0, 0]. Every observed win is zero, so those empirical intervals collapse. They do not establish population equality or rule out rare wins. There are only eight independent world clusters, with related benchmark opponents and intervals conditional on these three training seeds. No claim of leaderboard strength follows.

## What the records show

PPO completed 1,704 training games: 852 against independently written crop/livestock controls and 852 in current self-play. It lost all 852 control games. In self-play, 828 games ended with both players at zero cash. Across 2,556 learner-seat episodes, 2,520 ended at zero cash.

The optimizer changed weights, log probabilities replayed correctly, and the deployed actor matched the official interpreter. These checks establish that the implemented update and execution paths work on the tested cases; they do not establish effective learning. The training outcome shows that this terminal-reward setup did not learn profitable behavior at the tested budget. It does not show that PPO cannot solve the game with a different training design or more compute.

The 24 selected-PPO replays from the first test world contain 407 PLANT commands, four WATER commands and no HARVEST commands. Mean cash at the next day boundary was 1, with median zero. Those records show spending without a completed production cycle. They cover one world and do not isolate a causal defect. The LNS replays contain harvest, feeding, care, transport and sale sequences, consistent with their positive final cash.

## Scope and cost

E1 uses five six-day production, labor, land, reserve and sale targets, with a new observation-based feedback executor. Beam search constructs stages; LNS changes allocations. The planner uses no competitor action labels, weights, hidden inventory or future seed. It is a finite target grammar, not an optimal full-season scheduling algorithm.

E2 starts a 147,458-parameter GRU from random weights. It replays complete 719-turn sequences and uses an ordered conditional decoder for all 24 official commands, ten market slots and integer quantities within conservative resource bounds. Training uses terminal win/draw/loss rewards. PPO adds self-play to the simple-control mixture used by search, so this is a comparison of the two implemented systems rather than an isolated algorithm ablation.

| Family | Trials | Timed process CPU | Full training/search games |
|---|---:|---:|---:|
| Search: beam and LNS, three seeds each | 6 | 1,082.19 seconds | 100,608 |
| PPO: two entropy values, three seeds each | 6 | 1,086.66 seconds | 1,704 |

Each trial received 180 process CPU seconds and finished its last complete rollout/update. PPO collected 1,837,764 eligible learner turns. The 12-trial pipeline took 1,148.09 elapsed seconds with two concurrent workers. Timed CPU excludes model/optimizer initialization; elapsed pipeline time includes startup. Evaluation worker wall time sums to 3,792.09 seconds across two workers; that sum is neither elapsed time nor CPU time. No GPU was used.

The tested collector reached 2,046.88 eligible learner turns/second on a four-game correctness workload. Native plan simulation reached 177.64 full games/second in its separate check. These component measurements exclude work outside their timed regions and cannot stand in for a GPU training benchmark.

CP-SAT receives LNS proposals that have already passed deterministic feasibility repair. This pilot therefore cannot show a benefit from solver repair. It also does not isolate the UCB mutation controller. Both require separate ablations before making performance claims.

## Verification and evidence

- All 12 trials and all 1,664 evaluation games finished. There were zero failed evaluation games and no trial retries.
- The first world in each evaluation phase checked every observation for every candidate/opponent/seat against official 1.32.7: 208 complete games and 299,520 observation comparisons, with complete replay files.
- Five focused regression tests passed. Separate full-game checks covered shared resources, partial cash trades, capacity/overnight behavior, the final action, three plan controls and direct-action decoding. The direct decoder matched 3,065 menus.
- The policy replay check’s maximum log-probability error was 0.0000009536743. A real optimizer update changed 18 tensors, and checkpoint save/load and a full deployed-actor game passed.
- The final supplementary audit independently recomputed case coverage, configuration choice, score arithmetic, action-stream hashes, replay terminal cash and training totals. All 130 frozen file hashes and the 18-file source inventory matched.
- Local maximum test action times were 2.76 ms for LNS and 247.75 ms for PPO. All tested calls met the pilot’s local one-second gate. This does not certify Kaggle startup, memory or hosted execution.
- The original evaluation holdout supplied membership exclusions only. No holdout games or outcomes were used. The pilot test is now consumed; final confirmation still requires a fresh custodian release and current campaign audit.

The archive contains 293 files, including checkpoints, journals, game rows, replays, manifests, the tested binary and earlier development receipts. Its contents and hashes were checked after compression. See [evidence-manifest.json](evidence-manifest.json) and [reproduction instructions](README.md).

- Frozen source commit: `5e745a237e85baa22d1163f6b334d2f37ed1d36f`.
- Base evaluation commit: `f98cda9cbb3dfc0acfd69df84777bc68055ba827`.
- Input freeze SHA256: `a137b86b2917d888599671eae46c4350a6e328bf0f9f4ece5f500607d3dce39e`.
- Test rows SHA256: `4bff3383663dca1db098dfb27f0645ce3d4282a235851fe0a135cca6003b3136`.
- Evidence archive SHA256: `a25cb4232b50aa32c9da80c131f0af7e3872c527396671160446a9c4c9be3746`.

## Recommended next experiment

1. Run a bounded PPO learnability test using trajectories from our own profitable search plans. Compare a policy initialized from those trajectories with the current random-start control, using fresh seeds and the same downstream evaluation rule. Measure completed production cycles, retained cash and wins against the independent training controls. This tests whether an economic starting point fixes the observed failure; competitor actions are unnecessary.
2. Improve E1’s feedback executor and training-opponent mixture. Use stronger independently generated plans in development and measure delivery losses, idle time and investment payback. Keep strict wins against the strong benchmark opponents as the advancement rule. If testing CP-SAT, provide unrepaired proposals and isolate its effect.
3. Scale the better-supported training design on the team JAX simulator after matching the decoder and passing parity checks. The inspected 1.32.7 snapshot has a feed-forward actor with two market slots and eight quantity choices, which differs from this pilot. Its README’s RTX 3090 rates were not repeated here. Pin the complete source/table bundle, regenerate events for new seeds, port the ordered actor boundary, require zero bound diagnostics, and measure full rollout-plus-update throughput on an assigned GPU. See [JAX_AUDIT.md](../JAX_AUDIT.md).

These are proposed next experiments. This run stops at its frozen 12-trial budget and completed test; no larger training run has started.
