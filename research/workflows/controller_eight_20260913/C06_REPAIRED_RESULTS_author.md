# Final ZIP delivery identity

Verified at 2026-09-13T18:29:28.958062+00:00, before the original 2026-09-13T18:45:58.765Z deadline.

- File: `C06_round1_repaired_evidence.zip`
- Exact bytes: 71,527,816
- File members: 663
- SHA-256: `762423049128ed5d7a6e1165b7fa99d70e75deb17e15d454a381011f48f0cc33`
- Manifest hashes: all 661 entries matched; CRC checks passed.
- Inventory: five indexed attempt suites; all 70 completed-game replays present.
- Deep archive inspection: 50,400 state frames and 50,330 transitions scanned.
- All 54 runtime/preflight files match the clean-extracted, rebuilt checkpoint.

The original incomplete ZIP is preserved byte-for-byte inside the replacement.
No missing historical experiment has been invented or restored by assertion.

---

# C06 round 1: repaired delivery and post-recovery evidence

## Result and source boundary

**The unchanged delivered policy produced $204,465.921875 mean terminal cash across 64 complete PASS games: 32 predeclared distinct seeds, both seats.** Total terminal cash is $13,085,819. This local sample exceeds the $200,000 mean target. It is not a controller fresh-qualification result or a competitive-acceptance result.

The original approximately $209,142 / 16-game claim is withdrawn. The claimed historical research logs are absent and were not recovered. Every result in this report is supported by newly generated, timestamped post-recovery files. No missing result was inferred or reconstructed from memory.

The actual failed ZIP is preserved byte-for-byte at `audit/failures/C06_round1_strategy.zip`: 2,583,667 bytes; 144 members; SHA-256 `d30e9f251449dc84a7efbd53f564efec6c2c2340ef52e45348376fccfef35bdd`. Its original report is also preserved. Inspection confirms that old archive had no `research_snapshot/` directory and its experiment inventory was `[]`. These quarantined documents contain withdrawn claims, not recovery evidence.

## Exact economic results

| Panel | Distinct seeds | Full games | Mean terminal cash |
|---|---:|---:|---:|
| Mandatory first eight seeds, both seats | 8 | 16 | $196,455.375000 |
| Predeclared expansion | 24 | 48 | $207,136.104167 |
| **Combined predeclared panel** | **32** | **64** | **$204,465.921875** |

The mandatory subset alone was below target. The 24-seed expansion had already been declared before the first recovery game, and was run in full. All declared seed/seat pairs are included; no completed game was discarded. The policy and its settings were not tuned or changed between panels.

Seat 0 mean: $207,939.468750. Seat 1 mean: $200,992.375000. Minimum game cash: $123,175; maximum: $266,007; median: $204,085.50. The two seats on a seed are paired observations, not 64 independent seeds.

Per-game rows are in `audit/economic_games.csv`, `audit/all_game_attempts.csv`, and each suite's `results.jsonl`. Exact machine-readable calculations are in `audit/economic_summary.json` and `audit/economic_by_seed.json`.

### Seed and seat cash table

| Seed | Panel | Seat 0 cash | Seat 1 cash |
|---:|---|---:|---:|
| 986533592 | mandatory | $211,441 | $211,441 |
| 1845281115 | mandatory | $163,018 | $163,018 |
| 1361880447 | mandatory | $218,194 | $216,836 |
| 106519797 | mandatory | $204,168 | $204,168 |
| 430892602 | mandatory | $186,638 | $186,965 |
| 785533991 | mandatory | $153,736 | $176,944 |
| 1715573336 | mandatory | $184,333 | $184,333 |
| 566204725 | mandatory | $240,489 | $237,564 |
| 824829880 | expansion | $198,299 | $158,623 |
| 1631542730 | expansion | $241,036 | $191,344 |
| 481745539 | expansion | $259,753 | $259,753 |
| 490448923 | expansion | $249,677 | $253,729 |
| 1342908283 | expansion | $205,755 | $190,040 |
| 249403354 | expansion | $201,408 | $201,408 |
| 682972272 | expansion | $158,160 | $154,960 |
| 482178936 | expansion | $236,817 | $242,744 |
| 1291433730 | expansion | $241,264 | $241,737 |
| 1459745997 | expansion | $266,007 | $264,770 |
| 1626554043 | expansion | $213,038 | $213,038 |
| 1165360502 | expansion | $240,221 | $130,625 |
| 242575516 | expansion | $191,570 | $207,133 |
| 192423017 | expansion | $242,787 | $242,589 |
| 280508456 | expansion | $188,397 | $175,466 |
| 1966655268 | expansion | $204,003 | $187,787 |
| 163699382 | expansion | $199,308 | $200,018 |
| 132554512 | expansion | $155,474 | $123,175 |
| 961322289 | expansion | $212,989 | $208,237 |
| 1025707663 | expansion | $236,964 | $236,964 |
| 196407811 | expansion | $188,576 | $190,933 |
| 932766373 | expansion | $152,106 | $167,975 |
| 683755405 | expansion | $180,345 | $179,347 |
| 1291125580 | expansion | $228,092 | $228,092 |

## Full-game, replay and reset checks

All 64 economic games completed exactly 719 transitions. Both players first reached DONE at transition 719, and both final rewards equaled their final cash exactly. The PASS player used the supplied `pass_agent` on its own legal observation and finished at the configured starting cash of $3,000. There were no in-game exceptions in the economic panel.

The full replay for every completed game contains an initial state, all 719 actions/transitions with both players' complete states, policy-input hashes and timing, and a terminal-check record. The result JSON records each replay's byte count and SHA-256. Seed values exist only in evaluator metadata; policy configuration has `seed: null`. The policy receives only its own observation and sanitized configuration.

Four additional full games in one Python process test automatic step-0 reset and seat isolation, with no explicit reset between games. Each repeated seat reproduced its original action hash, full-state hash and terminal cash. Two further complete games from a clean extracted/rebuilt runtime reproduced both seats exactly. These six duplicate-seed checks are not included in the 64-game economic mean.

All **70 completed-game replays** were independently reconstructed with the frozen official referee: **50,330 transitions and 50,400 full state frames**, zero mismatches. Every recorded policy-input hash and every PASS action was also checked. Referee-only replay reconstruction does not count as another policy game.

## Failures and missing evidence

The first recovery validation launcher had a path-handling bug: a relative replay output path was passed to `Path.relative_to` with an absolute root. All 16 scheduled launches failed before `LocalGame` was constructed, so they have zero transitions and no terminal cash. The failed launcher, per-attempt stderr, seeds, statuses and elapsed subprocess timings are retained. Resolving the output directory fixed the harness; no policy code changed.

The archive therefore records **86 game-launch attempts: 16 pre-game harness failures, 64 economic games, and 6 duplicate validation games**. All 70 games that actually started completed. `audit/experiment_inventory.json` is a populated index of all five suites, including the failed one; it is not an empty placeholder. No full replay exists for a launch that failed before game initialization.

Historical research claims about labor-price ablations, reserve changes, fertilizer changes or service variants are not carried forward as verified findings. Those logs and experiment sources were not found in the provided current environment. The original failed archive is the complete preserved record of that delivery failure.

## Source, builds and measured resources

All 71 entries in the published `MANIFEST.json` matched their hashes. The input ZIP SHA-256 is `75356640e384979f504fa7393796ab3c4e8dd22b051d5284599d7e4266b2e76d`. The 44 runtime/build source files and settings remain byte-identical to the actual failed delivery and the published parent. The root policy is not an untested recovered experiment.

The compiler was `g++ (Debian 14.2.0-19) 14.2.0`. Measured cgroup CPU quota: `400000 100000` (four CPU cores), affinity CPUs [0, 1, 2, 3, 4]; memory limit: 4,294,967,296 bytes (4 GiB). Python: 3.13.5.

The root build receipt reports 25.746013 seconds for compilation and receipt generation. `/usr/bin/time` measured 26.34 seconds wall time and 570,408 KiB maximum resident memory for the build process tree. The clean-extraction rebuild receipt reports 26.294649 seconds. Both builds succeeded.

The first measured full game took 4.001665 seconds including replay logging. Across the 64 economic games, full-game wall time averaged 4.108423 seconds, maximum 4.666310 seconds. The longest measured policy call was 0.201224 seconds. Raw per-game timing components and memory measurements are preserved; some expansion tests overlapped independent verification/build work within the measured CPU quota.

The root and clean-extraction native libraries were byte-identical: `174ba2ad749cb3213a72a03302ef50b08a37a579848ba922bde132a19b8b6c23`. The native runtime, full source and offline build instructions are included. The clean build deleted the staged binary before recompilation, so it did not merely load a copied library.

## New storage observation, not an economic improvement claim

During exact referee reconstruction of the 64 economic games, the official automatic deposit operation discarded 1,015 item units across 83 of 1,856 own-farm day-boundary deposit events. Counts by item are in `audit/storage_observations.json`; every before/after deposit is in the replay-verification audit folders. The observer calls the original engine function and does not change game behavior. These are actual discarded units, not an estimate of recoverable cash or proof of an optimal storage intervention.

`STRATEGY.md` describes the inherited planning, storage, resource reservations, fertilizer logic and executor. No new economic modification is claimed in this repair.

## Reproduction and package contents

Build from the extracted root:

```bash
python3 -B build.py --cxx g++
```

Use `main.py:agent(observation, configuration)`. `main.create_agent()` also returns a separately closable instance. `OFFLINE_BUILD.md` gives exact validation and replay-verification commands. Runtime dependencies are Python's standard library and the platform native C++ runtime. No external download is required.

New evidence is under `research_snapshot/recovery/`; build, source, input, failure and verification receipts are under `audit/`. The frozen referee is under `referee/`; unchanged published material is under `published_inputs/`. Original historical competitive rows are not counted as new C06 results.

`MANIFEST.json` hashes every final package file except itself and `SHA256SUMS.txt`; the checksum file also hashes the manifest. The supplied final-ZIP inspection receipt checks the actual compressed archive, its member list, every manifest hash, the preserved failed ZIP, the populated experiment inventory and complete replay contents. Its ZIP hash is necessarily external to the ZIP.

## Timing and limitations

Recovery began at 2026-09-13T18:18:55.756514853Z. The original round deadline remained 2026-09-13T18:45:58.765Z. Final packaging and inspection timestamps are recorded in the external delivery receipt. This was a repair within the original round, not a new two-hour research round.

The supplied LocalGame host is not Kaggle's schema validator, timeout sandbox or competitive controller. Source-level observation restrictions, action-shape checks, timing and replay verification do not certify every Kaggle deployment constraint. The smoke seed 291307001 was excluded. Because prior research logs are lost, the new seed list cannot be certified never to have occurred in those missing runs. It is a declared post-recovery development panel, not a sealed holdout.

The reported sample exceeds the local economic target, but performance on the controller's fresh seed panel remains unmeasured. No round-robin, 1,536-game competitive acceptance result, or competitive win rate is claimed.
