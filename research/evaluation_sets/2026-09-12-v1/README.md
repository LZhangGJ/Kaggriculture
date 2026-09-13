# Shared Kaggriculture evaluation sets

The **stress panel is complete**: [overall results](stress_results/RESULTS.md), [win rates by opponent](stress_results/BY_OPPONENT.md), and [opponent-by-seat tables](stress_results/BY_OPPONENT_AND_SEAT.md). All 24,576 stress games passed validation. Stress measures selected extremes and must be reported separately from representative performance. Holdout results are pending.

The **representative panel is complete**: [overall results](representative_results/RESULTS.md), [win rates by opponent](representative_results/BY_OPPONENT.md), and [opponent-by-seat tables](representative_results/BY_OPPONENT_AND_SEAT.md). All 49,152 representative games passed validation. The stress panel is also complete; holdout is still running with all six policies frozen.

These panels give both research tasks a common comparison. Their seed manifests are fixed and reproducible. The initial seed-creation report is [REPORT.md](REPORT.md); the live publication stage is [PUBLICATION_STATUS.json](PUBLICATION_STATUS.json).

| Panel | Size | Use | Games per candidate |
|---|---:|---|---:|
| [Representative](manifests/representative.json) | 256 seeds | Estimate performance on a uniform sample from the eligible default 31-bit seed domain. Use for routine matched development comparisons. | 8,192 |
| [Stress](manifests/stress.json) | 128 seeds | Test selected extremes and contrasting economic conditions. Report separately from representative performance. | 4,096 |
| Holdout | 256 seeds | One final comparison after every candidate and the analysis are frozen. [Checksum commitment](holdout_receipt.json). | 8,192 |

Every panel crosses the same [16 pinned opponents](OPPONENTS.md) and both candidate seats. The separate delayed-seller diagnostic opponent is outside these counts. The holdout values and reproduction key remain local until the frozen comparison finishes; the branch publishes their checksums and release receipt. After the frozen comparison, the released holdout seeds and complete results will be added here. Once those results guide development, this holdout is retired from fresh-confirmation use.

## What the stress labels mean

A seed does not fix a shop or price path regardless of actions. The official engine uses the same daily RNG for weeds and shop choices, so changing the number of empty tiles can change the shop drawn. Stress selection uses 13 potential-randomness dimensions and 13 demand dimensions under a fixed, legal PASS/PASS reference controller. The reference labels are conditional on that controller. They are not guaranteed economic conditions in candidate games. Actual prices, supply gluts and sales depend on both players' actions.

The 128 stress seeds cover both extremes of 26 dimensions within a separately drawn 4,096-seed pool, all eight reference first-shop types, and 12 contrasting cells. Selection used no candidate wins, losses, cash or policy trajectories. See [FEATURE_DEFINITIONS.md](FEATURE_DEFINITIONS.md), [selection_protocol.json](selection_protocol.json), and [selection_result.json](selection_result.json).

## Validate and use

From this directory, using Python 3.11 or later:

```sh
python -B tools/check_bundle.py
python -B tools/validate_public.py
python -B tools/validate_public.py --reproduce
```

The last command recomputes all 4,352 development feature rows. It does not open the holdout or run agents. The original creation tools are retained for provenance; commands that require `sealed/` are for the local custodian and will not work in this pre-release checkout. Original `validation.json` and `reproducibility.json` are seed-creation receipts, not candidate evaluation results.

1. Use the exact manifest lists; do not replace them with a consecutive seed range. Existing `evaluate.py` CLI support for `--seed-start` is insufficient for these panels.
2. Reserve representative, stress, and the entire stress selection pool from training-data generation. Candidate development may inspect representative/stress results, but do not call them untouched confirmation afterward.
3. Freeze candidate runtime files, settings, evaluator, opponent versions, failure rules and analysis before any holdout run. Re-audit all known campaign seed stores. Earlier audit coverage and gaps are recorded in [audit/summary.json](audit/summary.json) and [audit_refresh_final.json](audit_refresh_final.json); the broader [campaign audit](campaign_audit/report.json) records both tasks' local and remote coverage before evaluation.
4. Reset both policies for every game. Keep the hidden seed out of policy inputs. Require all planned unique seed/opponent/seat cases and 719 transitions per terminal game. Record incomplete or failed cases.
5. Keep representative and stress metrics separate. Report wins, draws, losses, per-opponent, per-seat and economic-condition results. Use paired seed-cluster uncertainty, not independent-game intervals. Follow [EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md).

## Candidate comparison

The user selected promising candidates only. The frozen six-entry [roster](evaluation/roster.json) contains Day9, connected Day3 v1, connected Day3 v2, terminal suffix, v37 feed reserve, and original AFS R2. The two research tasks nominated them from earlier evidence, before any new evaluation result. Each receives all three panels, for 122,880 primary games. The v37 public-parent policy is a comparison arm; it does not establish the team's Three-Layer architecture goal.

The [analysis plan](evaluation/ANALYSIS_PLAN.md), [freeze receipt](evaluation/FREEZE.json), complete pinned runtime and manifest-aware runner are published before preflight. The [campaign audit](campaign_audit/report.json) adds both local research trees, remote seed registries, the replay archive index and NPZ seed arrays. It finds no overlap with any new set. Prior metrics in handoffs explain candidate selection; they are not scores on these panels.

The original [candidates.json](candidates.json) records the seed-creation task. The native 1.32.7 source pinned in [provenance.json](provenance.json) is the evaluation engine; the repository root's older 1.32.6 tooling is not interchangeable without parity evidence. The [repeated preflight](evaluation/verification/preflight-v2/STATUS.json) passed all 192 cases (512 full games and 276,480 native/official observation checks). A [documented optional-debug repair](evaluation/RUNNER_REPAIRS.json) preserves the failed first preflight. Representative and stress results are complete and published above. Holdout is running. Each panel retains separate opponent and seat tables.

Repeated tuning can overfit these development panels. Sixteen opponent names do not imply sixteen independent strategy families. No fixed panel proves robustness against every opponent. These evaluations provide shared evidence; they do not automatically promote a candidate or authorize a Kaggle submission.
