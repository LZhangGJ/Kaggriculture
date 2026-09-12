# Shared Kaggriculture evaluation sets

These panels give both research tasks a common comparison. Their seed manifests are fixed and reproducible. The initial seed-creation report is [REPORT.md](REPORT.md); the live publication stage is [PUBLICATION_STATUS.json](PUBLICATION_STATUS.json).

| Panel | Size | Use | Games per candidate |
|---|---:|---|---:|
| [Representative](manifests/representative.json) | 256 seeds | Estimate performance on a uniform sample from the eligible default 31-bit seed domain. Use for routine matched development comparisons. | 8,192 |
| [Stress](manifests/stress.json) | 128 seeds | Test selected extremes and contrasting economic conditions. Report separately from representative performance. | 4,096 |
| Holdout | 256 seeds | One final comparison after every candidate and the analysis are frozen. [Checksum commitment](holdout_receipt.json). | 8,192 |

Every panel crosses the same [16 pinned opponents](OPPONENTS.md) and both candidate seats. The separate delayed-seller diagnostic opponent is outside these counts. The holdout values and reproduction key remain local during roster preparation; this branch initially publishes only their checksums. After the frozen comparison, the released holdout seeds and complete results will be added here. Once those results guide development, this holdout is retired from fresh-confirmation use.

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
3. Freeze candidate runtime files, settings, evaluator, opponent versions, failure rules and analysis before any holdout run. Re-audit all known campaign seed stores. Earlier audit coverage and gaps are recorded in [audit/summary.json](audit/summary.json) and [audit_refresh_final.json](audit_refresh_final.json); the second task's separate/remote records are being added to the campaign-wide audit before evaluation.
4. Reset both policies for every game. Keep the hidden seed out of policy inputs. Require all planned unique seed/opponent/seat cases and 719 transitions per terminal game. Record incomplete or failed cases.
5. Keep representative and stress metrics separate. Report wins, draws, losses, per-opponent, per-seat and economic-condition results. Use paired seed-cluster uncertainty, not independent-game intervals. Follow [EVALUATION_CONTRACT.md](EVALUATION_CONTRACT.md).

## Candidate comparison

The user requested all distinct runnable candidates from both active research tasks, including unsuccessful versions, plus original AFS R2. Exact duplicates will share one evaluation only when full runtime/configuration identity establishes that they are the same candidate. Incomplete artifacts and non-deployable diagnostics will be listed with their status, not silently omitted or assigned a fabricated score.

The original [candidates.json](candidates.json) pins the three candidates in the seed-creation task. The expanded evaluation roster will be frozen separately before launch. No full candidate panel has run as of this initial publication. The native 1.32.7 source pinned in [provenance.json](provenance.json) is the intended evaluation engine; the repository root's older 1.32.6 tooling is not interchangeable without parity evidence.

Repeated tuning can overfit these development panels. Sixteen opponent names do not imply sixteen independent strategy families. No fixed panel proves robustness against every opponent. These evaluations provide shared evidence; they do not automatically promote a candidate or authorize a Kaggle submission.
