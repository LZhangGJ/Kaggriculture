# V2 verification

The complete preflight passed: **1,008 cases, 2,688 full games and 1,451,520 native/official observation comparisons**. It covers six candidates, four public development seeds, 16 primary opponents, five diagnostic opponents and both seats. There were no missing or invalid cases. All 23 regression tests passed.

The original published bundle was unchanged. A read-only check also matched its live runner, analysis, roster, opponent/seed manifests and all 102 runtime hashes. No holdout values were inspected. No primary candidate comparison or submission was launched.

[REPORT.json](REPORT.json) contains counts, hashes and local policy-latency diagnostics. [inputs/FREEZE.json](inputs/FREEZE.json) pins the exact inputs. The compressed [preflight journal](preflight/rows.jsonl.gz) expands to the SHA256 recorded in the receipt. For local replay of receipt checks, decompress it as `rows.jsonl` beside the preflight JSON files.

An earlier attempt recorded 56 valid cases before it was interrupted to remove an unfrozen future-holdout feature-cache path. Its partial records remain in `superseded/`; they add no weight to the passing result. The repaired implementation passed a new, complete preflight.

This verifies the tested execution paths. It does not measure agent strength, prove generalization to new opponent families, or certify Kaggle's packaged runtime limits.
