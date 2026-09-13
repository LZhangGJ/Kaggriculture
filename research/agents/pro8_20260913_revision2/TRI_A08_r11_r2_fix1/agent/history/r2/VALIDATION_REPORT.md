# Validation report — TRI_A08_r11_r2

The complete English narrative and scope limitations are in `README.md`.
The artifact is a source-tested candidate, not an accepted >85% strategy.

## Result index

- `analysis/parent_result_and_ledger_audit.json`: all 1,536 parent rows, all 210 own cash days.
- `analysis/hypothesis_audit.json`: rejected crop-service mismatch; observed collection work omissions.
- `analysis/observed_labor_example.json`: exact sheep example and actual own actions.
- `analysis/entry_probe_summary.json`: all 7 saved root probes, first divergences, full denominators.
- `probes/official_execution.json`: frozen official rule/physical execution cross-checks.
- `logs/unit_1.run.stdout`: corrected exhaustive-model checks.
- `logs/unit_0.run.stdout`: expected legacy negative-control discrepancies.
- `logs/unit_1_ubsan.run.stdout`: standalone UBSan result.
- `rerun/*/unit_sanitize.stdout`: combined ASan/UBSan result.
- `logs/entry_contract.stdout`: root entry/library/ABI/metadata/context checks.
- `probes/r2_clean_build_comparison.json`: independent default build byte identity.
- `probes/diagnostic_reproduction.json`: parent diagnostic reproduced from frozen source.
- `logs/parent_build_interrupted.txt`, `logs/clean_build_wrapper_incomplete.txt`: interrupted wrappers.
- `logs/audit_thomas_repeat.*`: completed replacement for the interrupted final diagnostic wrapper.

No new game, victory, recovered cash, candidate future ledger or preserved narrow
win is claimed by any of these tests. Recorded full root actions after a first
divergence are compatibility checks only. The two narrow wins remain unresolved
central-test gates; see their first changed steps 96 and 192 in the summary.

## Archive relocation check

A preflight ZIP was extracted to a new directory. Every packaged checksum, the
183-check actual root/library contract, and48 saved-observation calls through
that extracted `main.py` passed. Exact logs are `logs/zip_preflight_*` and
`probes/zip_preflight_probe.json`. Production hashes are unchanged afterward.
The final archive itself is additionally CRC/hash/entry verified; its external
delivery receipt records the archive SHA256 without a circular self-hash.
