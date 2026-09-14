# Raw evidence index

Final native: `49c98f2c1930cf741bdd9e2eb7da9af398117bb64f903f9375c7da9350e8fa64`.
Current actual-native/official checks: `raw/atomic_checks_gcc/`,
`raw/atomic_checks_clang/`. Current bounded branches: root
`raw/branch_*.json.gz`, `raw/own_branch_summary_*.json`, `raw/atomic_branches*`.
Portable repetitions are in `raw/portable_branches*/`; summaries compare equal
in `raw/portable_reproduction.json`. These do not add new independent cases.

Current full own-observation probes: `raw/seedguard_final_*`,
`raw/seedguard_clang_*`, `raw/seedguard_ablation_*`; actual default-root
outputs: `raw/root_default_*` and `raw/root_entry_check.json`.

The earlier `candidate_v1`, `candidate_final`, non-seedguard `clang`/`ablation`
and numbered candidate-build files are development checkpoints, not current
identity. `seedguard_checks_*` official execution outputs and early branches
were superseded by the atomic-PLANT harness correction; do not count them as
independent validating observations. Their actual-native animal gate and C++
property results are unaffected by that Python harness defect.

Absolute paths in command logs are the actual original work paths. They are
not required by the root offline builder/tests. `development/tools_original`
retains original investigation scripts, including work-path assumptions.
Use root `tests/run_checks.py`, `tests/probe_observations.py` and
`tests/own_branch_checks.py` for portable reproduction. The final ZIP
post-unpack check is reported outside the immutable ZIP.
