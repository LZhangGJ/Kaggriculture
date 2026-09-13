# Validation report — TRI_A08_r11_r2_fix1

## Verdict

The source/build repair is implemented, compiled at production `-O3`, and tested
with GCC 14.2 and Clang 17. The originally reported GCC 13.3 environment could not
be reproduced locally. Final root-cause classification (source UB, GCC optimizer,
or another compiler-specific issue) and the GCC 13.3 fixed-build result are open.
This archive is an honest candidate/checkpoint, not an assertion of central acceptance.

## Central failure, not rewritten as a local result

`fix1/central_failure.json` preserves the user's exact original ZIP/native hashes,
`unit_1` failure after 25,859 checks and the O3/O0/sanitized observations. The
original assertion remains in `history/r2/animal_collection_test.cpp`; the complete
parent production source and native are in `reference/r2/`. The current test still
asserts positive-return maintenance; neither the expectation nor its tolerance
has been relaxed. The original ZIP remains byte-identical in the working archive.

## Reduction and bounded source review

`fix1/repro/original_standalone.cpp` removes the game engine and imports only the
DP, numeric constants and rich/poor test. It is a reduction candidate, NOT an
independently reproduced GCC 13 failure. The matrix also runs the untouched full
original unit so that a reduction which loses the trigger cannot replace it.
Boundary-only, argmax-only and combined variants distinguish the two changes.

On each locally available compiler all 15 variants/modes compile and run: original
standalone, boundary-only, argmax-only, combined and original full unit, each in
O3, O0 and O1 ASan/UBSan. This consistency is useful regression evidence but cannot
identify a fault that these compilers never exhibit. The GCC 13 matrix reports
COMPILER_UNAVAILABLE rather than PASS. The attempted apt/HTTPS retrieval failures
are preserved under `fix1/logs/`.

Bounds review: kind 9..11 maps to species 0..2; hunger 0..1; pending bonus is
bounded by held-1 <= 5; only d <= 28 reads d+1; starvation is checked before
indexing; next bonus is capped; terminal tables are aggregate-zero-initialized.
No UB has been established by that bounded review. Sanitizer success does not
prove the absence of every possible UB. The inherited animal_path noipa boundary
is supporting context only, not proof of the cause of this new central symptom.

## Source repair invariant

One immutable candidate index supplies `(feed, care, value)` and the second value
table. All three legal service choices retain the same rewards, collection work,
care-cap restriction, action order and 1e-9 tie rule. A targeted noipa/noinline
boundary stops caller-specialized DP clones while leaving the body at O3.
No prices, strategy thresholds, investment logic, sales-DP branches or seeds were
changed. Production diff: `fix1/PRODUCTION_DIFF.patch`. Test diff: `fix1/TEST_DIFF.patch`.

## Executed checks

| Evidence | Actual result | Scope/limit |
|---|---|---|
| Expanded GCC14 O3 C++ unit | 148,821 checks; 5,120 brute-force cases; 4,608 path cases; 20,496 choice states; 0 repaired-model mismatches | Conditional service/path model, not a match |
| Expanded Clang17 O3 C++ unit | Same counts, PASS | Independent compiler, same platform libraries |
| GCC14 and Clang17 ASan/UBSan units | PASS, same 148,821 checks | Separate O1 evidence, not a substitute for O3 |
| Actual linked native gate, both compilers | 530 constructions; 29,456 validated states; 171,628 successful checks before mutation | Table value, Choice.value, independent argmax, Bellman edge, policy rollout |
| Rich control | value=65, Choice.value=65, feed=1, care=0 | Original positive expectation retained |
| Poor control | value=0, feed=0 | Non-profitable maintenance still stops |
| Deliberately mutated feed bit | Rejected by independent argmax check | Negative test, NOT a compiler reproduction |
| Official execution, both compilers | 4,608 local lifecycles; 16,128 day states; 26,389 effective actions; 90,914 checks | Single own farm; does not prove global scheduling feasibility |
| Root entry contract | 183 checks, PASS | Native/source receipts, config, ABI, reset, two-context isolation and rejected input |
| Original r2 native vs fix1 native | 5,033/5,033 action equality across 7 fixed own-observation traces | No new futures or matches |
| GCC14 fix1 vs Clang17 fix1 native | 5,033/5,033 action equality | Root factory using explicitly recorded alternate Clang native |
| Clean GCC14 build | Identical deployed native SHA256 | Same compiler/platform reproducibility only |
| Expected-failure build promotion | Rejected with positive assertion; previous native and receipt unchanged | Isolated checker mutation; raw failure retained |

The legacy `A08_ANIMAL_COLLECTION_LABOR=0` unit deliberately exhibits the known
r2 pre-fix economic mismatches (3,937 value, 5,236 labor-day, 3,521 stream-value
mismatches). Its PASS means the regression detector found the expected old defect
and action/value consistency held. It is NOT a production configuration and does
not mean those legacy economics are correct. The production feature remains 1.

New C++ checks validate the stored chosen action at every generated state, not
merely the optimal value. The native test is independent Python and reads the
exact deployed C++ table; it additionally rolls out the selected policy to its
terminal state. An action with the right score but wrong feed flag is no longer
allowed to pass. The native library and gate hashes are recorded by the builder.

## Trace and counterfactual limitations

The paired tests really execute both root agents on every one of the seven saved
own-visible observation sequences (719 calls per sequence), with fresh state per
case; full action rows are retained as gzip JSON. They are not a closed-loop game.
After any difference from a source trajectory, its saved future cannot be treated
as the candidate's true future. No recovered cash, new win or loss is inferred.

`market_smart_v8_915042141_seat0` and
`submission_56149565_915042141_seat1` each match original r2 719/719. This preserves
r2 behavior on those observations, not proof of preserving the source r1 +78/+324
wins in a new simulation. r2 was already different from r1; that historical risk
has not disappeared. The original parent panel's 1,054/1,536 is not a fix1 result.

New closed-loop matches = 0. Full new panel = 0. No winner-based acceptance claim.
Only central fresh-64-seed/12-opponent/two-seat play can measure the requested
strict >85% threshold (1,306/1,536 wins). No opponent code/private state was used,
no seed lookup or opponent-name branch was added, and no new game seeds were drawn.

## Raw result index

- `fix1/repro/matrix_gcc14/RESULT.json` and `matrix_clang17/RESULT.json` — 15-mode matrices and per-command output/time logs.
- `fix1/repro/matrix_gcc13_unavailable/RESULT.json` — unavailable compiler, not success.
- `fix1/repro/paired_original_fix1/RESULT.json` — actual original/fix paired calls and per-case action hashes.
- `fix1/repro/paired_gcc14_clang17/RESULT.json` — actual cross-compiler paired calls.
- `fix1/repro/portable_build_refusal/RESULT.json` — native AND receipt preservation under expected failure.
- `fix1/repro/build_transaction.json` — initial independent failure-injection run.
- `policy/tri_a08_r11_r2_fix1.BUILD.json` — final default native and gate identity.
- `fix1/clean_rebuild/policy/tri_a08_r11_r2_fix1.BUILD.json` — independently rebuilt same bytes.
- `fix1/logs/*.command.json`, `.stdout`, `.stderr`, `.time`, `.json` — actual commands, status, wall/RSS.
- `rerun/20260913T165058311690Z/` — final default GCC14 exact `python tests/run_checks.py --sanitize`, all commands PASS; complete outer wall/RSS record in `fix1/logs/final_candidate_exact_checks.*`.
- `rerun/20260913T164157736553Z/` — full Clang17 suite, native gate and default root contract.
- `rerun/20260913T163743144751Z/` — expanded GCC14 suite and all saved observations; inner suite finished, outer timed wrapper was interrupted (see timing limitations).
- `history/r2/` — original claims/manifests, retained as historical evidence, superseded by the explicit limitations above.

## Timing and record limitations

Original start/deadline: 2026-09-13 15:22:07.194 / 17:22:07.194 UTC. Repair resources
were measured at 16:26:28 UTC; repair began at approximately 16:26:13 UTC. The main
source repair was saved at 16:32:10 UTC, before the original 90-minute target.
Budget was not reset. Effective quota: 4 CPUs, cgroup limit 4 GiB, one worker,
30% memory headroom policy. MemAvailable is recorded separately and not treated
as the cgroup allowance. Actual measured compilation peak was below 600 MiB.

Two outer wrapper records are incomplete: attempted apt retrieval and the early
long GCC14 suite. Their raw stdout/stderr/inner results are preserved, rather than
fabricating completion times or peak RSS. Later bounded clean build, compiler
checks, paired calls and final unpacked-package tests provide separate complete
records. Intermediate receipts identify the bytes actually compiled then; they
must not be confused with the final default native. No time estimate is substituted
for a completed experiment.
