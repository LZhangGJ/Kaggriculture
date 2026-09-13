# Failures, rejected hypotheses and limits

## Retained failed checks (not hidden or relabeled as passes)

1. `atomic_market_v1`: the negative fixture used a WHEAT inventory shock -5000
   and asserted that fewer than7 hires must be affordable. Assertion failed:
   an individually unaffordable feed unit simply failed, leaving cash for
   hires. The correct production behavior was not an overdraft. Exact test
   source: `evidence/raw_logs/market_reservation_v1_failed.py`; stderr and exit
   receipt: `atomic_market_v1.*`.
2. `atomic_market_v2`: changing WHEAT shock to-650 still did not make the chosen
   negative fixture underfund wages because its fertilizer sale proceeds
   remained sufficient. That assertion also failed. Exact source/error/exit
   files with `v2` are retained. The final test explicitly also stresses
   FERTILIZER inventory by+600; it actually fills only6 hires with no overdraft.
   The final15-fixture test and32 assertions pass as `atomic_market_v3.*`.

These were errors in constructing a test expectation, not silently changed
production logic. The production source/library was unchanged between all
three fixture attempts.

## Runtime and bookkeeping problems

An initial request for a streaming container session was unsupported
(`StreamingExecNotEnabledContainerError`). Some initial command wrappers timed
out at the tool layer while their children continued and produced real successful
build/probe outputs. Thus `initial_compile.time` and `parent_all_probe.time` are
empty and are NOT used as timing evidence. The full historical parent probe
completed all7 case outputs; its aggregate result also records measured per-case
elapsed times. The parent compilation was repeated under a functioning timed
wrapper; its native remained byte-identical. All subsequent significant jobs
have explicit `.exit.json`, `.time`, stdout and stderr receipts.

Original parent source/native bytes never changed. Python did create two derived
pyc files despite chmod because the interpreter ran as root. This was recorded
in `parent_copy_cache_note.json`; generated caches were removed, subsequent tests
disabled bytecode, and an exact48-member immutable archive was created and
checked. A chmod-only claim of perfect directory write protection is not made.

## Hypotheses investigated, but not shipped as changes

The continuous-crop held-capacity/harvest-loss hypothesis was not supported in
the three highlighted large-loss cases: observed production was harvested and
sold. There were no uncollected units in those crop totals to relabel as
recoverable profit. The supplied ahmed trace does contain terminal leftovers;
that is not evidence that the same issue explains the other cases.

The parent r11 output-rescue mechanism excludes water-only immature crops.
Expanding that mechanism was considered but not implemented: this iteration
repairs the upstream planned wage ordering rather than adding another policy
module or a new hiring/value heuristic.

## What remains unverified

- Zero new complete games were run. No overall win-rate improvement,1306 wins,
  or preservation of narrow wins has been established.
- Both protected historical narrow-win cases differ at their first changed
  action (steps120 and168). Their subsequent saved observations were not used
  as candidate future trajectories.
- The atomic market and fixed-route fixtures are synthetic/component tests,
  not historical counterfactual recovery or actual candidate cash results.
- The route fixture freezes the parent's plan/context and uses the unchanged
  executor; the full candidate can choose/replan differently in a live match.
- Actual feed/input costs can still leave insufficient wage cash. The repair
  preserves an ordering priority, not a guaranteed future revenue source.
- Earlier hires can defer capital purchases to a later market frame and change
  the response to public supply. Profit and win-rate effects need real opponents.
- Dynamic-library compatibility was verified in this environment and with a
  portable x86-64 ISA target, not on every Linux distribution/glibc version.
