# TRI_A06_r12_r1

**Status: built, locally checked candidate; awaiting central win-rate acceptance.**
This is a single structural correction to the supplied **A06_r12 harvest-calendar
lineage**, not a replacement policy, a merged A06_r6, or an accepted >85% agent.
Use the root `main.py:agent`. It loads the included `policy/a06.so`; it does not
read the evidence, historical replays, test seeds, or reference opponent.

## Identity and scope

The input archive was `A06_r12_source_and_losses.zip`, SHA256
`caf5a9adb0eb4c0158bf1d69483808cad04fc3c1c97b84b5986072ed9d2612ea`.
The stated parent source is repository `LZhangGJ/Kaggriculture`, branch
`research/pro8-a06-a08-20260913`, commit
`d65325334067cc6c2b616f346bc43231b1c11190`. The supplied source and native were
used locally; no external strategy was incorporated and no repository was changed.

| Identity | SHA256 |
|---|---|
| Parent native | `1400cdd7ef2677dc16b9c2b46a34a7ad41419d038c6e9bf5ebf60137707add3a` |
| Delivered / final-tested native | `d642c5650c916365136bfc5ce5c47f82240c27dc34d9baf691a580376b56ec0b` |
| Unchanged production configuration | `ae611b3a052fae69ef971335583c5bacf9a9b6d08efd5bf43eed33262e2c5016` |

`SOURCE_FREEZE.json` and `BUILD.json` enumerate all 47 production/build inputs.
`PROVENANCE.json` records the lineage and changed files. `MANIFEST.json` hashes
the complete delivery payload, excluding the manifest itself. Original parent
receipts are historical evidence, not receipts for the new native.

## One structural cause: terminal calendar/executor disagreement

All day numbers below are **zero-based** (the final day is 29). The r12 calendar
can choose a finite crop with a shorter, still productive terminal maturity.
However, its executor previously required the *preferred* full harvest age when
compiling a PLANT action: four days for wheat or three for carrot. The intraday
admission stage also rejected the same project when a legacy repeated-cycle
output proxy was zero. A calendar-approved, productive final cycle could thus
be valued yet not be admitted or compiled.

The included regression reproduces three cases in the original source:
wheat planted on day 26 with a day-29 harvest, wheat planted on day 27 with a
day-29 harvest, and carrot planted on day 27 with a day-29 harvest. The calendar
returns positive conditional outputs (3, 2 and 2 units in this fixture), but
original compilation emits no PLANT. Those are **conditional model outputs**,
not earned cash or a recoverable-loss estimate.

The production correction has a narrow scope:

* `policy/executor/policy.hpp` uses legal finite maturity for planting-window
  feasibility when the finite r12 calendar is active. It does not overwrite the
  incumbent crop's selected harvest age or force an earliest harvest.
* `policy/executor/intraday_admission.hpp` no longer lets the zero legacy output
  proxy veto a finite, legal-window proposal with a finite positive **fully
  shared calendar value**. Cash, actual seeds, purchase receipt confirmation,
  paid/idle worker availability, route and water deadlines, reservations and
  market-order caps still apply. No projected quantity or cash is credited.
* `policy/triad.hpp` activates the behavior only for r12 calendar on and repeat
  off. The existing offline `td_r12_enable` bridge control synchronizes that
  flag for consistent ablations. The root does not call this control.
  `main.py` has only an identity/docstring change.

Calendar stopping/portfolio DP, current crop harvest decisions, paid successor
protection, finite fertilizer, ongoing-crop and animal management, positive-value
investment selection, public supply modeling, market execution, fleet packing,
rolling scheduling, labor recovery, cash receipts and sale guards remain in the
source. The configuration and compiler flags are unchanged. The complete audit
diff is `evidence/PRODUCTION_DIFF.patch`; this delivery also includes all sources.

The mismatch is reproduced, but it is **not proven to explain the entire historical
349-versus-374 win gap**. No alternative module was retuned to chase that gap.

## What was actually validated

| Evidence class | Result | Interpretation |
|---|---|---|
| All supplied historical rows | 349/480 wins; public 315/440; original R2 34/40 | Historical development data only; not new wins |
| Original action reproduction | 5,033/5,033 actions match across all seven histories | Parent identity and legal-observation reproduction; zero new games |
| New terminal regression | 317 positive/negative checks pass | Original source rejects the three positive cases; new source accepts them |
| Current unit suites | 7/7 pass | Six retained suites plus the new terminal regression |
| Final-native common-prefix checks | 4,459 identical actions; 4,466 comparisons | Each of seven cases stops at its first difference; no candidate suffix outcome |
| Final-native real-policy panel | 5 strict wins, 0 draws, 3 losses in 8 games against the exact parent | One opponent, four development seeds, both seats; not the central pool |
| Parent-parent controls | 4 complete games | Eight paired seat outcomes: 3 wins, 2 draws, 3 losses |
| Final panel integrity | 12/12 games reach 719 steps and 60 player-day cash records, zero runtime exceptions | Official interpreter transitions, not the full Kaggle sandbox |
| Root interface/reset checks | 80/80 comparisons pass | Both seats, new step-zero games and explicit reset |
| Clean delivery-directory rebuild | Same native bytes as the final tested library | 47 production/build-input hashes also match |

The retained unit suites exercise fleet packing (1,992 tour/subset comparisons
and 160 warehouse subsets), pending feed/actual receipts, execution-plan and
paid-commitment preservation (14 candidates and 168 known-day transitions),
short-sale DP (2,520 cases), labor recovery (80 exhaustive cases / 16,700 route
enumerations), and the r12 calendar (2,000 DAG cases / 16,000 labels and 12
exhaustive portfolio cases). Current commands and outputs are in
`tests/UNIT_RESULTS.json`; inherited results are separately labeled historical.
The new tests include calendar-off/repeat-on, all day/crop boundaries, incumbent
versus successor ages, deferred planting, zero cash, missing purchase receipts,
paid seeds, non-positive/absent shared values, busy workers, late hours and order
capacity. Synthetic fixtures are not claimed to be realized match profits.

The closed-loop development seeds are 2613091201–2613091204 and are used only by
the test driver, never by the policy. Both real agents act on fresh legal current
observations at every tick. There is no PASS opponent and no saved-action opponent
in these complete games. Paired with parent-parent control seats, none of three
old winning seats became a loss; two draw seats became wins. There was nevertheless
one margin regression of 189 (the seat remained a win), and three candidate losses.
Average own-cash change was +1,257.875 and average margin change +1,061.5; these
small-panel diagnostics are not acceptance metrics. No production change was
made in response to these final-panel outcomes.

The final panel contains 12 complete games. An earlier, explicitly labeled pilot
used a preliminary native, so **13 real-policy full games in total** were run
in this task, not 13 games of the final candidate. The preliminary pilot is excluded
from the 5/8 result. The first seed's pilot preceded the panel declaration;
this is development evidence, not a sealed holdout or preregistered experiment.

## Historical prefixes and narrow-win caution

For the final candidate, first differences occur at steps 624, 643 or 648
(days 26–27). The supplied official interpreter exactly reconstructs the common
prefix. A single same-tick probe is allowed with the other player's recorded
simultaneous action, because its observation is unchanged at that tick. No saved
future is used after that difference. The raw probes include legal observations,
action/state hashes and the resulting immediate cash, seeds and hands.

**Both supplied narrow wins (+7 and +363 historically) also diverge. Their new
terminal outcomes have not been verified.** Prefix equality is not proof that
those wins are protected. The actual named public/R2 opponent implementations
were not present in the supplied evidence bundle; no substitute was silently
labeled as those opponents. `evidence/logs/prefix_final/` holds the final-native
prefix probes, and `tests/reference_bundle/` preserves all seven audits/replays
and the complete 480-row historical set.

## Offline use and reproduction

Python's standard library and a Linux x86-64 C++20 compiler are sufficient.
The shipped ELF was built with GCC 14.2.0 and dynamically links the host C/C++
runtimes; inspect `evidence/native_linux_dependencies.txt` for ABI requirements.
A different platform/runtime may require a local rebuild. Bit identity across
unrelated compiler/runtime versions is not asserted.

```bash
# Start in a freshly extracted archive (main.py is at the root).
python -B tests/verify_bundle.py
python -B tests/test_entrypoint.py \
  --referee tests/reference_bundle/referee --out /tmp/a06_entrypoint.json

# Rebuild in a working copy. Rebuilding/running units updates build receipts/logs.
python -B build.py --cxx g++
python -B tests/run_units.py --cxx g++

# Focused terminal unit only; assertions must remain enabled for unit tests.
python -B tests/run_units.py --cxx g++ --only test_terminal_calendar

# One genuine complete game, not a replay suffix or a PASS baseline.
python -B tests/closed_loop.py --seed 2613091202 --seat 0 \
  --variant candidate --parent tests/reference_parent \
  --referee tests/reference_bundle/referee --out /tmp/a06_closed_loop

# A bounded, first-divergence historical prefix probe.
python -B tests/replay_prefix_probe.py --bundle tests/reference_bundle \
  --parent tests/reference_parent \
  --case aurax_reactive_v1_4145681841_seat1 --out /tmp/a06_prefix
```

Verify the frozen delivery before rebuilding: rebuilding intentionally modifies
logs/receipts, so a full-payload manifest comparison afterward is not a test of
whether the delivered archive was intact. `BUILD.md` gives the actual build
command, compiler, flags, timing and native hash. `evidence/EXPERIMENT_INDEX.md`
separates preliminary, final, expected-failure and historical evidence.

## Resource discipline and remaining work

Resource capture started at 02:20:34 UTC, within five minutes of the supplied
02:18:58.681 UTC dispatch. Affinity allowed five logical CPUs, but cgroup quota
was four CPU equivalents; memory limit was 4 GiB. Work was serial, reserving
at least 30% of the memory quota. A bounded short compile and 48 legal-observation
calls preceded production compilation and the small game panel. The final clean
build took 26.85 seconds wall time with 608,024 KiB peak RSS. The 12 final-panel
games used 106.090 seconds of game-driver wall time in aggregate. Full raw resource
records and the task timeline are included; host MemAvailable was not treated as
an allocation.

The central 64-seed / 12-opponent / both-seat **1,536-game** test was **not run
here**, and no >85% overall result is claimed. Acceptance requires at least
1,306 strict wins under the user's central protocol, with the updated frozen
identity/seeds. That pool is mixed development data, not a sealed holdout.
True closed-loop protection of the two supplied narrow wins also remains open.
The official lightweight host does not implement the full Kaggle sandbox,
JSON-schema validation or time enforcement. Zero runtime exceptions does not
mean that every submitted action filled or succeeded. Future crop events,
transport/labor demands and public rival supply remain modeled uncertainty;
a positive calendar value is not guaranteed realizable cash.
