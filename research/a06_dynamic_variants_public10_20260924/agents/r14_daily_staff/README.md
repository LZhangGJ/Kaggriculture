# A06 R14-DailyStaff

Based on the supplied A06 R12 r4 and the preceding R13 terminal-staffing candidate.
This is a rule-only experimental candidate, not a claim of reaching 80% or 90%.
See the parent package REPORT_ZH.md and results for the measured panel.

Entry point: `main.agent(observation, configuration)`; stdlib Python + native Linux x86-64 library.
`main.create_agent()` creates an independent context; call `.close()` after use.

Rebuild offline with a C++20 compiler:

```sh
python build.py
python tests/run_units.py
```

The included binary is a Linux x86-64 .so, not a Windows DLL.
No network, training weights, opponent program, seed lookup, or replay tape is used.

New enabled setting: `r14_staff=1`. On day 3..28 at hour zero, if the
selected baseline calls for at least 8 hires, compare that count with up to
4 fewer (never below 6). Preserve non-hire orders. Simulate one day with the
existing controller and public-flow assumptions. Reject candidates losing
projected surviving assets or worsening their water/feed deficit. Select
using conditional cash plus continuation of already existing assets.
These forecast constraints do NOT guarantee improved actual episode results.

The R13 terminal staffing option stays enabled. Setting `r14_staff=0`
disables this iteration's policy change. The complete local evaluation uses
the supplied official Python rules, but is not a full Kaggle SDK sandbox
resource/timeout acceptance test.

## Mechanism boundary

The implementation also sets the executor's **day capacity cap** (`core.p.max_hands`)
to the selected count. It reconstructs the market preparation queue by preserving
the relative order of non-HIRE orders and appending HIRE orders. This can move HIRE
orders that were interleaved with purchases by the parent's wage-reservation path.
The original-count forecast uses the same normalization/cap mechanism; it is not
an untouched R13 no-op candidate. Asset protection is relative to that conditional
reference, not a proof of preserving every asset or outcome of the unmodified parent.
A setting of `r14_staff_range=0` isolates cap/order normalization without selecting
a smaller count. The final provided setting is 4 and was not changed using the new-seed audit.

Native binary requirements observed locally include GLIBC 2.32 and GLIBCXX 3.4.31.
The binary is not guaranteed to load on every Linux image. Rebuild in the target
environment. Local tests recorded single calls above one second and did not enforce
the Kaggle SDK time/overage budget.
