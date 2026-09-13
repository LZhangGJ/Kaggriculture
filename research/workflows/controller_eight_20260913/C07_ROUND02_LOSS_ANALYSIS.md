C07 surviving r4 lost200.75 mean cash to the parent on the matched32-game validation panel. Its mean was187473.125; parent187673.875. The initial missing delivery remains a different, unverified result.

Seed 915988743, seat 0: 116,242 cash versus 125,331, a -9,089 change. First action divergence: transition 381. The trace contains 9 mode comparisons and 2 mode changes. The largest end-of-day cash deficit is -9,089 on day 29.

Seed 1160566399, seat 1: 147,879 cash versus 147,879, a +0 change. First action divergence: transition none. The trace contains 13 mode comparisons and 1 mode changes. The largest end-of-day cash deficit is 0 on day 0.

Seed 1196670171, seat 1: 160,411 cash versus 202,559, a -42,148 change. First action divergence: transition 235. The trace contains 12 mode comparisons and 1 mode changes. The largest end-of-day cash deficit is -42,148 on day 29.

Seed 1281132349, seat 0: 172,448 cash versus 196,561, a -24,113 change. First action divergence: transition 306. The trace contains 17 mode comparisons and 3 mode changes. The largest end-of-day cash deficit is -24,113 on day 29.

Next experiments:

- On the two regression seeds, disable only the first differing mode switch, then keep all later decisions unchanged where feasible. Run both seats as development tests; compare whole-game cash, not conditional score. This tests a mechanism without treating the observed gap as recoverable profit.
- Compare current-day realized cash and remaining-game tail separately when ranking admission modes. Test a minimum score margin and a once-per-day switch limit on the complete development panel. Retain all rejected configurations and timings.
- Test r4 against the simpler r1 on identical full-game cells. r1 development gained2001.50, while r4 gained5161.21875 but lost200.75 to parent on validation. The extra WAIT/NOW choice needs a matched validation test, not selection from the already-exposed validation losses.
- Keep rejected cargo-detour variants disabled: r2 and r3 development means were190878.84 and189698.44 versus195548 parent. Do not attribute their losses to a single forecast or assume leftover inventory is saleable profit.
- Freeze the next source before a fresh controller panel. These disclosed loss seeds are now development data; no competitive claim follows from this economic analysis.

Daily asset counts, worker use, pending-admission frames, exact mode comparisons, first differing actions, source hashes and record/replay IDs are in the JSON. These observations do not isolate a cause or establish a recoverable cash gain.

The two largest regressions first switched from mode 4 to WAIT/NOW mode 3 at observations 228 and 300, for conditional score gains of 155.29 and 60.25. Actions diverged at transitions 235 and 306. This is a specific ranking hypothesis to test, not proof that either switch caused the terminal loss.

In the low-cash seed 1160566399, seat 1, one mode change occurred but all actions stayed identical to the parent and terminal cash tied. A mode-change counter alone does not show a behavioral change.
