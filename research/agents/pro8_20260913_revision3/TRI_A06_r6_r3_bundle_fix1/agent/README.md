# TRI_A06_r6_r3_bundle_fix1

## Status and exact lineage
This is an independently named focused repair of the **6cb895 r3 bundle**, not a replacement of b10d or either other same-name delivery. Parent ZIP: `a6737f9f0f4a1d4963a70e5c4beafbf3d02d5d15238f3e2e5bdbd6c0c832f021`. Parent native: `6cb895db1484045699e8b81f11ed1d753507e02d9d57af81d037845dbf73441e`. Candidate native: `6f1ea0c50035b5c33a313cf7e61e2f98f7da826fda52863d67d7c086301250de`. The read-only production parent is under `parent_6cb895/`. Root `main.py:agent`, full production sources, fixed configuration and offline dependencies are included. No opponent code or private opponent state is used.

**No new competitive game was run. No win-rate acceptance or reversal of the R2 regression is claimed.** The complete b10d parent result is 1017/1536, not the old incomplete snapshot. The central evaluator's nine selected bundle games remain development evidence, not a representative acceptance panel.

## One production change
Only `policy/executor/policy.hpp` changes: remove the `day < 15` gate on buying the deficit of fertilizer for tasks that the existing planner has already selected. The requested quantity remains `max(0, need[F] - observed_shed[F])`, original priority 4. Existing value selection, wage/feed priority, order pagination, inventory accounting and cash admission remain intact. This does not force fertilizer usage or buy without a selected need. All other **40 of 41** production build inputs are byte-identical. In particular, `triad.hpp` whole land-plus-project comparison, `search.hpp` land options, `labor_funding.hpp` hiring recovery and `planner.hpp` correct price-floor calculation are untouched.

## Evidence and limits of attribution
Both true principal-case histories were reproduced with their own exact binaries, 719/719 each. Step0 is merely the first order rearrangement: the first substantive observation difference is step193/day8. At step194 both have cash 248 and six workers, but preparation paging and worker locations differ. The new bundle buys its third land on day12 rather than day10 and subsequently chooses different crop and maintenance portfolios. Its labor-recovery offer count is zero throughout this principal trace. The full cash deterioration is not attributed to step0 ordering or excess rescue wages.

At day25/step600 in the true 6cb895 trajectory, the selected maintenance count is **20 fertilizer services** and observed stock is **11**, while the preparation queue buys none because of the date gate. The actual day completes 12 services, including a later one-unit purchase. This is an actual selected-plan/input mismatch, but a substantial cash gap already exists before day25. The fix's first real-prefix action divergence on this case is exactly step600, so **earlier portfolio deterioration is explicitly unresolved**. Saved observations after first divergence are not fed as a candidate future.

## Executed focused validation
- Exact principal parent and bundle native trace reproduction: 1438/1438 calls total.
- New fertilizer order/admission tests: 40 cases, spanning days 0/14/15/25/28, inventory 0/11/20/25, selected need 0/20, zero cash rejection, sufficient cash acceptance and preserved feed priority.
- Nine true bundle prefixes: **4139** actual root calls, **4131** exact prior actions; stop immediately at first different action. Moon290316560 remains 719/719 unchanged. This is not a new win.
- Two independently initialized same-day own-only branches: **46/46** actual root calls match the instrumented production logic and **46/46** official unit/market prefixes. At R2 201542742 day25 the candidate actually hires 12 workers and completes 20 fertilizer actions. At R2 389573676 day9 it actually hires seven. No opponent orders, current shops only, no midnight or unknown future RNG. These are **cold-start execution checks**, not continuations of a full new game and not a measured 12-to-20 causal delta. No predicted cash is reported as recovered revenue.
- Original production parent rebuilt byte-identically. Candidate compiled from current sources, and the actual root loads that compiled library. The production build receipt includes all 41 source hashes.

Detailed outputs: `validation_fix1/focused/SUMMARY.json`, per-prefix gzip files, conditional branch records and raw stdout/stderr/time files. Complete supplied feedback and full denominator remain under `evidence/closed_loop_feedback/`.

## Build and test
Linux x86-64, Python standard library and GCC C++20; no network required.

```sh
python build.py
python tests/build_fix1.py
```

Actual compiler/flags/commands are in `BUILD`, `COMPILER_FLAGS.json`, `policy/a06.BUILD.json` and raw logs. `build.py --unit` is the preserved inherited suite; it was **not rerun** in this deadline-limited supplement. Earlier suite results under inherited evidence are historical, not newly claimed fix1 results. `tests/build_fix1.py` rebuilds and runs the new focused tests.

## Failures and remaining risks
One diagnostic harness initially attempted `json.loads()` on an already decoded debug dict; the failed script and traceback are retained and the corrected replay succeeded. Exploratory `physical_loss_scan` files are raw heuristics, not validated death attribution; none of the conclusions relies on them. Cold branch conditions cannot establish final cash or competitive outcomes. Later fertilizer purchases may compete with other investments and market supply; only the central new 64-seed/12-opponent/dual-seat panel can establish whether strict win rate exceeds 85% (at least 1306/1536). No seed/opponent special casing was introduced. All supplied development seeds are retained.

## Budget
Follow-up first actual resource probe: 2026-09-13T20:32:06.406639420Z. Original start 18:44:08Z and hard deadline 20:44:08Z are unchanged. Effective quota was four CPUs / four GiB, serial compilation, 30% memory headroom maintained per recorded cgroup observations. Production compile took about26.4 seconds /577116 KiB RSS; exact parent rebuild about27.1 seconds /585324 KiB. See resource and time receipts for actual values, not host capacity.
