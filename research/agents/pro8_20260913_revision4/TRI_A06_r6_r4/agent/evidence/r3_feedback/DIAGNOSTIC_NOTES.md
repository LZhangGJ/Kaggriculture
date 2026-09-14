# A06 r6: evidence for a further competitive revision

The tested parent is `TRI_A06_r6_r2` with native SHA-256 `b10d86c102f68e5863559e6c049369571cbc123aa4db71db908a10c1ab14eab1`. It is the search.hpp revision that retains original expansion proposals and adds matched no-new-land alternatives when all original proposals expand. The chat contains another ZIP with the same filename but native hash `79a769f34ee405a7ab3f6542c22eb6af59f62a76bff76b32074b0573d77cab73`; the current results do not evaluate that other binary. Use the supplied exact parent source package for this task.

Five complete games were selected from the first 16 frozen seeds: three losses and two narrow wins. Codex re-executed all 3,595 official transitions and reconciled 300 player-days of cash. The source full-game audits stay local; this package contains only own-visible observations, own actions, own ledgers, public match outcomes and public rules. All examples are development data.

| Case | Margin | Purpose |
|---|---:|---|
| thomas_955_v2_1612180274_seat0 | -11,217 | Large public loss, zero physical inventory loss |
| market_smart_v8_389573676_seat0 | -8,582 | Large public loss, zero physical inventory loss |
| submission_56149565_389573676_seat0 | -17,972 | Worst R2 loss in the first 16 seeds; 22 early strawberry deaths |
| moon_v215_290316560_seat0 | +16 | Narrow public win to protect |
| submission_56149565_201542742_seat0 | +1,003 | Narrow R2 win to protect |

## Confirmed funding and maintenance failure in the R2 loss

The R2 case lost 16 strawberry plants at step239 (zero-based day9), and six more at step335 (day13). They had not produced their first yield. All died after a second unwatered day; the official crop audit has zero state mismatches.

On day9, step216, the strategy issued sales, wheat purchase, land purchase, three cows, crop seeds and two HIRE orders. Cash fell from445 to1 and only two hands were present. At step217 it submitted five more HIRE orders, but hand count stayed two. At step218 it sold four milk and cash rose to644; no subsequent HIRE order was issued that day. Sixteen young strawberries died at the end of the day.

On day13, step312, it issued sales, wheat purchase, land purchase, seeds and four HIRE orders. It ended with16 cash and four hands. At step313, four more HIRE orders produced only two additional hands, leaving3 cash. At step314, further sales raised cash to2,490, but there was no further HIRE order that day. Six young strawberries died that night.

The exact own-visible sequence is supplied in `analysis/death_preparation.json`. Investigate whether planned labor and existing maintenance commitments remain affordable through the real sale/purchase order sequence, including the ten-order boundary and later sale receipts. A no-new-land option alone does not solve this execution failure when the selected investment plan still leaves promised hires unfunded. Preserve profitable investment and animal feeding; do not solve this by globally banning land or indiscriminately adding workers. Validate a coherent change to funding, admission or recovery on the real executor, with positive and negative cases. The observed deaths and later cash do not by themselves quantify recoverable terminal cash.

## Competitive weaknesses beyond that failure

The two large public losses have no physical stock loss, so warehouse overflow cannot explain them. Against Thomas, 302 milk realized34,376 and340 strawberries realized47,131;39 strawberries sold at the floor. Against market_smart,263 milk realized23,729 and298 strawberries realized28,851;53 strawberries sold at the floor. Both spent7,000 on land. This does not prove expansion is harmful or every floor sale avoidable. Check whether marginal supply and actual selling opportunities justify the admitted portfolio and its upkeep, and whether the planner and executor agree about realizable production and cash.

The R2 loss discarded only8 stock units, while272 harvested strawberries were all sold. Its drought failure is distinct from the major zero-overflow public defeats. Prioritize a substantial change supported by the data and source; document which weaknesses remain uncovered rather than claiming one patch fixes the entire pool.

## Validation and acceptance

The current frozen panel continues to completion. `PROGRESS.json` and `PARTIAL_ROWS.json` are a timestamped incomplete snapshot; they are not the final overall win rate. This revision round starts early only after the accumulated nonwins make1,306 wins in1,536 games mathematically impossible. Complete results will be supplied separately when ready, without extending the new author's two-hour deadline.

Preserve the parent floor-price correction and useful land alternatives. Test the actual packaged native and entry point. After the first changed action, old saved future observations are not the candidate's real trajectory; observation replay cannot establish a new win rate or counterfactual cash. Use public rules and bounded focused validation within the measured resource budget. Codex runs the actual full12-opponent panel on fresh shared64 seeds after freezing the delivered revision. Acceptance remains strictly more than85% wins, at least1,306/1,536; cash is diagnostic only.
