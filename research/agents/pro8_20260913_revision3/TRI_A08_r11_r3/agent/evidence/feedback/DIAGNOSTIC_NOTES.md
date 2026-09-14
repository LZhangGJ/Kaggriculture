# Provisional A08 evidence for the next author round

These are two selected completed losses within the first 16 seeds of the frozen round-three panel. They are development examples, not a random sample or a complete win-rate report. This analysis adds zero games. The complete panel continues; this author round may begin once PROGRESS.json proves mathematical failure.

The exact frozen parent is `TRI_A08_r11_r2_fix1`, native SHA-256 `91828b8a9f6882030ba5247a8e40f15d788a9219fe64667502ebd6a5d211beed`. Full saved-action audits reconciled all 719 official transitions and both players' 30 daily cash balances for each case. Only this agent's observations, actions and ledger are in this directory. Joint audits remain local with Codex.

| Case | Own final cash | Margin | Physical stock discarded | Strawberry produced / harvested / sold | Tomato produced / harvested / sold |
|---|---:|---:|---:|---:|---:|
| aurax_reactive_v1_1801029924_seat0 | 108,319 | -10,695 | 75 units | 270 / 270 / 264 | 110 / 110 / 110 |
| submission_56149565_1782098909_seat1 | 74,359 | -7,910 | 23 units | 229 / 229 / 227 | 102 / 102 / 100 |

Neither case has an ongoing-crop drought death, lost standing yield from digging, or crop yield-capacity loss. All produced strawberry and tomato units were harvested. Terminal stock was empty. Production survival alone does not explain these defeats.

## Confirmed warehouse mechanism

Every discarded unit in these two cases was lost during automatic end-of-day deposit. The shed was empty immediately before each deposit; aggregate worker cargo exceeded the 100-unit capacity. Explicit daytime DROP caused no loss. Full official re-execution of the seven affected days verified this event sequence and exactly matched the full cash auditor's stock-loss totals.

| Case | Zero-based day | Step | Cargo before automatic deposit | Discarded |
|---|---:|---:|---:|---:|
| aurax_reactive | 19 | 479 | 115 | 15 |
| aurax_reactive | 21 | 527 | 107 | 7 |
| aurax_reactive | 27 | 671 | 120 | 20 |
| aurax_reactive | 28 | 695 | 133 | 33 |
| R2 | 15 | 383 | 105 | 5 |
| R2 | 25 | 623 | 109 | 9 |
| R2 | 28 | 695 | 109 | 9 |

Read-only telemetry from the exact production native reproduced all 1,438 recorded parent actions. Midroute delivery ran 562 checks in each game. It inserted two deliveries totaling three units against aurax_reactive and none against R2. Thus the mechanism is enabled but does not prevent the observed overload.

Investigate the route and deposit forecast before it becomes too late to return: `policy/executor/policy.hpp`, `expected_auto_deposit()` and `dispatch_midroute()`. The current source rejects a route if it already contains a remaining DROP or if carried items are also needed by future tasks. It requires every remaining task to fit after the detour. These are source-level investigation targets; current telemetry does not prove which rejection condition blocked each worker. Preserve feed/fertilizer commitments and production deadlines while checking actual feasibility of earlier deposits and market clearing. Do not simply turn on a feature already enabled, remove productive tasks indiscriminately, or assign discarded units their displayed price as guaranteed recoverable cash.

The R2 loss also sold 7 wool, 28 melon and 9 strawberry units at the price floor. Those are actual fills, not failed orders. Investigate supply choices and realized selling opportunities alongside delivery; the audit does not prove every floor sale could profitably have been delayed. A repair must improve competitive outcomes, rather than only an internal score or one inventory metric.

Retain the exact native service-DP build gate from fix1. Make no seed-specific rules. Codex will validate a delivered revision against the complete fixed pool on fresh seeds; acceptance remains strict overall wins above 85%, with cash diagnostic only.
