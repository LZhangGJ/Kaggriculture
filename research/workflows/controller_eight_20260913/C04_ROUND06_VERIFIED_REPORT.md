# C04 Round06: verified economic revision

C04's frozen `finance3` strategy averaged **204,538.71875 terminal cash** across 16 declared seeds in both seats against legal PASS. It clears the 200,000 cash gate on this author panel, but trails its matched baseline by **8,892.3125**. This result does not establish controller qualification or competitive acceptance. The author kept the frozen revision after validation; no fallback or seed extension occurred.

| Validation measure | Verified result |
|---|---:|
| Candidate games | 32 |
| Candidate mean cash | 204,538.71875 |
| Matched baseline games | 32 |
| Baseline mean cash | 213,431.03125 |
| Candidate minus baseline | −8,892.3125 |

The worst paired validation loss was **97,556** at seed **231131086**, seat **0**: candidate cash **161,937**, baseline cash **259,493**. The development gain did not hold on this validation panel. Complete results preserve the losses as well as the gains.

## Source change

The selected revision starts from C04's verified Round04 source. Only `policy/search.hpp` changes in production; the configuration stays the same. The planner uses a common three-day public rollout when prepared spending and estimated service and hiring costs leave insufficient cash cover. Otherwise, its configured horizon remains one day. The cover calculation counts current cash and modeled proceeds from available shed goods, then subtracts prepared spending and estimated bills. It is a conditional estimate, not a proof that future costs or sales will match the model.

The complete crop and animal planner, route executor, inventory checks and daily scheduling remain present. No new C08 source or configuration component was copied. Earlier C03 calibration and inherited source attribution remain documented.

## Independent controller verification

The controller verified all **eight original archives** by SHA-256, ZIP CRC and safe path checks. The merged evidence contains **2,422 files**, including **2,421 checked inventory entries** and the inventory itself. All **251 supplied input manifest entries** match. The evidence files remained unchanged after the audit.

| Audit scope | Verified count |
|---|---:|
| New game attempts | 344 |
| Complete games | 342 |
| Official interpreter transitions in complete games | 245,898 |
| Interrupted attempts retained as failures | 2 |
| Official interpreter transitions in partial prefixes | 311 + 275 = 586 |

Every complete replay reached 719 transitions, with both players DONE and terminal cash equal to rewards. The controller reran the frozen official interpreter on the recorded actions and compared every frame. The truncated prefixes also match the interpreter, but have no terminal result. Their later completed retries remain separate records.

The controller rebuilt from source with GCC 11.4.0. All **11 supplied API and horizon checks** and **seven controller entry and reset checks** passed. On already used seed 231131086, both seats reproduced every recorded action and terminal cash: **161,937 / 216,257**, in **3.88 / 4.08 seconds** locally. Compilation and unit checks finished before the two-worker replay audit. No new evaluation seeds were drawn during this audit.

## Freeze, seeds and retained limits

The author declared validation seeds at **2026-09-13 22:29:49 UTC**, excluding **1,128 recoverable known IDs**. The source froze at **22:52:20.828521 UTC**, before all 32 candidate validation starts. Source manifests, declared cell coverage and the recomputed mean passed controller checks. The resulting controller inventory preserves **1,144 known seed IDs**.

The author reports an early checkpoint saved within five minutes. The controller did **not retrieve that checkpoint while the author was working**. Final evidence contains checkpoint records, but later retrieval does not establish that the controller held an early backup at the time.

Round02's revised source and complete records remain lost. Its reported failed validation is historical, not reconstructed evidence. Unknown seed usage in that lost history cannot be ruled out. The current audit verifies Round06 independently and preserves this gap.

## Published artifacts and identities

All eight original author archives are available in the existing [campaign release](https://github.com/LZhangGJ/Kaggriculture/releases/tag/controller-eight-round01-20260913): one source/runtime ZIP and seven evidence ZIPs. The evidence ZIPs are ordinary independent archives; extract them into one directory rather than concatenating them.

| Artifact | SHA-256 |
|---|---|
| Source/runtime ZIP, 1,672,490 bytes, 79 files | `7fef72d6d82da8f8b535c09b7a201107a2cb2b84106c4c2debaf93ff4835b450` |
| Author native library | `840c2723ba54d45b9c7d522867fd0dc2eb5d503fb40cf2becf4d12828cde89b2` |
| Controller GCC11 native library | `e23a12d42f2f499f06e14042fefd1481d21381c00af6a6e996a49c2e9a252c2c` |

`C04_ROUND06_IMPORT_VERIFICATION.json` contains the complete controller receipt, runtime and referee hash maps, validation chronology, partial-attempt records, test results and local action reproduction. The full author report and result files preserve development settings, failures, retries and source variants. Competitive acceptance still requires the separate fixed-pool evaluation; this report makes no competitive win-rate claim.
