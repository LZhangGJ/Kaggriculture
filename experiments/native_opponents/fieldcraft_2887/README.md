# Native Fieldcraft 2887 opponent

Offline-only C++ port of the Apache-2.0 Fieldcraft R1 release. The frozen
distribution is pinned to:

- `main.py`: `7fab02c889fde933e642ee9ceb35dd56ae9a0a073fffc0f6d7a689128367651d`
- `mirror_plan.py`: `20d79c8435814979ea6924e09026f3f83c5c646a331e2ce3bd940de426c1dbc4`
- `submission.tar.gz`: `99d277ac82fa43e29adb56db951a914ef20fb39758d245a9410ef563f147e8b8`

The binary asset holds the 719-frame base tape, all 20 routed tapes, and the
64 first-two-shop routing entries. Runtime code never imports Python or parses
JSON. Dynamic Fieldcraft layers remain C++ state machines and are not replaced
by the generic replay shell.

Implemented active semantics include route/substitution, bounded sale lead,
D29, SETomato7, InputHand beam planning, SheepPen lean/endgame, feed economics,
grain outlet, and final row reranking. Sparse null tape rows are discarded by
the asset generator exactly where the Python `plan()` discards them; preserving
them as `PASS` rows would silently change downstream ten-row feasibility.

```bash
/root/miniforge3/envs/torch-npu/bin/python generate_assets.py
bash build.sh
```

Standalone parity gates passed before any JobBatch wiring:

- `parity_archive_report.json`: 40 complete historical games from two
  non-overlapping seed blocks, all 20 officially reachable routing outcomes,
  and 28,760 / 28,760 full `PlayerAction` frames exact. Historical files only
  contain Fieldcraft as seat 1; these are active games against the deployed
  dynamic policy and include InputHand, garden, SheepPen, and D29 triggers.
- `parity_live_disjoint_report.json`: fresh `2632800000..2632800015`, both
  seats, 32 / 32 games and 23,008 / 23,008 frames exact against the frozen
  Python source. This range is disjoint from the v21 training range and from
  the reserved `2633000000..2633000511` blind range.

The earlier `2632500000..2632500007` live probe overlapped v21 training seeds
and was used only as development parity; it is not evidence for the disjoint
gate.

JobBatch code 5 is now wired with public-opponent route `-1`. The independent
`fieldcraft-jobbatch-b8-parity-2632800100-v2.json` gate replays four new seeds
in both seats through the standalone native module: 5,752 / 5,752 opponent
action frames, 8 / 8 terminals, and 8 / 8 job identities are exact, with zero
illegal actor choices and no fallback path. The report pins the JobBatch
module, standalone module, source, mirror, asset, student binary/weights, route
library, deployment, and R1 configuration hashes. The production agent and
R1 files remain untouched; formal trainer pool weighting is intentionally a
separate change.
