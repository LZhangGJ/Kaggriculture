# Agent maintenance guide

These instructions apply to work in `agents/route_clustering_switch_agent/`. Keep changes reproducible and preserve unrelated user work. Documentation introduced or revised for the two-approach handoff must be in English.

## Scope and implementation layers

By default, work inside this package. Cross-directory changes need a clear task reason and user authorization; the current two-approach publication explicitly authorizes repository-level documentation and the second agent family. Check Git status before editing and preserve unrelated changes.

- `main.py` is generated; do not edit its compressed payload by hand.
- `runtime/` is the exported deployment snapshot, including routes, action tapes and policy.
- `src/meta_agent/` is the development source for features, control and execution adaptation.
- `scripts/` contains offline data processing, search, evaluation and export.
- `fast_kaggriculture/` contains the native simulator with separate documentation.
- `docs/` contains explanations and charts; it is not required for inference.

For behavior changes, edit development source first and then export and verify a new candidate. Runtime snapshots and similarly named development files need not be identical; do not overwrite them in either direction without verification. Write replays, generated matrices, search caches and build products to an `OUTPUT_ROOT` outside tracked source.

## Invariants of the published runtime

1. The active selector is a shallow decision-tree route policy, without PPO, PyTorch or `.pt` weights.
2. The opening is forced to `G001`; switch checkpoints are 144, 168 and 216; at most one actual switch is allowed.
3. Search and evaluation use the route-intervention interface in `runtime/teammate_base.py`.
4. Features use `semantic_route_switch_v1`. A schema change must update training, export and runtime consistently.
5. Counts such as 275 families, 175 carriers, 11 targets and 63 checkpoint subsets describe the original data, not future invariants.
6. Large generated artifacts belong outside ordinary Git history.

An intentional change to a runtime invariant must update the README, runtime manifest, relevant tests and method guide, explaining compatibility implications. Documentation-only changes should preserve frozen deployment bytes.

## Clustering quality gate

A successful script exit does not establish route-library quality. Complete this review before round-robin, Nash or counterfactual search. The pipeline script does not automate the review or pause for it.

### Inspect family support

Distinguish per-team diagnostic clustering in `analyze_macro_route_library.py` from global intent clustering in `cluster_intended_macro_routes.py`. Do not mix their labels or percentages.

Calculate each family's support divided by replay seats. Record at least the largest ten families, cumulative coverage, singleton count and total family count. Inspect a recent window of about 20 valid games for each leading team as well.

A head-heavy per-team distribution, roughly 40%, then twenty-something and ten-something percent, can be a diagnostic reference. Three or four routes in a recent 20-game window can be plausible; six or more deserves investigation. These are empirical heuristics, not statistical confidence intervals or automatic rejection rules. Real policy changes, small samples and selection bias can explain deviations.

Unexpectedly uniform families, a small leading family, many singletons or a family swallowing almost everything require inspection of replay/seat selection, intention recovery, layout/time alignment, degenerate distance components and threshold choice.

### Audit representative quality

`export_macro_intent_route_specs.py` currently chooses a geometric medoid. It reports execution failures, outcomes and cash but does not automatically use them to choose the representative. An exported `intent_medoid_source` is not evidence that the carrier is good.

Record for every proposed family carrier:

| Field | Purpose |
|---|---|
| Family, support and share | Measure the impact of a poor representative |
| Medoid route and within-family distance rank | Keep the candidate near the strategy center |
| Failed production tasks | Inspect planting, construction and animal placement |
| Completion rate and reward agreement | Detect replay/executor semantic errors |
| Family and carrier wins/draws/losses | Avoid selecting an obviously failed carrier without explanation |
| Final cash/reward, minimum cash and capital gap | Inspect execution and funding viability |

Consider a small set of actual samples near the medoid. Prefer fewer hard production failures, better completion and exact reward replay. When those are comparable, consider outcomes, final/minimum cash and centrality. If a high-quality candidate is far from the center, review the cluster, split it or retain multiple candidate carriers; do not substitute an unrelated high-scoring outlier.

Outcome and cash are post-clustering quality criteria, never distance inputs. Record selected/rejected candidates and reasons in a quality JSON/CSV under `OUTPUT_ROOT`.

### Pause expensive search when evidence is missing

Review before proceeding if the support distribution is unexplained, a recent team window splits into six or more routes without evidence of policy change, leading medoids have many production failures, replay rewards disagree, leading carriers suffer funding failure, or the quality table is missing.

When overall performance is poor, investigate in this order: family distribution, leading-carrier quality, native/reference semantics, payoff matrix/opening selection, counterfactual labels, then decision trees. Tuning depth cannot repair a broken route carrier upstream.

## Work and validation sequence

Read the README and relevant method/source files. Inspect status, change only what the task requires, complete the quality gate for newly generated route libraries, and run the most relevant tests before broader checks. For re-exported agents, compare manifests, input provenance, export logs and final artifacts.

Documentation or light Python work requires at least:

```bash
python -I main.py
python -I runtime/main.py
PYTHONPATH=src python -m pytest -q tests
```

Native changes also require an external build and the native tests:

```bash
bash scripts/build_native.sh /path/to/output-root
python -m pytest -q fast_kaggriculture/tests
```

Changes to training, route counts, checkpoints or new performance claims require corresponding pipeline stages in an independent output directory and newly generated evidence. The full pipeline is expensive. State when it was not run; static checks are not experimental reproduction.

## Data and documentation conventions

Replay manifests follow `configs/manifest.example.json`; replay paths are resolved relative to the manifest. Local pipeline configurations follow `configs/pipeline.env.example` and are not committed. Update `SOURCE_PROVENANCE.md` when provenance or trust boundaries change, and `OMITTED_ARTIFACTS.md` when excluding large artifacts, preserving reasons, sizes and available hashes.

The README should keep entry points, checks, reconstruction steps and evidence boundaries easy to find. Put detailed formulas in `docs/METHOD.md` and native implementation details in `fast_kaggriculture/README.md`. Report validation performed, publication changes and unrun expensive stages at handoff.
