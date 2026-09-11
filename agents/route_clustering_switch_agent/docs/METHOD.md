# Route clustering and learned switching: complete method

[Both approaches](../../../docs/agent_approaches/README.md) · [Package entry](../README.md) · [Source provenance](../SOURCE_PROVENANCE.md)

## 1. The idea in ordinary terms

A replay contains both a farming strategy and the movements used to execute it. Two players can pursue almost the same production strategy while taking different paths, suffering different weeds or receiving different prices. Treating every replay as an independent strategy overweights common approaches and mistakes execution noise for strategic diversity.

This agent first groups historical games by **intended production and investment**. It keeps an executable representative from each retained family, simulates those representatives against one another and searches for situations where changing strategy helps. Finally, it trains small decision trees to recognize those situations from observations available during play.

For example, two replays might both invest in the same animal production at nearly the same times, although one worker detours and the other encounters a shortage. They can belong to the same family. A crop-heavy alternative can belong to a different family. At a switch checkpoint, public prices, stocks and other observable features help decide whether to keep the opening or use a different family's continuation. This is an illustration of the mechanism, not a literal rule asserted for every tree.

The result is a route library plus a learned selector and an existing reactive executor. It is **not** a policy network trained end to end, and it does not invent an arbitrary new farm plan during a match.

## 2. Components and where to inspect them

| Component | Responsibility | Source or asset |
|---|---|---|
| Replay feature extraction | Describe production intentions and time/layout structure | [analyze_macro_route_library.py](../scripts/analyze_macro_route_library.py), [merge_macro_route_caches.py](../scripts/merge_macro_route_caches.py) |
| Execution audit | Check whether intended actions actually succeed under the game rules | [audit_macro_execution_quality.py](../scripts/audit_macro_execution_quality.py) |
| Route distance and clustering | Group similar intentions independently of outcomes | [intent_distance.cpp](../fast_kaggriculture/tools/intent_distance.cpp), [cluster_intended_macro_routes.py](../scripts/cluster_intended_macro_routes.py) |
| Representative export | Turn selected replay seats into executable route carriers | [export_macro_intent_route_specs.py](../scripts/export_macro_intent_route_specs.py), [export_intent_route_carriers.py](../scripts/export_intent_route_carriers.py) |
| Simulation/search | Evaluate static routes and counterfactual switches | [round robin](../scripts/run_native_intent_round_robin.py), [switch search](../scripts/run_native_intent_switch_search.py) |
| Opening selection | Solve a maximin mixture from the payoff matrix | [solve_route_nash.py](../scripts/solve_route_nash.py) |
| Target reduction and training | Find complementary targets and learn robust switch decisions | [target selection](../scripts/analyze_compact_switch_search.py), [tree training](../scripts/train_robust_search_route_trees.py) |
| Checkpoint selection | Evaluate combinations of enabled checkpoints | [search_native_route_tree_sequences.py](../scripts/search_native_route_tree_sequences.py) |
| Online features and control | Build 147 features, traverse trees and enforce one switch | [route_switch_features.py](../src/meta_agent/src/route_switch_features.py), [search_route_policy.py](../src/meta_agent/src/search_route_policy.py) |
| Execution adapter | Select the route tape while retaining the reactive base policy | [teammate_expanded_routes.py](../src/meta_agent/src/teammate_expanded_routes.py), [teammate_base.py](../runtime/teammate_base.py) |
| Frozen deployment | Complete Python agent with embedded assets, or expanded equivalent | [main.py](../main.py), [runtime/](../runtime/) |

There are three representations of the implementation: editable research code under `src/`, a frozen expanded deployment under `runtime/`, and a generated compressed `main.py`. Changing the first does not automatically update the other two. This handoff preserves both frozen deployment forms.

## 3. Step one: build a dataset without counting duplicates twice

The archived experiment combined 1,165 older replay seats and 2,398 newer seats into **3,563 distinct episode/player pairs**, spanning episode identifiers 93803702 through 98440493. A seat means one player's trajectory in an episode, not one whole two-player game. The deduplication key is `(episode_id, player_index)`.

These counts describe that particular corpus. They are not constants that a new run should force. Preserve acquisition dates, team identity and episode/seat identity in the manifest, using [manifest.example.json](../configs/manifest.example.json). A second replay collection can be merged with the first.

The audit replays the 719 action steps through the native environment. It separates what was attempted from what became a real crop, building or animal. This distinction prevents a cash failure from making the intended strategy look like an unrelated low-investment strategy.

However, intention similarity must not excuse unusable representatives. The quality gate below checks execution separately before a carrier enters expensive search.

## 4. Step two: represent production intentions

The route description uses two main structures:

- Five intended layouts, each covering 100 farm tiles, at steps **168, 288, 432, 576 and 719**.
- Ten time windows of 72 steps, each containing counts for 12 kinds of macro action.

Layout categories cover five crop types, three animal types, coops and pastures. The schedule also describes land expansion and hiring. PLANT, construction and animal-placement intentions remain represented even when an individual action fails.

Worker movement, watering, weeding and routine market transactions are not clustering dimensions. Neither are final reward, win/loss or cash. Those quantities matter for execution and evaluation, but using them in the distance would conflate strategic identity with luck, prices or execution quality.

The combined distance is:

```text
D(i, j) = 0.45 * composition_difference
        + 0.35 * layout_difference
        + 0.20 * timing_difference
```

For composition, compare category counts at each of the five layouts: sum absolute count differences and divide by the larger total count, with denominator at least 8. Average over layouts. For layout, count mismatched tiles within the union of active tiles, divide by the union size and average over layouts; two empty layouts contribute zero.

For timing, cumulatively sum macro-action counts through each of the ten windows. At each window, divide the absolute count difference by the larger cumulative total, with denominator at least 6, and average over windows. Cumulative counts make an investment delay visible without comparing every worker step.

This distance deliberately emphasizes the eventual production composition and spatial commitment while retaining a smaller timing term. The weights are design choices, not learned economic laws.

## 5. Step three: cluster and choose executable representatives

The implementation uses average-linkage agglomerative clustering with the precomputed distance and a **0.12** distance threshold. The archived sensitivity study explored thresholds from 0.04 to 0.20: 0.04 produced 434 families, 0.12 produced 275, and 0.20 produced 178.

At 0.12, 211 families were singletons. The largest two families covered 50.21% of samples, the largest eight 84.51%, and the largest 56 covered 93.63%. Thus the dataset was large but highly concentrated. More replay files did not imply proportionally more independent strategies.

Family labels such as `G001` are identifiers, not model architectures. Their ordering uses support, then median reward and a deterministic tie-break. Reward in this naming order is different from reward in the clustering distance, where it is excluded.

The medoid is the actual sample with the smallest mean distance to other family members. A median macro schedule can describe a family, but it is not necessarily executable. Export therefore needs an actual replay carrier, with actions indexed by original time.

The archived zero-win filter retained **175 of 275 families**, covering **3,448 of 3,563 seats**. The 100 discarded families contained 115 seats. This removes weakly supported candidates for that search; zero observed wins, especially for a singleton, is not proof that an idea can never work.

### Mandatory quality review before large searches

Inspect both per-team diversity and the global intent families. They answer different questions. A team may contribute many replays of one policy while the global pool contains a long tail of rare ideas.

Recent maintenance guidance suggests inspecting the latest 20 games per top team: three or four routes can be plausible, while six or more deserves investigation. A rough 40%/20%/10% leading-family pattern can be a useful diagnostic reference. These are review heuristics, not statistical acceptance thresholds.

For each selected representative, record family support, centrality, hard failures of planting/construction/animal placement, replay completion, reward agreement, wins/draws/losses, final cash, minimum cash and capital demand. Examine top-10 coverage and singleton counts as well.

The existing pure-medoid exporter does **not** automatically optimize representative execution quality. Where the medoid has hard failures, inspect nearby candidates and choose a credible carrier using execution quality first, then outcome/cash and centrality. Record the choice. Do not add outcomes to the clustering distance to hide a poor representative.

The shell pipeline currently runs stages consecutively. This quality review is a **manual gate**: execute the preparation stages separately, inspect their outputs and resolve unexplained anomalies before starting the round robin. Do not describe the script as enforcing a gate it does not implement.

## 6. Step four: evaluate routes and choose an opening

Every retained route uses the same route-intervention base executor, so the study compares route choices through a common execution stack. It is not a tournament of 175 unrelated original notebook implementations.

The static round robin used 175 routes, 64 seeds in `[410000, 410064)` and both seats for every distinct route pair: **1,948,800 games**. Payoffs use 1 for a win, 0.5 for a draw and 0 for a loss. The self-play diagonal is restored to 0.5 for the matrix game.

SciPy's HiGHS linear-programming solver chooses nonnegative opening probabilities summing to one, maximizing the guaranteed payoff against matrix columns. The archived mixture put approximately 36.36% on `G001` and 63.64% on `G136`, with matrix-game value 0.5. This is an equilibrium calculation inside the finite evaluated pool, not a universal guarantee against new policies.

The **published runtime forces `G001`**. It does not sample the recorded Nash mixture on every new game. The mixture is retained as search provenance and as an available controller mechanism; it must not be mistaken for the final deployed opening policy.

## 7. Step five: generate training labels by counterfactual simulation

At a checkpoint, the search holds the opening trajectory, opponent, seed and seat fixed, then tries multiple continuations from that same state. A candidate switch uses the destination route's suffix at the **same absolute game step**. It does not replay the destination's opening, reset the farm or magically acquire its assets.

This matters: a promising route may rely on animals, land or supplies that the current farm lacks. Its continuation must be evaluated from the actual switch state through the execution layer. That evaluation exposes incompatibility, although it does not create a general planner that can repair every incompatibility.

The coarse search used two openings, 14 checkpoints, 175 destination routes, 175 opponent routes, eight seeds in `[420000, 420008)` and both seats: **13,720,000 continuations**. Checkpoints were 48, 72, 96, 120, 144, 168, 216, 264, 312, 360, 408, 456, 504 and 576.

For each matched situation, the training target is the destination producing the best outcome, with win/draw/loss score primary and final cash margin breaking ties. Labels therefore come from simulated alternative actions rather than a human annotator or the historical winner's action alone.

A greedy target reduction keeps the opening routes and adds destinations that improve the paired oracle by at least 0.0005. The archived result retained 11 complementary targets: `G001`, `G136`, `G006`, `G036`, `G126`, `G169`, `G227`, `G245`, `G093`, `G008` and `G013`. They are not simply the eleven best global win rates.

Fine search used two openings, seven checkpoints through step 216, those 11 targets, 175 opponents, 128 seeds in `[430000, 430128)` and both seats: **6,899,200 continuations**.

## 8. Step six: train small, robust decision trees

Each opening/checkpoint receives a supervised `DecisionTreeClassifier`. The input schema, `semantic_route_switch_v1`, contains **147 features**:

- Own and visible opponent assets, crops, animals, workers and weeds.
- Own private stock, seeds and bags; no private opponent inventory.
- Public market prices and inventory, shop availability and time.
- History derived from observations, including visible cash and weed/loss changes.
- Planned costs and capital slack over the next 24, 48 and 72 steps of the agent's **own known route**, including purchases, hiring, land and animals.

The last category uses a policy asset already stored in the agent. It is not access to the unseen future of the current match. Similarly, historical replay data used offline does not authorize the runtime to inspect future opponent actions.

The published pipeline searches depths 2, 3 and 4 and minimum leaf sizes 512, 1,024 and 2,048. Defaults exposed by individual scripts may differ; use the explicit pipeline arguments when reproducing that experiment.

Validation is payoff-based. A classifier can often predict the most common route correctly while failing in the relatively few states where a switch creates value. The code scores the simulated payoff of the route it chooses, not only classification accuracy.

There are two grouped cross-validation views, up to five folds each: grouping by seed and grouping by opponent family. Robustness checks shift thresholds by 5% of each feature's training 5th-to-95th-percentile range, with a small lower bound, using both uniform directions and eight deterministic sign patterns. Paired per-seed gains provide a one-sided 95% Student-t lower confidence bound.

For each validation view, the robust gain is the minimum of the worst threshold-perturbed improvement and the lower confidence bound; the final criterion also takes the worse grouped-validation view. Bootstrap refits check root-feature-group stability. The archived gate requires robust improvement of at least 0.01 and root-group stability of at least 0.35. Among candidates within 0.005 of the best score, the selection favors smaller trees and larger leaves.

This is supervised distillation of search results with stability checks. There is no PPO update, policy-gradient rollout or Torch checkpoint in the active runtime. Old recurrent/PPO artifacts belong to abandoned experiments and are listed in [omissions](../OMITTED_ARTIFACTS.md).

## 9. Step seven: select checkpoint sequences and evaluate a holdout

The surviving candidate checkpoints were 72, 96, 120, 144, 168 and 216. Searching all 63 nonempty subsets, plus the baseline, used 128 fresh seeds in `[450000, 450128)`, 175 opponents and both seats: **2,867,200 games**. This selects the schedule of opportunities to switch, not an unlimited sequence of route changes.

The selected checkpoints were **144, 168 and 216**, with at most one actual switch. The final evaluation used another 256 seeds in `[460000, 460256)`, both seats and the same 175-route opponent pool. Baseline and dynamic policies each played 89,600 games, totaling **179,200**.

| Archived final holdout metric | Value |
|---|---:|
| Fixed-opening baseline score | 81.176339% |
| Learned-switch score | 89.614955% |
| Paired improvement | 8.438616 percentage points |
| One-sided 95% lower bound on improvement | 7.643251 percentage points |
| Seed-level standard error of improvement | 0.481791 percentage points |
| Fraction of games with an actual switch | 42.015625% |

These values are retained in [runtime/MANIFEST.json](../runtime/MANIFEST.json). They are archived author results, not a tournament rerun during this publication. Large sample matrices and the source replay collection are omitted. Artifact checks can establish what is packaged, but cannot independently reconstruct these outcomes without the omitted inputs and computation.

## 10. What happens on every online step

1. On step zero, reset history and switch state, choose forced opening `G001` and select its route carrier.
2. Read the new observation and update public history.
3. If the current step is 144, 168 or 216 and no different route has yet been selected, build the feature vector and traverse the corresponding tree.
4. If the predicted family is the current route, continue; a later checkpoint can still act. If it is a valid different family, append a schedule change at this step and lock out further switches.
5. The adapter selects the correct route's tape entry for the current absolute step. The base policy receives that route through its intervention hook and produces the actual action using its reactive execution logic.

The frozen trees have depths **2, 4 and 2**, with 7, 17 and 7 nodes. Although the feature schema is broad, the current trees use a small subset: tomato, strawberry, milk or wool market inventory; milk or wool price; and yarn-shop availability. These are the actual named features behind the numerical indices in the JSON. They do not imply that every available opponent feature is used by this particular trained tree.

The adapter creates an isolated base-policy namespace, retains its execution behavior and disables the specified opening-pressure/late-route override hooks so the chosen route can be supplied consistently. The base can adjust actions reactively, but it is not the P16 complete-workflow scheduler. There is no claim of globally optimal routing, minimal hiring or guaranteed feasibility for any route suffix.

## 11. Run, reproduce or rebuild

To use the published policy, load [main.py](../main.py) and call `agent(observation, configuration)`. It embeds the route assets and needs NumPy; the training toolchain is not needed for inference. The expanded runtime is useful for inspecting the same deployed assets. Initialize independent policy instances/processes for separate simultaneous games.

From this package directory, basic checks are:

```bash
python -I main.py
python -I runtime/main.py
PYTHONPATH=src python -m pytest -q tests
```

Initialization is not a full match test. For offline reconstruction, use Linux/WSL, Python dependencies in [requirements.txt](../requirements.txt), and a C++20/OpenMP toolchain:

```bash
python -m venv /tmp/route-switch-env
source /tmp/route-switch-env/bin/activate
python -m pip install -r requirements.txt
cp configs/pipeline.env.example /tmp/route-switch.env
# Edit the replay manifests, replay roots, worker counts and OUTPUT_ROOT.
# Inspect the preparation stages and complete the quality gate before the search.
bash scripts/run_pipeline.sh /tmp/route-switch.env
```

`OUTPUT_ROOT` holds generated features, audit tables, matrices, policies and exported agents. The wrapper is a full, expensive run: it is not a smoke test. The original seed ranges and search sizes are explicit in the script. New replays can change family labels, counts, opening support and trained trees; matching the old aggregate counts is not the goal of a new experiment.

The package's native simulator targets the default `kaggle-environments==1.32.7` rules. Review [its rule audit](../fast_kaggriculture/RULE_AUDIT.md), install the matching official engine for differential testing, and validate the exact configuration before trusting new search results. The repository's older general lab describes 1.32.6 separately. Sparse custom `marketParams` overrides remain unsupported in this native package.

For a lightweight repack of the frozen expanded assets:

```bash
python scripts/export_single_file_submission.py \
  --source-dir runtime \
  --forced-opening G001 \
  --output /tmp/route-switch-repacked.py
python -I /tmp/route-switch-repacked.py
```

Always pass the forced opening when reproducing the published policy. The single-file exporter otherwise defaults to the controller's opening mixture. This branch repairs the full pipeline to pass its selected final opening to both exporters and fixes the multi-file exporter's `src/meta_agent` lookup. The original frozen `main.py`, `runtime/` assets and trained trees remain unchanged. [Publication record](../../../docs/agent_approaches/PUBLICATION.md).

## 12. Limits and next experiments

The route pool is highly homogeneous; held-out seeds do not create held-out strategic families. The grouped validation helps, but a final test against independently acquired live agents is still necessary to establish broader strength. Do not compare the 89.6% score directly with the P16 study's win rates.

Representative quality, route suffix compatibility, absolute timing, sparse checkpoints and simulator fidelity can all limit generalization. Market features may also change meaning when new opponents alter supply. One-switch control is intentionally small and stable, but cannot reverse a poor switch later.

Useful next experiments would audit alternative near-medoid carriers, hold out newly acquired policy families, test delayed or incompatible route transitions, and compare on one frozen live-opponent panel with the P16 agents. Combining a learned macro selector with a state-based worker planner is a possible future experiment; no such hybrid is implemented in this publication.
