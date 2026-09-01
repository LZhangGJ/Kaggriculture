# Source provenance

The reproducibility code was recovered on 2026-08-25 from the final experiment
workspace at:

`hw:/root/kaggriculture_transformer_ppo_starter/meta_agent_route_rl_submission_minimal`

That workspace is not a Git checkout, so no truthful upstream Git commit is
available. The imported route-search scripts are the final server copies. Two
legacy compatibility branches were deliberately removed from
`search_route_policy.py` and `train_robust_search_route_trees.py`; they depended
on the abandoned recurrent/PPO feature path and are not used by the 147-feature
DecisionTree router.

`runtime/teammate_base.py` is the route-intervention build used by the native
round-robin and counterfactual switch search. Its SHA-256 is
`71e8689abf5aff9161fd5957cda6bcea14e755bdc38ec4618702b3b6f0a73f7d`,
matching the server file
`teammate_meta_route_submission_v1/teammate_base.py`.

No Replay, PPO/recurrent weights, generated matrices, search caches, compiled
binaries, or server experiment outputs were imported in this update.
