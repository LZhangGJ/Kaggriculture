# Development checkpoint, not the selected agent

The root of this archive is the only r2 candidate.
`broad_matched_deferral/` is the first implementation of the same land-wait idea,
retained with its complete source and matching native. Its hash is recorded in
REFINEMENT.json. It added alternatives even when some original candidate already
avoided land. Saved-observation probes diverged in four cases, including both
narrow-win fixtures. Those are not measured new losses.

The final intervention is narrower: alternatives are added only when ALL original
prepared queues require BUY_LAND. This avoids widening an already two-sided
decision and reproduces both narrow-win own-action traces exactly. No new seed
sweep, score threshold, opponent-name gate, or seed table was introduced.

Raw broad-stage results are under validation/analysis/candidate_root* and
choice_units*. The final-stage results are explicitly named narrow_*.
Every supplied development seed and all 1536 outcomes remain in evidence.
