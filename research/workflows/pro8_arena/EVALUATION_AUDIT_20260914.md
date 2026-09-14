# Kaggle evaluation and the private arena

Reviewed September 14, 2026. Scope: game outcomes, draws, live skill updates, final Bradley–Terry (BT) ranking, eligible history, matchmaking, uncertainty, and selection of two submissions.

## Findings that change our arena

1. Kaggle staff confirms that final BT ties count as half wins. Our score tables and Elo already did this, but our BT fit used a Davidson model with a separate draw parameter. We replaced it with fractional-outcome logistic BT.
2. The final fit uses the whole competition's eligible history, not just the post-deadline fortnight. Both submitted versions must remain active. We added a cumulative active-version BT view that combines resolved daily and continuous games, excludes either retired endpoint, and keeps execution contracts separate.
3. Kaggle's live system models skill and uncertainty. Our fixed-K Elo is an internal metric, not a reproduction of Kaggle's score. It now says so in the site and report. Changing its starting number to 600 would not fix that difference.

## Evidence and authority

| Source | What it establishes | Limits |
|---|---|---|
| [Kaggle simulation documentation](https://www.kaggle.com/docs/competitions?lor=19), Simulation Evaluation | Gaussian skill estimate; initial mean 600; updates depend on surprise and uncertainty; draws bring means closer; cash margin does not affect rating; similar-skill matchmaking | Describes the process, not executable formulas or all constants. General documentation does not override competition-specific rules. |
| [Kaggriculture final-evaluation announcement](https://www.kaggle.com/competitions/kaggriculture/discussion/731587), María Cruz, staff | Games continue for two weeks after submissions close; a final BT fit determines ranking | Does not give the solver, prior, or guaranteed episode count. |
| [Which episodes count?](https://www.kaggle.com/competitions/kaggriculture/discussion/732931), Addison Howard and Bovard Doerschuk-Tiberi, staff | History spans the whole competition; both endpoints must still be active; games against deactivated versions do not count | Does not answer the question about carrying live ratings into a prior. |
| [Three final-scoring clarifications](https://www.kaggle.com/competitions/kaggriculture/discussion/739410), Addison Howard, staff | Ties are half wins; a team's better submission determines its rank; a team cannot occupy two ranks; no promised post-deadline play rate | Does not supply a numerical live-rating algorithm. |
| [Kore rating-system clarification](https://www.kaggle.com/competitions/kore-2022-beta/discussion/316935), Bovard Doerschuk-Tiberi, staff | Historical system described as similar to TrueSkill | Historical evidence, not proof that stock TrueSkill and its defaults reproduce Kaggriculture. |

The competition overview/evaluation page did not expose its full text through the research fetch. The relevant staff discussions above did, including replies. Searches also covered public discussions about convergence and rating formulas. I did not find an authoritative numerical specification or released server implementation for current live updates. This is a limit of this audit, not proof that no such implementation exists anywhere.

The popular [participant convergence analysis](https://www.kaggle.com/competitions/kaggriculture/discussion/736219) says the final fit is post-deadline-only. That conflicts with the direct staff clarification. Its estimates of convergence time and games/hour are not an evaluation contract and were not used to configure ours.

## Exact outcome treatment

For each valid game, the rating outcome is 1 for a win, 0.5 for a draw, and 0 for a loss. A 100-cash win and a 100,000-cash win have the same rating outcome. Final cash remains useful for diagnosing strategy and variance; it must not scale rating updates.

Our referee compares the official engine's terminal rewards exactly. Equal finite terminal cash yields a draw; it does not round cash or apply a tolerance. The validator checks the recorded outcome against cash. A single agent failure yields a forfeit loss, with no invented cash. Infrastructure failures and two failed agents remain unresolved and receive no rating points. The precise hosted handling of two simultaneous failures is not established by the sources reviewed; keeping them visible as failures is safer than recording normal-play draws.

Strict win rate is W/N. Competition score is (W + D/2)/N. Both remain visible. The website's opponent detail now shows draw counts and half-draw score as well. A field of identical agents could have zero strict wins and a 50% score; that is not a contradiction.

## The corrected BT fit

For strengths s_i and s_j:

    p(i over j) = sigmoid(s_i - s_j)
    loss = sum(log(1 + exp(s_i - s_j)) - y*(s_i - s_j))
           + sum(s_i**2)/18

Every draw uses y=0.5. Thus two draws contribute the same likelihood as one win and one loss. There is no separate draw category or fitted seat effect. Seat results remain diagnostics, and scheduled games still swap seats.

The Gaussian SD=3 penalty is an explicit local choice to keep estimates finite when an agent is undefeated. It is symmetric across agent identities, unlike penalizing only strengths relative to one unpenalized reference. Strengths are centered within each connected comparison group. Kaggle's regularization, solver tolerance, initialization, numerical scale, and any carryover prior remain unknown. We match the confirmed outcome semantics, not a claimed exact final numeric score.

Daily completed tournaments retain 500 seed-block bootstrap fits and 95% intervals. These describe sampling variation conditional on this roster and model. They do not cover unknown opponents, changing rules, or strategy-selection bias. Cumulative snapshots initially omit intervals; they do not borrow daily intervals or imply certainty from large game counts. Disconnected groups are not ranked against each other.

Rating method is versioned as `half-win-bt-v2`; old report caches are invalidated. Historical raw results stay unchanged, so a changed rating after this deployment does not imply the agent changed.

## Cumulative and daily views

The cumulative view includes all resolved daily and continuous games with both exact versions in the active roster. It keeps one fit per execution contract. It excludes smoke, placement, tuning and targeted gate games, whose selection differs from normal arena play. Eligible games are indexed once in a private SQLite database. Incomplete runs are revisited; resolved records are immutable. Ordered agent identities, seed and contract prevent the same game from being counted twice, while preserving the swapped seat as a distinct game.

Retiring an agent removes its games from this active-view fit, including their indirect influence on other ratings. It does not delete evidence. Re-adding the same exact version restores eligible history. A new version has a new identity and cannot inherit old results.

Daily tournaments remain frozen-roster comparisons on fresh seeds. They help show whether the cumulative result persists on a recent balanced sample. A cumulative snapshot is never labeled a completed daily tournament. The Best team agent card still selects from a completed daily tournament and now excludes retired team versions. The cumulative fit is available in the tournament selector for submission review.

## Live Elo: what is and is not replicated

Current internal rule:

    expected = 1 / (1 + 10**((R_other - R_self)/400))
    delta = 32 * (mean score across two swapped-seat games - expected)

Both agents start at 1500; one complete seat pair produces one equal-and-opposite update. This is not K=32 per game. Draws between equal ratings leave them equal; draws between unequal ratings move them closer. Unfinished pairs wait, daily games do not enter Elo, and a processed pair cannot update twice. The internal scale cannot be compared directly to a Kaggle score of 2700.

Kaggle describes per-episode updates with agent-specific uncertainty. The public sources do not specify current initial uncertainty, performance-noise scale, uncertainty floor/drift, draw margin, exact update equations, or all matchmaking weights. We should not pick stock TrueSkill defaults, call them Kaggle's algorithm, and replace a useful running metric. An optional approximation would need its own label and validation against hosted before/after ratings; that is not implemented here.

Elo also depends on update order. Our arrivals come from several CPU workers, and ratings retain historical influence from opponents since retired. That is appropriate for a live progress indicator but differs from the active-only batch fit. Use cumulative BT and the matchup matrix to resolve disagreements.

## Matchmaking, sample size and two submissions

Kaggle seeks similar-skill opponents; our round robin measures a broader, balanced pool. This is an intentional difference. Do not trade away broad coverage to mimic an unpublished scheduler. More local games reduce seed noise, but cannot make a narrow public roster representative of every private finalist. Similar-strength comparisons matter most when deciding between close finalists; retain coverage against distinct strategies too.

For submission selection, compare exact versions on cumulative active BT, recent completed daily BT, per-opponent score, per-seat score and runtime failures. Inspect disagreements: a candidate may exploit one common family or lose badly to a less common counter. Use fresh confirmation seeds after tuning. Do not promote purely on a live Elo streak or average cash.

Slot one should hold the strongest robust candidate. Slot two can hedge with a similarly strong agent whose weak matchups differ. Do not simply choose the top two internal rows when they are near-identical, nor choose a much weaker agent solely for diversity. The relevant goal is the better final rank of the two, not their average rating. No local sample guarantees a particular Kaggle placement.

Before the deadline, freeze a shortlist and the opponent versions used for final checks, then fit their whole eligible history and review recent fresh panels separately. Test the actual packaged submission under the runtime contract. The official final episode count is uncommitted; we must not promise a per-day rate based on another competition.

## Validation and remaining limits

Regression tests cover half-win equivalence, all draws, draws between unequal agents, symmetric identity handling, finite undefeated fits, strict-win versus score accounting, incomplete Elo pairs, idempotence, and removal of retired opponents from cumulative history. Existing intake, exports, scheduling and security tests remain required.

This change does not change engine rules, CPU concurrency, worker isolation, private access, agent code, or competition submissions. Remaining gaps are the private opponent population, exact hosted runtime configuration, live update equations, scheduler parameters and final-fit numerical choices. Those gaps are stated rather than filled with assumptions.
