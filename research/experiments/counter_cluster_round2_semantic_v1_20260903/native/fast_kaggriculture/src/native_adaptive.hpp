// Licensed under the Apache License, Version 2.0.
#pragma once

#include "adaptive_candidates.hpp"
#include "native_teammate.hpp"
#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fastkag {

// A compact, searchable genome for the general economic planner.  These are
// policy parameters, not a replay tape and not opponent-identity switches.
struct AdaptiveGenome {
  double cash_reserve = 180.0;
  double action_cost = 18.0;
  double move_cost = 7.0;
  double risk_multiplier = 1.08;
  double market_impact_weight = 1.0;
  double opponent_supply_weight = 0.65;
  double demand_drift_weight = 0.60;
  double sell_drop_limit = 0.16;
  double price_replan_fraction = 0.20;
  double task_stickiness = 18.0;
  double deadline_weight = 35.0;
  double fertilizer_value_fraction = 0.85;
  int max_hands = 10;
  int max_quadrants = 3;
  int max_total_animals = 18;
  int max_cows = 12;
  int max_sheep = 8;
  int max_geese = 4;
  int max_wheat = 32;
  int max_carrot = 12;
  int max_tomato = 16;
  int max_strawberry = 18;
  int max_melon = 12;
  int min_wheat_buffer = 4;
  int liquidation_day = 27;
  int stop_new_animals_day = 15;
  int stop_new_crops_day = 26;
  int replan_interval_steps = 24;
  // Scale for relative ordering inside the routine-task tier.  Ten reproduces
  // the original priority score exactly; smaller values let travel dominate.
  double routine_priority_scale = 10.0;
  // Converts task-local realized value / avoidable delayed loss into the same
  // soft scheduling score as distance and routine priority.  Zero preserves
  // the historical scheduler for clean ablation.
  double task_value_scale = 0.0;
  // Searchable urgency of satisfying an already selected crop commitment.
  // This is deliberately a policy parameter rather than a crop/day rule.
  double plant_priority = 520.0;
  // Minimum carried product value that justifies an explicit shed return.
  // Smaller cargos may still return when visible opponent supply creates a
  // first-seller race.
  // Disabled by default until an independent opponent pool proves a positive
  // own-score effect; the parameters remain searchable for W3/W5 calibration.
  double drop_value_threshold = 100000.0;
  int preempt_quantity = 100;
  // Fraction of shed capacity offered to the public town-consumption WHEAT
  // relay.  Zero is a clean ablation/default until multi-seed Arena evidence
  // proves that the extra working-capital loop is beneficial.
  double wheat_relay_capacity_fraction = 0.0;
  // Expected adverse opponent WHEAT supply used only when valuing the
  // one-step relay; kept separate from the long-horizon project-price model.
  double wheat_relay_opponent_risk = 0.0;
  // Soft preference for continuing the same crop/animal production chain.
  // Zero preserves the established scheduler; positive values are searched
  // and may always be preempted by hard-deadline work.
  double industry_affinity_scale = 0.0;
  // One-step route lookahead: prefer a destination with several nearby
  // productive follow-up tasks.  Zero preserves the established scheduler.
  double route_density_scale = 0.0;
  // Scheduling-priority points added for each action by which an authoritative
  // daily flow is behind its even within-day execution pace.
  double flow_pace_priority = 0.0;
  // Strength of prerequisite-aware local service chains.  Pickup timing and
  // quantity remain under the established scheduler; positive values only
  // keep an already carrying worker on its dependent local work.
  double prerequisite_chain_strength = 0.0;
  // Select authoritative-flow candidates by proximity to the currently
  // available workforce instead of truncating the map in row-major order.
  // Zero preserves the established ordering for controlled ablation.
  double local_candidate_order = 0.0;
  // Soft within-day spatial ownership.  Anchors are generated from the live
  // task map and remain fixed for the day; hard-deadline work may ignore them.
  double region_ownership_scale = 0.0;
  // Enforce the selected plan's daily animal service/product flow while hard
  // escape and capacity safety remain unconditional.
  double animal_flow_control = 0.0;
  // Use task-count-balanced cell ownership instead of point anchors for the
  // within-day region model.  This is a searchable execution capability.
  double workload_region_mode = 0.0;
  // At most one routine action per physical cell per step so one worker can
  // complete the local multi-action chain.  Hard tasks remain shareable.
  double routine_cell_exclusivity = 0.0;
  // Additional urgency for hard tasks based on travel-adjusted slack.
  double hard_travel_slack_scale = 0.0;
  // Reserve a searchable fraction of cells required by already selected
  // future animal capacity.  This prevents short-lived crops from occupying
  // the central service district before the structures are due, without
  // encoding any replay coordinate.
  double future_structure_reservation_fraction = 0.0;
  // Public-state audits expose two recurring economic decision windows. These
  // fields do not encode a route: they control how much *future*, still
  // reversible backbone capacity is released back to the economic planner
  // when the corresponding shop/price opportunity is visible.
  double yarn_animal_flex = 0.0;
  double yarn_wool_price_threshold = 219.0;
  // Public on-map sheep count above which a YARN opportunity is treated as
  // crowded rather than inviting another sheep-heavy commitment.
  double yarn_opponent_sheep_gate = 100.0;
  double pet_crop_flex = 0.0;
  double pet_carrot_price_threshold = 44.0;
  // Priority used when weeds are the binding capacity bottleneck for the
  // selected crop portfolio.  Zero preserves the legacy coarse rule.
  double weed_capacity_priority = 0.0;
  // Activate authoritative animal-flow limits only when remaining crop work
  // consumes at least this fraction of the day's estimated worker capacity.
  // Zero keeps the original always-active experiment.
  double animal_flow_pressure_gate = 0.0;
  // A live-state alternative to the calendar-pressure gate above.  It counts
  // only crop work that can actually be performed from the current farm state
  // (serviceable plants, available seeds/cells, and clearable capacity).  This
  // prevents a nominal-but-infeasible crop calendar from starving profitable
  // animal work.  Zero preserves the established behavior.
  double animal_flow_executable_gate = 0.0;
  // Suppress animal service whose economic effect cannot become a sellable
  // product before the final market window.  This is computed from the live
  // animal age/interval/capacity rather than a replay day or action count.
  double terminal_animal_economics = 0.0;
  // Generic crop-lot geometry.  Zero preserves nearest-shed radial fill, one
  // uses contiguous scanline lots, and two alternates scan direction by row
  // to create a serpentine service lane.  No replay coordinate is stored.
  double crop_lane_layout_mode = 0.0;
  // Assign daily spatial regions to the live worker positions by global
  // minimum Manhattan cost instead of binding region order to unit id.
  double region_assignment_mode = 0.0;
  // Reserve a worker lane when a hard task reaches its latest feasible start
  // (deadline slack minus the nearest live travel distance).  Unlike a broad
  // priority boost this activates only at the feasibility boundary.
  double hard_latest_start_reservation = 0.0;
  // Preserve ownership of an unfinished routine target while its assigned
  // worker is travelling to it.  Hard deadline work is still assigned first
  // and may preempt the route.  This closes a generic receding-horizon flaw:
  // without ownership, another worker can claim the target on the next step
  // and send the original worker across the farm toward a different task.
  double enroute_task_reservation = 0.0;
  // Solve the remaining routine unit-task batch jointly after hard deadline
  // reservations.  This replaces the myopic first-best greedy sequence with
  // a maximum-total-score matching while retaining the same live-state task
  // scores, legality checks and deterministic fallbacks.
  double global_routine_matching = 0.0;
  // Release a fraction of the backbone's not-yet-planted crop commitments to
  // the live economic allocator.  Existing plants remain hard obligations;
  // only reversible future crop capacity and seed spending are reconsidered.
  // Zero preserves the audited semantic plan, one asks EcoBot to rebuild the
  // future crop portfolio from current public economics.
  double backbone_crop_flex = 0.0;
  // Give each worker the first choice of executable routine work in its live,
  // workload-balanced spatial region before the global fallback.  Regions are
  // rebuilt from current tasks each day and hard deadlines remain unrestricted.
  double region_owner_first = 0.0;
  // Softly finish routine work in the worker's current farm quadrant before
  // crossing to another one.  This is shared-quadrant continuity, not fixed
  // worker ownership; any worker may enter any quadrant when value/deadline
  // evidence is strong enough.
  double quadrant_affinity_scale = 0.0;
  // Fraction of the normal strategic cash reserve protected on the opening
  // day.  A full-season reserve can otherwise block already-selected seeds
  // while workers sit idle.  Feed and same-day hire costs remain protected by
  // their explicit transaction guards, so this only controls deployable
  // residual capital.  One preserves the established behavior.
  double opening_cash_reserve_fraction = 1.0;
  // Continuous soft distance to the worker's workload-balanced daily region
  // anchor.  Unlike hard cell ownership, this still permits shared-resource
  // chains and local reassignment while discouraging repeated cross-farm
  // rescues.  Hard deadlines are never constrained by this term.
  double region_anchor_distance_scale = 0.0;
  // Public town-demand timing for all products.  Positive values allow shed
  // stock to wait a few steps for the next known consumption tick when the
  // forecast gain survives visible opponent supply.  Financing, capacity and
  // liquidation pressure always override the hold.
  double town_demand_sell_timing = 0.0;
  // Value one maintained crop slot over every replant cycle that can mature
  // before the terminal market, including repeated seed and action costs.
  // Zero preserves the former one-lifecycle estimator for clean ablation.
  double repeating_crop_economics = 0.0;
  // Reserve deployable capital for the next productive land block before
  // filling the current block with lower-value marginal projects.  The gate
  // uses only current liquid capital and the remaining official production
  // horizon; it does not encode a replay purchase day.
  double proactive_land_investment = 0.0;

  // Penalize projects that lock liquid capital for many days before their
  // first sellable output.  This is a generic cash-conversion-cycle term: it
  // uses only official production timing and is intentionally independent of
  // any Replay date, quantity, coordinate, or opponent identity.
  double capital_lockup_weight = 0.0;

  // Searchable release fraction for not-yet-purchased animal commitments in a
  // semantic operating prior.  Already owned animals are irreversible and
  // remain hard service obligations; the live project evaluator must earn any
  // replacement capacity from current public economics.
  double backbone_animal_flex = 0.0;

  // Compare the available cow-heavy and sheep-heavy operating priors using
  // current public prices, unlocked-shop demand and visible opponent capacity
  // before the first irreversible animal expansion.  Zero preserves the
  // legacy YARN-only branch gate; positive values scale the forward opponent
  // supply term.  This is a business-state selector, never an opponent-id
  // route.
  double animal_branch_economic_selector = 0.0;

  // Number of full public-state days for which irreversible cow/sheep
  // expansion is held at the common lower bound of the candidate priors.
  // This buys information when an opponent's future industry is still hidden
  // in private inventory.  Zero makes the economic decision at the original
  // expansion date.
  int animal_branch_observation_days = 0;

  // Order simultaneous seed commitments by live cash-conversion value instead
  // of the historical fixed crop enum order.  Zero is the rollback arm;
  // positive values activate the state-derived transaction ordering.
  double seed_transaction_value_order = 0.0;

  // Public capacity imbalance required to counter-specialise the next animal
  // investment.  For example, sufficiently more visible opponent cows than
  // sheep selects the sheep-heavy portfolio.  One hundred is the disabled
  // rollback value on the 10x10 board.
  int animal_counter_crowding_gate = 100;

  // Fraction of the generic strategic reserve that an already-due daily
  // planting commitment may spend.  This is narrower than lowering the whole
  // opening reserve: only seeds required by the current authoritative flow
  // receive the release, while speculative crops/animals/land remain guarded.
  double due_flow_cash_release = 0.0;

  // Accelerate an already-selected crop flow when the opponent's public map
  // shows same-product capacity that can reach the market in the same window
  // and materially depress its quote.  One coherent strength controls both
  // the narrowly-scoped seed cash release and the corresponding live task
  // urgency, so capital and execution do not drift apart.  Zero is a complete
  // rollback and no crop, date, route, coordinate or opponent id is encoded.
  double market_race_acceleration = 0.0;

  // Compare every loaded semantic operating prior with a fully autonomous
  // plan generated from the same public state.  Priors are candidates, never
  // authorities; no opponent identity is available to the selector.  Zero
  // preserves the audited legacy path, positive values enable selection.
  double operating_prior_selection = 0.0;
  // Required relative analytic-value gain before replacing yesterday's
  // selected operating prior.  This prevents harmless market noise from
  // oscillating the whole portfolio while still allowing event-driven change.
  double operating_prior_switch_margin = 0.05;

  // Fraction of yesterday's still-unrealized autonomous project commitments
  // carried into today's candidate construction.  This is public-state plan
  // memory, not a route prior: the project evaluator can still replace future
  // commitments when economics change.  Zero preserves the audited legacy
  // plan-from-scratch behavior.
  double autonomous_commitment_persistence = 0.0;

  // Buy the next quadrant when selected productive projects consume this
  // fraction of current capacity.  The land cost is charged to the marginal
  // project value; this advances expansion from the legacy nearly-full gate
  // without hard-coding a game day or a route-specific acreage schedule.
  // Zero is the exact legacy rollback.
  double land_capacity_trigger_fraction = 0.0;

  // Replace the single global crop stop day with official per-crop terminal
  // feasibility.  Short crops may keep cycling after long crops can no longer
  // return product.  Zero preserves the legacy scalar stop-day behavior.
  double crop_specific_terminal_horizon = 0.0;

  // Replace scalar carry-over with a live project state machine.  Each
  // outstanding crop, animal and land commitment is independently kept,
  // scaled, deferred, switched or cancelled from current public economics and
  // deployable cash.  Zero preserves autonomous_commitment_persistence as the
  // exact rollback path.
  double autonomous_project_state_machine = 0.0;

  // Expected demand from shops that have not unlocked yet.  Official rules
  // add one uniformly sampled shop every town_shop_unlock_interval days until
  // eight shops are present.  Zero is the exact historical rollback; one uses
  // the full rules-derived expectation.  This is uncertainty-aware project
  // valuation, not future-seed knowledge or an expert calendar.
  double future_shop_expectation_weight = 0.0;

  // Internalise the price impact of the portfolio currently being proposed,
  // including both live public commitments and not-yet-placed additions.
  // The legacy estimator discounted these terms as a coarse daily rate; once
  // a purchase reached the field it therefore became artificially cheap on
  // the next replan.  Zero keeps that exact rollback behaviour.  Positive
  // values price all own public-state commitments by expected supply, without
  // a Replay quantity, calendar or opponent identity.
  double portfolio_supply_impact_weight = 0.0;

  // Replace the coarse "visible tiles times horizon" supply proxy with a
  // public-state projection of the products already committed on both farms.
  // The projection uses each tile's age, official maturity/interval rules and
  // currently held yield.  Zero is the exact legacy rollback; one uses the
  // full live commitment forecast.  Hidden inventory and future RNG are never
  // inspected.
  double live_commitment_projection_weight = 0.0;

  // Evaluate complete, reversible portfolio edits after the marginal greedy
  // allocator has produced its draft.  A positive value is the minimum
  // relative analytic-value improvement required to accept a multi-unit
  // SCALE or SWITCH bundle.  Zero is an exact rollback: the extra candidate
  // generation and scoring path is not entered.  Only not-yet-purchased
  // animals and future crop targets are reversible; live obligations remain
  // hard lower bounds.
  double portfolio_bundle_switch_margin = 0.0;
  // Minimum full days between accepted portfolio switches.  Zero preserves
  // the original event-driven behavior; a positive value supplies hysteresis
  // without making the selected industry or date opponent-specific.
  int portfolio_switch_cooldown_days = 0;
  // Offline counterfactual arm: choose the Nth analytic bundle candidate at
  // the same public decision state.  Runtime/default is zero (best analytic
  // candidate); higher ranks exist only to calibrate candidate ordering.
  int portfolio_switch_candidate_rank = 0;
  // Maximum number of following full days for which the two targets touched
  // by an accepted local SWITCH remain directional commitments.  The lock is
  // released earlier once the destination has actually been commissioned and
  // survived its official first-production lead time.  Zero is a complete
  // rollback.  Unrelated projects remain event-driven throughout, so this is
  // a project transaction rather than a frozen operating calendar.
  int portfolio_local_edit_hold_days = 0;

  // Runtime-only offline-oracle flag.  It is deliberately absent from
  // names()/DIM and can therefore never be enabled by a submitted genome.
  // The counterfactual teacher uses it to expose every feasible reversible
  // SWITCH candidate, including analytically negative ones, so the learned
  // value model receives honest negative examples instead of inheriting the
  // weak analytic scorer's candidate filter.
  bool offline_expose_all_feasible_switches = false;

  // Runtime-only broad-candidate counterfactual controls.  They are absent
  // from names()/DIM and therefore cannot silently change any saved or
  // submitted genome.  Offline teachers use them to force one of the generic
  // replay-audited PlanDelta candidates at the first eligible public state.
  bool offline_candidate8_enabled = false;
  int offline_candidate8_rank = 0;
  int offline_candidate8_minimum_day = 0;
  bool offline_candidate8_use_feasible_pool = false;
  // Offline-only stable replay of a semantic Candidate8 edit. Unlike rank,
  // the complete delta identity survives candidate-pool reordering.
  bool offline_candidate8_use_explicit_delta = false;
  AdaptivePlanDelta offline_candidate8_explicit_delta{};

  // Runtime-only public-state market-race probe.  These fields are absent
  // from names()/DIM so every frozen R6 genome remains byte-for-byte
  // compatible.  A positive fraction sells a bounded share of our currently
  // sellable stock when the opponent has enough same-product yield visibly
  // ready on the map.  No opponent id, replay action or hidden inventory is
  // inspected.  The dedicated bindings use this only for causal ablation;
  // promotion into a submitted genome requires independent-pool acceptance.
  double runtime_market_preempt_fraction = 0.0;
  int runtime_market_preempt_min_ready = 4;
  double runtime_market_preempt_min_price_ratio = 0.0;
  int runtime_market_preempt_max_quantity = 30;

  // Runtime-only R8 execution experiment.  These fields are deliberately
  // absent from names()/DIM so every frozen R6/R7 genome remains byte-for-byte
  // compatible.  The dedicated R8 bindings enable them explicitly.
  bool r8_execution_enabled = false;
  int r8_day_horizon_steps = 24;
  int r8_joint_horizon_steps = 6;
  int r8_joint_candidate_limit = 8;
  double r8_lookahead_scale = 0.01;
  // Runtime-only local sequence solver.  Zero keeps a bounded deterministic
  // beam for high-throughput search; one uses the exact subset DP.  Neither
  // field is serialized into the frozen 89-dimensional policy genome.
  int r8_sequence_solver = 1;
  int r8_beam_width = 8;
  // 1=day ledger, 2=+exact current matching/en-route ownership,
  // 3=+persistent task ownership score, 4=+4-8 step route value.
  int r8_feature_level = 4;

  static constexpr int DIM = 89;
  static AdaptiveGenome from_vector(const std::vector<double>& values);
  static const std::array<const char*, DIM>& names();
};

struct AdaptivePlan {
  enum ProjectMode : int8_t {
    INACTIVE = 0,
    KEEP = 1,
    SCALE = 2,
    DEFER = 3,
    SWITCH = 4,
    CANCEL = 5,
  };
  // Irreversible purchase/ownership commitment.
  std::array<int16_t, N_ANIMALS> animal_targets{};
  // Animals that should be placed and serviced on this day.  This may lag the
  // ownership commitment when operating capacity is temporarily more valuable
  // elsewhere, and may fall late when release is economically intentional.
  std::array<int16_t, N_ANIMALS> animal_service_targets{};
  std::array<int16_t, N_CROPS> crop_targets{};
  // Deferred targets remain in plan memory but are not purchased today.
  // This lets a cash shortage postpone a still-profitable project without
  // making it disappear at the next daily replan.
  std::array<int16_t, N_ANIMALS> deferred_animal_targets{};
  std::array<int16_t, N_CROPS> deferred_crop_targets{};
  std::array<int8_t, N_ANIMALS> animal_project_modes{};
  std::array<int8_t, N_CROPS> crop_project_modes{};
  int16_t quadrant_target = 1;
  int16_t deferred_quadrant_target = 1;
  int8_t quadrant_project_mode = INACTIVE;
  int16_t hand_target = 0;
  int16_t wheat_buffer = 0;
  int16_t generated_day = -1;
  double expected_incremental_value = 0.0;
  int8_t bundle_switch_applied = 0;
  double bundle_predicted_gain = 0.0;
  // A SWITCH is a local project edit, not permission to rebuild the whole
  // portfolio differently on the next day.  Preserve the two edited targets
  // explicitly so AdaptivePlannerState can optionally keep just this decision
  // stable through a short, searchable execution window.  All unrelated
  // projects remain reactive.
  int8_t bundle_switch_source = -1;
  int8_t bundle_switch_destination = -1;
  int16_t bundle_switch_source_target = -1;
  int16_t bundle_switch_destination_target = -1;
  // Public/self-state snapshot for offline counterfactual calibration of the
  // analytic SWITCH decision.  It is diagnostic only: the runtime planner
  // never reads these values back and therefore cannot leak future RNG or an
  // opponent identity into its action.
  // 145 core state/plan/opponent fields plus 104 execution-feasibility fields,
  // 29 public opponent/market trend fields, 32 baseline/candidate value
  // decomposition fields, 30 baseline/candidate time-window feasibility fields,
  // 40 baseline/candidate current/future joint-scheduling fields and six
  // continuation-policy context fields:
  // self inventory and carried stock, per-project yield/age/stress/distance,
  // and public committed supply projected 2/4/8 days for both seats.
  static constexpr int SWITCH_FEATURE_DIM = 386;
  std::array<int32_t, SWITCH_FEATURE_DIM> bundle_switch_features{};

  // Public/self state used when Candidate8 is generated.  Candidate-only
  // fields are not enough to rank the same project edit under different
  // cash, market, capacity and deadline conditions.  Keep this compact
  // snapshot diagnostic-only; the runtime executor never reads it back.
  // 101 self/market fields plus 38 public-opponent snapshot/trend fields and
  // 71 deployable opponent-intent fields: exact last-day public deltas,
  // market drift, unit-to-project logistics, estimated sale lead, and public
  // committed supply over 2/4/8-day horizons.  Private opponent shed, seeds,
  // carried inventory, route identity and future actions remain excluded.
  // Opponent identity, private inventory and replay actions are deliberately
  // excluded so a selector trained on this context must generalise by state.
  static constexpr int CANDIDATE8_CONTEXT_FEATURE_DIM = 210;
  std::array<int32_t, CANDIDATE8_CONTEXT_FEATURE_DIM>
      candidate8_context_features{};

  // Sparse generic candidate overlay.  Targets above remain the single
  // economic-plan owner; these fields tell the executor how the selected
  // candidate should schedule, transact, or recover during this plan window.
  int8_t candidate8_family = -1;
  int8_t candidate8_schedule_profile = int8_t(ScheduleProfile::CURRENT);
  int8_t candidate8_market_profile = int8_t(MarketProfile::CURRENT);
  int8_t candidate8_recovery_profile = int8_t(RecoveryProfile::CURRENT);
  int8_t candidate8_market_item = -1;
  uint8_t candidate8_recovery_issue = RECOVERY_NONE;
  std::array<int16_t, ADAPTIVE_PROJECTS> candidate8_target_delta{};
  int8_t candidate8_hand_delta = 0;
  int8_t candidate8_quadrant_delta = 0;
  int8_t candidate8_effective_delay_days = 0;
  int8_t candidate8_suffix_project = -1;
  double candidate8_estimated_value = 0.0;
  double candidate8_estimated_cash_cost = 0.0;
  double candidate8_estimated_daily_action_load = 0.0;
  int16_t candidate8_raw_count = 0;
  int16_t candidate8_feasible_count = 0;
  int16_t candidate8_shortlist_count = 0;
  uint64_t candidate8_signature = 0;
};

// Searchable semantic operating plan used as the FOLLOW lower bound.  It is
// loaded from an audited plan artifact at executor construction time; no raw
// replay actions, author identity, future random events, or coordinates enter
// the native planner.
struct AdaptiveBackbone {
  bool enabled = false;
  std::array<int16_t, 30> hand_target{};
  std::array<int16_t, 30> quadrant_target{};
  std::array<std::array<int16_t, N_ANIMALS>, 30> animal_targets{};
  std::array<std::array<int16_t, N_ANIMALS>, 30> animal_service_targets{};
  std::array<std::array<int16_t, N_CROPS>, 30> crop_targets{};
  // Daily *flow* obligations are distinct from end-of-day stock targets.  A
  // route may harvest and replant ten cells while ending the day with exactly
  // the same crop count.  Keeping only crop_targets silently erased that
  // productive turnover and made high-value semantic plans inexpressible.
  std::array<std::array<int16_t, N_CROPS>, 30> crop_plant{};
  std::array<std::array<int16_t, N_CROPS>, 30> crop_water{};
  std::array<std::array<int16_t, N_CROPS>, 30> crop_harvest{};
  std::array<std::array<int16_t, N_CROPS>, 30> crop_fertilize{};
  std::array<std::array<int16_t, N_CROPS>, 30> crop_harvest_min_yield{};
  std::array<int16_t, 30> crop_clear{};
  std::array<std::array<int16_t, N_ANIMALS>, 30> animal_feed{};
  std::array<std::array<int16_t, N_ANIMALS>, 30> animal_care{};
  std::array<std::array<int16_t, N_ANIMALS>, 30> animal_product{};
  std::array<std::array<int16_t, N_ANIMALS>, 30> animal_fertilizer{};
  std::array<int16_t, 30> wheat_buffer{};
  std::array<int16_t, 30> sell_first_hour{};
  std::array<int16_t, 30> sell_last_hour{};
  std::array<int16_t, 30> buy_first_hour{};
  std::array<int16_t, 30> buy_last_hour{};
  int16_t liquidation_start_step = 648;
};

struct AdaptiveTask {
  Action action{};
  Position target{};
  int priority = 0;
  int deadline_step = 719;
  int required_item = -1;
  int reservation_key = -1;
  // Realized cash unlocked by completing this atomic step, and value that is
  // put at risk if it misses its deadline.  These are scheduling signals; the
  // project planner remains the owner of long-horizon project selection.
  double expected_cash_gain = 0.0;
  double loss_if_delayed = 0.0;
};

// Persistent, state-derived task memory used by R8.  It records obligations
// and ownership, not a replay action or a fixed worker route.  Nodes disappear
// when the live task no longer exists and may be reassigned at any step.
struct R8TaskNode {
  int32_t id = 0;
  int16_t cell = -1;
  int16_t reservation_key = -1;
  int16_t first_seen_step = -1;
  int16_t last_seen_step = -1;
  int16_t deadline_step = 719;
  int16_t priority = 0;
  int8_t op = int8_t(Op::PASS);
  int8_t semantic_group = -1;
  int8_t assigned_unit = -1;
  int8_t active = 0;
  double expected_cash_gain = 0.0;
  double loss_if_delayed = 0.0;
};

struct AdaptivePlannerState {
  int last_step = -1;
  int last_plan_step = -1;
  int last_shop_count = -1;
  std::array<int32_t, N_PRODUCTS> last_prices{};
  std::array<int32_t, N_PRODUCTS> last_inventory{};
  std::array<double, N_PRODUCTS> observed_daily_drift{};
  // Public opponent trajectory memory.  A single snapshot cannot distinguish
  // active expansion from steady production or liquidation.  These EWMAs are
  // updated once per day from public tiles, workforce, land and cash only.
  int16_t opponent_history_day = -1;
  int16_t opponent_history_samples = 0;
  std::array<int16_t, N_CROPS + N_ANIMALS> opponent_last_project_counts{};
  std::array<int16_t, N_CROPS + N_ANIMALS> opponent_last_project_yield{};
  std::array<int16_t, N_CROPS + N_ANIMALS>
      opponent_project_count_trend_x100{};
  std::array<int16_t, N_CROPS + N_ANIMALS>
      opponent_project_yield_trend_x100{};
  // Exact most-recent public change.  The EWMA above describes direction;
  // these fields preserve discrete expansion, harvest and release events that
  // would otherwise be blurred across several days.
  std::array<int16_t, N_CROPS + N_ANIMALS>
      opponent_recent_project_count_delta{};
  std::array<int16_t, N_CROPS + N_ANIMALS>
      opponent_recent_project_yield_delta{};
  int32_t opponent_last_cash = 0;
  int32_t opponent_cash_trend = 0;
  int32_t opponent_recent_cash_delta = 0;
  int16_t opponent_last_hands = 0;
  int16_t opponent_last_quadrants = 1;
  int16_t opponent_hands_trend_x100 = 0;
  int16_t opponent_quadrants_trend_x100 = 0;
  int16_t opponent_recent_hands_delta = 0;
  int16_t opponent_recent_quadrants_delta = 0;
  int flow_day = -1;
  std::array<int16_t, N_CROPS> crop_plant_actions{};
  std::array<int16_t, N_CROPS> crop_water_actions{};
  std::array<int16_t, N_CROPS> crop_harvest_actions{};
  std::array<int16_t, N_CROPS> crop_fertilize_actions{};
  int16_t crop_clear_actions = 0;
  std::array<int16_t, N_ANIMALS> animal_feed_actions{};
  std::array<int16_t, N_ANIMALS> animal_care_actions{};
  std::array<int16_t, N_ANIMALS> animal_product_actions{};
  std::array<int16_t, N_ANIMALS> animal_fertilizer_actions{};
  AdaptivePlan plan{};
  std::vector<int16_t> sticky_target;
  // Last physical task cell owned by each unit.  reservation_key alone only
  // preserves a multi-action chain on one exact tile; this anchor lets the
  // scheduler finish nearby work before sending the unit across the farm.
  std::vector<int16_t> sticky_cell;
  std::vector<int8_t> sticky_op;
  // Soft within-day industry affinity.  Unlike a fixed territory this can be
  // preempted by hard deadlines; it merely helps one worker finish a coherent
  // crop/animal chain before switching to another production system.
  std::vector<int8_t> sticky_group;
  std::vector<int16_t> region_anchor_cell;
  std::vector<int8_t> region_owner_by_cell;
  int16_t region_anchor_day = -1;
  // -1 undecided, 0 keep the current macro backbone, 1 enable the compatible
  // public-state branch.  Decisions are sticky so a short-lived price move
  // cannot oscillate the whole plan every step.
  int8_t animal_branch = -1;
  int8_t crop_suffix = -1;
  // -2 means the general selector has not run, -1 means the autonomous plan,
  // and non-negative values address one loaded semantic operating prior.
  int8_t operating_prior_index = -2;
  int16_t operating_prior_selection_day = -1;
  int operating_prior_switches = 0;
  int bundle_switches = 0;
  double bundle_predicted_gain = 0.0;
  int16_t first_bundle_switch_day = -1;
  int16_t last_bundle_switch_day = -1;
  // Exact target memory for the two projects touched by the most recent local
  // portfolio edit.  This prevents a 2-slot SWITCH from silently becoming an
  // 8-slot whole-plan change at the immediately following daily replan without
  // freezing the farm for the full switch cooldown.  Irreversible live
  // commitments always override these temporary targets.
  int8_t bundle_locked_source = -1;
  int8_t bundle_locked_destination = -1;
  int16_t bundle_locked_source_target = -1;
  int16_t bundle_locked_destination_target = -1;
  int16_t bundle_lock_until_day = -1;
  int candidate8_decisions = 0;
  int16_t first_candidate8_day = -1;
  int8_t first_candidate8_family = -1;
  uint64_t first_candidate8_signature = 0;
  int16_t first_candidate8_raw_count = 0;
  int16_t first_candidate8_feasible_count = 0;
  int16_t first_candidate8_shortlist_count = 0;
  // Sparse macro edits selected by Candidate8 must survive ordinary
  // day/shop/price replans.  Absolute targets avoid applying the same delta a
  // second time after newly purchased or planted assets enter live state.
  // -1 leaves the corresponding project under the ordinary planner.
  std::array<int16_t, ADAPTIVE_PROJECTS> candidate8_persistent_target{
      -1, -1, -1, -1, -1, -1, -1, -1};
  int16_t candidate8_persistent_hand_target = -1;
  int16_t candidate8_persistent_quadrant_target = -1;
  int8_t candidate8_persistent_schedule_profile =
      int8_t(ScheduleProfile::CURRENT);
  std::array<int16_t, ADAPTIVE_PROJECTS> first_candidate8_target_delta{};
  int8_t first_candidate8_hand_delta = 0;
  int8_t first_candidate8_quadrant_delta = 0;
  int8_t first_candidate8_effective_delay_days = 0;
  int8_t first_candidate8_schedule_profile = 0;
  int8_t first_candidate8_market_profile = 0;
  int8_t first_candidate8_recovery_profile = 0;
  int8_t first_candidate8_suffix_project = -1;
  int8_t first_candidate8_market_item = -1;
  uint8_t first_candidate8_recovery_issue = RECOVERY_NONE;
  double first_candidate8_estimated_value = 0.0;
  double first_candidate8_estimated_cash_cost = 0.0;
  double first_candidate8_estimated_daily_action_load = 0.0;
  std::array<int32_t, AdaptivePlan::CANDIDATE8_CONTEXT_FEATURE_DIM>
      first_candidate8_context_features{};
  std::array<int32_t, AdaptivePlan::SWITCH_FEATURE_DIM>
      first_bundle_switch_features{};
  int replans = 0;
  int override_actions = 0;
  int avoidable_crop_losses = 0;
  int avoidable_animal_losses = 0;

  // R8 24-step task-level plan and rolling 4-8-step joint-dispatch state.
  // The day plan is a live obligation ledger.  It is rebuilt at a day boundary
  // and incrementally reconciled on every following public state.
  int16_t r8_day_plan_day = -1;
  int16_t r8_day_plan_start_step = -1;
  int16_t r8_day_plan_end_step = -1;
  int32_t r8_next_task_id = 1;
  std::vector<R8TaskNode> r8_task_nodes;
  int r8_day_plan_rebuilds = 0;
  int r8_rolling_updates = 0;
  int r8_task_nodes_created = 0;
  int r8_task_reassignments = 0;
  int r8_joint_matches = 0;
  int r8_lookahead_evaluations = 0;
  int r8_idle_unit_actions = 0;
  int r8_peak_active_tasks = 0;
  int r8_move_unit_actions = 0;
  int r8_resolved_task_nodes = 0;
  int r8_unresolved_hard_day_tasks = 0;
  int r8_duplicate_reservation_violations = 0;
  std::array<int32_t, 25> r8_resolution_latency_histogram{};

  void reset();
};

struct AdaptiveMatchResult {
  std::array<double, 2> rewards{};
  std::array<int16_t, N_ANIMALS> max_animal_targets{};
  std::array<int16_t, N_CROPS> max_crop_targets{};
  std::array<int16_t, N_ANIMALS> final_animals{};
  int candidate_seat = 0;
  int replans = 0;
  int override_actions = 0;
  int avoidable_crop_losses = 0;
  int avoidable_animal_losses = 0;
  int end_overflow = 0;
  int final_operating_prior_index = -2;
  int operating_prior_switches = 0;
  int bundle_switches = 0;
  double bundle_predicted_gain = 0.0;
  int first_bundle_switch_day = -1;
  int candidate8_decisions = 0;
  int first_candidate8_day = -1;
  int first_candidate8_family = -1;
  uint64_t first_candidate8_signature = 0;
  int first_candidate8_raw_count = 0;
  int first_candidate8_feasible_count = 0;
  int first_candidate8_shortlist_count = 0;
  std::array<int16_t, ADAPTIVE_PROJECTS> first_candidate8_target_delta{};
  int first_candidate8_hand_delta = 0;
  int first_candidate8_quadrant_delta = 0;
  int first_candidate8_effective_delay_days = 0;
  int first_candidate8_schedule_profile = 0;
  int first_candidate8_market_profile = 0;
  int first_candidate8_recovery_profile = 0;
  int first_candidate8_suffix_project = -1;
  int first_candidate8_market_item = -1;
  int first_candidate8_recovery_issue = RECOVERY_NONE;
  double first_candidate8_estimated_value = 0.0;
  double first_candidate8_estimated_cash_cost = 0.0;
  double first_candidate8_estimated_daily_action_load = 0.0;
  std::array<int32_t, AdaptivePlan::CANDIDATE8_CONTEXT_FEATURE_DIM>
      first_candidate8_context_features{};
  std::array<int32_t, AdaptivePlan::SWITCH_FEATURE_DIM>
      first_bundle_switch_features{};
  int r8_day_plan_rebuilds = 0;
  int r8_rolling_updates = 0;
  int r8_task_nodes_created = 0;
  int r8_task_reassignments = 0;
  int r8_joint_matches = 0;
  int r8_lookahead_evaluations = 0;
  int r8_idle_unit_actions = 0;
  int r8_peak_active_tasks = 0;
  int r8_move_unit_actions = 0;
  int r8_resolved_task_nodes = 0;
  int r8_resolution_latency_p50 = 0;
  int r8_resolution_latency_p95 = 0;
  int r8_unresolved_hard_day_tasks = 0;
  int r8_duplicate_reservation_violations = 0;
  std::vector<std::array<PlayerAction, 2>> trace;
  std::vector<std::pair<int, AdaptivePlan>> plan_trace;
  std::vector<std::pair<int, int>> operating_prior_trace;
};

struct AdaptivePortfolioCounterfactualResult {
  bool decision_found = false;
  int decision_step = -1;
  int arms = 0;
  int future_count = 0;
  // Row-major [future, arm], arm zero is KEEP and arm n+1 is analytic rank n.
  std::vector<double> own_rewards;
  std::vector<int32_t> end_overflow;
  std::vector<int8_t> arm_available;
  std::vector<std::array<int32_t, AdaptivePlan::SWITCH_FEATURE_DIM>>
      arm_features;
};

struct AdaptiveCandidate8CounterfactualResult {
  static constexpr int CANDIDATE_FEATURE_DIM = 20;
  // Canonical, non-clairvoyant previews at 24 steps, 48 steps and terminal.
  // Each
  // horizon contains 40 realised-state fields and 36 cumulative execution
  // fields. The preview uses a fixed synthetic future RNG and a PASS
  // opponent, never the counterfactual label bank or the opponent route.
  static constexpr int CONSEQUENCE_FEATURE_DIM = 228;
  // Five fixed, identity-free rival continuations: PASS, balanced autonomous,
  // public-state expansion, immediate public-supply sale, and demand hold.
  // Raw features are retained per scenario so Python can audit and aggregate
  // mean/lower-tail/worst/variance without hiding a favourable weighting.
  static constexpr int RESPONSE_SCENARIO_COUNT = 5;
  static constexpr int RESPONSE_OUTCOME_DIM = 9;  // own/opp/margin x 3 horizons
  bool decision_found = false;
  int decision_step = -1;
  int arms = 0;
  int future_count = 0;
  // Row-major [future, arm].  Every arm starts from the exact same checkpoint;
  // only future official shop/weed RNG is resampled.
  std::vector<double> own_rewards;
  std::vector<double> opponent_rewards;
  std::vector<int32_t> end_overflow;
  std::vector<int8_t> arm_family;
  std::vector<uint64_t> arm_signature;
  std::vector<std::array<int32_t, CANDIDATE_FEATURE_DIM>> arm_features;
  std::vector<std::array<int32_t, CONSEQUENCE_FEATURE_DIM>>
      arm_consequence_features;
  // Flattened row-major [arm, scenario].  These are populated only when the
  // caller explicitly requests the O1.5 response-scenario preview.
  std::vector<std::array<int32_t, CONSEQUENCE_FEATURE_DIM>>
      arm_response_scenario_features;
  std::vector<std::array<int32_t, RESPONSE_OUTCOME_DIM>>
      arm_response_scenario_outcomes;
  // Offline effectiveness labels from future zero.  Each arm is compared
  // with KEEP from the exact same checkpoint and future event stream.  These
  // labels are never exposed to the runtime planner; a deployable model must
  // infer them from public candidate/context features.
  std::vector<int16_t> arm_first_action_change_offset;
  std::vector<int16_t> arm_action_change_count_24;
  std::vector<int16_t> arm_action_change_count_full;
  std::vector<int16_t> arm_first_state_change_offset;
  std::vector<int16_t> arm_state_change_count_24;
  std::vector<int16_t> arm_state_change_count_full;
  std::array<int32_t, AdaptivePlan::CANDIDATE8_CONTEXT_FEATURE_DIM>
      context_features{};
};

// Offline receding-horizon upper-bound audit.  At every requested day the
// complete feasible Candidate8 pool is evaluated with common, independently
// reseeded future event streams.  The best expected-cash arm is committed to
// the real trajectory, which is then advanced to the next requested day.
// This deliberately has access to far more simulation than an online policy;
// it measures candidate/executor headroom, not deployable playing strength.
struct AdaptiveCandidate8RollingOracleResult {
  std::array<double, 2> rewards{};
  int candidate_seat = 0;
  int end_overflow = 0;
  int avoidable_crop_losses = 0;
  int avoidable_animal_losses = 0;
  int complete_continuations = 0;
  std::vector<int16_t> decision_day;
  std::vector<int16_t> decision_step;
  std::vector<int16_t> feasible_count;
  std::vector<int16_t> selected_rank;
  std::vector<int8_t> selected_family;
  std::vector<uint64_t> selected_signature;
  std::vector<double> selected_expected_reward;
  std::vector<double> keep_expected_reward;
  std::vector<double> stage_expected_gain;
  std::vector<double> selected_expected_win_rate;
  std::vector<double> keep_expected_win_rate;
  std::vector<double> selected_expected_margin;
  std::vector<double> keep_expected_margin;
};

// Exact-future beam search over a sequence of Candidate8 edits.  Unlike the
// rolling Oracle above, this keeps several temporarily inferior prefixes alive
// so a later complementary edit can make the combined sequence competitive.
// It is an offline representational audit and is never reachable from a saved
// runtime genome.
struct AdaptiveCandidate8SequenceOracleResult {
  std::array<double, 2> rewards{};
  int candidate_seat = 0;
  int complete_continuations = 0;
  int expanded_nodes = 0;
  int maximum_live_beam = 0;
  // Terminal outcomes for every child expanded at the final decision depth,
  // before beam pruning.  These are complete decision-day sequences; unlike
  // complete_continuations they exclude terminal rollouts of shorter prefixes.
  std::vector<double> final_path_candidate_rewards;
  std::vector<double> final_path_opponent_rewards;
  int final_path_sequence_length = 0;
  std::vector<int16_t> final_path_selected_ranks;
  std::vector<int8_t> final_path_selected_families;
  std::vector<int16_t> decision_day;
  std::vector<int16_t> selected_rank;
  std::vector<int8_t> selected_family;
  std::vector<uint64_t> selected_signature;
  std::vector<int16_t> feasible_count;
};

// Fixed-budget open-loop MCTS over the same public Candidate8 edits used by
// candidate8_sequence_oracle().  Every simulation produces one complete
// terminal continuation.  The tree allocates later simulations adaptively;
// unsaved suffixes use a deterministic seeded rollout policy.  This remains an
// exact-future offline search audit and is never reachable from a runtime
// genome or submission policy.
struct AdaptiveCandidate8MctsResult {
  std::array<double, 2> rewards{};
  int candidate_seat = 0;
  int simulations = 0;
  int tree_nodes = 0;
  int maximum_depth = 0;
  int first_win_simulation = -1;
  int best_found_simulation = -1;
  int winning_simulations = 0;
  int unique_sampled_paths = 0;
  std::vector<int16_t> decision_day;
  std::vector<int16_t> selected_rank;
  std::vector<int8_t> selected_family;
  std::vector<uint64_t> selected_signature;
  std::vector<int16_t> feasible_count;
  std::vector<int16_t> root_rank;
  std::vector<int32_t> root_visits;
  std::vector<double> root_mean_value;
  std::vector<double> sampled_candidate_rewards;
  std::vector<double> sampled_opponent_rewards;
};

// Executes a caller-selected sequence of public Candidate8 ranks without any
// future lookahead.  This is the evaluation bridge for learned Day0/Day1
// selectors: the caller may inspect public candidate features, choose ranks,
// and then verify the committed sequence on an untouched actual seed.
struct AdaptiveCandidate8CommittedSequenceResult {
  std::array<double, 2> rewards{};
  int candidate_seat = 0;
  int end_overflow = 0;
  int avoidable_crop_losses = 0;
  int avoidable_animal_losses = 0;
  std::vector<int16_t> decision_day;
  std::vector<int16_t> selected_rank;
  std::vector<int8_t> selected_family;
  std::vector<uint64_t> selected_signature;
  std::vector<uint8_t> selected_matched;
  std::vector<AdaptivePlanDelta> selected_delta;
  std::vector<std::array<PlayerAction, 2>> trace;
};

class NativeAdaptivePlanner {
 public:
  explicit NativeAdaptivePlanner(AdaptiveGenome genome = {},
                                 std::vector<AdaptiveBackbone> backbones = {},
                                 bool force_backbone_exact = false)
      : genome_(genome), backbones_(std::move(backbones)),
        force_backbone_exact_(force_backbone_exact) {}

  PlayerAction action(const Simulator& env, int player,
                      AdaptivePlannerState& state) const;
  const AdaptiveGenome& genome() const { return genome_; }

 private:
  struct PlanValueBreakdown {
    double crop_gross = 0.0;
    double animal_gross = 0.0;
    double fertilizer_gross = 0.0;
    double seed_cost = 0.0;
    double feed_cost = 0.0;
    double animal_purchase_cost = 0.0;
    double action_cost = 0.0;
    double move_cost = 0.0;
    double hire_cost = 0.0;
    double land_cost = 0.0;
    double lockup_cost = 0.0;
    double crop_units = 0.0;
    double animal_product_units = 0.0;
    double fertilizer_used = 0.0;
    double fertilizer_sellable = 0.0;
    double setup_turns = 0.0;
    // Generic 2/4/8-day execution and cash-conversion projections.  These are
    // derived from official production lead times and the current public/self
    // state; no replay calendar or opponent identity is used.
    double first_cash_lag = 30.0;
    double immediate_commitment_actions = 0.0;
    double today_action_capacity = 0.0;
    double today_deadline_slack = 0.0;
    double peak_daily_utilization_x100 = 0.0;
    std::array<double, 3> window_action_demand{};
    std::array<double, 3> window_action_capacity{};
    std::array<double, 3> window_net_cash{};
    double minimum_window_slack = 0.0;
    // Current-state execution projection produced by the same task generator
    // and joint scheduler used at runtime.  It measures whether the candidate
    // can be started without hiding its logistics behind an aggregate workload
    // estimate.  These fields are diagnostics for offline candidate ranking;
    // they do not alter the atomic action selected by the scheduler.
    double current_task_count = 0.0;
    double current_hard_task_count = 0.0;
    double assigned_task_count = 0.0;
    double assigned_hard_task_count = 0.0;
    double total_assignment_distance = 0.0;
    double total_assignment_steps = 0.0;
    double minimum_assignment_slack = 0.0;
    double deadline_infeasible_task_count = 0.0;
    double unassigned_hard_task_count = 0.0;
    double assigned_expected_cash_gain = 0.0;
    double unassigned_delayed_loss = 0.0;
    double assigned_value_per_step = 0.0;
    // Synthetic commissioning chains for resources not yet purchased.  The
    // current task queue cannot contain them, but they still consume worker
    // lanes before the first possible cash return.
    double commissioning_job_count = 0.0;
    double commissioning_total_steps = 0.0;
    double commissioning_makespan_steps = 0.0;
    double commissioning_first_cash_slack_steps = 0.0;
    double commissioning_unreachable_count = 0.0;
    double new_service_distance = 0.0;
    double new_service_route_span = 0.0;
    double commissioning_steps_per_added_unit = 0.0;
  };

  AdaptiveGenome genome_;
  // One plan remains fully backwards compatible.  Four plans use the audited
  // latest-submission order: cow/carrot, cow/wheat, sheep/carrot,
  // sheep/wheat.  The public-state branch flags select the active prior; the
  // economic planner still rebuilds its project portfolio from live state.
  std::vector<AdaptiveBackbone> backbones_;
  // Offline O1.7 attribution only.  A normal backbone is deliberately a live
  // economic lower bound, so the marginal allocator may expand it.  The
  // forced semantic test needs the supplied targets to be both floor and cap
  // in order to isolate candidate generation from plan execution.  The flag
  // is never enabled by play(), Candidate8, R8, or a submission path.
  bool force_backbone_exact_ = false;

  const AdaptiveBackbone& active_backbone(
      const AdaptivePlannerState& state) const;

  bool should_replan(const Simulator& env, int player,
                     const AdaptivePlannerState& state) const;
  AdaptivePlan build_plan(const Simulator& env, int player,
                           const AdaptivePlannerState& state) const;
  double score_plan_candidate(const Simulator& env, int player,
                               const AdaptivePlan& plan,
                               PlanValueBreakdown* breakdown = nullptr,
                               const AdaptivePlannerState* planner_state = nullptr) const;
  std::vector<AdaptiveTask> build_tasks(const Simulator& env, int player,
                                        const AdaptivePlan& plan,
                                        const AdaptivePlannerState& state) const;
  std::vector<Action> assign_tasks(const Simulator& env, int player,
                                   const std::vector<AdaptiveTask>& tasks,
                                   AdaptivePlannerState& state) const;
  std::vector<Action> market_orders(const Simulator& env, int player,
                                     const AdaptivePlan& plan,
                                     const AdaptivePlannerState& state,
                                     const std::vector<Action>& unit_actions) const;
};

// Runs the dynamic planner entirely inside the native 719-step hot loop.  The
// opponent is one frozen route-family executor; its identity is not passed to
// the planner and therefore cannot be used for opponent-specific routing.
class NativeAdaptiveExecutor {
 public:
  explicit NativeAdaptiveExecutor(NativeTapeLibrary library,
                                  std::vector<AdaptiveBackbone> backbones = {})
      : route_executor_(std::move(library)), backbones_(std::move(backbones)) {}

  AdaptiveMatchResult play(const AdaptiveGenome& genome, int opponent_route,
                           uint64_t seed, int candidate_seat,
                           bool capture_trace = false) const;
  AdaptiveMatchResult play_forced_backbone(
      const AdaptiveGenome& genome, int opponent_route, uint64_t seed,
      int candidate_seat, bool capture_trace = false) const;
  // Diagnostic layer ablation.  The base route is fixed before the match and
  // is never selected from opponent identity.  Modes: 0=fully adaptive,
  // 1=base unit actions + adaptive market, 2=adaptive unit actions + base
  // market, 3=fully base, 4/5=offline WHEAT-shuttle ablations, and 6=execute
  // a configurable prefix of the base route before handing the settled state
  // to the adaptive planner.  Mode 6 is an offline opening ablation;
  // it does not initialize adaptive state from an action the planner did not
  // execute.  These modes isolate execution/scheduling losses without
  // changing the production policy API.
  AdaptiveMatchResult play_blend(const AdaptiveGenome& genome, int base_route,
                                 int opponent_route, uint64_t seed,
                                 int candidate_seat, int blend_mode,
                                 bool capture_trace = false,
                                 int prefix_steps = 1,
                                 bool force_backbone_exact = false) const;
  // Offline W3 oracle: freeze one public decision state, fork several legal
  // project edits, and average their continuation over resampled future event
  // seeds.  This is never part of online action selection.
  AdaptivePortfolioCounterfactualResult portfolio_counterfactual(
      const AdaptiveGenome& genome, int opponent_route, uint64_t prefix_seed,
      const std::vector<uint64_t>& future_seeds, int candidate_seat,
      int candidate_ranks, double switch_margin,
      int minimum_decision_day = 0) const;
  // Execute one real broad PlanDelta arm.  Rank addresses the deterministic
  // max-64 shortlist at the first eligible replan; an unavailable rank is a
  // safe KEEP.  This is the bridge used to produce C++ continuation labels.
  AdaptiveMatchResult play_candidate8(
      const AdaptiveGenome& genome, int opponent_route, uint64_t seed,
      int candidate_seat, int candidate_rank, int minimum_decision_day = 0,
      bool use_feasible_pool = false, bool capture_trace = false) const;
  // Multi-future Candidate8 oracle.  This removes the single-future
  // multiple-comparison bias: candidates are ranked by expected continuation
  // from one frozen visible state, not by luck in one hidden future event
  // sequence.  Offline-only; runtime policy cannot access future seeds.
  AdaptiveCandidate8CounterfactualResult candidate8_counterfactual(
      const AdaptiveGenome& genome, int opponent_route, uint64_t prefix_seed,
      const std::vector<uint64_t>& future_seeds, int candidate_seat,
      int minimum_decision_day = 0, bool use_feasible_pool = true,
      int maximum_arms = 4096,
      const std::vector<int>& committed_days = {},
      const std::vector<int>& committed_ranks = {},
      bool include_response_scenarios = false) const;
  // Greedy multi-stage Oracle used only to answer whether the expanded
  // candidate language and executor can reach a target cash ceiling.  Future
  // banks used for arm selection are independent from the actual match seed.
  AdaptiveCandidate8RollingOracleResult candidate8_rolling_oracle(
      const AdaptiveGenome& genome, int opponent_route, uint64_t actual_seed,
      const std::vector<int>& decision_days, uint64_t future_seed_base,
      int future_count, int candidate_seat,
      bool use_feasible_pool = true, int maximum_arms = 4096,
      bool clairvoyant_actual_future = false,
      bool competitive_objective = false) const;
  AdaptiveCandidate8SequenceOracleResult candidate8_sequence_oracle(
      const AdaptiveGenome& genome, int opponent_route, uint64_t actual_seed,
      const std::vector<int>& decision_days, int candidate_seat,
      int beam_width = 32, int per_node_arms = 64,
      bool use_feasible_pool = true,
      bool competitive_objective = true,
      const std::vector<int>& committed_days = {},
      const std::vector<std::vector<int>>& committed_rank_sequences = {},
      int prefix_route = -1, int prefix_steps = 0) const;
  AdaptiveCandidate8MctsResult candidate8_mcts_oracle(
      const AdaptiveGenome& genome, int opponent_route, uint64_t actual_seed,
      const std::vector<int>& decision_days, int candidate_seat,
      int simulation_budget = 12000, int per_node_arms = 64,
      bool use_feasible_pool = false,
      bool competitive_objective = true,
      double exploration_constant = 1.25,
      double progressive_widening_constant = 2.0,
      double progressive_widening_alpha = 0.5,
      int rollout_arms = 8, uint64_t search_seed = 1,
      int prefix_route = -1, int prefix_steps = 0) const;
  AdaptiveCandidate8CommittedSequenceResult candidate8_committed_sequence(
      const AdaptiveGenome& genome, int opponent_route, uint64_t actual_seed,
      const std::vector<int>& decision_days,
      const std::vector<int>& selected_ranks, int candidate_seat,
      bool use_feasible_pool = false, int prefix_route = -1,
      int prefix_steps = 0, bool capture_trace = false,
      const std::vector<AdaptivePlanDelta>& selected_deltas = {}) const;
  int route_count() const { return route_executor_.route_count(); }

 private:
  NativeTeammateExecutor route_executor_;
  std::vector<AdaptiveBackbone> backbones_;
};

}  // namespace fastkag
