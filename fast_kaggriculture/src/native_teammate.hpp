// Licensed under the Apache License, Version 2.0.
#pragma once

#include "repair.hpp"
#include "reactive_production.hpp"
#include "../../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/include/event_triggered_local_repair.hpp"
#include "../../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/include/day_horizon_planner.hpp"
#include "../../native_deps/findNewRoad/repairMechanismCpp/failureDebtSchedulerCpp/include/deferred_crop_scheduler.hpp"
#include "../../native_deps/findNewRoad/repairMechanismCpp/purchaseRecoveryLedgerCpp/include/purchase_recovery_ledger.hpp"
#include "native_general_market.hpp"
#include "native_phased_market.hpp"
#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace fastkag {

// Offline-only native executor for the frozen teammate route stack.  The
// submission remains Python; this class exists so counterfactual route search
// can keep the complete 719-turn inner loop in C++.
struct NativeTapeLibrary {
  struct ThomasPredictEvent {
    int16_t step{};
    int8_t item{};
    int16_t quantity{};
  };
  std::vector<std::vector<PlayerAction>> routes;
  std::vector<PlayerAction> r5_reference;
  std::vector<PlayerAction> md_reference;
  // Five current and five legacy Moon tapes, in K320 label order.
  std::array<std::vector<PlayerAction>, 5> moon{};
  std::array<std::vector<PlayerAction>, 5> moon_legacy{};
  // Optional offline Thomas V92 predictor streams, indexed by first-two-shop
  // pair (shop0 * 8 + shop1). Empty for every normal executor construction.
  std::array<std::vector<std::vector<ThomasPredictEvent>>, 64>
      thomas_predict_pairs{};
};

struct NativeAgentState {
  struct StationaryObligationSubmission {
    int actor{-1};
    int source_step{-1};
    Position tile{};
    Action intended{};
  };
  struct WeedRepair {
    bool active = false;
    int start = -1;
    Action intended{};
  };
  struct UnitRealignment {
    bool active = false;
    int start_step = -1;
    int skipped_source_step = -1;
    // DIG for a weed transaction, PASS for a delayed animal PICKUP.
    Action inserted{};
  };
  struct SuppressedUnitSource {
    int actor = -1;
    int source_step = -1;
  };
  struct RoomEvac {
    bool active = false;
    int actor = -1;
    Position target{};
    int day = -1;
  };
  struct Salvage {
    bool active = false;
    int actor = -1;
    Position target{};
    Item product = Item::NONE;
    int quantity = 0;
  };
  struct MoonRace {
    int last_step = -1;
    std::array<int, N_PRODUCTS> inventory{};
    std::array<int, N_PRODUCTS> prices{};
    std::array<int, 4> own_sells{};
    std::vector<int8_t> shops;
    std::array<std::array<double, 6>, 4> scores{};
    std::array<double, 4> evidence{};
    std::array<int, 4> horizon{1, 1, 1, 1};
    std::array<double, 6> policy_scores{};
    double policy_evidence = 0.0;
    int policy_horizon = 1;
  };
  struct PendingPurchaseReceipt {
    int market_slot{-1};
    Op operation{Op::PASS};
    Item item{Item::NONE};
    int requested{};
    int submitted_step{-1};
    std::vector<int> debt_tiles;
  };
  struct PendingRouteCursorCommit {
    bool active{};
    bool exact_production_manifest{};
    int submitted_step{-1};
    joint_fixed_move_oracle::production::RouteCursorProposal proposal;
    // Production-free cursor transitions used when only a subset of actors'
    // receipt effects are confirmed.
    joint_fixed_move_oracle::production::RouteCursorProposal source_base;
    joint_fixed_move_oracle::production::RouteCursorProposal source_final;
    std::vector<Action> final_units;
    std::vector<Position> positions_before;
    std::vector<Position> positions_expected;
    std::vector<Tile> tiles_before;
    std::vector<Tile> tiles_expected;
    std::vector<int> inventory_before;
  };
  struct PendingDeferredCropReceipt {
    bool active{};
    int submitted_step{-1};
    g001::failure_debt::deferred_crop::Proposal proposal;
    g001::failure_debt::deferred_crop::CropSnapshot before;
    Action emitted{};
    int crop_inventory_before{};
    int fertilizer_inventory_before{};
    bool day_end_water_effect_lower_bound{};
    std::uint64_t movement_hash{};
  };
  struct PendingEventLocalReceipt {
    std::uint64_t decision_id{};
    std::uint64_t transaction_id{};
    int actor{-1};
    std::uint64_t actor_generation{};
    Position tile{};
    Item desired_item{Item::NONE};
    int desired_inventory_before{};
    Tile before{};
    Op emitted{Op::PASS};
    int submitted_step{-1};
  };
  struct EventLocalPurchaseLink {
    std::uint64_t transaction_id{};
    std::uint64_t debt_id{};
    Item item{Item::NONE};
  };

  int last_step = -1;
  std::vector<WeedRepair> weed;
  std::vector<UnitRealignment> experimental_realign;
  std::vector<SuppressedUnitSource> experimental_suppressed;
  // Day-start declarations for the independent obligation-day fork. Native
  // never schedules or rewrites these; it only transfers stationary weed
  // work to the external owner.
  std::vector<StationaryObligationSubmission>
      experimental_stationary_obligations;
  g001::repair::AnimalRetryController experimental_animals{};
  std::array<int, 3> experimental_previous_placed{};
  bool experimental_animals_initialized = false;
  std::array<bool, 3> experimental_animal_retry_awaiting{};
  std::array<int, 4> k320_due{};
  bool r5_target = false;
  bool md_target = false;
  RoomEvac room_evac{};
  int wheat_credit = 0;
  Salvage salvage{};
  // Moon debts indexed by due step and premium product.
  std::array<std::array<int, 4>, 720> moon_debts{};
  MoonRace moon_race{};
  int moon_layout = -1;  // -1 undecided, 0 current, 1 legacy
  NativePhasedRuntimeState phased_market{};
  joint_fixed_move_oracle::production::RouteCursorState
      experimental_route_cursor{};
  joint_fixed_move_oracle::production::RouteCursorAudit
      experimental_route_cursor_audit{};
  PendingRouteCursorCommit experimental_route_cursor_pending{};
  std::optional<g001::failure_debt::deferred_crop::DeferredCropScheduler>
      experimental_deferred_crop_scheduler;
  std::vector<PendingDeferredCropReceipt> experimental_deferred_crop_pending;
  std::set<std::pair<int, int>> experimental_deferred_crop_enqueued_sources;
  // Shared exact-receipt ledger for seed and animal acquisition debt.  Crop
  // recovery is the first producer; animal recovery can use the same ABI.
  std::vector<PendingPurchaseReceipt> experimental_purchase_receipts;
  std::optional<g001::event_local_repair::Compiler>
      experimental_event_local_repair;
  int experimental_event_local_day{-1};
  std::vector<Item> experimental_event_local_last_crop;
  std::vector<std::uint64_t> experimental_event_local_actor_generations;
  std::vector<PendingEventLocalReceipt> experimental_event_local_pending;
  g001::purchase_recovery::Ledger experimental_event_local_purchase_ledger;
  std::vector<EventLocalPurchaseLink> experimental_event_local_purchase_links;
  struct DayHorizonObjective {
    std::uint64_t id{};
    Position tile{};
    Item desired{Item::NONE};
    Op source{Op::PASS};
    bool requires_first_yield{};
    bool complete{};
  };
  struct DayHorizonPending {
    std::uint64_t objective_id{};
    int actor{-1};
    Position tile{};
    Op emitted{Op::PASS};
    Tile before{};
    int submitted_step{-1};
  };
  struct DayHorizonRuntime {
    int day{-1};
    int source_step{-1};
    int actor_count{-1};
    bool active{};
    bool failed_closed{};
    int failure_code{};
    std::uint64_t next_objective_id{1};
    std::vector<std::vector<Action>> manifest;
    std::vector<std::vector<std::uint64_t>> objective_bindings;
    std::vector<DayHorizonObjective> objectives;
    std::vector<DayHorizonPending> pending;
  } experimental_day_horizon;
  struct DayHorizonV2Objective {
    std::uint64_t id{};
    Position tile{};
    Item desired{Item::NONE};
    Op source{Op::PASS};
    std::uint64_t source_epoch{};
    bool requires_harvest{};
    bool waiting_maturity{};
    std::vector<g001::event_local_repair::Action> remaining;
  };
  struct DayHorizonV2Pending {
    std::uint64_t objective_id{};
    int transition_index{};
    int actor{-1};
    std::uint64_t actor_generation{};
    Position tile{};
    Op emitted{Op::PASS};
    Tile before{};
    int submitted_step{-1};
    std::optional<g001::day_horizon_repair::HarvestProofToken>
        maturity_proof;
  };
  struct DayHorizonV2PurchaseLink {
    std::uint64_t objective_id{};
    std::uint64_t debt_id{};
    Item item{Item::NONE};
  };
  struct DayHorizonV2Runtime {
    int day{-1};
    int source_step{-1};
    int actor_count{-1};
    bool active{};
    bool resource_changed{};
    std::uint64_t next_objective_id{1};
    std::vector<g001::day_horizon_repair::ActorPlan> actors;
    g001::day_horizon_repair::TimeExpandedResult plan;
    std::vector<DayHorizonV2Objective> objectives;
    std::vector<DayHorizonV2Pending> pending;
    std::set<g001::day_horizon_repair::HarvestProofToken>
        consumed_maturity_proofs;
    g001::purchase_recovery::Ledger purchase_ledger;
    std::vector<DayHorizonV2PurchaseLink> purchase_links;
  } experimental_day_horizon_v2;
  struct RouteSkeletonMove {
    int source_step{-1};
    Action action{};
    // Set only when an accepted v3 manifest intentionally postpones a due
    // MOVE.  A past, unmarked source MOVE occurred while v3 declined the
    // whole proposal and therefore ran byte-identical baseline; it must not
    // be replayed on the next successful proposal.
    bool deferred_by_planner{};
  };
  struct RouteSkeletonActor {
    int actor{-1};
    std::uint64_t generation{};
    std::vector<RouteSkeletonMove> moves;
    std::size_t cursor{};
  };
  struct RouteSkeletonObjective {
    std::uint64_t id{};
    Position tile{};
    Item desired{Item::NONE};
    bool requires_harvest{};
    bool waiting_maturity{};
    std::vector<g001::event_local_repair::Action> remaining;
  };
  struct RouteSkeletonPending {
    std::uint64_t objective_id{};
    int actor{-1};
    std::uint64_t generation{};
    Position tile{};
    Op emitted{Op::PASS};
    Tile before{};
    int submitted_step{-1};
  };
  struct RouteSkeletonRuntime {
    int day{-1};
    bool planned_once{};
    std::uint64_t next_objective_id{1};
    std::vector<RouteSkeletonActor> actors;
    std::vector<RouteSkeletonObjective> objectives;
    std::vector<RouteSkeletonPending> pending;
    std::set<g001::day_horizon_repair::HarvestProofToken>
        consumed_maturity_proofs;
  } experimental_route_skeleton_v3;

  void reset();
};

// Offline-only experiment.  Every switch defaults off so the aligned
// final-G001 provider and submission behavior remain byte-for-byte unchanged.
struct NativeRepairOptions {
  bool weed_min_loss_realign = false;
  bool animal_buy_retry = false;
  bool empty_stall_reuse = false;
  // Development-only observable online production controller.  0 preserves
  // deployed behavior, 1 repairs only route-critical weed collisions, 2 also
  // admits receipt-confirmed missing-seed debt and bounded seed retries.
  int route_cursor_production = 0;
  // Independent final-composer source-ledger experiment.  It never invokes
  // the global RouteCursor and is mutually exclusive with it.
  // Research-only/non-promotable bit-8 receipt-layer experiment. It remains
  // default-off until a day-horizon compiler passes paired economic gates.
  bool state_driven_local_repair = false;
  // Independent experimental full-day compiler. Bit 16; mutually exclusive
  // with bit 8, RouteCursor, and legacy K320 ownership.
  bool day_horizon_repair = false;
  // Post-bit16 experimental replacement. Bit 32 has independent persistent
  // objective, receipt/replan, maturity, and purchase state.
  bool day_horizon_repair_v2 = false;
  // Rolling MOVE-source skeleton with plot-owned production objectives.
  // Independent experimental bit 64.
  bool rolling_route_skeleton_v3 = false;
  // Independent provider-only seam. Deliberately absent from the stable mask:
  // the default/native action ABI remains unchanged unless a fork sets it.
  bool weed_obligation_day_owner = false;
  int minimum_weed_realign_step = 240;
  int maximum_realign_lookahead = 16;
  int stationary_reuse_lookahead = 8;

  [[nodiscard]] bool enabled() const {
    return weed_min_loss_realign || animal_buy_retry || empty_stall_reuse ||
           route_cursor_production != 0 || state_driven_local_repair ||
           day_horizon_repair || day_horizon_repair_v2 ||
           rolling_route_skeleton_v3 || weed_obligation_day_owner;
  }
  [[nodiscard]] static NativeRepairOptions all_enabled() {
    NativeRepairOptions result;
    result.weed_min_loss_realign = true;
    result.animal_buy_retry = true;
    result.empty_stall_reuse = true;
    return result;
  }
};

struct NativeRepairAudit {
  int weed_triggers = 0;
  int weed_absorbed_pass = 0;
  int weed_absorbed_productive = 0;
  int weed_legacy_would_drop_move = 0;
  int weed_no_safe_alignment = 0;
  int movement_edits = 0;
  int animal_original_attempted = 0;
  int animal_inferred_filled = 0;
  int animal_inferred_partial = 0;
  int animal_inferred_zero = 0;
  int animal_retries_emitted = 0;
  int animal_retry_fills = 0;
  int animal_retry_pickup_realign = 0;
  int animal_retry_blocked_no_path = 0;
  int empty_stall_reused = 0;
  int empty_stall_unresolved = 0;
  int route_cursor_inserted_before_move = 0;
  int route_cursor_skipped_nonmoves = 0;
  int route_cursor_forced_moves = 0;
  int route_cursor_weed_events = 0;
  int route_cursor_digs = 0;
  int route_cursor_plants = 0;
  int route_cursor_waters = 0;
  int route_cursor_harvests = 0;
  int route_cursor_abandoned_cycles = 0;
  int route_cursor_completed_cycles = 0;
  int route_cursor_seed_retry_orders = 0;
  int route_cursor_seed_retry_units = 0;
  int route_cursor_seed_retry_fills = 0;
  int route_cursor_proposals = 0;
  int route_cursor_final_staged = 0;
  int route_cursor_overlay_overrides = 0;
  int route_cursor_effect_failures = 0;
  int route_cursor_first_effect_failure_step = -1;
  int route_cursor_first_effect_failure_actor = -1;
  int route_cursor_first_effect_failure_op = -1;
  std::array<int, 24> route_cursor_effect_failures_by_op{};
  int route_cursor_commits = 0;
  int route_cursor_deferred_nonmoves = 0;
  int local_repair_decisions = 0;
  int local_repair_commits = 0;
  int local_repair_receipts_confirmed = 0;
  int local_repair_receipts_failed = 0;
  int local_repair_fail_closed = 0;
  int local_repair_seed_orders = 0;
  int local_repair_seed_units = 0;
  int local_repair_seed_fills = 0;
  int local_repair_seed_zero_fills = 0;
  int local_repair_transaction_claims = 0;
  int local_repair_transaction_claim_rejections = 0;
  int day_horizon_plans = 0;
  int day_horizon_commits = 0;
  int day_horizon_recompiles = 0;
  int day_horizon_fail_closed = 0;
  int day_horizon_assignments = 0;
  int day_horizon_terminal_raw = 0;
  int day_horizon_v2_plans = 0;
  int day_horizon_v2_replans = 0;
  int day_horizon_v2_receipts_confirmed = 0;
  int day_horizon_v2_receipts_failed = 0;
  int day_horizon_v2_fail_closed = 0;
  int day_horizon_v2_assignments = 0;
  int day_horizon_v2_exact_plans = 0;
  int day_horizon_v2_fallback_plans = 0;
  int day_horizon_v2_budget_exhausted = 0;
  int day_horizon_v2_objectives_completed = 0;
  int day_horizon_v2_objectives_carried = 0;
  int day_horizon_v2_maturity_waits = 0;
  int day_horizon_v2_maturity_tokens_consumed = 0;
  int day_horizon_v2_seed_unscheduled = 0;
  int day_horizon_v2_seed_orders = 0;
  int day_horizon_v2_seed_zero_fills = 0;
  int day_horizon_v2_seed_fills = 0;
  int day_horizon_v2_terminal_raw = 0;
  std::array<int, 8> day_horizon_v2_fail_reasons{};
  int route_skeleton_v3_plans = 0;
  int route_skeleton_v3_replans = 0;
  int route_skeleton_v3_rebases = 0;
  int route_skeleton_v3_lcs_kept = 0;
  int route_skeleton_v3_receipts_confirmed = 0;
  int route_skeleton_v3_receipts_failed = 0;
  int route_skeleton_v3_fail_closed = 0;
  int route_skeleton_v3_assignments = 0;
  int route_skeleton_v3_objectives_completed = 0;
  int route_skeleton_v3_objectives_carried = 0;
  int route_skeleton_v3_moves_emitted = 0;
  int route_skeleton_v3_terminal_moves = 0;
  int route_skeleton_v3_fallback_plans = 0;
  int route_skeleton_v3_budget_exhausted = 0;
  std::array<int, 8> route_skeleton_v3_fail_reasons{};
};

// Stable C++/Python experiment mask ABI. Bits 1/2/4 retain their historical
// meanings; bit 8 selects only the research-only final-composer receipt
// ledger, bit 16 selects the rejected/frozen day compiler, and bit 32 selects
// its independently evaluated persistent-suffix successor. None is enabled by
// default or promotable before its own paired gates pass. Bit 64 selects the
// separate rolling route-skeleton v3 experiment.
[[nodiscard]] NativeRepairOptions native_repair_options_from_mask(int mask);

struct NativeMatchResult {
  std::array<double, 2> rewards{};
  std::array<int32_t, 2> macro_unit_failures{};
  std::array<int32_t, 2> macro_market_failures{};
  std::array<int32_t, 2> first_macro_failure_step{-1, -1};
  std::array<int32_t, 2> first200_unit_failures{};
  std::array<int32_t, 2> first200_market_failures{};
  std::vector<std::array<PlayerAction, 2>> trace;
  std::array<NativeRepairAudit, 2> repair_audit{};
  std::array<NativeGeneralMarketAudit, 2> economy_audit{};
  std::array<NativePhasedMarketAudit, 2> phased_market_audit{};
};

struct NativeMovementToken {
  int actor{-1};
  int source_step{-1};
  Action action{};
};

// The current provider can make an immutable statement only about the action
// it has already finalized.  remaining_day_complete is deliberately false
// unless a future provider owns and freezes every later overlay decision.
struct NativeMovementCommitment {
  int player{-1};
  int day{-1};
  std::uint64_t issuer_generation{};
  int issued_step{-1};
  int immutable_through_step{-1};
  bool remaining_day_complete{};
  bool revoked{};
  std::vector<NativeMovementToken> moves;
  std::uint64_t content_hash{};
};

struct NativeCommittedAction {
  PlayerAction action;
  NativeMovementCommitment movement;
};

[[nodiscard]] std::uint64_t native_movement_commitment_hash(
    const NativeMovementCommitment& commitment) noexcept;

class NativeTeammateExecutor {
 public:
  explicit NativeTeammateExecutor(NativeTapeLibrary library)
      : library_(std::move(library)) {}

  NativeMatchResult play(int route0, int route1, uint64_t seed,
                         int switch_step0 = -1, int switch_route0 = -1,
                         int switch_step1 = -1, int switch_route1 = -1,
                         bool capture_trace = false,
                         bool capture_audit = true,
                         bool neutral_special_economy = false,
                         NativeRepairOptions repair_options = {},
                         int repair_player = -1,
                         bool experimental_general_takeover = false,
                         int general_takeover_player = -1,
                         int experimental_market_arm = 0,
                         int experimental_market_player = -1,
                         int stop_after_steps = -1,
                         int evaluation_compaction_minimum_step = -1,
                         int evaluation_compaction_evidence_mode = 0,
                         int thomas_prefix_player = -2) const;
  std::array<float, 147> features_at(int route0, int route1, uint64_t seed,
                                     int checkpoint, int player,
                                     int feature_route) const;
  int route_count() const { return int(library_.routes.size()); }
  [[nodiscard]] const std::vector<PlayerAction>& route_tape(int route) const {
    if (route < 0 || route >= route_count())
      throw std::out_of_range("native route is outside library");
    return library_.routes[static_cast<std::size_t>(route)];
  }

  // Pure-C++ evaluation seam for matches whose other player is not another
  // NativeTeammateExecutor.  The caller owns one isolated state per episode.
  // This is the same action path used by play(); no opponent/provider identity
  // enters the policy ABI. neutral_special_economy is an explicit modular
  // handover mode: it preserves the production route and mandatory guards but
  // disables K320/R5/MD/Moon/FC trade observers/debts, allowing a separate
  // observation-confirmed market policy to own SELL continuation.
  PlayerAction action_external(
      const Simulator& env, int player, int route, NativeAgentState& state,
      NativeMarketArm experimental_market_arm = NativeMarketArm::LegacyDefault,
      NativePhasedMarketAudit* phased_market_audit = nullptr,
      NativePhasedStepAudit* phased_step_audit = nullptr,
      bool neutral_special_economy = false,
      NativeRepairOptions repair_options = {},
      NativeRepairAudit* repair_audit = nullptr) const;

  // Executes the identical default-off provider path and reports only what is
  // immutable after final overlay selection: the finalized current step.  It
  // does not claim that unknown future overlay choices are frozen.
  NativeCommittedAction action_external_committed(
      const Simulator& env, int player, int route, NativeAgentState& state,
      std::uint64_t issuer_generation,
      NativeMarketArm experimental_market_arm = NativeMarketArm::LegacyDefault,
      NativePhasedMarketAudit* phased_market_audit = nullptr,
      NativePhasedStepAudit* phased_step_audit = nullptr,
      bool neutral_special_economy = false,
      NativeRepairOptions repair_options = {},
      NativeRepairAudit* repair_audit = nullptr) const;

  // Default-off final-action commit seam for the independent weed/MOVE day
  // owner. Replays the native provider with only weed_obligation_day_owner
  // enabled, then substitutes only the authorized owned actor before any
  // final-manifest-dependent pending receipt is staged. Every other unit and
  // market slot must remain native-exact. The legacy external API never enters
  // this path.
  PlayerAction action_external_weed_owner_finalized(
      const Simulator& env, int player, int route, NativeAgentState& state,
      int owned_actor, std::uint64_t owner_authority_hash,
      const PlayerAction& final_action) const;

  // Same final-stage seam for a composed repair transaction. Every selected
  // unit is explicit in owned_actor_mask; market ownership is append-only so
  // native orders and their pending receipts cannot be invalidated.
  PlayerAction action_external_repair_owner_finalized(
      const Simulator& env, int player, int route, NativeAgentState& state,
      NativeRepairOptions repair_options, std::uint64_t owned_actor_mask,
      bool owns_market_tail,
      std::uint64_t owner_authority_hash,
      const PlayerAction& final_action) const;

 private:
  NativeTapeLibrary library_;

  PlayerAction action(const Simulator& env, int player, int route,
                      NativeAgentState& state,
                      bool neutral_special_economy = false,
                      const NativeRepairOptions& repair_options = {},
                      NativeRepairAudit* repair_audit = nullptr,
                      bool experimental_general_takeover = false,
                      NativeGeneralMarketAudit* economy_audit = nullptr,
                      int future_switch_step = -1,
                      NativeMarketArm experimental_market_arm =
                          NativeMarketArm::LegacyDefault,
                      NativePhasedMarketAudit* phased_market_audit = nullptr,
                      NativePhasedStepAudit* phased_step_audit = nullptr,
                      const PlayerAction* final_action_override = nullptr,
                      std::uint64_t final_action_owned_actor_mask = 0,
                      bool final_action_owns_market_tail = false,
                      std::uint64_t final_action_authority_hash = 0) const;
};

// Read-only evaluation counters shared by the built-in and external match
// loops.  Unit failures are inspected before Simulator::step; market failures
// are inspected immediately after it.
int native_macro_unit_failures(const Simulator& env, int player,
                               const PlayerAction& action);
int native_macro_market_failures(const Simulator& env, int player,
                                 const PlayerAction& action);

}  // namespace fastkag
