#pragma once

#include "simulator.hpp"

#include <cstdint>
#include <vector>

namespace joint_fixed_move_oracle::production {

struct TileDebt {
  bool active{};
  bool structure{};
  fastkag::Item crop{fastkag::Item::NONE};
  fastkag::Op structure_op{fastkag::Op::PASS};
  int created_step{-1};
  int seed_retry_attempts{};
  int last_seed_retry_step{-1};
  // A crop repair owns the delayed crop plus one replant first-yield suffix.
  // Zero means the first verified HARVEST is still due; one means the
  // re-planted crop's first verified HARVEST is the completion boundary.
  int confirmed_harvests{};
};

struct ReactiveState {
  std::vector<TileDebt> debts;
  std::vector<fastkag::Item> last_crop;
  int last_step{-1};
  void reset(int tile_count);
};

struct ReactiveAudit {
  int candidate_weed_events{};
  int weed_events{};
  int recovered_old_crop_weeds{};
  int digs{};
  int plants{};
  int waters{};
  int fertilizes{};
  int harvests{};
  int structures{};
  int seed_waits{};
  int seed_demand_suppressed{};
  int active_debt_steps{};
  int move_mismatches{};
  int abandoned_cycles{};
  int completed_cycles{};
};

struct ReactiveOptions {
  bool recover_old_crop_weeds{true};
  // Recompile lifecycle work from the observed tile state whenever a worker
  // is standing on a previously known crop tile.  This deliberately does not
  // require the current source-tape byte to be WATER/HARVEST/PLANT.
  bool manage_all_known_crops{false};
  // Retire a missing/weed-blocked crop cycle when even an immediate replant
  // cannot reach first maturity before the observable episode deadline.
  bool causal_cycle_guard{false};
  bool recover_missing_seed_debt{true};
  // Observable value gate for recovering a remembered crop after it became
  // weed. Zero preserves the oracle behavior.
  int minimum_old_crop_market_price{0};
  // Causal conservative admission: a newly blocked source intent is owned by
  // RouteCursor only when legacy's fixed 10-step transaction would discard a
  // MOVE. Otherwise the compiler deliberately abandons that one cycle.
  bool require_legacy_drop_move_for_new_debt{false};
  int latest_new_debt_step{1'000'000};
  // -1 enables every eligible event.  A non-negative value enables only that
  // causal event ordinal, permitting offline finite-oracle ablations without
  // changing the action compiler itself.
  int allowed_event_ordinal{-1};
  int allowed_event_ordinal_second{-1};
  int allowed_event_ordinal_third{-1};
};

struct RouteCursorState {
  struct DeferredNonMove {
    int source_step{-1};
    fastkag::Action action{};
    // Exact observed tile at which this source action was crossed. {-1,-1}
    // means the source expired at a day boundary and no causal origin can be
    // reconstructed from the live observation.
    fastkag::Position tile{-1, -1};
    fastkag::Item item{fastkag::Item::NONE};
    int deadline_step{-1};
  };
  ReactiveState production;
  std::vector<int> source_cursor;
  // Exact non-MOVE sources crossed by a forced end-of-day MOVE. They are not
  // claimed executed and are retained for a canonical replanner.
  std::vector<std::vector<DeferredNonMove>> deferred_nonmoves;
  int day{-1};
};

struct RouteCursorAudit {
  ReactiveAudit production;
  int inserted_before_move{};
  int skipped_nonmoves{};
  int forced_moves{};
  int deferred_nonmoves{};
};

// Side-effect-free RouteCursor proposal. `next_*` is tentative until the
// caller proves that the exact final unit vector was submitted and its effects
// were confirmed from the next observation.
struct RouteCursorProposal {
  fastkag::PlayerAction action;
  RouteCursorState prior_state;
  RouteCursorState next_state;
  RouteCursorAudit prior_audit;
  RouteCursorAudit next_audit;
};

// State-reactive production compiler for fields affected by weed or lifecycle
// drift.  MOVE bytes remain at their original absolute steps.  Non-MOVE slots
// on an affected tile are owned by a cross-day production debt state machine.
[[nodiscard]] fastkag::PlayerAction apply_reactive_fixed_move(
    const fastkag::PlayerAction& raw, const fastkag::Simulator& simulator,
    int player, ReactiveState& state, ReactiveAudit& audit,
    const ReactiveOptions& options = {});

// Time-dilated route compiler.  The route contract is the ordered MOVE
// subsequence of each actor within each day, not an absolute-tick byte match.
// Production work may delay a MOVE while slack remains; when the day deadline
// becomes tight, low-priority non-MOVE source slots are skipped so every MOVE
// in that day's skeleton is emitted before midnight.
[[nodiscard]] fastkag::PlayerAction apply_reactive_route_cursor(
    const std::vector<fastkag::PlayerAction>& tape,
    const fastkag::Simulator& simulator, int player, RouteCursorState& state,
    RouteCursorAudit& audit, const ReactiveOptions& options = {});

[[nodiscard]] RouteCursorProposal propose_reactive_route_cursor(
    const std::vector<fastkag::PlayerAction>& tape,
    const fastkag::Simulator& simulator, int player,
    const RouteCursorState& state, const RouteCursorAudit& audit,
    const ReactiveOptions& options = {});

// Reconcile the final composed unit manifest with the ordered route skeleton
// without accepting any tentative production transition. On a non-exact
// overlay, an actor may consume only its earliest remaining source MOVE. Every
// preceding non-MOVE is retained as an explicit deferred debt. A temporary
// MOVE that does not equal that earliest MOVE consumes nothing.
[[nodiscard]] RouteCursorProposal reconcile_reactive_route_cursor_final(
    const std::vector<fastkag::PlayerAction>& tape,
    const fastkag::Simulator& simulator, int player,
    const RouteCursorProposal& proposal,
    const std::vector<fastkag::Action>& final_units,
    bool accept_exact_production = true);

// Atomic state/audit commit. A mismatch or unconfirmed effect changes nothing.
[[nodiscard]] bool commit_reactive_route_cursor(
    const RouteCursorProposal& proposal,
    const std::vector<fastkag::Action>& final_units,
    bool effects_confirmed, RouteCursorState& state, RouteCursorAudit& audit);

}  // namespace joint_fixed_move_oracle::production
