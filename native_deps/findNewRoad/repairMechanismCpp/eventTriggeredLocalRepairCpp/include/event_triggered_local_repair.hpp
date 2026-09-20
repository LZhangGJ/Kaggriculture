#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace g001::event_local_repair {

enum class Op : std::uint8_t {
  Pass,
  Move,
  Dig,
  Plant,
  Build,
  Water,
  Harvest,
  Other,
};

struct Position {
  int row{-1};
  int column{-1};
  friend bool operator==(const Position&, const Position&) = default;
  friend bool operator<(const Position& left, const Position& right) noexcept {
    return left.row < right.row ||
           (left.row == right.row && left.column < right.column);
  }
};

// arg0/arg1 deliberately remain opaque.  MOVE direction and any native action
// payload are compared bit-for-bit by the isolated compiler.
struct Action {
  Op op{Op::Pass};
  int item{-1};
  int quantity{1};
  int arg0{};
  int arg1{};
  friend bool operator==(const Action&, const Action&) = default;
};

enum class TileKind : std::uint8_t { Empty, Weed, Crop, Structure, Other };

struct TileObservation {
  TileKind kind{TileKind::Other};
  int item{-1};
  bool watered_today{};
  bool harvest_legal{};
  friend bool operator==(const TileObservation&, const TileObservation&) = default;
};

struct PlannedAction {
  Action action;
  // PASS is always a zero-loss sink.  A caller may explicitly certify another
  // stationary action as discardable.  MOVE can never be a sink.
  int certified_stationary_loss{std::numeric_limits<int>::max()};
};

struct ActorDayPlan {
  int actor{-1};
  std::vector<PlannedAction> turns;
};

struct DayPlan {
  int day{-1};
  int turns{};
  std::vector<ActorDayPlan> actors;
};

struct ActorObservation {
  int actor{-1};
  Position position;
  TileObservation tile;
  // Inventory of the crop named by desired_item_hint.  A zero after
  // a failed market fill keeps the transaction open; it is never treated as
  // successful acquisition.
  int desired_item_hint{-1};
  int desired_item_inventory{};
  bool failed_fill_observed{};
  // Physical actor slots are recycled by the native simulator when farm
  // hands despawn and are rehired.  Generation is therefore part of causal
  // identity; zero preserves the original isolated ABI.
  std::uint64_t actor_generation{};
  // Final-manifest tile arbitration may expose the real observation while
  // denying this actor an effect lease for the current turn.
  bool tile_effect_eligible{true};
};

struct TurnInput {
  TurnInput() = default;
  TurnInput(int day_value, int turn_value,
            std::vector<ActorObservation> actor_values,
            std::vector<Action> composed_values = {})
      : day(day_value), turn(turn_value), actors(std::move(actor_values)),
        composed_baseline(std::move(composed_values)) {}
  int day{-1};
  int turn{-1};
  std::vector<ActorObservation> actors;
  // Optional exact multi-actor action vector already produced by the native
  // composer for this turn.  The local compiler may replace only a slot still
  // owned by its installed source plan; every other slot is copied verbatim.
  // Empty preserves the isolated DayPlan-only ABI.
  std::vector<Action> composed_baseline;
};

struct TransactionKey {
  int actor{-1};
  Position tile;
  std::uint64_t actor_generation{};
  friend bool operator==(const TransactionKey&, const TransactionKey&) = default;
  friend bool operator<(const TransactionKey& left,
                        const TransactionKey& right) noexcept {
    if (left.actor != right.actor) return left.actor < right.actor;
    if (left.actor_generation != right.actor_generation)
      return left.actor_generation < right.actor_generation;
    return left.tile < right.tile;
  }
};

enum class GoalKind : std::uint8_t { Crop, Structure };

struct TransactionView {
  std::uint64_t id{};
  // Persistent plot/source generation. Actor identity below records origin
  // provenance only; it is never the transaction owner.
  std::uint64_t source_epoch{};
  TransactionKey key;
  GoalKind goal{GoalKind::Crop};
  // -1 is unknown.  Zero is a valid native item id (WHEAT).
  int desired_item{-1};
  int origin_day{-1};
  int origin_turn{-1};
  int confirmed_harvests{};
  int confirmed_actions{};
  int failed_receipts{};
  int outstanding_source_actions{};
  bool completed{};
  bool superseded{};
  bool cancelled{};
};

struct RepairBinding {
  std::uint64_t transaction_id{};
  // Non-zero when this effect discharges an original planned source action.
  std::uint64_t source_action_id{};
  int actor{-1};
  std::uint64_t actor_generation{};
  Position tile;
  Action action;
  TileObservation before;
  bool changes_baseline{};
};

struct Decision {
  std::uint64_t id{};
  int day{-1};
  int turn{-1};
  std::vector<int> actor_ids;
  std::vector<Action> baseline;
  std::vector<Action> actions;
  std::vector<RepairBinding> bindings;
  std::vector<int> fail_closed_actors;
};

struct Receipt {
  std::uint64_t decision_id{};
  int actor{-1};
  Position tile;
  TileObservation after;
  int desired_item_inventory_delta{};
  bool generic_effect_confirmed{};
  // Needed for WATER on a day boundary, where watered_today may reset before
  // the next externally visible observation.
  bool day_end_water_effect_lower_bound{};
  std::uint64_t actor_generation{};
};

struct Audit {
  int transactions_created{};
  int local_rewrites{};
  int state_recompiles{};
  int source_debts_created{};
  int source_debts_confirmed{};
  int source_debts_externalized{};
  int source_intents_coalesced{};
  int objectives_superseded{};
  int objectives_cancelled{};
  int direct_slack_uses{};
  int delayed_move_rewrites{};
  int fail_closed{};
  int receipts_confirmed{};
  int receipts_failed{};
  int completed_transactions{};
  // Receipt-bound actor/generation leases committed for plot-owned debt.
  int plot_leases{};
};

struct Config {
  bool enabled{false};
  int maximum_stationary_sink_loss{0};
  // A crop repair owns the delayed generation and its replant generation.  It
  // closes only at this many observation-confirmed harvest effects.
  int crop_harvests_to_close{2};
};

// Event-local compiler.  It does not own a global source cursor.  A rewrite is
// scoped to one actor and one actor+tile transaction; all other actor bytes are
// copied directly from the installed day plan.
class Compiler {
 public:
  explicit Compiler(Config config = {});

  void install_day(DayPlan plan);
  [[nodiscard]] Decision decide(const TurnInput& input);

  // Stages a receipt only if the exact final multi-actor manifest equals the
  // proposal.  No lifecycle progress is credited at submission time.
  [[nodiscard]] bool commit(const Decision& decision,
                            const std::vector<Action>& final_actions);
  [[nodiscard]] bool observe(const Receipt& receipt);

  [[nodiscard]] std::vector<TransactionView> open_transactions() const;
  [[nodiscard]] const Audit& audit() const noexcept { return audit_; }
  [[nodiscard]] bool movement_invariant_holds(int actor) const;
  [[nodiscard]] bool protected_action_invariant_holds(int actor) const;

 private:
  struct Transaction {
    TransactionView view;
    std::map<std::uint64_t, Action> source_debts;
  };
  struct Pending {
    std::uint64_t decision_id{};
    RepairBinding binding;
  };

  [[nodiscard]] int actor_index(int actor) const;
  [[nodiscard]] const ActorObservation* observation_for(
      const TurnInput& input, int actor) const;
  [[nodiscard]] Transaction* find_plot_transaction(Position tile);
  [[nodiscard]] const Transaction* find_plot_transaction(Position tile) const;
  Transaction& ensure_transaction(const TransactionKey& key, GoalKind goal,
                                  int desired_item, int day, int turn);
  [[nodiscard]] std::optional<Action> required_action(
      const Transaction& transaction,
      const ActorObservation& observation) const;
  [[nodiscard]] bool action_is_blocked(const Action& action,
                                       const ActorObservation& observation,
                                       GoalKind& goal,
                                       int& desired_item) const;
  [[nodiscard]] const Action& effective_action(int actor_index,
                                               int turn) const;
  [[nodiscard]] std::uint64_t effective_source_id(int actor_index,
                                                  int turn) const;
  [[nodiscard]] int effective_sink_loss(int actor_index, int turn) const;
  [[nodiscard]] bool install_insertion(int actor_index, int turn,
                                       const Action& inserted,
                                       std::uint64_t inserted_source_id);
  [[nodiscard]] bool externalize_blocked_source(int actor_index, int turn,
                                                std::uint64_t source_id);
  [[nodiscard]] bool is_sink(int actor_index, int turn) const;
  void register_source_debt(Transaction& transaction, const Action& source,
                            std::uint64_t source_id);
  [[nodiscard]] std::uint64_t source_for_required(
      const Transaction& transaction, const Action& required) const;
  [[nodiscard]] bool action_confirms(const Pending& pending,
                                     const Receipt& receipt) const;

  Config config_;
  std::optional<DayPlan> day_plan_;
  // overlays_[actor-index][turn] is a fully compiled event-local replacement.
  std::vector<std::vector<std::optional<Action>>> overlays_;
  // Overlay actions retain the discard certificate of their baseline source.
  // Repair actions use max(), so they can never become absorption sinks.
  std::vector<std::vector<std::optional<int>>> overlay_sink_losses_;
  std::vector<std::vector<std::uint64_t>> baseline_source_ids_;
  std::vector<std::vector<std::optional<std::uint64_t>>> overlay_source_ids_;
  // Keyed by source epoch/id. Plot + desired target define persistent
  // ownership; actors only take receipt-bound execution leases.
  std::map<std::uint64_t, Transaction> transactions_;
  std::map<std::pair<int, std::uint64_t>, Pending> pending_by_actor_;
  // Needed only to prove current-day protected-action preservation after a
  // repeated source is coalesced into an existing plot objective. It is reset
  // at install_day and never becomes cross-day debt.
  std::map<std::uint64_t, bool> coalesced_current_day_sources_;
  std::uint64_t next_transaction_id_{1};
  std::uint64_t next_decision_id_{1};
  std::uint64_t next_source_action_id_{1};
  Audit audit_;
};

[[nodiscard]] const char* op_name(Op operation) noexcept;

}  // namespace g001::event_local_repair
