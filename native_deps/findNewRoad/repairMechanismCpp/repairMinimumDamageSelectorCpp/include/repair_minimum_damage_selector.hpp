#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace g001::minimum_damage {

struct Position {
  int x{};
  int y{};
  friend bool operator==(Position, Position) = default;
};

enum class TileKind : std::uint8_t {
  Empty = 0,
  Soil,
  Weed,
  Pasture,
  Crop,
  Animal,
  OtherStructure,
  Blocked,
};

enum class ActionKind : std::uint8_t {
  Pass = 0,
  North,
  South,
  East,
  West,
  Dig,
  BuildPasture,
  Pickup,
  Plant,
  PlaceAnimal,
  Feed,
  Care,
  BuySeed,
  BuyAnimal,
  BuyFeed,
  Water,
  CollectFertilizer,
  BuildCoop,
};

struct Action {
  ActionKind kind{ActionKind::Pass};
  int item{-1};
  int quantity{1};
  friend bool operator==(Action, Action) = default;
};

[[nodiscard]] bool is_move(ActionKind kind) noexcept;
[[nodiscard]] bool is_market(ActionKind kind) noexcept;
[[nodiscard]] const char* action_kind_name(ActionKind kind) noexcept;

// One immutable selected-route source slot.  Production metadata is used only
// when the raw action is non-PASS/non-MOVE; MOVE is always a hard token.
struct SourceSlot {
  Action action{};
  bool must_finish{};
  bool important{};
  int cascade_risk{};
};

struct DayStartSnapshot {
  int width{};
  int height{};
  std::vector<TileKind> tiles;
  std::vector<Position> actor_positions;
  std::vector<int> seed_inventory;
  std::vector<int> animal_inventory;
  std::vector<int> shed_inventory;
  std::vector<std::vector<int>> actor_inventory;
  int feed_inventory{};
  int cash{};
};

enum class ObligationKind : std::uint8_t {
  RawProduction = 0,
  WeedPastureRecovery,
  ReplantRecovery,
  PurchaseRetry,
  AnimalRecovery,
  AnimalCareRecovery,
};

enum class Channel : std::uint8_t { Unit = 0, Market };

struct Transition {
  Channel channel{Channel::Unit};
  Action action{};
  bool requires_position{};
  Position tile{};
  // BUY transitions are admitted only with a day-start guaranteed fill and
  // explicit worst-case cash bound. They never infer a future market result.
  bool guaranteed_fill{};
  int worst_case_cash_cost{};
};

struct TypedObligation {
  std::uint64_t id{};
  ObligationKind kind{ObligationKind::RawProduction};
  // Unit owner, or -1 when every transition is on the market channel.
  int actor{-1};
  int release_slot{};
  int deadline_slot{23};
  int source_slot{-1};
  bool must_finish{};
  bool important{};
  int cascade_risk{};
  std::vector<std::uint64_t> dependencies;
  std::vector<Transition> transitions;
};

[[nodiscard]] TypedObligation make_weed_pasture_obligation(
    std::uint64_t id, int actor, Position tile, int release_slot,
    bool must_finish, bool important, int cascade_risk);
[[nodiscard]] TypedObligation make_replant_obligation(
    std::uint64_t id, int actor, Position tile, int seed_item,
    bool dig_first, int release_slot, bool must_finish, bool important,
    int cascade_risk);
[[nodiscard]] TypedObligation make_seed_purchase_retry(
    std::uint64_t id, int seed_item, int quantity, int release_slot,
    int worst_case_cash_cost, bool guaranteed_fill, bool must_finish,
    bool important, int cascade_risk);
[[nodiscard]] TypedObligation make_animal_purchase_retry(
    std::uint64_t id, int animal_item, int quantity, int release_slot,
    int worst_case_cash_cost, bool guaranteed_fill, bool must_finish,
    bool important, int cascade_risk);
[[nodiscard]] TypedObligation make_feed_purchase_retry(
    std::uint64_t id, int quantity, int release_slot,
    int worst_case_cash_cost, bool guaranteed_fill, bool must_finish,
    bool important, int cascade_risk);
[[nodiscard]] TypedObligation make_animal_recovery_obligation(
    std::uint64_t id, int actor, Position tile, int animal_item,
    bool build_pasture_first, bool feed_after_place, bool care_after_place,
    int release_slot, bool must_finish, bool important, int cascade_risk,
    std::vector<std::uint64_t> dependencies = {});

struct Request {
  DayStartSnapshot day_start;
  // [actor][slot], and every actor lane must contain exactly horizon slots.
  std::vector<std::vector<SourceSlot>> raw_sources;
  std::vector<TypedObligation> repair_obligations;
  int horizon{24};
  std::size_t maximum_states{2'000'000};
};

struct Objective {
  int unfinished_must_finish{};
  int cross_day_cascade_risk{};
  int deferred_important{};
  int inserted_move_distance{};
  int move_slot_displacement{};
  int production_slot_disturbance{};
  friend bool operator==(const Objective&, const Objective&) = default;
};

[[nodiscard]] bool objective_less(const Objective& left,
                                  const Objective& right) noexcept;

struct ScheduledAction {
  Action action{};
  int raw_source_slot{-1};
  std::uint64_t obligation_id{};
  int transition_index{-1};
};

struct ScheduleSlot {
  std::vector<ScheduledAction> units;
  std::optional<ScheduledAction> market;
};

enum class DebtReason : std::uint8_t {
  Capacity = 0,
  ResourceUnavailable,
  UnreachableTile,
  UnsupportedIdentity,
  PartialDayBound,
};

struct DayBoundDebt {
  std::uint64_t obligation_id{};
  DebtReason reason{DebtReason::Capacity};
  int completed_transitions{};
  int remaining_transitions{};
  bool must_finish{};
  bool important{};
  int cascade_risk{};
  int bound_day_offset{1};
};

enum class PlanStatus : std::uint8_t {
  Planned = 0,
  InvalidRequest,
  HardMoveInfeasible,
  SearchLimit,
};

struct PlanResult {
  PlanStatus status{PlanStatus::InvalidRequest};
  Objective objective{};
  std::vector<ScheduleSlot> slots;
  std::vector<DayBoundDebt> debts;
  std::size_t explored_states{};
  std::string diagnostic;
  [[nodiscard]] bool planned() const noexcept {
    return status == PlanStatus::Planned;
  }
};

// Exact memoized day solver. No terminal reward, later observation, or market
// oracle enters the request or objective.
[[nodiscard]] PlanResult select_minimum_damage(const Request& request);

// Independent no-memo exhaustive enumeration intended only for tiny tests.
[[nodiscard]] PlanResult exhaustive_oracle_for_testing(
    const Request& request, std::size_t maximum_nodes = 5'000'000);

struct VerifyResult {
  bool valid{};
  std::string diagnostic;
  Objective recomputed_objective{};
  int checked_move_tokens{};
};

[[nodiscard]] VerifyResult verify_schedule(const Request& request,
                                           const PlanResult& plan);
[[nodiscard]] const char* debt_reason_name(DebtReason reason) noexcept;
[[nodiscard]] const char* plan_status_name(PlanStatus status) noexcept;

}  // namespace g001::minimum_damage
