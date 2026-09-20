#pragma once

#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace g001::failure_debt {

constexpr std::size_t kOperationCount = 24;

enum class FillClass : std::uint8_t { Pending, Full, Partial, Zero, Ambiguous };
enum class DebtKind : std::uint8_t { MarketPurchase, UnitAction };
enum class DebtStatus : std::uint8_t {
  Open,
  Attempted,
  Confirmed,
  Scheduled,
  Consumed,
  Failed,
  Retired,
};

struct OwnObservation {
  int step{};
  int money{};
  int protected_cash{};
  int shed_used{};
  int shed_capacity{100};
  int market_slots_used{};
  int maximum_market_slots{10};
  int turns_per_day{24};
  std::array<int, fastkag::N_CROPS> seeds{};
  // Total usable own inventory by item (shed + carried). Animal entries may
  // be zero because `animals` below is the authoritative animal total.
  std::array<int, fastkag::N_ITEMS> available_items{};
  std::array<int, fastkag::N_ITEMS> shed_items{};
  std::vector<std::array<int, fastkag::N_ITEMS>> actor_inventory;
  // Shed + carried + placed. This makes animal settlement invariant to a
  // simultaneous PICKUP/PLACE between observations.
  std::array<int, fastkag::N_ANIMALS> animals{};
  std::array<int, fastkag::N_ITEMS> known_outflow{};
  std::array<int, fastkag::N_ITEMS> known_other_inflow{};
};

struct UnitRequirement {
  fastkag::Action action{};
  int actor{};
  fastkag::Position position{};
  int earliest_step{};
  int deadline{};
  int economic_value{};
  int maximum_absorbed_value{4};
  bool allow_existing_equivalent{true};
  std::string provenance;
};

// One purchased unit owns one chain. Partial fill retires the leading filled
// chains and materializes debts only for the missing suffix.
struct RecoveryChain {
  int purchase_unit_index{};
  std::vector<UnitRequirement> requirements;
};

enum class LifecycleKind : std::uint8_t { Crop, Animal };

struct DayLifecycleSpec {
  int purchase_unit_index{};
  LifecycleKind kind{LifecycleKind::Crop};
  fastkag::Item item{fastkag::Item::NONE};
  fastkag::Position tile{};
  int tile_observed_step{-1};
  fastkag::TileKind observed_kind{fastkag::TileKind::EMPTY};
  int observed_planted_day{-1};
  int observed_placed_day{-1};
  int required_harvest_cycles{1};
  int required_care_days{};
  int next_occupancy_step{-1};
  // Empty tiles crossing a day boundary can acquire a weed. This may be true
  // only when the chain itself contains a certified DIG/recheck before PLANT.
  bool future_empty_tile_weed_risk_resolved{};
  std::string provenance;
};

struct PurchaseIntent {
  std::uint64_t attempt_id{};
  fastkag::Op operation{fastkag::Op::PASS};
  fastkag::Item item{fastkag::Item::NONE};
  int quantity{};
  int deadline{};
  int economic_value{};  // conservative value of fully completing the chain
  std::vector<RecoveryChain> chains;
  std::string provenance;
};

struct SettlementRecord {
  std::uint64_t attempt_id{};
  FillClass classification{FillClass::Pending};
  int requested{};
  int filled{};
  int missing{};
  std::string reason;
};

struct Debt {
  std::uint64_t id{};
  std::uint64_t transaction_id{};
  std::uint64_t source_attempt_id{};
  DebtKind kind{DebtKind::MarketPurchase};
  DebtStatus status{DebtStatus::Open};
  int purchase_unit_index{-1};
  fastkag::Action action{};
  int actor{-1};
  fastkag::Position position{};
  int earliest_step{};
  int deadline{};
  int economic_value{};
  int cash_required{};
  int shed_capacity_required{};
  int wheat_required{};
  std::vector<std::uint64_t> dependencies;
  std::string provenance;
};

struct PlannedMarketOrder {
  fastkag::Action action{};
  // Quantity already committed to the unmodified route. Only the remainder
  // may discharge a repair debt.
  int reserved_quantity{};
  int conservative_unit_cost{};
  bool guaranteed_fill{true};
};

struct PlannedUnitSlot {
  int actor{};
  fastkag::Position position{};
  fastkag::Action original{};
  int economic_value{};
  bool expected_success{true};
  bool critical{};
  bool absorbable{};
  std::array<bool, kOperationCount> legal_replacements{};
};

struct PlannedTurn {
  int step{};
  std::vector<PlannedMarketOrder> market;
  std::vector<PlannedUnitSlot> units;
  // Guaranteed effects of unmodified unit actions before this turn's market.
  // Positive shed delta consumes capacity; negative releases it.
  int guaranteed_shed_delta_before_market{};
  int guaranteed_cash_income_before_market{};
  int extra_cash_reserve{};
  int certified_actor_count{-1};
  bool actor_availability_certified{};
};

struct PlanWindow {
  std::vector<PlannedTurn> turns;
  // Rebuilt from the current observation on every receding-horizon call.
  std::vector<DayLifecycleSpec> day_lifecycles;
  // Purchases beyond the explicit horizon that the original route needs.
  int future_hard_purchase_reserve{};
};

struct MarketPatch {
  bool required{};
  int step{};
  fastkag::Action addition{};
  bool uses_existing_surplus{};
  int existing_slot{-1};
};

struct UnitPatch {
  int step{};
  int actor{};
  int source_slot{-1};
  fastkag::Action original{};
  fastkag::Action replacement{};
  int absorbed_value{};
  std::uint64_t debt_id{};
};

struct UnitSchedule {
  std::uint64_t debt_id{};
  int step{};
  int actor{};
  int source_slot{-1};
  bool uses_existing_equivalent{};
};

struct AuditEvent {
  int step{};
  std::uint64_t transaction_id{};
  std::string decision;
  std::string reason;
  std::string provenance;
  int requested{};
  int filled{};
  int missing{};
  int absorbed_value{};
};

struct UnitExecutionEvidence {
  int step{};
  fastkag::Action emitted{};
  fastkag::TileKind observed_tile_kind{fastkag::TileKind::EMPTY};
  fastkag::Item observed_tile_item{fastkag::Item::NONE};
  int own_inventory_delta{};
  int yield_before{};
  int yield_after{};
  bool watered_today{};
  bool fed_today{};
  bool cared_today{};
  bool generic_effect_verified{};
  std::string provenance;
};

struct RepairPlan {
  // Internal receding-horizon certificate. It is not a PlayerAction or market
  // queue. Modular integration must pass it through modular_repair_adapter and
  // then through modular_agent_core::compose.
  bool accepted{};
  bool day_lifecycle_complete{};
  std::uint64_t transaction_id{};
  std::uint64_t observation_fingerprint{};
  std::string reason;
  int purchase_cost{};
  int absorbed_value{};
  int net_value{};
  MarketPatch market;
  std::vector<UnitPatch> units;
  std::vector<UnitSchedule> unit_schedule;
  // Includes requirements discharged by an already-valid legacy action as
  // well as actual replacement patches.
  std::vector<std::uint64_t> scheduled_debt_ids;
  std::vector<AuditEvent> audit;
};

struct SchedulerConfig {
  // Research seam is deliberately default-off. The caller must opt in.
  bool enabled{false};
  int maximum_attempts{3};
  int maximum_horizon{240};
  int minimum_net_value{0};
  int maximum_total_absorbed_value{16};
  // A step-window-only experiment must explicitly disable this. Deployment
  // candidates fail closed without a day/tile lifecycle proof.
  bool require_day_lifecycle{true};
};

[[nodiscard]] bool is_movement(fastkag::Op operation);
[[nodiscard]] int purchase_unit_cost(fastkag::Op operation, fastkag::Item item,
                                     int fallback = 0);
[[nodiscard]] std::string audit_json(const AuditEvent& event);

class FailureDebtScheduler {
 public:
  explicit FailureDebtScheduler(SchedulerConfig config = {});
  void reset();

  // Must be called before record_attempt for the same step.
  std::vector<SettlementRecord> observe(const OwnObservation& observation);
  bool record_attempt(const PurchaseIntent& intent,
                      const OwnObservation& before_attempt);

  [[nodiscard]] RepairPlan plan(std::uint64_t transaction_id,
                                const OwnObservation& observation,
                                const PlanWindow& window);
  // `plan()` is side-effect free. Commit only the plan actually emitted at
  // the current step. A market plan enters Attempted and is settled by the
  // next `observe()`; a unit-only plan enters Scheduled immediately.
  bool commit(const RepairPlan& plan, const OwnObservation& before_execution);
  // Call after observing/verifying a scheduled unit patch or equivalent
  // legacy action. Failure reopens that node and its same-chain suffix.
  bool observe_unit_result(std::uint64_t debt_id,
                           const UnitExecutionEvidence& evidence);
  [[nodiscard]] const std::vector<Debt>& debts() const { return debts_; }
  [[nodiscard]] const std::vector<AuditEvent>& audit() const { return audit_; }
  [[nodiscard]] std::vector<std::uint64_t> open_transactions() const;

 private:
  struct AttemptState {
    PurchaseIntent intent;
    int step{};
    int holding_before{};
    int attempts{};
    int requested_quantity{};
    std::vector<int> purchase_unit_indices;
    std::vector<std::uint64_t> scheduled_debt_ids;
    bool awaiting{true};
    std::uint64_t transaction_id{};
  };

  struct TransactionState {
    struct ExecutedUnit {
      std::uint64_t debt_id{};
      fastkag::Action action{};
      int actor{};
      fastkag::Position position{};
      int step{};
    };
    std::uint64_t id{};
    PurchaseIntent intent;
    int filled{};
    int missing{};
    int attempts{};
    int remaining_purchase{};
    bool completed{};
    bool retired{};
    std::string retire_reason;
    std::vector<std::uint64_t> debt_ids;
    std::vector<ExecutedUnit> executed_units;
  };

  int holding(const OwnObservation& observation, fastkag::Op operation,
              fastkag::Item item) const;
  void materialize_transaction(const AttemptState& attempt, int filled, int missing,
                               int step);
  void append_audit(AuditEvent event);

  SchedulerConfig config_{};
  std::vector<AttemptState> attempts_;
  std::vector<TransactionState> transactions_;
  std::vector<Debt> debts_;
  std::vector<AuditEvent> audit_;
  std::uint64_t next_transaction_id_{1};
  std::uint64_t next_debt_id_{1};
  int latest_step_{-1};
};

}  // namespace g001::failure_debt
