#pragma once

#include "general_planner.hpp"
#include "public_belief_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace planner_input {

using Inventory = g001::market::Inventory;

struct CurrentOwnPublicState {
  int step{};
  int day{};
  std::int64_t own_money{};
  std::int64_t opponent_public_money{};
  int official_shed_capacity{100};
  Inventory own_shed{};
  Inventory public_market_inventory{};
  Inventory public_market_price{};
};

// Narrow adapter over productionForecastCpp::Result.  This avoids coupling
// the planner boundary to the evolving obligation compiler DTO.
struct CertifiedProductionForecast {
  g001::rolling::FixedForecast forecast;
  bool accepted{};
  bool cash_nodes_consistent{};
};

struct KnownTownDrain {
  int step{};
  Inventory quantity{};
};

struct CausalMarketSchedule {
  // Strictly increasing current/future market slots over the episode.
  std::vector<int> remaining_sale_windows;
  // Optional deterministic future drain.  If the production forecast already
  // contains the identical value it is recognized, never added twice.
  std::vector<KnownTownDrain> future_town_drain;
};

struct Input {
  public_belief_runtime::Snapshot public_belief;
  CurrentOwnPublicState current;
  CertifiedProductionForecast production;
  CausalMarketSchedule market_schedule;
  int episode_steps{720};
};

// Abstract causal stress used by the rolling proof. Quantities are deliberately
// unassigned to market slots: OpponentFirst is not an official same-slot rule.
struct CausalPressureScenario {
  std::uint32_t source_scenario_id{};
  g001::dump::BeliefBand belief_band{g001::dump::BeliefBand::Point};
  g001::dump::DumpTiming timing{g001::dump::DumpTiming::None};
  g001::rolling::SameTickOrder abstract_proof_order{
      g001::rolling::SameTickOrder::OpponentFirst};
  int source_total{};
  bool split_correlated_marginal{};
  Inventory quantity{};
};

struct Audit {
  bool accepted{};
  int forecast_capacity{};
  int critical_purchase_requirements{};
  int feed_requirements{};
  int route_hard_requirements{};
  std::int64_t total_cash_requirements{};
  int town_drain_inserted{};
  int town_drain_already_present{};
  // Historical frame drain is already reflected in observed inventory and is
  // intentionally never copied into the future forecast.
  int historical_town_drain_units_ignored{};
  bool marginal_upper_split{};
  std::vector<CausalPressureScenario> pressures;
};

struct Result {
  g001::general::Input general;
  // Independently generated from the public/causal planner boundary during
  // compile().  The selective runtime compares this complete weighted scenario
  // surface with general::plan().dump_audit before claiming an exact replay.
  g001::dump::Result causal_scenario_set;
  std::vector<CausalPressureScenario> causal_pressures;
  Audit audit;
};

// Throws on any discontinuity, inconsistent certification, interval, schedule
// or forecast.  No partial output is returned (fail closed).
[[nodiscard]] Result compile(const Input& input);

// Exact semantic equality for the weighted scenario surface consumed by the
// rolling optimizer: metadata plus every belief/dump/order/causality field.
// Intent/flow-history diagnostics are deliberately outside this proof object.
[[nodiscard]] bool exact_scenario_set_equal(
    const g001::dump::Result& expected,
    const g001::dump::Result& actual);

enum class PlacementGoal : std::uint8_t {
  SerialOpponentBefore,
  SameSlotLockstep,
};

enum class OverflowPolicy : std::uint8_t {
  FailClosed,
  SplitProductLocal,
};

enum class ExactRelation : std::uint8_t {
  OpponentBefore,
  Lockstep,
};

enum class MaterializeStatus : std::uint8_t {
  Exact,
  Split,
  Unrepresentable,
  NoPressure,
};

struct MaterializeInput {
  CausalPressureScenario pressure;
  std::vector<g001::market::Order> own_candidate_queue;
  int optional_slot{-1};
  int maximum_slots{10};
  // ReplaceTotal(0) has no executable own SELL slot.  In this explicit mode,
  // optional_slot must equal maximum_slots and denotes the virtual boundary
  // immediately after the official queue.  It can prove serial rival pressure
  // before "no selected SELL", but can never request Lockstep.
  bool virtual_no_own_sell_at_market_end{};
  PlacementGoal goal{PlacementGoal::SerialOpponentBefore};
  OverflowPolicy overflow_policy{OverflowPolicy::FailClosed};
};

struct ExactSlotPressure {
  std::vector<g001::market::Order> rival_queue;
  ExactRelation relation{ExactRelation::Lockstep};
  int optional_slot{-1};
  Inventory quantity{};
};

struct MaterializeResult {
  MaterializeStatus status{MaterializeStatus::Unrepresentable};
  std::vector<ExactSlotPressure> exact_slot_pressure;
  bool abstract_opponent_first_represented{};
  bool split_product_local{};
  std::string reason;
};

// Converts an abstract quantity scenario only after the complete own queue and
// optional slot are known. Earlier rival slots are serial opponent-before;
// the same slot is official lockstep. No placement can precede own slot 0.
[[nodiscard]] MaterializeResult materialize(const MaterializeInput& input);

}  // namespace planner_input
