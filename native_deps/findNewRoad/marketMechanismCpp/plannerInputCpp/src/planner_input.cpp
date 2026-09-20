#include "planner_input.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace planner_input {
namespace {

using g001::market::Operation;
using g001::market::Order;
using g001::rolling::Requirement;
using g001::rolling::RequirementKind;

bool zero(const Inventory& inventory) {
  return std::all_of(inventory.begin(), inventory.end(),
                     [](int value) { return value == 0; });
}

int total(const Inventory& inventory) {
  return std::accumulate(inventory.begin(), inventory.end(), 0);
}

void validate_inventory(const Inventory& inventory, const char* name) {
  if (std::any_of(inventory.begin(), inventory.end(),
                  [](int value) { return value < 0; })) {
    throw std::invalid_argument(name);
  }
}

void validate_belief(const g001::market::InventoryBelief& belief) {
  validate_inventory(belief.total, "negative opponent total point");
  validate_inventory(belief.total_interval.lower,
                     "negative opponent total lower");
  validate_inventory(belief.total_interval.upper,
                     "negative opponent total upper");
  validate_inventory(belief.shed, "negative opponent shed point");
  validate_inventory(belief.shed_interval.lower,
                     "negative opponent shed lower");
  validate_inventory(belief.shed_interval.upper,
                     "negative opponent shed upper");
  for (std::size_t p = 0; p < g001::market::product_count; ++p) {
    if (belief.total_interval.lower[p] > belief.total[p] ||
        belief.total[p] > belief.total_interval.upper[p] ||
        belief.shed_interval.lower[p] > belief.shed[p] ||
        belief.shed[p] > belief.shed_interval.upper[p]) {
      throw std::invalid_argument("opponent belief interval is not ordered");
    }
  }
}

void validate_requirement(const Requirement& requirement) {
  if (requirement.cash < 0 || requirement.quantity < 0 ||
      requirement.product < -1 ||
      requirement.product >= static_cast<int>(g001::market::product_count)) {
    throw std::invalid_argument("invalid production requirement");
  }
}

const Inventory& shed_band(const g001::market::InventoryBelief& belief,
                           g001::dump::BeliefBand band) {
  if (band == g001::dump::BeliefBand::Lower)
    return belief.shed_interval.lower;
  if (band == g001::dump::BeliefBand::Upper)
    return belief.shed_interval.upper;
  return belief.shed;
}

Order sell_order(std::size_t product, int quantity) {
  return {Operation::Sell, static_cast<g001::market::Product>(product),
          g001::market::Animal::Goose, quantity};
}

bool same_dump(const g001::rolling::OpponentDump& left,
               const g001::rolling::OpponentDump& right) {
  return left.step == right.step && left.product == right.product &&
         left.quantity == right.quantity;
}

bool same_scenario(const g001::rolling::Scenario& left,
                   const g001::rolling::Scenario& right) {
  if (left.belief_stock != right.belief_stock ||
      !std::isfinite(left.weight) || !std::isfinite(right.weight) ||
      left.weight != right.weight ||
      left.realized_future != right.realized_future ||
      left.same_tick_order != right.same_tick_order ||
      left.dumps.size() != right.dumps.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.dumps.size(); ++index)
    if (!same_dump(left.dumps[index], right.dumps[index])) return false;
  return true;
}

}  // namespace

bool exact_scenario_set_equal(const g001::dump::Result& expected,
                              const g001::dump::Result& actual) {
  if (expected.scenarios.size() != actual.scenarios.size()) return false;
  for (std::size_t index = 0; index < expected.scenarios.size(); ++index) {
    const auto& left = expected.scenarios[index];
    const auto& right = actual.scenarios[index];
    if (left.id != right.id || left.belief_band != right.belief_band ||
        left.timing != right.timing || !std::isfinite(left.weight) ||
        !std::isfinite(right.weight) || left.weight != right.weight ||
        !same_scenario(left.scenario, right.scenario)) {
      return false;
    }
  }
  return true;
}

Result compile(const Input& input) {
  if (!input.production.accepted || !input.production.cash_nodes_consistent) {
    throw std::invalid_argument("production forecast is not certified");
  }
  // general_planner.cpp currently uses the official fixed 720-step horizon
  // and step/24 day mapping internally.  Accepting a different geometry here
  // would make its regenerated dump scenarios disagree with this bridge.
  if (input.episode_steps != 720 || input.current.step < 0 ||
      input.current.step >= input.episode_steps || input.current.day < 0 ||
      input.current.day != input.current.step / 24 ||
      input.current.official_shed_capacity < 0) {
    throw std::invalid_argument("invalid current planner scalar");
  }
  validate_inventory(input.current.own_shed, "negative own shed");
  validate_inventory(input.current.public_market_inventory,
                     "negative current market inventory");
  validate_inventory(input.current.public_market_price,
                     "negative current market price");
  validate_belief(input.public_belief.belief);
  if (total(input.current.own_shed) > input.current.official_shed_capacity ||
      total(input.public_belief.belief.shed) >
          input.current.official_shed_capacity ||
      total(input.public_belief.belief.shed_interval.lower) >
          input.current.official_shed_capacity) {
    throw std::invalid_argument("observed/lower shed stock exceeds official capacity");
  }
  for (int upper : input.public_belief.belief.shed_interval.upper) {
    if (upper > input.current.official_shed_capacity) {
      throw std::invalid_argument("one-product shed upper exceeds official capacity");
    }
  }
  if (input.public_belief.belief.step != input.current.step) {
    throw std::invalid_argument("belief step does not match current state");
  }

  const auto& history = input.public_belief.public_market_history;
  if (history.empty()) throw std::invalid_argument("public history is empty");
  for (std::size_t i = 0; i < history.size(); ++i) {
    const auto& frame = history[i];
    validate_inventory(frame.market_inventory, "negative history market inventory");
    validate_inventory(frame.market_price, "negative history market price");
    validate_inventory(frame.known_town_drain, "negative historical town drain");
    for (std::size_t p = 0; p < g001::market::product_count; ++p) {
      if (frame.own_sell_filled_lower[p] > frame.own_sell_filled_point[p] ||
          frame.own_sell_filled_point[p] > frame.own_sell_filled_upper[p] ||
          frame.own_sell_filled_upper[p] > frame.own_sell_requested[p]) {
        throw std::invalid_argument("invalid history own SELL interval");
      }
      if (frame.opponent_clearance_valid[p] &&
          (frame.opponent_clearance_lower[p] >
               frame.opponent_clearance_point[p] ||
           frame.opponent_clearance_point[p] >
               frame.opponent_clearance_upper[p])) {
        throw std::invalid_argument("invalid history opponent clearance interval");
      }
    }
    if (i > 0 && (frame.step != history[i - 1].step + 1 ||
                  frame.day < history[i - 1].day)) {
      throw std::invalid_argument("public history is not continuous");
    }
  }
  if (history.back().step != input.current.step ||
      history.back().day != input.current.day ||
      history.back().market_inventory != input.current.public_market_inventory ||
      history.back().market_price != input.current.public_market_price) {
    throw std::invalid_argument("final public frame does not match current state");
  }

  auto forecast = input.production.forecast;
  if (forecast.ticks.size() < 24 || forecast.ticks.size() > 72 ||
      forecast.shed_capacity < 0 ||
      forecast.shed_capacity > input.current.official_shed_capacity) {
    throw std::invalid_argument("forecast lies outside general planner ABI");
  }
  Result result;
  result.audit.forecast_capacity = forecast.shed_capacity;
  for (std::size_t i = 0; i < forecast.ticks.size(); ++i) {
    auto& tick = forecast.ticks[i];
    if (tick.step != input.current.step + static_cast<int>(i)) {
      throw std::invalid_argument("production forecast is not continuous/current");
    }
    validate_inventory(tick.production, "negative forecast production");
    validate_inventory(tick.baseline_sale, "negative forecast baseline sale");
    validate_inventory(tick.town_drain, "negative forecast town drain");
    auto account = [&](const std::vector<Requirement>& requirements) {
      for (const auto& requirement : requirements) {
        validate_requirement(requirement);
        result.audit.total_cash_requirements += requirement.cash;
        if (requirement.kind == RequirementKind::CriticalPurchase)
          ++result.audit.critical_purchase_requirements;
        else if (requirement.kind == RequirementKind::Feed)
          ++result.audit.feed_requirements;
        else
          ++result.audit.route_hard_requirements;
      }
    };
    account(tick.pre_market_requirements);
    account(tick.requirements);
  }

  const auto& windows = input.market_schedule.remaining_sale_windows;
  if (!std::is_sorted(windows.begin(), windows.end()) ||
      std::adjacent_find(windows.begin(), windows.end()) != windows.end()) {
    throw std::invalid_argument("sale windows must be strictly increasing");
  }
  const int steps_remaining = input.episode_steps - 1 - input.current.step;
  for (int step : windows) {
    if (step < input.current.step || step > input.current.step + steps_remaining) {
      throw std::invalid_argument("sale window outside remaining episode");
    }
  }
  for (const auto& tick : forecast.ticks) {
    const bool listed = std::binary_search(windows.begin(), windows.end(), tick.step);
    if (tick.sale_window != listed) {
      throw std::invalid_argument("forecast and causal sale-window schedule disagree");
    }
  }

  std::set<int> drain_steps;
  for (const auto& drain : input.market_schedule.future_town_drain) {
    validate_inventory(drain.quantity, "negative future town drain");
    if (!drain_steps.insert(drain.step).second ||
        drain.step < input.current.step ||
        drain.step > input.current.step + steps_remaining ||
        drain.step >= input.current.step + static_cast<int>(forecast.ticks.size())) {
      throw std::invalid_argument("future town drain step is duplicate/outside forecast");
    }
    auto& target = forecast.ticks[
        static_cast<std::size_t>(drain.step - input.current.step)].town_drain;
    if (zero(target)) {
      target = drain.quantity;
      if (!zero(drain.quantity)) ++result.audit.town_drain_inserted;
    } else if (target == drain.quantity) {
      ++result.audit.town_drain_already_present;
    } else {
      throw std::invalid_argument("forecast has conflicting town drain; refuse double count");
    }
  }
  for (const auto& frame : history)
    result.audit.historical_town_drain_units_ignored += total(frame.known_town_drain);

  result.general.current.step = input.current.step;
  result.general.current.own_money = input.current.own_money;
  result.general.current.opponent_money = input.current.opponent_public_money;
  result.general.current.own_stock = input.current.own_shed;
  result.general.current.market_inventory = input.current.public_market_inventory;
  result.general.own_causal_forecast = std::move(forecast);
  result.general.public_market_history = history;
  result.general.opponent_stock_lower =
      input.public_belief.belief.total_interval.lower;
  result.general.opponent_stock_point = input.public_belief.belief.total;
  result.general.opponent_stock_upper =
      input.public_belief.belief.total_interval.upper;
  result.general.remaining_sale_windows = windows;

  g001::dump::Input dump_input;
  dump_input.public_history = history;
  dump_input.belief_lower = result.general.opponent_stock_lower;
  dump_input.belief_point = result.general.opponent_stock_point;
  dump_input.belief_upper = result.general.opponent_stock_upper;
  dump_input.current_step = input.current.step;
  dump_input.current_day = input.current.day;
  dump_input.steps_remaining = steps_remaining;
  dump_input.remaining_sale_windows = windows;
  const auto scenarios = g001::dump::generate(dump_input);
  result.causal_scenario_set = scenarios;

  auto append_pressure = [&](CausalPressureScenario audit) {
    const bool duplicate = std::any_of(
        result.causal_pressures.begin(), result.causal_pressures.end(),
        [&](const auto& prior) { return prior.quantity == audit.quantity; });
    if (!duplicate) {
      result.causal_pressures.push_back(audit);
      result.audit.pressures.push_back(std::move(audit));
    }
  };
  for (const auto& weighted : scenarios.scenarios) {
    Inventory pressure{};
    const auto& shed = shed_band(input.public_belief.belief,
                                 weighted.belief_band);
    for (const auto& dump : weighted.scenario.dumps) {
      if (dump.step != input.current.step || dump.product < 0 ||
          dump.product >= static_cast<int>(g001::market::product_count)) continue;
      const auto product = static_cast<std::size_t>(dump.product);
      pressure[product] = std::min(std::max(0, dump.quantity), shed[product]);
    }
    if (zero(pressure)) continue;

    CausalPressureScenario base;
    base.source_scenario_id = weighted.id;
    base.belief_band = weighted.belief_band;
    base.timing = weighted.timing;
    base.source_total = total(pressure);
    if (base.source_total <= input.current.official_shed_capacity) {
      base.quantity = pressure;
      append_pressure(std::move(base));
      continue;
    }

    // Per-product upper bounds are correlated.  Emitting their impossible sum
    // would make protected_queue invent an over-capacity rival shed.  Product-
    // local marginal scenarios preserve each conservative price stress.
    result.audit.marginal_upper_split = true;
    for (std::size_t p = 0; p < g001::market::product_count; ++p) {
      if (pressure[p] <= 0) continue;
      CausalPressureScenario local = base;
      local.split_correlated_marginal = true;
      local.quantity[p] = std::min(
          pressure[p], input.current.official_shed_capacity);
      append_pressure(std::move(local));
    }
  }
  result.audit.accepted = true;
  return result;
}

MaterializeResult materialize(const MaterializeInput& input) {
  MaterializeResult result;
  const bool virtual_end = input.virtual_no_own_sell_at_market_end;
  const bool valid_slot = virtual_end
      ? input.optional_slot == input.maximum_slots &&
            input.goal == PlacementGoal::SerialOpponentBefore
      : input.optional_slot >= 0 && input.optional_slot < input.maximum_slots &&
            input.optional_slot < static_cast<int>(input.own_candidate_queue.size());
  if (input.maximum_slots <= 0 || !valid_slot ||
      input.own_candidate_queue.size() >
          static_cast<std::size_t>(input.maximum_slots)) {
    throw std::invalid_argument("invalid optional slot/full own queue");
  }
  for (const auto& order : input.own_candidate_queue) {
    if (order.quantity < 0)
      throw std::invalid_argument("negative own candidate order quantity");
  }
  if (!virtual_end) {
    const auto& optional = input.own_candidate_queue[
        static_cast<std::size_t>(input.optional_slot)];
    if (optional.operation != Operation::Sell || optional.quantity <= 0) {
      throw std::invalid_argument("optional slot must identify a positive own SELL");
    }
  }
  validate_inventory(input.pressure.quantity, "negative causal pressure quantity");

  std::vector<std::size_t> products;
  for (std::size_t p = 0; p < g001::market::product_count; ++p)
    if (input.pressure.quantity[p] > 0) products.push_back(p);
  if (products.empty()) {
    result.status = MaterializeStatus::NoPressure;
    result.reason = "causal pressure has zero quantity";
    return result;
  }

  auto serial = [&](const std::vector<std::size_t>& assigned) {
    ExactSlotPressure exact;
    exact.relation = ExactRelation::OpponentBefore;
    exact.optional_slot = input.optional_slot;
    for (const std::size_t p : assigned) {
      exact.rival_queue.push_back(sell_order(p, input.pressure.quantity[p]));
      exact.quantity[p] = input.pressure.quantity[p];
    }
    return exact;
  };
  auto lockstep = [&](std::size_t product) {
    ExactSlotPressure exact;
    exact.relation = ExactRelation::Lockstep;
    exact.optional_slot = input.optional_slot;
    exact.rival_queue.resize(static_cast<std::size_t>(input.optional_slot + 1));
    exact.rival_queue[static_cast<std::size_t>(input.optional_slot)] =
        sell_order(product, input.pressure.quantity[product]);
    exact.quantity[product] = input.pressure.quantity[product];
    return exact;
  };

  if (input.goal == PlacementGoal::SerialOpponentBefore) {
    if (input.optional_slot == 0) {
      result.status = MaterializeStatus::Unrepresentable;
      result.reason =
          "own optional slot 0 has no earlier rival slot; serial OpponentFirst is impossible";
      return result;
    }
    if (products.size() <= static_cast<std::size_t>(input.optional_slot)) {
      result.exact_slot_pressure.push_back(serial(products));
      result.status = MaterializeStatus::Exact;
      result.abstract_opponent_first_represented = true;
      result.reason = virtual_end
          ? "all rival SELLs occupy official slots before the virtual no-own-SELL boundary"
          : "all rival SELLs occupy slots strictly before the optional SELL";
      return result;
    }
    if (input.overflow_policy == OverflowPolicy::FailClosed) {
      result.status = MaterializeStatus::Unrepresentable;
      result.reason =
          "not enough earlier slots for all pressure products";
      return result;
    }
    for (const std::size_t product : products)
      result.exact_slot_pressure.push_back(serial({product}));
    result.status = MaterializeStatus::Split;
    result.split_product_local = true;
    result.abstract_opponent_first_represented = true;
    result.reason =
        "multi-product pressure split into exact product-local earlier-slot scenarios";
    return result;
  }

  if (products.size() == 1) {
    result.exact_slot_pressure.push_back(lockstep(products.front()));
    result.status = MaterializeStatus::Exact;
    result.reason =
        "rival and own optional SELL share a slot and use official pre-commit lockstep quotes";
    return result;
  }
  if (input.overflow_policy == OverflowPolicy::FailClosed) {
    result.status = MaterializeStatus::Unrepresentable;
    result.reason = "one official slot cannot hold multiple rival products";
    return result;
  }
  for (const std::size_t product : products)
    result.exact_slot_pressure.push_back(lockstep(product));
  result.status = MaterializeStatus::Split;
  result.split_product_local = true;
  result.reason =
      "multi-product pressure split into product-local same-slot lockstep scenarios";
  return result;
}

}  // namespace planner_input
