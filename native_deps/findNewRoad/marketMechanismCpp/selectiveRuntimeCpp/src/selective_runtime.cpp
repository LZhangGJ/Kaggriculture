#include "selective_runtime.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace selective_runtime {
namespace {

using Clock = std::chrono::steady_clock;
using g001::market::Order;
using g001::market::Product;
using protected_queue::SlotOrigin;

std::int64_t micros(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

int item_index(production_obligation::Item item) {
  return static_cast<int>(item);
}

g001::market::PlayerMarketState own_market_state(const Input& input) {
  const auto& current = input.production.current;
  g001::market::PlayerMarketState own;
  own.money = input.current.own_money;
  for (int p = 0; p < production_obligation::kProducts; ++p)
    own.shed[static_cast<std::size_t>(p)] = current.shed[p];
  for (int crop = 0; crop < production_obligation::kCrops; ++crop)
    own.seeds[static_cast<std::size_t>(crop)] = current.seeds[crop];
  for (int animal = 0; animal < production_obligation::kAnimals; ++animal)
    own.animals[static_cast<std::size_t>(animal)] = current.shed[9 + animal];
  own.hires_today = current.hires_today;
  own.hands = std::max(0, static_cast<int>(current.actor_positions.size()) - 1);
  own.unlocked_quadrants = std::popcount(current.unlocked_mask & 0x0fU);
  return own;
}

g001::market::Inventory protected_stock(
    const production_obligation::CompileResult& production,
    const g001::market::PlayerMarketState& own) {
  g001::market::Inventory out{};
  std::set<int> seen;
  for (const auto& node : production.nodes) {
    if (node.kind != production_obligation::NodeKind::Pickup) continue;
    if (node.id >= 0 && !seen.insert(node.id).second) continue;
    const int item = item_index(node.item);
    if (item >= 0 && item < production_obligation::kProducts)
      out[static_cast<std::size_t>(item)] += std::max(0, node.quantity);
  }
  for (std::size_t p = 0; p < out.size(); ++p)
    out[p] = std::min(out[p], std::max(0, own.shed[p]));
  return out;
}

std::optional<economic_intent_bridge::Continuation> settle_prior(
    const Input& input, bool& confirmed) {
  confirmed = true;
  if (!input.continuation) {
    if (input.confirmed_prior_fill)
      throw std::invalid_argument("confirmed fill supplied without continuation");
    return std::nullopt;
  }
  auto prior = *input.continuation;
  if (!prior.pending) {
    if (input.confirmed_prior_fill)
      throw std::invalid_argument("confirmed fill supplied without pending proposal");
    return prior;
  }
  if (!input.confirmed_prior_fill) {
    confirmed = false;
    return prior;
  }
  prior.execution = g001::general_econ::settle_interval(
      prior.execution, *prior.pending, *input.confirmed_prior_fill);
  prior.pending.reset();
  return prior;
}

void validate_input(const Input& input) {
  if (input.production.current.step != input.current.step)
    throw std::invalid_argument("production/current step mismatch");
  if (input.own_market_frames.empty() ||
      input.own_market_frames.front().step != input.current.step)
    throw std::invalid_argument("current legacy queue is not explicit");
  if (input.maximum_market_orders <= 0 || input.maximum_market_orders > 10 ||
      input.shed_capacity < 0 || input.episode_steps <= 1 ||
      input.relaxation_step < 0 || input.compaction_minimum_step < 0 ||
      input.uncertainty_cash_buffer < 0 ||
      input.capacity_buffer < 0)
    throw std::invalid_argument("invalid selective runtime scalar");
  if (input.current.official_shed_capacity != input.shed_capacity ||
      input.production.current.shed_capacity != input.shed_capacity)
    throw std::invalid_argument("shed capacity mismatch");
  if (!std::isfinite(input.production.current.money) ||
      input.current.own_money < 0 ||
      input.current.own_money > std::numeric_limits<int>::max() ||
      input.production.current.money !=
          static_cast<double>(input.current.own_money))
    throw std::invalid_argument("own money is negative, nonintegral, out of compiler range, or mismatched");
  if (input.production.current.max_market_orders !=
      input.maximum_market_orders)
    throw std::invalid_argument("market order limit mismatch");
  if (input.public_belief.opponent_supply_horizon_steps < 0 ||
      input.public_belief.opponent_supply_horizon_steps > 24)
    throw std::invalid_argument("invalid public opponent supply horizon");

  std::int64_t shed_used = 0;
  for (const int quantity : input.production.current.shed) {
    if (quantity < 0)
      throw std::invalid_argument("negative current shed quantity");
    shed_used += quantity;
  }
  if (shed_used > input.shed_capacity)
    throw std::invalid_argument("current products and animals exceed shed capacity");
  for (const int quantity : input.production.current.seeds)
    if (quantity < 0)
      throw std::invalid_argument("negative current seed quantity");
  for (const auto& carried : input.production.current.carried)
    for (const int quantity : carried.quantity)
      if (quantity < 0)
        throw std::invalid_argument("negative carried quantity");
  for (const auto& frame : input.production.future_units)
    for (const auto& action : frame.actor_actions)
      if (action.quantity < 0)
        throw std::invalid_argument("negative future unit quantity");
  for (std::size_t p = 0; p < g001::market::product_count; ++p) {
    if (input.production.current.shed[p] != input.current.own_shed[p])
      throw std::invalid_argument("own shed mismatch");
    if (input.production.current.product_price[p] !=
        input.current.public_market_price[p])
      throw std::invalid_argument("market quote mismatch");
    if (input.current.public_market_inventory[p] < 0)
      throw std::invalid_argument("negative public market inventory");
    const int harvestable =
        input.public_belief.opponent_harvestable_now_upper[p];
    const int new_production = input.public_belief
        .opponent_new_production_within_horizon_upper[p];
    const int sellable =
        input.public_belief.opponent_sellable_within_horizon_upper[p];
    if (harvestable < 0 || new_production < 0 || sellable < 0)
      throw std::invalid_argument("negative public opponent supply bound");
    if (input.public_belief.opponent_supply_horizon_steps > 0) {
      const std::int64_t required = std::min<std::int64_t>(
          std::numeric_limits<int>::max(),
          static_cast<std::int64_t>(
              input.public_belief.belief.total_interval.upper[p]) +
              harvestable + new_production);
      if (sellable < required)
        throw std::invalid_argument(
            "public opponent horizon supply bound omits current or visible-producer exposure");
    }
  }
}

protected_queue::InputV2 queue_input(
    const Input& input,
    const production_obligation::CompileResult& production,
    const g001::market::PlayerMarketState& own,
    const g001::market::Inventory& reserve,
    const std::vector<Order>& legacy,
    const std::optional<protected_queue::OptionalSell>& optional,
    const std::vector<std::vector<Order>>& pressures,
    Product selected_product,
    int minimum_optional_slot) {
  protected_queue::InputV2 out;
  out.step = input.current.step;
  out.production_dag = production.nodes;
  out.queue.maximum_slots = input.maximum_market_orders;
  out.queue.shed_capacity = input.shed_capacity;
  out.queue.hire_cost_multiplier = input.production.current.farm_hand_cost_mult;
  out.queue.market_inventory = input.current.public_market_inventory;
  out.queue.own = own;
  out.queue.protected_stock = reserve;
  out.queue.legacy_orders = legacy;
  if (optional) out.queue.optional_sells.push_back(*optional);
  out.queue.restrict_legacy_sell_replacement = true;
  out.queue.replaceable_legacy_sell[static_cast<std::size_t>(selected_product)] = true;
  out.queue.opponent_first_pressure = pressures;
  out.queue.maximum_search_states = input.maximum_search_states;
  out.queue.minimum_optional_slot = minimum_optional_slot;
  return out;
}

std::vector<Order> orders(const protected_queue::ResultV2& value) {
  std::vector<Order> out;
  out.reserve(value.slots.size());
  for (const auto& slot : value.slots) out.push_back(slot.order);
  return out;
}

bool has_causal_inventory_band(
    const g001::dump::Result& scenarios,
    const g001::market::Inventory& inventory) {
  const bool exact = std::any_of(
      scenarios.scenarios.begin(), scenarios.scenarios.end(),
      [&](const auto& value) {
        return !value.scenario.realized_future &&
            value.scenario.belief_stock == inventory;
      });
  if (exact) return true;

  // dump::generate intentionally omits an all-zero belief band from the dump
  // Cartesian product.  In rolling::simulate that band has no executable
  // opponent dump; every no-dump scenario is outcome-identical because belief
  // stock is consulted only to cap explicit scenario.dumps.  Recognize that
  // exact semantic representative instead of rejecting a proven empty lower
  // band merely because its metadata was deduplicated.
  const bool empty = std::all_of(
      inventory.begin(), inventory.end(), [](int quantity) {
        return quantity == 0;
      });
  return empty && std::any_of(
      scenarios.scenarios.begin(), scenarios.scenarios.end(),
      [](const auto& value) {
        return !value.scenario.realized_future &&
            value.scenario.dumps.empty();
      });
}

}  // namespace

const char* fallback_name(Fallback value) {
  switch (value) {
    case Fallback::None: return "none";
    case Fallback::InvalidInput: return "invalid-input";
    case Fallback::ProductionUncertified: return "production-uncertified";
    case Fallback::LegacyForecastUncertified: return "legacy-forecast-uncertified";
    case Fallback::PlannerNoStrictImprovement: return "planner-no-strict-improvement";
    case Fallback::AwaitingObservation: return "awaiting-observation";
    case Fallback::OptionSwitchUncertified: return "option-switch-uncertified";
    case Fallback::NoOptionalIntent: return "no-optional-intent";
    case Fallback::ProvisionalQueueRejected: return "provisional-queue-rejected";
    case Fallback::OptionalSlotAmbiguous: return "optional-slot-ambiguous";
    case Fallback::ExactPressureUnrepresentable: return "exact-pressure-unrepresentable";
    case Fallback::FinalQueueRejected: return "final-queue-rejected";
    case Fallback::OptionalQuantityChanged: return "optional-quantity-changed";
    case Fallback::QueueInvariantFailed: return "queue-invariant-failed";
    case Fallback::RobustCertificateRejected: return "robust-certificate-rejected";
    case Fallback::PhaseGateRejected: return "phase-gate-rejected";
    case Fallback::OpponentBeliefTooWide: return "opponent-belief-too-wide";
    case Fallback::AccelerationOpportunityCostRejected:
      return "acceleration-opportunity-cost-rejected";
    case Fallback::CompactionOnlyNoCandidate:
      return "compaction-only-no-candidate";
    case Fallback::FullModeInvariant: return "full-mode-invariant";
  }
  return "unknown";
}

OpportunityCostAudit evaluate_acceleration_opportunity_cost(
    Product product,
    int desired_total,
    int legacy_total,
    const g001::market::Inventory& current_market_inventory,
    const g001::rolling::FixedForecast& forecast,
    bool require_equilibrium_boundary_value) {
  OpportunityCostAudit audit;
  if (desired_total <= legacy_total) return audit;
  audit.evaluated = true;

  const auto product_index = static_cast<std::size_t>(product);
  const std::int64_t current_inventory =
      std::max(0, current_market_inventory[product_index]);
  const std::int64_t marginal_inventory = std::min<std::int64_t>(
      std::numeric_limits<int>::max(),
      current_inventory + static_cast<std::int64_t>(desired_total) - 1);
  audit.current_marginal_quote = g001::market::price(
      product, static_cast<int>(marginal_inventory));

  // The official phase order is market, then town clearing.  Thus a drain on
  // tick i can first affect a sale window on tick i+1.  Ignore future player
  // transactions: this is an optimistic opportunity quote, which makes the
  // deployment gate deliberately conservative and fully causal.
  std::int64_t future_inventory = current_inventory;
  for (std::size_t index = 0; index + 1 < forecast.ticks.size(); ++index) {
    future_inventory = std::max<std::int64_t>(
        0, future_inventory -
               std::max(0, forecast.ticks[index].town_drain[product_index]));
    const auto& future_tick = forecast.ticks[index + 1];
    if (!future_tick.sale_window) continue;
    const int quote = g001::market::price(
        product, static_cast<int>(std::min<std::int64_t>(
                     std::numeric_limits<int>::max(), future_inventory)));
    if (!audit.has_future_sale_window ||
        quote > audit.best_town_only_future_quote) {
      audit.has_future_sale_window = true;
      audit.best_town_only_future_quote = quote;
      audit.best_future_step = future_tick.step;
    }
  }
  if (require_equilibrium_boundary_value) {
    const int equilibrium_quote = g001::market::parameters(product).base;
    if (equilibrium_quote > audit.best_town_only_future_quote) {
      audit.best_town_only_future_quote = equilibrium_quote;
      audit.best_future_step = -1;
      audit.used_equilibrium_boundary_value = true;
    }
  }
  audit.passed = !audit.has_future_sale_window ||
      audit.current_marginal_quote >= audit.best_town_only_future_quote;
  if (audit.used_equilibrium_boundary_value)
    audit.passed = audit.current_marginal_quote >=
        audit.best_town_only_future_quote;
  return audit;
}

Result run(const Input& input) noexcept {
  const auto total_begin = Clock::now();
  Result out;
  // Exception paths must never erase an active/pending option.  Replacement
  // occurs only after observation settlement succeeds.
  out.continuation = input.compaction_only
      ? std::nullopt : input.continuation;
  auto fallback = [&](Fallback reason, std::string message) -> Result {
    if (!input.own_market_frames.empty())
      out.orders = input.own_market_frames.front().orders;
    out.audit.selected = false;
    out.audit.fallback = reason;
    out.audit.reason = std::move(message);
    out.audit.timing.total_us = micros(total_begin, Clock::now());
    return out;
  };

  try {
    validate_input(input);
    out.orders = input.own_market_frames.front().orders;

    bool prior_confirmed = input.compaction_only;
    std::optional<economic_intent_bridge::Continuation> settled_prior;
    if (!input.compaction_only) {
      settled_prior = settle_prior(input, prior_confirmed);
      out.continuation = settled_prior;
      out.audit.previous_execution_observation_confirmed = prior_confirmed;
      if (!prior_confirmed)
        return fallback(Fallback::AwaitingObservation,
                        "previous optional SELL has no observation-confirmed fill interval");
    }

    auto begin = Clock::now();
    const auto obligation = production_obligation::compile(input.production);
    out.audit.timing.obligation_us = micros(begin, Clock::now());
    if (!obligation.feasible)
      return fallback(Fallback::ProductionUncertified,
                      "own non-market production obligation DAG is infeasible");

    begin = Clock::now();
    production_forecast::Input production_input;
    production_input.production = input.production;
    production_input.obligations = obligation;
    production_input.config = input.production_config;
    const auto production = production_forecast::compile(production_input);
    out.audit.timing.production_forecast_us = micros(begin, Clock::now());
    out.audit.production = production.audit;
    out.audit.production_certified = production.audit.accepted &&
                                     production.audit.cash_nodes_consistent;
    if (!out.audit.production_certified)
      return fallback(Fallback::ProductionUncertified,
                      "production forecast did not certify the raw own unit plan");

    begin = Clock::now();
    legacy_baseline_forecast::Input legacy_input;
    legacy_input.current_step = input.current.step;
    legacy_input.fixed_forecast_certified = out.audit.production_certified;
    legacy_input.fixed_forecast = production.forecast;
    legacy_input.own_market_frames = input.own_market_frames;
    const auto legacy = legacy_baseline_forecast::compile(legacy_input);
    out.audit.timing.legacy_forecast_us = micros(begin, Clock::now());
    out.audit.legacy = legacy.audit;
    out.audit.legacy_forecast_certified = legacy.audit.accepted;
    if (!legacy.audit.accepted)
      return fallback(Fallback::LegacyForecastUncertified, legacy.audit.reason);
    const auto own = own_market_state(input);

    // For products that cannot be bought during the market phase, public
    // inventory can only stay fixed or increase between two own SELL slots.
    // Moving an unchanged total quantity into its earliest existing SELL slot
    // therefore weakly increases own receipts and weakly decreases any rival
    // receipt at the same/later slots.  Across later turns, however, the price
    // change can alter an endogenous legacy branch.  The proved-lower mode
    // consequently requires a positive public lower bound on rival shed
    // stock; point/no-evidence modes are explicitly offline empirical probes,
    // not theorem-backed deployment gates.  WHEAT and FERTILIZER are excluded
    // because an intervening BUY_PRODUCT can reverse even the within-queue
    // monotonicity.
    if (input.current.step >= input.relaxation_step &&
        input.current.step >= input.compaction_minimum_step &&
        input.compaction_rival_evidence != CompactionRivalEvidence::Disabled) {
      out.audit.monotone_sell_compaction_evaluated = true;
      g001::market::Inventory available_now = own.shed;
      if (!legacy.forecast.ticks.empty()) {
        const auto& delta = legacy.forecast.ticks.front().pre_market_unit_delta;
        for (std::size_t p = 0; p < available_now.size(); ++p) {
          const std::int64_t adjusted =
              static_cast<std::int64_t>(available_now[p]) + delta[p];
          available_now[p] = static_cast<int>(std::clamp<std::int64_t>(
              adjusted, 0, std::numeric_limits<int>::max()));
        }
      }
      std::array<int, g001::market::product_count> first_slot{};
      std::array<int, g001::market::product_count> sell_orders{};
      std::array<std::int64_t, g001::market::product_count> sell_units{};
      first_slot.fill(-1);
      for (std::size_t slot = 0; slot < out.orders.size(); ++slot) {
        const auto& order = out.orders[slot];
        if (order.operation != g001::market::Operation::Sell ||
            order.quantity <= 0 || order.product == Product::Wheat ||
            order.product == Product::Fertilizer) continue;
        const auto p = static_cast<std::size_t>(order.product);
        if (first_slot[p] < 0) first_slot[p] = static_cast<int>(slot);
        ++sell_orders[p];
        sell_units[p] += order.quantity;
        if (sell_units[p] > std::numeric_limits<int>::max())
          throw std::overflow_error("same-product SELL compaction overflows int");
      }
      for (std::size_t p = 0; p < sell_orders.size(); ++p) {
        int rival_evidence = 1;
        if (input.compaction_rival_evidence ==
            CompactionRivalEvidence::ProvedShedLowerBound) {
          rival_evidence = input.public_belief.belief.shed_interval.lower[p];
        } else if (input.compaction_rival_evidence ==
                   CompactionRivalEvidence::CausalShedPointEstimate) {
          rival_evidence = input.public_belief.belief.shed[p];
        } else if (input.compaction_rival_evidence ==
                   CompactionRivalEvidence::RecentClearancePointEstimate) {
          rival_evidence =
              input.public_belief.belief.recent_clearance[p];
        } else if (input.compaction_rival_evidence ==
                   CompactionRivalEvidence::RecentClearanceProvedLowerBound) {
          rival_evidence = input.public_belief.belief
              .recent_clearance_interval.lower[p];
        }
        const auto product = static_cast<Product>(p);
        const std::int64_t rival_same_tick_upper = std::min<std::int64_t>(
            std::numeric_limits<int>::max(),
            static_cast<std::int64_t>(
                input.public_belief.belief.total_interval.upper[p]) +
                input.public_belief.opponent_harvestable_now_upper[p]);
        // `pre_market_unit_delta` is a production projection, not a sound
        // upper bound on every same-tick DROP/PICKUP acceptance.  Use the
        // official shed capacity for our side, as in the terminal plateau
        // proof, so the floor guard covers every incumbent committed SELL.
        const std::int64_t possible_units =
            static_cast<std::int64_t>(input.shed_capacity) +
            rival_same_tick_upper;
        // At the official $1 floor a successful SELL consumes stock without
        // increasing public inventory. Reordering the same total can then
        // change how many units enter the market and permanently change the
        // future price path.  Compaction is eligible only when every possible
        // same-tick unit remains strictly above that floor; with this guard,
        // unchanged product totals imply an identical post-tick inventory.
        const bool floor_free = possible_units > 0 &&
            g001::market::price(product,
                                input.current.public_market_inventory[p]) > 1 &&
            possible_units - 1 <= std::numeric_limits<int>::max() -
                static_cast<std::int64_t>(
                    input.current.public_market_inventory[p]) &&
            g001::market::price(
                product,
                input.current.public_market_inventory[p] +
                    static_cast<int>(possible_units - 1)) > 1;
        if (rival_evidence <= 0 || !floor_free ||
            sell_orders[p] < 2 ||
            available_now[p] <= 0 ||
            sell_units[p] < available_now[p]) continue;
        const int first = first_slot[p];
        out.orders[static_cast<std::size_t>(first)].quantity =
            static_cast<int>(sell_units[p]);
        for (std::size_t slot = static_cast<std::size_t>(first + 1);
             slot < out.orders.size(); ++slot) {
          auto& order = out.orders[slot];
          if (order.operation != g001::market::Operation::Sell ||
              static_cast<std::size_t>(order.product) != p ||
              order.quantity <= 0) continue;
          order = {g001::market::Operation::Pass, Product::Wheat,
                   g001::market::Animal::Goose, 1};
          ++out.audit.monotone_sell_compaction_orders;
        }
        ++out.audit.monotone_sell_compaction_products;
        out.audit.monotone_sell_compaction_units +=
            static_cast<int>(sell_units[p]);
      }
      out.audit.monotone_sell_compaction_selected =
          out.audit.monotone_sell_compaction_products > 0;
    }

    // Exact final-step theorem: after this market phase own inventory has no
    // terminal value.  Appending SELLs after every incumbent slot preserves
    // all purchases and their funding/order semantics.  Every committed unit
    // pays at least one coin, and adding own supply can only weakly reduce a
    // rival's same/later quote, never increase it.  This proof needs no rival
    // inventory estimate and is therefore valid even when the generic belief
    // band is too wide for exact rolling search.
    if (input.current.step == input.episode_steps - 2) {
      out.audit.exact_terminal_liquidation_evaluated = true;
      g001::market::Inventory available = own.shed;
      if (!legacy.forecast.ticks.empty()) {
        const auto& delta = legacy.forecast.ticks.front().pre_market_unit_delta;
        for (std::size_t p = 0; p < available.size(); ++p) {
          const std::int64_t adjusted = static_cast<std::int64_t>(available[p]) +
              delta[p];
          available[p] = static_cast<int>(std::clamp<std::int64_t>(
              adjusted, 0, std::numeric_limits<int>::max()));
        }
      }
      for (const auto& order : out.orders) {
        if (order.operation != g001::market::Operation::Sell ||
            order.quantity <= 0) continue;
        const auto p = static_cast<std::size_t>(order.product);
        available[p] = std::max(0, available[p] - order.quantity);
      }

      struct TerminalSale {
        Product product{};
        int quantity{};
        std::int64_t nominal_revenue{};
      };
      std::vector<TerminalSale> sales;
      for (std::size_t p = 0; p < available.size(); ++p) {
        if (available[p] <= 0) continue;
        const auto product = static_cast<Product>(p);
        sales.push_back({product, available[p], g001::market::sell_revenue(
            product, input.current.public_market_inventory[p], available[p])});
      }
      std::stable_sort(sales.begin(), sales.end(), [](const auto& left,
                                                       const auto& right) {
        return std::tie(left.nominal_revenue, left.quantity, left.product) >
               std::tie(right.nominal_revenue, right.quantity, right.product);
      });

      for (const auto& sale : sales) {
        if (out.orders.size() >=
            static_cast<std::size_t>(input.maximum_market_orders)) break;
        if (sale.product == Product::Wheat ||
            sale.product == Product::Fertilizer) continue;
        const auto p = static_cast<std::size_t>(sale.product);
        const std::int64_t rival_upper = std::min<std::int64_t>(
            input.shed_capacity,
            static_cast<std::int64_t>(
                input.public_belief.belief.total_interval.upper[p]) +
                input.public_belief.opponent_harvestable_now_upper[p]);
        // `pre_market_unit_delta` is a production-feasibility projection, not
        // an upper bound on same-tick DROP/PICKUP acceptance.  The official
        // shed capacity is the sound bound on all incumbent plus appended own
        // SELL commits.  Cover it and the rival upper jointly so no earlier or
        // same-slot unit can leave the certified rounded-price plateau.
        const std::int64_t possible_units =
            static_cast<std::int64_t>(input.shed_capacity) + rival_upper;
        const std::int64_t last_inventory =
            static_cast<std::int64_t>(input.current.public_market_inventory[p]) +
                possible_units - 1;
        if (possible_units <= 0 ||
            last_inventory > std::numeric_limits<int>::max() ||
            g001::market::price(sale.product,
                               input.current.public_market_inventory[p]) !=
                g001::market::price(sale.product,
                                    static_cast<int>(last_inventory))) {
          continue;
        }
        out.orders.push_back({g001::market::Operation::Sell, sale.product,
                              g001::market::Animal::Goose, sale.quantity});
        ++out.audit.exact_terminal_liquidation_orders;
        out.audit.exact_terminal_liquidation_units += sale.quantity;
        out.audit.exact_terminal_nominal_revenue += sale.nominal_revenue;
      }
      if (out.audit.exact_terminal_liquidation_orders > 0) {
        out.audit.exact_terminal_liquidation_selected = true;
        out.audit.selected = true;
        out.audit.fallback = Fallback::None;
        out.audit.phase_decision.mode =
            phased_takeover::Mode::SelectiveSellOverlay;
        out.audit.reason = out.audit.monotone_sell_compaction_selected
            ? "experimental same-product SELL compaction plus plateau-certified final-step liquidation"
            : "final-step liquidation appended a non-buyable-product SELL on a price plateau covering all public rival stock";
        out.continuation.reset();
        out.audit.timing.total_us = micros(total_begin, Clock::now());
        return out;
      }
    }

    if (out.audit.monotone_sell_compaction_selected) {
      out.audit.selected = true;
      out.audit.fallback = Fallback::None;
      out.audit.phase_decision.mode = phased_takeover::Mode::SelectiveSellOverlay;
      out.audit.reason =
          "compacted repeated non-buyable-product SELLs into their earliest incumbent slots with unchanged product totals";
      out.continuation.reset();
      out.audit.timing.total_us = micros(total_begin, Clock::now());
      return out;
    }

    if (input.compaction_only) {
      out.continuation.reset();
      return fallback(
          Fallback::CompactionOnlyNoCandidate,
          "compaction-only profile found no certified terminal or compaction candidate");
    }

    std::int64_t opponent_upper_units = 0;
    for (const int quantity : input.public_belief.belief.total_interval.upper)
      opponent_upper_units += quantity;
    if (input.maximum_opponent_units_for_exact_search <= 0 ||
        opponent_upper_units > input.maximum_opponent_units_for_exact_search) {
      return fallback(
          Fallback::OpponentBeliefTooWide,
          "public opponent inventory band is too wide for bounded exact rollout");
    }

    begin = Clock::now();
    planner_input::Input planner_boundary;
    planner_boundary.public_belief = input.public_belief;
    planner_boundary.current = input.current;
    planner_boundary.production = {legacy.forecast, true,
                                   production.audit.cash_nodes_consistent};
    planner_boundary.market_schedule = input.market_schedule;
    planner_boundary.episode_steps = input.episode_steps;
    const auto planner_compiled = planner_input::compile(planner_boundary);
    out.audit.timing.planner_input_us = micros(begin, Clock::now());
    out.audit.planner_input = planner_compiled.audit;

    begin = Clock::now();
    auto planner_config = input.planner_config;
    planner_config.rolling.threads = 1;
    const auto planner = g001::general::plan(planner_compiled.general, planner_config);
    out.audit.timing.planner_us = micros(begin, Clock::now());
    out.audit.planner_scenarios = planner.dump_audit;
    auto expected_planner_scenarios = planner_compiled.causal_scenario_set;
    if (planner_config.mechanisms.same_tick_order_robust) {
      for (auto& weighted : expected_planner_scenarios.scenarios) {
        weighted.scenario.same_tick_order =
            g001::rolling::SameTickOrder::OpponentFirst;
      }
    }
    out.audit.expected_planner_scenarios =
        expected_planner_scenarios.scenarios.size();
    out.audit.actual_planner_scenarios = planner.dump_audit.scenarios.size();
    out.audit.planner_scenario_set_replayed =
        planner_config.mechanisms.terminal_robust &&
        planner_input::exact_scenario_set_equal(expected_planner_scenarios,
                                                planner.dump_audit);
    if (planner.plan.score.worst_margin <=
        planner.plan.baseline_reference.worst_margin)
      return fallback(Fallback::PlannerNoStrictImprovement,
                      "general planner found no strict worst-margin improvement");

    const auto reserve = protected_stock(obligation, own);
    begin = Clock::now();
    economic_intent_bridge::Input bridge_input;
    bridge_input.own_stock = own.shed;
    bridge_input.protected_stock = reserve;
    bridge_input.public_market_inventory = input.current.public_market_inventory;
    bridge_input.sale_window = !legacy.forecast.ticks.empty() &&
                               legacy.forecast.ticks.front().sale_window;
    bridge_input.remaining_sale_windows = static_cast<int>(std::count_if(
        input.market_schedule.remaining_sale_windows.begin(),
        input.market_schedule.remaining_sale_windows.end(),
        [&](int step) { return step >= input.current.step; }));
    bridge_input.switch_directive = input.allow_observation_confirmed_pause_switch
        ? g001::general_econ::SwitchDirective::PausePrevious
        : g001::general_econ::SwitchDirective::RequireSame;
    bridge_input.continuation = settled_prior;
    const auto bridge = economic_intent_bridge::make_sell_intent(planner, bridge_input);
    out.audit.timing.bridge_us = micros(begin, Clock::now());
    out.audit.bridge = bridge.audit;
    out.audit.intent_mode = bridge.mode;
    out.audit.desired_total = bridge.desired_total;
    if (bridge.audit.reason == economic_intent_bridge::Reason::AwaitingExplicitSwitch)
      return fallback(Fallback::OptionSwitchUncertified, bridge.audit.explanation);

    const auto commit_state_only = [&]() -> Result {
      // State-only is deliberately narrower than a market replacement.  It
      // may record a causally selected option, but must not pause/switch an
      // active option or leave an executable pending proposal behind.
      if (bridge.desired_total != 0 || bridge.optional_sell ||
          bridge.continuation.pending || bridge.audit.switched ||
          bridge.audit.paused_previous || bridge.audit.resumed ||
          bridge.audit.cancelled_outstanding != 0) {
        return fallback(Fallback::NoOptionalIntent,
                        "state-only intent is not a safe exact-legacy continuation update");
      }
      out.continuation = bridge.continuation;
      out.audit.state_only_continuation_committed = true;
      out.audit.selected = false;
      out.audit.fallback = Fallback::None;
      out.audit.reason = "state-only continuation committed; market queue remains exact legacy";
      out.audit.timing.total_us = micros(total_begin, Clock::now());
      return out;
    };

    if (bridge.mode == economic_intent_bridge::IntentMode::StateOnlyContinuation)
      return commit_state_only();
    if (bridge.mode !=
            economic_intent_bridge::IntentMode::ReplaceSelectedProductTotal ||
        !bridge.selected_product || bridge.desired_total < 0)
      return fallback(Fallback::NoOptionalIntent, bridge.audit.explanation);

    const Product selected = *bridge.selected_product;
    if (!input.allow_buyable_product_overlay &&
        (selected == Product::Wheat || selected == Product::Fertilizer)) {
      return fallback(
          Fallback::RobustCertificateRejected,
          "native deployment excludes buyable production inputs from the sell-only opponent scenario model");
    }
    const bool zero_target = bridge.desired_total == 0;
    if ((zero_target && bridge.optional_sell) ||
        (!zero_target && (!bridge.optional_sell ||
                         bridge.optional_sell->product != selected ||
                         bridge.optional_sell->maximum_quantity !=
                             bridge.desired_total))) {
      return fallback(Fallback::NoOptionalIntent,
                      "bridge replacement mode and executable quantity disagree");
    }

    int selected_legacy_sell_units = 0;
    for (const auto& order : out.orders) {
      if (order.operation != g001::market::Operation::Sell ||
          order.product != selected || order.quantity <= 0)
        continue;
      if (selected_legacy_sell_units >
          std::numeric_limits<int>::max() - order.quantity)
        throw std::overflow_error("selected legacy SELL total overflow");
      selected_legacy_sell_units += order.quantity;
    }
    // Delaying an incumbent sale leaves the recovered price available to the
    // rival.  A finite dump-timing catalog cannot prove safety against every
    // adaptive drip schedule.  Permit a reduction only when public causal
    // inventory inference proves the rival has zero units of this product;
    // additions/accelerations remain eligible for the exact robust proof.
    const auto selected_index = static_cast<std::size_t>(selected);
    const std::int64_t rival_deferral_exposure =
        input.public_belief.opponent_supply_horizon_steps > 0
            ? input.public_belief
                  .opponent_sellable_within_horizon_upper[selected_index]
            : static_cast<std::int64_t>(
                  input.public_belief.belief.total_interval.upper[selected_index]) +
                  input.public_belief
                      .opponent_harvestable_now_upper[selected_index];
    if (!input.allow_sale_deferral &&
        bridge.audit.reason != economic_intent_bridge::Reason::NoSellableStock &&
        bridge.desired_total < selected_legacy_sell_units) {
      return fallback(
          Fallback::RobustCertificateRejected,
          "native deployment forbids reducing an incumbent executable SELL beyond the causal forecast horizon");
    }
    if (bridge.audit.reason != economic_intent_bridge::Reason::NoSellableStock &&
        bridge.desired_total < selected_legacy_sell_units &&
        rival_deferral_exposure > 0) {
      return fallback(
          Fallback::RobustCertificateRejected,
          "selected-product deferral exposes a recovered quote to rival current or visible-producer horizon supply");
    }
    out.audit.opportunity_cost = evaluate_acceleration_opportunity_cost(
        selected, bridge.desired_total, selected_legacy_sell_units,
        input.current.public_market_inventory,
        planner_compiled.general.own_causal_forecast,
        input.production_config.liquidate_own_at_end);
    if (!out.audit.opportunity_cost.passed) {
      std::ostringstream reason;
      reason << "selected-product acceleration has current marginal quote "
             << out.audit.opportunity_cost.current_marginal_quote
             << " below "
             << (out.audit.opportunity_cost.used_equilibrium_boundary_value
                     ? "the equilibrium boundary quote "
                     : "town-only future opportunity quote ")
             << out.audit.opportunity_cost.best_town_only_future_quote;
      if (!out.audit.opportunity_cost.used_equilibrium_boundary_value)
        reason << " at step " << out.audit.opportunity_cost.best_future_step;
      reason
             << " (desired=" << bridge.desired_total
             << ", legacy=" << selected_legacy_sell_units << ')';
      return fallback(Fallback::AccelerationOpportunityCostRejected,
                      reason.str());
    }
    // Replacing an already-zero selected-product total changes no executable
    // order.  Treat it as a state-only update instead of manufacturing a
    // queue/certificate proof for a nonexistent suppression.
    if (zero_target && selected_legacy_sell_units == 0)
      return commit_state_only();

    int minimum_optional_slot = 0;
    for (const auto& pressure : planner_compiled.causal_pressures) {
      const int positive_products = static_cast<int>(std::count_if(
          pressure.quantity.begin(), pressure.quantity.end(),
          [](int quantity) { return quantity > 0; }));
      minimum_optional_slot = std::max(minimum_optional_slot, positive_products);
    }
    if (!zero_target && minimum_optional_slot >= input.maximum_market_orders)
      return fallback(Fallback::ExactPressureUnrepresentable,
                      "causal pressure needs more earlier slots than the queue permits");
    if (zero_target) minimum_optional_slot = 0;
    begin = Clock::now();
    const auto provisional = protected_queue::compose_v2(queue_input(
        input, obligation, own, reserve, out.orders, bridge.optional_sell, {},
        selected, minimum_optional_slot));
    out.audit.timing.provisional_queue_us = micros(begin, Clock::now());
    out.audit.provisional_queue = provisional.audit;
    if (provisional.exact_legacy_fallback || !provisional.audit.certificate_passed)
      return fallback(Fallback::ProvisionalQueueRejected, provisional.reason);
    int optional_slot = -1;
    int provisional_selected_total = 0;
    for (std::size_t slot = 0; slot < provisional.slots.size(); ++slot) {
      if (provisional.slots[slot].origin == SlotOrigin::OptionalSell) {
        if (optional_slot >= 0) optional_slot = -2;
        else optional_slot = static_cast<int>(slot);
      }
      if (provisional.slots[slot].origin == SlotOrigin::OptionalSell ||
          provisional.slots[slot].origin == SlotOrigin::RequiredFundingSell)
        provisional_selected_total += provisional.slots[slot].order.quantity;
    }
    if (zero_target) {
      if (optional_slot != -1)
        return fallback(Fallback::OptionalSlotAmbiguous,
                        "zero replacement unexpectedly produced an optional SELL slot");
      // Explicit virtual boundary after all official slots.  It is not an
      // executable slot and can only prove serial OpponentBefore pressure for
      // a no-own-SELL replacement.
      optional_slot = input.maximum_market_orders;
      out.audit.virtual_no_sell_slot = true;
    } else if (optional_slot < 0) {
      return fallback(Fallback::OptionalSlotAmbiguous,
                      "provisional queue has zero or multiple optional slots");
    }
    if (provisional_selected_total != bridge.desired_total)
      return fallback(Fallback::OptionalQuantityChanged,
                      "provisional RequiredFunding+Optional total changed the planner target");

    begin = Clock::now();
    std::vector<std::vector<Order>> exact_pressures;
    std::vector<planner_input::ExactRelation> exact_relations;
    for (const auto& pressure : planner_compiled.causal_pressures) {
      PressureAudit pressure_audit;
      pressure_audit.source_scenario_id = pressure.source_scenario_id;
      pressure_audit.belief_band = pressure.belief_band;
      pressure_audit.abstract_order = pressure.abstract_proof_order;

      planner_input::MaterializeInput materialize_input;
      materialize_input.pressure = pressure;
      materialize_input.own_candidate_queue = orders(provisional);
      materialize_input.optional_slot = optional_slot;
      materialize_input.maximum_slots = input.maximum_market_orders;
      materialize_input.virtual_no_own_sell_at_market_end = zero_target;
      materialize_input.goal = planner_input::PlacementGoal::SerialOpponentBefore;
      materialize_input.overflow_policy = planner_input::OverflowPolicy::FailClosed;
      auto materialized = planner_input::materialize(materialize_input);
      pressure_audit.status = materialized.status;
      pressure_audit.reason = materialized.reason;
      for (const auto& exact : materialized.exact_slot_pressure) {
        exact_pressures.push_back(exact.rival_queue);
        exact_relations.push_back(exact.relation);
        pressure_audit.exact_relations.push_back(exact.relation);
      }
      out.audit.pressures.push_back(std::move(pressure_audit));
      if (materialized.status == planner_input::MaterializeStatus::Unrepresentable)
        return fallback(Fallback::ExactPressureUnrepresentable,
                        "a public causal pressure cannot be represented in exact slots");
      if (!materialized.abstract_opponent_first_represented ||
          materialized.exact_slot_pressure.empty() ||
          std::any_of(materialized.exact_slot_pressure.begin(),
                      materialized.exact_slot_pressure.end(),
                      [](const auto& exact) {
                        return exact.relation !=
                            planner_input::ExactRelation::OpponentBefore;
                      })) {
        return fallback(Fallback::ExactPressureUnrepresentable,
                        "materialized pressure lacks explicit serial OpponentBefore proof");
      }
    }
    out.audit.timing.pressure_materialize_us = micros(begin, Clock::now());

    begin = Clock::now();
    const auto candidate = protected_queue::compose_v2(queue_input(
        input, obligation, own, reserve, out.orders, bridge.optional_sell,
        exact_pressures, selected, minimum_optional_slot));
    out.audit.timing.final_queue_us = micros(begin, Clock::now());
    out.audit.final_queue = candidate.audit;
    if (candidate.exact_legacy_fallback || !candidate.audit.certificate_passed)
      return fallback(Fallback::FinalQueueRejected, candidate.reason);
    if (exact_pressures.size() != exact_relations.size() ||
        candidate.audit.scenarios.size() != exact_pressures.size() + 1)
      return fallback(Fallback::FinalQueueRejected,
                      "final exact pressure scenarios are not one-to-one with replay audits");

    int final_optional_slot = -1;
    for (std::size_t slot = 0; slot < candidate.slots.size(); ++slot) {
      if (candidate.slots[slot].origin != SlotOrigin::OptionalSell) continue;
      if (final_optional_slot >= 0) final_optional_slot = -2;
      else final_optional_slot = static_cast<int>(slot);
    }
    if (zero_target && final_optional_slot == -1)
      final_optional_slot = input.maximum_market_orders;
    if (final_optional_slot != optional_slot)
      return fallback(Fallback::ExactPressureUnrepresentable,
                      "final optional slot differs from the slot used to materialize pressure");
    for (const auto& rival : exact_pressures) {
      int last_active_sell = -1;
      for (std::size_t slot = 0; slot < rival.size(); ++slot)
        if (rival[slot].operation == g001::market::Operation::Sell &&
            rival[slot].quantity > 0)
          last_active_sell = static_cast<int>(slot);
      if (last_active_sell >= final_optional_slot)
        return fallback(Fallback::ExactPressureUnrepresentable,
                        "final queue no longer places every rival pressure SELL before optional SELL");
    }

    queue_invariant_proof::Input queue_proof_input;
    queue_proof_input.legacy_queue = out.orders;
    queue_proof_input.candidate = candidate;
    queue_proof_input.selected_product = selected;
    queue_proof_input.bridge_emitted_total_target = bridge.desired_total;
    queue_proof_input.previous_execution_observation_confirmed = prior_confirmed;
    queue_proof_input.movement_unchanged = true;
    queue_proof_input.exact_pressure_facts.push_back(
        {0, queue_invariant_proof::ExactRelation::NoPressure, true, false});
    for (std::size_t index = 0; index < exact_relations.size(); ++index) {
      const bool opponent_before =
          exact_relations[index] == planner_input::ExactRelation::OpponentBefore;
      queue_proof_input.exact_pressure_facts.push_back(
          {index + 1,
           opponent_before ? queue_invariant_proof::ExactRelation::OpponentBefore
                           : queue_invariant_proof::ExactRelation::Lockstep,
           true, opponent_before});
    }
    const auto queue_proof = queue_invariant_proof::derive(queue_proof_input);
    out.audit.queue_proof = queue_proof.audit;
    out.audit.invariants.bridge_quantity = bridge.desired_total;
    out.audit.invariants.final_optional_quantity =
        queue_proof.audit.selected_actual_units;
    out.audit.invariants.optional_slot = final_optional_slot;
    out.audit.invariants.baseline_slot_failures =
        queue_proof.slots.baseline_failures;
    out.audit.invariants.candidate_slot_failures =
        queue_proof.slots.candidate_failures;
    out.audit.invariants.required_funding_sells_identical =
        queue_proof.execution.required_funding_sells_unchanged;
    out.audit.invariants.legacy_non_sell_byte_identical =
        queue_proof.execution.legacy_non_sell_orders_unchanged;
    out.audit.invariants.purchase_timing_byte_identical =
        queue_proof.execution.purchase_timing_unchanged;
    out.audit.invariants.land_hire_timing_byte_identical =
        queue_proof.execution.land_hire_timing_unchanged;
    out.audit.invariants.only_selected_product_sell_changed =
        queue_proof.execution.only_optional_sell_changed;
    out.audit.invariants.non_target_sells_byte_identical =
        queue_proof.audit.accepted;
    out.audit.invariants.non_target_sell_origins_preserved =
        queue_proof.audit.accepted;
    if (queue_proof.audit.selected_actual_units != bridge.desired_total)
      return fallback(Fallback::OptionalQuantityChanged,
                      "final RequiredFunding+Optional total changed the planner target");
    if (!queue_proof.audit.accepted ||
        !queue_proof.audit.all_pressure_scenarios_abstract_opponent_first)
      return fallback(Fallback::QueueInvariantFailed,
                      !queue_proof.audit.accepted
                          ? queue_proof.audit.reason
                          : "exact queue facts do not all represent serial OpponentBefore pressure");
    if (zero_target &&
        (queue_proof.audit.selected_legacy_sell_units <= 0 ||
         queue_proof.audit.required_funding_units != 0 ||
         queue_proof.audit.optional_units != 0 ||
         !queue_proof.execution.replacement_materialized ||
         queue_proof.execution.selected_total != 0)) {
      return fallback(Fallback::QueueInvariantFailed,
                      "zero replacement lacks factual selected-SELL suppression proof");
    }

    begin = Clock::now();
    robust_certificate::Proof proof;
    proof.scenarios.public_observations_only = true;
    proof.scenarios.lower_band_evaluated =
        has_causal_inventory_band(
            planner.dump_audit, planner_compiled.general.opponent_stock_lower);
    proof.scenarios.point_band_evaluated =
        has_causal_inventory_band(
            planner.dump_audit, planner_compiled.general.opponent_stock_point);
    proof.scenarios.upper_band_evaluated =
        has_causal_inventory_band(
            planner.dump_audit, planner_compiled.general.opponent_stock_upper);
    proof.scenarios.exact_planner_scenario_set_replayed =
        out.audit.planner_scenario_set_replayed;
    proof.scenarios.replayed_scenario_count =
        out.audit.planner_scenario_set_replayed
            ? expected_planner_scenarios.scenarios.size() : 0;
    proof.execution = queue_proof.execution;
    if (zero_target && planner.plan.first.kind != g001::option::Kind::Hold) {
      const auto& provenance = bridge.audit.option;
      const auto& selected_option = planner.plan.first;
      const bool exact_option =
          provenance.kind == selected_option.kind &&
          provenance.product == selected_option.product &&
          provenance.quota == selected_option.quota &&
          provenance.window_steps == selected_option.window_steps &&
          provenance.reservation_price == selected_option.reservation_price &&
          provenance.target_inventory == selected_option.target_inventory &&
          provenance.impact_limit == selected_option.impact_limit;
      const bool causal_zero_action =
          bridge.audit.reason ==
              economic_intent_bridge::Reason::ReservationPause ||
          bridge.audit.reason == economic_intent_bridge::Reason::ImpactPause;
      proof.execution.zero_option_action_replayed =
          exact_option && causal_zero_action &&
          bridge.audit.requested_by_option > 0 &&
          bridge.audit.requested_after_stock_cap > 0 &&
          bridge.audit.emitted_quantity == 0;
    }
    proof.slots = queue_proof.slots;
    const auto robust = robust_certificate::compile_selective_sell(planner, proof);
    out.audit.timing.certificate_us = micros(begin, Clock::now());
    out.audit.robust = robust.audit;
    if (!robust.audit.accepted)
      return fallback(Fallback::RobustCertificateRejected, robust.audit.reason);

    begin = Clock::now();
    phase_input::Input phase;
    phase.step = input.current.step;
    phase.relaxation_step = input.relaxation_step;
    phase.shed_capacity = input.shed_capacity;
    phase.maximum_market_orders = input.maximum_market_orders;
    phase.uncertainty_cash_buffer = input.uncertainty_cash_buffer;
    phase.capacity_buffer = input.capacity_buffer;
    phase.pending_unconfirmed_execution = false;
    phase.future_plan_certified = out.audit.production_certified &&
                                   out.audit.legacy_forecast_certified;
    phase.own = own;
    phase.production = obligation;
    phase.protected_queue.evaluated = true;
    phase.protected_queue.certificate_passed = candidate.audit.certificate_passed;
    phase.protected_queue.pending_unconfirmed_execution = false;
    for (std::size_t scenario = 0; scenario < candidate.audit.scenarios.size(); ++scenario) {
      const auto& source = candidate.audit.scenarios[scenario];
      phase_input::FundingScenarioAudit target;
      target.evaluated = true;
      target.opponent_first = scenario == 0 ||
          (scenario - 1 < exact_relations.size() &&
           exact_relations[scenario - 1] == planner_input::ExactRelation::OpponentBefore);
      target.required_funding_requested_units =
          source.required_funding_requested_units;
      target.required_funding_committed_units =
          source.required_funding_committed_units;
      target.committed_required_funding_revenue =
          source.committed_required_funding_revenue;
      target.post_queue_shed_used =
          input.shed_capacity - source.candidate_capacity_margin;
      phase.protected_queue.scenarios.push_back(target);
    }
    phase.selective_sell = robust.certificate;
    // Full is intentionally left unevaluated: this module can never authorize it.
    const auto phase_built = phase_input::build(phase);
    out.audit.phase_input = phase_built.audit;
    const auto decision = phased_takeover::decide(phase_built.phased);
    out.audit.timing.phase_gate_us = micros(begin, Clock::now());
    out.audit.phase_decision = decision;
    out.audit.full_mode_observed = decision.mode == phased_takeover::Mode::FullTakeover;
    if (out.audit.full_mode_observed)
      return fallback(Fallback::FullModeInvariant,
                      "selective runtime observed forbidden FullTakeover mode");
    if (decision.mode != phased_takeover::Mode::SelectiveSellOverlay)
      return fallback(Fallback::PhaseGateRejected, decision.reason);

    if (zero_target && bridge.continuation.pending)
      return fallback(Fallback::QueueInvariantFailed,
                      "zero replacement cannot commit a tentative positive SELL");
    out.orders = orders(candidate);
    out.continuation = bridge.continuation;
    out.audit.tentative_pending_committed = bridge.continuation.pending.has_value();
    out.audit.selected = true;
    out.audit.fallback = Fallback::None;
    out.audit.reason = "all causal, exact-slot, queue, robust and phase certificates passed";
    out.audit.timing.total_us = micros(total_begin, Clock::now());
    return out;
  } catch (const std::exception& error) {
    return fallback(Fallback::InvalidInput, error.what());
  } catch (...) {
    return fallback(Fallback::InvalidInput, "unknown selective runtime error");
  }
}

}  // namespace selective_runtime
