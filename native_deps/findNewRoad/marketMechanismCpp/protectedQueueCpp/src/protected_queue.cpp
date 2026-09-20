#include "protected_queue.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace protected_queue {
namespace {

namespace market = g001::market;

bool is_sell(const market::Order& order) {
  return order.operation == market::Operation::Sell && order.quantity > 0;
}

bool is_protected(const market::Order& order) {
  return order.operation != market::Operation::Sell &&
         order.operation != market::Operation::Pass;
}

bool respects_protected_stock(const Input& input,
                              const std::vector<market::Order>& queue) {
  market::Inventory sold{};
  for (const auto& order : queue) {
    if (!is_sell(order)) continue;
    const auto product = static_cast<std::size_t>(order.product);
    if (product >= market::product_count) return false;
    sold[product] += order.quantity;
  }
  for (std::size_t product = 0; product < market::product_count; ++product) {
    const int usable = std::max(
        0, input.own.shed[product] - input.protected_stock[product]);
    if (sold[product] > usable) return false;
  }
  return true;
}

int required_fill(const market::Order& order) {
  if (order.operation == market::Operation::Hire ||
      order.operation == market::Operation::BuyLand) return 1;
  return std::max(0, order.quantity);
}

int shed_used(const market::PlayerMarketState& player) {
  int used = 0;
  for (const int value : player.shed) used += std::max(0, value);
  for (const int value : player.animals) used += std::max(0, value);
  return used;
}

std::int64_t marked_value(const market::PlayerMarketState& player,
                          const market::Inventory& inventory) {
  static constexpr std::array<int, 5> seed_cost{10, 20, 50, 100, 80};
  static constexpr std::array<int, 3> animal_cost{300, 400, 500};
  std::int64_t value = player.money;
  for (std::size_t product = 0; product < market::product_count; ++product) {
    value += static_cast<std::int64_t>(std::max(0, player.shed[product])) *
             market::price(static_cast<market::Product>(product), inventory[product]);
  }
  for (std::size_t crop = 0; crop < seed_cost.size(); ++crop)
    value += static_cast<std::int64_t>(std::max(0, player.seeds[crop])) * seed_cost[crop];
  for (std::size_t animal = 0; animal < animal_cost.size(); ++animal)
    value += static_cast<std::int64_t>(std::max(0, player.animals[animal])) *
             animal_cost[animal];
  return value;
}

struct Pressure {
  std::vector<market::Order> queue;
  market::PlayerMarketState rival;
};

Pressure pressure_state(const std::vector<market::Order>& queue,
                        int maximum_slots, int shed_capacity) {
  if (queue.size() > static_cast<std::size_t>(maximum_slots))
    throw std::invalid_argument("pressure queue exceeds market slot limit");
  Pressure result;
  result.queue = queue;
  int total = 0;
  for (const auto& order : queue) {
    if (order.operation == market::Operation::Pass || order.quantity == 0) continue;
    if (order.operation != market::Operation::Sell || order.quantity < 0)
      throw std::invalid_argument("pressure queue may contain only nonnegative SELLs");
    const auto product = static_cast<std::size_t>(order.product);
    if (product >= market::product_count)
      throw std::invalid_argument("pressure SELL has invalid product");
    result.rival.shed[product] += order.quantity;
    total += order.quantity;
  }
  if (total > shed_capacity)
    throw std::invalid_argument("pressure queue implies more rival stock than shed capacity");
  return result;
}

struct Evaluation {
  std::vector<int> protected_failures;
  int required_sell_unfilled{};
  int optional_sell_unfilled{};
  market::QueueResult exact;
};

Evaluation evaluate(const Input& input, const Pressure& pressure,
                    const std::vector<market::Order>& queue,
                    const std::vector<bool>& required_sell,
                    const std::vector<bool>& optional_sell) {
  std::array<market::PlayerMarketState, 2> players{pressure.rival, input.own};
  std::array<std::vector<market::Order>, 2> queues{pressure.queue, queue};
  Evaluation result;
  result.exact = market::simulate_queue(
      input.market_inventory, players, queues, input.maximum_slots,
      input.shed_capacity, input.hire_cost_multiplier);
  const auto committed_at = [&](std::size_t slot) {
    return slot < result.exact.committed[1].size() ? result.exact.committed[1][slot] : 0;
  };
  for (std::size_t slot = 0; slot < input.legacy_orders.size(); ++slot) {
    if (!is_protected(input.legacy_orders[slot])) continue;
    result.protected_failures.push_back(
        std::max(0, required_fill(input.legacy_orders[slot]) - committed_at(slot)));
  }
  for (std::size_t slot = 0; slot < queue.size(); ++slot) {
    if (slot < required_sell.size() && required_sell[slot])
      result.required_sell_unfilled +=
          std::max(0, queue[slot].quantity - committed_at(slot));
    if (slot < optional_sell.size() && optional_sell[slot])
      result.optional_sell_unfilled +=
          std::max(0, queue[slot].quantity - committed_at(slot));
  }
  return result;
}

bool hard_passes(const Evaluation& evaluation, bool require_sells) {
  const bool protected_ok = std::all_of(
      evaluation.protected_failures.begin(), evaluation.protected_failures.end(),
      [](int value) { return value == 0; });
  return protected_ok && (!require_sells || evaluation.required_sell_unfilled == 0);
}

std::vector<market::Order> skeleton(const Input& input,
                                    const std::vector<int>& sell_slots,
                                    const std::vector<int>& quantities) {
  auto queue = input.legacy_orders;
  for (std::size_t index = 0; index < sell_slots.size(); ++index) {
    auto& order = queue[static_cast<std::size_t>(sell_slots[index])];
    if (quantities[index] <= 0) order = market::Order{};
    else order.quantity = quantities[index];
  }
  return queue;
}

bool all_hard_pass(const Input& input, const std::vector<Pressure>& pressures,
                   const std::vector<market::Order>& queue,
                   const std::vector<bool>& required_sell,
                   bool require_sells, bool enforce_protected_stock = true) {
  if (enforce_protected_stock && !respects_protected_stock(input, queue))
    return false;
  const std::vector<bool> no_optional(queue.size(), false);
  for (const auto& pressure : pressures) {
    if (!hard_passes(evaluate(input, pressure, queue, required_sell, no_optional),
                     require_sells)) return false;
  }
  return true;
}

std::vector<market::Order> trim_trailing_passes(std::vector<market::Order> queue) {
  while (!queue.empty() && queue.back().operation == market::Operation::Pass)
    queue.pop_back();
  return queue;
}

void validate(const Input& input) {
  if (input.maximum_slots <= 0 || input.shed_capacity < 0 ||
      input.hire_cost_multiplier <= 0 || input.maximum_search_states == 0)
    throw std::invalid_argument("invalid protected queue limits");
  if (input.minimum_optional_slot < 0 ||
      input.minimum_optional_slot >= input.maximum_slots)
    throw std::invalid_argument(
        "minimum optional slot lies outside market slot limit");
  if (input.legacy_orders.size() > static_cast<std::size_t>(input.maximum_slots))
    throw std::invalid_argument("legacy queue exceeds market slot limit");
  for (const auto& order : input.legacy_orders) {
    if (order.quantity < 0) throw std::invalid_argument("negative legacy quantity");
  }
  for (const auto& sell : input.optional_sells) {
    if (sell.maximum_quantity < 0 || sell.reservation_price < 0 ||
        sell.impact_limit < 0)
      throw std::invalid_argument("negative optional SELL field");
  }
  for (std::size_t product = 0; product < market::product_count; ++product) {
    if (input.protected_stock[product] < 0 ||
        input.protected_stock[product] > input.own.shed[product])
      throw std::invalid_argument("protected stock lies outside own shed");
  }
}

}  // namespace

Result compose(const Input& input) {
  validate(input);
  Result result;
  result.orders = input.legacy_orders;
  result.audit.worst_marked_margin = std::numeric_limits<std::int64_t>::max();
  for (const auto& order : input.legacy_orders)
    result.audit.protected_non_sell_orders += is_protected(order);

  std::vector<Pressure> pressures;
  pressures.push_back(pressure_state({}, input.maximum_slots, input.shed_capacity));
  for (const auto& queue : input.opponent_first_pressure)
    pressures.push_back(pressure_state(queue, input.maximum_slots, input.shed_capacity));

  const std::vector<bool> legacy_sell_flags(input.legacy_orders.size(), false);
  const std::vector<bool> legacy_optional_flags(input.legacy_orders.size(), false);
  auto legacy_fallback = [&](std::string reason) {
    result.orders = input.legacy_orders;
    result.origins.assign(result.orders.size(), SlotOrigin::ExactLegacyFallback);
    result.audit.used_legacy_fallback = true;
    result.audit.certificate_passed = false;
    result.audit.minimum_proven = false;
    result.audit.required_funding_sell_orders = 0;
    result.audit.required_funding_sell_units = 0;
    result.audit.optional_replacement_orders = 0;
    result.audit.optional_replacement_units = 0;
    result.audit.worst_marked_margin = 0;
    result.audit.scenarios.clear();
    for (const auto& pressure : pressures) {
      const auto legacy = evaluate(input, pressure, input.legacy_orders,
                                   legacy_sell_flags, legacy_optional_flags);
      ScenarioAudit audit;
      audit.legacy_failure_vector = legacy.protected_failures;
      audit.candidate_failure_vector = legacy.protected_failures;
      audit.legacy_final_money = legacy.exact.players[1].money;
      audit.candidate_final_money = legacy.exact.players[1].money;
      audit.legacy_capacity_margin = input.shed_capacity - shed_used(legacy.exact.players[1]);
      audit.candidate_capacity_margin = audit.legacy_capacity_margin;
      audit.marked_margin_vs_legacy = 0;
      result.audit.scenarios.push_back(std::move(audit));
    }
    result.audit.reason = std::move(reason);
    return result;
  };
  for (const auto& pressure : pressures) {
    if (!hard_passes(evaluate(input, pressure, input.legacy_orders,
                              legacy_sell_flags, legacy_optional_flags), false)) {
      return legacy_fallback(
          "legacy protected non-SELL already fails a pressure scenario; preserve legacy exactly");
    }
  }

  std::vector<int> sell_slots;
  std::vector<int> caps;
  for (std::size_t slot = 0; slot < input.legacy_orders.size(); ++slot) {
    const auto& order = input.legacy_orders[slot];
    if (!is_sell(order)) continue;
    const auto product = static_cast<std::size_t>(order.product);
    if (input.restrict_legacy_sell_replacement &&
        !input.replaceable_legacy_sell[product]) {
      ++result.audit.preserved_legacy_sell_orders;
      result.audit.preserved_legacy_sell_units += order.quantity;
      continue;
    }
    sell_slots.push_back(static_cast<int>(slot));
    caps.push_back(order.quantity);
  }

  std::vector<int> best;
  int best_units = std::numeric_limits<int>::max();
  bool exhausted = false;
  std::vector<int> selected(sell_slots.size(), 0);
  std::function<void(std::size_t, int)> search = [&](std::size_t at, int units) {
    if (exhausted || units >= best_units) return;
    if (++result.audit.search_states > input.maximum_search_states) {
      exhausted = true;
      return;
    }
    std::vector<int> optimistic = selected;
    for (std::size_t index = at; index < optimistic.size(); ++index)
      optimistic[index] = caps[index];
    auto optimistic_queue = skeleton(input, sell_slots, optimistic);
    std::vector<bool> optimistic_flags(optimistic_queue.size(), false);
    for (std::size_t index = 0; index < sell_slots.size(); ++index)
      optimistic_flags[static_cast<std::size_t>(sell_slots[index])] = optimistic[index] > 0;
    if (!all_hard_pass(input, pressures, optimistic_queue, optimistic_flags,
                       false, false)) return;

    if (at == selected.size()) {
      auto queue = skeleton(input, sell_slots, selected);
      std::vector<bool> flags(queue.size(), false);
      for (std::size_t index = 0; index < sell_slots.size(); ++index)
        flags[static_cast<std::size_t>(sell_slots[index])] = selected[index] > 0;
      if (all_hard_pass(input, pressures, queue, flags, true, true)) {
        best = selected;
        best_units = units;
      }
      return;
    }
    const int upper = std::min(caps[at], best_units - units - 1);
    for (int quantity = 0; quantity <= upper; ++quantity) {
      selected[at] = quantity;
      search(at + 1, units + quantity);
      if (exhausted) return;
    }
    selected[at] = 0;
  };
  search(0, 0);

  if (exhausted || (best.empty() && !sell_slots.empty() && best_units ==
                                       std::numeric_limits<int>::max())) {
    return legacy_fallback(exhausted
        ? "exact minimum search budget exhausted; preserve legacy exactly"
        : "no retained legacy SELL subset certifies protected orders; preserve legacy exactly");
  }
  if (sell_slots.empty()) best.clear();
  result.audit.minimum_proven = true;

  auto queue = skeleton(input, sell_slots, best);
  std::vector<bool> required_flags(queue.size(), false);
  std::vector<bool> optional_flags(queue.size(), false);
  std::vector<int> optional_reservation(queue.size(), 0);
  std::vector<int> optional_impact(
      queue.size(), std::numeric_limits<int>::max());
  for (std::size_t index = 0; index < sell_slots.size(); ++index) {
    if (best[index] <= 0) continue;
    required_flags[static_cast<std::size_t>(sell_slots[index])] = true;
    ++result.audit.required_funding_sell_orders;
    result.audit.required_funding_sell_units += best[index];
  }

  auto candidates = input.optional_sells;
  std::stable_sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
    return std::tie(left.priority, left.maximum_quantity) >
           std::tie(right.priority, right.maximum_quantity);
  });
  std::array<int, market::product_count> reserved{};
  for (std::size_t slot = 0; slot < queue.size(); ++slot) {
    if (required_flags[slot])
      reserved[static_cast<std::size_t>(queue[slot].product)] += queue[slot].quantity;
  }

  // Distinguish a true slot-budget failure from a planner proposal whose
  // quantity is already covered by RequiredFundingSell. In the former case we
  // must not silently move the optional order into the reserved serial prefix.
  bool optional_tail_needed = false;
  for (const auto& candidate : candidates) {
    const auto product = static_cast<std::size_t>(candidate.product);
    const int available = std::max(
        0, input.own.shed[product] - input.protected_stock[product] -
               reserved[product]);
    const int remaining_target = std::max(
        0, candidate.maximum_quantity - reserved[product]);
    if (std::min(remaining_target, available) > 0) {
      optional_tail_needed = true;
      break;
    }
  }
  const std::size_t minimum_optional_slot =
      static_cast<std::size_t>(input.minimum_optional_slot);
  bool has_eligible_optional_slot = false;
  for (std::size_t slot = minimum_optional_slot; slot < queue.size(); ++slot) {
    if (queue[slot].operation == market::Operation::Pass) {
      has_eligible_optional_slot = true;
      break;
    }
  }
  if (!has_eligible_optional_slot) {
    const std::size_t append_slot =
        std::max(queue.size(), minimum_optional_slot);
    has_eligible_optional_slot =
        append_slot < static_cast<std::size_t>(input.maximum_slots);
  }
  if (optional_tail_needed && !has_eligible_optional_slot) {
    return legacy_fallback(
        "no market slot at or after minimum optional slot; preserve legacy exactly");
  }

  std::size_t next_candidate = 0;
  auto try_slot = [&](std::size_t slot) {
    while (next_candidate < candidates.size()) {
      const auto candidate = candidates[next_candidate++];
      const auto product = static_cast<std::size_t>(candidate.product);
      const int available = std::max(
          0, input.own.shed[product] - input.protected_stock[product] -
                 reserved[product]);
      // OptionalSell is the planner's total selected-product target, not an
      // amount to add on top of a legacy prefix that must remain for funding.
      // Count retained same-product funding units toward that target.
      const int remaining_target = std::max(
          0, candidate.maximum_quantity - reserved[product]);
      const int quantity = std::min(remaining_target, available);
      if (quantity <= 0) continue;
      if (slot >= queue.size()) {
        queue.resize(slot + 1);
        required_flags.resize(slot + 1, false);
        optional_flags.resize(slot + 1, false);
        optional_reservation.resize(slot + 1, 0);
        optional_impact.resize(slot + 1, std::numeric_limits<int>::max());
      }
      const auto previous = queue[slot];
      int selected_quantity = 0;
      for (int trial = quantity; trial >= 1 && selected_quantity == 0; --trial) {
        queue[slot] = {market::Operation::Sell, candidate.product,
                       market::Animal::Goose, trial};
        optional_flags[slot] = true;
        bool passed = true;
        const int initial_quote = market::price(
            candidate.product, input.market_inventory[product]);
        for (const auto& pressure : pressures) {
          const auto evaluation = evaluate(
              input, pressure, queue, required_flags, optional_flags);
          const int minimum_quote =
              slot < evaluation.exact.minimum_committed_quote[1].size()
                  ? evaluation.exact.minimum_committed_quote[1][slot] : -1;
          if (!hard_passes(evaluation, true) ||
              evaluation.optional_sell_unfilled != 0 ||
              minimum_quote < candidate.reservation_price ||
              initial_quote - minimum_quote > candidate.impact_limit) {
            passed = false;
            break;
          }
        }
        if (passed) selected_quantity = trial;
      }
      if (selected_quantity <= 0) {
        queue[slot] = previous;
        optional_flags[slot] = false;
        continue;
      }
      queue[slot].quantity = selected_quantity;
      optional_reservation[slot] = candidate.reservation_price;
      optional_impact[slot] = candidate.impact_limit;
      reserved[product] += selected_quantity;
      ++result.audit.optional_replacement_orders;
      result.audit.optional_replacement_units += selected_quantity;
      return;
    }
  };
  for (std::size_t slot = minimum_optional_slot;
       slot < queue.size() && next_candidate < candidates.size(); ++slot) {
    if (queue[slot].operation == market::Operation::Pass) try_slot(slot);
  }
  while (next_candidate < candidates.size()) {
    const auto before = queue.size();
    const auto slot = std::max(before, minimum_optional_slot);
    if (slot >= static_cast<std::size_t>(input.maximum_slots)) break;
    try_slot(slot);
    if (queue.size() == before && next_candidate >= candidates.size()) break;
  }
  queue = trim_trailing_passes(std::move(queue));
  required_flags.resize(queue.size());
  optional_flags.resize(queue.size());
  optional_reservation.resize(queue.size());
  optional_impact.resize(queue.size());

  result.audit.certificate_passed = true;
  for (std::size_t scenario = 0; scenario < pressures.size(); ++scenario) {
    const auto legacy = evaluate(input, pressures[scenario], input.legacy_orders,
                                 legacy_sell_flags, legacy_optional_flags);
    const auto candidate = evaluate(input, pressures[scenario], queue,
                                    required_flags, optional_flags);
    int price_band_failures = 0;
    for (std::size_t slot = 0; slot < queue.size(); ++slot) {
      if (!optional_flags[slot]) continue;
      const auto product = static_cast<std::size_t>(queue[slot].product);
      const int initial_quote = market::price(
          queue[slot].product, input.market_inventory[product]);
      const int minimum_quote =
          slot < candidate.exact.minimum_committed_quote[1].size()
              ? candidate.exact.minimum_committed_quote[1][slot] : -1;
      price_band_failures += minimum_quote < optional_reservation[slot] ||
          initial_quote - minimum_quote > optional_impact[slot];
    }
    if (!hard_passes(candidate, true) || candidate.optional_sell_unfilled != 0 ||
        price_band_failures > 0) {
      return legacy_fallback("final exact certificate failed; preserve legacy exactly");
    }
    ScenarioAudit audit;
    audit.legacy_failure_vector = legacy.protected_failures;
    audit.candidate_failure_vector = candidate.protected_failures;
    audit.required_sell_unfilled = candidate.required_sell_unfilled;
    audit.optional_sell_unfilled = candidate.optional_sell_unfilled;
    audit.optional_price_band_failures = price_band_failures;
    for (std::size_t slot = 0; slot < queue.size(); ++slot) {
      if (slot >= required_flags.size() || !required_flags[slot]) continue;
      audit.required_funding_requested_units += queue[slot].quantity;
      if (slot < candidate.exact.committed[1].size())
        audit.required_funding_committed_units +=
            candidate.exact.committed[1][slot];
      if (slot < candidate.exact.committed_cash[1].size())
        audit.committed_required_funding_revenue +=
            std::max<std::int64_t>(0, candidate.exact.committed_cash[1][slot]);
    }
    audit.legacy_final_money = legacy.exact.players[1].money;
    audit.candidate_final_money = candidate.exact.players[1].money;
    audit.legacy_capacity_margin = input.shed_capacity - shed_used(legacy.exact.players[1]);
    audit.candidate_capacity_margin = input.shed_capacity - shed_used(candidate.exact.players[1]);
    audit.marked_margin_vs_legacy =
        marked_value(candidate.exact.players[1], candidate.exact.market_inventory) -
        marked_value(legacy.exact.players[1], legacy.exact.market_inventory);
    result.audit.worst_marked_margin = std::min(
        result.audit.worst_marked_margin, audit.marked_margin_vs_legacy);
    result.audit.scenarios.push_back(std::move(audit));
  }
  if (result.audit.worst_marked_margin == std::numeric_limits<std::int64_t>::max())
    result.audit.worst_marked_margin = 0;
  result.orders = std::move(queue);
  result.origins.resize(result.orders.size(), SlotOrigin::Pass);
  for (std::size_t slot = 0; slot < result.orders.size(); ++slot) {
    if (slot < optional_flags.size() && optional_flags[slot]) {
      result.origins[slot] = SlotOrigin::OptionalSell;
    } else if (slot < required_flags.size() && required_flags[slot]) {
      result.origins[slot] = SlotOrigin::RequiredFundingSell;
    } else if (is_protected(result.orders[slot])) {
      result.origins[slot] = SlotOrigin::LegacyNonSell;
    } else if (is_sell(result.orders[slot])) {
      result.origins[slot] = SlotOrigin::PreservedLegacySell;
    }
  }
  result.audit.reason =
      "protected non-SELLs and retained/optional SELL fills certified in every exact slot-aligned pressure scenario";
  return result;
}

namespace {

bool due_purchase(const production_obligation::ObligationNode& node, int step) {
  using production_obligation::NodeKind;
  const bool purchase = node.kind == NodeKind::BuySeed ||
      node.kind == NodeKind::BuyProduct || node.kind == NodeKind::BuyAnimal ||
      node.kind == NodeKind::Hire || node.kind == NodeKind::BuyLand;
  return purchase && node.execution_step == step;
}

bool legacy_matches(const market::Order& order,
                    const production_obligation::ObligationNode& node) {
  using production_obligation::NodeKind;
  const int item = static_cast<int>(node.item);
  switch (node.kind) {
    case NodeKind::BuySeed:
      return order.operation == market::Operation::BuySeed && item >= 0 &&
             item < 5 && static_cast<int>(order.product) == item;
    case NodeKind::BuyProduct:
      return order.operation == market::Operation::BuyProduct &&
             static_cast<int>(order.product) == item;
    case NodeKind::BuyAnimal:
      return order.operation == market::Operation::BuyAnimal && item >= 9 &&
             item <= 11 && static_cast<int>(order.animal) == item - 9;
    case NodeKind::Hire:
      return order.operation == market::Operation::Hire;
    case NodeKind::BuyLand:
      return order.operation == market::Operation::BuyLand;
    default:
      return false;
  }
}

}  // namespace

ResultV2 compose_v2(const InputV2& input) {
  if (input.abi_version != InputV2::supported_abi_version)
    throw std::invalid_argument("unsupported protected queue ABI version");
  if (input.step < 0) throw std::invalid_argument("negative V2 step");

  ResultV2 output;
  auto exact_fallback = [&](std::string reason) {
    output.slots.clear();
    output.slots.reserve(input.queue.legacy_orders.size());
    for (const auto& order : input.queue.legacy_orders) {
      output.slots.push_back(
          {order, SlotOrigin::ExactLegacyFallback, -1});
    }
    output.audit.used_legacy_fallback = true;
    output.audit.certificate_passed = false;
    output.audit.reason = reason;
    output.exact_legacy_fallback = true;
    output.reason = std::move(reason);
    return output;
  };

  std::vector<const production_obligation::ObligationNode*> due;
  for (const auto& node : input.production_dag)
    if (due_purchase(node, input.step)) due.push_back(&node);
  std::stable_sort(due.begin(), due.end(), [](const auto* left, const auto* right) {
    return std::tie(left->order_slot, left->id) <
           std::tie(right->order_slot, right->id);
  });

  std::vector<int> remaining(input.queue.legacy_orders.size(), 0);
  for (std::size_t slot = 0; slot < input.queue.legacy_orders.size(); ++slot) {
    if (is_protected(input.queue.legacy_orders[slot]))
      remaining[slot] = required_fill(input.queue.legacy_orders[slot]);
  }
  std::vector<int> first_node_for_slot(input.queue.legacy_orders.size(), -1);
  for (const auto* node : due) {
    const int needed = node->kind == production_obligation::NodeKind::Hire ||
                               node->kind == production_obligation::NodeKind::BuyLand
                           ? 1
                           : std::max(0, node->quantity);
    if (needed <= 0)
      return exact_fallback("current hard acquisition has nonpositive quantity");
    int match = -1;
    for (std::size_t slot = 0; slot < input.queue.legacy_orders.size(); ++slot) {
      if (remaining[slot] < needed ||
          !legacy_matches(input.queue.legacy_orders[slot], *node)) continue;
      match = static_cast<int>(slot);
      break;
    }
    if (match < 0)
      return exact_fallback(
          "current hard acquisition cannot be stable-bound to an adequate legacy non-SELL");
    remaining[static_cast<std::size_t>(match)] -= needed;
    if (first_node_for_slot[static_cast<std::size_t>(match)] < 0)
      first_node_for_slot[static_cast<std::size_t>(match)] = node->id;
    output.due_legacy_bindings.push_back({node->id, match, needed});
  }

  const Result candidate = compose(input.queue);
  output.audit = candidate.audit;
  if (candidate.audit.used_legacy_fallback)
    return exact_fallback(candidate.audit.reason);
  if (candidate.orders.size() != candidate.origins.size())
    return exact_fallback("protected queue returned incomplete slot provenance");
  for (std::size_t slot = 0; slot < input.queue.legacy_orders.size(); ++slot) {
    if (!is_protected(input.queue.legacy_orders[slot])) continue;
    if (slot >= candidate.orders.size() ||
        candidate.orders[slot].operation != input.queue.legacy_orders[slot].operation ||
        candidate.orders[slot].product != input.queue.legacy_orders[slot].product ||
        candidate.orders[slot].animal != input.queue.legacy_orders[slot].animal ||
        candidate.orders[slot].quantity != input.queue.legacy_orders[slot].quantity) {
      return exact_fallback("candidate changed a protected legacy non-SELL byte or slot");
    }
  }
  output.slots.reserve(candidate.orders.size());
  for (std::size_t slot = 0; slot < candidate.orders.size(); ++slot) {
    const int node = slot < first_node_for_slot.size()
        ? first_node_for_slot[slot] : -1;
    output.slots.push_back({candidate.orders[slot], candidate.origins[slot], node});
  }
  output.exact_legacy_fallback = false;
  output.reason = candidate.audit.reason;
  return output;
}

}  // namespace protected_queue
