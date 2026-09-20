#include "unified_market_allocator.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace unified_market {
namespace {

using g001::market::Animal;
using g001::market::Operation;
using g001::market::Order;
using g001::market::Product;
using production_obligation::Item;
using production_obligation::NodeKind;
using production_obligation::ObligationNode;

int product_index(Product product) { return static_cast<int>(product); }

bool purchase_kind(NodeKind kind) {
  return kind == NodeKind::BuySeed || kind == NodeKind::BuyProduct ||
         kind == NodeKind::BuyAnimal || kind == NodeKind::Hire ||
         kind == NodeKind::BuyLand;
}

constexpr std::array<int, 5> seed_costs{10, 20, 50, 100, 80};
constexpr std::array<int, 3> animal_costs{300, 400, 500};
constexpr std::array<int, 3> land_costs{1000, 2000, 4000};

int required_units(const Order& order) {
  return order.operation == Operation::Hire || order.operation == Operation::BuyLand
             ? 1
             : std::max(0, order.quantity);
}

int shed_used(const g001::market::PlayerMarketState& player) {
  int used = 0;
  for (const int quantity : player.shed) used += std::max(0, quantity);
  for (const int quantity : player.animals) used += std::max(0, quantity);
  return used;
}

int fibonacci(int index) {
  int left = 1;
  int right = 1;
  for (int count = 0; count < std::max(0, index); ++count) {
    const int next = left + right;
    left = right;
    right = next;
  }
  return left;
}

std::int64_t remaining_cash_cost(
    const Order& order, int remaining,
    const g001::market::PlayerMarketState& state,
    g001::market::Inventory market_inventory) {
  remaining = std::max(0, remaining);
  if (remaining == 0) return 0;
  switch (order.operation) {
    case Operation::BuySeed:
      return std::int64_t(seed_costs.at(static_cast<std::size_t>(order.product))) * remaining;
    case Operation::BuyAnimal:
      return std::int64_t(animal_costs.at(static_cast<std::size_t>(order.animal))) * remaining;
    case Operation::Hire:
      return fibonacci(state.hires_today);
    case Operation::BuyLand: {
      const int index = state.unlocked_quadrants - 1;
      return index >= 0 && index < static_cast<int>(land_costs.size())
                 ? land_costs[index]
                 : 0;
    }
    case Operation::BuyProduct: {
      const int product = product_index(order.product);
      std::int64_t cost = 0;
      for (int unit = 0; unit < remaining; ++unit) {
        cost += g001::market::price(order.product, market_inventory[product] - 1);
        --market_inventory[product];
      }
      return cost;
    }
    case Operation::Pass:
    case Operation::Sell:
      return 0;
  }
  return 0;
}

Order convert(const ObligationNode& node) {
  Order order;
  order.quantity = node.quantity;
  const int item = static_cast<int>(node.item);
  switch (node.kind) {
    case NodeKind::BuySeed:
      if (item < 0 || item >= 5) throw std::invalid_argument("invalid seed obligation");
      order.operation = Operation::BuySeed;
      order.product = static_cast<Product>(item);
      break;
    case NodeKind::BuyProduct:
      if (item != 0 && item != 8) throw std::invalid_argument("invalid product obligation");
      order.operation = Operation::BuyProduct;
      order.product = static_cast<Product>(item);
      break;
    case NodeKind::BuyAnimal:
      if (item < 9 || item > 11) throw std::invalid_argument("invalid animal obligation");
      order.operation = Operation::BuyAnimal;
      order.animal = static_cast<Animal>(item - 9);
      break;
    case NodeKind::Hire:
      if (node.quantity != 1) throw std::invalid_argument("HIRE node must have quantity one");
      order.operation = Operation::Hire;
      break;
    case NodeKind::BuyLand:
      if (node.quantity != 1) throw std::invalid_argument("BUY_LAND node must have quantity one");
      order.operation = Operation::BuyLand;
      break;
    default:
      throw std::invalid_argument("non-purchase obligation converted to market order");
  }
  return order;
}

struct CandidateUnit {
  int intent{};
  int quote{};
  int product{};
};

std::vector<Order> build_sell_prefix(const Input& input,
                                     const std::vector<Order>& purchases,
                                     int maximum_sell_orders) {
  std::array<int, g001::market::product_count> selected{};
  std::array<int, g001::market::product_count> allowed{};
  std::array<int, g001::market::product_count> priority{};
  std::array<int, g001::market::product_count> reservation{};
  reservation.fill(std::numeric_limits<int>::max());
  priority.fill(std::numeric_limits<int>::min());
  for (const SellIntent& intent : input.sell_intents) {
    const int product = product_index(intent.product);
    const int owned = std::max(0, input.own.shed[product]);
    const int usable = std::max(0, std::min(intent.maximum_quantity,
                                            owned - intent.protected_quantity));
    allowed[product] = std::max(allowed[product], usable);
    priority[product] = std::max(priority[product], intent.priority);
    reservation[product] = std::min(reservation[product],
                                    std::max(1, intent.reservation_price));
  }

  auto simulate = [&](const std::array<int, g001::market::product_count>& quantities) {
    std::vector<Order> queue;
    for (int product = 0; product < int(g001::market::product_count); ++product) {
      if (quantities[product] <= 0) continue;
      queue.push_back({Operation::Sell, static_cast<Product>(product), Animal::Goose,
                       quantities[product]});
    }
    queue.insert(queue.end(), purchases.begin(), purchases.end());
    std::array<g001::market::PlayerMarketState, 2> players{input.own, {}};
    std::array<std::vector<Order>, 2> queues{queue, {}};
    return g001::market::simulate_queue(input.market_inventory, players, queues,
                                        input.maximum_slots, input.shed_capacity);
  };

  auto all_purchases_fill = [&](const g001::market::QueueResult& result,
                                int sell_orders) {
    if (result.committed[0].size() < static_cast<std::size_t>(sell_orders + purchases.size()))
      return false;
    for (std::size_t index = 0; index < purchases.size(); ++index) {
      const int required = purchases[index].operation == Operation::Hire ||
                                   purchases[index].operation == Operation::BuyLand
                               ? 1
                               : purchases[index].quantity;
      if (result.committed[0][sell_orders + int(index)] != required) return false;
    }
    return true;
  };

  for (;;) {
    int sell_orders = 0;
    for (const int quantity : selected) sell_orders += quantity > 0;
    const auto current = simulate(selected);
    if (all_purchases_fill(current, sell_orders)) break;
    CandidateUnit best{-1, std::numeric_limits<int>::min(), -1};
    for (int product = 0; product < int(g001::market::product_count); ++product) {
      if (selected[product] >= allowed[product]) continue;
      const int opened = selected[product] > 0;
      if (!opened && sell_orders >= maximum_sell_orders) continue;
      const auto typed = static_cast<Product>(product);
      const int quote = g001::market::price(
          typed, input.market_inventory[product] + selected[product]);
      if (quote < reservation[product]) continue;
      // Prefer an already-open order, then policy priority, then exact cash.
      const int score = opened * 100000000 + priority[product] * 100000 + quote;
      if (score > best.quote || (score == best.quote && product < best.product))
        best = {product, score, product};
    }
    if (best.intent < 0) return {};
    ++selected[best.product];
  }

  std::vector<Order> result;
  for (int product = 0; product < int(g001::market::product_count); ++product) {
    if (selected[product] <= 0) continue;
    result.push_back({Operation::Sell, static_cast<Product>(product), Animal::Goose,
                      selected[product]});
  }
  return result;
}

void populate_failure_audit(
    const Input& input, const std::vector<const ObligationNode*>& due,
    const std::vector<Order>& purchases, const std::vector<Order>& sell_prefix,
    Audit& audit) {
  std::vector<Order> full_queue = sell_prefix;
  full_queue.insert(full_queue.end(), purchases.begin(), purchases.end());
  std::array<g001::market::PlayerMarketState, 2> players{input.own, {}};
  std::array<std::vector<Order>, 2> queues{full_queue, {}};
  const auto full = g001::market::simulate_queue(
      input.market_inventory, players, queues, input.maximum_slots, input.shed_capacity);

  int first = -1;
  int committed = 0;
  for (std::size_t index = 0; index < purchases.size(); ++index) {
    const int queue_slot = static_cast<int>(sell_prefix.size() + index);
    committed = queue_slot < static_cast<int>(full.committed[0].size())
                    ? full.committed[0][queue_slot]
                    : 0;
    if (committed < required_units(purchases[index])) {
      first = static_cast<int>(index);
      break;
    }
  }
  if (first < 0) return;

  const int hard_slot = static_cast<int>(sell_prefix.size()) + first;
  const int replay_count = std::min<int>(hard_slot + 1, input.maximum_slots);
  std::vector<Order> prefix(full_queue.begin(),
                            full_queue.begin() + std::min<int>(replay_count, full_queue.size()));
  queues = {prefix, {}};
  const auto replay = g001::market::simulate_queue(
      input.market_inventory, players, queues, input.maximum_slots, input.shed_capacity);
  const int prefix_committed = hard_slot < static_cast<int>(replay.committed[0].size())
                                   ? replay.committed[0][hard_slot]
                                   : 0;
  const int required = required_units(purchases[first]);
  const int remaining = std::max(0, required - prefix_committed);
  const auto& failure_state = replay.players[0];

  audit.has_first_unfilled_hard_order = true;
  audit.first_unfilled_kind = due[first]->kind;
  audit.first_unfilled_item = due[first]->item;
  audit.first_unfilled_quantity = required;
  audit.first_unfilled_committed = prefix_committed;
  audit.first_unfilled_remaining = remaining;
  const std::int64_t cash_needed = remaining_cash_cost(
      purchases[first], remaining, failure_state, replay.market_inventory);
  audit.cash_shortfall = std::max<std::int64_t>(0, cash_needed - failure_state.money);
  if (purchases[first].operation == Operation::BuyProduct ||
      purchases[first].operation == Operation::BuyAnimal) {
    audit.capacity_shortfall = std::max(
        0, shed_used(failure_state) + remaining - input.shed_capacity);
  }

  std::array<int, g001::market::product_count> sold{};
  for (std::size_t index = 0; index < sell_prefix.size(); ++index) {
    const int committed_sale = index < replay.committed[0].size()
                                   ? replay.committed[0][index]
                                   : 0;
    sold[product_index(sell_prefix[index].product)] += committed_sale;
  }
  std::array<int, g001::market::product_count> protected_blocked{};
  for (const SellIntent& intent : input.sell_intents) {
    const int product = product_index(intent.product);
    const int remaining_policy = std::max(0, intent.maximum_quantity - sold[product]);
    const int quote = g001::market::price(intent.product, replay.market_inventory[product]);
    if (remaining_policy == 0 || quote < std::max(1, intent.reservation_price)) continue;
    const int stock = std::max(0, failure_state.shed[product]);
    const int eligible = std::min(remaining_policy,
                                  std::max(0, stock - intent.protected_quantity));
    audit.sellable_unprotected_by_product[product] = std::max(
        audit.sellable_unprotected_by_product[product], eligible);
    const int without_protection = std::min(remaining_policy, stock);
    protected_blocked[product] = std::max(
        protected_blocked[product], without_protection - eligible);
  }
  for (std::size_t product = 0; product < g001::market::product_count; ++product) {
    audit.sellable_unprotected_inventory +=
        audit.sellable_unprotected_by_product[product];
    audit.protected_blocked_inventory += protected_blocked[product];
  }
  audit.remaining_sell_slots = std::max(
      0, input.maximum_slots - static_cast<int>(sell_prefix.size() + purchases.size()));

  if (hard_slot >= input.maximum_slots) {
    audit.failure_cause = FailureCause::HardOrderSlotBudget;
  } else if (audit.remaining_sell_slots == 0 &&
             audit.sellable_unprotected_inventory > 0 &&
             (audit.cash_shortfall > 0 || audit.capacity_shortfall > 0)) {
    audit.failure_cause = FailureCause::SellSlot;
  } else if (audit.protected_blocked_inventory > 0 &&
             audit.sellable_unprotected_inventory == 0 &&
             (audit.cash_shortfall > 0 || audit.capacity_shortfall > 0)) {
    audit.failure_cause = FailureCause::ProtectedStock;
  } else if (audit.capacity_shortfall > 0) {
    audit.failure_cause = FailureCause::Capacity;
  } else if (audit.cash_shortfall > 0) {
    audit.failure_cause = FailureCause::Cash;
  } else {
    audit.failure_cause = FailureCause::InvalidOrder;
  }

  std::ostringstream reason;
  reason << "first unfilled hard order: kind="
         << production_obligation::node_kind_name(audit.first_unfilled_kind)
         << " item=" << production_obligation::item_name(audit.first_unfilled_item)
         << " required=" << audit.first_unfilled_quantity
         << " committed=" << audit.first_unfilled_committed
         << " cash_shortfall=" << audit.cash_shortfall
         << " capacity_shortfall=" << audit.capacity_shortfall
         << " sellable_unprotected=" << audit.sellable_unprotected_inventory
         << " remaining_sell_slots=" << audit.remaining_sell_slots
         << " cause=" << failure_cause_name(audit.failure_cause);
  audit.reason = reason.str();
}

}  // namespace

const char* failure_cause_name(FailureCause cause) {
  switch (cause) {
    case FailureCause::None: return "NONE";
    case FailureCause::Cash: return "CASH";
    case FailureCause::Capacity: return "CAPACITY";
    case FailureCause::ProtectedStock: return "PROTECTED_STOCK";
    case FailureCause::SellSlot: return "SELL_SLOT";
    case FailureCause::HardOrderSlotBudget: return "HARD_ORDER_SLOT_BUDGET";
    case FailureCause::InvalidOrder: return "INVALID_ORDER";
  }
  return "UNKNOWN";
}

Result allocate(const Input& input) {
  if (input.maximum_slots < 0 || input.shed_capacity < 0)
    throw std::invalid_argument("negative allocator limit");
  Result result;
  std::vector<const ObligationNode*> due;
  for (const ObligationNode& node : input.production_dag) {
    if (purchase_kind(node.kind) && node.execution_step == input.step) due.push_back(&node);
  }
  std::stable_sort(due.begin(), due.end(), [](const auto* left, const auto* right) {
    return std::tie(left->order_slot, left->id) < std::tie(right->order_slot, right->id);
  });
  std::vector<Order> purchases;
  purchases.reserve(due.size());
  for (const ObligationNode* node : due) purchases.push_back(convert(*node));
  result.audit.due_purchase_orders = static_cast<int>(purchases.size());
  if (purchases.size() > static_cast<std::size_t>(input.maximum_slots)) {
    result.audit.unmet_purchase_orders = static_cast<int>(purchases.size()) - input.maximum_slots;
    populate_failure_audit(input, due, purchases, {}, result.audit);
    return result;
  }

  const int sell_budget = input.maximum_slots - static_cast<int>(purchases.size());
  auto sell_prefix = build_sell_prefix(input, purchases, sell_budget);
  result.orders = sell_prefix;
  result.orders.insert(result.orders.end(), purchases.begin(), purchases.end());
  result.audit.funding_sell_orders = static_cast<int>(sell_prefix.size());
  for (const Order& order : sell_prefix) result.audit.funding_sell_units += order.quantity;
  result.audit.slots_used = static_cast<int>(result.orders.size());

  std::array<g001::market::PlayerMarketState, 2> players{input.own, {}};
  std::array<std::vector<Order>, 2> queues{result.orders, {}};
  const auto exact = g001::market::simulate_queue(
      input.market_inventory, players, queues, input.maximum_slots, input.shed_capacity);
  result.predicted_own = exact.players[0];
  result.predicted_market = exact.market_inventory;
  bool filled = exact.committed[0].size() == result.orders.size();
  for (std::size_t index = sell_prefix.size(); filled && index < result.orders.size(); ++index) {
    const Order& order = result.orders[index];
    const int required = order.operation == Operation::Hire ||
                                 order.operation == Operation::BuyLand
                             ? 1
                             : order.quantity;
    filled = exact.committed[0][index] == required;
  }
  result.audit.exact_queue_verified = filled;
  result.feasible = filled;
  if (!filled) {
    result.audit.unmet_purchase_orders = 1;
    populate_failure_audit(input, due, purchases, sell_prefix, result.audit);
    result.orders.clear();
  } else {
    result.audit.reason = "all due production purchases fill in exact one-player queue replay";
  }
  return result;
}

}  // namespace unified_market
