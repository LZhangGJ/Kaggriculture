// Offline prototype: fixed NN planting/procurement, optimise only one product's SELL quantity.
// g++ -std=c++20 -O2 -Ipolicy/r1 experiments/sale_postprocess_one_item.cpp -o /tmp/sale_postprocess_one_item
#include "conditional_market.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct State { int market, shed, cash, rival_cash; };
struct Period {
  int landed = 0, reserved = 0, rival_sell = 0, town_demand = 0;
  bool sale_before_buys = true;
  std::vector<int> fixed_costs; // ordered, prequoted one-unit purchases/hires; never modified
};
struct Receipt {
  State next;
  int own_sell = 0, own_income = 0, rival_income = 0;
  int own_admitted = 0, rival_admitted = 0, paid = 0, filled_costs = 0;
  bool complete = false;
};

// Same-slot SELL/SELL: the official engine quotes both sides BEFORE either commits each unit.
// ConditionalMarket::execute is exact for one side; two concurrent sides need shared prequotes.
std::optional<Receipt> settle(int item, State start, const Period& p, int sell) {
  if (item < 0 || item >= 9 || start.shed < 0 || start.shed > 100 || start.cash < 0 ||
      p.reserved < 0 || p.rival_sell < 0 ||
      p.town_demand < 0 || start.shed + p.landed < 0 || start.shed + p.landed > 100 || sell < 0 ||
      sell > start.shed + p.landed - p.reserved || (!p.sale_before_buys && sell)) return {};
  Receipt r;
  r.next = start;
  r.next.shed += p.landed; // already landed after fixed NN unit actions, not predicted crop flow
  r.own_sell = sell;
  for (int u = 0; u < std::max(sell, p.rival_sell); ++u) {
    const int quote = dp7::price(item, r.next.market);
    if (u < sell) {
      --r.next.shed;
      r.next.cash += quote;
      r.own_income += quote;
      if (quote > 1) { ++r.next.market; ++r.own_admitted; }
    }
    if (u < p.rival_sell) {
      r.next.rival_cash += quote;
      r.rival_income += quote;
      if (quote > 1) { ++r.next.market; ++r.rival_admitted; }
    }
  }
  for (int cost : p.fixed_costs) {
    if (cost < 0) return {};
    if (r.next.cash < cost) break; // official partial fill; reject this plan below
    r.next.cash -= cost;
    r.paid += cost;
    ++r.filled_costs;
  }
  r.complete = r.filled_costs == int(p.fixed_costs.size());
  r.next.market -= p.town_demand; // town consumption follows all market orders
  return r;
}

struct Plan { bool feasible = false; int margin = 0; std::vector<int> sales; };
Plan choose(int item, State start, const std::vector<Period>& periods) {
  Plan best;
  std::vector<int> path;
  // ponytail: exhaustive short-horizon oracle; memoise (period,I,shed,cash) before extending horizon.
  std::function<void(size_t, State)> visit = [&](size_t k, State s) {
    if (k == periods.size()) {
      int margin = s.cash - s.rival_cash;
      if (!best.feasible || margin > best.margin) best = {true, margin, path};
      return;
    }
    const auto& p = periods[k];
    int limit = p.sale_before_buys ? s.shed + p.landed - p.reserved : 0;
    for (int q = 0; q <= limit; ++q) {
      auto receipt = settle(item, s, p, q);
      if (!receipt || !receipt->complete) continue;
      path.push_back(q);
      visit(k + 1, receipt->next);
      path.pop_back();
    }
  };
  visit(0, start);
  return best;
}
} // namespace

int main(int argc, char** argv) {
  using competitive::ConditionalMarket;
  if (argc == 2 && std::strcmp(argv[1], "--project") == 0) {
    std::string line;
    while (std::getline(std::cin, line)) {
      std::istringstream input(line);
      int item, market, shed, cash, sell, rival, demand, count;
      if (!(input >> item >> market >> shed >> cash >> sell >> rival >> demand >> count) || count < 0) {
        std::puts("REJECT"); continue;
      }
      Period p; p.rival_sell = rival; p.town_demand = demand;
      for (int i = 0, cost; i < count; ++i) {
        if (!(input >> cost)) return 2;
        p.fixed_costs.push_back(cost);
      }
      input >> p.landed; // optional observed unit-phase shed delta; old fixtures default to zero
      auto receipt = settle(item, {market, shed, cash, 0}, p, sell);
      if (!receipt) { std::puts("REJECT"); continue; }
      const auto& r = *receipt;
      std::printf("%s %d %d %d %d %d %d %d %d\n", r.complete ? "OK" : "INCOMPLETE",
                  r.next.market, r.next.shed,
                  r.next.cash, r.own_sell, r.own_income, r.own_admitted,
                  r.rival_admitted, r.filled_costs);
    }
    return 0;
  }
  // Keep one wheat for later: sell one now to fund a fixed $20 order, then sell after demand.
  State start{10000, 2, 0, 0};
  Period first{0, 0, 0, 100, true, {20}}, second;
  auto plan = choose(dp7::W, start, {first, second});
  assert(plan.feasible && plan.sales == std::vector<int>({1, 1}));
  auto one = settle(dp7::W, start, first, plan.sales[0]);
  assert(one && one->complete && one->filled_costs == 1);
  assert(one->next.shed == start.shed - one->own_sell);
  assert(one->next.cash == start.cash + one->own_income - one->paid);
  assert(one->next.market == start.market + one->own_admitted + one->rival_admitted - first.town_demand);
  double single_inventory = start.market;
  assert(ConditionalMarket::execute(dp7::W, single_inventory, 1.) == one->own_income);
  assert(int(single_inventory) == start.market + one->own_admitted);

  // At the $1 floor the sale pays, removes shed goods, and admits nothing to the market.
  int floor = int(ConditionalMarket::saturation(dp7::WO));
  auto low = settle(dp7::WO, {floor, 2, 0, 0}, Period{}, 2);
  assert(low && low->next.shed == 0 && low->next.cash == 2 && low->next.market == floor);
  // Both players quote the same pre-commit price; a shared slot can cross saturation by two.
  Period rival; rival.rival_sell = 1;
  auto pair = settle(dp7::WO, {floor - 1, 1, 0, 0}, rival, 1);
  assert(pair && pair->own_admitted == 1 && pair->rival_admitted == 1 && pair->next.market == floor + 1);
  Period unit_drop; unit_drop.landed = 2;
  auto after_drop = settle(dp7::W, {10000, 0, 0, 0}, unit_drop, 2);
  assert(after_drop && after_drop->next.shed == 0 && after_drop->own_income > 0);
  Period unit_pickup; unit_pickup.landed = -1;
  assert(!settle(dp7::W, {10000, 1, 0, 0}, unit_pickup, 1));
  assert(settle(dp7::W, {10000, 1, 0, 0}, unit_pickup, 0));
  first.sale_before_buys = false;
  assert(!choose(dp7::W, start, {first, second}).feasible); // no sale slot before the fixed buy
  std::printf("PASS sales=%d,%d cash_after_first=%d floor_cash=%d pair_inventory=%d\n",
              plan.sales[0], plan.sales[1], one->next.cash, low->next.cash, pair->next.market);
}
