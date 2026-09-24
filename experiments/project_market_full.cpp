// Offline market-only projector after fixed unit actions; no NN/RL integration.
// g++ -std=c++20 -O2 -Ipolicy/r1 experiments/project_market_full.cpp -o /tmp/project_market_full
#include "conditional_market.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
enum Op { SELL, BUY_PRODUCT, BUY_SEED, BUY_ANIMAL, HIRE, BUY_LAND, OTHER };
struct Order { int op, item, quantity; };
struct Player {
  int cash, hires, land;
  std::array<int, 12> shed;
  std::vector<std::vector<std::pair<int, int>>> bags; // worker order, then insertion order
  int overflow = 0;
  std::vector<Order> orders;
  std::vector<int> fills;
};
struct Market {
  std::array<int, 9> inventory;
  std::array<int, 9> town;
  std::array<Player, 2> players;
  bool end_day = false;
};
constexpr int seed_cost[5] = {10, 20, 50, 100, 80};
constexpr int animal_cost[3] = {300, 400, 500};
constexpr int land_cost[3] = {1000, 2000, 4000};

int fib(int n) {
  int a = 1, b = 1;
  while (n--) { const int c = a + b; a = b; b = c; }
  return a;
}
int shed_sum(const Player& p) {
  return std::accumulate(p.shed.begin(), p.shed.end(), 0);
}
int quote(const Market& m, const Order& o) {
  if (o.op == SELL && o.item >= 0 && o.item < 9)
    return dp7::price(o.item, m.inventory[o.item]);
  if (o.op == BUY_PRODUCT && (o.item == 0 || o.item == 8))
    return dp7::price(o.item, m.inventory[o.item] - 1);
  if (o.op == BUY_SEED && o.item >= 0 && o.item < 5) return seed_cost[o.item];
  if (o.op == BUY_ANIMAL && o.item >= 9 && o.item < 12) return animal_cost[o.item - 9];
  return -1;
}
bool commit(Market& m, int who, const Order& o, int price) {
  auto& p = m.players[who];
  if (o.op == SELL) {
    if (p.shed[o.item] <= 0) return false;
    --p.shed[o.item]; p.cash += price;
    if (price > 1) ++m.inventory[o.item];
    return true;
  }
  if (o.op == BUY_PRODUCT) {
    if (p.cash < price || shed_sum(p) >= 100) return false;
    p.cash -= price; ++p.shed[o.item]; --m.inventory[o.item];
    return true;
  }
  if (o.op == BUY_SEED) {
    if (p.cash < price) return false;
    p.cash -= price; return true;
  }
  if (o.op == BUY_ANIMAL) {
    if (p.cash < price || shed_sum(p) >= 100) return false;
    p.cash -= price; ++p.shed[o.item]; return true;
  }
  return false;
}
void project(Market& m) {
  const size_t slots = std::min<size_t>(10, std::max(m.players[0].orders.size(),
                                                   m.players[1].orders.size()));
  for (auto& p : m.players) p.fills.assign(p.orders.size(), 0);
  for (size_t slot = 0; slot < slots; ++slot) {
    Order o[2] = {{OTHER, -1, 0}, {OTHER, -1, 0}};
    int remaining[2] = {};
    bool active[2] = {};
    for (int p = 0; p < 2; ++p) if (slot < m.players[p].orders.size()) {
      o[p] = m.players[p].orders[slot];
      remaining[p] = o[p].quantity;
      active[p] = remaining[p] > 0;
    }
    // Both hire/land orders settle before the per-unit market loop.
    for (int p = 0; p < 2; ++p) if (active[p] && (o[p].op == HIRE || o[p].op == BUY_LAND)) {
      auto& player = m.players[p];
      if (o[p].op == HIRE) {
        const int cost = fib(player.hires);
        if (player.cash >= cost) {
          player.cash -= cost; ++player.hires; player.fills[slot] = 1;
        }
      } else if (player.land < 3 && player.cash >= land_cost[player.land]) {
        player.cash -= land_cost[player.land]; ++player.land; player.fills[slot] = 1;
      }
      active[p] = false;
    }
    while (active[0] || active[1]) {
      int prices[2] = {-1, -1};
      for (int p = 0; p < 2; ++p) if (active[p]) {
        prices[p] = quote(m, o[p]); // prequote both before either commit
        if (prices[p] < 0) active[p] = false;
      }
      if (prices[0] < 0 && prices[1] < 0) break;
      bool any = false;
      for (int p = 0; p < 2; ++p) if (prices[p] >= 0) {
        if (commit(m, p, o[p], prices[p])) {
          ++m.players[p].fills[slot]; any = true;
          if (--remaining[p] == 0) active[p] = false;
        } else active[p] = false;
      }
      if (!any) break;
    }
  }
  for (int i = 0; i < 9; ++i) m.inventory[i] -= m.town[i];
  if (m.end_day) for (auto& p : m.players) {
    for (const auto& bag : p.bags) for (auto [item, count] : bag) {
      const int moved = std::min(count, 100 - shed_sum(p));
      p.shed[item] += moved;
      p.overflow += count - moved;
    }
  }
}
bool read(Market& m) {
  for (int& x : m.inventory) if (!(std::cin >> x)) return false;
  for (int& x : m.town) if (!(std::cin >> x)) return false;
  for (auto& p : m.players) {
    p.overflow = 0;
    if (!(std::cin >> p.cash >> p.hires >> p.land)) return false;
    for (int& x : p.shed) if (!(std::cin >> x)) return false;
    int count;
    if (!(std::cin >> count) || count < 0 || count > 64) return false;
    p.orders.resize(count);
    for (auto& o : p.orders)
      if (!(std::cin >> o.op >> o.item >> o.quantity)) return false;
  }
  int end_day;
  if (!(std::cin >> end_day)) return false;
  m.end_day = end_day != 0;
  for (auto& p : m.players) {
    int workers;
    if (!(std::cin >> workers) || workers < 0 || workers > 64) return false;
    p.bags.resize(workers);
    for (auto& bag : p.bags) {
      int count;
      if (!(std::cin >> count) || count < 0 || count > 12) return false;
      bag.resize(count);
      for (auto& pair : bag)
        if (!(std::cin >> pair.first >> pair.second)) return false;
    }
  }
  return true;
}
void write(const Market& m) {
  for (int x : m.inventory) std::cout << x << ' ';
  for (const auto& p : m.players) {
    std::cout << p.cash << ' ';
    for (int x : p.shed) std::cout << x << ' ';
    std::cout << p.fills.size() << ' ';
    for (int x : p.fills) std::cout << x << ' ';
    std::cout << p.overflow << ' ';
  }
  std::cout << '\n';
}
} // namespace

int main() {
  Market m;
  while (read(m)) {
    project(m);
    write(m);
  }
}
