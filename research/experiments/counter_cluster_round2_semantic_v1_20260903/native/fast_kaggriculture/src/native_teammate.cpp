// Licensed under the Apache License, Version 2.0.
#include "native_teammate.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace fastkag {
namespace {

constexpr std::array<int, 4> PREMIUM{3, 4, 6, 7};
constexpr std::array<int, 4> COUNTER_ITEMS{4, 6, 3, 7};
constexpr std::array<int, 3> K320_PREMIUM{3, 6, 7};
constexpr std::array<int, 9> LIQUIDATION{1, 5, 8, 4, 6, 3, 2, 0, 7};
constexpr std::array<int, 9> ROOM_PRIORITY{7, 6, 5, 4, 3, 2, 1, 8, 0};
constexpr std::array<Position, 4> SHED_ACCESS{{{4, 4}, {5, 4}, {5, 5}, {4, 5}}};

int premium_slot(int item) {
  for (int i = 0; i < 4; ++i) if (PREMIUM[i] == item) return i;
  return -1;
}

int quantity(const Action& a) { return std::max(0, int(a.quantity)); }
bool sell(const Action& a) {
  return a.op == Op::SELL && int(a.item) >= 0 && int(a.item) < N_PRODUCTS;
}
bool at(Position a, Position b) { return a.x == b.x && a.y == b.y; }
int distance(Position a, Position b) {
  return std::abs(int(a.x) - int(b.x)) + std::abs(int(a.y) - int(b.y));
}
bool shed_adjacent(Position p) {
  return (p.x == 4 || p.x == 5) && (p.y == 4 || p.y == 5);
}

int shed_sum(const PrivateState& p) {
  return std::accumulate(p.shed.begin(), p.shed.end(), 0);
}

PlayerAction aligned(PlayerAction a, const Simulator& env, int player) {
  const size_t expected = env.farms()[player].hands.size() + 1;
  a.units.resize(expected);
  return a;
}

const Tile* tile_at(const Simulator& env, int player, Position p) {
  const int n = env.config().board_size;
  if (p.x < 0 || p.y < 0 || p.x >= n || p.y >= n) return nullptr;
  return &env.farms()[player].tiles[int(p.y) * n + int(p.x)];
}

std::vector<Position> positions(const Simulator& env, int player) {
  std::vector<Position> out{env.farms()[player].farmer};
  out.insert(out.end(), env.farms()[player].hands.begin(),
             env.farms()[player].hands.end());
  return out;
}

std::array<int, N_ITEMS> projected_shed(const Simulator& env, int player,
                                         const PlayerAction& action) {
  auto projected = env.privates()[player].shed;
  const auto pos = positions(env, player);
  const auto& pr = env.privates()[player];
  for (size_t u = 0; u < action.units.size() && u < pos.size() &&
                     u < pr.inventories.size(); ++u) {
    if (!shed_adjacent(pos[u])) continue;
    const auto& a = action.units[u];
    const auto& inv = pr.inventories[u];
    if (a.op == Op::DROP) {
      // CPython dict insertion order is preserved by inventory_order.
      for (int item : pr.inventory_order[u]) {
        const int room = std::max(0, 100 - std::accumulate(
            projected.begin(), projected.end(), 0));
        projected[item] += std::min(std::max(0, inv[item]), room);
      }
    } else if (a.op == Op::PLACE) {
      const int item = int(a.item);
      if (item < 0 || item >= N_ITEMS) continue;
      const Tile* t = tile_at(env, player, pos[u]);
      if (item >= 9 && t &&
          ((item == 9 && t->kind == TileKind::COOP) ||
           (item >= 10 && t->kind == TileKind::PASTURE)) &&
          t->animal == Item::NONE) continue;
      const int room = std::max(0, 100 - std::accumulate(
          projected.begin(), projected.end(), 0));
      projected[item] += std::min({quantity(a), std::max(0, inv[item]), room});
    }
  }
  return projected;
}

struct PublicSignature {
  int hands = 0, lands = 0;
  std::array<int, 11> counts{};
};

PublicSignature signature(const Farm& farm) {
  PublicSignature s;
  s.hands = int(farm.hands.size());
  s.lands = std::popcount(unsigned(farm.unlocked_mask));
  for (const Tile& t : farm.tiles) {
    if (t.kind == TileKind::PLANT) s.counts[int(t.crop)]++;
    else if (t.kind == TileKind::ANIMAL) s.counts[5 + int(t.animal) - 9]++;
    else if (t.kind == TileKind::PASTURE) s.counts[8]++;
    else if (t.kind == TileKind::COOP) s.counts[9]++;
    else if (t.kind == TileKind::WEED) s.counts[10]++;
  }
  return s;
}

int clone_distance(const Simulator& env) {
  const auto a = signature(env.farms()[0]);
  const auto b = signature(env.farms()[1]);
  int d = std::abs(a.hands - b.hands) + 3 * std::abs(a.lands - b.lands);
  for (size_t i = 0; i < a.counts.size(); ++i) d += std::abs(a.counts[i] - b.counts[i]);
  return d;
}

int animal_count(const Farm& farm, Item item) {
  return std::count_if(farm.tiles.begin(), farm.tiles.end(),
                       [item](const Tile& t) {
                         return t.kind == TileKind::ANIMAL && t.animal == item;
                       });
}

int shop_demand(const Simulator& env, int item, int step) {
  int demand = item != 8 && step % 24 == 0 ? 1 : 0;
  if (step % 4 != 0) return demand;
  // BAKERY, BRUNCH, FARMERS, ICE_CREAM, PET, PIZZA, SMOOTHIE, YARN.
  static constexpr uint16_t masks[8] = {
      (1u << 5) | (1u << 0),
      (1u << 5) | (1u << 0) | (1u << 3),
      (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3),
      (1u << 3) | (1u << 6) | (1u << 0),
      (1u << 1),
      (1u << 6) | (1u << 2) | (1u << 0),
      (1u << 3) | (1u << 6),
      (1u << 7)};
  for (int sh : env.shops()) {
    if (sh >= 0 && sh < 8 && (masks[sh] & (1u << item)))
      demand += std::popcount(unsigned(masks[sh])) == 1 ? 2 : 1;
  }
  return demand;
}

double curve(const char* name, double value) {
  value = std::max(0.0, value);
  if (name[0] == 's' && name[1] == 'q' && name[2] == '\0') return value * value;
  if (name[0] == 's') return std::sqrt(value);
  if (name[0] == 'l' && name[3] == '\0') return std::log1p(value);
  return value;
}

int k320_market_price(int item, int inventory) {
  struct P { int base, eq, scale; const char *below, *above; double bt, at; };
  static constexpr P p[9] = {
      {25,10000,400,"sqrt","log",.8,.2}, {35,10000,450,"log","sqrt",.2,.7},
      {60,10000,200,"linear","sqrt",.4,.6}, {120,10000,100,"sqrt","linear",.7,1.6},
      {250,10000,300,"log","sq",.2,3.6}, {50,10000,332,"linear","log",.4,.2},
      {160,10000,122,"sqrt","linear",.6,1.6}, {200,10000,105,"log","sq",.2,3.2},
      {100,10000,200,"linear","linear",.4,.4}};
  const auto& x = p[item];
  double value;
  if (inventory < x.eq)
    value = x.base + x.bt * x.base / curve(x.below, x.scale) *
                         curve(x.below, x.eq - inventory);
  else
    value = x.base - x.at * x.base / curve(x.above, x.scale) *
                         curve(x.above, inventory - x.eq);
  return std::max(1, int(std::nearbyint(value)));
}

double order_score(const Simulator& env, const Action& a) {
  if (!sell(a)) return -std::numeric_limits<double>::infinity();
  const int item = int(a.item), q = quantity(a), inv = env.market().inventory[item];
  const double current = env.market().prices[item];
  const double later = k320_market_price(item, inv + q);
  double score = q * std::max(0.0, current - later);
  if (score <= 0) return score;
  static constexpr uint16_t masks[8] = {33,41,15,73,2,69,72,128};
  double demand = item == 8 ? 0.0 : 1.0;
  for (int sh : env.shops()) if (masks[sh] & (1u << item))
    demand += 6.0 * (std::popcount(unsigned(masks[sh])) == 1 ? 2 : 1);
  demand = std::max(0.25, demand);
  const double excess = std::max(0, inv + q - 10000);
  const double urgency = std::min(1.0, excess / demand / 10.0);
  return score * (1.0 + .25 * urgency);
}

void rank_sell_slots(const Simulator& env, PlayerAction& a) {
  struct Row { double score; int neg_index; Action action; };
  std::vector<Row> rows;
  for (size_t i = 0; i < a.market.size(); ++i)
    if (sell(a.market[i])) rows.push_back({order_score(env, a.market[i]), -int(i), a.market[i]});
  if (rows.size() < 2) return;
  std::sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) {
    return std::tie(x.score, x.neg_index) > std::tie(y.score, y.neg_index);
  });
  size_t r = 0;
  for (auto& order : a.market) if (sell(order)) order = rows[r++].action;
}

int planned_sell(const PlayerAction& a, int item) {
  int n = 0;
  for (const auto& x : a.market) if (sell(x) && int(x.item) == item) n += quantity(x);
  return n;
}

int pickup_reserve(const PlayerAction& a, int item) {
  int n = 0;
  for (const auto& x : a.units)
    if (x.op == Op::PICKUP && int(x.item) == item) n += quantity(x);
  return n;
}

void merge_sale(PlayerAction& a, int item, int q) {
  if (q <= 0) return;
  for (auto& x : a.market) if (sell(x) && int(x.item) == item) {
    x.quantity = quantity(x) + q;
    return;
  }
  if (a.market.size() < 10)
    a.market.push_back(Action{Op::SELL, Item(item), q});
}

Action move_toward(Position p, Position target) {
  if (p.x < target.x) return Action{Op::EAST};
  if (p.x > target.x) return Action{Op::WEST};
  if (p.y < target.y) return Action{Op::SOUTH};
  if (p.y > target.y) return Action{Op::NORTH};
  return Action{};
}

int moon_route(const Simulator& env) {
  if (!env.shops().empty() && env.shops()[0] == 7) return 3;
  if (env.shops().size() >= 2 &&
      (env.shops()[0] == 7 || env.shops()[1] == 7)) return 4;
  if (env.shops().size() >= 3 &&
      (env.shops()[0] == 7 || env.shops()[1] == 7 || env.shops()[2] == 7)) return 2;
  for (size_t i = 0; i < std::min<size_t>(3, env.shops().size()); ++i)
    if (env.shops()[i] == 3 || env.shops()[i] == 5 || env.shops()[i] == 6) return 0;
  return 1;
}

int tape_sell(const std::vector<PlayerAction>& tape, int step, int item) {
  if (step < 0 || step >= int(tape.size())) return 0;
  return planned_sell(tape[step], item);
}

int macro_unit_failures(const Simulator& env, int player,
                        const PlayerAction& actions) {
  auto farm = env.farms()[player];
  auto private_state = env.privates()[player];
  std::array<int, N_CROPS> demand{};
  for (const auto& action : actions.units)
    if (action.op == Op::PLANT && int(action.item) >= 0 && int(action.item) < N_CROPS)
      demand[int(action.item)]++;
  std::array<bool, N_CROPS> blocked{};
  for (int item = 0; item < N_CROPS; ++item)
    blocked[item] = demand[item] > private_state.seeds[item];
  int failures = 0;
  for (size_t actor = 0; actor < actions.units.size(); ++actor) {
    const auto& action = actions.units[actor];
    const bool macro = action.op == Op::PLANT || action.op == Op::BUILD_COOP ||
        action.op == Op::BUILD_PASTURE ||
        (action.op == Op::PLACE && int(action.item) >= int(Item::GOOSE) &&
         int(action.item) <= int(Item::SHEEP));
    if (!macro) continue;
    if (actor > farm.hands.size()) { failures++; continue; }
    const Position position = actor == 0 ? farm.farmer : farm.hands[actor - 1];
    const int tile_index = position.y * env.config().board_size + position.x;
    const Tile& tile = farm.tiles[tile_index];
    if (action.op == Op::PLANT) {
      const int item = int(action.item);
      if (item < 0 || item >= N_CROPS || blocked[item] || tile.kind != TileKind::EMPTY) {
        failures++;
      } else {
        private_state.seeds[item]--;
        farm.tiles[tile_index].kind = TileKind::PLANT;
      }
    } else if (action.op == Op::BUILD_COOP || action.op == Op::BUILD_PASTURE) {
      if (tile.kind != TileKind::EMPTY) failures++;
      else farm.tiles[tile_index].kind = action.op == Op::BUILD_COOP
          ? TileKind::COOP : TileKind::PASTURE;
    } else {
      const int item = int(action.item);
      const TileKind required = item == int(Item::GOOSE) ? TileKind::COOP : TileKind::PASTURE;
      if (tile.kind != required || tile.animal != Item::NONE ||
          actor >= private_state.inventories.size() ||
          private_state.inventories[actor][item] <= 0) {
        failures++;
      } else {
        private_state.inventories[actor][item]--;
        farm.tiles[tile_index].kind = TileKind::ANIMAL;
        farm.tiles[tile_index].animal = action.item;
      }
    }
  }
  return failures;
}

int macro_market_failures(const Simulator& env, int player,
                          const PlayerAction& actions) {
  const auto& fills = env.last_market_fills()[player];
  int failures = 0;
  for (size_t order = 0; order < actions.market.size(); ++order) {
    const auto& action = actions.market[order];
    if (action.op != Op::BUY_SEED && action.op != Op::BUY_ANIMAL &&
        action.op != Op::HIRE && action.op != Op::BUY_LAND) continue;
    const int requested = action.op == Op::HIRE || action.op == Op::BUY_LAND
        ? int(action.quantity > 0) : quantity(action);
    const int filled = order < fills.size() ? fills[order] : 0;
    failures += std::max(0, requested - filled);
  }
  return failures;
}

struct FeatureCounts {
  std::array<int, 5> crops{}, crop_yield{};
  std::array<int, 3> animals{}, animal_yield{};
  std::array<int, 3> structures{};
  int weeds = 0, empty = 0, crop_stress = 0, animal_stress = 0;
};

FeatureCounts feature_counts(const Farm& farm) {
  FeatureCounts c;
  for (const Tile& t : farm.tiles) {
    if (t.kind == TileKind::EMPTY) c.empty++;
    else if (t.kind == TileKind::WEED) c.weeds++;
    else if (t.kind == TileKind::COOP) c.structures[1]++;
    else if (t.kind == TileKind::PASTURE) c.structures[2]++;
    else if (t.kind == TileKind::PLANT) {
      const int item = int(t.crop); c.crops[item]++; c.crop_yield[item] += t.yield_units;
      c.crop_stress += !t.watered_today; c.crop_stress += t.consecutive_unwatered;
    } else if (t.kind == TileKind::ANIMAL) {
      const int item = int(t.animal) - 9; c.animals[item]++; c.animal_yield[item] += t.yield_units;
      c.structures[t.animal == Item::GOOSE ? 1 : 2]++;
      c.animal_stress += !t.fed_today; c.animal_stress += t.consecutive_unfed;
    }
  }
  return c;
}

struct FeatureHistory {
  int steps = 0, tile_samples = 0;
  std::array<double, 2> min_money{
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  std::array<int, 2> low100{}, low300{}, max_weeds{}, weed_area{}, max_crops{}, max_animals{};
  void update(const Simulator& env) {
    const int step = env.step_count();
    const bool sample = steps == 0 || step % 24 == 0;
    for (int p = 0; p < 2; ++p) {
      const auto& farm = env.farms()[p];
      min_money[p] = std::min(min_money[p], farm.money);
      low100[p] += farm.money < 100; low300[p] += farm.money < 300;
      if (sample) {
        const auto c = feature_counts(farm);
        const int crops = std::accumulate(c.crops.begin(), c.crops.end(), 0);
        const int animals = std::accumulate(c.animals.begin(), c.animals.end(), 0);
        max_weeds[p] = std::max(max_weeds[p], c.weeds); weed_area[p] += c.weeds;
        max_crops[p] = std::max(max_crops[p], crops);
        max_animals[p] = std::max(max_animals[p], animals);
      }
    }
    tile_samples += sample; steps++;
  }
};

int fib_cost(int index) {
  int left = 1, right = 1;
  while (index-- > 0) { const int next = left + right; left = right; right = next; }
  return left;
}

std::array<float, 147> build_features(const Simulator& env, int player,
                                      const FeatureHistory& history,
                                      const std::vector<PlayerAction>& tape) {
  std::array<float, 147> result{}; size_t at_index = 0;
  auto push = [&](double value) { result.at(at_index++) = float(value); };
  auto farm_vector = [&](int p) {
    const auto& farm = env.farms()[p]; const auto c = feature_counts(farm);
    push(farm.money); push(farm.hands.size()); push(std::popcount(unsigned(farm.unlocked_mask)));
    push(farm.hires_today); push(c.weeds); push(c.empty);
    for (int x : c.crops) push(x); for (int x : c.animals) push(x);
    for (int x : c.structures) push(x); for (int x : c.crop_yield) push(x);
    for (int x : c.animal_yield) push(x); push(c.crop_stress); push(c.animal_stress);
  };
  farm_vector(player); farm_vector(1 - player);
  const auto& pr = env.privates()[player];
  for (int x : pr.shed) push(x); for (int x : pr.seeds) push(x);
  std::array<int, N_ITEMS> carried{};
  for (const auto& inv : pr.inventories)
    for (int item = 0; item < N_ITEMS; ++item) carried[item] += inv[item];
  for (int x : carried) push(x);
  const int shed_total = shed_sum(pr);
  push(shed_total); push(std::accumulate(carried.begin(), carried.end(), 0)); push(100 - shed_total);
  for (int item = 0; item < N_PRODUCTS; ++item) {
    push(env.market().inventory[item]); push(env.market().prices[item]);
  }
  std::array<bool, 8> shops{}; for (int sh : env.shops()) shops[sh] = true;
  for (bool value : shops) push(value);
  const int step = env.step_count(); push(step); push(step / 24); push(step % 24);
  for (int p : {player, 1 - player}) {
    const auto current = feature_counts(env.farms()[p]);
    const int crops = std::accumulate(current.crops.begin(), current.crops.end(), 0);
    const int animals = std::accumulate(current.animals.begin(), current.animals.end(), 0);
    push(std::isfinite(history.min_money[p]) ? history.min_money[p] : 0);
    push(double(history.low100[p]) / std::max(1, history.steps));
    push(double(history.low300[p]) / std::max(1, history.steps));
    push(history.max_weeds[p]);
    push(double(history.weed_area[p]) / std::max(1, history.tile_samples));
    push(std::max(0, history.max_crops[p] - crops));
    push(std::max(0, history.max_animals[p] - animals));
  }
  static constexpr int seed_cost[5] = {10,20,50,100,80};
  static constexpr int animal_cost[3] = {300,400,500};
  static constexpr int land_cost[3] = {1000,2000,4000};
  for (int horizon : {24, 48, 72}) {
    double cumulative = 0, requirement = 0, expense = 0;
    int hires = env.farms()[player].hires_today;
    int land = std::popcount(unsigned(env.farms()[player].unlocked_mask));
    int hires_planned = 0, lands_planned = 0, animals_planned = 0;
    int previous_day = step / 24;
    for (int future = step; future < std::min(int(tape.size()), step + horizon); ++future) {
      const int day = future / 24; if (day != previous_day) { hires = 0; previous_day = day; }
      for (const auto& order : tape[future].market) {
        double spend = 0, revenue = 0;
        const int item = int(order.item), q = std::max(1, quantity(order));
        if (order.op == Op::HIRE) { spend = fib_cost(hires++); hires_planned++; }
        else if (order.op == Op::BUY_LAND) {
          const int extra = std::max(0, land - 1);
          if (extra < 3) { spend = land_cost[extra]; land++; lands_planned++; }
        } else if (order.op == Op::BUY_SEED && item >= 0 && item < 5) spend = q * seed_cost[item];
        else if (order.op == Op::BUY_ANIMAL && item >= 9 && item < 12) {
          spend = q * animal_cost[item - 9]; animals_planned += q;
        } else if (order.op == Op::BUY_PRODUCT && item >= 0 && item < N_PRODUCTS)
          spend = q * env.market().prices[item];
        else if (order.op == Op::SELL && item >= 0 && item < N_PRODUCTS)
          revenue = q * env.market().prices[item];
        expense += spend; cumulative += spend - revenue; requirement = std::max(requirement, cumulative);
      }
    }
    push(expense); push(requirement); push(env.farms()[player].money - requirement);
    push(hires_planned); push(lands_planned); push(animals_planned);
  }
  if (at_index != result.size()) throw std::runtime_error("native route feature dimension mismatch");
  return result;
}

}  // namespace

void NativeAgentState::reset() {
  *this = NativeAgentState{};
}

PlayerAction NativeTeammateExecutor::action(const Simulator& env, int player,
                                             int route,
                                             NativeAgentState& state) const {
  const int step = env.step_count();
  if (step == 0 || step < state.last_step) state.reset();
  state.last_step = step;
  if (route < 0 || route >= int(library_.routes.size()) ||
      library_.routes[route].empty()) return {};
  const auto& tape = library_.routes[route];
  PlayerAction out = aligned(tape[std::min(step, int(tape.size()) - 1)], env, player);
  const auto pos = positions(env, player);

  // K320 weed repair and replay catch-up.
  state.weed.resize(out.units.size());
  int hires = 0;
  for (int ahead = 0; ahead < 3; ++ahead) {
    const auto& x = tape[std::min(step + ahead, int(tape.size()) - 1)];
    hires += std::count_if(x.market.begin(), x.market.end(),
                           [](const Action& a) { return a.op == Op::HIRE; });
  }
  const bool farmer_barrier = hires >= 5;
  for (size_t u = 0; u < state.weed.size(); ++u) {
    auto& w = state.weed[u];
    if (farmer_barrier && u == 0) { w.active = false; continue; }
    if (!w.active) continue;
    const int age = step - w.start;
    if (age == 1) out.units[u] = w.intended;
    else if (age >= 2 && age <= 9) {
      const auto& previous = tape[std::max(0, step - 1)];
      out.units[u] = u < previous.units.size() ? previous.units[u] : Action{};
    } else w.active = false;
  }
  for (size_t u = 0; u < out.units.size() && u < pos.size(); ++u) {
    auto& w = state.weed[u];
    if (w.active || (farmer_barrier && u == 0)) continue;
    const auto op = out.units[u].op;
    if (op != Op::BUILD_PASTURE && op != Op::PLANT) continue;
    const Tile* t = tile_at(env, player, pos[u]);
    if (t && t->kind == TileKind::WEED) {
      w = {true, step, out.units[u]};
      out.units[u] = Action{Op::DIG};
    }
  }

  // Late capacity evacuation.
  if (step >= 648) {
    const int day = step / 24, hour = step % 24;
    if (state.room_evac.day != day) state.room_evac = {{}, -1, {}, day};
    if (hour >= 21) {
      const auto& pr = env.privates()[player];
      int total = shed_sum(pr);
      for (const auto& inv : pr.inventories)
        total += std::accumulate(inv.begin(), inv.end(), 0);
      if (hour == 21 && !state.room_evac.active && total > 100) {
        std::tuple<int, int, int> best{999, 0, 999};
        Position best_target{};
        for (size_t u = 0; u < pos.size() && u < pr.inventories.size(); ++u) {
          int saleable = std::accumulate(pr.inventories[u].begin(),
                                         pr.inventories[u].begin() + N_PRODUCTS, 0);
          if (saleable <= 0 || u >= out.units.size() || out.units[u].op != Op::PASS) continue;
          Position target = SHED_ACCESS[0];
          for (auto p : SHED_ACCESS) if (distance(pos[u], p) < distance(pos[u], target)) target = p;
          const int d = distance(pos[u], target);
          auto candidate = std::tuple{d, -saleable, int(u)};
          if (d <= 2 && candidate < best) { best = candidate; best_target = target; }
        }
        if (std::get<2>(best) != 999) {
          state.room_evac.active = true;
          state.room_evac.actor = std::get<2>(best);
          state.room_evac.target = best_target;
        }
      }
      if (state.room_evac.active) {
        const int u = state.room_evac.actor;
        if (u < 0 || u >= int(pos.size()) || u >= int(pr.inventories.size()))
          state.room_evac.active = false;
        else if (!at(pos[u], state.room_evac.target)) out.units[u] = move_toward(pos[u], state.room_evac.target);
        else if (hour == 23) {
          out.units[u] = Action{Op::DROP};
          int needed = std::max(0, total - 100);
          for (int item : ROOM_PRIORITY) {
            const int available = std::max(0, pr.inventories[u][item] - planned_sell(out, item));
            const int q = std::min(needed, available);
            merge_sale(out, item, q); needed -= q;
            if (needed <= 0) break;
          }
        }
      }
    }
  }

  // K320 rolling repayment.
  for (auto it = out.market.begin(); it != out.market.end();) {
    const int slot = premium_slot(int(it->item));
    if (sell(*it) && slot >= 0 && state.k320_due[slot] > 0) {
      const int reduce = std::min(quantity(*it), state.k320_due[slot]);
      it->quantity = quantity(*it) - reduce; state.k320_due[slot] -= reduce;
      if (it->quantity <= 0) { it = out.market.erase(it); continue; }
    }
    ++it;
  }
  rank_sell_slots(env, out);

  // K320 four-step preemption.
  const int clone = clone_distance(env);
  const bool exact = clone == 0;
  if (step >= (exact ? 120 : 216) && step < 680 && clone <= (exact ? 6 : 100) &&
      std::accumulate(state.k320_due.begin(), state.k320_due.end(), 0) == 0 &&
      out.market.size() < 10) {
    auto remaining = projected_shed(env, player, out);
    for (const auto& a : out.market) if (sell(a)) remaining[int(a.item)] =
        std::max(0, remaining[int(a.item)] - quantity(a));
    const std::array<int, 4> choices{3, 4, 6, 7};
    for (int item : choices) {
      if (exact && item == 4) continue;
      int future = 0;
      for (int h = 1; h <= 4 && step + h < int(tape.size()); ++h)
        future += planned_sell(tape[step + h], item);
      if (future < 4 || out.market.size() >= 10) continue;
      const int q = std::min({remaining[item], future, exact ? 32 : 12});
      if (q > 0) {
        out.market.push_back(Action{Op::SELL, Item(item), q});
        state.k320_due[premium_slot(item)] = q;
      }
    }
  }

  // R5 and MD opponent-family counters.
  const Farm& opponent = env.farms()[1 - player];
  const int cows = animal_count(opponent, Item::COW);
  const int sheep = animal_count(opponent, Item::SHEEP);
  if (!state.r5_target && step >= 24 && sheep >= 4 && cows <= 3) state.r5_target = true;
  if (!state.md_target && step >= 160 &&
      ((std::popcount(unsigned(opponent.unlocked_mask)) >= 2 && cows >= 4 && sheep <= 2) || cows >= 9))
    state.md_target = true;
  auto reference_counter = [&](const std::vector<PlayerAction>& ref, int future,
                               double fraction, bool enabled) {
    if (!enabled || future < 0 || future >= int(ref.size())) return;
    const auto& pr = env.privates()[player];
    for (int item : COUNTER_ITEMS) {
      const int target = planned_sell(ref[future], item);
      if (target <= 0) continue;
      if (fraction == .5 &&
          (shop_demand(env, item, step) > 0 || shop_demand(env, item, step + 1) > 0)) continue;
      const int available = std::max(0, pr.shed[item] - planned_sell(out, item) - pickup_reserve(out, item));
      const int desired = std::max(1, int(std::nearbyint(target * fraction)));
      merge_sale(out, item, std::min(available, desired));
    }
    if (out.market.size() > 10) out.market.resize(10);
  };
  reference_counter(library_.r5_reference, step + 3, .5, state.r5_target);
  reference_counter(library_.md_reference, step + 1, 2.0, state.md_target);

  // End-of-day capacity guard.
  if (step % 24 == 23) {
    const auto& pr = env.privates()[player];
    int carried = 0; for (const auto& inv : pr.inventories) carried += std::accumulate(inv.begin(), inv.end(), 0);
    int produced = 0, consumed = 0;
    for (size_t u = 0; u < out.units.size() && u < pos.size(); ++u) {
      const Tile* t = tile_at(env, player, pos[u]); const auto& x = out.units[u];
      if (x.op == Op::HARVEST && t) produced += std::max(0, int(t->yield_units));
      else if (x.op == Op::COLLECT_FERTILIZER && t && t->fertilizer_available) produced++;
      else if (x.op == Op::FEED || x.op == Op::FERTILIZE) consumed++;
      else if (x.op == Op::PLACE && int(x.item) >= 9) consumed++;
    }
    int actual_sells = 0, buys = 0;
    for (int item = 0; item < N_PRODUCTS; ++item)
      actual_sells += std::min(pr.shed[item], planned_sell(out, item));
    for (const auto& x : out.market)
      if (x.op == Op::BUY_PRODUCT || x.op == Op::BUY_ANIMAL) buys += quantity(x);
    int needed = std::max(0, shed_sum(pr) + carried + produced - consumed + buys - actual_sells - 100);
    for (int item : ROOM_PRIORITY) {
      const int q = std::min(needed, std::max(0, pr.shed[item] - planned_sell(out, item)));
      merge_sale(out, item, q); needed -= q;
      if (needed <= 0) break;
    }
    if (out.market.size() > 10) out.market.resize(10);
  }

  // Terminal liquidation.
  if (step >= 716) {
    const auto& shed = env.privates()[player].shed;
    for (int item : LIQUIDATION) {
      const int extra = step >= 718 ? shed[item] : std::max(0, shed[item] - planned_sell(out, item));
      if (extra > 0 && out.market.size() < 10)
        out.market.push_back(Action{Op::SELL, Item(item), extra});
    }
  }
  out = aligned(std::move(out), env, player);

  // Do not buy one-shot seeds that can no longer be planted and harvested.
  auto seeds = env.privates()[player].seeds;
  for (auto it = out.market.begin(); it != out.market.end();) {
    const int item = int(it->item);
    if (it->op == Op::BUY_SEED && (item == 0 || item == 1)) {
      const int last_day = 29 - 2;
      int need = std::count_if(out.units.begin(), out.units.end(), [item](const Action& a) {
        return a.op == Op::PLANT && int(a.item) == item;
      });
      for (int ahead = step + 1; ahead < int(tape.size()) && ahead / 24 <= last_day; ++ahead)
        need += std::count_if(tape[ahead].units.begin(), tape[ahead].units.end(), [item](const Action& a) {
          return a.op == Op::PLANT && int(a.item) == item;
        });
      const int q = std::min(quantity(*it), std::max(0, need - seeds[item]));
      if (q <= 0) { it = out.market.erase(it); continue; }
      it->quantity = q; seeds[item] += q;
    }
    ++it;
  }

  // FC15 residual premium sale.
  if (step >= 120 && clone <= 6) {
    auto projected = projected_shed(env, player, out);
    static constexpr int bases[9] = {25,35,60,120,250,50,160,200,100};
    for (int item : K320_PREMIUM) {
      const int available = std::max(0, projected[item] - planned_sell(out, item) - pickup_reserve(out, item));
      if (available > 0 && env.market().prices[item] >= bases[item]) merge_sale(out, item, 1);
    }
    if (out.market.size() > 10) out.market.resize(10);
  }

  // Moon opponent-market observer.  The overlay uses its own five-route tape.
  auto& race = state.moon_race;
  if (step == 0 || step < race.last_step) {
    race = NativeAgentState::MoonRace{};
    race.horizon = {1,1,1,1};
  }
  const int moon_label = moon_route(env);
  if (state.moon_layout < 0 && step >= 24 && step < 72) {
    const Farm& opp = env.farms()[1 - player];
    const auto sig = signature(opp);
    state.moon_layout = sig.counts[0] == 5 && sig.counts[4] == 5 &&
                        animal_count(opp, Item::COW) == 1 &&
                        animal_count(opp, Item::SHEEP) == 4 && sig.counts[8] == 0 &&
                        opp.money <= 12 ? 1 : 0;
  }
  const auto& moon_tape = (state.moon_layout == 1 ? library_.moon_legacy : library_.moon)[moon_label];
  auto moon_planned = [&](int at_step, int item) { return tape_sell(moon_tape, at_step, item); };
  for (double& x : race.evidence) x *= .999;
  for (auto& row : race.scores) for (double& x : row) x *= .999;
  race.policy_evidence *= .999; for (double& x : race.policy_scores) x *= .999;
  if (race.last_step == step - 1) {
    for (int item : PREMIUM) {
      const int slot = premium_slot(item);
      if (race.prices[item] <= 1 || env.market().prices[item] <= 1) continue;
      int drain = 0;
      if ((step - 1) % 4 == 0) {
        // Reconstruct prior-shop consumption from the stored shop list.
        Simulator const& same = env;
        (void)same;
        static constexpr uint16_t masks[8] = {
          33,41,15,73,2,69,72,128};
        for (int sh : race.shops) if (masks[sh] & (1u << item))
          drain += std::popcount(unsigned(masks[sh])) == 1 ? 2 : 1;
      }
      if ((step - 1) % 24 == 0) drain++;
      const int opponent_supply = env.market().inventory[item] - race.inventory[item] + drain - race.own_sells[slot];
      const int extra = opponent_supply - moon_planned(step - 1, item);
      // The embedded Python Moon observer always advances/decays its public
      // snapshot, but suppresses horizon evidence while the two public farms
      // are not clone-like.  Keep the snapshot work above and gate only the
      // learning update here.
      if (extra < 4 || clone > 6) continue;
      race.evidence[slot] += 1; race.policy_evidence += 1;
      for (int h = 1; h <= 6; ++h) {
        const int expected = moon_planned(step - 1 + h, item);
        const double delta = expected > 0 ? 1.0 + double(std::min(extra, expected)) / std::max(extra, expected) : -.15;
        race.scores[slot][h-1] += delta; race.policy_scores[h-1] += delta;
      }
      if (race.evidence[slot] >= 1.5) {
        std::array<int,6> order{0,1,2,3,4,5};
        std::sort(order.begin(), order.end(), [&](int a, int b) {
          if (race.scores[slot][a] != race.scores[slot][b])
            return race.scores[slot][a] > race.scores[slot][b];
          return a < b;
        });
        if (race.scores[slot][order[0]] >= race.scores[slot][order[1]] + .25)
          race.horizon[slot] = std::min(6, order[0] + 2);
      }
    }
  }
  if (race.policy_evidence >= 1.5) {
    std::array<int,6> order{0,1,2,3,4,5};
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      if (race.policy_scores[a] != race.policy_scores[b])
        return race.policy_scores[a] > race.policy_scores[b];
      return a < b;
    });
    if (race.policy_scores[order[0]] >= race.policy_scores[order[1]] + .25) {
      race.policy_horizon = std::min(6, order[0] + 2);
      for (int& h : race.horizon) if (h == 1) h = race.policy_horizon;
    }
  }
  race.last_step = step; race.inventory = env.market().inventory; race.prices = env.market().prices;
  race.shops = env.shops();

  // Moon repayment.
  for (auto it = out.market.begin(); it != out.market.end();) {
    const int slot = premium_slot(int(it->item));
    if (sell(*it) && slot >= 0 && state.moon_debts[step][slot] > 0) {
      const int r = std::min(quantity(*it), state.moon_debts[step][slot]);
      it->quantity = quantity(*it) - r; state.moon_debts[step][slot] -= r;
      if (it->quantity <= 0) { it = out.market.erase(it); continue; }
    }
    ++it;
  }
  if (step >= 120 && step < 680 && clone <= 6 && out.market.size() < 10) {
    if (step >= 120 && clone <= 2) for (int& h : race.horizon) h = std::max(h, 4);
    auto remaining = projected_shed(env, player, out);
    for (const auto& x : out.market) if (sell(x)) remaining[int(x.item)] =
        std::max(0, remaining[int(x.item)] - quantity(x));
    struct Choice { double value; int item, q, horizon; };
    std::vector<Choice> choices;
    for (int item : PREMIUM) {
      const int slot = premium_slot(item);
      for (int h = race.horizon[slot]; h >= 1; --h) {
        const int future = moon_planned(step + h, item);
        if (future < 4) continue;
        const int q = std::min({remaining[item], future, 12});
        if (q > 0) choices.push_back({double(env.market().prices[item]) * q, item, q, h});
        break;
      }
    }
    std::vector<Choice> selected;
    for (auto x : choices) if (x.horizon > 1) selected.push_back(x);
    if (!selected.empty()) {
      const Choice best = *std::max_element(selected.begin(), selected.end(),
          [](auto a, auto b) { return std::tie(a.value,a.item,a.q,a.horizon) < std::tie(b.value,b.item,b.q,b.horizon); });
      selected.clear(); selected.push_back(best);
    }
    else selected = choices;
    for (auto x : selected) if (out.market.size() < 10) {
      out.market.push_back(Action{Op::SELL, Item(x.item), x.q});
      if (step + x.horizon < 720) state.moon_debts[step + x.horizon][premium_slot(x.item)] += x.q;
    }
  }
  // Record projected actual own premium sells for the next observation.
  race.own_sells.fill(0);
  {
    auto remaining = projected_shed(env, player, out);
    for (const auto& x : out.market) {
      const int slot = premium_slot(int(x.item));
      if (!sell(x) || slot < 0) continue;
      const int q = std::min(quantity(x), remaining[int(x.item)]);
      race.own_sells[slot] += q; remaining[int(x.item)] -= q;
    }
  }

  // Opening and cash guards.
  for (auto& x : out.market) {
    if (step == 0 && x.op == Op::BUY_SEED && x.item == Item::WHEAT) x.quantity = 8;
    if (step > 72 && step < 192 && x.op == Op::BUY_SEED && x.item == Item::CARROT && quantity(x) >= 5)
      x.quantity = std::max(0, quantity(x) - 1);
  }
  if (out.market.size() > 10) out.market.resize(10);

  // Feed-value guard and wheat purchase credit.
  const auto& pr = env.privates()[player];
  int skipped = 0;
  if (step / 24 >= 10) for (size_t u = 0; u < out.units.size() && u < pos.size(); ++u) {
    if (out.units[u].op != Op::FEED) continue;
    const Tile* t = tile_at(env, player, pos[u]);
    if (!t || t->kind != TileKind::ANIMAL || t->fed_today || t->consecutive_unfed != 0) continue;
    const int product = t->animal == Item::GOOSE ? 5 : t->animal == Item::COW ? 6 : 7;
    if (env.market().prices[product] * (1 + t->pending_care_bonus) < env.market().prices[0]) {
      out.units[u] = Action{}; skipped++;
    }
  }
  state.wheat_credit += skipped;
  for (auto it = out.market.begin(); it != out.market.end();) {
    if (state.wheat_credit > 0 && it->op == Op::BUY_PRODUCT && it->item == Item::WHEAT) {
      const int take = std::min(quantity(*it), state.wheat_credit);
      it->quantity = quantity(*it) - take; state.wheat_credit -= take;
      if (it->quantity <= 0) { it = out.market.erase(it); continue; }
    }
    ++it;
  }

  // Terminal crop salvage overlay.
  if (!state.salvage.active && step >= 696) {
    long best = std::numeric_limits<long>::min();
    for (size_t u = 0; u < pos.size() && u < out.units.size() && u < pr.inventories.size(); ++u) {
      const Op op = out.units[u].op;
      if (op != Op::NORTH && op != Op::SOUTH && op != Op::EAST && op != Op::WEST) continue;
      const Tile* t = tile_at(env, player, pos[u]);
      if (!t || t->kind != TileKind::PLANT || t->yield_units <= 0) continue;
      static constexpr int first[5] = {2,2,8,10,10};
      if (step / 24 - t->planted_day < first[int(t->crop)]) continue;
      Position target = SHED_ACCESS[0];
      for (auto p : SHED_ACCESS) if (distance(pos[u], p) < distance(pos[u], target)) target = p;
      if (distance(pos[u], target) + 2 != 719 - step) continue;
      int carried_value = 0;
      for (int item = 0; item < N_PRODUCTS; ++item) carried_value += pr.inventories[u][item] * env.market().prices[item];
      const int crop_value = t->yield_units * env.market().prices[int(t->crop)];
      if (crop_value < 2 * carried_value) continue;
      const long score = long(crop_value) * 100 - int(u);
      if (score > best) {
        best = score; state.salvage = {true, int(u), target, t->crop, int(t->yield_units)};
      }
    }
    if (state.salvage.active) out.units[state.salvage.actor] = Action{Op::HARVEST};
  } else if (state.salvage.active && state.salvage.actor < int(pos.size())) {
    const int u = state.salvage.actor;
    out.units[u] = at(pos[u], state.salvage.target) ? Action{Op::DROP} : move_toward(pos[u], state.salvage.target);
  }
  if (state.salvage.active && state.salvage.actor < int(pos.size()) && step == 718 &&
      at(pos[state.salvage.actor], state.salvage.target)) {
    const auto projected = projected_shed(env, player, out);
    const int item = int(state.salvage.product);
    merge_sale(out, item, std::min(state.salvage.quantity,
               std::max(0, projected[item] - planned_sell(out, item))));
  }

  if (out.market.size() > 10) out.market.resize(10);
  return aligned(std::move(out), env, player);
}

NativeMatchResult NativeTeammateExecutor::play(int route0, int route1,
                                                uint64_t seed,
                                                int switch_step0,
                                                int switch_route0,
                                                int switch_step1,
                                                int switch_route1,
                                                bool capture_trace,
                                                bool capture_audit) const {
  Simulator env(Config{}, seed);
  NativeAgentState states[2];
  NativeMatchResult result;
  if (capture_trace) result.trace.reserve(719);
  while (!env.done()) {
    const int active0 = switch_step0 >= 0 && env.step_count() >= switch_step0
                            ? switch_route0 : route0;
    const int active1 = switch_step1 >= 0 && env.step_count() >= switch_step1
                            ? switch_route1 : route1;
    std::array<PlayerAction, 2> actions{
        action(env, 0, active0, states[0]), action(env, 1, active1, states[1])};
    std::array<int, 2> unit_failures{};
    if (capture_audit) unit_failures = {
        macro_unit_failures(env, 0, actions[0]),
        macro_unit_failures(env, 1, actions[1])};
    if (capture_trace) result.trace.push_back(actions);
    env.step(actions);
    if (capture_audit) for (int player = 0; player < 2; ++player) {
      const int market_failures = macro_market_failures(env, player, actions[player]);
      result.macro_unit_failures[player] += unit_failures[player];
      result.macro_market_failures[player] += market_failures;
      if ((unit_failures[player] || market_failures) &&
          result.first_macro_failure_step[player] < 0)
        result.first_macro_failure_step[player] = env.step_count() - 1;
    }
  }
  result.rewards = {env.farms()[0].money, env.farms()[1].money};
  return result;
}

NativeMatchResult NativeTeammateExecutor::play_tape_suffix(
    int opening, int opponent,
    const std::vector<PlayerAction>& candidate_tape, uint64_t seed,
    int candidate_seat, int prefix_steps, bool capture_trace) const {
  const Config config{};
  const int action_steps = config.episode_steps - 1;
  if (opening < 0 || opening >= route_count() ||
      opponent < 0 || opponent >= route_count() ||
      candidate_seat < 0 || candidate_seat > 1 ||
      prefix_steps < 0 || prefix_steps > action_steps ||
      int(candidate_tape.size()) != action_steps)
    throw std::invalid_argument(
        "play_tape_suffix requires valid routes/seat/prefix and one full action tape");

  Simulator env(config, seed);
  NativeAgentState opening_state, opponent_state;
  NativeMatchResult result;
  if (capture_trace) result.trace.reserve(action_steps);
  while (!env.done()) {
    const int step = env.step_count();
    std::array<PlayerAction, 2> joint;
    joint[candidate_seat] = step < prefix_steps
        ? action(env, candidate_seat, opening, opening_state)
        : candidate_tape[step];
    joint[1 - candidate_seat] = action(
        env, 1 - candidate_seat, opponent, opponent_state);
    if (capture_trace) result.trace.push_back(joint);
    env.step(joint);
  }
  result.rewards = {env.farms()[0].money, env.farms()[1].money};
  return result;
}

NativeMatchResult NativeTeammateExecutor::play_route_suffix(
    int opening, int opponent, int tape_route, uint64_t seed,
    int candidate_seat, int prefix_steps, bool capture_trace) const {
  if (tape_route < 0 || tape_route >= route_count())
    throw std::invalid_argument("invalid suffix tape route");
  return play_tape_suffix(
      opening, opponent, library_.routes[tape_route], seed,
      candidate_seat, prefix_steps, capture_trace);
}

NativeSwitchCaseResult NativeTeammateExecutor::switch_case(
    int opening, int opponent, uint64_t seed, int checkpoint, int seat,
    const std::vector<int>& targets) const {
  if (opening < 0 || opening >= int(library_.routes.size()) || opponent < 0 ||
      opponent >= int(library_.routes.size()) || seat < 0 || seat > 1 ||
      checkpoint < 0 || checkpoint >= 719)
    throw std::invalid_argument("invalid native switch case");
  for (int target : targets)
    if (target < 0 || target >= int(library_.routes.size()))
      throw std::invalid_argument("invalid native switch target");

  const int route0 = seat == 0 ? opening : opponent;
  const int route1 = seat == 0 ? opponent : opening;
  Simulator prefix(Config{}, seed);
  NativeAgentState prefix_states[2];
  FeatureHistory history;
  while (!prefix.done() && prefix.step_count() < checkpoint) {
    history.update(prefix);
    std::array<PlayerAction, 2> actions{
        action(prefix, 0, route0, prefix_states[0]),
        action(prefix, 1, route1, prefix_states[1])};
    prefix.step(actions);
  }
  if (prefix.done() || prefix.step_count() != checkpoint)
    throw std::runtime_error("checkpoint was not reached");
  history.update(prefix);

  NativeSwitchCaseResult result;
  result.features =
      build_features(prefix, seat, history, library_.routes[opening]);
  result.outcome.resize(targets.size());
  result.margin.resize(targets.size());
  for (size_t ti = 0; ti < targets.size(); ++ti) {
    Simulator env = prefix;
    NativeAgentState states[2]{prefix_states[0], prefix_states[1]};
    while (!env.done()) {
      const int active0 = seat == 0 ? targets[ti] : opponent;
      const int active1 = seat == 1 ? targets[ti] : opponent;
      std::array<PlayerAction, 2> actions{
          action(env, 0, active0, states[0]),
          action(env, 1, active1, states[1])};
      env.step(actions);
    }
    const double own = env.farms()[seat].money;
    const double other = env.farms()[1 - seat].money;
    const double difference = own - other;
    result.outcome[ti] = difference > 0 ? 2 : difference == 0 ? 1 : 0;
    result.margin[ti] = float(difference);
  }
  return result;
}

std::array<float, 147> NativeTeammateExecutor::features_at(
    int route0, int route1, uint64_t seed, int checkpoint, int player,
    int feature_route) const {
  if (player < 0 || player > 1 || feature_route < 0 ||
      feature_route >= int(library_.routes.size()))
    throw std::invalid_argument("invalid native feature player or route");
  Simulator env(Config{}, seed);
  NativeAgentState states[2]; FeatureHistory history;
  while (!env.done()) {
    history.update(env);
    if (env.step_count() == checkpoint)
      return build_features(env, player, history, library_.routes[feature_route]);
    std::array<PlayerAction, 2> actions{
        action(env, 0, route0, states[0]), action(env, 1, route1, states[1])};
    env.step(actions);
  }
  throw std::invalid_argument("checkpoint is outside the episode");
}

}  // namespace fastkag
