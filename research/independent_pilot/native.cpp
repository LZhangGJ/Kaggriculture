#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace py = pybind11;
using std::array;
using std::string;
using std::vector;
constexpr int NI = 12, NV = 27, NF = 59;
const array<string, NI> items = {"WHEAT",      "CARROT", "TOMATO", "STRAWBERRY",
                                 "MELON",      "EGG",    "MILK",   "WOOL",
                                 "FERTILIZER", "GOOSE",  "COW",    "SHEEP"};
const array<string, NV> verbs = {
    "WAIT",        "CONTINUE",      "CANCEL",
    "NORTH",       "SOUTH",         "EAST",
    "WEST",        "PLANT",         "WATER",
    "HARVEST",     "FERTILIZE",     "DIG",
    "BUILD_COOP",  "BUILD_PASTURE", "PLACE",
    "FEED",        "CARE",          "COLLECT_FERTILIZER",
    "DROP",        "PICKUP",        "STOP",
    "HIRE",        "BUY_LAND",      "BUY_SEED",
    "BUY_PRODUCT", "BUY_ANIMAL",    "SELL"};
enum V {
  WAIT,
  CONTINUE,
  CANCEL,
  NORTH,
  SOUTH,
  EAST,
  WEST,
  PLANT,
  WATER,
  HARVEST,
  FERTILIZE,
  DIG,
  COOP,
  PASTURE,
  PLACE,
  FEED,
  CARE,
  FERT,
  DROP,
  PICKUP,
  STOP,
  HIRE,
  LAND,
  SEED,
  PRODUCT,
  ANIMAL,
  SELL
};
const array<string, 8> shops = {
    "BAKERY",   "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP",  "SMOOTHIE_SHOP",  "YARN_STORE"};
const vector<vector<int>> shop_items = {
    {5, 0}, {5, 0, 3}, {0, 1, 2, 3}, {3, 6, 0}, {1}, {6, 2, 0}, {3, 6}, {7}};
const int seedcost[5] = {10, 20, 50, 100, 80}, firstday[5] = {2, 2, 8, 10, 10},
          maxday[5] = {4, 3, 8, 10, 12}, interval[5] = {0, 0, 1, 2, 0},
          maxyield[5] = {6, 4, 4, 4, 6};
const int acost[3] = {300, 400, 500}, astructure[3] = {3, 4, 4},
          afirst[3] = {4, 8, 6}, ainterval[3] = {1, 2, 3}, amax[3] = {4, 6, 6};
struct Pos {
  int x = 4, y = 4;
  bool operator==(const Pos &) const = default;
};
const array<Pos, 4> shed_access = {Pos{4, 4}, Pos{5, 4}, Pos{4, 5}, Pos{5, 5}};
bool adjacent(Pos p) { return p.x >= 4 && p.x <= 5 && p.y >= 4 && p.y <= 5; }
int distance(Pos a, Pos b) { return abs(a.x - b.x) + abs(a.y - b.y); }
Pos nearest(Pos p) {
  return *std::min_element(
      shed_access.begin(), shed_access.end(),
      [&](Pos a, Pos b) { return distance(p, a) < distance(p, b); });
}
int itemid(string s) {
  auto p = std::find(items.begin(), items.end(), s);
  return p == items.end() ? -1 : int(p - items.begin());
}
int verbid(string s) {
  auto p = std::find(verbs.begin(), verbs.end(), s);
  return p == verbs.end() ? WAIT : int(p - verbs.begin());
}
// CPython's integer-seeded MT19937, random(), and getrandbits rejection path.
// Town draws depend on how many empty tiles consumed random draws that day.
struct RNG {
  array<uint32_t, 624> mt{};
  int ix = 624;
  explicit RNG(uint64_t seed) {
    mt[0] = 19650218U;
    for (int i = 1; i < 624; i++)
      mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
    uint32_t key[2] = {uint32_t(seed), uint32_t(seed >> 32)};
    int nk = key[1] ? 2 : 1, i = 1, j = 0;
    for (int k = 624; k; k--) {
      mt[i] =
          (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + key[j] + j;
      if (++i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
      if (++j >= nk)
        j = 0;
    }
    for (int k = 623; k; k--) {
      mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - i;
      if (++i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
    }
    mt[0] = 0x80000000U;
  }
  uint32_t next() {
    if (ix >= 624) {
      for (int k = 0; k < 624; k++) {
        uint32_t y = (mt[k] & 0x80000000U) | (mt[(k + 1) % 624] & 0x7fffffffU);
        mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0);
      }
      ix = 0;
    }
    uint32_t y = mt[ix++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= y >> 18;
    return y;
  }
  double random() {
    uint32_t a = next() >> 5, b = next() >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }
  int shop() {
    uint32_t r;
    do {
      r = next() >> 28;
    } while (r >= 8);
    return r;
  }
};
struct Inv {
  array<int, NI> q{};
  vector<int> order;
  int sum() const {
    int s = 0;
    for (auto n : q)
      s += n;
    return s;
  }
  void add(int item, int n) {
    if (item < 0 || item >= NI || n <= 0)
      return;
    if (!q[item])
      order.push_back(item);
    q[item] += n;
  }
  bool take(int item, int n = 1) {
    if (item < 0 || item >= NI || n <= 0 || q[item] < n)
      return false;
    q[item] -= n;
    if (!q[item])
      order.erase(std::find(order.begin(), order.end(), item));
    return true;
  }
  void clear() {
    q.fill(0);
    order.clear();
  }
};
struct Tile {
  int kind = 0, item = -1, day = 0, yield = 0, life = -1, unwatered = 0,
      fertuntil = -1, unfed = 0, bonus = 0;
  bool watered = false, fed = false, cared = false, fertilizer = false;
};
struct Farm {
  double cash = 3000;
  array<Tile, 100> tiles{};
  vector<Pos> positions = {Pos{4, 4}};
  vector<Inv> invs = {Inv{}};
  array<int, NI> shed{};
  array<int, 5> seeds{};
  int unlocked = 1, hires = 0;
  Farm() {
    for (int y = 0; y < 10; y++)
      for (int x = 0; x < 10; x++)
        if (x >= 5 || y >= 5)
          tiles[y * 10 + x].kind = -1;
  }
  int stock() const {
    int n = 0;
    for (int q : shed)
      n += q;
    return n;
  }
};
struct Price {
  double base, I0, T, belowtarget, abovetarget;
  int below, above;
};
// shape: linear=0, square=1,sqrt=2,log=3,log10=4,hinge=5
const array<Price, 9> defaults = {Price{25, 10000, 400, .8, .2, 2, 3},
                                  Price{35, 10000, 450, 1, .7, 5, 2},
                                  Price{60, 10000, 200, .4, .6, 5, 2},
                                  Price{120, 10000, 100, .7, 1.6, 2, 0},
                                  Price{250, 10000, 300, .2, 3.6, 3, 1},
                                  Price{50, 10000, 332, .4, .2, 5, 3},
                                  Price{160, 10000, 122, .6, 1.6, 2, 0},
                                  Price{200, 10000, 105, .2, 3.2, 3, 1},
                                  Price{100, 10000, 200, .4, .4, 0, 0}};
double shape(int f, double x, double T) {
  x = std::max(0., x);
  switch (f) {
  case 1:
    return x * x;
  case 2:
    return sqrt(x);
  case 3:
    return log(1. + x);
  case 4:
    return log10(1. + x);
  case 5: {
    if (T <= 0)
      return x;
    double u = x / T;
    return u + 8 * pow(std::max(0., u - 1), 2);
  }
  default:
    return x;
  }
}
int quote(int item, int inventory, const array<Price, 9> &params = defaults) {
  const auto &p = params[item];
  bool low = inventory < p.I0;
  int f = low ? p.below : p.above;
  double amp =
      (low ? p.belowtarget : p.abovetarget) * p.base / shape(f, p.T, p.T);
  double v =
      p.base + (low ? 1 : -1) * amp * shape(f, abs(p.I0 - inventory), p.T);
  return std::max(1, int(std::nearbyint(v)));
}
struct Action {
  int verb = WAIT, item = -1, q = 1;
};
struct Turn {
  vector<Action> units, market;
};
double hirecost(int n, int mult = 1) {
  double a = 1, b = 1;
  for (int i = 0; i < n; i++) {
    double c = a + b;
    a = b;
    b = c;
  }
  return mult * a;
}
void hire(Farm &f, int mult = 1) {
  double c = hirecost(f.hires, mult);
  if (f.cash < c)
    return;
  f.cash -= c;
  f.hires++;
  int best = 0, minimum = 1000000;
  for (int k = 0; k < 4; k++) {
    int n = 0;
    for (auto p : f.positions)
      n += p == shed_access[k];
    if (n < minimum) {
      minimum = n;
      best = k;
    }
  }
  f.positions.push_back(shed_access[best]);
  f.invs.emplace_back();
}
void land(Farm &f) {
  const int cost[3] = {1000, 2000, 4000};
  if (f.unlocked >= 4 || f.cash < cost[f.unlocked - 1])
    return;
  f.cash -= cost[f.unlocked - 1];
  int k = f.unlocked++;
  for (int y = 0; y < 10; y++)
    for (int x = 0; x < 10; x++)
      if ((x >= 5) + 2 * (y >= 5) == k && f.tiles[y * 10 + x].kind == -1)
        f.tiles[y * 10 + x] = Tile{};
}
void drop(Farm &f, Inv &inv, int cap) {
  for (int item : inv.order) {
    int n = std::min(inv.q[item], std::max(0, cap - f.stock()));
    f.shed[item] += n;
  }
  inv.clear();
}
void unit(Farm &f, int worker, Action a, int day, int tpd = 24, int cap = 100) {
  if (worker < 0 || worker >= int(f.positions.size()))
    return;
  auto &p = f.positions[worker];
  auto &inv = f.invs[worker];
  int v = a.verb, it = a.item;
  if (v >= NORTH && v <= WEST) {
    Pos n = p;
    if (v == NORTH)
      n.y--;
    if (v == SOUTH)
      n.y++;
    if (v == EAST)
      n.x++;
    if (v == WEST)
      n.x--;
    if (n.x >= 0 && n.x < 10 && n.y >= 0 && n.y < 10)
      p = n;
    return;
  }
  if (v == WAIT)
    return;
  auto &t = f.tiles[p.y * 10 + p.x];
  if (v == DROP) {
    if (adjacent(p))
      drop(f, inv, cap);
    return;
  }
  if (v == PICKUP) {
    if (adjacent(p) && it >= 0 && a.q > 0) {
      int n = std::min(a.q, f.shed[it]);
      f.shed[it] -= n;
      inv.add(it, n);
    }
    return;
  }
  if (v == PLACE) {
    if (it >= 9 && it < 12 && t.kind == astructure[it - 9] && t.item < 0) {
      if (inv.take(it)) {
        t = Tile{};
        t.kind = astructure[it - 9];
        t.item = it;
        t.day = day;
      }
      return;
    }
    if (adjacent(p) && it >= 0 && a.q > 0) {
      int n = std::min({a.q, inv.q[it], std::max(0, cap - f.stock())});
      if (n > 0) {
        inv.take(it, n);
        f.shed[it] += n;
      }
    }
    return;
  }
  if (t.kind == -1)
    return;
  if (v == PLANT) {
    if (it >= 0 && it < 5 && t.kind == 0 && f.seeds[it] > 0) {
      f.seeds[it]--;
      t = Tile{};
      t.kind = 2;
      t.item = it;
      t.day = day;
      t.unwatered = 1;
      t.yield = interval[it] ? 0 : 1;
      t.life = interval[it] ? -1 : (day + maxday[it] + 1) * tpd;
    }
    return;
  }
  if (v == WATER) {
    if (t.kind != 2 || t.watered)
      return;
    t.watered = true;
    int age = day - t.day;
    if (!interval[t.item] && age >= (maxday[t.item] + 1) / 2 &&
        age <= maxday[t.item])
      t.yield =
          std::min(maxyield[t.item], t.yield + (t.fertuntil >= day ? 2 : 1));
    return;
  }
  if (v == HARVEST) {
    if (t.yield <= 0)
      return;
    if (t.kind == 2) {
      if (day - t.day < firstday[t.item])
        return;
      inv.add(t.item, t.yield);
      t.yield = 0;
      if (!interval[t.item])
        t = Tile{};
    } else if (t.item >= 9) {
      inv.add(5 + t.item - 9, t.yield);
      t.yield = 0;
    }
    return;
  }
  if (v == FERTILIZE) {
    if (t.kind == 2 && inv.take(8))
      t.fertuntil = std::max(t.fertuntil, day + 2);
    return;
  }
  if (v == DIG) {
    if (t.item < 9)
      t = Tile{};
    return;
  }
  if (v == COOP || v == PASTURE) {
    if (t.kind == 0)
      t.kind = v == COOP ? 3 : 4;
    return;
  }
  if (t.item >= 9) {
    if (v == FEED && !t.fed && inv.take(0))
      t.fed = true;
    if (v == CARE)
      t.cared = true;
    if (v == FERT && t.fertilizer) {
      t.fertilizer = false;
      inv.add(8, 1);
    }
  }
}
bool trade(Farm &f, array<int, 9> &market, Action a, int price, int cap = 100) {
  int it = a.item;
  if (it < 0)
    return false;
  if (a.verb == SELL) {
    if (it >= 9 || f.shed[it] <= 0)
      return false;
    f.shed[it]--;
    f.cash += price;
    if (price > 1)
      market[it]++;
    return true;
  }
  if (f.cash < price)
    return false;
  if (a.verb == SEED && it < 5) {
    f.cash -= price;
    f.seeds[it]++;
    return true;
  }
  if ((a.verb == PRODUCT && (it == 0 || it == 8)) ||
      (a.verb == ANIMAL && it >= 9)) {
    if (f.stock() >= cap)
      return false;
    f.cash -= price;
    f.shed[it]++;
    if (a.verb == PRODUCT)
      market[it]--;
    return true;
  }
  return false;
}
struct State {
  uint64_t seed = 0;
  int step = 0;
  array<Farm, 2> farms;
  array<int, 9> market{};
  vector<int> town;
  array<Price, 9> params = defaults;
  int tpd = 24, cap = 100, shop_interval = 4, center_interval = 24,
      unlock_interval = 3, hire_mult = 1, terminal = 719;
  double weed = .005;
  explicit State(uint64_t s = 0) : seed(s) {
    for (int i = 0; i < 9; i++)
      market[i] = 10000;
  }
  bool done() const { return step >= terminal; }
  void endday() {
    int day = step / tpd;
    RNG rng((seed * 1000003ULL) ^ uint64_t(day));
    for (auto &f : farms) {
      for (auto &t : f.tiles) {
        if (t.kind == 2) {
          bool watered = t.watered;
          t.unwatered = watered ? 0 : t.unwatered + 1;
          t.watered = false;
          if (t.unwatered >= 2) {
            t = Tile{};
            t.kind = 1;
            continue;
          }
          int it = t.item;
          if (interval[it]) {
            int d = day + 1 - t.day - firstday[it];
            if (d >= 0 && d % interval[it] == 0) {
              int count = d / interval[it] + 1;
              if (count <= maxyield[it]) {
                t.yield =
                    std::min(maxyield[it],
                             t.yield + (watered && t.fertuntil >= day ? 2 : 1));
                if (count == maxyield[it])
                  t.life = (day + 2) * tpd;
              }
            }
          }
        }
      }
      for (auto &t : f.tiles)
        if (t.item >= 9) {
          int it = t.item - 9;
          t.unfed = t.fed ? 0 : t.unfed + 1;
          if (t.unfed >= 2) {
            t = Tile{};
            t.kind = astructure[it];
            continue;
          }
          int d = day + 1 - t.day - afirst[it];
          if (d >= 0 && d % ainterval[it] == 0) {
            t.yield = std::min(amax[it], t.yield + 1 + (t.fed ? t.bonus : 0));
            t.bonus = 0;
          }
          if (t.cared && t.fed)
            t.bonus++;
          t.fertilizer = true;
          t.fed = t.cared = false;
        }
      for (auto &t : f.tiles)
        if (t.kind == 0 && rng.random() < weed)
          t.kind = 1;
      for (auto &inv : f.invs)
        drop(f, inv, cap);
      f.positions = {Pos{4, 4}};
      f.invs = {Inv{}};
      f.hires = 0;
    }
    if ((day + 1) % unlock_interval == 0 && town.size() < 8)
      town.push_back(rng.shop());
  }
  void advance(array<Turn, 2> &actions) {
    if (done())
      throw std::runtime_error("step after terminal");
    int day = step / tpd;
    for (int s = 0; s < 2; s++) {
      array<int, 5> demand{};
      array<bool, 5> blocked{};
      for (auto a : actions[s].units)
        if (a.verb == PLANT && a.item >= 0 && a.item < 5)
          demand[a.item]++;
      for (int i = 0; i < 5; i++)
        blocked[i] = demand[i] > farms[s].seeds[i];
      for (int w = 0; w < int(actions[s].units.size()); w++) {
        auto a = actions[s].units[w];
        if (a.verb == PLANT && a.item >= 0 && a.item < 5 && blocked[a.item])
          continue;
        unit(farms[s], w, a, day, tpd, cap);
      }
    }
    for (int k = 0; k < 10; k++) {
      array<Action, 2> a{};
      array<int, 2> remaining{};
      for (int s = 0; s < 2; s++)
        if (k < int(actions[s].market.size())) {
          a[s] = actions[s].market[k];
          if (a[s].verb == HIRE)
            hire(farms[s], hire_mult);
          else if (a[s].verb == LAND)
            land(farms[s]);
          else
            remaining[s] = std::max(0, a[s].q);
        }
      for (int iters = 0; (remaining[0] || remaining[1]) && iters < 99999;
           iters++) {
        int prices[2] = {-1, -1};
        for (int s = 0; s < 2; s++)
          if (remaining[s]) {
            int it = a[s].item, v = a[s].verb;
            if (v == SELL && it >= 0 && it < 9)
              prices[s] = quote(it, market[it], params);
            else if (v == PRODUCT && (it == 0 || it == 8))
              prices[s] = quote(it, market[it] - 1, params);
            else if (v == SEED && it >= 0 && it < 5)
              prices[s] = seedcost[it];
            else if (v == ANIMAL && it >= 9 && it < 12)
              prices[s] = acost[it - 9];
            else
              remaining[s] = 0;
          }
        for (int s = 0; s < 2; s++)
          if (remaining[s]) {
            if (trade(farms[s], market, a[s], prices[s], cap))
              remaining[s]--;
            else
              remaining[s] = 0;
          }
      }
    }
    if (step % shop_interval == 0)
      for (int sh : town)
        for (int it : shop_items[sh])
          market[it] -= shop_items[sh].size() == 1 ? 2 : 1;
    if (step % center_interval == 0)
      for (int it = 0; it < 8; it++)
        market[it]--;
    for (auto &f : farms)
      for (auto &t : f.tiles)
        if (t.kind == 2 && t.life >= 0 && step >= t.life &&
            (step - t.life) % 2 == 0) {
          if (--t.yield <= 0) {
            t = Tile{};
            t.kind = 1;
          }
        }
    if ((step + 1) % tpd == 0)
      endday();
    step++;
  }
};
struct Choice {
  int verb = WAIT;
  Pos target{-1, -1};
  int item = -1, q = 1;
  bool operator==(const Choice &) const = default;
};
struct Tasks {
  vector<Choice> tasks;
  vector<bool> valid;
  int day = -1;
  void refresh(int d, int n) {
    if (d != day) {
      for (int i = 1; i < int(valid.size()); i++)
        valid[i] = false;
      day = d;
    }
    tasks.resize(n);
    valid.resize(n, false);
  }
};
struct Ledger {
  Farm f;
  array<int, 9> market;
  array<Price, 9> params;
  Tasks *tasks = nullptr;
  vector<Choice> reserved;
  array<int, 5> seeds_reserved{};
  Turn action;
  int day = 0, hour = 0;
  bool full_actions = false;
  bool uncertain = false, started_market = false;
  double upper_cash = 0;
  double guaranteed_cash = 0;
  array<int, NI> upper_shed{};
  void begin(const State &s, int seat, Tasks &t) {
    f = s.farms[seat];
    market = s.market;
    params = s.params;
    tasks = &t;
    day = s.step / 24;
    hour = s.step % 24;
    t.refresh(day, f.positions.size());
    reserved.clear();
    seeds_reserved.fill(0);
    action.units.clear();
    action.market.clear();
    uncertain = started_market = false;
  }
  bool feasible(Choice c, int w) const {
    int v = c.verb, it = c.item;
    Pos p = f.positions[w];
    if (v == WAIT || v == CANCEL)
      return true;
    if (v >= NORTH && v <= WEST)
      return v == NORTH   ? p.y > 0
             : v == SOUTH ? p.y < 9
             : v == EAST  ? p.x < 9
                          : p.x > 0;
    if (v != DROP && v != PICKUP &&
        std::find(reserved.begin(), reserved.end(), c) != reserved.end())
      return false;
    if (c.target.x < 0 || c.target.y < 0 || c.target.x >= 10 ||
        c.target.y >= 10)
      return false;
    const auto &t = f.tiles[c.target.y * 10 + c.target.x];
    const auto &inv = f.invs[w];
    bool adj = adjacent(c.target);
    if (v == DROP)
      return adj && !inv.order.empty();
    if (v == PICKUP)
      return adj && c.q > 0 && it >= 0 && f.shed[it] > 0;
    if (v == PLACE) {
      if (it < 0)
        return false;
      bool carried = inv.q[it] > 0, borrowed = !carried && f.shed[it] > 0;
      if (!carried && !borrowed)
        return false;
      if (it >= 9 && t.kind == astructure[it - 9] && t.item < 0)
        return true;
      return adj && c.q > 0 && f.stock() - int(borrowed) < 100;
    }
    if (t.kind == -1)
      return false;
    if (v == PLANT)
      return t.kind == 0 && it >= 0 && it < 5 &&
             f.seeds[it] > seeds_reserved[it];
    if (v == COOP || v == PASTURE)
      return t.kind == 0;
    if (v == DIG)
      return t.kind != 0 && t.item < 9;
    if (v == WATER)
      return t.kind == 2 && !t.watered;
    if (v == FERTILIZE)
      return t.kind == 2 && (inv.q[8] > 0 || f.shed[8] > 0);
    if (v == FEED)
      return t.item >= 9 && !t.fed && (inv.q[0] > 0 || f.shed[0] > 0);
    if (v == CARE)
      return t.item >= 9 && !t.cared;
    if (v == FERT)
      return t.item >= 9 && t.fertilizer;
    if (v == HARVEST)
      return t.yield > 0 &&
             (t.item >= 9 || (t.kind == 2 && day - t.day >= firstday[t.item]));
    return false;
  }
  // Direct game commands: no routes, reservations, crop preferences or plans.
  void direct_menu(int w, vector<Choice> &out) {
    out.push_back(Choice{});
    Pos p = f.positions[w];
    const auto &t = f.tiles[p.y * 10 + p.x];
    const auto &inv = f.invs[w];
    auto add = [&](int v, int it = -1, int q = 1) {
      out.push_back(Choice{v, p, it, q});
    };
    if (p.y > 0) add(NORTH);
    if (p.y < 9) add(SOUTH);
    if (p.x < 9) add(EAST);
    if (p.x > 0) add(WEST);
    if (adjacent(p)) {
      if (inv.sum() > 0) add(DROP);
      for (int it = 0; it < NI; ++it) {
        if (f.shed[it] > 0) add(PICKUP, it, f.shed[it]);
        bool animal_slot = it >= 9 && t.kind == astructure[it - 9] && t.item < 0;
        if (!animal_slot && inv.q[it] > 0 && f.stock() < 100)
          add(PLACE, it, std::min(inv.q[it], 100 - f.stock()));
      }
    }
    if (t.kind == -1) return;
    if (t.kind == 0) {
      for (int it = 0; it < 5; ++it)
        if (f.seeds[it] > 0) add(PLANT, it);
      add(COOP); add(PASTURE);
    }
    if (t.item < 9) add(DIG);
    if (t.kind == 2) {
      if (!t.watered) add(WATER);
      if (inv.q[8] > 0) add(FERTILIZE);
    }
    if (t.yield > 0 && (t.item >= 9 || (t.kind == 2 && day - t.day >= firstday[t.item])))
      add(HARVEST);
    for (int it = 9; it < NI; ++it)
      if (inv.q[it] > 0 && t.kind == astructure[it - 9] && t.item < 0) add(PLACE, it);
    if (t.item >= 9) {
      if (!t.fed && inv.q[0] > 0) add(FEED);
      add(CARE);
      if (t.fertilizer) add(FERT);
    }
  }
  void menu(int w, vector<Choice> &out) {
    out.clear();
    if (w < 0) {
      market_menu(out);
      return;
    }
    if (full_actions) { direct_menu(w, out); return; }
    out.push_back(Choice{});
    if (tasks->valid[w] && feasible(tasks->tasks[w], w)) {
      out.push_back(Choice{CONTINUE});
      out.push_back(Choice{CANCEL});
    } else
      tasks->valid[w] = false;
    for (int v = NORTH; v <= WEST; v++)
      if (feasible(Choice{v}, w))
        out.push_back(Choice{v});
    const auto &inv = f.invs[w];
    if (inv.sum() > 0) {
      for (auto p : shed_access)
        out.push_back(Choice{DROP, p});
      if (f.stock() < 100)
        for (auto p : shed_access)
          for (int it : inv.order)
            out.push_back(
                Choice{PLACE, p, it, std::min(inv.q[it], 100 - f.stock())});
    }
    auto add = [&](Choice c) {
      if (feasible(c, w))
        out.push_back(c);
    };
    for (int y = 0; y < 10; y++)
      for (int x = 0; x < 10; x++) {
        const auto &t = f.tiles[y * 10 + x];
        Pos p{x, y};
        if (t.kind == 0) {
          for (int it = 0; it < 5; it++)
            add(Choice{PLANT, p, it});
          add(Choice{COOP, p});
          add(Choice{PASTURE, p});
        } else if (t.kind > 0) {
          if (t.kind == 2) {
            for (int v : {WATER, HARVEST, FERTILIZE, DIG})
              add(Choice{v, p});
          } else if (t.item >= 9) {
            for (int v : {FEED, CARE, HARVEST, FERT})
              add(Choice{v, p});
          } else {
            add(Choice{DIG, p});
            if (t.kind == 3 || t.kind == 4)
              for (int it = 9; it < 12; it++)
                if (astructure[it - 9] == t.kind)
                  add(Choice{PLACE, p, it});
          }
        }
      }
    for (int it = 0; it < NI; it++)
      if (f.shed[it] > 0)
        out.push_back(Choice{PICKUP, nearest(f.positions[w]), it, f.shed[it]});
  }
  std::pair<Action, bool> primitive(Choice c, int w) {
    int v = c.verb, it = c.item;
    Pos pos = f.positions[w], target = c.target;
    if (v == WAIT || v == CANCEL)
      return {Action{}, true};
    if (v >= NORTH && v <= WEST)
      return {Action{v}, true};
    int req = (v == PLACE || v == PICKUP) ? it
              : v == FEED                 ? 0
              : v == FERTILIZE            ? 8
                                          : -1;
    if (v == PICKUP || (req >= 0 && f.invs[w].q[req] == 0)) {
      if (f.shed[req] <= 0)
        return {Action{}, true};
      if (!adjacent(pos))
        target = nearest(pos);
      else
        return {Action{PICKUP, req, std::min(c.q, f.shed[req])}, v == PICKUP};
    }
    if (pos != target) {
      int dx = target.x - pos.x, dy = target.y - pos.y;
      return {Action{dx > 0   ? EAST
                     : dx < 0 ? WEST
                     : dy > 0 ? SOUTH
                              : NORTH},
              false};
    }
    return {Action{v, it, c.q}, true};
  }
  void apply(Choice c, int w) {
    if (w < 0) {
      apply_market(c);
      return;
    }
    if (full_actions) {
      Action a{c.verb, c.item, c.q};
      unit(f, w, a, day);
      action.units.push_back(a);
      tasks->valid[w] = false;
      return;
    }
    if (c.verb == WAIT) {
      action.units.push_back(Action{});
      return;
    }
    if (c.verb == CONTINUE)
      c = tasks->tasks[w];
    auto [a, complete] = primitive(c, w);
    unit(f, w, a, day);
    if (!complete) {
      tasks->tasks[w] = c;
      tasks->valid[w] = true;
      reserved.push_back(c);
      if (c.verb == PLANT)
        seeds_reserved[c.item]++;
    } else
      tasks->valid[w] = false;
    action.units.push_back(a);
  }
  void market_start() {
    if (!started_market) {
      started_market = true;
      upper_cash = f.cash;
      guaranteed_cash = f.cash;
      upper_shed = f.shed;
    }
  }
  void market_menu(vector<Choice> &out) {
    market_start();
    out.push_back(Choice{STOP});
    double cash = full_actions ? guaranteed_cash : (uncertain ? upper_cash : f.cash);
    int room = (uncertain && !full_actions) ? 100 : std::max(0, 100 - f.stock());
    const auto &shed = (uncertain && !full_actions) ? upper_shed : f.shed;
    if (cash >= hirecost(f.hires) && (!full_actions || f.positions.size() < 17))
      out.push_back(Choice{HIRE});
    const int lc[3] = {1000, 2000, 4000};
    if (f.unlocked < 4 && cash >= lc[f.unlocked - 1])
      out.push_back(Choice{LAND});
    for (int it = 0; it < 9; it++)
      if (shed[it] > 0)
        out.push_back(Choice{SELL, {-1, -1}, it, shed[it]});
    for (int it = 0; it < 5; it++) {
      int n = std::min(99999., floor(cash / seedcost[it]));
      if (n > 0)
        out.push_back(Choice{SEED, {-1, -1}, it, n});
    }
    for (int it = 9; it < 12; it++) {
      int n = std::min(double(room), floor(cash / acost[it - 9]));
      if (n > 0)
        out.push_back(Choice{ANIMAL, {-1, -1}, it, n});
    }
    for (int it : {0, 8}) {
      int affordable=room;
      if(full_actions) {
        double remaining=cash; affordable=0;
        // The rival can buy at most a shed's capacity in each market slot.
        // Quote the worst attainable price over that public bound.
        for(int j=0;j<room;j++) {
          int price=quote(it,market[it]-100*int(action.market.size()+1)-j-1,params);
          if(remaining<price)break;remaining-=price;affordable++;
        }
      }
      if(cash>=1 && affordable>0) out.push_back(Choice{PRODUCT,{-1,-1},it,affordable});
    }
  }
  void apply_market(Choice c) {
    market_start();
    int v = c.verb, it = c.item, q = c.q;
    if (v == STOP)
      return;
    if(full_actions) {
      if(v==HIRE) guaranteed_cash-=hirecost(f.hires);
      else if(v==LAND) {const int costs[3]={1000,2000,4000};guaranteed_cash-=costs[f.unlocked-1];}
      else if(v==SEED) guaranteed_cash-=q*seedcost[it];
      else if(v==ANIMAL) guaranteed_cash-=q*acost[it-9];
      else if(v==SELL) guaranteed_cash+=q; // The official price floor is one.
      else if(v==PRODUCT) for(int j=0;j<q;j++)
        guaranteed_cash-=quote(it,market[it]-100*int(action.market.size()+1)-j-1,params);
      if(guaranteed_cash < -1e-6) throw std::runtime_error("joint cash lower bound violated");
    }
    action.market.push_back(Action{v, it, q});
    if (v == HIRE)
      hire(f);
    else if (v == LAND)
      land(f);
    else if (v == SELL) {
      upper_cash += q * double(quote(it, market[it] - 2000, params));
      upper_shed[it] = std::max(0, upper_shed[it] - q);
      for (int j = 0; j < q; j++)
        if (!trade(f, market, Action{v, it}, quote(it, market[it], params)))
          break;
      uncertain = true;
    } else if (v == SEED || v == ANIMAL) {
      int price = v == SEED ? seedcost[it] : acost[it - 9];
      int n = std::min(double(q), std::max(0., floor(f.cash / price)));
      if (v == ANIMAL)
        n = std::min(n, std::max(0, 100 - f.stock()));
      f.cash -= n * price;
      if (v == SEED)
        f.seeds[it] += n;
      else {
        f.shed[it] += n;
        upper_shed[it] = std::min(100, upper_shed[it] + q);
      }
    } else if (v == PRODUCT) {
      upper_shed[it] = std::min(100, upper_shed[it] + q);
      for (int j = 0; j < q; j++)
        if (!trade(f, market, Action{v, it}, quote(it, market[it] - 1, params)))
          break;
      uncertain = true;
    }
    if (!uncertain) {
      upper_cash = f.cash;
      upper_shed = f.shed;
    }
  }
  void features(const vector<Choice> &menu, int w, float *out) const {
    const float cash_feature = log1p(f.cash) / 15;
    const float stock_feature = f.stock() / 100.f;
    const float hour_feature = hour / 24.f;
    const float day_feature = day / 30.f;
    const float market_feature = action.market.size() / 10.f;
    const float units_feature = action.units.size() / 40.f;
    const Pos p = w >= 0 ? f.positions[w] : Pos{0, 0};
    for (auto c : menu) {
      std::fill(out, out + NF, 0);
      out[c.verb] = 1;
      Choice actual = c;
      if (c.verb == CONTINUE && w >= 0 && tasks->valid[w])
        actual = tasks->tasks[w];
      if (actual.item >= 0)
        out[NV + actual.item] = 1;
      float *z = out + NV + NI;
      z[0] = actual.target.x / 9.f;
      z[1] = actual.target.y / 9.f;
      z[2] = p.x / 9.f;
      z[3] = p.y / 9.f;
      z[4] = distance(p, actual.target) / 18.f;
      z[5] = log1p(actual.q) / 10;
      z[6] = cash_feature;
      z[7] = stock_feature;
      z[8] = hour_feature;
      z[9] = day_feature;
      z[10] = market_feature;
      z[11] = units_feature;
      z[12] = w >= 0 && tasks->valid[w] && tasks->tasks[w] == actual;
      if (w >= 0 && actual.target.x >= 0) {
        const auto &t = f.tiles[actual.target.y * 10 + actual.target.x];
        if (t.kind > 0) {
          z[13] = t.yield / 6.f;
          z[14] = t.watered;
          z[15] = t.fed;
          z[16] = t.kind == 2 ? (day - t.day) / 30.f : 0;
        }
      }
      z[17] = w < 0;
      z[18] = uncertain;
      z[19] = actual.item >= 0 && actual.item < 5 ? f.seeds[actual.item] / 100.f
                                                  : 0;
      out += NF;
    }
  }
};
void encode(const State &s, int seat, const Tasks &tasks, float *tiles,
            float *global) {
  std::fill(tiles, tiles + 4800, 0);
  std::fill(global, global + 128, 0);
  int day = s.step / 24;
  for (int side = 0; side < 2; side++) {
    const Farm &f = s.farms[side == 0 ? seat : 1 - seat];
    for (int k = 0; k < 100; k++) {
      const auto &t = f.tiles[k];
      auto put = [&](int c, float v) { tiles[(side * 24 + c) * 100 + k] = v; };
      put(0, t.kind == 0);
      put(1, t.kind == -1);
      if (t.kind > 0) {
        put(2, t.kind == 1);
        put(3, t.kind == 2);
        put(4, t.kind == 3);
        put(5, t.kind == 4);
        if (t.item >= 0)
          put(6 + t.item, 1);
        put(18, t.yield / 6.f);
        put(19, (t.kind == 2 || t.item >= 9) ? (day - t.day) / 30.f : 0);
        put(20, t.watered);
        put(21, t.fed);
        put(22, t.fertilizer);
        put(23, t.cared);
      }
    }
    for (auto p : f.positions)
      tiles[(side * 24 + 23) * 100 + p.y * 10 + p.x] += .1f;
  }
  int k = 0;
  global[k++] = s.step / 719.f;
  global[k++] = day / 30.f;
  global[k++] = (s.step % 24) / 24.f;
  for (int who : {seat, 1 - seat}) {
    const auto &f = s.farms[who];
    global[k++] = log1p(f.cash) / 15;
    global[k++] = (f.positions.size() - 1) / 40.f;
    global[k++] = f.unlocked / 4.f;
    global[k++] = f.hires / 40.f;
  }
  const auto &f = s.farms[seat];
  for (int it = 0; it < NI; it++) {
    global[k++] = f.shed[it] / 100.f;
    int n = 0;
    for (const auto &i : f.invs)
      n += i.q[it];
    global[k++] = n / 100.f;
  }
  for (int n : f.seeds)
    global[k++] = n / 100.f;
  for (int it = 0; it < 9; it++) {
    global[k++] = log1p(quote(it, s.market[it], s.params)) / 10;
    global[k++] = (s.market[it] - 10000) / 1000.f;
  }
  for (int sh = 0; sh < 8; sh++)
    global[k++] = std::count(s.town.begin(), s.town.end(), sh) / 8.f;
  for (int v = 0; v < NV; v++) {
    int n = 0;
    for (int w = 0; w < int(tasks.valid.size()); w++)
      if (tasks.valid[w] && tasks.tasks[w].verb == v)
        n++;
    global[k++] = n / 40.f;
  }
}
// Reusable threads; no per-step process creation and no Python callbacks in
// jobs.
class Pool {
  vector<std::thread> threads;
  std::mutex mutex;
  std::condition_variable ready, finished;
  std::function<void(int)> fn;
  std::atomic<int> next{0};
  int count = 0, pending = 0;
  size_t generation = 0;
  bool quit = false;

public:
  explicit Pool(int n) {
    for (int i = 0; i < std::max(1, n); i++)
      threads.emplace_back([this] {
        size_t seen = 0;
        for (;;) {
          std::unique_lock lock(mutex);
          ready.wait(lock, [&] { return quit || generation != seen; });
          if (quit)
            return;
          seen = generation;
          lock.unlock();
          for (int k; (k = next.fetch_add(1)) < count;)
            fn(k);
          lock.lock();
          if (--pending == 0)
            finished.notify_one();
        }
      });
  }
  void run(int n, std::function<void(int)> f) {
    std::unique_lock lock(mutex);
    fn = std::move(f);
    count = n;
    next = 0;
    pending = threads.size();
    generation++;
    ready.notify_all();
    finished.wait(lock, [&] { return pending == 0; });
  }
  ~Pool() {
    {
      std::lock_guard lock(mutex);
      quit = true;
    }
    ready.notify_all();
    for (auto &t : threads)
      t.join();
  }
};
py::dict invdict(const Inv &i) {
  py::dict d;
  for (int it : i.order)
    d[py::str(items[it])] = i.q[it];
  return d;
}
py::object tiledict(const Tile &t) {
  if (t.kind == 0)
    return py::none();
  if (t.kind == -1)
    return py::str("LOCKED");
  py::dict d;
  d["kind"] = t.kind == 1   ? "WEED"
              : t.kind == 2 ? "PLANT"
              : t.kind == 3 ? "COOP"
                            : "PASTURE";
  if (t.kind == 2) {
    d["crop"] = items[t.item];
    d["planted_day"] = t.day;
    d["watered_today"] = t.watered;
    d["consecutive_unwatered"] = t.unwatered;
    d["yield_units"] = t.yield;
    d["max_lifespan_step"] = t.life;
    d["fertilized_until_day"] = t.fertuntil;
  }
  if (t.item >= 9) {
    d["animal"] = items[t.item];
    d["placed_day"] = t.day;
    d["yield_units"] = t.yield;
    d["consecutive_unfed"] = t.unfed;
    d["fed_today"] = t.fed;
    d["cared_today"] = t.cared;
    d["fertilizer_available"] = t.fertilizer;
    d["pending_care_bonus"] = t.bonus;
  }
  return d;
}
py::dict farmdict(const Farm &f) {
  py::dict d;
  d["money"] = f.cash;
  py::list rows;
  for (int y = 0; y < 10; y++) {
    py::list row;
    for (int x = 0; x < 10; x++)
      row.append(tiledict(f.tiles[y * 10 + x]));
    rows.append(row);
  }
  d["tiles"] = rows;
  d["farmer"] = py::cast(vector<int>{f.positions[0].x, f.positions[0].y});
  py::list hands;
  for (int i = 1; i < int(f.positions.size()); i++)
    hands.append(vector<int>{f.positions[i].x, f.positions[i].y});
  d["hands"] = hands;
  py::list quadrants;
  const char *q[] = {"NW", "NE", "SW", "SE"};
  for (int i = 0; i < f.unlocked; i++)
    quadrants.append(q[i]);
  d["unlocked_quadrants"] = quadrants;
  d["hires_today"] = f.hires;
  return d;
}
py::dict privatedict(const Farm &f) {
  py::dict d, shed, seeds;
  py::list inv;
  for (int it = 0; it < NI; it++)
    shed[py::str(items[it])] = f.shed[it];
  for (int it = 0; it < 5; it++)
    seeds[py::str(items[it])] = f.seeds[it];
  for (auto &i : f.invs)
    inv.append(invdict(i));
  d["shed"] = shed;
  d["seeds"] = seeds;
  d["inventories"] = inv;
  return d;
}
py::dict observation(const State &s, int seat) {
  py::dict d;
  d["player"] = seat;
  d["step"] = s.step;
  d["day"] = s.step / 24;
  d["hour"] = s.step % 24;
  py::list farms;
  for (auto &f : s.farms)
    farms.append(farmdict(f));
  d["farms"] = farms;
  d["private"] = privatedict(s.farms[seat]);
  py::dict m, inv, prices;
  for (int it = 0; it < 9; it++) {
    inv[py::str(items[it])] = s.market[it];
    prices[py::str(items[it])] = quote(it, s.market[it], s.params);
  }
  m["inventory"] = inv;
  m["prices"] = prices;
  d["market"] = m;
  py::dict town;
  py::list sh;
  for (int i : s.town)
    sh.append(shops[i]);
  town["unlocked_shops"] = sh;
  d["town"] = town;
  return d;
}
int getint(py::dict d, const char *k, int fallback = 0) {
  return d.contains(k) ? py::cast<int>(d[k]) : fallback;
}
Tile readtile(py::handle o) {
  Tile t;
  if (o.is_none())
    return t;
  if (py::isinstance<py::str>(o)) {
    t.kind = -1;
    return t;
  }
  auto d = py::cast<py::dict>(o);
  string k = py::cast<string>(d["kind"]);
  t.kind = k == "WEED" ? 1 : k == "PLANT" ? 2 : k == "COOP" ? 3 : 4;
  if (d.contains("crop"))
    t.item = itemid(py::cast<string>(d["crop"]));
  if (d.contains("animal"))
    t.item = itemid(py::cast<string>(d["animal"]));
  t.day = getint(d, t.kind == 2 ? "planted_day" : "placed_day");
  t.yield = getint(d, "yield_units");
  t.life = getint(d, "max_lifespan_step", -1);
  t.fertuntil = getint(d, "fertilized_until_day", -1);
  t.unwatered = getint(d, "consecutive_unwatered");
  t.unfed = getint(d, "consecutive_unfed");
  t.bonus = getint(d, "pending_care_bonus");
  t.watered = getint(d, "watered_today");
  t.fed = getint(d, "fed_today");
  t.cared = getint(d, "cared_today");
  t.fertilizer = getint(d, "fertilizer_available");
  return t;
}
void loadstate(State &s, py::list observations) {
  if (py::len(observations) != 2)
    throw std::runtime_error("two observations required for simulator restore");
  auto o0 = py::cast<py::dict>(observations[0]);
  s.step = getint(o0, "step");
  auto farms = py::cast<py::list>(o0["farms"]);
  for (int seat = 0; seat < 2; seat++) {
    Farm f;
    auto d = py::cast<py::dict>(farms[seat]);
    f.cash = py::cast<double>(d["money"]);
    f.hires = getint(d, "hires_today");
    f.unlocked = py::len(d["unlocked_quadrants"]);
    auto rows = py::cast<py::list>(d["tiles"]);
    if (py::len(rows) != 10)
      throw std::runtime_error("only competition 10x10 board is supported");
    for (int y = 0; y < 10; y++) {
      auto row = py::cast<py::list>(rows[y]);
      for (int x = 0; x < 10; x++)
        f.tiles[y * 10 + x] = readtile(row[x]);
    }
    f.positions.clear();
    auto p = py::cast<vector<int>>(d["farmer"]);
    f.positions.push_back(Pos{p[0], p[1]});
    for (auto h : py::cast<py::list>(d["hands"])) {
      auto p = py::cast<vector<int>>(h);
      f.positions.push_back(Pos{p[0], p[1]});
    }
    auto priv =
        py::cast<py::dict>(py::cast<py::dict>(observations[seat])["private"]);
    for (auto kv : py::cast<py::dict>(priv["shed"])) {
      int i = itemid(py::cast<string>(kv.first));
      if (i >= 0)
        f.shed[i] = py::cast<int>(kv.second);
    }
    for (auto kv : py::cast<py::dict>(priv["seeds"])) {
      int i = itemid(py::cast<string>(kv.first));
      if (i >= 0 && i < 5)
        f.seeds[i] = py::cast<int>(kv.second);
    }
    f.invs.clear();
    for (auto inv : py::cast<py::list>(priv["inventories"])) {
      Inv i;
      for (auto kv : py::cast<py::dict>(inv))
        i.add(itemid(py::cast<string>(kv.first)), py::cast<int>(kv.second));
      f.invs.push_back(i);
    }
    s.farms[seat] = f;
  }
  auto m = py::cast<py::dict>(o0["market"]);
  if (m.contains("params"))
    throw std::runtime_error(
        "custom market overrides are not supported by v2 competition engine");
  auto inv = py::cast<py::dict>(m["inventory"]);
  for (int i = 0; i < 9; i++)
    s.market[i] = py::cast<int>(inv[py::str(items[i])]);
  s.town.clear();
  for (auto sh :
       py::cast<py::list>(py::cast<py::dict>(o0["town"])["unlocked_shops"])) {
    string x = py::cast<string>(sh);
    auto p = std::find(shops.begin(), shops.end(), x);
    if (p == shops.end())
      throw std::runtime_error("unknown shop");
    s.town.push_back(p - shops.begin());
  }
}
Action readaction(py::handle h) {
  if (!py::isinstance<py::list>(h) || py::len(h) == 0)
    return Action{};
  auto a = py::cast<py::list>(h);
  Action r;
  r.verb = verbid(py::cast<string>(a[0]));
  if (py::len(a) > 1 && py::isinstance<py::str>(a[1]))
    r.item = itemid(py::cast<string>(a[1]));
  if (py::len(a) > 2)
    r.q = py::cast<int>(a[2]);
  return r;
}
Turn readturn(py::dict d) {
  Turn t;
  t.units.push_back(d.contains("farmer") ? readaction(d["farmer"]) : Action{});
  if (d.contains("hands"))
    for (auto a : py::cast<py::list>(d["hands"]))
      t.units.push_back(readaction(a));
  if (d.contains("market"))
    for (auto a : py::cast<py::list>(d["market"]))
      t.market.push_back(readaction(a));
  return t;
}
py::list actionlist(Action a) {
  py::list l;
  l.append(a.verb == WAIT ? "PASS" : verbs[a.verb]);
  if (a.item >= 0) {
    l.append(items[a.item]);
    l.append(a.q);
  }
  return l;
}
py::dict turndict(const Turn &t) {
  py::dict d;
  d["farmer"] = t.units.empty() ? actionlist(Action{}) : actionlist(t.units[0]);
  py::list h, m;
  for (size_t i = 1; i < t.units.size(); i++)
    h.append(actionlist(t.units[i]));
  for (auto a : t.market)
    m.append(actionlist(a));
  d["hands"] = h;
  d["market"] = m;
  return d;
}
// Scripted curricula use the same legal task decoder as the learner. IDs 2..6
// vary crops, developed area and daily labor, rather than altering game rules.
int scripted(const Ledger &l, const vector<Choice> &menu, int worker,
             int level) {
  if (level == 1)
    return 0;
  int crop = level == 2   ? 1
             : level == 3 ? 0
             : level == 4 ? 1
             : level == 6 ? 0
                          : 2,
      area = level == 2   ? 4
             : level == 3 ? 8
             : level == 4 ? 12
             : level == 8 ? 50
                          : 25;
  if (worker < 0) {
    for (int i = 0; i < int(menu.size()); i++)
      if (menu[i].verb == SELL)
        return i;
    int hands = level == 2   ? 0
                : level == 3 ? 1
                : level == 4 ? 3
                : level == 5 ? 6
                : level == 6 ? 8
                             : 12;
    if (level >= 3 && int(l.f.positions.size()) < hands + 1 &&
        l.f.cash > 300 + hirecost(l.f.hires) && l.day < 28)
      for (int i = 0; i < int(menu.size()); i++)
        if (menu[i].verb == HIRE)
          return i;
    if (level == 8 && l.f.unlocked < 2 && l.f.cash > 2200 && l.day < 20)
      for (int i = 0; i < int(menu.size()); i++)
        if (menu[i].verb == LAND)
          return i;
    if (level == 6 && l.day < 20 && l.f.cash > 600 && l.f.shed[9] < 1)
      for (int i = 0; i < int(menu.size()); i++)
        if (menu[i].verb == ANIMAL && menu[i].item == 9)
          return i;
    if (level == 6 && l.f.shed[0] < 5 && l.f.cash > 200)
      for (int i = 0; i < int(menu.size()); i++)
        if (menu[i].verb == PRODUCT && menu[i].item == 0)
          return i;
    for (int wanted : {crop, (level == 5 || level == 8) ? 1 : crop})
      if (l.day + firstday[wanted] < 29 && l.f.seeds[wanted] < 2)
        for (int i = 0; i < int(menu.size()); i++)
          if (menu[i].verb == SEED && menu[i].item == wanted)
            return i;
    return 0;
  }
  if (menu.size() > 1 && menu[1].verb == CONTINUE)
    return 1;
  if (l.f.invs[worker].sum() > 0) {
    for (int i = 0; i < int(menu.size()); i++)
      if (menu[i].verb == DROP)
        return i;
  }
  int best = 0;
  double score = -1e9;
  for (int i = 0; i < int(menu.size()); i++) {
    auto c = menu[i];
    if (c.target.x < 0)
      continue;
    const auto &t = l.f.tiles[c.target.y * 10 + c.target.x];
    int rank = c.target.y * (level == 8 ? 10 : 5) + c.target.x;
    if (c.target.x >= (level == 8 ? 10 : 5) || c.target.y >= 5 || rank >= area)
      continue;
    double v = -1e6;
    if (c.verb == HARVEST &&
        (t.item >= 9 || interval[t.item] || t.yield >= maxyield[t.item] ||
         l.day - t.day >= maxday[t.item] || l.day >= 28))
      v = 120;
    if (c.verb == WATER)
      v = 100;
    if (c.verb == FEED)
      v = 115;
    if (c.verb == CARE)
      v = 80;
    if (c.verb == PLANT &&
        c.item == ((level == 5 || level == 8) ? (rank % 2 ? 1 : 2) : crop) &&
        l.day + firstday[c.item] < 29)
      v = 60;
    if (c.verb == DIG && t.kind == 1)
      v = 50;
    if (level == 6) {
      if (c.verb == COOP && rank % 3 == 0)
        v = 65;
      if (c.verb == PLACE && c.item == 9)
        v = 90;
    }
    v -= distance(l.f.positions[worker], c.target);
    if (v > score && v > -1000) {
      score = v;
      best = i;
    }
  }
  return best;
}
struct Arena {
  Tasks tasks;
  Ledger ledger;
  vector<Choice> menu;
  int worker = 0, markets = 0, script = 0;
  bool finished = false;
};
Turn plan_action(const Farm&,const array<int,9>&,const array<Price,9>&,int,const vector<vector<int>>&);
void check_plan(const vector<vector<int>>&);
class Batch {
  Pool pool;
  vector<State> states;
  vector<Arena> arenas;
  vector<int> active;
  vector<array<Turn, 2>> overrides;
  vector<array<bool, 2>> has_override;

public:
  explicit Batch(vector<uint64_t> seeds, int threads = 8) : pool(threads) {
    for (auto seed : seeds)
      states.emplace_back(seed);
    arenas.resize(seeds.size() * 2);
    overrides.resize(seeds.size());
    has_override.resize(seeds.size());
  }
  int size() const { return states.size(); }
  void planned(int env,int seat,const vector<vector<int>>& p) {
    check_plan(p);const auto& s=states.at(env);
    overrides.at(env)[seat]=plan_action(s.farms.at(seat),s.market,s.params,s.step,p);
    has_override.at(env)[seat]=true;
  }
  void action_modes(const vector<int> &modes) {
    if (modes.size() != arenas.size()) throw std::runtime_error("action mode shape");
    for (size_t k = 0; k < modes.size(); ++k) arenas[k].ledger.full_actions = modes[k] != 0;
  }
  void reset(int env, uint64_t seed) {
    states.at(env) = State(seed);
    arenas[2 * env] = Arena{};
    arenas[2 * env + 1] = Arena{};
    has_override[env] = {false, false};
  }
  void scripts(vector<int> ids) {
    if (ids.size() != arenas.size())
      throw std::runtime_error("script id shape");
    for (size_t i = 0; i < ids.size(); i++)
      arenas[i].script = ids[i];
  }
  void encode_only(py::array_t<float, py::array::c_style> tiles,
                   py::array_t<float, py::array::c_style> global) {
    if (tiles.size() != int(arenas.size()) * 4800 ||
        global.size() != int(arenas.size()) * 128)
      throw std::runtime_error("encode shape");
    float *t = tiles.mutable_data(), *g = global.mutable_data();
    py::gil_scoped_release release;
    pool.run(arenas.size(), [&](int k) {
      encode(states[k / 2], k % 2, arenas[k].tasks, t + k * 4800, g + k * 128);
    });
  }
  void begin(py::array_t<float, py::array::c_style> tiles,
             py::array_t<float, py::array::c_style> global) {
    if (tiles.size() != int(arenas.size()) * 4800 ||
        global.size() != int(arenas.size()) * 128)
      throw std::runtime_error("encode shape");
    float *t = tiles.mutable_data(), *g = global.mutable_data();
    py::gil_scoped_release release;
    pool.run(arenas.size(), [&](int k) {
      auto &a = arenas[k];
      encode(states[k / 2], k % 2, a.tasks, t + k * 4800, g + k * 128);
      a.ledger.begin(states[k / 2], k % 2, a.tasks);
      a.worker = 0;
      a.markets = 0;
      a.finished = has_override[k / 2][k % 2];
    });
  }
  void prepare(int k) {
    auto &a = arenas[k];
    while (!a.finished) {
      int w = a.worker < int(a.ledger.f.positions.size()) && a.markets == 0
                  ? a.worker
                  : -1;
      // Unit count is the pre-market worker count: hires act on the following
      // turn.
      if (a.worker >= int(states[k / 2].farms[k % 2].positions.size()))
        w = -1;
      if (w < 0 && a.markets >= 10) {
        a.finished = true;
        break;
      }
      a.ledger.menu(w, a.menu);
      if (!a.script)
        break;
      int ix = scripted(a.ledger, a.menu, w, a.script);
      auto c = a.menu[ix];
      if (w < 0 && c.verb == SEED)
        c.q = std::min(c.q, 2);
      if (w < 0 && c.verb == ANIMAL)
        c.q = 1;
      if (w < 0 && c.verb == PRODUCT)
        c.q = std::min(c.q, 5);
      a.ledger.apply(c, w);
      if (w >= 0)
        a.worker++;
      else {
        a.markets++;
        if (c.verb == STOP)
          a.finished = true;
      }
    }
  }
  py::tuple next(py::array_t<float, py::array::c_style> features,
                 py::array_t<int64_t, py::array::c_style> owners,
                 py::array_t<int64_t, py::array::c_style> bounds,
                 py::array_t<int64_t, py::array::c_style> offsets,
                 py::array_t<int64_t, py::array::c_style> ids) {
    {
      py::gil_scoped_release release;
      pool.run(arenas.size(), [&](int k) { prepare(k); });
    }
    active.clear();
    int rows = 0;
    for (int k = 0; k < int(arenas.size()); k++)
      if (!arenas[k].finished) {
        active.push_back(k);
        rows += arenas[k].menu.size();
      }
    if (features.size() < rows * NF || owners.size() < rows ||
        bounds.size() < rows || offsets.size() < int(active.size() + 1) ||
        ids.size() < int(active.size()))
      throw std::runtime_error("menu buffer capacity exceeded");
    int off = 0;
    auto *own = owners.mutable_data();
    auto *bd = bounds.mutable_data();
    auto *os = offsets.mutable_data();
    auto *is = ids.mutable_data();
    float *f = features.mutable_data();
    for (int i = 0; i < int(active.size()); i++) {
      is[i] = active[i];
      os[i] = off;
      off += arenas[active[i]].menu.size();
    }
    os[active.size()] = off;
    auto emit = [&](int i) {
      int k = active[i];
      auto &a = arenas[k];
      int row = os[i];
      int w = a.worker < int(states[k / 2].farms[k % 2].positions.size())
                  ? a.worker : -1;
      a.ledger.features(a.menu, w, f + row * NF);
      for (auto c : a.menu) {
        own[row] = i;
        bd[row] = (w < 0 || c.verb == PICKUP || c.verb == PLACE)
                      ? std::max(1, c.q) : 1;
        row++;
      }
    };
    {
      py::gil_scoped_release release;
      if (rows >= 4096)
        pool.run(active.size(), emit);
      else
        for (int i = 0; i < int(active.size()); i++) emit(i);
    }
    return py::make_tuple(active.size(), rows);
  }
  void apply(py::array_t<int64_t, py::array::c_style> selected,
             py::array_t<int64_t, py::array::c_style> quantity) {
    if (selected.size() < int(active.size()) ||
        quantity.size() < int(active.size()))
      throw std::runtime_error("selection shape");
    for (int i = 0; i < int(active.size()); i++) {
      auto &a = arenas[active[i]];
      if (selected.data()[i] < 0 ||
          selected.data()[i] >= int64_t(a.menu.size()))
        throw std::runtime_error("selection out of support");
      auto c = a.menu[selected.data()[i]];
      if (quantity.data()[i] < 1 || quantity.data()[i] > std::max(1, c.q))
        throw std::runtime_error("quantity out of support");
    }
    auto *sel = selected.data();
    auto *q = quantity.data();
    py::gil_scoped_release release;
    pool.run(active.size(), [&](int i) {
      int k = active[i];
      auto &a = arenas[k];
      auto c = a.menu[sel[i]];
      c.q = q[i];
      int w = a.worker < int(states[k / 2].farms[k % 2].positions.size())
                  ? a.worker
                  : -1;
      a.ledger.apply(c, w);
      if (w >= 0)
        a.worker++;
      else {
        a.markets++;
        if (c.verb == STOP)
          a.finished = true;
      }
    });
  }
  py::array_t<double> step() {
    for (auto &a : arenas)
      if (!a.finished)
        throw std::runtime_error("incomplete conditional action");
    py::array_t<double> result({size(), 5});
    auto *r = result.mutable_data();
    {
      py::gil_scoped_release release;
      pool.run(size(), [&](int k) {
        array<Turn, 2> a;
        for (int s = 0; s < 2; s++)
          a[s] = has_override[k][s] ? overrides[k][s]
                                    : arenas[2 * k + s].ledger.action;
        for (int s = 0; s < 2; s++)
          arenas[2 * k + s].ledger.action = a[s];
        states[k].advance(a);
        r[k * 5] = states[k].farms[0].cash;
        r[k * 5 + 1] = states[k].farms[1].cash;
        r[k * 5 + 2] = states[k].done();
        r[k * 5 + 3] = states[k].step;
        r[k * 5 + 4] = states[k].seed;
        has_override[k] = {false, false};
      });
    }
    return result;
  }
  py::dict obs(int env, int seat) { return observation(states.at(env), seat); }
  void restore(int env, uint64_t seed, py::list obs) {
    reset(env, seed);
    loadstate(states[env], obs);
  }
  void override_action(int env, int seat, py::dict a) {
    overrides.at(env)[seat] = readturn(a);
    has_override.at(env)[seat] = true;
  }
  void primitive_step(int env, py::list actions) {
    array<Turn, 2> a = {readturn(py::cast<py::dict>(actions[0])),
                        readturn(py::cast<py::dict>(actions[1]))};
    states.at(env).advance(a);
  }
  py::dict action(int env, int seat) {
    return turndict(arenas.at(2 * env + seat).ledger.action);
  }
  py::list task_state(int env) {
    py::list out;
    for (int s = 0; s < 2; s++) {
      const auto &t = arenas.at(env * 2 + s).tasks;
      py::dict d;
      d["day"] = t.day;
      py::list rows;
      for (size_t i = 0; i < t.tasks.size(); i++) {
        const auto &c = t.tasks[i];
        rows.append(vector<int>{int(t.valid[i]), c.verb, c.target.x, c.target.y,
                                c.item, c.q});
      }
      d["tasks"] = rows;
      out.append(d);
    }
    return out;
  }
  void restore_tasks(int env, py::list saved) {
    for (int s = 0; s < 2; s++) {
      auto d = py::cast<py::dict>(saved[s]);
      auto &t = arenas.at(env * 2 + s).tasks;
      t.day = getint(d, "day", -1);
      t.tasks.clear();
      t.valid.clear();
      for (auto r : py::cast<py::list>(d["tasks"])) {
        auto v = py::cast<vector<int>>(r);
        if (v.size() != 6)
          throw std::runtime_error("invalid task snapshot");
        t.valid.push_back(v[0]);
        t.tasks.push_back(Choice{v[1], {v[2], v[3]}, v[4], v[5]});
      }
    }
  }
  py::array_t<int64_t> suggestions(int level) {
    py::array_t<int64_t> out({int(active.size()), 2});
    auto *p = out.mutable_data();
    for (int i = 0; i < int(active.size()); i++) {
      int k = active[i];
      auto &a = arenas[k];
      int w = a.worker < int(states[k / 2].farms[k % 2].positions.size())
                  ? a.worker
                  : -1;
      int ix = scripted(a.ledger, a.menu, w, level);
      auto c = a.menu[ix];
      int q = c.q;
      if (w < 0 && c.verb == SEED)
        q = std::min(q, 2);
      if (w < 0 && c.verb == ANIMAL)
        q = 1;
      if (w < 0 && c.verb == PRODUCT)
        q = std::min(q, 5);
      p[i * 2] = ix;
      p[i * 2 + 1] = q;
    }
    return out;
  }
};
#include "actor.hpp"
#include "planner.hpp"
PYBIND11_MODULE(native, m) {
  py::class_<Actor>(m,"Actor").def(py::init<py::dict>()).def("encode",&Actor::encode_actor)
      .def("menu",&Actor::menu).def("apply",&Actor::apply).def("action",&Actor::action);
  m.def("plan_action",&act_plan);
  m.def("plan_games",&plan_games);
  m.attr("engine_version") = "kaggle-environments-1.32.7-default-config";
  m.attr("feature_dim") = NF;
  py::class_<Batch>(m, "Batch")
      .def(py::init<vector<uint64_t>, int>(), py::arg("seeds"),
           py::arg("threads") = 8)
      .def("reset", &Batch::reset)
      .def("planned", &Batch::planned)
      .def("scripts", &Batch::scripts)
      .def("action_modes", &Batch::action_modes)
      .def("begin", &Batch::begin)
      .def("encode", &Batch::encode_only)
      .def("next", &Batch::next)
      .def("apply", &Batch::apply)
      .def("step", &Batch::step)
      .def("observation", &Batch::obs)
      .def("restore", &Batch::restore)
      .def("override_action", &Batch::override_action)
      .def("primitive_step", &Batch::primitive_step)
      .def("action", &Batch::action)
      .def("task_state", &Batch::task_state)
      .def("restore_tasks", &Batch::restore_tasks)
      .def("suggestions", &Batch::suggestions);
  m.def("random_sequence", [](uint64_t seed, int n) {
    RNG r(seed);
    vector<double> v;
    for (int i = 0; i < n; i++)
      v.push_back(r.random());
    return v;
  });
  m.def("quote", [](int i, int inventory) { return quote(i, inventory); });
}
