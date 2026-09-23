#include "fieldcraft_2887.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fieldcraft_2887 {
namespace {

using fastkag::Action;
using fastkag::Farm;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::PrivateState;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;

constexpr int kFrames = 719;
constexpr int kHalf = 5;
constexpr int kRoutePairs = 64;
constexpr int kMaxOrders = 10;
constexpr int kShedCapacity = 100;
constexpr std::array<Position, 4> kAccess{{{4, 4}, {5, 4}, {4, 5}, {5, 5}}};
constexpr std::array<int, 7> kHarvested{{
    int(Item::STRAWBERRY), int(Item::MILK), int(Item::WOOL),
    int(Item::EGG), int(Item::MELON), int(Item::CARROT), int(Item::TOMATO)}};
constexpr std::array<int, 9> kProducts{{0, 1, 2, 3, 4, 5, 6, 7, 8}};

class Reader {
 public:
  explicit Reader(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open Fieldcraft asset");
    bytes_ = {std::istreambuf_iterator<char>(stream), {}};
  }

  template <class T>
  T read() {
    if (offset_ + sizeof(T) > bytes_.size())
      throw std::runtime_error("truncated Fieldcraft asset");
    T value{};
    std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  void expect(const char* value, std::size_t size) {
    if (offset_ + size > bytes_.size() ||
        std::memcmp(bytes_.data() + offset_, value, size) != 0)
      throw std::runtime_error("invalid Fieldcraft asset magic");
    offset_ += size;
  }

  void skip(std::size_t size) {
    if (offset_ + size > bytes_.size())
      throw std::runtime_error("truncated Fieldcraft asset header");
    offset_ += size;
  }

  bool done() const { return offset_ == bytes_.size(); }

 private:
  std::vector<char> bytes_;
  std::size_t offset_{};
};

Action read_action(Reader& reader) {
  const int op = reader.read<std::int8_t>();
  const int item = reader.read<std::int8_t>();
  const int quantity = reader.read<std::int32_t>();
  if (op < int(Op::PASS) || op > int(Op::SELL) || item < -1 ||
      item >= fastkag::N_ITEMS)
    throw std::runtime_error("invalid action in Fieldcraft asset");
  return {static_cast<Op>(op), static_cast<Item>(item), quantity};
}

PlayerAction read_player_action(Reader& reader) {
  const int units = reader.read<std::uint8_t>();
  const int market = reader.read<std::uint8_t>();
  if (units < 1 || market > kMaxOrders)
    throw std::runtime_error("invalid Fieldcraft action width");
  PlayerAction result;
  result.units.reserve(units);
  result.market.reserve(market);
  for (int i = 0; i < units; ++i) result.units.push_back(read_action(reader));
  for (int i = 0; i < market; ++i) result.market.push_back(read_action(reader));
  return result;
}

struct TapeLibrary {
  std::vector<PlayerAction> base;
  std::vector<int> route_ids;
  std::vector<std::vector<PlayerAction>> routes;
  std::array<std::int8_t, kRoutePairs> pair_routes{};
};

TapeLibrary load_library(const std::string& path) {
  Reader reader(path);
  reader.expect("FC2887\0\0", 8);
  if (reader.read<std::uint32_t>() != 1)
    throw std::runtime_error("unsupported Fieldcraft asset version");
  reader.skip(64);  // pinned main.py and mirror_plan.py SHA-256 values
  const int frames = reader.read<std::uint16_t>();
  const int route_count = reader.read<std::uint16_t>();
  if (frames != kFrames || route_count < 1 || route_count > 64)
    throw std::runtime_error("invalid Fieldcraft route dimensions");
  TapeLibrary library;
  library.route_ids.reserve(route_count);
  for (int i = 0; i < route_count; ++i)
    library.route_ids.push_back(reader.read<std::int16_t>());
  for (auto& route : library.pair_routes) {
    route = reader.read<std::int8_t>();
    if (route < -1 || route >= route_count)
      throw std::runtime_error("invalid Fieldcraft shop-pair route");
  }
  library.base.reserve(frames);
  for (int i = 0; i < frames; ++i)
    library.base.push_back(read_player_action(reader));
  library.routes.resize(route_count);
  for (auto& route : library.routes) {
    route.reserve(frames);
    for (int i = 0; i < frames; ++i)
      route.push_back(read_player_action(reader));
  }
  if (!reader.done()) throw std::runtime_error("trailing Fieldcraft asset data");
  return library;
}

std::shared_ptr<const TapeLibrary> shared_library(const std::string& path) {
  static std::mutex mutex;
  static std::unordered_map<
      std::string, std::shared_ptr<const TapeLibrary>> cache;
  const std::lock_guard lock(mutex);
  auto& value = cache[path];
  if (!value)
    value = std::make_shared<const TapeLibrary>(load_library(path));
  return value;
}

int count(const Action& action) { return std::max(0, action.quantity); }

bool same_position(Position left, Position right) {
  return left.x == right.x && left.y == right.y;
}

bool position_less(Position left, Position right) {
  return std::tie(left.x, left.y) < std::tie(right.x, right.y);
}

int distance(Position left, Position right) {
  return std::abs(int(left.x) - int(right.x)) +
         std::abs(int(left.y) - int(right.y));
}

bool access(Position position) {
  return std::any_of(kAccess.begin(), kAccess.end(), [&](Position candidate) {
    return same_position(position, candidate);
  });
}

Position nearest_access(Position position) {
  return *std::min_element(kAccess.begin(), kAccess.end(),
      [&](Position left, Position right) {
        return distance(position, left) < distance(position, right);
      });
}

Action toward(Position from, Position to) {
  if (to.x != from.x) return {to.x > from.x ? Op::EAST : Op::WEST};
  if (to.y != from.y) return {to.y > from.y ? Op::SOUTH : Op::NORTH};
  return {};
}

Position moved(Position position, Op op) {
  if (op == Op::NORTH) --position.y;
  if (op == Op::SOUTH) ++position.y;
  if (op == Op::EAST) ++position.x;
  if (op == Op::WEST) --position.x;
  position.x = std::clamp<int>(position.x, 0, 9);
  position.y = std::clamp<int>(position.y, 0, 9);
  return position;
}

bool movement(Op op) {
  return op == Op::NORTH || op == Op::SOUTH ||
         op == Op::EAST || op == Op::WEST;
}

int shed_sum(const std::array<int32_t, fastkag::N_ITEMS>& values) {
  int total = 0;
  for (int value : values) total += std::max(0, value);
  return total;
}

int inventory_sum(const std::array<int32_t, fastkag::N_ITEMS>& values) {
  int total = 0;
  for (int value : values) total += std::max(0, value);
  return total;
}

std::vector<Position> unit_positions(const Farm& farm) {
  std::vector<Position> result{farm.farmer};
  result.insert(result.end(), farm.hands.begin(), farm.hands.end());
  return result;
}

const Tile* tile_at(const Farm& farm, Position position, int board_size = 10) {
  if (position.x < 0 || position.y < 0 ||
      position.x >= board_size || position.y >= board_size)
    return nullptr;
  return &farm.tiles[position.y * board_size + position.x];
}

bool unlocked(const Farm& farm, int quadrant) {
  return (farm.unlocked_mask & (1u << quadrant)) != 0;
}

int unlocked_count(const Farm& farm) {
  return __builtin_popcount(unsigned(farm.unlocked_mask));
}

int shop_count(const Simulator& env, int shop, int prefix) {
  int total = 0;
  for (int i = 0; i < std::min<int>(prefix, env.shops().size()); ++i)
    total += env.shops()[i] == shop;
  return total;
}

bool milk_shop(int shop) { return shop == 5 || shop == 3 || shop == 6; }
bool tomato_shop(int shop) { return shop == 5 || shop == 2; }

double shape(int kind, double x, double threshold) {
  x = std::max(0.0, x);
  if (kind == 0) return x;                    // linear
  if (kind == 1) return x * x;                // square
  if (kind == 2) return std::sqrt(x);         // sqrt
  if (kind == 3) return std::log1p(x);        // log
  const double u = x / threshold;             // hinge
  return u + 8.0 * std::pow(std::max(0.0, u - 1.0), 2);
}

int market_price(int item, int inventory) {
  struct Parameters { int base, threshold, below, above; double bt, at; };
  static constexpr std::array<Parameters, fastkag::N_PRODUCTS> parameters{{
      {25, 400, 2, 3, .8, .2}, {35, 450, 4, 2, 1., .7},
      {60, 200, 4, 2, .4, .6}, {120, 100, 2, 0, .7, 1.6},
      {250, 300, 3, 1, .2, 3.6}, {50, 332, 4, 3, .4, .2},
      {160, 122, 2, 0, .6, 1.6}, {200, 105, 3, 1, .2, 3.2},
      {100, 200, 0, 0, .4, .4}}};
  const auto& p = parameters.at(item);
  double value = p.base;
  if (inventory < 10000) {
    value += p.bt * p.base / shape(p.below, p.threshold, p.threshold) *
             shape(p.below, 10000 - inventory, p.threshold);
  } else {
    value -= p.at * p.base / shape(p.above, p.threshold, p.threshold) *
             shape(p.above, inventory - 10000, p.threshold);
  }
  return std::max(1, int(std::nearbyint(value)));
}

int animal_product(Item animal) {
  if (animal == Item::GOOSE) return int(Item::EGG);
  if (animal == Item::COW) return int(Item::MILK);
  if (animal == Item::SHEEP) return int(Item::WOOL);
  return -1;
}

std::array<int, fastkag::N_ITEMS> projected_shed(
    const Simulator& env, int player, const PlayerAction& action) {
  auto result = env.privates()[player].shed;
  const auto positions = unit_positions(env.farms()[player]);
  const auto& inventories = env.privates()[player].inventories;
  const int limit = std::min<int>(positions.size(), action.units.size());
  for (int actor = 0; actor < limit; ++actor) {
    if (!access(positions[actor])) continue;
    const auto& unit = action.units[actor];
    const auto& inventory = inventories[actor];
    if (unit.op == Op::DROP) {
      for (int item = 0; item < fastkag::N_ITEMS; ++item)
        result[item] += std::max(0, inventory[item]);
    } else if (unit.op == Op::PLACE && unit.item < Item::GOOSE &&
               unit.item != Item::NONE) {
      const int item = int(unit.item);
      result[item] += std::min(count(unit), std::max(0, inventory[item]));
    } else if (unit.op == Op::PICKUP && unit.item != Item::NONE) {
      const int item = int(unit.item);
      result[item] = std::max(0, result[item] - count(unit));
    }
  }
  return result;
}

int exposure(const Simulator& env, int item, int quantity, int batch = 8) {
  const int inventory = env.market().inventory[item];
  int current = 0, delayed = 0;
  for (int i = 0; i < quantity; ++i) {
    current += market_price(item, inventory + i);
    delayed += market_price(item, inventory + batch + i);
  }
  return current - delayed;
}

void order_rows(PlayerAction& action, const Simulator& env, int player,
                bool sales_first, bool projected_quote, bool hygiene) {
  std::vector<Action> rows;
  rows.reserve(action.market.size());
  std::array<int, fastkag::N_ITEMS> sell_index{};
  sell_index.fill(-1);
  for (auto row : action.market) {
    if (hygiene && row.op == Op::SELL) {
      if (count(row) <= 0) continue;
      const int item = int(row.item);
      if (item >= 0 && sell_index[item] >= 0) {
        rows[sell_index[item]].quantity += count(row);
        continue;
      }
      if (item >= 0) sell_index[item] = rows.size();
    }
    rows.push_back(row);
  }
  if (sales_first) {
    for (int index = 0; index < int(rows.size()); ++index) {
      if (rows[index].op != Op::SELL) continue;
      int cursor = index;
      while (cursor > 0 && rows[cursor - 1].op != Op::SELL &&
             !((rows[cursor - 1].op == Op::BUY_PRODUCT ||
                rows[cursor - 1].op == Op::BUY_ANIMAL) &&
               rows[cursor - 1].item == rows[cursor].item)) {
        std::swap(rows[cursor - 1], rows[cursor]);
        --cursor;
      }
    }
  }
  if (projected_quote) {
    const auto stock = projected_shed(env, player, action);
    int begin = 0;
    while (begin < int(rows.size())) {
      if (rows[begin].op != Op::SELL) { ++begin; continue; }
      int end = begin;
      std::set<int> unique;
      while (end < int(rows.size()) && rows[end].op == Op::SELL) {
        unique.insert(int(rows[end].item));
        ++end;
      }
      if (int(unique.size()) == end - begin) {
        std::stable_sort(rows.begin() + begin, rows.begin() + end,
            [&](const Action& left, const Action& right) {
              const int li = int(left.item), ri = int(right.item);
              const int ls = li >= 0 && li < fastkag::N_PRODUCTS
                  ? exposure(env, li, std::min(count(left), stock[li])) : 0;
              const int rs = ri >= 0 && ri < fastkag::N_PRODUCTS
                  ? exposure(env, ri, std::min(count(right), stock[ri])) : 0;
              return ls > rs;
            });
      }
      begin = end;
    }
  }
  action.market = std::move(rows);
}

std::vector<Position> se_tiles() {
  std::vector<Position> result;
  for (int x = kHalf; x < 2 * kHalf; ++x)
    for (int y = kHalf; y < 2 * kHalf; ++y)
      if (x != kHalf || y != kHalf) result.push_back({int16_t(x), int16_t(y)});
  std::sort(result.begin(), result.end(), [](Position left, Position right) {
    return std::tuple{std::abs(int(left.x) - kHalf) +
                          std::abs(int(left.y) - kHalf), left.y, left.x} <
           std::tuple{std::abs(int(right.x) - kHalf) +
                          std::abs(int(right.y) - kHalf), right.y, right.x};
  });
  return result;
}

std::vector<std::vector<Position>> clusters(
    const std::vector<Position>& input, int groups) {
  auto tiles = input;
  std::sort(tiles.begin(), tiles.end(), position_less);
  std::vector<std::vector<Position>> result;
  int begin = 0;
  groups = std::max(1, groups);
  for (int group = 0; group < groups; ++group) {
    const int end = begin + (int(tiles.size()) - begin) / (groups - group);
    result.emplace_back(tiles.begin() + begin, tiles.begin() + end);
    begin = end;
  }
  return result;
}

int fib(int n) {
  int left = 1, right = 1;
  while (n-- > 0) { const int next = left + right; left = right; right = next; }
  return left;
}

}  // namespace

struct Opponent::Impl {
  struct RivalPrevious {
    int step{-1};
    std::array<int, fastkag::N_PRODUCTS> inventory{};
    std::array<int, fastkag::N_PRODUCTS> prices{};
    std::array<int, fastkag::N_ITEMS> projected{};
  };

  struct GardenState {
    bool blocked{};
    bool decided{};
    int day_seen{-1};
    int n_before{-1};
    int k_today{};
    int late_crew{3};
    std::vector<Position> tiles;
    std::vector<std::vector<Position>> groups;
  };

  struct InputTarget {
    Position position;
    Item crop{Item::NONE};
    int birth{};
    int yield{};
    int fertilized_until{-1};
    bool watered{};
    std::vector<int> water_steps;
    int harvest_step{-1};
    int first{};
    int last{};
    int cap{};
  };

  struct InputPlan {
    std::vector<InputTarget> path;
    int quantity{};
    bool loaded{};
  };

  struct SheepPending {
    int first{};
    bool initial{};
    int workers{};
  };

  struct SheepWork {
    int step{-1};
    Action command;
    std::array<int32_t, fastkag::N_ITEMS> inventory{};
  };

  struct SeatState {
    int action_step{-1};
    bool route_chosen{};
    int route_index{-1};
    int route_id{-1};
    std::set<int> returning;
    std::map<int, std::array<int, fastkag::N_PRODUCTS>> lead_debts;
    std::optional<bool> s2c_ok;
    double board_similarity{};
    bool same_family{};
    std::optional<RivalPrevious> rival_previous;
    std::vector<int> rival_horizons;
    std::map<int, Position> d29_assignments;
    GardenState garden;
    int feed_saved{};
    int feed_pending{};
    int feed_day{};
    int input_day{-1};
    int input_requested_day{-1};
    std::map<int, InputPlan> input_workers;
    std::map<int, InputPlan> input_pending;
    bool sheep_committed{};
    int sheep_day{-1};
    int sheep_requested_day{-1};
    int sheep_rescue_today{};
    std::optional<SheepPending> sheep_pending;
    std::map<int, std::vector<Position>> sheep_workers;
    std::map<int, SheepWork> sheep_work;
    std::array<int, 2> sheep_credit{};  // WOOL, FERTILIZER
  };

  explicit Impl(const std::string& path)
      : shared(shared_library(path)), library(*shared) {
    route_lookup.fill(-1);
    for (int index = 0; index < int(library.route_ids.size()); ++index) {
      const int route = library.route_ids[index];
      if (route >= 0 && route < int(route_lookup.size())) route_lookup[route] = index;
    }
    for (auto& state : seats) reset_state(state);
  }

  std::shared_ptr<const TapeLibrary> shared;
  const TapeLibrary& library;
  std::array<int, 256> route_lookup{};
  std::array<SeatState, 2> seats{};

  static void reset_state(SeatState& state) {
    state = {};
    state.action_step = -1;
    state.route_id = -1;
    state.route_index = -1;
    state.garden.tiles = se_tiles();
    state.garden.tiles.resize(12);
    state.garden.groups = clusters(state.garden.tiles, 1);
  }

  void reset() {
    for (auto& state : seats) reset_state(state);
  }

  void select_route(const Simulator& env, SeatState& state) const {
    if (env.step_count() < 144 || state.route_chosen) return;
    state.route_chosen = true;
    if (env.shops().size() < 2 || env.shops()[0] == 7 || env.shops()[1] == 7)
      return;
    const int pair = int(env.shops()[0]) * 8 + int(env.shops()[1]);
    state.route_index = library.pair_routes[pair];
    if (state.route_index >= 0)
      state.route_id = library.route_ids[state.route_index];
    else {
      state.route_id = 100;
      state.route_index = route_lookup[100];
    }
  }

  const PlayerAction& tape_action(const SeatState& state, int step) const {
    step = std::clamp(step, 0, kFrames - 1);
    return state.route_index < 0 ? library.base[step]
                                 : library.routes[state.route_index][step];
  }

  static Action substitute(Action action,
                           const std::map<Item, Item>& replacements) {
    if ((action.op == Op::PICKUP || action.op == Op::PLACE) &&
        replacements.contains(action.item))
      action.item = replacements.at(action.item);
    if (action.op == Op::BUILD_COOP && replacements.contains(Item::GOOSE))
      action.op = Op::BUILD_PASTURE;
    return action;
  }

  bool d29_on(const SeatState& state) const {
    return state.garden.decided && !state.garden.blocked;
  }

  PlayerAction plan(const Simulator& env, int player, const SeatState& state,
                    int target_step, int flags_step = -1) const {
    const int flag = flags_step < 0 ? target_step : flags_step;
    const bool yarn6 = flag >= 144 && shop_count(env, 7, 2) >= 2;
    const bool yarn7 = flag >= 168 && shop_count(env, 7, 2) >= 1;
    const bool yarn10 = flag >= 240 && shop_count(env, 7, 3) >= 1;
    int milk = 0;
    for (int i = 0; i < std::min<int>(3, env.shops().size()); ++i)
      milk += milk_shop(env.shops()[i]);
    const bool sheep_to_cow = flag >= 216 && shop_count(env, 7, 3) == 0 &&
        milk >= 2 && state.s2c_ok.value_or(false);
    std::map<Item, Item> replacements;
    if (yarn6 && target_step >= 144 && target_step < 168)
      replacements[Item::COW] = Item::SHEEP;
    if (yarn7 && target_step >= 168 && target_step < 192)
      replacements[Item::COW] = Item::SHEEP;
    if (yarn10 && target_step >= 240 && target_step < 288)
      replacements[Item::GOOSE] = Item::SHEEP;
    if (sheep_to_cow && target_step >= 216 && target_step < 264)
      replacements[Item::SHEEP] = Item::COW;

    PlayerAction result = tape_action(state, target_step);
    for (auto& unit : result.units) unit = substitute(unit, replacements);
    const bool wool_sales = yarn6 || yarn7 || yarn10;
    for (auto& order : result.market) {
      if (order.op == Op::BUY_ANIMAL && replacements.contains(order.item))
        order.item = replacements.at(order.item);
      if (order.op == Op::SELL && order.item == Item::EGG && yarn10) {
        order.item = Item::WOOL;
        order.quantity = 999;
      } else if (order.op == Op::SELL && order.item == Item::WOOL && wool_sales) {
        order.quantity = 999;
      } else if (order.op == Op::SELL && order.item == Item::MILK && sheep_to_cow) {
        order.quantity = 999;
      }
    }
    if (target_step >= 696 && d29_on(state)) {
      std::erase_if(result.market, [](const Action& order) {
        return order.op == Op::HIRE || order.op == Op::BUY_PRODUCT ||
               order.op == Op::BUY_SEED;
      });
      if (target_step == 696)
        result.market.insert(result.market.end(), 7, Action{Op::HIRE});
      else if (target_step == 697)
        result.market.insert(result.market.end(), 2, Action{Op::HIRE});
      result.units.clear();
      result.units.push_back({});
      const int hands = target_step == 697 ? 7 : target_step > 697 ? 9 : 0;
      result.units.insert(result.units.end(), hands, Action{});
    }
    return result;
  }

  static double board_similarity_robust(const Simulator& env, int player) {
    const auto& own = env.farms()[player].tiles;
    const auto& rival = env.farms()[1 - player].tiles;
    int active = 0, equal = 0;
    for (int index = 0; index < int(own.size()); ++index) {
      const int own_crop = own[index].kind == TileKind::PLANT ? int(own[index].crop) : -1;
      const int rival_crop = rival[index].kind == TileKind::PLANT ? int(rival[index].crop) : -1;
      const int own_animal = own[index].kind == TileKind::ANIMAL ? int(own[index].animal) : -1;
      const int rival_animal = rival[index].kind == TileKind::ANIMAL ? int(rival[index].animal) : -1;
      if (own_crop >= 0 || rival_crop >= 0 || own_animal >= 0 || rival_animal >= 0) {
        ++active;
        equal += own_crop == rival_crop && own_animal == rival_animal;
      }
    }
    return active >= 8 ? double(equal) / active : 0.0;
  }

  void lead_layer(PlayerAction& action, const Simulator& env, int player,
                  SeatState& state, int step, int horizon) const {
    std::array<int, fastkag::N_PRODUCTS> debt{};
    if (auto it = state.lead_debts.find(step); it != state.lead_debts.end()) {
      debt = it->second;
      state.lead_debts.erase(it);
    }
    std::vector<Action> market;
    market.reserve(action.market.size());
    for (auto order : action.market) {
      const int item = int(order.item);
      if (order.op == Op::SELL && item >= 0 && item < fastkag::N_PRODUCTS &&
          debt[item] > 0) {
        const int reduction = std::min(count(order), debt[item]);
        order.quantity = count(order) - reduction;
        debt[item] -= reduction;
        if (order.quantity <= 0) continue;
      }
      market.push_back(order);
    }
    action.market = std::move(market);
    if (step > 695 || step >= 712) return;
    // Public Fieldcraft never pulls a sale across its 72-step planning block.
    const int end = std::min({695, step + horizon,
                              (step / 72 + 1) * 72 - 1});
    if (end <= step) return;
    auto stock = projected_shed(env, player, action);
    std::array<int, fastkag::N_PRODUCTS> already_sold{};
    std::set<int> bought, picked;
    for (const auto& order : action.market) {
      if (order.op == Op::SELL && int(order.item) < fastkag::N_PRODUCTS)
        already_sold[int(order.item)] += count(order);
      if (order.op == Op::BUY_PRODUCT) bought.insert(int(order.item));
    }
    for (const auto& unit : action.units)
      if (unit.op == Op::PICKUP) picked.insert(int(unit.item));

    std::vector<int> items(kHarvested.begin(), kHarvested.end());
    if (step % 24 >= 6) items.push_back(int(Item::FERTILIZER));
    for (const int item : items) {
      if (bought.contains(item) || picked.contains(item) ||
          env.market().prices[item] < 2) continue;
      int available = std::max(0, stock[item] - already_sold[item]);
      if (available <= 0) continue;
      std::vector<std::pair<int, int>> moved;
      for (int future = step + 1; future <= end && available > 0; ++future) {
        const auto next = plan(env, player, state, future, step);
        bool stop = false;
        for (const auto& unit : next.units)
          stop |= unit.op == Op::PICKUP && int(unit.item) == item;
        for (const auto& order : next.market)
          stop |= order.op == Op::BUY_PRODUCT && int(order.item) == item;
        if (stop) break;
        int sold = 0;
        for (const auto& order : next.market)
          if (order.op == Op::SELL && int(order.item) == item)
            sold += count(order);
        const int reserved = state.lead_debts.contains(future)
            ? state.lead_debts.at(future)[item] : 0;
        const int amount = std::min(available, std::max(0, sold - reserved));
        if (amount > 0) { moved.emplace_back(future, amount); available -= amount; }
      }
      int total = 0;
      for (const auto [future, amount] : moved) total += amount;
      if (total <= 0) continue;
      auto existing = std::find_if(action.market.begin(), action.market.end(),
          [&](const Action& order) {
            return order.op == Op::SELL && int(order.item) == item;
          });
      if (existing != action.market.end()) existing->quantity += total;
      else if (action.market.size() < kMaxOrders)
        action.market.push_back({Op::SELL, static_cast<Item>(item), total});
      else continue;
      for (const auto [future, amount] : moved)
        state.lead_debts[future][item] += amount;
    }
  }

  int rival_lead_horizon(const Simulator& env, int player, SeatState& state,
                         int step, int base = 8) const {
    if (state.rival_previous && state.rival_previous->step == step - 1 &&
        (step - 1) % 4 != 0 && (step - 1) % 24 != 23 &&
        step - 1 >= 288 && step - 1 < 696) {
      static constexpr std::array<int, 5> items{{
          int(Item::STRAWBERRY), int(Item::MILK), int(Item::EGG),
          int(Item::MELON), int(Item::CARROT)}};
      const auto& previous = *state.rival_previous;
      const auto& shed = env.privates()[player].shed;
      for (const int item : items) {
        const int deposited = std::max(0, previous.projected[item] - shed[item]);
        const int flow = env.market().inventory[item] - previous.inventory[item] -
            (previous.prices[item] > 1 ? deposited : 0);
        if (flow < 2 || deposited > 0 || shed[item] <= 0) continue;
        int cumulative = 0;
        std::optional<int> observed;
        for (int future = step; future <= std::min(step - 1 + 17, 711); ++future) {
          const auto next = plan(env, player, state, future, step);
          int sold = 0;
          for (const auto& order : next.market)
            if (order.op == Op::SELL && int(order.item) == item)
              sold += count(order);
          if (sold == 0) continue;
          const int reserved = state.lead_debts.contains(future)
              ? state.lead_debts.at(future)[item] : 0;
          if (cumulative == 0 && (reserved >= sold || flow < sold - 1)) break;
          cumulative += sold;
          if (cumulative <= flow + 1) observed = future - (step - 1);
          else break;
        }
        if (observed) state.rival_horizons.push_back(*observed);
      }
    }
    if (state.rival_horizons.size() < 2) return base;
    auto values = state.rival_horizons;
    std::sort(values.begin(), values.end());
    int selected = values[1];
    for (int index = 0; index + 1 < int(values.size()); ++index) {
      if (values[index] - values[index + 1] <= 1) {
        selected = values[index];
        break;
      }
    }
    return std::max(base, std::min(20, selected + 1));
  }

  void save_rival_previous(const Simulator& env, int player,
                           const PlayerAction& action, SeatState& state) const {
    RivalPrevious previous;
    previous.step = env.step_count();
    previous.inventory = env.market().inventory;
    previous.prices = env.market().prices;
    previous.projected = projected_shed(env, player, action);
    state.rival_previous = previous;
  }

  void d29_route(PlayerAction& action, const Simulator& env, int player,
                 SeatState& state) const {
    struct Target { Position position; double value; bool water; };
    const int step = env.step_count(), day = env.day();
    const auto& farm = env.farms()[player];
    const auto& prices = env.market().prices;
    std::vector<Target> targets;
    for (int y = 0; y < env.config().board_size; ++y) {
      for (int x = 0; x < env.config().board_size; ++x) {
        if (x >= kHalf && y >= kHalf) continue;
        const auto& tile = farm.tiles[y * env.config().board_size + x];
        if (tile.kind == TileKind::PLANT) {
          int first = -1, last = -1, cap = -1;
          if (tile.crop == Item::WHEAT) std::tie(first, last, cap) = std::tuple{2,4,6};
          if (tile.crop == Item::CARROT) std::tie(first, last, cap) = std::tuple{2,3,4};
          if (tile.crop == Item::MELON) std::tie(first, last, cap) = std::tuple{10,12,6};
          if (first >= 0) {
            const int age = day - tile.planted_day;
            if (age < first) continue;
            const bool water = (last + 1) / 2 <= age && age <= last &&
                !tile.watered_today && tile.yield_units < cap;
            const int yield = tile.yield_units + int(water);
            if (yield <= 0) continue;
            const int multiplier = age > last ? 2 : 1;
            targets.push_back({{int16_t(x), int16_t(y)},
                double(multiplier * yield * std::max(1, prices[int(tile.crop)])),
                water});
          } else if (tile.yield_units >= 1) {
            targets.push_back({{int16_t(x), int16_t(y)},
                double(tile.yield_units * std::max(1, prices[int(tile.crop)])), false});
          }
        } else if (tile.kind == TileKind::ANIMAL && tile.yield_units >= 1) {
          const int product = animal_product(tile.animal);
          targets.push_back({{int16_t(x), int16_t(y)},
              double(tile.yield_units * std::max(1, prices[product])), false});
        }
      }
    }
    const int hands = std::min<int>(9, farm.hands.size());
    std::set<std::pair<int, int>> valid;
    for (const auto& target : targets)
      valid.emplace(target.position.x, target.position.y);
    for (auto it = state.d29_assignments.begin(); it != state.d29_assignments.end();) {
      const auto key = std::pair{int(it->second.x), int(it->second.y)};
      if (it->first > hands || !valid.contains(key))
        it = state.d29_assignments.erase(it);
      else ++it;
    }
    std::vector<Action> units = action.units;
    units.resize(hands + 1);
    auto positions = unit_positions(farm);
    const auto& inventories = env.privates()[player].inventories;
    for (int actor = 0; actor <= hands; ++actor) {
      const auto position = positions[actor];
      const auto& inventory = inventories[actor];
      int carried = 0;
      for (int item = 0; item < fastkag::N_PRODUCTS; ++item)
        carried += std::max(0, inventory[item]);
      const Position access_tile = nearest_access(position);
      const int access_distance = distance(position, access_tile);
      if (carried && step + access_distance >= 718) {
        units[actor] = access_distance == 0 ? Action{Op::DROP}
                                           : toward(position, access_tile);
        state.d29_assignments.erase(actor);
        continue;
      }
      auto assignment = state.d29_assignments.find(actor);
      if (assignment == state.d29_assignments.end()) {
        std::set<std::pair<int, int>> occupied;
        for (const auto& [other, target] : state.d29_assignments)
          if (other != actor) occupied.emplace(target.x, target.y);
        const Target* best = nullptr;
        std::tuple<int, double> best_score{
            std::numeric_limits<int>::min(), -1.0};
        for (const auto& target : targets) {
          if (occupied.contains({target.position.x, target.position.y})) continue;
          const int travel = distance(position, target.position);
          const int exit = distance(target.position, nearest_access(target.position));
          if (step + travel + (target.water ? 2 : 1) + exit > 718) continue;
          // D29_NEAREST is enabled in the frozen source: distance dominates,
          // then target value breaks ties; board insertion order breaks the rest.
          const auto score = std::tuple{-travel, target.value};
          if (!best || score > best_score) { best = &target; best_score = score; }
        }
        if (best) {
          state.d29_assignments[actor] = best->position;
          assignment = state.d29_assignments.find(actor);
        }
      }
      if (assignment != state.d29_assignments.end()) {
        const Position target = assignment->second;
        if (same_position(position, target)) {
          const auto found = std::find_if(targets.begin(), targets.end(),
              [&](const Target& value) { return same_position(value.position, target); });
          units[actor] = found != targets.end() && found->water
              ? Action{Op::WATER} : Action{Op::HARVEST};
        } else units[actor] = toward(position, target);
      } else if (carried) {
        units[actor] = access_distance == 0 ? Action{Op::DROP}
                                            : toward(position, access_tile);
      } else units[actor] = {};
    }
    action.units = std::move(units);
  }

  PlayerAction base_agent(const Simulator& env, int player, SeatState& state) const {
    const int step = env.step_count();
    if (step >= 216 && !state.s2c_ok && shop_count(env, 7, 3) == 0) {
      int milk = 0;
      for (int i = 0; i < std::min<int>(3, env.shops().size()); ++i)
        milk += milk_shop(env.shops()[i]);
      if (milk >= 2)
        state.s2c_ok = env.market().prices[int(Item::MILK)] >=
                       env.market().prices[int(Item::WOOL)];
    }
    PlayerAction result = plan(env, player, state, step);
    if (step >= 696 && d29_on(state)) d29_route(result, env, player, state);
    if (step < 712) {
      if (step % 24 == 0) {
        state.board_similarity = board_similarity_robust(env, player);
        state.same_family = state.board_similarity >= .7 || step < 48;
      }
      const int horizon = state.same_family
          ? rival_lead_horizon(env, player, state, step, 8) : 4;
      lead_layer(result, env, player, state, step, horizon);
      save_rival_previous(env, player, result, state);
      order_rows(result, env, player, true, true, false);
      return result;
    }
    const auto positions = unit_positions(env.farms()[player]);
    const auto& inventories = env.privates()[player].inventories;
    result.units.resize(std::max(result.units.size(), positions.size()));
    for (int actor = 0; actor < int(positions.size()); ++actor) {
      int products = 0;
      bool only_fertilizer = true;
      for (const int item : kProducts) {
        if (inventories[actor][item] <= 0) continue;
        products += inventories[actor][item];
        only_fertilizer &= item == int(Item::FERTILIZER);
      }
      if (products && only_fertilizer) {
        const auto target = nearest_access(positions[actor]);
        const int travel = distance(positions[actor], target);
        result.units[actor] = travel == 0 ? Action{Op::DROP}
                                          : toward(positions[actor], target);
        state.returning.insert(actor);
      } else if (state.returning.contains(actor) && products == 0) {
        result.units[actor] = {};
      }
    }
    std::vector<int> sell_order;
    for (const auto& order : tape_action(state, step).market)
      if (order.op == Op::SELL &&
          std::find(sell_order.begin(), sell_order.end(), int(order.item)) == sell_order.end())
        sell_order.push_back(int(order.item));
    for (const int item : kProducts)
      if (std::find(sell_order.begin(), sell_order.end(), item) == sell_order.end())
        sell_order.push_back(item);
    result.market.clear();
    for (const int item : sell_order)
      result.market.push_back({Op::SELL, static_cast<Item>(item), 999});
    return result;
  }

  static int deficit_by(const Simulator& env, int day = 27) {
    int result = day;
    for (int index = 0; index < int(env.shops().size()); ++index)
      if (tomato_shop(env.shops()[index]))
        result += 6 * std::max(0, day - (3 * (index + 1) - 1));
    return result;
  }

  Action garden_decide(const Simulator& env, int player, SeatState& state,
                       int group, Position position,
                       const std::array<int32_t, fastkag::N_ITEMS>& inventory) const {
    const int step = env.step_count(), day = env.day(), hour = env.hour();
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    const auto& targets = group < int(state.garden.groups.size())
        ? state.garden.groups[group] : std::vector<Position>{};
    const int tomato = inventory[int(Item::TOMATO)];
    const int fertilizer = inventory[int(Item::FERTILIZER)];
    const int room = 100 - shed_sum(private_state.shed);
    const auto access_tile = nearest_access(position);
    const int access_distance = distance(position, access_tile);
    const bool final = day >= 29;
    auto return_to_shed = [&]() {
      if (access(position))
        return room >= 1 ? Action{Op::PLACE, Item::TOMATO,
                                  std::min(tomato, room)} : Action{};
      return toward(position, access_tile);
    };
    if (tomato && step + access_distance + 1 >= 717) return return_to_shed();
    if (tomato && (tomato >= 12 || hour + access_distance >= 22))
      return return_to_shed();
    if ((day == 25 || day == 28) && fertilizer == 0 && access(position) &&
        hour <= 4 && private_state.shed[int(Item::FERTILIZER)] > 0) {
      int need = 0;
      for (const auto target : targets) {
        const auto* tile = tile_at(farm, target);
        need += tile && tile->kind == TileKind::PLANT &&
                tile->crop == Item::TOMATO && tile->fertilized_until_day < day;
      }
      if (need) return {Op::PICKUP, Item::FERTILIZER,
                        std::min(need, private_state.shed[int(Item::FERTILIZER)])};
    }
    if (fertilizer && access(position)) {
      bool need = false;
      for (const auto target : targets) {
        const auto* tile = tile_at(farm, target);
        need |= tile && tile->kind == TileKind::PLANT &&
                tile->crop == Item::TOMATO && tile->fertilized_until_day < day;
      }
      if (!need) return room >= 1
          ? Action{Op::PLACE, Item::FERTILIZER, fertilizer} : Action{};
    }
    struct Choice { double priority; int travel; Position target; Action action; };
    std::vector<Choice> choices;
    for (const auto target : targets) {
      const auto* tile = tile_at(farm, target);
      if (!tile) continue;
      const int travel = distance(position, target);
      if (tile->kind == TileKind::LOCKED) continue;
      if (tile->kind == TileKind::PLANT && tile->crop == Item::TOMATO) {
        const int exit = distance(target, nearest_access(target));
        const bool can_exit = hour + travel + 1 + exit <= 22 || step >= 697;
        if (final) {
          if (tile->yield_units >= 1 && step + travel + 1 + exit + 1 < 717)
            choices.push_back({.5, travel, target, {Op::HARVEST}});
          continue;
        }
        if (!tile->watered_today && tile->consecutive_unwatered >= 1)
          choices.push_back({0., travel, target, {Op::WATER}});
        if (tile->yield_units >= 3 && can_exit)
          choices.push_back({.5, travel, target, {Op::HARVEST}});
        if (fertilizer > 0 && (day == 25 || day == 28) &&
            tile->fertilized_until_day < day)
          choices.push_back({1., travel, target, {Op::FERTILIZE}});
        if (!tile->watered_today && tile->consecutive_unwatered == 0)
          choices.push_back({(day >= 25 && day <= 28) ? 1.1 : 4.,
                             travel, target, {Op::WATER}});
      } else if (tile->kind == TileKind::WEED && day <= 19) {
        choices.push_back({3., travel, target, {Op::DIG}});
      } else if (tile->kind == TileKind::EMPTY &&
                 private_state.seeds[int(Item::TOMATO)] > 0 && day <= 19) {
        choices.push_back({2., travel, target, {Op::PLANT, Item::TOMATO}});
      }
    }
    if (choices.empty()) return tomato ? return_to_shed() : Action{};
    const auto choice = *std::min_element(choices.begin(), choices.end(),
        [](const Choice& left, const Choice& right) {
          return std::tie(left.priority, left.travel, left.target.x, left.target.y,
                          left.action.op) <
                 std::tie(right.priority, right.travel, right.target.x, right.target.y,
                          right.action.op);
        });
    return choice.travel == 0 ? choice.action : toward(position, choice.target);
  }

  PlayerAction garden_layer(const Simulator& env, int player, SeatState& state,
                            PlayerAction result) const {
    auto& garden = state.garden;
    const int day = env.day(), hour = env.hour();
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    if (day != garden.day_seen) { garden.day_seen = day; garden.n_before = -1; }
    if (garden.blocked) return result;
    std::vector<Action> market = result.market;
    std::vector<Action> extra;
    if (day == 18 && hour == 1 && !unlocked(farm, 3) && farm.money >= 5500 &&
        deficit_by(env, 27) >= 300) {
      extra.push_back({Op::BUY_LAND});
      extra.push_back({Op::BUY_SEED, Item::TOMATO, 14});
    }
    if (unlocked(farm, 3) && day == 18 && hour == 2 && !garden.decided) {
      garden.decided = true;
      const bool rival_four = unlocked_count(env.farms()[1 - player]) >= 4;
      const bool large = deficit_by(env, 27) >= 400 ||
                         (rival_four && deficit_by(env, 27) >= 400);
      garden.tiles = se_tiles();
      garden.tiles.resize(large ? 16 : 12);
      garden.late_crew = large ? 4 : 3;
      if (large) extra.push_back({Op::BUY_SEED, Item::TOMATO, 4});
    }
    auto crew = [&](int target_day) {
      if (target_day < 18 || target_day > 29) return 0;
      if (target_day == 18) return 2;
      if (target_day < 25) return 1;
      if (target_day - 18 == 8) return std::max(1, garden.late_crew - 1);
      return garden.late_crew;
    };
    if (unlocked(farm, 3) && garden.decided && hour == 2 && crew(day) > 0) {
      garden.k_today = crew(day);
      garden.n_before = farm.hands.size();
      extra.insert(extra.end(), garden.k_today, Action{Op::HIRE});
      if (day == 25 || day == 28)
        extra.push_back({Op::BUY_PRODUCT, Item::FERTILIZER,
                         int(garden.tiles.size())});
    }
    std::vector<Action> new_workers;
    if (garden.n_before >= 0 && hour > 2) {
      const int count_new = std::max(0, std::min<int>(
          int(farm.hands.size()) - garden.n_before, garden.k_today));
      garden.groups = clusters(garden.tiles, count_new);
      std::vector<Action> hands;
      for (int index = 1; index < int(result.units.size()) &&
                          int(hands.size()) < garden.n_before; ++index)
        hands.push_back(result.units[index]);
      hands.resize(garden.n_before);
      for (int worker = 0; worker < count_new; ++worker) {
        const int hand = garden.n_before + worker;
        const int inventory = hand + 1;
        auto unit = garden_decide(env, player, state, worker, farm.hands[hand],
            inventory < int(private_state.inventories.size())
                ? private_state.inventories[inventory]
                : std::array<int32_t, fastkag::N_ITEMS>{});
        new_workers.push_back(unit);
        hands.push_back(unit);
      }
      const Action farmer = result.units.empty() ? Action{} : result.units[0];
      result.units.clear();
      result.units.push_back(farmer);
      result.units.insert(result.units.end(), hands.begin(), hands.end());
    }
    bool tomato_place = std::any_of(new_workers.begin(), new_workers.end(),
        [](const Action& unit) {
          return unit.op == Op::PLACE && unit.item == Item::TOMATO;
        });
    if (tomato_place || private_state.shed[int(Item::TOMATO)] > 0) {
      std::erase_if(market, [](const Action& order) {
        return order.op == Op::SELL && order.item == Item::TOMATO;
      });
      market.insert(market.begin(), {Op::SELL, Item::TOMATO, 999});
    }
    market.insert(market.end(), extra.begin(), extra.end());
    if (market.size() > kMaxOrders) market.resize(kMaxOrders);
    result.market = std::move(market);
    return result;
  }

  std::map<std::pair<int, int>, InputTarget> input_forecast(
      const Simulator& env, int player, const SeatState& state,
      int expected) const {
    const int step = env.step_count(), day = env.day();
    const auto& farm = env.farms()[player];
    std::vector<Position> positions{farm.farmer};
    positions.insert(positions.end(), farm.hands.begin(),
                     farm.hands.begin() + std::min<int>(expected,
                                                        farm.hands.size()));
    std::map<std::pair<int, int>, InputTarget> targets;
    for (int y = 0; y < env.config().board_size; ++y) {
      for (int x = 0; x < env.config().board_size; ++x) {
        const auto& tile = farm.tiles[y * env.config().board_size + x];
        int first = 0, last = 0, cap = 0;
        if (tile.kind != TileKind::PLANT) continue;
        if (tile.crop == Item::WHEAT)
          std::tie(first, last, cap) = std::tuple{2, 4, 6};
        else if (tile.crop == Item::CARROT)
          std::tie(first, last, cap) = std::tuple{2, 3, 4};
        else continue;
        const int age = day - tile.planted_day;
        if (age < 1 || age >= last) continue;
        targets[{x, y}] = {{int16_t(x), int16_t(y)}, tile.crop,
                           tile.planted_day, tile.yield_units,
                           tile.fertilized_until_day, tile.watered_today,
                           {}, -1, first, last, cap};
      }
    }
    std::set<std::tuple<int, int, int>> watered;
    const int end = std::min(712, (day + 4) * 24);
    for (int future = step; future < end; ++future) {
      const auto planned = plan(env, player, state, future, step);
      for (int actor = 0;
           actor < std::min<int>(positions.size(), planned.units.size());
           ++actor) {
        const auto& unit = planned.units[actor];
        const auto key = std::pair{int(positions[actor].x),
                                   int(positions[actor].y)};
        auto found = targets.find(key);
        if (found != targets.end() && found->second.harvest_step < 0) {
          auto& target = found->second;
          if (unit.op == Op::WATER &&
              watered.insert({future / 24, key.first, key.second}).second) {
            const int water_day = future / 24;
            const bool already_watered = water_day == day && target.watered;
            const int age = water_day - target.birth;
            if (!already_watered && age >= target.first && age <= target.last)
              target.water_steps.push_back(future);
          }
          if (unit.op == Op::HARVEST) target.harvest_step = future;
        }
        if (movement(unit.op)) positions[actor] = moved(positions[actor], unit.op);
      }
      for (const auto& row : planned.market) {
        if (row.op != Op::HIRE) continue;
        Position spawn = kAccess.front();
        int best = std::numeric_limits<int>::max();
        for (const auto candidate : kAccess) {
          const int occupancy = std::count_if(
              positions.begin(), positions.end(), [&](Position value) {
                return same_position(value, candidate);
              });
          if (occupancy < best) { best = occupancy; spawn = candidate; }
        }
        positions.push_back(spawn);
      }
      if ((future + 1) % 24 == 0) positions = {{4, 4}};
    }
    return targets;
  }

  static int input_gain(const InputTarget& target, int arrival, int day) {
    if (target.harvest_step < 0 || target.harvest_step <= arrival) return 0;
    int usable = 0;
    int predicted_yield = target.yield;
    for (const int water : target.water_steps) {
      const int water_day = water / 24;
      if (arrival < water && water <= target.harvest_step &&
          water_day >= day && water_day <= day + 2 &&
          water_day > target.fertilized_until)
        ++usable;
      predicted_yield += water_day <= target.fertilized_until ? 2 : 1;
    }
    return std::max(0, std::min(usable, target.cap - predicted_yield));
  }

  struct BeamResult {
    std::vector<InputTarget> path;
    std::array<int, 2> gains{};  // WHEAT, CARROT
  };

  static bool input_path_less(const std::vector<InputTarget>& left,
                              const std::vector<InputTarget>& right) {
    return std::lexicographical_compare(
        left.begin(), left.end(), right.begin(), right.end(),
        [](const InputTarget& a, const InputTarget& b) {
          return std::tuple{a.position.x, a.position.y, int(a.crop), a.birth} <
                 std::tuple{b.position.x, b.position.y, int(b.crop), b.birth};
        });
  }

  BeamResult input_beam(
      const Simulator& env, int player, const SeatState& state,
      const PlayerAction& current,
      const std::map<std::pair<int, int>, InputTarget>& targets,
      int worker_index) const {
    std::vector<Position> positions = unit_positions(env.farms()[player]);
    for (int actor = 0;
         actor < std::min<int>(positions.size(), current.units.size()); ++actor)
      if (movement(current.units[actor].op))
        positions[actor] = moved(positions[actor], current.units[actor].op);
    int hires = 0;
    for (const auto& row : current.market) hires += row.op == Op::HIRE;
    Position spawn = kAccess.front();
    for (int index = 0; index < hires + worker_index + 1; ++index) {
      int best = std::numeric_limits<int>::max();
      for (const auto candidate : kAccess) {
        const int occupancy = std::count_if(
            positions.begin(), positions.end(), [&](Position value) {
              return same_position(value, candidate);
            });
        if (occupancy < best) { best = occupancy; spawn = candidate; }
      }
      positions.push_back(spawn);
    }

    struct Beam {
      double score{};
      int gross{};
      int time{};
      Position position;
      std::vector<InputTarget> path;
      std::set<std::pair<int, int>> visited;
      std::array<int, 2> gains{};
    };
    const int day = env.day();
    const std::array<int, 2> crop_prices{{
        std::max(1, env.market().prices[int(Item::WHEAT)] - 2),
        std::max(1, env.market().prices[int(Item::CARROT)] - 2)}};
    const int fertilizer_cost = std::max(
        1, market_price(int(Item::FERTILIZER),
                        env.market().inventory[int(Item::FERTILIZER)] - 16) + 2);
    std::vector<Beam> beam{{0., 0, env.step_count() + 2, spawn, {}, {}, {}}};
    std::optional<Beam> best;
    auto better = [](const Beam& left, const Beam& right) {
      if (left.score != right.score) return left.score > right.score;
      if (left.gross != right.gross) return left.gross > right.gross;
      if (left.time != right.time) return left.time < right.time;
      return input_path_less(left.path, right.path);
    };
    for (int depth = 0; depth < 8; ++depth) {
      std::vector<Beam> expanded;
      for (const auto& node : beam) {
        for (const auto& [key, target] : targets) {
          if (node.visited.contains(key)) continue;
          const int arrival = node.time + distance(node.position, target.position);
          if (arrival >= day * 24 + 23) continue;
          const int gain = input_gain(target, arrival, day);
          if (gain <= 0) continue;
          Beam next = node;
          next.time = arrival + 1;
          next.position = target.position;
          next.path.push_back(target);
          next.visited.insert(key);
          const int crop = target.crop == Item::WHEAT ? 0 : 1;
          next.gross += gain * crop_prices[crop];
          next.gains[crop] += gain;
          next.score = next.gross - 1.5 * fertilizer_cost * next.path.size();
          expanded.push_back(std::move(next));
        }
      }
      if (expanded.empty()) break;
      std::sort(expanded.begin(), expanded.end(), better);
      if (expanded.size() > 8) expanded.resize(8);
      beam = std::move(expanded);
      if (depth >= 2 && (!best || better(beam.front(), *best)))
        best = beam.front();
    }
    if (!best) return {};
    return {best->path, best->gains};
  }

  std::vector<InputPlan> input_joint_plans(
      const Simulator& env, int player, const SeatState& state,
      const PlayerAction& current,
      const std::map<std::pair<int, int>, InputTarget>& all_targets,
      const std::array<int, fastkag::N_ITEMS>& stock, int purchases,
      int topup) const {
    struct Portfolio {
      int net{}, gross{}, negative_cost{}, negative_workers{}, negative_scenario{};
      std::vector<InputPlan> plans;
    };
    std::optional<Portfolio> best;
    int nonempty_rows = current.market.size();
    for (int scenario = 0; scenario < 3; ++scenario) {
      auto targets = all_targets;
      std::vector<InputPlan> plans;
      int fertilizer = 0, cost = 0, revenue = 0;
      std::array<int, 2> prior_gains{};
      for (int worker = 0; worker < 2; ++worker) {
        auto eligible = targets;
        if (worker == 0 && scenario > 0) {
          const Item crop = scenario == 1 ? Item::WHEAT : Item::CARROT;
          std::erase_if(eligible, [&](const auto& pair) {
            return pair.second.crop != crop;
          });
        }
        auto result = input_beam(env, player, state, current, eligible, worker);
        const int tiles = result.path.size();
        if (tiles < 3 || nonempty_rows + 2 + worker > kMaxOrders ||
            shed_sum(stock) + purchases + fertilizer + tiles + topup > 95)
          break;
        const int price = market_price(
            int(Item::FERTILIZER),
            env.market().inventory[int(Item::FERTILIZER)] -
                fertilizer - tiles - topup);
        const int worker_cost =
            (tiles + (worker == 0 ? topup : 0)) * (price + 2) +
            fib(env.farms()[player].hires_today + worker);
        int worker_revenue = 0;
        for (int crop = 0; crop < 2; ++crop) {
          const int item = crop == 0 ? int(Item::WHEAT) : int(Item::CARROT);
          const int units = result.gains[crop];
          worker_revenue += units * std::max(
              1, market_price(item, env.market().inventory[item] +
                                      prior_gains[crop] + units) - 2);
        }
        if (worker_revenue < 1.5 * worker_cost + 50 ||
            env.farms()[player].money < cost + worker_cost + 3000)
          break;
        plans.push_back({result.path, tiles, false});
        fertilizer += tiles;
        cost += worker_cost;
        revenue += worker_revenue;
        for (int crop = 0; crop < 2; ++crop)
          prior_gains[crop] += result.gains[crop];
        for (const auto& target : result.path)
          targets.erase({target.position.x, target.position.y});
      }
      Portfolio portfolio{revenue - cost, revenue, -cost,
                          -int(plans.size()), -scenario, plans};
      const auto score = [](const Portfolio& value) {
        return std::tuple{value.net, value.gross, value.negative_cost,
                          value.negative_workers, value.negative_scenario};
      };
      if (!best || score(portfolio) > score(*best)) best = std::move(portfolio);
    }
    return best ? best->plans : std::vector<InputPlan>{};
  }

  PlayerAction input_hand_layer(const Simulator& env, int player,
                                SeatState& state, PlayerAction result) const {
    const int step = env.step_count(), day = env.day(), hour = env.hour();
    auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    if (day != state.input_day) {
      state.input_day = day;
      state.input_workers.clear();
      state.input_pending.clear();
    }
    if (!state.input_pending.empty()) {
      auto pending = std::move(state.input_pending);
      state.input_pending.clear();
      for (auto& [actor, plan] : pending)
        if (int(farm.hands.size()) >= actor)
          state.input_workers[actor] = std::move(plan);
    }
    if (!state.input_workers.empty()) {
      result.units.resize(farm.hands.size() + 1);
      for (auto& [actor, worker] : state.input_workers) {
        if (actor > int(farm.hands.size())) continue;
        Action command{};
        const auto position = farm.hands[actor - 1];
        const auto& inventory = private_state.inventories[actor];
        if (!worker.loaded) {
          const auto stock = projected_shed(env, player, result);
          const int quantity = std::min(
              worker.quantity,
              std::max(0, stock[int(Item::FERTILIZER)]));
          if (quantity > 0 && access(position)) {
            command = {Op::PICKUP, Item::FERTILIZER, quantity};
            worker.loaded = true;
          }
        } else if (inventory[int(Item::FERTILIZER)] > 0) {
          while (!worker.path.empty()) {
            const auto& target = worker.path.front();
            const auto* tile = tile_at(farm, target.position);
            if (!tile || tile->kind != TileKind::PLANT ||
                tile->crop != target.crop || tile->planted_day != target.birth ||
                tile->fertilized_until_day >= day + 2) {
              worker.path.erase(worker.path.begin());
              continue;
            }
            command = same_position(position, target.position)
                ? Action{Op::FERTILIZE}
                : toward(position, target.position);
            if (command.op == Op::FERTILIZE)
              worker.path.erase(worker.path.begin());
            break;
          }
        }
        result.units[actor] = command;
      }
      return result;
    }
    if ((hour < 1 || hour > 3) || day < 12 || day > 28 ||
        state.input_requested_day == day)
      return result;
    if (std::any_of(result.market.begin(), result.market.end(),
                    [](const Action& row) { return row.op == Op::HIRE; }))
      return result;
    int expected = 0;
    bool future_hire = false;
    for (int future = day * 24; future < std::min((day + 1) * 24, kFrames);
         ++future) {
      const auto planned = plan(env, player, state, future, step);
      expected = std::max<int>(expected, planned.units.size() - 1);
      if (future >= day * 24 + hour)
        future_hire |= std::any_of(
            planned.market.begin(), planned.market.end(),
            [](const Action& row) { return row.op == Op::HIRE; });
    }
    if (future_hire) return result;
    const auto targets = input_forecast(env, player, state, expected);
    const auto stock = projected_shed(env, player, result);
    int purchases = 0;
    int available_fertilizer = stock[int(Item::FERTILIZER)];
    for (const auto& row : result.market) {
      if (row.op == Op::BUY_PRODUCT || row.op == Op::BUY_ANIMAL)
        purchases += count(row);
      if (row.item == Item::FERTILIZER && row.op == Op::SELL)
        available_fertilizer = std::max(0, available_fertilizer - count(row));
      else if (row.item == Item::FERTILIZER && row.op == Op::BUY_PRODUCT)
        available_fertilizer += count(row);
    }
    int next_pickup = 0;
    if (step + 1 < kFrames) {
      const auto next = plan(env, player, state, step + 1, step);
      for (const auto& unit : next.units)
        if (unit.op == Op::PICKUP && unit.item == Item::FERTILIZER)
          next_pickup += count(unit);
    }
    const int topup = std::max(0, next_pickup - available_fertilizer);
    auto plans = input_joint_plans(env, player, state, result, targets,
                                   stock, purchases, topup);
    if (plans.empty()) return result;
    int fertilizer = 0;
    for (int index = 0; index < int(plans.size()); ++index) {
      fertilizer += plans[index].quantity;
      state.input_pending[int(farm.hands.size()) + 1 + index] = plans[index];
    }
    state.input_requested_day = day;
    result.market.push_back(
        {Op::BUY_PRODUCT, Item::FERTILIZER, fertilizer + topup});
    result.market.insert(result.market.end(), plans.size(), Action{Op::HIRE});
    return result;
  }

  static PlayerAction input_surplus(const Simulator& env, int player,
                                    PlayerAction result) {
    const int step = env.step_count();
    if (step >= 712 || env.day() < 12) return result;
    auto& rows = result.market;
    const auto& private_state = env.privates()[player];
    const int wheat = private_state.shed[int(Item::WHEAT)];
    const bool wheat_order = std::any_of(rows.begin(), rows.end(),
        [](const Action& row) {
          return (row.op == Op::SELL || row.op == Op::BUY_PRODUCT) &&
                 row.item == Item::WHEAT;
        });
    if (wheat > 60 && rows.size() < kMaxOrders && !wheat_order)
      rows.push_back({Op::SELL, Item::WHEAT, wheat - 60});
    if (env.hour() != 23) return result;

    const auto projected = projected_shed(env, player, result);
    const int carried = [&] {
      int total = 0;
      for (const auto& inventory : private_state.inventories)
        total += inventory_sum(inventory);
      return total;
    }();
    const int deposited = std::max(0, shed_sum(projected) -
                                      shed_sum(private_state.shed));
    std::array<int, fastkag::N_ITEMS> sold{};
    for (const auto& row : rows)
      if (row.op == Op::SELL && row.item != Item::NONE)
        sold[int(row.item)] += count(row);
    int executable_sales = 0;
    for (int item = 0; item < fastkag::N_ITEMS; ++item)
      executable_sales += std::min(projected[item], sold[item]);
    int overflow = shed_sum(projected) + (carried - deposited) -
                   executable_sales - 99;
    if (overflow <= 0) return result;

    std::vector<int> priority{int(Item::WHEAT), int(Item::FERTILIZER),
                              int(Item::CARROT), int(Item::EGG)};
    std::vector<int> rest;
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item)
      if (std::find(priority.begin(), priority.end(), item) == priority.end())
        rest.push_back(item);
    std::stable_sort(rest.begin(), rest.end(), [&](int left, int right) {
      return env.market().prices[left] < env.market().prices[right];
    });
    priority.insert(priority.end(), rest.begin(), rest.end());
    for (const int item : priority) {
      const int available = std::max(0, projected[item] - sold[item]);
      const int quantity = std::min(overflow, available);
      if (quantity <= 0 || env.market().prices[item] < 1) continue;
      auto existing = std::find_if(rows.begin(), rows.end(),
          [&](const Action& row) {
            return row.op == Op::SELL && int(row.item) == item;
          });
      if (existing != rows.end()) existing->quantity += quantity;
      else {
        if (rows.size() >= kMaxOrders) break;
        rows.push_back({Op::SELL, static_cast<Item>(item), quantity});
      }
      sold[item] += quantity;
      overflow -= quantity;
      if (overflow <= 0) break;
    }
    return result;
  }

  bool sheep_eligible(const Simulator& env, int player,
                      const SeatState& state) const {
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    if (farm.tiles.size() != 100 || farm.unlocked_mask != 0b0111) return false;
    if (shop_count(env, 7, env.shops().size()) < 2 ||
        env.market().prices[int(Item::WOOL)] < 220 ||
        env.market().prices[int(Item::WHEAT)] > 45)
      return false;
    for (int y = 5; y <= 6; ++y)
      for (int x = 5; x <= 7; ++x)
        if (farm.tiles[y * 10 + x].kind != TileKind::LOCKED) return false;
    if (private_state.shed[int(Item::SHEEP)] > 0) return false;
    for (const auto& inventory : private_state.inventories)
      if (inventory[int(Item::SHEEP)] > 0) return false;
    for (int future = 12 * 24; future < kFrames; ++future) {
      const auto planned = plan(env, player, state, future, env.step_count());
      for (const auto& row : planned.market)
        if (row.op == Op::BUY_LAND ||
            (row.op == Op::BUY_ANIMAL && row.item == Item::SHEEP))
          return false;
      for (const auto& unit : planned.units)
        if ((unit.op == Op::PICKUP || unit.op == Op::PLACE) &&
            unit.item == Item::SHEEP)
          return false;
    }
    return true;
  }

  static int sheep_hires_today(const Simulator& env, int player,
                               bool initial) {
    if (!initial) {
      bool finished = true;
      const auto& farm = env.farms()[player];
      const int day = env.day();
      for (int y = 5; y <= 6; ++y) {
        for (int x = 5; x <= 7; ++x) {
          const auto& tile = farm.tiles[y * 10 + x];
          if (tile.kind != TileKind::ANIMAL || tile.animal != Item::SHEEP ||
              tile.yield_units >= 1) {
            finished = false;
            continue;
          }
          for (int future_day = day; future_day < 29; ++future_day) {
            const int offset = future_day + 1 - tile.placed_day - 6;
            if (offset >= 0 && offset % 3 == 0) {
              finished = false;
              break;
            }
          }
        }
      }
      if (finished) return 0;
    }
    if (initial || env.day() <= 13) return 2;
    const auto& farm = env.farms()[player];
    for (int y = 5; y <= 6; ++y)
      for (int x = 5; x <= 7; ++x) {
        const auto& tile = farm.tiles[y * 10 + x];
        if (tile.kind != TileKind::ANIMAL || tile.animal != Item::SHEEP ||
            tile.yield_units >= 1)
          return 2;
      }
    return 1;
  }

  PlayerAction sheep_request(const Simulator& env, int player,
                             SeatState& state, PlayerAction result) const {
    const int day = env.day(), hour = env.hour();
    if (hour > (state.sheep_committed ? 2 : 1) ||
        state.sheep_requested_day == day)
      return result;
    if (!state.sheep_committed &&
        (day != 12 || !sheep_eligible(env, player, state)))
      return result;
    int expected = 0;
    bool future_hire = false;
    for (int future = day * 24; future < std::min((day + 1) * 24, kFrames);
         ++future) {
      const auto planned = plan(env, player, state, future, env.step_count());
      expected = std::max<int>(expected, planned.units.size() - 1);
      if (future > day * 24 + hour)
        future_hire |= std::any_of(
            planned.market.begin(), planned.market.end(),
            [](const Action& row) { return row.op == Op::HIRE; });
    }
    if (future_hire) return result;
    int current_hires = 0;
    for (const auto& row : result.market) current_hires += row.op == Op::HIRE;
    const auto& farm = env.farms()[player];
    if (int(farm.hands.size()) + current_hires != expected) return result;
    const bool initial = !state.sheep_committed;
    const int workers = sheep_hires_today(env, player, initial);
    if (workers == 0) {
      state.sheep_requested_day = day;
      return result;
    }
    const int extras = (initial ? 2 : 0) + 1 + workers;
    if (int(result.market.size()) + extras > kMaxOrders) return result;
    const auto stock = projected_shed(env, player, result);
    int added_capacity = 6 + (initial ? 6 : 0);
    double cost = initial ? 4000 + 400 * 6 : 0;
    cost += 6 * (env.market().prices[int(Item::WHEAT)] + 10);
    for (int index = farm.hires_today;
         index < farm.hires_today + current_hires + workers; ++index)
      cost += fib(index);
    static constexpr std::array<int, fastkag::N_CROPS> seed_cost{{
        10, 20, 50, 100, 80}};
    static constexpr std::array<int, fastkag::N_ANIMALS> animal_cost{{
        300, 500, 400}};
    for (const auto& row : result.market) {
      if (row.op == Op::BUY_LAND) return result;
      if (row.op == Op::BUY_PRODUCT && row.item != Item::NONE) {
        added_capacity += count(row);
        cost += count(row) * (env.market().prices[int(row.item)] + 10);
      } else if (row.op == Op::BUY_ANIMAL && row.item >= Item::GOOSE) {
        added_capacity += count(row);
        cost += count(row) * animal_cost[int(row.item) - int(Item::GOOSE)];
      } else if (row.op == Op::BUY_SEED && row.item >= Item::WHEAT &&
                 row.item <= Item::MELON) {
        cost += count(row) * seed_cost[int(row.item)];
      }
    }
    if (shed_sum(stock) + added_capacity > kShedCapacity ||
        farm.money < cost + (initial ? 3000 : 1000))
      return result;
    state.sheep_requested_day = day;
    state.sheep_pending = SheepPending{expected + 1, initial, workers};
    if (initial) {
      result.market.push_back({Op::BUY_LAND});
      result.market.push_back({Op::BUY_ANIMAL, Item::SHEEP, 6});
    }
    result.market.push_back({Op::BUY_PRODUCT, Item::WHEAT, 6});
    result.market.insert(result.market.end(), workers, Action{Op::HIRE});
    return result;
  }

  static Position nearest_access_lex(Position position) {
    return *std::min_element(kAccess.begin(), kAccess.end(),
        [&](Position left, Position right) {
          return std::tuple{distance(position, left), left.x, left.y} <
                 std::tuple{distance(position, right), right.x, right.y};
        });
  }

  static Action sheep_worker(const Simulator& env, int player, int actor,
                             const std::vector<Position>& targets) {
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    const Position position = farm.hands[actor - 1];
    const auto& inventory = private_state.inventories[actor];
    const Position access_tile = nearest_access_lex(position);
    const int access_distance = distance(position, access_tile);
    for (const Item item : {Item::WOOL, Item::FERTILIZER}) {
      if (inventory[int(item)] > 0 &&
          env.hour() >= (env.day() == 29 ? 22 : 23) - access_distance)
        return access_distance == 0
            ? Action{Op::PLACE, item, inventory[int(item)]}
            : toward(position, access_tile);
    }
    int missing_animals = 0, unfed = 0;
    bool needs_primary_work = false;
    for (const auto target : targets) {
      const auto* tile = tile_at(farm, target);
      const bool animal = tile && tile->kind == TileKind::ANIMAL &&
                          tile->animal == Item::SHEEP;
      missing_animals += !animal;
      unfed += !(animal && tile->fed_today);
      needs_primary_work |= targets.size() > 3 && animal &&
                            !(tile->fed_today && tile->cared_today);
    }
    if (missing_animals && inventory[int(Item::SHEEP)] == 0 &&
        private_state.shed[int(Item::SHEEP)] > 0)
      return access_distance == 0
          ? Action{Op::PICKUP, Item::SHEEP,
                   std::min(missing_animals,
                            private_state.shed[int(Item::SHEEP)])}
          : toward(position, access_tile);
    if (unfed && inventory[int(Item::WHEAT)] == 0 &&
        private_state.shed[int(Item::WHEAT)] > 0)
      return access_distance == 0
          ? Action{Op::PICKUP, Item::WHEAT,
                   std::min(unfed, private_state.shed[int(Item::WHEAT)])}
          : toward(position, access_tile);

    struct Choice { int travel, index; Position target; Action command; };
    std::vector<Choice> choices;
    for (int index = 0; index < int(targets.size()); ++index) {
      const Position target = targets[index];
      const auto* tile = tile_at(farm, target);
      Action command{};
      bool valid = false;
      if (tile && tile->kind == TileKind::EMPTY) {
        command = {Op::BUILD_PASTURE}; valid = true;
      } else if (tile && tile->kind == TileKind::WEED) {
        command = {Op::DIG}; valid = true;
      } else if (tile && tile->kind == TileKind::PASTURE &&
                 tile->animal == Item::NONE &&
                 inventory[int(Item::SHEEP)] > 0) {
        command = {Op::PLACE, Item::SHEEP}; valid = true;
      } else if (tile && tile->kind == TileKind::ANIMAL &&
                 tile->animal == Item::SHEEP) {
        if (!tile->fed_today && inventory[int(Item::WHEAT)] > 0)
          command = {Op::FEED}, valid = true;
        else if (!tile->cared_today)
          command = {Op::CARE}, valid = true;
        else if (tile->yield_units > 0)
          command = {Op::HARVEST}, valid = true;
        else if (tile->fertilizer_available && !needs_primary_work)
          command = {Op::COLLECT_FERTILIZER}, valid = true;
      }
      if (valid) choices.push_back(
          {distance(position, target), index, target, command});
    }
    if (!choices.empty()) {
      const auto choice = *std::min_element(
          choices.begin(), choices.end(), [](const Choice& left,
                                              const Choice& right) {
            return std::tuple{left.travel, left.index} <
                   std::tuple{right.travel, right.index};
          });
      return choice.travel == 0 ? choice.command
                                : toward(position, choice.target);
    }
    for (const Item item : {Item::WOOL, Item::FERTILIZER})
      if (inventory[int(item)] > 0)
        return access_distance == 0
            ? Action{Op::PLACE, item, inventory[int(item)]}
            : toward(position, access_tile);
    return {};
  }

  static PlayerAction sheep_rescue(const Simulator& env, int player,
                                   SeatState& state, PlayerAction result) {
    if (state.sheep_workers.empty() || env.hour() > 14 ||
        result.market.size() >= kMaxOrders)
      return result;
    for (const auto& row : result.market)
      if (row.op == Op::HIRE || row.op == Op::BUY_LAND ||
          row.op == Op::BUY_ANIMAL || row.op == Op::BUY_PRODUCT ||
          row.op == Op::BUY_SEED || row.item == Item::WHEAT)
        return result;
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    int need = 0, carried = 0;
    for (const auto& [actor, targets] : state.sheep_workers) {
      const Action command = actor < int(result.units.size())
          ? result.units[actor] : Action{};
      if (command.op == Op::FEED ||
          (command.op == Op::PICKUP && command.item == Item::WHEAT))
        return result;
      if (actor < int(private_state.inventories.size()))
        carried += private_state.inventories[actor][int(Item::WHEAT)];
      for (const auto target : targets) {
        const auto* tile = tile_at(farm, target);
        need += tile && tile->kind == TileKind::ANIMAL &&
                tile->animal == Item::SHEEP && !tile->fed_today;
      }
    }
    const auto stock = projected_shed(env, player, result);
    const int quantity = need - carried - stock[int(Item::WHEAT)];
    if (quantity <= 0 || quantity > 6 ||
        state.sheep_rescue_today + quantity > 6 ||
        env.market().prices[int(Item::WHEAT)] < 1 ||
        farm.money < 1000 +
                         quantity * (env.market().prices[int(Item::WHEAT)] + 10) ||
        shed_sum(stock) + quantity > kShedCapacity)
      return result;
    result.market.push_back({Op::BUY_PRODUCT, Item::WHEAT, quantity});
    state.sheep_rescue_today += quantity;
    return result;
  }

  PlayerAction sheep_layer(const Simulator& env, int player, SeatState& state,
                           PlayerAction result) const {
    const int step = env.step_count(), day = env.day();
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    if (day < 12) return result;
    if (day != state.sheep_day) {
      state.sheep_day = day;
      state.sheep_workers.clear();
      state.sheep_work.clear();
      state.sheep_rescue_today = 0;
    }
    for (const auto& [actor, work] : state.sheep_work) {
      if (work.step != step - 1 ||
          actor >= int(private_state.inventories.size()))
        continue;
      int credit = -1;
      if (work.command.op == Op::HARVEST) credit = 0;
      if (work.command.op == Op::COLLECT_FERTILIZER) credit = 1;
      if (credit < 0) continue;
      const Item item = credit == 0 ? Item::WOOL : Item::FERTILIZER;
      state.sheep_credit[credit] += std::max(
          0, private_state.inventories[actor][int(item)] -
                 work.inventory[int(item)]);
    }
    if (state.sheep_pending) {
      const auto pending = *state.sheep_pending;
      state.sheep_pending.reset();
      const bool southeast = unlocked(farm, 3);
      const bool ready = southeast &&
          (!pending.initial || private_state.shed[int(Item::SHEEP)] >= 6);
      if (ready && int(farm.hands.size()) >=
                       pending.first + pending.workers - 1) {
        const std::vector<Position> upper{{5, 5}, {6, 5}, {7, 5}};
        const std::vector<Position> lower{{5, 6}, {6, 6}, {7, 6}};
        if (pending.workers == 1) {
          auto both = upper;
          both.insert(both.end(), lower.begin(), lower.end());
          state.sheep_workers[pending.first] = std::move(both);
        } else {
          state.sheep_workers[pending.first] = upper;
          state.sheep_workers[pending.first + 1] = lower;
        }
        if (pending.initial) {
          state.sheep_committed = true;
          state.garden.blocked = true;
        }
      }
    }
    result = sheep_request(env, player, state, std::move(result));
    if (!state.sheep_committed) return result;
    result.units.resize(farm.hands.size() + 1);
    state.sheep_work.clear();
    for (const auto& [actor, targets] : state.sheep_workers) {
      if (actor > int(farm.hands.size())) continue;
      const Action command = sheep_worker(env, player, actor, targets);
      result.units[actor] = command;
      state.sheep_work[actor] = {
          step, command,
          actor < int(private_state.inventories.size())
              ? private_state.inventories[actor]
              : std::array<int32_t, fastkag::N_ITEMS>{}};
    }
    result = sheep_rescue(env, player, state, std::move(result));
    auto stock = projected_shed(env, player, result);
    for (int credit = 0; credit < 2; ++credit) {
      const Item item = credit == 0 ? Item::WOOL : Item::FERTILIZER;
      int already_sold = 0;
      for (const auto& row : result.market)
        if (row.op == Op::SELL && row.item == item) already_sold += count(row);
      const int quantity = std::min(
          state.sheep_credit[credit],
          std::max(0, stock[int(item)] - already_sold));
      if (quantity > 0 && result.market.size() < kMaxOrders) {
        result.market.push_back({Op::SELL, item, quantity});
        state.sheep_credit[credit] -= quantity;
      }
    }
    return result;
  }

  std::set<std::pair<int, int>> planned_feed_tiles(
      const SeatState& state, int day) const {
    std::vector<Position> positions{{4, 4}};
    std::vector<int> wheat{0};
    std::set<std::pair<int, int>> result;
    for (int hour = 0; hour < 24; ++hour) {
      const int step = day * 24 + hour;
      if (step >= kFrames) break;
      const auto& planned = tape_action(state, step);
      for (int actor = 0;
           actor < std::min<int>(positions.size(), planned.units.size());
           ++actor) {
        const auto& unit = planned.units[actor];
        const Position position = positions[actor];
        if (movement(unit.op)) {
          positions[actor] = moved(position, unit.op);
        } else if (unit.op == Op::PICKUP && unit.item == Item::WHEAT &&
                   access(position)) {
          wheat[actor] += count(unit);
        } else if (unit.op == Op::FEED && wheat[actor] > 0) {
          --wheat[actor];
          if (hour <= 21) result.emplace(position.x, position.y);
        } else if (unit.op == Op::DROP && access(position)) {
          wheat[actor] = 0;
        } else if (unit.op == Op::PLACE && unit.item == Item::WHEAT &&
                   access(position)) {
          wheat[actor] = std::max(0, wheat[actor] - count(unit));
        }
      }
      for (const auto& row : planned.market) {
        if (row.op != Op::HIRE) continue;
        const Position spawn = *std::min_element(
            kAccess.begin(), kAccess.end(), [&](Position left, Position right) {
              const int left_count = std::count_if(
                  positions.begin(), positions.end(), [&](Position value) {
                    return same_position(value, left);
                  });
              const int right_count = std::count_if(
                  positions.begin(), positions.end(), [&](Position value) {
                    return same_position(value, right);
                  });
              const auto left_index = std::find_if(
                  kAccess.begin(), kAccess.end(), [&](Position value) {
                    return same_position(value, left);
                  }) - kAccess.begin();
              const auto right_index = std::find_if(
                  kAccess.begin(), kAccess.end(), [&](Position value) {
                    return same_position(value, right);
                  }) - kAccess.begin();
              return std::tuple{left_count, left_index} <
                     std::tuple{right_count, right_index};
            });
        positions.push_back(spawn);
        wheat.push_back(0);
      }
    }
    return result;
  }

  static int feed_bonus_cost(const Tile& tile, int day) {
    int first = 0, period = 1;
    if (tile.animal == Item::GOOSE) std::tie(first, period) = std::tuple{4, 1};
    else if (tile.animal == Item::COW) std::tie(first, period) = std::tuple{8, 2};
    else if (tile.animal == Item::SHEEP) std::tie(first, period) = std::tuple{6, 3};
    first += tile.placed_day;
    const int tomorrow = day + 1;
    const bool produces = tomorrow >= first &&
                          (tomorrow - first) % period == 0;
    const int pending = produces ? std::max(0, int(tile.pending_care_bonus)) : 0;
    while (first <= tomorrow) first += period;
    return pending + int(first <= 29);
  }

  PlayerAction feed_economics(const Simulator& env, int player,
                              SeatState& state, PlayerAction result) const {
    const int step = env.step_count();
    const int day = env.day();
    if (day >= 10 && day <= 28 && env.hour() <= 21) {
      int planned_hands = 0;
      for (int future = day * 24;
           future < std::min(kFrames, day * 24 + 24); ++future)
        planned_hands = std::max<int>(
            planned_hands, tape_action(state, future).units.size() - 1);
      const auto positions = unit_positions(env.farms()[player]);
      const auto& inventories = env.privates()[player].inventories;
      auto units = result.units;
      const auto tomorrow_feeds = day < 28
          ? planned_feed_tiles(state, day + 1)
          : std::set<std::pair<int, int>>{};
      bool changed = false;
      for (int actor = 0;
           actor < std::min<int>({int(units.size()), planned_hands + 1,
                                  int(positions.size()), int(inventories.size())});
           ++actor) {
        if (units[actor].op != Op::FEED) continue;
        const auto* tile = tile_at(env.farms()[player], positions[actor]);
        if (!tile || animal_product(tile->animal) < 0 || tile->fed_today ||
            tile->consecutive_unfed != 0 ||
            inventories[actor][int(Item::WHEAT)] <= 0)
          continue;
        const int product = animal_product(tile->animal);
        const int bonus = feed_bonus_cost(*tile, day);
        if (bonus * (double(env.market().prices[product]) + 5.0) * 1.25 >=
            double(env.market().prices[int(Item::WHEAT)]))
          continue;
        if (day < 28 && !tomorrow_feeds.contains(
                {positions[actor].x, positions[actor].y}))
          continue;
        units[actor] = {};
        ++state.feed_saved;
        changed = true;
      }
      if (changed) result.units = std::move(units);
    }

    if (day != state.feed_day) {
      state.feed_pending += state.feed_saved;
      state.feed_saved = 0;
      state.feed_day = day;
    }
    if (state.feed_pending <= 0 || env.hour() < 1 || env.hour() > 20 ||
        step >= 712)
      return result;
    auto& rows = result.market;
    if (std::any_of(rows.begin(), rows.end(), [](const Action& row) {
          return row.op == Op::BUY_PRODUCT && row.item == Item::WHEAT;
        }))
      return result;
    const auto projected = projected_shed(env, player, result);
    int sold = 0;
    for (const auto& row : rows)
      if (row.op == Op::SELL && row.item == Item::WHEAT) sold += count(row);
    const int quantity = std::min(
        state.feed_pending,
        std::max(0, projected[int(Item::WHEAT)] - sold));
    if (quantity <= 0) return result;
    auto existing = std::find_if(rows.begin(), rows.end(), [](const Action& row) {
      return row.op == Op::SELL && row.item == Item::WHEAT;
    });
    if (existing != rows.end()) existing->quantity += quantity;
    else {
      if (rows.size() >= kMaxOrders) return result;
      rows.push_back({Op::SELL, Item::WHEAT, quantity});
    }
    state.feed_pending -= quantity;
    return result;
  }

  PlayerAction action(const Simulator& env, int player) {
    if (player < 0 || player > 1) throw std::invalid_argument("seat must be 0 or 1");
    const int step = env.step_count();
    if (step < 0 || step >= kFrames)
      throw std::out_of_range("Fieldcraft action requested outside live game");
    auto& state = seats[player];
    if (state.action_step < 0 && step != 0)
      throw std::logic_error("Fieldcraft episode must start at step 0");
    if (step == 0) reset_state(state);
    if (state.action_step >= 0 && step != state.action_step + 1)
      throw std::logic_error("Fieldcraft action stream is not monotonic");
    state.action_step = step;
    select_route(env, state);
    auto result = base_agent(env, player, state);
    result = garden_layer(env, player, state, std::move(result));
    result = input_hand_layer(env, player, state, std::move(result));
    result = input_surplus(env, player, std::move(result));
    result = sheep_layer(env, player, state, std::move(result));
    result = feed_economics(env, player, state, std::move(result));
    // Match the frozen wrapper's final projected quote + row hygiene pass.
    order_rows(result, env, player, true, true, true);
    return result;
  }
};

Opponent::Opponent(const std::string& asset_path)
    : impl_(std::make_unique<Impl>(asset_path)) {}
Opponent::~Opponent() = default;
Opponent::Opponent(Opponent&&) noexcept = default;
Opponent& Opponent::operator=(Opponent&&) noexcept = default;
fastkag::PlayerAction Opponent::action(const fastkag::Simulator& env, int player) {
  return impl_->action(env, player);
}
int Opponent::route(int player) const {
  if (player < 0 || player > 1) throw std::invalid_argument("seat must be 0 or 1");
  return impl_->seats[player].route_id;
}
void Opponent::reset() { impl_->reset(); }

}  // namespace fieldcraft_2887
