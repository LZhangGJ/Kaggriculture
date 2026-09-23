// Reuse the repository's exact replay executor and Thomas-family overlays.
// Including the implementation is intentional: the family overlay is kept in
// its anonymous namespace and has no public ABI.  This experiment builds a
// separate DSO and does not change the production executor.
#include "../../../fast_kaggriculture/src/native_teammate.cpp"

#include "metav4_2965.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metav4_2965 {
namespace {

class Reader {
 public:
  explicit Reader(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open 2965 asset: " + path);
    const auto size = stream.tellg();
    if (size < 0) throw std::runtime_error("cannot size 2965 asset");
    bytes_.resize(static_cast<std::size_t>(size));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(bytes_.data()), size);
    if (!stream) throw std::runtime_error("cannot read 2965 asset");
  }

  template <class T>
  T little() {
    static_assert(std::is_integral_v<T>);
    using U = std::make_unsigned_t<T>;
    need(sizeof(T));
    U value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i)
      value |= U(bytes_[offset_++]) << (8 * i);
    return static_cast<T>(value);
  }

  void expect(const char* text, std::size_t size) {
    need(size);
    if (!std::equal(bytes_.begin() + offset_, bytes_.begin() + offset_ + size,
                    reinterpret_cast<const std::uint8_t*>(text)))
      throw std::runtime_error("2965 asset magic mismatch");
    offset_ += size;
  }

  void skip(std::size_t size) { need(size); offset_ += size; }
  bool done() const noexcept { return offset_ == bytes_.size(); }

 private:
  void need(std::size_t count) const {
    if (count > bytes_.size() - offset_)
      throw std::runtime_error("truncated 2965 asset");
  }
  std::vector<std::uint8_t> bytes_;
  std::size_t offset_{};
};

fastkag::Action read_action(Reader& reader) {
  const int op = reader.little<std::int8_t>();
  const int item = reader.little<std::int8_t>();
  const int quantity = reader.little<std::int32_t>();
  if (op < int(fastkag::Op::PASS) || op > int(fastkag::Op::SELL) ||
      item < int(fastkag::Item::NONE) || item > int(fastkag::Item::SHEEP))
    throw std::runtime_error("invalid action in 2965 asset");
  return {fastkag::Op(op), fastkag::Item(item), quantity};
}

fastkag::PlayerAction read_player_action(Reader& reader) {
  fastkag::PlayerAction result;
  const int units = reader.little<std::uint8_t>();
  const int market = reader.little<std::uint8_t>();
  result.units.reserve(units);
  result.market.reserve(market);
  for (int i = 0; i < units; ++i) result.units.push_back(read_action(reader));
  for (int i = 0; i < market; ++i) result.market.push_back(read_action(reader));
  return result;
}

struct Assets {
  fastkag::NativeTapeLibrary library;
  std::array<int, 129> route_to_slot{};
  std::array<std::int16_t, 64> new_routes{};
  std::array<std::int16_t, 64> old_routes{};
  std::array<std::int16_t, 64> v92_routes{};
  struct RivalOverride {
    std::int64_t money_milli{};
    std::int32_t wheat{};
    std::int16_t route{};
  };
  std::vector<RivalOverride> rival_overrides;
};

Assets load_assets(const std::string& path) {
  Reader reader(path);
  reader.expect("MV42965\0", 8);
  if (reader.little<std::uint32_t>() != 1)
    throw std::runtime_error("unsupported 2965 asset version");
  reader.skip(32);  // authoritative Python source SHA-256

  Assets result;
  result.route_to_slot.fill(-1);
  const int route_count = reader.little<std::uint16_t>();
  if (route_count != 41) throw std::runtime_error("unexpected 2965 route count");
  result.library.routes.reserve(route_count);
  for (int slot = 0; slot < route_count; ++slot) {
    const int route_id = reader.little<std::int16_t>();
    const int frame_count = reader.little<std::uint16_t>();
    if (route_id < 0 || route_id >= int(result.route_to_slot.size()) ||
        frame_count != 719 || result.route_to_slot[route_id] >= 0)
      throw std::runtime_error("invalid 2965 route header");
    result.route_to_slot[route_id] = slot;
    auto& tape = result.library.routes.emplace_back();
    tape.reserve(frame_count);
    for (int frame = 0; frame < frame_count; ++frame)
      tape.push_back(read_player_action(reader));
  }
  if (result.route_to_slot[0] < 0 || result.route_to_slot[2] < 0)
    throw std::runtime_error("2965 asset lacks mandatory routes 0/2");

  for (auto& pair : result.library.thomas_predict_pairs) {
    const int stream_count = reader.little<std::uint16_t>();
    pair.reserve(stream_count);
    for (int stream = 0; stream < stream_count; ++stream) {
      const int event_count = reader.little<std::uint16_t>();
      auto& events = pair.emplace_back();
      events.reserve(event_count);
      for (int event = 0; event < event_count; ++event) {
        const auto step = reader.little<std::int16_t>();
        const auto item = reader.little<std::int8_t>();
        const auto quantity = reader.little<std::int16_t>();
        if (step < 0 || step >= 719 || item < 0 || item >= fastkag::N_PRODUCTS)
          throw std::runtime_error("invalid 2965 predictor event");
        events.push_back({step, item, quantity});
      }
    }
  }
  for (auto* table : {&result.new_routes, &result.old_routes, &result.v92_routes})
    for (auto& value : *table) value = reader.little<std::int16_t>();

  const int override_count = reader.little<std::uint16_t>();
  result.rival_overrides.reserve(override_count);
  for (int i = 0; i < override_count; ++i)
    result.rival_overrides.push_back({reader.little<std::int64_t>(),
                                      reader.little<std::int32_t>(),
                                      reader.little<std::int16_t>()});
  if (!reader.done()) throw std::runtime_error("trailing bytes in 2965 asset");

  // These references are irrelevant to the public 2965 stack.  Empty 719
  // frame tapes keep the shared executor's optional family counters inert.
  result.library.r5_reference.resize(719);
  result.library.md_reference.resize(719);
  for (auto& tape : result.library.moon) tape.resize(719);
  for (auto& tape : result.library.moon_legacy) tape.resize(719);
  return result;
}

struct SharedAssets {
  explicit SharedAssets(Assets value)
      : assets(std::move(value)), executor(assets.library) {}

  Assets assets;
  fastkag::NativeTeammateExecutor executor;
};

std::shared_ptr<const SharedAssets> shared_assets(const std::string& path) {
  static std::mutex mutex;
  static std::unordered_map<
      std::string, std::shared_ptr<const SharedAssets>> cache;
  const std::lock_guard lock(mutex);
  auto& value = cache[path];
  if (!value)
    value = std::make_shared<const SharedAssets>(load_assets(path));
  return value;
}

bool same(const fastkag::Action& action, fastkag::Op op,
          fastkag::Item item = fastkag::Item::NONE) {
  return action.op == op && (item == fastkag::Item::NONE || action.item == item);
}

}  // namespace

struct Opponent::Impl {
  struct SheepPending {
    int first{};
    bool initial{};
    int count{2};
    std::vector<fastkag::Position> targets;
  };
  struct SheepWork {
    int step{-1};
    fastkag::Action command{};
    int wool_before{};
    int fertilizer_before{};
  };
  struct SheepState {
    int day{-1};
    int requested_day{-1};
    int rescue_today{};
    bool committed{};
    std::optional<SheepPending> pending;
    std::vector<std::pair<int, std::vector<fastkag::Position>>> workers;
    std::vector<std::pair<int, SheepWork>> work;
    int wool_credit{};
    int fertilizer_credit{};
  };
  struct V219Labor {
    std::vector<std::vector<fastkag::Position>> paths;
    std::vector<fastkag::Position> spawns;
    int workers{};
    bool fertilizer{};
  };
  struct V219Pending {
    int step{-1};
    int first_actor{};
    int count{};
    int crop_workers{};
    bool fertilizer{};
    std::optional<V219Labor> labor;
  };
  struct V219Role {
    bool fertilizer_worker{};
    std::vector<fastkag::Position> targets;
    bool needs_fertilizer{};
    bool loaded{};
    bool pickup_requested{};
    int fertilizer_available{};
    int fertilizer_quantity{-1};
  };
  struct V219LastWork {
    int step{-1};
    fastkag::Action command{};
    int tomatoes{};
  };
  struct V219State {
    int last_step{-1};
    int day{-1};
    int requested_day{-1};
    bool eligible{};
    bool committed{};
    std::optional<V219Pending> pending;
    std::vector<std::pair<int, V219Role>> workers;
    std::vector<std::pair<int, V219LastWork>> last_work;
    std::array<bool, 100> seen_plants{};
    std::array<bool, 100> lost{};
  };
  struct CarrotState {
    int step{-1};
    int spare_wheat{};
    int spare_carrot{};
    int credit{};
    std::vector<std::pair<fastkag::Position, int>> tiles;
  };
  struct Or2Tile {
    bool present{};
    bool plant{};
    int item{-1};
    int born{-1};
    int yield{};
  };
  struct Or2State {
    int step{-1};
    std::array<int, 8> stock{};
    bool has_previous{};
    int previous_step{-1};
    std::array<Or2Tile, 100> previous_tiles{};
    std::array<int, 8> previous_inventory{};
    std::array<int, 8> previous_own{};
    std::array<int, 8> previous_prices{};
    std::vector<std::int8_t> previous_shops;
  };
  struct InputTarget {
    fastkag::Position position{};
    fastkag::Item crop{fastkag::Item::NONE};
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
  struct InputPathEntry {
    fastkag::Position position{};
    fastkag::Item crop{fastkag::Item::NONE};
    int birth{};
  };
  struct InputPlan {
    std::vector<InputPathEntry> path;
    int quantity{};
    bool loaded{};
  };
  struct InputState {
    int last_step{-1};
    int day{-1};
    std::vector<std::pair<int, InputPlan>> workers;
    std::vector<std::pair<int, InputPlan>> pending;
  };
  struct V9HerdState {
    int last_step{-1};
    bool decided{};
    fastkag::Item species{fastkag::Item::NONE};
  };
  struct CattleSite {
    fastkag::Position site{};
    int day{-1};
  };
  struct CattleCreditState {
    int last_step{-1};
    std::vector<CattleSite> sites;
    int milk_credit{};
  };
  struct WeedPending {
    fastkag::Position position{};
    fastkag::Action action{};
  };
  struct SeatState {
    fastkag::NativeAgentState native;
    fastkag::ThomasPrefixMarketState prefix;
    int route_id{0};
    bool routed{};
    bool rival_key_seen{};
    std::int64_t rival_money_milli{};
    std::int32_t rival_wheat{};
    int last_step{-1};
    SheepState sheep;
    V219State tomato;
    CarrotState carrot;
    Or2State or2;
    InputState input;
    V9HerdState v9_herd;
    CattleCreditState cattle_credit;
    std::vector<std::vector<WeedPending>> weed_pending;
    std::array<std::array<int, fastkag::N_PRODUCTS>, 720> racepx_debts{};
    std::array<std::array<int, fastkag::N_PRODUCTS>, 720> sale_debts{};
  };

  explicit Impl(const std::string& path)
      : shared(shared_assets(path)), assets(shared->assets),
        executor(shared->executor) {}

  void reset() { seats = {}; }

  fastkag::PlayerAction fixture_route_action(int route_id, int step) const {
    const auto& tape = assets.library.routes[slot_for(route_id)];
    if (step < 0 || step >= int(tape.size()))
      throw std::out_of_range("2965 fixture step outside route tape");
    return tape[step];
  }

  int slot_for(int route_id) const {
    if (route_id < 0 || route_id >= int(assets.route_to_slot.size()) ||
        assets.route_to_slot[route_id] < 0)
      throw std::runtime_error("2965 router selected a missing route");
    return assets.route_to_slot[route_id];
  }

  void update_route(const fastkag::Simulator& env, int player, SeatState& state) {
    const int step = env.step_count();
    if (step == 2 && !state.rival_key_seen) {
      state.rival_money_milli = static_cast<std::int64_t>(
          std::nearbyint(env.farms()[1 - player].money * 1000.0));
      state.rival_wheat = env.market().inventory[int(fastkag::Item::WHEAT)];
      state.rival_key_seen = true;
    }
    if (step >= 144 && !state.routed) {
      int selected = 100;
      if (env.shops().size() >= 2) {
        const int pair = int(env.shops()[0]) * 8 + int(env.shops()[1]);
        const bool yarn = env.shops()[0] == 7 || env.shops()[1] == 7;
        selected = yarn ? assets.old_routes[pair] : assets.new_routes[pair];
        if (selected < 0) selected = yarn ? 0 : 100;
        if (assets.v92_routes[pair] >= 0) selected = assets.v92_routes[pair];
      }
      for (const auto& override : assets.rival_overrides)
        if (state.rival_key_seen && override.money_milli == state.rival_money_milli &&
            override.wheat == state.rival_wheat)
          selected = override.route;
      state.route_id = selected;
      state.routed = true;
    }
    if (step >= 648) state.route_id = 2;
  }

  static int distance(fastkag::Position left, fastkag::Position right) {
    return std::abs(int(left.x) - int(right.x)) +
           std::abs(int(left.y) - int(right.y));
  }

  static fastkag::Action walk(fastkag::Position from, fastkag::Position to) {
    if (from.x != to.x)
      return {from.x < to.x ? fastkag::Op::EAST : fastkag::Op::WEST};
    if (from.y != to.y)
      return {from.y < to.y ? fastkag::Op::SOUTH : fastkag::Op::NORTH};
    return {};
  }

  static fastkag::Position home(fastkag::Position position) {
    static constexpr std::array<fastkag::Position, 4> access{{
        {4, 4}, {5, 4}, {4, 5}, {5, 5}}};
    auto best = access[0];
    int best_distance = distance(position, best);
    for (std::size_t i = 1; i < access.size(); ++i) {
      const int candidate = distance(position, access[i]);
      if (candidate < best_distance) {
        best = access[i];
        best_distance = candidate;
      }
    }
    return best;
  }

  static int fib(int n) {
    int a = 1, b = 1;
    for (int i = 0; i < n; ++i) {
      const int next = a + b;
      a = b;
      b = next;
    }
    return a;
  }

  static int count_market(const fastkag::PlayerAction& action, fastkag::Op op) {
    return std::count_if(action.market.begin(), action.market.end(),
                         [op](const auto& order) { return order.op == op; });
  }

  static int sale_quantity(const fastkag::PlayerAction& action,
                           fastkag::Item item) {
    int total = 0;
    for (const auto& order : action.market)
      if (same(order, fastkag::Op::SELL, item))
        total += std::max(0, order.quantity);
    return total;
  }

  static void cap_sale_quantity(fastkag::PlayerAction& action,
                                fastkag::Item item, int keep) {
    for (auto iterator = action.market.begin(); iterator != action.market.end();) {
      if (!same(*iterator, fastkag::Op::SELL, item)) {
        ++iterator;
        continue;
      }
      const int quantity = std::max(0, iterator->quantity);
      const int retained = std::min(quantity, std::max(0, keep));
      keep -= retained;
      if (retained == 0) iterator = action.market.erase(iterator);
      else {
        iterator->quantity = retained;
        ++iterator;
      }
    }
  }

  static int count_item(const std::array<int, fastkag::N_ITEMS>& values) {
    return std::accumulate(values.begin(), values.end(), 0);
  }

  static std::array<int, fastkag::N_ITEMS> python_projected_shed(
      const fastkag::Simulator& env, int player,
      const fastkag::PlayerAction& action) {
    auto projected = env.privates()[player].shed;
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    std::vector<fastkag::Position> positions{farm.farmer};
    positions.insert(positions.end(), farm.hands.begin(), farm.hands.end());
    auto adjacent = [](fastkag::Position p) {
      return (p.x == 4 || p.x == 5) && (p.y == 4 || p.y == 5);
    };
    int total = count_item(projected);
    for (std::size_t actor = 0;
         actor < action.units.size() && actor < positions.size() &&
         actor < private_state.inventories.size(); ++actor) {
      if (!adjacent(positions[actor])) continue;
      const auto& command = action.units[actor];
      const auto& inventory = private_state.inventories[actor];
      const int item = int(command.item);
      if (command.op == fastkag::Op::PICKUP && item >= 0 &&
          item < fastkag::N_ITEMS) {
        const int take = std::min(projected[item], std::max(0, command.quantity));
        projected[item] -= take;
        total -= take;
      } else if (command.op == fastkag::Op::DROP) {
        for (int raw : private_state.inventory_order[actor]) {
          const int take = std::min(std::max(0, inventory[raw]),
                                    std::max(0, 100 - total));
          projected[raw] += take;
          total += take;
        }
      } else if (command.op == fastkag::Op::PLACE && item >= 0 &&
                 item < int(fastkag::Item::GOOSE)) {
        const int take = std::min({std::max(0, command.quantity),
                                   std::max(0, inventory[item]),
                                   std::max(0, 100 - total)});
        projected[item] += take;
        total += take;
      }
    }
    return projected;
  }

  static const fastkag::Tile& tile(const fastkag::Simulator& env, int player,
                                   fastkag::Position position) {
    return env.farms()[player].tiles[
        int(position.y) * env.config().board_size + int(position.x)];
  }

  static bool is_sheep(const fastkag::Tile& value) {
    return value.kind == fastkag::TileKind::ANIMAL &&
           value.animal == fastkag::Item::SHEEP;
  }

  static bool unit_noop(const fastkag::Simulator& env, int player,
                        std::size_t actor, const fastkag::Action& action) {
    const auto positions = fastkag::positions(env, player);
    if (actor >= positions.size()) return true;
    const auto position = positions[actor];
    const auto& value = tile(env, player, position);
    const auto& private_state = env.privates()[player];
    static const std::array<int, fastkag::N_ITEMS> empty{};
    const auto& inventory = actor < private_state.inventories.size()
        ? private_state.inventories[actor] : empty;
    const int item = int(action.item);
    if (action.op == fastkag::Op::NORTH || action.op == fastkag::Op::SOUTH ||
        action.op == fastkag::Op::EAST || action.op == fastkag::Op::WEST) {
      auto target = position;
      if (action.op == fastkag::Op::NORTH) --target.y;
      else if (action.op == fastkag::Op::SOUTH) ++target.y;
      else if (action.op == fastkag::Op::EAST) ++target.x;
      else --target.x;
      return target.x < 0 || target.x >= env.config().board_size ||
             target.y < 0 || target.y >= env.config().board_size;
    }
    if (action.op == fastkag::Op::PASS) return true;
    const bool adjacent = fastkag::shed_adjacent(position);
    if (action.op == fastkag::Op::DROP)
      return !adjacent || !std::any_of(
          inventory.begin(), inventory.end(), [](int amount) {
            return amount > 0;
          });
    if (action.op == fastkag::Op::PICKUP) return !adjacent;
    if (action.op == fastkag::Op::PLACE) {
      if (item >= int(fastkag::Item::GOOSE) &&
          item <= int(fastkag::Item::SHEEP)) {
        const auto structure = action.item == fastkag::Item::GOOSE
            ? fastkag::TileKind::COOP : fastkag::TileKind::PASTURE;
        if (value.kind == structure && value.animal == fastkag::Item::NONE)
          return item < 0 || inventory[item] <= 0;
      }
      return !adjacent || item < 0 || item >= fastkag::N_ITEMS ||
             inventory[item] <= 0;
    }
    if (value.kind == fastkag::TileKind::LOCKED) return true;
    if (action.op == fastkag::Op::PLANT)
      return value.kind != fastkag::TileKind::EMPTY || item < 0 ||
             item >= fastkag::N_CROPS || private_state.seeds[item] <= 0;
    if (action.op == fastkag::Op::WATER)
      return value.kind != fastkag::TileKind::PLANT || value.watered_today;
    if (action.op == fastkag::Op::HARVEST)
      return value.kind == fastkag::TileKind::EMPTY || value.yield_units <= 0;
    if (action.op == fastkag::Op::FERTILIZE)
      return value.kind != fastkag::TileKind::PLANT ||
             inventory[int(fastkag::Item::FERTILIZER)] <= 0;
    if (action.op == fastkag::Op::DIG)
      return value.kind == fastkag::TileKind::EMPTY ||
             value.kind == fastkag::TileKind::ANIMAL;
    if (action.op == fastkag::Op::BUILD_COOP ||
        action.op == fastkag::Op::BUILD_PASTURE)
      return value.kind != fastkag::TileKind::EMPTY;
    if (action.op == fastkag::Op::FEED)
      return value.kind != fastkag::TileKind::ANIMAL || value.fed_today ||
             inventory[int(fastkag::Item::WHEAT)] <= 0;
    if (action.op == fastkag::Op::COLLECT_FERTILIZER)
      return value.kind != fastkag::TileKind::ANIMAL ||
             !value.fertilizer_available;
    if (action.op == fastkag::Op::CARE)
      return value.kind != fastkag::TileKind::ANIMAL || value.cared_today;
    return true;
  }

  static void apply_exact_weed_repair(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action, SeatState& state) {
    const auto positions = fastkag::positions(env, player);
    action.units.resize(positions.size());
    state.weed_pending.resize(positions.size());
    const int step = env.step_count();
    const auto movement = [](fastkag::Op op) {
      return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
             op == fastkag::Op::EAST || op == fastkag::Op::WEST;
    };
    for (std::size_t actor = 0; actor < positions.size(); ++actor) {
      const auto raw = actor < tape[step].units.size()
          ? tape[step].units[actor] : fastkag::Action{};
      const auto next = step + 1 < int(tape.size()) &&
                                actor < tape[step + 1].units.size()
          ? tape[step + 1].units[actor] : fastkag::Action{};
      auto& queue = state.weed_pending[actor];
      if (!queue.empty() &&
          (queue.front().position.x != positions[actor].x ||
           queue.front().position.y != positions[actor].y))
        queue.clear();
      const auto& value = tile(env, player, positions[actor]);
      const bool weed = value.kind == fastkag::TileKind::WEED;
      const bool structural = raw.op == fastkag::Op::PLANT ||
                              raw.op == fastkag::Op::BUILD_COOP ||
                              raw.op == fastkag::Op::BUILD_PASTURE;
      if (structural && weed) {
        queue.push_back({positions[actor], raw});
        action.units[actor] = {fastkag::Op::DIG};
      } else if (!queue.empty() && unit_noop(env, player, actor, raw)) {
        const auto replay = queue.front().action;
        if (replay.op == fastkag::Op::PLANT && movement(next.op)) {
          queue.clear();
          action.units[actor] = raw;
        } else {
          queue.erase(queue.begin());
          if (raw.op != fastkag::Op::PASS && !movement(raw.op))
            queue.push_back({positions[actor], raw});
          action.units[actor] = replay;
          if (queue.empty()) queue.clear();
        }
      } else if (weed && unit_noop(env, player, actor, raw)) {
        action.units[actor] = {fastkag::Op::DIG};
      }
    }
  }

  static void undo_shared_feed_guard(
      const fastkag::Simulator& env, int player,
      const fastkag::PlayerAction& tape_action,
      fastkag::PlayerAction& action,
      fastkag::NativeAgentState& native) {
    // NativeTeammateExecutor contains an older unconditional feed-value
    // overlay which is not in this public script.  Restore exactly the FEEDs
    // it suppresses; the later R85 implementation then applies the canonical
    // phase-aware value test.  Its associated wheat-purchase credit is also
    // private to that obsolete overlay.
    const auto positions = fastkag::positions(env, player);
    int restored = 0;
    for (std::size_t actor = 0;
         actor < tape_action.units.size() && actor < action.units.size() &&
         actor < positions.size(); ++actor) {
      if (tape_action.units[actor].op != fastkag::Op::FEED ||
          action.units[actor].op != fastkag::Op::PASS)
        continue;
      const auto& value = tile(env, player, positions[actor]);
      if (value.kind != fastkag::TileKind::ANIMAL || value.fed_today ||
          value.consecutive_unfed != 0)
        continue;
      const int product = value.animal == fastkag::Item::GOOSE ? 5 :
                          value.animal == fastkag::Item::COW ? 6 : 7;
      if (env.market().prices[product] *
              (1 + std::max(0, int(value.pending_care_bonus))) >=
          env.market().prices[int(fastkag::Item::WHEAT)])
        continue;
      action.units[actor] = tape_action.units[actor];
      ++restored;
    }
    if (restored > 0) {
      int tape_buy = 0, actual_buy = 0;
      for (const auto& order : tape_action.market)
        if (same(order, fastkag::Op::BUY_PRODUCT, fastkag::Item::WHEAT))
          tape_buy += std::max(0, order.quantity);
      for (const auto& order : action.market)
        if (same(order, fastkag::Op::BUY_PRODUCT, fastkag::Item::WHEAT))
          actual_buy += std::max(0, order.quantity);
      int missing = std::min(restored, std::max(0, tape_buy - actual_buy));
      if (missing > 0) {
        auto found = std::find_if(action.market.begin(), action.market.end(),
            [](const auto& order) {
              return same(order, fastkag::Op::BUY_PRODUCT,
                          fastkag::Item::WHEAT);
            });
        if (found != action.market.end()) {
          found->quantity += missing;
        } else {
          const auto source = std::find_if(
              tape_action.market.begin(), tape_action.market.end(),
              [](const auto& order) {
                return same(order, fastkag::Op::BUY_PRODUCT,
                            fastkag::Item::WHEAT);
              });
          const std::size_t index = source == tape_action.market.end()
              ? action.market.size()
              : std::min<std::size_t>(
                    std::distance(tape_action.market.begin(), source),
                    action.market.size());
          action.market.insert(action.market.begin() + index,
                               {fastkag::Op::BUY_PRODUCT,
                                fastkag::Item::WHEAT, missing});
        }
      }
    }
    native.wheat_credit = 0;
  }

  static std::vector<fastkag::Position> shortest_path(
      fastkag::Position start, std::vector<fastkag::Position> targets) {
    std::sort(targets.begin(), targets.end(), [](auto left, auto right) {
      return std::tie(left.x, left.y) < std::tie(right.x, right.y);
    });
    std::vector<fastkag::Position> best;
    int best_length = std::numeric_limits<int>::max();
    do {
      int length = targets.empty() ? 0 : distance(start, targets[0]);
      for (std::size_t i = 1; i < targets.size(); ++i)
        length += distance(targets[i - 1], targets[i]);
      if (length < best_length || (length == best_length &&
          std::lexicographical_compare(targets.begin(), targets.end(),
              best.begin(), best.end(), [](auto left, auto right) {
                return std::tie(left.x, left.y) < std::tie(right.x, right.y);
              }))) {
        best_length = length;
        best = targets;
      }
    } while (std::next_permutation(targets.begin(), targets.end(),
        [](auto left, auto right) {
          return std::tie(left.x, left.y) < std::tie(right.x, right.y);
        }));
    return best;
  }

  static fastkag::Position extra_hire_spawn(
      const fastkag::Simulator& env, int player,
      const fastkag::PlayerAction& parent, int extra_index) {
    auto positions = fastkag::positions(env, player);
    for (std::size_t actor = 0;
         actor < positions.size() && actor < parent.units.size(); ++actor) {
      const auto op = parent.units[actor].op;
      if (op == fastkag::Op::NORTH) positions[actor].y = std::max<int>(0, positions[actor].y - 1);
      else if (op == fastkag::Op::SOUTH) positions[actor].y = std::min<int>(9, positions[actor].y + 1);
      else if (op == fastkag::Op::WEST) positions[actor].x = std::max<int>(0, positions[actor].x - 1);
      else if (op == fastkag::Op::EAST) positions[actor].x = std::min<int>(9, positions[actor].x + 1);
    }
    static constexpr std::array<fastkag::Position, 4> access{{
        {4, 4}, {5, 4}, {4, 5}, {5, 5}}};
    fastkag::Position chosen{};
    const int rounds = count_market(parent, fastkag::Op::HIRE) + extra_index + 1;
    for (int round = 0; round < rounds; ++round) {
      int best = 0;
      int best_count = std::numeric_limits<int>::max();
      for (int candidate = 0; candidate < int(access.size()); ++candidate) {
        const int occupied = std::count_if(positions.begin(), positions.end(),
            [&](auto p) { return p.x == access[candidate].x && p.y == access[candidate].y; });
        if (occupied < best_count) {
          best = candidate;
          best_count = occupied;
        }
      }
      chosen = access[best];
      positions.push_back(chosen);
    }
    return chosen;
  }

  bool sheep_eligible(const fastkag::Simulator& env, int player,
                      const std::vector<fastkag::PlayerAction>& tape) const {
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    if (env.config().board_size != 10 || farm.unlocked_mask != 0b0111) return false;
    if (std::count(env.shops().begin(), env.shops().end(), std::int8_t(7)) < 2 ||
        env.market().prices[int(fastkag::Item::WOOL)] < 220 ||
        env.market().prices[int(fastkag::Item::WHEAT)] > 45) return false;
    for (int y : {5, 6})
      for (int x = 5; x < 8; ++x)
        if (farm.tiles[y * 10 + x].kind != fastkag::TileKind::LOCKED) return false;
    int held = private_state.shed[int(fastkag::Item::SHEEP)];
    for (const auto& inventory : private_state.inventories)
      held += inventory[int(fastkag::Item::SHEEP)];

    // The day-11 VE wrapper permits only sheep already accounted for by the
    // tape's still-future pickups.  Day 12 uses the original strict test.
    if (env.day() == 11) {
      int pickups = 0;
      for (int step = env.step_count(); step < 12 * 24; ++step)
        for (const auto& command : tape[step].units)
          if (same(command, fastkag::Op::PICKUP, fastkag::Item::SHEEP))
            pickups += std::max(0, command.quantity);
      if (held > pickups) return false;
    } else if (held != 0) {
      return false;
    }
    for (int step = 12 * 24; step < 719; ++step) {
      for (const auto& order : tape[step].market)
        if (order.op == fastkag::Op::BUY_LAND ||
            same(order, fastkag::Op::BUY_ANIMAL, fastkag::Item::SHEEP))
          return false;
      for (const auto& command : tape[step].units)
        if ((command.op == fastkag::Op::PICKUP ||
             command.op == fastkag::Op::PLACE) &&
            command.item == fastkag::Item::SHEEP)
          return false;
    }
    if (env.day() != 11) return true;

    int spend = 0;
    std::array<int, 30> hires{};
    static constexpr std::array<int, 5> seed_cost{10, 20, 50, 100, 80};
    static constexpr std::array<int, 3> animal_cost{300, 400, 500};
    for (int step = env.step_count() + 1; step < 13 * 24; ++step) {
      for (const auto& order : tape[step].market) {
        const int item = int(order.item);
        const int quantity = std::max(0, order.quantity);
        if (order.op == fastkag::Op::BUY_LAND ||
            same(order, fastkag::Op::BUY_ANIMAL, fastkag::Item::SHEEP))
          return false;
        if (order.op == fastkag::Op::BUY_SEED && item >= 0 && item < 5)
          spend += quantity * seed_cost[item];
        else if (order.op == fastkag::Op::BUY_PRODUCT && item >= 0 && item < 9)
          spend += quantity * (env.market().prices[item] + 10);
        else if (order.op == fastkag::Op::BUY_ANIMAL && item >= 9 && item < 12)
          spend += quantity * animal_cost[item - 9];
        else if (order.op == fastkag::Op::HIRE)
          spend += fib(hires[step / 24]++);
      }
    }
    return farm.money >= 10000 + spend;
  }

  static int expected_hands(const std::vector<fastkag::PlayerAction>& tape,
                            int day) {
    int expected = 0;
    for (int step = day * 24; step < std::min((day + 1) * 24, 719); ++step)
      expected = std::max(expected, int(tape[step].units.size()) - 1);
    return expected;
  }

  fastkag::PlayerAction sheep_request(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction action, SheepState& state) {
    const int step = env.step_count(), day = env.day(), hour = env.hour();
    if (state.requested_day == day) return action;
    if (!state.committed &&
        (hour > (day == 11 ? 3 : 1) || (day != 11 && day != 12) ||
         !sheep_eligible(env, player, tape)))
      return action;

    int deadline = state.committed ? 2 : (day == 11 ? 3 : 1);
    int last_native_hire = 0;
    for (int offset = 0; offset < 24 && day * 24 + offset < 719; ++offset)
      if (count_market(tape[day * 24 + offset], fastkag::Op::HIRE) > 0)
        last_native_hire = offset;
    if (state.committed && last_native_hire > 2 && last_native_hire <= 6)
      deadline = 6;
    if (hour > deadline) return action;
    for (int offset = hour + 1; offset < 24 && day * 24 + offset < 719; ++offset)
      if (count_market(tape[day * 24 + offset], fastkag::Op::HIRE) > 0)
        return action;

    const auto& farm = env.farms()[player];
    const int parent_hires = count_market(action, fastkag::Op::HIRE);
    const int expected = expected_hands(tape, day);
    if (int(farm.hands.size()) + parent_hires != expected) return action;
    const bool initial = !state.committed;
    const int extra_count = initial ? 5 : 3;
    if (int(action.market.size()) + extra_count > 10) return action;
    auto stock = python_projected_shed(env, player, action);
    int incoming = 6 + (initial ? 6 : 0);
    int budget = (initial ? 7000 : 0) +
                 6 * (env.market().prices[int(fastkag::Item::WHEAT)] + 10);
    budget += fib(farm.hires_today + parent_hires) +
              fib(farm.hires_today + parent_hires + 1);
    for (const auto& order : action.market) {
      const int quantity = std::max(0, order.quantity);
      const int item = int(order.item);
      if (order.op == fastkag::Op::BUY_LAND) return action;
      if (order.op == fastkag::Op::BUY_PRODUCT && item >= 0 && item < 9) {
        incoming += quantity;
        budget += quantity * (env.market().prices[item] + 10);
      } else if (order.op == fastkag::Op::BUY_ANIMAL && item >= 9 && item < 12) {
        static constexpr std::array<int, 3> costs{300, 400, 500};
        incoming += quantity;
        budget += quantity * costs[item - 9];
      } else if (order.op == fastkag::Op::BUY_SEED && item >= 0 && item < 5) {
        static constexpr std::array<int, 5> costs{10, 20, 50, 100, 80};
        budget += quantity * costs[item];
      }
    }
    if (count_item(stock) + incoming > 100 ||
        farm.money < budget + (initial ? 3000 : 1000))
      return action;

    state.requested_day = day;
    state.pending = SheepPending{expected + 1, initial, 2, {}};
    if (initial) {
      action.market.push_back({fastkag::Op::BUY_LAND});
      action.market.push_back(
          {fastkag::Op::BUY_ANIMAL, fastkag::Item::SHEEP, 6});
    }
    action.market.push_back(
        {fastkag::Op::BUY_PRODUCT, fastkag::Item::WHEAT, 6});
    action.market.push_back({fastkag::Op::HIRE});
    action.market.push_back({fastkag::Op::HIRE});

    // SL compaction of an already-committed, non-harvest service day.
    if (!initial) {
      static const std::vector<fastkag::Position> all{{5,5},{6,5},{7,5},{7,6},{6,6},{5,6}};
      bool quiet = true;
      for (auto target : all) {
        const auto& value = tile(env, player, target);
        quiet = quiet && is_sheep(value) && value.yield_units == 0;
      }
      if (quiet) {
        const auto spawn = extra_hire_spawn(env, player,
            fastkag::PlayerAction{action.units,
                std::vector<fastkag::Action>(action.market.begin(), action.market.end() - 2)}, 0);
        const auto path = shortest_path(spawn, all);
        int travel = distance(spawn, path[0]);
        for (std::size_t i = 1; i < path.size(); ++i)
          travel += distance(path[i - 1], path[i]);
        int mandatory = 0;
        for (auto target : all) {
          const auto& value = tile(env, player, target);
          mandatory += !value.fed_today;
          mandatory += !value.cared_today;
        }
        const int ready = step + 2;
        const int available = std::min((day + 1) * 24, 719) - ready;
        const int native_hires = parent_hires;
        const int saved = fib(farm.hires_today + native_hires + 1);
        const int delivery = day == 29 ? distance(path.back(), home(path.back())) + 1 : 0;
        const int possible = std::max(0, std::min(6,
            available - travel - mandatory - delivery));
        if (travel + mandatory <= available &&
            saved > (6 - possible) *
                env.market().prices[int(fastkag::Item::FERTILIZER)]) {
          action.market.pop_back();
          state.pending->count = 1;
          state.pending->targets = path;
        }
      }
    }

    // VT owns the final-day request.  It discards feed/two hires and keeps at
    // most one harvest worker, or nothing when no wool exists.
    if (day == 29 && state.committed) {
      std::vector<fastkag::Position> wool;
      static constexpr std::array<fastkag::Position, 6> targets{{
          {5,5},{6,5},{7,5},{5,6},{6,6},{7,6}}};
      for (auto target : targets)
        if (is_sheep(tile(env, player, target)) && tile(env, player, target).yield_units > 0)
          wool.push_back(target);
      // Recover the parent action by removing the request's suffix.
      const int remove = initial ? 5 : (state.pending->count == 1 ? 2 : 3);
      auto parent = action;
      parent.market.resize(parent.market.size() - remove);
      if (wool.empty()) {
        state.pending.reset();
        return parent;
      }
      action = parent;
      action.market.push_back({fastkag::Op::HIRE});
      state.pending->count = 1;
      state.pending->targets = shortest_path(extra_hire_spawn(env, player, parent, 0), wool);
    }
    return action;
  }

  fastkag::Action original_sheep_worker(
      const fastkag::Simulator& env, int player, int actor,
      const std::vector<fastkag::Position>& targets) const {
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    if (actor <= 0 || actor > int(farm.hands.size()) ||
        actor >= int(private_state.inventories.size())) return {};
    const auto position = farm.hands[actor - 1];
    const auto& inventory = private_state.inventories[actor];
    const auto shed_home = home(position);
    const int home_distance = distance(position, shed_home);
    const bool wool = inventory[int(fastkag::Item::WOOL)] > 0;
    const bool fertilizer = inventory[int(fastkag::Item::FERTILIZER)] > 0;
    if ((wool || fertilizer) &&
        env.hour() >= (env.day() == 29 ? 22 : 23) - home_distance) {
      const auto move = walk(position, shed_home);
      if (move.op != fastkag::Op::PASS) return move;
      const auto item = wool ? fastkag::Item::WOOL : fastkag::Item::FERTILIZER;
      return {fastkag::Op::PLACE, item, inventory[int(item)]};
    }
    int missing = 0;
    for (auto target : targets) missing += !is_sheep(tile(env, player, target));
    if (missing && inventory[int(fastkag::Item::SHEEP)] == 0 &&
        private_state.shed[int(fastkag::Item::SHEEP)] > 0) {
      const auto move = walk(position, shed_home);
      if (move.op != fastkag::Op::PASS) return move;
      return {fastkag::Op::PICKUP, fastkag::Item::SHEEP,
              std::min(missing, private_state.shed[int(fastkag::Item::SHEEP)])};
    }
    int hungry = 0;
    for (auto target : targets) {
      const auto& value = tile(env, player, target);
      // The public worker deliberately preloads feed while the target cells
      // are still empty/pasture: Python's `not(dict and fed_today)` counts
      // every not-yet-occupied target as hungry.
      hungry += !is_sheep(value) || !value.fed_today;
    }
    if (hungry && inventory[int(fastkag::Item::WHEAT)] == 0 &&
        private_state.shed[int(fastkag::Item::WHEAT)] > 0) {
      const auto move = walk(position, shed_home);
      if (move.op != fastkag::Op::PASS) return move;
      return {fastkag::Op::PICKUP, fastkag::Item::WHEAT,
              std::min(hungry, private_state.shed[int(fastkag::Item::WHEAT)])};
    }

    int best_distance = std::numeric_limits<int>::max();
    int best_index = std::numeric_limits<int>::max();
    fastkag::Position best_target{};
    fastkag::Action best_command{};
    for (int index = 0; index < int(targets.size()); ++index) {
      const auto target = targets[index];
      const auto& value = tile(env, player, target);
      fastkag::Action command{};
      if (value.kind == fastkag::TileKind::EMPTY)
        command = {fastkag::Op::BUILD_PASTURE};
      else if (value.kind == fastkag::TileKind::WEED)
        command = {fastkag::Op::DIG};
      else if (value.kind == fastkag::TileKind::PASTURE &&
               inventory[int(fastkag::Item::SHEEP)] > 0)
        command = {fastkag::Op::PLACE, fastkag::Item::SHEEP, 1};
      else if (is_sheep(value)) {
        if (!value.fed_today && inventory[int(fastkag::Item::WHEAT)] > 0)
          command = {fastkag::Op::FEED};
        else if (!value.cared_today)
          command = {fastkag::Op::CARE};
        else if (value.yield_units > 0)
          command = {fastkag::Op::HARVEST};
        else if (value.fertilizer_available)
          command = {fastkag::Op::COLLECT_FERTILIZER};
      }
      if (command.op == fastkag::Op::PASS) continue;
      const int candidate_distance = distance(position, target);
      if (std::tie(candidate_distance, index) <
          std::tie(best_distance, best_index)) {
        best_distance = candidate_distance;
        best_index = index;
        best_target = target;
        best_command = command;
      }
    }
    if (best_index != std::numeric_limits<int>::max()) {
      const auto move = walk(position, best_target);
      return move.op == fastkag::Op::PASS ? best_command : move;
    }
    if (wool || fertilizer) {
      const auto move = walk(position, shed_home);
      if (move.op != fastkag::Op::PASS) return move;
      const auto item = wool ? fastkag::Item::WOOL : fastkag::Item::FERTILIZER;
      return {fastkag::Op::PLACE, item, inventory[int(item)]};
    }
    return {};
  }

  fastkag::Action compact_sheep_worker(
      const fastkag::Simulator& env, int player, int actor,
      const std::vector<fastkag::Position>& targets) const {
    if (targets.size() != 6)
      return original_sheep_worker(env, player, actor, targets);
    for (auto target : targets)
      if (!is_sheep(tile(env, player, target)))
        return original_sheep_worker(env, player, actor, targets);
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    const auto position = farm.hands[actor - 1];
    const auto& inventory = private_state.inventories[actor];
    int hungry = 0;
    std::vector<fastkag::Position> needed;
    for (auto target : targets) {
      const auto& value = tile(env, player, target);
      hungry += !value.fed_today;
      if (!value.fed_today || !value.cared_today) needed.push_back(target);
    }
    const auto shed_home = home(position);
    if (hungry > inventory[int(fastkag::Item::WHEAT)]) {
      const auto move = walk(position, shed_home);
      if (move.op != fastkag::Op::PASS) return move;
      return {fastkag::Op::PICKUP, fastkag::Item::WHEAT,
              std::min(hungry, private_state.shed[int(fastkag::Item::WHEAT)])};
    }
    if (!needed.empty()) {
      const auto path = shortest_path(position, needed);
      const auto& current = tile(env, player, position);
      const bool on_target = std::any_of(targets.begin(), targets.end(),
          [&](auto target) { return target.x == position.x && target.y == position.y; });
      if (on_target && current.fed_today && current.cared_today &&
          current.fertilizer_available) {
        int travel = distance(position, path[0]);
        for (std::size_t i = 1; i < path.size(); ++i)
          travel += distance(path[i - 1], path[i]);
        int work = 0;
        for (auto target : path) {
          const auto& value = tile(env, player, target);
          work += !value.fed_today;
          work += !value.cared_today;
        }
        const int delivery = env.day() == 29
            ? distance(path.back(), home(path.back())) + 1 : 0;
        const int remaining = std::min((env.day() + 1) * 24, 719) - env.step_count();
        if (1 + travel + work + delivery <= remaining)
          return {fastkag::Op::COLLECT_FERTILIZER};
      }
      const auto target = path[0];
      const auto move = walk(position, target);
      if (move.op != fastkag::Op::PASS) return move;
      return {tile(env, player, target).fed_today
                  ? fastkag::Op::CARE : fastkag::Op::FEED};
    }
    const int remaining = env.day() == 29 ? 719 - env.step_count() : 24 - env.hour();
    int best_distance = std::numeric_limits<int>::max();
    fastkag::Position best{};
    bool found = false;
    for (auto target : targets) {
      const auto& value = tile(env, player, target);
      if (!value.fertilizer_available) continue;
      const int outward = distance(position, target);
      const int back = env.day() == 29 ? distance(target, home(target)) + 1 : 0;
      if (outward + 1 + back <= remaining &&
          std::tie(outward, target.x, target.y) <
              std::tie(best_distance, best.x, best.y)) {
        best_distance = outward;
        best = target;
        found = true;
      }
    }
    if (found) {
      const auto move = walk(position, best);
      return move.op == fastkag::Op::PASS
          ? fastkag::Action{fastkag::Op::COLLECT_FERTILIZER} : move;
    }
    if (inventory[int(fastkag::Item::FERTILIZER)] > 0) {
      const auto move = walk(position, shed_home);
      if (move.op != fastkag::Op::PASS) return move;
      return {fastkag::Op::PLACE, fastkag::Item::FERTILIZER,
              inventory[int(fastkag::Item::FERTILIZER)]};
    }
    return {};
  }

  fastkag::Action final_day_sheep_worker(
      const fastkag::Simulator& env, int player, int actor,
      const std::vector<fastkag::Position>& targets) const {
    if (env.day() != 28 && env.day() != 29)
      return compact_sheep_worker(env, player, actor, targets);
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    const auto position = farm.hands[actor - 1];
    const auto& inventory = private_state.inventories[actor];
    const auto shed_home = home(position);
    const bool wool = inventory[int(fastkag::Item::WOOL)] > 0;
    const bool fertilizer = inventory[int(fastkag::Item::FERTILIZER)] > 0;
    const int last = env.day() == 29 ? 717 : env.day() * 24 + 23;
    if ((wool || fertilizer) &&
        env.step_count() >= last - distance(position, shed_home)) {
      const auto move = walk(position, shed_home);
      if (move.op != fastkag::Op::PASS) return move;
      const auto item = wool ? fastkag::Item::WOOL : fastkag::Item::FERTILIZER;
      return {fastkag::Op::PLACE, item, inventory[int(item)]};
    }
    std::vector<fastkag::Position> sheep;
    for (auto target : targets)
      if (is_sheep(tile(env, player, target))) sheep.push_back(target);
    if (env.day() == 28) {
      int hungry = 0;
      for (auto target : sheep) hungry += !tile(env, player, target).fed_today;
      if (hungry && inventory[int(fastkag::Item::WHEAT)] == 0 &&
          private_state.shed[int(fastkag::Item::WHEAT)] > 0) {
        const auto move = walk(position, shed_home);
        if (move.op != fastkag::Op::PASS) return move;
        return {fastkag::Op::PICKUP, fastkag::Item::WHEAT,
                std::min(hungry, private_state.shed[int(fastkag::Item::WHEAT)])};
      }
    }
    int best_distance = std::numeric_limits<int>::max();
    fastkag::Position best{};
    fastkag::Action command{};
    for (auto target : sheep) {
      const auto& value = tile(env, player, target);
      fastkag::Action candidate{};
      if (env.day() == 28 && !value.fed_today &&
          inventory[int(fastkag::Item::WHEAT)] > 0)
        candidate = {fastkag::Op::FEED};
      else if (value.yield_units > 0)
        candidate = {fastkag::Op::HARVEST};
      else if (env.day() == 28 && value.fertilizer_available)
        candidate = {fastkag::Op::COLLECT_FERTILIZER};
      if (candidate.op == fastkag::Op::PASS) continue;
      const int candidate_distance = distance(position, target);
      if (std::tie(candidate_distance, target.x, target.y) <
          std::tie(best_distance, best.x, best.y)) {
        best_distance = candidate_distance;
        best = target;
        command = candidate;
      }
    }
    if (command.op != fastkag::Op::PASS) {
      const auto move = walk(position, best);
      return move.op == fastkag::Op::PASS ? command : move;
    }
    if (wool || fertilizer) {
      const auto move = walk(position, shed_home);
      if (move.op != fastkag::Op::PASS) return move;
      const auto item = wool ? fastkag::Item::WOOL : fastkag::Item::FERTILIZER;
      return {fastkag::Op::PLACE, item, inventory[int(item)]};
    }
    return {};
  }

  fastkag::PlayerAction apply_sheep(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction action, SheepState& state) {
    const int step = env.step_count(), day = env.day();
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    if (day < 11) return action;
    if (state.day != day) {
      state.day = day;
      state.work.clear();
      state.workers.clear();
      state.rescue_today = 0;
    }
    for (const auto& [actor, previous] : state.work) {
      if (previous.step != step - 1 ||
          actor >= int(private_state.inventories.size())) continue;
      const auto& inventory = private_state.inventories[actor];
      if (previous.command.op == fastkag::Op::HARVEST)
        state.wool_credit += std::max(0,
            inventory[int(fastkag::Item::WOOL)] - previous.wool_before);
      else if (previous.command.op == fastkag::Op::COLLECT_FERTILIZER)
        state.fertilizer_credit += std::max(0,
            inventory[int(fastkag::Item::FERTILIZER)] -
            previous.fertilizer_before);
    }
    if (state.pending) {
      auto pending = *state.pending;
      state.pending.reset();
      const bool funded = (farm.unlocked_mask & (1 << 3)) != 0 &&
          (!pending.initial ||
           private_state.shed[int(fastkag::Item::SHEEP)] >= 6);
      if (funded && int(farm.hands.size()) >= pending.first + pending.count - 1) {
        if (pending.count == 1) {
          state.workers.push_back({pending.first, std::move(pending.targets)});
        } else {
          state.workers.push_back({pending.first, {{5,5},{6,5},{7,5}}});
          state.workers.push_back({pending.first + 1, {{5,6},{6,6},{7,6}}});
        }
        if (pending.initial) state.committed = true;
      }
    }
    action = sheep_request(env, player, tape, std::move(action), state);
    if (!state.committed) return action;

    if (action.units.size() < farm.hands.size() + 1)
      action.units.resize(farm.hands.size() + 1);
    state.work.clear();
    for (const auto& [actor, targets] : state.workers) {
      if (actor <= 0 || actor >= int(action.units.size()) ||
          actor >= int(private_state.inventories.size())) continue;
      const auto command = final_day_sheep_worker(env, player, actor, targets);
      action.units[actor] = command;
      const auto& inventory = private_state.inventories[actor];
      state.work.push_back({actor, SheepWork{
          step, command, inventory[int(fastkag::Item::WOOL)],
          inventory[int(fastkag::Item::FERTILIZER)]}});
    }

    // V234 bounded emergency feed purchase (disabled on day 29).
    if (day != 29 && !state.workers.empty() && env.hour() <= 14 &&
        action.market.size() < 10) {
      bool barrier = false;
      for (const auto& order : action.market) {
        if (order.op == fastkag::Op::HIRE || order.op == fastkag::Op::BUY_LAND ||
            order.op == fastkag::Op::BUY_ANIMAL ||
            order.op == fastkag::Op::BUY_PRODUCT ||
            order.op == fastkag::Op::BUY_SEED ||
            order.item == fastkag::Item::WHEAT) barrier = true;
      }
      int hungry = 0, carried = 0;
      for (const auto& [actor, targets] : state.workers) {
        if (actor >= int(action.units.size()) ||
            actor >= int(private_state.inventories.size())) continue;
        const auto command = action.units[actor];
        if (command.op == fastkag::Op::FEED ||
            same(command, fastkag::Op::PICKUP, fastkag::Item::WHEAT))
          barrier = true;
        carried += private_state.inventories[actor][int(fastkag::Item::WHEAT)];
        for (auto target : targets) {
          const auto& value = tile(env, player, target);
          hungry += is_sheep(value) && !value.fed_today;
        }
      }
      if (!barrier) {
        const auto stock = python_projected_shed(env, player, action);
        const int shortage = hungry - carried - stock[int(fastkag::Item::WHEAT)];
        const int quote = env.market().prices[int(fastkag::Item::WHEAT)];
        if (shortage > 0 && shortage <= 6 &&
            state.rescue_today + shortage <= 6 && quote >= 1 &&
            farm.money >= 1000 + shortage * (quote + 10) &&
            count_item(stock) + shortage <= 100) {
          action.market.push_back(
              {fastkag::Op::BUY_PRODUCT, fastkag::Item::WHEAT, shortage});
          state.rescue_today += shortage;
        }
      }
    }

    // Credit liquidation runs at the V233 wrapper point.  Several later
    // wrappers may rewrite unrelated workers (for example to DROP); letting
    // those future-in-chain deposits fund V233 would be a causality error.
    // Reconstruct the route-stage units, retaining only V233's own workers.
    auto credit_action = action;
    credit_action.units = tape[step].units;
    credit_action.units.resize(farm.hands.size() + 1);
    for (const auto& [actor, targets] : state.workers) {
      (void)targets;
      if (actor >= 0 && actor < int(action.units.size()) &&
          actor < int(credit_action.units.size()))
        credit_action.units[actor] = action.units[actor];
    }
    auto stock = python_projected_shed(env, player, credit_action);
    for (const auto item : {fastkag::Item::WOOL, fastkag::Item::FERTILIZER}) {
      int scheduled = 0;
      for (const auto& order : action.market)
        if (same(order, fastkag::Op::SELL, item))
          scheduled += std::max(0, order.quantity);
      int& credit = item == fastkag::Item::WOOL
          ? state.wool_credit : state.fertilizer_credit;
      const int amount = std::min(credit,
          std::max(0, stock[int(item)] - scheduled));
      if (amount > 0 && action.market.size() < 10) {
        action.market.push_back({fastkag::Op::SELL, item, amount});
        credit -= amount;
      }
    }
    return action;
  }

  static bool input_same_position(fastkag::Position left,
                                  fastkag::Position right) {
    return left.x == right.x && left.y == right.y;
  }

  static bool input_is_move(fastkag::Op op) {
    return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
           op == fastkag::Op::EAST || op == fastkag::Op::WEST;
  }

  static fastkag::Position input_move(fastkag::Position position,
                                      fastkag::Op op, int board) {
    if (op == fastkag::Op::NORTH) --position.y;
    else if (op == fastkag::Op::SOUTH) ++position.y;
    else if (op == fastkag::Op::WEST) --position.x;
    else if (op == fastkag::Op::EAST) ++position.x;
    position.x = std::clamp<int>(position.x, 0, board - 1);
    position.y = std::clamp<int>(position.y, 0, board - 1);
    return position;
  }

  static fastkag::Position input_spawn(
      const std::vector<fastkag::Position>& positions) {
    static constexpr std::array<fastkag::Position, 4> access{{
        {4, 4}, {5, 4}, {4, 5}, {5, 5}}};
    auto best = access.front();
    int best_count = std::count_if(
        positions.begin(), positions.end(), [&](auto position) {
          return input_same_position(position, best);
        });
    for (const auto candidate : access) {
      const int count = std::count_if(
          positions.begin(), positions.end(), [&](auto position) {
            return input_same_position(position, candidate);
          });
      if (count < best_count) {
        best = candidate;
        best_count = count;
      }
    }
    return best;
  }

  const fastkag::PlayerAction& input_frame(
      const std::vector<fastkag::PlayerAction>& current_tape,
      int step) const {
    const auto& tape = step >= 648
        ? assets.library.routes[slot_for(2)] : current_tape;
    static const fastkag::PlayerAction empty;
    return step >= 0 && step < int(tape.size()) ? tape[step] : empty;
  }

  std::vector<InputTarget> input_forecast(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& current_tape,
      int expected) const {
    const int step = env.step_count(), day = env.day();
    const auto& farm = env.farms()[player];
    std::vector<InputTarget> targets;
    for (int y = 0; y < env.config().board_size; ++y) {
      for (int x = 0; x < env.config().board_size; ++x) {
        const auto& value =
            farm.tiles[y * env.config().board_size + x];
        if (value.kind != fastkag::TileKind::PLANT ||
            (value.crop != fastkag::Item::WHEAT &&
             value.crop != fastkag::Item::CARROT))
          continue;
        const int last = value.crop == fastkag::Item::WHEAT ? 4 : 3;
        const int age = day - value.planted_day;
        if (age < 1 || age >= last) continue;
        targets.push_back({
            {std::int16_t(x), std::int16_t(y)}, value.crop,
            value.planted_day, value.yield_units,
            value.fertilized_until_day, value.watered_today, {}, -1,
            2, last, value.crop == fastkag::Item::WHEAT ? 6 : 4});
      }
    }

    auto positions = fastkag::positions(env, player);
    if (int(positions.size()) > expected + 1)
      positions.resize(expected + 1);
    std::array<std::array<bool, 100>, 30> seen{};
    for (int future = step;
         future < std::min(712, (day + 4) * 24); ++future) {
      const auto& frame = input_frame(current_tape, future);
      for (int actor = 0; actor < int(positions.size()); ++actor) {
        const auto command = actor < int(frame.units.size())
            ? frame.units[actor] : fastkag::Action{};
        const auto position = positions[actor];
        auto target = std::find_if(
            targets.begin(), targets.end(), [&](const auto& value) {
              return input_same_position(value.position, position);
            });
        if (target != targets.end() && target->harvest_step < 0) {
          const int index = int(position.y) * env.config().board_size +
                            int(position.x);
          if (command.op == fastkag::Op::WATER &&
              !seen[future / 24][index]) {
            seen[future / 24][index] = true;
            const int age = future / 24 - target->birth;
            if (!(future / 24 == day && target->watered) &&
                age >= target->first && age <= target->last)
              target->water_steps.push_back(future);
          }
          if (command.op == fastkag::Op::HARVEST)
            target->harvest_step = future;
        }
        if (input_is_move(command.op))
          positions[actor] = input_move(
              positions[actor], command.op, env.config().board_size);
      }
      for (const auto& order : frame.market)
        if (order.op == fastkag::Op::HIRE)
          positions.push_back(input_spawn(positions));
      if ((future + 1) % 24 == 0) positions = {{4, 4}};
    }
    return targets;
  }

  static int input_gain(const InputTarget& target, int arrival, int day) {
    if (target.harvest_step < 0 || target.harvest_step <= arrival) return 0;
    int extra = 0, baseline = target.yield;
    for (const int water : target.water_steps) {
      const int water_day = water / 24;
      extra += arrival < water && water <= target.harvest_step &&
               water_day >= day && water_day <= day + 2 &&
               water_day > target.fertilized_until;
      baseline += water_day <= target.fertilized_until ? 2 : 1;
    }
    return std::max(0, std::min(extra, target.cap - baseline));
  }

  static bool input_path_less(const std::vector<int>& left,
                              const std::vector<int>& right,
                              const std::vector<InputTarget>& targets) {
    const auto key = [&](int index) {
      const auto& value = targets[index];
      // Python compares the crop strings, so CARROT sorts before WHEAT.
      const int crop = value.crop == fastkag::Item::CARROT ? 0 : 1;
      return std::tuple<int, int, int, int>(
          value.position.x, value.position.y, crop, value.birth);
    };
    return std::lexicographical_compare(
        left.begin(), left.end(), right.begin(), right.end(),
        [&](int a, int b) { return key(a) < key(b); });
  }

  std::pair<InputPlan, std::array<int, 2>> input_path(
      const fastkag::Simulator& env, int player,
      const fastkag::PlayerAction& action,
      const std::vector<InputTarget>& targets,
      const std::vector<bool>& allowed, int worker_index) const {
    const int step = env.step_count(), day = env.day();
    auto positions = fastkag::positions(env, player);
    for (int actor = 0;
         actor < int(positions.size()) && actor < int(action.units.size());
         ++actor)
      if (input_is_move(action.units[actor].op))
        positions[actor] = input_move(
            positions[actor], action.units[actor].op,
            env.config().board_size);
    int native_hires = 0;
    for (const auto& order : action.market)
      native_hires += order.op == fastkag::Op::HIRE;
    fastkag::Position start{4, 4};
    for (int index = 0; index < native_hires + worker_index + 1; ++index) {
      start = input_spawn(positions);
      positions.push_back(start);
    }

    const int crop_price[2] = {
        std::max(1, env.market().prices[int(fastkag::Item::WHEAT)] - 2),
        std::max(1, env.market().prices[int(fastkag::Item::CARROT)] - 2)};
    const int fertilizer = std::max(
        1, public_market_price(
               int(fastkag::Item::FERTILIZER),
               env.market().inventory[int(fastkag::Item::FERTILIZER)] - 16) +
               2);
    struct Node {
      double score{};
      int gross{};
      int now{};
      fastkag::Position position{};
      std::vector<int> path;
      std::vector<bool> used;
      std::array<int, 2> units{};
    };
    std::vector<Node> beam{{0.0, 0, step + 2, start, {},
                            std::vector<bool>(targets.size()), {}}};
    std::optional<Node> best;
    const auto better = [&](const Node& left, const Node& right) {
      if (left.score != right.score) return left.score > right.score;
      if (left.gross != right.gross) return left.gross > right.gross;
      if (left.now != right.now) return left.now < right.now;
      return input_path_less(left.path, right.path, targets);
    };
    for (int depth = 0; depth < 8; ++depth) {
      std::vector<Node> expanded;
      for (const auto& node : beam) {
        for (int index = 0; index < int(targets.size()); ++index) {
          if (!allowed[index] || node.used[index]) continue;
          const auto& target = targets[index];
          const int arrival = node.now +
              std::abs(int(node.position.x) - int(target.position.x)) +
              std::abs(int(node.position.y) - int(target.position.y));
          if (arrival >= day * 24 + 23) continue;
          const int gain = input_gain(target, arrival, day);
          if (gain <= 0) continue;
          Node next = node;
          next.gross += gain * crop_price[int(target.crop)];
          next.path.push_back(index);
          next.used[index] = true;
          next.now = arrival + 1;
          next.position = target.position;
          next.units[int(target.crop)] += gain;
          next.score = next.gross -
                       1.5 * fertilizer * int(next.path.size());
          expanded.push_back(std::move(next));
        }
      }
      if (expanded.empty()) break;
      std::stable_sort(expanded.begin(), expanded.end(), better);
      if (expanded.size() > 8) expanded.resize(8);
      beam = std::move(expanded);
      if (depth >= 2 && (!best || better(beam.front(), *best)))
        best = beam.front();
    }
    if (!best) return {};
    InputPlan plan;
    plan.quantity = int(best->path.size());
    for (const int index : best->path) {
      const auto& target = targets[index];
      plan.path.push_back({target.position, target.crop, target.birth});
    }
    return {std::move(plan), best->units};
  }

  std::vector<InputPlan> input_joint_plans(
      const fastkag::Simulator& env, int player,
      const fastkag::PlayerAction& action,
      const std::vector<InputTarget>& targets,
      const std::array<int, fastkag::N_ITEMS>& stock,
      int purchases, int topup) const {
    struct Candidate {
      int net{};
      int value{};
      int cost{};
      int mode{};
      int quantity{};
      std::vector<InputPlan> plans;
    };
    std::optional<Candidate> best;
    const auto better = [](const Candidate& left, const Candidate& right) {
      return std::tuple<int, int, int, int, int>(
                 left.net, left.value, -left.cost,
                 -int(left.plans.size()), -left.mode) >
             std::tuple<int, int, int, int, int>(
                 right.net, right.value, -right.cost,
                 -int(right.plans.size()), -right.mode);
    };
    for (int mode = 0; mode < 3; ++mode) {
      std::vector<bool> remaining(targets.size(), true);
      Candidate candidate;
      candidate.mode = mode;
      std::array<int, 2> all_units{};
      for (int worker = 0; worker < 2; ++worker) {
        auto allowed = remaining;
        if (worker == 0 && mode != 0) {
          const auto crop = mode == 1
              ? fastkag::Item::WHEAT : fastkag::Item::CARROT;
          for (int index = 0; index < int(targets.size()); ++index)
            allowed[index] = allowed[index] && targets[index].crop == crop;
        }
        auto [plan, units] = input_path(
            env, player, action, targets, allowed, worker);
        const int quantity = int(plan.path.size());
        if (quantity < 3 || action.market.size() + 2 + worker > 10 ||
            count_item(stock) + purchases + candidate.quantity + quantity +
                    topup >
                95)
          break;
        const int quote = public_market_price(
            int(fastkag::Item::FERTILIZER),
            env.market().inventory[int(fastkag::Item::FERTILIZER)] -
                candidate.quantity - quantity - topup);
        const int cost = (quantity + (worker == 0 ? topup : 0)) *
                             (quote + 2) +
                         fib(env.farms()[player].hires_today + worker);
        int value = 0;
        for (int crop = 0; crop < 2; ++crop)
          value += units[crop] * std::max(
              1, public_market_price(
                     crop, env.market().inventory[crop] + all_units[crop] +
                               units[crop]) -
                     2);
        if (value < 1.5 * cost + 50 ||
            env.farms()[player].money < candidate.cost + cost + 3000)
          break;
        for (const auto& entry : plan.path)
          for (int index = 0; index < int(targets.size()); ++index)
            if (input_same_position(targets[index].position,
                                    entry.position))
              remaining[index] = false;
        candidate.quantity += quantity;
        candidate.cost += cost;
        candidate.value += value;
        candidate.net = candidate.value - candidate.cost;
        candidate.plans.push_back(std::move(plan));
        for (int crop = 0; crop < 2; ++crop)
          all_units[crop] += units[crop];
      }
      if (!best || better(candidate, *best)) best = std::move(candidate);
    }
    return best ? std::move(best->plans) : std::vector<InputPlan>{};
  }

  void apply_input51(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action, SeatState& seat) const {
    auto& state = seat.input;
    const int step = env.step_count(), day = env.day(), hour = env.hour();
    if (step == 0 || step <= state.last_step) state = {};
    state.last_step = step;
    if (state.day != day) {
      state.day = day;
      state.workers.clear();
      state.pending.clear();
    }
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    if (!state.pending.empty()) {
      for (auto& row : state.pending)
        if (int(farm.hands.size()) >= row.first)
          state.workers.push_back(std::move(row));
      state.pending.clear();
    }
    if (!state.workers.empty()) {
      action.units.resize(farm.hands.size() + 1);
      for (auto& [actor, plan] : state.workers) {
        if (actor <= 0 || actor >= int(action.units.size()) ||
            actor >= int(private_state.inventories.size()))
          continue;
        const auto position = farm.hands[actor - 1];
        const auto& inventory = private_state.inventories[actor];
        fastkag::Action command{};
        if (!plan.loaded) {
          const auto stock = python_projected_shed(env, player, action);
          const int quantity = std::min(
              plan.quantity,
              std::max(0, stock[int(fastkag::Item::FERTILIZER)]));
          const bool adjacent = (position.x == 4 || position.x == 5) &&
                                (position.y == 4 || position.y == 5);
          if (quantity > 0 && adjacent) {
            command = {fastkag::Op::PICKUP,
                       fastkag::Item::FERTILIZER, quantity};
            plan.loaded = true;
          }
        } else if (inventory[int(fastkag::Item::FERTILIZER)] > 0) {
          while (!plan.path.empty()) {
            const auto entry = plan.path.front();
            const auto& value = tile(env, player, entry.position);
            if (value.kind != fastkag::TileKind::PLANT ||
                value.crop != entry.crop ||
                int(value.planted_day) != entry.birth ||
                value.fertilized_until_day >= day + 2) {
              plan.path.erase(plan.path.begin());
              continue;
            }
            command = walk(position, entry.position);
            if (command.op == fastkag::Op::PASS) {
              command = {fastkag::Op::FERTILIZE};
              plan.path.erase(plan.path.begin());
            }
            break;
          }
        }
        action.units[actor] = command;
      }
      return;
    }
    if ((hour != 1 && hour != 2 && hour != 3) || day < 12 || day > 28)
      return;

    int expected = 0;
    for (int offset = 0;
         offset < 24 && day * 24 + offset < int(tape.size()); ++offset)
      expected = std::max(
          expected, int(tape[day * 24 + offset].units.size()) - 1);
    for (int offset = hour;
         offset < 24 && day * 24 + offset < int(tape.size()); ++offset)
      for (const auto& order : tape[day * 24 + offset].market)
        if (order.op == fastkag::Op::HIRE) return;
    if (std::any_of(seat.native.weed.begin(), seat.native.weed.end(),
                    [](const auto& pending) { return pending.active; }))
      return;
    if (day == 12 || day == 18 ||
        (seat.tomato.committed && seat.tomato.requested_day != day) ||
        (seat.sheep.committed && seat.sheep.requested_day != day) ||
        seat.tomato.pending || seat.sheep.pending)
      return;
    if (std::any_of(action.market.begin(), action.market.end(),
                    [](const auto& order) {
                      return order.op == fastkag::Op::HIRE;
                    }))
      return;
    if (expected > int(farm.hands.size())) return;
    std::vector<bool> owned(farm.hands.size() + 1);
    for (int actor = 1; actor <= expected; ++actor) owned[actor] = true;
    for (const auto& row : seat.tomato.workers) {
      if (row.first <= 0 || row.first >= int(owned.size()) ||
          owned[row.first])
        return;
      owned[row.first] = true;
    }
    for (const auto& row : seat.sheep.workers) {
      if (row.first <= 0 || row.first >= int(owned.size()) ||
          owned[row.first])
        return;
      owned[row.first] = true;
    }
    for (int actor = 1; actor < int(owned.size()); ++actor)
      if (!owned[actor]) return;

    const auto targets = input_forecast(env, player, tape, expected);
    const auto stock = python_projected_shed(env, player, action);
    int purchases = 0;
    for (const auto& order : action.market)
      if (order.op == fastkag::Op::BUY_PRODUCT ||
          order.op == fastkag::Op::BUY_ANIMAL)
        purchases += std::max(0, order.quantity);
    int available = std::max(
        0, stock[int(fastkag::Item::FERTILIZER)]);
    for (const auto& order : action.market) {
      if (order.item != fastkag::Item::FERTILIZER) continue;
      if (order.op == fastkag::Op::SELL)
        available = std::max(0, available - std::max(0, order.quantity));
      else if (order.op == fastkag::Op::BUY_PRODUCT)
        available += std::max(0, order.quantity);
    }
    int native_pickups = 0;
    if (hour + 1 < 24 && day * 24 + hour + 1 < int(tape.size()))
      for (const auto& command : tape[day * 24 + hour + 1].units)
        if (same(command, fastkag::Op::PICKUP,
                 fastkag::Item::FERTILIZER))
          native_pickups += std::max(0, command.quantity);
    const int topup = std::max(0, native_pickups - available);
    auto plans = input_joint_plans(
        env, player, action, targets, stock, purchases, topup);
    if (plans.empty()) return;
    int total = topup;
    const int first_actor = int(farm.hands.size()) + 1;
    for (int index = 0; index < int(plans.size()); ++index) {
      total += plans[index].quantity;
      state.pending.push_back({first_actor + index,
                               std::move(plans[index])});
    }
    action.market.push_back({fastkag::Op::BUY_PRODUCT,
                             fastkag::Item::FERTILIZER, total});
    for (std::size_t index = 0; index < state.pending.size(); ++index)
      action.market.push_back({fastkag::Op::HIRE});
  }

  static void suppress_reserved_sales(fastkag::PlayerAction& action,
                                      SeatState& state, int step) {
    if (step < 0 || step >= int(state.sale_debts.size())) return;
    auto& due = state.sale_debts[step];
    for (auto& order : action.market) {
      const int item = int(order.item);
      if (order.op != fastkag::Op::SELL || item < 0 ||
          item >= fastkag::N_PRODUCTS) continue;
      const int removed = std::min(std::max(0, order.quantity), due[item]);
      order.quantity -= removed;
      due[item] -= removed;
    }
  }

  static void suppress_racepx_sales(fastkag::PlayerAction& action,
                                    SeatState& state, int step) {
    if (step < 0 || step >= int(state.racepx_debts.size())) return;
    auto& due = state.racepx_debts[step];
    for (auto& order : action.market) {
      const int item = int(order.item);
      if (order.op != fastkag::Op::SELL || item < 0 ||
          item >= fastkag::N_PRODUCTS)
        continue;
      const int removed = std::min(std::max(0, order.quantity), due[item]);
      order.quantity -= removed;
      due[item] -= removed;
    }
  }

  static void apply_racepx_lead(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action, SeatState& state) {
    const int step = env.step_count(), next = step + 1;
    if (next > 718 || next % 72 == 0 || step % 4 == 0 ||
        next >= int(tape.size()))
      return;
    static constexpr std::array<int, fastkag::N_PRODUCTS> bases{
        25, 35, 60, 120, 250, 50, 160, 200, 100};
    bool any_blocked = false;
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item)
      any_blocked = any_blocked || env.market().prices[item] <= bases[item];
    std::array<int, fastkag::N_PRODUCTS> planned{};
    for (const auto& order : tape[next].market) {
      const int item = int(order.item);
      if (order.op == fastkag::Op::SELL && item >= 0 &&
          item < fastkag::N_PRODUCTS)
        planned[item] += std::max(0, order.quantity);
    }
    std::array<bool, fastkag::N_PRODUCTS> already{};
    for (const auto& order : action.market) {
      const int item = int(order.item);
      if (order.op == fastkag::Op::SELL && item >= 0 &&
          item < fastkag::N_PRODUCTS)
        already[item] = true;
    }
    auto projected = python_projected_shed(env, player, action);
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item) {
      if (env.market().prices[item] <= bases[item] || already[item] ||
          planned[item] <= 0 || env.market().prices[item] < 2)
        continue;
      // With no glutted book RACEPX delegates to the original lead helper,
      // which deliberately excludes the two purchasable products.  Its
      // custom partial-gate branch permits them when some other book is
      // blocked.
      if (!any_blocked &&
          (item == int(fastkag::Item::WHEAT) ||
           item == int(fastkag::Item::FERTILIZER)))
        continue;
      const int quantity = std::min(std::max(0, projected[item]), planned[item]);
      if (quantity <= 0 || action.market.size() >= 10) continue;
      action.market.push_back(
          {fastkag::Op::SELL, fastkag::Item(item), quantity});
      projected[item] -= quantity;
      state.racepx_debts[next][item] += quantity;
    }
  }

  static void apply_v9_fertilizer(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action) {
    const int step = env.step_count(), day = env.day();
    if (day < 14 || step >= 700) return;
    const auto positions = fastkag::positions(env, player);
    const auto& private_state = env.privates()[player];
    std::vector<int> planned(positions.size());
    for (int future = step; future < std::min<int>(tape.size(), (day + 1) * 24);
         ++future)
      for (std::size_t actor = 0;
           actor < tape[future].units.size() && actor < planned.size(); ++actor)
        planned[actor] += tape[future].units[actor].op == fastkag::Op::FERTILIZE;
    std::vector<fastkag::Position> targeted;
    for (std::size_t actor = 0;
         actor < action.units.size() && actor < positions.size(); ++actor) {
      if (action.units[actor].op != fastkag::Op::WATER) continue;
      const auto position = positions[actor];
      const auto& value = tile(env, player, position);
      if (value.kind != fastkag::TileKind::PLANT ||
          (value.crop != fastkag::Item::WHEAT &&
           value.crop != fastkag::Item::CARROT) ||
          day - value.planted_day != 1 || value.watered_today ||
          value.consecutive_unwatered != 0 ||
          value.fertilized_until_day >= day)
        continue;
      const int held = actor < private_state.inventories.size()
          ? private_state.inventories[actor][int(fastkag::Item::FERTILIZER)] : 0;
      const bool duplicate = std::any_of(targeted.begin(), targeted.end(),
          [&](auto prior) { return prior.x == position.x && prior.y == position.y; });
      if (held - planned[actor] <= 0 || duplicate) continue;
      action.units[actor] = {fastkag::Op::FERTILIZE};
      targeted.push_back(position);
    }
  }

  void reserve_future_sales(const fastkag::Simulator& env, int player,
                            const std::vector<fastkag::PlayerAction>& tape,
                            fastkag::PlayerAction& action, SeatState& state) {
    const int step = env.step_count();
    if (step < 192 || step >= 696) return;
    // Final EXP389 item horizon.  Its public-history ladder starts at 41 and can
    // only rise after a detected rival race; the PASS parity fixture remains
    // at this canonical floor.
    constexpr int horizon = 41;
    const int end = std::min(695, step + horizon);
    if (end <= step) return;
    const auto& private_state = env.privates()[player];
    for (std::size_t actor = 0;
         actor < action.units.size() && actor < private_state.inventories.size(); ++actor) {
      const auto& command = action.units[actor];
      const int item = int(command.item);
      if (command.op == fastkag::Op::PLACE && item >= int(fastkag::Item::GOOSE) &&
          item <= int(fastkag::Item::SHEEP) &&
          private_state.inventories[actor][item] > 0)
        return;
    }
    auto stock = python_projected_shed(env, player, action);
    std::array<bool, fastkag::N_PRODUCTS> blocked{};
    for (const auto& order : action.market) {
      const int item = int(order.item);
      if (item >= 0 && item < fastkag::N_PRODUCTS &&
          (order.op == fastkag::Op::SELL ||
           order.op == fastkag::Op::BUY_PRODUCT))
        blocked[item] = true;
    }
    for (const auto& command : action.units) {
      const int item = int(command.item);
      if (command.op == fastkag::Op::PICKUP && item >= 0 &&
          item < fastkag::N_PRODUCTS) blocked[item] = true;
    }
    static constexpr std::array<int, fastkag::N_PRODUCTS> bases{
        25, 35, 60, 120, 250, 50, 160, 200, 100};
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item) {
      if (blocked[item] || env.market().prices[item] <= bases[item] ||
          env.market().prices[item] < 2 || stock[item] <= 0 ||
          action.market.size() >= 10) continue;
      int available = stock[item];
      std::vector<std::pair<int, int>> reservations;
      for (int due = step + 1; due <= end; ++due) {
        bool barrier = false;
        for (const auto& command : tape[due].units)
          if (command.op == fastkag::Op::PICKUP && int(command.item) == item)
            barrier = true;
        for (const auto& order : tape[due].market)
          if (order.op == fastkag::Op::BUY_PRODUCT && int(order.item) == item)
            barrier = true;
        if (barrier) break;
        int planned = 0;
        for (const auto& order : tape[due].market)
          if (order.op == fastkag::Op::SELL && int(order.item) == item)
            planned += std::max(0, order.quantity);
        const int amount = std::min(available,
            std::max(0, planned - state.sale_debts[due][item]));
        if (amount > 0) {
          reservations.push_back({due, amount});
          available -= amount;
        }
        if (available == 0) break;
      }
      int quantity = 0;
      for (auto [due, amount] : reservations) quantity += amount;
      if (quantity == 0) continue;
      action.market.push_back(
          {fastkag::Op::SELL, fastkag::Item(item), quantity});
      for (auto [due, amount] : reservations)
        state.sale_debts[due][item] += amount;
    }
  }

  static double market_shape(int kind, double value, double scale) {
    value = std::max(0.0, value);
    if (kind == 0) return value;
    if (kind == 1) return value * value;
    if (kind == 2) return std::sqrt(value);
    if (kind == 3) return std::log1p(value);
    const double unit = value / scale;
    return unit + 8.0 * std::pow(std::max(0.0, unit - 1.0), 2.0);
  }

  static int public_market_price(int item, int inventory) {
    // shape: 0 linear, 1 square, 2 sqrt, 3 log, 4 hinge
    struct Params { int base, equilibrium, scale, below, above; double bt, at; };
    static constexpr std::array<Params, fastkag::N_PRODUCTS> params{{
        {25,10000,400,2,3,.8,.2}, {35,10000,450,4,2,1.,.7},
        {60,10000,200,4,2,.4,.6}, {120,10000,100,2,0,.7,1.6},
        {250,10000,300,3,1,.2,3.6}, {50,10000,332,4,3,.4,.2},
        {160,10000,122,2,0,.6,1.6}, {200,10000,105,3,1,.2,3.2},
        {100,10000,200,0,0,.4,.4}}};
    const auto& p = params[item];
    double price = p.base;
    if (inventory < p.equilibrium) {
      const double amplitude = p.bt * p.base /
          market_shape(p.below, p.scale, p.scale);
      price += amplitude * market_shape(
          p.below, p.equilibrium - inventory, p.scale);
    } else {
      const double amplitude = p.at * p.base /
          market_shape(p.above, p.scale, p.scale);
      price -= amplitude * market_shape(
          p.above, inventory - p.equilibrium, p.scale);
    }
    return std::max(1, int(std::nearbyint(price)));
  }

  bool v219_qualifies(const fastkag::Simulator& env, int player) const {
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    if (env.config().board_size != 10 || farm.unlocked_mask != 0x7 ||
        farm.money < 12000 ||
        env.market().prices[int(fastkag::Item::TOMATO)] < 70)
      return false;
    const int tomato_shops = std::count(env.shops().begin(), env.shops().end(), 2) +
                              std::count(env.shops().begin(), env.shops().end(), 5);
    if (tomato_shops < 3 ||
        private_state.seeds[int(fastkag::Item::TOMATO)] != 0 ||
        private_state.shed[int(fastkag::Item::TOMATO)] != 0)
      return false;
    for (int y : {5, 6})
      for (int x = 5; x < 10; ++x)
        if (farm.tiles[y * 10 + x].kind != fastkag::TileKind::LOCKED)
          return false;
    for (const auto& value : farm.tiles)
      if (value.kind == fastkag::TileKind::PLANT &&
          value.crop == fastkag::Item::TOMATO)
        return false;
    for (const auto& route : assets.library.routes)
      for (int step = 432; step < std::min(719, int(route.size())); ++step) {
        if (count_market(route[step], fastkag::Op::BUY_LAND)) return false;
        if (std::any_of(route[step].units.begin(), route[step].units.end(),
                        [](const auto& command) {
          return same(command, fastkag::Op::PLANT, fastkag::Item::TOMATO);
        })) return false;
      }
    return true;
  }

  static bool v219_fertilizer_worthwhile(
      const fastkag::Simulator& env, int player,
      const fastkag::PlayerAction& action) {
    const int fertilizer_item = int(fastkag::Item::FERTILIZER);
    if (env.market().prices[fertilizer_item] <= 30) return true;
    const auto& farm = env.farms()[player];
    const int day = env.day();
    int bonus = 0;
    for (int y : {5, 6})
      for (int x = 5; x < 10; ++x) {
        const auto& value = farm.tiles[y * 10 + x];
        if (value.kind != fastkag::TileKind::PLANT ||
            value.crop != fastkag::Item::TOMATO)
          continue;
        for (int future = day; future < day + 3; ++future)
          if (value.fertilized_until_day < future &&
              future + 1 - value.planted_day >= 8 &&
              future + 1 - value.planted_day <= 11)
            ++bonus;
      }
    if (bonus == 0) return false;
    const int tomato_item = int(fastkag::Item::TOMATO);
    const int tomato_price = std::max(
        1, public_market_price(tomato_item,
             env.market().inventory[tomato_item] + bonus + 10) - 2);
    const int fertilizer_price = std::max(
        1, public_market_price(fertilizer_item,
             env.market().inventory[fertilizer_item] - 10) + 2);
    const int native_hires = count_market(action, fastkag::Op::HIRE);
    const int labor = fib(env.farms()[player].hires_today + native_hires + 3);
    return bonus * tomato_price >= 2 * (10 * fertilizer_price + labor) + 100;
  }

  static bool position_less(fastkag::Position left,
                            fastkag::Position right) {
    return std::tie(left.x, left.y) < std::tie(right.x, right.y);
  }

  static bool paths_less(
      const std::vector<std::vector<fastkag::Position>>& left,
      const std::vector<std::vector<fastkag::Position>>& right) {
    return std::lexicographical_compare(
        left.begin(), left.end(), right.begin(), right.end(),
        [](const auto& a, const auto& b) {
          return std::lexicographical_compare(
              a.begin(), a.end(), b.begin(), b.end(), position_less);
        });
  }

  static std::optional<V219Labor> v219_labor_assignment(
      const fastkag::Simulator& env, int player,
      const fastkag::PlayerAction& action, bool fertilizer) {
    const int day = env.day(), hour = env.hour();
    if ((day != 26 && day != 27 && day != 28) || hour > 2 ||
        (day == 27 && !fertilizer))
      return std::nullopt;
    const int count = fertilizer ? 3 : 2;
    auto positions = fastkag::positions(env, player);
    for (std::size_t actor = 0;
         actor < positions.size() && actor < action.units.size(); ++actor) {
      const auto command = action.units[actor].op;
      if (command == fastkag::Op::NORTH)
        positions[actor].y = std::max<int>(0, positions[actor].y - 1);
      else if (command == fastkag::Op::SOUTH)
        positions[actor].y = std::min<int>(9, positions[actor].y + 1);
      else if (command == fastkag::Op::WEST)
        positions[actor].x = std::max<int>(0, positions[actor].x - 1);
      else if (command == fastkag::Op::EAST)
        positions[actor].x = std::min<int>(9, positions[actor].x + 1);
    }
    static constexpr std::array<fastkag::Position, 4> access{{
        {4,4}, {5,4}, {4,5}, {5,5}}};
    std::vector<fastkag::Position> spawns;
    const int native_hires = count_market(action, fastkag::Op::HIRE);
    for (int index = 0; index < native_hires + count; ++index) {
      int best = 0, best_count = std::numeric_limits<int>::max();
      for (int candidate = 0; candidate < int(access.size()); ++candidate) {
        const int occupied = std::count_if(
            positions.begin(), positions.end(), [&](auto position) {
              return position.x == access[candidate].x &&
                     position.y == access[candidate].y;
            });
        if (occupied < best_count) {
          best = candidate;
          best_count = occupied;
        }
      }
      positions.push_back(access[best]);
      if (index >= native_hires) spawns.push_back(access[best]);
    }

    std::vector<std::vector<fastkag::Position>> groups;
    if (fertilizer) {
      groups = {{{5,5},{6,5},{7,5},{8,5}},
                {{9,5},{9,6},{8,6}},
                {{5,6},{6,6},{7,6}}};
    } else {
      groups.resize(2);
      for (int x = 5; x < 10; ++x) {
        groups[0].push_back({int16_t(x),5});
        groups[1].push_back({int16_t(x),6});
      }
    }
    std::vector<int> permutation(groups.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    const int remaining = 23 - hour;
    bool found = false;
    int best_max = 0, best_sum = 0;
    std::vector<std::vector<fastkag::Position>> best_paths;
    do {
      std::vector<std::vector<fastkag::Position>> paths;
      std::vector<int> costs;
      for (int worker = 0; worker < count; ++worker) {
        const auto& path = groups[permutation[worker]];
        int travel = distance(spawns[worker], path.front());
        for (std::size_t index = 1; index < path.size(); ++index)
          travel += distance(path[index - 1], path[index]);
        int return_home = std::numeric_limits<int>::max();
        for (const auto target : access)
          return_home = std::min(return_home, distance(path.back(), target));
        costs.push_back(travel + return_home +
                        (fertilizer ? 3 : 2) * int(path.size()) + 1 +
                        int(fertilizer));
        paths.push_back(path);
      }
      const int maximum = *std::max_element(costs.begin(), costs.end());
      const int sum = std::accumulate(costs.begin(), costs.end(), 0);
      if (maximum <= remaining &&
          (!found || std::tie(maximum, sum) < std::tie(best_max, best_sum) ||
           (std::tie(maximum, sum) == std::tie(best_max, best_sum) &&
            paths_less(paths, best_paths)))) {
        found = true;
        best_max = maximum;
        best_sum = sum;
        best_paths = std::move(paths);
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    if (!found) return std::nullopt;
    return V219Labor{std::move(best_paths), std::move(spawns), count,
                     fertilizer};
  }

  static int v219_parent_fertilizer_quantity(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      const fastkag::PlayerAction& action) {
    auto stock = python_projected_shed(env, player, action);
    int total = count_item(stock);
    for (const auto& order : action.market) {
      const int item = int(order.item);
      if (item < 0 || item >= fastkag::N_ITEMS) continue;
      const int requested = std::max(0, order.quantity);
      if (order.op == fastkag::Op::SELL) {
        const int quantity = std::min(requested, std::max(0, stock[item]));
        stock[item] -= quantity;
        total -= quantity;
      } else if (order.op == fastkag::Op::BUY_PRODUCT ||
                 order.op == fastkag::Op::BUY_ANIMAL) {
        const int quantity = std::min(requested, std::max(0, 100 - total));
        stock[item] += quantity;
        total += quantity;
      }
    }
    int native_need = 0;
    const int next = env.step_count() + 1;
    if (next < int(tape.size()))
      for (const auto& command : tape[next].units)
        if (same(command, fastkag::Op::PICKUP,
                 fastkag::Item::FERTILIZER))
          native_need += std::max(0, command.quantity);
    const int fertilizer = int(fastkag::Item::FERTILIZER);
    const int quantity = std::max(10, 10 + native_need -
                                      std::max(0, stock[fertilizer]));
    return quantity > std::max(0, 100 - total) ? 10 : quantity;
  }

  fastkag::Action v219_worker(const fastkag::Simulator& env, int player,
                              V219State& state, int actor,
                              V219Role& role) const {
    const auto positions = fastkag::positions(env, player);
    if (actor < 0 || actor >= int(positions.size()) ||
        actor >= int(env.privates()[player].inventories.size()))
      return {};
    const auto position = positions[actor];
    const auto& inventory = env.privates()[player].inventories[actor];
    const int fertilizer = int(fastkag::Item::FERTILIZER);
    const int tomato = int(fastkag::Item::TOMATO);
    if (role.needs_fertilizer && !role.loaded) {
      const auto target = home(position);
      if (position.x != target.x || position.y != target.y)
        return walk(position, target);
      const int desired = role.fertilizer_quantity >= 0
          ? role.fertilizer_quantity : (role.fertilizer_worker ? 10 : 5);
      if (inventory[fertilizer] >= desired) {
        role.loaded = true;
      } else if (role.pickup_requested) {
        role.loaded = true;
        role.fertilizer_available = inventory[fertilizer];
      } else if (env.privates()[player].shed[fertilizer] >= desired) {
        role.pickup_requested = true;
        return {fastkag::Op::PICKUP, fastkag::Item::FERTILIZER, desired};
      } else {
        role.loaded = true;
      }
    }

    struct Work { fastkag::Position target; fastkag::Action command; };
    std::vector<Work> todo;
    for (const auto target : role.targets) {
      const auto& value = tile(env, player, target);
      const bool is_tomato = value.kind == fastkag::TileKind::PLANT &&
                             value.crop == fastkag::Item::TOMATO;
      const int key = int(target.y) * 10 + int(target.x);
      if (is_tomato) state.seen_plants[key] = true;
      if (state.seen_plants[key] && !is_tomato) state.lost[key] = true;
      fastkag::Action command{};
      if (role.fertilizer_worker) {
        if (is_tomato && value.fertilized_until_day < env.day() + 2 &&
            inventory[fertilizer] > 0)
          command = {fastkag::Op::FERTILIZE};
      } else if (env.day() == 18 && !is_tomato) {
        if (value.kind == fastkag::TileKind::EMPTY &&
            env.privates()[player].seeds[tomato] > 0)
          command = {fastkag::Op::PLANT, fastkag::Item::TOMATO};
        else if (value.kind == fastkag::TileKind::WEED)
          command = {fastkag::Op::DIG};
      } else if (is_tomato) {
        if (env.day() < 29 && !value.watered_today)
          command = {fastkag::Op::WATER};
        else if (role.needs_fertilizer &&
                 value.fertilized_until_day < env.day() + 2 &&
                 inventory[fertilizer] > 0)
          command = {fastkag::Op::FERTILIZE};
        else if (value.yield_units > 0)
          command = {fastkag::Op::HARVEST};
      }
      if (command.op != fastkag::Op::PASS)
        todo.push_back({target, command});
    }
    const auto shed = home(position);
    const int to_shed = distance(position, shed);
    if (env.step_count() >= 718 - to_shed && inventory[tomato] > 0) {
      if (to_shed > 0) return walk(position, shed);
      return {fastkag::Op::PLACE, fastkag::Item::TOMATO,
              inventory[tomato]};
    }
    if (!todo.empty()) {
      auto selected = todo.begin();
      int best_distance = distance(position, selected->target);
      for (auto iterator = std::next(todo.begin()); iterator != todo.end();
           ++iterator) {
        const int candidate = distance(position, iterator->target);
        if (candidate < best_distance) {
          selected = iterator;
          best_distance = candidate;
        }
      }
      if (best_distance > 0) return walk(position, selected->target);
      return selected->command;
    }
    if (inventory[tomato] > 0) {
      if (to_shed > 0) return walk(position, shed);
      return {fastkag::Op::PLACE, fastkag::Item::TOMATO,
              inventory[tomato]};
    }
    if (std::any_of(inventory.begin(), inventory.end(),
                    [](int quantity) { return quantity != 0; })) {
      if (to_shed > 0) return walk(position, shed);
      return {fastkag::Op::DROP};
    }
    return {};
  }

  void apply_v219(const fastkag::Simulator& env, int player,
                  const std::vector<fastkag::PlayerAction>& tape,
                  fastkag::PlayerAction& action, V219State& state) const {
    const int step = env.step_count(), day = env.day(), hour = env.hour();
    if (step == 432) state.eligible = v219_qualifies(env, player);
    if (!state.eligible || day < 18) return;
    if (state.day != day) {
      state.day = day;
      state.workers.clear();
      state.last_work.clear();
    }
    const auto& farm = env.farms()[player];
    if (state.pending) {
      const auto pending = std::move(*state.pending);
      state.pending.reset();
      if (int(farm.hands.size()) + 1 >=
              pending.first_actor + pending.count &&
          (farm.unlocked_mask & 0x8)) {
        for (int index = 0; index < pending.count; ++index) {
          V219Role role;
          role.fertilizer_worker = index == pending.crop_workers;
          if (role.fertilizer_worker) {
            for (int y : {5, 6})
              for (int x = 5; x < 10; ++x)
                role.targets.push_back({int16_t(x), int16_t(y)});
          } else if (pending.crop_workers == 1) {
            for (int y : {5, 6})
              for (int x = 5; x < 10; ++x)
                role.targets.push_back({int16_t(x), int16_t(y)});
          } else if (pending.crop_workers == 2) {
            const int y = 5 + index;
            for (int x = 5; x < 10; ++x)
              role.targets.push_back({int16_t(x), int16_t(y)});
          } else {
            if (index == 0)
              role.targets = {{5,5},{6,5},{7,5}};
            else if (index == 1)
              role.targets = {{8,5},{9,5},{9,6},{8,6}};
            else
              role.targets = {{5,6},{6,6},{7,6}};
          }
          role.needs_fertilizer = pending.fertilizer &&
                                  (day == 24 || role.fertilizer_worker);
          if (pending.labor) {
            role.targets = pending.labor->paths[index];
            role.needs_fertilizer = pending.labor->fertilizer;
            role.fertilizer_quantity = int(role.targets.size());
          }
          state.workers.push_back(
              {pending.first_actor + index, std::move(role)});
        }
      }
    }

    // The final v13 wrapper intentionally skips the pre-production watering
    // crews on alternating safe days.  It monkey-patches `_v219_request`, so
    // this guard must run before the original request logic.
    if (state.committed && state.requested_day != day &&
        (day == 19 || day == 21 || day == 23)) {
      int tomatoes = 0;
      bool safe = true;
      for (int y : {5, 6})
        for (int x = 5; x < 10; ++x) {
          const auto& value = farm.tiles[y * 10 + x];
          if (value.kind == fastkag::TileKind::PLANT &&
              value.crop == fastkag::Item::TOMATO) {
            ++tomatoes;
            if (value.consecutive_unwatered != 0) safe = false;
          }
        }
      if (tomatoes > 0 && safe) state.requested_day = day;
    }

    if ((!state.committed && day == 18) || state.committed) {
      bool can_request = state.requested_day != day;
      int latest_hire = -1;
      const int begin = day * 24;
      const int end = std::min((day + 1) * 24, 719);
      for (int at = begin; at < end; ++at)
        if (count_market(tape[at], fastkag::Op::HIRE) > 0)
          latest_hire = at - begin;
      int deadline = state.committed && latest_hire > 3 && latest_hire <= 6
          ? 6 : 3;
      if (hour > deadline) can_request = false;
      for (int at = step + 1; can_request && at < end; ++at)
        if (count_market(tape[at], fastkag::Op::HIRE) > 0)
          can_request = false;
      const int parent_hires = count_market(action, fastkag::Op::HIRE);
      const int expected = expected_hands(tape, day);
      if (int(farm.hands.size()) + parent_hires != expected)
        can_request = false;
      if (can_request) {
        const bool fertilizer = (day == 24 || day == 27) &&
            v219_fertilizer_worthwhile(env, player, action);
        int crop_workers =
            ((day >= 19 && day <= 23) || day == 25) && hour <= 2
                ? 1 : ((day >= 26 && day <= 28) ? 3 : 2);
        auto labor = v219_labor_assignment(
            env, player, action, fertilizer);
        if (labor) crop_workers = labor->workers;
        const int count = crop_workers +
                          int(fertilizer && day == 27 && !labor);
        std::vector<fastkag::Action> extra;
        if (!state.committed) {
          extra.push_back({fastkag::Op::BUY_LAND});
          extra.push_back({fastkag::Op::BUY_SEED,
                           fastkag::Item::TOMATO, 10});
        }
        const int fertilizer_quantity = fertilizer
            ? v219_parent_fertilizer_quantity(env, player, tape, action) : 0;
        if (fertilizer)
          extra.push_back({fastkag::Op::BUY_PRODUCT,
                           fastkag::Item::FERTILIZER,
                           fertilizer_quantity});
        for (int index = 0; index < count; ++index)
          extra.push_back({fastkag::Op::HIRE});
        if (action.market.size() + extra.size() <= 10) {
          std::int64_t budget = 0;
          for (int index = farm.hires_today;
               index < farm.hires_today + parent_hires + count; ++index)
            budget += fib(index);
          if (!state.committed) budget += 4500;
          if (fertilizer)
            budget += std::int64_t(fertilizer_quantity) *
                      (env.market().prices[int(fastkag::Item::FERTILIZER)] + 5);
          static constexpr std::array<int, fastkag::N_CROPS> seed_cost{
              10, 20, 50, 100, 80};
          static constexpr std::array<int, fastkag::N_ANIMALS> animal_cost{
              300, 400, 500};
          bool valid = true;
          for (const auto& order : action.market) {
            const int item = int(order.item);
            const int quantity = std::max(0, order.quantity);
            if (order.op == fastkag::Op::BUY_PRODUCT &&
                item >= 0 && item < fastkag::N_PRODUCTS)
              budget += std::int64_t(quantity) *
                        (env.market().prices[item] + 10);
            else if (order.op == fastkag::Op::BUY_ANIMAL &&
                     item >= int(fastkag::Item::GOOSE) &&
                     item <= int(fastkag::Item::SHEEP))
              budget += std::int64_t(quantity) *
                        animal_cost[item - int(fastkag::Item::GOOSE)];
            else if (order.op == fastkag::Op::BUY_SEED &&
                     item >= 0 && item < fastkag::N_CROPS)
              budget += std::int64_t(quantity) * seed_cost[item];
            else if ((order.op == fastkag::Op::BUY_PRODUCT ||
                      order.op == fastkag::Op::BUY_ANIMAL ||
                      order.op == fastkag::Op::BUY_SEED) && quantity > 0)
              valid = false;
          }
          if (valid && farm.money >= double(budget + 3000)) {
            state.pending = V219Pending{
                step, expected + 1, count, crop_workers, fertilizer,
                std::move(labor)};
            state.requested_day = day;
            state.committed = true;
            action.market.insert(action.market.end(),
                                 extra.begin(), extra.end());
          }
        }
      }
    }

    if (!state.workers.empty()) {
      action.units.resize(std::max(action.units.size(), farm.hands.size() + 1));
      for (auto& [actor, role] : state.workers) {
        if (actor >= int(action.units.size())) continue;
        const auto command = v219_worker(env, player, state, actor, role);
        action.units[actor] = command;
        const auto& inventory = env.privates()[player].inventories[actor];
        const int tomatoes = inventory[int(fastkag::Item::TOMATO)];
        auto found = std::find_if(state.last_work.begin(), state.last_work.end(),
            [&](const auto& row) { return row.first == actor; });
        V219LastWork work{step, command, tomatoes};
        if (found == state.last_work.end())
          state.last_work.push_back({actor, work});
        else
          found->second = work;
      }
    }
    if (state.committed && action.market.size() < 10 &&
        sale_quantity(action, fastkag::Item::TOMATO) == 0) {
      const int quantity = python_projected_shed(env, player, action)
          [int(fastkag::Item::TOMATO)];
      if (quantity > 0)
        action.market.push_back({fastkag::Op::SELL,
                                 fastkag::Item::TOMATO, quantity});
    }
    state.last_step = step;
  }

  static void v224_sales_first(fastkag::PlayerAction& action) {
    action.market.erase(std::remove_if(action.market.begin(), action.market.end(),
        [](const auto& order) {
          return order.op == fastkag::Op::PASS ||
                 (order.op != fastkag::Op::HIRE &&
                  order.op != fastkag::Op::BUY_LAND && order.quantity <= 0);
        }), action.market.end());
    if (action.market.size() > 10) action.market.resize(10);
    for (std::size_t index = 0; index < action.market.size(); ++index) {
      if (action.market[index].op != fastkag::Op::SELL) continue;
      std::size_t cursor = index;
      while (cursor > 0) {
        const auto& previous = action.market[cursor - 1];
        if (previous.op == fastkag::Op::SELL ||
            ((previous.op == fastkag::Op::BUY_PRODUCT ||
              previous.op == fastkag::Op::BUY_ANIMAL) &&
             previous.item == action.market[cursor].item))
          break;
        std::swap(action.market[cursor - 1], action.market[cursor]);
        --cursor;
      }
    }
  }

  static double r37_quote_priority(
      const fastkag::Simulator& env, int player,
      const std::array<int, fastkag::N_ITEMS>& stock,
      const fastkag::Action& order) {
    const int item = int(order.item);
    if (order.op != fastkag::Op::SELL || item < 0 ||
        item >= fastkag::N_PRODUCTS)
      return 0.0;
    const int quantity = std::min(std::max(0, order.quantity),
                                  std::max(0, stock[item]));
    if (quantity == 0) return 0.0;
    int standing = 0;
    for (const auto& value : env.farms()[1 - player].tiles) {
      if (item < fastkag::N_CROPS &&
          value.kind == fastkag::TileKind::PLANT && int(value.crop) == item)
        standing += std::max(0, int(value.yield_units));
      else if (item >= int(fastkag::Item::EGG) &&
               item <= int(fastkag::Item::WOOL) &&
               value.kind == fastkag::TileKind::ANIMAL &&
               int(value.animal) == item - int(fastkag::Item::EGG) +
                                      int(fastkag::Item::GOOSE))
        standing += std::max(0, int(value.yield_units));
    }
    const int batch = std::min(24, std::max(8, standing));
    const int inventory = env.market().inventory[item];
    double score = 0.0;
    for (int unit = 0; unit < quantity; ++unit)
      score += public_market_price(item, inventory + unit) -
               public_market_price(item, inventory + batch + unit);
    return score;
  }

  static void r37_reorder_sales(const fastkag::Simulator& env, int player,
                                fastkag::PlayerAction& action) {
    const auto stock = python_projected_shed(env, player, action);
    for (int begin = 0; begin < int(action.market.size());) {
      if (action.market[begin].op != fastkag::Op::SELL) {
        ++begin;
        continue;
      }
      int end = begin;
      while (end < int(action.market.size()) &&
             action.market[end].op == fastkag::Op::SELL)
        ++end;
      std::array<bool, fastkag::N_ITEMS> seen{};
      bool distinct = true;
      for (int index = begin; index < end; ++index) {
        const int item = int(action.market[index].item);
        if (item < 0 || item >= fastkag::N_ITEMS || seen[item]) {
          distinct = false;
          break;
        }
        seen[item] = true;
      }
      if (distinct)
        std::stable_sort(action.market.begin() + begin,
                         action.market.begin() + end,
            [&](const auto& left, const auto& right) {
              return r37_quote_priority(env, player, stock, left) >
                     r37_quote_priority(env, player, stock, right);
            });
      begin = end;
    }
  }

  static double or2_exposure(const fastkag::Simulator& env, int item,
                             int quantity, int batch) {
    if (item < 0 || item >= fastkag::N_PRODUCTS || quantity <= 0 || batch <= 0)
      return 0.0;
    const int inventory = env.market().inventory[item];
    double result = 0.0;
    for (int unit = 0; unit < quantity; ++unit)
      result += public_market_price(item, inventory + unit) -
                public_market_price(item, inventory + batch + unit);
    return result;
  }

  static std::array<Or2Tile, 100> or2_tiles(
      const fastkag::Farm& farm) {
    std::array<Or2Tile, 100> result{};
    for (std::size_t index = 0;
         index < result.size() && index < farm.tiles.size(); ++index) {
      const auto& value = farm.tiles[index];
      auto& row = result[index];
      if (value.kind == fastkag::TileKind::PLANT &&
          value.crop != fastkag::Item::NONE) {
        row = {true, true, int(value.crop), int(value.planted_day),
               int(value.yield_units)};
      } else if (value.kind == fastkag::TileKind::ANIMAL &&
                 value.animal >= fastkag::Item::GOOSE &&
                 value.animal <= fastkag::Item::SHEEP) {
        row = {true, false,
               int(value.animal) - int(fastkag::Item::GOOSE) +
                   int(fastkag::Item::EGG),
               int(value.placed_day), int(value.yield_units)};
      }
    }
    return result;
  }

  static std::array<int, 8> or2_draw(
      const std::vector<std::int8_t>& shops, int step) {
    std::array<int, 8> result{};
    static constexpr std::array<std::array<int, 4>, 8> products{{
        {{int(fastkag::Item::EGG), int(fastkag::Item::WHEAT), -1, -1}},
        {{int(fastkag::Item::EGG), int(fastkag::Item::WHEAT),
          int(fastkag::Item::STRAWBERRY), -1}},
        {{int(fastkag::Item::WHEAT), int(fastkag::Item::CARROT),
          int(fastkag::Item::TOMATO), int(fastkag::Item::STRAWBERRY)}},
        {{int(fastkag::Item::STRAWBERRY), int(fastkag::Item::MILK),
          int(fastkag::Item::WHEAT), -1}},
        {{int(fastkag::Item::CARROT), -1, -1, -1}},
        {{int(fastkag::Item::MILK), int(fastkag::Item::TOMATO),
          int(fastkag::Item::WHEAT), -1}},
        {{int(fastkag::Item::STRAWBERRY), int(fastkag::Item::MILK), -1, -1}},
        {{int(fastkag::Item::WOOL), -1, -1, -1}},
    }};
    if (step % 4 == 0)
      for (const int raw_shop : shops) {
        if (raw_shop < 0 || raw_shop >= int(products.size())) continue;
        int count = 0;
        for (const int item : products[raw_shop]) count += item >= 0;
        for (const int item : products[raw_shop])
          if (item >= 0) result[item] += count == 1 ? 2 : 1;
      }
    if (step % 24 == 0)
      for (auto& quantity : result) ++quantity;
    return result;
  }

  void apply_or2(const fastkag::Simulator& env, int player,
                 const std::vector<fastkag::PlayerAction>& tape,
                 fastkag::PlayerAction& action, SeatState& seat,
                 double slot_margin = 20.0) const {
    const int step = env.step_count();
    auto& state = seat.or2;
    if (step == 0 || step <= state.step) state = {};
    const auto current_tiles = or2_tiles(env.farms()[1 - player]);
    std::array<int, 8> inventory{};
    for (int item = 0; item < 8; ++item)
      inventory[item] = env.market().inventory[item];

    if (state.has_previous && state.previous_step == step - 1) {
      const auto& rival = env.farms()[1 - player];
      for (std::size_t index = 0; index < state.previous_tiles.size(); ++index) {
        const auto& old = state.previous_tiles[index];
        if (!old.present || old.yield <= 0) continue;
        const auto& current = current_tiles[index];
        int harvested = 0;
        const bool ongoing = old.plant &&
            (old.item == int(fastkag::Item::TOMATO) ||
             old.item == int(fastkag::Item::STRAWBERRY));
        if (old.plant && !ongoing) {
          const bool weed = index < rival.tiles.size() &&
                            rival.tiles[index].kind == fastkag::TileKind::WEED;
          if ((!current.present && !weed) ||
              (current.present && current.born != old.born))
            harvested = old.yield;
        } else if (current.present && current.plant == old.plant &&
                   current.item == old.item && current.born == old.born &&
                   current.yield < old.yield) {
          if (step % 24 != 0) harvested = old.yield - current.yield;
          else if (current.yield == 0) harvested = old.yield;
        }
        if (harvested > 0 && old.item >= 0 && old.item < 8)
          state.stock[old.item] += harvested;
      }
      const auto draw = or2_draw(state.previous_shops,
                                 state.previous_step);
      for (int item = 0; item < 8; ++item) {
        if (state.previous_prices[item] <= 1) continue;
        const int moved = inventory[item] - state.previous_inventory[item] +
                          draw[item] - state.previous_own[item];
        if (moved > 0)
          state.stock[item] = std::max(0, state.stock[item] - moved);
      }
    }

    const auto projected = python_projected_shed(env, player, action);
    if (step >= 288 && step < 694 && action.market.size() >= 10) {
      static constexpr std::array<int, 5> candidates{{
          int(fastkag::Item::MILK), int(fastkag::Item::STRAWBERRY),
          int(fastkag::Item::WOOL), int(fastkag::Item::MELON),
          int(fastkag::Item::EGG)}};
      std::array<bool, fastkag::N_ITEMS> selling{}, bought{};
      std::vector<int> sale_indices;
      for (int index = 0; index < int(action.market.size()); ++index) {
        const auto& order = action.market[index];
        const int item = int(order.item);
        if (order.op == fastkag::Op::SELL) {
          sale_indices.push_back(index);
          if (item >= 0 && item < fastkag::N_ITEMS) selling[item] = true;
        } else if ((order.op == fastkag::Op::BUY_PRODUCT ||
                    order.op == fastkag::Op::BUY_ANIMAL) &&
                   item >= 0 && item < fastkag::N_ITEMS) {
          bought[item] = true;
        }
      }
      struct SlotCandidate {
        bool found{};
        double value{};
        int item{-1};
        int quantity{};
        std::vector<std::pair<int, int>> plan;
      } best;
      for (const int item : candidates) {
        if (selling[item] || bought[item] || projected[item] <= 0 ||
            env.market().prices[item] < 2)
          continue;
        SlotCandidate candidate;
        candidate.item = item;
        const int stop = std::min(694, step + 6);
        for (int future = step + 1;
             future <= stop && candidate.quantity < projected[item]; ++future) {
          const auto& future_tape = future >= 648
              ? assets.library.routes[slot_for(2)] : tape;
          int planned = 0;
          for (const auto& order : future_tape[future].market)
            if (order.op == fastkag::Op::SELL && int(order.item) == item)
              planned += std::max(0, order.quantity);
          const int quantity = std::min(
              planned - seat.sale_debts[future][item],
              projected[item] - candidate.quantity);
          if (quantity > 0) {
            candidate.plan.push_back({future, quantity});
            candidate.quantity += quantity;
          }
        }
        if (candidate.quantity <= 0) continue;
        candidate.value = or2_exposure(
            env, item, candidate.quantity,
            std::max(1, std::min(30, state.stock[item])));
        candidate.found = true;
        if (!best.found || candidate.value > best.value)
          best = std::move(candidate);
      }
      if (best.found && !sale_indices.empty()) {
        int weakest = sale_indices.front();
        auto sale_value = [&](int index) {
          const auto& order = action.market[index];
          const int item = int(order.item);
          const int quantity = item >= 0 && item < fastkag::N_ITEMS
              ? std::min(std::max(0, order.quantity),
                         std::max(0, projected[item])) : 0;
          const int batch = item >= 0 && item < 8
              ? std::max(1, std::min(30, state.stock[item])) : 1;
          return or2_exposure(env, item, quantity, batch);
        };
        for (const int index : sale_indices)
          if (sale_value(index) < sale_value(weakest)) weakest = index;
        const auto removed = action.market[weakest];
        const int removed_item = int(removed.item);
        if (best.value > sale_value(weakest) + slot_margin &&
            ((removed_item != int(fastkag::Item::WHEAT) &&
              removed_item != int(fastkag::Item::FERTILIZER)) ||
             removed.quantity <= 2)) {
          int refund = std::max(0, removed.quantity);
          if (removed_item >= 0 && removed_item < fastkag::N_PRODUCTS)
            for (int future = step + 1;
                 future < step + 49 && refund > 0 && future < 720; ++future) {
              const int back = std::min(
                  seat.sale_debts[future][removed_item], refund);
              seat.sale_debts[future][removed_item] -= back;
              refund -= back;
            }
          action.market.erase(action.market.begin() + weakest);
          action.market.push_back(
              {fastkag::Op::SELL, fastkag::Item(best.item), best.quantity});
          for (const auto [future, quantity] : best.plan)
            seat.sale_debts[future][best.item] += quantity;
        }
      }
    }

    std::array<bool, fastkag::N_ITEMS> bought{};
    struct Row {
      int index;
      double exposure;
      double quote;
      fastkag::Action order;
    };
    std::vector<Row> movable;
    std::vector<fastkag::Action> fixed;
    for (int index = 0; index < int(action.market.size()); ++index) {
      const auto order = action.market[index];
      const int item = int(order.item);
      if ((order.op == fastkag::Op::BUY_PRODUCT ||
           order.op == fastkag::Op::BUY_ANIMAL) &&
          item >= 0 && item < fastkag::N_ITEMS)
        bought[item] = true;
      if (order.op == fastkag::Op::SELL && order.quantity > 0 && item >= 0 &&
          item < fastkag::N_ITEMS && !bought[item]) {
        const int quantity = std::min(
            std::max(0, order.quantity), std::max(0, projected[item]));
        const int batch = item < 8 ? std::min(30, state.stock[item]) : 0;
        movable.push_back({index, or2_exposure(env, item, quantity, batch),
                           r37_quote_priority(env, player, projected, order),
                           order});
      } else {
        fixed.push_back(order);
      }
    }
    std::stable_sort(movable.begin(), movable.end(),
        [](const auto& left, const auto& right) {
          if (left.exposure != right.exposure)
            return left.exposure > right.exposure;
          if (left.quote != right.quote) return left.quote > right.quote;
          return left.index < right.index;
        });
    action.market.clear();
    for (const auto& row : movable) action.market.push_back(row.order);
    action.market.insert(action.market.end(), fixed.begin(), fixed.end());

    std::array<int, 8> own{};
    auto left = projected;
    for (std::size_t index = 0;
         index < action.market.size() && index < 10; ++index) {
      const auto& order = action.market[index];
      const int item = int(order.item);
      if (order.op != fastkag::Op::SELL || item < 0 || item >= 8) continue;
      const int sold = std::min(std::max(0, order.quantity),
                                std::max(0, left[item]));
      left[item] -= sold;
      own[item] += sold;
    }
    state.has_previous = true;
    state.previous_step = step;
    state.previous_tiles = current_tiles;
    state.previous_inventory = inventory;
    state.previous_own = own;
    for (int item = 0; item < 8; ++item)
      state.previous_prices[item] = env.market().prices[item];
    state.previous_shops = env.shops();
    state.step = step;
  }

  static void apply_shedroom(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action) {
    const int step = env.step_count(), hour = env.hour();
    if ((hour != 21 && hour != 22 && hour != 23) || step >= 717) return;
    auto left = python_projected_shed(env, player, action);
    int night_shed = 0;
    for (int item = 0; item < fastkag::N_ITEMS; ++item)
      night_shed += std::max(0, left[item]);
    for (const auto& order : action.market) {
      const int item = int(order.item);
      if (order.op == fastkag::Op::SELL && item >= 0 &&
          item < fastkag::N_ITEMS) {
        const int filled = std::min(std::max(0, order.quantity),
                                    std::max(0, left[item]));
        left[item] -= filled;
        night_shed -= filled;
      } else if (order.op == fastkag::Op::BUY_PRODUCT ||
                 order.op == fastkag::Op::BUY_ANIMAL) {
        night_shed += std::max(0, order.quantity);
      }
    }
    const auto unit_positions = fastkag::positions(env, player);
    const auto& private_state = env.privates()[player];
    int carried = 0;
    for (std::size_t actor = 0; actor < unit_positions.size(); ++actor) {
      int held = 0;
      if (actor < private_state.inventories.size())
        for (int amount : private_state.inventories[actor])
          held += std::max(0, amount);
      const fastkag::Action command = actor < action.units.size()
          ? action.units[actor] : fastkag::Action{};
      const auto position = unit_positions[actor];
      const auto& value = tile(env, player, position);
      if (command.op == fastkag::Op::DROP &&
          fastkag::shed_adjacent(position))
        held = 0;
      else if (command.op == fastkag::Op::HARVEST)
        held += std::max(0, int(value.yield_units));
      else if (command.op == fastkag::Op::COLLECT_FERTILIZER &&
               value.fertilizer_available)
        ++held;
      else if ((command.op == fastkag::Op::FEED ||
                command.op == fastkag::Op::FERTILIZE) && held > 0)
        --held;
      else if (command.op == fastkag::Op::PICKUP &&
               fastkag::shed_adjacent(position))
        held += std::max(1, command.quantity);
      carried += held;
    }
    int overflow = night_shed + carried - env.config().shed_capacity + 8;
    if (overflow <= 0) return;
    std::array<int, fastkag::N_PRODUCTS> need{};
    const int stop = std::min(719, step + 25);
    for (int future = step + 1;
         future < stop && future < int(tape.size()); ++future) {
      for (const auto& command : tape[future].units) {
        if (command.op == fastkag::Op::FEED) ++need[int(fastkag::Item::WHEAT)];
        else if (command.op == fastkag::Op::FERTILIZE)
          ++need[int(fastkag::Item::FERTILIZER)];
      }
    }
    struct Candidate { int price, item, spare; };
    std::vector<Candidate> candidates;
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item) {
      const int spare = left[item] - need[item];
      if (spare > 0 && env.market().prices[item] >= 2)
        candidates.push_back({env.market().prices[item], item, spare});
    }
    std::sort(candidates.begin(), candidates.end(), [](auto left, auto right) {
      if (left.price != right.price) return left.price < right.price;
      return std::string_view(fastkag::item_name(left.item)) <
             std::string_view(fastkag::item_name(right.item));
    });
    for (const auto candidate : candidates) {
      if (overflow <= 0) break;
      const int quantity = std::min(candidate.spare, overflow);
      auto existing = std::find_if(action.market.begin(), action.market.end(),
          [&](const auto& order) {
            return order.op == fastkag::Op::SELL &&
                   int(order.item) == candidate.item;
          });
      if (existing != action.market.end())
        existing->quantity += quantity;
      else {
        if (action.market.size() >= 10) continue;
        action.market.push_back({fastkag::Op::SELL,
                                 fastkag::Item(candidate.item), quantity});
      }
      overflow -= quantity;
    }
  }

  static bool overflow_budget_ok(const fastkag::Simulator& env, int player,
                                 const fastkag::PlayerAction& action) {
    std::int64_t cost = 0;
    int hires = env.farms()[player].hires_today;
    static constexpr std::array<int, fastkag::N_CROPS> seed_cost{
        10, 20, 50, 100, 80};
    static constexpr std::array<int, fastkag::N_ANIMALS> animal_cost{
        300, 400, 500};
    for (const auto& order : action.market) {
      const int item = int(order.item);
      const int quantity = std::max(0, order.quantity);
      if (order.op == fastkag::Op::HIRE) {
        cost += fib(hires++);
      } else if (order.op == fastkag::Op::BUY_LAND) {
        cost += 4000;
      } else if (order.op == fastkag::Op::BUY_PRODUCT) {
        // R97 deliberately defines conservative quotes only for its two
        // protected inputs.  A different product raises in Python and makes
        // the enclosing OVERFLOW wrapper leave the parent action unchanged.
        if (item != int(fastkag::Item::WHEAT) &&
            item != int(fastkag::Item::FERTILIZER))
          return false;
        cost += std::int64_t(quantity) *
                public_market_price(item, env.market().inventory[item] - 2000);
      } else if (order.op == fastkag::Op::BUY_ANIMAL) {
        const int animal = item - int(fastkag::Item::GOOSE);
        if (animal < 0 || animal >= fastkag::N_ANIMALS) return false;
        cost += std::int64_t(quantity) * animal_cost[animal];
      } else if (order.op == fastkag::Op::BUY_SEED) {
        if (item < 0 || item >= fastkag::N_CROPS) return false;
        cost += std::int64_t(quantity) * seed_cost[item];
      }
    }
    return double(cost) <= env.farms()[player].money;
  }

  static std::array<int, fastkag::N_ITEMS> overflow_market_stock(
      std::array<int, fastkag::N_ITEMS> stock,
      const std::vector<fastkag::Action>& orders) {
    int total = count_item(stock);
    for (const auto& order : orders) {
      const int item = int(order.item);
      if (item < 0 || item >= fastkag::N_ITEMS) continue;
      const int requested = std::max(0, order.quantity);
      if (order.op == fastkag::Op::SELL) {
        const int quantity = std::min(requested, std::max(0, stock[item]));
        stock[item] -= quantity;
        total -= quantity;
      } else if (order.op == fastkag::Op::BUY_PRODUCT ||
                 order.op == fastkag::Op::BUY_ANIMAL) {
        const int quantity = std::min(requested, std::max(0, 100 - total));
        stock[item] += quantity;
        total += quantity;
      }
    }
    return stock;
  }

  static std::pair<std::array<int, fastkag::N_ITEMS>, bool>
  overflow_delivery(std::array<int, fastkag::N_ITEMS> stock,
                    const fastkag::PrivateState& private_state) {
    int total = count_item(stock);
    bool lost = false;
    for (std::size_t actor = 0;
         actor < private_state.inventories.size(); ++actor) {
      const auto& inventory = private_state.inventories[actor];
      const auto& order = private_state.inventory_order[actor];
      for (const int raw : order) {
        const int quantity = std::max(0, inventory[raw]);
        const int take = std::min(quantity, std::max(0, 100 - total));
        stock[raw] += take;
        total += take;
        lost = lost || quantity > take;
      }
    }
    return {stock, lost};
  }

  static void apply_overflow(const fastkag::Simulator& env, int player,
                             fastkag::PlayerAction& action) {
    if (env.hour() != 23 || action.market.size() >= 10 ||
        !overflow_budget_ok(env, player, action))
      return;

    std::array<fastkag::PlayerAction, 2> joint{};
    joint[player] = action;
    const auto after_units = env.preview_unit_phase(joint);
    const auto& private_state = after_units.privates()[player];
    const auto base_stock = private_state.shed;
    const auto market_stock = overflow_market_stock(base_stock, action.market);
    const auto [original, has_loss] =
        overflow_delivery(market_stock, private_state);
    if (!has_loss) return;

    int remaining = std::max(0, 100 - count_item(market_stock));
    std::vector<int> tail;
    for (std::size_t actor = 0;
         actor < private_state.inventories.size(); ++actor) {
      const auto& inventory = private_state.inventories[actor];
      for (const int raw : private_state.inventory_order[actor]) {
        const int quantity = std::max(0, inventory[raw]);
        const int take = std::min(quantity, remaining);
        remaining -= take;
        tail.insert(tail.end(), quantity - take, raw);
      }
    }

    std::array<int, fastkag::N_ITEMS> released{};
    std::vector<int> release_order;
    std::optional<std::vector<fastkag::Action>> best;
    for (const int item : tail) {
      if (released[item]++ == 0) release_order.push_back(item);
      if (item < 0 || item >= fastkag::N_PRODUCTS ||
          released[item] > market_stock[item])
        break;
      if (action.market.size() + release_order.size() > 10) break;
      auto proposed = action.market;
      for (const int product : release_order)
        proposed.push_back({fastkag::Op::SELL, fastkag::Item(product),
                            released[product]});
      const auto after_market = overflow_market_stock(base_stock, proposed);
      const auto final = overflow_delivery(after_market, private_state).first;
      if (final == original) best = std::move(proposed);
    }
    if (best) action.market = std::move(*best);
  }

  bool r86_next_feed(const fastkag::Simulator& env,
                     const std::vector<fastkag::PlayerAction>& current_tape,
                     fastkag::Position target) const {
    // Python R86 treats the last playable dawn specially: after day 28 there
    // is no second dawn before termination, so today's saved wheat cannot
    // forfeit a later feeding opportunity.
    if (env.day() == 28) return true;
    const int tomorrow = env.day() + 1;
    const auto& tape = tomorrow >= 27
        ? assets.library.routes[slot_for(2)] : current_tape;
    std::vector<fastkag::Position> positions{{4, 4}};
    std::vector<int> wheat{0};
    static constexpr std::array<fastkag::Position, 4> access{{
        {4,4},{5,4},{4,5},{5,5}}};
    const auto at_access = [&](fastkag::Position position) {
      return std::any_of(access.begin(), access.end(),
          [&](auto value) { return value.x == position.x && value.y == position.y; });
    };
    bool target_fed = false;
    for (int hour = 0; hour < 24; ++hour) {
      const int step = tomorrow * 24 + hour;
      if (step >= int(tape.size())) break;
      const auto& planned = tape[step];
      for (std::size_t actor = 0;
           actor < positions.size() && actor < planned.units.size(); ++actor) {
        const auto command = planned.units[actor];
        auto& position = positions[actor];
        if (command.op == fastkag::Op::NORTH)
          position.y = std::max<int>(0, position.y - 1);
        else if (command.op == fastkag::Op::SOUTH)
          position.y = std::min<int>(9, position.y + 1);
        else if (command.op == fastkag::Op::WEST)
          position.x = std::max<int>(0, position.x - 1);
        else if (command.op == fastkag::Op::EAST)
          position.x = std::min<int>(9, position.x + 1);
        else if (same(command, fastkag::Op::PICKUP, fastkag::Item::WHEAT) &&
                 at_access(position))
          wheat[actor] += std::max(0, command.quantity);
        else if (command.op == fastkag::Op::FEED && wheat[actor] > 0) {
          --wheat[actor];
          if (hour <= 21 && position.x == target.x && position.y == target.y)
            target_fed = true;
        } else if (command.op == fastkag::Op::DROP && at_access(position))
          wheat[actor] = 0;
        else if (same(command, fastkag::Op::PLACE, fastkag::Item::WHEAT) &&
                 at_access(position))
          wheat[actor] = std::max(0, wheat[actor] -
                                      std::max(0, command.quantity));
      }
      for (const auto& order : planned.market) {
        if (order.op != fastkag::Op::HIRE) continue;
        int best = 0, best_count = std::numeric_limits<int>::max();
        for (int index = 0; index < int(access.size()); ++index) {
          const int count = std::count_if(positions.begin(), positions.end(),
              [&](auto value) {
                return value.x == access[index].x &&
                       value.y == access[index].y;
              });
          if (count < best_count) {
            best = index;
            best_count = count;
          }
        }
        positions.push_back(access[best]);
        wheat.push_back(0);
      }
    }
    return target_fed;
  }

  void apply_r85_feed(const fastkag::Simulator& env, int player,
                      const std::vector<fastkag::PlayerAction>& tape,
                      fastkag::PlayerAction& action) const {
    const int day = env.day();
    if (day < 10 || day > 28 || env.hour() > 21) return;
    const int expected = expected_hands(tape, day);
    const auto positions = fastkag::positions(env, player);
    const auto& private_state = env.privates()[player];
    for (int actor = 0; actor <= expected &&
                        actor < int(action.units.size()) &&
                        actor < int(positions.size()) &&
                        actor < int(private_state.inventories.size()); ++actor) {
      if (action.units[actor].op != fastkag::Op::FEED) continue;
      const auto& value = tile(env, player, positions[actor]);
      if (value.kind != fastkag::TileKind::ANIMAL || value.fed_today ||
          value.consecutive_unfed != 0 ||
          private_state.inventories[actor][int(fastkag::Item::WHEAT)] <= 0)
        continue;
      int first = 0, interval = 0, product = -1;
      if (value.animal == fastkag::Item::GOOSE) {
        first = 4; interval = 1; product = int(fastkag::Item::EGG);
      } else if (value.animal == fastkag::Item::COW) {
        first = 8; interval = 2; product = int(fastkag::Item::MILK);
      } else if (value.animal == fastkag::Item::SHEEP) {
        first = 6; interval = 3; product = int(fastkag::Item::WOOL);
      } else continue;
      first += value.placed_day;
      const int tomorrow = day + 1;
      const bool produces = tomorrow >= first &&
                            (tomorrow - first) % interval == 0;
      int bonus = produces ? std::max(0, int(value.pending_care_bonus)) : 0;
      int next_use = first;
      if (next_use <= tomorrow)
        next_use += ((tomorrow - next_use) / interval + 1) * interval;
      if (next_use <= 29) ++bonus;
      if (double(bonus) * (env.market().prices[product] + 5) * 1.25 >=
          env.market().prices[int(fastkag::Item::WHEAT)])
        continue;
      if (!r86_next_feed(env, tape, positions[actor])) continue;
      action.units[actor] = {};
    }
  }

  void apply_r51_close_warehouse(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action) const {
    const int step = env.step_count();
    if (env.hour() != 23 || env.day() < 12 || env.day() > 28)
      return;
    if (std::any_of(action.market.begin(), action.market.end(),
                    [](const auto& order) {
                      return order.op != fastkag::Op::SELL;
                    }))
      return;

    std::array<fastkag::PlayerAction, 2> joint{};
    joint[player] = action;
    const auto preview = env.preview_unit_phase(joint);
    auto post = preview.privates()[player].shed;
    for (const auto& order : action.market) {
      const int item = int(order.item);
      if (order.op == fastkag::Op::SELL && item >= 0 &&
          item < fastkag::N_ITEMS)
        post[item] = std::max(0, post[item] - std::max(0, order.quantity));
    }
    const auto& inventories = preview.privates()[player].inventories;
    int needed = count_item(post) - env.config().shed_capacity;
    for (const auto& inventory : inventories)
      for (const int quantity : inventory)
        needed += std::max(0, quantity);
    if (needed <= 0) return;

    std::vector<int> products;
    for (int item = int(fastkag::Item::CARROT);
         item <= int(fastkag::Item::WOOL); ++item)
      products.push_back(item);
    std::stable_sort(products.begin(), products.end(), [&](int left, int right) {
      return env.market().prices[left] > env.market().prices[right];
    });
    for (const int item : products) {
      const int quantity = std::min(needed, std::max(0, post[item]));
      if (quantity <= 0) continue;
      auto sale = std::find_if(action.market.begin(), action.market.end(),
          [&](const auto& order) {
            return order.op == fastkag::Op::SELL && int(order.item) == item;
          });
      if (sale != action.market.end())
        sale->quantity += quantity;
      else {
        if (action.market.size() >= 10) continue;
        action.market.push_back(
            {fastkag::Op::SELL, fastkag::Item(item), quantity});
      }
      needed -= quantity;
      post[item] -= quantity;
      if (needed <= 0) return;
    }

    int reserve = 0;
    for (int future = step + 1; future < 719; ++future) {
      const auto& planned = future >= 648
          ? assets.library.routes[slot_for(2)][future] : tape[future];
      for (const auto& command : planned.units)
        if (same(command, fastkag::Op::PICKUP, fastkag::Item::WHEAT))
          reserve += std::max(0, command.quantity > 0
                                  ? command.quantity : 1);
      if (std::any_of(planned.market.begin(), planned.market.end(),
                      [](const auto& order) {
                        return same(order, fastkag::Op::BUY_PRODUCT,
                                    fastkag::Item::WHEAT);
                      }))
        break;
    }
    int incoming = 0, others = 0;
    for (const auto& inventory : inventories)
      for (int item = 0; item < fastkag::N_ITEMS; ++item)
        if (item == int(fastkag::Item::WHEAT))
          incoming += std::max(0, inventory[item]);
        else
          others += std::max(0, inventory[item]);
    for (int item = 1; item < fastkag::N_ITEMS; ++item)
      others += std::max(0, post[item]);
    if (env.config().shed_capacity - others < reserve) return;
    const int wheat = int(fastkag::Item::WHEAT);
    const int quantity = std::min(
        {needed, std::max(0, post[wheat]),
         std::max(0, post[wheat] + incoming - reserve)});
    if (quantity <= 0) return;
    auto sale = std::find_if(action.market.begin(), action.market.end(),
        [](const auto& order) {
          return same(order, fastkag::Op::SELL, fastkag::Item::WHEAT);
        });
    if (sale != action.market.end())
      sale->quantity += quantity;
    else if (action.market.size() < 10)
      action.market.push_back(
          {fastkag::Op::SELL, fastkag::Item::WHEAT, quantity});
  }

  static void apply_v9_herd(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action, V9HerdState& state) {
    const int step = env.step_count();
    if (step == 0 || step <= state.last_step) state = {};
    state.last_step = step;
    if (!state.decided && state.species == fastkag::Item::NONE && step >= 216) {
      const bool buying_goose = std::any_of(
          action.market.begin(), action.market.end(), [](const auto& order) {
            return same(order, fastkag::Op::BUY_ANIMAL,
                        fastkag::Item::GOOSE);
          });
      if (buying_goose) {
        state.decided = true;
        int egg_shops = 0, milk_shops = 0;
        bool yarn = false;
        for (const int shop : env.shops()) {
          egg_shops += shop == 0 || shop == 1;
          milk_shops += shop == 3 || shop == 5 || shop == 6;
          yarn = yarn || shop == 7;
        }
        if (egg_shops <= 1 && yarn &&
            env.market().prices[int(fastkag::Item::WOOL)] >= 150)
          state.species = fastkag::Item::SHEEP;
        else if (egg_shops == 0 && milk_shops >= 3 &&
                 env.market().prices[int(fastkag::Item::MILK)] >= 150)
          state.species = fastkag::Item::COW;
      }
    }
    if (state.species == fastkag::Item::NONE) return;

    for (auto& command : action.units) {
      if (command.op == fastkag::Op::BUILD_COOP)
        command.op = fastkag::Op::BUILD_PASTURE;
      else if ((command.op == fastkag::Op::PICKUP ||
                command.op == fastkag::Op::PLACE) &&
               command.item == fastkag::Item::GOOSE)
        command.item = state.species;
    }
    for (auto& order : action.market)
      if (same(order, fastkag::Op::BUY_ANIMAL, fastkag::Item::GOOSE))
        order.item = state.species;

    const bool owns_goose = std::any_of(
        env.farms()[player].tiles.begin(), env.farms()[player].tiles.end(),
        [](const auto& value) {
          return value.kind == fastkag::TileKind::ANIMAL &&
                 value.animal == fastkag::Item::GOOSE;
        });
    if (!owns_goose)
      action.market.erase(std::remove_if(
          action.market.begin(), action.market.end(), [](const auto& order) {
            return same(order, fastkag::Op::SELL, fastkag::Item::EGG);
          }), action.market.end());

    if (step >= 718) return;
    const auto product = state.species == fastkag::Item::SHEEP
        ? fastkag::Item::WOOL : fastkag::Item::MILK;
    const auto stock = python_projected_shed(env, player, action);
    int selling = 0, planned = 0;
    for (const auto& order : action.market)
      if (same(order, fastkag::Op::SELL, product))
        selling += std::max(0, order.quantity);
    for (int future = step + 1; future < int(tape.size()); ++future)
      for (const auto& order : tape[future].market)
        if (same(order, fastkag::Op::SELL, product))
          planned += std::max(0, order.quantity);
    const int extra = stock[int(product)] - selling - planned;
    if (extra > 0 && action.market.size() < 10 &&
        env.market().prices[int(product)] >= 2)
      action.market.insert(
          action.market.begin(), {fastkag::Op::SELL, product, extra});
  }

  int r85_reserve(const fastkag::Simulator& env, int player,
                  const SeatState& state) const {
    std::array<int, 720> reserve{};
    const int route2 = slot_for(2);
    const int selected = slot_for(state.route_id);
    for (int future = 718; future >= 0; --future) {
      const auto& planned = assets.library.routes[
          future >= 648 ? route2 : selected][future];
      int pickup = 0, purchase = 0;
      for (const auto& command : planned.units)
        if (same(command, fastkag::Op::PICKUP,
                 fastkag::Item::FERTILIZER))
          pickup += std::max(0, command.quantity > 0
                                  ? command.quantity : 1);
      for (const auto& order : planned.market)
        if (same(order, fastkag::Op::BUY_PRODUCT,
                 fastkag::Item::FERTILIZER))
          purchase += std::max(0, order.quantity);
      reserve[future] = pickup +
          std::max(0, (future < 719 ? reserve[future + 1] : 0) - purchase);
    }

    int dedicated = 0;
    const auto& inventories = env.privates()[player].inventories;
    for (const auto& [actor, role] : state.tomato.workers) {
      if (!role.needs_fertilizer || role.loaded) continue;
      const int desired = role.fertilizer_quantity >= 0
          ? role.fertilizer_quantity : (role.fertilizer_worker ? 10 : 5);
      const int carried = actor >= 0 && actor < int(inventories.size())
          ? inventories[actor][int(fastkag::Item::FERTILIZER)] : 0;
      dedicated += std::max(0, desired - carried);
    }
    if (state.tomato.pending && state.tomato.pending->fertilizer)
      dedicated += 10;
    const auto reserve_input = [&](const auto& plans) {
      int result = 0;
      for (const auto& [actor, plan] : plans)
        if (!plan.loaded) result += std::max(0, plan.quantity);
      return result;
    };
    dedicated += reserve_input(state.input.workers);
    dedicated += reserve_input(state.input.pending);
    return std::max(14, reserve[std::min(719, env.step_count() + 1)] +
                            dedicated);
  }

  void apply_r85_fertilizer(const fastkag::Simulator& env, int player,
                            fastkag::PlayerAction& action,
                            const SeatState& state) const {
    if (env.day() < 6 || env.day() > 28 || action.market.size() >= 10)
      return;
    if (std::any_of(action.market.begin(), action.market.end(),
                    [](const auto& order) {
                      return order.op != fastkag::Op::SELL;
                    }))
      return;
    const auto stock = python_projected_shed(env, player, action);
    int sold = 0;
    for (const auto& order : action.market)
      if (same(order, fastkag::Op::SELL, fastkag::Item::FERTILIZER))
        sold += std::max(0, order.quantity);
    const int extra = stock[int(fastkag::Item::FERTILIZER)] - sold -
                      r85_reserve(env, player, state);
    if (extra > 0)
      action.market.push_back(
          {fastkag::Op::SELL, fastkag::Item::FERTILIZER, extra});
  }

  static double v44_item_margin(
      const std::vector<fastkag::Action>& mine,
      const std::vector<fastkag::Action>& opponent, int item,
      int initial_inventory, int initial_stock) {
    int inventory = initial_inventory;
    std::array<int, 2> stock{initial_stock, initial_stock};
    std::array<double, 2> revenue{};
    const std::size_t slots = std::max(mine.size(), opponent.size());
    for (std::size_t slot = 0; slot < slots; ++slot) {
      struct Remaining { fastkag::Op op{fastkag::Op::PASS}; int quantity{}; };
      std::array<Remaining, 2> remaining{};
      for (int side = 0; side < 2; ++side) {
        const auto& orders = side == 0 ? mine : opponent;
        if (slot >= orders.size()) continue;
        const auto& order = orders[slot];
        if (int(order.item) == item && order.quantity > 0 &&
            (order.op == fastkag::Op::SELL ||
             order.op == fastkag::Op::BUY_PRODUCT))
          remaining[side] = {order.op, order.quantity};
      }
      for (int guard = 0; guard < 5000; ++guard) {
        std::array<int, 2> quote{-1, -1};
        for (int side = 0; side < 2; ++side) {
          if (remaining[side].quantity <= 0) continue;
          if (remaining[side].op == fastkag::Op::SELL)
            quote[side] = public_market_price(item, inventory);
          else if (item == int(fastkag::Item::WHEAT) ||
                   item == int(fastkag::Item::FERTILIZER))
            quote[side] = public_market_price(item, inventory - 1);
          else
            remaining[side].quantity = 0;
        }
        if (quote[0] < 0 && quote[1] < 0) break;
        bool committed = false;
        for (int side = 0; side < 2; ++side) {
          if (quote[side] < 0) continue;
          if (remaining[side].op == fastkag::Op::SELL) {
            if (stock[side] <= 0) {
              remaining[side].quantity = 0;
              continue;
            }
            --stock[side];
            revenue[side] += quote[side];
            if (quote[side] > 1) ++inventory;
          } else {
            ++stock[side];
            revenue[side] -= quote[side];
            --inventory;
          }
          --remaining[side].quantity;
          committed = true;
        }
        if (!committed) break;
      }
    }
    return revenue[0] - revenue[1];
  }

  static double v44_margin(
      const std::vector<fastkag::Action>& candidate,
      const std::vector<fastkag::Action>& opponent,
      const fastkag::Market& market,
      const std::array<int, fastkag::N_ITEMS>& stock) {
    double total = 0.0;
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item) {
      const bool present = std::any_of(candidate.begin(), candidate.end(),
          [item](const auto& order) {
            return int(order.item) == item &&
                (order.op == fastkag::Op::SELL ||
                 order.op == fastkag::Op::BUY_PRODUCT);
          });
      if (present)
        total += v44_item_margin(candidate, opponent, item,
                                 market.inventory[item], stock[item]);
    }
    return total;
  }

  static void v44_reorder(const fastkag::Simulator& env, int player,
                          fastkag::PlayerAction& action) {
    if (env.step_count() < 216 || action.market.size() < 2) return;
    std::vector<std::pair<int, int>> blocks;
    for (int index = 0; index < int(action.market.size());) {
      if (action.market[index].op != fastkag::Op::SELL) {
        ++index;
        continue;
      }
      int end = index;
      while (end < int(action.market.size()) &&
             action.market[end].op == fastkag::Op::SELL) ++end;
      if (end - index >= 2 && end - index <= 6)
        blocks.push_back({index, end});
      index = end;
    }
    if (blocks.empty()) return;
    const auto opponent = action.market;
    const auto stock = python_projected_shed(env, player, action);
    const double baseline = v44_margin(
        action.market, opponent, env.market(), stock);
    double best = baseline;
    for (auto [begin, end] : blocks) {
      const int size = end - begin;
      std::vector<int> permutation(size);
      std::iota(permutation.begin(), permutation.end(), 0);
      std::vector<fastkag::Action> accepted;
      do {
        // Duplicate `(item, quantity)` permutations only repeat an identical
        // candidate; evaluating them again is harmless at n<=6.
        auto candidate = action.market;
        for (int offset = 0; offset < size; ++offset)
          candidate[begin + offset] = action.market[begin + permutation[offset]];
        const double value = v44_margin(candidate, opponent, env.market(), stock);
        if (value > best + 0.5) {
          best = value;
          accepted = std::move(candidate);
        }
      } while (std::next_permutation(permutation.begin(), permutation.end()));
      if (!accepted.empty()) action.market = std::move(accepted);
    }
  }

  static void e335_compact(const fastkag::Simulator& env, int player,
                           fastkag::PlayerAction& action) {
    if (env.step_count() < 144 || action.market.size() < 2) return;
    const auto is_cash = [](int item) {
      return item >= int(fastkag::Item::CARROT) &&
             item <= int(fastkag::Item::WOOL);
    };
    const bool all_sales = std::all_of(
        action.market.begin(), action.market.end(), [](const auto& order) {
          return order.op == fastkag::Op::SELL;
        });
    std::array<fastkag::PlayerAction, 2> joint{};
    joint[player] = action;
    const auto after_units = env.preview_unit_phase(joint);
    auto remaining = after_units.privates()[player].shed;
    if (all_sales) {
      std::vector<fastkag::Action> effective;
      effective.reserve(action.market.size());
      for (const auto& order : action.market) {
        const int item = int(order.item);
        const int quantity = item >= 0 && item < fastkag::N_ITEMS
            ? std::min(std::max(0, order.quantity),
                       std::max(0, remaining[item])) : 0;
        if (item >= 0 && item < fastkag::N_ITEMS)
          remaining[item] -= quantity;
        effective.push_back(quantity > 0
            ? fastkag::Action{fastkag::Op::SELL, order.item, quantity}
            : fastkag::Action{});
      }
      auto revised = effective;
      for (int begin = 0; begin < int(effective.size());) {
        if (effective[begin].op == fastkag::Op::SELL &&
            !is_cash(int(effective[begin].item))) {
          ++begin;
          continue;
        }
        int end = begin + 1;
        while (end < int(effective.size()) &&
               (effective[end].op == fastkag::Op::PASS ||
                is_cash(int(effective[end].item))))
          ++end;
        std::vector<fastkag::Action> kept;
        for (int index = begin; index < end; ++index) {
          const auto order = effective[index];
          if (order.op != fastkag::Op::SELL) continue;
          auto prior = std::find_if(kept.begin(), kept.end(),
              [&](const auto& value) { return value.item == order.item; });
          if (prior == kept.end()) kept.push_back(order);
          else prior->quantity += order.quantity;
        }
        for (int offset = 0; offset < end - begin; ++offset)
          revised[begin + offset] = offset < int(kept.size())
              ? kept[offset] : fastkag::Action{};
        begin = end;
      }
      action.market = std::move(revised);
      return;
    }

    // Original E334 path: only compact runs of at least two cash-product
    // sales, leaving BUY/HIRE and buyable-product sale indices untouched.
    for (int begin = 0; begin < int(action.market.size());) {
      if (action.market[begin].op != fastkag::Op::SELL ||
          !is_cash(int(action.market[begin].item))) {
        ++begin;
        continue;
      }
      int end = begin + 1;
      while (end < int(action.market.size()) &&
             action.market[end].op == fastkag::Op::SELL &&
             is_cash(int(action.market[end].item)))
        ++end;
      if (end - begin >= 2) {
        std::vector<fastkag::Action> kept;
        for (int index = begin; index < end; ++index) {
          const auto order = action.market[index];
          auto prior = std::find_if(kept.begin(), kept.end(),
              [&](const auto& value) { return value.item == order.item; });
          if (prior == kept.end()) kept.push_back(order);
          else prior->quantity += std::max(0, order.quantity);
        }
        for (auto& order : kept) {
          const int item = int(order.item);
          order.quantity = std::min(std::max(0, order.quantity),
                                    std::max(0, remaining[item]));
          remaining[item] -= order.quantity;
        }
        kept.erase(std::remove_if(kept.begin(), kept.end(),
            [](const auto& order) { return order.quantity <= 0; }), kept.end());
        for (int offset = 0; offset < end - begin; ++offset)
          action.market[begin + offset] = offset < int(kept.size())
              ? kept[offset] : fastkag::Action{};
      }
      begin = end;
    }
  }

  void adv_apply(const fastkag::Simulator& env, int player,
                 const std::vector<fastkag::PlayerAction>& current_tape,
                 fastkag::PlayerAction& action) const {
    const int step = env.step_count();
    if (env.hour() == 23 || step < 144 || step >= 718) return;
    struct Planned { int step, item, quantity; };
    std::vector<Planned> plan;
    std::optional<fastkag::Action> first;
    for (int offset = 1; offset <= 3 && step + offset <= 718; ++offset) {
      const int future = step + offset;
      const auto& tape = future >= 648
          ? assets.library.routes[slot_for(2)] : current_tape;
      if (future >= int(tape.size())) continue;
      for (const auto& order : tape[future].market) {
        if (order.op == fastkag::Op::PASS) continue;
        if (!first) first = order;
        const int item = int(order.item);
        if (order.op == fastkag::Op::SELL &&
            item >= int(fastkag::Item::CARROT) &&
            item <= int(fastkag::Item::WOOL) && order.quantity > 0)
          plan.push_back({future, item, order.quantity});
      }
    }
    if (first && first->op == fastkag::Op::SELL) {
      const int protected_item = int(first->item);
      plan.erase(std::remove_if(plan.begin(), plan.end(),
          [&](const auto& row) { return row.item == protected_item; }),
          plan.end());
    }
    if (plan.empty() || std::any_of(action.market.begin(), action.market.end(),
        [](const auto& order) {
          return order.op == fastkag::Op::BUY_PRODUCT;
        }))
      return;
    auto stock = python_projected_shed(env, player, action);
    std::array<int, fastkag::N_ITEMS> selling{};
    for (const auto& order : action.market) {
      const int item = int(order.item);
      if (order.op == fastkag::Op::SELL && item >= 0 &&
          item < fastkag::N_ITEMS)
        selling[item] += std::max(0, order.quantity);
    }
    std::array<bool, fastkag::N_ITEMS> picked{};
    for (const auto& command : action.units) {
      const int item = int(command.item);
      if (command.op == fastkag::Op::PICKUP && item >= 0 &&
          item < fastkag::N_ITEMS)
        picked[item] = true;
    }
    std::vector<int> items;
    for (const auto& row : plan)
      if (std::find(items.begin(), items.end(), row.item) == items.end())
        items.push_back(row.item);
    std::stable_sort(items.begin(), items.end(), [&](int left, int right) {
      return env.market().prices[left] > env.market().prices[right];
    });
    std::vector<fastkag::Action> extra;
    for (int item : items) {
      if (picked[item] || env.market().prices[item] < 2) continue;
      int available = stock[item] - selling[item];
      if (available < 1) continue;
      auto existing = std::find_if(action.market.begin(), action.market.end(),
          [&](const auto& order) {
            return order.op == fastkag::Op::SELL && int(order.item) == item;
          });
      if (existing == action.market.end() &&
          action.market.size() + extra.size() >= 10)
        continue;
      int quantity = 0;
      for (const auto& row : plan) {
        if (row.item != item || available <= 0) continue;
        const int take = std::min(row.quantity, available);
        quantity += take;
        available -= take;
      }
      if (quantity <= 0) continue;
      if (existing != action.market.end()) existing->quantity += quantity;
      else extra.push_back(
          {fastkag::Op::SELL, fastkag::Item(item), quantity});
    }
    if (!extra.empty()) {
      extra.insert(extra.end(), action.market.begin(), action.market.end());
      action.market = std::move(extra);
    }
  }

  static void drop_unwatered_last_hour_plants(
      const fastkag::Simulator& env, int player,
      fastkag::PlayerAction& action) {
    if (env.hour() != 23 || !std::any_of(
            action.units.begin(), action.units.end(), [](const auto& command) {
              return command.op == fastkag::Op::PLANT;
            }))
      return;
    const auto positions = fastkag::positions(env, player);
    for (std::size_t pass = 0; pass <= action.units.size(); ++pass) {
      std::array<fastkag::PlayerAction, 2> joint{};
      joint[player] = action;
      const auto after = env.preview_unit_phase(joint);
      std::vector<std::size_t> rejected;
      for (std::size_t actor = 0;
           actor < action.units.size(); ++actor) {
        const auto& command = action.units[actor];
        if (command.op != fastkag::Op::PLANT) continue;
        bool valid = false;
        if (actor < positions.size()) {
          const auto position = positions[actor];
          const auto& tile = after.farms()[player].tiles[
              int(position.y) * env.config().board_size + int(position.x)];
          valid = tile.kind == fastkag::TileKind::PLANT &&
                  tile.crop == command.item &&
                  tile.planted_day == env.day() && tile.watered_today;
        }
        if (!valid) rejected.push_back(actor);
      }
      if (rejected.empty()) break;
      for (const auto actor : rejected) action.units[actor] = {};
    }
  }

  static void apply_capharv(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& state) {
    const int step = env.step_count(), day = env.day();
    auto positions = fastkag::positions(env, player);
    using Visit = std::pair<fastkag::Position, fastkag::Op>;
    std::vector<Visit> visits;
    static constexpr std::array<fastkag::Position, 4> access{{
        {4, 4}, {5, 4}, {4, 5}, {5, 5}}};
    const auto is_move = [](fastkag::Op op) {
      return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
             op == fastkag::Op::EAST || op == fastkag::Op::WEST;
    };
    for (int future = step; future < (day + 1) * 24 &&
                              future < int(tape.size()); ++future) {
      const auto& turn = future == step ? action : tape[future];
      for (std::size_t actor = 0; actor < positions.size(); ++actor) {
        const fastkag::Action command = actor < turn.units.size()
            ? turn.units[actor] : fastkag::Action{};
        if (is_move(command.op)) {
          auto next = positions[actor];
          if (command.op == fastkag::Op::NORTH) --next.y;
          else if (command.op == fastkag::Op::SOUTH) ++next.y;
          else if (command.op == fastkag::Op::EAST) ++next.x;
          else --next.x;
          if (next.x >= 0 && next.x < env.config().board_size &&
              next.y >= 0 && next.y < env.config().board_size)
            positions[actor] = next;
        } else if (future > step) {
          visits.push_back({positions[actor], command.op});
        }
      }
      for (const auto& order : turn.market) {
        if (order.op != fastkag::Op::HIRE) continue;
        std::array<int, 4> occupancy{};
        for (const auto position : positions)
          for (int index = 0; index < 4; ++index)
            if (position.x == access[index].x &&
                position.y == access[index].y)
              ++occupancy[index];
        int best = 0;
        for (int index = 1; index < 4; ++index)
          if (occupancy[index] < occupancy[best]) best = index;
        positions.push_back(access[best]);
      }
    }

    const auto current_positions = fastkag::positions(env, player);
    const auto& private_state = env.privates()[player];
    int carried = 0;
    for (const auto& inventory : private_state.inventories)
      for (int quantity : inventory) carried += std::max(0, quantity);
    for (std::size_t actor = 0;
         actor < action.units.size() && actor < current_positions.size();
         ++actor) {
      auto& command = action.units[actor];
      if (command.op != fastkag::Op::CARE &&
          command.op != fastkag::Op::COLLECT_FERTILIZER)
        continue;
      const auto& value = tile(env, player, current_positions[actor]);
      int product = -1, capacity = 0, first = 0, interval = 0;
      if (value.kind == fastkag::TileKind::ANIMAL &&
          value.animal == fastkag::Item::GOOSE) {
        product = int(fastkag::Item::EGG); capacity = 4; first = 4; interval = 1;
      } else if (value.kind == fastkag::TileKind::ANIMAL &&
                 value.animal == fastkag::Item::COW) {
        product = int(fastkag::Item::MILK); capacity = 6; first = 8; interval = 2;
      } else if (value.kind == fastkag::TileKind::ANIMAL &&
                 value.animal == fastkag::Item::SHEEP) {
        product = int(fastkag::Item::WOOL); capacity = 6; first = 6; interval = 3;
      } else {
        continue;
      }
      const int since = day + 1 - value.placed_day - first;
      if (since < 0 || since % interval != 0) continue;
      bool future_harvest = false, future_feed = false,
           future_collect = false;
      for (const auto& [position, op] : visits) {
        if (position.x != current_positions[actor].x ||
            position.y != current_positions[actor].y)
          continue;
        future_harvest |= op == fastkag::Op::HARVEST;
        future_feed |= op == fastkag::Op::FEED;
        future_collect |= op == fastkag::Op::COLLECT_FERTILIZER;
      }
      if (future_harvest) continue;
      const bool fed = value.fed_today || future_feed;
      const int production = 1 +
          (fed ? std::max(0, int(value.pending_care_bonus)) : 0);
      const int overflow = value.yield_units + production - capacity;
      if (overflow <= 0 || value.yield_units <= 0) continue;
      int saved = 0;
      if (command.op == fastkag::Op::COLLECT_FERTILIZER) {
        if (overflow * env.market().prices[product] <=
            env.market().prices[int(fastkag::Item::FERTILIZER)])
          continue;
        saved = overflow;
      } else {
        if (future_collect || overflow <= 1) continue;
        saved = overflow - 1;
      }
      if (count_item(private_state.shed) + carried + value.yield_units >= 90)
        continue;
      command = {fastkag::Op::HARVEST};
      state.capharv_credit[product] += saved;
    }
  }

  static void merge_cowswap_credit(
      const fastkag::Simulator& env, int player,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& state) {
    if (state.cowswap_credit <= 0) return;
    const auto stock = python_projected_shed(env, player, action);
    const int extra = std::min(
        state.cowswap_credit,
        std::max(0, stock[int(fastkag::Item::EGG)] -
                    sale_quantity(action, fastkag::Item::EGG)));
    if (extra <= 0) return;
    auto sale = std::find_if(action.market.begin(), action.market.end(),
        [](const auto& order) {
          return same(order, fastkag::Op::SELL, fastkag::Item::EGG);
        });
    if (sale != action.market.end()) sale->quantity += extra;
    else if (action.market.size() < 10)
      action.market.insert(action.market.begin(),
                           {fastkag::Op::SELL,
                            fastkag::Item::EGG, extra});
    else return;
    state.cowswap_credit -= extra;
  }

  static void sell_capharv_credit(
      const fastkag::Simulator& env, int player,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& state) {
    const auto stock = python_projected_shed(env, player, action);
    for (int product = 0; product < fastkag::N_PRODUCTS; ++product) {
      int& credit = state.capharv_credit[product];
      if (credit <= 0 || action.market.size() >= 10 ||
          env.market().prices[product] < 2)
        continue;
      const int extra = std::min(
          credit, std::max(0, stock[product] -
                                  sale_quantity(action,
                                                fastkag::Item(product))));
      if (extra <= 0) continue;
      auto sale = std::find_if(action.market.begin(), action.market.end(),
          [&](const auto& order) {
            return same(order, fastkag::Op::SELL,
                        fastkag::Item(product));
          });
      if (sale != action.market.end()) sale->quantity += extra;
      else action.market.insert(action.market.begin(),
                                {fastkag::Op::SELL,
                                 fastkag::Item(product), extra});
      credit -= extra;
    }
  }

  static void apply_v231_cattle(
      const fastkag::Simulator& env, int player,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& prefix,
      CattleCreditState& state) {
    const int step = env.step_count();
    if (step == 0 || step <= state.last_step) state = {};
    state.last_step = step;

    // The shared helper already implements the buy/pick/place substitution.
    // Preserve its pending-placement evidence before it consumes that queue,
    // then add V231's omitted harvest-credit and sale semantics here.
    for (const auto& pending : prefix.cattle_pending_places) {
      const auto& value = tile(env, player, pending.site);
      if (value.kind != fastkag::TileKind::ANIMAL ||
          value.animal != fastkag::Item::COW ||
          value.placed_day != pending.day)
        continue;
      auto site = std::find_if(
          state.sites.begin(), state.sites.end(), [&](const auto& entry) {
            return input_same_position(entry.site, pending.site);
          });
      if (site == state.sites.end())
        state.sites.push_back({pending.site, pending.day});
      else
        site->day = pending.day;
    }

    fastkag::apply_thomas_cattle_substitution(
        env, player, action, prefix);

    const auto positions = fastkag::positions(env, player);
    std::vector<fastkag::Position> harvested;
    for (std::size_t actor = 0;
         actor < action.units.size() && actor < positions.size(); ++actor) {
      if (action.units[actor].op != fastkag::Op::HARVEST) continue;
      const auto position = positions[actor];
      const auto site = std::find_if(
          state.sites.begin(), state.sites.end(), [&](const auto& entry) {
            return input_same_position(entry.site, position);
          });
      if (site == state.sites.end() ||
          std::any_of(harvested.begin(), harvested.end(), [&](auto prior) {
            return input_same_position(prior, position);
          }))
        continue;
      const auto& value = tile(env, player, position);
      if (value.kind != fastkag::TileKind::ANIMAL ||
          value.animal != fastkag::Item::COW ||
          value.placed_day != site->day)
        continue;
      state.milk_credit += std::max(0, int(value.yield_units));
      harvested.push_back(position);
    }

    if (state.milk_credit <= 0) return;
    const auto stock = python_projected_shed(env, player, action);
    int planned = 0;
    for (const auto& order : action.market)
      if (same(order, fastkag::Op::SELL, fastkag::Item::MILK))
        planned += std::max(0, order.quantity);
    const int extra = std::min(
        state.milk_credit,
        std::max(0, stock[int(fastkag::Item::MILK)] - planned));
    if (extra <= 0) return;
    for (auto& order : action.market) {
      if (!same(order, fastkag::Op::SELL, fastkag::Item::MILK) ||
          order.quantity <= 0)
        continue;
      order.quantity += extra;
      state.milk_credit -= extra;
      break;
    }
  }

  static void apply_family_prefix(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& state) {
    const auto sales_first = [&]() {
      for (std::size_t index = 0; index < action.market.size(); ++index) {
        if (action.market[index].op != fastkag::Op::SELL) continue;
        std::size_t cursor = index;
        while (cursor > 0) {
          const auto& previous = action.market[cursor - 1];
          if (previous.op == fastkag::Op::SELL ||
              ((previous.op == fastkag::Op::BUY_PRODUCT ||
                previous.op == fastkag::Op::BUY_ANIMAL) &&
               previous.item == action.market[cursor].item))
            break;
          std::swap(action.market[cursor - 1], action.market[cursor]);
          --cursor;
        }
      }
    };
    const int step = env.step_count();
    if (step == 0) {
      action.market = {
          {fastkag::Op::BUY_PRODUCT, fastkag::Item::WHEAT, 20},
          {fastkag::Op::SELL, fastkag::Item::WHEAT, 15}};
    } else if (step == 1) {
      action.market.erase(std::remove_if(action.market.begin(), action.market.end(),
          [](const auto& order) {
            return order.item == fastkag::Item::WHEAT &&
                   (order.op == fastkag::Op::BUY_PRODUCT ||
                    order.op == fastkag::Op::SELL);
          }), action.market.end());
    }
    fastkag::apply_thomas_wheat_replenishment_trim(
        env, player, tape, action);
    fastkag::apply_thomas_supply_guard(env, player, tape, action);
    fastkag::apply_thomas_courier(env, player, tape, action, state);
    sales_first();
  }

  static void apply_capharv_outer(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& state) {
    apply_capharv(env, player, tape, action, state);
    sell_capharv_credit(env, player, action, state);
  }

  static void apply_herd_outer(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& state) {
    fastkag::apply_thomas_herd2(env, player, tape, action, state);
    fastkag::apply_thomas_cowswap(env, player, tape, action, state);
    fastkag::observe_thomas_cowswap_harvest(env, player, action, state);
    merge_cowswap_credit(env, player, action, state);
  }

  struct CropVisit { int step, actor; fastkag::Op op; };

  std::vector<CropVisit> crop_visits(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      const fastkag::PlayerAction& action, fastkag::Position target,
      int start, int end) const {
    auto positions = fastkag::positions(env, player);
    std::vector<CropVisit> result;
    static constexpr std::array<fastkag::Position, 4> access{{
        {4, 4}, {5, 4}, {4, 5}, {5, 5}}};
    const auto is_move = [](fastkag::Op op) {
      return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
             op == fastkag::Op::EAST || op == fastkag::Op::WEST;
    };
    for (int future = env.step_count();
         future <= std::min(end, 718); ++future) {
      const auto& future_tape = future >= 648
          ? assets.library.routes[slot_for(2)] : tape;
      const auto& turn = future == env.step_count()
          ? action : future_tape[future];
      for (std::size_t actor = 0; actor < positions.size(); ++actor) {
        const auto command = actor < turn.units.size()
            ? turn.units[actor] : fastkag::Action{};
        if (is_move(command.op)) {
          auto next = positions[actor];
          if (command.op == fastkag::Op::NORTH) --next.y;
          else if (command.op == fastkag::Op::SOUTH) ++next.y;
          else if (command.op == fastkag::Op::EAST) ++next.x;
          else --next.x;
          if (next.x >= 0 && next.x < env.config().board_size &&
              next.y >= 0 && next.y < env.config().board_size)
            positions[actor] = next;
        } else if (future >= start && positions[actor].x == target.x &&
                   positions[actor].y == target.y) {
          result.push_back({future, int(actor), command.op});
        }
      }
      for (const auto& order : turn.market) {
        if (order.op != fastkag::Op::HIRE) continue;
        std::array<int, 4> occupancy{};
        for (const auto position : positions)
          for (int index = 0; index < 4; ++index)
            if (position.x == access[index].x &&
                position.y == access[index].y)
              ++occupancy[index];
        int best = 0;
        for (int index = 1; index < 4; ++index)
          if (occupancy[index] < occupancy[best]) best = index;
        positions.push_back(access[best]);
      }
      if (future % 24 == 23) positions.assign(1, {4, 4});
    }
    return result;
  }

  struct CarrotYield {
    int harvest{};
    int rescue{};
    int rescue_step{-1};
  };

  static CarrotYield carrot_yield_path(
      fastkag::Item crop, int planted_day,
      const std::vector<CropVisit>& visits, int initial_yield = 1,
      int fertilized_until = -1, int watered_day = -1,
      int now_step = 0) {
    const int max_day = crop == fastkag::Item::WHEAT ? 4 : 3;
    const int capacity = crop == fastkag::Item::WHEAT ? 6 : 4;
    const int water_start = (max_day + 1) / 2;
    const int lifespan = (planted_day + max_day + 1) * 24;
    int yield = initial_yield;
    CarrotYield result;
    const auto decays = [](int first_decay, int begin, int end) {
      begin = std::max(begin, first_decay);
      if (end <= begin) return 0;
      const int first = (begin - first_decay) % 2 == 0 ? begin : begin + 1;
      return first >= end ? 0 : (end - 1 - first) / 2 + 1;
    };
    for (const auto& visit : visits) {
      const int day = visit.step / 24;
      const int age = day - planted_day;
      const int current = yield - decays(lifespan, now_step, visit.step);
      if (current <= 0 && visit.step > lifespan) return result;
      if (visit.op == fastkag::Op::HARVEST) {
        result.harvest = age >= 2 ? std::max(0, current) : 0;
        return result;
      }
      if (visit.op == fastkag::Op::PLANT || visit.op == fastkag::Op::DIG ||
          visit.op == fastkag::Op::BUILD_COOP ||
          visit.op == fastkag::Op::BUILD_PASTURE)
        return result;
      if (age >= 2 && current > result.rescue && visit.step > now_step) {
        result.rescue = current;
        result.rescue_step = visit.step;
      }
      if (visit.op == fastkag::Op::WATER && age >= water_start &&
          age <= max_day && day != watered_day) {
        watered_day = day;
        yield = std::min(capacity, yield +
            (fertilized_until >= day ? 2 : 1));
      }
    }
    return result;
  }

  static int carrot_wheat_total(const fastkag::Simulator& env, int player) {
    const auto& private_state = env.privates()[player];
    int total = private_state.shed[int(fastkag::Item::WHEAT)];
    for (const auto& inventory : private_state.inventories)
      total += inventory[int(fastkag::Item::WHEAT)];
    return total;
  }

  int carrot_feed_need(const fastkag::Simulator& env,
                       const std::vector<fastkag::PlayerAction>& tape,
                       int feed_days = 1) const {
    int need = 0;
    const int stop = std::min(719, env.step_count() + 24 * feed_days);
    for (int future = env.step_count(); future <= stop; ++future) {
      if (future >= 719) continue;
      const auto& future_tape = future >= 648
          ? assets.library.routes[slot_for(2)] : tape;
      for (const auto& command : future_tape[future].units)
        if (command.op == fastkag::Op::FEED) ++need;
    }
    return need;
  }

  void apply_carrot2(const fastkag::Simulator& env, int player,
                     const std::vector<fastkag::PlayerAction>& tape,
                     fastkag::PlayerAction& action,
                     CarrotState& state, int feed_days = 1) const {
    const int step = env.step_count();
    if (step == 0 || step <= state.step) state = {};
    state.step = step;
    if (step > 717) return;
    const int day = env.day();
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    const auto positions = fastkag::positions(env, player);
    auto units = action.units;
    auto market = action.market;
    bool changed = false;
    const auto is_move = [](fastkag::Op op) {
      return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
             op == fastkag::Op::EAST || op == fastkag::Op::WEST;
    };

    for (auto iterator = state.tiles.begin(); iterator != state.tiles.end();) {
      const auto position = iterator->first;
      const int planted = iterator->second;
      const auto& value = tile(env, player, position);
      if (value.kind != fastkag::TileKind::PLANT ||
          value.crop != fastkag::Item::CARROT ||
          value.planted_day != planted) {
        iterator = state.tiles.erase(iterator);
        continue;
      }
      int actor = -1;
      for (int index = 0;
           index < int(positions.size()) && index < int(units.size()); ++index)
        if (positions[index].x == position.x &&
            positions[index].y == position.y) {
          actor = index;
          break;
        }
      if (actor < 0) {
        ++iterator;
        continue;
      }
      const auto command = units[actor];
      const int yield = value.yield_units;
      if (command.op == fastkag::Op::HARVEST) {
        if (day - planted >= 2 && yield > 0) {
          state.credit += yield;
          iterator = state.tiles.erase(iterator);
        } else {
          ++iterator;
        }
        continue;
      }
      if (is_move(command.op) || day - planted < 2 || yield <= 0) {
        ++iterator;
        continue;
      }
      const auto visits = crop_visits(
          env, player, tape, action, position, step,
          (planted + 5) * 24);
      const auto path = carrot_yield_path(
          fastkag::Item::CARROT, planted, visits, yield,
          value.fertilized_until_day, value.watered_today ? day : -1, step);
      if (yield > std::max(path.harvest, path.rescue)) {
        units[actor] = {fastkag::Op::HARVEST};
        state.credit += yield;
        iterator = state.tiles.erase(iterator);
        changed = true;
      } else {
        ++iterator;
      }
    }

    const int carrot = int(fastkag::Item::CARROT);
    const int wheat = int(fastkag::Item::WHEAT);
    const int carrot_price = env.market().prices[carrot];
    const int wheat_price = env.market().prices[wheat];
    const bool pays_now = 3 * carrot_price - 20 > 4 * wheat_price - 15;
    if (day >= 6 && day <= 28 && pays_now) {
      int planned_carrots = std::count_if(units.begin(), units.end(),
          [](const auto& command) {
            return same(command, fastkag::Op::PLANT,
                        fastkag::Item::CARROT);
          });
      int carrot_seeds = std::min(
          state.spare_carrot,
          private_state.seeds[carrot] - planned_carrots);
      std::optional<bool> wheat_ok;
      for (std::size_t actor = 0; actor < units.size(); ++actor) {
        if (!same(units[actor], fastkag::Op::PLANT,
                  fastkag::Item::WHEAT) ||
            actor >= positions.size() || carrot_seeds <= 0)
          continue;
        const auto position = positions[actor];
        if (tile(env, player, position).kind != fastkag::TileKind::EMPTY)
          continue;
        if (!wheat_ok)
          wheat_ok = carrot_wheat_total(env, player) >=
                     carrot_feed_need(env, tape, feed_days);
        if (!*wheat_ok) break;
        const auto visits = crop_visits(
            env, player, tape, action, position, step + 1,
            (day + 6) * 24);
        const auto wheat_path = carrot_yield_path(
            fastkag::Item::WHEAT, day, visits);
        const auto carrot_path = carrot_yield_path(
            fastkag::Item::CARROT, day, visits);
        const int carrot_units = std::max(
            carrot_path.harvest, carrot_path.rescue);
        if (carrot_units * carrot_price - 20 >
            wheat_path.harvest * wheat_price - 15) {
          units[actor] = {fastkag::Op::PLANT, fastkag::Item::CARROT};
          --carrot_seeds;
          --state.spare_carrot;
          auto found = std::find_if(state.tiles.begin(), state.tiles.end(),
              [&](const auto& row) {
                return row.first.x == position.x &&
                       row.first.y == position.y;
              });
          if (found == state.tiles.end())
            state.tiles.push_back({position, day});
          else
            found->second = day;
          ++state.spare_wheat;
          changed = true;
        }
      }
    }

    std::vector<fastkag::Action> revised;
    revised.reserve(market.size() + 1);
    for (auto order : market) {
      if (order.op == fastkag::Op::BUY_SEED &&
          (order.item == fastkag::Item::WHEAT ||
           order.item == fastkag::Item::CARROT)) {
        int& spare = order.item == fastkag::Item::WHEAT
            ? state.spare_wheat : state.spare_carrot;
        const int cut = std::min(std::max(0, order.quantity), spare);
        if (cut > 0) {
          spare -= cut;
          order.quantity -= cut;
          changed = true;
          if (order.quantity <= 0) continue;
        }
      }
      revised.push_back(order);
    }
    market = std::move(revised);
    if (day >= 6 && day <= 27 && pays_now && market.size() < 10) {
      const int planned = std::count_if(units.begin(), units.end(),
          [](const auto& command) {
            return same(command, fastkag::Op::PLANT,
                        fastkag::Item::CARROT);
          });
      const int have = private_state.seeds[carrot] - planned;
      int buying = 0;
      for (const auto& order : market)
        if (same(order, fastkag::Op::BUY_SEED,
                 fastkag::Item::CARROT))
          buying += order.quantity;
      const int quantity = 8 - have - buying;
      if (quantity > 0 && int(farm.money) >= 800 + 20 * quantity) {
        market.push_back({fastkag::Op::BUY_SEED,
                          fastkag::Item::CARROT, quantity});
        state.spare_carrot += quantity;
        changed = true;
      }
    }
    if (state.credit > 0 && carrot_price >= 2 && market.size() < 10) {
      fastkag::PlayerAction projected_action{units, market};
      const int stock = python_projected_shed(
          env, player, projected_action)[carrot];
      int selling = 0;
      for (const auto& order : market)
        if (same(order, fastkag::Op::SELL, fastkag::Item::CARROT))
          selling += order.quantity;
      const int quantity = std::min(state.credit, stock - selling);
      if (quantity > 0) {
        market.insert(market.begin(),
                      {fastkag::Op::SELL,
                       fastkag::Item::CARROT, quantity});
        state.credit -= quantity;
        changed = true;
      }
    }
    if (changed) {
      action.units = std::move(units);
      if (market.size() > 10) market.resize(10);
      action.market = std::move(market);
    }
  }

  static int crop_harvest_yield(
      fastkag::Item crop, int planted_day,
      const std::vector<CropVisit>& visits, int initial_yield,
      int fertilized_until, int watered_day, int now_step) {
    const int max_day = crop == fastkag::Item::WHEAT ? 4 : 3;
    const int capacity = crop == fastkag::Item::WHEAT ? 6 : 4;
    const int water_start = (max_day + 1) / 2;
    const int lifespan = (planted_day + max_day + 1) * 24;
    int yield = initial_yield;
    const auto decays = [](int first_decay, int begin, int end) {
      begin = std::max(begin, first_decay);
      if (end <= begin) return 0;
      const int first = (begin - first_decay) % 2 == 0 ? begin : begin + 1;
      return first >= end ? 0 : (end - 1 - first) / 2 + 1;
    };
    for (const auto& visit : visits) {
      const int day = visit.step / 24;
      const int age = day - planted_day;
      const int current = yield - decays(lifespan, now_step, visit.step);
      if (current <= 0 && visit.step > lifespan) return 0;
      if (visit.op == fastkag::Op::HARVEST)
        return age >= 2 ? std::max(0, current) : 0;
      if (visit.op == fastkag::Op::PLANT || visit.op == fastkag::Op::DIG ||
          visit.op == fastkag::Op::BUILD_COOP ||
          visit.op == fastkag::Op::BUILD_PASTURE)
        return 0;
      if (visit.op == fastkag::Op::WATER &&
          age >= water_start && age <= max_day && day != watered_day) {
        watered_day = day;
        yield = std::min(capacity, yield +
            (fertilized_until >= day ? 2 : 1));
      }
    }
    return 0;
  }

  void apply_e410(const fastkag::Simulator& env, int player,
                  const std::vector<fastkag::PlayerAction>& tape,
                  fastkag::PlayerAction& action) const {
    if (!std::any_of(action.units.begin(), action.units.end(),
                     [](const auto& command) {
                       return command.op == fastkag::Op::FERTILIZE;
                     }))
      return;
    int expected = 0;
    const int begin = env.day() * 24;
    for (int future = begin;
         future < std::min(begin + 24, int(tape.size())); ++future)
      expected = std::max(expected, int(tape[future].units.size()) - 1);
    const auto positions = fastkag::positions(env, player);
    fastkag::PlayerAction prefix = action;
    std::fill(prefix.units.begin(), prefix.units.end(), fastkag::Action{});
    for (std::size_t actor = 0;
         actor < action.units.size() && actor < positions.size(); ++actor) {
      auto command = action.units[actor];
      std::array<fastkag::PlayerAction, 2> joint{};
      joint[player] = prefix;
      const auto before = env.preview_unit_phase(joint);
      const auto& value = tile(before, player, positions[actor]);
      const auto& inventories = before.privates()[player].inventories;
      if (command.op == fastkag::Op::FERTILIZE &&
          value.kind == fastkag::TileKind::PLANT &&
          (value.crop == fastkag::Item::WHEAT ||
           value.crop == fastkag::Item::CARROT) &&
          actor < inventories.size() &&
          inventories[actor][int(fastkag::Item::FERTILIZER)] > 0) {
        const bool covered = value.fertilized_until_day >= env.day() + 2;
        bool skip = covered;
        if (!skip && int(actor) <= expected) {
          const int end = std::min(718, (int(value.planted_day) + 6) * 24);
          const auto visits = crop_visits(
              env, player, tape, action, positions[actor],
              env.step_count() + 1, end);
          const int watered_day = value.watered_today ? env.day() : -1;
          const int old_yield = crop_harvest_yield(
              value.crop, value.planted_day, visits, value.yield_units,
              value.fertilized_until_day, watered_day, env.step_count());
          const int new_yield = crop_harvest_yield(
              value.crop, value.planted_day, visits, value.yield_units,
              std::max<int>(value.fertilized_until_day, env.day() + 2),
              watered_day, env.step_count());
          skip = old_yield > 0 && old_yield == new_yield;
        }
        if (skip) command = {};
      }
      action.units[actor] = command;
      prefix.units[actor] = command;
    }
  }

  void final_overlays(const fastkag::Simulator& env, int player,
                      const std::vector<fastkag::PlayerAction>& tape,
                      fastkag::PlayerAction& action) {
    const int step = env.step_count();
    // HybridOpening: v9 rewrites the wheat round-trip, then the final opening
    // adds the one seed used by the productive idle worker.
    if (step == 0 && action.market.size() < 10)
      action.market.push_back(
          {fastkag::Op::BUY_SEED, fastkag::Item::WHEAT, 1});

    // If the temporary wheat failed to materialize, restore the original
    // pasture job rather than watering an empty/weed tile.
    if (step == 29 && action.units.size() > 3 &&
        action.units[3].op == fastkag::Op::WATER) {
      const auto& tile = env.farms()[player].tiles[4 * env.config().board_size + 2];
      const bool valid = tile.kind == fastkag::TileKind::PLANT &&
                         tile.crop == fastkag::Item::WHEAT && tile.planted_day == 0;
      if (!valid) action.units[3] = {fastkag::Op::BUILD_PASTURE};
    }

    // Matured temporary-wheat delivery.  The immediately following visible
    // price guard owns the final decision, exactly as in Python.
    if (step == 91 && !env.farms()[player].hands.empty() &&
        env.farms()[player].hands[0].x == 4 &&
        env.farms()[player].hands[0].y == 4 && action.units.size() > 1 &&
        action.units[1].op == fastkag::Op::DROP) {
      const auto& inventories = env.privates()[player].inventories;
      const int amount = inventories.size() > 1
          ? inventories[1][int(fastkag::Item::WHEAT)] : 0;
      if (amount > 0) {
        auto it = std::find_if(action.market.begin(), action.market.end(),
            [](const fastkag::Action& order) {
              return same(order, fastkag::Op::SELL, fastkag::Item::WHEAT);
            });
        if (it != action.market.end()) it->quantity += amount;
        else if (action.market.size() < 10)
          action.market.push_back(
              {fastkag::Op::SELL, fastkag::Item::WHEAT, amount});
      }
    }
    if (step == 91 && env.market().prices[int(fastkag::Item::WHEAT)] < 31)
      action.market.erase(std::remove_if(action.market.begin(), action.market.end(),
          [](const fastkag::Action& order) {
            return same(order, fastkag::Op::SELL, fastkag::Item::WHEAT);
          }), action.market.end());

    // R127 rejects last-hour planting requests unless the post-unit state has
    // the newly planted crop watered on the same turn.
    drop_unwatered_last_hour_plants(env, player, action);

    // v44y's public lockstep best-response ordering runs before ADV.
    v44_reorder(env, player, action);

    // E334/E335 enforce the post-unit physical stock bound while preserving
    // external market indices with explicit empty slots.
    e335_compact(env, player, action);

    // E410 removes fertilizer uses that cannot improve the crop's planned
    // harvest under the future route visit schedule.
    apply_e410(env, player, tape, action);

    // ADV pulls already-held cash products at most three tape turns forward;
    // the shipped build does not book these pulls into R36's debt ledger.
    adv_apply(env, player, tape, action);

    // ADV's final stable partition: unrelated cash sales, then product buys
    // plus same-product sales in their original order, then all other orders.
    // This is stronger than the shared Thomas prefix's sales-only bubbling.
    if (step >= 144 && action.market.size() >= 2) {
      std::vector<fastkag::Action> filtered;
      filtered.reserve(action.market.size());
      for (const auto& order : action.market)
        if (order.op != fastkag::Op::PASS) filtered.push_back(order);
      std::array<bool, fastkag::N_ITEMS> product_buys{};
      for (const auto& order : filtered)
        if (order.op == fastkag::Op::BUY_PRODUCT && int(order.item) >= 0)
          product_buys[int(order.item)] = true;
      std::vector<fastkag::Action> front, middle, rest;
      front.reserve(filtered.size());
      middle.reserve(filtered.size());
      rest.reserve(filtered.size());
      for (const auto& order : filtered) {
        const int item = int(order.item);
        const bool bought = item >= 0 && item < fastkag::N_ITEMS &&
                            product_buys[item];
        if (order.op == fastkag::Op::SELL && !bought)
          front.push_back(order);
        else if (order.op == fastkag::Op::BUY_PRODUCT ||
                 (order.op == fastkag::Op::SELL && bought))
          middle.push_back(order);
        else
          rest.push_back(order);
      }
      front.insert(front.end(), middle.begin(), middle.end());
      front.insert(front.end(), rest.begin(), rest.end());
      const auto equal = [](const fastkag::Action& left,
                            const fastkag::Action& right) {
        return left.op == right.op && left.item == right.item &&
               left.quantity == right.quantity;
      };
      if (front.size() != filtered.size() ||
          !std::equal(front.begin(), front.end(), filtered.begin(), equal))
        action.market = std::move(front);
    }

    // IG final queue closure: an unavailable cash-product sale becomes an
    // explicit no-op slot, then later executable cash sales are pulled left
    // into those holes.  Keeping the holes is important for simultaneous
    // market-list priority against a non-PASS opponent.
    if (action.market.size() >= 2) {
      auto remaining = python_projected_shed(env, player, action);
      const auto cash = [](int item) {
        return item >= int(fastkag::Item::CARROT) &&
               item <= int(fastkag::Item::WOOL);
      };
      std::vector<std::size_t> holes;
      for (std::size_t index = 0; index < action.market.size(); ++index) {
        auto& order = action.market[index];
        const int item = int(order.item);
        if (order.op == fastkag::Op::SELL && cash(item)) {
          const int executed = std::min(std::max(0, order.quantity),
                                        std::max(0, remaining[item]));
          remaining[item] -= executed;
          if (executed <= 0) order = {};
        }
        if (order.op == fastkag::Op::PASS) {
          holes.push_back(index);
          continue;
        }
        const int revised_item = int(order.item);
        const bool movable = order.op == fastkag::Op::SELL &&
                             cash(revised_item) && order.quantity > 0;
        if (!movable || holes.empty()) continue;
        const auto target = holes.front();
        holes.erase(holes.begin());
        action.market[target] = order;
        order = {};
        holes.push_back(index);
      }
    }

    // Final-day seed-float trim.  CARROT2 intentionally maintains an
    // eight-seed buffer, but from step 648 onward only the route-2 plant
    // commands through step 671 can still yield before the episode ends.
    // Cap today's buys to that exact remaining demand (plus the script's
    // two-unit wheat hedge).
    if (step >= 648) {
      const auto& route2 = assets.library.routes[slot_for(2)];
      int wheat_ahead = 0, carrot_ahead = 0;
      for (int future = step + 1;
           future <= 671 && future < int(route2.size()); ++future)
        for (const auto& command : route2[future].units) {
          if (same(command, fastkag::Op::PLANT, fastkag::Item::WHEAT))
            ++wheat_ahead;
          else if (same(command, fastkag::Op::PLANT,
                        fastkag::Item::CARROT))
            ++carrot_ahead;
        }
      int planting_wheat = 0, planting_carrot = 0;
      for (const auto& command : action.units) {
        planting_wheat += same(command, fastkag::Op::PLANT,
                               fastkag::Item::WHEAT);
        planting_carrot += same(command, fastkag::Op::PLANT,
                                fastkag::Item::CARROT);
      }
      const int wheat_price =
          env.market().prices[int(fastkag::Item::WHEAT)];
      const int carrot_price =
          env.market().prices[int(fastkag::Item::CARROT)];
      const bool swapping =
          3 * carrot_price - 20 > 4 * wheat_price - 15 && env.day() <= 28;
      const int need_carrot =
          carrot_ahead + (swapping ? wheat_ahead : 0);
      const int need_wheat = swapping ? 0 : wheat_ahead;
      int carrot_left =
          env.privates()[player].seeds[int(fastkag::Item::CARROT)] -
          planting_carrot;
      const int wheat_left =
          env.privates()[player].seeds[int(fastkag::Item::WHEAT)] -
          planting_wheat;
      int allow_carrot = std::max(0, need_carrot - carrot_left);
      std::vector<fastkag::Action> carrot_trimmed;
      carrot_trimmed.reserve(action.market.size());
      for (auto order : action.market) {
        if (same(order, fastkag::Op::BUY_SEED,
                 fastkag::Item::CARROT)) {
          const int quantity = std::min(order.quantity, allow_carrot);
          allow_carrot -= quantity;
          carrot_left += quantity;
          if (quantity <= 0) continue;
          order.quantity = quantity;
        }
        carrot_trimmed.push_back(order);
      }
      const int carrot_shortage = std::max(0, need_carrot - carrot_left);
      int allow_wheat = std::max(
          0, std::min(wheat_ahead, need_wheat + carrot_shortage + 2) -
                 wheat_left);
      std::vector<fastkag::Action> final;
      final.reserve(carrot_trimmed.size());
      for (auto order : carrot_trimmed) {
        if (same(order, fastkag::Op::BUY_SEED,
                 fastkag::Item::WHEAT)) {
          const int quantity = std::min(order.quantity, allow_wheat);
          allow_wheat -= quantity;
          if (quantity <= 0) continue;
          order.quantity = quantity;
        }
        final.push_back(order);
      }
      action.market = std::move(final);
    }

    // Final public knockout layer.
    if (step >= 696 && step <= 719)
      action.market.erase(std::remove_if(action.market.begin(), action.market.end(),
          [](const fastkag::Action& order) {
            return same(order, fastkag::Op::BUY_PRODUCT,
                        fastkag::Item::FERTILIZER);
          }), action.market.end());
  }

  // The SHOP wrapper replaces the complete action on the final playable
  // turn: every loaded unit at shed access drops, then all projected shed
  // stock is sold by descending gross value.  Python's sort is stable, so
  // product enum order is the tie-breaker.
  static void apply_terminal_rescue(const fastkag::Simulator& env, int player,
                                    fastkag::PlayerAction& action) {
    if (env.step_count() < 718) return;
    const auto positions = fastkag::positions(env, player);
    const auto& private_state = env.privates()[player];
    action.units.assign(positions.size(), {});
    for (std::size_t actor = 0;
         actor < positions.size() && actor < private_state.inventories.size();
         ++actor) {
      const bool loaded = std::any_of(
          private_state.inventories[actor].begin(),
          private_state.inventories[actor].end(),
          [](int quantity) { return quantity > 0; });
      if (loaded && fastkag::shed_adjacent(positions[actor]))
        action.units[actor] = {fastkag::Op::DROP, fastkag::Item::NONE, 1};
    }
    const auto projected = python_projected_shed(env, player, action);
    action.market.clear();
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item)
      if (projected[item] > 0)
        action.market.push_back({fastkag::Op::SELL, fastkag::Item(item),
                                 projected[item]});
    std::stable_sort(action.market.begin(), action.market.end(),
                     [&](const auto& left, const auto& right) {
      const auto value = [&](const auto& order) {
        return std::int64_t(env.market().prices[int(order.item)]) *
               order.quantity;
      };
      return value(left) > value(right);
    });
  }

  // EXP-154 is an old wrapper immediately outside SHOP.  It deliberately
  // reasons from the observed shed plus carried inventory (not projected
  // post-unit stock), then appends enough high-price sales to leave one slot
  // of headroom for the dawn deposit.  Later wrappers may rewrite these
  // orders, so preserve its exact position in the public call chain.
  static void apply_pre_room_guard(const fastkag::Simulator& env, int player,
                                   fastkag::PlayerAction& action) {
    if (env.hour() != 23) return;
    const auto& private_state = env.privates()[player];
    int carried = 0;
    for (const auto& inventory : private_state.inventories)
      for (int quantity : inventory) carried += std::max(0, quantity);
    int needed = carried;
    for (int quantity : private_state.shed)
      needed += std::max(0, quantity);
    needed -= 99;
    if (needed <= 0) return;

    std::array<int, fastkag::N_PRODUCTS> planned{};
    for (const auto& order : action.market) {
      const int item = int(order.item);
      if (order.op == fastkag::Op::SELL && item >= 0 &&
          item < fastkag::N_PRODUCTS)
        planned[item] += std::max(0, order.quantity);
    }
    std::array<int, fastkag::N_PRODUCTS> items{};
    std::iota(items.begin(), items.end(), 0);
    std::stable_sort(items.begin(), items.end(), [&](int left, int right) {
      return env.market().prices[left] > env.market().prices[right];
    });
    for (int item : items) {
      const int quantity = std::min(
          needed,
          std::max(0, private_state.shed[item] - planned[item]));
      if (quantity <= 0) continue;
      if (action.market.size() >= 10) break;
      action.market.push_back(
          {fastkag::Op::SELL, fastkag::Item(item), quantity});
      needed -= quantity;
      if (needed <= 0) break;
    }
  }

  fastkag::PlayerAction action(const fastkag::Simulator& env, int player) {
    if (player < 0 || player > 1) throw std::invalid_argument("player must be 0 or 1");
    auto& state = seats[player];
    const int step = env.step_count();
    if (step == 0 && state.last_step >= 0) state = {};
    if (step < state.last_step)
      throw std::logic_error("2965 opponent observed a non-monotonic episode");
    update_route(env, player, state);
    const int slot = slot_for(state.route_id);
    state.native.wheat_credit = 0;
    // The shared executor's legacy weed transaction unconditionally replays a
    // blocked action one turn later.  The public Chassis keeps a queue and
    // drops a queued PLANT when the following raw command moves away.  Keep
    // shared weed state stateless and apply that exact queue below.
    for (auto& transaction : state.native.weed) transaction = {};
    // The repository executor has a generic late-capacity evacuation that is
    // not part of the public 2965 Chassis.  Its hour-23 DROP/sale is owned by
    // the script's later SHEDROOM wrapper, so suppress only this terminal arm.
    if (env.hour() == 23) state.native.room_evac.active = false;
    auto result = executor.action_external(
        env, player, slot, state.native,
        fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true);
    for (auto& transaction : state.native.weed) transaction = {};
    undo_shared_feed_guard(
        env, player, assets.library.routes[slot][step], result, state.native);
    apply_exact_weed_repair(
        env, player, assets.library.routes[slot], result, state);
    // NativeTeammateExecutor has a repository-wide hour-23 room guard that is
    // not part of this public script's Chassis.  In particular it may sell a
    // product which the same tape action is buying.  Preserve any tape-native
    // sale, but remove only that executor-added same-product quantity; the
    // canonical SHEDROOM wrapper is applied later with its future-use reserve.
    if (env.hour() == 23) {
      std::array<bool, fastkag::N_ITEMS> bought{};
      std::array<int, fastkag::N_ITEMS> tape_sales{};
      for (const auto& order : assets.library.routes[slot][step].market) {
        const int item = int(order.item);
        if (order.op == fastkag::Op::BUY_PRODUCT && item >= 0 &&
            item < fastkag::N_ITEMS)
          bought[item] = true;
        else if (order.op == fastkag::Op::SELL && item >= 0 &&
                 item < fastkag::N_ITEMS)
          tape_sales[item] += std::max(0, order.quantity);
      }
      for (int item = 0; item < fastkag::N_ITEMS; ++item)
        // R36 disables Chassis' lead-sale arm on [288,696), so every
        // above-tape sale in this hour-23 window comes from the repository
        // executor's unrelated capacity guard.  Preserve genuine Chassis
        // suppressions by capping rather than replacing the quantity.
        if (bought[item] || (step >= 288 && step < 696))
          cap_sale_quantity(result, fastkag::Item(item), tape_sales[item]);
    }
    // RACEPX is a Chassis hook in Python, hence it executes before all of the
    // outer wrappers, including SHOP and EXP-154.  R36 monkeypatches the
    // Chassis suppression hook: first consume RACEPX's one-step suppression,
    // then its own reservation debts, then open today's RACEPX lead.
    suppress_racepx_sales(result, state, step);
    suppress_reserved_sales(result, state, step);
    apply_racepx_lead(
        env, player, assets.library.routes[slot], result, state);
    // SHOP is the first wrapper outside Chassis.  Its step-718 liquidation is
    // intentionally still visible to every later market-order wrapper.
    apply_terminal_rescue(env, player, result);
    apply_pre_room_guard(env, player, result);
    // R36's new reservations are still added by its outer wrapper below.
    // V219 and V231 are inner parents of R36 in the authoritative stack.
    // In particular V231 credit must not consume a sale that R36 adds later.
    apply_v219(env, player, assets.library.routes[slot], result, state.tomato);
    apply_v231_cattle(
        env, player, result, state.prefix, state.cattle_credit);
    // The first v224 wrapper is outside Chassis but inside R36: it removes
    // zero/dead tape slots before the generic reservation tries to append a
    // new sale.
    if (step >= 288) v224_sales_first(result);
    reserve_future_sales(
        env, player, assets.library.routes[slot], result, state);
    // R36's outer wrapper immediately applies the second v224 pass and then
    // R37.  Both precede R97 and the other family overlays below.  In
    // particular R97 may deliberately turn a WHEAT sale into a zero-quantity
    // slot; that hole must remain visible to CARROT2 until the later queue
    // compaction wrappers run.
    if (step >= 288) {
      v224_sales_first(result);
      r37_reorder_sales(env, player, result);
    }
    const int base_fertilizer_sale =
        sale_quantity(result, fastkag::Item::FERTILIZER);
    static const std::vector<std::vector<fastkag::NativeTapeLibrary::ThomasPredictEvent>>
        empty_predictors;
    const auto& predictors = env.shops().size() >= 2
        ? assets.library.thomas_predict_pairs[
              int(env.shops()[0]) * 8 + int(env.shops()[1])]
        : empty_predictors;
    apply_family_prefix(
        env, player, assets.library.routes[slot], result, state.prefix);
    // The shared Thomas overlay has an unconditional drop-credit fertilizer
    // sale.  2965's later RACEGATE forbids only the overlay-added quantity
    // while the fertilizer book is at/below its $100 base; tape-native sales
    // remain untouched.
    if (env.market().prices[int(fastkag::Item::FERTILIZER)] <= 100)
      cap_sale_quantity(result, fastkag::Item::FERTILIZER,
                        base_fertilizer_sale);
    result = apply_sheep(
        env, player, assets.library.routes[slot], std::move(result), state.sheep);
    apply_input51(env, player, assets.library.routes[slot], result, state);
    apply_r51_close_warehouse(
        env, player, assets.library.routes[slot], result);
    apply_r85_feed(env, player, assets.library.routes[slot], result);
    apply_r85_fertilizer(env, player, result, state);
    apply_v9_herd(
        env, player, assets.library.routes[slot], result, state.v9_herd);
    apply_v9_fertilizer(env, player, assets.library.routes[slot], result);
    // V92 observes the action after V9 HERD/FERT.  This lets a removed EGG
    // sale free a slot for its predicted premium sale and keeps its public
    // inventory ledger aligned with the action that is actually emitted.
    fastkag::apply_thomas_predict(
        env, player, assets.library.routes[slot], predictors, result,
        state.prefix);
    apply_overflow(env, player, result);
    // These four wrappers are ordered exactly as in the public Python stack:
    // CARROT2 -> ORDERPRI2 -> CAPHARV -> SHEDROOM -> HERD2/COWSWAP.
    apply_carrot2(
        env, player, assets.library.routes[slot], result, state.carrot);
    apply_or2(env, player, assets.library.routes[slot], result, state);
    apply_capharv_outer(
        env, player, assets.library.routes[slot], result, state.prefix);
    apply_shedroom(env, player, assets.library.routes[slot], result);
    apply_herd_outer(
        env, player, assets.library.routes[slot], result, state.prefix);
    final_overlays(env, player, assets.library.routes[slot], result);
    state.last_step = step;
    return result;
  }

  std::shared_ptr<const SharedAssets> shared;
  const Assets& assets;
  const fastkag::NativeTeammateExecutor& executor;
  std::array<SeatState, 2> seats{};
};

Opponent::Opponent(const std::string& asset_path)
    : impl_(std::make_unique<Impl>(asset_path)) {}
Opponent::~Opponent() = default;
Opponent::Opponent(Opponent&&) noexcept = default;
Opponent& Opponent::operator=(Opponent&&) noexcept = default;
fastkag::PlayerAction Opponent::action(const fastkag::Simulator& env, int player) {
  return impl_->action(env, player);
}
fastkag::PlayerAction Opponent::fixture_route_action(int route_id,
                                                     int step) const {
  return impl_->fixture_route_action(route_id, step);
}
int Opponent::route(int player) const {
  if (player < 0 || player > 1) throw std::invalid_argument("player must be 0 or 1");
  return impl_->seats[player].route_id;
}
void Opponent::reset() { impl_->reset(); }

}  // namespace metav4_2965
