#include "salemali7_2900.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <vector>

namespace salemali7_2900 {
namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;

class Reader {
 public:
  explicit Reader(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open Salemali7 asset");
    bytes_ = {std::istreambuf_iterator<char>(stream), {}};
  }

  template <class T>
  T read() {
    if (offset_ + sizeof(T) > bytes_.size())
      throw std::runtime_error("truncated Salemali7 asset");
    T value{};
    std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  void expect(const char* value, std::size_t size) {
    if (offset_ + size > bytes_.size() ||
        std::memcmp(bytes_.data() + offset_, value, size) != 0)
      throw std::runtime_error("invalid Salemali7 asset magic");
    offset_ += size;
  }

  void skip(std::size_t size) {
    if (offset_ + size > bytes_.size())
      throw std::runtime_error("truncated Salemali7 asset header");
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
    throw std::runtime_error("invalid action in Salemali7 asset");
  return {static_cast<Op>(op), static_cast<Item>(item), quantity};
}

std::vector<PlayerAction> load_tape(const std::string& path) {
  Reader reader(path);
  reader.expect("SL72900\0", 8);
  if (reader.read<std::uint32_t>() != 1)
    throw std::runtime_error("unsupported Salemali7 asset version");
  reader.skip(32);  // authoritative Python source SHA-256
  const int frames = reader.read<std::uint16_t>();
  if (frames != 720)
    throw std::runtime_error("Salemali7 asset must contain 720 frames");
  std::vector<PlayerAction> tape;
  tape.reserve(frames);
  for (int frame = 0; frame < frames; ++frame) {
    const int units = reader.read<std::uint8_t>();
    const int market = reader.read<std::uint8_t>();
    if (units < 1 || units > 64 || market > 10)
      throw std::runtime_error("invalid action count in Salemali7 asset");
    PlayerAction action;
    action.units.reserve(units);
    action.market.reserve(market);
    for (int i = 0; i < units; ++i) action.units.push_back(read_action(reader));
    for (int i = 0; i < market; ++i) action.market.push_back(read_action(reader));
    tape.push_back(std::move(action));
  }
  if (!reader.done()) throw std::runtime_error("trailing Salemali7 asset data");
  return tape;
}

int quantity(const Action& action) { return std::max(0, action.quantity); }

}  // namespace

struct Opponent::Impl {
  struct WeedTransaction {
    int start{-1};
    Action intended{};
  };
  struct SeatState {
    int action_step{-1};
    int weed_step{-1};
    int front_run_step{-1};
    int due_step{-1};
    std::vector<std::optional<WeedTransaction>> weed;
    std::array<int, fastkag::N_PRODUCTS> due{};
  };

  explicit Impl(const std::string& path) : tape(load_tape(path)) {}

  std::vector<PlayerAction> tape;
  std::array<SeatState, 2> seats{};

  void reset() { seats = {}; }

  static const fastkag::Tile* tile_at(const Simulator& env, int player,
                                      fastkag::Position position) {
    const int n = env.config().board_size;
    if (position.x < 0 || position.y < 0 || position.x >= n || position.y >= n)
      return nullptr;
    return &env.farms()[player].tiles[position.y * n + position.x];
  }

  Action source_action(int step, std::size_t actor) const {
    step = std::clamp(step, 0, int(tape.size()) - 1);
    return actor < tape[step].units.size() ? tape[step].units[actor] : Action{};
  }

  void weed_repair(PlayerAction& action, const Simulator& env, int player,
                   int step, SeatState& state) const {
    const std::size_t expected = env.farms()[player].hands.size() + 1;
    action.units.resize(expected);
    if (step <= 0 || step < state.weed_step) {
      state.weed.clear();
      state.weed_step = step;
    }
    state.weed_step = step;
    state.weed.resize(expected);

    for (std::size_t actor = 0; actor < expected; ++actor) {
      auto& transaction = state.weed[actor];
      if (!transaction) continue;
      const int elapsed = step - transaction->start;
      if (elapsed == 1) {
        action.units[actor] = transaction->intended;
      } else if (elapsed >= 2 && elapsed <= 9) {
        action.units[actor] = source_action(step - 1, actor);
      } else {
        transaction.reset();
      }
    }

    for (std::size_t actor = 0; actor < expected; ++actor) {
      if (state.weed[actor]) continue;
      const Action intended = action.units[actor];
      if (intended.op != Op::BUILD_PASTURE && intended.op != Op::PLANT) continue;
      const auto position = actor == 0 ? env.farms()[player].farmer
                                      : env.farms()[player].hands[actor - 1];
      const auto* tile = tile_at(env, player, position);
      if (!tile || tile->kind != fastkag::TileKind::WEED) continue;
      state.weed[actor] = WeedTransaction{step, intended};
      action.units[actor] = {Op::DIG};
    }
  }

  void front_run_state(int step, SeatState& state) const {
    if (step <= 0 || step < state.front_run_step) {
      state.front_run_step = step;
      state.due_step = -1;
      state.due.fill(0);
    }
    state.front_run_step = step;
    if (state.due_step >= 0 && state.due_step < step) {
      state.due_step = -1;
      state.due.fill(0);
    }
  }

  static void repay(PlayerAction& action, int step, SeatState& state) {
    if (state.due_step != step) return;
    std::vector<Action> market;
    market.reserve(action.market.size());
    auto debt = state.due;
    for (auto order : action.market) {
      const int item = int(order.item);
      if (order.op == Op::SELL && item >= 0 && item < fastkag::N_PRODUCTS &&
          debt[item] > 0) {
        int request = quantity(order);
        const int reduction = std::min(request, debt[item]);
        request -= reduction;
        debt[item] -= reduction;
        if (request == 0) continue;
        order.quantity = request;
      }
      if (market.size() < 10) market.push_back(order);
    }
    action.market = std::move(market);
    state.due_step = -1;
    state.due.fill(0);
  }

  static bool shop_demands(int shop, int item) {
    switch (shop) {
      case 0: return item == int(Item::EGG) || item == int(Item::WHEAT);
      case 1: return item == int(Item::EGG) || item == int(Item::WHEAT) ||
                     item == int(Item::STRAWBERRY);
      case 2: return item >= int(Item::WHEAT) && item <= int(Item::STRAWBERRY);
      case 3: return item == int(Item::STRAWBERRY) || item == int(Item::MILK) ||
                     item == int(Item::WHEAT);
      case 4: return item == int(Item::CARROT);
      case 5: return item == int(Item::MILK) || item == int(Item::TOMATO) ||
                     item == int(Item::WHEAT);
      case 6: return item == int(Item::STRAWBERRY) || item == int(Item::MILK);
      case 7: return item == int(Item::WOOL);
      default: return false;
    }
  }

  static int town_demand(const Simulator& env, int item, int step) {
    int demand = item != int(Item::FERTILIZER) && step % 24 == 0 ? 1 : 0;
    if (step % 4 != 0) return demand;
    for (const int shop : env.shops())
      if (shop_demands(shop, item)) demand += shop == 4 || shop == 7 ? 2 : 1;
    return demand;
  }

  int future_quantity(int step, int item) const {
    if (++step < 0 || step >= int(tape.size())) return 0;
    int total = 0;
    for (const auto& order : tape[step].market)
      if (order.op == Op::SELL && int(order.item) == item)
        total += quantity(order);
    return total;
  }

  static int reserved(const PlayerAction& action, int item) {
    int total = 0;
    for (const auto& unit : action.units)
      if (unit.op == Op::PICKUP && int(unit.item) == item)
        total += quantity(unit);
    for (const auto& order : action.market)
      if (order.op == Op::SELL && int(order.item) == item)
        total += quantity(order);
    return total;
  }

  void front_run(PlayerAction& action, const Simulator& env, int player,
                 int step, SeatState& state) const {
    static constexpr std::array<int, 4> items{
        int(Item::MELON), int(Item::MILK), int(Item::STRAWBERRY), int(Item::WOOL)};
    std::array<int, fastkag::N_PRODUCTS> moved{};
    for (const int item : items) {
      const int target = future_quantity(step, item);
      if (target <= 0 || town_demand(env, item, step) > 0) continue;
      const int stock = std::max(0, env.privates()[player].shed[item]);
      const int amount = std::min(target, std::max(0, stock - reserved(action, item)));
      if (amount <= 0) continue;
      auto existing = std::find_if(action.market.begin(), action.market.end(),
          [item](const Action& order) {
            return order.op == Op::SELL && int(order.item) == item;
          });
      if (existing != action.market.end()) {
        existing->quantity = quantity(*existing) + amount;
      } else if (action.market.size() < 10) {
        action.market.push_back({Op::SELL, static_cast<Item>(item), amount});
      } else {
        continue;
      }
      moved[item] += amount;
    }
    if (std::any_of(moved.begin(), moved.end(), [](int value) { return value > 0; })) {
      state.due_step = step + 1;
      state.due = moved;
    }
  }

  PlayerAction action(const Simulator& env, int player) {
    if (player < 0 || player > 1) throw std::invalid_argument("seat must be 0 or 1");
    const int step = env.step_count();
    if (step < 0 || step >= 719)
      throw std::out_of_range("Salemali7 action requested outside live game");
    auto& state = seats[player];
    if (state.action_step < 0 && step != 0)
      throw std::logic_error("Salemali7 episode must start at step 0");
    if (step == 0) state = {};
    if (state.action_step >= 0 && step != state.action_step + 1)
      throw std::logic_error("Salemali7 action stream is not monotonic");
    state.action_step = step;

    PlayerAction result = tape[step];
    weed_repair(result, env, player, step, state);
    front_run_state(step, state);
    repay(result, step, state);
    front_run(result, env, player, step, state);
    result.units.resize(env.farms()[player].hands.size() + 1);
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
void Opponent::reset() { impl_->reset(); }

}  // namespace salemali7_2900
