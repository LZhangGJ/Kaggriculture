#include "production_obligation.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace production_obligation {
namespace {

constexpr std::array<int, kCrops> kSeedCost{10, 20, 50, 100, 80};
constexpr std::array<int, kAnimals> kAnimalCost{300, 400, 500};
constexpr std::array<int, 3> kLandCost{1000, 2000, 4000};
constexpr std::array<int, kCrops> kCropFirstDay{2, 2, 8, 10, 10};

struct Need {
  NodeKind kind = NodeKind::BuySeed;
  Item item = Item::None;
  int quantity = 0;
  int consumer_step = -1;
  int deadline = -1;
  int earliest = -1;
  int unit_cost = 0;
  int actor = -1;
  int quadrant = -1;
  int predecessor = -1;
  int successor = -1;
  int execution = -1;
  int slot = -1;
  std::string reason;
};

struct PickupOpportunity {
  int step = -1;
  int actor = -1;
  Item item = Item::None;
  int requested = 0;
  int filled = 0;
};

struct Failure {
  bool present = false;
  int step = -1;
  int actor = -1;
  Item item = Item::None;
  NodeKind purchase_kind = NodeKind::BuyProduct;
  int quantity = 1;
  int pickup_step = -1;
  std::string message;
};

int as_int(Item item) { return static_cast<int>(item); }
bool valid_item(Item item) { return as_int(item) >= 0 && as_int(item) < kItems; }
bool is_crop(Item item) { return as_int(item) >= 0 && as_int(item) < kCrops; }
bool is_animal(Item item) { return as_int(item) >= 9 && as_int(item) < 12; }
bool ongoing_crop(Item item) {
  return item == Item::Tomato || item == Item::Strawberry;
}
int fib_cost(int already_hired, int multiplier) {
  int a = 1;
  int b = 1;
  while (already_hired-- > 0) {
    const int c = a + b;
    a = b;
    b = c;
  }
  return a * multiplier;
}

int quadrant(Position p, int board_size) {
  const int half = board_size / 2;
  return (p.y < half ? 0 : 2) + (p.x < half ? 0 : 1);
}

bool shed_adjacent(Position p, int board_size) {
  const int half = board_size / 2;
  return (p.x == half - 1 || p.x == half) &&
         (p.y == half - 1 || p.y == half);
}

Position default_spawn(int board_size) {
  const int half = board_size / 2;
  return {half - 1, half - 1};
}

Position spawn_hand(const std::vector<Position>& positions, int board_size) {
  const int half = board_size / 2;
  const std::array<Position, 4> candidates{{
      {half - 1, half - 1}, {half, half - 1},
      {half - 1, half}, {half, half}}};
  std::array<int, 4> counts{};
  for (const Position p : positions) {
    for (int i = 0; i < 4; ++i) {
      if (p.x == candidates[i].x && p.y == candidates[i].y) ++counts[i];
    }
  }
  int best = 0;
  for (int i = 1; i < 4; ++i) if (counts[i] < counts[best]) best = i;
  return candidates[best];
}

int shed_sum(const std::array<int, kItems>& shed) {
  int total = 0;
  for (const int quantity : shed) total += quantity;
  return total;
}

void inventory_add(Inventory& inventory, Item item, int quantity) {
  if (!valid_item(item) || quantity <= 0) return;
  const int index = as_int(item);
  if (inventory.quantity[index] == 0) inventory.insertion_order.push_back(item);
  inventory.quantity[index] += quantity;
}

int inventory_take(Inventory& inventory, Item item, int quantity) {
  if (!valid_item(item) || quantity <= 0) return 0;
  const int index = as_int(item);
  const int taken = std::min(quantity, inventory.quantity[index]);
  inventory.quantity[index] -= taken;
  if (inventory.quantity[index] == 0) {
    inventory.insertion_order.erase(
        std::remove(inventory.insertion_order.begin(),
                    inventory.insertion_order.end(), item),
        inventory.insertion_order.end());
  }
  return taken;
}

void normalize_inventory(Inventory& inventory) {
  std::array<bool, kItems> seen{};
  std::vector<Item> normalized;
  normalized.reserve(kItems);
  for (const Item item : inventory.insertion_order) {
    if (!valid_item(item)) continue;
    const int index = as_int(item);
    if (!seen[index] && inventory.quantity[index] > 0) {
      seen[index] = true;
      normalized.push_back(item);
    }
  }
  for (int index = 0; index < kItems; ++index) {
    if (!seen[index] && inventory.quantity[index] > 0) {
      normalized.push_back(static_cast<Item>(index));
    }
  }
  inventory.insertion_order = std::move(normalized);
}

int tile_index(Position p, int board_size) {
  return p.y * board_size + p.x;
}

bool inside(Position p, int board_size) {
  return p.x >= 0 && p.y >= 0 && p.x < board_size && p.y < board_size;
}

void move(Position& p, UnitOp op, int board_size) {
  Position next = p;
  if (op == UnitOp::North) --next.y;
  if (op == UnitOp::South) ++next.y;
  if (op == UnitOp::West) --next.x;
  if (op == UnitOp::East) ++next.x;
  if (inside(next, board_size)) p = next;
}

bool is_move(UnitOp op) {
  return op == UnitOp::North || op == UnitOp::South ||
         op == UnitOp::East || op == UnitOp::West;
}

bool requires_unlocked_tile(UnitOp op) {
  switch (op) {
    case UnitOp::Plant:
    case UnitOp::Water:
    case UnitOp::Harvest:
    case UnitOp::Fertilize:
    case UnitOp::Dig:
    case UnitOp::BuildCoop:
    case UnitOp::BuildPasture:
    case UnitOp::Feed:
    case UnitOp::CollectFertilizer:
    case UnitOp::Care:
      return true;
    case UnitOp::Pass:
    case UnitOp::North:
    case UnitOp::South:
    case UnitOp::East:
    case UnitOp::West:
    case UnitOp::Drop:
    case UnitOp::Pickup:
    case UnitOp::Place:
      return false;
  }
  return false;
}

void add_diag(CompileResult& result, DiagnosticCode code, int step, int actor,
              Item item, int quantity, std::string message, bool fatal = true) {
  const auto duplicate = std::find_if(
      result.diagnostics.begin(), result.diagnostics.end(),
      [&](const Diagnostic& d) {
        return d.code == code && d.step == step && d.actor == actor &&
               d.item == item && d.message == message;
      });
  if (duplicate == result.diagnostics.end()) {
    result.diagnostics.push_back(
        {code, step, actor, item, quantity, std::move(message)});
  }
  if (fatal) result.feasible = false;
}

struct Builder {
  const CompilerInput& input;
  CompileResult result;
  std::vector<UnitFrame> frames;
  std::unordered_map<int, std::size_t> frame_at;
  std::vector<Need> needs;
  std::uint8_t virtual_unlocked = 1;

  explicit Builder(const CompilerInput& value) : input(value) {}

  int unit_cost(NodeKind kind, Item item, int ordinal = 0) const {
    if (kind == NodeKind::BuySeed && is_crop(item)) return kSeedCost[as_int(item)];
    if (kind == NodeKind::BuyAnimal && is_animal(item)) return kAnimalCost[as_int(item) - 9];
    if (kind == NodeKind::BuyProduct && valid_item(item)) {
      const int index = as_int(item);
      return index < kProducts ? std::max(1, input.current.product_price[index]) : 0;
    }
    if (kind == NodeKind::BuyLand && ordinal >= 0 && ordinal < 3) return kLandCost[ordinal];
    return 0;
  }

  void validate() {
    const CurrentState& current = input.current;
    if (current.step < 0 || current.turns_per_day <= 0 ||
        current.board_size <= 1 || current.board_size % 2 != 0 ||
        current.shed_capacity < 0 || current.max_market_orders <= 0) {
      add_diag(result, DiagnosticCode::InvalidInput, current.step, -1,
               Item::None, 0, "invalid scalar configuration");
    }
    if (input.maximum_resource_derivation_iterations < 0) {
      add_diag(result, DiagnosticCode::InvalidInput, current.step, -1,
               Item::None, 0,
               "maximum resource derivation iterations cannot be negative");
    }
    frames = input.future_units;
    std::sort(frames.begin(), frames.end(),
              [](const UnitFrame& a, const UnitFrame& b) { return a.step < b.step; });
    for (std::size_t index = 0; index < frames.size(); ++index) {
      if (frames[index].step < current.step ||
          (index > 0 && frames[index - 1].step == frames[index].step)) {
        add_diag(result, DiagnosticCode::InvalidInput, frames[index].step, -1,
                 Item::None, 0, "future unit steps must be unique and not precede current step");
      } else {
        frame_at[frames[index].step] = index;
      }
    }
    const int prefix = std::popcount(static_cast<unsigned>(current.unlocked_mask));
    const std::uint8_t expected = static_cast<std::uint8_t>((1u << prefix) - 1u);
    if (current.unlocked_mask == 0 || current.unlocked_mask != expected || prefix > 4) {
      add_diag(result, DiagnosticCode::InvalidLandPrefix, current.step, -1,
               Item::None, 0, "unlocked quadrants are not an official NW->NE->SW->SE prefix");
    }
    virtual_unlocked = current.unlocked_mask;
  }

  int add_need(NodeKind kind, Item item, int quantity, int consumer_step,
               int deadline, int earliest, int actor, int quadrant_value,
               int cost, std::string reason, bool merge = true) {
    if (quantity <= 0) return -1;
    if (deadline < earliest) {
      add_diag(result, DiagnosticCode::MissedAcquisitionDeadline,
               consumer_step, actor, item, quantity,
               "unit executes before any legal market acquisition slot");
      return -1;
    }
    if (merge) {
      for (std::size_t index = 0; index < needs.size(); ++index) {
        Need& need = needs[index];
        if (need.kind == kind && need.item == item && need.deadline == deadline &&
            need.earliest == earliest &&
            need.quadrant == quadrant_value && need.reason == reason) {
          need.quantity += quantity;
          if (need.actor != actor) need.actor = -1;
          return static_cast<int>(index);
        }
      }
    }
    needs.push_back({kind, item, quantity, consumer_step, deadline, earliest,
                     cost, actor, quadrant_value, -1, -1, -1, -1,
                     std::move(reason)});
    return static_cast<int>(needs.size()) - 1;
  }

  void derive_actor_and_land() {
    if (frames.empty()) return;
    const CurrentState& current = input.current;
    std::vector<Position> positions = current.actor_positions;
    if (positions.empty()) positions.push_back(default_spawn(current.board_size));
    std::vector<TileState> tiles = current.tiles;
    if (tiles.size() != static_cast<std::size_t>(current.board_size * current.board_size)) {
      tiles.resize(current.board_size * current.board_size);
      for (int y = 0; y < current.board_size; ++y) {
        for (int x = 0; x < current.board_size; ++x) {
          const int q = quadrant({x, y}, current.board_size);
          if ((current.unlocked_mask & (1u << q)) == 0) {
            tiles[y * current.board_size + x].kind = TileKind::Locked;
          }
        }
      }
    }
    int hires = current.hires_today;
    int day = current.step / current.turns_per_day;
    int land_tail = -1;
    std::map<int, int> hire_tail_by_day;

    auto add_land_through = [&](int target, int consumer_step, int actor) {
      while ((virtual_unlocked & (1u << target)) == 0) {
        const int extras = std::popcount(static_cast<unsigned>(virtual_unlocked)) - 1;
        if (extras >= 3) break;
        const int quadrant_to_buy = extras + 1;
        const int index = add_need(
            NodeKind::BuyLand, Item::None, 1, consumer_step, consumer_step - 1,
            current.step, actor, quadrant_to_buy, kLandCost[extras],
            "unlock ordered quadrant before first planned access", false);
        if (index >= 0 && land_tail >= 0) {
          needs[index].predecessor = land_tail;
          needs[land_tail].successor = index;
        }
        if (index >= 0) land_tail = index;
        virtual_unlocked |= static_cast<std::uint8_t>(1u << quadrant_to_buy);
        for (int y = 0; y < current.board_size; ++y) {
          for (int x = 0; x < current.board_size; ++x) {
            if (quadrant({x, y}, current.board_size) == quadrant_to_buy) {
              TileState& tile = tiles[y * current.board_size + x];
              if (tile.kind == TileKind::Locked) tile = TileState{};
            }
          }
        }
      }
    };

    const int last_step = frames.back().step;
    for (int step = current.step; step <= last_step; ++step) {
      const int now_day = step / current.turns_per_day;
      if (now_day != day) {
        positions.resize(1);
        positions[0] = default_spawn(current.board_size);
        hires = 0;
        day = now_day;
      }
      const auto frame_it = frame_at.find(step);
      if (frame_it == frame_at.end()) continue;
      const UnitFrame& frame = frames[frame_it->second];
      const int required = static_cast<int>(frame.actor_actions.size());
      if (required == 0) continue;
      if (required > static_cast<int>(positions.size())) {
        const int missing = required - static_cast<int>(positions.size());
        if (step % current.turns_per_day == 0) {
          add_diag(result, DiagnosticCode::ActorUnavailableAtDayStart, step,
                   static_cast<int>(positions.size()), Item::None, missing,
                   "hands hired on the preceding step are deleted by this day boundary");
        }
        for (int count = 0; count < missing; ++count) {
          const int earliest = std::max(current.step, now_day * current.turns_per_day);
          const int index = add_need(
              NodeKind::Hire, Item::None, 1, step, step - 1, earliest,
              static_cast<int>(positions.size()), -1,
              fib_cost(hires, current.farm_hand_cost_mult),
              "create actor before this day's first required action", false);
          const auto tail_it = hire_tail_by_day.find(now_day);
          if (index >= 0 && tail_it != hire_tail_by_day.end()) {
            needs[index].predecessor = tail_it->second;
            needs[tail_it->second].successor = index;
          }
          if (index >= 0) hire_tail_by_day[now_day] = index;
          ++hires;
          positions.push_back(spawn_hand(positions, current.board_size));
        }
      }
      for (int actor = 0; actor < required; ++actor) {
        Position& position = positions[actor];
        const UnitAction& action = frame.actor_actions[actor];
        if (is_move(action.op)) {
          move(position, action.op, current.board_size);
        } else if (requires_unlocked_tile(action.op) &&
                   tiles[tile_index(position, current.board_size)].kind == TileKind::Locked) {
          add_land_through(quadrant(position, current.board_size), step, actor);
        }
      }
    }
  }

  void add_resource_need(const Failure& failure) {
    const int deadline = failure.purchase_kind == NodeKind::BuySeed
                             ? failure.step - 1
                             : failure.pickup_step - 1;
    const int consumer = failure.step;
    const int cost = unit_cost(failure.purchase_kind, failure.item);
    add_need(failure.purchase_kind, failure.item, failure.quantity, consumer,
             deadline, input.current.step, failure.actor, -1, cost,
             failure.purchase_kind == NodeKind::BuySeed
                 ? "seed must exist before PLANT unit phase"
                 : "stock shed before the actor's planned PICKUP");
  }

  struct Simulation {
    std::array<int, kItems> shed{};
    std::array<int, kCrops> seeds{};
    std::vector<Position> positions;
    std::vector<Inventory> carried;
    std::vector<TileState> tiles;
    std::vector<PickupOpportunity> pickups;
    Failure failure;
    std::vector<Diagnostic> soft_diagnostics;
  };

  Simulation initial_simulation() const {
    Simulation simulation;
    simulation.shed = input.current.shed;
    simulation.seeds = input.current.seeds;
    simulation.positions = input.current.actor_positions;
    if (simulation.positions.empty()) {
      simulation.positions.push_back(default_spawn(input.current.board_size));
    }
    simulation.carried = input.current.carried;
    simulation.carried.resize(simulation.positions.size());
    for (Inventory& inventory : simulation.carried) normalize_inventory(inventory);
    simulation.tiles = input.current.tiles;
    if (simulation.tiles.size() != static_cast<std::size_t>(
                                       input.current.board_size * input.current.board_size)) {
      simulation.tiles.assign(input.current.board_size * input.current.board_size,
                              TileState{});
    }
    return simulation;
  }

  void apply_purchases_at(int step, Simulation& simulation) const {
    for (const Need& need : needs) {
      if (need.deadline != step || need.quantity <= 0) continue;
      if (need.kind == NodeKind::BuySeed && is_crop(need.item)) {
        simulation.seeds[as_int(need.item)] += need.quantity;
      } else if ((need.kind == NodeKind::BuyProduct ||
                  need.kind == NodeKind::BuyAnimal) && valid_item(need.item)) {
        // Virtual supply assumes the emitted capacity prerequisite is honored.
        simulation.shed[as_int(need.item)] += need.quantity;
      } else if (need.kind == NodeKind::BuyLand && need.quadrant >= 1 &&
                 need.quadrant <= 3) {
        for (int y = 0; y < input.current.board_size; ++y) {
          for (int x = 0; x < input.current.board_size; ++x) {
            if (quadrant({x, y}, input.current.board_size) == need.quadrant) {
              TileState& tile = simulation.tiles[y * input.current.board_size + x];
              if (tile.kind == TileKind::Locked) tile = TileState{};
            }
          }
        }
      }
    }
  }

  void end_day(int step, Simulation& simulation, bool collect_soft) const {
    const int capacity = input.current.shed_capacity;
    for (std::size_t actor = 0; actor < simulation.carried.size(); ++actor) {
      Inventory& inventory = simulation.carried[actor];
      normalize_inventory(inventory);
      for (const Item item : inventory.insertion_order) {
        const int index = as_int(item);
        const int room = std::max(0, capacity - shed_sum(simulation.shed));
        const int accepted = std::min(inventory.quantity[index], room);
        simulation.shed[index] += accepted;
        const int overflow = inventory.quantity[index] - accepted;
        if (collect_soft && overflow > 0) {
          simulation.soft_diagnostics.push_back(
              {DiagnosticCode::EndOfDayOverflow, step, static_cast<int>(actor),
               item, overflow,
               "official day boundary discards carried units that do not fit the shed"});
        }
      }
    }
    simulation.positions.resize(1);
    simulation.positions[0] = default_spawn(input.current.board_size);
    simulation.carried.assign(1, Inventory{});
    for (TileState& tile : simulation.tiles) {
      if (tile.kind == TileKind::Animal) {
        tile.consecutive_unfed = tile.fed_today ? 0 : tile.consecutive_unfed + 1;
        tile.fed_today = false;
        tile.fertilizer_available = true;
        if (tile.consecutive_unfed >= 2) {
          tile.kind = tile.item == Item::Goose ? TileKind::Coop : TileKind::Pasture;
          tile.item = Item::None;
          tile.yield_units = 0;
        }
      }
    }
  }

  PickupOpportunity* latest_pickup(Simulation& simulation, int step, int actor,
                                    Item item) const {
    const int day = step / input.current.turns_per_day;
    for (auto it = simulation.pickups.rbegin(); it != simulation.pickups.rend(); ++it) {
      if (it->actor == actor && it->item == item &&
          it->step / input.current.turns_per_day == day &&
          it->filled < it->requested) {
        return &*it;
      }
    }
    return nullptr;
  }

  bool consume_or_fail(Simulation& simulation, int step, int actor, Item item,
                       NodeKind purchase_kind, std::string message) const {
    if (actor < 0 || actor >= static_cast<int>(simulation.carried.size())) return false;
    if (inventory_take(simulation.carried[actor], item, 1) == 1) return true;
    PickupOpportunity* pickup = latest_pickup(simulation, step, actor, item);
    simulation.failure = {true, step, actor, item, purchase_kind, 1,
                          pickup == nullptr ? -1 : pickup->step,
                          std::move(message)};
    return false;
  }

  Simulation simulate(bool collect_soft) const {
    Simulation simulation = initial_simulation();
    if (frames.empty()) return simulation;
    const int last_step = frames.back().step;
    int day = input.current.step / input.current.turns_per_day;
    for (int step = input.current.step; step <= last_step; ++step) {
      const int now_day = step / input.current.turns_per_day;
      if (now_day != day) day = now_day;
      const auto frame_it = frame_at.find(step);
      if (frame_it != frame_at.end()) {
        const UnitFrame& frame = frames[frame_it->second];
        while (simulation.positions.size() < frame.actor_actions.size()) {
          simulation.positions.push_back(
              spawn_hand(simulation.positions, input.current.board_size));
          simulation.carried.push_back(Inventory{});
        }
        for (std::size_t actor = 0; actor < frame.actor_actions.size(); ++actor) {
          const UnitAction& action = frame.actor_actions[actor];
          Position& position = simulation.positions[actor];
          if (is_move(action.op)) {
            move(position, action.op, input.current.board_size);
            continue;
          }
          if (action.op == UnitOp::Pass || !inside(position, input.current.board_size)) continue;
          TileState& tile = simulation.tiles[tile_index(position, input.current.board_size)];
          const int quantity = std::max(0, action.quantity);
          if (action.op == UnitOp::Pickup) {
            int filled = 0;
            if (shed_adjacent(position, input.current.board_size) &&
                valid_item(action.item) && quantity > 0) {
              const int index = as_int(action.item);
              filled = std::min(quantity, simulation.shed[index]);
              simulation.shed[index] -= filled;
              inventory_add(simulation.carried[actor], action.item, filled);
            }
            simulation.pickups.push_back(
                {step, static_cast<int>(actor), action.item, quantity, filled});
          } else if (action.op == UnitOp::Drop) {
            if (!shed_adjacent(position, input.current.board_size)) continue;
            Inventory& inventory = simulation.carried[actor];
            normalize_inventory(inventory);
            const std::vector<Item> order = inventory.insertion_order;
            for (const Item item : order) {
              const int index = as_int(item);
              const int room = std::max(0, input.current.shed_capacity - shed_sum(simulation.shed));
              const int accepted = std::min(inventory.quantity[index], room);
              simulation.shed[index] += accepted;
              inventory_take(inventory, item, inventory.quantity[index]);
            }
          } else if (action.op == UnitOp::Plant) {
            if (is_crop(action.item) && tile.kind == TileKind::Empty) {
              const int index = as_int(action.item);
              if (simulation.seeds[index] <= 0) {
                simulation.failure = {true, step, static_cast<int>(actor), action.item,
                                      NodeKind::BuySeed, 1, -1,
                                      "PLANT lacks a seed before unit execution"};
                return simulation;
              }
              --simulation.seeds[index];
              tile.kind = TileKind::Plant;
              tile.item = action.item;
              tile.planted_day = step / input.current.turns_per_day;
            }
          } else if (action.op == UnitOp::Fertilize) {
            if (tile.kind == TileKind::Plant &&
                !consume_or_fail(simulation, step, static_cast<int>(actor),
                                 Item::Fertilizer, NodeKind::BuyProduct,
                                 "FERTILIZE lacks carried fertilizer")) return simulation;
          } else if (action.op == UnitOp::Feed) {
            if (tile.kind == TileKind::Animal && !tile.fed_today) {
              if (!consume_or_fail(simulation, step, static_cast<int>(actor),
                                   Item::Wheat, NodeKind::BuyProduct,
                                   "FEED lacks carried wheat")) return simulation;
              tile.fed_today = true;
            }
          } else if (action.op == UnitOp::Place) {
            if (is_animal(action.item) &&
                ((action.item == Item::Goose && tile.kind == TileKind::Coop) ||
                 (action.item != Item::Goose && tile.kind == TileKind::Pasture))) {
              if (!consume_or_fail(simulation, step, static_cast<int>(actor),
                                   action.item, NodeKind::BuyAnimal,
                                   "PLACE lacks the requested animal in this actor's carried inventory")) {
                return simulation;
              }
              tile.kind = TileKind::Animal;
              tile.item = action.item;
              tile.fed_today = false;
              tile.consecutive_unfed = 0;
            } else if (!is_animal(action.item) && valid_item(action.item) &&
                       shed_adjacent(position, input.current.board_size)) {
              Inventory& inventory = simulation.carried[actor];
              const int room = std::max(0, input.current.shed_capacity - shed_sum(simulation.shed));
              const int accepted = std::min({quantity,
                                             inventory.quantity[as_int(action.item)], room});
              inventory_take(inventory, action.item, accepted);
              simulation.shed[as_int(action.item)] += accepted;
            }
          } else if (action.op == UnitOp::Harvest && tile.yield_units > 0 &&
                     (tile.kind == TileKind::Plant || tile.kind == TileKind::Animal)) {
            const bool plant = tile.kind == TileKind::Plant;
            if (plant && (!is_crop(tile.item) ||
                          step / input.current.turns_per_day - tile.planted_day <
                              kCropFirstDay[as_int(tile.item)])) {
              continue;
            }
            Item product = plant ? tile.item : Item::None;
            if (!plant) {
              if (tile.item == Item::Goose) product = Item::Egg;
              if (tile.item == Item::Cow) product = Item::Milk;
              if (tile.item == Item::Sheep) product = Item::Wool;
            }
            inventory_add(simulation.carried[actor], product, tile.yield_units);
            if (plant && !ongoing_crop(tile.item)) {
              tile = TileState{};
            } else {
              tile.yield_units = 0;
            }
          } else if (action.op == UnitOp::CollectFertilizer &&
                     tile.kind == TileKind::Animal && tile.fertilizer_available) {
            tile.fertilizer_available = false;
            inventory_add(simulation.carried[actor], Item::Fertilizer, 1);
          } else if (action.op == UnitOp::Dig) {
            if (tile.kind != TileKind::Empty && tile.kind != TileKind::Animal) tile = TileState{};
          } else if (action.op == UnitOp::BuildCoop) {
            if (tile.kind == TileKind::Empty) tile.kind = TileKind::Coop;
          } else if (action.op == UnitOp::BuildPasture) {
            if (tile.kind == TileKind::Empty) tile.kind = TileKind::Pasture;
          }
        }
      }
      apply_purchases_at(step, simulation);
      if ((step + 1) % input.current.turns_per_day == 0) {
        end_day(step, simulation, collect_soft);
      }
    }
    return simulation;
  }

  void derive_resources() {
    int guard = 0;
    int total_actions = 0;
    for (const UnitFrame& frame : frames) total_actions += static_cast<int>(frame.actor_actions.size());
    int guard_limit = std::max(32, total_actions * 3 + 32);
    if (input.maximum_resource_derivation_iterations > 0) {
      guard_limit = std::min(
          guard_limit, input.maximum_resource_derivation_iterations);
    }
    while (guard++ < guard_limit) {
      Simulation simulation = simulate(false);
      if (!simulation.failure.present) return;
      const Failure& failure = simulation.failure;
      if (failure.purchase_kind != NodeKind::BuySeed && failure.pickup_step < 0) {
        add_diag(result, DiagnosticCode::MissingPickupPath, failure.step,
                 failure.actor, failure.item, failure.quantity,
                 failure.message + "; no earlier underfilled PICKUP can carry a purchase to it");
        return;
      }
      const std::size_t before = needs.size();
      int old_quantity = 0;
      if (failure.purchase_kind == NodeKind::BuySeed) {
        for (const Need& need : needs) {
          if (need.kind == failure.purchase_kind && need.item == failure.item &&
              need.deadline == failure.step - 1) old_quantity += need.quantity;
        }
      } else {
        for (const Need& need : needs) {
          if (need.kind == failure.purchase_kind && need.item == failure.item &&
              need.deadline == failure.pickup_step - 1) old_quantity += need.quantity;
        }
      }
      add_resource_need(failure);
      int new_quantity = 0;
      for (const Need& need : needs) {
        const int expected_deadline = failure.purchase_kind == NodeKind::BuySeed
                                          ? failure.step - 1
                                          : failure.pickup_step - 1;
        if (need.kind == failure.purchase_kind && need.item == failure.item &&
            need.deadline == expected_deadline) {
          new_quantity += need.quantity;
        }
      }
      if (needs.size() == before && new_quantity <= old_quantity) return;
      if (!result.feasible &&
          (failure.purchase_kind == NodeKind::BuySeed
               ? failure.step - 1
               : failure.pickup_step - 1) < input.current.step) return;
    }
    add_diag(result, DiagnosticCode::InvalidInput, input.current.step, -1,
             Item::None, 0, "resource derivation did not converge");
  }

  void chain_cumulative_resources() {
    using Key = std::pair<int, int>;
    std::map<Key, std::vector<int>> groups;
    for (std::size_t index = 0; index < needs.size(); ++index) {
      const Need& need = needs[index];
      if (need.kind == NodeKind::BuySeed || need.kind == NodeKind::BuyProduct ||
          need.kind == NodeKind::BuyAnimal) {
        groups[{static_cast<int>(need.kind), as_int(need.item)}].push_back(
            static_cast<int>(index));
      }
    }
    for (auto& [key, indices] : groups) {
      (void)key;
      std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return std::tie(needs[lhs].deadline, needs[lhs].consumer_step) <
               std::tie(needs[rhs].deadline, needs[rhs].consumer_step);
      });
      for (std::size_t pos = 1; pos < indices.size(); ++pos) {
        const int before = indices[pos - 1];
        const int after = indices[pos];
        if (needs[after].predecessor < 0 && needs[before].successor < 0) {
          needs[after].predecessor = before;
          needs[before].successor = after;
        }
      }
    }
  }

  void schedule() {
    if (needs.empty()) return;
    const int slots_per_step = input.current.max_market_orders;
    int max_deadline = input.current.step;
    for (const Need& need : needs) max_deadline = std::max(max_deadline, need.deadline);
    std::set<int> available;
    for (int step = input.current.step; step <= max_deadline; ++step) {
      for (int slot = 0; slot < slots_per_step; ++slot) {
        available.insert(step * slots_per_step + slot);
      }
    }
    std::vector<int> order(needs.size());
    for (std::size_t index = 0; index < needs.size(); ++index) order[index] = static_cast<int>(index);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      if (needs[lhs].deadline != needs[rhs].deadline)
        return needs[lhs].deadline > needs[rhs].deadline;
      // A successor must be placed first so its predecessor can be bounded.
      if (needs[lhs].predecessor == rhs) return true;
      if (needs[rhs].predecessor == lhs) return false;
      return lhs > rhs;
    });
    std::vector<int> assigned_key(needs.size(), -1);
    bool progress = true;
    while (progress) {
      progress = false;
      for (const int index : order) {
        Need& need = needs[index];
        // -1 is the only pending state.  -2 is a terminal, already-diagnosed
        // scheduling failure; revisiting it used to set progress forever when
        // no slot (or no slot after earliest) existed.
        if (assigned_key[index] != -1) continue;
        if (need.successor >= 0 && assigned_key[need.successor] < 0) continue;
        int upper = need.deadline * slots_per_step + slots_per_step - 1;
        if (need.successor >= 0) upper = std::min(upper, assigned_key[need.successor] - 1);
        auto it = available.upper_bound(upper);
        if (it == available.begin()) {
          add_diag(result, DiagnosticCode::MarketSlotInfeasible,
                   need.consumer_step, need.actor, need.item, need.quantity,
                   "no market order slot exists before this obligation's deadline");
          assigned_key[index] = -2;
          progress = true;
          continue;
        }
        --it;
        const int key = *it;
        if (key / slots_per_step < need.earliest) {
          add_diag(result, DiagnosticCode::MarketSlotInfeasible,
                   need.consumer_step, need.actor, need.item, need.quantity,
                   "10-slot schedule cannot place this order between earliest and deadline");
          assigned_key[index] = -2;
          progress = true;
          continue;
        }
        assigned_key[index] = key;
        need.execution = key / slots_per_step;
        need.slot = key % slots_per_step;
        available.erase(it);
        progress = true;
      }
    }
    for (std::size_t index = 0; index < needs.size(); ++index) {
      if (assigned_key[index] == -1) {
        add_diag(result, DiagnosticCode::MarketSlotInfeasible,
                 needs[index].consumer_step, needs[index].actor,
                 needs[index].item, needs[index].quantity,
                 "precedence cycle prevented market scheduling");
      }
    }
  }

  int push_node(ObligationNode node) {
    node.id = static_cast<int>(result.nodes.size());
    result.nodes.push_back(std::move(node));
    return result.nodes.back().id;
  }

  void materialize() {
    std::vector<int> need_node(needs.size(), -1);
    std::map<int, std::vector<int>> by_execution;
    std::map<std::pair<int, int>, int> cumulative_quantity;
    std::vector<int> cumulative_for_need(needs.size(), 0);
    std::vector<int> deadline_order(needs.size());
    for (std::size_t index = 0; index < needs.size(); ++index) {
      deadline_order[index] = static_cast<int>(index);
    }
    std::sort(deadline_order.begin(), deadline_order.end(), [&](int lhs, int rhs) {
      return std::tie(needs[lhs].deadline, lhs) < std::tie(needs[rhs].deadline, rhs);
    });
    for (const int index : deadline_order) {
      const auto key = std::make_pair(static_cast<int>(needs[index].kind),
                                      as_int(needs[index].item));
      cumulative_quantity[key] += needs[index].quantity;
      cumulative_for_need[index] = cumulative_quantity[key];
    }
    for (std::size_t index = 0; index < needs.size(); ++index) {
      const Need& need = needs[index];
      ObligationNode node;
      node.kind = need.kind;
      node.item = need.item;
      node.quantity = need.quantity;
      node.cumulative_quantity = cumulative_for_need[index];
      node.actor = need.actor;
      node.consumer_step = need.consumer_step;
      node.deadline_step = need.deadline;
      node.earliest_step = need.earliest;
      node.execution_step = need.execution;
      node.order_slot = need.slot;
      node.quadrant = need.quadrant;
      node.unit_cost_quote = need.unit_cost;
      node.cash_quote = need.unit_cost * need.quantity;
      node.free_capacity_required =
          (need.kind == NodeKind::BuyProduct || need.kind == NodeKind::BuyAnimal)
              ? need.quantity
              : 0;
      node.reason = need.reason;
      need_node[index] = push_node(std::move(node));
      result.purchase_units += need.quantity;
      ++result.market_order_nodes;
      result.quoted_cash += need.unit_cost * need.quantity;
      if (need.execution >= 0) by_execution[need.execution].push_back(static_cast<int>(index));
    }
    for (std::size_t index = 0; index < needs.size(); ++index) {
      if (needs[index].predecessor >= 0) {
        result.edges.push_back({need_node[needs[index].predecessor], need_node[index],
                                "ordered/cumulative acquisition precedence"});
      }
    }

    int cumulative_cash = 0;
    for (auto& [step, indices] : by_execution) {
      std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return needs[lhs].slot < needs[rhs].slot;
      });
      int cash = 0;
      int capacity = 0;
      for (const int index : indices) {
        cash += needs[index].unit_cost * needs[index].quantity;
        if (needs[index].kind == NodeKind::BuyProduct ||
            needs[index].kind == NodeKind::BuyAnimal) {
          capacity += needs[index].quantity;
        }
      }
      cumulative_cash += cash;
      ObligationNode slot_node;
      slot_node.kind = NodeKind::SlotBudget;
      slot_node.quantity = static_cast<int>(indices.size());
      slot_node.deadline_step = step;
      slot_node.execution_step = step;
      slot_node.reason = "reserve distinct market-order positions; HIRE/LAND are one unit per slot";
      const int slot_id = push_node(std::move(slot_node));
      result.peak_orders_in_step = std::max(result.peak_orders_in_step,
                                            static_cast<int>(indices.size()));

      ObligationNode cash_node;
      cash_node.kind = NodeKind::CashReserve;
      cash_node.quantity = cash;
      cash_node.cash_quote = cash;
      cash_node.cumulative_cash_quote = cumulative_cash;
      cash_node.cash_shortfall_quote = std::max(
          0, static_cast<int>(std::ceil(cumulative_cash - input.current.money)));
      cash_node.deadline_step = step;
      cash_node.execution_step = step;
      cash_node.reason = "cash needed at this execution step (products use current-price quote)";
      const int cash_id = push_node(std::move(cash_node));

      int capacity_id = -1;
      if (capacity > 0) {
        ObligationNode capacity_node;
        capacity_node.kind = NodeKind::CapacityRelease;
        capacity_node.quantity = capacity;
        capacity_node.free_capacity_required = capacity;
        capacity_node.deadline_step = step;
        capacity_node.execution_step = step;
        capacity_node.reason = "prove this much shed room immediately before capacity-limited buys";
        capacity_id = push_node(std::move(capacity_node));
      }
      for (const int index : indices) {
        result.edges.push_back({slot_id, need_node[index], "market slot reservation"});
        result.edges.push_back({cash_id, need_node[index], "cash reservation"});
        if (capacity_id >= 0 &&
            (needs[index].kind == NodeKind::BuyProduct ||
             needs[index].kind == NodeKind::BuyAnimal)) {
          result.edges.push_back({capacity_id, need_node[index], "shed capacity prerequisite"});
        }
      }
    }

    // Audit the non-market consumers and connect the acquisitions that can
    // causally feed them. These nodes are explanatory; scheduling used only
    // acquisition nodes above.
    std::vector<Position> positions = input.current.actor_positions;
    if (positions.empty()) positions.push_back(default_spawn(input.current.board_size));
    std::vector<TileState> audit_tiles = input.current.tiles;
    if (audit_tiles.size() != static_cast<std::size_t>(
                                  input.current.board_size * input.current.board_size)) {
      audit_tiles.resize(input.current.board_size * input.current.board_size);
      for (int y = 0; y < input.current.board_size; ++y) {
        for (int x = 0; x < input.current.board_size; ++x) {
          const int q = quadrant({x, y}, input.current.board_size);
          if ((input.current.unlocked_mask & (1u << q)) == 0) {
            audit_tiles[y * input.current.board_size + x].kind = TileKind::Locked;
          }
        }
      }
    }
    std::map<std::pair<int, int>, int> last_pickup_node;
    int previous_day = input.current.step / input.current.turns_per_day;
    for (const UnitFrame& frame : frames) {
      const int day = frame.step / input.current.turns_per_day;
      if (day != previous_day) {
        positions.resize(1);
        positions[0] = default_spawn(input.current.board_size);
        last_pickup_node.clear();
        ObligationNode boundary;
        boundary.kind = NodeKind::DayBoundary;
        boundary.consumer_step = day * input.current.turns_per_day - 1;
        boundary.deadline_step = boundary.consumer_step;
        boundary.reason = "carried inventory returns to shed; hands and hire counter reset";
        push_node(std::move(boundary));
        previous_day = day;
      }
      while (positions.size() < frame.actor_actions.size()) {
        positions.push_back(spawn_hand(positions, input.current.board_size));
      }
      for (std::size_t actor = 0; actor < frame.actor_actions.size(); ++actor) {
        const UnitAction& action = frame.actor_actions[actor];
        if (is_move(action.op)) move(positions[actor], action.op, input.current.board_size);
        const int access_quadrant = quadrant(positions[actor], input.current.board_size);
        if (requires_unlocked_tile(action.op) &&
            audit_tiles[tile_index(positions[actor], input.current.board_size)].kind ==
                TileKind::Locked) {
          ObligationNode access;
          access.kind = NodeKind::LandAccess;
          access.actor = static_cast<int>(actor);
          access.consumer_step = frame.step;
          access.deadline_step = frame.step - 1;
          access.quadrant = access_quadrant;
          access.reason = "planned productive unit action is after the engine's LOCKED guard";
          const int access_id = push_node(std::move(access));
          for (std::size_t index = 0; index < needs.size(); ++index) {
            if (needs[index].kind == NodeKind::BuyLand &&
                needs[index].quadrant <= access_quadrant) {
              result.edges.push_back({need_node[index], access_id,
                                      "ordered land must unlock before access"});
            }
          }
          for (int q = 1; q <= access_quadrant; ++q) {
            for (int y = 0; y < input.current.board_size; ++y) {
              for (int x = 0; x < input.current.board_size; ++x) {
                if (quadrant({x, y}, input.current.board_size) == q) {
                  TileState& tile = audit_tiles[y * input.current.board_size + x];
                  if (tile.kind == TileKind::Locked) tile = TileState{};
                }
              }
            }
          }
        }
        if (action.op == UnitOp::Pickup && valid_item(action.item)) {
          ObligationNode pickup;
          pickup.kind = NodeKind::Pickup;
          pickup.item = action.item;
          pickup.quantity = std::max(0, action.quantity);
          pickup.actor = static_cast<int>(actor);
          pickup.consumer_step = frame.step;
          pickup.deadline_step = frame.step - 1;
          pickup.reason = "unit-phase transfer from shed to this actor";
          const int pickup_id = push_node(std::move(pickup));
          last_pickup_node[{static_cast<int>(actor), as_int(action.item)}] = pickup_id;
          for (std::size_t index = 0; index < needs.size(); ++index) {
            if ((needs[index].kind == NodeKind::BuyProduct ||
                 needs[index].kind == NodeKind::BuyAnimal) &&
                needs[index].item == action.item &&
                needs[index].deadline <= frame.step - 1) {
              result.edges.push_back({need_node[index], pickup_id,
                                      "purchase must precede unit-phase PICKUP"});
            }
          }
        }
        Item consumed = Item::None;
        if (action.op == UnitOp::Plant) consumed = action.item;
        if (action.op == UnitOp::Feed) consumed = Item::Wheat;
        if (action.op == UnitOp::Fertilize) consumed = Item::Fertilizer;
        if (action.op == UnitOp::Place && is_animal(action.item)) consumed = action.item;
        if (consumed != Item::None) {
          ObligationNode consume;
          consume.kind = NodeKind::Consume;
          consume.item = consumed;
          consume.quantity = 1;
          consume.actor = static_cast<int>(actor);
          consume.consumer_step = frame.step;
          consume.deadline_step = frame.step - 1;
          consume.reason = "planned production resource consumption";
          const int consume_id = push_node(std::move(consume));
          if (action.op == UnitOp::Plant) {
            for (std::size_t index = 0; index < needs.size(); ++index) {
              if (needs[index].kind == NodeKind::BuySeed && needs[index].item == consumed &&
                  needs[index].deadline <= frame.step - 1) {
                result.edges.push_back({need_node[index], consume_id,
                                        "seed acquisition before PLANT"});
              }
            }
          } else {
            const auto pickup = last_pickup_node.find(
                {static_cast<int>(actor), as_int(consumed)});
            if (pickup != last_pickup_node.end()) {
              result.edges.push_back({pickup->second, consume_id,
                                      "same actor must carry resource after PICKUP"});
            }
          }
        }
      }
    }

    Simulation final_simulation = simulate(true);
    for (Diagnostic& diagnostic : final_simulation.soft_diagnostics) {
      ObligationNode capacity_node;
      capacity_node.kind = NodeKind::CapacityRelease;
      capacity_node.item = diagnostic.item;
      capacity_node.quantity = diagnostic.quantity;
      capacity_node.actor = diagnostic.actor;
      capacity_node.consumer_step = diagnostic.step;
      capacity_node.deadline_step = diagnostic.step;
      capacity_node.execution_step = diagnostic.step;
      capacity_node.free_capacity_required = diagnostic.quantity;
      capacity_node.reason = "free shed room before end-of-day automatic carried-inventory return";
      push_node(std::move(capacity_node));
      result.diagnostics.push_back(std::move(diagnostic));
    }
  }

  CompileResult run() {
    validate();
    if (!result.feasible) return std::move(result);
    derive_actor_and_land();
    derive_resources();
    chain_cumulative_resources();
    schedule();
    materialize();
    return std::move(result);
  }
};

std::string escape_json(const std::string& value) {
  std::ostringstream out;
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (character < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(character) << std::dec;
        } else {
          out << character;
        }
    }
  }
  return out.str();
}

}  // namespace

CompileResult compile(const CompilerInput& input) { return Builder(input).run(); }

namespace {

struct SoftWorkingState {
  std::array<int, kItems> shed{};
  std::array<int, kCrops> seeds{};
  std::vector<Position> positions;
  std::vector<Inventory> carried;
  std::vector<TileState> tiles;
};

SoftWorkingState soft_working_state(const CurrentState& current) {
  SoftWorkingState state;
  state.shed = current.shed;
  state.seeds = current.seeds;
  state.positions = current.actor_positions;
  if (state.positions.empty()) state.positions.push_back(default_spawn(current.board_size));
  state.carried = current.carried;
  state.carried.resize(state.positions.size());
  for (Inventory& inventory : state.carried) normalize_inventory(inventory);
  state.tiles = current.tiles;
  if (state.tiles.size() !=
      static_cast<std::size_t>(current.board_size * current.board_size)) {
    state.tiles.assign(current.board_size * current.board_size, TileState{});
    for (int y = 0; y < current.board_size; ++y) {
      for (int x = 0; x < current.board_size; ++x) {
        const int q = quadrant({x, y}, current.board_size);
        if ((current.unlocked_mask & (1u << q)) == 0) {
          state.tiles[y * current.board_size + x].kind = TileKind::Locked;
        }
      }
    }
  }
  return state;
}

struct SoftCandidate {
  UnitReplacement replacement;
  int downstream = 0;
};

bool is_resource_soft_op(UnitOp op) {
  return op == UnitOp::Plant || op == UnitOp::Feed ||
         op == UnitOp::Fertilize || op == UnitOp::Place;
}

Item consumed_item(const UnitAction& action) {
  if (action.op == UnitOp::Plant || action.op == UnitOp::Place) return action.item;
  if (action.op == UnitOp::Feed) return Item::Wheat;
  if (action.op == UnitOp::Fertilize) return Item::Fertilizer;
  return Item::None;
}

bool resource_action_would_consume(const UnitAction& action, const TileState& tile) {
  if (tile.kind == TileKind::Locked) return false;
  if (action.op == UnitOp::Plant) {
    return is_crop(action.item) && tile.kind == TileKind::Empty;
  }
  if (action.op == UnitOp::Feed) {
    return tile.kind == TileKind::Animal && !tile.fed_today;
  }
  if (action.op == UnitOp::Fertilize) return tile.kind == TileKind::Plant;
  if (action.op == UnitOp::Place && is_animal(action.item)) {
    return (action.item == Item::Goose && tile.kind == TileKind::Coop) ||
           (action.item != Item::Goose && tile.kind == TileKind::Pasture);
  }
  return false;
}

UnitAction soft_replacement_for(const CurrentState& current,
                                const SoftWorkingState& state, int actor,
                                UnitOp original_op, Item item) {
  if (actor >= 0 && actor < static_cast<int>(state.positions.size()) &&
      valid_item(item) && original_op != UnitOp::Plant &&
      shed_adjacent(state.positions[actor], current.board_size) &&
      state.shed[as_int(item)] > 0) {
    return {UnitOp::Pickup, item, 1};
  }
  return {UnitOp::Pass, Item::None, 1};
}

void execute_soft_current_action(const CurrentState& current, SoftWorkingState& state,
                                 int actor, const UnitAction& action) {
  if (actor < 0 || actor >= static_cast<int>(state.positions.size())) return;
  Position& position = state.positions[actor];
  if (is_move(action.op)) {
    move(position, action.op, current.board_size);
    return;
  }
  if (action.op == UnitOp::Pass || !inside(position, current.board_size)) return;
  Inventory& carried = state.carried[actor];
  TileState& tile = state.tiles[tile_index(position, current.board_size)];
  const int quantity = std::max(0, action.quantity);
  if (action.op == UnitOp::Pickup) {
    if (!shed_adjacent(position, current.board_size) || !valid_item(action.item)) return;
    const int index = as_int(action.item);
    const int taken = std::min(quantity, state.shed[index]);
    state.shed[index] -= taken;
    inventory_add(carried, action.item, taken);
    return;
  }
  if (action.op == UnitOp::Drop) {
    if (!shed_adjacent(position, current.board_size)) return;
    normalize_inventory(carried);
    const std::vector<Item> order = carried.insertion_order;
    for (const Item item : order) {
      const int index = as_int(item);
      const int room = std::max(0, current.shed_capacity - shed_sum(state.shed));
      const int accepted = std::min(carried.quantity[index], room);
      state.shed[index] += accepted;
      inventory_take(carried, item, carried.quantity[index]);
    }
    return;
  }
  if (action.op == UnitOp::Place) {
    if (is_animal(action.item)) {
      if (resource_action_would_consume(action, tile) &&
          inventory_take(carried, action.item, 1) == 1) {
        tile.kind = TileKind::Animal;
        tile.item = action.item;
        tile.fed_today = false;
        tile.consecutive_unfed = 0;
      }
    } else if (valid_item(action.item) && shed_adjacent(position, current.board_size)) {
      const int index = as_int(action.item);
      const int room = std::max(0, current.shed_capacity - shed_sum(state.shed));
      const int accepted = std::min({quantity, carried.quantity[index], room});
      inventory_take(carried, action.item, accepted);
      state.shed[index] += accepted;
    }
    return;
  }
  if (tile.kind == TileKind::Locked) return;
  if (action.op == UnitOp::Plant && is_crop(action.item) &&
      tile.kind == TileKind::Empty && state.seeds[as_int(action.item)] > 0) {
    --state.seeds[as_int(action.item)];
    tile.kind = TileKind::Plant;
    tile.item = action.item;
    tile.planted_day = current.step / current.turns_per_day;
  } else if (action.op == UnitOp::Feed && tile.kind == TileKind::Animal &&
             !tile.fed_today && inventory_take(carried, Item::Wheat, 1) == 1) {
    tile.fed_today = true;
  } else if (action.op == UnitOp::Fertilize && tile.kind == TileKind::Plant) {
    inventory_take(carried, Item::Fertilizer, 1);
  } else if (action.op == UnitOp::Harvest && tile.yield_units > 0 &&
             (tile.kind == TileKind::Plant || tile.kind == TileKind::Animal)) {
    const bool plant = tile.kind == TileKind::Plant;
    if (plant && (!is_crop(tile.item) ||
                  current.step / current.turns_per_day - tile.planted_day <
                      kCropFirstDay[as_int(tile.item)])) return;
    Item product = plant ? tile.item : Item::None;
    if (!plant) {
      if (tile.item == Item::Goose) product = Item::Egg;
      if (tile.item == Item::Cow) product = Item::Milk;
      if (tile.item == Item::Sheep) product = Item::Wool;
    }
    inventory_add(carried, product, tile.yield_units);
    if (plant && !ongoing_crop(tile.item)) tile = TileState{};
    else tile.yield_units = 0;
  } else if (action.op == UnitOp::BuildCoop && tile.kind == TileKind::Empty) {
    tile.kind = TileKind::Coop;
  } else if (action.op == UnitOp::BuildPasture && tile.kind == TileKind::Empty) {
    tile.kind = TileKind::Pasture;
  } else if (action.op == UnitOp::Dig && tile.kind != TileKind::Empty &&
             tile.kind != TileKind::Animal) {
    tile = TileState{};
  }
}

struct LocatedAction {
  int step = -1;
  int actor = -1;
  Position position{-1, -1};
  UnitAction action;
};

std::vector<LocatedAction> locate_actions(const CompilerInput& input) {
  std::vector<UnitFrame> frames = input.future_units;
  std::sort(frames.begin(), frames.end(),
            [](const UnitFrame& lhs, const UnitFrame& rhs) { return lhs.step < rhs.step; });
  std::vector<Position> positions = input.current.actor_positions;
  if (positions.empty()) positions.push_back(default_spawn(input.current.board_size));
  int day = input.current.step / input.current.turns_per_day;
  std::vector<LocatedAction> located;
  for (const UnitFrame& frame : frames) {
    const int now_day = frame.step / input.current.turns_per_day;
    if (now_day != day) {
      positions.resize(1);
      positions[0] = default_spawn(input.current.board_size);
      day = now_day;
    }
    while (positions.size() < frame.actor_actions.size()) {
      positions.push_back(spawn_hand(positions, input.current.board_size));
    }
    for (std::size_t actor = 0; actor < frame.actor_actions.size(); ++actor) {
      located.push_back({frame.step, static_cast<int>(actor), positions[actor],
                         frame.actor_actions[actor]});
      if (is_move(frame.actor_actions[actor].op)) {
        move(positions[actor], frame.actor_actions[actor].op,
             input.current.board_size);
      }
    }
  }
  return located;
}

bool same_position(Position lhs, Position rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool downstream_of(UnitOp missed, UnitOp future) {
  if (missed == UnitOp::Plant) {
    return future == UnitOp::Water || future == UnitOp::Fertilize ||
           future == UnitOp::Harvest || future == UnitOp::Dig;
  }
  if (missed == UnitOp::Feed) {
    return future == UnitOp::Care || future == UnitOp::Harvest ||
           future == UnitOp::CollectFertilizer;
  }
  if (missed == UnitOp::Fertilize) {
    return future == UnitOp::Water || future == UnitOp::Harvest;
  }
  if (missed == UnitOp::Place) {
    return future == UnitOp::Feed || future == UnitOp::Care ||
           future == UnitOp::Harvest || future == UnitOp::CollectFertilizer;
  }
  return false;
}

bool lifecycle_reset(UnitOp missed, UnitOp future) {
  if (missed == UnitOp::Plant || missed == UnitOp::Fertilize) {
    return future == UnitOp::Plant || future == UnitOp::Dig;
  }
  if (missed == UnitOp::Place) {
    return future == UnitOp::Place || future == UnitOp::Dig ||
           future == UnitOp::BuildCoop || future == UnitOp::BuildPasture;
  }
  return missed == UnitOp::Feed && future == UnitOp::Feed;
}

}  // namespace

SoftCompileResult compile_soft_current(const CompilerInput& input) {
  CompilerInput repaired = input;
  auto current_frame = std::find_if(
      repaired.future_units.begin(), repaired.future_units.end(),
      [&](const UnitFrame& frame) { return frame.step == input.current.step; });
  if (current_frame == repaired.future_units.end()) {
    return {compile(input), {}};
  }

  SoftWorkingState state = soft_working_state(input.current);
  std::vector<SoftCandidate> candidates;
  const std::size_t actors = std::min(current_frame->actor_actions.size(),
                                     state.positions.size());
  for (std::size_t actor = 0; actor < actors; ++actor) {
    UnitAction& action = current_frame->actor_actions[actor];
    if (!is_resource_soft_op(action.op) || is_move(action.op) ||
        !inside(state.positions[actor], input.current.board_size)) {
      execute_soft_current_action(input.current, state, static_cast<int>(actor), action);
      continue;
    }
    TileState& tile = state.tiles[tile_index(state.positions[actor],
                                            input.current.board_size)];
    const Item item = consumed_item(action);
    bool missing = false;
    if (resource_action_would_consume(action, tile)) {
      if (action.op == UnitOp::Plant) {
        missing = !is_crop(item) || state.seeds[as_int(item)] <= 0;
      } else {
        missing = !valid_item(item) ||
                  state.carried[actor].quantity[as_int(item)] <= 0;
      }
    }
    if (!missing) {
      execute_soft_current_action(input.current, state, static_cast<int>(actor), action);
      continue;
    }

    UnitAction replacement = soft_replacement_for(
        input.current, state, static_cast<int>(actor), action.op, item);
    SoftCandidate candidate;
    candidate.replacement = {input.current.step, static_cast<int>(actor), action,
                             replacement, state.positions[actor], -1};
    candidates.push_back(candidate);
    action = replacement;
    execute_soft_current_action(input.current, state, static_cast<int>(actor), action);
  }

  SoftCompileResult output;
  output.dag = compile(repaired);
  if (candidates.empty()) return output;
  const std::vector<LocatedAction> located = locate_actions(input);
  auto append_node = [&](ObligationNode node) {
    node.id = static_cast<int>(output.dag.nodes.size());
    output.dag.nodes.push_back(std::move(node));
    return output.dag.nodes.back().id;
  };

  for (SoftCandidate& candidate : candidates) {
    std::vector<LocatedAction> dependents;
    const UnitReplacement& patch = candidate.replacement;
    for (const LocatedAction& action : located) {
      if (action.step <= input.current.step ||
          !same_position(action.position, patch.position)) continue;
      if (patch.original.op == UnitOp::Feed &&
          action.step / input.current.turns_per_day !=
              input.current.step / input.current.turns_per_day) break;
      if (lifecycle_reset(patch.original.op, action.action.op)) break;
      if (downstream_of(patch.original.op, action.action.op)) {
        dependents.push_back(action);
      }
    }
    candidate.downstream = static_cast<int>(dependents.size());

    ObligationNode miss;
    miss.kind = NodeKind::SoftMiss;
    miss.item = consumed_item(patch.original);
    miss.quantity = 1;
    miss.actor = patch.actor;
    miss.consumer_step = patch.step;
    miss.deadline_step = patch.step - 1;
    miss.position = patch.position;
    miss.original_unit_op = patch.original.op;
    miss.suggested_unit_op = patch.replacement.op;
    miss.downstream_dependents = candidate.downstream;
    miss.downstream_loss_proxy = 1 + candidate.downstream;
    miss.reason = "current unit lacks its carried/global resource; same-step market executes too late";
    const int miss_id = append_node(std::move(miss));
    candidate.replacement.soft_miss_node = miss_id;
    ++output.dag.soft_misses;

    for (const LocatedAction& dependent : dependents) {
      ObligationNode node;
      node.kind = NodeKind::DownstreamAction;
      node.item = dependent.action.item;
      node.quantity = 1;
      node.actor = dependent.actor;
      node.consumer_step = dependent.step;
      node.position = dependent.position;
      node.original_unit_op = dependent.action.op;
      node.suggested_unit_op = dependent.action.op;
      node.reason = "planned lifecycle action may no-op after the current soft miss";
      const int dependent_id = append_node(std::move(node));
      output.dag.edges.push_back(
          {miss_id, dependent_id, "soft miss invalidates a lifecycle precondition"});
    }
    output.current_unit_replacements.push_back(candidate.replacement);
  }
  return output;
}

const char* item_name(Item item) {
  static constexpr std::array<const char*, kItems> names{
      "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
      "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"};
  return valid_item(item) ? names[as_int(item)] : "NONE";
}

const char* node_kind_name(NodeKind kind) {
  switch (kind) {
    case NodeKind::BuySeed: return "BUY_SEED";
    case NodeKind::BuyProduct: return "BUY_PRODUCT";
    case NodeKind::BuyAnimal: return "BUY_ANIMAL";
    case NodeKind::Hire: return "HIRE";
    case NodeKind::BuyLand: return "BUY_LAND";
    case NodeKind::Pickup: return "PICKUP";
    case NodeKind::Consume: return "CONSUME";
    case NodeKind::LandAccess: return "LAND_ACCESS";
    case NodeKind::CashReserve: return "CASH_RESERVE";
    case NodeKind::CapacityRelease: return "CAPACITY_RELEASE";
    case NodeKind::SlotBudget: return "SLOT_BUDGET";
    case NodeKind::DayBoundary: return "DAY_BOUNDARY";
    case NodeKind::SoftMiss: return "SOFT_MISS";
    case NodeKind::DownstreamAction: return "DOWNSTREAM_ACTION";
  }
  return "UNKNOWN";
}

const char* unit_op_name(UnitOp op) {
  switch (op) {
    case UnitOp::Pass: return "PASS";
    case UnitOp::North: return "NORTH";
    case UnitOp::South: return "SOUTH";
    case UnitOp::East: return "EAST";
    case UnitOp::West: return "WEST";
    case UnitOp::Drop: return "DROP";
    case UnitOp::Pickup: return "PICKUP";
    case UnitOp::Place: return "PLACE";
    case UnitOp::Plant: return "PLANT";
    case UnitOp::Water: return "WATER";
    case UnitOp::Harvest: return "HARVEST";
    case UnitOp::Fertilize: return "FERTILIZE";
    case UnitOp::Dig: return "DIG";
    case UnitOp::BuildCoop: return "BUILD_COOP";
    case UnitOp::BuildPasture: return "BUILD_PASTURE";
    case UnitOp::Feed: return "FEED";
    case UnitOp::CollectFertilizer: return "COLLECT_FERTILIZER";
    case UnitOp::Care: return "CARE";
  }
  return "UNKNOWN";
}

const char* diagnostic_code_name(DiagnosticCode code) {
  switch (code) {
    case DiagnosticCode::InvalidInput: return "INVALID_INPUT";
    case DiagnosticCode::MissedAcquisitionDeadline: return "MISSED_ACQUISITION_DEADLINE";
    case DiagnosticCode::MissingPickupPath: return "MISSING_PICKUP_PATH";
    case DiagnosticCode::ActorUnavailableAtDayStart: return "ACTOR_UNAVAILABLE_AT_DAY_START";
    case DiagnosticCode::MarketSlotInfeasible: return "MARKET_SLOT_INFEASIBLE";
    case DiagnosticCode::InvalidLandPrefix: return "INVALID_LAND_PREFIX";
    case DiagnosticCode::UnitWouldNotExecute: return "UNIT_WOULD_NOT_EXECUTE";
    case DiagnosticCode::EndOfDayOverflow: return "END_OF_DAY_OVERFLOW";
  }
  return "UNKNOWN";
}

std::string CompileResult::to_json() const {
  std::ostringstream out;
  out << "{\"feasible\":" << (feasible ? "true" : "false")
      << ",\"summary\":{\"purchase_units\":" << purchase_units
      << ",\"market_order_nodes\":" << market_order_nodes
      << ",\"quoted_cash\":" << quoted_cash
      << ",\"peak_orders_in_step\":" << peak_orders_in_step
      << ",\"soft_misses\":" << soft_misses << "},\"nodes\":[";
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (index != 0) out << ',';
    const ObligationNode& node = nodes[index];
    out << "{\"id\":" << node.id
        << ",\"kind\":\"" << node_kind_name(node.kind) << "\""
        << ",\"item\":\"" << item_name(node.item) << "\""
        << ",\"quantity\":" << node.quantity
        << ",\"cumulative_quantity\":" << node.cumulative_quantity
        << ",\"actor\":" << node.actor
        << ",\"consumer_step\":" << node.consumer_step
        << ",\"deadline_step\":" << node.deadline_step
        << ",\"earliest_step\":" << node.earliest_step
        << ",\"execution_step\":" << node.execution_step
        << ",\"order_slot\":" << node.order_slot
        << ",\"quadrant\":" << node.quadrant
        << ",\"unit_cost_quote\":" << node.unit_cost_quote
        << ",\"cash_quote\":" << node.cash_quote
        << ",\"cumulative_cash_quote\":" << node.cumulative_cash_quote
        << ",\"cash_shortfall_quote\":" << node.cash_shortfall_quote
        << ",\"free_capacity_required\":" << node.free_capacity_required
        << ",\"position\":[" << node.position.x << ',' << node.position.y << ']'
        << ",\"original_unit_op\":\"" << unit_op_name(node.original_unit_op) << "\""
        << ",\"suggested_unit_op\":\"" << unit_op_name(node.suggested_unit_op) << "\""
        << ",\"downstream_dependents\":" << node.downstream_dependents
        << ",\"downstream_loss_proxy\":" << node.downstream_loss_proxy
        << ",\"reason\":\"" << escape_json(node.reason) << "\"}";
  }
  out << "],\"edges\":[";
  for (std::size_t index = 0; index < edges.size(); ++index) {
    if (index != 0) out << ',';
    out << "{\"before\":" << edges[index].before
        << ",\"after\":" << edges[index].after
        << ",\"reason\":\"" << escape_json(edges[index].reason) << "\"}";
  }
  out << "],\"diagnostics\":[";
  for (std::size_t index = 0; index < diagnostics.size(); ++index) {
    if (index != 0) out << ',';
    const Diagnostic& diagnostic = diagnostics[index];
    out << "{\"code\":\"" << diagnostic_code_name(diagnostic.code) << "\""
        << ",\"step\":" << diagnostic.step
        << ",\"actor\":" << diagnostic.actor
        << ",\"item\":\"" << item_name(diagnostic.item) << "\""
        << ",\"quantity\":" << diagnostic.quantity
        << ",\"message\":\"" << escape_json(diagnostic.message) << "\"}";
  }
  out << "]}";
  return out.str();
}

std::uint64_t CompileResult::fingerprint() const {
  const std::string json = to_json();
  std::uint64_t hash = 1469598103934665603ull;
  for (const unsigned char value : json) {
    hash ^= value;
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace production_obligation
