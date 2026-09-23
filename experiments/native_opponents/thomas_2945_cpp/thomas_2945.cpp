#include "thomas_2945.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

// Thomas is the exact Python prefix of MetaV4.  Expose that experiment's
// implementation only inside this translation unit so the prefix can stop at
// the correct wrapper boundary; no MetaV4 source or ABI is modified.
#define private public
#include "../metav4_2965/metav4_2965.hpp"
#undef private
#include "../metav4_2965/metav4_2965.cpp"

namespace thomas_2945 {

struct Opponent::Impl : metav4_2965::Opponent::Impl {
  using Base = metav4_2965::Opponent::Impl;
  explicit Impl(const std::string& path) : Base(path) {}

  struct WeedPending {
    fastkag::Position position{};
    fastkag::Action action{};
  };
  struct ThomasChassisState {
    std::vector<std::vector<WeedPending>> pending;
  };
  struct CarrotVisit {
    int step{};
    int actor{};
    fastkag::Op op{fastkag::Op::PASS};
  };
  struct CarrotState {
    int last_step{-1};
    int spare_wheat{};
    int spare_carrot{};
    int credit{};
    std::array<int, 100> planted{};
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
    std::vector<fastkag::Position> placed;
  };
  struct V9HerdState {
    int last_step{-1};
    bool decided{};
    fastkag::Item species{fastkag::Item::NONE};
  };
  struct V9CarrotState {
    int last_step{-1};
    bool swapped{};
    std::unordered_map<int, int> planted;
  };
  struct Herd2Pending {
    fastkag::Position site{};
    int day{-1};
  };
  struct Herd2State {
    int last_step{-1};
    std::vector<Herd2Pending> pending;
    std::unordered_map<int, int> sites;
    int credit{};
  };
  struct CattlePlace {
    int actor{-1};
    fastkag::Position site{};
    int day{-1};
  };
  struct CattleSite {
    fastkag::Position site{};
    int day{-1};
  };
  struct CattleState {
    int last_step{-1};
    int confirmed{};
    int reserved{};
    bool pending_buy{};
    int pending_before{};
    int pending_quantity{};
    std::array<int, 32> carrying{};
    std::vector<CattlePlace> pending_places;
    std::vector<CattleSite> sites;
    int milk_credit{};
  };

  static bool same_position(fastkag::Position left,
                            fastkag::Position right) {
    return left.x == right.x && left.y == right.y;
  }

  static bool is_move(fastkag::Op op) {
    return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
           op == fastkag::Op::EAST || op == fastkag::Op::WEST;
  }

  static fastkag::Position move_position(
      fastkag::Position position, fastkag::Op op, int board) {
    if (op == fastkag::Op::NORTH) --position.y;
    else if (op == fastkag::Op::SOUTH) ++position.y;
    else if (op == fastkag::Op::WEST) --position.x;
    else if (op == fastkag::Op::EAST) ++position.x;
    position.x = std::clamp<int>(position.x, 0, board - 1);
    position.y = std::clamp<int>(position.y, 0, board - 1);
    return position;
  }

  static fastkag::Position carrot_spawn(
      const std::vector<fastkag::Position>& positions) {
    static constexpr std::array<fastkag::Position, 4> access{{
        {4, 4}, {5, 4}, {4, 5}, {5, 5}}};
    auto best = access.front();
    int count = std::count_if(positions.begin(), positions.end(),
        [&](auto value) { return same_position(value, best); });
    for (const auto candidate : access) {
      const int candidate_count = std::count_if(
          positions.begin(), positions.end(),
          [&](auto value) { return same_position(value, candidate); });
      if (candidate_count < count) {
        best = candidate;
        count = candidate_count;
      }
    }
    return best;
  }

  static void suppress_inner_tape_sales(
      fastkag::PlayerAction& action,
      std::array<int, fastkag::N_PRODUCTS>& due,
      const fastkag::PlayerAction& raw_tape) {
    // Both suppression ledgers live inside Chassis/R36, while the shared
    // executor returns an action that already contains several outer-wrapper
    // sales.  Suppression may therefore consume only the quantity originating
    // in the inner tape.  Otherwise, for example, OVERFLOW's fresh WHEAT sale
    // at step 575 is incorrectly mistaken for an old lead-sale debt.
    std::array<int, fastkag::N_PRODUCTS> inner{};
    for (const auto& order : raw_tape.market) {
      const int item = int(order.item);
      if (order.op == fastkag::Op::SELL && item >= 0 &&
          item < fastkag::N_PRODUCTS)
        inner[item] += std::max(0, order.quantity);
    }
    for (auto& order : action.market) {
      const int item = int(order.item);
      if (order.op != fastkag::Op::SELL || item < 0 ||
          item >= fastkag::N_PRODUCTS)
        continue;
      const int removed = std::min(
          {std::max(0, order.quantity), std::max(0, due[item]), inner[item]});
      order.quantity -= removed;
      due[item] -= removed;
      inner[item] -= removed;
    }
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
            return order.op == fastkag::Op::BUY_ANIMAL &&
                   order.item == fastkag::Item::GOOSE;
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
      if (order.op == fastkag::Op::BUY_ANIMAL &&
          order.item == fastkag::Item::GOOSE)
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
            return order.op == fastkag::Op::SELL &&
                   order.item == fastkag::Item::EGG;
          }), action.market.end());

    if (step >= 718) return;
    const fastkag::Item product = state.species == fastkag::Item::SHEEP
        ? fastkag::Item::WOOL : fastkag::Item::MILK;
    const int product_index = int(product);
    const auto stock = python_projected_shed(env, player, action);
    int selling = 0, planned = 0;
    for (const auto& order : action.market)
      if (order.op == fastkag::Op::SELL && order.item == product)
        selling += std::max(0, order.quantity);
    for (int future = step + 1; future < int(tape.size()); ++future)
      for (const auto& order : tape[future].market)
        if (order.op == fastkag::Op::SELL && order.item == product)
          planned += std::max(0, order.quantity);
    const int extra = stock[product_index] - selling - planned;
    if (const char* requested = std::getenv("THOMAS_DEBUG_STEP");
        requested && step == std::atoi(requested))
      std::fprintf(stderr,
                   "thomas-v9-herd-state: species=%d stock=%d selling=%d "
                   "planned=%d extra=%d\n",
                   int(state.species), stock[product_index], selling, planned,
                   extra);
    if (extra > 0 && action.market.size() < 10 &&
        env.market().prices[product_index] >= 2)
      action.market.insert(action.market.begin(),
                           {fastkag::Op::SELL, product, extra});
  }

  static void apply_v9_carrot(
      const fastkag::Simulator& env, int player,
      fastkag::PlayerAction& action, V9CarrotState& state) {
    const int step = env.step_count();
    if (step == 0 || step <= state.last_step) state = {};
    state.last_step = step;

    auto candidate = action;
    const int day = env.day();
    const auto positions = fastkag::positions(env, player);
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    bool changed = false;

    // Swapped carrots expire one day before the replay's wheat harvest.  The
    // Python wrapper turns their age-three WATER visit into the rescue harvest
    // but intentionally keeps the sticky tile ledger afterwards.
    if (!state.planted.empty()) {
      for (int actor = 0;
           actor < int(candidate.units.size()) && actor < int(positions.size());
           ++actor) {
        auto& command = candidate.units[actor];
        if (command.op != fastkag::Op::WATER) continue;
        const auto position = positions[actor];
        const int index = int(position.y) * env.config().board_size +
                          int(position.x);
        const auto planted = state.planted.find(index);
        if (planted == state.planted.end() || planted->second != day - 3)
          continue;
        const auto& value = farm.tiles[index];
        if (value.kind == fastkag::TileKind::PLANT &&
            value.crop == fastkag::Item::CARROT &&
            int(value.planted_day) == day - 3 && value.yield_units > 0) {
          command = {fastkag::Op::HARVEST};
          changed = true;
        }
      }
    }

    const int carrot = int(fastkag::Item::CARROT);
    const int wheat = int(fastkag::Item::WHEAT);
    const int carrot_price = env.market().prices[carrot];
    const int wheat_price = env.market().prices[wheat];
    int wheat_held = private_state.shed[wheat];
    for (const auto& inventory : private_state.inventories)
      wheat_held += inventory[wheat];
    const double ratio = double(carrot_price) / std::max(1, wheat_price);
    const bool boom = ratio >= 3.5;
    const int reserve = boom ? 10 : 40;

    if (boom && day >= 10 && day <= 23 && wheat_held < 40) {
      const int topup = 40 - wheat_held;
      const double budget = double(farm.money) - 1500.0;
      const int affordable = int(std::floor(
          budget / std::max(1, wheat_price + 5)));
      const int quantity = std::min(topup, affordable);
      const bool already_buying = std::any_of(
          candidate.market.begin(), candidate.market.end(),
          [](const auto& order) {
            return order.op == fastkag::Op::BUY_PRODUCT &&
                   order.item == fastkag::Item::WHEAT;
          });
      if (quantity > 0 && candidate.market.size() < 10 && !already_buying) {
        candidate.market.push_back(
            {fastkag::Op::BUY_PRODUCT, fastkag::Item::WHEAT, quantity});
        changed = true;
      }
    }

    if (day >= 10 && day <= 23 && wheat_held >= reserve && ratio >= 1.8) {
      int carrot_seeds = private_state.seeds[carrot];
      for (const auto& command : candidate.units)
        carrot_seeds -= command.op == fastkag::Op::PLANT &&
                        command.item == fastkag::Item::CARROT;
      for (int actor = 0;
           actor < int(candidate.units.size()) && carrot_seeds > 0; ++actor) {
        auto& command = candidate.units[actor];
        if (command.op != fastkag::Op::PLANT ||
            command.item != fastkag::Item::WHEAT)
          continue;
        command.item = fastkag::Item::CARROT;
        --carrot_seeds;
        state.swapped = true;
        changed = true;
        if (actor < int(positions.size())) {
          const auto position = positions[actor];
          const int index = int(position.y) * env.config().board_size +
                            int(position.x);
          state.planted[index] = day;
        }
      }
      for (auto& order : candidate.market) {
        if (order.op != fastkag::Op::BUY_SEED ||
            order.item != fastkag::Item::WHEAT)
          continue;
        order.item = fastkag::Item::CARROT;
        state.swapped = true;
        changed = true;
      }
    }

    if (state.swapped && day < 24) {
      const auto stock = python_projected_shed(env, player, candidate);
      int selling = 0;
      for (const auto& order : candidate.market)
        if (order.op == fastkag::Op::SELL &&
            order.item == fastkag::Item::CARROT)
          selling += std::max(0, order.quantity);
      const int quantity = stock[carrot] - selling;
      if (quantity > 0 && candidate.market.size() < 10 &&
          carrot_price >= 2) {
        candidate.market.insert(
            candidate.market.begin(),
            {fastkag::Op::SELL, fastkag::Item::CARROT, quantity});
        changed = true;
      }
    }
    if (changed) action = std::move(candidate);
  }

  static bool chassis_noop(const fastkag::Simulator& env, int player,
                           int actor, fastkag::Position position,
                           const fastkag::Action& command) {
    const int board = env.config().board_size;
    const auto& value = tile(env, player, position);
    const auto& private_state = env.privates()[player];
    static const std::array<int, fastkag::N_ITEMS> empty_inventory{};
    const auto& inventory = actor < int(private_state.inventories.size())
        ? private_state.inventories[actor] : empty_inventory;
    const int item = int(command.item);
    if (is_move(command.op)) {
      int x = position.x, y = position.y;
      if (command.op == fastkag::Op::NORTH) --y;
      else if (command.op == fastkag::Op::SOUTH) ++y;
      else if (command.op == fastkag::Op::WEST) --x;
      else if (command.op == fastkag::Op::EAST) ++x;
      return x < 0 || y < 0 || x >= board || y >= board;
    }
    if (command.op == fastkag::Op::PASS) return true;
    if (command.op == fastkag::Op::DROP) {
      const bool loaded = std::any_of(
          inventory.begin(), inventory.end(), [](int quantity) {
            return quantity > 0;
          });
      return !fastkag::shed_adjacent(position) || !loaded;
    }
    if (command.op == fastkag::Op::PICKUP)
      return !fastkag::shed_adjacent(position);
    if (command.op == fastkag::Op::PLACE) {
      const bool animal = item >= int(fastkag::Item::GOOSE) &&
                          item <= int(fastkag::Item::SHEEP);
      const auto structure = item == int(fastkag::Item::GOOSE)
          ? fastkag::TileKind::COOP : fastkag::TileKind::PASTURE;
      if (animal && value.kind == structure)
        return inventory[item] <= 0;
      return !fastkag::shed_adjacent(position) || item < 0 ||
             item >= fastkag::N_ITEMS || inventory[item] <= 0;
    }
    if (value.kind == fastkag::TileKind::LOCKED) return true;
    if (command.op == fastkag::Op::PLANT)
      return value.kind != fastkag::TileKind::EMPTY || item < 0 ||
             item >= fastkag::N_CROPS || private_state.seeds[item] <= 0;
    if (command.op == fastkag::Op::WATER)
      return value.kind != fastkag::TileKind::PLANT || value.watered_today;
    if (command.op == fastkag::Op::HARVEST)
      return value.yield_units <= 0;
    if (command.op == fastkag::Op::FERTILIZE)
      return value.kind != fastkag::TileKind::PLANT ||
             inventory[int(fastkag::Item::FERTILIZER)] <= 0;
    if (command.op == fastkag::Op::DIG)
      return value.kind == fastkag::TileKind::EMPTY ||
             value.kind == fastkag::TileKind::ANIMAL;
    if (command.op == fastkag::Op::BUILD_COOP ||
        command.op == fastkag::Op::BUILD_PASTURE)
      return value.kind != fastkag::TileKind::EMPTY;
    if (command.op == fastkag::Op::FEED)
      return value.kind != fastkag::TileKind::ANIMAL || value.fed_today ||
             inventory[int(fastkag::Item::WHEAT)] <= 0;
    if (command.op == fastkag::Op::COLLECT_FERTILIZER)
      return value.kind != fastkag::TileKind::ANIMAL ||
             !value.fertilizer_available;
    if (command.op == fastkag::Op::CARE)
      return value.kind != fastkag::TileKind::ANIMAL || value.cared_today;
    return true;
  }

  static void apply_v231_cattle(
      const fastkag::Simulator& env, int player,
      fastkag::PlayerAction& action, CattleState& state) {
    const int step = env.step_count();
    if (step == 0 || step <= state.last_step) state = {};
    state.last_step = step;
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    const auto positions = fastkag::positions(env, player);

    if (state.pending_buy) {
      const int gained = std::max(
          0, private_state.shed[int(fastkag::Item::COW)] -
                 state.pending_before);
      const int confirmed = std::min(state.pending_quantity, gained);
      state.confirmed += confirmed;
      state.reserved += confirmed;
      state.pending_buy = false;
    }
    for (const auto& pending : state.pending_places) {
      const auto& value = tile(env, player, pending.site);
      if (value.kind != fastkag::TileKind::ANIMAL ||
          value.animal != fastkag::Item::COW ||
          value.placed_day != pending.day)
        continue;
      auto site = std::find_if(
          state.sites.begin(), state.sites.end(), [&](const auto& entry) {
            return same_position(entry.site, pending.site);
          });
      if (site == state.sites.end())
        state.sites.push_back({pending.site, pending.day});
      else
        site->day = pending.day;
      if (pending.actor >= 0 && pending.actor < int(state.carrying.size()))
        state.carrying[pending.actor] =
            std::max(0, state.carrying[pending.actor] - 1);
    }
    state.pending_places.clear();

    std::vector<fastkag::Position> harvested;
    std::vector<fastkag::Position> occupied;
    int cow_available = private_state.shed[int(fastkag::Item::COW)];
    for (std::size_t actor = 0;
         actor < action.units.size() && actor < positions.size(); ++actor) {
      auto& command = action.units[actor];
      const auto position = positions[actor];
      const auto& value = tile(env, player, position);
      static const std::array<int, fastkag::N_ITEMS> empty_inventory{};
      const auto& inventory = actor < private_state.inventories.size()
          ? private_state.inventories[actor] : empty_inventory;
      const auto site = std::find_if(
          state.sites.begin(), state.sites.end(), [&](const auto& entry) {
            return same_position(entry.site, position);
          });
      const bool already_harvested = std::any_of(
          harvested.begin(), harvested.end(), [&](const auto& seen) {
            return same_position(seen, position);
          });
      if (command.op == fastkag::Op::HARVEST &&
          site != state.sites.end() && !already_harvested &&
          value.kind == fastkag::TileKind::ANIMAL &&
          value.animal == fastkag::Item::COW &&
          value.placed_day == site->day) {
        state.milk_credit += std::max(0, int(value.yield_units));
        harvested.push_back(position);
      }

      if (command.op == fastkag::Op::PICKUP &&
          command.item == fastkag::Item::SHEEP) {
        const int quantity = std::max(0, int(command.quantity));
        const int center = env.config().board_size / 2;
        int animal_cargo = 0;
        for (int animal = int(fastkag::Item::GOOSE);
             animal <= int(fastkag::Item::SHEEP); ++animal)
          animal_cargo += inventory[animal];
        if (quantity > 0 && state.reserved >= quantity &&
            cow_available >= quantity &&
            (position.x == center - 1 || position.x == center) &&
            (position.y == center - 1 || position.y == center) &&
            animal_cargo == 0) {
          command.item = fastkag::Item::COW;
          state.reserved -= quantity;
          cow_available -= quantity;
          if (actor < state.carrying.size())
            state.carrying[actor] += quantity;
        }
      }
      const bool position_occupied = std::any_of(
          occupied.begin(), occupied.end(), [&](const auto& seen) {
            return same_position(seen, position);
          });
      if (command.op == fastkag::Op::PLACE &&
          command.item == fastkag::Item::SHEEP &&
          actor < state.carrying.size() && state.carrying[actor] > 0 &&
          inventory[int(fastkag::Item::COW)] > 0 &&
          value.kind == fastkag::TileKind::PASTURE &&
          value.animal == fastkag::Item::NONE && !position_occupied) {
        command.item = fastkag::Item::COW;
        state.pending_places.push_back(
            {int(actor), position, env.day()});
      }
      const int item = int(command.item);
      if (command.op == fastkag::Op::PLACE &&
          item >= int(fastkag::Item::GOOSE) &&
          item <= int(fastkag::Item::SHEEP) && inventory[item] > 0)
        occupied.push_back(position);
    }

    std::vector<fastkag::Action*> animal_orders;
    for (auto& order : action.market)
      if (order.op == fastkag::Op::BUY_ANIMAL)
        animal_orders.push_back(&order);
    int cows = 0, sheep = 0;
    for (const auto& value : farm.tiles) {
      cows += value.kind == fastkag::TileKind::ANIMAL &&
              value.animal == fastkag::Item::COW;
      sheep += value.kind == fastkag::TileKind::ANIMAL &&
                value.animal == fastkag::Item::SHEEP;
    }
    int cargo = 0;
    for (const auto& inventory : private_state.inventories)
      for (int animal = int(fastkag::Item::GOOSE);
           animal <= int(fastkag::Item::SHEEP); ++animal)
        cargo += inventory[animal];
    for (int animal = int(fastkag::Item::GOOSE);
         animal <= int(fastkag::Item::SHEEP); ++animal)
      cargo += private_state.shed[animal];
    int milk_shops = 0;
    bool yarn = false;
    for (const int shop : env.shops()) {
      milk_shops += shop == 3 || shop == 5 || shop == 6;
      yarn = yarn || shop == 7;
    }
    const bool carrying = std::any_of(
        state.carrying.begin(), state.carrying.end(),
        [](int quantity) { return quantity != 0; });
    if (step >= 216 && step <= 227 && env.shops().size() >= 3 &&
        state.confirmed < 4 && state.reserved == 0 && !carrying &&
        state.pending_places.empty() && cargo == 0 &&
        animal_orders.size() == 1 &&
        animal_orders[0]->item == fastkag::Item::SHEEP &&
        milk_shops >= 2 && !yarn &&
        env.market().prices[int(fastkag::Item::MILK)] >=
            env.market().prices[int(fastkag::Item::WOOL)] &&
        cows >= 4 && sheep >= 2) {
      const int quantity = std::max(0, int(animal_orders[0]->quantity));
      if (quantity >= 1 && quantity <= 2 &&
          quantity <= 4 - state.confirmed) {
        animal_orders[0]->item = fastkag::Item::COW;
        state.pending_buy = true;
        state.pending_before =
            private_state.shed[int(fastkag::Item::COW)];
        state.pending_quantity = quantity;
      }
    }

    if (state.milk_credit <= 0) return;
    const auto stock = python_projected_shed(env, player, action);
    int planned = 0;
    for (const auto& order : action.market)
      if (order.op == fastkag::Op::SELL &&
          order.item == fastkag::Item::MILK)
        planned += std::max(0, int(order.quantity));
    const int extra = std::min(
        state.milk_credit,
        std::max(0, stock[int(fastkag::Item::MILK)] - planned));
    if (extra <= 0) return;
    for (auto& order : action.market) {
      if (order.op != fastkag::Op::SELL ||
          order.item != fastkag::Item::MILK || order.quantity <= 0)
        continue;
      order.quantity += extra;
      state.milk_credit -= extra;
      break;
    }
  }

  const std::array<int, 720>& r85_native_reserve(int route) {
    const auto found = r85_native_reserves.find(route);
    if (found != r85_native_reserves.end()) return found->second;
    std::array<int, 720> reserve{};
    const int fertilizer = int(fastkag::Item::FERTILIZER);
    for (int step = 718; step >= 0; --step) {
      const int tape_route = step >= 648 ? 2 : route;
      const auto& planned = assets.library.routes[slot_for(tape_route)][step];
      int pickup = 0, purchase = 0;
      for (const auto& command : planned.units)
        if (command.op == fastkag::Op::PICKUP &&
            int(command.item) == fertilizer)
          pickup += std::max(0, command.quantity);
      for (const auto& order : planned.market)
        if (order.op == fastkag::Op::BUY_PRODUCT &&
            int(order.item) == fertilizer)
          purchase += std::max(0, order.quantity);
      reserve[step] = pickup + std::max(0, reserve[step + 1] - purchase);
    }
    return r85_native_reserves.emplace(route, reserve).first->second;
  }

  int r85_fertilizer_reserve(const fastkag::Simulator& env, int player,
                             int route, const SeatState& seat,
                             const InputState& input_state) {
    const int next = std::min(719, env.step_count() + 1);
    int dedicated = 0;
    const auto& inventories = env.privates()[player].inventories;
    for (const auto& [actor, role] : seat.tomato.workers) {
      if (!role.needs_fertilizer || role.loaded) continue;
      const int desired = role.fertilizer_quantity >= 0
          ? role.fertilizer_quantity : (role.fertilizer_worker ? 10 : 5);
      const int carried = actor >= 0 && actor < int(inventories.size())
          ? inventories[actor][int(fastkag::Item::FERTILIZER)] : 0;
      dedicated += std::max(0, desired - carried);
    }
    if (seat.tomato.pending && seat.tomato.pending->fertilizer)
      dedicated += 10;

    // Python merges {**workers, **pending}; a pending plan for the same actor
    // replaces, rather than adds to, its confirmed worker reservation.
    for (const auto& [actor, plan] : input_state.workers) {
      const bool shadowed = std::any_of(
          input_state.pending.begin(), input_state.pending.end(),
          [&](const auto& row) { return row.first == actor; });
      if (!shadowed && !plan.loaded) dedicated += std::max(0, plan.quantity);
    }
    for (const auto& [actor, plan] : input_state.pending)
      if (!plan.loaded) dedicated += std::max(0, plan.quantity);
    return std::max(14, r85_native_reserve(route)[next] + dedicated);
  }

  void apply_r85_fertilizer(const fastkag::Simulator& env, int player,
                            int route, fastkag::PlayerAction& action,
                            const SeatState& seat,
                            const InputState& input_state) {
    if (env.day() < 6 || env.day() > 28 || action.market.size() >= 10 ||
        std::any_of(action.market.begin(), action.market.end(),
                    [](const auto& order) {
                      return order.op != fastkag::Op::SELL;
                    }))
      return;
    const int fertilizer = int(fastkag::Item::FERTILIZER);
    const auto stock = python_projected_shed(env, player, action);
    int sold = 0;
    for (const auto& order : action.market)
      if (order.op == fastkag::Op::SELL && int(order.item) == fertilizer)
        sold += std::max(0, order.quantity);
    const int extra = std::max(0, stock[fertilizer]) - sold -
        r85_fertilizer_reserve(env, player, route, seat, input_state);
    if (extra > 0)
      action.market.push_back({fastkag::Op::SELL,
                               fastkag::Item::FERTILIZER, extra});
  }

  void apply_r51_close_warehouse(const fastkag::Simulator& env, int player,
                                 int route,
                                 fastkag::PlayerAction& action) const {
    const int step = env.step_count(), day = env.day();
    if (env.hour() != 23 || day < 12 || day > 28 ||
        std::any_of(action.market.begin(), action.market.end(),
                    [](const auto& order) {
                      return order.op != fastkag::Op::SELL;
                    }))
      return;

    auto projected_action = action;
    std::array<int, fastkag::N_CROPS> demand{};
    for (const auto& command : projected_action.units)
      if (command.op == fastkag::Op::PLANT && int(command.item) >= 0 &&
          int(command.item) < fastkag::N_CROPS)
        ++demand[int(command.item)];
    const auto& seeds = env.privates()[player].seeds;
    for (auto& command : projected_action.units)
      if (command.op == fastkag::Op::PLANT && int(command.item) >= 0 &&
          int(command.item) < fastkag::N_CROPS &&
          demand[int(command.item)] > seeds[int(command.item)])
        command = {};
    std::array<fastkag::PlayerAction, 2> joint{};
    joint[player] = std::move(projected_action);
    const auto after_units = env.preview_unit_phase(joint);
    const auto& private_state = after_units.privates()[player];
    auto post = private_state.shed;
    for (const auto& order : action.market) {
      const int item = int(order.item);
      if (order.op == fastkag::Op::SELL && item >= 0 &&
          item < fastkag::N_ITEMS)
        post[item] = std::max(0, post[item] - std::max(0, order.quantity));
    }
    int carried = 0;
    for (const auto& inventory : private_state.inventories)
      for (const int quantity : inventory)
        carried += std::max(0, quantity);
    int needed = count_item(post) + carried - 100;
    if (needed <= 0) return;

    std::array<int, fastkag::N_PRODUCTS - 2> cash{};
    int cursor = 0;
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item)
      if (item != int(fastkag::Item::WHEAT) &&
          item != int(fastkag::Item::FERTILIZER))
        cash[cursor++] = item;
    std::stable_sort(cash.begin(), cash.end(), [&](int left, int right) {
      return env.market().prices[left] > env.market().prices[right];
    });
    const auto add_sale = [&](int item, int quantity) {
      auto existing = std::find_if(
          action.market.begin(), action.market.end(), [&](const auto& order) {
            return order.op == fastkag::Op::SELL && int(order.item) == item;
          });
      if (existing != action.market.end()) {
        existing->quantity = std::max(0, existing->quantity) + quantity;
        return true;
      }
      if (action.market.size() >= 10) return false;
      action.market.push_back(
          {fastkag::Op::SELL, fastkag::Item(item), quantity});
      return true;
    };
    for (const int item : cash) {
      const int quantity = std::min(needed, std::max(0, post[item]));
      if (quantity <= 0 || !add_sale(item, quantity)) continue;
      needed -= quantity;
      post[item] -= quantity;
      if (needed <= 0) break;
    }
    if (needed <= 0) return;

    int reserve = 0;
    for (int future = step + 1; future < 719; ++future) {
      const int future_route = future >= 648 ? 2 : route;
      const auto& frame =
          assets.library.routes[slot_for(future_route)][future];
      for (const auto& command : frame.units)
        if (command.op == fastkag::Op::PICKUP &&
            command.item == fastkag::Item::WHEAT)
          reserve += std::max(0, command.quantity);
      if (std::any_of(frame.market.begin(), frame.market.end(),
                      [](const auto& order) {
                        return order.op == fastkag::Op::BUY_PRODUCT &&
                               order.item == fastkag::Item::WHEAT;
                      }))
        break;
    }
    const int wheat = int(fastkag::Item::WHEAT);
    int incoming = 0, others = 0;
    for (const auto& inventory : private_state.inventories)
      for (int item = 0; item < fastkag::N_ITEMS; ++item)
        if (item == wheat) incoming += std::max(0, inventory[item]);
        else others += std::max(0, inventory[item]);
    for (int item = 0; item < fastkag::N_ITEMS; ++item)
      if (item != wheat) others += post[item];
    const int quantity = 100 - others >= reserve
        ? std::min({needed, std::max(0, post[wheat]),
                    std::max(0, post[wheat] + incoming - reserve)})
        : 0;
    if (quantity > 0) add_sale(wheat, quantity);
  }

  void apply_exact_chassis_units(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action, ThomasChassisState& state) {
    const int step = env.step_count();
    const auto positions = fastkag::positions(env, player);
    action.units = tape[step].units;
    action.units.resize(positions.size());
    state.pending.resize(positions.size());
    const auto& next = step + 1 < int(tape.size())
        ? tape[step + 1].units : std::vector<fastkag::Action>{};
    for (std::size_t actor = 0; actor < action.units.size(); ++actor) {
      auto& queue = state.pending[actor];
      if (!queue.empty() &&
          !same_position(queue.front().position, positions[actor]))
        queue.clear();
      auto command = action.units[actor];
      const auto& value = tile(env, player, positions[actor]);
      const bool weed = value.kind == fastkag::TileKind::WEED;
      const bool noop = chassis_noop(
          env, player, int(actor), positions[actor], command);
      const auto next_op = actor < next.size()
          ? next[actor].op : fastkag::Op::PASS;
      if ((command.op == fastkag::Op::PLANT ||
           command.op == fastkag::Op::BUILD_COOP ||
           command.op == fastkag::Op::BUILD_PASTURE) && weed) {
        queue.push_back({positions[actor], command});
        command = {fastkag::Op::DIG};
      } else if (!queue.empty() && noop) {
        const auto replay = queue.front().action;
        if (replay.op == fastkag::Op::PLANT && is_move(next_op)) {
          queue.clear();
        } else {
          queue.erase(queue.begin());
          if (command.op != fastkag::Op::PASS && !is_move(command.op))
            queue.push_back({positions[actor], command});
          command = replay;
          if (queue.empty()) queue.clear();
        }
      } else if (weed && noop) {
        command = {fastkag::Op::DIG};
      }
      action.units[actor] = command;
    }
  }

  const fastkag::PlayerAction& carrot_frame(
      const std::vector<fastkag::PlayerAction>& current_tape,
      int step) const {
    const auto& tape = step >= 648
        ? assets.library.routes[slot_for(2)] : current_tape;
    static const fastkag::PlayerAction empty;
    return step >= 0 && step < int(tape.size()) ? tape[step] : empty;
  }

  std::vector<CarrotVisit> carrot_visits(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& current_tape,
      const fastkag::PlayerAction& action, fastkag::Position target,
      int end, int start) const {
    const int step = env.step_count(), board = env.config().board_size;
    auto positions = fastkag::positions(env, player);
    std::vector<CarrotVisit> result;
    for (int future = step; future <= std::min(end, 719); ++future) {
      const auto& frame = future == step
          ? action : carrot_frame(current_tape, future);
      for (int actor = 0; actor < int(positions.size()); ++actor) {
        const auto command = actor < int(frame.units.size())
            ? frame.units[actor] : fastkag::Action{};
        if (is_move(command.op)) {
          positions[actor] = move_position(positions[actor], command.op, board);
        } else if (same_position(positions[actor], target) && future >= start) {
          result.push_back({future, actor, command.op});
        }
      }
      for (const auto& order : frame.market)
        if (order.op == fastkag::Op::HIRE)
          positions.push_back(carrot_spawn(positions));
      if (future % 24 == 23) positions = {{4, 4}};
    }
    return result;
  }

  static int carrot_decays(int mature_loss_step, int begin, int end) {
    begin = std::max(begin, mature_loss_step);
    if (end <= begin) return 0;
    const int first = (begin - mature_loss_step) % 2 == 0
        ? begin : begin + 1;
    return first >= end ? 0 : (end - 1 - first) / 2 + 1;
  }

  struct CarrotYield {
    int harvest{};
    int rescue{};
  };

  static CarrotYield carrot_yield_path(
      fastkag::Item crop, int planted,
      const std::vector<CarrotVisit>& visits, int initial_yield = 1,
      int fertilized_until = -1, int watered_day = -1,
      int now_step = 0) {
    const int max_days = crop == fastkag::Item::WHEAT ? 4 : 3;
    const int cap = crop == fastkag::Item::WHEAT ? 6 : 4;
    const int low = (max_days + 1) / 2;
    const int mature_loss_step = (planted + max_days + 1) * 24;
    int yield = initial_yield, rescue = 0;
    for (const auto& visit : visits) {
      const int day = visit.step / 24;
      const int age = day - planted;
      const int current = yield -
          carrot_decays(mature_loss_step, now_step, visit.step);
      if (current <= 0 && visit.step > mature_loss_step)
        return {0, rescue};
      if (visit.op == fastkag::Op::HARVEST)
        return {age >= 2 ? std::max(0, current) : 0, rescue};
      if (visit.op == fastkag::Op::PLANT ||
          visit.op == fastkag::Op::DIG ||
          visit.op == fastkag::Op::BUILD_COOP ||
          visit.op == fastkag::Op::BUILD_PASTURE)
        return {0, rescue};
      if (age >= 2 && current > rescue && visit.step > now_step)
        rescue = current;
      if (visit.op == fastkag::Op::WATER && age >= low &&
          age <= max_days && day != watered_day) {
        watered_day = day;
        yield = std::min(cap, yield +
            (fertilized_until >= day ? 2 : 1));
      }
    }
    return {0, rescue};
  }

  int carrot_feed_need(
      const std::vector<fastkag::PlayerAction>& current_tape,
      int step) const {
    int need = 0;
    for (int future = step;
         future <= std::min(719, step + 48); ++future)
      for (const auto& command : carrot_frame(current_tape, future).units)
        need += command.op == fastkag::Op::FEED;
    return need;
  }

  void apply_carrot2(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& current_tape,
      fastkag::PlayerAction& action, CarrotState& state) const {
    const int step = env.step_count(), day = env.day();
    if (step == 0 || step <= state.last_step) {
      state = {};
      state.planted.fill(-1);
    }
    state.last_step = step;
    if (step > 717) return;

    const auto positions = fastkag::positions(env, player);
    action.units.resize(positions.size());
    const auto& farm = env.farms()[player];
    const auto& private_state = env.privates()[player];
    const int carrot = int(fastkag::Item::CARROT);
    const int wheat = int(fastkag::Item::WHEAT);
    const int carrot_price = env.market().prices[carrot];
    const int wheat_price = env.market().prices[wheat];

    // Bookkeeping and rescue of fields that this wrapper swapped earlier.
    for (int index = 0; index < int(state.planted.size()); ++index) {
      const int planted = state.planted[index];
      if (planted < 0) continue;
      const fastkag::Position position{
          std::int8_t(index % env.config().board_size),
          std::int8_t(index / env.config().board_size)};
      const auto& value = farm.tiles[index];
      if (value.kind != fastkag::TileKind::PLANT ||
          value.crop != fastkag::Item::CARROT ||
          int(value.planted_day) != planted) {
        state.planted[index] = -1;
        continue;
      }
      const auto actor = std::find_if(
          positions.begin(), positions.end(),
          [&](auto value_) { return same_position(value_, position); });
      if (actor == positions.end()) continue;
      const int unit = int(actor - positions.begin());
      auto& command = action.units[unit];
      const int yield = std::max(0, int(value.yield_units));
      if (command.op == fastkag::Op::HARVEST) {
        if (day - planted >= 2 && yield > 0) {
          state.credit += yield;
          state.planted[index] = -1;
        }
        continue;
      }
      if (is_move(command.op) || day - planted < 2 || yield <= 0)
        continue;
      const auto visits = carrot_visits(
          env, player, current_tape, action, position,
          (planted + 5) * 24, step);
      const auto future = carrot_yield_path(
          fastkag::Item::CARROT, planted, visits, yield,
          value.fertilized_until_day,
          value.watered_today ? day : -1, step);
      if (yield > std::max(future.harvest, future.rescue)) {
        command = {fastkag::Op::HARVEST};
        state.credit += yield;
        state.planted[index] = -1;
      }
    }

    const bool pays_now = 3 * carrot_price - 20 >
                          4 * wheat_price - 15;
    if (day >= 6 && day <= 28 && pays_now) {
      int planting_carrot = 0;
      for (const auto& command : action.units)
        planting_carrot += command.op == fastkag::Op::PLANT &&
                           command.item == fastkag::Item::CARROT;
      int seeds = std::min(
          state.spare_carrot,
          private_state.seeds[carrot] - planting_carrot);
      std::optional<bool> wheat_ok;
      for (int actor = 0;
           actor < int(action.units.size()) && actor < int(positions.size()) &&
           seeds > 0; ++actor) {
        auto& command = action.units[actor];
        if (command.op != fastkag::Op::PLANT ||
            command.item != fastkag::Item::WHEAT)
          continue;
        const auto position = positions[actor];
        const int index = int(position.y) * env.config().board_size +
                          int(position.x);
        if (farm.tiles[index].kind != fastkag::TileKind::EMPTY) continue;
        if (!wheat_ok) {
          int held = private_state.shed[wheat];
          for (const auto& inventory : private_state.inventories)
            held += inventory[wheat];
          wheat_ok = held >= carrot_feed_need(current_tape, step);
        }
        if (!*wheat_ok) break;
        const auto visits = carrot_visits(
            env, player, current_tape, action, position,
            (day + 6) * 24, step + 1);
        const auto wheat_yield = carrot_yield_path(
            fastkag::Item::WHEAT, day, visits);
        const auto carrot_yield = carrot_yield_path(
            fastkag::Item::CARROT, day, visits);
        const int carrot_units = std::max(
            carrot_yield.harvest, carrot_yield.rescue);
        if (carrot_units * carrot_price - 20 >
            wheat_yield.harvest * wheat_price - 15) {
          command.item = fastkag::Item::CARROT;
          --seeds;
          --state.spare_carrot;
          state.planted[index] = day;
          ++state.spare_wheat;
        }
      }
    }

    // Reconcile seed purchases against substitutions already made.
    std::vector<fastkag::Action> market;
    market.reserve(action.market.size() + 1);
    for (auto order : action.market) {
      if (order.op == fastkag::Op::BUY_SEED &&
          (order.item == fastkag::Item::WHEAT ||
           order.item == fastkag::Item::CARROT)) {
        int& spare = order.item == fastkag::Item::WHEAT
            ? state.spare_wheat : state.spare_carrot;
        const int cut = std::min(std::max(0, order.quantity), spare);
        spare -= cut;
        order.quantity -= cut;
        if (order.quantity <= 0) continue;
      }
      market.push_back(order);
    }
    action.market = std::move(market);

    if (day >= 6 && day <= 27 && pays_now && action.market.size() < 10) {
      int planting = 0, buying = 0;
      for (const auto& command : action.units)
        planting += command.op == fastkag::Op::PLANT &&
                    command.item == fastkag::Item::CARROT;
      for (const auto& order : action.market)
        if (order.op == fastkag::Op::BUY_SEED &&
            order.item == fastkag::Item::CARROT)
          buying += std::max(0, order.quantity);
      const int quantity = 8 -
          (private_state.seeds[carrot] - planting) - buying;
      if (quantity > 0 && farm.money >= 800 + 20 * quantity) {
        action.market.push_back(
            {fastkag::Op::BUY_SEED, fastkag::Item::CARROT, quantity});
        state.spare_carrot += quantity;
      }
    }

    if (state.credit > 0 && carrot_price >= 2 &&
        action.market.size() < 10) {
      const auto stock = python_projected_shed(env, player, action);
      int selling = 0;
      for (const auto& order : action.market)
        if (order.op == fastkag::Op::SELL &&
            order.item == fastkag::Item::CARROT)
          selling += std::max(0, order.quantity);
      const int quantity = std::min(
          state.credit, std::max(0, stock[carrot] - selling));
      if (quantity > 0) {
        action.market.insert(
            action.market.begin(),
            {fastkag::Op::SELL, fastkag::Item::CARROT, quantity});
        state.credit -= quantity;
      }
    }
    if (action.market.size() > 10) action.market.resize(10);
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
        const auto& value = farm.tiles[y * env.config().board_size + x];
        if (value.kind != fastkag::TileKind::PLANT ||
            (value.crop != fastkag::Item::WHEAT &&
             value.crop != fastkag::Item::CARROT))
          continue;
        const int first = 2;
        const int last = value.crop == fastkag::Item::WHEAT ? 4 : 3;
        const int age = day - value.planted_day;
        if (age < 1 || age >= last) continue;
        targets.push_back({
            {std::int8_t(x), std::int8_t(y)}, value.crop,
            value.planted_day, value.yield_units,
            value.fertilized_until_day, value.watered_today, {}, -1,
            first, last, value.crop == fastkag::Item::WHEAT ? 6 : 4});
      }
    }

    auto positions = fastkag::positions(env, player);
    if (int(positions.size()) > expected + 1)
      positions.resize(expected + 1);
    std::array<std::array<bool, 100>, 30> seen{};
    for (int future = step;
         future < std::min(712, (day + 4) * 24); ++future) {
      const auto& frame = carrot_frame(current_tape, future);
      for (int actor = 0; actor < int(positions.size()); ++actor) {
        const auto command = actor < int(frame.units.size())
            ? frame.units[actor] : fastkag::Action{};
        const auto position = positions[actor];
        auto target = std::find_if(
            targets.begin(), targets.end(), [&](const auto& value) {
              return same_position(value.position, position);
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
        if (is_move(command.op))
          positions[actor] = move_position(
              positions[actor], command.op, env.config().board_size);
      }
      for (const auto& order : frame.market)
        if (order.op == fastkag::Op::HIRE)
          positions.push_back(carrot_spawn(positions));
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
      if (is_move(action.units[actor].op))
        positions[actor] = move_position(
            positions[actor], action.units[actor].op,
            env.config().board_size);
    int native_hires = 0;
    for (const auto& order : action.market)
      native_hires += order.op == fastkag::Op::HIRE;
    fastkag::Position start{4, 4};
    for (int index = 0; index < native_hires + worker_index + 1; ++index) {
      start = carrot_spawn(positions);
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
      double net{};
      int value{};
      int cost{};
      int mode{};
      int quantity{};
      std::vector<InputPlan> plans;
    };
    std::optional<Candidate> best;
    const auto better = [](const Candidate& left, const Candidate& right) {
      return std::tuple<double, int, int, int, int>(
                 left.net, left.value, -left.cost,
                 -int(left.plans.size()), -left.mode) >
             std::tuple<double, int, int, int, int>(
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
                     crop,
                     env.market().inventory[crop] + all_units[crop] +
                         units[crop]) -
                     2);
        if (value < 1.5 * cost + 50 ||
            env.farms()[player].money <
                candidate.cost + cost + 3000)
          break;
        for (const auto& entry : plan.path)
          for (int index = 0; index < int(targets.size()); ++index)
            if (same_position(targets[index].position, entry.position))
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
      fastkag::PlayerAction& action, SeatState& seat,
      InputState& state, const ThomasChassisState& chassis_state) const {
    const int step = env.step_count(), day = env.day(), hour = env.hour();
    if (step == 0 || step <= state.last_step) state = {};
    state.last_step = step;
    if (state.day != day) {
      state.day = day;
      state.workers.clear();
      state.pending.clear();
      state.placed.clear();
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
          if (quantity > 0 && fastkag::shed_adjacent(position)) {
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
              state.placed.push_back(entry.position);
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
    for (int offset = 0; offset < 24 && day * 24 + offset < int(tape.size());
         ++offset)
      expected = std::max(
          expected, int(tape[day * 24 + offset].units.size()) - 1);
    for (int offset = hour; offset < 24 && day * 24 + offset < int(tape.size());
         ++offset)
      for (const auto& order : tape[day * 24 + offset].market)
        if (order.op == fastkag::Op::HIRE) return;
    if (std::any_of(chassis_state.pending.begin(), chassis_state.pending.end(),
                    [](const auto& queue) { return !queue.empty(); }))
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
    std::vector<bool> owned(farm.hands.size() + 1);
    for (int actor = 1; actor <= expected && actor < int(owned.size()); ++actor)
      owned[actor] = true;
    for (const auto& row : seat.tomato.workers) {
      if (row.first < int(owned.size()) && owned[row.first]) return;
      if (row.first < int(owned.size())) owned[row.first] = true;
    }
    for (const auto& row : seat.sheep.workers) {
      if (row.first < int(owned.size()) && owned[row.first]) return;
      if (row.first < int(owned.size())) owned[row.first] = true;
    }
    for (int actor = 1; actor < int(owned.size()); ++actor)
      if (!owned[actor]) return;

    const auto targets = input_forecast(env, player, tape, expected);
    auto stock = python_projected_shed(env, player, action);
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
        if (command.op == fastkag::Op::PICKUP &&
            command.item == fastkag::Item::FERTILIZER)
          native_pickups += std::max(0, command.quantity);
    const int topup = std::max(0, native_pickups - available);
    auto plans = input_joint_plans(
        env, player, action, targets, stock, purchases, topup);
    if (plans.empty()) return;
    int total = topup;
    const int first_actor = int(farm.hands.size()) + 1;
    for (int index = 0; index < int(plans.size()); ++index) {
      total += plans[index].quantity;
      state.pending.push_back({first_actor + index, std::move(plans[index])});
    }
    action.market.push_back({fastkag::Op::BUY_PRODUCT,
                             fastkag::Item::FERTILIZER, total});
    for (std::size_t index = 0; index < state.pending.size(); ++index)
      action.market.push_back({fastkag::Op::HIRE});
  }

  void apply_v219_thomas(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action, V219State& state) const {
    const int day = env.day(), hour = env.hour();
    const bool committed_before = state.committed;
    const int requested_before = state.requested_day;
    const std::size_t market_before = action.market.size();
    apply_v219(env, player, tape, action, state);

    // MetaV4 later monkey-patches V219 to skip safe watering crews on days
    // 19/21/23.  Thomas predates that wrapper and always executes the original
    // request.  Detect only that exact shared-runtime skip, then run the frozen
    // request semantics; every other V219 transition stays shared.
    if (!committed_before || (day != 19 && day != 21 && day != 23) ||
        requested_before == day || state.requested_day != day ||
        state.pending || action.market.size() != market_before)
      return;
    state.requested_day = requested_before;
    int latest_hire = -1;
    const int begin = day * 24;
    const int end = std::min((day + 1) * 24, 719);
    for (int at = begin; at < end; ++at)
      if (count_market(tape[at], fastkag::Op::HIRE) > 0)
        latest_hire = at - begin;
    const int deadline = latest_hire > 3 && latest_hire <= 6 ? 6 : 3;
    if (hour > deadline) return;
    for (int at = env.step_count() + 1; at < end; ++at)
      if (count_market(tape[at], fastkag::Op::HIRE) > 0) return;
    const int parent_hires = count_market(action, fastkag::Op::HIRE);
    const int expected = expected_hands(tape, day);
    if (int(env.farms()[player].hands.size()) + parent_hires != expected)
      return;
    const int count = hour <= 2 ? 1 : 2;
    if (action.market.size() + count > 10) return;
    std::int64_t budget = 0;
    for (int index = env.farms()[player].hires_today;
         index < env.farms()[player].hires_today + parent_hires + count;
         ++index)
      budget += fib(index);
    static constexpr std::array<int, fastkag::N_CROPS> seed_cost{
        10, 20, 50, 100, 80};
    static constexpr std::array<int, fastkag::N_ANIMALS> animal_cost{
        300, 400, 500};
    bool valid = true;
    for (const auto& order : action.market) {
      const int item = int(order.item);
      const int quantity = std::max(0, order.quantity);
      if (order.op == fastkag::Op::BUY_PRODUCT && item >= 0 &&
          item < fastkag::N_PRODUCTS)
        budget += std::int64_t(quantity) *
                  (env.market().prices[item] + 10);
      else if (order.op == fastkag::Op::BUY_ANIMAL &&
               item >= int(fastkag::Item::GOOSE) &&
               item <= int(fastkag::Item::SHEEP))
        budget += std::int64_t(quantity) *
                  animal_cost[item - int(fastkag::Item::GOOSE)];
      else if (order.op == fastkag::Op::BUY_SEED && item >= 0 &&
               item < fastkag::N_CROPS)
        budget += std::int64_t(quantity) * seed_cost[item];
      else if ((order.op == fastkag::Op::BUY_PRODUCT ||
                order.op == fastkag::Op::BUY_ANIMAL ||
                order.op == fastkag::Op::BUY_SEED) && quantity > 0)
        valid = false;
    }
    if (!valid || env.farms()[player].money < double(budget + 3000))
      return;
    state.pending = V219Pending{
        env.step_count(), expected + 1, count, count, false, std::nullopt};
    state.requested_day = day;
    for (int index = 0; index < count; ++index)
      action.market.push_back({fastkag::Op::HIRE});
  }

  void reserve_future_sales_thomas(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action, SeatState& state) {
    const int step = env.step_count();
    if (step < 192 || step >= 696) return;
    // Thomas v9/2 starts at 40 and can rise to 48 only after public evidence of
    // a longer rival lead.  The passive parity oracle has no such evidence.
    const int end = std::min(695, step + 40);
    const auto& private_state = env.privates()[player];
    for (std::size_t actor = 0;
         actor < action.units.size() && actor < private_state.inventories.size();
         ++actor) {
      const int item = int(action.units[actor].item);
      if (action.units[actor].op == fastkag::Op::PLACE &&
          item >= int(fastkag::Item::GOOSE) &&
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
          item < fastkag::N_PRODUCTS)
        blocked[item] = true;
    }
    static constexpr std::array<int, fastkag::N_PRODUCTS> bases{
        25, 35, 60, 120, 250, 50, 160, 200, 100};
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item) {
      if (blocked[item] || env.market().prices[item] <= bases[item] ||
          stock[item] <= 0 || action.market.size() >= 10)
        continue;
      int available = stock[item];
      std::vector<std::pair<int, int>> reservations;
      for (int due = step + 1; due <= end; ++due) {
        bool barrier = false;
        for (const auto& command : tape[due].units)
          barrier |= command.op == fastkag::Op::PICKUP &&
                     int(command.item) == item;
        for (const auto& order : tape[due].market)
          barrier |= order.op == fastkag::Op::BUY_PRODUCT &&
                     int(order.item) == item;
        if (barrier) break;
        int planned = 0;
        for (const auto& order : tape[due].market)
          if (order.op == fastkag::Op::SELL && int(order.item) == item)
            planned += std::max(0, order.quantity);
        const int amount = std::min(
            available, std::max(0, planned - state.sale_debts[due][item]));
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

  void apply_prefix_and_r36(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action, SeatState& seat) {
    auto& state = seat.prefix;
    const int step = env.step_count();
    if (step == 0) {
      action.market = {{fastkag::Op::BUY_PRODUCT, fastkag::Item::WHEAT, 20},
                       {fastkag::Op::SELL, fastkag::Item::WHEAT, 15}};
    } else if (step == 1) {
      action.market.erase(std::remove_if(
          action.market.begin(), action.market.end(), [](const auto& order) {
            return order.item == fastkag::Item::WHEAT &&
                   (order.op == fastkag::Op::BUY_PRODUCT ||
                    order.op == fastkag::Op::SELL);
          }), action.market.end());
    }

    if (state.due_step == step) {
      for (auto& order : action.market) {
        if (!fastkag::sell(order)) continue;
        const int item = int(order.item);
        const int removed = std::min(
            std::max(0, order.quantity), state.due[item]);
        order.quantity -= removed;
        state.due[item] -= removed;
      }
      state.due_step = -1;
      state.due.fill(0);
    }
    // V231 is the last wrapper before R36. Both one-step and R36 debt
    // suppression have already happened; reserve against this causal parent
    // before later credit-sale wrappers consume the same stock.
    apply_v231_cattle(env, player, action, cattle[player]);
    reserve_future_sales_thomas(env, player, tape, action, seat);

    const int next = step + 1;
    if (step < 288 && next < int(tape.size()) && next % 72 != 0 &&
        step % 4 != 0) {
      static constexpr std::array<int, fastkag::N_PRODUCTS> bases{
          25, 35, 60, 120, 250, 50, 160, 200, 100};
      auto projected = fastkag::projected_shed(env, player, action);
      for (int item = 0;
           item < fastkag::N_PRODUCTS && action.market.size() < 10; ++item) {
        const int planned = fastkag::planned_sell(tape[next], item);
        if (planned <= 0 || fastkag::planned_sell(action, item) > 0 ||
            env.market().prices[item] <= bases[item])
          continue;
        const int quantity = std::min(projected[item], planned);
        if (quantity <= 0) continue;
        action.market.push_back(
            {fastkag::Op::SELL, fastkag::Item(item), quantity});
        projected[item] -= quantity;
        state.due[item] += quantity;
        state.due_step = next;
      }
    }
  }

  static void sell_outer_credit(
      const fastkag::Simulator& env, int player,
      fastkag::PlayerAction& action, int product, int& credit,
      bool merge_when_full = false) {
    if (credit <= 0 || env.market().prices[product] < 2 ||
        (!merge_when_full && action.market.size() >= 10))
      return;
    const auto stock = python_projected_shed(env, player, action);
    int planned = 0;
    for (const auto& order : action.market)
      if (order.op == fastkag::Op::SELL && int(order.item) == product)
        planned += std::max(0, order.quantity);
    const int extra = std::min(
        credit, std::max(0, stock[product] - planned));
    if (extra <= 0) return;
    for (auto& order : action.market) {
      if (order.op != fastkag::Op::SELL || int(order.item) != product)
        continue;
      order.quantity += extra;
      credit -= extra;
      return;
    }
    if (action.market.size() >= 10) return;
    action.market.insert(action.market.begin(),
                         {fastkag::Op::SELL,
                          fastkag::Item(product), extra});
    credit -= extra;
  }

  static void apply_capharv_outer(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& state) {
    apply_capharv(env, player, tape, action, state);
    for (int product = 0; product < fastkag::N_PRODUCTS; ++product)
      sell_outer_credit(
          env, player, action, product, state.capharv_credit[product]);
  }

  static void apply_herd2_exact(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& prefix, Herd2State& state) {
    const int step = env.step_count();
    if (step == 0 || step <= state.last_step) state = {};
    state.last_step = step;

    const int board = env.config().board_size;
    if (prefix.herd2_mode != fastkag::Item::NONE) {
      std::vector<Herd2Pending> keep;
      for (const auto& pending : state.pending) {
        const auto& value = tile(env, player, pending.site);
        if (value.kind == fastkag::TileKind::ANIMAL &&
            value.animal == prefix.herd2_mode &&
            int(value.placed_day) == pending.day) {
          const int index = int(pending.site.y) * board + int(pending.site.x);
          state.sites[index] = pending.day;
        } else if (env.day() <= pending.day + 1) {
          keep.push_back(pending);
        }
      }
      state.pending = std::move(keep);
    }

    // Keep the exact parent commands: HERD2 records only substitutions that
    // originated as goose actions, while the shared primitive mutates them in
    // place to the selected species.
    const auto parent = action;
    fastkag::apply_thomas_herd2(env, player, tape, action, prefix);
    if (prefix.herd2_mode == fastkag::Item::NONE) return;

    const auto positions = fastkag::positions(env, player);
    for (std::size_t actor = 0;
         actor < parent.units.size() && actor < positions.size(); ++actor) {
      const auto& command = parent.units[actor];
      const auto position = positions[actor];
      if (command.op == fastkag::Op::PLACE &&
          command.item == fastkag::Item::GOOSE) {
        state.pending.push_back({position, env.day()});
      } else if (command.op == fastkag::Op::HARVEST) {
        const int index = int(position.y) * board + int(position.x);
        if (state.sites.find(index) == state.sites.end()) continue;
        const auto& value = tile(env, player, position);
        if (value.kind == fastkag::TileKind::ANIMAL &&
            value.animal == prefix.herd2_mode)
          state.credit += std::max(0, int(value.yield_units));
      }
    }

    if (state.credit <= 0) return;
    const int product = prefix.herd2_mode == fastkag::Item::COW
        ? int(fastkag::Item::MILK) : int(fastkag::Item::WOOL);
    const auto stock = python_projected_shed(env, player, action);
    int planned = 0;
    for (const auto& order : action.market)
      if (order.op == fastkag::Op::SELL && int(order.item) == product)
        planned += std::max(0, order.quantity);
    int extra = std::min(
        state.credit, std::max(0, stock[product] - planned));
    if (extra <= 0) return;
    for (auto& order : action.market) {
      if (order.op != fastkag::Op::SELL || int(order.item) != product)
        continue;
      order.quantity += extra;
      state.credit -= extra;
      return;
    }
    if (action.market.size() < 10) {
      action.market.insert(
          action.market.begin(),
          {fastkag::Op::SELL, fastkag::Item(product), extra});
      state.credit -= extra;
    }
  }

  static void apply_herd_outer(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& tape,
      fastkag::PlayerAction& action,
      fastkag::ThomasPrefixMarketState& state, Herd2State& herd2) {
    apply_herd2_exact(env, player, tape, action, state, herd2);
    fastkag::apply_thomas_cowswap(env, player, tape, action, state);
    fastkag::observe_thomas_cowswap_harvest(env, player, action, state);
    sell_outer_credit(env, player, action, int(fastkag::Item::EGG),
                      state.cowswap_credit, true);
  }

  void apply_shedroom_thomas(
      const fastkag::Simulator& env, int player,
      const std::vector<fastkag::PlayerAction>& current_tape,
      fastkag::PlayerAction& action) {
    const int step = env.step_count(), hour = env.hour();
    // Thomas' terminal SHEDROOM wrapper predates MetaV4's more aggressive
    // hour-21/+8 variant.  Keep these as profile constants, not observations
    // learned from a parity fixture.
    if ((hour != 22 && hour != 23) || step >= 717) return;
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
    int overflow = night_shed + carried - env.config().shed_capacity + 4;
    if (overflow <= 0) return;

    std::array<int, fastkag::N_PRODUCTS> need{};
    const int stop = std::min(719, step + 25);
    for (int future = step + 1; future < stop; ++future) {
      // Python's _sr_tape forces route 2 for every future frame at/after 648,
      // even when this 24-step lookahead crosses the route boundary.
      const auto& tape = future >= 648
          ? assets.library.routes[slot_for(2)] : current_tape;
      if (future >= int(tape.size())) continue;
      for (const auto& command : tape[future].units) {
        if (command.op == fastkag::Op::FEED)
          ++need[int(fastkag::Item::WHEAT)];
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

  static void apply_v28_room_guard(const fastkag::Simulator& env, int player,
                                   fastkag::PlayerAction& action) {
    if (env.hour() != 23) return;
    const auto& private_state = env.privates()[player];
    int carried = 0;
    for (const auto& inventory : private_state.inventories)
      for (const int quantity : inventory)
        carried += std::max(0, quantity);
    int needed = count_item(private_state.shed) + carried - 99;
    if (needed <= 0) return;
    std::array<int, fastkag::N_PRODUCTS> planned{};
    for (const auto& order : action.market)
      if (order.op == fastkag::Op::SELL && int(order.item) >= 0 &&
          int(order.item) < fastkag::N_PRODUCTS)
        planned[int(order.item)] += std::max(0, order.quantity);
    std::array<int, fastkag::N_PRODUCTS> products{};
    std::iota(products.begin(), products.end(), 0);
    std::stable_sort(products.begin(), products.end(), [&](int left, int right) {
      return env.market().prices[left] > env.market().prices[right];
    });
    for (const int item : products) {
      const int quantity = std::min(
          needed, std::max(0, private_state.shed[item] - planned[item]));
      if (quantity <= 0) continue;
      if (action.market.size() >= 10) break;
      action.market.push_back(
          {fastkag::Op::SELL, fastkag::Item(item), quantity});
      needed -= quantity;
      if (needed <= 0) break;
    }
  }

  fastkag::PlayerAction action(const fastkag::Simulator& env, int player) {
    if (player < 0 || player > 1)
      throw std::invalid_argument("player must be 0 or 1");
    auto& state = seats[player];
    const int step = env.step_count();
    const auto debug = [&](const char* stage,
                           const fastkag::PlayerAction& value) {
      const char* requested = std::getenv("THOMAS_DEBUG_STEP");
      if (!requested || step != std::atoi(requested)) return;
      std::fprintf(stderr, "thomas-%s:", stage);
      for (std::size_t actor = 0; actor < value.units.size(); ++actor)
        std::fprintf(stderr, " u%zu=(%d,%d,%d)", actor,
                     int(value.units[actor].op), int(value.units[actor].item),
                     value.units[actor].quantity);
      std::fprintf(stderr, " market:");
      for (const auto& order : value.market)
        std::fprintf(stderr, " (%d,%d,%d)", int(order.op), int(order.item),
                     order.quantity);
      std::fprintf(stderr, "\n");
    };
    if (step == 0 && state.last_step >= 0) {
      state = {};
      chassis[player] = {};
    }
    if (step < state.last_step)
      throw std::logic_error("Thomas opponent observed a non-monotonic episode");
    update_route(env, player, state);
    const int slot = slot_for(state.route_id);
    auto result = executor.action_external(
        env, player, slot, state.native,
        fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true);
    apply_exact_chassis_units(
        env, player, assets.library.routes[slot], result, chassis[player]);
    // The shared executor always enables its generic late-capacity market
    // layer.  Thomas' Chassis explicitly freezes room_guard=false and wraps it
    // with the older v28 guard below, whose state equation is different.
    // Restore the raw parent market at this exact wrapper boundary; every
    // Thomas market overlay is then replayed in source order below.
    result.market = assets.library.routes[slot][step].market;
    // Both sale-debt suppressors and RACEPX's one-step lead are Chassis hooks,
    // despite being installed by wrappers later in the source file.  They act
    // on the raw tape before EXP154/V219 or any other outer sale is appended.
    // Delaying either suppressor lets a second debt consume an unrelated outer
    // sale after the native tape slot has already been reduced to zero.
    suppress_inner_tape_sales(
        result, state.racepx_debts[step], assets.library.routes[slot][step]);
    suppress_inner_tape_sales(
        result, state.sale_debts[step], assets.library.routes[slot][step]);
    apply_racepx_lead(env, player, assets.library.routes[slot], result, state);
    debug("executor", result);
    // SHOP's terminal liquidation is an early wrapper.  Keeping it here lets
    // R37/ORDERPRI2 and every later market wrapper see the real quantities.
    apply_terminal_rescue(env, player, result);
    apply_v28_room_guard(env, player, result);
    debug("terminal-rescue", result);
    // V219 is the first policy wrapper outside Chassis.  Keep it ahead of
    // V224/V231/R36 so its hires, virtual workers and tomato cash commitment
    // participate in every downstream feasibility check.
    apply_v219_thomas(
        env, player, assets.library.routes[slot], result, state.tomato);
    debug("v219", result);
    debug("racepx", result);
    // The inner V224 wrapper drops zero placeholders before R36 evaluates its
    // blocked-item set.  Reproduce that boundary after causal suppression.
    if (step >= 144) v224_sales_first(result);
    const int base_fertilizer_sale =
        sale_quantity(result, fastkag::Item::FERTILIZER);
    static const std::vector<std::vector<
        fastkag::NativeTapeLibrary::ThomasPredictEvent>> empty_predictors;
    const auto& predictors = env.shops().size() >= 2
        ? assets.library.thomas_predict_pairs[
              int(env.shops()[0]) * 8 + int(env.shops()[1])]
        : empty_predictors;
    apply_prefix_and_r36(
        env, player, assets.library.routes[slot], result, state);
    debug("prefix", result);
    if (env.market().prices[int(fastkag::Item::FERTILIZER)] <= 100)
      cap_sale_quantity(result, fastkag::Item::FERTILIZER,
                        base_fertilizer_sale);
    debug("r36", result);
    if (step >= 288) {
      v224_sales_first(result);
      r37_reorder_sales(env, player, result);
    }
    debug("r37", result);
    result = apply_sheep(env, player, assets.library.routes[slot],
                         std::move(result), state.sheep);
    debug("sheep", result);
    apply_input51(env, player, assets.library.routes[slot], result, state,
                  input[player], chassis[player]);
    debug("input51", result);
    apply_r51_close_warehouse(env, player, state.route_id, result);
    debug("warehouse51", result);
    // R85 wraps the sheep worker layer in Thomas.  It can turn uneconomic
    // native FEED commands into PASS, but deliberately considers only the
    // tape's native worker range (not V233's newly hired sheep workers).
    apply_r85_feed(env, player, assets.library.routes[slot], result);
    apply_r85_fertilizer(
        env, player, state.route_id, result, state, input[player]);
    apply_r51_close_warehouse(env, player, state.route_id, result);
    debug("r85-feed", result);
    // R95, R97 and COURIER are outer wrappers of the sheep/input/R85 chain in
    // the public file.  They must inspect those wrappers' final resource and
    // worker actions; applying them inside the earlier R36 bundle changes both
    // wheat protection and delivered-sale attribution.
    fastkag::apply_thomas_wheat_replenishment_trim(
        env, player, assets.library.routes[slot], result);
    debug("r95-trim", result);
    fastkag::apply_thomas_supply_guard(
        env, player, assets.library.routes[slot], result);
    debug("r97-supply", result);
    fastkag::apply_thomas_courier(
        env, player, assets.library.routes[slot], result, state.prefix);
    debug("courier", result);
    // After R95/R97/COURIER, V9 CARROT and V9 HERD each keep cross-turn state:
    // the former attributes its own crop substitutions and the latter freezes
    // the species choice at the first goose purchase.
    apply_v9_carrot(env, player, result, v9_carrot[player]);
    debug("v9-carrot", result);
    apply_v9_herd(
        env, player, assets.library.routes[slot], result, v9_herd[player]);
    debug("v9-herd", result);
    if (env.day() >= 16)
      apply_v9_fertilizer(env, player, assets.library.routes[slot], result);
    debug("fert", result);
    // PREDICT wraps V9 HERD/FERT in Python.  Besides changing this turn's
    // action it stores our actually visible premium sales for the next public
    // inventory-delta update, so running it inside the earlier R36 bundle
    // corrupts active-rival belief whenever HERD removes an EGG sale.
    fastkag::apply_thomas_predict(
        env, player, assets.library.routes[slot], predictors, result,
        state.prefix);
    debug("predict", result);
    // OVERFLOW is outside the generic hour-23 room guard.  Reapply its exact
    // post-market/dawn invariant after removing the shared guard's excess so
    // legitimate outer sales (notably route 9 step 575) are preserved.
    apply_overflow(env, player, result);
    debug("overflow", result);
    apply_carrot2(
        env, player, assets.library.routes[slot], result, carrot[player]);
    debug("carrot2", result);
    // Continue through Thomas' outer wrapper chain. ORDERPRI2 owns a public
    // cross-turn belief of rival harvests/sales, so the passive-fixture
    // shortcut is not valid when this opponent is used for RL rollouts.
    // Thomas freezes a wider replacement hysteresis than MetaV4 (50 vs 20).
    // Keep the common OR2 state machine, but pass the source-profile constant.
    apply_or2(env, player, assets.library.routes[slot], result, state, 50.0);
    debug("or2", result);
    // CAPHARV is outside ORDERPRI2 in the frozen Python chain.  Its credited
    // sale is therefore inserted after ordering, rather than being sorted as
    // if it had existed in the inner action.
    apply_capharv_outer(
        env, player, assets.library.routes[slot], result, state.prefix);
    debug("capharv", result);
    if (const char* requested = std::getenv("THOMAS_DEBUG_STEP");
        requested && step == std::atoi(requested)) {
      std::fprintf(stderr, "thomas-capharv-credit:");
      for (int product = 0; product < fastkag::N_PRODUCTS; ++product)
        std::fprintf(stderr, " %d", state.prefix.capharv_credit[product]);
      std::fprintf(stderr, "\n");
    }
    apply_shedroom_thomas(env, player, assets.library.routes[slot], result);
    debug("shedroom", result);
    // HERD2 and COWSWAP are the two outermost policy wrappers.
    apply_herd_outer(
        env, player, assets.library.routes[slot], result, state.prefix,
        herd2[player]);
    debug("herd", result);
    if (const char* requested = std::getenv("THOMAS_DEBUG_STEP");
        requested && step == std::atoi(requested)) {
      std::fprintf(stderr,
                   "thomas-herd-state: herd=%d cowswap=%d broken=%d credit=%d\n",
                   int(state.prefix.herd2_mode), state.prefix.cowswap_active,
                   state.prefix.cowswap_broken, state.prefix.cowswap_credit);
    }
    state.last_step = step;
    return result;
  }

  void reset() {
    Base::reset();
    chassis = {};
    carrot = {};
    input = {};
    v9_carrot = {};
    v9_herd = {};
    herd2 = {};
    cattle = {};
  }

  std::array<ThomasChassisState, 2> chassis{};
  std::array<CarrotState, 2> carrot{};
  std::array<InputState, 2> input{};
  std::array<V9CarrotState, 2> v9_carrot{};
  std::array<V9HerdState, 2> v9_herd{};
  std::array<Herd2State, 2> herd2{};
  std::array<CattleState, 2> cattle{};
  std::unordered_map<int, std::array<int, 720>> r85_native_reserves;
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
  if (player < 0 || player > 1)
    throw std::invalid_argument("player must be 0 or 1");
  return impl_->seats[player].route_id;
}
void Opponent::reset() { impl_->reset(); }

}  // namespace thomas_2945
