#include "route_loader.hpp"
#include "native_teammate.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::Position;
using fastkag::TileKind;

struct FailedPurchase {
  std::uint64_t seed{};
  int seat{};
  int step{-1};
  int slot{-1};
  Op operation{Op::PASS};
  Item item{Item::NONE};
  int requested{};
  int filled{};
};

struct ConsumerFailure {
  std::uint64_t seed{};
  int seat{};
  int step{-1};
  int actor{-1};
  Op operation{Op::PASS};
  Item item{Item::NONE};
  Position tile{};
};

const fastkag::Tile* tile_at(const fastkag::Simulator& env, int player,
                             Position position) {
  if (position.x < 0 || position.y < 0 ||
      position.x >= env.config().board_size ||
      position.y >= env.config().board_size)
    return nullptr;
  const auto index = static_cast<std::size_t>(position.y) *
                         static_cast<std::size_t>(env.config().board_size) +
                     static_cast<std::size_t>(position.x);
  const auto& tiles = env.farms()[player].tiles;
  return index < tiles.size() ? &tiles[index] : nullptr;
}

Position actor_position(const fastkag::Simulator& env, int player, int actor) {
  if (actor == 0) return env.farms()[player].farmer;
  const auto& hands = env.farms()[player].hands;
  return actor > 0 && static_cast<std::size_t>(actor - 1) < hands.size()
             ? hands[static_cast<std::size_t>(actor - 1)]
             : Position{-1, -1};
}

bool consumer_effect(const fastkag::Simulator& before,
                     const fastkag::Simulator& after, int player, int actor,
                     Action action) {
  const auto position = actor_position(before, player, actor);
  const auto* tile = tile_at(after, player, position);
  if (!tile) return false;
  if (action.op == Op::PLANT)
    return tile->kind == TileKind::PLANT && tile->crop == action.item;
  if (action.op == Op::PLACE)
    return tile->kind == TileKind::ANIMAL && tile->animal == action.item;
  return true;
}

int main_impl(int argc, char** argv) {
  const int seeds = argc > 1 ? std::stoi(argv[1]) : 256;
  const std::uint64_t seed_begin =
      argc > 2 ? std::stoull(argv[2]) : 990000ULL;
  fastkag::NativeTapeLibrary library;
  library.routes = {
      g001::repair::load_route(PURCHASE_AUDIT_G001_TAPES,
                               PURCHASE_AUDIT_G001_LIBRARY, "G001"),
      g001::repair::load_route(PURCHASE_AUDIT_G001_TAPES,
                               PURCHASE_AUDIT_G001_LIBRARY, "G096")};
  fastkag::NativeTeammateExecutor executor(std::move(library));

  int games = 0;
  int purchase_attempts = 0;
  int failed_seed = 0;
  int failed_animal = 0;
  int plant_failures = 0;
  int place_failures = 0;
  int linked_seed = 0;
  int linked_animal = 0;
  std::optional<FailedPurchase> first_purchase;
  std::optional<ConsumerFailure> first_seed_consumer;
  std::optional<ConsumerFailure> first_animal_consumer;
  std::optional<FailedPurchase> first_linked_seed_purchase;
  std::optional<FailedPurchase> first_linked_animal_purchase;
  for (int offset = 0; offset < seeds; ++offset) {
    const auto seed = seed_begin + static_cast<std::uint64_t>(offset);
    for (int seat = 0; seat < 2; ++seat) {
      ++games;
      fastkag::Simulator env({}, seed);
      std::array<fastkag::NativeAgentState, 2> states;
      std::vector<FailedPurchase> open;
      while (!env.done()) {
        std::array<fastkag::PlayerAction, 2> actions;
        actions[seat] = executor.action_external(env, seat, 0, states[seat]);
        actions[1 - seat] =
            executor.action_external(env, 1 - seat, 1, states[1 - seat]);
        const auto before = env;
        const auto unit_after = before.preview_unit_phase(actions);
        env.step(actions);
        const auto& fills = env.last_market_fills()[seat];
        for (std::size_t slot = 0; slot < actions[seat].market.size(); ++slot) {
          const auto order = actions[seat].market[slot];
          if (order.op != Op::BUY_SEED && order.op != Op::BUY_ANIMAL) continue;
          ++purchase_attempts;
          const int filled = slot < fills.size() ? fills[slot] : 0;
          if (filled >= order.quantity) continue;
          FailedPurchase failed{seed, seat, before.step_count(),
                                static_cast<int>(slot), order.op, order.item,
                                order.quantity, filled};
          open.push_back(failed);
          if (order.op == Op::BUY_SEED)
            ++failed_seed;
          else
            ++failed_animal;
          if (!first_purchase) first_purchase = failed;
        }
        for (std::size_t actor = 0; actor < actions[seat].units.size(); ++actor) {
          const auto unit = actions[seat].units[actor];
          if (unit.op != Op::PLANT && unit.op != Op::PLACE) continue;
          if (consumer_effect(before, unit_after, seat,
                              static_cast<int>(actor), unit))
            continue;
          if (unit.op == Op::PLANT)
            ++plant_failures;
          else
            ++place_failures;
          const auto linked = std::find_if(
              open.rbegin(), open.rend(), [&](const auto& failed) {
                const bool kind =
                    (unit.op == Op::PLANT && failed.operation == Op::BUY_SEED) ||
                    (unit.op == Op::PLACE &&
                     failed.operation == Op::BUY_ANIMAL);
                return kind && failed.item == unit.item &&
                       failed.step < before.step_count() &&
                       before.step_count() - failed.step <= 48;
              });
          if (linked == open.rend()) continue;
          if (unit.op == Op::PLANT)
            ++linked_seed;
          else
            ++linked_animal;
          const ConsumerFailure failure{
              seed, seat, before.step_count(), static_cast<int>(actor), unit.op,
              unit.item,
              actor_position(before, seat, static_cast<int>(actor))};
          if (unit.op == Op::PLANT && !first_seed_consumer) {
            first_seed_consumer = failure;
            first_linked_seed_purchase = *linked;
          }
          if (unit.op == Op::PLACE && !first_animal_consumer) {
            first_animal_consumer = failure;
            first_linked_animal_purchase = *linked;
          }
        }
      }
    }
  }

  std::cout << "games=" << games << " purchase_attempts=" << purchase_attempts
            << " failed_seed=" << failed_seed
            << " failed_animal=" << failed_animal
            << " plant_failures=" << plant_failures
            << " place_failures=" << place_failures
            << " linked_seed=" << linked_seed
            << " linked_animal=" << linked_animal << '\n';
  if (first_purchase)
    std::cout << "first_purchase seed=" << first_purchase->seed
              << " seat=" << first_purchase->seat
              << " step=" << first_purchase->step
              << " slot=" << first_purchase->slot
              << " op=" << static_cast<int>(first_purchase->operation)
              << " item=" << static_cast<int>(first_purchase->item)
              << " requested=" << first_purchase->requested
              << " filled=" << first_purchase->filled << '\n';
  const auto print_link = [](const char* label,
                             const std::optional<FailedPurchase>& purchase,
                             const std::optional<ConsumerFailure>& consumer) {
    if (!purchase || !consumer) return;
    std::cout << label << " seed=" << consumer->seed
              << " seat=" << consumer->seat
              << " purchase_step=" << purchase->step
              << " purchase_slot=" << purchase->slot
              << " requested=" << purchase->requested
              << " filled=" << purchase->filled
              << " consumer_step=" << consumer->step
              << " actor=" << consumer->actor
              << " op=" << static_cast<int>(consumer->operation)
              << " item=" << static_cast<int>(consumer->item)
              << " tile=(" << consumer->tile.x << ',' << consumer->tile.y
              << ")\n";
  };
  print_link("first_linked_seed", first_linked_seed_purchase,
             first_seed_consumer);
  print_link("first_linked_animal", first_linked_animal_purchase,
             first_animal_consumer);
  return first_purchase && (first_seed_consumer || first_animal_consumer) ? 0
                                                                         : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "purchase_failure_real_trigger_audit: " << error.what()
              << '\n';
    return 1;
  }
}
