#include "static_dependencies.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <map>

namespace g001::failure_debt {
namespace {

struct Token {
  std::size_t dependency{};
};

bool crop_item(fastkag::Item item) {
  const int value = static_cast<int>(item);
  return value >= 0 && value < fastkag::N_CROPS;
}

bool animal_item(fastkag::Item item) {
  const int value = static_cast<int>(item);
  return value >= 9 && value < 12;
}

}  // namespace

std::vector<StaticPurchaseDependency> extract_static_dependencies(
    const std::vector<fastkag::PlayerAction>& tape) {
  std::vector<StaticPurchaseDependency> result;
  std::array<std::deque<Token>, fastkag::N_CROPS> seeds;
  std::array<std::deque<Token>, fastkag::N_ANIMALS> animals;
  std::map<std::pair<int, int>, std::deque<Token>> carried;

  for (int step = 0; step < static_cast<int>(tape.size()); ++step) {
    const auto& turn = tape[static_cast<std::size_t>(step)];
    // Unit phase precedes market phase in the official environment.
    for (int actor = 0; actor < static_cast<int>(turn.units.size()); ++actor) {
      const auto& action = turn.units[static_cast<std::size_t>(actor)];
      const int item = static_cast<int>(action.item);
      if (action.op == fastkag::Op::PLANT && crop_item(action.item)) {
        auto& queue = seeds[static_cast<std::size_t>(item)];
        if (!queue.empty()) {
          const auto token = queue.front();
          queue.pop_front();
          auto& dependency = result[token.dependency];
          dependency.first_unit_step = step;
          dependency.first_actor = actor;
          dependency.terminal_unit_step = step;
          dependency.terminal_actor = actor;
          dependency.complete = true;
          dependency.reason = "BUY_SEED->PLANT";
        }
      } else if (action.op == fastkag::Op::PICKUP && animal_item(action.item)) {
        auto& queue = animals[static_cast<std::size_t>(item - 9)];
        int quantity = std::max(0, int(action.quantity));
        while (quantity-- > 0 && !queue.empty()) {
          const auto token = queue.front();
          queue.pop_front();
          auto& dependency = result[token.dependency];
          dependency.first_unit_step = step;
          dependency.first_actor = actor;
          dependency.reason = "BUY_ANIMAL->PICKUP; awaiting PLACE";
          carried[{actor, item}].push_back(token);
        }
      } else if (action.op == fastkag::Op::PLACE && animal_item(action.item)) {
        auto& queue = carried[{actor, item}];
        if (!queue.empty()) {
          const auto token = queue.front();
          queue.pop_front();
          auto& dependency = result[token.dependency];
          dependency.terminal_unit_step = step;
          dependency.terminal_actor = actor;
          dependency.complete = true;
          dependency.reason = "BUY_ANIMAL->PICKUP->PLACE; runtime care proof required";
        }
      }
    }

    for (int slot = 0; slot < static_cast<int>(turn.market.size()); ++slot) {
      const auto& action = turn.market[static_cast<std::size_t>(slot)];
      if ((action.op != fastkag::Op::BUY_SEED || !crop_item(action.item)) &&
          (action.op != fastkag::Op::BUY_ANIMAL || !animal_item(action.item))) {
        continue;
      }
      for (int unit = 0; unit < std::max(0, int(action.quantity)); ++unit) {
        const auto dependency_index = result.size();
        result.push_back({action.op, action.item, step, slot, unit,
                          -1, -1, -1, -1, false,
                          action.op == fastkag::Op::BUY_SEED
                              ? "BUY_SEED has no later matched PLANT"
                              : "BUY_ANIMAL has no later matched PICKUP/PLACE"});
        if (action.op == fastkag::Op::BUY_SEED) {
          seeds[static_cast<std::size_t>(static_cast<int>(action.item))]
              .push_back({dependency_index});
        } else {
          animals[static_cast<std::size_t>(static_cast<int>(action.item) - 9)]
              .push_back({dependency_index});
        }
      }
    }
  }
  return result;
}

}  // namespace g001::failure_debt
