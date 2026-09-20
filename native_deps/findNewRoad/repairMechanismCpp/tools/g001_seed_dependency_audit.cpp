#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fastkag::Item;
using fastkag::Op;

constexpr int kCropCount = fastkag::N_CROPS;

struct SeedUnit {
  int buy_step{};
  int buy_slot{};
  int unit_index{};
};

struct Dependency {
  int buy_step{};
  int buy_slot{};
  int buy_unit{};
  int plant_step{};
  int actor{};
  int crop{};
};

const char* crop_name(int crop) {
  static constexpr std::array<const char*, kCropCount> names{
      "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"};
  return crop >= 0 && crop < kCropCount ? names[static_cast<std::size_t>(crop)]
                                       : "INVALID";
}

void usage(const char* executable) {
  std::cerr << "usage: " << executable
            << " ROUTE_ACTIONS_ZLIB ROUTE_LIBRARY_JSON [FAMILY]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4) {
      usage(argv[0]);
      return 2;
    }
    const std::string family = argc == 4 ? argv[3] : "G001";
    const auto tape = g001::repair::load_route(argv[1], argv[2], family);

    std::array<std::queue<SeedUnit>, kCropCount> available;
    std::array<int, kCropCount> bought{};
    std::array<int, kCropCount> planted{};
    std::array<int, kCropCount> unmatched_plants{};
    std::vector<Dependency> dependencies;

    // Unit actions execute before market actions. A BUY_SEED at step t can
    // therefore satisfy only PLANT actions from t+1 onward.
    for (int step = 0; step < static_cast<int>(tape.size()); ++step) {
      const auto& turn = tape[static_cast<std::size_t>(step)];
      for (int actor = 0; actor < static_cast<int>(turn.units.size()); ++actor) {
        const auto& action = turn.units[static_cast<std::size_t>(actor)];
        if (action.op != Op::PLANT) continue;
        const int crop = static_cast<int>(action.item);
        if (crop < 0 || crop >= kCropCount) continue;
        ++planted[static_cast<std::size_t>(crop)];
        if (available[static_cast<std::size_t>(crop)].empty()) {
          ++unmatched_plants[static_cast<std::size_t>(crop)];
          continue;
        }
        const auto seed = available[static_cast<std::size_t>(crop)].front();
        available[static_cast<std::size_t>(crop)].pop();
        dependencies.push_back({seed.buy_step, seed.buy_slot, seed.unit_index,
                                step, actor, crop});
      }
      for (int slot = 0; slot < static_cast<int>(turn.market.size()); ++slot) {
        const auto& order = turn.market[static_cast<std::size_t>(slot)];
        if (order.op != Op::BUY_SEED) continue;
        const int crop = static_cast<int>(order.item);
        if (crop < 0 || crop >= kCropCount) continue;
        const int quantity = std::max(0, int(order.quantity));
        bought[static_cast<std::size_t>(crop)] += quantity;
        for (int unit = 0; unit < quantity; ++unit) {
          available[static_cast<std::size_t>(crop)].push(
              SeedUnit{step, slot, unit});
        }
      }
    }

    std::cout << "{\"family\":\"" << family << "\",\"steps\":"
              << tape.size() << ",\"summary\":[";
    for (int crop = 0; crop < kCropCount; ++crop) {
      if (crop != 0) std::cout << ',';
      std::cout << "{\"crop\":\"" << crop_name(crop)
                << "\",\"bought\":" << bought[static_cast<std::size_t>(crop)]
                << ",\"planted\":" << planted[static_cast<std::size_t>(crop)]
                << ",\"unmatched_plants\":"
                << unmatched_plants[static_cast<std::size_t>(crop)]
                << ",\"unused_seed_units\":"
                << available[static_cast<std::size_t>(crop)].size() << '}';
    }
    std::cout << "],\"dependencies\":[";
    for (std::size_t index = 0; index < dependencies.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto& row = dependencies[index];
      std::cout << "{\"crop\":\"" << crop_name(row.crop)
                << "\",\"buy_step\":" << row.buy_step
                << ",\"buy_slot\":" << row.buy_slot
                << ",\"buy_unit\":" << row.buy_unit
                << ",\"plant_step\":" << row.plant_step
                << ",\"actor\":" << row.actor
                << ",\"lead_steps\":" << row.plant_step - row.buy_step
                << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "g001_seed_dependency_audit: " << error.what() << '\n';
    return 1;
  }
}
