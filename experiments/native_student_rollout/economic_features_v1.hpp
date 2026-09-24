#pragma once

#include "../../fast_kaggriculture/src/simulator.hpp"

#include <array>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace economic_v1 {

constexpr int kObservationExtra = 71;
constexpr int kResourceExtra = 27;
constexpr int kShopResourceExtra = 9;
constexpr int kOldObservation = 3074;
constexpr int kOldResource = 347;
#ifdef ECONOMIC_FEATURES_MATURE_STORED
constexpr int kSemantics = 2;
#else
constexpr int kSemantics = 1;
#endif
constexpr std::array<int, 3> kHorizonDays{2, 6, 30};
constexpr std::array<std::array<int, 9>, 8> kShopDemand{{
    {{1, 0, 0, 0, 0, 1, 0, 0, 0}},
    {{1, 0, 0, 1, 0, 1, 0, 0, 0}},
    {{1, 1, 1, 1, 0, 0, 0, 0, 0}},
    {{1, 0, 0, 1, 0, 0, 1, 0, 0}},
    {{0, 2, 0, 0, 0, 0, 0, 0, 0}},
    {{1, 0, 1, 0, 0, 0, 1, 0, 0}},
    {{0, 0, 0, 1, 0, 0, 1, 0, 0}},
    {{0, 0, 0, 0, 0, 0, 0, 2, 0}},
}};
constexpr std::array<int, 5> kFirstCrop{2, 2, 8, 10, 10};
constexpr std::array<int, 3> kFirstAnimal{4, 8, 6};

inline std::array<float, kObservationExtra> observation(
    const fastkag::Simulator& env, int seat) {
  if (seat < 0 || seat > 1) throw std::invalid_argument("economic seat");
  std::array<float, kObservationExtra> out{};
  std::array<int, 8> counts{};
  for (const int shop : env.shops()) {
    if (shop < 0 || shop >= 8) throw std::invalid_argument("economic shop");
    ++counts[shop];
  }
  std::array<int, 9> rate{};
  for (int shop = 0; shop < 8; ++shop) {
    out[shop] = counts[shop] / 8.0f;
    for (int item = 0; item < 9; ++item)
      rate[item] += counts[shop] * kShopDemand[shop][item];
  }
  struct Asset { int product, first; float held; };
  std::vector<Asset> assets;
  std::array<float, 9> stored{};
  for (const auto& tile : env.farms()[1 - seat].tiles) {
    int product = -1, first = 0;
    if (tile.kind == fastkag::TileKind::PLANT &&
        int(tile.crop) >= 0 && int(tile.crop) < 5) {
      product = int(tile.crop);
      first = tile.planted_day + kFirstCrop[product];
    } else if (tile.kind == fastkag::TileKind::ANIMAL &&
               int(tile.animal) >= 9 && int(tile.animal) < 12) {
      product = int(tile.animal) - 4;
      first = tile.placed_day + kFirstAnimal[int(tile.animal) - 9];
    }
    if (product < 0) continue;
    float held = std::max(0, int(tile.yield_units));
#ifdef ECONOMIC_FEATURES_MATURE_STORED
    if (first > env.day()) held = 0;
#endif
    stored[product] += held;
    assets.push_back({product, first, held});
  }
  const int step = env.step_count();
  for (int h = 0; h < 3; ++h) {
    const int end = std::min(720, step + 24 * kHorizonDays[h]);
    int town_ticks = 0, center_ticks = 0;
    for (int tick = step; tick < end; ++tick) {
      town_ticks += tick % 4 == 0;
      center_ticks += tick % 24 == 0;
    }
    for (int item = 0; item < 9; ++item) {
      const int demand = rate[item] * town_ticks +
                         (item < 8 ? center_ticks : 0);
      out[8 + h * 9 + item] =
          (env.market().inventory[item] - 10000 - demand) / 500.0f;
    }
    for (const auto& asset : assets)
      if (asset.first < end / 24 || asset.held > 0)
        out[35 + h * 9 + asset.product] += 1.0f / 25.0f;
  }
  for (int item = 0; item < 9; ++item)
    out[62 + item] = stored[item] / 100.0f;
  return out;
}

inline void prefix_flow(const double* values, int day, float* output) {
  if (!values || !output || day < 0 || day >= 30)
    throw std::invalid_argument("economic prefix flow");
  for (int h = 0; h < 3; ++h)
    for (int item = 0; item < 9; ++item) {
      double sum = 0;
      for (int d = day; d < std::min(30, day + kHorizonDays[h]); ++d)
        sum += values[17 + 11 * d + item];
      output[h * 9 + item] = float(sum / 500.0);
    }
}

inline void shop_rate(const fastkag::Simulator& env, float* output) {
  if (!output) throw std::invalid_argument("economic shop rate output");
  std::array<int, 9> demand{};
  for (const int shop : env.shops()) {
    if (shop < 0 || shop >= 8) throw std::invalid_argument("economic shop");
    for (int item = 0; item < 9; ++item)
      demand[item] += kShopDemand[shop][item];
  }
  for (int item = 0; item < 9; ++item)
    output[item] = demand[item] / 8.0f;
}

}  // namespace economic_v1
