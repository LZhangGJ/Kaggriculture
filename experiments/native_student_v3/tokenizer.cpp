#include "tokenizer.hpp"

#include "../../fast_kaggriculture/src/simulator.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace student_v3 {
namespace {

constexpr std::uint32_t kContinuous = 24;
constexpr std::uint32_t kCategories = 7;
constexpr std::uint32_t kPositionIds = 64;
constexpr std::uint32_t kOwnerSelf = 1;
constexpr std::uint32_t kOwnerOpponent = 2;
constexpr std::uint32_t kOwnerShared = 3;
constexpr std::uint32_t kTokenGlobal = 1;
constexpr std::uint32_t kTokenFarm = 2;
constexpr std::uint32_t kTokenUnit = 3;
constexpr std::uint32_t kTokenTile = 4;
constexpr std::uint32_t kTokenMarket = 5;
constexpr std::uint32_t kTokenTown = 6;

float ratio(double value, double scale) {
  return static_cast<float>(std::clamp(value / scale, -10.0, 10.0));
}

std::uint32_t position_id(int value) {
  return std::min<std::uint32_t>(kPositionIds - 1,
                                 std::max(1, value + 1));
}

struct Builder {
  explicit Builder(std::uint32_t capacity)
      : capacity(capacity), continuous(std::size_t(capacity) * kContinuous),
        categories(std::size_t(kCategories) * capacity) {}

  void add(std::uint32_t kind, std::initializer_list<float> values = {},
           std::uint32_t a = 0, std::uint32_t b = 0,
           std::uint32_t c = 0, std::uint32_t x = 0,
           std::uint32_t y = 0,
           std::uint32_t owner = kOwnerShared) {
    if (count >= capacity) throw std::runtime_error("native token capacity");
    std::size_t column = 0;
    for (const float value : values) {
      if (column == kContinuous) break;
      continuous[std::size_t(count) * kContinuous + column++] = value;
    }
    const std::array<std::uint32_t, kCategories> values_by_category{
        kind, a, b, c, x, y, owner};
    for (std::uint32_t category = 0; category < kCategories; ++category)
      categories[std::size_t(category) * capacity + count] =
          values_by_category[category];
    ++count;
  }

  std::uint32_t capacity;
  std::vector<float> continuous;
  std::vector<std::uint32_t> categories;
  std::uint32_t count{};
};

}  // namespace

EncodedTokens tokenize(const fastkag::Simulator& env, int seat,
                       std::uint32_t capacity) {
  if (seat < 0 || seat > 1 || !capacity)
    throw std::invalid_argument("native tokenizer seat/capacity");
  Builder output(capacity);
  const int opponent = 1 - seat;
  const auto& own_farm = env.farms()[seat];
  const auto& rival_farm = env.farms()[opponent];
  const auto& private_state = env.privates()[seat];
  const double step = env.step_count();
  const double day = env.day();
  const double hour = env.hour();
  const double own_money = own_farm.money;
  const double rival_money = rival_farm.money;
  output.add(kTokenGlobal, {
      ratio(day, 30), ratio(hour, 24), ratio(own_money, 10000),
      ratio(rival_money, 10000), ratio(2, 2),
      ratio(env.shops().size(), 8),
      static_cast<float>(std::clamp(step / 720.0, 0.0, 1.0)),
      static_cast<float>(std::clamp((719.0 - step) / 720.0, 0.0, 1.0))});

  for (int logical = 0; logical < 2; ++logical) {
    const bool self = logical == 0;
    const auto& farm = env.farms()[self ? seat : opponent];
    const std::uint32_t owner = self ? kOwnerSelf : kOwnerOpponent;
    const int unlocked = std::popcount(unsigned(farm.unlocked_mask));
    const int empty = std::count_if(
        farm.tiles.begin(), farm.tiles.end(), [](const auto& tile) {
          return tile.kind == fastkag::TileKind::EMPTY;
        });
    output.add(kTokenFarm, {
        ratio(farm.money, 10000), ratio(farm.hands.size(), 10),
        ratio(unlocked, 4), ratio(farm.hires_today, 10), ratio(empty, 100)},
        std::min(31, unlocked + 1), 0, 0, 0, 0, owner);

    auto add_unit = [&](std::size_t unit, fastkag::Position position) {
      const std::array<int32_t, fastkag::N_ITEMS>* inventory = nullptr;
      if (self && unit < private_state.inventories.size())
        inventory = &private_state.inventories[unit];
      int total = 0;
      if (inventory)
        for (const int value : *inventory) total += value;
      std::array<float, 14> values{};
      values[0] = unit == 0 ? 1.0f : 0.0f;
      values[1] = ratio(unit, 10);
      values[2] = ratio(total, 100);
      for (int item = 1; item < fastkag::N_ITEMS; ++item)
        values[2 + item] = ratio(inventory ? (*inventory)[item] : 0, 20);
      if (output.count >= output.capacity)
        throw std::runtime_error("native token capacity");
      const auto index = output.count;
      output.add(kTokenUnit, {}, unit == 0 ? 1 : 2, 0, 0,
                 position_id(position.x), position_id(position.y), owner);
      std::copy(values.begin(), values.end(),
                output.continuous.begin() + std::size_t(index) * kContinuous);
    };
    add_unit(0, farm.farmer);
    for (std::size_t hand = 0; hand < farm.hands.size(); ++hand)
      add_unit(hand + 1, farm.hands[hand]);

    if (farm.tiles.size() != 100)
      throw std::runtime_error("native tokenizer board width");
    for (int y = 0; y < 10; ++y) {
      for (int x = 0; x < 10; ++x) {
        const auto& tile = farm.tiles[y * 10 + x];
        std::uint32_t kind = 1, item = 0;
        std::array<float, 7> values{};
        std::size_t value_count = 0;
        switch (tile.kind) {
          case fastkag::TileKind::LOCKED:
            kind = 2;
            break;
          case fastkag::TileKind::WEED:
            kind = 3;
            break;
          case fastkag::TileKind::PLANT:
            kind = 4;
            item = std::uint32_t(int(tile.crop) + 1);
            values = {ratio(day - tile.planted_day, 30),
                      ratio(tile.yield_units, 6),
                      tile.watered_today ? 1.0f : 0.0f,
                      ratio(tile.consecutive_unwatered, 2),
                      ratio(tile.fertilized_until_day - day, 3), 0, 0};
            value_count = 5;
            break;
          case fastkag::TileKind::COOP:
            kind = 5;
            break;
          case fastkag::TileKind::PASTURE:
            kind = 6;
            break;
          case fastkag::TileKind::ANIMAL:
            kind = int(tile.animal) == 9 ? 5 : 6;
            item = std::uint32_t(int(tile.animal) + 1);
            values = {ratio(day - tile.placed_day, 30),
                      ratio(tile.yield_units, 6),
                      tile.fed_today ? 1.0f : 0.0f,
                      ratio(tile.consecutive_unfed, 2),
                      tile.cared_today ? 1.0f : 0.0f,
                      tile.fertilizer_available ? 1.0f : 0.0f,
                      ratio(tile.pending_care_bonus, 6)};
            value_count = 7;
            break;
          case fastkag::TileKind::EMPTY:
            break;
        }
        if (output.count >= output.capacity)
          throw std::runtime_error("native token capacity");
        const auto index = output.count;
        output.add(kTokenTile, {}, kind, item,
                   tile.kind == fastkag::TileKind::LOCKED ? 2 : 1,
                   position_id(x), position_id(y), owner);
        std::copy_n(values.begin(), value_count,
                    output.continuous.begin() +
                        std::size_t(index) * kContinuous);
      }
    }
  }

  for (int item = 0; item < fastkag::N_PRODUCTS; ++item) {
    output.add(kTokenMarket, {
        ratio(env.market().inventory[item], 10000),
        ratio(env.market().prices[item], 300),
        ratio(private_state.shed[item], 100),
        ratio(item < fastkag::N_CROPS ? private_state.seeds[item] : 0, 100),
        ratio(own_money, 10000)}, std::uint32_t(item + 1));
  }
  std::array<int, 8> shop_counts{};
  for (const int shop : env.shops())
    if (shop >= 0 && shop < int(shop_counts.size())) ++shop_counts[shop];
  for (int shop = 0; shop < int(shop_counts.size()); ++shop)
    output.add(kTokenTown, {ratio(shop_counts[shop], 8)}, shop + 1);

#ifdef STUDENT_EXPLICIT_SHOP_TOKENS
#ifndef SHOP_TOKEN_GAIN
#define SHOP_TOKEN_GAIN 1
#endif
  // The old mean pool loses the association between shop ID and count:
  // all eight IDs are always present, while their counts sum to a constant.
  // These previously unused continuous slots survive that mean exactly.
  for (int shop = 0; shop < int(shop_counts.size()); ++shop)
    output.continuous[8 + shop] =
        ratio(shop_counts[shop], 8) * output.count * SHOP_TOKEN_GAIN;
#endif

  return {std::move(output.continuous), std::move(output.categories),
          output.count};
}

}  // namespace student_v3
