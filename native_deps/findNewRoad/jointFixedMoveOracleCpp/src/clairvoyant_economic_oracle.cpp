#include "clairvoyant_economic_oracle.hpp"

#include <algorithm>
#include <limits>

namespace joint_fixed_move_oracle::economic {

std::array<int, fastkag::N_PRODUCTS> demand_at_step(
    const std::vector<std::int8_t>& shops, int step,
    int town_shop_sell_interval, int town_center_sell_interval) {
  std::array<int, fastkag::N_PRODUCTS> out{};
  auto take = [&](int product, int units = 1) { out[product] += units; };
  if (step % std::max(1, town_shop_sell_interval) == 0) {
    for (const int shop : shops) {
      switch (shop) {
        case 0: take(5); take(0); break;
        case 1: take(5); take(0); take(3); break;
        case 2: take(0); take(1); take(2); take(3); break;
        case 3: take(3); take(6); take(0); break;
        case 4: take(1, 2); break;
        case 5: take(6); take(2); take(0); break;
        case 6: take(3); take(6); break;
        case 7: take(7, 2); break;
        default: break;
      }
    }
  }
  if (step % std::max(1, town_center_sell_interval) == 0)
    for (int product = 0; product < 8; ++product) ++out[product];
  return out;
}

FutureDemand build_future_demand(
    const std::vector<ReferenceFrame>& realized_future,
    int town_shop_sell_interval, int town_center_sell_interval) {
  FutureDemand out;
  out.suffix_units.resize(realized_future.size() + 1);
  out.suffix_peak_price.resize(realized_future.size() + 1);
  out.suffix_peak_price.back().fill(1);
  for (std::size_t reverse = realized_future.size(); reverse > 0; --reverse) {
    const std::size_t step = reverse - 1;
    out.suffix_units[step] = out.suffix_units[step + 1];
    out.suffix_peak_price[step] = out.suffix_peak_price[step + 1];
    const auto demand = demand_at_step(realized_future[step].shops,
                                       static_cast<int>(step),
                                       town_shop_sell_interval,
                                       town_center_sell_interval);
    for (int product = 0; product < fastkag::N_PRODUCTS; ++product) {
      out.suffix_units[step][product] += demand[product];
      out.suffix_peak_price[step][product] = std::max(
          out.suffix_peak_price[step][product],
          realized_future[step].prices[product]);
    }
  }
  return out;
}

namespace {
std::uint32_t mix(std::uint32_t value) noexcept {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}
}  // namespace

fastkag::Item target_for_tile(const CropPolicy& policy, int tile) noexcept {
  const int bucket = static_cast<int>(mix(static_cast<std::uint32_t>(tile) +
                                          0x9e3779b9U) % 100U);
  return bucket < std::clamp(policy.primary_share_percent, 0, 100)
             ? policy.primary
             : policy.secondary;
}

bool tile_selected(const CropPolicy& policy, int tile) noexcept {
  const std::uint32_t salt = static_cast<std::uint32_t>(policy.tile_selector_salt);
  const int bucket = static_cast<int>(mix(static_cast<std::uint32_t>(tile) +
                                          0x85ebca6bU + salt * 0x9e3779b9U) %
                                      100U);
  return bucket < std::clamp(policy.selected_tile_percent, 0, 100);
}

int sell_quantity(SellPace pace, int own_stock,
                  int opponent_stock) noexcept {
  own_stock = std::max(0, own_stock);
  opponent_stock = std::max(0, opponent_stock);
  switch (pace) {
    case SellPace::All: return own_stock;
    case SellPace::Half: return (own_stock + 1) / 2;
    case SellPace::Cap1: return std::min(own_stock, 1);
    case SellPace::Cap2: return std::min(own_stock, 2);
    case SellPace::Cap4: return std::min(own_stock, 4);
    case SellPace::Cap8: return std::min(own_stock, 8);
    case SellPace::OpponentPressure:
      return std::min(own_stock, std::max(1, opponent_stock));
  }
  return 0;
}

const char* sell_pace_name(SellPace pace) noexcept {
  switch (pace) {
    case SellPace::All: return "all";
    case SellPace::Half: return "half";
    case SellPace::Cap1: return "cap1";
    case SellPace::Cap2: return "cap2";
    case SellPace::Cap4: return "cap4";
    case SellPace::Cap8: return "cap8";
    case SellPace::OpponentPressure: return "opponent-pressure";
  }
  return "unknown";
}

const char* baseline_sell_mode_name(BaselineSellMode mode) noexcept {
  switch (mode) {
    case BaselineSellMode::Replace: return "replace";
    case BaselineSellMode::Gate: return "gate";
    case BaselineSellMode::Augment: return "augment";
  }
  return "unknown";
}

}  // namespace joint_fixed_move_oracle::economic
