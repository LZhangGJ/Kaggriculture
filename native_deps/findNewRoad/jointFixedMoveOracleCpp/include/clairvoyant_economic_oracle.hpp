#pragma once

#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace joint_fixed_move_oracle::economic {

// This API deliberately consumes a realized reference future.  It is an
// offline oracle feature and must not be linked into a submitted agent.
struct ReferenceFrame {
  std::array<int, fastkag::N_PRODUCTS> prices{};
  std::vector<std::int8_t> shops;
};

struct FutureDemand {
  std::vector<std::array<int, fastkag::N_PRODUCTS>> suffix_units;
  std::vector<std::array<int, fastkag::N_PRODUCTS>> suffix_peak_price;
};

enum class SellPace : std::uint8_t {
  All,
  Half,
  Cap1,
  Cap2,
  Cap4,
  Cap8,
  OpponentPressure,
};

enum class BaselineSellMode : std::uint8_t {
  Replace,
  Gate,
  Augment,
};

struct TradePolicy {
  int cutoff_step{220};
  int future_peak_ratio_bps{9000};
  int terminal_steps{24};
  SellPace pace{SellPace::Cap8};
  BaselineSellMode baseline_mode{BaselineSellMode::Replace};
};

struct CropPolicy {
  bool enabled{true};
  int cutoff_step{220};
  fastkag::Item primary{fastkag::Item::STRAWBERRY};
  fastkag::Item secondary{fastkag::Item::STRAWBERRY};
  int primary_share_percent{100};
  int selected_tile_percent{100};
  int tile_selector_salt{};
  bool harvest_before_switch{true};
  int seed_buffer{8};
};

[[nodiscard]] std::array<int, fastkag::N_PRODUCTS> demand_at_step(
    const std::vector<std::int8_t>& shops, int step,
    int town_shop_sell_interval = 4, int town_center_sell_interval = 24);

[[nodiscard]] FutureDemand build_future_demand(
    const std::vector<ReferenceFrame>& realized_future,
    int town_shop_sell_interval = 4, int town_center_sell_interval = 24);

[[nodiscard]] fastkag::Item target_for_tile(const CropPolicy& policy,
                                            int tile) noexcept;
[[nodiscard]] bool tile_selected(const CropPolicy& policy, int tile) noexcept;

[[nodiscard]] int sell_quantity(SellPace pace, int own_stock,
                                int opponent_stock) noexcept;

[[nodiscard]] const char* sell_pace_name(SellPace pace) noexcept;
[[nodiscard]] const char* baseline_sell_mode_name(BaselineSellMode mode) noexcept;

}  // namespace joint_fixed_move_oracle::economic
