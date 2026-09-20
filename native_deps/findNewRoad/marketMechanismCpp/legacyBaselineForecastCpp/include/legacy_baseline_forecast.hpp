#pragma once

#include "market.hpp"
#include "rolling_optimizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace legacy_baseline_forecast {

struct OwnMarketFrame {
  int step{-1};
  std::vector<g001::market::Order> orders;
  // Must be true for the current selected queue and every stable own future
  // tape frame.  No unknown frame is filled from a route or replay lookup.
  bool causal_queue_known{};
  // An unresolved production/route switch invalidates the fixed legacy
  // baseline rather than silently splicing two tapes.
  bool crosses_route_switch{};
};

// Deliberately has no route ID, opponent field, or realized-future field.
struct Input {
  int current_step{};
  bool fixed_forecast_certified{};
  g001::rolling::FixedForecast fixed_forecast;
  std::vector<OwnMarketFrame> own_market_frames;
};

struct SellSlotAudit {
  int slot{-1};
  g001::market::Product product{g001::market::Product::Wheat};
  int quantity{};
};

struct NonSellSlotAudit {
  int slot{-1};
  g001::market::Operation operation{g001::market::Operation::Pass};
  g001::market::Product product{g001::market::Product::Wheat};
  g001::market::Animal animal{g001::market::Animal::Goose};
  int quantity{};
};

struct FrameAudit {
  int step{-1};
  int original_slot_count{};
  g001::market::Inventory sell_by_product{};
  int sell_units{};
  std::vector<SellSlotAudit> sell_slots;
  std::vector<NonSellSlotAudit> non_sell_slots;
};

struct Audit {
  bool accepted{};
  int current_sell_units{};
  int future_sell_units{};
  int current_non_sell_orders{};
  int future_non_sell_orders{};
  g001::market::Inventory current_sell_by_product{};
  g001::market::Inventory future_sell_by_product{};
  std::vector<FrameAudit> frames;
  std::string reason;
};

struct Result {
  g001::rolling::FixedForecast forecast;
  Audit audit;
};

[[nodiscard]] Result compile(const Input& input);

}  // namespace legacy_baseline_forecast
