#include "legacy_baseline_forecast.hpp"

#include <limits>
#include <sstream>

namespace legacy_baseline_forecast {
namespace {

constexpr std::size_t kOfficialMarketSlots = 10;

bool valid_operation(g001::market::Operation operation) {
  const int value = static_cast<int>(operation);
  return value >= static_cast<int>(g001::market::Operation::Pass) &&
         value <= static_cast<int>(g001::market::Operation::BuyLand);
}
bool valid_product(g001::market::Product product) {
  const auto value = static_cast<std::size_t>(product);
  return value < g001::market::product_count;
}
bool valid_animal(g001::market::Animal animal) {
  const int value = static_cast<int>(animal);
  return value >= static_cast<int>(g001::market::Animal::Goose) &&
         value <= static_cast<int>(g001::market::Animal::Sheep);
}

bool add_checked(int& target, int quantity) {
  if (quantity < 0 || target > std::numeric_limits<int>::max() - quantity)
    return false;
  target += quantity;
  return true;
}

Result reject(const Input& input, std::string reason) {
  Result result;
  result.forecast = input.fixed_forecast;
  result.audit.reason = "rejected: " + std::move(reason);
  return result;
}

}  // namespace

Result compile(const Input& input) {
  if (input.current_step < 0)
    return reject(input, "negative current step");
  if (!input.fixed_forecast_certified)
    return reject(input, "FixedForecast is not certified");
  if (input.fixed_forecast.ticks.size() < 24 ||
      input.fixed_forecast.ticks.size() > 72)
    return reject(input, "FixedForecast must contain 24..72 ticks");
  if (input.own_market_frames.size() != input.fixed_forecast.ticks.size())
    return reject(input, "own market frames do not cover the full forecast window");
  if (input.own_market_frames.empty() ||
      input.own_market_frames.front().step != input.current_step)
    return reject(input, "current selected legacy queue is not explicitly included");
  if (input.current_step > std::numeric_limits<int>::max() -
                               static_cast<int>(input.fixed_forecast.ticks.size()))
    return reject(input, "forecast step range overflows int");

  for (std::size_t index = 0; index < input.fixed_forecast.ticks.size(); ++index) {
    const int expected_step = input.current_step + static_cast<int>(index);
    if (input.fixed_forecast.ticks[index].step != expected_step)
      return reject(input, "FixedForecast steps are not contiguous from current step");
    if (input.own_market_frames[index].step != expected_step)
      return reject(input, "own market frames contain a gap or out-of-order step");
    if (!input.own_market_frames[index].causal_queue_known)
      return reject(input, "current/future own market queue is unknown");
    if (input.own_market_frames[index].crosses_route_switch)
      return reject(input, "forecast crosses an unresolved route/production switch");
    if (input.own_market_frames[index].orders.size() > kOfficialMarketSlots)
      return reject(input, "legacy queue exceeds the official 10-slot limit");
    for (const int existing : input.fixed_forecast.ticks[index].baseline_sale) {
      if (existing != 0)
        return reject(input, "input FixedForecast already contains baseline sales");
    }
  }

  Result result;
  result.forecast = input.fixed_forecast;
  result.audit.frames.reserve(input.own_market_frames.size());
  for (std::size_t frame_index = 0;
       frame_index < input.own_market_frames.size(); ++frame_index) {
    const OwnMarketFrame& frame = input.own_market_frames[frame_index];
    result.forecast.ticks[frame_index].legacy_market_orders = frame.orders;
    result.forecast.ticks[frame_index].legacy_market_queue_known = true;
    FrameAudit frame_audit;
    frame_audit.step = frame.step;
    frame_audit.original_slot_count = static_cast<int>(frame.orders.size());
    for (std::size_t slot = 0; slot < frame.orders.size(); ++slot) {
      const g001::market::Order& order = frame.orders[slot];
      if (!valid_operation(order.operation))
        return reject(input, "legacy queue contains an invalid operation enum");
      if (order.quantity < 0)
        return reject(input, "legacy queue contains a negative quantity");
      if (!valid_product(order.product) || !valid_animal(order.animal))
        return reject(input, "legacy queue contains an out-of-range typed item");

      if (order.operation == g001::market::Operation::Sell) {
        const std::size_t product = static_cast<std::size_t>(order.product);
        if (!add_checked(result.forecast.ticks[frame_index].baseline_sale[product],
                         order.quantity) ||
            !add_checked(frame_audit.sell_by_product[product], order.quantity) ||
            !add_checked(frame_audit.sell_units, order.quantity)) {
          return reject(input, "aggregated legacy SELL quantity overflows int");
        }
        frame_audit.sell_slots.push_back(
            {static_cast<int>(slot), order.product, order.quantity});
      } else {
        frame_audit.non_sell_slots.push_back(
            {static_cast<int>(slot), order.operation, order.product,
             order.animal, order.quantity});
      }
    }

    const bool current = frame_index == 0;
    int& sell_total = current ? result.audit.current_sell_units
                              : result.audit.future_sell_units;
    int& non_sell_total = current ? result.audit.current_non_sell_orders
                                  : result.audit.future_non_sell_orders;
    auto& sell_products = current ? result.audit.current_sell_by_product
                                  : result.audit.future_sell_by_product;
    if (!add_checked(sell_total, frame_audit.sell_units) ||
        !add_checked(non_sell_total,
                     static_cast<int>(frame_audit.non_sell_slots.size()))) {
      return reject(input, "legacy baseline audit total overflows int");
    }
    for (std::size_t product = 0; product < g001::market::product_count; ++product) {
      if (!add_checked(sell_products[product],
                       frame_audit.sell_by_product[product]))
        return reject(input, "legacy baseline product total overflows int");
    }
    result.audit.frames.push_back(std::move(frame_audit));
  }

  result.audit.accepted = true;
  std::ostringstream reason;
  reason << "accepted " << result.audit.frames.size()
         << " contiguous own causal queues; current SELL units="
         << result.audit.current_sell_units << " future SELL units="
         << result.audit.future_sell_units << " current non-SELL slots="
         << result.audit.current_non_sell_orders << " future non-SELL slots="
         << result.audit.future_non_sell_orders;
  result.audit.reason = reason.str();
  return result;
}

}  // namespace legacy_baseline_forecast
