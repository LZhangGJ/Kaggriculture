#include "protected_queue.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
  namespace market = g001::market;
  protected_queue::Input input;
  input.market_inventory.fill(10000);
  input.own.money = 0;
  input.own.shed[3] = 10;
  input.own.shed[5] = 10;
  input.legacy_orders = {
      {},
      {market::Operation::Sell, market::Product::Strawberry,
       market::Animal::Goose, 10},
      {market::Operation::Sell, market::Product::Egg,
       market::Animal::Goose, 10},
      {market::Operation::BuySeed, market::Product::Strawberry,
       market::Animal::Goose, 1},
  };
  input.optional_sells = {{market::Product::Egg, 3, 10}};
  input.opponent_first_pressure = {{
      {market::Operation::Sell, market::Product::Strawberry,
       market::Animal::Goose, 11}}};
  constexpr int iterations = 10000;
  std::int64_t checksum = 0;
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    input.optional_sells[0].priority = iteration & 1;
    const auto result = protected_queue::compose(input);
    checksum += result.audit.required_funding_sell_units +
                result.audit.optional_replacement_units +
                static_cast<int>(result.orders.size());
  }
  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  std::cout << std::fixed << std::setprecision(3)
            << "iterations=" << iterations
            << " elapsed_ms=" << elapsed * 1000.0
            << " us_per_compose=" << elapsed * 1e6 / iterations
            << " checksum=" << checksum << '\n';
}
