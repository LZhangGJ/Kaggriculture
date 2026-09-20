#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4) {
      std::cerr << "usage: " << argv[0]
                << " ROUTE_ACTIONS_ZLIB ROUTE_LIBRARY_JSON [FAMILY]\n";
      return 2;
    }
    const std::string family = argc == 4 ? argv[3] : "G001";
    const auto tape = g001::repair::load_route(argv[1], argv[2], family);
    constexpr int kMaxSlots = 10;
    std::array<int, kMaxSlots + 1> sell_slot_histogram{};
    std::array<int, fastkag::N_PRODUCTS + 1> sell_product_histogram{};
    std::array<int, kMaxSlots + 1> total_market_slot_histogram{};
    int steps_with_sells = 0;
    int steps_with_multiple_sell_products = 0;
    int maximum_sell_slots = 0;
    int maximum_sell_products = 0;
    int maximum_market_slots = 0;
    for (const auto& turn : tape) {
      int sell_slots = 0;
      std::set<int> products;
      for (const auto& order : turn.market) {
        if (order.op != fastkag::Op::SELL) continue;
        ++sell_slots;
        const int product = static_cast<int>(order.item);
        if (product >= 0 && product < fastkag::N_PRODUCTS)
          products.insert(product);
      }
      const int sell_products = static_cast<int>(products.size());
      const int market_slots = static_cast<int>(turn.market.size());
      ++sell_slot_histogram[static_cast<std::size_t>(
          std::min(kMaxSlots, sell_slots))];
      ++sell_product_histogram[static_cast<std::size_t>(
          std::min(fastkag::N_PRODUCTS, sell_products))];
      ++total_market_slot_histogram[static_cast<std::size_t>(
          std::min(kMaxSlots, market_slots))];
      steps_with_sells += sell_slots > 0;
      steps_with_multiple_sell_products += sell_products > 1;
      maximum_sell_slots = std::max(maximum_sell_slots, sell_slots);
      maximum_sell_products = std::max(maximum_sell_products, sell_products);
      maximum_market_slots = std::max(maximum_market_slots, market_slots);
    }
    auto emit_histogram = [](const auto& histogram) {
      std::cout << '[';
      for (std::size_t index = 0; index < histogram.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << histogram[index];
      }
      std::cout << ']';
    };
    std::cout << "{\"family\":\"" << family << "\",\"steps\":"
              << tape.size() << ",\"steps_with_sells\":" << steps_with_sells
              << ",\"steps_with_multiple_sell_products\":"
              << steps_with_multiple_sell_products
              << ",\"maximum_sell_slots\":" << maximum_sell_slots
              << ",\"maximum_sell_products\":" << maximum_sell_products
              << ",\"maximum_market_slots\":" << maximum_market_slots
              << ",\"sell_slot_histogram\":";
    emit_histogram(sell_slot_histogram);
    std::cout << ",\"sell_product_histogram\":";
    emit_histogram(sell_product_histogram);
    std::cout << ",\"total_market_slot_histogram\":";
    emit_histogram(total_market_slot_histogram);
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "g001_market_structure_audit: " << error.what() << '\n';
    return 1;
  }
}
