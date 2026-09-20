#include "route_loader.hpp"
#include "static_dependencies.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

const char* operation_name(fastkag::Op operation) {
  return operation == fastkag::Op::BUY_SEED ? "BUY_SEED" : "BUY_ANIMAL";
}

const char* item_name(fastkag::Item item) {
  static constexpr const char* names[]{
      "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
      "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"};
  const int index = static_cast<int>(item);
  return index >= 0 && index < 12 ? names[index] : "INVALID";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4) {
      std::cerr << "usage: " << argv[0]
                << " ROUTE_ACTIONS_ZLIB ROUTE_LIBRARY_JSON [FAMILY]\n";
      return 2;
    }
    const std::string family = argc == 4 ? argv[3] : "G001";
    const auto tape = g001::repair::load_route(argv[1], argv[2], family);
    const auto dependencies =
        g001::failure_debt::extract_static_dependencies(tape);
    int seed_units = 0, animal_units = 0, complete_seed = 0,
        complete_animal = 0;
    for (const auto& dependency : dependencies) {
      const bool seed = dependency.purchase_operation == fastkag::Op::BUY_SEED;
      (seed ? seed_units : animal_units)++;
      if (dependency.complete) (seed ? complete_seed : complete_animal)++;
    }
    std::cout << "{\"schema\":\"g001-failure-dependency-v1\",\"family\":\""
              << family << "\",\"steps\":" << tape.size()
              << ",\"seed_units\":" << seed_units
              << ",\"complete_seed_units\":" << complete_seed
              << ",\"animal_units\":" << animal_units
              << ",\"complete_animal_units\":" << complete_animal
              << ",\"dependencies\":[";
    for (std::size_t index = 0; index < dependencies.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto& row = dependencies[index];
      std::cout << "{\"operation\":\"" << operation_name(row.purchase_operation)
                << "\",\"item\":\"" << item_name(row.item)
                << "\",\"purchase_step\":" << row.purchase_step
                << ",\"purchase_slot\":" << row.purchase_slot
                << ",\"purchase_unit\":" << row.purchase_unit
                << ",\"first_unit_step\":" << row.first_unit_step
                << ",\"first_actor\":" << row.first_actor
                << ",\"terminal_unit_step\":" << row.terminal_unit_step
                << ",\"terminal_actor\":" << row.terminal_actor
                << ",\"complete\":" << (row.complete ? "true" : "false")
                << ",\"reason\":\"" << row.reason << "\"}";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "g001_failure_dependency_audit: " << error.what() << '\n';
    return 1;
  }
}
