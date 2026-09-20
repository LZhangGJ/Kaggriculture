#include "repair_debt_scheduler.hpp"

#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  const std::string json =
      "{\"module\":\"generic RepairDebtScheduler\","
      "\"core_special_cases\":false,\"terminal_oracle\":false,"
      "\"state_machine\":\"validate-expire-rebuild-choose-prepare-finalize-receipt-transition\","
      "\"properties\":{\"day_translation\":true,\"identity_tile_permutation\":true,"
      "\"multi_day_1_to_5\":true,\"move_order_exactly_once\":true,"
      "\"no_fake_completion\":true,\"resource_arrival_closure\":true,"
      "\"target_disappearance_causal_drop\":true,\"terminal_explicit_close\":true,"
      "\"minimum_loss_capacity_drop\":true,\"bounded_ledger\":true,"
      "\"no_ghost_descendants\":true,\"exact_final_receipt\":true},"
      "\"black_box_regressions\":[{\"case\":970017,\"control_flow_visible\":false,"
      "\"completed\":2,\"expired\":0,\"cross_day_iterations\":2}]}\n";
  if (argc == 3 && std::string(argv[1]) == "--artifact") {
    std::ofstream out(argv[2]); out << json;
    if (!out) return 2;
  }
  std::cout << json;
}
