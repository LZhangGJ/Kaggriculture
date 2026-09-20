#pragma once

#include "simulator.hpp"

#include <string>
#include <vector>

namespace g001::failure_debt {

struct StaticPurchaseDependency {
  fastkag::Op purchase_operation{fastkag::Op::PASS};
  fastkag::Item item{fastkag::Item::NONE};
  int purchase_step{-1};
  int purchase_slot{-1};
  int purchase_unit{-1};
  int first_unit_step{-1};
  int first_actor{-1};
  int terminal_unit_step{-1};
  int terminal_actor{-1};
  bool complete{};
  std::string reason;
};

// Static, own-tape-only dependency scan. Runtime legality and state still have
// to be certified by FailureDebtScheduler before a repair is emitted.
[[nodiscard]] std::vector<StaticPurchaseDependency> extract_static_dependencies(
    const std::vector<fastkag::PlayerAction>& tape);

}  // namespace g001::failure_debt
