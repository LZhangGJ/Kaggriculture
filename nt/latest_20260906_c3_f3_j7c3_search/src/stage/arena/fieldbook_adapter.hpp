#pragma once
// Opponent-only bridge. Never used by the candidate's economic planner.
#include "policy.hpp"
#include <memory>

namespace fieldbook {
class Controller {
 public:
  Controller();
  ~Controller();
  Controller(const Controller&); // Deep snapshot for OFFLINE branch validation.
  Controller& operator=(const Controller&) = delete;
  fastkag::PlayerAction act(const dp7::View& observation, int seat);
  std::array<int,5> segments() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
// Diagnostic order matches flattening the original Python PackedObservation.
std::vector<double> input_fields(const dp7::View& observation, int seat);
}
