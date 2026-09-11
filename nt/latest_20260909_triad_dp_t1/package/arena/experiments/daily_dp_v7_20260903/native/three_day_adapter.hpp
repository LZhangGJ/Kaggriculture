#pragma once
// Frozen opponent only; no dependency from our economic planner.
#include "policy.hpp"
#include <memory>
namespace threeday {
class Controller {
 public:
  Controller();~Controller();
  Controller(const Controller&); // Deep snapshot for OFFLINE branch validation.
  Controller& operator=(const Controller&)=delete;
  fastkag::PlayerAction act(const dp7::View&,int seat);
  int route() const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
std::vector<double> input_fields(const dp7::View&,int seat);
}
