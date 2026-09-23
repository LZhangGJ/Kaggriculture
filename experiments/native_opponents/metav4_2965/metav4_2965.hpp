// Offline-only native port of the public "2965 Master Hybrid" opponent.
#pragma once

#include "../../../fast_kaggriculture/src/simulator.hpp"

#include <memory>
#include <string>

namespace metav4_2965 {

// One instance owns one match.  It may control either or both seats; all
// mutable policy state is isolated by seat.  No opponent identity is exposed
// to the policy -- this class is only a training-environment opponent.
class Opponent {
 public:
  explicit Opponent(const std::string& asset_path);
  ~Opponent();
  Opponent(Opponent&&) noexcept;
  Opponent& operator=(Opponent&&) noexcept;
  Opponent(const Opponent&) = delete;
  Opponent& operator=(const Opponent&) = delete;

  fastkag::PlayerAction action(const fastkag::Simulator& env, int player);
  // Deterministic active-rival fixture used only by the parity harness.
  fastkag::PlayerAction fixture_route_action(int route_id, int step) const;
  int route(int player) const;
  void reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace metav4_2965
