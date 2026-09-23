#pragma once

#include "simulator.hpp"

#include <memory>
#include <string>

namespace fieldcraft_2887 {

// Offline-only native port of the frozen Fieldcraft R1 public opponent.
// One instance may own independent state for either seat of one episode.
class Opponent {
 public:
  explicit Opponent(const std::string& asset_path);
  ~Opponent();
  Opponent(Opponent&&) noexcept;
  Opponent& operator=(Opponent&&) noexcept;
  Opponent(const Opponent&) = delete;
  Opponent& operator=(const Opponent&) = delete;

  fastkag::PlayerAction action(const fastkag::Simulator& env, int player);
  int route(int player) const;
  void reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fieldcraft_2887
