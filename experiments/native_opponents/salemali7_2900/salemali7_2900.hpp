#pragma once

#include "simulator.hpp"

#include <memory>
#include <string>

namespace salemali7_2900 {

class Opponent {
 public:
  explicit Opponent(const std::string& asset_path);
  ~Opponent();
  Opponent(Opponent&&) noexcept;
  Opponent& operator=(Opponent&&) noexcept;
  Opponent(const Opponent&) = delete;
  Opponent& operator=(const Opponent&) = delete;

  fastkag::PlayerAction action(const fastkag::Simulator& env, int player);
  void reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace salemali7_2900
