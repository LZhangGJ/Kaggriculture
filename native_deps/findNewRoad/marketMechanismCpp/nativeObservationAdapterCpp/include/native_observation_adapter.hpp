#pragma once

#include "dump_scenarios.hpp"
#include "public_belief_runtime.hpp"
#include "simulator.hpp"

#include <cstddef>
#include <optional>

namespace native_observation_adapter {

// Pure mapping of the official observation surface plus the focal player's
// own private state. No temporal state is changed.
[[nodiscard]] public_belief_runtime::Observation map_observation(
    const fastkag::Simulator& simulator, int player);

struct Config {
  std::size_t history_capacity{64};
};

struct UpdateResult {
  public_belief_runtime::Snapshot snapshot;
  public_belief_runtime::TransitionEvidence evidence;
  g001::dump::DerivedOwnSellFill derived_fill;
};

// Explicit two-phase boundary. stage_submission reads only the focal player's
// selected action/current legal state. observe accepts the next official
// state and settles by own-shed conservation before updating the public belief.
class Adapter {
 public:
  explicit Adapter(Config config = {});

  [[nodiscard]] public_belief_runtime::Snapshot reset(
      const fastkag::Simulator& simulator, int player);
  void stage_submission(const fastkag::Simulator& simulator,
                        const fastkag::PlayerAction& own_submission);
  [[nodiscard]] UpdateResult observe(const fastkag::Simulator& simulator);

  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] bool awaiting_observation() const noexcept {
    return pending_.has_value();
  }
  [[nodiscard]] int player() const noexcept { return player_; }

 private:
  struct Pending {
    int step{-1};
    g001::dump::OwnShedTransition shed_transition;
    public_belief_runtime::TransitionEvidence evidence;
  };

  Config config_;
  bool initialized_{};
  int player_{-1};
  int last_observed_step_{-1};
  std::optional<public_belief_runtime::Runtime> runtime_;
  std::optional<Pending> pending_;
};

}  // namespace native_observation_adapter
