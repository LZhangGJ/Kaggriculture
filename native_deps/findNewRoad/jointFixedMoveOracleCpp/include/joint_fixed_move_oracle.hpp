#pragma once

#include "simulator.hpp"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace joint_fixed_move_oracle {

using Tape = std::vector<fastkag::PlayerAction>;

enum class Arm : std::uint8_t {
  Baseline,
  WeedRepair,
  RepairAndDebt,
  TradeOnlyClairvoyant,
  CropAndTrade,
  AllFour,
};

enum class TradePolicy : std::uint8_t {
  Baseline,
  HoldAll,
  ClearExisting,
  HoldThenClear360,
  HoldThenClear480,
  HoldThenClear600,
  TerminalClear,
};

struct MoveCertificate {
  bool valid{true};
  int checked_steps{};
  int checked_actor_slots{};
  int mismatches{};
  std::string reason;
};

// A blocked unit action observed on a weed tile.  Production repair is
// allowed to rewrite non-MOVE slots, but MOVE stays at its original absolute
// step.  `position` is the actor position immediately before event_step.
struct WeedCollision {
  int event_step{};
  int actor{};
  fastkag::Position position{};
  fastkag::Action intended{};
};

struct AbsoluteWeedPatch {
  Tape tape;
  int event_step{-1};
  int replay_step{-1};
  int water_step{-1};
  fastkag::Action displaced_replay{};
  fastkag::Action displaced_water{};
};

[[nodiscard]] bool is_move(fastkag::Op op) noexcept;
[[nodiscard]] bool action_equal(const fastkag::Action& left,
                                const fastkag::Action& right) noexcept;
[[nodiscard]] MoveCertificate certify_absolute_moves(const Tape& baseline,
                                                     const Tape& candidate);
void require_absolute_moves(const Tape& baseline, const Tape& candidate);

// Enumerate fixed-MOVE production recompilations for one collision.  PLANT is
// emitted only when a later same-day, same-position non-MOVE slot can also
// water the newly planted crop.  Without that companion WATER the simulator
// kills a just-planted crop at day end, so a DIG+late-PLANT branch is not a
// valid production repair.  BUILD_PASTURE needs only one replay slot.
[[nodiscard]] std::vector<AbsoluteWeedPatch> enumerate_absolute_weed_patches(
    const Tape& baseline, const Tape& source,
    const std::vector<std::vector<fastkag::Position>>& positions,
    const WeedCollision& collision, int turns_per_day,
    int maximum_candidates = 32);

// Changes only market orders. The current simulator state is used for the
// terminal-clear quantity; no opponent/provider identity enters this API.
[[nodiscard]] fastkag::PlayerAction apply_trade_policy(
    const fastkag::PlayerAction& source, const fastkag::Simulator& simulator,
    int player, TradePolicy policy);

[[nodiscard]] const char* arm_name(Arm arm) noexcept;
[[nodiscard]] const char* trade_policy_name(TradePolicy policy) noexcept;

}  // namespace joint_fixed_move_oracle
