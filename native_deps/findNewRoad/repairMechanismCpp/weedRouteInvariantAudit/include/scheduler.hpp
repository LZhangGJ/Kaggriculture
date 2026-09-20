#pragma once

#include "repair.hpp"

#include <optional>
#include <vector>

namespace weed_audit {

enum class Policy { LegacyNine, MinimumLoss, StrictSlack };

struct Choice {
  int skip_step{-1};
  fastkag::Op skip_op{fastkag::Op::PASS};
  bool crossed_day{};
  bool day_end_truncated{};
  bool used_pass{};
};

fastkag::Action tape_unit(const std::vector<fastkag::PlayerAction>& tape,
                          int step, int actor);

// Reproduces the current native min-loss selection exactly: at most 16
// sources and never beyond the current day.
std::optional<Choice> choose_minimum_loss(
    const std::vector<fastkag::PlayerAction>& tape, int start_step, int actor,
    fastkag::Position actual_position, int maximum_lookahead = 16);

// Strict slack policy: never absorbs MOVE. It waits for the first PASS in the
// remainder of the current day; only if none exists does it choose the
// least-valued non-MOVE. No actor crosses day close: the official simulator
// resets coordinates there, so source-position semantics would not survive.
std::optional<Choice> choose_strict_slack(
    const std::vector<fastkag::PlayerAction>& tape, int start_step, int actor,
    int maximum_lookahead = 23);

// Preserve every MOVE source in the current day. Prefer the first later PASS;
// otherwise absorb the last later non-MOVE so the remaining tail MOVEs execute
// at their original source indices. If t+1..day_end is all MOVE, return null.
std::optional<Choice> choose_tail_preserve_move(
    const std::vector<fastkag::PlayerAction>& tape, int start_step, int actor);

std::vector<int> emitted_move_sources(int start_step, int skip_step,
                                      const std::vector<fastkag::PlayerAction>& tape,
                                      int actor);
bool exact_move_source_invariant(int start_step, int skip_step,
                                 const std::vector<fastkag::PlayerAction>& tape,
                                 int actor);

}  // namespace weed_audit
