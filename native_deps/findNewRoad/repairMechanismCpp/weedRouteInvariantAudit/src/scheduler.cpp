#include "scheduler.hpp"

#include <algorithm>
#include <limits>

namespace weed_audit {
namespace {

fastkag::Position moved(fastkag::Position p, fastkag::Op op) {
  if (op == fastkag::Op::NORTH) --p.y;
  else if (op == fastkag::Op::SOUTH) ++p.y;
  else if (op == fastkag::Op::EAST) ++p.x;
  else if (op == fastkag::Op::WEST) --p.x;
  return p;
}

}  // namespace

fastkag::Action tape_unit(const std::vector<fastkag::PlayerAction>& tape,
                          int step, int actor) {
  if (step < 0 || step >= static_cast<int>(tape.size()) || actor < 0 ||
      actor >= static_cast<int>(tape[static_cast<std::size_t>(step)].units.size()))
    return {};
  return tape[static_cast<std::size_t>(step)].units[static_cast<std::size_t>(actor)];
}

std::optional<Choice> choose_minimum_loss(
    const std::vector<fastkag::PlayerAction>& tape, int start_step, int actor,
    fastkag::Position actual_position, int maximum_lookahead) {
  if (start_step < 0 || start_step >= static_cast<int>(tape.size())) return std::nullopt;
  const int day_end = (start_step / 24 + 1) * 24 - 1;
  const int unconstrained = std::min(static_cast<int>(tape.size()) - 1,
                                     start_step + std::max(0, maximum_lookahead));
  const int last = std::min(day_end, unconstrained);
  std::vector<g001::repair::TimedAction> window;
  auto position = actual_position;
  for (int source = start_step; source <= last; ++source) {
    const auto action = tape_unit(tape, source, actor);
    g001::repair::TimedAction timed;
    timed.action = action;
    timed.required_position = position;
    timed.earliest_step = 0;
    timed.latest_step = last - start_step + 1;
    timed.economic_value = g001::repair::default_economic_value(action.op);
    timed.expected_fill = timed.economic_value > 0;
    window.push_back(timed);
    position = moved(position, action.op);
  }
  if (window.empty()) return std::nullopt;
  const auto plan = g001::repair::search_minimum_loss_realign(
      window, actual_position, static_cast<int>(window.size()) - 1);
  if (plan.metrics.skipped_source_index < 0 || plan.metrics.movement_edits != 0)
    return std::nullopt;
  const int skip = start_step + plan.metrics.skipped_source_index;
  const auto op = tape_unit(tape, skip, actor).op;
  return Choice{skip, op, false, last == day_end && unconstrained > day_end,
                op == fastkag::Op::PASS};
}

std::optional<Choice> choose_strict_slack(
    const std::vector<fastkag::PlayerAction>& tape, int start_step, int actor,
    int maximum_lookahead) {
  if (start_step < 0 || start_step >= static_cast<int>(tape.size())) return std::nullopt;
  const int day_end = (start_step / 24 + 1) * 24 - 1;
  (void)actor;
  const int lifecycle_end = std::min({static_cast<int>(tape.size()) - 1, day_end,
                                      start_step + std::max(0, maximum_lookahead)});
  int best = -1;
  int best_value = std::numeric_limits<int>::max();
  for (int source = start_step; source <= lifecycle_end; ++source) {
    const auto op = tape_unit(tape, source, actor).op;
    if (g001::repair::is_movement(op)) continue;
    if (op == fastkag::Op::PASS)
      return Choice{source, op, false, lifecycle_end == day_end, true};
    const int value = g001::repair::default_economic_value(op);
    if (value < best_value) {
      best = source;
      best_value = value;
    }
  }
  if (best < 0) return std::nullopt;
  const auto op = tape_unit(tape, best, actor).op;
  return Choice{best, op, false, lifecycle_end == day_end, false};
}

std::optional<Choice> choose_tail_preserve_move(
    const std::vector<fastkag::PlayerAction>& tape, int start_step, int actor) {
  if (start_step < 0 || start_step >= static_cast<int>(tape.size())) return std::nullopt;
  const int day_end = std::min(static_cast<int>(tape.size()) - 1,
                               (start_step / 24 + 1) * 24 - 1);
  for (int source = start_step + 1; source <= day_end; ++source) {
    const auto op = tape_unit(tape, source, actor).op;
    if (op == fastkag::Op::PASS)
      return Choice{source, op, false, source == day_end, true};
  }
  for (int source = day_end; source > start_step; --source) {
    const auto op = tape_unit(tape, source, actor).op;
    if (!g001::repair::is_movement(op))
      return Choice{source, op, false, source == day_end, false};
  }
  return std::nullopt;
}

std::vector<int> emitted_move_sources(int start_step, int skip_step,
                                      const std::vector<fastkag::PlayerAction>& tape,
                                      int actor) {
  std::vector<int> out;
  for (int source = start_step; source <= skip_step; ++source)
    if (g001::repair::is_movement(tape_unit(tape, source, actor).op))
      out.push_back(source);
  return out;
}

bool exact_move_source_invariant(int start_step, int skip_step,
                                 const std::vector<fastkag::PlayerAction>& tape,
                                 int actor) {
  if (skip_step < start_step ||
      g001::repair::is_movement(tape_unit(tape, skip_step, actor).op)) return false;
  const auto expected = emitted_move_sources(start_step, skip_step, tape, actor);
  // Insert DIG, replay every source before skip, then resume at skip+1. Thus
  // the emitted MOVE source indices must equal the raw indices exactly.
  std::vector<int> actual;
  for (int source = start_step; source < skip_step; ++source)
    if (g001::repair::is_movement(tape_unit(tape, source, actor).op))
      actual.push_back(source);
  // skip is certified non-MOVE, so <=skip and <skip have identical MOVE sets.
  return actual == expected && std::is_sorted(actual.begin(), actual.end()) &&
         std::adjacent_find(actual.begin(), actual.end()) == actual.end();
}

}  // namespace weed_audit
