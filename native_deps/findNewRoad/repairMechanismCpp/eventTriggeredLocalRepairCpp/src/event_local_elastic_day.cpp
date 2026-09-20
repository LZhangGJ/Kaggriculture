#include "../include/event_local_elastic_day.hpp"

#include <algorithm>
#include <stdexcept>

namespace g001::day_horizon_repair {
namespace {

using event_local_repair::Op;
using event_local_repair::TileKind;

Position after_move(Position position, const Action& action) {
  if (action.op != Op::Move) return position;
  if (action.arg0 == 0) --position.row;
  else if (action.arg0 == 1) ++position.row;
  else if (action.arg0 == 2) --position.column;
  else if (action.arg0 == 3) ++position.column;
  else throw std::invalid_argument("invalid elastic MOVE direction");
  return position;
}

bool crop_source(const Action& action) {
  return action.op == Op::Dig || action.op == Op::Plant ||
      action.op == Op::Water || action.op == Op::Harvest;
}

int transitions_needed(const event_local_repair::TileObservation& state,
                       int desired) {
  if (state.kind == TileKind::Weed) return 3;
  if (state.kind == TileKind::Empty) return 2;
  if (state.kind != TileKind::Crop || state.item != desired) return 3;
  return state.watered_today ? 0 : 1;
}

}  // namespace

ElasticDayResult compile_event_local_elastic_day(
    const ElasticDayInput& input) {
  const auto turns = input.actor.raw.size();
  if (input.actor.actor < 0 || turns == 0 ||
      input.actor.certified_sink.size() != turns ||
      input.actor.trigger_source.size() != turns ||
      input.scenario_seed_receipt_by_turn.size() != turns ||
      input.desired_crop < 0)
    throw std::invalid_argument("invalid elastic day input");
  if (!input.actor.trigger_source.front())
    throw std::invalid_argument("elastic suffix must start at observed trigger");
  for (std::size_t turn = 0; turn < turns; ++turn)
    if (input.actor.raw[turn].op == Op::Move &&
        (input.actor.certified_sink[turn] || input.actor.trigger_source[turn]))
      throw std::invalid_argument("MOVE cannot be an elastic sink/trigger");

  ElasticDayResult result;
  result.manifest.resize(turns);
  result.positions_before.reserve(turns);
  result.debt = {1, input.plot, input.desired_crop, false, input.day};
  auto state = input.plot_state;
  auto position = input.actor.start;
  std::size_t cursor = 0;
  bool triggered = false;
  std::vector<int> raw_move_turns;
  std::vector<Action> raw_moves;
  for (std::size_t turn = 0; turn < turns; ++turn)
    if (input.actor.raw[turn].op == Op::Move) {
      raw_move_turns.push_back(static_cast<int>(turn));
      raw_moves.push_back(input.actor.raw[turn]);
    }

  for (std::size_t turn = 0; turn < turns; ++turn) {
    result.positions_before.push_back(position);
    while (cursor < turns && input.actor.certified_sink[cursor] &&
           !input.actor.trigger_source[cursor] &&
           !(triggered && position == input.plot &&
             transitions_needed(state, input.desired_crop) > 0)) {
      if (input.actor.raw[cursor].op == Op::Pass)
        ++result.absorbed_pass_sinks;
      else
        ++result.absorbed_no_effect_sinks;
      ++cursor;
    }

    if (cursor < turns && input.actor.trigger_source[cursor]) {
      if (!crop_source(input.actor.raw[cursor]))
        throw std::invalid_argument("elastic trigger is not crop production");
      triggered = true;
      ++result.absorbed_no_effect_sinks;
      ++cursor;
    }

    Action chosen;
    bool serviced = false;
    if (triggered && position == input.plot) {
      const int needed = transitions_needed(state, input.desired_crop);
      int hard_tokens = 0;
      for (std::size_t source = cursor; source < turns; ++source)
        hard_tokens += !input.actor.certified_sink[source];
      const int remaining_ticks = static_cast<int>(turns - turn);
      const bool seed_ready = state.kind != TileKind::Empty ||
          input.scenario_seed_receipt_by_turn[turn] > 0;
      const bool capacity = needed > 0 && seed_ready &&
          remaining_ticks >= hard_tokens + needed;
      if (state.kind == TileKind::Weed) {
        chosen.op = Op::Dig;
        state = {TileKind::Empty, -1, false, false};
        serviced = true;
      } else if (capacity && state.kind == TileKind::Empty) {
        chosen = {Op::Plant, input.desired_crop, 1};
        state = {TileKind::Crop, input.desired_crop, false, false};
        serviced = true;
      } else if (capacity && state.kind == TileKind::Crop &&
                 state.item == input.desired_crop &&
                 !state.watered_today) {
        chosen = {Op::Water, input.desired_crop, 1};
        state.watered_today = true;
        serviced = true;
      }
      if (state.kind == TileKind::Empty &&
          input.scenario_seed_receipt_by_turn[turn] <= 0 &&
          !result.purchase_requested) {
        result.purchase_requested = true;
        result.purchase_request_turn = static_cast<int>(turn);
      }
    }
    if (serviced) {
      result.manifest[turn] = chosen;
      ++result.assignments;
      continue;
    }

    while (cursor < turns && input.actor.certified_sink[cursor]) {
      if (input.actor.raw[cursor].op == Op::Pass)
        ++result.absorbed_pass_sinks;
      else
        ++result.absorbed_no_effect_sinks;
      ++cursor;
    }
    if (cursor < turns) {
      result.manifest[turn] = input.actor.raw[cursor++];
      position = after_move(position, result.manifest[turn]);
    }
  }

  result.has_debt = triggered && transitions_needed(state, input.desired_crop) > 0;
  for (; cursor < turns; ++cursor) {
    if (input.actor.certified_sink[cursor]) continue;
    if (input.actor.raw[cursor].op == Op::Move) ++result.terminal_move_tokens;
    else ++result.terminal_hard_raw_tokens;
  }
  std::size_t move_cursor = 0;
  result.move_source_order_exact = true;
  result.move_payload_exact = true;
  for (std::size_t turn = 0; turn < turns; ++turn) {
    if (result.manifest[turn].op != Op::Move) continue;
    if (move_cursor >= raw_moves.size()) {
      result.move_source_order_exact = false;
      result.move_payload_exact = false;
      continue;
    }
    result.move_source_order.push_back(move_cursor + 1);
    if (!(result.manifest[turn] == raw_moves[move_cursor]))
      result.move_payload_exact = false;
    const int delay = static_cast<int>(turn) - raw_move_turns[move_cursor];
    if (delay > 0) {
      ++result.delayed_moves;
      result.total_move_delay += delay;
      result.maximum_move_delay = std::max(result.maximum_move_delay, delay);
    }
    ++move_cursor;
  }
  result.move_source_order_exact = result.move_source_order_exact &&
      move_cursor == raw_moves.size() && result.terminal_move_tokens == 0;
  result.capacity_proof_held = result.terminal_move_tokens == 0 &&
      result.terminal_hard_raw_tokens == 0 &&
      result.move_source_order_exact && result.move_payload_exact;
  return result;
}

}  // namespace g001::day_horizon_repair
