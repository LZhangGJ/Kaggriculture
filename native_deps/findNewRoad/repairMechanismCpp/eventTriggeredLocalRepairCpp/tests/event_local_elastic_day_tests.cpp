#include "../include/event_local_elastic_day.hpp"

#include <iostream>
#include <stdexcept>

namespace local = g001::day_horizon_repair;
using g001::event_local_repair::Action;
using g001::event_local_repair::Op;
using g001::event_local_repair::Position;
using g001::event_local_repair::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Action move(int direction) { return {Op::Move, -1, 1, direction, 0}; }

local::ElasticDayInput input(std::vector<int> seeds, int turns = 4) {
  local::ElasticDayInput value;
  value.actor.actor = 0;
  value.actor.start = {2, 2};
  value.actor.raw = {{Op::Water, 0, 1}, move(3), {}, {}};
  value.actor.raw.resize(static_cast<std::size_t>(turns));
  value.actor.certified_sink = {true, false, true, true};
  value.actor.certified_sink.resize(static_cast<std::size_t>(turns));
  value.actor.trigger_source = {true, false, false, false};
  value.actor.trigger_source.resize(static_cast<std::size_t>(turns));
  value.plot = {2, 2};
  value.plot_state = {TileKind::Weed, -1, false, false};
  value.desired_crop = 0;
  value.scenario_seed_receipt_by_turn = std::move(seeds);
  return value;
}

void zero_fill_does_not_delay_move() {
  const auto result = local::compile_event_local_elastic_day(
      input({0, 0, 0, 0}));
  check(result.manifest[0].op == Op::Dig &&
            result.manifest[1] == move(3) && result.delayed_moves == 0 &&
            result.purchase_requested && result.has_debt &&
            result.terminal_move_tokens == 0 && result.move_source_order_exact,
        "zero fill delayed/dropped MOVE or lost debt");
}

void confirmed_fill_and_two_sinks_close_before_move() {
  const auto result = local::compile_event_local_elastic_day(
      input({0, 1, 1, 1}));
  check(result.manifest[0].op == Op::Dig &&
            result.manifest[1].op == Op::Plant &&
            result.manifest[2].op == Op::Water &&
            result.manifest[3] == move(3) && !result.has_debt &&
            result.delayed_moves == 1 && result.maximum_move_delay == 2 &&
            result.move_source_order_exact && result.move_payload_exact &&
            result.terminal_move_tokens == 0 &&
            result.terminal_hard_raw_tokens == 0,
        "confirmed fill did not close DIG/PLANT/WATER before ordered MOVE");
}

void no_slack_keeps_move_and_cross_day_debt() {
  auto value = input({0, 1}, 2);
  const auto result = local::compile_event_local_elastic_day(value);
  check(result.manifest[0].op == Op::Dig && result.manifest[1] == move(3) &&
            result.has_debt && result.delayed_moves == 0 &&
            result.terminal_move_tokens == 0,
        "no-slack branch delayed MOVE or lost cross-day debt");
}

void pretrigger_prefix_cannot_be_consumed_or_advanced() {
  auto value = input({0, 0, 0, 0});
  value.actor.trigger_source = {false, true, false, false};
  bool rejected = false;
  try {
    static_cast<void>(local::compile_event_local_elastic_day(value));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "elastic API accepted a pre-trigger prefix");
}

}  // namespace

int main() try {
  zero_fill_does_not_delay_move();
  confirmed_fill_and_two_sinks_close_before_move();
  no_slack_keeps_move_and_cross_day_debt();
  pretrigger_prefix_cannot_be_consumed_or_advanced();
  std::cout << "event local elastic day: 4 fixtures passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
