#include "scheduler.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {
void require(bool value, std::string_view message) {
  if (!value) { std::cerr << "FAILED: " << message << '\n'; std::exit(1); }
}
fastkag::PlayerAction turn(fastkag::Op op) {
  fastkag::PlayerAction out; out.units.push_back({op}); return out;
}
}

int main() {
  std::vector<fastkag::PlayerAction> tape;
  tape.push_back(turn(fastkag::Op::PLANT));
  for (int i = 1; i < 12; ++i) tape.push_back(turn(fastkag::Op::EAST));
  tape.push_back(turn(fastkag::Op::PASS));
  const auto strict = weed_audit::choose_strict_slack(tape, 0, 0, 20);
  require(strict && strict->skip_step == 12 && strict->used_pass,
          "strict policy waits past legacy source nine for PASS");
  require(weed_audit::exact_move_source_invariant(0, strict->skip_step, tape, 0),
          "strict schedule preserves exact MOVE source indices");

  std::vector<fastkag::PlayerAction> cross(30, turn(fastkag::Op::WATER));
  cross[22] = turn(fastkag::Op::PLANT);
  cross[23] = turn(fastkag::Op::EAST);
  cross[24] = turn(fastkag::Op::WEST);
  cross[26] = turn(fastkag::Op::PASS);
  const auto farmer = weed_audit::choose_strict_slack(cross, 22, 0, 8);
  require(farmer && farmer->skip_step <= 23 && !farmer->crossed_day,
          "farmer cannot cross official end-of-day coordinate reset");
  const auto hand = weed_audit::choose_strict_slack(cross, 22, 1, 8);
  require(hand && hand->skip_step <= 23 && !hand->crossed_day,
          "hand transaction cannot outlive end-of-day destruction");

  std::vector<fastkag::PlayerAction> tail;
  tail.push_back(turn(fastkag::Op::PLANT));
  tail.push_back(turn(fastkag::Op::WATER));
  tail.push_back(turn(fastkag::Op::EAST));
  tail.push_back(turn(fastkag::Op::NORTH));
  const auto tail_choice = weed_audit::choose_tail_preserve_move(tail, 0, 0);
  require(tail_choice && tail_choice->skip_step == 1 &&
              tail_choice->skip_op == fastkag::Op::WATER,
          "tail preserve absorbs last non-MOVE before tail MOVEs");
  std::vector<fastkag::PlayerAction> all_moves{
      turn(fastkag::Op::PLANT), turn(fastkag::Op::EAST), turn(fastkag::Op::NORTH)};
  require(!weed_audit::choose_tail_preserve_move(all_moves, 0, 0),
          "all-MOVE remainder declines DIG instead of editing movement");
  std::cout << "scheduler tests passed\n";
}
