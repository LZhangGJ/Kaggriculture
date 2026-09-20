#include "native_teammate.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::NativeAgentState;
using fastkag::NativeRepairAudit;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using Move = std::tuple<int, int, int>;  // source/execution step, hour, op

bool movement(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
      op == Op::WEST;
}

void print_moves(const char* label, const std::vector<Move>& moves) {
  std::cout << label << "=[";
  for (std::size_t index = 0; index < moves.size(); ++index) {
    if (index) std::cout << ',';
    const auto [step, hour, op] = moves[index];
    std::cout << "{s:" << step << ",h:" << hour << ",op:" << op << '}';
  }
  std::cout << "]\n";
}

}  // namespace

int main() try {
  constexpr std::uint64_t seed = 25771838701ULL;
  const auto tape = g001::repair::load_route(
      NATIVE_G001_TAPES, NATIVE_G001_LIBRARY, "G001");
  if (tape.size() != 719) throw std::runtime_error("G001 length != 719");
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  fastkag::Config config;
  config.episode_steps = 720;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, seed);
  std::array<NativeAgentState, 2> states;
  NativeRepairAudit audit;
  const auto repair = fastkag::native_repair_options_from_mask(64);
  std::map<std::pair<int, int>, std::vector<Move>> raw;
  std::map<std::pair<int, int>, std::vector<Move>> final;
  std::map<int, int> reason2_by_day;
  std::map<int, int> reason2_unsupported_ops;
  int reason2_with_other_crop = 0;
  int reason2_with_other_move = 0;
  while (!simulator.done()) {
    const int step = simulator.step_count();
    const int day = simulator.day();
    const int hour = simulator.hour();
    const int reason2_before = audit.route_skeleton_v3_fail_reasons[2];
    std::array<PlayerAction, 2> actions{
        executor.action_external(
            simulator, 0, 0, states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true,
            repair, &audit),
        executor.action_external(simulator, 1, 0, states[1])};
    const auto& source = tape[static_cast<std::size_t>(step)].units;
    const bool watched = (step >= 16 && step <= 23) ||
        (step >= 157 && step <= 167);
    if (watched) {
      const int actor = step < 24 ? 2 : 5;
      const auto source_op = actor < static_cast<int>(source.size())
          ? static_cast<int>(source[static_cast<std::size_t>(actor)].op) : -1;
      const auto final_op = actor < static_cast<int>(actions[0].units.size())
          ? static_cast<int>(actions[0].units[static_cast<std::size_t>(actor)].op)
          : -1;
      std::cout << "watch step=" << step << " actor=" << actor
                << " source_op=" << source_op << " final_op=" << final_op
                << " reason2="
                << (audit.route_skeleton_v3_fail_reasons[2] != reason2_before)
                << " remaining=[";
      const auto& actors = states[0].experimental_route_skeleton_v3.actors;
      const auto found = std::find_if(
          actors.begin(), actors.end(), [&](const auto& candidate) {
            return candidate.actor == actor;
          });
      if (found != actors.end())
        for (std::size_t cursor = found->cursor; cursor < found->moves.size();
             ++cursor)
          std::cout << "{src:" << found->moves[cursor].source_step
                    << ",op:" << static_cast<int>(found->moves[cursor].action.op)
                    << ",deferred:" << found->moves[cursor].deferred_by_planner
                    << "},";
      std::cout << "]\n";
    }
    const std::size_t width = std::max(source.size(), actions[0].units.size());
    for (std::size_t actor = 0; actor < width; ++actor) {
      const Action source_action = actor < source.size() ? source[actor] : Action{};
      const Action final_action = actor < actions[0].units.size()
          ? actions[0].units[actor] : Action{};
      if (movement(source_action.op))
        raw[{day, static_cast<int>(actor)}].push_back(
            {step, hour, static_cast<int>(source_action.op)});
      if (movement(final_action.op))
        final[{day, static_cast<int>(actor)}].push_back(
            {step, hour, static_cast<int>(final_action.op)});
    }
    if (audit.route_skeleton_v3_fail_reasons[2] != reason2_before) {
      ++reason2_by_day[day];
      bool has_crop = false;
      bool has_move = false;
      for (const auto& unit : actions[0].units) {
        has_crop = has_crop || unit.op == Op::DIG || unit.op == Op::PLANT ||
            unit.op == Op::WATER || unit.op == Op::HARVEST;
        has_move = has_move || movement(unit.op);
        if (unit.op != Op::PASS && !movement(unit.op) && unit.op != Op::DIG &&
            unit.op != Op::PLANT && unit.op != Op::WATER &&
            unit.op != Op::HARVEST)
          ++reason2_unsupported_ops[static_cast<int>(unit.op)];
      }
      reason2_with_other_crop += has_crop;
      reason2_with_other_move += has_move;
      if ((day == 0 || day == 6)) {
        std::cout << "reason2 step=" << step << " day=" << day
                  << " hour=" << hour << " source_width=" << source.size()
                  << " final_width=" << actions[0].units.size()
                  << " hands=" << simulator.farms()[0].hands.size()
                  << " source_ops=[";
        for (std::size_t actor = 0; actor < source.size(); ++actor) {
          if (actor) std::cout << ',';
          std::cout << static_cast<int>(source[actor].op);
        }
        std::cout << "] final_ops=[";
        for (std::size_t actor = 0; actor < actions[0].units.size(); ++actor) {
          if (actor) std::cout << ',';
          std::cout << static_cast<int>(actions[0].units[actor].op);
        }
        std::cout << "]\n";
      }
    }
    simulator.step(actions);
  }
  std::set<std::pair<int, int>> keys;
  for (const auto& [key, unused] : raw) {
    static_cast<void>(unused);
    keys.insert(key);
  }
  for (const auto& [key, unused] : final) {
    static_cast<void>(unused);
    keys.insert(key);
  }
  int mismatches = 0;
  for (const auto& key : keys) {
    std::vector<int> raw_ops;
    std::vector<int> final_ops;
    for (const auto& move : raw[key]) raw_ops.push_back(std::get<2>(move));
    for (const auto& move : final[key]) final_ops.push_back(std::get<2>(move));
    if (raw_ops == final_ops) continue;
    ++mismatches;
    std::cout << "mismatch day=" << key.first << " actor=" << key.second
              << "\n";
    print_moves("raw", raw[key]);
    print_moves("final", final[key]);
  }
  std::cout << "reason2_by_day=[";
  for (const auto& [day, count] : reason2_by_day)
    std::cout << "{d:" << day << ",n:" << count << "},";
  std::cout << "] total=" << audit.route_skeleton_v3_fail_reasons[2]
            << " with_crop=" << reason2_with_other_crop
            << " with_move=" << reason2_with_other_move
            << " mismatches=" << mismatches << " unsupported_ops=[";
  for (const auto& [op, count] : reason2_unsupported_ops)
    std::cout << "{op:" << op << ",n:" << count << "},";
  std::cout << "]\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "forced audit failure: " << error.what() << '\n';
  return 1;
}
