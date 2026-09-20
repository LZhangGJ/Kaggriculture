#include "day_horizon_planner.hpp"

#include <iostream>
#include <stdexcept>

using namespace g001::day_horizon_repair;
using g001::event_local_repair::Op;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Action pass() { return {Op::Pass, -1, 1, 0, 0}; }
Action move(int direction) { return {Op::Move, -1, 1, direction, 71}; }
Action water(int crop) { return {Op::Water, crop, 1, 0, 0}; }
Action harvest(int crop) { return {Op::Harvest, crop, 1, 0, 0}; }
Action plant(int crop) { return {Op::Plant, crop, 1, 0, 0}; }

void move_timeline_is_hard_frozen() {
  ActorPlan actor{0, {0, 0}, {pass(), move(3), pass()}, {false, false, false}};
  const auto result = compile(
      {actor}, {{1, {0, 1}, {water(0)}, 2, 10, true}});
  check(result.move_slots_exact && result.manifest[0][1] == move(3),
        "MOVE slot changed");
  check(result.positions_before[0][2] == Position{0, 1} &&
            result.manifest[0][2] == water(0),
        "position timeline did not schedule at the reached plot");
}

void critical_objective_wins_contested_capacity() {
  ActorPlan actor{0, {2, 2}, {pass()}, {false}};
  const auto result = compile(
      {actor}, {{10, {2, 2}, {water(0)}, 0, 100, false},
                {11, {2, 2}, {harvest(0)}, 0, 1, true}});
  check(result.manifest[0][0] == harvest(0) &&
            result.unscheduled_objectives == std::vector<std::uint64_t>{10},
        "critical objective did not win contested capacity");
}

void only_certified_blocked_production_is_replaceable() {
  ActorPlan denied{0, {4, 4}, {harvest(1)}, {false}};
  auto denied_result = compile(
      {denied}, {{20, {4, 4}, {water(1)}, 0, 5, true}});
  check(denied_result.assignments.empty() &&
            denied_result.manifest[0][0] == harvest(1),
        "uncertified production was replaced");

  denied.blocked_equivalent_capacity[0] = true;
  auto allowed_result = compile(
      {denied}, {{20, {4, 4}, {water(1)}, 0, 5, true}});
  check(allowed_result.manifest[0][0] == water(1),
        "certified state-equivalent capacity was not used");
}

void objective_move_and_unfunded_plant_fail_closed() {
  ActorPlan actor{0, {1, 1}, {pass(), pass()}, {false, false}};
  const auto move_result = compile(
      {actor}, {{30, {1, 1}, {move(3), water(0)}, 1, 10, true}});
  check(move_result.assignments.empty() && move_result.move_slots_exact &&
            move_result.unscheduled_objectives ==
                std::vector<std::uint64_t>{30},
        "objective MOVE was not rejected");

  const auto no_seed = compile(
      {actor}, {{31, {1, 1}, {plant(2)}, 1, 10, true}});
  check(no_seed.assignments.empty() &&
            no_seed.unscheduled_objectives ==
                std::vector<std::uint64_t>{31},
        "zero-seed PLANT was scheduled");
  const auto funded = compile(
      {actor}, {{31, {1, 1}, {plant(2)}, 1, 10, true}},
      ResourceSnapshot{{{2, 1}}});
  check(funded.assignments.size() == 1 &&
            funded.manifest[0][0] == plant(2),
        "reserved seed was not used exactly once");
}

void failed_receipt_reallocates_only_future_suffix() {
  ActorPlan left{0, {0, 0}, {pass(), move(3), pass()},
                 {false, false, false}};
  ActorPlan right{1, {0, 0}, {Action{Op::Other, -1, 1, 9, 0}, pass(), pass()},
                  {false, false, false}};
  const Objective objective{40, {0, 0}, {water(0)}, 2, 10, true};
  const auto prior = compile({left, right}, {objective});
  check(prior.assignments.size() == 1 && prior.assignments[0].actor == 0 &&
            prior.assignments[0].turn == 0,
        "initial objective was not assigned to earliest capacity");
  const auto replanned = recompile_suffix(
      {left, right}, {objective}, {}, 0, prior, {{40, 0, false}});
  check(replanned.manifest[0][0] == water(0) &&
            replanned.manifest[0][1] == move(3) &&
            replanned.move_slots_exact && replanned.assignments.size() == 2 &&
            replanned.assignments[1].actor == 1 &&
            replanned.assignments[1].turn == 1,
        "failed receipt rewrote history or did not reallocate future suffix");
}

void post_receipt_resource_snapshot_controls_suffix() {
  ActorPlan actor{0, {3, 3}, {pass(), pass(), pass()},
                  {false, false, false}};
  const Objective objective{50, {3, 3}, {plant(1), water(1)}, 2, 10, true};
  const ResourceSnapshot one_seed{{{1, 1}}};
  const auto prior = compile({actor}, {objective}, one_seed);
  check(prior.assignments.size() == 2,
        "funded objective suffix was not initially scheduled");

  const auto confirmed = recompile_suffix(
      {actor}, {objective}, ResourceSnapshot{}, 0, prior, {{50, 0, true}});
  check(confirmed.manifest[0][0] == plant(1) &&
            confirmed.manifest[0][1] == water(1) &&
            confirmed.unscheduled_objectives.empty(),
        "confirmed PLANT did not retire before zero-seed suffix re-solve");

  const auto failed = recompile_suffix(
      {actor}, {objective}, ResourceSnapshot{}, 0, prior, {{50, 0, false}});
  check(failed.manifest[0][0] == plant(1) &&
            failed.manifest[0][1] == pass() &&
            failed.unscheduled_objectives ==
                std::vector<std::uint64_t>{50},
        "failed zero-seed PLANT did not freeze history and fail closed");
}

void route_sequence_delays_move_and_recomputes_position() {
  ActorPlan actor{0, {7, 7},
                  {pass(), move(3), pass(), harvest(0)},
                  {false, false, false, false}};
  const Objective objective{60, {7, 7}, {plant(0), water(0)}, 3, 10, true};
  const auto result = compile_route_sequence(
      {actor}, {objective}, ResourceSnapshot{{{0, 1}}});
  check(result.completed_objectives == 1 &&
            result.manifest[0] ==
                std::vector<Action>{plant(0), water(0), move(3), harvest(0)} &&
            result.positions_before[0][3] == Position{7, 8} &&
            result.ordered_move_sequences_exact && !result.move_slots_exact &&
            result.move_timing_deviation == 1 &&
            result.terminal_unexecuted_moves == 0 &&
            result.terminal_penalty == 0,
        "route-sequence mode did not delay/recompute the whole suffix");
}

void exact_dp_improves_value_over_greedy_priority() {
  ActorPlan actor{0, {1, 1}, {pass(), pass()}, {false, false}};
  const std::vector<Objective> objectives{
      {70, {1, 1}, {water(0), harvest(0)}, 1, 1, true},
      {71, {1, 1}, {harvest(0)}, 1, 10, false}};
  const auto greedy = compile_route_sequence({actor}, objectives);
  const auto exact = compile_time_expanded({actor}, objectives);
  check(greedy.completed_objectives == 1 &&
            greedy.assignments.front().objective_id == 70,
        "greedy comparison fixture did not select critical suffix");
  check(exact.exact && !exact.fell_back_to_greedy &&
            exact.completed_objectives == 1 &&
            exact.completed_objective_value == 10 &&
            exact.assignments.front().objective_id == 71 &&
            exact.total_cost < 10 && exact.expanded_states > 0,
        "time-expanded DP did not minimize unfinished objective value");
}

void exact_dp_matches_deadline_seed_route_oracle() {
  ActorPlan actor{0, {3, 3},
                  {pass(), pass(), move(3), move(2), pass()},
                  {false, false, false, false, false}};
  const std::vector<Objective> objectives{
      {1144, {3, 3}, {plant(0)}, 1, 9, true},
      {1145, {3, 3}, {water(0)}, 2, 1, true},
      {1146, {3, 3}, {water(0)}, 1, 7, false}};
  const auto result = compile_time_expanded(
      {actor}, objectives, ResourceSnapshot{{{0, 2}}});
  if (result.total_cost != 1) {
    std::cerr << "oracle regression cost=" << result.total_cost
              << " completed_value=" << result.completed_objective_value
              << " timing=" << result.timing_deviation_cost
              << " assignments=" << result.assignments.size() << '\n';
  }
  check(result.exact && result.total_cost == 1 &&
            result.completed_objective_value == 16 &&
            result.timing_deviation_cost == 0,
        "exact DP missed deadline/seed/route oracle");
}

void maturity_wait_requires_fresh_harvest_proof() {
  ActorPlan actor{0, {5, 5}, {pass(), pass()}, {false, false}};
  Objective water_then_wait{
      80, {5, 5}, {water(0), harvest(0)}, 1, 10, true};
  const auto waiting = compile_time_expanded(
      {actor}, {water_then_wait}, {},
      ObservationReadiness{41, {{{80, 1}}}, {}, {}});
  check(waiting.assignments.size() == 1 &&
            waiting.assignments[0].action == water(0) &&
            waiting.completed_objectives == 0 &&
            waiting.waiting_objectives == std::vector<std::uint64_t>{80},
        "WATERED_IMMATURE did not stop at the observation wait node");

  Objective mature{80, {5, 5}, {harvest(0)}, 1, 10, true};
  ObservationReadiness fresh{42, {{{80, 0}}}, {}, {}};
  fresh.harvest_legal.insert({42, 80, 0});
  const auto harvested = compile_time_expanded(
      {actor}, {mature}, {}, fresh);
  check(harvested.assignments.size() == 1 &&
            harvested.assignments[0].action == harvest(0) &&
            harvested.completed_objectives == 1 &&
            harvested.waiting_objectives.empty() &&
            harvested.observation_epoch == 42 &&
            harvested.consumed_harvest_legal == fresh.harvest_legal,
        "fresh harvest_legal proof did not unlock suffix re-solve");
  fresh.consumed_harvest_legal = harvested.consumed_harvest_legal;
  const auto replay = compile_time_expanded({actor}, {mature}, {}, fresh);
  check(replay.completed_objectives == 0 &&
            replay.waiting_objectives == std::vector<std::uint64_t>{80},
        "consumed harvest proof was replayed");
  bool epochless_rejected = false;
  try {
    ObservationReadiness epochless{0, {{{80, 0}}}, {}, {}};
    epochless.harvest_legal.insert({0, 80, 0});
    (void)compile_time_expanded({actor}, {mature}, {}, epochless);
  } catch (const std::invalid_argument&) {
    epochless_rejected = true;
  }
  check(epochless_rejected,
        "epochless harvest proof was accepted as a fresh observation");
}

void multi_actor_joint_search_and_large_fallback_are_safe() {
  ActorPlan left{0, {0, 0}, {pass(), pass()}, {false, false}};
  ActorPlan right{1, {0, 0}, {pass(), pass()}, {false, false}};
  const std::vector<Objective> objectives{
      {90, {0, 0}, {water(0), harvest(0)}, 1, 1, true},
      {91, {0, 0}, {harvest(0)}, 1, 10, false}};
  const auto joint = compile_time_expanded({left, right}, objectives);
  check(!joint.exact && !joint.fell_back_to_greedy &&
            joint.joint_search_components == 1 &&
            joint.exact_component_objectives == 0 &&
            joint.joint_search_objectives == 2 &&
            joint.planning_elapsed_us < 1'000'000 &&
            joint.completed_objective_value == 10 &&
            joint.assignments.size() == 1 &&
            joint.assignments[0].objective_id == 91,
        "bounded multi-actor subset search did not improve objective value");

  ActorPlan third{2, {0, 0}, {pass(), pass()}, {false, false}};
  ActorPlan fourth{3, {0, 0}, {pass(), pass()}, {false, false}};
  const auto greedy = compile_route_sequence(
      {left, right, third, fourth}, objectives);
  const auto bounded = compile_time_expanded(
      {left, right, third, fourth}, objectives);
  check(!bounded.exact && bounded.fell_back_to_greedy &&
            bounded.fallback_components == 1 &&
            bounded.within_time_budget &&
            bounded.manifest == greedy.manifest &&
            bounded.assignments == greedy.assignments,
        "large multi-actor fallback was not bit-identical to greedy baseline");

  ActorPlan crop_left{10, {3, 3}, {pass()}, {false}};
  ActorPlan crop_right{11, {7, 7}, {pass()}, {false}};
  const auto reserved = compile_time_expanded(
      {crop_left, crop_right},
      {{100, {3, 3}, {plant(0)}, 0, 5, true},
       {101, {7, 7}, {plant(0)}, 0, 4, true}},
      ResourceSnapshot{{{0, 1}}});
  check(reserved.assignments.size() == 1 &&
            reserved.assignments[0].action == plant(0) &&
            reserved.unscheduled_objectives.size() == 1,
        "decomposed components overbooked globally shared seed inventory");
}

void objective_ids_are_nonzero_and_unique_at_every_entry() {
  ActorPlan actor{0, {0, 0}, {pass()}, {false}};
  const Objective one{120, {0, 0}, {water(0)}, 0, 1, true};
  const std::vector<Objective> duplicate{one, one};
  const Objective zero{0, {0, 0}, {water(0)}, 0, 1, true};
  auto rejects = [](const auto& call) {
    try {
      call();
      return false;
    } catch (const std::invalid_argument&) {
      return true;
    }
  };
  check(rejects([&] { (void)compile({actor}, duplicate); }) &&
            rejects([&] {
              (void)compile_route_sequence({actor}, duplicate);
            }) &&
            rejects([&] {
              (void)compile_time_expanded({actor}, duplicate);
            }) &&
            rejects([&] { (void)compile({actor}, {zero}); }),
        "planner accepted a zero or duplicate objective id");
  const auto prior = compile({actor}, {one});
  check(rejects([&] {
          (void)recompile_suffix({actor}, duplicate, {}, -1, prior, {});
        }),
        "suffix planner accepted duplicate objective ids");
}

}  // namespace

int main() try {
  move_timeline_is_hard_frozen();
  critical_objective_wins_contested_capacity();
  only_certified_blocked_production_is_replaceable();
  objective_move_and_unfunded_plant_fail_closed();
  failed_receipt_reallocates_only_future_suffix();
  post_receipt_resource_snapshot_controls_suffix();
  route_sequence_delays_move_and_recomputes_position();
  exact_dp_improves_value_over_greedy_priority();
  exact_dp_matches_deadline_seed_route_oracle();
  maturity_wait_requires_fresh_harvest_proof();
  multi_actor_joint_search_and_large_fallback_are_safe();
  objective_ids_are_nonzero_and_unique_at_every_entry();
  std::cout << "day-horizon planner: 12 MVP/DP fixtures passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
