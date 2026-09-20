#include "../include/day_horizon_planner.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

namespace g001::day_horizon_repair {
namespace {

using event_local_repair::Op;

Position after_move(Position position, const Action& action) {
  if (action.op != Op::Move) return position;
  switch (action.arg0) {
    case 0: --position.row; break;
    case 1: ++position.row; break;
    case 2: --position.column; break;
    case 3: ++position.column; break;
    default: throw std::invalid_argument("unknown MOVE direction");
  }
  return position;
}

bool higher_priority(const Objective& left, const Objective& right) {
  if (left.critical != right.critical) return left.critical > right.critical;
  if (left.deadline_turn != right.deadline_turn)
    return left.deadline_turn < right.deadline_turn;
  if (left.value != right.value) return left.value > right.value;
  return left.id < right.id;
}

bool valid_transition(const Action& action) {
  return action.op == Op::Dig || action.op == Op::Plant ||
      action.op == Op::Build || action.op == Op::Water ||
      action.op == Op::Harvest;
}

void validate_objective_ids(const std::vector<Objective>& objectives) {
  std::set<std::uint64_t> ids;
  for (const auto& objective : objectives)
    if (objective.id == 0 || !ids.insert(objective.id).second)
      throw std::invalid_argument("objective ids must be nonzero and unique");
}

}  // namespace

Result compile(const std::vector<ActorPlan>& actors,
               std::vector<Objective> objectives,
               const ResourceSnapshot& resources) {
  validate_objective_ids(objectives);
  if (actors.empty()) throw std::invalid_argument("empty actor horizon");
  const auto turns = actors.front().raw.size();
  if (turns == 0) throw std::invalid_argument("empty day horizon");

  Result result;
  result.manifest.reserve(actors.size());
  result.positions_before.resize(actors.size());
  std::set<int> actor_ids;
  for (std::size_t actor = 0; actor < actors.size(); ++actor) {
    const auto& plan = actors[actor];
    if (plan.actor < 0 || plan.raw.size() != turns ||
        plan.blocked_equivalent_capacity.size() != turns ||
        !actor_ids.insert(plan.actor).second)
      throw std::invalid_argument("inconsistent actor horizon");
    result.manifest.push_back(plan.raw);
    auto position = plan.start;
    auto& timeline = result.positions_before[actor];
    timeline.reserve(turns);
    for (const auto& action : plan.raw) {
      timeline.push_back(position);
      position = after_move(position, action);
    }
  }

  std::sort(objectives.begin(), objectives.end(), higher_priority);
  auto available_seeds = resources.seeds;
  // Raw PLANT demand owns inventory before repair. Conservatively reserve all
  // such demand, including a blocked slot that may ultimately be replaced.
  for (const auto& actor : actors)
    for (const auto& action : actor.raw)
      if (action.op == Op::Plant && action.item >= 0 && action.quantity > 0)
        available_seeds[action.item] = std::max(
            0, available_seeds[action.item] - action.quantity);
  std::set<std::pair<int, int>> occupied_slots;
  std::set<std::tuple<int, int, int>> occupied_tile_turns;
  std::map<std::tuple<int, int, int>, int> raw_tile_effect_owner;
  // Retained raw production owns its tile receipt before repair is scheduled.
  // Otherwise an earlier PASS actor could receive a repair effect on the same
  // tile/turn as a later actor's raw effect, making settlement order-dependent.
  for (std::size_t actor = 0; actor < actors.size(); ++actor)
    for (std::size_t turn = 0; turn < turns; ++turn)
      if (valid_transition(actors[actor].raw[turn])) {
        const auto& tile = result.positions_before[actor][turn];
        const auto key =
            std::tuple{tile.row, tile.column, static_cast<int>(turn)};
        const int replaceable_owner =
            actors[actor].blocked_equivalent_capacity[turn]
            ? static_cast<int>(actor) : -1;
        const auto [entry, inserted] =
            raw_tile_effect_owner.emplace(key, replaceable_owner);
        // Multiple raw effects already make receipt causality ambiguous. No
        // repair is then allowed to replace only one member of that group.
        if (!inserted) entry->second = -1;
      }
  for (const auto& objective : objectives) {
    if (objective.id == 0 || objective.remaining_transitions.empty() ||
        objective.deadline_turn < 0 ||
        !std::all_of(objective.remaining_transitions.begin(),
                     objective.remaining_transitions.end(),
                     valid_transition)) {
      result.unscheduled_objectives.push_back(objective.id);
      continue;
    }
    std::vector<Assignment> tentative;
    std::map<int, int> tentative_seed_demand;
    int after_turn = -1;
    for (std::size_t transition = 0;
         transition < objective.remaining_transitions.size(); ++transition) {
      const auto& transition_action =
          objective.remaining_transitions[transition];
      if (transition_action.op == Op::Plant) {
        const int item = transition_action.item;
        const int demand = transition_action.quantity;
        if (item < 0 || demand <= 0 ||
            available_seeds[item] - tentative_seed_demand[item] < demand) {
          tentative.clear();
          break;
        }
        tentative_seed_demand[item] += demand;
      }
      bool found = false;
      for (int turn = after_turn + 1;
           turn < static_cast<int>(turns) && turn <= objective.deadline_turn &&
           !found;
           ++turn) {
        for (std::size_t actor = 0; actor < actors.size(); ++actor) {
          if (!(result.positions_before[actor][static_cast<std::size_t>(turn)] ==
                objective.tile))
            continue;
          const auto tile_turn = std::tuple{
              objective.tile.row, objective.tile.column, turn};
          const auto raw_owner = raw_tile_effect_owner.find(tile_turn);
          if (occupied_slots.contains({static_cast<int>(actor), turn}) ||
              occupied_tile_turns.contains(tile_turn) ||
              (raw_owner != raw_tile_effect_owner.end() &&
               raw_owner->second != static_cast<int>(actor)))
            continue;
          const auto& raw = actors[actor].raw[static_cast<std::size_t>(turn)];
          const bool capacity = raw.op == Op::Pass ||
              actors[actor].blocked_equivalent_capacity[
                  static_cast<std::size_t>(turn)];
          if (!capacity || raw.op == Op::Move) continue;
          tentative.push_back(
              {objective.id, static_cast<int>(transition), actors[actor].actor,
               turn, objective.remaining_transitions[transition]});
          after_turn = turn;
          found = true;
          break;
        }
      }
      if (!found) {
        tentative.clear();
        break;
      }
    }
    if (tentative.empty()) {
      result.unscheduled_objectives.push_back(objective.id);
      continue;
    }
    for (const auto& [item, demand] : tentative_seed_demand)
      available_seeds[item] -= demand;
    for (const auto& assignment : tentative) {
      const auto actor = static_cast<std::size_t>(std::distance(
          actors.begin(), std::find_if(
              actors.begin(), actors.end(), [&](const ActorPlan& plan) {
                return plan.actor == assignment.actor;
              })));
      if (actor == actors.size()) throw std::logic_error("actor disappeared");
      result.manifest[actor][static_cast<std::size_t>(assignment.turn)] =
          assignment.action;
      occupied_slots.insert({static_cast<int>(actor), assignment.turn});
      occupied_tile_turns.insert(
          {objective.tile.row, objective.tile.column, assignment.turn});
      result.assignments.push_back(assignment);
    }
  }

  result.move_slots_exact = true;
  for (std::size_t actor = 0; actor < actors.size(); ++actor)
    for (std::size_t turn = 0; turn < turns; ++turn)
      if ((actors[actor].raw[turn].op == Op::Move ||
           result.manifest[actor][turn].op == Op::Move) &&
          !(result.manifest[actor][turn] == actors[actor].raw[turn]))
        result.move_slots_exact = false;
  return result;
}

Result recompile_suffix(const std::vector<ActorPlan>& actors,
                        std::vector<Objective> objectives,
                        const ResourceSnapshot& resources,
                        int frozen_through_turn, const Result& prior,
                        const std::vector<ReceiptOutcome>& receipts) {
  validate_objective_ids(objectives);
  if (actors.empty()) throw std::invalid_argument("empty actor horizon");
  const int turns = static_cast<int>(actors.front().raw.size());
  if (frozen_through_turn < -1 || frozen_through_turn >= turns)
    throw std::invalid_argument("invalid frozen prefix");
  if (prior.manifest.size() != actors.size())
    throw std::invalid_argument("prior manifest actor mismatch");
  for (std::size_t actor = 0; actor < actors.size(); ++actor)
    if (prior.manifest[actor].size() != actors[actor].raw.size())
      throw std::invalid_argument("prior manifest horizon mismatch");

  for (auto& objective : objectives) {
    std::vector<bool> confirmed(objective.remaining_transitions.size(), false);
    for (const auto& receipt : receipts) {
      if (receipt.objective_id != objective.id || !receipt.confirmed) continue;
      if (receipt.transition_index < 0 ||
          receipt.transition_index >= static_cast<int>(confirmed.size()))
        throw std::invalid_argument("receipt transition outside objective");
      confirmed[static_cast<std::size_t>(receipt.transition_index)] = true;
    }
    std::size_t prefix = 0;
    while (prefix < confirmed.size() && confirmed[prefix]) ++prefix;
    objective.remaining_transitions.erase(
        objective.remaining_transitions.begin(),
        objective.remaining_transitions.begin() +
            static_cast<std::ptrdiff_t>(prefix));
  }
  objectives.erase(
      std::remove_if(objectives.begin(), objectives.end(),
                     [](const Objective& objective) {
                       return objective.remaining_transitions.empty();
                     }),
      objectives.end());

  const int suffix_begin = frozen_through_turn + 1;
  auto full_raw = compile(actors, {});
  Result result = full_raw;
  for (std::size_t actor = 0; actor < actors.size(); ++actor)
    for (int turn = 0; turn <= frozen_through_turn; ++turn)
      result.manifest[actor][static_cast<std::size_t>(turn)] =
          prior.manifest[actor][static_cast<std::size_t>(turn)];
  for (const auto& assignment : prior.assignments)
    if (assignment.turn <= frozen_through_turn)
      result.assignments.push_back(assignment);

  if (suffix_begin < turns && !objectives.empty()) {
    std::vector<ActorPlan> suffix_actors;
    suffix_actors.reserve(actors.size());
    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      ActorPlan suffix;
      suffix.actor = actors[actor].actor;
      suffix.start = full_raw.positions_before[actor][
          static_cast<std::size_t>(suffix_begin)];
      suffix.raw.assign(actors[actor].raw.begin() + suffix_begin,
                        actors[actor].raw.end());
      suffix.blocked_equivalent_capacity.assign(
          actors[actor].blocked_equivalent_capacity.begin() + suffix_begin,
          actors[actor].blocked_equivalent_capacity.end());
      suffix_actors.push_back(std::move(suffix));
    }
    for (auto& objective : objectives)
      objective.deadline_turn -= suffix_begin;
    const auto suffix = compile(suffix_actors, objectives, resources);
    for (std::size_t actor = 0; actor < actors.size(); ++actor)
      for (std::size_t turn = 0; turn < suffix.manifest[actor].size(); ++turn)
        result.manifest[actor][static_cast<std::size_t>(suffix_begin) + turn] =
            suffix.manifest[actor][turn];
    for (auto assignment : suffix.assignments) {
      assignment.turn += suffix_begin;
      result.assignments.push_back(std::move(assignment));
    }
    result.unscheduled_objectives = suffix.unscheduled_objectives;
  } else {
    for (const auto& objective : objectives)
      result.unscheduled_objectives.push_back(objective.id);
  }

  result.move_slots_exact = true;
  for (std::size_t actor = 0; actor < actors.size(); ++actor)
    for (int turn = 0; turn < turns; ++turn)
      if ((actors[actor].raw[static_cast<std::size_t>(turn)].op == Op::Move ||
           result.manifest[actor][static_cast<std::size_t>(turn)].op ==
               Op::Move) &&
          !(actors[actor].raw[static_cast<std::size_t>(turn)] ==
            result.manifest[actor][static_cast<std::size_t>(turn)]))
        result.move_slots_exact = false;
  return result;
}

RouteSequenceResult compile_route_sequence(
    const std::vector<ActorPlan>& actors, std::vector<Objective> objectives,
    const ResourceSnapshot& resources, RouteSequenceConfig config) {
  validate_objective_ids(objectives);
  if (actors.empty()) throw std::invalid_argument("empty actor horizon");
  if (config.move_timing_deviation_cost < 0 ||
      config.unfinished_move_penalty < 0)
    throw std::invalid_argument("negative route-sequence cost");
  const int turns = static_cast<int>(actors.front().raw.size());
  if (turns <= 0) throw std::invalid_argument("empty day horizon");

  struct QueuedRaw { Action action; int original_turn{}; };
  struct ActorState {
    Position position;
    std::vector<QueuedRaw> queue;
    std::size_t cursor{};
    std::optional<std::size_t> objective;
    std::size_t objective_transition{};
  };

  RouteSequenceResult result;
  result.manifest.assign(actors.size(),
                         std::vector<Action>(static_cast<std::size_t>(turns)));
  result.positions_before.resize(actors.size());
  std::set<int> actor_ids;
  std::vector<ActorState> states(actors.size());
  auto available_seeds = resources.seeds;
  for (std::size_t actor = 0; actor < actors.size(); ++actor) {
    if (actors[actor].actor < 0 ||
        static_cast<int>(actors[actor].raw.size()) != turns ||
        static_cast<int>(actors[actor].blocked_equivalent_capacity.size()) !=
            turns ||
        !actor_ids.insert(actors[actor].actor).second)
      throw std::invalid_argument("inconsistent actor horizon");
    states[actor].position = actors[actor].start;
    for (int turn = 0; turn < turns; ++turn) {
      const auto action = actors[actor].raw[static_cast<std::size_t>(turn)];
      if (action.op == Op::Pass) continue;
      states[actor].queue.push_back({action, turn});
      if (action.op == Op::Plant && action.item >= 0 && action.quantity > 0)
        available_seeds[action.item] = std::max(
            0, available_seeds[action.item] - action.quantity);
    }
    result.positions_before[actor].reserve(static_cast<std::size_t>(turns));
  }
  std::sort(objectives.begin(), objectives.end(), higher_priority);
  std::vector<bool> claimed(objectives.size(), false);
  std::vector<bool> completed(objectives.size(), false);
  std::set<Position> active_objective_tiles;

  auto objective_funded = [&](const Objective& objective) {
    std::map<int, int> demand;
    for (const auto& action : objective.remaining_transitions) {
      if (!valid_transition(action)) return false;
      if (action.op != Op::Plant) continue;
      if (action.item < 0 || action.quantity <= 0) return false;
      demand[action.item] += action.quantity;
      if (demand[action.item] > available_seeds[action.item]) return false;
    }
    return true;
  };
  auto reserve_objective = [&](const Objective& objective) {
    for (const auto& action : objective.remaining_transitions)
      if (action.op == Op::Plant)
        available_seeds[action.item] -= action.quantity;
  };
  auto suffix_has_raw_tile_conflict =
      [&](std::size_t claiming_actor, int begin_turn,
          const Objective& candidate) {
        const int end_turn = begin_turn + static_cast<int>(
            candidate.remaining_transitions.size()) - 1;
        for (std::size_t actor = 0; actor < states.size(); ++actor) {
          const auto& state = states[actor];
          Position position = state.position;
          int next_turn = begin_turn;
          // Existing complete objective leases and the candidate itself consume
          // consecutive slots before that actor can resume its ordered raw
          // queue. Production does not change position.
          if (actor == claiming_actor) {
            next_turn = end_turn + 1;
          } else if (state.objective.has_value()) {
            const auto& active = objectives[*state.objective];
            next_turn += static_cast<int>(active.remaining_transitions.size() -
                                          state.objective_transition);
          }
          for (std::size_t cursor = state.cursor;
               cursor < state.queue.size(); ++cursor) {
            const auto& queued = state.queue[cursor];
            const int execution_turn =
                std::max(next_turn, queued.original_turn);
            if (execution_turn > end_turn) break;
            if (execution_turn >= begin_turn &&
                valid_transition(queued.action) &&
                position == candidate.tile)
              return true;
            position = after_move(position, queued.action);
            next_turn = execution_turn + 1;
          }
        }
        return false;
      };

  for (int turn = 0; turn < turns; ++turn) {
    std::set<Position> raw_reserved_tiles;
    std::set<Position> committed_tile_effects;
    // Raw queue effects have priority over repair for receipt causality. A due
    // raw effect is reserved up front, independent of actor iteration order.
    // This may conservatively leave a slot unused if another repair delays the
    // raw action, but never emits two effects on one tile/turn.
    for (const auto& state : states)
      if (state.cursor < state.queue.size() &&
          state.queue[state.cursor].original_turn <= turn &&
          valid_transition(state.queue[state.cursor].action))
        raw_reserved_tiles.insert(state.position);
    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      auto& state = states[actor];
      result.positions_before[actor].push_back(state.position);
      Action selected;
      bool objective_action = false;

      if (state.objective.has_value()) {
        const auto objective = *state.objective;
        const auto& active = objectives[objective];
        const int transitions_left = static_cast<int>(
            active.remaining_transitions.size() -
            state.objective_transition);
        if (turn + transitions_left - 1 > active.deadline_turn) {
          for (std::size_t transition = state.objective_transition;
               transition < active.remaining_transitions.size(); ++transition) {
            const auto& action = active.remaining_transitions[transition];
            if (action.op == Op::Plant)
              available_seeds[action.item] += action.quantity;
          }
          active_objective_tiles.erase(active.tile);
          state.objective.reset();
        }
      }

      if (!state.objective.has_value()) {
        const int raw_remaining = static_cast<int>(state.queue.size() -
                                                   state.cursor);
        const int slack = turns - turn - raw_remaining;
        for (std::size_t objective = 0; objective < objectives.size();
             ++objective) {
          const auto& candidate = objectives[objective];
          if (claimed[objective] || candidate.remaining_transitions.empty() ||
              !(candidate.tile == state.position) ||
              active_objective_tiles.contains(candidate.tile) ||
              raw_reserved_tiles.contains(candidate.tile) ||
              committed_tile_effects.contains(candidate.tile) ||
              candidate.deadline_turn <
                  turn + static_cast<int>(
                             candidate.remaining_transitions.size()) - 1 ||
              static_cast<int>(candidate.remaining_transitions.size()) >
                  slack ||
              suffix_has_raw_tile_conflict(actor, turn, candidate) ||
              !objective_funded(candidate))
            continue;
          claimed[objective] = true;
          active_objective_tiles.insert(candidate.tile);
          reserve_objective(candidate);
          state.objective = objective;
          state.objective_transition = 0;
          break;
        }
      }

      if (state.objective.has_value()) {
        const auto objective = *state.objective;
        const auto& candidate = objectives[objective];
        if (!committed_tile_effects.contains(candidate.tile)) {
          selected = candidate.remaining_transitions[
              state.objective_transition];
          objective_action = true;
          committed_tile_effects.insert(candidate.tile);
          result.assignments.push_back(
              {candidate.id, static_cast<int>(state.objective_transition),
               actors[actor].actor, turn, selected});
          ++state.objective_transition;
          if (state.objective_transition ==
              candidate.remaining_transitions.size()) {
            ++result.completed_objectives;
            completed[objective] = true;
            active_objective_tiles.erase(candidate.tile);
            state.objective.reset();
          }
        }
      }

      if (!objective_action && !state.objective.has_value() &&
          state.cursor < state.queue.size() &&
          state.queue[state.cursor].original_turn <= turn) {
        const auto& due = state.queue[state.cursor];
        const bool effect_conflict = valid_transition(due.action) &&
            (active_objective_tiles.contains(state.position) ||
             committed_tile_effects.contains(state.position));
        if (!effect_conflict) {
          const auto queued = state.queue[state.cursor++];
          selected = queued.action;
          if (valid_transition(selected))
            committed_tile_effects.insert(state.position);
          if (selected.op == Op::Move) {
            const int deviation = turn - queued.original_turn;
            result.move_timing_deviation += deviation;
          }
        }
      }
      result.manifest[actor][static_cast<std::size_t>(turn)] = selected;
      state.position = after_move(state.position, selected);
    }
  }

  for (std::size_t objective = 0; objective < objectives.size(); ++objective)
    if (!completed[objective])
      result.unscheduled_objectives.push_back(objectives[objective].id);

  result.ordered_move_sequences_exact = true;
  result.move_slots_exact = true;
  for (std::size_t actor = 0; actor < actors.size(); ++actor) {
    std::vector<Action> raw_moves;
    std::vector<Action> compiled_moves;
    for (const auto& action : actors[actor].raw)
      if (action.op == Op::Move) raw_moves.push_back(action);
    for (std::size_t turn = 0; turn < result.manifest[actor].size(); ++turn) {
      const auto& action = result.manifest[actor][turn];
      if (action.op == Op::Move) compiled_moves.push_back(action);
      if ((actors[actor].raw[turn].op == Op::Move || action.op == Op::Move) &&
          !(actors[actor].raw[turn] == action))
        result.move_slots_exact = false;
    }
    if (raw_moves != compiled_moves) result.ordered_move_sequences_exact = false;
    for (std::size_t cursor = states[actor].cursor;
         cursor < states[actor].queue.size(); ++cursor) {
      ++result.terminal_unexecuted_raw_actions;
      if (states[actor].queue[cursor].action.op == Op::Move)
        ++result.terminal_unexecuted_moves;
    }
  }
  result.timing_deviation_cost = result.move_timing_deviation *
      config.move_timing_deviation_cost;
  result.terminal_penalty = result.terminal_unexecuted_moves *
      config.unfinished_move_penalty;
  if (config.fail_on_unfinished_route &&
      (!result.ordered_move_sequences_exact ||
       result.terminal_unexecuted_raw_actions != 0))
    throw std::runtime_error("route-sequence completion hard gate failed");
  return result;
}

TimeExpandedResult compile_time_expanded(
    const std::vector<ActorPlan>& actors, std::vector<Objective> objectives,
    const ResourceSnapshot& resources, const ObservationReadiness& readiness,
    TimeExpandedConfig config) {
  validate_objective_ids(objectives);
  if (config.move_timing_deviation_cost < 0 ||
      config.unfinished_objective_value_cost < 0 ||
      config.terminal_route_hard_penalty <= 0 || config.max_states == 0 ||
      config.time_budget_ms <= 0)
    throw std::invalid_argument("invalid time-expanded configuration");
  if ((!readiness.harvest_legal_required.empty() ||
       !readiness.harvest_legal.empty()) &&
      readiness.observation_epoch == 0)
    throw std::invalid_argument("maturity proof has no observation epoch");
  for (const auto& proof : readiness.harvest_legal)
    if (proof.observation_epoch != readiness.observation_epoch ||
        !readiness.harvest_legal_required.contains(
            {proof.objective_id, proof.transition_index}))
      throw std::invalid_argument("unrequested harvest proof");
  for (const auto& [objective_id, transition] :
       readiness.harvest_legal_required) {
    const auto objective = std::find_if(
        objectives.begin(), objectives.end(), [&](const Objective& candidate) {
          return candidate.id == objective_id;
        });
    if (objective == objectives.end() || transition < 0 ||
        transition >= static_cast<int>(
            objective->remaining_transitions.size()) ||
        objective->remaining_transitions[static_cast<std::size_t>(transition)].op !=
            Op::Harvest)
      throw std::invalid_argument("invalid maturity wait transition");
  }

  struct PlanningObjective {
    Objective objective;
    std::size_t original_transition_count{};
    bool reaches_terminal{};
    int progress_value{};
  };
  std::vector<PlanningObjective> planning;
  std::vector<std::uint64_t> initially_waiting;
  int total_objective_value = 0;
  for (auto objective : objectives) {
    total_objective_value += std::max(0, objective.value);
    if (objective.remaining_transitions.empty()) {
      planning.push_back(
          {std::move(objective), 0, true, 0});
      continue;
    }
    std::size_t ready = 0;
    for (; ready < objective.remaining_transitions.size(); ++ready) {
      const auto transition =
          std::pair{objective.id, static_cast<int>(ready)};
      const HarvestProofToken proof{
          readiness.observation_epoch, objective.id, static_cast<int>(ready)};
      if (readiness.harvest_legal_required.contains(transition) &&
          (!readiness.harvest_legal.contains(proof) ||
           readiness.consumed_harvest_legal.contains(proof)))
        break;
      if (readiness.harvest_legal_required.contains(transition) &&
          objective.remaining_transitions[ready].op != Op::Harvest)
        throw std::invalid_argument("harvest proof gates a non-HARVEST action");
    }
    if (ready == 0) {
      initially_waiting.push_back(objective.id);
      continue;
    }
    const auto original_count = objective.remaining_transitions.size();
    const bool terminal = ready == original_count;
    objective.remaining_transitions.resize(ready);
    const int progress_value = terminal ? std::max(0, objective.value) :
        std::max(1, std::max(0, objective.value) *
                        static_cast<int>(ready) /
                        static_cast<int>(original_count));
    planning.push_back(
        {std::move(objective), original_count, terminal, progress_value});
  }

  std::vector<Objective> ready_objectives;
  ready_objectives.reserve(planning.size());
  for (const auto& item : planning) ready_objectives.push_back(item.objective);
  const auto greedy = compile_route_sequence(
      actors, ready_objectives, resources,
      RouteSequenceConfig{config.move_timing_deviation_cost,
                          config.terminal_route_hard_penalty, true});

  auto summarize = [&](const RouteSequenceResult& base,
                       bool exact, bool fallback,
                       std::size_t expanded, bool within_budget) {
    TimeExpandedResult out;
    static_cast<RouteSequenceResult&>(out) = base;
    out.exact = exact;
    out.fell_back_to_greedy = fallback;
    out.expanded_states = expanded;
    out.within_time_budget = within_budget;
    out.observation_epoch = readiness.observation_epoch;
    out.waiting_objectives = initially_waiting;
    std::map<std::uint64_t, int> assignment_counts;
    for (const auto& assignment : out.assignments)
      ++assignment_counts[assignment.objective_id];
    for (const auto& assignment : out.assignments) {
      const auto transition = std::pair{
          assignment.objective_id, assignment.transition_index};
      if (!readiness.harvest_legal_required.contains(transition)) continue;
      const HarvestProofToken proof{
          readiness.observation_epoch, assignment.objective_id,
          assignment.transition_index};
      if (readiness.harvest_legal.contains(proof) &&
          !readiness.consumed_harvest_legal.contains(proof))
        out.consumed_harvest_legal.insert(proof);
    }
    int progress_value = 0;
    out.completed_objectives = 0;
    out.completed_objective_value = 0;
    for (const auto& item : planning) {
      const bool scheduled = !item.objective.remaining_transitions.empty() &&
          assignment_counts[item.objective.id] ==
          static_cast<int>(item.objective.remaining_transitions.size());
      if (!scheduled) continue;
      progress_value += item.progress_value;
      if (item.reaches_terminal) {
        ++out.completed_objectives;
        out.completed_objective_value += std::max(0, item.objective.value);
      } else {
        out.waiting_objectives.push_back(item.objective.id);
      }
    }
    std::sort(out.waiting_objectives.begin(), out.waiting_objectives.end());
    out.waiting_objectives.erase(
        std::unique(out.waiting_objectives.begin(),
                    out.waiting_objectives.end()),
        out.waiting_objectives.end());
    out.unfinished_objective_value =
        std::max(0, total_objective_value - progress_value);
    out.total_cost = out.timing_deviation_cost +
        out.unfinished_objective_value *
            config.unfinished_objective_value_cost +
        out.terminal_unexecuted_raw_actions *
            config.terminal_route_hard_penalty;
    return out;
  };

  if (actors.size() > 1) {
    const auto decomposition_started = std::chrono::steady_clock::now();
    const auto budget_exhausted = [&] {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - decomposition_started).count() >=
          config.time_budget_ms;
    };
    std::vector<std::set<Position>> reachable(actors.size());
    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      auto position = actors[actor].start;
      reachable[actor].insert(position);
      for (const auto& action : actors[actor].raw) {
        position = after_move(position, action);
        reachable[actor].insert(position);
      }
    }
    std::vector<std::vector<std::size_t>> eligible(planning.size());
    for (std::size_t objective = 0; objective < planning.size(); ++objective)
      for (std::size_t actor = 0; actor < actors.size(); ++actor)
        if (reachable[actor].contains(planning[objective].objective.tile))
          eligible[objective].push_back(actor);

    std::vector<std::size_t> parent(planning.size());
    for (std::size_t index = 0; index < parent.size(); ++index)
      parent[index] = index;
    std::function<std::size_t(std::size_t)> root = [&](std::size_t value) {
      return parent[value] == value ? value : parent[value] = root(parent[value]);
    };
    auto unite = [&](std::size_t left, std::size_t right) {
      left = root(left);
      right = root(right);
      if (left != right) parent[right] = left;
    };
    for (std::size_t left = 0; left < planning.size(); ++left)
      for (std::size_t right = left + 1; right < planning.size(); ++right) {
        bool shares_actor = false;
        for (const auto actor : eligible[left])
          if (std::find(eligible[right].begin(), eligible[right].end(), actor) !=
              eligible[right].end()) {
            shares_actor = true;
            break;
          }
        bool shares_seed_resource = false;
        for (const auto& left_action :
             planning[left].objective.remaining_transitions) {
          if (left_action.op != Op::Plant || left_action.item < 0) continue;
          if (std::any_of(
                  planning[right].objective.remaining_transitions.begin(),
                  planning[right].objective.remaining_transitions.end(),
                  [&](const Action& right_action) {
                    return right_action.op == Op::Plant &&
                        right_action.item == left_action.item;
                  })) {
            shares_seed_resource = true;
            break;
          }
        }
        if (shares_actor || planning[left].objective.tile ==
                                planning[right].objective.tile ||
            shares_seed_resource)
          unite(left, right);
      }
    std::map<std::size_t, std::vector<std::size_t>> component_map;
    for (std::size_t index = 0; index < planning.size(); ++index)
      component_map[root(index)].push_back(index);
    std::vector<std::vector<std::size_t>> components;
    for (auto& [unused, component] : component_map) {
      (void)unused;
      components.push_back(std::move(component));
    }
    std::sort(components.begin(), components.end(), [&](const auto& left,
                                                        const auto& right) {
      const auto value = [&](const auto& component) {
        int result = 0;
        for (const auto index : component)
          result += std::max(0, planning[index].objective.value);
        return result;
      };
      return value(left) > value(right);
    });

    auto remaining_seeds = resources.seeds;
    for (const auto& actor : actors)
      for (const auto& action : actor.raw)
        if (action.op == Op::Plant && action.item >= 0 && action.quantity > 0)
          remaining_seeds[action.item] = std::max(
              0, remaining_seeds[action.item] - action.quantity);
    RouteSequenceResult merged;
    merged.manifest.reserve(actors.size());
    for (const auto& actor : actors) merged.manifest.push_back(actor.raw);
    merged.positions_before.assign(actors.size(), {});
    int single_exact = 0;
    int joint_exact = 0;
    int joint_objectives = 0;
    int fallback_components = 0;
    int exact_objectives = 0;
    std::size_t expanded = 0;

    for (const auto& component : components) {
      std::set<std::size_t> actor_set;
      for (const auto objective : component)
        actor_set.insert(eligible[objective].begin(), eligible[objective].end());
      std::vector<std::size_t> actor_indices(actor_set.begin(), actor_set.end());
      if (actor_indices.empty()) {
        ++fallback_components;
        for (const auto objective : component)
          merged.unscheduled_objectives.push_back(
              planning[objective].objective.id);
        continue;
      }
      std::vector<ActorPlan> local_actors;
      for (const auto actor : actor_indices)
        local_actors.push_back(actors[actor]);
      ResourceSnapshot local_resources{remaining_seeds};
      // The component compiler reserves its own raw planting. Add that amount
      // back because remaining_seeds already reflects the global raw reserve.
      for (const auto& actor : local_actors)
        for (const auto& action : actor.raw)
          if (action.op == Op::Plant && action.item >= 0 && action.quantity > 0)
            local_resources.seeds[action.item] += action.quantity;
      std::vector<Objective> local_objectives;
      for (const auto objective : component)
        local_objectives.push_back(planning[objective].objective);

      RouteSequenceResult selected;
      bool component_exact = false;
      if (actor_indices.size() == 1 && component.size() <= 20 &&
          !budget_exhausted() && expanded < config.max_states) {
        auto local_config = config;
        local_config.max_states = config.max_states - expanded;
        local_config.time_budget_ms = std::max(
            1, config.time_budget_ms - static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - decomposition_started)
                    .count()));
        const auto exact = compile_time_expanded(
            local_actors, local_objectives, local_resources, {}, local_config);
        static_cast<RouteSequenceResult&>(selected) = exact;
        expanded += exact.expanded_states;
        component_exact = exact.exact;
        if (component_exact) ++single_exact;
      } else if (actor_indices.size() <= 3 && component.size() <= 7 &&
                 !budget_exhausted() && expanded < config.max_states) {
        bool have_best = false;
        int best_cost = std::numeric_limits<int>::max();
        const std::uint64_t subsets = std::uint64_t{1} << component.size();
        std::uint64_t searched = 0;
        for (std::uint64_t mask = 0; mask < subsets; ++mask) {
          if (budget_exhausted() || expanded >= config.max_states) break;
          std::vector<Objective> subset;
          int selected_value = 0;
          for (std::size_t bit = 0; bit < component.size(); ++bit)
            if ((mask & (std::uint64_t{1} << bit)) != 0) {
              subset.push_back(local_objectives[bit]);
              selected_value += std::max(0, local_objectives[bit].value);
            }
          const auto candidate = compile_route_sequence(
              local_actors, subset, local_resources);
          std::set<std::uint64_t> completed_ids;
          std::map<std::uint64_t, int> counts;
          for (const auto& assignment : candidate.assignments)
            ++counts[assignment.objective_id];
          int completed_value = 0;
          for (const auto& objective : subset)
            if (counts[objective.id] == static_cast<int>(
                    objective.remaining_transitions.size())) {
              completed_ids.insert(objective.id);
              completed_value += std::max(0, objective.value);
            }
          const int component_value = [&] {
            int value = 0;
            for (const auto& objective : local_objectives)
              value += std::max(0, objective.value);
            return value;
          }();
          const int cost = (component_value - completed_value) *
                  config.unfinished_objective_value_cost +
              candidate.move_timing_deviation *
                  config.move_timing_deviation_cost;
          if (!have_best || cost < best_cost) {
            selected = candidate;
            best_cost = cost;
            have_best = true;
            selected.unscheduled_objectives.clear();
            for (const auto& objective : local_objectives)
              if (!completed_ids.contains(objective.id))
                selected.unscheduled_objectives.push_back(objective.id);
          }
          (void)selected_value;
          ++searched;
          ++expanded;
        }
        component_exact = searched == subsets;
        if (component_exact) {
          ++joint_exact;
          joint_objectives += static_cast<int>(component.size());
        }
      }
      if (!component_exact) {
        selected = compile_route_sequence(
            local_actors, local_objectives, local_resources);
        ++fallback_components;
      } else if (actor_indices.size() == 1) {
        exact_objectives += static_cast<int>(component.size());
      }

      for (std::size_t local_actor = 0; local_actor < actor_indices.size();
           ++local_actor)
        merged.manifest[actor_indices[local_actor]] =
            selected.manifest[local_actor];
      merged.assignments.insert(merged.assignments.end(),
                                selected.assignments.begin(),
                                selected.assignments.end());
      merged.unscheduled_objectives.insert(
          merged.unscheduled_objectives.end(),
          selected.unscheduled_objectives.begin(),
          selected.unscheduled_objectives.end());
      for (const auto& assignment : selected.assignments)
        if (assignment.action.op == Op::Plant)
          remaining_seeds[assignment.action.item] -= assignment.action.quantity;
    }

    bool merge_valid = true;
    std::set<std::tuple<int, int, int>> tile_turns;
    std::set<std::pair<int, int>> repair_slots;
    for (const auto& assignment : merged.assignments)
      repair_slots.insert({assignment.actor, assignment.turn});
    merged.positions_before.assign(actors.size(), {});
    merged.ordered_move_sequences_exact = true;
    merged.move_slots_exact = true;
    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      Position position = actors[actor].start;
      std::vector<Action> raw_moves;
      std::vector<Action> compiled_moves;
      std::vector<Action> raw_non_pass;
      std::vector<Action> compiled_raw_non_pass;
      for (const auto& action : actors[actor].raw)
        if (action.op != Op::Pass) {
          raw_non_pass.push_back(action);
          if (action.op == Op::Move) raw_moves.push_back(action);
        }
      for (std::size_t turn = 0; turn < actors[actor].raw.size(); ++turn) {
        merged.positions_before[actor].push_back(position);
        const auto& action = merged.manifest[actor][turn];
        if (action.op == Op::Move) compiled_moves.push_back(action);
        if (action.op != Op::Pass &&
            !repair_slots.contains(
                {actors[actor].actor, static_cast<int>(turn)}))
          compiled_raw_non_pass.push_back(action);
        if ((actors[actor].raw[turn].op == Op::Move || action.op == Op::Move) &&
            !(actors[actor].raw[turn] == action))
          merged.move_slots_exact = false;
        if (valid_transition(action) &&
            !tile_turns.insert({position.row, position.column,
                                static_cast<int>(turn)}).second)
          merge_valid = false;
        position = after_move(position, action);
      }
      if (raw_moves != compiled_moves) {
        merge_valid = false;
        merged.ordered_move_sequences_exact = false;
      }
      if (raw_non_pass != compiled_raw_non_pass) merge_valid = false;
      std::size_t raw_move = 0;
      for (std::size_t turn = 0; turn < merged.manifest[actor].size(); ++turn)
        if (merged.manifest[actor][turn].op == Op::Move) {
          while (raw_move < actors[actor].raw.size() &&
                 actors[actor].raw[raw_move].op != Op::Move)
            ++raw_move;
          if (raw_move >= actors[actor].raw.size()) {
            merge_valid = false;
            break;
          }
          if (turn < raw_move) {
            merge_valid = false;
          } else {
            merged.move_timing_deviation +=
                static_cast<int>(turn - raw_move);
          }
          ++raw_move;
        }
    }
    for (const auto& [item, count] : remaining_seeds) {
      (void)item;
      if (count < 0) merge_valid = false;
    }
    merged.timing_deviation_cost = merged.move_timing_deviation *
        config.move_timing_deviation_cost;
    merged.completed_objectives = static_cast<int>(planning.size()) -
        static_cast<int>(merged.unscheduled_objectives.size());

    auto output = merge_valid
        ? summarize(merged, fallback_components == 0 && joint_exact == 0,
                    fallback_components != 0, expanded, !budget_exhausted())
        : summarize(greedy, false, true, expanded, !budget_exhausted());
    const auto greedy_summary = summarize(
        greedy, false, true, expanded, !budget_exhausted());
    if (!merge_valid ||
        output.completed_objectives < greedy_summary.completed_objectives ||
        output.completed_objective_value <
            greedy_summary.completed_objective_value ||
        output.timing_deviation_cost >
            greedy_summary.timing_deviation_cost ||
        output.terminal_unexecuted_raw_actions >
            greedy_summary.terminal_unexecuted_raw_actions)
      output = greedy_summary;
    output.conflict_components = static_cast<int>(components.size());
    output.single_actor_exact_components = single_exact;
    output.joint_search_components = joint_exact;
    output.joint_search_objectives = joint_objectives;
    output.fallback_components = fallback_components + !merge_valid;
    output.exact_component_objectives = exact_objectives;
    for (const auto& component : components) {
      output.largest_component_objectives = std::max(
          output.largest_component_objectives,
          static_cast<int>(component.size()));
      ++output.component_objective_size_histogram[
          static_cast<int>(component.size())];
    }
    output.planning_elapsed_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - decomposition_started).count());
    return output;
  }

  // The exact kernel uses a 64-bit handled-objective mask. Larger components
  // retain the verified greedy manifest.
  if (planning.size() > 20 || actors.front().raw.size() > 32)
    return summarize(greedy, false, true, 0, true);

  struct QueuedRaw { Action action; int original_turn{}; };
  std::vector<QueuedRaw> raw_queue;
  auto available_seeds = resources.seeds;
  for (int turn = 0; turn < static_cast<int>(actors[0].raw.size()); ++turn) {
    const auto& action = actors[0].raw[static_cast<std::size_t>(turn)];
    if (action.op == Op::Pass) continue;
    raw_queue.push_back({action, turn});
    if (action.op == Op::Plant && action.item >= 0 && action.quantity > 0)
      available_seeds[action.item] = std::max(
          0, available_seeds[action.item] - action.quantity);
  }
  std::vector<int> seed_items;
  for (const auto& item : planning)
    for (const auto& action : item.objective.remaining_transitions)
      if (action.op == Op::Plant && action.item >= 0 &&
          std::find(seed_items.begin(), seed_items.end(), action.item) ==
              seed_items.end())
        seed_items.push_back(action.item);
  std::sort(seed_items.begin(), seed_items.end());
  std::vector<int> initial_seed_counts;
  for (int item : seed_items) initial_seed_counts.push_back(available_seeds[item]);

  struct State {
    int turn{};
    std::size_t route_cursor{};
    Position position;
    std::uint64_t objective_mask{};
    std::vector<int> seeds;
  };
  struct StateLess {
    bool operator()(const State& left, const State& right) const {
      return std::tie(left.turn, left.route_cursor, left.position.row,
                      left.position.column, left.objective_mask, left.seeds) <
          std::tie(right.turn, right.route_cursor, right.position.row,
                   right.position.column, right.objective_mask, right.seeds);
    }
  };
  enum class Choice : std::uint8_t { None, Pass, Raw, Objective };
  struct Best {
    bool valid{};
    int progress_value{std::numeric_limits<int>::min()};
    int move_delay{std::numeric_limits<int>::max()};
    Choice choice{Choice::None};
    int objective{-1};
  };
  std::map<State, Best, StateLess> memo;
  const auto started = std::chrono::steady_clock::now();
  bool aborted = false;
  auto seed_index = [&](int item) {
    const auto found = std::lower_bound(seed_items.begin(), seed_items.end(), item);
    return static_cast<std::size_t>(found - seed_items.begin());
  };
  auto better = [&](const Best& candidate, const Best& incumbent) {
    const auto utility = [&](const Best& value) {
      return static_cast<long long>(value.progress_value) *
              config.unfinished_objective_value_cost -
          static_cast<long long>(value.move_delay) *
              config.move_timing_deviation_cost;
    };
    return candidate.valid && (!incumbent.valid ||
        utility(candidate) > utility(incumbent) ||
        (utility(candidate) == utility(incumbent) &&
         (candidate.progress_value > incumbent.progress_value ||
          (candidate.progress_value == incumbent.progress_value &&
           candidate.move_delay < incumbent.move_delay))));
  };
  std::function<Best(const State&)> solve = [&](const State& state) -> Best {
    if (aborted) return {};
    if (memo.size() >= config.max_states ||
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count() >=
            config.time_budget_ms) {
      aborted = true;
      return {};
    }
    if (const auto found = memo.find(state); found != memo.end())
      return found->second;
    const int turns = static_cast<int>(actors[0].raw.size());
    if (state.turn == turns) {
      Best terminal;
      terminal.valid = state.route_cursor == raw_queue.size();
      terminal.progress_value = terminal.valid ? 0 :
          std::numeric_limits<int>::min();
      terminal.move_delay = terminal.valid ? 0 :
          std::numeric_limits<int>::max();
      memo.emplace(state, terminal);
      return terminal;
    }
    Best best;
    auto consider = [&](State next, Choice choice, int objective,
                        int reward, int delay) {
      auto suffix = solve(next);
      if (!suffix.valid) return;
      suffix.progress_value += reward;
      suffix.move_delay += delay;
      suffix.choice = choice;
      suffix.objective = objective;
      if (better(suffix, best)) best = suffix;
    };

    State pass_state = state;
    ++pass_state.turn;
    consider(std::move(pass_state), Choice::Pass, -1, 0, 0);

    if (state.route_cursor < raw_queue.size() &&
        raw_queue[state.route_cursor].original_turn <= state.turn) {
      State next = state;
      const auto& queued = raw_queue[state.route_cursor];
      ++next.turn;
      ++next.route_cursor;
      next.position = after_move(next.position, queued.action);
      const int delay = queued.action.op == Op::Move
          ? state.turn - queued.original_turn : 0;
      consider(std::move(next), Choice::Raw, -1, 0, delay);
    }

    for (std::size_t index = 0; index < planning.size(); ++index) {
      if ((state.objective_mask & (std::uint64_t{1} << index)) != 0) continue;
      const auto& item = planning[index];
      const auto length = static_cast<int>(
          item.objective.remaining_transitions.size());
      if (!(item.objective.tile == state.position) || length <= 0 ||
          state.turn + length > turns ||
          state.turn + length - 1 > item.objective.deadline_turn ||
          turns - (state.turn + length) <
              static_cast<int>(raw_queue.size() - state.route_cursor))
        continue;
      State next = state;
      bool funded = true;
      for (const auto& action : item.objective.remaining_transitions) {
        if (!valid_transition(action)) { funded = false; break; }
        if (action.op != Op::Plant) continue;
        if (action.item < 0 || action.quantity <= 0) {
          funded = false;
          break;
        }
        const auto which = seed_index(action.item);
        if (which >= next.seeds.size() ||
            next.seeds[which] < action.quantity) {
          funded = false;
          break;
        }
        next.seeds[which] -= action.quantity;
      }
      if (!funded) continue;
      next.turn += length;
      next.objective_mask |= std::uint64_t{1} << index;
      consider(std::move(next), Choice::Objective,
               static_cast<int>(index), item.progress_value, 0);
    }
    memo.emplace(state, best);
    return best;
  };

  State initial{0, 0, actors[0].start, 0, initial_seed_counts};
  const auto optimum = solve(initial);
  if (aborted || !optimum.valid)
    return summarize(greedy, false, true, memo.size(), !aborted);

  RouteSequenceResult exact_result;
  const int turns = static_cast<int>(actors[0].raw.size());
  exact_result.manifest.assign(1, std::vector<Action>(turns));
  exact_result.positions_before.assign(1, {});
  exact_result.positions_before[0].reserve(turns);
  State state = initial;
  std::vector<bool> scheduled(planning.size(), false);
  while (state.turn < turns) {
    const auto found = memo.find(state);
    if (found == memo.end() || !found->second.valid)
      throw std::logic_error("time-expanded reconstruction lost state");
    const auto decision = found->second;
    if (decision.choice == Choice::Objective) {
      const auto index = static_cast<std::size_t>(decision.objective);
      const auto& item = planning[index];
      scheduled[index] = true;
      for (std::size_t transition = 0;
           transition < item.objective.remaining_transitions.size();
           ++transition) {
        exact_result.positions_before[0].push_back(state.position);
        exact_result.manifest[0][static_cast<std::size_t>(state.turn)] =
            item.objective.remaining_transitions[transition];
        exact_result.assignments.push_back(
            {item.objective.id, static_cast<int>(transition), actors[0].actor,
             state.turn, item.objective.remaining_transitions[transition]});
        if (item.objective.remaining_transitions[transition].op == Op::Plant) {
          const auto which = seed_index(
              item.objective.remaining_transitions[transition].item);
          state.seeds[which] -=
              item.objective.remaining_transitions[transition].quantity;
        }
        ++state.turn;
      }
      state.objective_mask |= std::uint64_t{1} << index;
    } else {
      exact_result.positions_before[0].push_back(state.position);
      if (decision.choice == Choice::Raw) {
        const auto& queued = raw_queue[state.route_cursor++];
        exact_result.manifest[0][static_cast<std::size_t>(state.turn)] =
            queued.action;
        if (queued.action.op == Op::Move)
          exact_result.move_timing_deviation +=
              state.turn - queued.original_turn;
        state.position = after_move(state.position, queued.action);
      }
      ++state.turn;
    }
  }
  exact_result.ordered_move_sequences_exact =
      state.route_cursor == raw_queue.size();
  exact_result.move_slots_exact = exact_result.manifest[0] == actors[0].raw;
  exact_result.timing_deviation_cost = exact_result.move_timing_deviation *
      config.move_timing_deviation_cost;
  for (std::size_t index = 0; index < planning.size(); ++index)
    if (!scheduled[index])
      exact_result.unscheduled_objectives.push_back(
          planning[index].objective.id);

  auto exact_summary = summarize(exact_result, true, false, memo.size(), true);
  auto greedy_summary = summarize(greedy, false, true, memo.size(), true);
  if (exact_summary.total_cost > greedy_summary.total_cost)
    return greedy_summary;
  return exact_summary;
}

}  // namespace g001::day_horizon_repair
