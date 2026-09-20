#include "component_shadow_mapper.hpp"
#include "exact_move_slot_planner.hpp"
#include "event_local_elastic_day.hpp"
#include "native_teammate.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace shadow = fastkag::component_shadow;
namespace local = g001::day_horizon_repair;
using fastkag::Action;
using fastkag::Config;
using fastkag::NativeAgentState;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;

struct Options {
  std::uint64_t forced_seed{25772138000ULL};
  std::uint64_t normal_seed_begin{25772138701ULL};
  int normal_seeds{32};
  std::string output{"component-shadow-eval.json"};
  std::string tapes{NATIVE_G001_TAPES};
  std::string library{NATIVE_G001_LIBRARY};
};

struct CertificateCounts {
  std::array<std::array<std::uint64_t, 8>, 18> by_op{};
};

struct Summary {
  struct ElasticAggregate {
    std::uint64_t windows{};
    std::uint64_t assignments{};
    std::uint64_t closed{};
    std::uint64_t terminal_debt{};
    std::uint64_t purchase_requests{};
    std::uint64_t delayed_moves{};
    std::uint64_t total_move_delay{};
    std::uint64_t maximum_move_delay{};
    std::uint64_t absorbed_pass_sinks{};
    std::uint64_t absorbed_no_effect_sinks{};
    std::uint64_t route_failures{};
    std::uint64_t terminal_raw_failures{};
    std::uint64_t capacity_proof_failures{};
  };
  int games{};
  std::uint64_t unit_actions{};
  std::uint64_t segments{};
  std::uint64_t merge_safe_segments{};
  std::uint64_t sealed_components{};
  std::uint64_t global_opaque_segments{};
  std::uint64_t objectives{};
  std::uint64_t safe_repair_assignments{};
  std::uint64_t committed_objectives{};
  std::uint64_t unscheduled_objectives{};
  std::uint64_t conservation_failures{};
  std::uint64_t route_validator_failures{};
  std::uint64_t position_validator_failures{};
  std::uint64_t potential_move_delay{};
  std::uint64_t potentially_delayed_moves{};
  std::uint64_t maximum_potential_move_delay{};
  std::map<int, std::uint64_t> component_node_size_histogram;
  CertificateCounts certificates;
  std::uint64_t prefix_authority_failures{};
  std::uint64_t exact_slot_segments{};
  std::uint64_t exact_slot_absorbed_intents{};
  std::uint64_t exact_slot_assignments{};
  std::uint64_t exact_slot_completed_intents{};
  std::uint64_t exact_slot_cross_day_debt_created{};
  std::uint64_t exact_slot_cross_day_debt_retired{};
  std::uint64_t exact_slot_debt_lost{};
  std::uint64_t exact_slot_terminal_active_debt{};
  std::uint64_t exact_slot_move_slot_changes{};
  std::uint64_t exact_slot_move_payload_changes{};
  std::uint64_t exact_slot_position_failures{};
  std::map<local::ExactSlotReject, std::uint64_t> exact_slot_rejections;
  std::uint64_t deviation_segments{};
  std::uint64_t deviation_confirmed_triggers{};
  std::uint64_t deviation_assignments{};
  std::uint64_t deviation_completed_intents{};
  std::uint64_t deviation_debt_lost{};
  std::uint64_t deviation_terminal_active_debt{};
  std::uint64_t deviation_cross_day_debt_created{};
  std::uint64_t deviation_cross_day_debt_retired{};
  std::uint64_t deviation_move_slot_changes{};
  std::uint64_t deviation_move_payload_changes{};
  std::uint64_t deviation_position_failures{};
  std::uint64_t deviation_state_equivalent_changes{};
  std::uint64_t deviation_certified_service_changes{};
  std::map<g001::event_local_repair::Op, std::uint64_t>
      deviation_raw_nonmove_changes;
  std::map<local::ExactSlotReject, std::uint64_t> deviation_rejections;
  std::set<std::tuple<int, int, int>> deviation_affected_actor_days;
  std::set<std::tuple<int, int, int, int>> deviation_affected_tiles;
  std::uint64_t seed_upper_segments{};
  std::uint64_t seed_upper_confirmed_triggers{};
  std::uint64_t seed_upper_assignments{};
  std::uint64_t seed_upper_completed_intents{};
  std::uint64_t seed_upper_cross_day_debt_created{};
  std::uint64_t seed_upper_cross_day_debt_retired{};
  std::uint64_t seed_upper_debt_lost{};
  std::uint64_t seed_upper_terminal_active_debt{};
  std::uint64_t seed_upper_move_failures{};
  std::map<local::ExactSlotReject, std::uint64_t> seed_upper_rejections;
  ElasticAggregate elastic_observed;
  ElasticAggregate elastic_seed_upper;
};

struct Frame {
  Simulator before;
  PlayerAction raw;
};

local::Position local_position(Position position) {
  return {position.y, position.x};
}

Position actor_position(const Simulator& simulator, int player, int actor) {
  if (actor == 0) return simulator.farms()[player].farmer;
  return simulator.farms()[player]
      .hands[static_cast<std::size_t>(actor - 1)];
}

local::Position after_move(local::Position position,
                           const local::Action& action) {
  if (action.op != g001::event_local_repair::Op::Move) return position;
  if (action.arg0 == 0) --position.row;
  else if (action.arg0 == 1) ++position.row;
  else if (action.arg0 == 2) --position.column;
  else if (action.arg0 == 3) ++position.column;
  else throw std::invalid_argument("invalid shadow MOVE direction");
  return position;
}

bool same_transitions(const std::vector<local::Action>& left,
                      const std::vector<local::Action>& right) {
  return left == right;
}

local::Action local_action(const Action& action) {
  local::Action result;
  result.item = static_cast<int>(action.item);
  result.quantity = action.quantity;
  switch (action.op) {
    case Op::PASS: result.op = g001::event_local_repair::Op::Pass; break;
    case Op::NORTH: result.op = g001::event_local_repair::Op::Move;
      result.arg0 = 0; break;
    case Op::SOUTH: result.op = g001::event_local_repair::Op::Move;
      result.arg0 = 1; break;
    case Op::WEST: result.op = g001::event_local_repair::Op::Move;
      result.arg0 = 2; break;
    case Op::EAST: result.op = g001::event_local_repair::Op::Move;
      result.arg0 = 3; break;
    case Op::DIG: result.op = g001::event_local_repair::Op::Dig; break;
    case Op::PLANT: result.op = g001::event_local_repair::Op::Plant; break;
    case Op::WATER: result.op = g001::event_local_repair::Op::Water; break;
    case Op::HARVEST: result.op = g001::event_local_repair::Op::Harvest; break;
    default: result.op = g001::event_local_repair::Op::Other;
      result.arg0 = static_cast<int>(action.op); break;
  }
  return result;
}

g001::event_local_repair::TileObservation local_tile(
    const fastkag::Tile& tile, int day) {
  using Kind = g001::event_local_repair::TileKind;
  g001::event_local_repair::TileObservation result;
  if (tile.kind == fastkag::TileKind::EMPTY) result.kind = Kind::Empty;
  else if (tile.kind == fastkag::TileKind::WEED) result.kind = Kind::Weed;
  else if (tile.kind == fastkag::TileKind::PLANT) result.kind = Kind::Crop;
  else if (tile.kind == fastkag::TileKind::COOP ||
           tile.kind == fastkag::TileKind::PASTURE)
    result.kind = Kind::Structure;
  else result.kind = Kind::Other;
  if (tile.kind == fastkag::TileKind::PLANT) {
    static constexpr int first_day[fastkag::N_CROPS] = {2, 2, 8, 10, 10};
    result.item = static_cast<int>(tile.crop);
    result.harvest_legal = result.item >= 0 && result.item < fastkag::N_CROPS &&
        tile.yield_units > 0 && day - tile.planted_day >= first_day[result.item];
  }
  result.watered_today = tile.watered_today;
  return result;
}

const char* exact_reject_name(local::ExactSlotReject reason) {
  switch (reason) {
    case local::ExactSlotReject::MissingDesiredCrop: return "missing_desired_crop";
    case local::ExactSlotReject::SeedUnavailable: return "seed_unavailable";
    case local::ExactSlotReject::MaturityWait: return "maturity_wait";
    case local::ExactSlotReject::TileUnsupported: return "tile_unsupported";
    case local::ExactSlotReject::TileSerialized: return "tile_serialized";
    case local::ExactSlotReject::OngoingHarvestUnsupported:
      return "ongoing_harvest_unsupported";
    case local::ExactSlotReject::GlobalSeedManifestUnsafe:
      return "global_seed_manifest_unsafe";
  }
  return "invalid";
}

const char* local_op_name(g001::event_local_repair::Op op) {
  using RepairOp = g001::event_local_repair::Op;
  switch (op) {
    case RepairOp::Pass: return "PASS";
    case RepairOp::Move: return "MOVE";
    case RepairOp::Dig: return "DIG";
    case RepairOp::Plant: return "PLANT";
    case RepairOp::Build: return "BUILD";
    case RepairOp::Water: return "WATER";
    case RepairOp::Harvest: return "HARVEST";
    case RepairOp::Other: return "OTHER";
  }
  return "INVALID";
}

Options parse(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) throw std::invalid_argument("missing option value");
      return argv[index];
    };
    if (argument == "--forced-seed") options.forced_seed = std::stoull(value());
    else if (argument == "--normal-seed-begin")
      options.normal_seed_begin = std::stoull(value());
    else if (argument == "--normal-seeds")
      options.normal_seeds = std::stoi(value());
    else if (argument == "--output") options.output = value();
    else if (argument == "--tapes") options.tapes = value();
    else if (argument == "--library") options.library = value();
    else throw std::invalid_argument("unknown option: " + argument);
  }
  if (options.normal_seeds <= 0)
    throw std::invalid_argument("normal-seeds must be positive");
  return options;
}

std::vector<Frame> collect(
    const NativeTeammateExecutor& executor, std::uint64_t seed, int seat,
    double weed_rate) {
  Config config;
  config.episode_steps = 720;
  config.weed_spawn_chance = weed_rate;
  Simulator simulator(config, seed);
  std::array<NativeAgentState, 2> states;
  std::vector<Frame> frames;
  frames.reserve(719);
  while (!simulator.done()) {
    std::array<PlayerAction, 2> actions{
        executor.action_external(simulator, 0, 0, states[0]),
        executor.action_external(simulator, 1, 0, states[1])};
    frames.push_back({simulator, actions[static_cast<std::size_t>(seat)]});
    simulator.step(actions);
  }
  return frames;
}

std::map<int, int> component_sizes(
    const std::vector<local::ComponentActorPlan>& actors,
    const std::vector<local::Objective>& objectives) {
  const std::size_t nodes = actors.size() + objectives.size();
  std::vector<std::size_t> parent(nodes);
  for (std::size_t node = 0; node < nodes; ++node) parent[node] = node;
  const auto root = [&](auto&& self, std::size_t node) -> std::size_t {
    return parent[node] == node ? node : parent[node] = self(self, parent[node]);
  };
  auto unite = [&](std::size_t left, std::size_t right) {
    left = root(root, left);
    right = root(root, right);
    if (left != right) parent[right] = left;
  };
  std::vector<std::set<local::Position>> reachable(actors.size());
  std::vector<std::set<int>> resources(actors.size());
  bool global_opaque = false;
  for (std::size_t actor = 0; actor < actors.size(); ++actor) {
    auto position = actors[actor].start;
    reachable[actor].insert(position);
    for (const auto& source : actors[actor].ordered_raw) {
      position = after_move(position, source.action);
      reachable[actor].insert(position);
      if (source.action.op == g001::event_local_repair::Op::Plant &&
          source.action.item >= 0)
        resources[actor].insert(source.action.item);
      if (source.effect_certificate.has_value()) {
        const auto& certificate = *source.effect_certificate;
        resources[actor].insert(certificate.resource_reads.begin(),
                                certificate.resource_reads.end());
        for (const auto& [item, unused] : certificate.resource_consumption) {
          (void)unused;
          resources[actor].insert(item);
        }
        for (const auto& [item, unused] : certificate.resource_production) {
          (void)unused;
          resources[actor].insert(item);
        }
      } else if (source.unsupported &&
                 source.action.op == g001::event_local_repair::Op::Other) {
        global_opaque = true;
      }
    }
  }
  if (global_opaque && nodes > 0)
    for (std::size_t node = 1; node < nodes; ++node) unite(0, node);
  for (std::size_t left = 0; left < actors.size(); ++left)
    for (std::size_t right = left + 1; right < actors.size(); ++right)
      for (const auto& tile : reachable[left])
        if (reachable[right].contains(tile)) unite(left, right);
  for (std::size_t objective = 0; objective < objectives.size(); ++objective) {
    std::set<int> items;
    for (const auto& action : objectives[objective].remaining_transitions)
      if (action.op == g001::event_local_repair::Op::Plant)
        items.insert(action.item);
    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      bool resource_edge = false;
      for (const int item : items)
        resource_edge = resource_edge || resources[actor].contains(item);
      if (reachable[actor].contains(objectives[objective].tile) || resource_edge)
        unite(actor, actors.size() + objective);
    }
    for (std::size_t other = objective + 1; other < objectives.size(); ++other) {
      bool shared = objectives[objective].tile == objectives[other].tile;
      for (const auto& action : objectives[other].remaining_transitions)
        shared = shared ||
            (action.op == g001::event_local_repair::Op::Plant &&
             items.contains(action.item));
      if (shared) unite(actors.size() + objective, actors.size() + other);
    }
  }
  std::map<std::size_t, int> sizes;
  for (std::size_t node = 0; node < nodes; ++node)
    ++sizes[root(root, node)];
  std::map<int, int> histogram;
  for (const auto& [unused, size] : sizes) {
    (void)unused;
    ++histogram[size];
  }
  return histogram;
}

bool validate_positions(const local::ComponentScopedResult& result,
                        int board_size) {
  for (std::size_t actor = 0; actor < result.manifest.size(); ++actor)
    for (std::size_t turn = 0; turn < result.manifest[actor].size(); ++turn) {
      const auto position = result.positions_before[actor][turn];
      if (position.row < 0 || position.column < 0 ||
          position.row >= board_size || position.column >= board_size)
        return false;
      const auto after = after_move(position, result.manifest[actor][turn]);
      if (after.row < 0 || after.column < 0 || after.row >= board_size ||
          after.column >= board_size)
        return false;
    }
  return true;
}

void analyze_segment(const std::vector<Frame>& frames, int begin, int end,
                     int seat, Summary& summary,
                     std::vector<local::Objective>& carried) {
  const int turns = end - begin;
  const int actor_count = static_cast<int>(frames[static_cast<std::size_t>(begin)]
                                               .raw.units.size());
  std::vector<local::ComponentActorPlan> actors;
  actors.reserve(static_cast<std::size_t>(actor_count));
  for (int actor = 0; actor < actor_count; ++actor) {
    local::ComponentActorPlan plan;
    plan.actor = actor;
    plan.start = local_position(actor_position(
        frames[static_cast<std::size_t>(begin)].before, seat, actor));
    plan.turns = turns;
    local::BoardLegalityCertificate board;
    board.rows = frames[static_cast<std::size_t>(begin)].before.config().board_size;
    board.columns = board.rows;
    board.complete = true;
    plan.board = std::move(board);
    actors.push_back(std::move(plan));
  }
  std::vector<local::Objective> objectives = carried;
  for (auto& objective : objectives) objective.deadline_turn = turns - 1;
  std::size_t new_objectives = 0;
  std::uint64_t next_objective =
      static_cast<std::uint64_t>(begin + 1) * 1'000'000ULL;
  bool global_opaque = false;
  for (int turn = 0; turn < turns; ++turn) {
    const auto& frame = frames[static_cast<std::size_t>(begin + turn)];
    for (int actor = 0; actor < actor_count; ++actor) {
      const auto source_id =
          static_cast<std::uint64_t>(begin + turn + 1) * 64ULL +
          static_cast<std::uint64_t>(actor + 1);
      const auto mapped = shadow::map_current_unit(
          frame.before, seat, actor, frame.raw, source_id, turn);
      ++summary.unit_actions;
      ++summary.certificates.by_op[static_cast<std::size_t>(mapped.native_op)]
                                      [static_cast<std::size_t>(mapped.kind)];
      if (mapped.kind == shadow::CertificateKind::TypedEffect &&
          (!mapped.typed_effect.has_value() ||
           !mapped.typed_effect->lower_slot_prefix_bound ||
           mapped.typed_effect->prefix_hash == 0 ||
           mapped.typed_effect->actor_generation == 0))
        ++summary.prefix_authority_failures;
      if (mapped.raw_source.has_value()) {
        actors[static_cast<std::size_t>(actor)]
            .ordered_raw.push_back(*mapped.raw_source);
        if (mapped.kind == shadow::CertificateKind::GlobalOpaque)
          global_opaque = true;
        if (mapped.kind == shadow::CertificateKind::MoveTransition) {
          const auto from = local_position(actor_position(frame.before, seat, actor));
          const auto to = after_move(from, mapped.raw_source->action);
          actors[static_cast<std::size_t>(actor)]
              .board->move_transitions[source_id] = {from, to};
        }
      }
      if (!mapped.objective.has_value()) continue;
      const auto duplicate = std::find_if(
          objectives.begin(), objectives.end(), [&](const auto& objective) {
            return objective.tile == mapped.objective->tile &&
                same_transitions(objective.remaining_transitions,
                                 mapped.objective->transitions);
          });
      if (duplicate == objectives.end()) {
        objectives.push_back({++next_objective, mapped.objective->tile,
                              mapped.objective->transitions, turns - 1,
                              mapped.objective->value, true});
        ++new_objectives;
      }
    }
  }
  local::ResourceSnapshot resources;
  for (int crop = 0; crop < fastkag::N_CROPS; ++crop)
    resources.seeds[crop] = frames[static_cast<std::size_t>(begin)]
                                .before.privates()[seat].seeds[crop];
  const auto result = local::compile_component_scoped_route(
      actors, objectives, resources);
  ++summary.segments;
  summary.merge_safe_segments += result.merge_safe;
  summary.sealed_components += result.sealed_components;
  summary.global_opaque_segments += global_opaque;
  summary.objectives += new_objectives;
  summary.safe_repair_assignments +=
      result.merge_safe ? result.assignments.size() : 0;
  summary.committed_objectives += result.committed_objectives.size();
  summary.unscheduled_objectives += result.unscheduled_objectives.size();
  summary.conservation_failures += !result.objective_conservation;
  summary.route_validator_failures +=
      !result.raw_source_order_exact ||
      result.terminal_unexecuted_raw_actions != 0 ||
      !result.global_route_geometry_safe;
  summary.position_validator_failures +=
      !validate_positions(result,
          frames[static_cast<std::size_t>(begin)].before.config().board_size);
  for (const auto& [size, count] : component_sizes(actors, objectives))
    summary.component_node_size_histogram[size] += count;

  std::set<std::uint64_t> unscheduled(
      result.unscheduled_objectives.begin(),
      result.unscheduled_objectives.end());
  carried.clear();
  for (auto objective : objectives)
    if (unscheduled.contains(objective.id)) {
      objective.deadline_turn = 0;
      carried.push_back(std::move(objective));
    }

  std::map<std::uint64_t, int> move_release;
  for (const auto& actor : actors)
    for (const auto& source : actor.ordered_raw)
      if (source.action.op == g001::event_local_repair::Op::Move)
        move_release[source.source_id] = source.earliest_turn;
  for (std::size_t actor = 0; actor < result.raw_source_manifest.size(); ++actor)
    for (int turn = 0; turn < turns; ++turn) {
      const auto source = result.raw_source_manifest[actor]
          [static_cast<std::size_t>(turn)];
      const auto found = move_release.find(source);
      if (found == move_release.end()) continue;
      const int delay = turn - found->second;
      if (delay > 0) {
        summary.potential_move_delay += static_cast<std::uint64_t>(delay);
        ++summary.potentially_delayed_moves;
        summary.maximum_potential_move_delay = std::max(
            summary.maximum_potential_move_delay,
            static_cast<std::uint64_t>(delay));
      }
    }
}

void analyze_exact_slot_segment(
    const std::vector<Frame>& frames, int begin, int end, int seat,
    Summary& summary, std::vector<local::PersistentPlotIntent>& carried,
    bool deviation_only, bool seed_upper_bound = false) {
  const int turns = end - begin;
  const auto& initial = frames[static_cast<std::size_t>(begin)].before;
  const int actor_count = static_cast<int>(
      frames[static_cast<std::size_t>(begin)].raw.units.size());
  std::vector<local::ExactMoveSlotActor> actors;
  actors.reserve(static_cast<std::size_t>(actor_count));
  for (int actor = 0; actor < actor_count; ++actor) {
    local::ExactMoveSlotActor plan;
    plan.actor = actor;
    plan.start = local_position(actor_position(initial, seat, actor));
    plan.raw.resize(static_cast<std::size_t>(turns));
    plan.service_slot.resize(static_cast<std::size_t>(turns));
    if (deviation_only)
      plan.trigger_source.resize(static_cast<std::size_t>(turns));
    for (int turn = 0; turn < turns; ++turn) {
      const auto& native = frames[static_cast<std::size_t>(begin + turn)]
                               .raw.units[static_cast<std::size_t>(actor)];
      plan.raw[static_cast<std::size_t>(turn)] = local_action(native);
      if (deviation_only) {
        const auto& frame = frames[static_cast<std::size_t>(begin + turn)];
        const auto mapped = shadow::map_current_unit(
            frame.before, seat, actor, frame.raw,
            static_cast<std::uint64_t>(begin + turn + 1) * 64ULL + actor + 1,
            turn);
        const bool trigger =
            mapped.kind == shadow::CertificateKind::ReplaceableObjective;
        plan.trigger_source[static_cast<std::size_t>(turn)] = trigger;
        plan.service_slot[static_cast<std::size_t>(turn)] =
            native.op == Op::PASS || trigger;
      } else {
        plan.service_slot[static_cast<std::size_t>(turn)] =
            native.op == Op::PASS || native.op == Op::DIG ||
            native.op == Op::PLANT || native.op == Op::WATER ||
            native.op == Op::HARVEST;
      }
    }
    actors.push_back(std::move(plan));
  }
  std::map<local::Position, g001::event_local_repair::TileObservation> tiles;
  const int board = initial.config().board_size;
  for (int row = 0; row < board; ++row)
    for (int column = 0; column < board; ++column)
      tiles[{row, column}] = local_tile(
          initial.farms()[seat].tiles[static_cast<std::size_t>(row * board +
                                                               column)],
          initial.day());
  std::map<int, int> seeds;
  for (int crop = 0; crop < fastkag::N_CROPS; ++crop)
    seeds[crop] = seed_upper_bound ? 1'000'000 :
        initial.privates()[seat].seeds[crop];
  const auto incoming = carried.size();
  std::map<std::uint64_t, int> incoming_origin;
  for (const auto& intent : carried)
    incoming_origin[intent.id] = intent.origin_day;
  auto result = local::compile_exact_move_slots(
      actors, std::move(tiles), std::move(seeds), std::move(carried),
      initial.day());
  if (seed_upper_bound) {
    ++summary.seed_upper_segments;
    summary.seed_upper_confirmed_triggers += result.absorbed_raw_intents;
    summary.seed_upper_assignments += result.assignments;
    summary.seed_upper_completed_intents += result.completed_intents.size();
    summary.seed_upper_move_failures +=
        !result.move_slots_exact || !result.move_positions_exact ||
        result.original_move_slot_changes != 0 ||
        result.original_move_payload_changes != 0;
    for (const auto& [reason, count] : result.rejected)
      summary.seed_upper_rejections[reason] += count;
  } else if (deviation_only) {
    ++summary.deviation_segments;
    summary.deviation_confirmed_triggers += result.absorbed_raw_intents;
    summary.deviation_assignments += result.assignments;
    summary.deviation_completed_intents += result.completed_intents.size();
    summary.deviation_move_slot_changes += result.original_move_slot_changes;
    summary.deviation_move_payload_changes +=
        result.original_move_payload_changes;
    summary.deviation_position_failures +=
        !result.move_slots_exact || !result.move_positions_exact;
    summary.deviation_state_equivalent_changes +=
        result.state_equivalent_changes;
    summary.deviation_certified_service_changes +=
        result.certified_service_changes;
    for (const auto& [op, count] : result.raw_nonmove_changes)
      summary.deviation_raw_nonmove_changes[op] += count;
    for (const auto& [reason, count] : result.rejected)
      summary.deviation_rejections[reason] += count;
    for (const int actor : result.affected_actors)
      summary.deviation_affected_actor_days.insert(
          {summary.games, initial.day(), actor});
    for (const auto tile : result.affected_tiles)
      summary.deviation_affected_tiles.insert(
          {summary.games, initial.day(), tile.row, tile.column});
  } else {
    ++summary.exact_slot_segments;
    summary.exact_slot_absorbed_intents += result.absorbed_raw_intents;
    summary.exact_slot_assignments += result.assignments;
    summary.exact_slot_completed_intents += result.completed_intents.size();
    summary.exact_slot_move_slot_changes += result.original_move_slot_changes;
    summary.exact_slot_move_payload_changes +=
        result.original_move_payload_changes;
    summary.exact_slot_position_failures +=
        !result.move_slots_exact || !result.move_positions_exact;
    for (const auto& [reason, count] : result.rejected)
      summary.exact_slot_rejections[reason] += count;
  }
  for (const auto id : result.completed_intents)
    if (incoming_origin.contains(id) &&
        incoming_origin.at(id) < initial.day()) {
      if (seed_upper_bound) ++summary.seed_upper_cross_day_debt_retired;
      else if (deviation_only) ++summary.deviation_cross_day_debt_retired;
      else ++summary.exact_slot_cross_day_debt_retired;
    }
  const auto conserved = incoming +
      static_cast<std::size_t>(result.absorbed_raw_intents) ==
      result.active_intents.size() + result.completed_intents.size();
  if (seed_upper_bound) summary.seed_upper_debt_lost += !conserved;
  else if (deviation_only) summary.deviation_debt_lost += !conserved;
  else summary.exact_slot_debt_lost += !conserved;
  carried = std::move(result.active_intents);
}

void analyze_elastic_triggers(const std::vector<Frame>& frames, int begin,
                              int end, int seat, Summary& summary) {
  const int actor_count = static_cast<int>(
      frames[static_cast<std::size_t>(begin)].raw.units.size());
  for (int trigger_turn = begin; trigger_turn < end; ++trigger_turn) {
    const auto& trigger_frame = frames[static_cast<std::size_t>(trigger_turn)];
    for (int actor = 0; actor < actor_count; ++actor) {
      const auto mapped = shadow::map_current_unit(
          trigger_frame.before, seat, actor, trigger_frame.raw,
          static_cast<std::uint64_t>(trigger_turn + 1) * 64ULL + actor + 1,
          0);
      if (mapped.kind != shadow::CertificateKind::ReplaceableObjective ||
          !mapped.objective.has_value())
        continue;
      int desired = -1;
      for (const auto& transition : mapped.objective->transitions)
        if (transition.op == g001::event_local_repair::Op::Plant)
          desired = transition.item;
      if (desired < 0)
        desired = static_cast<int>(trigger_frame.raw.units[
            static_cast<std::size_t>(actor)].item);
      if (desired < 0 || desired >= fastkag::N_CROPS) continue;

      local::ElasticDayInput input;
      input.actor.actor = actor;
      input.actor.start = local_position(
          actor_position(trigger_frame.before, seat, actor));
      input.plot = input.actor.start;
      input.plot_state = local_tile(
          trigger_frame.before.farms()[seat].tiles[static_cast<std::size_t>(
              input.plot.row * trigger_frame.before.config().board_size +
              input.plot.column)],
          trigger_frame.before.day());
      input.desired_crop = desired;
      input.day = trigger_frame.before.day();
      for (int turn = trigger_turn; turn < end; ++turn) {
        const auto& frame = frames[static_cast<std::size_t>(turn)];
        const auto& raw = frame.raw.units[static_cast<std::size_t>(actor)];
        input.actor.raw.push_back(local_action(raw));
        input.actor.certified_sink.push_back(raw.op == Op::PASS ||
                                             turn == trigger_turn);
        input.actor.trigger_source.push_back(turn == trigger_turn);
        input.scenario_seed_receipt_by_turn.push_back(
            frame.before.privates()[seat].seeds[desired]);
      }
      const auto accumulate = [](Summary::ElasticAggregate& aggregate,
                                 const local::ElasticDayResult& result) {
        ++aggregate.windows;
        aggregate.assignments += result.assignments;
        aggregate.closed += !result.has_debt;
        aggregate.terminal_debt += result.has_debt;
        aggregate.purchase_requests += result.purchase_requested;
        aggregate.delayed_moves += result.delayed_moves;
        aggregate.total_move_delay += result.total_move_delay;
        aggregate.maximum_move_delay = std::max(
            aggregate.maximum_move_delay,
            static_cast<std::uint64_t>(result.maximum_move_delay));
        aggregate.absorbed_pass_sinks += result.absorbed_pass_sinks;
        aggregate.absorbed_no_effect_sinks +=
            result.absorbed_no_effect_sinks;
        aggregate.route_failures +=
            !result.move_source_order_exact || !result.move_payload_exact;
        aggregate.terminal_raw_failures +=
            result.terminal_move_tokens != 0 ||
            result.terminal_hard_raw_tokens != 0;
        aggregate.capacity_proof_failures += !result.capacity_proof_held;
      };
      accumulate(summary.elastic_observed,
                 local::compile_event_local_elastic_day(input));
      std::fill(input.scenario_seed_receipt_by_turn.begin(),
                input.scenario_seed_receipt_by_turn.end(), 1'000'000);
      accumulate(summary.elastic_seed_upper,
                 local::compile_event_local_elastic_day(input));
    }
  }
}

void analyze_game(const NativeTeammateExecutor& executor, std::uint64_t seed,
                  int seat, double weed_rate, Summary& summary) {
  const auto frames = collect(executor, seed, seat, weed_rate);
  ++summary.games;
  const auto opaque_tick = [&](int frame_index) {
    const auto& frame = frames[static_cast<std::size_t>(frame_index)];
    for (std::size_t actor = 0; actor < frame.raw.units.size(); ++actor) {
      const auto source =
          static_cast<std::uint64_t>(frame_index + 1) * 64ULL + actor + 1;
      if (shadow::map_current_unit(frame.before, seat,
                                  static_cast<int>(actor), frame.raw,
                                  source, 0).kind ==
          shadow::CertificateKind::GlobalOpaque)
        return true;
    }
    return false;
  };
  int begin = 0;
  int carried_day = -1;
  int exact_day = -1;
  std::vector<local::Objective> carried;
  std::vector<local::PersistentPlotIntent> exact_carried;
  std::vector<local::PersistentPlotIntent> deviation_carried;
  std::vector<local::PersistentPlotIntent> seed_upper_carried;
  while (begin < static_cast<int>(frames.size())) {
    const int day = frames[static_cast<std::size_t>(begin)].before.day();
    if (day != carried_day) {
      carried.clear();
      carried_day = day;
    }
    if (exact_day >= 0 && day != exact_day)
      summary.exact_slot_cross_day_debt_created += exact_carried.size();
    if (exact_day >= 0 && day != exact_day)
      summary.deviation_cross_day_debt_created += deviation_carried.size();
    if (exact_day >= 0 && day != exact_day)
      summary.seed_upper_cross_day_debt_created += seed_upper_carried.size();
    exact_day = day;
    const auto width = frames[static_cast<std::size_t>(begin)].raw.units.size();
    const bool opaque = opaque_tick(begin);
    int end = begin + 1;
    while (!opaque && end < static_cast<int>(frames.size()) &&
           frames[static_cast<std::size_t>(end)].before.day() == day &&
           frames[static_cast<std::size_t>(end)].raw.units.size() == width &&
           !opaque_tick(end))
      ++end;
    analyze_segment(frames, begin, end, seat, summary, carried);
    analyze_exact_slot_segment(frames, begin, end, seat, summary,
                               exact_carried, false);
    analyze_exact_slot_segment(frames, begin, end, seat, summary,
                               deviation_carried, true);
    analyze_exact_slot_segment(frames, begin, end, seat, summary,
                               seed_upper_carried, true, true);
    analyze_elastic_triggers(frames, begin, end, seat, summary);
    begin = end;
  }
  summary.exact_slot_terminal_active_debt += exact_carried.size();
  summary.deviation_terminal_active_debt += deviation_carried.size();
  summary.seed_upper_terminal_active_debt += seed_upper_carried.size();
}

void write_summary(std::ostream& output, const Summary& summary) {
  output << "{\"games\":" << summary.games
         << ",\"unit_actions\":" << summary.unit_actions
         << ",\"segments\":" << summary.segments
         << ",\"merge_safe_segments\":" << summary.merge_safe_segments
         << ",\"sealed_components\":" << summary.sealed_components
         << ",\"global_opaque_segments\":"
         << summary.global_opaque_segments
         << ",\"objectives\":" << summary.objectives
         << ",\"safe_repair_assignments\":"
         << summary.safe_repair_assignments
         << ",\"committed_objectives\":"
         << summary.committed_objectives
         << ",\"unscheduled_objectives\":"
         << summary.unscheduled_objectives
         << ",\"conservation_failures\":"
         << summary.conservation_failures
         << ",\"route_validator_failures\":"
         << summary.route_validator_failures
         << ",\"position_validator_failures\":"
         << summary.position_validator_failures
         << ",\"prefix_authority_failures\":"
         << summary.prefix_authority_failures
         << ",\"potential_move_delay\":"
         << summary.potential_move_delay
         << ",\"potentially_delayed_moves\":"
         << summary.potentially_delayed_moves
         << ",\"maximum_potential_move_delay\":"
         << summary.maximum_potential_move_delay
         << ",\"component_node_size_histogram\":{";
  bool first = true;
  for (const auto& [size, count] : summary.component_node_size_histogram) {
    if (!first) output << ',';
    first = false;
    output << '"' << size << "\":" << count;
  }
  output << "},\"exact_move_slot\":{\"enabled\":true"
         << ",\"segments\":" << summary.exact_slot_segments
         << ",\"absorbed_raw_intents\":"
         << summary.exact_slot_absorbed_intents
         << ",\"assignments\":" << summary.exact_slot_assignments
         << ",\"completed_intents\":"
         << summary.exact_slot_completed_intents
         << ",\"cross_day_debt_created\":"
         << summary.exact_slot_cross_day_debt_created
         << ",\"cross_day_debt_retired\":"
         << summary.exact_slot_cross_day_debt_retired
         << ",\"debt_lost\":" << summary.exact_slot_debt_lost
         << ",\"terminal_active_debt\":"
         << summary.exact_slot_terminal_active_debt
         << ",\"original_move_slot_changes\":"
         << summary.exact_slot_move_slot_changes
         << ",\"original_move_payload_changes\":"
         << summary.exact_slot_move_payload_changes
         << ",\"position_validator_failures\":"
         << summary.exact_slot_position_failures
         << ",\"rejections\":{";
  first = true;
  for (const auto& [reason, count] : summary.exact_slot_rejections) {
    if (!first) output << ',';
    first = false;
    output << '"' << exact_reject_name(reason) << "\":" << count;
  }
  output << "}},\"deviation_only_exact_move_slot\":{\"enabled\":true"
         << ",\"segments\":" << summary.deviation_segments
         << ",\"confirmed_triggers\":"
         << summary.deviation_confirmed_triggers
         << ",\"assignments\":" << summary.deviation_assignments
         << ",\"completed_intents\":"
         << summary.deviation_completed_intents
         << ",\"affected_tiles\":"
         << summary.deviation_affected_tiles.size()
         << ",\"affected_actor_days\":"
         << summary.deviation_affected_actor_days.size()
         << ",\"state_equivalent_changes\":"
         << summary.deviation_state_equivalent_changes
         << ",\"certified_service_changes\":"
         << summary.deviation_certified_service_changes
         << ",\"cross_day_debt_created\":"
         << summary.deviation_cross_day_debt_created
         << ",\"cross_day_debt_retired\":"
         << summary.deviation_cross_day_debt_retired
         << ",\"debt_lost\":" << summary.deviation_debt_lost
         << ",\"terminal_active_debt\":"
         << summary.deviation_terminal_active_debt
         << ",\"original_move_slot_changes\":"
         << summary.deviation_move_slot_changes
         << ",\"original_move_payload_changes\":"
         << summary.deviation_move_payload_changes
         << ",\"position_validator_failures\":"
         << summary.deviation_position_failures
         << ",\"raw_nonmove_changes_by_op\":{";
  first = true;
  for (const auto& [op, count] : summary.deviation_raw_nonmove_changes) {
    if (!first) output << ',';
    first = false;
    output << '"' << local_op_name(op) << "\":" << count;
  }
  output << "},\"rejections\":{";
  first = true;
  for (const auto& [reason, count] : summary.deviation_rejections) {
    if (!first) output << ',';
    first = false;
    output << '"' << exact_reject_name(reason) << "\":" << count;
  }
  output << "}},\"diagnostic_seed_upper_bound\":{"
         << "\"executable\":false,\"adds_seed_to_actions\":false"
         << ",\"segments\":" << summary.seed_upper_segments
         << ",\"confirmed_triggers\":"
         << summary.seed_upper_confirmed_triggers
         << ",\"assignments\":" << summary.seed_upper_assignments
         << ",\"completed_intents\":"
         << summary.seed_upper_completed_intents
         << ",\"cross_day_debt_created\":"
         << summary.seed_upper_cross_day_debt_created
         << ",\"cross_day_debt_retired\":"
         << summary.seed_upper_cross_day_debt_retired
         << ",\"debt_lost\":" << summary.seed_upper_debt_lost
         << ",\"terminal_active_debt\":"
         << summary.seed_upper_terminal_active_debt
         << ",\"move_invariant_failures\":"
         << summary.seed_upper_move_failures
         << ",\"rejections\":{";
  first = true;
  for (const auto& [reason, count] : summary.seed_upper_rejections) {
    if (!first) output << ',';
    first = false;
    output << '"' << exact_reject_name(reason) << "\":" << count;
  }
  const auto write_elastic = [&](const Summary::ElasticAggregate& elastic,
                                 bool seed_upper) {
    output << "{\"executable\":false,\"scenario_shadow_only\":true"
           << ",\"future_receipt_trace\":true,\"seed_upper_bound\":"
           << (seed_upper ? "true" : "false")
           << ",\"windows\":" << elastic.windows
           << ",\"assignments\":" << elastic.assignments
           << ",\"closed\":" << elastic.closed
           << ",\"terminal_debt\":" << elastic.terminal_debt
           << ",\"purchase_requests\":" << elastic.purchase_requests
           << ",\"delayed_moves\":" << elastic.delayed_moves
           << ",\"total_move_delay\":" << elastic.total_move_delay
           << ",\"maximum_move_delay\":" << elastic.maximum_move_delay
           << ",\"absorbed_pass_sinks\":"
           << elastic.absorbed_pass_sinks
           << ",\"absorbed_no_effect_sinks\":"
           << elastic.absorbed_no_effect_sinks
           << ",\"route_failures\":" << elastic.route_failures
           << ",\"terminal_raw_failures\":"
           << elastic.terminal_raw_failures
           << ",\"capacity_proof_failures\":"
           << elastic.capacity_proof_failures << '}';
  };
  output << "}},\"event_local_elastic_day_observed\":";
  write_elastic(summary.elastic_observed, false);
  output << ",\"event_local_elastic_day_seed_upper\":";
  write_elastic(summary.elastic_seed_upper, true);
  output << ",\"op_certificates\":{";
  first = true;
  for (int op = static_cast<int>(Op::PASS);
       op <= static_cast<int>(Op::CARE); ++op) {
    if (!first) output << ',';
    first = false;
    output << '"' << shadow::unit_op_name(static_cast<Op>(op)) << "\":{";
    for (int kind = 0; kind < 8; ++kind) {
      if (kind) output << ',';
      output << '"' << shadow::certificate_kind_name(
          static_cast<shadow::CertificateKind>(kind)) << "\":"
             << summary.certificates.by_op[static_cast<std::size_t>(op)]
                                           [static_cast<std::size_t>(kind)];
    }
    output << '}';
  }
  output << "}}";
}

}  // namespace

int main(int argc, char** argv) try {
  const auto options = parse(argc, argv);
  const auto tape = g001::repair::load_route(
      options.tapes, options.library, "G001");
  if (tape.size() != 719)
    throw std::runtime_error("G001 tape is not exactly 719 actions");
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  Summary forced;
  analyze_game(executor, options.forced_seed, 0, 1.0, forced);
  Summary normal;
  for (int seed = 0; seed < options.normal_seeds; ++seed)
    for (int seat = 0; seat < 2; ++seat)
      analyze_game(executor, options.normal_seed_begin +
                   static_cast<std::uint64_t>(seed), seat, 0.005, normal);
  const auto parent = std::filesystem::path(options.output).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream output(options.output);
  if (!output) throw std::runtime_error("cannot open shadow report");
  output << "{\n\"schema\":\"g001-component-shadow-v1\","
         << "\n\"applies_actions\":false,"
         << "\n\"reward_metrics_present\":false,"
         << "\n\"forced\":{\"seed\":" << options.forced_seed
         << ",\"weed_rate\":1.0,\"summary\":";
  write_summary(output, forced);
  output << "},\n\"normal\":{\"seed_begin\":"
         << options.normal_seed_begin << ",\"seed_count\":"
         << options.normal_seeds
         << ",\"seats\":2,\"weed_rate\":0.005,\"summary\":";
  write_summary(output, normal);
  output << "}\n}\n";
  std::cout << "component_shadow_eval output=" << options.output << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "component_shadow_eval failure: " << error.what() << '\n';
  return 1;
}
