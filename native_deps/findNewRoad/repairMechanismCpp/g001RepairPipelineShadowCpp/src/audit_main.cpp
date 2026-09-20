#include "g001_repair_pipeline_shadow.hpp"

#include "native_teammate.hpp"
#include "repair_fork_evaluator.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace shadow = g001::repair_pipeline_shadow;
namespace day_issuer = g001::day_start_issuer;
namespace obligation_day = g001::obligation_day;
namespace typed = g001::typed_intent;
namespace production = production_obligation;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;

namespace {

struct Options {
  std::string output{"g001-repair-pipeline-shadow.json"};
  std::string reproducer{"g001-repair-pipeline-first-invalid.json"};
  std::uint64_t seed{970017};
  std::size_t seeds{4};
};

Options parse(int argc, char** argv) {
  Options output;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help") {
      std::cout << "usage: " << argv[0]
                << " [--output FILE] [--seed N] [--seeds N]\n";
      std::exit(0);
    }
    if (++index >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    const std::string value = argv[index];
    if (option == "--output") {
      output.output = value;
    } else if (option == "--reproducer") {
      output.reproducer = value;
    } else if (option == "--seed") {
      const auto parsed = std::from_chars(
          value.data(), value.data() + value.size(), output.seed);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("invalid --seed");
      }
    } else if (option == "--seeds") {
      const auto parsed = std::from_chars(
          value.data(), value.data() + value.size(), output.seeds);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size() || output.seeds == 0) {
        throw std::invalid_argument("invalid --seeds");
      }
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  return output;
}

fastkag::NativeTapeLibrary load_g001() {
  fastkag::NativeTapeLibrary output;
  output.routes.push_back(
      g001::repair::load_route(G001_SHADOW_TAPES, G001_SHADOW_LIBRARY,
                               "G001"));
  output.r5_reference =
      g001::repair::load_route(G001_SHADOW_REFS, G001_SHADOW_LIBRARY, "R5");
  output.md_reference =
      g001::repair::load_route(G001_SHADOW_REFS, G001_SHADOW_LIBRARY, "MD");
  constexpr std::array<std::string_view, 5> labels{
      "10C4S_3Q", "8C6S_3Q", "6C8S_3Q", "6C12S_4Q_FIRST_YARN",
      "6C12S_4Q_SECOND_YARN"};
  for (std::size_t index = 0; index < labels.size(); ++index) {
    output.moon[index] = g001::repair::load_route(
        G001_SHADOW_REFS, G001_SHADOW_LIBRARY,
        "MOON_" + std::string(labels[index]));
    output.moon_legacy[index] = g001::repair::load_route(
        G001_SHADOW_REFS, G001_SHADOW_LIBRARY,
        "MOON_LEGACY_" + std::string(labels[index]));
  }
  return output;
}

fastkag::Position actor_position(const fastkag::Simulator& simulator,
                                 int player, int actor) {
  const auto& farm = simulator.farms()[player];
  if (actor == 0) return farm.farmer;
  if (actor > 0 && actor <= static_cast<int>(farm.hands.size())) {
    return farm.hands[static_cast<std::size_t>(actor - 1)];
  }
  return {-1, -1};
}

fastkag::TileKind target_kind(const fastkag::Simulator& simulator, int player,
                              int actor) {
  const auto position = actor_position(simulator, player, actor);
  const int size = simulator.config().board_size;
  if (position.x < 0 || position.y < 0 || position.x >= size ||
      position.y >= size) {
    return fastkag::TileKind::LOCKED;
  }
  return simulator.farms()[player].tiles[
      static_cast<std::size_t>(position.y * size + position.x)].kind;
}

fastkag::Position moved(fastkag::Position position, Op operation) {
  if (operation == Op::NORTH) --position.y;
  if (operation == Op::SOUTH) ++position.y;
  if (operation == Op::WEST) --position.x;
  if (operation == Op::EAST) ++position.x;
  return position;
}

bool dynamic_suffix(const fastkag::NativeAgentState& state) {
  return std::any_of(state.weed.begin(), state.weed.end(),
                     [](const auto& value) { return value.active; }) ||
         std::any_of(state.experimental_realign.begin(),
                     state.experimental_realign.end(),
                     [](const auto& value) { return value.active; }) ||
         state.room_evac.active || state.salvage.active;
}

bool actor_overlay_active(const fastkag::NativeAgentState& state, int actor) {
  return (actor >= 0 && actor < static_cast<int>(state.weed.size()) &&
          state.weed[static_cast<std::size_t>(actor)].active) ||
         (actor >= 0 &&
          actor < static_cast<int>(state.experimental_realign.size()) &&
          state.experimental_realign[static_cast<std::size_t>(actor)].active) ||
         (state.room_evac.active && state.room_evac.actor == actor) ||
         (state.salvage.active && state.salvage.actor == actor);
}

std::string overlay_description(const fastkag::NativeAgentState& state,
                                int actor) {
  if (actor >= 0 && actor < static_cast<int>(state.weed.size()) &&
      state.weed[static_cast<std::size_t>(actor)].active) {
    const auto& weed = state.weed[static_cast<std::size_t>(actor)];
    return "weed_repair:start=" + std::to_string(weed.start) +
           ":intended_op=" +
           std::to_string(static_cast<int>(weed.intended.op));
  }
  if (actor >= 0 &&
      actor < static_cast<int>(state.experimental_realign.size()) &&
      state.experimental_realign[static_cast<std::size_t>(actor)].active) {
    return "unit_realignment";
  }
  if (state.room_evac.active && state.room_evac.actor == actor) {
    return "room_evac";
  }
  if (state.salvage.active && state.salvage.actor == actor) return "salvage";
  if (state.r5_target) return "r5_reference_route";
  if (state.md_target) return "md_reference_route";
  if (state.moon_layout >= 0) return "moon_reference_route";
  return "baseline_or_cleared_overlay";
}

bool has_effect(const fastkag::Simulator& simulator, int player,
                std::span<const Action> units, int actor) {
  std::array<PlayerAction, 2> prefix;
  prefix[player].units.assign(units.begin(), units.begin() + actor);
  const auto before = simulator.preview_unit_phase(prefix);
  prefix[player].units.push_back(units[static_cast<std::size_t>(actor)]);
  const auto after = simulator.preview_unit_phase(prefix);
  return g001::repair_fork::full_unit_phase_state_fingerprint(before) !=
         g001::repair_fork::full_unit_phase_state_fingerprint(after);
}

bool action_equal(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

const char* issue_reject_name(day_issuer::IssueReject reject) {
  switch (reject) {
    case day_issuer::IssueReject::None:
      return "none";
    case day_issuer::IssueReject::InvalidInput:
      return "invalid_input";
    case day_issuer::IssueReject::NotDayStart:
      return "not_day_start";
    case day_issuer::IssueReject::RouteTooShort:
      return "route_too_short";
  }
  return "unknown";
}

const char* plan_reject_name(obligation_day::PlanReject reject) {
  switch (reject) {
    case obligation_day::PlanReject::None:
      return "none";
    case obligation_day::PlanReject::InvalidState:
      return "invalid_state";
    case obligation_day::PlanReject::NotDayStart:
      return "not_day_start";
    case obligation_day::PlanReject::InvalidIdentity:
      return "invalid_identity";
    case obligation_day::PlanReject::InvalidMoveToken:
      return "invalid_move_token";
    case obligation_day::PlanReject::InvalidDag:
      return "invalid_dag";
    case obligation_day::PlanReject::InternalProofFailure:
      return "internal_proof_failure";
  }
  return "unknown";
}

struct DayRecord {
  std::uint64_t seed{};
  int player{};
  int day{};
  int observed_steps{};
  day_issuer::IssueReject issue_reject{day_issuer::IssueReject::InvalidInput};
  bool fully_proven{};
  int moves{};
  int obligations{};
  int crop_obligations{};
  int animal_obligations{};
  int unified_targets{};
  int unified_groups{};
  int unified_admitted_groups{};
  int unified_demands{};
  int unified_diagnostics{};
  bool unified_planned{};
  bool unified_takeover_eligible{};
  std::map<std::string, int> unsupported;
  obligation_day::PlanReject plan_reject{
      obligation_day::PlanReject::InvalidState};
  bool planned{};
  bool partial_plan{};
  bool locally_full_certificate_eligible{};
  bool full_certificate_eligible{};
  bool certificate_valid{};
  obligation_day::PlanReject verification_reject{
      obligation_day::PlanReject::InvalidState};
  int verification_checked_slots{};
  int certificate_move_closure{};
  int debts{};
  int must_finish_debts{};
  std::map<std::string, int> debt_reasons;
  bool provider_audited{};
  bool provider_valid{};
  int provider_mismatches{};
  int provider_move_exact{};
  int provider_move_total{};
  bool runtime_move_authorized{};
  struct MoveMismatch {
    int actor{-1};
    int source_step{-1};
    int expected_op{-1};
    int actual_op{-1};
    std::string runtime_overlay;
  };
  std::vector<MoveMismatch> move_mismatches;
  int current_evidence{};
  int current_exact{};
  int combined_exact{};
  int complete_crop_payloads{};
  int complete_animal_payloads{};
  int crop_lineage_writes{};
  int animal_lineage_writes{};
  int persistent_crop_recoveries{};
  int weed_no_effect_wh{};
  int weed_no_effect_wh_current_exact{};
  int weed_no_effect_wh_lineage{};
  int weed_no_effect_wh_lineage_additional{};
  int weed_no_effect_wh_combined_exact{};
  std::map<std::string, int> current_unresolved;
  std::map<std::string, int> weed_remaining_causes;
};

struct ActiveDay {
  DayRecord record;
  std::uint64_t generation{};
  day_issuer::IssueResult issued;
  std::vector<PlayerAction> actual;
  std::vector<std::vector<std::string>> runtime_overlays;
  std::vector<fastkag::Position> raw_route_positions;
};

struct Totals {
  int days{};
  int full_days{};
  int terminal_short_days{};
  int issued_days{};
  int fully_proven_days{};
  int planned_days{};
  int partial_plan_days{};
  int locally_full_certificate_eligible_days{};
  int full_certificate_eligible_days{};
  int certificate_valid_days{};
  int provider_valid_days{};
  int provider_invalid_days{};
  int moves{};
  int certificate_move_closure{};
  int provider_move_exact{};
  int provider_move_total{};
  int retrospective_move_exact_days{};
  int runtime_move_authorized_days{};
  int runtime_move_fail_closed_days{};
  int obligations{};
  int crop_obligations{};
  int animal_obligations{};
  int unified_targets{};
  int unified_groups{};
  int unified_admitted_groups{};
  int unified_demands{};
  int unified_diagnostics{};
  int unified_planned_days{};
  int unified_takeover_eligible_days{};
  int current_evidence{};
  int current_exact{};
  int combined_exact{};
  int complete_crop_payloads{};
  int complete_animal_payloads{};
  int must_finish_debts{};
  int weed_no_effect_wh{};
  int weed_no_effect_wh_current_exact{};
  int weed_no_effect_wh_lineage{};
  int weed_no_effect_wh_lineage_additional{};
  int weed_no_effect_wh_combined_exact{};
  std::map<std::string, int> unsupported;
  std::map<std::string, int> current_unresolved;
  std::map<std::string, int> weed_remaining_causes;
  std::map<std::string, int> verification_rejects;
  std::map<std::string, int> debt_reasons;
};

void merge(std::map<std::string, int>& target,
           const std::map<std::string, int>& source) {
  for (const auto& [name, count] : source) target[name] += count;
}

void accumulate(Totals& totals, const DayRecord& record) {
  ++totals.days;
  totals.full_days += record.observed_steps == 24;
  totals.terminal_short_days += record.observed_steps != 24;
  totals.issued_days += record.issue_reject == day_issuer::IssueReject::None;
  totals.fully_proven_days += record.fully_proven;
  totals.planned_days += record.planned;
  totals.partial_plan_days += record.partial_plan;
  totals.locally_full_certificate_eligible_days +=
      record.locally_full_certificate_eligible;
  totals.full_certificate_eligible_days += record.full_certificate_eligible;
  totals.certificate_valid_days += record.certificate_valid;
  totals.provider_valid_days += record.provider_audited && record.provider_valid;
  totals.provider_invalid_days +=
      record.provider_audited && !record.provider_valid;
  totals.moves += record.moves;
  totals.certificate_move_closure += record.certificate_move_closure;
  totals.provider_move_exact += record.provider_move_exact;
  totals.provider_move_total += record.provider_move_total;
  totals.retrospective_move_exact_days +=
      record.provider_audited &&
      record.provider_move_exact == record.provider_move_total;
  totals.runtime_move_authorized_days += record.runtime_move_authorized;
  totals.runtime_move_fail_closed_days +=
      record.issue_reject == day_issuer::IssueReject::None &&
      !record.runtime_move_authorized;
  totals.obligations += record.obligations;
  totals.crop_obligations += record.crop_obligations;
  totals.animal_obligations += record.animal_obligations;
  totals.unified_targets += record.unified_targets;
  totals.unified_groups += record.unified_groups;
  totals.unified_admitted_groups += record.unified_admitted_groups;
  totals.unified_demands += record.unified_demands;
  totals.unified_diagnostics += record.unified_diagnostics;
  totals.unified_planned_days += record.unified_planned;
  totals.unified_takeover_eligible_days += record.unified_takeover_eligible;
  totals.current_evidence += record.current_evidence;
  totals.current_exact += record.current_exact;
  totals.combined_exact += record.combined_exact;
  totals.complete_crop_payloads += record.complete_crop_payloads;
  totals.complete_animal_payloads += record.complete_animal_payloads;
  totals.must_finish_debts += record.must_finish_debts;
  totals.weed_no_effect_wh += record.weed_no_effect_wh;
  totals.weed_no_effect_wh_current_exact +=
      record.weed_no_effect_wh_current_exact;
  totals.weed_no_effect_wh_lineage += record.weed_no_effect_wh_lineage;
  totals.weed_no_effect_wh_lineage_additional +=
      record.weed_no_effect_wh_lineage_additional;
  totals.weed_no_effect_wh_combined_exact +=
      record.weed_no_effect_wh_combined_exact;
  merge(totals.unsupported, record.unsupported);
  merge(totals.current_unresolved, record.current_unresolved);
  merge(totals.weed_remaining_causes, record.weed_remaining_causes);
  merge(totals.debt_reasons, record.debt_reasons);
  if (!record.certificate_valid && record.planned) {
    ++totals.verification_rejects[
        plan_reject_name(record.verification_reject)];
  }
}

void finalize(ActiveDay& active, Totals& totals) {
  auto& record = active.record;
  record.observed_steps = static_cast<int>(active.actual.size());
  if (active.issued.issued() && active.actual.size() == 24) {
    const auto audit = day_issuer::audit_final_provider_day(
        active.issued, record.day * 24, active.actual);
    record.provider_audited = true;
    record.provider_valid = audit.day_valid;
    record.provider_mismatches = audit.mismatches;
    for (const auto& move : active.issued.moves) {
      ++record.provider_move_total;
      const int tick = move.source_step - record.day * 24;
      if (tick >= 0 && tick < static_cast<int>(active.actual.size()) &&
          move.actor >= 0 &&
          move.actor < static_cast<int>(active.actual[tick].units.size()) &&
          action_equal(active.actual[tick].units[move.actor], move.action)) {
        ++record.provider_move_exact;
      } else {
        const int actual_op =
            tick >= 0 && tick < static_cast<int>(active.actual.size()) &&
                    move.actor >= 0 &&
                    move.actor <
                        static_cast<int>(active.actual[tick].units.size())
                ? static_cast<int>(active.actual[tick].units[move.actor].op)
                : -1;
        std::string overlay = "missing_runtime_snapshot";
        if (tick >= 0 &&
            tick < static_cast<int>(active.runtime_overlays.size()) &&
            move.actor >= 0 &&
            move.actor < static_cast<int>(
                             active.runtime_overlays[tick].size())) {
          overlay = active.runtime_overlays[tick][move.actor];
        }
        record.move_mismatches.push_back(
            {move.actor, move.source_step,
             static_cast<int>(move.action.op), actual_op, overlay});
      }
    }
  }
  record.full_certificate_eligible =
      record.locally_full_certificate_eligible && record.provider_valid &&
      record.provider_move_exact == record.provider_move_total;
  accumulate(totals, record);
}

void write_map(std::ostream& output,
               const std::map<std::string, int>& values) {
  output << '{';
  bool first = true;
  for (const auto& [name, count] : values) {
    if (!first) output << ',';
    first = false;
    output << '"' << name << "\":" << count;
  }
  output << '}';
}

double ratio(int numerator, int denominator) {
  return denominator ? static_cast<double>(numerator) / denominator : 0.0;
}

void write_first_invalid_reproducer(
    const std::string& path, std::uint64_t seed,
    const fastkag::Simulator& day_start, int player,
    std::uint64_t generation, const day_issuer::IssueResult& issued,
    const obligation_day::VerifyResult& original_verification) {
  obligation_day::DayPlanRequest minimized{
      &day_start, player, generation, issued.moves, issued.obligations};
  const auto remains_invalid = [](const auto& request) {
    const auto plan = obligation_day::internal::plan_day_unchecked(request);
    return plan.planned() &&
           !obligation_day::verify_day_schedule(request, *plan.certificate)
                .valid;
  };
  for (std::size_t index = 0; index < minimized.obligations.size();) {
    auto candidate = minimized;
    candidate.obligations.erase(candidate.obligations.begin() +
                                static_cast<std::ptrdiff_t>(index));
    if (remains_invalid(candidate)) {
      minimized.obligations = std::move(candidate.obligations);
    } else {
      ++index;
    }
  }
  for (std::size_t index = 0; index < minimized.moves.size();) {
    auto candidate = minimized;
    candidate.moves.erase(candidate.moves.begin() +
                          static_cast<std::ptrdiff_t>(index));
    if (remains_invalid(candidate)) {
      minimized.moves = std::move(candidate.moves);
    } else {
      ++index;
    }
  }
  const auto plan = obligation_day::internal::plan_day_unchecked(minimized);
  const auto verification = obligation_day::verify_day_schedule(
      minimized, *plan.certificate);
  std::ofstream output(path, std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create invalid reproducer");
  output << "{\n  \"schema\":\"g001-day-scheduler-invalid-request-v1\",\n"
         << "  \"replay\":{\"seed\":" << seed
         << ",\"provider\":\"G001_vs_G001_default_off\",\"player\":"
         << player << ",\"day\":" << day_start.day()
         << ",\"step\":" << day_start.step_count() << "},\n"
         << "  \"issuer_generation\":" << generation << ",\n"
         << "  \"focal_start_fingerprint\":"
         << g001::production_suffix::focal_unit_state_fingerprint(
                day_start, player)
         << ",\n  \"full_start_fingerprint\":"
         << g001::repair_fork::full_unit_phase_state_fingerprint(day_start)
         << ",\n  \"original\":{\"moves\":" << issued.moves.size()
         << ",\"obligations\":" << issued.obligations.size()
         << ",\"verification_checked_slots\":"
         << original_verification.checked_slots << "},\n"
         << "  \"minimized\":{\"moves\":[";
  for (std::size_t index = 0; index < minimized.moves.size(); ++index) {
    const auto& move = minimized.moves[index];
    output << (index == 0 ? "" : ",") << "{\"actor\":" << move.actor
           << ",\"source_step\":" << move.source_step
           << ",\"op\":" << static_cast<int>(move.action.op)
           << ",\"item\":" << static_cast<int>(move.action.item)
           << ",\"quantity\":" << move.action.quantity << '}';
  }
  output << "],\"obligations\":[";
  for (std::size_t index = 0; index < minimized.obligations.size(); ++index) {
    const auto& value = minimized.obligations[index];
    output << (index == 0 ? "" : ",") << "{\"id\":" << value.id
           << ",\"actor\":" << value.actor << ",\"tile\":["
           << value.tile.x << ',' << value.tile.y << "],\"goal\":"
           << static_cast<int>(value.goal) << ",\"item\":"
           << static_cast<int>(value.item) << ",\"quantity\":"
           << value.quantity << ",\"earliest\":" << value.earliest_step
           << ",\"deadline\":" << value.deadline_step
           << ",\"priority\":" << value.priority
           << ",\"must_finish\":"
           << (value.must_finish_today ? "true" : "false")
           << ",\"dependencies\":[";
    for (std::size_t dependency = 0;
         dependency < value.dependencies.size(); ++dependency) {
      output << (dependency == 0 ? "" : ",")
             << value.dependencies[dependency];
    }
    output << "]}";
  }
  output << "],\"verification_checked_slots\":"
         << verification.checked_slots
         << ",\"failure_reason\":\""
         << obligation_day::verify_failure_reason_name(
                verification.failure_reason)
         << "\",\"failure_step\":" << verification.failure_step
         << ",\"failure_actor\":" << verification.failure_actor
         << ",\"failure_obligation_id\":"
         << verification.failure_obligation_id
         << ",\"failure_index\":" << verification.failure_index
         << ",\"failure_stage\":\""
         << (verification.checked_slots < 24 ? "slot_replay" : "post_slots")
         << "\",\"failing_slot\":" << verification.checked_slots;
  if (verification.checked_slots < 24 && plan.certificate) {
    const auto& slot =
        plan.certificate->slots[verification.checked_slots];
    output << ",\"failing_actions\":[";
    for (std::size_t actor = 0; actor < slot.actions.size(); ++actor) {
      output << (actor == 0 ? "" : ",")
             << "{\"actor\":" << actor << ",\"op\":"
             << static_cast<int>(slot.actions[actor].op)
             << ",\"item\":" << static_cast<int>(slot.actions[actor].item)
             << ",\"quantity\":" << slot.actions[actor].quantity
             << ",\"obligation_id\":" << slot.obligation_ids[actor]
             << ",\"source_step\":" << slot.sources[actor].source_step
             << '}';
    }
    output << ']';
  }
  output << ",\"obligation_statuses\":[";
  if (plan.certificate) {
    for (std::size_t index = 0;
         index < plan.certificate->obligation_statuses.size(); ++index) {
      const auto& status = plan.certificate->obligation_statuses[index];
      output << (index == 0 ? "" : ",")
             << "{\"id\":" << status.obligation_id
             << ",\"disposition\":"
             << static_cast<int>(status.disposition)
             << ",\"assigned_actor\":" << status.assigned_actor
             << ",\"remaining\":" << status.remaining_transitions
             << ",\"transition_steps\":[";
      for (std::size_t step = 0; step < status.transition_steps.size();
           ++step) {
        output << (step == 0 ? "" : ",") << status.transition_steps[step];
      }
      output << "]}";
    }
  }
  output << ']';
  output << "}\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse(argc, argv);
    const auto library = load_g001();
    fastkag::NativeTeammateExecutor executor(library);
    const auto& raw_tape = executor.route_tape(0);
    std::vector<DayRecord> records;
    Totals totals;
    bool invalid_reproducer_written = false;

    for (std::size_t seed_offset = 0; seed_offset < options.seeds;
         ++seed_offset) {
      const std::uint64_t seed = options.seed + seed_offset;
      fastkag::Simulator simulator({}, seed);
      std::array<fastkag::NativeAgentState, 2> states;
      std::array<day_issuer::PersistentRouteIntentRegistry, 2> lineage;
      std::array<std::map<int, std::set<std::pair<int, int>>>, 2>
          exact_plant_tiles;
      std::array<std::optional<ActiveDay>, 2> active;

      while (!simulator.done()) {
        if (simulator.hour() == 0) {
          for (int player = 0; player < 2; ++player) {
            if (active[player]) {
              finalize(*active[player], totals);
              records.push_back(std::move(active[player]->record));
            }
            const auto generation =
                (seed << 16) |
                (static_cast<std::uint64_t>(player + 1) << 12) |
                static_cast<std::uint64_t>(simulator.day() + 1);
            const auto composed = shadow::plan_and_verify_day_shadow(
                {&simulator, &raw_tape, player, generation,
                 &lineage[player]});
            const auto unified = shadow::plan_unified_day_shadow(
                {&simulator, &raw_tape, player, generation,
                 &lineage[player]});
            if (!invalid_reproducer_written && composed.plan.planned() &&
                !composed.verification.valid) {
              write_first_invalid_reproducer(
                  options.reproducer, seed, simulator, player, generation,
                  composed.issued, composed.verification);
              invalid_reproducer_written = true;
            }
            ActiveDay next;
            next.record.seed = seed;
            next.record.player = player;
            next.record.day = simulator.day();
            next.raw_route_positions.push_back(
                simulator.farms()[player].farmer);
            next.raw_route_positions.insert(
                next.raw_route_positions.end(),
                simulator.farms()[player].hands.begin(),
                simulator.farms()[player].hands.end());
            next.record.issue_reject = composed.issued.reject;
            next.generation = generation;
            next.record.fully_proven = composed.issued.fully_proven();
            next.record.moves = static_cast<int>(composed.issued.moves.size());
            next.record.obligations =
                static_cast<int>(composed.issued.obligations.size());
            next.record.crop_obligations = static_cast<int>(
                composed.issued.crop_owner_obligations.size());
            next.record.animal_obligations = static_cast<int>(
                composed.issued.animal_owner_obligations.size());
            next.record.unified_targets =
                static_cast<int>(unified.targets.size());
            next.record.unified_groups =
                static_cast<int>(unified.compiled.groups.size());
            next.record.unified_admitted_groups = static_cast<int>(
                std::count_if(unified.admission.groups.begin(),
                              unified.admission.groups.end(),
                              [](const auto& group) { return group.admitted; }));
            next.record.unified_demands =
                static_cast<int>(unified.compiled.demands.size());
            next.record.unified_diagnostics =
                static_cast<int>(unified.compiled.diagnostics.size());
            next.record.unified_planned = unified.admission.planned();
            next.record.unified_takeover_eligible =
                unified.takeover_eligible;
            for (const auto& unsupported : composed.issued.unsupported) {
              ++next.record.unsupported[
                  day_issuer::unsupported_reason_name(unsupported.reason)];
            }
            next.record.plan_reject = composed.plan.reject;
            next.record.planned = composed.plan.planned();
            next.record.partial_plan = composed.partial_plan;
            next.record.locally_full_certificate_eligible =
                composed.locally_full_certificate_eligible;
            next.record.certificate_valid = composed.verification.valid;
            next.record.verification_reject = composed.verification.reject;
            next.record.verification_checked_slots =
                composed.verification.checked_slots;
            if (composed.verification.valid && composed.plan.certificate) {
              next.record.certificate_move_closure = static_cast<int>(
                  composed.plan.certificate->move_replays.size());
            }
            next.record.debts = static_cast<int>(composed.plan.debts.size());
            for (const auto& debt : composed.plan.debts) {
              ++next.record.debt_reasons[
                  obligation_day::debt_reason_name(debt.reason)];
              const auto found = std::find_if(
                  composed.issued.obligations.begin(),
                  composed.issued.obligations.end(), [&](const auto& value) {
                    return value.id == debt.obligation_id;
                  });
              if (found != composed.issued.obligations.end() &&
                  found->must_finish_today) {
                ++next.record.must_finish_debts;
              }
            }
            next.issued = composed.issued;
            active[player] = std::move(next);
          }
        }

        std::array<PlayerAction, 2> actions;
        for (int player = 0; player < 2; ++player) {
          const auto committed = executor.action_external_committed(
              simulator, player, 0, states[player],
              active[player]->generation,
              fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr,
              false, {}, nullptr);
          actions[player] = committed.action;
          const auto movement_authority =
              shadow::verify_movement_suffix_authority(
                  player, simulator.day(), active[player]->generation,
                  simulator.step_count(), active[player]->issued.moves,
                  &committed.movement);
          active[player]->record.runtime_move_authorized =
              active[player]->record.runtime_move_authorized ||
              movement_authority.authorized;
          active[player]->actual.push_back(actions[player]);
          std::vector<std::string> overlays;
          overlays.reserve(actions[player].units.size());
          for (std::size_t actor = 0; actor < actions[player].units.size();
               ++actor) {
            overlays.push_back(overlay_description(
                states[player], static_cast<int>(actor)));
          }
          active[player]->runtime_overlays.push_back(std::move(overlays));

          const bool dynamic = dynamic_suffix(states[player]);
          const int step = simulator.step_count();
          const int end = std::min(
              {step + 24, 718, static_cast<int>(raw_tape.size()) - 1});
          std::vector<fastkag::NativeFutureUnitFrame> future;
          for (int next = step + 1; next <= end; ++next) {
            future.push_back(
                {next, raw_tape[static_cast<std::size_t>(next)].units});
          }
          const auto dag = fastkag::compile_native_production_obligations(
              simulator, player, actions[player].units, future);

          std::vector<bool> trigger(actions[player].units.size());
          std::vector<bool> lineage_before(actions[player].units.size());
          std::vector<bool> actor_has_prior_plant(
              actions[player].units.size());
          std::vector<bool> actor_dynamic(actions[player].units.size());
          std::vector<bool> actor_present_at_day_start(
              actions[player].units.size());
          std::vector<bool> residual_route_position_drift(
              actions[player].units.size());
          for (std::size_t actor = 0; actor < actions[player].units.size();
               ++actor) {
            const auto& action = actions[player].units[actor];
            if (action.op != Op::WATER && action.op != Op::HARVEST) continue;
            const bool weed = target_kind(simulator, player,
                                          static_cast<int>(actor)) ==
                              fastkag::TileKind::WEED;
            const bool no_effect = !has_effect(
                simulator, player, actions[player].units,
                static_cast<int>(actor));
            trigger[actor] = weed || no_effect;
            const auto prior = lineage[player].crop_at(
                actor_position(simulator, player, static_cast<int>(actor)));
            lineage_before[actor] = prior.has_value();
            actor_has_prior_plant[actor] =
                !exact_plant_tiles[player][static_cast<int>(actor)].empty();
            actor_dynamic[actor] =
                actor_overlay_active(states[player], static_cast<int>(actor));
            actor_present_at_day_start[actor] =
                actor < active[player]->raw_route_positions.size();
            if (actor_present_at_day_start[actor]) {
              const auto live = actor_position(
                  simulator, player, static_cast<int>(actor));
              const auto raw = active[player]->raw_route_positions[actor];
              residual_route_position_drift[actor] =
                  live.x != raw.x || live.y != raw.y;
            }
          }

          const auto generation =
              (seed << 20) |
              (static_cast<std::uint64_t>(player + 1) << 16) |
              static_cast<std::uint64_t>(step + 1);
          const auto observed = shadow::observe_final_current_units(
              {&simulator, player, generation, actions[player].units,
               dag.nodes, !dynamic && dag.feasible, &lineage[player]});
          auto& record = active[player]->record;
          record.complete_crop_payloads +=
              static_cast<int>(observed.crops.size());
          record.complete_animal_payloads +=
              static_cast<int>(observed.animals.size());
          record.crop_lineage_writes += observed.crop_lineage_records;
          record.animal_lineage_writes += observed.animal_lineage_records;
          record.persistent_crop_recoveries +=
              observed.persistent_crop_recoveries;
          for (const auto& evidence : observed.evidence) {
            ++record.current_evidence;
            record.current_exact += evidence.current.exact;
            record.combined_exact += evidence.exact;
            if (!evidence.exact) {
              ++record.current_unresolved[
                  typed::proof_name(evidence.current.proof)];
            }
            const int actor = evidence.current.actor;
            if (actor < 0 || actor >= static_cast<int>(trigger.size()) ||
                !trigger[static_cast<std::size_t>(actor)]) {
              continue;
            }
            ++record.weed_no_effect_wh;
            record.weed_no_effect_wh_current_exact += evidence.current.exact;
            record.weed_no_effect_wh_lineage +=
                lineage_before[static_cast<std::size_t>(actor)];
            record.weed_no_effect_wh_lineage_additional +=
                lineage_before[static_cast<std::size_t>(actor)] &&
                !evidence.current.exact;
            record.weed_no_effect_wh_combined_exact += evidence.exact;
            if (!evidence.exact) {
              const auto position = evidence.current.tile;
              const bool valid_actor = position.x >= 0 && position.y >= 0;
              const bool has_prior =
                  actor_has_prior_plant[static_cast<std::size_t>(actor)];
              const bool dynamic_actor =
                  actor_dynamic[static_cast<std::size_t>(actor)];
              const auto cause = shadow::classify_missing_lineage(
                  {valid_actor,
                   actor_present_at_day_start[static_cast<std::size_t>(actor)],
                   residual_route_position_drift[
                       static_cast<std::size_t>(actor)],
                   dynamic_actor, has_prior, actor,
                   evidence.current.proof});
              ++record.weed_remaining_causes[
                  shadow::missing_lineage_cause_name(cause)];
            }
          }
          for (const auto& evidence : observed.evidence) {
            if (evidence.current.exact &&
                evidence.current.source_action.op == Op::PLANT) {
              exact_plant_tiles[player][evidence.current.actor].insert(
                  {evidence.current.tile.x, evidence.current.tile.y});
            }
          }
          const auto& raw_units =
              raw_tape[static_cast<std::size_t>(step)].units;
          for (std::size_t actor = 0;
               actor < active[player]->raw_route_positions.size() &&
               actor < raw_units.size();
               ++actor) {
            const auto operation = raw_units[actor].op;
            if (operation == Op::NORTH || operation == Op::SOUTH ||
                operation == Op::EAST || operation == Op::WEST) {
              active[player]->raw_route_positions[actor] = moved(
                  active[player]->raw_route_positions[actor], operation);
            }
          }
        }
        simulator.step(actions);
      }

      for (int player = 0; player < 2; ++player) {
        if (active[player]) {
          finalize(*active[player], totals);
          records.push_back(std::move(active[player]->record));
        }
      }
    }

    constexpr std::array<std::string_view, 8> remaining_categories{
        "actor_unavailable",
        "actor_unavailable_at_day_start_or_hired_midday",
        "lineage_tile_mismatch_from_residual_move_drift",
        "dynamic_move_lineage_tile_mismatch",
        "hired_actor_no_prior_exact_plant",
        "planned_route_tile_without_prior_lineage",
        "suffix_authority_only_no_lineage",
        "no_prior_exact_plant_lineage"};
    for (const auto name : remaining_categories) {
      totals.weed_remaining_causes.try_emplace(std::string(name), 0);
    }
    totals.weed_remaining_causes.try_emplace(
        "crop_switch_or_registry_stale", 0);
    int categorized_remaining = 0;
    for (const auto& [name, count] : totals.weed_remaining_causes) {
      (void)name;
      categorized_remaining += count;
    }
    if (categorized_remaining !=
        totals.weed_no_effect_wh - totals.weed_no_effect_wh_combined_exact) {
      throw std::runtime_error("weed remaining causes are not a partition");
    }
    if (options.seed == 970017 && options.seeds == 4 &&
        totals.weed_no_effect_wh != 581) {
      throw std::runtime_error("typed-audit 581-trigger parity changed");
    }
    if (options.seed == 970017 && options.seeds == 4) {
      int move_mismatch_records = 0;
      bool expected_move_mismatch = false;
      for (const auto& record : records) {
        move_mismatch_records +=
            static_cast<int>(record.move_mismatches.size());
        for (const auto& mismatch : record.move_mismatches) {
          expected_move_mismatch =
              expected_move_mismatch ||
              (record.seed == 970017 && record.player == 1 &&
               record.day == 7 && mismatch.actor == 0 &&
               mismatch.source_step == 187 &&
               mismatch.expected_op == static_cast<int>(Op::SOUTH) &&
               mismatch.actual_op == static_cast<int>(Op::CARE));
        }
      }
      if (totals.provider_move_exact != 1671 ||
          totals.provider_move_total != 1672 ||
          move_mismatch_records != 1 || !expected_move_mismatch ||
          totals.weed_remaining_causes[
              "actor_unavailable_at_day_start_or_hired_midday"] != 82 ||
          totals.weed_remaining_causes[
              "no_prior_exact_plant_lineage"] != 30) {
        throw std::runtime_error("real shadow regression coverage changed");
      }
    }

    std::ofstream output(options.output, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create audit output");
    output << std::setprecision(12)
           << "{\n  \"schema\":\"g001-repair-pipeline-shadow-v1\",\n"
           << "  \"provider\":\"NativeTeammateExecutor::action_external G001 vs G001\",\n"
           << "  \"seed_begin\":" << options.seed
           << ",\n  \"seed_count\":" << options.seeds
           << ",\n  \"seats\":2,\n"
           << "  \"shadow_only\":true,\n"
           << "  \"native_repair_options_enabled\":false,\n"
           << "  \"native_bit_opened\":false,\n"
           << "  \"actions_rewritten\":false,\n"
           << "  \"win_rate_panel\":false,\n"
           << "  \"abi_adapter\":{\"current_evidence_available_only_at_current_step\":true,\"future_current_evidence_used_at_day_start\":false,\"persistent_lineage_is_day_start_bridge\":true},\n"
           << "  \"totals\":{\"days\":" << totals.days
           << ",\"full_days\":" << totals.full_days
           << ",\"terminal_short_days\":" << totals.terminal_short_days
           << ",\"issued_days\":" << totals.issued_days
           << ",\"fully_proven_days\":" << totals.fully_proven_days
           << ",\"planned_days\":" << totals.planned_days
           << ",\"partial_plan_days\":" << totals.partial_plan_days
           << ",\"locally_full_certificate_eligible_days\":"
           << totals.locally_full_certificate_eligible_days
           << ",\"full_certificate_eligible_days\":"
           << totals.full_certificate_eligible_days
           << ",\"certificate_valid_days\":"
           << totals.certificate_valid_days
           << ",\"provider_valid_days\":" << totals.provider_valid_days
           << ",\"provider_invalid_days\":" << totals.provider_invalid_days
           << ",\"move_tokens\":" << totals.moves
           << ",\"certificate_move_exact_closure\":"
           << totals.certificate_move_closure
           << ",\"provider_move_exact\":" << totals.provider_move_exact
           << ",\"provider_move_total\":" << totals.provider_move_total
           << ",\"retrospective_move_exact_days\":"
           << totals.retrospective_move_exact_days
           << ",\"runtime_move_authorized_days\":"
           << totals.runtime_move_authorized_days
           << ",\"runtime_move_fail_closed_days\":"
           << totals.runtime_move_fail_closed_days
           << ",\"obligations\":" << totals.obligations
           << ",\"typed_crop_obligations\":" << totals.crop_obligations
           << ",\"typed_animal_obligations\":"
           << totals.animal_obligations
           << ",\"unified_targets\":" << totals.unified_targets
           << ",\"unified_groups\":" << totals.unified_groups
           << ",\"unified_admitted_groups\":"
           << totals.unified_admitted_groups
           << ",\"unified_demands\":" << totals.unified_demands
           << ",\"unified_diagnostics\":" << totals.unified_diagnostics
           << ",\"unified_planned_days\":" << totals.unified_planned_days
           << ",\"unified_takeover_eligible_days\":"
           << totals.unified_takeover_eligible_days
           << ",\"current_evidence\":" << totals.current_evidence
           << ",\"current_exact\":" << totals.current_exact
           << ",\"combined_exact\":" << totals.combined_exact
           << ",\"complete_crop_payloads\":"
           << totals.complete_crop_payloads
           << ",\"complete_animal_payloads\":"
           << totals.complete_animal_payloads
           << ",\"must_finish_debts\":" << totals.must_finish_debts
           << ",\"weed_no_effect_water_harvest\":"
           << totals.weed_no_effect_wh
           << ",\"weed_no_effect_current_exact\":"
           << totals.weed_no_effect_wh_current_exact
           << ",\"weed_no_effect_persistent_lineage\":"
           << totals.weed_no_effect_wh_lineage
           << ",\"weed_no_effect_lineage_additional\":"
           << totals.weed_no_effect_wh_lineage_additional
           << ",\"weed_no_effect_combined_exact\":"
           << totals.weed_no_effect_wh_combined_exact
           << ",\"weed_no_effect_lineage_ratio\":"
           << ratio(totals.weed_no_effect_wh_lineage,
                    totals.weed_no_effect_wh)
           << ",\"weed_no_effect_combined_exact_ratio\":"
           << ratio(totals.weed_no_effect_wh_combined_exact,
                    totals.weed_no_effect_wh)
           << ",\"unsupported\":";
    write_map(output, totals.unsupported);
    output << ",\"current_unresolved\":";
    write_map(output, totals.current_unresolved);
    output << ",\"weed_remaining_causes\":";
    write_map(output, totals.weed_remaining_causes);
    output << ",\"verification_rejects\":";
    write_map(output, totals.verification_rejects);
    output << ",\"debt_reasons\":";
    write_map(output, totals.debt_reasons);
    output << "},\n  \"days\":[\n";
    for (std::size_t index = 0; index < records.size(); ++index) {
      const auto& record = records[index];
      output << (index == 0 ? "    " : ",\n    ")
             << "{\"seed\":" << record.seed << ",\"player\":"
             << record.player << ",\"day\":" << record.day
             << ",\"observed_steps\":" << record.observed_steps
             << ",\"issue_reject\":\""
             << issue_reject_name(record.issue_reject)
             << "\",\"fully_proven\":"
             << (record.fully_proven ? "true" : "false")
             << ",\"moves\":" << record.moves
             << ",\"obligations\":" << record.obligations
             << ",\"typed_crop\":" << record.crop_obligations
             << ",\"typed_animal\":" << record.animal_obligations
             << ",\"unified_targets\":" << record.unified_targets
             << ",\"unified_groups\":" << record.unified_groups
             << ",\"unified_admitted_groups\":"
             << record.unified_admitted_groups
             << ",\"unified_demands\":" << record.unified_demands
             << ",\"unified_diagnostics\":" << record.unified_diagnostics
             << ",\"unified_planned\":"
             << (record.unified_planned ? "true" : "false")
             << ",\"unified_takeover_eligible\":"
             << (record.unified_takeover_eligible ? "true" : "false")
             << ",\"unsupported\":";
      write_map(output, record.unsupported);
      output << ",\"plan_reject\":\""
             << plan_reject_name(record.plan_reject)
             << "\",\"planned\":" << (record.planned ? "true" : "false")
             << ",\"partial_plan\":"
             << (record.partial_plan ? "true" : "false")
             << ",\"locally_full_certificate_eligible\":"
             << (record.locally_full_certificate_eligible ? "true" : "false")
             << ",\"full_certificate_eligible\":"
             << (record.full_certificate_eligible ? "true" : "false")
             << ",\"certificate_valid\":"
             << (record.certificate_valid ? "true" : "false")
             << ",\"verification_reject\":\""
             << plan_reject_name(record.verification_reject)
             << "\",\"verification_checked_slots\":"
             << record.verification_checked_slots
             << ",\"certificate_move_closure\":"
             << record.certificate_move_closure
             << ",\"debts\":" << record.debts
             << ",\"must_finish_debts\":" << record.must_finish_debts
             << ",\"debt_reasons\":";
      write_map(output, record.debt_reasons);
      output << ",\"provider_audited\":"
             << (record.provider_audited ? "true" : "false")
             << ",\"provider_valid\":"
             << (record.provider_valid ? "true" : "false")
             << ",\"provider_mismatches\":" << record.provider_mismatches
             << ",\"provider_move_exact\":" << record.provider_move_exact
             << ",\"provider_move_total\":" << record.provider_move_total
             << ",\"move_mismatches\":[";
      for (std::size_t mismatch = 0;
           mismatch < record.move_mismatches.size(); ++mismatch) {
        const auto& value = record.move_mismatches[mismatch];
        output << (mismatch == 0 ? "" : ",")
               << "{\"actor\":" << value.actor
               << ",\"source_step\":" << value.source_step
               << ",\"expected_op\":" << value.expected_op
               << ",\"actual_op\":" << value.actual_op
               << ",\"runtime_overlay\":\"" << value.runtime_overlay
               << "\"}";
      }
      output << ']'
             << ",\"current_exact\":" << record.current_exact
             << ",\"combined_exact\":" << record.combined_exact
             << ",\"persistent_crop_recoveries\":"
             << record.persistent_crop_recoveries
             << ",\"weed_no_effect_wh\":" << record.weed_no_effect_wh
             << ",\"weed_no_effect_wh_lineage\":"
             << record.weed_no_effect_wh_lineage
             << ",\"weed_no_effect_wh_combined_exact\":"
             << record.weed_no_effect_wh_combined_exact
             << ",\"current_unresolved\":";
      write_map(output, record.current_unresolved);
      output << ",\"weed_remaining_causes\":";
      write_map(output, record.weed_remaining_causes);
      output << '}';
    }
    output << "\n  ]\n}\n";
    std::cout << "days=" << totals.days << " full=" << totals.full_days
              << " planned=" << totals.planned_days
              << " verified=" << totals.certificate_valid_days
              << " triggers=" << totals.weed_no_effect_wh
              << " lineage=" << totals.weed_no_effect_wh_lineage
              << " combined=" << totals.weed_no_effect_wh_combined_exact
              << " unified_takeover="
              << totals.unified_takeover_eligible_days
              << '\n';
  } catch (const std::exception& error) {
    std::cerr << "g001 repair pipeline shadow audit failed: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
