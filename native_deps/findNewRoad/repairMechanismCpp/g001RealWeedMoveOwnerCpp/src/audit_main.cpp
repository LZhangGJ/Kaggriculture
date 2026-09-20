#include "g001_real_weed_move_owner.hpp"

#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

using fastkag::Action;
using fastkag::NativeAgentState;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
namespace owner = g001::real_weed_move_owner;

namespace {

void hash_value(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (8 * byte)) & 255U;
    hash *= 1099511628211ULL;
  }
}

void hash_action(std::uint64_t& hash, const PlayerAction& action) {
  hash_value(hash, action.units.size());
  for (const auto unit : action.units) {
    hash_value(hash, static_cast<std::uint8_t>(unit.op));
    hash_value(hash, static_cast<std::uint8_t>(unit.item));
    hash_value(hash, static_cast<std::uint32_t>(unit.quantity));
  }
  hash_value(hash, action.market.size());
  for (const auto market : action.market) {
    hash_value(hash, static_cast<std::uint8_t>(market.op));
    hash_value(hash, static_cast<std::uint8_t>(market.item));
    hash_value(hash, static_cast<std::uint32_t>(market.quantity));
  }
}

const char* op_name(Op op) {
  switch (op) {
    case Op::PASS: return "PASS";
    case Op::NORTH: return "NORTH";
    case Op::SOUTH: return "SOUTH";
    case Op::EAST: return "EAST";
    case Op::WEST: return "WEST";
    case Op::DIG: return "DIG";
    case Op::BUILD_PASTURE: return "BUILD_PASTURE";
    case Op::PICKUP: return "PICKUP";
    case Op::PLACE: return "PLACE";
    case Op::FEED: return "FEED";
    case Op::CARE: return "CARE";
    default: return "OTHER";
  }
}

const char* goal_name(g001::obligation_day::GoalKind goal) {
  using Goal = g001::obligation_day::GoalKind;
  switch (goal) {
    case Goal::CropReady: return "crop_ready";
    case Goal::Pickup: return "pickup";
    case Goal::Place: return "place";
    case Goal::Feed: return "feed";
    case Goal::Care: return "care";
    case Goal::Harvest: return "harvest";
    case Goal::BuildPasture: return "build_pasture";
    case Goal::CollectFertilizer: return "collect_fertilizer";
    case Goal::BuildCoop: return "build_coop";
  }
  return "unknown";
}

const char* disposition_name(
    g001::obligation_day::ObligationDisposition value) {
  using D = g001::obligation_day::ObligationDisposition;
  switch (value) {
    case D::Completed: return "completed";
    case D::BlockedOnReceipt: return "blocked_on_receipt";
    case D::CapacityDebt: return "capacity_debt";
    case D::DependencyDebt: return "dependency_debt";
    case D::DeadlineDebt: return "deadline_debt";
    case D::UnsupportedStateDebt: return "unsupported_state_debt";
    case D::PolicyDeferredDebt: return "policy_deferred_debt";
  }
  return "unknown";
}

struct EndSnapshot {
  fastkag::Farm farm;
  fastkag::PrivateState private_state;
  int unit_failures{};
  int market_failures{};
};

EndSnapshot snapshot(const Simulator& simulator, int unit_failures,
                     int market_failures) {
  return {simulator.farms()[1], simulator.privates()[1], unit_failures,
          market_failures};
}

bool same_tile(const fastkag::Tile& lhs, const fastkag::Tile& rhs) {
  return lhs.kind == rhs.kind && lhs.crop == rhs.crop &&
      lhs.animal == rhs.animal && lhs.planted_day == rhs.planted_day &&
      lhs.placed_day == rhs.placed_day && lhs.yield_units == rhs.yield_units &&
      lhs.consecutive_unwatered == rhs.consecutive_unwatered &&
      lhs.consecutive_unfed == rhs.consecutive_unfed &&
      lhs.fertilized_until_day == rhs.fertilized_until_day &&
      lhs.pending_care_bonus == rhs.pending_care_bonus &&
      lhs.max_lifespan_step == rhs.max_lifespan_step &&
      lhs.watered_today == rhs.watered_today &&
      lhs.fed_today == rhs.fed_today && lhs.cared_today == rhs.cared_today &&
      lhs.fertilizer_available == rhs.fertilizer_available;
}

void write_tile(std::ostream& output, const fastkag::Tile& tile) {
  output << "{\"kind\":" << static_cast<int>(tile.kind)
         << ",\"crop\":" << static_cast<int>(tile.crop)
         << ",\"animal\":" << static_cast<int>(tile.animal)
         << ",\"yield\":" << tile.yield_units
         << ",\"unwatered\":" << tile.consecutive_unwatered
         << ",\"unfed\":" << tile.consecutive_unfed
         << ",\"watered\":" << (tile.watered_today ? "true" : "false")
         << ",\"fed\":" << (tile.fed_today ? "true" : "false")
         << ",\"cared\":" << (tile.cared_today ? "true" : "false")
         << '}';
}

template <class Values>
void write_array(std::ostream& output, const Values& values) {
  output << '[';
  for (std::size_t i = 0; i < values.size(); ++i)
    output << (i ? "," : "") << values[i];
  output << ']';
}

void write_snapshot(std::ostream& output, const EndSnapshot& value) {
  output << "{\"farmer\":[" << value.farm.farmer.x << ','
         << value.farm.farmer.y << "],\"hands\":" << value.farm.hands.size()
         << ",\"money\":" << value.farm.money << ",\"shed\":";
  write_array(output, value.private_state.shed);
  output << ",\"seeds\":";
  write_array(output, value.private_state.seeds);
  output << ",\"inventories\":[";
  for (std::size_t actor = 0; actor < value.private_state.inventories.size();
       ++actor) {
    output << (actor ? "," : "");
    write_array(output, value.private_state.inventories[actor]);
  }
  output << "],\"unit_failures\":" << value.unit_failures
         << ",\"market_failures\":" << value.market_failures << '}';
}

struct BaselineResult {
  std::uint64_t hash{1469598103934665603ULL};
  std::uint64_t seat1_prefix_hash{1469598103934665603ULL};
  Op step187{Op::PASS};
  std::uint64_t day7_start_fingerprint{};
  int day7_farmer_x{-1};
  int day7_farmer_y{-1};
  int day7_hands{-1};
  int day7_money{};
  EndSnapshot day7_end;
  EndSnapshot terminal;
  double opponent_terminal_money{};
  int post_day_unit_failures{};
  int post_day_market_failures{};
  bool source191_effect{};
  std::map<int, bool> actor0_effects;
  std::map<int, Op> actor0_ops;
};

BaselineResult baseline(const NativeTeammateExecutor& executor) {
  Simulator simulator({}, 970017);
  std::array<NativeAgentState, 2> states;
  BaselineResult result;
  int day7_unit_failures = 0;
  int day7_market_failures = 0;
  for (int step = 0; step < simulator.config().episode_steps - 1; ++step) {
    if (step == 168) {
      result.day7_start_fingerprint =
          g001::production_suffix::focal_unit_state_fingerprint(simulator, 1);
      result.day7_farmer_x = simulator.farms()[1].farmer.x;
      result.day7_farmer_y = simulator.farms()[1].farmer.y;
      result.day7_hands = static_cast<int>(simulator.farms()[1].hands.size());
      result.day7_money = static_cast<int>(simulator.farms()[1].money);
    }
    std::array<PlayerAction, 2> actions;
    for (int player = 0; player < 2; ++player) {
      actions[player] = executor.action_external(
          simulator, player, 0, states[player]);
      hash_action(result.hash, actions[player]);
      if (player == 1 && step < 168)
        hash_action(result.seat1_prefix_hash, actions[player]);
    }
    if (step >= 168 && step <= 191) {
      day7_unit_failures +=
          fastkag::native_macro_unit_failures(simulator, 1, actions[1]);
      {
        std::array<PlayerAction, 2> isolated;
        isolated[1].units.resize(actions[1].units.size());
        isolated[1].units[0] = actions[1].units[0];
        const auto preview = simulator.preview_unit_phase(isolated);
        const bool effect =
            g001::production_suffix::focal_unit_state_fingerprint(
                simulator, 1, step) !=
            g001::production_suffix::focal_unit_state_fingerprint(
                preview, 1, step);
        result.actor0_effects[step] = effect;
        result.actor0_ops[step] = actions[1].units[0].op;
        if (step == 191) result.source191_effect = effect;
      }
    }
    if (step == 187) result.step187 = actions[1].units.at(0).op;
    if (step > 191)
      result.post_day_unit_failures +=
          fastkag::native_macro_unit_failures(simulator, 1, actions[1]);
    simulator.step(actions);
    if (step >= 168 && step <= 191)
      day7_market_failures +=
          fastkag::native_macro_market_failures(simulator, 1, actions[1]);
    if (step == 191)
      result.day7_end = snapshot(simulator, day7_unit_failures,
                                 day7_market_failures);
    if (step > 191)
      result.post_day_market_failures +=
          fastkag::native_macro_market_failures(simulator, 1, actions[1]);
  }
  result.terminal = snapshot(simulator, result.post_day_unit_failures,
                             result.post_day_market_failures);
  result.opponent_terminal_money = simulator.farms()[0].money;
  return result;
}

struct ReplayCapture {
  owner::DayAudit day7;
  bool found{};
  int source187_emitted_step{-1};
  Op source187_emitted_op{Op::PASS};
  int day7_start_step{-1};
  int day7_farmer_x{-1};
  int day7_farmer_y{-1};
  int day7_hands{-1};
  int day7_money{};
  std::uint64_t default_prefix_hash{1469598103934665603ULL};
  EndSnapshot day7_end;
  EndSnapshot terminal;
  double opponent_terminal_money{};
  int post_day_unit_failures{};
  int post_day_market_failures{};
  int deferred_source_step{-1};
  bool fail_stopped{};
  bool native_state_leaked{};
  bool native_proposal_discarded{};
};

ReplayCapture owned_replay(const NativeTeammateExecutor& executor,
                           int deferred_source_step = -1,
                           std::optional<owner::State::OverlayTestInjection>
                               overlay_injection = std::nullopt) {
  Simulator simulator({}, 970017);
  NativeAgentState player0;
  owner::State player1;
  player1.overlay_test_injection = overlay_injection;
  ReplayCapture result;
  result.deferred_source_step = deferred_source_step;
  int day7_unit_failures = 0;
  int day7_market_failures = 0;
  for (int step = 0; step < simulator.config().episode_steps - 1; ++step) {
    if (step == 192) owner::finish_day(player1);
    if (step == 168) {
      result.day7_start_step = simulator.step_count();
      result.day7_farmer_x = simulator.farms()[1].farmer.x;
      result.day7_farmer_y = simulator.farms()[1].farmer.y;
      result.day7_hands = static_cast<int>(simulator.farms()[1].hands.size());
      result.day7_money = static_cast<int>(simulator.farms()[1].money);
    }
    std::array<PlayerAction, 2> actions;
    actions[0] = executor.action_external(simulator, 0, 0, player0);
    if (step < 168) {
      actions[1] = executor.action_external(
          simulator, 1, 0, player1.native);
      owner::observe_final_provider_action(simulator, 1, actions[1], player1);
    } else if (step <= 191) {
      if (step == 168 && deferred_source_step >= 0) {
        const auto generation =
            (970017ULL << 20) | (2ULL << 16) |
            static_cast<std::uint64_t>(simulator.day() + 1);
        const auto issued = g001::day_start_issuer::issue_day_start(
            {&simulator, &executor.route_tape(0), 1, generation,
             &player1.persistent_lineage, {}, {}});
        const auto obligation = std::find_if(
            issued.obligations.begin(), issued.obligations.end(),
            [&](const auto& value) {
              return value.actor == 0 &&
                     value.source_step == deferred_source_step;
            });
        const auto source = std::find_if(
            issued.raw_sources.begin(), issued.raw_sources.end(),
            [&](const auto& value) {
              return value.actor == 0 &&
                     value.source_step == deferred_source_step;
            });
        if (!issued.issued() || obligation == issued.obligations.end() ||
            source == issued.raw_sources.end())
          throw std::runtime_error(
              "typed debt selection source was not issued");
        owner::DebtSelectionIdentity identity;
        identity.player = 1;
        identity.day = simulator.day();
        identity.issuer_generation = generation;
        identity.obligation_id = obligation->id;
        identity.obligation_content_hash =
            owner::production_obligation_identity_hash(*obligation);
        identity.actor = obligation->actor;
        identity.source_step = obligation->source_step;
        identity.source_action = source->action;
        identity.goal = obligation->goal;
        identity.item = obligation->item;
        identity.tile = obligation->tile;
        identity.quantity = obligation->quantity;
        identity.content_hash = owner::debt_selection_identity_hash(identity);
        player1.policy_deferred_authorization = identity;
      }
      const int native_last_step_before = player1.native.last_step;
      try {
        actions[1] = owner::action_external(
            executor, simulator, 1, 0, player1,
            (970017ULL << 20) | (2ULL << 16) |
                static_cast<std::uint64_t>(simulator.day() + 1));
      } catch (const std::runtime_error&) {
        result.fail_stopped = true;
        result.native_state_leaked =
            player1.native.last_step != native_last_step_before;
        result.native_proposal_discarded =
            player1.audit.native_proposal_discarded;
        return result;
      }
    } else {
      actions[1] = executor.action_external(
          simulator, 1, 0, player1.native);
    }
    if (step < 168) hash_action(result.default_prefix_hash, actions[1]);
    if (player1.active && player1.audit.day == 7 &&
        player1.owned_actor >= 0 &&
        player1.owned_actor < static_cast<int>(actions[1].units.size())) {
      PlayerAction isolated;
      isolated.units.resize(actions[1].units.size());
      isolated.units[player1.owned_actor] =
          actions[1].units[player1.owned_actor];
      if (fastkag::native_macro_unit_failures(simulator, 1, isolated) > 0)
        ++player1.audit.runtime_unit_failures;
    }
    if (step >= 168)
      day7_unit_failures +=
          step <= 191
              ? fastkag::native_macro_unit_failures(simulator, 1, actions[1])
              : 0;
    if (step > 191)
      result.post_day_unit_failures +=
          fastkag::native_macro_unit_failures(simulator, 1, actions[1]);
    simulator.step(actions);
    if (step >= 168)
      day7_market_failures +=
          step <= 191
              ? fastkag::native_macro_market_failures(simulator, 1, actions[1])
              : 0;
    if (step > 191)
      result.post_day_market_failures +=
          fastkag::native_macro_market_failures(simulator, 1, actions[1]);
    if (step == 191)
      result.day7_end = snapshot(simulator, day7_unit_failures,
                                 day7_market_failures);
  }
  owner::finish_day(player1);
  result.terminal = snapshot(simulator, result.post_day_unit_failures,
                             result.post_day_market_failures);
  result.opponent_terminal_money = simulator.farms()[0].money;
  for (const auto& day : player1.completed_days) {
    if (day.day != 7) continue;
    result.day7 = day;
    result.found = true;
    for (const auto& replay : day.move_replays) {
      if (replay.actor == 0 && replay.source_step == 187) {
        result.source187_emitted_step = replay.emitted_step;
        result.source187_emitted_op = replay.action.op;
      }
    }
  }
  return result;
}

bool feasible_candidate(const ReplayCapture& value) {
  return value.found && value.day7.planned && value.day7.certificate_valid &&
      value.day7.scoped_manifest_proven &&
      value.day7.exclusive_owner_admitted &&
      !value.day7.admission_rejected &&
      !value.day7.overlay_conflict_fail_stop &&
      value.day7.owned_source_coverage_total == 24 &&
      value.day7.owned_source_move_coverage +
              value.day7.owned_source_obligation_coverage +
              value.day7.owned_source_pass_coverage ==
          24 &&
      value.day7.owned_source_missing == 0 &&
      value.day7.owned_source_duplicates == 0 &&
      value.day7.owned_source_unsupported == 0 &&
      value.day7.stationary_completed == 1 &&
      value.day7.stationary_debts == 0 &&
      !value.day7.unowned_overlay_suppressed &&
      value.day7.runtime_unit_failures == 0 &&
      value.day7.move_emitted == value.day7.move_tokens &&
      value.day7.completed == value.day7.obligations - 1 &&
      value.day7.debts == 1;
}

int animal_quality(const EndSnapshot& value) {
  int result = 0;
  for (const auto& tile : value.farm.tiles) {
    if (tile.kind != fastkag::TileKind::ANIMAL) continue;
    result += 4 + tile.fed_today + tile.cared_today;
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string output = argc > 1 ? argv[1] : "g001-real-owner.json";
    NativeTapeLibrary library;
    library.routes.push_back(g001::repair::load_route(
        G001_OWNER_TAPES, G001_OWNER_LIBRARY, "G001"));
    NativeTeammateExecutor executor(std::move(library));

    const auto base1 = baseline(executor);
    const auto base2 = baseline(executor);
    if (base1.hash != base2.hash || base1.step187 != Op::CARE)
      throw std::runtime_error("real baseline replay/hash regression changed");
    const auto natural = owned_replay(executor);
    owner::State::OverlayTestInjection positive_overlay;
    positive_overlay.step = 170;
    positive_overlay.actor = 0;
    positive_overlay.action = Action{Op::CARE};
    const auto overlay_fail_stop =
        owned_replay(executor, -1, positive_overlay);
    auto mutation_overlay = positive_overlay;
    mutation_overlay.mutate_candidate_native_before_resign = true;
    const auto mutation_fail_stop =
        owned_replay(executor, -1, mutation_overlay);
    auto tampered_overlay = positive_overlay;
    tampered_overlay.tamper_move_source_before_resign = true;
    const auto tampered_fail_stop =
        owned_replay(executor, -1, tampered_overlay);
    if (natural.day7.interphase_resigns != 24 ||
        natural.day7.remaining_resigns != 24 ||
        natural.day7.remaining_resign_fail_stop ||
        !overlay_fail_stop.fail_stopped ||
        overlay_fail_stop.native_state_leaked ||
        !overlay_fail_stop.native_proposal_discarded ||
        !mutation_fail_stop.fail_stopped ||
        mutation_fail_stop.native_state_leaked ||
        !mutation_fail_stop.native_proposal_discarded ||
        !tampered_fail_stop.fail_stopped ||
        tampered_fail_stop.native_state_leaked)
      throw std::runtime_error("remaining-day suffix resign gate failed");
    std::vector<ReplayCapture> candidates;
    candidates.push_back(natural);
    for (const auto& obligation : natural.day7.obligation_audits) {
      if (obligation.obligation.goal ==
          g001::obligation_day::GoalKind::BuildPasture) continue;
      candidates.push_back(
          owned_replay(executor, obligation.obligation.source_step));
    }
    const auto selected = std::max_element(
        candidates.begin(), candidates.end(), [&](const auto& lhs,
                                                    const auto& rhs) {
          const auto score = [&](const auto& value) {
            const int failures = value.day7_end.unit_failures +
                value.day7_end.market_failures +
                value.post_day_unit_failures + value.post_day_market_failures;
            const double margin = value.terminal.farm.money -
                                  value.opponent_terminal_money;
            const bool baseline_sacrifice_no_effect =
                value.deferred_source_step >= 0 &&
                !base1.actor0_effects.at(value.deferred_source_step);
            return std::tuple{feasible_candidate(value), -failures, margin,
                              animal_quality(value.terminal),
                              baseline_sacrifice_no_effect,
                              value.deferred_source_step >= 0,
                              -value.deferred_source_step};
          };
          return score(lhs) < score(rhs);
        });
    if (selected == candidates.end())
      throw std::runtime_error("no owner candidate was evaluated");
    const auto& replay = *selected;
    const auto debt_obligation = std::find_if(
        replay.day7.obligation_audits.begin(),
        replay.day7.obligation_audits.end(), [](const auto& value) {
          return value.disposition !=
                 g001::obligation_day::ObligationDisposition::Completed;
        });
    if (!replay.found || replay.day7.stationary_submissions != 1 ||
        replay.day7.owned_actor != 0 || replay.day7.move_tokens != 7 ||
        !replay.day7.planned || !replay.day7.certificate_valid ||
        replay.day7.move_emitted != 7 ||
        replay.source187_emitted_op != Op::SOUTH ||
        replay.source187_emitted_step < 187 ||
        replay.source187_emitted_step > 191 ||
        replay.default_prefix_hash != base1.seat1_prefix_hash ||
        replay.day7.start_fingerprint != base1.day7_start_fingerprint ||
        replay.day7_farmer_x != base1.day7_farmer_x ||
        replay.day7_farmer_y != base1.day7_farmer_y ||
        replay.day7_hands != base1.day7_hands ||
        replay.day7_money != base1.day7_money ||
        replay.day7.unowned_overlay_suppressed ||
        replay.day7.admission_rejected ||
        replay.day7.overlay_conflict_fail_stop ||
        !replay.day7.scoped_manifest_proven ||
        !replay.day7.exclusive_owner_admitted ||
        replay.day7.owned_source_coverage_total != 24 ||
        replay.day7.owned_source_move_coverage +
                replay.day7.owned_source_obligation_coverage +
                replay.day7.owned_source_pass_coverage !=
            24 ||
        replay.day7.owned_source_missing != 0 ||
        replay.day7.owned_source_duplicates != 0 ||
        replay.day7.owned_source_unsupported != 0 ||
        replay.day7.runtime_unit_failures != 0 ||
        replay.day7.stationary_completed != 1 ||
        replay.day7.stationary_debts != 0 ||
        debt_obligation == replay.day7.obligation_audits.end() ||
        !feasible_candidate(replay))
      throw std::runtime_error("real day7 weed+MOVE ownership gate failed");

    std::ofstream json(output, std::ios::trunc);
    if (!json) throw std::runtime_error("cannot create owner audit artifact");
    const auto& day = replay.day7;
    json << "{\n  \"schema\":\"g001-real-weed-move-owner-v1\",\n"
         << "  \"replay\":{\"seed\":970017,\"seat\":1,\"day\":7,"
            "\"provider\":\"NativeTeammateExecutor+default-off-obligation-owner\"},\n"
         << "  \"oracle_scope\":{\"offline\":true,"
            "\"uses_terminal_future_information\":true,"
            "\"online_deployable\":false},\n"
         << "  \"default_off\":{\"native_mask_bit_added\":false,"
            "\"baseline_step187\":\"" << op_name(base1.step187)
         << "\",\"action_hash_1\":" << base1.hash
         << ",\"action_hash_2\":" << base2.hash
         << ",\"byte_deterministic\":true},\n"
         << "  \"real_day_start\":{\"step\":" << replay.day7_start_step
         << ",\"fingerprint\":" << day.start_fingerprint
         << ",\"farmer\":[" << replay.day7_farmer_x << ','
         << replay.day7_farmer_y << "],\"hands\":" << replay.day7_hands
         << ",\"money\":" << replay.day7_money << "},\n"
         << "  \"owner\":{\"actor\":" << day.owned_actor
         << ",\"stationary_submissions\":" << day.stationary_submissions
         << ",\"move_tokens\":" << day.move_tokens
         << ",\"obligations\":" << day.obligations
         << ",\"completed\":" << day.completed
         << ",\"explicit_debts\":" << day.debts
         << ",\"move_delays\":" << day.move_delays
         << ",\"planned\":" << (day.planned ? "true" : "false")
         << ",\"certificate_valid\":"
         << (day.certificate_valid ? "true" : "false")
         << ",\"scoped_manifest_proven\":"
         << (day.scoped_manifest_proven ? "true" : "false")
         << ",\"exclusive_owner_admitted\":"
         << (day.exclusive_owner_admitted ? "true" : "false")
         << ",\"admission_rejected\":"
         << (day.admission_rejected ? "true" : "false")
         << ",\"overlay_conflict_fail_stop\":"
         << (day.overlay_conflict_fail_stop ? "true" : "false")
         << ",\"unowned_overlay_suppressed\":"
         << (day.unowned_overlay_suppressed ? "true" : "false")
         << ",\"runtime_unit_failures\":" << day.runtime_unit_failures
         << ",\"stationary_completed\":" << day.stationary_completed
         << ",\"stationary_debts\":" << day.stationary_debts
         << "},\n  \"owned_source_coverage\":{\"total\":"
         << day.owned_source_coverage_total << ",\"move\":"
         << day.owned_source_move_coverage << ",\"obligation\":"
         << day.owned_source_obligation_coverage << ",\"pass\":"
         << day.owned_source_pass_coverage << ",\"missing\":"
         << day.owned_source_missing << ",\"duplicates\":"
         << day.owned_source_duplicates << ",\"unsupported\":"
         << day.owned_source_unsupported
         << "},\n  \"remaining_suffix_resign_tests\":{"
            "\"real_observation_resigns\":"
         << natural.day7.interphase_resigns
         << ",\"overlay_injected_step\":170,\"overlay_fail_stop\":"
         << (overlay_fail_stop.fail_stopped ? "true" : "false")
         << ",\"overlay_native_state_leaked\":"
         << (overlay_fail_stop.native_state_leaked ? "true" : "false")
         << ",\"mutation_injection_fail_stop\":"
         << (mutation_fail_stop.fail_stopped ? "true" : "false")
         << ",\"mutation_native_state_leaked\":"
         << (mutation_fail_stop.native_state_leaked ? "true" : "false")
         << ",\"candidate_native_discarded\":"
         << (mutation_fail_stop.native_proposal_discarded ? "true" : "false")
         << ",\"tampered_move_source_fail_stop\":"
         << (tampered_fail_stop.fail_stopped ? "true" : "false")
         << "},\n  \"stationary\":[";
    for (std::size_t i = 0; i < day.stationary.size(); ++i) {
      const auto& value = day.stationary[i];
      json << (i ? "," : "") << "{\"actor\":" << value.actor
           << ",\"source_step\":" << value.source_step
           << ",\"tile\":[" << value.tile.x << ',' << value.tile.y
           << "],\"op\":\"" << op_name(value.intended.op) << "\"}";
    }
    json << "],\n  \"raw_move_tokens\":[";
    for (std::size_t i = 0; i < day.raw_move_tokens.size(); ++i) {
      const auto& value = day.raw_move_tokens[i];
      json << (i ? "," : "") << "{\"actor\":" << value.actor
           << ",\"source_step\":" << value.source_step
           << ",\"direction\":\"" << op_name(value.action.op) << "\"}";
    }
    json << "],\n  \"move_replays\":[";
    for (std::size_t i = 0; i < day.move_replays.size(); ++i) {
      const auto& value = day.move_replays[i];
      json << (i ? "," : "") << "{\"actor\":" << value.actor
           << ",\"source_step\":" << value.source_step
           << ",\"emitted_step\":" << value.emitted_step
           << ",\"direction\":\"" << op_name(value.action.op) << "\"}";
    }
    json << "],\n  \"debts\":[";
    for (std::size_t i = 0; i < day.explicit_debts.size(); ++i) {
      const auto& value = day.explicit_debts[i];
      json << (i ? "," : "") << "{\"obligation_id\":"
           << value.obligation_id << ",\"reason\":\""
           << g001::obligation_day::debt_reason_name(value.reason)
           << "\",\"remaining_transitions\":"
           << value.remaining_transitions << '}';
    }
    json << "],\n  \"obligations\":[";
    for (std::size_t i = 0; i < day.obligation_audits.size(); ++i) {
      const auto& value = day.obligation_audits[i];
      json << (i ? "," : "") << "{\"id\":" << value.obligation.id
           << ",\"goal\":\"" << goal_name(value.obligation.goal)
           << "\",\"item\":" << static_cast<int>(value.obligation.item)
           << ",\"tile\":[" << value.obligation.tile.x << ','
           << value.obligation.tile.y << "],\"source_step\":"
           << value.obligation.source_step << ",\"status\":\""
           << disposition_name(value.disposition)
           << "\",\"remaining_transitions\":"
           << value.remaining_transitions << ",\"transition_steps\":";
      write_array(json, value.transition_steps);
      json << '}';
    }
    const int debt_source = debt_obligation->obligation.source_step;
    json << "],\n  \"debt_selection_identity\":";
    if (day.debt_selection_identity) {
      const auto& identity = *day.debt_selection_identity;
      json << "{\"player\":" << identity.player << ",\"day\":"
           << identity.day << ",\"issuer_generation\":"
           << identity.issuer_generation << ",\"obligation_id\":"
           << identity.obligation_id << ",\"obligation_content_hash\":"
           << identity.obligation_content_hash << ",\"actor\":"
           << identity.actor << ",\"source_step\":"
           << identity.source_step << ",\"source_op\":\""
           << op_name(identity.source_action.op) << "\",\"goal\":\""
           << goal_name(identity.goal) << "\",\"item\":"
           << static_cast<int>(identity.item) << ",\"tile\":["
           << identity.tile.x << ',' << identity.tile.y
           << "],\"quantity\":" << identity.quantity
           << ",\"content_hash\":" << identity.content_hash << '}';
    } else {
      json << "null";
    }
    json << ",\n  \"debt_provenance\":{\"source_step\":"
         << debt_obligation->obligation.source_step
         << ",\"raw_op\":\"" << op_name(base1.actor0_ops.at(debt_source))
         << "\",\"baseline_had_effect\":"
         << (base1.actor0_effects.at(debt_source) ? "true" : "false")
         << ",\"owner_status\":\""
         << disposition_name(debt_obligation->disposition) << "\"},\n"
            "  \"post_day_state\":{\"baseline\":";
    write_snapshot(json, base1.day7_end);
    json << ",\"owner\":";
    write_snapshot(json, replay.day7_end);
    json << ",\"tile_diffs\":[";
    bool first_diff = true;
    for (std::size_t tile = 0; tile < base1.day7_end.farm.tiles.size(); ++tile) {
      if (same_tile(base1.day7_end.farm.tiles[tile],
                    replay.day7_end.farm.tiles[tile])) continue;
      json << (first_diff ? "" : ",") << "{\"tile\":["
           << tile % 10 << ',' << tile / 10 << "],\"baseline\":";
      write_tile(json, base1.day7_end.farm.tiles[tile]);
      json << ",\"owner\":";
      write_tile(json, replay.day7_end.farm.tiles[tile]);
      json << '}';
      first_diff = false;
    }
    json << "]},\n  \"candidate_selection\":{\"selected_deferred_source\":"
         << replay.deferred_source_step << ",\"objective\":"
            "\"feasible_then_min_failures_then_terminal_margin_then_animal_quality\","
            "\"candidates\":[";
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const auto& value = candidates[i];
      json << (i ? "," : "") << "{\"kind\":\"policy_deferred\","
           << "\"source_step\":" << value.deferred_source_step
           << ",\"feasible\":"
           << (feasible_candidate(value) ? "true" : "false")
           << ",\"completed\":" << value.day7.completed
           << ",\"debts\":" << value.day7.debts
           << ",\"day7_unit_failures\":"
           << value.day7_end.unit_failures
           << ",\"day7_market_failures\":"
           << value.day7_end.market_failures
           << ",\"post_day_unit_failures\":"
           << value.post_day_unit_failures
           << ",\"post_day_market_failures\":"
           << value.post_day_market_failures
           << ",\"terminal_money\":" << value.terminal.farm.money
           << ",\"terminal_margin\":"
           << value.terminal.farm.money - value.opponent_terminal_money
           << ",\"terminal_animal_quality\":"
           << animal_quality(value.terminal) << '}';
    }
    json << "],\"cross_actor_transfer\":{\"feasible\":false,"
            "\"reason\":\"real_day_start_has_zero_hands\"}},\n"
            "  \"terminal_comparison\":{\"baseline\":";
    write_snapshot(json, base1.terminal);
    json << ",\"selected_owner\":";
    write_snapshot(json, replay.terminal);
    json << ",\"baseline_margin\":"
         << base1.terminal.farm.money - base1.opponent_terminal_money
         << ",\"owner_margin\":"
         << replay.terminal.farm.money - replay.opponent_terminal_money
         << ",\"money_delta\":"
         << replay.terminal.farm.money - base1.terminal.farm.money
         << ",\"margin_delta\":"
         << (replay.terminal.farm.money - replay.opponent_terminal_money) -
                (base1.terminal.farm.money - base1.opponent_terminal_money)
         << "},\n"
            "  \"baseline_terminal\":";
    write_snapshot(json, base1.terminal);
    json << ",\n  \"selected_terminal_delta\":{\"money\":"
         << replay.terminal.farm.money - base1.terminal.farm.money
         << ",\"margin\":"
         << (replay.terminal.farm.money - replay.opponent_terminal_money) -
                (base1.terminal.farm.money - base1.opponent_terminal_money)
         << ",\"unit_failures\":"
         << replay.terminal.unit_failures - base1.terminal.unit_failures
         << ",\"market_failures\":"
         << replay.terminal.market_failures - base1.terminal.market_failures
         << "},\n"
            "  \"source187\":{\"raw\":\"SOUTH\","
            "\"legacy_final\":\"CARE\",\"owner_emitted_step\":"
         << replay.source187_emitted_step << ",\"owner_final\":\""
         << op_name(replay.source187_emitted_op) << "\"}\n}\n";
    std::cout << "g001_real_weed_move_owner_audit: PASS output=" << output
              << " hash=" << base1.hash << " source187_emitted="
              << replay.source187_emitted_step << " debts=" << day.debts
              << " runtime_failures=" << day.runtime_unit_failures << '\n';
  } catch (const std::exception& error) {
    std::cerr << "g001_real_weed_move_owner_audit: " << error.what() << '\n';
    return 1;
  }
}
