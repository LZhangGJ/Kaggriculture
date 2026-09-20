#include "g001_continuous_runtime_receipt_adapter.hpp"
#include "g001_continuous_rolling_runtime_receipt_adapter.hpp"
#include "g001_day_start_obligation_issuer.hpp"
#include "g001_real_weed_move_owner.hpp"
#include "native_final_action_commit.hpp"
#include "repair_fork_evaluator.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace owner = g001::real_weed_move_owner;
namespace issuer = g001::day_start_issuer;
namespace runtime_receipt = g001::day_runtime_receipt;
namespace receipt_adapter = g001::continuous_runtime_receipt;
namespace rolling_receipt = g001::continuous_rolling_runtime_receipt;
namespace final_commit = g001::native_final_commit;
using fastkag::Action;
using fastkag::NativeAgentState;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;

namespace {

constexpr std::uint64_t kSeed = 970017;
constexpr int kSeat = 1;
constexpr int kOwnerStart = 168;

bool same_action(Action left, Action right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

const char* op_name(Op op) {
  switch (op) {
    case Op::PASS: return "PASS";
    case Op::NORTH: return "NORTH";
    case Op::SOUTH: return "SOUTH";
    case Op::EAST: return "EAST";
    case Op::WEST: return "WEST";
    case Op::DIG: return "DIG";
    case Op::PLANT: return "PLANT";
    case Op::WATER: return "WATER";
    case Op::HARVEST: return "HARVEST";
    case Op::BUILD_PASTURE: return "BUILD_PASTURE";
    case Op::PICKUP: return "PICKUP";
    case Op::PLACE: return "PLACE";
    case Op::FEED: return "FEED";
    case Op::CARE: return "CARE";
    case Op::HIRE: return "HIRE";
    case Op::BUY_LAND: return "BUY_LAND";
    case Op::BUY_SEED: return "BUY_SEED";
    case Op::BUY_PRODUCT: return "BUY_PRODUCT";
    case Op::BUY_ANIMAL: return "BUY_ANIMAL";
    case Op::SELL: return "SELL";
    default: return "OTHER";
  }
}

std::uint64_t generation(int day) {
  return (kSeed << 20) | (2ULL << 16) |
         static_cast<std::uint64_t>(day + 1);
}

int active_weeds(const NativeAgentState& state) {
  return static_cast<int>(std::count_if(
      state.weed.begin(), state.weed.end(),
      [](const auto& value) { return value.active; }));
}

int active_realignments(const NativeAgentState& state) {
  return static_cast<int>(std::count_if(
      state.experimental_realign.begin(), state.experimental_realign.end(),
      [](const auto& value) { return value.active; }));
}

const fastkag::Tile* tile_at(const Simulator& env, int player,
                             fastkag::Position value) {
  const int size = env.config().board_size;
  if (value.x < 0 || value.y < 0 || value.x >= size || value.y >= size)
    return nullptr;
  return &env.farms()[player].tiles[
      static_cast<std::size_t>(value.y * size + value.x)];
}

bool day_start_weed(const NativeTeammateExecutor& executor,
                    const Simulator& env, int player) {
  if (env.hour() != 0) return false;
  std::vector<fastkag::Position> projected{env.farms()[player].farmer};
  projected.insert(projected.end(), env.farms()[player].hands.begin(),
                   env.farms()[player].hands.end());
  const auto& tape = executor.route_tape(0);
  const int end = std::min(env.step_count() + 23,
                           static_cast<int>(tape.size()) - 1);
  for (int source = env.step_count(); source <= end; ++source) {
    const auto& units = tape[static_cast<std::size_t>(source)].units;
    for (std::size_t actor = 0;
         actor < projected.size() && actor < units.size(); ++actor) {
      const auto action = units[actor];
      const auto* tile = tile_at(env, player, projected[actor]);
      if (action.op == Op::BUILD_PASTURE && tile &&
          tile->kind == fastkag::TileKind::WEED)
        return true;
      if (action.op == Op::NORTH) --projected[actor].y;
      if (action.op == Op::SOUTH) ++projected[actor].y;
      if (action.op == Op::WEST) --projected[actor].x;
      if (action.op == Op::EAST) ++projected[actor].x;
    }
  }
  return false;
}

struct DayInstall {
  int day{-1};
  int step{-1};
  int actors{};
  int stationary{};
  int native_weed_before{};
  int native_weed_after{};
  int native_realign_before{};
  int native_realign_after{};
  bool owner_active{};
  int owned_actor{-1};
  bool planned{};
  bool certificate_valid{};
  bool fail_closed{};
  bool authorization_present{};
  bool authorization_matches_day{};
  bool authorization_present_after{};
  bool owner_receipt_pending_before{};
  bool owner_receipt_pending_after{};
  int finalized_prior_day{-1};
  int finalized_prior_receipt_checks{};
  int finalized_prior_receipt_failures{};
  int persistent_unsupported{};
  int empty_lineage_unsupported{};
  bool persistent_fully_proven{};
  bool empty_lineage_fully_proven{};
  int source_total{};
  int source_moves{};
  int source_obligations{};
  int source_passes{};
  int source_missing{};
  int source_duplicates{};
  int source_unsupported{};
  bool scoped_manifest_proven{};
  bool exclusive_owner_admitted{};
  bool admission_rejected{};
  bool overlay_conflict_fail_stop{};
  bool receipt_fingerprint_diagnostic_only{true};
  std::map<std::string, int> persistent_reasons;
  std::map<std::string, int> empty_reasons;
};

struct Conflict {
  int step{-1};
  int actor{-1};
  Action raw{};
  Action native_final{};
  Action owner_final{};
  bool raw_native{};
  bool raw_owner{};
};

struct FirstDiff {
  int step{-1};
  std::uint64_t baseline_before{};
  std::uint64_t owner_before{};
  std::uint64_t baseline_after{};
  std::uint64_t owner_after{};
  Action baseline_action{};
  Action owner_action{};
  int baseline_x{-1};
  int baseline_y{-1};
  int owner_x{-1};
  int owner_y{-1};
  double baseline_money{};
  double owner_money{};
};

struct MoveClosure {
  int day{-1};
  bool applicable{};
  int raw_tokens{};
  int replay_tokens{};
  int emitted{};
  bool exact{};
  int first_bad_source{-1};
  int receipt_checks{};
  int receipt_failures{};
  int first_receipt_failure_step{-1};
};

MoveClosure closure(const owner::DayAudit& day) {
  MoveClosure out;
  out.day = day.day;
  out.applicable = day.exclusive_owner_admitted;
  out.raw_tokens = static_cast<int>(day.raw_move_tokens.size());
  out.replay_tokens = static_cast<int>(day.move_replays.size());
  out.emitted = day.move_emitted;
  std::set<std::pair<int, int>> seen;
  std::size_t cursor = 0;
  for (const auto& raw : day.raw_move_tokens) {
    while (cursor < day.move_replays.size() &&
           day.move_replays[cursor].actor != raw.actor)
      ++cursor;
    if (cursor >= day.move_replays.size() ||
        day.move_replays[cursor].source_step != raw.source_step ||
        !same_action(day.move_replays[cursor].action, raw.action) ||
        !seen.insert({raw.actor, raw.source_step}).second) {
      out.first_bad_source = raw.source_step;
      break;
    }
    ++cursor;
  }
  out.exact = out.first_bad_source < 0 &&
              out.raw_tokens == out.replay_tokens &&
              out.raw_tokens == out.emitted;
  out.receipt_checks = day.receipt_checks;
  out.receipt_failures = day.receipt_failures;
  out.first_receipt_failure_step = day.first_receipt_failure_step;
  return out;
}

void add_reasons(const issuer::IssueResult& issued,
                 std::map<std::string, int>& reasons) {
  for (const auto& unsupported : issued.unsupported)
    ++reasons[issuer::unsupported_reason_name(unsupported.reason)];
}

void json_string(std::ostream& out, const std::string& value) {
  out << '"';
  for (const char ch : value) {
    if (ch == '"' || ch == '\\') out << '\\';
    if (ch == '\n') out << "\\n";
    else out << ch;
  }
  out << '"';
}

void write_action(std::ostream& out, Action action) {
  out << "{\"op\":\"" << op_name(action.op) << "\",\"item\":"
      << static_cast<int>(action.item) << ",\"quantity\":"
      << action.quantity << '}';
}

void write_component_diff(std::ostream& out,
                          const receipt_adapter::ComponentDiff& value) {
  out << "{\"empty\":" << (value.empty() ? "true" : "false")
      << ",\"money\":" << (value.money ? "true" : "false")
      << ",\"farmer\":" << (value.farmer ? "true" : "false")
      << ",\"hand_positions\":" << value.hand_positions
      << ",\"tiles\":" << value.tiles
      << ",\"shed_cells\":" << value.shed_cells
      << ",\"seed_cells\":" << value.seed_cells
      << ",\"inventory_cells\":" << value.inventory_cells
      << ",\"inventory_orders\":" << value.inventory_orders << '}';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string output = argc > 1
                                   ? argv[1]
                                   : "g001-continuous-owner-audit.json";
    NativeTapeLibrary library;
    library.routes.push_back(g001::repair::load_route(
        G001_CONTINUOUS_TAPES, G001_CONTINUOUS_LIBRARY, "G001"));
    NativeTeammateExecutor executor(std::move(library));
    const auto& tape = executor.route_tape(0);

    Simulator baseline({}, kSeed);
    Simulator candidate({}, kSeed);
    std::array<NativeAgentState, 2> baseline_state;
    NativeAgentState candidate_opponent;
    owner::State candidate_owner;
    std::vector<DayInstall> installs;
    std::vector<Conflict> conflicts;
    std::optional<FirstDiff> first_diff;
    std::optional<int> first_owner_exception_step;
    std::string first_owner_exception;
    std::uint64_t baseline_start_fingerprint{};
    std::uint64_t owner_start_fingerprint{};
    int owner_calls{};
    int direct_default_calls_after_start{};
    int first_raw_native_conflict{-1};
    int first_raw_owner_conflict{-1};
    int first_raw_native_op_conflict{-1};
    int first_raw_owner_op_conflict{-1};
    int first_pending_native_weed{-1};
    int first_pending_native_realign{-1};
    int final_commit_attempts{};
    int final_commit_successes{};
    int final_commit_failures{};
    int final_commit_stale_before{};
    int final_commit_stale_after{};
    int final_commit_state_leaks{};
    int stateful_overlay_injections{};
    int stateful_overlay_recoveries{};
    std::uint64_t injected_final_action_hash{};
    std::uint64_t injected_binding_hash{};
    int injected_owned_actor{-1};
    std::uint64_t injected_owner_generation{};
    std::uint64_t injected_owner_certificate_hash{};
    std::string first_final_commit_failure;
    bool stale_authorization_seen_after_day7 = false;
    bool stale_authorization_changed_install = false;
    bool stale_authorization_cleared_by_install = false;
    receipt_adapter::Adapter physical_adapter;
    rolling_receipt::Adapter rolling_adapter;
    rolling_receipt::Adapter final_action_probe;

    while (!baseline.done() && !candidate.done()) {
      const int step = candidate.step_count();
      const auto baseline_before =
          g001::repair_fork::full_unit_phase_state_fingerprint(baseline);
      const auto candidate_before =
          g001::repair_fork::full_unit_phase_state_fingerprint(candidate);
      if (step == kOwnerStart) {
        baseline_start_fingerprint = baseline_before;
        owner_start_fingerprint = candidate_before;
      }

      std::array<PlayerAction, 2> baseline_actions;
      for (int player = 0; player < 2; ++player)
        baseline_actions[player] = executor.action_external(
            baseline, player, 0, baseline_state[player]);

      std::array<PlayerAction, 2> candidate_actions;
      candidate_actions[0] = executor.action_external(
          candidate, 0, 0, candidate_opponent);
      if (step < kOwnerStart) {
        candidate_actions[1] = executor.action_external(
            candidate, 1, 0, candidate_owner.native);
        owner::observe_final_provider_action(
            candidate, kSeat, candidate_actions[1], candidate_owner);
      } else {
        ++owner_calls;
        const bool at_day_start = candidate.hour() == 0;
        std::optional<DayInstall> install;
        issuer::IssueResult persistent_issue;
        issuer::IssueResult empty_issue;
        if (at_day_start) {
          install.emplace();
          install->day = candidate.day();
          install->step = step;
          install->actors = static_cast<int>(
              1 + candidate.farms()[kSeat].hands.size());
          install->native_weed_before = active_weeds(candidate_owner.native);
          install->native_realign_before =
              active_realignments(candidate_owner.native);
          persistent_issue = issuer::issue_day_start(
              {&candidate, &tape, kSeat, generation(candidate.day()),
               &candidate_owner.persistent_lineage, {}, {}});
          issuer::PersistentRouteIntentRegistry empty_lineage;
          empty_issue = issuer::issue_day_start(
              {&candidate, &tape, kSeat, generation(candidate.day()),
               &empty_lineage, {}, {}});
          install->persistent_unsupported =
              static_cast<int>(persistent_issue.unsupported.size());
          install->empty_lineage_unsupported =
              static_cast<int>(empty_issue.unsupported.size());
          install->persistent_fully_proven = persistent_issue.fully_proven();
          install->empty_lineage_fully_proven = empty_issue.fully_proven();
          add_reasons(persistent_issue, install->persistent_reasons);
          add_reasons(empty_issue, install->empty_reasons);

          if (candidate.day() == 7) {
            const auto deferred = std::find_if(
                empty_issue.obligations.begin(), empty_issue.obligations.end(),
                [](const auto& obligation) {
                  return obligation.source_step == 191 &&
                         obligation.goal !=
                             g001::obligation_day::GoalKind::BuildPasture;
                });
            if (deferred == empty_issue.obligations.end())
              throw std::runtime_error(
                  "day7 immutable source 191 is no longer defer-eligible");
            owner::PolicyDeferredAuthorization authorization;
            authorization.player = kSeat;
            authorization.day = candidate.day();
            authorization.issuer_generation = generation(candidate.day());
            authorization.obligation_id = deferred->id;
            authorization.obligation_content_hash =
                owner::production_obligation_identity_hash(*deferred);
            authorization.actor = deferred->actor;
            authorization.source_step = deferred->source_step;
            authorization.source_action =
                tape[static_cast<std::size_t>(deferred->source_step)]
                    .units[static_cast<std::size_t>(deferred->actor)];
            authorization.goal = deferred->goal;
            authorization.item = deferred->item;
            authorization.tile = deferred->tile;
            authorization.quantity = deferred->quantity;
            authorization.content_hash =
                owner::policy_deferred_authorization_hash(authorization);
            candidate_owner.policy_deferred_authorization = authorization;
          }
          install->authorization_present =
              candidate_owner.policy_deferred_authorization.has_value();
          install->authorization_matches_day =
              install->authorization_present &&
              candidate_owner.policy_deferred_authorization->day ==
                  candidate.day();
          install->owner_receipt_pending_before =
              candidate_owner.pending.active;
          if (candidate.day() > 7 && install->authorization_present &&
              !install->authorization_matches_day)
            stale_authorization_seen_after_day7 = true;
        }

        const bool final_commit_day7 = candidate.day() == 7 &&
            step >= kOwnerStart && step < kOwnerStart + 24;
        std::optional<final_commit::Request> final_commit_request;
        std::optional<final_commit::Proposal> final_commit_proposal;
        int final_commit_base_last_step = candidate_owner.native.last_step;
        int final_commit_base_wheat_credit =
            candidate_owner.native.wheat_credit;

        NativeAgentState native_probe = candidate_owner.native;
        fastkag::NativeRepairOptions probe_options;
        probe_options.weed_obligation_day_owner =
            (candidate_owner.active &&
             candidate_owner.active_day == candidate.day()) ||
            day_start_weed(executor, candidate, kSeat);
        if (final_commit_day7) {
          if (step > kOwnerStart &&
              candidate_owner.native.last_step != step - 1)
            ++final_commit_stale_before;
          final_commit::Request request{
              &executor, &candidate, kSeat, 0, probe_options, std::nullopt};
          if (step == 188) {
            request.injection = final_commit::StatefulOverlayInjection{
                0, Action{Op::CARE}, 777777,
                final_commit::OverlayReplayBehavior::RepeatsOnReplay};
            ++stateful_overlay_injections;
          }
          final_commit_request = request;
          final_commit_proposal =
              final_commit::propose(request, candidate_owner.native);
        }
        const auto native_final = executor.action_external(
            candidate, kSeat, 0, native_probe,
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, false,
            probe_options, nullptr);
        try {
          candidate_actions[1] = owner::action_external(
              executor, candidate, kSeat, 0, candidate_owner,
              generation(candidate.day()));
          if (final_commit_proposal && final_commit_request) {
            ++final_commit_attempts;
            const auto binding = final_commit::bind_final_action(
                *final_commit_proposal, candidate_owner.owned_actor,
                generation(candidate.day()),
                candidate_owner.plan.certificate
                    ? candidate_owner.plan.certificate->content_hash
                    : 0,
                candidate_actions[1]);
            auto committed_native = final_commit_proposal->base_state;
            const auto result = final_commit::commit_weed_owner_finalized(
                *final_commit_request, *final_commit_proposal, binding,
                candidate_actions[1], committed_native);
            if (!result.committed) {
              ++final_commit_failures;
              if (first_final_commit_failure.empty())
                first_final_commit_failure =
                    final_commit::commit_reject_name(result.reject);
              if (committed_native.last_step != final_commit_base_last_step ||
                  committed_native.wheat_credit !=
                      final_commit_base_wheat_credit)
                ++final_commit_state_leaks;
              throw std::runtime_error(
                  std::string("final-action native commit rejected: ") +
                  final_commit::commit_reject_name(result.reject));
            }
            ++final_commit_successes;
            if (committed_native.last_step != step)
              ++final_commit_stale_after;
            if (step == 188) {
              injected_final_action_hash = binding.action_fingerprint;
              injected_binding_hash = binding.content_hash;
              injected_owned_actor = binding.owned_actor;
              injected_owner_generation = binding.owner_generation;
              injected_owner_certificate_hash =
                  binding.owner_certificate_hash;
              const bool no_phantom =
                  final_commit_proposal->candidate_state.last_step == 999999 &&
                  committed_native.last_step == step &&
                  final_commit_proposal->candidate_state.wheat_credit ==
                      committed_native.wheat_credit + 777777;
              if (result.candidate_discarded &&
                  result.proposal_stateful_mutation && no_phantom)
                ++stateful_overlay_recoveries;
              else
                ++final_commit_state_leaks;
            }
            candidate_owner.native = std::move(committed_native);
          }
        } catch (const std::exception& error) {
          first_owner_exception_step = step;
          first_owner_exception = error.what();
          if (install) {
            install->fail_closed = true;
            install->native_weed_after = active_weeds(candidate_owner.native);
            install->native_realign_after =
                active_realignments(candidate_owner.native);
            install->stationary = static_cast<int>(
                candidate_owner.native.experimental_stationary_obligations
                    .size());
            install->owner_active = candidate_owner.active;
            install->owned_actor = candidate_owner.owned_actor;
            install->planned = candidate_owner.audit.planned;
            install->certificate_valid =
                candidate_owner.audit.certificate_valid;
            install->authorization_present_after =
                candidate_owner.policy_deferred_authorization.has_value();
            install->owner_receipt_pending_after =
                candidate_owner.pending.active;
            install->source_total =
                candidate_owner.audit.owned_source_coverage_total;
            install->source_moves =
                candidate_owner.audit.owned_source_move_coverage;
            install->source_obligations =
                candidate_owner.audit.owned_source_obligation_coverage;
            install->source_passes =
                candidate_owner.audit.owned_source_pass_coverage;
            install->source_missing =
                candidate_owner.audit.owned_source_missing;
            install->source_duplicates =
                candidate_owner.audit.owned_source_duplicates;
            install->source_unsupported =
                candidate_owner.audit.owned_source_unsupported;
            install->scoped_manifest_proven =
                candidate_owner.audit.scoped_manifest_proven;
            install->exclusive_owner_admitted =
                candidate_owner.audit.exclusive_owner_admitted;
            install->admission_rejected =
                candidate_owner.audit.admission_rejected;
            install->overlay_conflict_fail_stop =
                candidate_owner.audit.overlay_conflict_fail_stop;
            install->receipt_fingerprint_diagnostic_only =
                candidate_owner.audit.receipt_fingerprint_diagnostic_only;
            installs.push_back(*install);
          }
          break;
        }

        if (at_day_start && candidate.day() == 7 && candidate_owner.active &&
            candidate_owner.plan.certificate) {
          physical_adapter.open(
              candidate, kSeat, candidate_owner.owned_actor,
              generation(candidate.day()), persistent_issue,
              candidate_owner.policy_deferred_authorization,
              *candidate_owner.plan.certificate);
          rolling_adapter.open(
              kSeat, candidate_owner.owned_actor,
              generation(candidate.day()),
              *candidate_owner.plan.certificate, candidate_owner.audit);
          final_action_probe.open(
              kSeat, candidate_owner.owned_actor,
              generation(candidate.day()),
              *candidate_owner.plan.certificate, candidate_owner.audit);
        }

        if (active_weeds(candidate_owner.native) > 0 &&
            first_pending_native_weed < 0)
          first_pending_native_weed = step;
        if (active_realignments(candidate_owner.native) > 0 &&
            first_pending_native_realign < 0)
          first_pending_native_realign = step;

        const auto& raw = tape[static_cast<std::size_t>(step)].units;
        const std::size_t actors = std::max(
            {raw.size(), native_final.units.size(),
             candidate_actions[1].units.size()});
        for (std::size_t actor = 0; actor < actors; ++actor) {
          const Action raw_action = actor < raw.size() ? raw[actor] : Action{};
          const Action native_action = actor < native_final.units.size()
                                           ? native_final.units[actor]
                                           : Action{};
          const Action owner_action = actor < candidate_actions[1].units.size()
                                          ? candidate_actions[1].units[actor]
                                          : Action{};
          const bool raw_native = !same_action(raw_action, native_action);
          const bool raw_owner = !same_action(raw_action, owner_action);
          if (!raw_native && !raw_owner) continue;
          conflicts.push_back({step, static_cast<int>(actor), raw_action,
                               native_action, owner_action, raw_native,
                               raw_owner});
          if (raw_native && first_raw_native_conflict < 0)
            first_raw_native_conflict = step;
          if (raw_owner && first_raw_owner_conflict < 0)
            first_raw_owner_conflict = step;
          if (raw_native && raw_action.op != native_action.op &&
              first_raw_native_op_conflict < 0)
            first_raw_native_op_conflict = step;
          if (raw_owner && raw_action.op != owner_action.op &&
              first_raw_owner_op_conflict < 0)
            first_raw_owner_op_conflict = step;
        }

        if (install) {
          install->native_weed_after = active_weeds(candidate_owner.native);
          install->native_realign_after =
              active_realignments(candidate_owner.native);
          install->stationary = static_cast<int>(
              candidate_owner.native.experimental_stationary_obligations
                  .size());
          install->owner_active = candidate_owner.active;
          install->owned_actor = candidate_owner.owned_actor;
          install->planned = candidate_owner.audit.planned;
          install->certificate_valid =
              candidate_owner.audit.certificate_valid;
          install->authorization_present_after =
              candidate_owner.policy_deferred_authorization.has_value();
          install->owner_receipt_pending_after =
              candidate_owner.pending.active;
          if (!candidate_owner.completed_days.empty()) {
            const auto& prior = candidate_owner.completed_days.back();
            if (prior.day == candidate.day() - 1) {
              install->finalized_prior_day = prior.day;
              install->finalized_prior_receipt_checks = prior.receipt_checks;
              install->finalized_prior_receipt_failures =
                  prior.receipt_failures;
            }
          }
          install->source_total =
              candidate_owner.audit.owned_source_coverage_total;
          install->source_moves =
              candidate_owner.audit.owned_source_move_coverage;
          install->source_obligations =
              candidate_owner.audit.owned_source_obligation_coverage;
          install->source_passes =
              candidate_owner.audit.owned_source_pass_coverage;
          install->source_missing =
              candidate_owner.audit.owned_source_missing;
          install->source_duplicates =
              candidate_owner.audit.owned_source_duplicates;
          install->source_unsupported =
              candidate_owner.audit.owned_source_unsupported;
          install->scoped_manifest_proven =
              candidate_owner.audit.scoped_manifest_proven;
          install->exclusive_owner_admitted =
              candidate_owner.audit.exclusive_owner_admitted;
          install->admission_rejected =
              candidate_owner.audit.admission_rejected;
          install->overlay_conflict_fail_stop =
              candidate_owner.audit.overlay_conflict_fail_stop;
          install->receipt_fingerprint_diagnostic_only =
              candidate_owner.audit.receipt_fingerprint_diagnostic_only;
          install->fail_closed =
              candidate_owner.audit.unowned_overlay_suppressed;
          if (candidate.day() > 7 && install->authorization_present &&
              !install->authorization_matches_day && install->fail_closed)
            stale_authorization_changed_install = true;
          if (candidate.day() > 7 && install->authorization_present &&
              !install->authorization_present_after)
            stale_authorization_cleared_by_install = true;
          installs.push_back(*install);
        }
      }

      if (step >= kOwnerStart && step < kOwnerStart + 24) {
        physical_adapter.observe_unit_step(candidate, candidate_actions);
        if (!candidate_owner.remaining_certificate)
          throw std::runtime_error(
              "owner omitted fresh remaining certificate on day7");
        if (!rolling_adapter.observe_unit_step(
                candidate, candidate_actions,
                *candidate_owner.remaining_certificate))
          throw std::runtime_error(
              std::string("rolling runtime receipt rejected step ") +
              std::to_string(step) + ": " +
              rolling_receipt::failure_name(
                  rolling_adapter.audit().first_failure));
        if (step <= 175) {
          auto probe_actions = candidate_actions;
          if (step == 175)
            probe_actions[kSeat].units[static_cast<std::size_t>(
                candidate_owner.owned_actor)] = Action{Op::SOUTH};
          const bool probe_accepted = final_action_probe.observe_unit_step(
              candidate, probe_actions,
              *candidate_owner.remaining_certificate);
          if ((step < 175 && !probe_accepted) ||
              (step == 175 && probe_accepted))
            throw std::runtime_error(
                "final-action rolling receipt probe result changed");
        }
      }

      baseline.step(baseline_actions);
      candidate.step(candidate_actions);
      if (step >= kOwnerStart && step < kOwnerStart + 24) {
        physical_adapter.observe_full_step(step, candidate);
        rolling_adapter.observe_full_step(step, candidate);
        if (step < 175) final_action_probe.observe_full_step(step, candidate);
      }
      const auto baseline_after =
          g001::repair_fork::full_unit_phase_state_fingerprint(baseline);
      const auto candidate_after =
          g001::repair_fork::full_unit_phase_state_fingerprint(candidate);
      if (!first_diff && baseline_after != candidate_after) {
        FirstDiff diff;
        diff.step = step;
        diff.baseline_before = baseline_before;
        diff.owner_before = candidate_before;
        diff.baseline_after = baseline_after;
        diff.owner_after = candidate_after;
        if (!baseline_actions[1].units.empty())
          diff.baseline_action = baseline_actions[1].units[0];
        if (!candidate_actions[1].units.empty())
          diff.owner_action = candidate_actions[1].units[0];
        diff.baseline_x = baseline.farms()[kSeat].farmer.x;
        diff.baseline_y = baseline.farms()[kSeat].farmer.y;
        diff.owner_x = candidate.farms()[kSeat].farmer.x;
        diff.owner_y = candidate.farms()[kSeat].farmer.y;
        diff.baseline_money = baseline.farms()[kSeat].money;
        diff.owner_money = candidate.farms()[kSeat].money;
        first_diff = diff;
      }
    }

    owner::finish_day(candidate_owner);
    std::vector<MoveClosure> closures;
    for (const auto& day : candidate_owner.completed_days)
      closures.push_back(closure(day));
    const auto& physical_receipts = physical_adapter.audit();
    const auto& rolling_receipts = rolling_adapter.audit();
    const auto& final_action_receipt_probe = final_action_probe.audit();

    const auto day7 = std::find_if(
        installs.begin(), installs.end(),
        [](const auto& value) { return value.day == 7; });
    const auto day8 = std::find_if(
        installs.begin(), installs.end(),
        [](const auto& value) { return value.day == 8; });
    if (baseline_start_fingerprint == 0 ||
        baseline_start_fingerprint != owner_start_fingerprint)
      throw std::runtime_error("same-start full-state proof failed");
    if (day7 == installs.end() || day8 == installs.end())
      throw std::runtime_error("day7/day8 install observation missing");
    if (owner_calls == 0 || direct_default_calls_after_start != 0)
      throw std::runtime_error("continuous owner call-path instrumentation failed");
    const auto day7_closure = std::find_if(
        closures.begin(), closures.end(),
        [](const auto& value) { return value.day == 7; });
    if (!day7->owner_active || !day7->scoped_manifest_proven ||
        !day7->exclusive_owner_admitted || day7->source_total != 24 ||
        day7->source_moves + day7->source_obligations +
                day7->source_passes !=
            24 ||
        day7->source_missing != 0 || day7->source_duplicates != 0 ||
        day7->source_unsupported != 0 || day7_closure == closures.end() ||
        !day7_closure->applicable || !day7_closure->exact)
      throw std::runtime_error("day7 scoped ownership diagnosis changed");
    if (day8->stationary != 0 || day8->owner_active ||
        !day8->owner_receipt_pending_before ||
        day8->owner_receipt_pending_after || day8->finalized_prior_day != 7 ||
        !day8->authorization_present || day8->authorization_present_after ||
        day8->fail_closed || stale_authorization_changed_install ||
        !stale_authorization_cleared_by_install)
      throw std::runtime_error("day8 settle/fallback diagnosis changed");
    if (!first_diff || first_diff->step != 190 ||
        first_pending_native_weed != -1 ||
        first_pending_native_realign != -1)
      throw std::runtime_error("first state/native pending diagnosis changed");
    if (final_commit_attempts != 24 || final_commit_successes != 24 ||
        final_commit_failures != 0 || final_commit_stale_before != 0 ||
        final_commit_stale_after != 0 || final_commit_state_leaks != 0 ||
        stateful_overlay_injections != 1 ||
        stateful_overlay_recoveries != 1 ||
        !first_final_commit_failure.empty())
      throw std::runtime_error(
          "day7 final-action native commit lifecycle changed");
    if (!physical_receipts.opened ||
        physical_receipts.slots.size() != 24U ||
        physical_receipts.projector_checks != 24 ||
        physical_receipts.projector_failures != 0 ||
        physical_receipts.unit_phase_step_failures != 0 ||
        physical_receipts.full_step_advance_failures != 0)
      throw std::runtime_error("player unit-phase receipt projector invalid");
    if (physical_receipts.day_closed) {
      if (physical_receipts.accepted_slots != 24 ||
          physical_receipts.first_failure_step != -1 ||
          !physical_receipts.runtime_move_closure_accepted)
        throw std::runtime_error("closed runtime receipt proof is incomplete");
    } else if (physical_receipts.first_failure_step < kOwnerStart ||
               physical_receipts.first_failure ==
                   runtime_receipt::Failure::None ||
               physical_receipts.accepted_slots >= 24) {
      throw std::runtime_error("runtime receipt failure lacks first evidence");
    }
    if (physical_receipts.accepted_slots != 1 ||
        physical_receipts.first_failure_step != 169 ||
        physical_receipts.first_failure !=
            runtime_receipt::Failure::StateFingerprintFork ||
        !physical_receipts.actual_move_sources_exactly_once ||
        physical_receipts.runtime_move_closure_accepted ||
        physical_receipts.day_closed || physical_receipts.slots.size() < 2U ||
        !physical_receipts.slots[1].expected_before_diff.money ||
        physical_receipts.slots[1].expected_before_diff.hand_positions != 6 ||
        physical_receipts.slots[1].expected_before_diff.shed_cells != 1 ||
        physical_receipts.slots[1].expected_before_diff.seed_cells != 1 ||
        physical_receipts.slots[1].expected_before_diff.inventory_cells !=
            72 ||
        physical_receipts.slots[1].expected_before_diff.inventory_orders !=
            6)
      throw std::runtime_error(
          "day7 inter-phase certificate drift diagnosis changed");
    if (!rolling_receipts.opened ||
        rolling_receipts.accepted_slots != 24 ||
        rolling_receipts.slots.size() != 24U ||
        rolling_receipts.first_failure != rolling_receipt::Failure::None ||
        rolling_receipts.first_failure_step != -1 ||
        rolling_receipts.expected_moves != 7 ||
        rolling_receipts.accepted_moves != 7 ||
        !rolling_receipts.move_sources_exactly_once_and_ordered ||
        rolling_receipts.predecessor_checks != 23 ||
        rolling_receipts.predecessor_failures != 0 ||
        rolling_receipts.projector_checks != 24 ||
        rolling_receipts.projector_failures != 0 ||
        rolling_receipts.full_step_records != 24 ||
        rolling_receipts.full_step_advance_failures != 0 ||
        !rolling_receipts.midnight_attempted ||
        !rolling_receipts.day_closed ||
        rolling_receipts.midnight_failure != rolling_receipt::Failure::None ||
        rolling_receipts.slots[20].step != 188 ||
        rolling_receipts.slots[20].source_step != 187 ||
        rolling_receipts.slots[20].emitted.op != Op::SOUTH)
      throw std::runtime_error("rolling runtime receipt closure changed");
    if (final_action_receipt_probe.accepted_slots != 7 ||
        final_action_receipt_probe.slots.size() != 8U ||
        final_action_receipt_probe.first_failure_step != 175 ||
        final_action_receipt_probe.first_failure !=
            rolling_receipt::Failure::ActionMismatch ||
        final_action_receipt_probe.slots.back().emitted.op != Op::SOUTH ||
        rolling_receipts.slots[7].step != 175 ||
        rolling_receipts.slots[7].emitted.op != Op::CARE)
      throw std::runtime_error(
          "receipt did not consume injected final submitted action");

    std::ofstream json(output, std::ios::trunc);
    if (!json) throw std::runtime_error("cannot create audit artifact");
    json << "{\n  \"schema\":\"g001-continuous-owner-audit-v2\",\n"
         << "  \"scope\":{\"seed\":" << kSeed
         << ",\"seat\":" << kSeat
         << ",\"owner_start_step\":" << kOwnerStart
         << ",\"future_observation_used\":false,"
            "\"terminal_or_profit_selection\":false},\n"
         << "  \"same_start\":{\"baseline_fingerprint\":"
         << baseline_start_fingerprint << ",\"owner_fingerprint\":"
         << owner_start_fingerprint << ",\"equal\":true},\n"
         << "  \"call_path\":{\"owner_calls_from_day7\":" << owner_calls
         << ",\"direct_default_calls_after_day7\":"
         << direct_default_calls_after_start
         << ",\"first_owner_exception_step\":"
         << (first_owner_exception_step ? *first_owner_exception_step : -1)
         << ",\"first_owner_exception\":";
    json_string(json, first_owner_exception);
    json << "},\n  \"native_final_action_commit\":{"
         << "\"mode\":\"default_off_fixed_weed_owner\","
         << "\"attempts\":" << final_commit_attempts
         << ",\"successes\":" << final_commit_successes
         << ",\"failures\":" << final_commit_failures
         << ",\"stale_before\":" << final_commit_stale_before
         << ",\"stale_after\":" << final_commit_stale_after
         << ",\"state_leaks\":" << final_commit_state_leaks
         << ",\"stateful_overlay\":{\"step\":188,"
         << "\"proposal\":\"CARE\",\"final\":\"SOUTH\","
         << "\"injections\":" << stateful_overlay_injections
         << ",\"recoveries\":" << stateful_overlay_recoveries
         << ",\"owned_actor\":" << injected_owned_actor
         << ",\"owner_generation\":" << injected_owner_generation
         << ",\"owner_certificate_hash\":"
         << injected_owner_certificate_hash
         << ",\"final_action_hash\":" << injected_final_action_hash
         << ",\"binding_hash\":" << injected_binding_hash
         << ",\"phantom_progress\":false,\"next_hand_stale\":false},"
         << "\"unsupported_policy\":\"fail_stop_without_base_state_commit\","
         << "\"future_observation_or_terminal_oracle_used\":false},\n"
         << "  \"rolling_final_action_probe\":{"
         << "\"step\":175,\"certificate_action\":";
    write_action(json, rolling_receipts.slots[7].emitted);
    json << ",\"injected_final_action\":";
    write_action(json, final_action_receipt_probe.slots.back().emitted);
    json << ",\"accepted\":"
         << (final_action_receipt_probe.slots.back().accepted ? "true"
                                                              : "false")
         << ",\"failure\":\""
         << rolling_receipt::failure_name(
                final_action_receipt_probe.slots.back().failure)
         << "\",\"receipt_records_final_south_not_proposed_care\":true},\n"
         << "  \"rolling_runtime_receipts\":{"
         << "\"scope\":\"owned_actor_unit_phase_per_real_observation\","
         << "\"fresh_remaining_certificate_each_step\":true,"
         << "\"future_suffix_used_as_current_authority\":false,"
         << "\"cross_step_link\":\"evaluator_owned_full_step_predecessor\","
         << "\"unit_post_compared_to_next_before\":false,"
         << "\"owner_diagnostic_receipt_used_for_physical_acceptance\":false,"
         << "\"opened\":"
         << (rolling_receipts.opened ? "true" : "false")
         << ",\"player\":" << rolling_receipts.player
         << ",\"day\":" << rolling_receipts.day
         << ",\"actor\":" << rolling_receipts.actor
         << ",\"issuer_generation\":"
         << rolling_receipts.issuer_generation
         << ",\"prior_certificate_hash\":"
         << rolling_receipts.prior_certificate_hash
         << ",\"expected_slots\":" << rolling_receipts.expected_slots
         << ",\"accepted_slots\":" << rolling_receipts.accepted_slots
         << ",\"first_failure\":{\"step\":"
         << rolling_receipts.first_failure_step << ",\"reason\":\""
         << rolling_receipt::failure_name(rolling_receipts.first_failure)
         << "\"},\"move_closure\":{\"expected\":"
         << rolling_receipts.expected_moves << ",\"accepted\":"
         << rolling_receipts.accepted_moves
         << ",\"sources_exactly_once_and_ordered\":"
         << (rolling_receipts.move_sources_exactly_once_and_ordered ? "true"
                                                                    : "false")
         << "},\"phase_checks\":{\"predecessor_checks\":"
         << rolling_receipts.predecessor_checks
         << ",\"predecessor_failures\":"
         << rolling_receipts.predecessor_failures
         << ",\"projector_checks\":"
         << rolling_receipts.projector_checks
         << ",\"projector_failures\":"
         << rolling_receipts.projector_failures
         << ",\"full_step_records\":"
         << rolling_receipts.full_step_records
         << ",\"full_step_advance_failures\":"
         << rolling_receipts.full_step_advance_failures
         << "},\"midnight\":{\"attempted\":"
         << (rolling_receipts.midnight_attempted ? "true" : "false")
         << ",\"day_closed\":"
         << (rolling_receipts.day_closed ? "true" : "false")
         << ",\"failure\":\""
         << rolling_receipt::failure_name(rolling_receipts.midnight_failure)
         << "\"},\"slots\":[";
    for (std::size_t i = 0; i < rolling_receipts.slots.size(); ++i) {
      const auto& value = rolling_receipts.slots[i];
      if (i) json << ',';
      json << "{\"step\":" << value.step << ",\"accepted\":"
           << (value.accepted ? "true" : "false")
           << ",\"failure\":\""
           << rolling_receipt::failure_name(value.failure)
           << "\",\"remaining_certificate_hash\":"
           << value.remaining_certificate_hash
           << ",\"observation_fingerprint\":"
           << value.observation_fingerprint
           << ",\"actual_before_fingerprint\":"
           << value.actual_before_fingerprint
           << ",\"projected_after_fingerprint\":"
           << value.projected_after_fingerprint
           << ",\"certificate_after_fingerprint\":"
           << value.certificate_after_fingerprint
           << ",\"full_step_after_fingerprint\":"
           << value.full_step_after_fingerprint
           << ",\"full_step_predecessor_checked\":"
           << (value.full_step_predecessor_checked ? "true" : "false")
           << ",\"full_step_predecessor_matched\":"
           << (value.full_step_predecessor_matched ? "true" : "false")
           << ",\"actor_prefix_projector_matched\":"
           << (value.actor_prefix_projector_matched ? "true" : "false")
           << ",\"unit_phase_step_unchanged\":"
           << (value.unit_phase_step_unchanged ? "true" : "false")
           << ",\"full_step_recorded\":"
           << (value.full_step_recorded ? "true" : "false")
           << ",\"full_step_advanced\":"
           << (value.full_step_advanced ? "true" : "false")
           << ",\"source_step\":" << value.source_step
           << ",\"obligation_id\":" << value.obligation_id
           << ",\"remaining_slots\":" << value.remaining_slots
           << ",\"remaining_moves\":" << value.remaining_moves
           << ",\"emitted\":";
      write_action(json, value.emitted);
      json << '}';
    }
    json << "]},\n  \"physical_runtime_receipts\":{"
         << "\"scope\":\"player_unit_phase\","
         << "\"projector\":\"Simulator::preview_unit_phase with only "
            "actual final player units\","
         << "\"validated_against_actual_joint_unit_phase\":true,"
         << "\"full_step_snapshot_used_as_receipt_after\":false,"
         << "\"owner_diagnostic_receipt_used_for_physical_acceptance\":"
            "false,"
         << "\"opened\":"
         << (physical_receipts.opened ? "true" : "false")
         << ",\"opening_failure\":\""
         << runtime_receipt::failure_name(
                physical_receipts.opening_failure)
         << "\",\"certificate_hash\":"
         << physical_receipts.certificate_hash
         << ",\"issuer_generation\":"
         << physical_receipts.issuer_generation
         << ",\"expected_slots\":" << physical_receipts.expected_slots
         << ",\"accepted_slots\":" << physical_receipts.accepted_slots
         << ",\"accepted_actor_slots\":"
         << physical_receipts.accepted_actor_slots
         << ",\"first_failure\":{\"step\":"
         << physical_receipts.first_failure_step << ",\"actor\":"
         << physical_receipts.first_failure_actor << ",\"reason\":\""
         << runtime_receipt::failure_name(physical_receipts.first_failure)
         << "\",\"classification\":\"legal_inter_phase_state_change_"
            "outside_day_start_unit_only_certificate\",\"next_gate\":\""
            "per_observation_suffix_resign_and_physical_receipt_or_verified_"
            "inter_phase_transition_proof\"},\"move_closure\":{\"expected\":"
         << physical_receipts.expected_moves << ",\"emitted\":"
         << physical_receipts.emitted_moves
         << ",\"actual_sources_exactly_once\":"
         << (physical_receipts.actual_move_sources_exactly_once ? "true"
                                                                : "false")
         << ",\"runtime_closure_accepted\":"
         << (physical_receipts.runtime_move_closure_accepted ? "true"
                                                             : "false")
         << "},\"midnight\":{\"attempted\":"
         << (physical_receipts.midnight_attempted ? "true" : "false")
         << ",\"day_closed\":"
         << (physical_receipts.day_closed ? "true" : "false")
         << ",\"failure\":\""
         << runtime_receipt::failure_name(
                physical_receipts.midnight_failure)
         << "\"},\"phase_checks\":{\"projector_checks\":"
         << physical_receipts.projector_checks
         << ",\"projector_failures\":"
         << physical_receipts.projector_failures
         << ",\"unit_phase_step_failures\":"
         << physical_receipts.unit_phase_step_failures
         << ",\"full_step_advance_failures\":"
         << physical_receipts.full_step_advance_failures
         << "},\"slots\":[";
    for (std::size_t i = 0; i < physical_receipts.slots.size(); ++i) {
      const auto& value = physical_receipts.slots[i];
      if (i) json << ',';
      json << "{\"step\":" << value.step << ",\"accepted\":"
           << (value.accepted ? "true" : "false")
           << ",\"failure\":\""
           << runtime_receipt::failure_name(value.failure)
           << "\",\"player_projector_matches_joint\":"
           << (value.player_projector_matches_joint ? "true" : "false")
           << ",\"unit_phase_step_unchanged\":"
           << (value.unit_phase_step_unchanged ? "true" : "false")
           << ",\"full_step_advanced\":"
           << (value.full_step_advanced ? "true" : "false")
           << ",\"actual_before\":" << value.actual_before
           << ",\"projected_after\":" << value.projected_after
           << ",\"certificate_after\":" << value.certificate_after
           << ",\"source_step\":" << value.source_step
           << ",\"obligation_id\":" << value.obligation_id
           << ",\"money_before\":" << value.money_before
           << ",\"money_after_full_step\":"
           << value.money_after_full_step << ",\"actors_before\":"
           << value.actors_before << ",\"actors_after_full_step\":"
           << value.actors_after_full_step
           << ",\"emitted\":";
      write_action(json, value.emitted);
      json << ",\"submitted_market\":[";
      for (std::size_t market = 0; market < value.submitted_market.size();
           ++market) {
        if (market) json << ',';
        write_action(json, value.submitted_market[market]);
      }
      json << "],\"market_fills\":[";
      for (std::size_t fill = 0; fill < value.market_fills.size(); ++fill) {
        if (fill) json << ',';
        json << value.market_fills[fill];
      }
      json << ']';
      json << ",\"expected_before_diff\":";
      write_component_diff(json, value.expected_before_diff);
      json << ",\"expected_after_diff\":";
      write_component_diff(json, value.expected_after_diff);
      json << '}';
    }
    json << "]},\n  \"policy_deferred_lifetime\":{"
         << "\"stale_identity_present_at_day8_entry\":"
         << (stale_authorization_seen_after_day7 ? "true" : "false")
         << ",\"authorization_cleared_by_next_install\":"
         << (stale_authorization_cleared_by_install ? "true" : "false")
         << ",\"stale_authorization_changed_install\":"
         << (stale_authorization_changed_install ? "true" : "false")
         << "},\n  \"native_pending\":{\"first_weed_step\":"
         << first_pending_native_weed
         << ",\"first_realign_step\":" << first_pending_native_realign
         << "},\n  \"overlay_summary\":{\"first_raw_native_conflict\":"
         << first_raw_native_conflict
         << ",\"first_raw_owner_conflict\":"
         << first_raw_owner_conflict
         << ",\"first_raw_native_op_conflict\":"
         << first_raw_native_op_conflict
         << ",\"first_raw_owner_op_conflict\":"
         << first_raw_owner_op_conflict << "},\n  \"installs\":[";
    for (std::size_t i = 0; i < installs.size(); ++i) {
      const auto& value = installs[i];
      if (i) json << ',';
      json << "{\"day\":" << value.day << ",\"step\":" << value.step
           << ",\"actors\":" << value.actors
           << ",\"stationary_submissions\":" << value.stationary
           << ",\"native_pending_weed_before\":"
           << value.native_weed_before
           << ",\"native_pending_weed_after\":" << value.native_weed_after
           << ",\"native_pending_realign_before\":"
           << value.native_realign_before
           << ",\"native_pending_realign_after\":"
           << value.native_realign_after
           << ",\"owner_active\":"
           << (value.owner_active ? "true" : "false")
           << ",\"owned_actor\":" << value.owned_actor
           << ",\"planned\":" << (value.planned ? "true" : "false")
           << ",\"certificate_valid\":"
           << (value.certificate_valid ? "true" : "false")
           << ",\"owned_source_coverage\":{\"total\":"
           << value.source_total << ",\"moves\":" << value.source_moves
           << ",\"obligations\":" << value.source_obligations
           << ",\"passes\":" << value.source_passes
           << ",\"missing\":" << value.source_missing
           << ",\"duplicates\":" << value.source_duplicates
           << ",\"unsupported\":" << value.source_unsupported << '}'
           << ",\"scoped_manifest_proven\":"
           << (value.scoped_manifest_proven ? "true" : "false")
           << ",\"exclusive_owner_admitted\":"
           << (value.exclusive_owner_admitted ? "true" : "false")
           << ",\"admission_rejected\":"
           << (value.admission_rejected ? "true" : "false")
           << ",\"overlay_conflict_fail_stop\":"
           << (value.overlay_conflict_fail_stop ? "true" : "false")
           << ",\"receipt_fingerprint_diagnostic_only\":"
           << (value.receipt_fingerprint_diagnostic_only ? "true" : "false")
           << ",\"fail_closed\":"
           << (value.fail_closed ? "true" : "false")
           << ",\"authorization_present\":"
           << (value.authorization_present ? "true" : "false")
           << ",\"authorization_matches_day\":"
           << (value.authorization_matches_day ? "true" : "false")
           << ",\"authorization_present_after\":"
           << (value.authorization_present_after ? "true" : "false")
           << ",\"owner_receipt_pending_before\":"
           << (value.owner_receipt_pending_before ? "true" : "false")
           << ",\"owner_receipt_pending_after\":"
           << (value.owner_receipt_pending_after ? "true" : "false")
           << ",\"finalized_prior_day\":" << value.finalized_prior_day
           << ",\"finalized_prior_receipt_checks\":"
           << value.finalized_prior_receipt_checks
           << ",\"finalized_prior_receipt_failures\":"
           << value.finalized_prior_receipt_failures
           << ",\"lineage\":{\"persistent\":{\"unsupported\":"
           << value.persistent_unsupported << ",\"fully_proven\":"
           << (value.persistent_fully_proven ? "true" : "false")
           << ",\"reasons\":{";
      bool first = true;
      for (const auto& [reason, count] : value.persistent_reasons) {
        if (!first) json << ',';
        first = false;
        json_string(json, reason);
        json << ':' << count;
      }
      json << "}},\"owner_empty_registry\":{\"unsupported\":"
           << value.empty_lineage_unsupported << ",\"fully_proven\":"
           << (value.empty_lineage_fully_proven ? "true" : "false")
           << ",\"reasons\":{";
      first = true;
      for (const auto& [reason, count] : value.empty_reasons) {
        if (!first) json << ',';
        first = false;
        json_string(json, reason);
        json << ':' << count;
      }
      json << "}}}}";
    }
    json << "],\n  \"move_closure\":[";
    for (std::size_t i = 0; i < closures.size(); ++i) {
      const auto& value = closures[i];
      if (i) json << ',';
      json << "{\"day\":" << value.day
           << ",\"applicable\":"
           << (value.applicable ? "true" : "false")
           << ",\"raw_tokens\":" << value.raw_tokens
           << ",\"replay_tokens\":" << value.replay_tokens
           << ",\"emitted\":" << value.emitted
           << ",\"exact\":" << (value.exact ? "true" : "false")
           << ",\"first_bad_source\":" << value.first_bad_source
           << ",\"receipt_checks\":" << value.receipt_checks
           << ",\"receipt_failures\":" << value.receipt_failures
           << ",\"first_receipt_failure_step\":"
           << value.first_receipt_failure_step << '}';
    }
    json << "],\n  \"conflicts\":[";
    for (std::size_t i = 0; i < conflicts.size(); ++i) {
      const auto& value = conflicts[i];
      if (i) json << ',';
      json << "{\"step\":" << value.step << ",\"actor\":" << value.actor
           << ",\"raw_native\":"
           << (value.raw_native ? "true" : "false")
           << ",\"raw_owner\":"
           << (value.raw_owner ? "true" : "false") << ",\"raw\":";
      write_action(json, value.raw);
      json << ",\"native_final\":";
      write_action(json, value.native_final);
      json << ",\"owner_final\":";
      write_action(json, value.owner_final);
      json << '}';
    }
    json << "],\n  \"first_state_difference\":";
    if (!first_diff) {
      json << "null";
    } else {
      const auto& value = *first_diff;
      json << "{\"step\":" << value.step
           << ",\"baseline_before\":" << value.baseline_before
           << ",\"owner_before\":" << value.owner_before
           << ",\"baseline_after\":" << value.baseline_after
           << ",\"owner_after\":" << value.owner_after
           << ",\"baseline_action\":";
      write_action(json, value.baseline_action);
      json << ",\"owner_action\":";
      write_action(json, value.owner_action);
      json << ",\"baseline_farmer\":[" << value.baseline_x << ','
           << value.baseline_y << "],\"owner_farmer\":[" << value.owner_x
           << ',' << value.owner_y << "],\"baseline_money\":"
           << value.baseline_money << ",\"owner_money\":"
           << value.owner_money << '}';
    }
    json << "\n}\n";

    std::cout << "g001_continuous_owner_audit: DIAGNOSTIC_REPRODUCED"
              << " owner_calls=" << owner_calls
              << " first_diff=" << (first_diff ? first_diff->step : -1)
              << " day7_active=" << day7->owner_active
              << " day8_active=" << day8->owner_active
              << " stale_policy=" << stale_authorization_seen_after_day7
              << " stale_changed_install="
              << stale_authorization_changed_install
              << " output=" << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "g001_continuous_owner_audit: " << error.what() << '\n';
    return 1;
  }
}
