#include "native_final_action_commit.hpp"

#include "g001_real_weed_move_owner.hpp"
#include "route_loader.hpp"

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using fastkag::Action;
using fastkag::NativeAgentState;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
namespace seam = g001::native_final_commit;
namespace owner = g001::real_weed_move_owner;

namespace {

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

const char* op_name(Op op) {
  switch (op) {
    case Op::PASS: return "PASS";
    case Op::NORTH: return "NORTH";
    case Op::SOUTH: return "SOUTH";
    case Op::EAST: return "EAST";
    case Op::WEST: return "WEST";
    case Op::CARE: return "CARE";
    default: return "OTHER";
  }
}

std::uint64_t generation(const Simulator& simulator) {
  return (970017ULL << 20) | (2ULL << 16) |
         static_cast<std::uint64_t>(simulator.day() + 1);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string artifact =
        argc > 1 ? argv[1] : "native-final-action-commit.json";
    NativeTapeLibrary library;
    library.routes.push_back(g001::repair::load_route(
        NATIVE_COMMIT_TAPES, NATIVE_COMMIT_LIBRARY, "G001"));
    NativeTeammateExecutor executor(std::move(library));
    Simulator simulator({}, 970017);
    NativeAgentState player0;
    owner::State player1;

    for (int step = 0; step < 188; ++step) {
      std::array<PlayerAction, 2> actions;
      actions[0] = executor.action_external(simulator, 0, 0, player0);
      if (step < 168) {
        actions[1] = executor.action_external(
            simulator, 1, 0, player1.native);
        owner::observe_final_provider_action(
            simulator, 1, actions[1], player1);
      } else {
        actions[1] = owner::action_external(
            executor, simulator, 1, 0, player1, generation(simulator));
      }
      simulator.step(actions);
    }
    require(simulator.step_count() == 188 && player1.active &&
                player1.owned_actor == 0,
            "real owner did not reach active step188");
    const Action scheduler_action = player1.plan.manifest[0][20];
    require(scheduler_action.op == Op::SOUTH,
            "real owner step188 action is not SOUTH");

    fastkag::NativeRepairOptions repair;
    repair.weed_obligation_day_owner = true;
    seam::Request proposal_request{
        &executor, &simulator, 1, 0, repair,
        seam::StatefulOverlayInjection{
            0, Action{Op::CARE}, 777777,
            seam::OverlayReplayBehavior::ProposalOnly}};
    const int base_last_step = player1.native.last_step;
    const int base_wheat_credit = player1.native.wheat_credit;
    const auto proposal = seam::propose(proposal_request, player1.native);
    require(proposal.action.units.at(0).op == Op::CARE &&
                proposal.candidate_state.last_step == 999999,
            "stateful CARE proposal injection was not active");
    auto final_action = proposal.action;
    final_action.units[0] = scheduler_action;
    auto committed_state = player1.native;
    const auto committed = seam::commit(
        proposal_request, proposal, final_action, committed_state);
    require(committed.committed &&
                committed.reject == seam::CommitReject::None &&
                committed.candidate_discarded &&
                committed.proposal_stateful_mutation &&
                committed.replayed_action.units.at(0).op == Op::SOUTH &&
                committed_state.last_step == simulator.step_count() &&
                committed_state.last_step != base_last_step &&
                committed_state.last_step != 999999 &&
                proposal.candidate_state.wheat_credit ==
                    committed_state.wheat_credit + 777777,
            "proposal-only overlay was not replay-committed exactly");
    const int committed_last_step = committed_state.last_step;
    const int committed_wheat_credit = committed_state.wheat_credit;

    auto repeating_request = proposal_request;
    repeating_request.injection->replay_behavior =
        seam::OverlayReplayBehavior::RepeatsOnReplay;
    const auto repeating = seam::propose(repeating_request, player1.native);
    auto finalized_state = player1.native;
    require(player1.plan.certificate.has_value(),
            "real owner certificate missing at step188");
    const auto final_binding = seam::bind_final_action(
        repeating, player1.owned_actor, generation(simulator),
        player1.plan.certificate->content_hash, final_action);
    const auto finalized = seam::commit_weed_owner_finalized(
        repeating_request, repeating, final_binding, final_action,
        finalized_state);
    require(finalized.committed &&
                finalized.reject == seam::CommitReject::None &&
                finalized.candidate_discarded &&
                finalized.proposal_stateful_mutation &&
                seam::same_action(finalized.replayed_action, final_action) &&
                finalized_state.last_step == simulator.step_count() &&
                finalized_state.last_step != 999999 &&
                repeating.candidate_state.wheat_credit ==
                    finalized_state.wheat_credit + 777777,
            "fixed weed-owner final commit did not discard stateful overlay");

    require(final_action.units.size() > 1 && final_action.market.size() < 10,
            "real action cannot exercise composed repair authority");
    auto composed_action = final_action;
    composed_action.units[1] = Action{Op::PASS};
    if (seam::same_action(composed_action, final_action))
      composed_action.units[1] = Action{Op::NORTH};
    composed_action.market.push_back(
        Action{Op::BUY_SEED, fastkag::Item::WHEAT, 1});
    const auto composed_binding = seam::bind_repair_final_action(
        repeating, 3, true, generation(simulator),
        player1.plan.certificate->content_hash, composed_action);
    auto composed_state = player1.native;
    const auto composed = seam::commit_repair_owner_finalized(
        repeating_request, repeating, composed_binding, composed_action,
        composed_state);
    require(composed.committed && seam::same_action(
                composed.replayed_action, composed_action) &&
                composed_state.last_step == simulator.step_count(),
            "multi-actor plus append-only market commit failed");

    auto market_only_action = final_action;
    market_only_action.market.push_back(
        Action{Op::BUY_SEED, fastkag::Item::WHEAT, 1});
    const auto market_only_binding = seam::bind_repair_final_action(
        repeating, 0, true, generation(simulator),
        player1.plan.certificate->content_hash, market_only_action);
    auto market_only_state = player1.native;
    const auto market_only = seam::commit_repair_owner_finalized(
        repeating_request, repeating, market_only_binding,
        market_only_action, market_only_state);
    require(market_only.committed && seam::same_action(
                market_only.replayed_action, market_only_action),
            "market-only purchase recovery commit failed");

    auto tampered_binding = final_binding;
    ++tampered_binding.action_fingerprint;
    auto binding_rejected_state = player1.native;
    const int binding_last_step = binding_rejected_state.last_step;
    const int binding_wheat_credit = binding_rejected_state.wheat_credit;
    const auto binding_rejected = seam::commit_weed_owner_finalized(
        repeating_request, repeating, tampered_binding, final_action,
        binding_rejected_state);
    require(!binding_rejected.committed &&
                binding_rejected.reject ==
                    seam::CommitReject::BindingMismatch &&
                binding_rejected_state.last_step == binding_last_step &&
                binding_rejected_state.wheat_credit == binding_wheat_credit,
            "tampered final-action binding did not fail without state leak");

    auto market_final = final_action;
    require(!market_final.market.empty(), "real action has no market envelope");
    market_final.market[0] = Action{Op::BUY_SEED, fastkag::Item::WHEAT, 1};
    const auto market_binding =
        seam::bind_final_action(
            repeating, player1.owned_actor, generation(simulator),
            player1.plan.certificate->content_hash, market_final);
    auto market_rejected_state = player1.native;
    const int market_last_step = market_rejected_state.last_step;
    const int market_wheat_credit = market_rejected_state.wheat_credit;
    const auto market_rejected = seam::commit_weed_owner_finalized(
        repeating_request, repeating, market_binding, market_final,
        market_rejected_state);
    require(!market_rejected.committed &&
                market_rejected.reject ==
                    seam::CommitReject::ReplayMismatch &&
                market_rejected_state.last_step == market_last_step &&
                market_rejected_state.wheat_credit == market_wheat_credit,
            "market replacement did not fail-stop without state leak");

    auto other_actor_final = final_action;
    require(other_actor_final.units.size() > 1,
            "real action has no unowned actor for authority test");
    other_actor_final.units[1] = Action{Op::SOUTH};
    if (seam::same_action(other_actor_final, final_action))
      other_actor_final.units[1] = Action{Op::NORTH};
    const auto other_actor_binding = seam::bind_final_action(
        repeating, player1.owned_actor, generation(simulator),
        player1.plan.certificate->content_hash, other_actor_final);
    auto other_actor_rejected_state = player1.native;
    const int other_actor_last_step = other_actor_rejected_state.last_step;
    const int other_actor_wheat_credit =
        other_actor_rejected_state.wheat_credit;
    const auto other_actor_rejected = seam::commit_weed_owner_finalized(
        repeating_request, repeating, other_actor_binding,
        other_actor_final, other_actor_rejected_state);
    require(!other_actor_rejected.committed &&
                other_actor_rejected.reject ==
                    seam::CommitReject::ReplayMismatch &&
                other_actor_rejected_state.last_step ==
                    other_actor_last_step &&
                other_actor_rejected_state.wheat_credit ==
                    other_actor_wheat_credit,
            "unowned actor replacement did not fail without state leak");

    auto expanded_request = repeating_request;
    expanded_request.repair_options.weed_min_loss_realign = true;
    auto expanded_rejected_state = player1.native;
    const int expanded_last_step = expanded_rejected_state.last_step;
    const int expanded_wheat_credit = expanded_rejected_state.wheat_credit;
    const auto expanded_rejected = seam::commit_weed_owner_finalized(
        expanded_request, repeating, final_binding, final_action,
        expanded_rejected_state);
    require(!expanded_rejected.committed &&
                expanded_rejected.reject ==
                    seam::CommitReject::InvalidRequest &&
                expanded_rejected_state.last_step == expanded_last_step &&
                expanded_rejected_state.wheat_credit == expanded_wheat_credit,
            "expanded repair configuration did not fail without state leak");

    auto rejected_state = player1.native;
    const int rejected_last_step = rejected_state.last_step;
    const int rejected_wheat_credit = rejected_state.wheat_credit;
    const auto rejected = seam::commit(
        repeating_request, repeating, final_action, rejected_state);
    require(!rejected.committed &&
                rejected.reject == seam::CommitReject::ReplayMismatch &&
                rejected.candidate_discarded &&
                rejected_state.last_step == rejected_last_step &&
                rejected_state.wheat_credit == rejected_wheat_credit,
            "persistent stateful overlay did not fail closed");

    std::array<PlayerAction, 2> step188;
    step188[0] = executor.action_external(simulator, 0, 0, player0);
    step188[1] = finalized.replayed_action;
    simulator.step(step188);
    require(simulator.step_count() == 189,
            "real simulator did not advance after committed final action");
    const auto next = executor.action_external(
        simulator, 1, 0, finalized_state,
        fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, false,
        repair, nullptr);
    require(finalized_state.last_step == 189 && !next.units.empty(),
            "replayed native state was stale on the next hand");

    std::ofstream json(artifact, std::ios::trunc);
    require(static_cast<bool>(json), "cannot create commit artifact");
    json << "{\n  \"schema\":\"native-final-action-commit-v2\",\n"
            "  \"replay\":{\"seed\":970017,\"seat\":1,\"day\":7,"
            "\"step\":188},\n"
            "  \"proposal\":{\"unit0\":\""
         << op_name(proposal.action.units[0].op)
         << "\",\"candidate_last_step\":"
         << proposal.candidate_state.last_step
         << ",\"candidate_wheat_credit\":"
         << proposal.candidate_state.wheat_credit << "},\n"
            "  \"final\":{\"unit0\":\""
         << op_name(final_action.units[0].op) << "\"},\n"
            "  \"recoverable_proposal_only_overlay\":{\"committed\":true,"
            "\"candidate_discarded\":true,\"phantom_progress\":false,"
            "\"committed_last_step\":"
         << committed_last_step
         << ",\"committed_wheat_credit\":"
         << committed_wheat_credit
         << ",\"base_wheat_credit\":" << base_wheat_credit
         << ",\"next_hand_last_step\":" << finalized_state.last_step
         << ",\"next_hand_not_stale\":true},\n"
            "  \"fixed_weed_owner_final_commit\":{\"committed\":true,"
            "\"candidate_discarded\":true,\"phantom_progress\":false,"
            "\"binding\":{\"player\":1,\"route\":0,\"step\":188,"
            "\"observation_fingerprint\":"
         << final_binding.observation_fingerprint
         << ",\"owned_actor\":" << final_binding.owned_actor
         << ",\"owner_generation\":" << final_binding.owner_generation
         << ",\"owner_certificate_hash\":"
         << final_binding.owner_certificate_hash
         << ",\"action_fingerprint\":" << final_binding.action_fingerprint
         << ",\"content_hash\":" << final_binding.content_hash
         << "},\"next_hand_not_stale\":true},\n"
            "  \"unsupported_market_replacement\":{\"committed\":false,"
            "\"reject\":\""
         << seam::commit_reject_name(market_rejected.reject)
         << "\",\"state_leaked\":false,\"policy\":\"fail_stop\"},\n"
            "  \"unsupported_unowned_actor_replacement\":{"
            "\"committed\":false,\"reject\":\""
         << seam::commit_reject_name(other_actor_rejected.reject)
         << "\",\"state_leaked\":false,\"policy\":\"fail_stop\"},\n"
            "  \"tampered_binding\":{\"committed\":false,\"reject\":\""
         << seam::commit_reject_name(binding_rejected.reject)
         << "\",\"state_leaked\":false,\"policy\":\"fail_stop\"},\n"
            "  \"unsupported_repair_configuration\":{\"committed\":false,"
            "\"reject\":\""
         << seam::commit_reject_name(expanded_rejected.reject)
         << "\",\"state_leaked\":false,\"policy\":\"fail_stop\"},\n"
            "  \"persistent_overlay\":{\"committed\":false,\"reject\":\""
         << seam::commit_reject_name(rejected.reject)
         << "\",\"state_leaked\":false,\"policy\":\"fail_stop\"},\n"
            "  \"scope\":{\"native_submission_default_path_modified\":false,"
            "\"default_off_native_api_added\":true,"
            "\"requires_exact_full_action_replay\":true,"
            "\"terminal_oracle_used\":false,\"benefit_claimed\":false},\n"
            "  \"source_mutation_audit\":{"
            "\"observation_clock\":[\"last_step\",\"receipt_ledgers\"],"
            "\"proposal_controllers\":[\"weed\",\"experimental_realign\","
            "\"room_evac\",\"k320_moon_debts\",\"targets\",\"salvage\","
            "\"wheat_credit\",\"animal_retry\",\"route_cursor\","
            "\"phased_market\"],"
            "\"final_action_staging\":[\"route_cursor_pending\","
            "\"deferred_crop_pending\",\"purchase_receipts\"],"
            "\"field_whitelist_used\":false}\n}\n";
    std::cout << "native_final_action_commit_audit: PASS output="
              << artifact << " proposal=CARE final=SOUTH next_step="
              << finalized_state.last_step << '\n';
  } catch (const std::exception& error) {
    std::cerr << "native_final_action_commit_audit: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
