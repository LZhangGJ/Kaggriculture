#include "native_final_action_commit.hpp"

#include <stdexcept>

namespace g001::native_final_commit {
namespace {

void hash_add(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (8 * byte)) & 255U;
    hash *= 1099511628211ULL;
  }
}

std::uint64_t observation_hash(const fastkag::Simulator& env) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, env.seed());
  hash_add(hash, env.step_count());
  hash_add(hash, env.done());
  hash_add(hash, production_suffix::focal_unit_state_fingerprint(env, 0));
  hash_add(hash, production_suffix::focal_unit_state_fingerprint(env, 1));
  for (const auto value : env.market().inventory) hash_add(hash, value);
  for (const auto value : env.market().prices) hash_add(hash, value);
  hash_add(hash, env.shops().size());
  for (const auto value : env.shops()) hash_add(hash, value);
  return hash;
}

bool same_unit(fastkag::Action lhs, fastkag::Action rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

void apply_injection(const StatefulOverlayInjection& injection,
                     fastkag::PlayerAction& action,
                     fastkag::NativeAgentState& candidate) {
  if (injection.actor < 0 ||
      injection.actor >= static_cast<int>(action.units.size()))
    throw std::invalid_argument("overlay injection actor is unavailable");
  action.units[injection.actor] = injection.proposal_action;
  candidate.wheat_credit += injection.wheat_credit_delta;
  // Deliberately action-coupled proposal progress. A commit that copied this
  // clone would be both phantom progress and temporally corrupt.
  candidate.last_step = 999999;
}

fastkag::PlayerAction run_native(const Request& request,
                                 fastkag::NativeAgentState& state) {
  return request.executor->action_external(
      *request.observation, request.player, request.route, state,
      fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, false,
      request.repair_options, nullptr);
}

bool request_valid(const Request& request) {
  return request.executor && request.observation && request.player >= 0 &&
         request.player < 2 && request.route >= 0 &&
         request.route < request.executor->route_count();
}

bool weed_owner_only(const fastkag::NativeRepairOptions& options) {
  return options.weed_obligation_day_owner &&
         !options.weed_min_loss_realign && !options.animal_buy_retry &&
         !options.empty_stall_reuse && options.route_cursor_production == 0 &&
         !options.state_driven_local_repair && !options.day_horizon_repair &&
         !options.day_horizon_repair_v2 &&
         !options.rolling_route_skeleton_v3 &&
         options.maximum_realign_lookahead == 16 &&
         options.stationary_reuse_lookahead == 8;
}

bool same_repair_options(const fastkag::NativeRepairOptions& lhs,
                         const fastkag::NativeRepairOptions& rhs) {
  return lhs.weed_min_loss_realign == rhs.weed_min_loss_realign &&
         lhs.animal_buy_retry == rhs.animal_buy_retry &&
         lhs.empty_stall_reuse == rhs.empty_stall_reuse &&
         lhs.route_cursor_production == rhs.route_cursor_production &&
         lhs.state_driven_local_repair == rhs.state_driven_local_repair &&
         lhs.day_horizon_repair == rhs.day_horizon_repair &&
         lhs.day_horizon_repair_v2 == rhs.day_horizon_repair_v2 &&
         lhs.rolling_route_skeleton_v3 == rhs.rolling_route_skeleton_v3 &&
         lhs.weed_obligation_day_owner == rhs.weed_obligation_day_owner &&
         lhs.maximum_realign_lookahead == rhs.maximum_realign_lookahead &&
         lhs.stationary_reuse_lookahead == rhs.stationary_reuse_lookahead;
}

std::uint64_t binding_hash(const FinalActionBinding& binding) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, static_cast<std::uint32_t>(binding.player));
  hash_add(hash, static_cast<std::uint32_t>(binding.route));
  hash_add(hash, static_cast<std::uint32_t>(binding.step));
  hash_add(hash, static_cast<std::uint32_t>(binding.owned_actor));
  hash_add(hash, binding.owned_actor_mask);
  hash_add(hash, binding.owns_market_tail);
  hash_add(hash, binding.owner_generation);
  hash_add(hash, binding.owner_certificate_hash);
  hash_add(hash, binding.observation_fingerprint);
  hash_add(hash, binding.action_fingerprint);
  return hash;
}

}  // namespace

bool same_action(const fastkag::PlayerAction& lhs,
                 const fastkag::PlayerAction& rhs) noexcept {
  if (lhs.units.size() != rhs.units.size() ||
      lhs.market.size() != rhs.market.size())
    return false;
  for (std::size_t index = 0; index < lhs.units.size(); ++index)
    if (!same_unit(lhs.units[index], rhs.units[index])) return false;
  for (std::size_t index = 0; index < lhs.market.size(); ++index)
    if (!same_unit(lhs.market[index], rhs.market[index])) return false;
  return true;
}

std::uint64_t action_fingerprint(
    const fastkag::PlayerAction& action) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, action.units.size());
  for (const auto& value : action.units) {
    hash_add(hash, static_cast<std::uint8_t>(value.op));
    hash_add(hash, static_cast<std::uint8_t>(value.item));
    hash_add(hash, static_cast<std::uint32_t>(value.quantity));
  }
  hash_add(hash, action.market.size());
  for (const auto& value : action.market) {
    hash_add(hash, static_cast<std::uint8_t>(value.op));
    hash_add(hash, static_cast<std::uint8_t>(value.item));
    hash_add(hash, static_cast<std::uint32_t>(value.quantity));
  }
  return hash;
}

FinalActionBinding bind_final_action(
    const Proposal& proposal, int owned_actor,
    std::uint64_t owner_generation, std::uint64_t owner_certificate_hash,
    const fastkag::PlayerAction& final_action) {
  const std::uint64_t mask = owned_actor >= 0 && owned_actor < 64
                                 ? 1ULL << owned_actor
                                 : 0;
  auto binding = bind_repair_final_action(
      proposal, mask, false, owner_generation, owner_certificate_hash,
      final_action);
  binding.owned_actor = owned_actor;
  binding.content_hash = binding_hash(binding);
  return binding;
}

FinalActionBinding bind_repair_final_action(
    const Proposal& proposal, std::uint64_t owned_actor_mask,
    bool owns_market_tail, std::uint64_t owner_generation,
    std::uint64_t owner_certificate_hash,
    const fastkag::PlayerAction& final_action) {
  FinalActionBinding binding;
  binding.player = proposal.player;
  binding.route = proposal.route;
  binding.step = proposal.step;
  binding.owned_actor_mask = owned_actor_mask;
  binding.owns_market_tail = owns_market_tail;
  binding.owner_generation = owner_generation;
  binding.owner_certificate_hash = owner_certificate_hash;
  binding.observation_fingerprint = proposal.observation_fingerprint;
  binding.action_fingerprint = action_fingerprint(final_action);
  binding.content_hash = binding_hash(binding);
  return binding;
}

Proposal propose(const Request& request,
                 const fastkag::NativeAgentState& state) {
  if (!request_valid(request))
    throw std::invalid_argument("invalid final-action proposal request");
  Proposal proposal;
  proposal.player = request.player;
  proposal.route = request.route;
  proposal.step = request.observation->step_count();
  proposal.observation_fingerprint =
      observation_hash(*request.observation);
  proposal.repair_options = request.repair_options;
  proposal.base_state = state;
  proposal.candidate_state = state;
  proposal.action = run_native(request, proposal.candidate_state);
  proposal.injection = request.injection;
  if (proposal.injection &&
      proposal.injection->replay_behavior != OverlayReplayBehavior::None)
    apply_injection(*proposal.injection, proposal.action,
                    proposal.candidate_state);
  return proposal;
}

CommitResult commit(const Request& request, const Proposal& proposal,
                    const fastkag::PlayerAction& final_action,
                    fastkag::NativeAgentState& state) {
  CommitResult result;
  if (!request_valid(request) || proposal.player != request.player ||
      proposal.route != request.route ||
      proposal.step != request.observation->step_count()) {
    result.reject = CommitReject::InvalidRequest;
    return result;
  }
  if (proposal.observation_fingerprint !=
      observation_hash(*request.observation)) {
    result.reject = CommitReject::ObservationChanged;
    return result;
  }
  if (final_action.units.size() != proposal.action.units.size() ||
      final_action.market.size() != proposal.action.market.size()) {
    result.reject = CommitReject::FinalShape;
    return result;
  }

  auto replay_state = proposal.base_state;
  auto replay_request = request;
  replay_request.repair_options = proposal.repair_options;
  replay_request.injection = proposal.injection;
  result.replayed_action = run_native(replay_request, replay_state);
  if (proposal.injection &&
      proposal.injection->replay_behavior ==
          OverlayReplayBehavior::RepeatsOnReplay)
    apply_injection(*proposal.injection, result.replayed_action, replay_state);
  result.proposal_stateful_mutation =
      proposal.candidate_state.last_step != replay_state.last_step ||
      proposal.candidate_state.wheat_credit != replay_state.wheat_credit;
  result.candidate_discarded = true;
  if (!same_action(result.replayed_action, final_action)) {
    result.reject = CommitReject::ReplayMismatch;
    return result;
  }
  state = std::move(replay_state);
  result.committed = true;
  return result;
}

CommitResult commit_weed_owner_finalized(
    const Request& request, const Proposal& proposal,
    const FinalActionBinding& binding,
    const fastkag::PlayerAction& final_action,
    fastkag::NativeAgentState& state) {
  if (binding.owned_actor < 0 || binding.owned_actor >= 64 ||
      binding.owned_actor_mask != (1ULL << binding.owned_actor) ||
      binding.owns_market_tail) {
    CommitResult result;
    result.reject = CommitReject::BindingMismatch;
    return result;
  }
  if (!weed_owner_only(request.repair_options) ||
      !weed_owner_only(proposal.repair_options)) {
    CommitResult result;
    result.reject = CommitReject::InvalidRequest;
    return result;
  }
  return commit_repair_owner_finalized(request, proposal, binding,
                                       final_action, state);
}

CommitResult commit_repair_owner_finalized(
    const Request& request, const Proposal& proposal,
    const FinalActionBinding& binding,
    const fastkag::PlayerAction& final_action,
    fastkag::NativeAgentState& state) {
  CommitResult result;
  if (!request_valid(request) || proposal.player != request.player ||
      proposal.route != request.route ||
      proposal.step != request.observation->step_count() ||
      !same_repair_options(request.repair_options,
                           proposal.repair_options)) {
    result.reject = CommitReject::InvalidRequest;
    return result;
  }
  if (proposal.observation_fingerprint !=
      observation_hash(*request.observation)) {
    result.reject = CommitReject::ObservationChanged;
    return result;
  }
  if (final_action.units.size() != proposal.action.units.size() ||
      final_action.market.size() < proposal.action.market.size() ||
      (!binding.owns_market_tail &&
       final_action.market.size() != proposal.action.market.size())) {
    result.reject = CommitReject::FinalShape;
    return result;
  }
  if (binding.player != proposal.player ||
      binding.route != proposal.route || binding.step != proposal.step ||
      (binding.owned_actor_mask == 0 && !binding.owns_market_tail) ||
      final_action.units.size() > 64 ||
      (final_action.units.size() < 64 &&
       (binding.owned_actor_mask >> final_action.units.size()) != 0) ||
      binding.owner_generation == 0 || binding.owner_certificate_hash == 0 ||
      binding.observation_fingerprint != proposal.observation_fingerprint ||
      binding.action_fingerprint != action_fingerprint(final_action) ||
      binding.content_hash != binding_hash(binding)) {
    result.reject = CommitReject::BindingMismatch;
    return result;
  }
  auto replay_state = proposal.base_state;
  try {
    result.replayed_action =
        request.executor->action_external_repair_owner_finalized(
            *request.observation, request.player, request.route,
            replay_state, proposal.repair_options,
            binding.owned_actor_mask,
            binding.owns_market_tail, binding.content_hash, final_action);
  } catch (const std::exception&) {
    result.reject = CommitReject::ReplayMismatch;
    result.candidate_discarded = true;
    return result;
  }
  result.proposal_stateful_mutation =
      proposal.candidate_state.last_step != replay_state.last_step ||
      proposal.candidate_state.wheat_credit != replay_state.wheat_credit;
  result.candidate_discarded = true;
  if (!same_action(result.replayed_action, final_action)) {
    result.reject = CommitReject::ReplayMismatch;
    return result;
  }
  state = std::move(replay_state);
  result.committed = true;
  return result;
}

const char* commit_reject_name(CommitReject reject) noexcept {
  switch (reject) {
    case CommitReject::None: return "none";
    case CommitReject::InvalidRequest: return "invalid_request";
    case CommitReject::ObservationChanged: return "observation_changed";
    case CommitReject::FinalShape: return "final_shape";
    case CommitReject::BindingMismatch: return "binding_mismatch";
    case CommitReject::ReplayMismatch: return "replay_mismatch";
  }
  return "unknown";
}

}  // namespace g001::native_final_commit
