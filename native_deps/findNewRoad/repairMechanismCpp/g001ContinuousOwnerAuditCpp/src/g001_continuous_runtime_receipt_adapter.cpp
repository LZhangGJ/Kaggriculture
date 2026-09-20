#include "g001_continuous_runtime_receipt_adapter.hpp"

#include "production_suffix_scheduler.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace g001::continuous_runtime_receipt {
namespace {

bool position_equal(fastkag::Position lhs, fastkag::Position rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool tile_equal(const fastkag::Tile& lhs, const fastkag::Tile& rhs) {
  return lhs.kind == rhs.kind && lhs.crop == rhs.crop &&
         lhs.animal == rhs.animal && lhs.planted_day == rhs.planted_day &&
         lhs.placed_day == rhs.placed_day &&
         lhs.yield_units == rhs.yield_units &&
         lhs.consecutive_unwatered == rhs.consecutive_unwatered &&
         lhs.consecutive_unfed == rhs.consecutive_unfed &&
         lhs.fertilized_until_day == rhs.fertilized_until_day &&
         lhs.pending_care_bonus == rhs.pending_care_bonus &&
         lhs.max_lifespan_step == rhs.max_lifespan_step &&
         lhs.watered_today == rhs.watered_today &&
         lhs.fed_today == rhs.fed_today &&
         lhs.cared_today == rhs.cared_today &&
         lhs.fertilizer_available == rhs.fertilizer_available;
}

template <class Left, class Right, class Equal>
int vector_diff_count(const Left& lhs, const Right& rhs, Equal equal) {
  const std::size_t common = std::min(lhs.size(), rhs.size());
  int count = static_cast<int>(std::max(lhs.size(), rhs.size()) - common);
  for (std::size_t i = 0; i < common; ++i) count += !equal(lhs[i], rhs[i]);
  return count;
}

ComponentDiff component_diff(const fastkag::Simulator& expected,
                             const fastkag::Simulator& actual, int player) {
  ComponentDiff out;
  const auto& expected_farm = expected.farms()[player];
  const auto& actual_farm = actual.farms()[player];
  out.money = expected_farm.money != actual_farm.money;
  out.farmer = !position_equal(expected_farm.farmer, actual_farm.farmer);
  out.hand_positions = vector_diff_count(
      expected_farm.hands, actual_farm.hands,
      [](auto lhs, auto rhs) { return position_equal(lhs, rhs); });
  out.tiles = vector_diff_count(expected_farm.tiles, actual_farm.tiles,
                                [](const auto& lhs, const auto& rhs) {
                                  return tile_equal(lhs, rhs);
                                });
  const auto& expected_private = expected.privates()[player];
  const auto& actual_private = actual.privates()[player];
  for (std::size_t i = 0; i < expected_private.shed.size(); ++i)
    out.shed_cells += expected_private.shed[i] != actual_private.shed[i];
  for (std::size_t i = 0; i < expected_private.seeds.size(); ++i)
    out.seed_cells += expected_private.seeds[i] != actual_private.seeds[i];
  const std::size_t inventories = std::max(expected_private.inventories.size(),
                                           actual_private.inventories.size());
  for (std::size_t actor = 0; actor < inventories; ++actor) {
    if (actor >= expected_private.inventories.size() ||
        actor >= actual_private.inventories.size()) {
      out.inventory_cells += fastkag::N_ITEMS;
      continue;
    }
    for (std::size_t item = 0;
         item < static_cast<std::size_t>(fastkag::N_ITEMS); ++item) {
      out.inventory_cells += expected_private.inventories[actor][item] !=
                             actual_private.inventories[actor][item];
    }
  }
  out.inventory_orders = vector_diff_count(
      expected_private.inventory_order, actual_private.inventory_order,
      [](const auto& lhs, const auto& rhs) { return lhs == rhs; });
  return out;
}

struct Projection {
  fastkag::Simulator player_after;
  bool matches_joint{};
  bool step_unchanged{};
};

Projection project_player_unit_phase(
    const fastkag::Simulator& before,
    const std::array<fastkag::PlayerAction, 2>& actual_joint, int player,
    int logical_step) {
  std::array<fastkag::PlayerAction, 2> isolated;
  isolated[player].units = actual_joint[player].units;
  fastkag::Simulator player_after = before.preview_unit_phase(isolated);
  const auto joint_after = before.preview_unit_phase(actual_joint);
  const auto player_fingerprint =
      production_suffix::focal_unit_state_fingerprint(
          player_after, player, logical_step);
  const auto joint_fingerprint =
      production_suffix::focal_unit_state_fingerprint(
          joint_after, player, logical_step);
  const bool step_unchanged =
      player_after.step_count() == before.step_count() &&
      joint_after.step_count() == before.step_count();
  return {std::move(player_after),
          player_fingerprint == joint_fingerprint, step_unchanged};
}

obligation_day::DayPlanRequest reconstruct_request(
    const fastkag::Simulator& day_start, int player,
    std::uint64_t issuer_generation, int actor,
    const day_start_issuer::IssueResult& issued,
    const std::optional<real_weed_move_owner::DebtSelectionIdentity>&
        authorization) {
  obligation_day::DayPlanRequest request;
  request.day_start = &day_start;
  request.player = player;
  request.issuer_generation = issuer_generation;
  for (const auto& move : issued.moves)
    if (move.actor == actor) request.moves.push_back(move);
  for (const auto& obligation : issued.obligations)
    if (obligation.actor == actor) request.obligations.push_back(obligation);
  if (authorization) {
    const auto selected = std::find_if(
        request.obligations.begin(), request.obligations.end(),
        [&](const auto& obligation) {
          return obligation.id == authorization->obligation_id &&
                 obligation.actor == authorization->actor &&
                 obligation.source_step == authorization->source_step;
        });
    if (selected == request.obligations.end() ||
        authorization->content_hash !=
            real_weed_move_owner::debt_selection_identity_hash(
                *authorization) ||
        authorization->player != player ||
        authorization->issuer_generation != issuer_generation)
      throw std::runtime_error(
          "runtime receipt request cannot bind debt selection identity");
    selected->policy_deferred = true;
  }
  return request;
}

bool movement(fastkag::Op op) {
  return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
         op == fastkag::Op::EAST || op == fastkag::Op::WEST;
}

}  // namespace

bool ComponentDiff::empty() const {
  return !money && !farmer && hand_positions == 0 && tiles == 0 &&
         shed_cells == 0 && seed_cells == 0 && inventory_cells == 0 &&
         inventory_orders == 0;
}

struct Adapter::Impl {
  int player{-1};
  int owned_actor{-1};
  int day_start{-1};
  Audit audit;
  std::optional<obligation_day::DayScheduleCertificate> certificate;
  std::unique_ptr<day_runtime_receipt::RuntimeReceiptVerifier> verifier;
  std::optional<fastkag::Simulator> certificate_cursor;
  std::set<std::pair<int, int>> actual_moves;
};

Adapter::Adapter() : impl_(std::make_unique<Impl>()) {}
Adapter::~Adapter() = default;
Adapter::Adapter(Adapter&&) noexcept = default;
Adapter& Adapter::operator=(Adapter&&) noexcept = default;

void Adapter::open(
    const fastkag::Simulator& day_start, int player, int owned_actor,
    std::uint64_t issuer_generation,
    const day_start_issuer::IssueResult& issued,
    const std::optional<real_weed_move_owner::DebtSelectionIdentity>&
        authorization,
    const obligation_day::DayScheduleCertificate& certificate) {
  if (impl_->verifier) throw std::runtime_error("receipt adapter reopened");
  auto request = reconstruct_request(day_start, player, issuer_generation,
                                     owned_actor, issued, authorization);
  impl_->player = player;
  impl_->owned_actor = owned_actor;
  impl_->day_start = certificate.day * 24;
  impl_->certificate = certificate;
  impl_->verifier =
      std::make_unique<day_runtime_receipt::RuntimeReceiptVerifier>(
          request, certificate);
  impl_->certificate_cursor = day_start;
  impl_->audit.opened = impl_->verifier->ready();
  impl_->audit.opening_failure = impl_->verifier->opening_failure();
  impl_->audit.certificate_hash = certificate.content_hash;
  impl_->audit.issuer_generation = certificate.issuer_generation;
  impl_->audit.expected_slots = static_cast<int>(certificate.slots.size());
  impl_->audit.expected_moves =
      static_cast<int>(certificate.move_replays.size());
}

void Adapter::observe_unit_step(
    const fastkag::Simulator& before,
    const std::array<fastkag::PlayerAction, 2>& actual_joint) {
  if (!impl_->verifier || !impl_->certificate ||
      !impl_->certificate_cursor)
    throw std::runtime_error("receipt adapter is not open");
  const int step = before.step_count();
  const int tick = step - impl_->day_start;
  if (tick < 0 || tick >= static_cast<int>(impl_->certificate->slots.size()))
    throw std::runtime_error("runtime receipt tick escaped certificate day");
  const auto& certificate_slot =
      impl_->certificate->slots[static_cast<std::size_t>(tick)];
  const auto projection =
      project_player_unit_phase(before, actual_joint, impl_->player, step);
  SlotAudit slot;
  slot.step = step;
  slot.player_projector_matches_joint = projection.matches_joint;
  slot.unit_phase_step_unchanged = projection.step_unchanged;
  ++impl_->audit.projector_checks;
  impl_->audit.projector_failures += !projection.matches_joint;
  impl_->audit.unit_phase_step_failures += !projection.step_unchanged;
  slot.actual_before = production_suffix::focal_unit_state_fingerprint(
      before, impl_->player, step);
  slot.projected_after = production_suffix::focal_unit_state_fingerprint(
      projection.player_after, impl_->player, step);
  slot.certificate_after = certificate_slot.focal_post_fingerprint;
  slot.money_before = before.farms()[impl_->player].money;
  slot.actors_before =
      1 + static_cast<int>(before.farms()[impl_->player].hands.size());
  slot.submitted_market = actual_joint[impl_->player].market;
  slot.expected_before_diff = component_diff(
      *impl_->certificate_cursor, before, impl_->player);

  std::array<fastkag::PlayerAction, 2> certified_joint;
  certified_joint[impl_->player].units = certificate_slot.actions;
  auto certified_after =
      impl_->certificate_cursor->preview_unit_phase(certified_joint);
  slot.expected_after_diff = component_diff(
      certified_after, projection.player_after, impl_->player);

  std::vector<day_runtime_receipt::ActorExecution> actor_records;
  actor_records.reserve(certificate_slot.actions.size());
  for (std::size_t actor = 0; actor < certificate_slot.actions.size();
       ++actor) {
    const auto emitted = actor < actual_joint[impl_->player].units.size()
                             ? actual_joint[impl_->player].units[actor]
                             : fastkag::Action{};
    actor_records.push_back(
        {impl_->player,
         impl_->certificate->day,
         step,
         static_cast<int>(actor),
         certificate_slot.sources[actor].source_step,
         certificate_slot.obligation_ids[actor],
         impl_->certificate->content_hash,
         impl_->certificate->issuer_generation,
         certificate_slot.sources[actor].source_action,
         emitted});
    if (static_cast<int>(actor) == impl_->owned_actor) {
      slot.emitted = emitted;
      slot.source_step = certificate_slot.sources[actor].source_step;
      slot.obligation_id = certificate_slot.obligation_ids[actor];
    }
    if (movement(emitted.op)) {
      ++impl_->audit.emitted_moves;
      impl_->actual_moves.insert(
          {static_cast<int>(actor),
           certificate_slot.sources[actor].source_step});
    }
  }
  day_runtime_receipt::StepExecution execution{
      &before, &projection.player_after, slot.actual_before,
      slot.projected_after, actor_records};
  const auto receipt = impl_->verifier->accept(execution);
  slot.accepted = receipt.accepted;
  slot.failure = receipt.failure;
  impl_->audit.accepted_slots = receipt.checked_steps;
  impl_->audit.accepted_actor_slots = receipt.checked_actor_slots;
  if (!receipt.accepted && impl_->audit.first_failure_step < 0) {
    impl_->audit.first_failure_step = receipt.failure_step;
    impl_->audit.first_failure_actor = receipt.failure_actor;
    impl_->audit.first_failure = receipt.failure;
  }
  impl_->audit.slots.push_back(slot);
  impl_->certificate_cursor = std::move(certified_after);

  if (tick == 23) {
    impl_->audit.midnight_attempted = true;
    const auto closed = impl_->verifier->close_midnight();
    impl_->audit.day_closed = closed.accepted && closed.day_closed;
    impl_->audit.midnight_failure = closed.failure;
    std::set<std::pair<int, int>> expected_moves;
    for (const auto& move : impl_->certificate->move_replays)
      expected_moves.insert({move.actor, move.source_step});
    impl_->audit.actual_move_sources_exactly_once =
        impl_->actual_moves == expected_moves &&
        impl_->audit.emitted_moves == impl_->audit.expected_moves;
    impl_->audit.runtime_move_closure_accepted =
        impl_->audit.day_closed &&
        impl_->audit.actual_move_sources_exactly_once;
  }
}

void Adapter::observe_full_step(int submitted_step,
                                const fastkag::Simulator& full_after) {
  if (impl_->audit.slots.empty() ||
      impl_->audit.slots.back().step != submitted_step)
    throw std::runtime_error("full-step receipt does not follow unit receipt");
  auto& slot = impl_->audit.slots.back();
  slot.full_step_advanced = full_after.step_count() == submitted_step + 1;
  slot.money_after_full_step = full_after.farms()[impl_->player].money;
  slot.actors_after_full_step =
      1 + static_cast<int>(full_after.farms()[impl_->player].hands.size());
  slot.market_fills = full_after.last_market_fills()[impl_->player];
  impl_->audit.full_step_advance_failures += !slot.full_step_advanced;
}

const Audit& Adapter::audit() const noexcept { return impl_->audit; }

}  // namespace g001::continuous_runtime_receipt
