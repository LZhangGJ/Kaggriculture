#include "../include/production_suffix_scheduler.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <deque>
#include <functional>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

namespace g001::production_suffix {
namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using online_elastic::FrozenDaySuffixCertificate;
using online_elastic::FrozenSlot;
using online_elastic::FrozenSlotKind;

constexpr int kCertificateTurns = 24;

bool action_equal(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

void hash_add(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffULL;
    hash *= 1099511628211ULL;
  }
}

void hash_action(std::uint64_t& hash, const Action& action) {
  hash_add(hash, static_cast<std::uint8_t>(action.op));
  hash_add(hash, static_cast<std::uint8_t>(action.item));
  hash_add(hash, static_cast<std::uint32_t>(action.quantity));
}

void hash_position(std::uint64_t& hash, Position position) {
  hash_add(hash, static_cast<std::uint16_t>(position.x));
  hash_add(hash, static_cast<std::uint16_t>(position.y));
}

std::uint64_t raw_route_hash(
    const std::vector<std::vector<Action>>& route) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, route.size());
  for (const auto& tick : route) {
    hash_add(hash, tick.size());
    for (const auto& action : tick) hash_action(hash, action);
  }
  return hash;
}

bool position_equal(Position left, Position right) {
  return left.x == right.x && left.y == right.y;
}

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
      op == Op::WEST;
}

bool is_unit_action(Op op) {
  return static_cast<int>(op) >= static_cast<int>(Op::PASS) &&
      static_cast<int>(op) <= static_cast<int>(Op::CARE);
}

bool is_production(Op op) {
  return op == Op::DROP || op == Op::PICKUP || op == Op::PLACE ||
      op == Op::PLANT || op == Op::WATER || op == Op::HARVEST ||
      op == Op::FERTILIZE || op == Op::DIG || op == Op::BUILD_COOP ||
      op == Op::BUILD_PASTURE || op == Op::FEED ||
      op == Op::COLLECT_FERTILIZER || op == Op::CARE;
}

std::uint64_t expected_actor_generation(int day, int actor) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(day + 1))
          << 32U) |
      static_cast<std::uint32_t>(actor + 1);
}

Position actor_position(const Simulator& state, int player, int actor) {
  const auto& farm = state.farms()[static_cast<std::size_t>(player)];
  if (actor == 0) return farm.farmer;
  return farm.hands[static_cast<std::size_t>(actor - 1)];
}

int action_cost(const WorkBudget& budget, const Action& action) {
  if (action.op == Op::PASS) return 0;
  if (is_move(action.op)) return budget.move_cost;
  if (is_production(action.op)) return budget.production_cost;
  return budget.other_cost;
}

Simulator apply_tick(const Simulator& before, const IssueRequest& request,
                     int tick, const Action& target) {
  std::array<PlayerAction, 2> joint;
  joint[static_cast<std::size_t>(request.player)].units =
      request.raw_units_by_tick[static_cast<std::size_t>(tick)];
  joint[static_cast<std::size_t>(request.player)]
      .units[static_cast<std::size_t>(request.actor)] = target;
  return before.preview_unit_phase(joint);
}

bool has_exact_effect(const Simulator& before, const IssueRequest& request,
                      int tick, const Action& action, Simulator& after) {
  after = apply_tick(before, request, tick, action);
  const auto pass = apply_tick(before, request, tick, Action{});
  const int logical_step = request.phase_start->day() *
      request.phase_start->config().turns_per_day + tick;
  return focal_unit_state_fingerprint(after, request.player, logical_step) !=
      focal_unit_state_fingerprint(pass, request.player, logical_step);
}

struct MoveToken {
  int source_step{-1};
  Action action{};
};

struct SearchState {
  Simulator phase;
  int tick{};
  std::size_t insertion_cursor{};
  std::deque<MoveToken> delayed;
  int stamina_used{};
  std::vector<WitnessSlot> witness;
  std::vector<ProofSlot> proof;
};

struct SearchAudit {
  bool saw_unsafe_sink{};
  bool saw_failed_precondition{};
  bool saw_stamina_failure{};
  bool saw_effectful_insertion{};
};

struct MemoKey {
  int tick{};
  std::size_t insertion_cursor{};
  int stamina_used{};
  std::uint64_t phase_hash{};
  std::vector<int> delayed_sources;

  bool operator<(const MemoKey& other) const {
    return std::tie(tick, insertion_cursor, stamina_used, phase_hash,
                    delayed_sources) <
        std::tie(other.tick, other.insertion_cursor, other.stamina_used,
                 other.phase_hash, other.delayed_sources);
  }
};

bool valid_position(const Simulator& state, Position position) {
  const int size = state.config().board_size;
  return position.x >= 0 && position.y >= 0 && position.x < size &&
      position.y < size;
}

}  // namespace

std::uint64_t focal_unit_state_fingerprint(const Simulator& state,
                                           int player) {
  return focal_unit_state_fingerprint(state, player, state.step_count());
}

std::uint64_t focal_unit_state_fingerprint(const Simulator& state, int player,
                                           int logical_step) {
  if (player < 0 || player >= 2)
    return 0;
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, static_cast<std::uint32_t>(player));
  hash_add(hash, static_cast<std::uint32_t>(logical_step));
  hash_add(hash, state.done());
  const auto& config = state.config();
  hash_add(hash, static_cast<std::uint32_t>(config.episode_steps));
  hash_add(hash, static_cast<std::uint32_t>(config.board_size));
  hash_add(hash, static_cast<std::uint32_t>(config.starting_money));
  hash_add(hash, static_cast<std::uint32_t>(config.max_market_orders));
  hash_add(hash, static_cast<std::uint32_t>(config.turns_per_day));
  hash_add(hash, static_cast<std::uint32_t>(config.shed_capacity));
  hash_add(hash, std::bit_cast<std::uint64_t>(config.weed_spawn_chance));
  hash_add(hash,
           static_cast<std::uint32_t>(config.town_shop_unlock_interval));
  hash_add(hash,
           static_cast<std::uint32_t>(config.town_shop_sell_interval));
  hash_add(hash,
           static_cast<std::uint32_t>(config.town_center_sell_interval));
  hash_add(hash, static_cast<std::uint32_t>(config.farm_hand_cost_mult));

  const auto& farm = state.farms()[static_cast<std::size_t>(player)];
  hash_add(hash, std::bit_cast<std::uint64_t>(farm.money));
  hash_position(hash, farm.farmer);
  hash_add(hash, farm.hands.size());
  for (const auto position : farm.hands) hash_position(hash, position);
  hash_add(hash, farm.unlocked_mask);
  hash_add(hash, static_cast<std::uint16_t>(farm.hires_today));
  hash_add(hash, farm.tiles.size());
  for (const auto& tile : farm.tiles) {
    hash_add(hash, static_cast<std::uint8_t>(tile.kind));
    hash_add(hash, static_cast<std::uint8_t>(tile.crop));
    hash_add(hash, static_cast<std::uint8_t>(tile.animal));
    hash_add(hash, static_cast<std::uint16_t>(tile.planted_day));
    hash_add(hash, static_cast<std::uint16_t>(tile.placed_day));
    hash_add(hash, static_cast<std::uint16_t>(tile.yield_units));
    hash_add(hash, static_cast<std::uint16_t>(tile.consecutive_unwatered));
    hash_add(hash, static_cast<std::uint16_t>(tile.consecutive_unfed));
    hash_add(hash, static_cast<std::uint16_t>(tile.fertilized_until_day));
    hash_add(hash, static_cast<std::uint16_t>(tile.pending_care_bonus));
    hash_add(hash, static_cast<std::uint32_t>(tile.max_lifespan_step));
    hash_add(hash, tile.watered_today);
    hash_add(hash, tile.fed_today);
    hash_add(hash, tile.cared_today);
    hash_add(hash, tile.fertilizer_available);
  }

  const auto& private_state =
      state.privates()[static_cast<std::size_t>(player)];
  for (const auto count : private_state.shed)
    hash_add(hash, static_cast<std::uint32_t>(count));
  for (const auto count : private_state.seeds)
    hash_add(hash, static_cast<std::uint32_t>(count));
  hash_add(hash, private_state.inventories.size());
  for (const auto& inventory : private_state.inventories)
    for (const auto count : inventory)
      hash_add(hash, static_cast<std::uint32_t>(count));
  hash_add(hash, private_state.inventory_order.size());
  for (const auto& order : private_state.inventory_order) {
    hash_add(hash, order.size());
    for (const auto item : order)
      hash_add(hash, static_cast<std::uint8_t>(item));
  }
  return hash;
}

IssueResult ProductionSuffixScheduler::issue(const IssueRequest& request) {
  IssueResult result;
  if (request.phase_start == nullptr) {
    result.reject = RejectReason::NullState;
    return result;
  }
  const auto& initial = *request.phase_start;
  const int day = initial.day();
  const int day_start = day * initial.config().turns_per_day;
  result.focal_phase_start_fingerprint =
      focal_unit_state_fingerprint(initial, request.player);
  if (initial.hour() != 0 || initial.step_count() != day_start) {
    result.reject = RejectReason::NotAtDayStart;
    return result;
  }
  if (initial.config().turns_per_day != kCertificateTurns) {
    result.reject = RejectReason::UnsupportedTurnsPerDay;
    return result;
  }
  if (initial.done() || day_start + kCertificateTurns - 1 >
      initial.config().episode_steps - 2) {
    result.reject = RejectReason::TerminalDay;
    return result;
  }
  const auto& farm = initial.farms()[static_cast<std::size_t>(
      std::clamp(request.player, 0, 1))];
  const int actors = static_cast<int>(farm.hands.size()) + 1;
  if (request.player < 0 || request.player >= 2 || request.actor < 0 ||
      request.actor >= actors || request.actor_generation !=
          expected_actor_generation(day, request.actor) ||
      request.issuer_generation == 0) {
    result.reject = RejectReason::InvalidIdentity;
    return result;
  }
  if (request.raw_units_by_tick.size() != kCertificateTurns ||
      std::any_of(request.raw_units_by_tick.begin(),
                  request.raw_units_by_tick.end(), [&](const auto& tick) {
                    return static_cast<int>(tick.size()) != actors;
                  })) {
    result.reject = RejectReason::InvalidRouteShape;
    return result;
  }
  for (const auto& tick : request.raw_units_by_tick)
    for (const auto& action : tick)
      if (!is_unit_action(action.op)) {
        result.reject = RejectReason::InvalidSourceAction;
        return result;
      }
  for (const int step : request.discardable_no_effect_steps)
    if (step < day_start || step >= day_start + kCertificateTurns) {
      result.reject = RejectReason::CrossDay;
      return result;
    }
  if (request.budget.available_stamina < 0 ||
      request.budget.move_cost < 0 ||
      request.budget.production_cost < 0 ||
      request.budget.other_cost < 0) {
    result.reject = RejectReason::InvalidInsertion;
    return result;
  }
  std::set<std::uint64_t> semantic_ids;
  for (const auto& insertion : request.insertions) {
    if (insertion.semantic_id == 0 ||
        !semantic_ids.insert(insertion.semantic_id).second ||
        !valid_position(initial, insertion.position) ||
        !is_unit_action(insertion.action.op) ||
        insertion.action.op == Op::PASS || is_move(insertion.action.op)) {
      result.reject = RejectReason::InvalidInsertion;
      return result;
    }
  }
  const auto key = std::tuple{request.player, day, request.actor};
  if (issued_hashes_.contains(key)) {
    result.reject = RejectReason::AlreadyIssued;
    return result;
  }

  std::vector<FrozenSlotKind> kinds;
  kinds.reserve(kCertificateTurns);
  for (int tick = 0; tick < kCertificateTurns; ++tick) {
    const int step = day_start + tick;
    const auto& action = request.raw_units_by_tick[static_cast<std::size_t>(tick)]
                                         [static_cast<std::size_t>(request.actor)];
    if (is_move(action.op))
      kinds.push_back(FrozenSlotKind::MoveToken);
    else if (action.op == Op::PASS ||
             request.discardable_no_effect_steps.contains(step))
      kinds.push_back(FrozenSlotKind::CertifiedSink);
    else {
      kinds.push_back(FrozenSlotKind::HardSemanticObligation);
      ++result.hard_obligations;
    }
    result.certified_sinks +=
        kinds.back() == FrozenSlotKind::CertifiedSink;
  }

  SearchAudit search_audit;
  std::set<MemoKey> visited;
  std::optional<SearchState> solution;
  std::function<bool(SearchState)> search = [&](SearchState state) -> bool {
    if (state.tick == kCertificateTurns) {
      if (state.insertion_cursor == request.insertions.size() &&
          state.delayed.empty()) {
        solution = std::move(state);
        return true;
      }
      return false;
    }
    MemoKey memo{state.tick, state.insertion_cursor, state.stamina_used,
                 focal_unit_state_fingerprint(
                     state.phase, request.player, day_start + state.tick),
                 {}};
    for (const auto& token : state.delayed)
      memo.delayed_sources.push_back(token.source_step);
    if (!visited.insert(std::move(memo)).second) return false;

    const int tick = state.tick;
    const int step = day_start + tick;
    const auto& raw = request.raw_units_by_tick[static_cast<std::size_t>(tick)]
                                         [static_cast<std::size_t>(request.actor)];
    const auto kind = kinds[static_cast<std::size_t>(tick)];

    auto emit = [&](const Action& action, int source_step,
                    std::uint64_t semantic_id, bool require_effect,
                    SearchState next) -> bool {
      const int cost = action_cost(request.budget, action);
      if (next.stamina_used > request.budget.available_stamina - cost) {
        search_audit.saw_stamina_failure = true;
        return false;
      }
      Simulator after = next.phase;
      if (require_effect &&
          !has_exact_effect(next.phase, request, tick, action, after)) {
        search_audit.saw_failed_precondition = true;
        return false;
      }
      if (source_step == -1) search_audit.saw_effectful_insertion = true;
      if (!require_effect) after = apply_tick(next.phase, request, tick, action);
      const auto pre_hash = focal_unit_state_fingerprint(
          next.phase, request.player, step);
      const auto position = actor_position(next.phase, request.player,
                                           request.actor);
      const int stamina_before = next.stamina_used;
      next.phase = std::move(after);
      next.stamina_used += cost;
      next.witness.push_back({step, action, source_step, semantic_id});
      next.proof.push_back(
          {step, raw, kind, position, action, source_step, semantic_id,
           stamina_before, next.stamina_used, pre_hash,
           focal_unit_state_fingerprint(next.phase, request.player, step)});
      ++next.tick;
      return search(std::move(next));
    };

    auto emit_insertion = [&](SearchState next) -> bool {
      if (next.insertion_cursor >= request.insertions.size()) return false;
      const auto& insertion = request.insertions[next.insertion_cursor];
      if (!position_equal(actor_position(next.phase, request.player,
                                         request.actor),
                          insertion.position)) {
        search_audit.saw_failed_precondition = true;
        return false;
      }
      ++next.insertion_cursor;
      return emit(insertion.action, -1, insertion.semantic_id, true,
                  std::move(next));
    };

    if (kind == FrozenSlotKind::HardSemanticObligation) {
      if (!state.delayed.empty()) return false;
      return emit(raw, step, 0, true, std::move(state));
    }
    if (kind == FrozenSlotKind::MoveToken) {
      // Prefer useful work at the current plot; the alternate branch keeps
      // the original MOVE in its slot when later prerequisites need it.
      if (state.insertion_cursor < request.insertions.size()) {
        auto delayed = state;
        delayed.delayed.push_back({step, raw});
        if (emit_insertion(std::move(delayed))) return true;
      }
      if (!state.delayed.empty()) {
        auto shifted = state;
        const auto token = shifted.delayed.front();
        shifted.delayed.pop_front();
        shifted.delayed.push_back({step, raw});
        return emit(token.action, token.source_step, 0, true,
                    std::move(shifted));
      }
      return emit(raw, step, 0, true, std::move(state));
    }

    // A non-PASS source may be consumed only after an exact, current-state
    // no-effect proof under the same lower/higher actor prefix used by the
    // witness.  This is never inferred from a future receipt.
    if (raw.op != Op::PASS) {
      Simulator raw_after = state.phase;
      if (has_exact_effect(state.phase, request, tick, raw, raw_after)) {
        search_audit.saw_unsafe_sink = true;
        return false;
      }
    }
    if (state.insertion_cursor < request.insertions.size() &&
        emit_insertion(state))
      return true;
    if (!state.delayed.empty()) {
      auto replay = state;
      const auto token = replay.delayed.front();
      replay.delayed.pop_front();
      if (emit(token.action, token.source_step, 0, true,
               std::move(replay)))
        return true;
    }
    return emit(Action{}, raw.op == Op::PASS ? step : -2, 0, false,
                std::move(state));
  };

  SearchState initial_search{initial, 0, 0, {}, 0, {}, {}};
  if (!search(std::move(initial_search))) {
    if (search_audit.saw_unsafe_sink)
      result.reject = RejectReason::UnsafeSink;
    else if (search_audit.saw_stamina_failure)
      result.reject = RejectReason::StaminaExceeded;
    else if (search_audit.saw_effectful_insertion)
      result.reject = RejectReason::SlotCapacity;
    else if (search_audit.saw_failed_precondition)
      result.reject = RejectReason::PreconditionsUnsatisfied;
    else
      result.reject = RejectReason::SlotCapacity;
    return result;
  }
  result.witness = solution->witness;
  result.stamina_used = solution->stamina_used;

  std::vector<std::pair<int, Action>> expected_moves;
  std::vector<std::pair<int, Action>> emitted_moves;
  std::set<int> emitted_hard;
  for (int tick = 0; tick < kCertificateTurns; ++tick) {
    const int step = day_start + tick;
    const auto& raw = request.raw_units_by_tick[static_cast<std::size_t>(tick)]
                                         [static_cast<std::size_t>(request.actor)];
    if (is_move(raw.op)) expected_moves.emplace_back(step, raw);
  }
  for (const auto& slot : result.witness) {
    if (slot.source_step >= 0 && is_move(slot.emitted.op))
      emitted_moves.emplace_back(slot.source_step, slot.emitted);
    if (slot.source_step >= 0) {
      const int source_tick = slot.source_step - day_start;
      if (source_tick >= 0 && source_tick < kCertificateTurns &&
          kinds[static_cast<std::size_t>(source_tick)] ==
              FrozenSlotKind::HardSemanticObligation)
        emitted_hard.insert(slot.source_step);
    }
  }
  bool proof_ok = emitted_moves.size() == expected_moves.size();
  for (std::size_t index = 0;
       proof_ok && index < expected_moves.size(); ++index) {
    proof_ok = emitted_moves[index].first == expected_moves[index].first &&
        action_equal(emitted_moves[index].second,
                     expected_moves[index].second);
    if (proof_ok) {
      const auto found = std::find_if(
          result.witness.begin(), result.witness.end(), [&](const auto& slot) {
            return slot.source_step == emitted_moves[index].first;
          });
      const int delay = found->emitted_step - found->source_step;
      if (delay > 0) {
        ++result.delayed_moves;
        result.maximum_move_delay = std::max(result.maximum_move_delay, delay);
      }
    }
  }
  for (int tick = 0; proof_ok && tick < kCertificateTurns; ++tick)
    if (kinds[static_cast<std::size_t>(tick)] ==
        FrozenSlotKind::HardSemanticObligation)
      proof_ok = emitted_hard.contains(day_start + tick) &&
          action_equal(result.witness[static_cast<std::size_t>(tick)].emitted,
                       request.raw_units_by_tick[static_cast<std::size_t>(tick)]
                                                [static_cast<std::size_t>(request.actor)]) &&
          result.witness[static_cast<std::size_t>(tick)].source_step ==
              day_start + tick;
  proof_ok = proof_ok && result.witness.size() == kCertificateTurns &&
      solution->insertion_cursor == request.insertions.size() &&
      solution->delayed.empty();
  if (!proof_ok) {
    result.witness.clear();
    result.reject = RejectReason::InternalProofFailure;
    return result;
  }

  FrozenDaySuffixCertificate projection;
  projection.player = request.player;
  projection.day = day;
  projection.actor = request.actor;
  projection.actor_generation = request.actor_generation;
  projection.suffix_start_step = day_start;
  projection.issuer_generation = request.issuer_generation;
  projection.sinks_irrevocable = true;
  for (int tick = 0; tick < kCertificateTurns; ++tick) {
    projection.slots.push_back(
        {day_start + tick,
         request.raw_units_by_tick[static_cast<std::size_t>(tick)]
                                  [static_cast<std::size_t>(request.actor)],
         kinds[static_cast<std::size_t>(tick)]});
  }
  projection.content_hash = online_elastic::frozen_suffix_hash(projection);

  ProductionFrozenDaySuffixCertificate certificate;
  certificate.player = request.player;
  certificate.day = day;
  certificate.actor = request.actor;
  certificate.actor_generation = request.actor_generation;
  certificate.issuer_generation = request.issuer_generation;
  certificate.focal_phase_start_fingerprint =
      result.focal_phase_start_fingerprint;
  certificate.raw_day_route_hash = raw_route_hash(request.raw_units_by_tick);
  certificate.budget = request.budget;
  certificate.insertions = request.insertions;
  certificate.proof_slots = solution->proof;
  certificate.owner_projection = projection;
  for (int tick = 0; tick < kCertificateTurns; ++tick) {
    if (kinds[static_cast<std::size_t>(tick)] !=
        FrozenSlotKind::HardSemanticObligation)
      continue;
    const auto& proof = certificate.proof_slots[static_cast<std::size_t>(tick)];
    std::uint64_t obligation_id = request.issuer_generation;
    hash_add(obligation_id, static_cast<std::uint32_t>(day_start + tick));
    if (obligation_id == 0) obligation_id = 1;
    certificate.hard_obligations.push_back(
        {obligation_id, day_start + tick, proof.raw_source,
         proof.actor_position_before, proof.pre_state_fingerprint,
         proof.post_state_fingerprint});
  }
  for (const auto& proof : certificate.proof_slots)
    if (proof.emitted_source_step >= 0 && is_move(proof.emitted.op))
      certificate.authorized_move_replays.push_back(
          {proof.emitted_source_step, proof.step, proof.emitted});
  certificate.content_hash = production_certificate_hash(certificate);
  issued_hashes_.emplace(key, certificate.content_hash);
  result.owner_projection = projection;
  result.certificate = std::move(certificate);
  result.reject = RejectReason::None;
  return result;
}

std::uint64_t production_certificate_hash(
    const ProductionFrozenDaySuffixCertificate& certificate) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, static_cast<std::uint32_t>(certificate.player));
  hash_add(hash, static_cast<std::uint32_t>(certificate.day));
  hash_add(hash, static_cast<std::uint32_t>(certificate.actor));
  hash_add(hash, certificate.actor_generation);
  hash_add(hash, certificate.issuer_generation);
  hash_add(hash, certificate.focal_phase_start_fingerprint);
  hash_add(hash, certificate.raw_day_route_hash);
  hash_add(hash, static_cast<std::uint32_t>(certificate.budget.available_stamina));
  hash_add(hash, static_cast<std::uint32_t>(certificate.budget.move_cost));
  hash_add(hash,
           static_cast<std::uint32_t>(certificate.budget.production_cost));
  hash_add(hash, static_cast<std::uint32_t>(certificate.budget.other_cost));
  hash_add(hash, certificate.insertions.size());
  for (const auto& insertion : certificate.insertions) {
    hash_add(hash, insertion.semantic_id);
    hash_position(hash, insertion.position);
    hash_action(hash, insertion.action);
  }
  hash_add(hash, certificate.hard_obligations.size());
  for (const auto& obligation : certificate.hard_obligations) {
    hash_add(hash, obligation.obligation_id);
    hash_add(hash, static_cast<std::uint32_t>(obligation.source_step));
    hash_action(hash, obligation.action);
    hash_position(hash, obligation.position_before);
    hash_add(hash, obligation.pre_state_fingerprint);
    hash_add(hash, obligation.post_state_fingerprint);
  }
  hash_add(hash, certificate.proof_slots.size());
  for (const auto& slot : certificate.proof_slots) {
    hash_add(hash, static_cast<std::uint32_t>(slot.step));
    hash_action(hash, slot.raw_source);
    hash_add(hash, static_cast<std::uint8_t>(slot.source_kind));
    hash_position(hash, slot.actor_position_before);
    hash_action(hash, slot.emitted);
    hash_add(hash, static_cast<std::uint32_t>(slot.emitted_source_step));
    hash_add(hash, slot.semantic_id);
    hash_add(hash, static_cast<std::uint32_t>(slot.stamina_before));
    hash_add(hash, static_cast<std::uint32_t>(slot.stamina_after));
    hash_add(hash, slot.pre_state_fingerprint);
    hash_add(hash, slot.post_state_fingerprint);
  }
  hash_add(hash, certificate.authorized_move_replays.size());
  for (const auto& replay : certificate.authorized_move_replays) {
    hash_add(hash, static_cast<std::uint32_t>(replay.source_step));
    hash_add(hash, static_cast<std::uint32_t>(replay.emitted_step));
    hash_action(hash, replay.exact_move);
  }
  hash_add(hash, certificate.owner_projection.content_hash);
  return hash;
}

VerificationResult verify_certificate(
    const ProductionFrozenDaySuffixCertificate& certificate,
    const Simulator& phase_start,
    const std::vector<std::vector<Action>>& raw_units_by_tick) {
  VerificationResult out;
  auto reject = [&](RejectReason reason) {
    out.valid = false;
    out.reject = reason;
    return out;
  };
  if (certificate.content_hash != production_certificate_hash(certificate))
    return reject(RejectReason::InternalProofFailure);
  if (phase_start.hour() != 0 || phase_start.day() != certificate.day)
    return reject(RejectReason::NotAtDayStart);
  if (phase_start.config().turns_per_day != kCertificateTurns)
    return reject(RejectReason::UnsupportedTurnsPerDay);
  const int day_start = certificate.day * kCertificateTurns;
  if (phase_start.done() || day_start + kCertificateTurns - 1 >
      phase_start.config().episode_steps - 2)
    return reject(RejectReason::TerminalDay);
  if (certificate.player < 0 || certificate.player >= 2 ||
      certificate.actor < 0 || certificate.actor_generation !=
          expected_actor_generation(certificate.day, certificate.actor) ||
      certificate.issuer_generation == 0)
    return reject(RejectReason::InvalidIdentity);
  const int actors = static_cast<int>(
      phase_start.farms()[static_cast<std::size_t>(certificate.player)]
          .hands.size()) + 1;
  if (certificate.actor >= actors ||
      raw_units_by_tick.size() != kCertificateTurns ||
      std::any_of(raw_units_by_tick.begin(), raw_units_by_tick.end(),
                  [&](const auto& tick) {
                    return static_cast<int>(tick.size()) != actors;
                  }) ||
      certificate.proof_slots.size() != kCertificateTurns)
    return reject(RejectReason::InvalidRouteShape);
  for (const auto& tick : raw_units_by_tick)
    for (const auto& action : tick)
      if (!is_unit_action(action.op))
        return reject(RejectReason::InvalidSourceAction);
  if (certificate.budget.available_stamina < 0 ||
      certificate.budget.move_cost < 0 ||
      certificate.budget.production_cost < 0 ||
      certificate.budget.other_cost < 0)
    return reject(RejectReason::StaminaExceeded);
  std::set<std::uint64_t> insertion_ids;
  for (const auto& insertion : certificate.insertions)
    if (insertion.semantic_id == 0 ||
        !insertion_ids.insert(insertion.semantic_id).second ||
        !valid_position(phase_start, insertion.position) ||
        !is_unit_action(insertion.action.op) ||
        insertion.action.op == Op::PASS || is_move(insertion.action.op))
      return reject(RejectReason::InvalidInsertion);
  if (certificate.focal_phase_start_fingerprint !=
          focal_unit_state_fingerprint(phase_start, certificate.player) ||
      certificate.raw_day_route_hash != raw_route_hash(raw_units_by_tick))
    return reject(RejectReason::InternalProofFailure);
  const auto& projection = certificate.owner_projection;
  if (projection.player != certificate.player ||
      projection.day != certificate.day ||
      projection.actor != certificate.actor ||
      projection.actor_generation != certificate.actor_generation ||
      projection.issuer_generation != certificate.issuer_generation ||
      !projection.sinks_irrevocable ||
      projection.suffix_start_step != day_start ||
      projection.slots.size() != kCertificateTurns ||
      projection.content_hash != online_elastic::frozen_suffix_hash(projection))
    return reject(RejectReason::InternalProofFailure);

  IssueRequest replay_request;
  replay_request.phase_start = &phase_start;
  replay_request.player = certificate.player;
  replay_request.actor = certificate.actor;
  replay_request.raw_units_by_tick = raw_units_by_tick;
  Simulator state = phase_start;
  std::size_t insertion_cursor = 0;
  std::size_t hard_cursor = 0;
  int stamina = 0;
  std::vector<std::pair<int, Action>> expected_moves;
  std::vector<std::pair<int, Action>> emitted_moves;
  for (int tick = 0; tick < kCertificateTurns; ++tick) {
    const int step = day_start + tick;
    const auto& raw = raw_units_by_tick[static_cast<std::size_t>(tick)]
                                      [static_cast<std::size_t>(certificate.actor)];
    const auto& proof = certificate.proof_slots[static_cast<std::size_t>(tick)];
    const auto& projected = projection.slots[static_cast<std::size_t>(tick)];
    if (proof.step != step || projected.source_step != step ||
        !action_equal(proof.raw_source, raw) ||
        !action_equal(projected.source_action, raw) ||
        proof.source_kind != projected.kind ||
        proof.pre_state_fingerprint != focal_unit_state_fingerprint(
            state, certificate.player, step) ||
        !position_equal(proof.actor_position_before,
                        actor_position(state, certificate.player,
                                       certificate.actor)) ||
        proof.stamina_before != stamina)
      return reject(RejectReason::InternalProofFailure);

    if (projected.kind == FrozenSlotKind::MoveToken) {
      if (!is_move(raw.op)) return reject(RejectReason::InvalidSourceAction);
      expected_moves.emplace_back(step, raw);
    } else if (projected.kind ==
               FrozenSlotKind::HardSemanticObligation) {
      if (raw.op == Op::PASS || is_move(raw.op) ||
          proof.emitted_source_step != step ||
          !action_equal(proof.emitted, raw) ||
          hard_cursor >= certificate.hard_obligations.size())
        return reject(RejectReason::InternalProofFailure);
      const auto& obligation = certificate.hard_obligations[hard_cursor++];
      if (obligation.obligation_id == 0 ||
          obligation.source_step != step ||
          !action_equal(obligation.action, raw) ||
          !position_equal(obligation.position_before,
                          proof.actor_position_before) ||
          obligation.pre_state_fingerprint != proof.pre_state_fingerprint ||
          obligation.post_state_fingerprint != proof.post_state_fingerprint)
        return reject(RejectReason::InternalProofFailure);
    } else {
      if (is_move(raw.op)) return reject(RejectReason::InvalidSourceAction);
      if (raw.op != Op::PASS) {
        Simulator raw_after = state;
        if (has_exact_effect(state, replay_request, tick, raw, raw_after))
          return reject(RejectReason::UnsafeSink);
      }
    }

    if (proof.emitted_source_step == -1) {
      if (insertion_cursor >= certificate.insertions.size())
        return reject(RejectReason::InvalidInsertion);
      const auto& insertion = certificate.insertions[insertion_cursor++];
      if (proof.semantic_id != insertion.semantic_id ||
          !action_equal(proof.emitted, insertion.action) ||
          !position_equal(proof.actor_position_before, insertion.position))
        return reject(RejectReason::InvalidInsertion);
    } else if (proof.emitted_source_step == -2) {
      if (proof.emitted.op != Op::PASS ||
          projected.kind != FrozenSlotKind::CertifiedSink ||
          raw.op == Op::PASS)
        return reject(RejectReason::InternalProofFailure);
    } else {
      const int source_tick = proof.emitted_source_step - day_start;
      if (source_tick < 0 || source_tick >= kCertificateTurns ||
          !action_equal(
              proof.emitted,
              raw_units_by_tick[static_cast<std::size_t>(source_tick)]
                               [static_cast<std::size_t>(certificate.actor)]))
        return reject(RejectReason::InternalProofFailure);
      if (is_move(proof.emitted.op))
        emitted_moves.emplace_back(proof.emitted_source_step, proof.emitted);
      else if (proof.emitted_source_step != step)
        return reject(RejectReason::InternalProofFailure);
    }
    if (proof.emitted_source_step != -1 && proof.semantic_id != 0)
      return reject(RejectReason::InternalProofFailure);

    const int cost = action_cost(certificate.budget, proof.emitted);
    if (stamina > certificate.budget.available_stamina - cost)
      return reject(RejectReason::StaminaExceeded);
    Simulator after = state;
    if (proof.emitted.op == Op::PASS) {
      after = apply_tick(state, replay_request, tick, proof.emitted);
    } else if (!has_exact_effect(state, replay_request, tick, proof.emitted,
                                 after)) {
      return reject(RejectReason::PreconditionsUnsatisfied);
    }
    stamina += cost;
    if (proof.stamina_after != stamina ||
        proof.post_state_fingerprint != focal_unit_state_fingerprint(
            after, certificate.player, step))
      return reject(RejectReason::InternalProofFailure);
    state = std::move(after);
    ++out.checked_slots;
  }
  if (insertion_cursor != certificate.insertions.size() ||
      hard_cursor != certificate.hard_obligations.size() ||
      emitted_moves.size() != expected_moves.size() ||
      certificate.authorized_move_replays.size() != emitted_moves.size())
    return reject(RejectReason::SlotCapacity);
  for (std::size_t index = 0; index < expected_moves.size(); ++index) {
    if (emitted_moves[index].first != expected_moves[index].first ||
        !action_equal(emitted_moves[index].second,
                      expected_moves[index].second))
      return reject(RejectReason::InternalProofFailure);
    const auto& authorization = certificate.authorized_move_replays[index];
    const auto proof = std::find_if(
        certificate.proof_slots.begin(), certificate.proof_slots.end(),
        [&](const auto& slot) {
          return slot.emitted_source_step == authorization.source_step &&
              is_move(slot.emitted.op);
        });
    if (proof == certificate.proof_slots.end() ||
        authorization.source_step != expected_moves[index].first ||
        authorization.emitted_step != proof->step ||
        authorization.emitted_step < authorization.source_step ||
        authorization.source_step / kCertificateTurns != certificate.day ||
        authorization.emitted_step / kCertificateTurns != certificate.day ||
        !action_equal(authorization.exact_move, expected_moves[index].second))
      return reject(RejectReason::InternalProofFailure);
  }
  out.valid = true;
  out.reject = RejectReason::None;
  return out;
}

bool ProductionSuffixScheduler::was_issued(int player, int day,
                                           int actor) const {
  return issued_hashes_.contains(std::tuple{player, day, actor});
}

const char* reject_reason_name(RejectReason reason) {
  switch (reason) {
    case RejectReason::None: return "none";
    case RejectReason::NullState: return "null_state";
    case RejectReason::NotAtDayStart: return "not_at_day_start";
    case RejectReason::UnsupportedTurnsPerDay:
      return "unsupported_turns_per_day";
    case RejectReason::TerminalDay: return "terminal_day";
    case RejectReason::InvalidIdentity: return "invalid_identity";
    case RejectReason::InvalidRouteShape: return "invalid_route_shape";
    case RejectReason::InvalidSourceAction: return "invalid_source_action";
    case RejectReason::InvalidInsertion: return "invalid_insertion";
    case RejectReason::UnsafeSink: return "unsafe_sink";
    case RejectReason::PreconditionsUnsatisfied:
      return "preconditions_unsatisfied";
    case RejectReason::SlotCapacity: return "slot_capacity";
    case RejectReason::StaminaExceeded: return "stamina_exceeded";
    case RejectReason::CrossDay: return "cross_day";
    case RejectReason::AlreadyIssued: return "already_issued";
    case RejectReason::InternalProofFailure: return "internal_proof_failure";
  }
  return "unknown";
}

}  // namespace g001::production_suffix
