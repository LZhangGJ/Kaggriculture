#include "repair_fork_evaluator.hpp"

#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace g001::repair_fork {
namespace {

using fastkag::Action;
using fastkag::NativeAgentState;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;

bool action_equal(const Action& a, const Action& b) {
  return a.op == b.op && a.item == b.item && a.quantity == b.quantity;
}

bool actions_equal(std::span<const Action> a, std::span<const Action> b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), action_equal);
}

bool player_action_equal(const PlayerAction& a, const PlayerAction& b) {
  return actions_equal(a.units, b.units) && actions_equal(a.market, b.market);
}

bool pair_action_equal(const std::array<PlayerAction, 2>& a,
                       const std::array<PlayerAction, 2>& b) {
  return player_action_equal(a[0], b[0]) && player_action_equal(a[1], b[1]);
}

bool receipt_equal(const ActionReceipt& a, const ActionReceipt& b) {
  return a.submitted_step == b.submitted_step && a.actor == b.actor &&
         a.manifest_generation == b.manifest_generation &&
         a.prefix_manifest_hash == b.prefix_manifest_hash &&
         a.post_prefix_state_fingerprint == b.post_prefix_state_fingerprint &&
         action_equal(a.emitted, b.emitted);
}

bool purchase_receipt_equal(const PurchaseReceipt& a,
                            const PurchaseReceipt& b) {
  return a.debt_id == b.debt_id && a.submitted_step == b.submitted_step &&
         a.operation == b.operation && a.item == b.item &&
         a.requested == b.requested && a.filled == b.filled &&
         a.market_slot == b.market_slot &&
         a.compile_status == b.compile_status;
}

bool tile_equal(const fastkag::Tile& a, const fastkag::Tile& b) {
  return a.kind == b.kind && a.crop == b.crop && a.animal == b.animal &&
         a.planted_day == b.planted_day && a.placed_day == b.placed_day &&
         a.yield_units == b.yield_units &&
         a.consecutive_unwatered == b.consecutive_unwatered &&
         a.consecutive_unfed == b.consecutive_unfed &&
         a.fertilized_until_day == b.fertilized_until_day &&
         a.pending_care_bonus == b.pending_care_bonus &&
         a.max_lifespan_step == b.max_lifespan_step &&
         a.watered_today == b.watered_today && a.fed_today == b.fed_today &&
         a.cared_today == b.cared_today &&
         a.fertilizer_available == b.fertilizer_available;
}

bool exact_environment_equal(const Simulator& a, const Simulator& b) {
  if (a.step_count() != b.step_count() || a.done() != b.done() ||
      a.market().inventory != b.market().inventory ||
      a.market().prices != b.market().prices || a.shops() != b.shops() ||
      a.last_market_fills() != b.last_market_fills() ||
      a.last_market_cash_shortfalls() != b.last_market_cash_shortfalls() ||
      a.last_end_of_day_overflow() != b.last_end_of_day_overflow())
    return false;
  for (int player = 0; player < 2; ++player) {
    const auto& x = a.farms()[player];
    const auto& y = b.farms()[player];
    if (x.money != y.money || x.farmer.x != y.farmer.x ||
        x.farmer.y != y.farmer.y || x.hands.size() != y.hands.size() ||
        x.tiles.size() != y.tiles.size() || x.unlocked_mask != y.unlocked_mask ||
        x.hires_today != y.hires_today)
      return false;
    for (std::size_t i = 0; i < x.hands.size(); ++i)
      if (x.hands[i].x != y.hands[i].x || x.hands[i].y != y.hands[i].y)
        return false;
    for (std::size_t i = 0; i < x.tiles.size(); ++i)
      if (!tile_equal(x.tiles[i], y.tiles[i])) return false;
    const auto& px = a.privates()[player];
    const auto& py = b.privates()[player];
    if (px.shed != py.shed || px.seeds != py.seeds ||
        px.inventories != py.inventories ||
        px.inventory_order != py.inventory_order)
      return false;
  }
  return true;
}

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST || op == Op::WEST;
}

double score(double own, double opponent) {
  return own > opponent ? 1.0 : own < opponent ? 0.0 : 0.5;
}

fastkag::Config config_for(Panel panel) {
  fastkag::Config config;
  if (panel == Panel::ForcedWeed) config.weed_spawn_chance = 1.0;
  return config;
}

std::uint64_t compute_full_unit_phase_state_fingerprint(const Simulator& env) {
  // A stable audit token, not a security primitive.  It binds authority to
  // the exact public/private phase-start facts used by the owner.
  std::uint64_t h = 1469598103934665603ULL;
  auto add = [&](const auto& value) {
    const auto* p = reinterpret_cast<const unsigned char*>(&value);
    for (std::size_t i = 0; i < sizeof(value); ++i) {
      h ^= p[i];
      h *= 1099511628211ULL;
    }
  };
  const auto& cfg = env.config();
  add(cfg.episode_steps); add(cfg.board_size); add(cfg.starting_money);
  add(cfg.max_market_orders); add(cfg.turns_per_day); add(cfg.shed_capacity);
  add(cfg.weed_spawn_chance); add(cfg.town_shop_unlock_interval);
  add(cfg.town_shop_sell_interval); add(cfg.town_center_sell_interval);
  add(cfg.farm_hand_cost_mult);
  add(env.seed()); add(env.step_count()); add(env.done());
  for (const auto v : env.market().inventory) add(v);
  for (const auto v : env.market().prices) add(v);
  add(env.shops().size());
  for (const auto v : env.shops()) add(v);
  for (int player = 0; player < 2; ++player) {
    add(player);
    const auto& farm = env.farms()[player];
    add(farm.money); add(farm.farmer.x); add(farm.farmer.y);
    add(farm.unlocked_mask); add(farm.hires_today); add(farm.hands.size());
    for (const auto& p : farm.hands) { add(p.x); add(p.y); }
    add(farm.tiles.size());
    for (const auto& tile : farm.tiles) {
      add(tile.kind); add(tile.crop); add(tile.animal); add(tile.planted_day);
      add(tile.placed_day); add(tile.yield_units);
      add(tile.consecutive_unwatered); add(tile.consecutive_unfed);
      add(tile.fertilized_until_day); add(tile.pending_care_bonus);
      add(tile.max_lifespan_step); add(tile.watered_today);
      add(tile.fed_today); add(tile.cared_today); add(tile.fertilizer_available);
    }
    const auto& pr = env.privates()[player];
    for (const auto v : pr.shed) add(v);
    for (const auto v : pr.seeds) add(v);
    add(pr.inventories.size());
    for (const auto& inv : pr.inventories) for (const auto v : inv) add(v);
    add(pr.inventory_order.size());
    for (const auto& order : pr.inventory_order) {
      add(order.size());
      for (const auto v : order) add(v);
    }
    const auto& fills = env.last_market_fills()[player];
    add(fills.size());
    for (const auto v : fills) add(v);
    const auto& shortfalls = env.last_market_cash_shortfalls()[player];
    add(shortfalls.size());
    for (const auto v : shortfalls) add(v);
    add(env.last_end_of_day_overflow()[player]);
  }
  return h;
}

struct MoveToken {
  int actor{};
  int emitted_step{};
  int source_step{};
  Op emitted{Op::PASS};
  Op source{Op::PASS};
};

struct PendingPurchase {
  PurchaseBinding binding;
  int submitted_step{};
};

std::vector<PurchaseReceipt> settle_receipts(
    const Simulator& after, int player, std::span<const PendingPurchase> pending,
    GameResult& result) {
  std::vector<PurchaseReceipt> receipts;
  const auto& fills = after.last_market_fills()[player];
  std::vector<int> remaining_fill(fills.begin(), fills.end());
  for (const auto& p : pending) {
    int filled = 0;
    if (p.binding.market_slot >= 0 &&
        p.binding.market_slot < static_cast<int>(remaining_fill.size())) {
      auto& available = remaining_fill[static_cast<std::size_t>(p.binding.market_slot)];
      filled = std::min(p.binding.requested, available);
      available -= filled;
    }
    receipts.push_back({p.binding.debt_id, p.submitted_step,
                        p.binding.operation, p.binding.item,
                        p.binding.requested, filled, p.binding.market_slot,
                        p.binding.status});
    if (p.binding.status == PurchaseCompileStatus::RejectedInvalid ||
        p.binding.status == PurchaseCompileStatus::RejectedNoSlot)
      continue;
    if (filled <= 0) ++result.purchase_zero_fills;
    else if (filled < p.binding.requested) ++result.purchase_partial_fills;
    else ++result.purchase_full_fills;
  }
  return receipts;
}

void audit_sources(const RepairContext& context, const RepairDecision& decision,
                   const std::vector<std::vector<Action>>& raw_history,
                   std::vector<MoveToken>& moves, GameResult& result) {
  if (decision.sources.size() != decision.units.size()) {
    ++result.move_source_failures;
    return;
  }
  for (std::size_t actor = 0; actor < decision.units.size(); ++actor) {
    const auto& emitted = decision.units[actor];
    const auto& source = decision.sources[actor];
    if (!is_move(emitted.op)) continue;
    const bool history_valid = source.source_step >= 0 &&
        source.source_step < static_cast<int>(raw_history.size()) &&
        source.actor >= 0 &&
        source.actor < static_cast<int>(
            raw_history[static_cast<std::size_t>(source.source_step)].size()) &&
        action_equal(source.source_action,
                     raw_history[static_cast<std::size_t>(source.source_step)]
                                [static_cast<std::size_t>(source.actor)]);
    const bool source_valid = source.actor == static_cast<int>(actor) &&
        source.source_step >= 0 &&
        action_equal(source.source_action, emitted) &&
        is_move(source.source_action.op) && history_valid;
    result.move_source_failures += !source_valid;
    if (!source_valid) continue;
    result.move_direction_failures += source.source_action.op != emitted.op;
    result.move_day_failures += source.source_step / 24 != context.step / 24;
    result.move_slot_failures += source.source_step % 24 != context.step % 24;
    moves.push_back({static_cast<int>(actor), context.step, source.source_step,
                     emitted.op, source.source_action.op});
  }
}

struct Loaded {
  fastkag::NativeTapeLibrary library;
  int focal{};
  int opponent{};
};

Loaded load_tapes(const EvaluatorOptions& options) {
  Loaded out;
  out.library.routes.push_back(
      g001::repair::load_route(options.tapes, options.library, "G001"));
  out.focal = 0;
  out.library.routes.push_back(
      g001::repair::load_route(options.tapes, options.library, options.opponent));
  out.opponent = 1;
  out.library.r5_reference =
      g001::repair::load_route(options.references, options.library, "R5");
  out.library.md_reference =
      g001::repair::load_route(options.references, options.library, "MD");
  constexpr std::array<std::string_view, 5> labels{
      "10C4S_3Q", "8C6S_3Q", "6C8S_3Q", "6C12S_4Q_FIRST_YARN",
      "6C12S_4Q_SECOND_YARN"};
  for (std::size_t i = 0; i < labels.size(); ++i) {
    out.library.moon[i] = g001::repair::load_route(
        options.references, options.library, "MOON_" + std::string(labels[i]));
    out.library.moon_legacy[i] = g001::repair::load_route(
        options.references, options.library,
        "MOON_LEGACY_" + std::string(labels[i]));
  }
  return out;
}

GameResult run_game(const fastkag::NativeTeammateExecutor& executor,
                    int focal_route, int opponent_route, std::uint64_t seed,
                    int seat, Panel panel,
                    const fastkag::NativeRepairOptions& candidate_repair,
                    RepairOwner& owner) {
  Simulator baseline(config_for(panel), seed);
  Simulator candidate(config_for(panel), seed);
  std::array<NativeAgentState, 2> base_state;
  std::array<NativeAgentState, 2> cand_state;
  ExactMarketCompiler market_compiler;
  std::vector<PurchaseReceipt> previous_receipts;
  std::vector<ActionReceipt> previous_action_receipts;
  std::vector<PendingPurchase> pending;
  std::vector<MoveToken> candidate_moves;
  std::vector<MoveToken> baseline_moves;
  std::vector<std::vector<Op>> baseline_routes;
  std::vector<std::vector<Op>> candidate_routes;
  std::vector<std::vector<Action>> raw_history;
  std::vector<std::uint64_t> last_manifest_generation;
  std::set<std::uint64_t> unresolved_purchase_debts;
  GameResult out;
  out.seed = seed;
  out.seat = seat;
  out.panel = panel;

  while (!baseline.done()) {
    if (candidate.done() || baseline.step_count() != candidate.step_count())
      throw std::runtime_error("paired branch clocks diverged");
    const int step = baseline.step_count();
    const int route0 = seat == 0 ? focal_route : opponent_route;
    const int route1 = seat == 1 ? focal_route : opponent_route;
    std::array<PlayerAction, 2> base_actions{
        executor.action_external(baseline, 0, route0, base_state[0]),
        executor.action_external(baseline, 1, route1, base_state[1])};
    std::array<PlayerAction, 2> cand_actions{
        executor.action_external(candidate, 0, route0, cand_state[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, false,
            seat == 0 ? candidate_repair : fastkag::NativeRepairOptions{}),
        executor.action_external(candidate, 1, route1, cand_state[1],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, false,
            seat == 1 ? candidate_repair : fastkag::NativeRepairOptions{})};

    auto& raw = cand_actions[seat];
    if (last_manifest_generation.size() < raw.units.size())
      last_manifest_generation.resize(raw.units.size());
    for (const auto& weed : cand_state[seat].weed)
      if (weed.active && weed.start == step) ++out.observed_legacy_weed_triggers;
    raw_history.push_back(raw.units);
    const RepairContext context{candidate, seat, step, raw, cand_actions,
                                previous_receipts, previous_action_receipts};
    auto decision = owner.decide(context);
    if (decision.units.empty() && !raw.units.empty())
      throw std::runtime_error("repair owner returned an empty unit manifest");
    audit_sources(context, decision, raw_history, candidate_moves, out);

    if (decision.receipt_acks.size() != previous_action_receipts.size()) {
      ++out.receipts_failed;
      throw std::runtime_error("repair owner did not acknowledge every action receipt");
    }
    for (std::size_t i = 0; i < previous_action_receipts.size(); ++i) {
      if (!receipt_equal(decision.receipt_acks[i], previous_action_receipts[i])) {
        ++out.receipts_failed;
        throw std::runtime_error("repair owner action receipt token mismatch");
      }
      ++out.action_receipts_acked;
    }
    if (decision.purchase_receipt_acks.size() != previous_receipts.size()) {
      ++out.receipts_failed;
      throw std::runtime_error(
          "repair owner did not acknowledge every purchase receipt");
    }
    for (std::size_t i = 0; i < previous_receipts.size(); ++i) {
      if (!purchase_receipt_equal(decision.purchase_receipt_acks[i],
                                  previous_receipts[i])) {
        ++out.receipts_failed;
        throw std::runtime_error("repair owner purchase receipt token mismatch");
      }
      ++out.purchase_receipts_acked;
    }

    const bool changed_units = !actions_equal(decision.units, raw.units);
    const bool changed_market = !decision.required_purchases.empty();
    const bool changed = changed_units || changed_market;
    std::vector<int> changed_actors;
    const auto common = std::min(decision.units.size(), raw.units.size());
    for (std::size_t actor = 0; actor < common; ++actor)
      if (!action_equal(decision.units[actor], raw.units[actor]))
        changed_actors.push_back(static_cast<int>(actor));
    if (decision.units.size() != raw.units.size())
      throw std::runtime_error("repair owner changed the actor manifest cardinality");
    if (decision.prefix_authority.size() != changed_actors.size()) {
      out.prefix_authority_failures +=
          std::max(decision.prefix_authority.size(), changed_actors.size());
      throw std::runtime_error("per-actor PrefixAuthority cardinality mismatch");
    }
    auto final_joint = cand_actions;
    final_joint[seat].units = decision.units;
    previous_action_receipts.clear();
    for (std::size_t i = 0; i < changed_actors.size(); ++i) {
      ++out.prefix_authority_checks;
      const int actor = changed_actors[i];
      const auto& authority = decision.prefix_authority[i];
      const bool authority_ok = authority.actor == actor &&
          authority.manifest_generation >
              last_manifest_generation[static_cast<std::size_t>(actor)] &&
          authority.prefix_manifest_hash ==
              unit_prefix_manifest_hash(decision.units, actor) &&
          authority.post_prefix_state_fingerprint ==
              post_unit_prefix_state_fingerprint(candidate, seat, final_joint,
                                                 actor);
      out.prefix_authority_failures += !authority_ok;
      if (!authority_ok)
        throw std::runtime_error("changed actor lacks exact PrefixAuthority");
      last_manifest_generation[static_cast<std::size_t>(actor)] =
          authority.manifest_generation;
      previous_action_receipts.push_back(
          {step, actor, authority.manifest_generation,
           authority.prefix_manifest_hash,
           authority.post_prefix_state_fingerprint,
           decision.units[static_cast<std::size_t>(actor)]});
      ++out.action_receipts_issued;
    }
    raw.units = std::move(decision.units);
    const auto compiled = market_compiler.compile(
        candidate, seat, raw.market, decision.required_purchases);
    raw.market = compiled.market;
    out.purchase_required += static_cast<int>(compiled.bindings.size());
    out.purchase_compile_rejected_invalid += compiled.rejected_invalid;
    out.purchase_compile_rejected_no_slot += compiled.rejected_no_slot;
    pending.clear();
    std::set<std::uint64_t> submitted_debts;
    for (const auto& binding : compiled.bindings)
      if (!submitted_debts.insert(binding.debt_id).second && binding.debt_id != 0)
        throw std::runtime_error("duplicate purchase debt id in one manifest");
      else {
        pending.push_back({binding, step});
        if (binding.debt_id != 0)
          unresolved_purchase_debts.insert(binding.debt_id);
      }

    out.repair_triggers += decision.telemetry.triggers;
    out.debts_opened += decision.telemetry.debts_opened;
    out.debts_closed += decision.telemetry.debts_closed;
    out.receipts_confirmed += decision.telemetry.receipts_confirmed;
    out.receipts_failed += decision.telemetry.receipts_failed;
    if (baseline.hour() == 23) {
      out.hour23_triggers += decision.telemetry.triggers;
      out.hour23_commits += changed;
      out.hour23_fail_closed += decision.telemetry.fail_closed;
    }

    const auto record_routes = [](std::span<const Action> units,
                                  std::vector<std::vector<Op>>& routes) {
      routes.resize(std::max(routes.size(), units.size()));
      for (std::size_t actor = 0; actor < units.size(); ++actor)
        if (is_move(units[actor].op)) routes[actor].push_back(units[actor].op);
    };
    record_routes(base_actions[seat].units, baseline_routes);
    record_routes(cand_actions[seat].units, candidate_routes);
    for (int actor = 0; actor < static_cast<int>(base_actions[seat].units.size()); ++actor) {
      const auto& a = base_actions[seat].units[static_cast<std::size_t>(actor)];
      if (is_move(a.op))
        baseline_moves.push_back({actor, step, step, a.op, a.op});
    }
    out.baseline_unit_failures +=
        fastkag::native_macro_unit_failures(baseline, seat, base_actions[seat]);
    out.candidate_unit_failures +=
        fastkag::native_macro_unit_failures(candidate, seat, cand_actions[seat]);
    out.action_mismatches += !pair_action_equal(base_actions, cand_actions);
    baseline.step(base_actions);
    candidate.step(cand_actions);
    out.baseline_market_failures +=
        fastkag::native_macro_market_failures(baseline, seat, base_actions[seat]);
    out.candidate_market_failures +=
        fastkag::native_macro_market_failures(candidate, seat, cand_actions[seat]);
    out.baseline_overflow += baseline.last_end_of_day_overflow()[seat];
    out.candidate_overflow += candidate.last_end_of_day_overflow()[seat];
    previous_receipts = settle_receipts(candidate, seat, pending, out);
    out.purchase_receipts_issued += static_cast<int>(previous_receipts.size());
    for (const auto& receipt : previous_receipts)
      if (receipt.debt_id != 0 && receipt.filled >= receipt.requested &&
          receipt.compile_status != PurchaseCompileStatus::RejectedInvalid &&
          receipt.compile_status != PurchaseCompileStatus::RejectedNoSlot)
        unresolved_purchase_debts.erase(receipt.debt_id);
    out.environment_mismatches += !exact_environment_equal(baseline, candidate);
    ++out.steps;
  }
  if (!candidate.done()) throw std::runtime_error("candidate branch did not finish");
  out.baseline_own = baseline.farms()[seat].money;
  out.baseline_opponent = baseline.farms()[1 - seat].money;
  out.candidate_own = candidate.farms()[seat].money;
  out.candidate_opponent = candidate.farms()[1 - seat].money;
  out.reward_mismatches = out.baseline_own != out.candidate_own ||
                          out.baseline_opponent != out.candidate_opponent;
  out.debts_outstanding = out.debts_opened - out.debts_closed;
  out.terminal_unacked_action_receipts =
      static_cast<int>(previous_action_receipts.size());
  out.terminal_unacked_purchase_receipts =
      static_cast<int>(previous_receipts.size());
  out.exact_purchase_debts_outstanding =
      static_cast<int>(unresolved_purchase_debts.size());

  if (baseline_moves.size() != candidate_moves.size()) {
    ++out.move_sequence_failures;
  } else {
    for (std::size_t i = 0; i < baseline_moves.size(); ++i) {
      const auto& a = baseline_moves[i];
      const auto& b = candidate_moves[i];
      if (a.actor != b.actor || a.emitted != b.emitted ||
          a.source_step != b.source_step) {
        ++out.move_sequence_failures;
        break;
      }
    }
  }
  baseline_routes.resize(std::max(baseline_routes.size(), candidate_routes.size()));
  candidate_routes.resize(baseline_routes.size());
  for (std::size_t actor = 0; actor < baseline_routes.size(); ++actor)
    out.move_route_failures += baseline_routes[actor] != candidate_routes[actor];
  return out;
}

std::string game_json(const GameResult& g) {
  const double base_margin = g.baseline_own - g.baseline_opponent;
  const double cand_margin = g.candidate_own - g.candidate_opponent;
  std::ostringstream os;
  os << std::fixed << std::setprecision(6)
     << "{\"panel\":\"" << panel_name(g.panel) << "\",\"seed\":" << g.seed
     << ",\"seat\":" << g.seat << ",\"steps\":" << g.steps
     << ",\"baseline\":{\"own\":" << g.baseline_own
     << ",\"opponent\":" << g.baseline_opponent << ",\"margin\":" << base_margin
     << ",\"score\":" << score(g.baseline_own, g.baseline_opponent)
     << ",\"unit_failures\":" << g.baseline_unit_failures
     << ",\"market_failures\":" << g.baseline_market_failures
     << ",\"overflow\":" << g.baseline_overflow << "}"
     << ",\"candidate\":{\"own\":" << g.candidate_own
     << ",\"opponent\":" << g.candidate_opponent << ",\"margin\":" << cand_margin
     << ",\"score\":" << score(g.candidate_own, g.candidate_opponent)
     << ",\"unit_failures\":" << g.candidate_unit_failures
     << ",\"market_failures\":" << g.candidate_market_failures
     << ",\"overflow\":" << g.candidate_overflow << "}"
     << ",\"delta\":{\"own\":" << g.candidate_own - g.baseline_own
     << ",\"opponent\":" << g.candidate_opponent - g.baseline_opponent
     << ",\"margin\":" << cand_margin - base_margin
     << ",\"score\":" << score(g.candidate_own, g.candidate_opponent) -
                                score(g.baseline_own, g.baseline_opponent) << "}"
     << ",\"parity\":{\"action_mismatches\":" << g.action_mismatches
     << ",\"environment_mismatches\":" << g.environment_mismatches
     << ",\"reward_mismatches\":" << g.reward_mismatches << "}"
     << ",\"move_invariants\":{\"direction_failures\":"
     << g.move_direction_failures << ",\"day_failures\":" << g.move_day_failures
     << ",\"slot_failures\":" << g.move_slot_failures
     << ",\"source_failures\":" << g.move_source_failures
     << ",\"sequence_failures\":" << g.move_sequence_failures
     << ",\"per_actor_route_failures\":" << g.move_route_failures << "}"
     << ",\"repair\":{\"legacy_weed_triggers\":"
     << g.observed_legacy_weed_triggers << ",\"typed_triggers\":" << g.repair_triggers
     << ",\"purchase_required\":" << g.purchase_required
     << ",\"purchase_zero_fills\":" << g.purchase_zero_fills
     << ",\"purchase_partial_fills\":" << g.purchase_partial_fills
     << ",\"purchase_full_fills\":" << g.purchase_full_fills
     << ",\"purchase_compile_rejected_invalid\":"
     << g.purchase_compile_rejected_invalid
     << ",\"purchase_compile_rejected_no_slot\":"
     << g.purchase_compile_rejected_no_slot
     << ",\"debts_opened\":" << g.debts_opened
     << ",\"debts_closed\":" << g.debts_closed
     << ",\"debts_outstanding\":" << g.debts_outstanding
     << ",\"receipts_confirmed\":" << g.receipts_confirmed
     << ",\"receipts_failed\":" << g.receipts_failed
     << ",\"action_receipts_issued\":" << g.action_receipts_issued
     << ",\"action_receipts_acked\":" << g.action_receipts_acked
     << ",\"purchase_receipts_issued\":" << g.purchase_receipts_issued
     << ",\"purchase_receipts_acked\":" << g.purchase_receipts_acked
     << ",\"terminal_unacked_action_receipts\":"
     << g.terminal_unacked_action_receipts
     << ",\"terminal_unacked_purchase_receipts\":"
     << g.terminal_unacked_purchase_receipts
     << ",\"exact_purchase_debts_outstanding\":"
     << g.exact_purchase_debts_outstanding
     << ",\"prefix_authority_checks\":" << g.prefix_authority_checks
     << ",\"prefix_authority_failures\":" << g.prefix_authority_failures
     << ",\"hour23_triggers\":" << g.hour23_triggers
     << ",\"hour23_commits\":" << g.hour23_commits
     << ",\"hour23_fail_closed\":" << g.hour23_fail_closed << "}}";
  return os.str();
}

// Compact, dependency-free SHA-256 used only to identify canonical reports.
constexpr std::array<std::uint32_t, 64> kSha256{
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};

}  // namespace

const char* panel_name(Panel panel) {
  return panel == Panel::Normal ? "normal" : "forced_weed";
}

std::uint64_t phase_start_fingerprint(const Simulator& env, int player) {
  if (player < 0 || player >= 2)
    throw std::invalid_argument("phase fingerprint player must be 0 or 1");
  // player remains validated for ABI misuse, while the token intentionally
  // commits to the complete two-player unit-phase state.
  return full_unit_phase_state_fingerprint(env);
}

std::uint64_t full_unit_phase_state_fingerprint(const Simulator& env) {
  return compute_full_unit_phase_state_fingerprint(env);
}

std::uint64_t unit_prefix_manifest_hash(std::span<const Action> final_units,
                                        int actor_exclusive) {
  if (actor_exclusive < 0 ||
      actor_exclusive > static_cast<int>(final_units.size()))
    throw std::invalid_argument("actor prefix is outside unit manifest");
  std::uint64_t hash = 1469598103934665603ULL;
  auto add = [&](std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      hash ^= static_cast<std::uint8_t>(value >> (8 * i));
      hash *= 1099511628211ULL;
    }
  };
  add(static_cast<std::uint64_t>(actor_exclusive));
  for (int actor = 0; actor < actor_exclusive; ++actor) {
    const auto& action = final_units[static_cast<std::size_t>(actor)];
    add(static_cast<std::uint8_t>(action.op));
    add(static_cast<std::uint8_t>(action.item));
    add(static_cast<std::uint32_t>(action.quantity));
  }
  return hash;
}

std::uint64_t post_unit_prefix_state_fingerprint(
    const Simulator& phase_start, int player,
    const std::array<PlayerAction, 2>& final_joint, int actor_exclusive) {
  if (player < 0 || player >= 2)
    throw std::invalid_argument("prefix fingerprint player must be 0 or 1");
  if (actor_exclusive < 0 ||
      actor_exclusive > static_cast<int>(final_joint[player].units.size()))
    throw std::invalid_argument("actor prefix is outside final manifest");
  auto prefix = final_joint;
  for (std::size_t actor = static_cast<std::size_t>(actor_exclusive);
       actor < prefix[player].units.size(); ++actor)
    prefix[player].units[actor] = {Op::PASS};
  // Official unit order is player 0 then player 1.  A player-0 prefix must not
  // include any later player-1 effects; player 1 must include all player-0
  // effects because they are part of its exact phase prefix.
  if (player == 0)
    for (auto& action : prefix[1].units) action = {Op::PASS};
  const auto preview = phase_start.preview_unit_phase(prefix);
  return phase_start_fingerprint(preview, player);
}

std::string PassThroughOwner::name() const { return "pass_through"; }

RepairDecision PassThroughOwner::decide(const RepairContext& context) {
  RepairDecision out;
  out.units = context.raw_g001.units;
  out.sources.reserve(out.units.size());
  for (std::size_t actor = 0; actor < out.units.size(); ++actor)
    out.sources.push_back({static_cast<int>(actor), context.step, out.units[actor]});
  return out;
}

MarketCompileResult ExactMarketCompiler::compile(
    const Simulator& phase_start, int, std::span<const Action> legacy_market,
    std::span<const RequiredPurchase> required) const {
  MarketCompileResult out;
  out.market.assign(legacy_market.begin(), legacy_market.end());
  std::vector<int> allocated(out.market.size());
  for (const auto& request : required) {
    const int item = static_cast<int>(request.item);
    const bool typed_item =
        (request.operation == Op::BUY_SEED && item >= 0 &&
         item < fastkag::N_CROPS) ||
        (request.operation == Op::BUY_ANIMAL && item >= 9 &&
         item < fastkag::N_ITEMS);
    if (!typed_item || request.quantity <= 0 || request.debt_id == 0) {
      ++out.rejected_invalid;
      out.bindings.push_back({request.debt_id, -1, request.operation,
                              request.item, request.quantity,
                              PurchaseCompileStatus::RejectedInvalid});
      continue;
    }
    int slot = -1;
    for (std::size_t i = 0; i < out.market.size(); ++i) {
      const auto& action = out.market[i];
      if (action.op == request.operation && action.item == request.item &&
          action.quantity - allocated[i] >= request.quantity) {
        slot = static_cast<int>(i);
        break;
      }
    }
    auto status = PurchaseCompileStatus::BoundExisting;
    if (slot < 0) {
      if (out.market.size() >=
          static_cast<std::size_t>(phase_start.config().max_market_orders)) {
        ++out.rejected_no_slot;
        out.bindings.push_back({request.debt_id, -1, request.operation,
                                request.item, request.quantity,
                                PurchaseCompileStatus::RejectedNoSlot});
        continue;
      }
      slot = static_cast<int>(out.market.size());
      out.market.push_back(
          {request.operation, request.item, request.quantity});
      allocated.push_back(0);
      status = PurchaseCompileStatus::Appended;
    }
    allocated[static_cast<std::size_t>(slot)] += request.quantity;
    out.bindings.push_back({request.debt_id, slot, request.operation,
                            request.item, request.quantity, status});
  }
  return out;
}

std::string sha256_hex(std::string_view input) {
  std::vector<std::uint8_t> bytes(input.begin(), input.end());
  const std::uint64_t bit_size = static_cast<std::uint64_t>(bytes.size()) * 8ULL;
  bytes.push_back(0x80U);
  while (bytes.size() % 64 != 56) bytes.push_back(0);
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(bit_size >> shift));
  std::array<std::uint32_t, 8> h{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                                 0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
  for (std::size_t offset = 0; offset < bytes.size(); offset += 64) {
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
      const auto j = offset + static_cast<std::size_t>(4 * i);
      w[i] = (static_cast<std::uint32_t>(bytes[j]) << 24) |
             (static_cast<std::uint32_t>(bytes[j + 1]) << 16) |
             (static_cast<std::uint32_t>(bytes[j + 2]) << 8) | bytes[j + 3];
    }
    for (int i = 16; i < 64; ++i) {
      const auto s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const auto s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
    for (int i = 0; i < 64; ++i) {
      const auto s1=std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25);
      const auto ch=(e&f)^((~e)&g);
      const auto t1=z+s1+ch+kSha256[i]+w[i];
      const auto s0=std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22);
      const auto maj=(a&b)^(a&c)^(b&c);
      const auto t2=s0+maj;
      z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
  }
  std::ostringstream os;
  os << std::hex << std::setfill('0');
  for (const auto value : h) os << std::setw(8) << value;
  return os.str();
}

EvaluationReport evaluate(
    const EvaluatorOptions& options,
    const std::function<std::unique_ptr<RepairOwner>()>& owner_factory) {
  if (options.seeds <= 0 || options.threads <= 0 || options.panels.empty())
    throw std::invalid_argument("seeds, threads, and panels must be non-empty");
  if (!owner_factory) throw std::invalid_argument("owner factory is required");
  const auto loaded = load_tapes(options);
  const fastkag::NativeTeammateExecutor executor(loaded.library);
  struct Task { Panel panel; std::uint64_t seed; int seat; };
  std::vector<Task> tasks;
  for (const auto panel : options.panels)
    for (int offset = 0; offset < options.seeds; ++offset)
      for (int seat = 0; seat < 2; ++seat)
        tasks.push_back({panel, options.seed_begin +
                                    static_cast<std::uint64_t>(offset), seat});
  std::vector<GameResult> games(tasks.size());
  std::atomic<std::size_t> cursor{};
  std::mutex error_mutex;
  std::exception_ptr first_error;
  std::vector<std::thread> workers;
  for (int i = 0; i < std::min<int>(options.threads, tasks.size()); ++i)
    workers.emplace_back([&] {
      try {
        for (;;) {
          const auto index = cursor.fetch_add(1);
          if (index >= tasks.size()) break;
          auto owner = owner_factory();
          if (!owner) throw std::runtime_error("owner factory returned null");
          const auto& task = tasks[index];
          games[index] = run_game(
              executor, loaded.focal, loaded.opponent, task.seed, task.seat,
              task.panel, fastkag::native_repair_options_from_mask(
                              options.candidate_repair_mask),
              *owner);
        }
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (!first_error) first_error = std::current_exception();
        cursor.store(tasks.size());
      }
    });
  for (auto& worker : workers) worker.join();
  if (first_error) std::rethrow_exception(first_error);
  std::sort(games.begin(), games.end(), [](const auto& a, const auto& b) {
    return std::tuple{a.panel, a.seed, a.seat} <
           std::tuple{b.panel, b.seed, b.seat};
  });
  EvaluationReport report;
  report.owner = owner_factory()->name();
  report.opponent = options.opponent;
  report.games = games;
  for (const auto& game : games) report.cases_jsonl += game_json(game) + "\n";

  long long steps=0, action_mismatch=0, env_mismatch=0, reward_mismatch=0;
  long long move_direction=0,move_day=0,move_slot=0,move_source=0,move_sequence=0;
  long long move_route=0;
  double own_delta=0,margin_delta=0,score_delta=0;
  for (const auto& g : games) {
    steps += g.steps; action_mismatch += g.action_mismatches;
    env_mismatch += g.environment_mismatches; reward_mismatch += g.reward_mismatches;
    move_direction += g.move_direction_failures; move_day += g.move_day_failures;
    move_slot += g.move_slot_failures; move_source += g.move_source_failures;
    move_sequence += g.move_sequence_failures;
    move_route += g.move_route_failures;
    own_delta += g.candidate_own-g.baseline_own;
    margin_delta += (g.candidate_own-g.candidate_opponent)-
                    (g.baseline_own-g.baseline_opponent);
    score_delta += score(g.candidate_own,g.candidate_opponent)-
                   score(g.baseline_own,g.baseline_opponent);
  }
  const double n = static_cast<double>(games.size());
  std::ostringstream summary;
  summary << std::fixed << std::setprecision(6)
          << "{\n  \"schema\":\"repair_fork_evaluator_v2\",\n"
          << "  \"mutates_deployed_native_agent\":false,\n"
          << "  \"candidate_branch_actions_applied\":true,\n"
          << "  \"owner\":\"" << report.owner << "\",\n"
          << "  \"opponent\":\"" << report.opponent << "\",\n";
  if (options.candidate_repair_mask != 0)
    summary << "  \"candidate_repair_mask\":"
            << options.candidate_repair_mask << ",\n";
  summary << "  \"games\":" << games.size() << ",\n"
          << "  \"steps\":" << steps << ",\n"
          << "  \"mean_own_delta\":" << own_delta/n << ",\n"
          << "  \"mean_margin_delta\":" << margin_delta/n << ",\n"
          << "  \"mean_score_delta\":" << score_delta/n << ",\n"
          << "  \"parity\":{\"action_mismatches\":" << action_mismatch
          << ",\"environment_mismatches\":" << env_mismatch
          << ",\"reward_mismatches\":" << reward_mismatch << "},\n"
          << "  \"move_invariants\":{\"direction_failures\":" << move_direction
          << ",\"day_failures\":" << move_day
          << ",\"slot_failures\":" << move_slot
          << ",\"source_failures\":" << move_source
          << ",\"sequence_failures\":" << move_sequence
          << ",\"per_actor_route_failures\":" << move_route << "}\n}"
          << '\n';
  report.summary_json = summary.str();
  report.deterministic_payload_sha256 =
      sha256_hex(report.summary_json + report.cases_jsonl);
  return report;
}

}  // namespace g001::repair_fork
