#include "repair_fork_evaluator.hpp"

#include <iostream>
#include <stdexcept>

#ifndef REPAIR_FORK_REPO_ROOT
#error REPAIR_FORK_REPO_ROOT must be defined
#endif

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

g001::repair_fork::EvaluatorOptions options() {
  const std::string root = REPAIR_FORK_REPO_ROOT;
  g001::repair_fork::EvaluatorOptions out;
  out.tapes = root + "/meta_agent_route_rl_submission_minimal/teammate_meta_route_submission_v1/route_actions.json.zlib";
  out.library = root + "/meta_agent_route_rl_submission_minimal/teammate_meta_route_submission_v1/route_library.json";
  out.references = root + "/findNewRoad/marketMechanismCpp/nativeSelectiveNtEval/artifacts/native_teammate_refs.json.zlib";
  out.seed_begin = 25772238701ULL;
  out.seeds = 1;
  out.threads = 2;
  out.panels = {g001::repair_fork::Panel::Normal,
                g001::repair_fork::Panel::ForcedWeed};
  return out;
}

void sha256_known_vector() {
  require(g001::repair_fork::sha256_hex("abc") ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 known vector mismatch");
}

void full_unit_phase_fingerprint_covers_hidden_causal_state() {
  fastkag::Simulator base({}, 991);
  {
    auto changed = base;
    auto& tile = const_cast<fastkag::Farm&>(changed.farms()[0]).tiles[0];
    tile.fertilized_until_day = 17;
    tile.consecutive_unwatered = 3;
    require(g001::repair_fork::full_unit_phase_state_fingerprint(base) !=
                g001::repair_fork::full_unit_phase_state_fingerprint(changed),
            "hidden FERTILIZE/tile state was omitted from full fingerprint");
  }
  {
    auto changed = base;
    auto& order = const_cast<fastkag::PrivateState&>(
        changed.privates()[0]).inventory_order[0];
    order.push_back(static_cast<std::int8_t>(fastkag::Item::WHEAT));
    require(g001::repair_fork::full_unit_phase_state_fingerprint(base) !=
                g001::repair_fork::full_unit_phase_state_fingerprint(changed),
            "DROP inventory_order was omitted from full fingerprint");
  }
  {
    auto changed = base;
    const_cast<fastkag::PrivateState&>(changed.privates()[1]).shed[0] = 9;
    require(g001::repair_fork::full_unit_phase_state_fingerprint(base) !=
                g001::repair_fork::full_unit_phase_state_fingerprint(changed),
            "other-player private unit state was omitted from full fingerprint");
  }
  {
    std::array<fastkag::PlayerAction, 2> pass;
    pass[0].units = {{fastkag::Op::PASS}};
    pass[1].units = {{fastkag::Op::PASS}};
    auto moved = pass;
    moved[0].units[0] = {fastkag::Op::NORTH};
    require(g001::repair_fork::post_unit_prefix_state_fingerprint(
                base, 1, pass, 0) !=
                g001::repair_fork::post_unit_prefix_state_fingerprint(
                    base, 1, moved, 0),
            "player-1 prefix did not bind earlier player-0 unit effects");
  }
}

void exact_pass_through_and_determinism() {
  const auto factory = [] {
    return std::make_unique<g001::repair_fork::PassThroughOwner>();
  };
  const auto first = g001::repair_fork::evaluate(options(), factory);
  const auto second = g001::repair_fork::evaluate(options(), factory);
  require(first.games.size() == 4, "dual-panel, dual-seat game count mismatch");
  require(first.cases_jsonl == second.cases_jsonl,
          "parallel cases report is nondeterministic");
  require(first.summary_json == second.summary_json,
          "parallel summary report is nondeterministic");
  require(first.deterministic_payload_sha256 ==
              second.deterministic_payload_sha256,
          "report hash is nondeterministic");
  require(first.deterministic_payload_sha256 ==
              "b383a25e9af17d18be48c07f421b6bb42393c21eb6b9a8d62d40f84eda52e84f",
          "pass-through canonical report hash changed");
  for (const auto& game : first.games) {
    require(game.steps == 719, "official episode did not emit 719 actions");
    require(game.action_mismatches == 0, "pass-through action parity failed");
    require(game.environment_mismatches == 0, "pass-through state parity failed");
    require(game.reward_mismatches == 0, "pass-through reward parity failed");
    require(game.move_direction_failures == 0 && game.move_day_failures == 0 &&
                game.move_slot_failures == 0 && game.move_source_failures == 0 &&
                game.move_sequence_failures == 0 &&
                game.move_route_failures == 0,
            "pass-through MOVE invariant failed");
  }
}

class UnauthorizedOwner final : public g001::repair_fork::RepairOwner {
 public:
  std::string name() const override { return "unauthorized"; }
  g001::repair_fork::RepairDecision decide(
      const g001::repair_fork::RepairContext& context) override {
    g001::repair_fork::RepairDecision out;
    out.units = context.raw_g001.units;
    out.sources.reserve(out.units.size());
    for (std::size_t actor = 0; actor < out.units.size(); ++actor)
      out.sources.push_back(
          {static_cast<int>(actor), context.step, out.units[actor]});
    if (!out.units.empty()) out.units[0] = {fastkag::Op::PASS};
    return out;
  }
};

class OneShotAuthorizedOwner final : public g001::repair_fork::RepairOwner {
 public:
  std::string name() const override { return "one_shot_authorized"; }
  g001::repair_fork::RepairDecision decide(
      const g001::repair_fork::RepairContext& context) override {
    g001::repair_fork::RepairDecision out;
    out.units = context.raw_g001.units;
    out.receipt_acks.assign(context.previous_action_receipts.begin(),
                            context.previous_action_receipts.end());
    out.purchase_receipt_acks.assign(
        context.previous_purchase_receipts.begin(),
        context.previous_purchase_receipts.end());
    int changed = -1;
    if (!committed_) {
      for (int actor = 0; actor < static_cast<int>(out.units.size()); ++actor) {
        const auto op = out.units[static_cast<std::size_t>(actor)].op;
        if (op != fastkag::Op::PASS && op != fastkag::Op::NORTH &&
            op != fastkag::Op::SOUTH && op != fastkag::Op::EAST &&
            op != fastkag::Op::WEST) {
          changed = actor;
          out.units[static_cast<std::size_t>(actor)] = {fastkag::Op::PASS};
          committed_ = true;
          break;
        }
      }
    }
    out.sources.reserve(out.units.size());
    for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
      const bool inserted = static_cast<int>(actor) == changed;
      out.sources.push_back({static_cast<int>(actor),
                             inserted ? -1 : context.step,
                             inserted ? out.units[actor]
                                      : context.raw_g001.units[actor]});
    }
    if (changed >= 0) {
      auto final_joint = context.raw_joint;
      final_joint[context.player].units = out.units;
      out.prefix_authority.push_back(
          {changed, 1,
           g001::repair_fork::unit_prefix_manifest_hash(out.units, changed),
           g001::repair_fork::post_unit_prefix_state_fingerprint(
               context.phase_start, context.player, final_joint, changed)});
    }
    return out;
  }

 private:
  bool committed_{};
};

class TerminalAuthorizedOwner final : public g001::repair_fork::RepairOwner {
 public:
  std::string name() const override { return "terminal_authorized"; }
  g001::repair_fork::RepairDecision decide(
      const g001::repair_fork::RepairContext& context) override {
    g001::repair_fork::RepairDecision out;
    out.units = context.raw_g001.units;
    out.receipt_acks.assign(context.previous_action_receipts.begin(),
                            context.previous_action_receipts.end());
    out.purchase_receipt_acks.assign(
        context.previous_purchase_receipts.begin(),
        context.previous_purchase_receipts.end());
    int changed = -1;
    if (context.step == 718 && !out.units.empty()) {
      changed = 0;
      out.units[0] = context.raw_g001.units[0].op == fastkag::Op::PASS
                         ? fastkag::Action{fastkag::Op::WATER}
                         : fastkag::Action{fastkag::Op::PASS};
      for (std::uint64_t debt = 1; debt <= 11; ++debt)
        out.required_purchases.push_back(
            {debt, fastkag::Op::BUY_SEED, fastkag::Item::WHEAT, 1, 718});
    }
    out.sources.reserve(out.units.size());
    for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
      const bool inserted = static_cast<int>(actor) == changed;
      out.sources.push_back({static_cast<int>(actor),
                             inserted ? -1 : context.step,
                             inserted ? out.units[actor]
                                      : context.raw_g001.units[actor]});
    }
    if (changed >= 0) {
      auto final_joint = context.raw_joint;
      final_joint[context.player].units = out.units;
      out.prefix_authority.push_back(
          {changed, 1,
           g001::repair_fork::unit_prefix_manifest_hash(out.units, changed),
           g001::repair_fork::post_unit_prefix_state_fingerprint(
               context.phase_start, context.player, final_joint, changed)});
    }
    return out;
  }
};

void unauthorized_change_fails_closed() {
  auto local = options();
  local.panels = {g001::repair_fork::Panel::Normal};
  bool rejected = false;
  try {
    (void)g001::repair_fork::evaluate(local, [] {
      return std::make_unique<UnauthorizedOwner>();
    });
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "unit rewrite without PrefixAuthority did not fail closed");
}

void per_actor_authority_round_trip() {
  auto local = options();
  local.panels = {g001::repair_fork::Panel::Normal};
  local.threads = 1;
  const auto report = g001::repair_fork::evaluate(local, [] {
    return std::make_unique<OneShotAuthorizedOwner>();
  });
  for (const auto& game : report.games) {
    require(game.prefix_authority_checks == 1 &&
                game.prefix_authority_failures == 0,
            "valid per-actor PrefixAuthority was not accepted");
    require(game.action_receipts_issued == 1 &&
                game.action_receipts_acked == 1,
            "per-actor action receipt token did not round-trip");
  }
}

void terminal_receipts_are_not_silently_confirmed() {
  auto local = options();
  local.panels = {g001::repair_fork::Panel::Normal};
  local.threads = 1;
  const auto report = g001::repair_fork::evaluate(local, [] {
    return std::make_unique<TerminalAuthorizedOwner>();
  });
  for (const auto& game : report.games) {
    require(game.terminal_unacked_action_receipts == 1,
            "terminal action receipt was silently confirmed");
    require(game.terminal_unacked_purchase_receipts == 11,
            "terminal purchase receipts were silently confirmed");
    require(game.exact_purchase_debts_outstanding > 0,
            "terminal rejected/unfilled purchase debt disappeared");
  }
}

void exact_market_compiler_contract() {
  fastkag::Simulator simulator({}, 7);
  g001::repair_fork::ExactMarketCompiler compiler;
  const std::vector<fastkag::Action> legacy{
      {fastkag::Op::BUY_SEED, fastkag::Item::CARROT, 3}};
  const std::vector<g001::repair_fork::RequiredPurchase> required{
      {1, fastkag::Op::BUY_SEED, fastkag::Item::CARROT, 2, 8},
      {2, fastkag::Op::BUY_ANIMAL, fastkag::Item::GOOSE, 1, 9},
      {3, fastkag::Op::SELL, fastkag::Item::WHEAT, 1, 9}};
  const auto result = compiler.compile(simulator, 0, legacy, required);
  require(result.market.size() == 2, "required purchase append mismatch");
  require(result.bindings.size() == 3, "required purchase binding mismatch");
  require(result.bindings[0].market_slot == 0 &&
              result.bindings[1].market_slot == 1,
          "required purchase slots are not exact");
  require(result.rejected_invalid == 1, "generic market rewrite was admitted");
  require(result.bindings[2].market_slot == -1 &&
              result.bindings[2].status ==
                  g001::repair_fork::PurchaseCompileStatus::RejectedInvalid,
          "rejected purchase lacks an exact compile receipt");
}

}  // namespace

int main(int argc, char** argv) try {
  const bool unit_only = argc == 2 && std::string_view(argv[1]) == "--unit-only";
  sha256_known_vector();
  full_unit_phase_fingerprint_covers_hidden_causal_state();
  exact_market_compiler_contract();
  if (unit_only) {
    std::cout << "repair_fork_evaluator_unit_tests: PASS\n";
    return 0;
  }
  exact_pass_through_and_determinism();
  unauthorized_change_fails_closed();
  per_actor_authority_round_trip();
  terminal_receipts_are_not_silently_confirmed();
  std::cout << "repair_fork_evaluator_tests: PASS\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "repair_fork_evaluator_tests: FAIL: " << error.what() << '\n';
  return 1;
}
