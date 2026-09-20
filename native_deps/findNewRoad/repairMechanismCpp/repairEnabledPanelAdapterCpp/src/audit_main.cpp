#include "repair_enabled_panel_adapter.hpp"

#include "g001_day_start_obligation_issuer.hpp"
#include "minimum_damage_remaining_day_audit_adapter.hpp"
#include "native_final_action_commit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool same(fastkag::Action left, fastkag::Action right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

void record_lineage(
    const fastkag::Simulator& observation,
    const fastkag::PlayerAction& action,
    g001::day_start_issuer::PersistentRouteIntentRegistry& lineage) {
  std::vector<fastkag::Position> positions{observation.farms()[0].farmer};
  positions.insert(positions.end(), observation.farms()[0].hands.begin(),
                   observation.farms()[0].hands.end());
  for (std::size_t actor = 0;
       actor < positions.size() && actor < action.units.size(); ++actor)
    (void)lineage.record_exact_source(static_cast<int>(actor),
                                      observation.step_count(),
                                      positions[actor], action.units[actor]);
}

void successful_buy_seed_crop_reissue_smoke() {
  namespace adapter = g001::minimum_damage_remaining_audit;
  namespace bridge = g001::minimum_damage_bridge;
  namespace commit = g001::native_final_commit;
  namespace issuer = g001::day_start_issuer;
  using fastkag::Action;
  using fastkag::Item;
  using fastkag::Op;
  using fastkag::PlayerAction;

  constexpr std::uint64_t seed = 990034;
  constexpr std::uint64_t generation = (seed << 20U) | (2ULL << 16U) | 2ULL;
  fastkag::Config config;
  config.episode_steps = 72;
  config.starting_money = 100'000;
  config.weed_spawn_chance = 0.0;
  std::vector<PlayerAction> tape(static_cast<std::size_t>(config.episode_steps));
  for (auto& action : tape) action.units.resize(1);
  tape[0].units[0] = {Op::PLANT, Item::TOMATO, 1};
  tape[1].units[0] = {Op::WATER, Item::NONE, 1};
  tape[24].units[0] = {Op::WATER, Item::NONE, 1};
  tape[24].market.push_back({Op::BUY_SEED, Item::TOMATO, 1});
  tape[25].units[0] = {Op::EAST, Item::NONE, 1};
  tape[26].units[0] = {Op::WEST, Item::NONE, 1};

  fastkag::NativeTapeLibrary library;
  library.routes.push_back(tape);
  fastkag::NativeTeammateExecutor executor(std::move(library));
  fastkag::Simulator env(config, seed);
  auto& private_states = const_cast<std::array<fastkag::PrivateState, 2>&>(
      env.privates());
  private_states[0].seeds[static_cast<std::size_t>(Item::TOMATO)] = 1;
  fastkag::NativeAgentState state;
  issuer::PersistentRouteIntentRegistry lineage;
  PlayerAction opponent;
  opponent.units.resize(1);

  while (env.step_count() < 24) {
    auto provider = executor.action_external(env, 0, 0, state);
    record_lineage(env, provider, lineage);
    env.step(std::array<PlayerAction, 2>{provider, opponent});
  }
  const auto target = env.farms()[0].farmer;
  const auto target_index = static_cast<std::size_t>(
      target.y * env.config().board_size + target.x);
  auto& farms = const_cast<std::array<fastkag::Farm, 2>&>(env.farms());
  if (farms[0].tiles[target_index].kind != fastkag::TileKind::PLANT ||
      farms[0].tiles[target_index].crop != Item::TOMATO ||
      env.privates()[0].seeds[static_cast<std::size_t>(Item::TOMATO)] != 0)
    throw std::runtime_error("crop reissue fixture lacks past planted crop");
  farms[0].tiles[target_index] = {};
  farms[0].tiles[target_index].kind = fastkag::TileKind::WEED;

  const auto issued = issuer::issue_day_start(
      {&env, &executor.route_tape(0), 0, generation, &lineage, {}, {}});
  if (!issued.fully_proven())
    throw std::runtime_error("crop reissue issuer did not prove the day");
  const auto crop = std::find_if(
      issued.obligations.begin(), issued.obligations.end(),
      [&](const auto& obligation) {
        return obligation.actor == 0 &&
               obligation.goal ==
                   g001::obligation_day::GoalKind::CropReady &&
               obligation.item == Item::TOMATO &&
               obligation.tile.x == target.x && obligation.tile.y == target.y;
      });
  if (crop == issued.obligations.end())
    throw std::runtime_error("crop reissue lacks typed crop obligation");
  std::vector<bridge::ObligationPolicy> policies;
  for (const auto& obligation : issued.obligations)
    if (obligation.actor == 0)
      policies.push_back({obligation.id, true, 100});
  adapter::Adapter rolling(issued, 0, generation, std::move(policies));

  std::vector<Op> production;
  std::vector<int> production_steps;
  std::vector<Op> moves;
  int native_commits = 0;
  int buy_fill = 0;
  int seed_after_buy = 0;
  bool crop_closed = false;
  const double money_before_buy = env.farms()[0].money;
  while (env.step_count() < 48) {
    const auto before = env;
    const auto provider_state_before = state;
    const auto provider = executor.action_external(env, 0, 0, state);
    const auto hand = rolling.plan(env);
    if (!hand.valid)
      throw std::runtime_error("crop reissue rejected hand " +
                               std::to_string(env.step_count()) + ":" +
                               hand.diagnostic);
    if (hand.candidate_market)
      throw std::runtime_error("crop reissue unexpectedly replaced market");

    auto selected = provider;
    selected.units[0] = hand.candidate_unit;
    if (before.step_count() == 24 &&
        (before.farms()[0].money < 50.0 || selected.market.size() != 1 ||
         selected.market[0].op != Op::BUY_SEED ||
         selected.market[0].item != Item::TOMATO ||
         selected.market[0].quantity != 1))
      throw std::runtime_error(
          "crop reissue lacks affordable exact BUY_SEED submission cash=" +
          std::to_string(before.farms()[0].money) + " market=" +
          std::to_string(selected.market.size()) + " op=" +
          std::to_string(selected.market.empty()
                             ? -1
                             : static_cast<int>(selected.market[0].op)));
    fastkag::NativeRepairOptions repair;
    repair.weed_obligation_day_owner = true;
    commit::Request request{&executor, &env, 0, 0, repair, std::nullopt};
    const auto native = commit::propose(request, provider_state_before);
    if (native.action.market.size() != provider.market.size() ||
        !std::equal(native.action.market.begin(), native.action.market.end(),
                    provider.market.begin(), same))
      throw std::runtime_error("crop reissue changed native market prefix");
    const auto binding = commit::bind_repair_final_action(
        native, 1ULL, false, generation, hand.certificate_hash, selected);
    auto committed_state = provider_state_before;
    const auto committed = commit::commit_repair_owner_finalized(
        request, native, binding, selected, committed_state);
    if (!committed.committed ||
        !commit::same_action(committed.replayed_action, selected))
      throw std::runtime_error("crop reissue native exact commit failed");

    env.step(std::array<PlayerAction, 2>{selected, opponent});
    state = std::move(committed_state);
    if (!rolling.observe_final(before, hand.candidate_unit, env))
      throw std::runtime_error("crop reissue real receipt was rejected");
    ++native_commits;
    if (hand.candidate_unit.op == Op::DIG ||
        hand.candidate_unit.op == Op::PLANT ||
        hand.candidate_unit.op == Op::WATER) {
      production.push_back(hand.candidate_unit.op);
      production_steps.push_back(before.step_count());
    }
    if (hand.candidate_unit.op == Op::EAST ||
        hand.candidate_unit.op == Op::WEST)
      moves.push_back(hand.candidate_unit.op);
    if (before.step_count() == 24) {
      const auto& fills = env.last_market_fills()[0];
      buy_fill = fills.empty() ? 0 : fills[0];
      seed_after_buy = env.privates()[0].seeds[
          static_cast<std::size_t>(Item::TOMATO)];
    }
    const auto& tile = env.farms()[0].tiles[target_index];
    crop_closed = crop_closed ||
                  (tile.kind == fastkag::TileKind::PLANT &&
                   tile.crop == Item::TOMATO && tile.watered_today);
  }

  const auto report = rolling.finish();
  const std::vector<Op> expected_production{Op::DIG, Op::PLANT, Op::WATER};
  const std::vector<Op> expected_moves{Op::EAST, Op::WEST};
  if (buy_fill != 1 || seed_after_buy != 1 ||
      env.farms()[0].money >= money_before_buy ||
      production != expected_production || production_steps.back() >= 47 ||
      moves != expected_moves ||
      !crop_closed || native_commits != 24 || report.hands != 24 ||
      report.initial_move_tokens != 2 ||
      report.observed_move_receipts != 2 ||
      report.duplicate_move_receipts != 0 ||
      report.failed_move_receipts != 0 ||
      report.move_early_violations != 0 ||
      report.outstanding_obligations != 0 ||
      !report.terminal_debts.empty())
    throw std::runtime_error(
        "successful BUY_SEED crop reissue gate failed fill=" +
        std::to_string(buy_fill) + " production=" +
        std::to_string(production.size()) + "@" +
        (production_steps.empty()
             ? std::string("-")
             : std::to_string(production_steps.front())) + "," +
        (production_steps.size() < 2
             ? std::string("-")
             : std::to_string(production_steps[1])) + "," +
        (production_steps.size() < 3
             ? std::string("-")
             : std::to_string(production_steps[2])) + " moves=" +
        std::to_string(moves.size()) + " closed=" +
        std::to_string(crop_closed) + " commits=" +
        std::to_string(native_commits) + " hands=" +
        std::to_string(report.hands) + " move_receipts=" +
        std::to_string(report.observed_move_receipts) + '/' +
        std::to_string(report.initial_move_tokens) + " outstanding=" +
        std::to_string(report.outstanding_obligations) + " debts=" +
        std::to_string(report.terminal_debts.size()));
  std::cout << "repair_enabled_panel_adapter_audit: SUCCESSFUL BUY CROP "
               "PASS fill="
            << buy_fill << " ops=DIG,PLANT,WATER moves=EAST,WEST commits="
            << native_commits << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    fastkag::Tile empty;
    empty.kind = fastkag::TileKind::EMPTY;
    const fastkag::Action plant{fastkag::Op::PLANT,
                                fastkag::Item::STRAWBERRY, 1};
    if (!g001::repair_enabled_panel_adapter::detail::disrupted_crop_source(
            plant, fastkag::Item::STRAWBERRY, empty, 10, 11) ||
        g001::repair_enabled_panel_adapter::detail::disrupted_crop_source(
            plant, fastkag::Item::STRAWBERRY, empty, 10, 10))
      throw std::runtime_error("failed-PLANT causal trigger changed");
    fastkag::Tile weed;
    weed.kind = fastkag::TileKind::WEED;
    const fastkag::Action water{fastkag::Op::WATER,
                                fastkag::Item::STRAWBERRY, 1};
    if (g001::repair_enabled_panel_adapter::detail::
            crop_recovery_fits_pass_capacity(
                water, fastkag::Item::STRAWBERRY, weed, 227, 216, 1) ||
        !g001::repair_enabled_panel_adapter::detail::
            crop_recovery_fits_pass_capacity(
                water, fastkag::Item::STRAWBERRY, weed, 227, 216, 2) ||
        g001::repair_enabled_panel_adapter::detail::
            crop_recovery_fits_pass_capacity(
                water, fastkag::Item::STRAWBERRY, weed, 215, 216, 2))
      throw std::runtime_error("crop recovery PASS-capacity gate changed");
    if (argc > 1 &&
        std::string(argv[1]) == "--transactional-crop-smoke") {
      const auto smoke =
          g001::repair_enabled_panel_adapter::
              evaluate_transactional_crop_smoke();
      if (!smoke.passed())
        throw std::runtime_error("transactional crop smoke gate failed");
      std::cout << "repair_enabled_panel_adapter_audit: TRANSACTIONAL CROP "
                   "PASS ops=DIG,PLANT,WATER moves="
                << smoke.moves_emitted << '/' << smoke.moves_expected
                << " action_receipts=" << smoke.action_receipts_acked
                << " purchase_receipts=" << smoke.purchase_receipts_acked
                << " debt_continuations=" << smoke.debt_continuations
                << " native_rejects=" << smoke.native_rejections << '\n';
      return 0;
    }
    if (argc > 1 &&
        std::string(argv[1]) == "--successful-buy-crop-reissue-smoke") {
      successful_buy_seed_crop_reissue_smoke();
      return 0;
    }
    if (argc == 7 && (std::string(argv[1]) == "--crop-at" ||
                      std::string(argv[1]) == "--crop-empty-at")) {
      const auto report =
          g001::repair_enabled_panel_adapter::evaluate_forced_crop_at(
              REPAIR_ENABLED_TAPES, REPAIR_ENABLED_LIBRARY,
              std::stoull(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]),
              std::stoi(argv[6]), std::string(argv[1]) == "--crop-at");
      std::ofstream file(argv[2], std::ios::trunc);
      if (!file) throw std::runtime_error("cannot create crop-grid artifact");
      file << report.json() << '\n';
      return 0;
    }
    const bool versus_g096 =
        argc > 1 && std::string(argv[1]) == "--g096";
    const bool sweep = argc > 1 && std::string(argv[1]) == "--sweep";
    const bool artifact_mode = sweep || versus_g096;
    const std::uint64_t seed = artifact_mode && argc > 3
        ? std::stoull(argv[3])
        : argc > 2 ? std::stoull(argv[2]) : 970017;
    if (argc > 1 && std::string(argv[1]) == "--focused") {
      const auto game = g001::repair_enabled_panel_adapter::evaluate_focused(
          REPAIR_ENABLED_TAPES, REPAIR_ENABLED_LIBRARY, seed);
      if (game.steps != 719 || game.causal_activations != 1 ||
          game.trigger_source_step != 183 || game.commits != 24 ||
          game.provider_state_commits != 24 ||
          game.provider_state_leaks != 0 || game.receipts_accepted != 24 ||
          game.receipts_failed != 0 || game.moves_expected != 7 ||
          game.moves_emitted != 7 || game.move_early != 0 ||
          game.move_duplicates != 0 || game.move_drops != 0 ||
          game.unowned_unit_failures != 0 ||
          game.market_exact_failures != 0 ||
          !game.trigger_effect_observed)
        throw std::runtime_error("focused sanitizer gate failed");
      std::cout << "repair_enabled_panel_adapter_audit: FOCUSED PASS "
                   "commits=24 receipts=24 moves=7/7\n";
      return 0;
    }
    const std::string output = artifact_mode ? argv[2] : argc > 1 ? argv[1]
        : "repair-enabled-panel-adapter-seed970017.json";
    const auto report = versus_g096
        ? g001::repair_enabled_panel_adapter::evaluate_vs_g096(
              REPAIR_ENABLED_TAPES, REPAIR_ENABLED_LIBRARY, seed)
        : g001::repair_enabled_panel_adapter::evaluate(
              REPAIR_ENABLED_TAPES, REPAIR_ENABLED_LIBRARY, seed);
    if (!report.disabled_parity || !report.fail_stop_probe ||
        report.games.size() != 6) {
      std::cerr << report.json() << '\n';
      throw std::runtime_error("disabled parity or smoke matrix changed");
    }
    int total_commits = 0;
    for (const auto& game : report.games) {
      const int expected_provider_commits =
          game.commits + game.purchase_retries -
          game.joint_weed_purchase_commits;
      if (game.steps != 719 ||
          game.provider_state_commits != expected_provider_commits ||
          game.provider_state_leaks != 0 ||
          game.unowned_unit_failures != 0 ||
          game.market_exact_failures != 0 ||
          game.receipts_failed != 0 ||
          game.receipts_accepted != game.commits ||
          game.move_failures != 0 ||
          game.move_early != 0 || game.move_duplicates != 0 ||
          game.move_drops != 0 ||
          game.moves_expected != game.moves_emitted ||
          game.purchase_finalize_failures != 0 ||
          game.purchase_move_mismatches != 0) {
        if (artifact_mode) std::cerr << report.json() << '\n';
        throw std::runtime_error("transaction/receipt/MOVE gate failed");
      }
      if (game.first_legacy_trade_exposure_step >= 0 &&
          (game.first_legacy_trade_exposure_product < 0 ||
           game.first_legacy_trade_exposure_product >= fastkag::N_PRODUCTS ||
           game.first_legacy_trade_requested <= 0 ||
           game.first_legacy_trade_incremental_fill == 0 ||
           game.first_legacy_trade_incremental_fill !=
               std::min(game.first_legacy_trade_requested,
                        game.first_legacy_trade_repair_shed) -
                   std::min(game.first_legacy_trade_requested,
                            game.first_legacy_trade_baseline_shed) ||
           !std::isfinite(game.first_legacy_trade_baseline_asset_value) ||
           !std::isfinite(game.first_legacy_trade_repair_asset_value)))
        throw std::runtime_error("legacy trade exposure audit is inconsistent");
      if (game.causal_activations == 0) {
        if (game.purchase_retries == 0 &&
            (game.attempts != 0 || game.action_divergences != 0 ||
             game.terminal_debts != 0))
          throw std::runtime_error("non-causal takeover detected");
      } else if (game.scenario ==
                     g001::repair_whole_game_panel::Scenario::ForcedWeed &&
                 (game.causal_activations != 1 ||
                 game.trigger_source_step != 183 || game.trigger_x != 6 ||
                 game.trigger_y != 1 || !game.trigger_observed_weed ||
                 !game.trigger_effect_observed || game.commits != 24 ||
                 game.moves_expected != 7 || game.moves_emitted != 7)) {
        std::cerr << report.json() << '\n';
        throw std::runtime_error("typed weed trigger/effect gate failed");
      }
      if (game.scenario ==
              g001::repair_whole_game_panel::Scenario::ForcedWeed &&
          (game.forced_events != 1 || game.forced_baseline_applied != 1 ||
           game.forced_repair_applied != 1 ||
           game.causal_activations != 1))
        throw std::runtime_error("frozen forced-weed schedule changed");
      if (game.scenario ==
              g001::repair_whole_game_panel::Scenario::ForcedCropWeed &&
          (game.forced_events != 1 || game.forced_baseline_applied != 1 ||
           game.forced_repair_applied != 1 ||
           game.crop_repair_activations == 0 ||
           !game.trigger_observed_weed || game.terminal_debts != 0 ||
           (!game.trigger_effect_observed &&
            (game.baseline_terminal.own != game.repair_terminal.own ||
             game.baseline_terminal.margin != game.repair_terminal.margin)))) {
        std::cerr << report.json() << '\n';
        throw std::runtime_error("forced crop-repair gate failed");
      }
      total_commits += game.commits;
    }
    if (total_commits == 0)
      throw std::runtime_error("repair-enabled smoke produced zero commits");
    std::ofstream file(output, std::ios::trunc);
    if (!file) throw std::runtime_error("cannot create adapter artifact");
    file << report.json() << '\n';
    std::cout << "repair_enabled_panel_adapter_audit: PASS games=6 commits="
              << total_commits << " output=" << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "repair_enabled_panel_adapter_audit: " << error.what()
              << '\n';
    return 1;
  }
}
