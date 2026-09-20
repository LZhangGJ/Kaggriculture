#include "g001_day_start_obligation_issuer.hpp"
#include "native_final_action_commit.hpp"
#include "native_general_market.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool same(Action left, Action right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool same_units(const std::vector<Action>& left,
                const std::vector<Action>& right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin(), same);
}

fastkag::NativeTapeLibrary real_library() {
  fastkag::NativeTapeLibrary library;
  library.routes.push_back(g001::repair::load_route(
      ANIMAL_RECOVERY_TAPES, ANIMAL_RECOVERY_LIBRARY, "G001"));
  library.routes.push_back(g001::repair::load_route(
      ANIMAL_RECOVERY_TAPES, ANIMAL_RECOVERY_LIBRARY, "G096"));
  library.r5_reference = g001::repair::load_route(
      ANIMAL_RECOVERY_REFERENCES, ANIMAL_RECOVERY_LIBRARY, "R5");
  library.md_reference = g001::repair::load_route(
      ANIMAL_RECOVERY_REFERENCES, ANIMAL_RECOVERY_LIBRARY, "MD");
  constexpr std::array<const char*, 5> labels{
      "10C4S_3Q", "8C6S_3Q", "6C8S_3Q", "6C12S_4Q_FIRST_YARN",
      "6C12S_4Q_SECOND_YARN"};
  for (std::size_t index = 0; index < labels.size(); ++index) {
    library.moon[index] = g001::repair::load_route(
        ANIMAL_RECOVERY_REFERENCES, ANIMAL_RECOVERY_LIBRARY,
        "MOON_" + std::string(labels[index]));
    library.moon_legacy[index] = g001::repair::load_route(
        ANIMAL_RECOVERY_REFERENCES, ANIMAL_RECOVERY_LIBRARY,
        "MOON_LEGACY_" + std::string(labels[index]));
  }
  return library;
}

g001::day_start_issuer::FinalMarketReceipt receipt_for(
    const fastkag::Simulator& after, int player, int submitted_step,
    const PlayerAction& submitted) {
  const auto& fills = after.last_market_fills()[player];
  return {player, submitted_step, submitted.market,
          std::vector<std::int32_t>(fills.begin(), fills.end())};
}

std::uint64_t generation(std::uint64_t seed, int step) {
  return (seed << 20U) ^ (2ULL << 16U) ^
         static_cast<std::uint64_t>(step + 1);
}

void run() {
  namespace binder_ns = g001::day_start_issuer;
  namespace commit = g001::native_final_commit;

  constexpr std::uint64_t seed = 990034;
  constexpr int player = 1;
  constexpr int actor = 1;
  auto library = real_library();
  fastkag::NativeTeammateExecutor executor(std::move(library));
  fastkag::Simulator env({}, seed);
  std::array<fastkag::NativeAgentState, 2> states;
  binder_ns::CrossTickAnimalIntentBinder binder;
  PlayerAction failed_submission;

  while (env.step_count() <= 96) {
    std::array<PlayerAction, 2> actions;
    actions[player] = executor.action_external(env, player, 0, states[player]);
    actions[0] = executor.action_external(env, 0, 1, states[0]);
    if (env.step_count() == 96) {
      failed_submission = actions[player];
      const auto staged = binder.stage_final_submission(
          env, player, failed_submission, executor.route_tape(0),
          generation(seed, 96));
      require(staged.status == binder_ns::CrossTickAnimalStatus::Staged &&
                  binder.pending(),
              "real step96 animal intent did not stage");
    }
    env.step(actions);
  }

  const auto bound = binder.bind_next(
      env, receipt_for(env, player, 96, failed_submission));
  require(bound.status == binder_ns::CrossTickAnimalStatus::Bound &&
              bound.obligation && !binder.pending(),
          "real failed BUY receipt did not bind exactly once");
  const auto obligation = *bound.obligation;
  require(obligation.actor == actor && obligation.animal == Item::COW &&
              obligation.place_step == 106 && obligation.target.row == 1 &&
              obligation.target.column == 2 &&
              obligation.acquisition_fill == 0 &&
              env.last_market_fills()[player] ==
                  std::vector<std::int32_t>({1, 1, 1, 0}),
          "real step96 fill/actor/PLACE binding changed");

  const auto& tape = executor.route_tape(0);
  require(env.step_count() == 97 && env.farms()[player].money == 369.0 &&
              env.privates()[player].shed[static_cast<int>(Item::WHEAT)] == 4 &&
              env.privates()[player]
                      .shed[static_cast<int>(Item::FERTILIZER)] == 1 &&
              tape[97].units[actor].op == Op::PICKUP &&
              tape[97].units[actor].item == Item::COW &&
              tape[98].units[actor].op == Op::PICKUP &&
              tape[98].units[actor].item == Item::WHEAT,
          "real two-hand recovery precondition changed");
  const auto target_index = static_cast<std::size_t>(
      obligation.target.row * env.config().board_size +
      obligation.target.column);
  require(env.farms()[player].tiles[target_index].kind ==
              fastkag::TileKind::EMPTY,
          "target must remain a real EMPTY tile before BUILD_PASTURE");
  // ponytail: keep this default-off slice stateless; add a persistent owner
  // only when it accepts receipt-proven future BUILD_PASTURE provenance.

  int native_commits = 0;
  std::array<PlayerAction, 2> actions;
  const auto state_before_97 = states[player];
  actions[player] = executor.action_external(env, player, 0, states[player]);
  actions[0] = executor.action_external(env, 0, 1, states[0]);
  require(same_units(actions[player].units, tape[97].units) &&
              actions[player].market.empty(),
          "step97 final native provider drifted from the immutable route");

  auto certified_units = actions[player].units;
  certified_units[actor] = {Op::PICKUP, Item::WHEAT, 1};
  std::vector<fastkag::NativeFutureUnitFrame> certified_future;
  for (int step = 98; step <= 108; ++step) {
    auto units = tape[static_cast<std::size_t>(step)].units;
    if (step == 98) units[actor] = {Op::PICKUP, Item::COW, 1};
    certified_future.push_back({step, std::move(units)});
  }
  const auto compiled = fastkag::compile_native_general_market(
      env, player, certified_units, certified_future);
  require(compiled.feasible && compiled.audit.compiler_feasible &&
              compiled.audit.allocator_feasible &&
              compiled.audit.first_due_kind == "BUY_ANIMAL" &&
              compiled.audit.first_due_item == static_cast<int>(Item::COW) &&
              compiled.audit.funding_sell_orders == 1 &&
              compiled.audit.funding_sell_units == 1 &&
              compiled.current_unit_replacements.empty() &&
              compiled.market.size() == 2 &&
              same(compiled.market[0],
                   {Op::SELL, Item::FERTILIZER, 1}) &&
              same(compiled.market[1],
                   {Op::BUY_ANIMAL, Item::COW, 1}),
          "certified swapped plan compiler mismatch feasible=" +
              std::to_string(compiled.feasible) + " compiler=" +
              std::to_string(compiled.audit.compiler_feasible) +
              " allocator=" +
              std::to_string(compiled.audit.allocator_feasible) +
              " due=" + compiled.audit.first_due_kind + "/" +
              std::to_string(compiled.audit.first_due_item) + " sells=" +
              std::to_string(compiled.audit.funding_sell_orders) + "/" +
              std::to_string(compiled.audit.funding_sell_units) +
              " market=" + std::to_string(compiled.market.size()) +
              " diagnostic=" + compiled.audit.diagnostic_code + ":" +
              compiled.audit.reason);

  auto selected_97 = actions[player];
  selected_97.units = std::move(certified_units);
  selected_97.market = compiled.market;
  commit::Request request_97{&executor, &env, player, 0, {}, {}};
  const auto proposal_97 = commit::propose(request_97, state_before_97);
  require(commit::same_action(proposal_97.action, actions[player]),
          "step97 exact native proposal drifted");
  const auto binding_97 = commit::bind_repair_final_action(
      proposal_97, 1ULL << actor, true, generation(seed, 97), obligation.id,
      selected_97);
  auto committed_state_97 = state_before_97;
  const auto committed_97 = commit::commit_repair_owner_finalized(
      request_97, proposal_97, binding_97, selected_97, committed_state_97);
  require(committed_97.committed &&
              commit::same_action(committed_97.replayed_action, selected_97),
          "step97 two-hand recovery failed native exact commit");
  states[player] = std::move(committed_state_97);
  const auto before_97 = env;
  actions[player] = selected_97;
  env.step(actions);
  ++native_commits;
  require(env.last_market_fills()[player] ==
              std::vector<std::int32_t>({1, 1}) &&
              env.privates()[player].shed[static_cast<int>(Item::COW)] == 1 &&
              env.privates()[player]
                      .inventories[actor][static_cast<int>(Item::WHEAT)] ==
                  before_97.privates()[player]
                          .inventories[actor][static_cast<int>(Item::WHEAT)] +
                      1,
          "step97 funding sale/BUY_COW/PICKUP_WHEAT receipts did not close");

  const auto state_before_98 = states[player];
  actions = {};
  actions[player] = executor.action_external(env, player, 0, states[player]);
  actions[0] = executor.action_external(env, 0, 1, states[0]);
  require(same_units(actions[player].units, tape[98].units) &&
              actions[player].units[actor].op == Op::PICKUP &&
              actions[player].units[actor].item == Item::WHEAT,
          "step98 native PICKUP_WHEAT source changed");
  auto selected_98 = actions[player];
  selected_98.units[actor] = {Op::PICKUP, Item::COW, 1};
  commit::Request request_98{&executor, &env, player, 0, {}, {}};
  const auto proposal_98 = commit::propose(request_98, state_before_98);
  require(commit::same_action(proposal_98.action, actions[player]),
          "step98 exact native proposal drifted");
  const auto binding_98 = commit::bind_repair_final_action(
      proposal_98, 1ULL << actor, false, generation(seed, 98), obligation.id,
      selected_98);
  auto committed_state_98 = state_before_98;
  const auto committed_98 = commit::commit_repair_owner_finalized(
      request_98, proposal_98, binding_98, selected_98, committed_state_98);
  require(committed_98.committed &&
              commit::same_action(committed_98.replayed_action, selected_98),
          "step98 PICKUP_COW failed native exact commit");
  states[player] = std::move(committed_state_98);
  const int cow_before_98 = env.privates()[player]
                               .inventories[actor][static_cast<int>(Item::COW)];
  actions[player] = selected_98;
  env.step(actions);
  ++native_commits;
  require(env.privates()[player]
                  .inventories[actor][static_cast<int>(Item::COW)] ==
              cow_before_98 + 1 &&
              env.privates()[player].shed[static_cast<int>(Item::COW)] == 0,
          "step98 real PICKUP_COW effect missing");

  std::vector<Op> observed_moves;
  std::vector<Op> expected_moves;
  bool pasture_built = false;
  bool cow_placed = false;
  bool cow_fed = false;
  bool cow_cared = false;
  bool wheat_seed_fill = false;
  while (env.step_count() <= 108) {
    const int step = env.step_count();
    actions = {};
    actions[player] =
        executor.action_external(env, player, 0, states[player]);
    actions[0] = executor.action_external(env, 0, 1, states[0]);
    require(same_units(actions[player].units,
                       tape[static_cast<std::size_t>(step)].units),
            "step" + std::to_string(step) +
                " changed a unit byte after the two-hand swap");
    const auto raw = tape[static_cast<std::size_t>(step)].units[actor];
    if (raw.op == Op::NORTH || raw.op == Op::SOUTH || raw.op == Op::EAST ||
        raw.op == Op::WEST) {
      expected_moves.push_back(raw.op);
      observed_moves.push_back(actions[player].units[actor].op);
    }
    env.step(actions);
    const auto& target = env.farms()[player].tiles[target_index];
    if (step == 105)
      pasture_built = target.kind == fastkag::TileKind::PASTURE;
    if (step == 106) {
      cow_placed = target.kind == fastkag::TileKind::ANIMAL &&
                   target.animal == Item::COW;
      for (std::size_t slot = 0; slot < actions[player].market.size(); ++slot)
        if (same(actions[player].market[slot],
                 {Op::BUY_SEED, Item::WHEAT, 5}) &&
            slot < env.last_market_fills()[player].size() &&
            env.last_market_fills()[player][slot] == 5)
          wheat_seed_fill = true;
    }
    if (step == 107)
      cow_fed = target.kind == fastkag::TileKind::ANIMAL &&
                target.animal == Item::COW && target.fed_today;
    if (step == 108)
      cow_cared = target.kind == fastkag::TileKind::ANIMAL &&
                  target.animal == Item::COW && target.cared_today;
  }

  const std::vector<Op> frozen_moves{Op::WEST, Op::NORTH, Op::WEST,
                                     Op::NORTH, Op::WEST, Op::NORTH};
  require(observed_moves == expected_moves && expected_moves == frozen_moves &&
              pasture_built && cow_placed && cow_fed && cow_cared &&
              wheat_seed_fill && native_commits == 2,
          "two-hand recovery did not preserve MOVE order or close the real "
          "BUILD/PLACE/FEED/CARE chain");

  while (!env.done()) {
    actions = {};
    actions[player] =
        executor.action_external(env, player, 0, states[player]);
    actions[0] = executor.action_external(env, 0, 1, states[0]);
    env.step(actions);
  }
  const double repair_own = env.farms()[player].money;
  const double repair_margin =
      repair_own - env.farms()[1 - player].money;

  fastkag::NativeTeammateExecutor baseline_executor(real_library());
  fastkag::Simulator baseline({}, seed);
  std::array<fastkag::NativeAgentState, 2> baseline_states;
  while (!baseline.done()) {
    actions = {};
    actions[player] = baseline_executor.action_external(
        baseline, player, 0, baseline_states[player]);
    actions[0] = baseline_executor.action_external(
        baseline, 0, 1, baseline_states[0]);
    baseline.step(actions);
  }
  const double baseline_own = baseline.farms()[player].money;
  const double baseline_margin =
      baseline_own - baseline.farms()[1 - player].money;

  std::cout << "repair_enabled_two_hand_animal_recovery_smoke: PASS "
               "default_off=1 failed_buy=0 recovered_buy=1 "
               "swap=97:PICKUP_WHEAT,98:PICKUP_COW "
               "moves=W,N,W,N,W,N place=106 feed=107 care=108 "
               "native_commits="
            << native_commits << " terminal_own=" << baseline_own << "->"
            << repair_own << " delta=" << repair_own - baseline_own
            << " terminal_margin=" << baseline_margin << "->"
            << repair_margin << " delta=" << repair_margin - baseline_margin
            << '\n';
}

}  // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "repair_enabled_two_hand_animal_recovery_smoke: "
              << error.what() << '\n';
    return 1;
  }
}
