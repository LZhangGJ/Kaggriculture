#include "g001_day_start_obligation_issuer.hpp"

#include "native_teammate.hpp"
#include "route_loader.hpp"

#include <array>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace issuer = g001::day_start_issuer;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<PlayerAction> tape_with(Action action, int step = 0) {
  std::vector<PlayerAction> tape(48);
  for (auto& turn : tape) {
    turn.units.push_back({});
  }
  tape[step].units[0] = action;
  return tape;
}

issuer::IssueRequest request_for(const Simulator& simulator,
                                 const std::vector<PlayerAction>& tape,
                                 std::uint64_t generation) {
  issuer::IssueRequest request;
  request.day_start = &simulator;
  request.route_tape = &tape;
  request.player = 0;
  request.issuer_generation = generation;
  return request;
}

void persistent_crop_lineage_survives_weed() {
  Simulator simulator({}, 91);
  const Position tile = simulator.farms()[0].farmer;
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  farm.tiles[tile.y * simulator.config().board_size + tile.x].kind =
      TileKind::WEED;
  auto tape = tape_with({Op::WATER, Item::NONE, 1});
  issuer::PersistentRouteIntentRegistry registry;
  check(registry.record_exact_source(
            0, 7, tile, {Op::PLANT, Item::STRAWBERRY, 1}),
        "exact prior PLANT was not recorded");
  auto request = request_for(simulator, tape, 100);
  request.persistent_intents = &registry;
  const auto result = issuer::issue_day_start(request);
  check(result.issued() && result.unsupported.empty() &&
            result.crop_owner_obligations.size() == 1 &&
            result.crop_owner_obligations[0].desired == Item::STRAWBERRY,
        "weed erased persistent crop lineage");

  auto& observed = farm.tiles[tile.y * simulator.config().board_size + tile.x];
  observed.kind = TileKind::PLANT;
  observed.crop = Item::MELON;
  const auto reconciled = issuer::issue_day_start(request);
  check(reconciled.crop_owner_obligations.size() == 1 &&
            reconciled.crop_owner_obligations[0].desired == Item::MELON,
        "newer public crop did not reconcile stale registry lineage");
}

void itemless_crop_without_lineage_fails_closed() {
  Simulator simulator({}, 92);
  const Position tile = simulator.farms()[0].farmer;
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  farm.tiles[tile.y * simulator.config().board_size + tile.x].kind =
      TileKind::WEED;
  const auto tape = tape_with({Op::WATER, Item::NONE, 1});
  const auto result =
      issuer::issue_day_start(request_for(simulator, tape, 101));
  check(result.issued() && result.obligations.empty() &&
            result.unsupported.size() == 1 &&
            result.unsupported[0].reason ==
                issuer::UnsupportedReason::MissingCropIdentity,
        "item-less WATER guessed a crop identity");
}

void exact_action_shapes_emit_typed_abis() {
  Simulator simulator({}, 93);
  const Position start = simulator.farms()[0].farmer;
  std::vector<PlayerAction> tape(48);
  for (auto& turn : tape) {
    turn.units.push_back({});
  }
  tape[0].units[0] = {Op::PLANT, Item::MELON, 1};
  tape[1].units[0] = {Op::WATER, Item::NONE, 1};
  tape[2].units[0] = {Op::EAST, Item::NONE, 1};
  tape[3].units[0] = {Op::PICKUP, Item::GOOSE, 1};
  tape[4].units[0] = {Op::PLACE, Item::GOOSE, 1};
  const auto result =
      issuer::issue_day_start(request_for(simulator, tape, 102));
  check(result.issued() && result.unsupported.empty(),
        "exact action shapes unexpectedly failed closed");
  check(result.moves.size() == 1 && result.moves[0].source_step == 2 &&
            result.moves[0].action.op == Op::EAST,
        "immutable MOVE token changed");
  check(result.crop_owner_obligations.size() == 2 &&
            result.crop_owner_obligations[1].desired == Item::MELON,
        "item-less WATER did not bind same-day exact PLANT");
  check(result.animal_owner_obligations.size() == 1 &&
            result.animal_owner_obligations[0].target.column == start.x + 1 &&
            result.animal_owner_obligations[0].target.row == start.y &&
            result.animal_owner_obligations[0].animal == Item::GOOSE,
        "PLACE target/item ABI lost route position");
}

void mismatched_authority_fails_closed() {
  Simulator simulator({}, 94);
  const Position tile = simulator.farms()[0].farmer;
  const auto tape = tape_with({Op::WATER, Item::NONE, 1});
  g001::transactional_crop_repair::TypedCropObligation binding;
  binding.id = 1;
  binding.player = 0;
  binding.actor = 0;
  binding.source_step = 0;
  binding.tile = {static_cast<std::int16_t>(tile.x + 1), tile.y};
  binding.source_action = tape[0].units[0];
  binding.desired = Item::TOMATO;
  auto request = request_for(simulator, tape, 103);
  request.exact_crop_bindings = std::span(&binding, 1);
  const auto result = issuer::issue_day_start(request);
  check(result.issued() && result.unsupported.size() == 1 &&
            result.unsupported[0].reason ==
                issuer::UnsupportedReason::AuthorityMismatch,
        "mismatched compiler authority was accepted");

  binding.tile = tile;
  binding.source_action = {Op::HARVEST, Item::NONE, 1};
  const auto wrong_action = issuer::issue_day_start(request);
  check(wrong_action.unsupported.size() == 1 &&
            wrong_action.unsupported[0].reason ==
                issuer::UnsupportedReason::AuthorityMismatch,
        "non-exact source action authority was accepted");

  binding.source_action = tape[0].units[0];
  auto conflicting = binding;
  conflicting.id = 2;
  conflicting.desired = Item::MELON;
  std::array bindings{binding, conflicting};
  request.exact_crop_bindings = bindings;
  const auto forward = issuer::issue_day_start(request);
  std::reverse(bindings.begin(), bindings.end());
  const auto reversed = issuer::issue_day_start(request);
  check(forward.unsupported.size() == 1 &&
            reversed.unsupported.size() == 1 &&
            forward.unsupported[0].reason ==
                issuer::UnsupportedReason::AuthorityMismatch &&
            reversed.unsupported[0].reason ==
                issuer::UnsupportedReason::AuthorityMismatch,
        "duplicate authority depended on input order");
}

fastkag::NativeTapeLibrary real_g001_library(const std::string& root) {
  const std::string tapes = root +
      "/meta_agent_route_rl_submission_minimal/"
      "teammate_meta_route_submission_v1/route_actions.json.zlib";
  const std::string library = root +
      "/meta_agent_route_rl_submission_minimal/"
      "teammate_meta_route_submission_v1/route_library.json";
  const std::string references = root +
      "/findNewRoad/marketMechanismCpp/nativeSelectiveNtEval/artifacts/"
      "native_teammate_refs.json.zlib";
  fastkag::NativeTapeLibrary result;
  result.routes.push_back(g001::repair::load_route(tapes, library, "G001"));
  result.routes.push_back(g001::repair::load_route(tapes, library, "G096"));
  result.r5_reference = g001::repair::load_route(references, library, "R5");
  result.md_reference = g001::repair::load_route(references, library, "MD");
  constexpr std::array<const char*, 5> labels{
      "10C4S_3Q", "8C6S_3Q", "6C8S_3Q", "6C12S_4Q_FIRST_YARN",
      "6C12S_4Q_SECOND_YARN"};
  for (std::size_t index = 0; index < labels.size(); ++index) {
    result.moon[index] = g001::repair::load_route(
        references, library, "MOON_" + std::string(labels[index]));
    result.moon_legacy[index] = g001::repair::load_route(
        references, library, "MOON_LEGACY_" + std::string(labels[index]));
  }
  return result;
}

issuer::FinalMarketReceipt receipt_for(
    const Simulator& after, int player, int submitted_step,
    const PlayerAction& submitted) {
  const auto& fills = after.last_market_fills()[player];
  return {player,
          submitted_step,
          submitted.market,
          std::vector<std::int32_t>(fills.begin(), fills.end())};
}

void cross_tick_animal_binding_fails_closed() {
  std::vector<PlayerAction> route(24);
  for (auto& action : route) action.units.resize(3);
  route[1].units[1] = {Op::PLACE, Item::COW, 1};
  PlayerAction final;
  final.units.push_back({});
  final.market = {{Op::HIRE, Item::NONE, 1},
                  {Op::BUY_ANIMAL, Item::COW, 1}};

  Simulator missing({}, 201);
  issuer::CrossTickAnimalIntentBinder binder;
  auto hire_only = final;
  hire_only.market.pop_back();
  check(binder.stage_final_submission(missing, 0, hire_only, route, 1).status ==
            issuer::CrossTickAnimalStatus::InvalidSubmission &&
            !binder.pending(),
        "missing BUY_ANIMAL was staged");
  auto buy_only = final;
  buy_only.market.erase(buy_only.market.begin());
  check(binder.stage_final_submission(missing, 0, buy_only, route, 1).status ==
            issuer::CrossTickAnimalStatus::InvalidSubmission &&
            !binder.pending(),
        "missing HIRE was staged");

  check(binder.stage_final_submission(missing, 0, final, route, 2).status ==
            issuer::CrossTickAnimalStatus::Staged,
        "valid synthetic cross-tick request was not staged");
  std::array<PlayerAction, 2> actions;
  actions[0] = final;
  missing.step(actions);
  auto forged = receipt_for(missing, 0, 0, final);
  forged.submitted_market.back().item = Item::SHEEP;
  check(binder.bind_next(missing, forged).status ==
            issuer::CrossTickAnimalStatus::ReceiptMismatch &&
            !binder.pending(),
        "forged BUY_ANIMAL receipt created an obligation");

  Simulator wrong_game({}, 204);
  check(binder.stage_final_submission(missing, 0, final, route, 20).status ==
            issuer::CrossTickAnimalStatus::Staged,
        "cross-game fixture was not staged");
  actions = {};
  actions[0] = final;
  wrong_game.step(actions);
  check(binder.bind_next(wrong_game,
                         receipt_for(wrong_game, 0, 0, final))
                .status == issuer::CrossTickAnimalStatus::StaleReceipt &&
            !binder.pending(),
        "receipt from another simulator seed consumed a stale intent");

  Simulator wrong_player({}, 205);
  check(binder.stage_final_submission(wrong_player, 0, final, route, 21)
                .status == issuer::CrossTickAnimalStatus::Staged,
        "wrong-player fixture was not staged");
  actions = {};
  actions[0] = final;
  wrong_player.step(actions);
  auto player_one = receipt_for(wrong_player, 0, 0, final);
  player_one.player = 1;
  check(binder.bind_next(wrong_player, player_one).status ==
            issuer::CrossTickAnimalStatus::StaleReceipt &&
            binder.bind_next(wrong_player,
                             receipt_for(wrong_player, 0, 0, final))
                    .status == issuer::CrossTickAnimalStatus::NoPending,
        "wrong-player or repeated receipt did not fail closed");

  Simulator day_end({}, 206);
  std::array<PlayerAction, 2> pass;
  while (day_end.step_count() < 23) day_end.step(pass);
  check(binder.stage_final_submission(day_end, 0, final, route, 22).status ==
            issuer::CrossTickAnimalStatus::InvalidSubmission &&
            !binder.pending(),
        "day-end submission created a cross-day animal intent");

  Simulator no_cash({}, 202);
  const_cast<fastkag::Farm&>(no_cash.farms()[0]).money = 0;
  check(binder.stage_final_submission(no_cash, 0, final, route, 3).status ==
            issuer::CrossTickAnimalStatus::Staged,
        "zero-cash request was not staged as unresolved");
  actions = {};
  actions[0] = final;
  no_cash.step(actions);
  check(binder.bind_next(no_cash, receipt_for(no_cash, 0, 0, final)).status ==
            issuer::CrossTickAnimalStatus::MissingHireEvidence,
        "unfilled HIRE was treated as a real actor");

  Simulator ambiguous({}, 203);
  auto two_hires = final;
  two_hires.market.insert(two_hires.market.begin(),
                          {Op::HIRE, Item::NONE, 1});
  route[1].units[2] = {Op::PLACE, Item::COW, 1};
  check(binder.stage_final_submission(ambiguous, 0, two_hires, route, 4)
                .status == issuer::CrossTickAnimalStatus::Staged,
        "ambiguous request did not reach bind gate");
  actions = {};
  actions[0] = two_hires;
  ambiguous.step(actions);
  check(binder.bind_next(ambiguous,
                         receipt_for(ambiguous, 0, 0, two_hires))
                .status == issuer::CrossTickAnimalStatus::PlaceNotUnique,
        "two matching PLACE candidates selected one by order");
}

void real_seed990034_seat1_cross_tick_animal_binding() {
  const std::string root = G001_ISSUER_REPO_ROOT;
  auto library = real_g001_library(root);
  const auto g001_tape = library.routes[0];
  fastkag::NativeTeammateExecutor executor(std::move(library));
  std::array<fastkag::NativeAgentState, 2> state;
  Simulator simulator({}, 990034);
  issuer::CrossTickAnimalIntentBinder binder;
  PlayerAction submitted;
  while (simulator.step_count() <= 96) {
    std::array<PlayerAction, 2> actions;
    actions[1] = executor.action_external(simulator, 1, 0, state[1]);
    actions[0] = executor.action_external(simulator, 0, 1, state[0]);
    if (simulator.step_count() == 96) {
      submitted = actions[1];
      const auto staged = binder.stage_final_submission(
          simulator, 1, submitted, g001_tape, 990034);
      check(staged.status == issuer::CrossTickAnimalStatus::Staged &&
                !staged.obligation,
            "seed990034 step96 guessed a hired actor before receipts");
    }
    simulator.step(actions);
  }
  const auto result =
      binder.bind_next(simulator, receipt_for(simulator, 1, 96, submitted));
  const auto& fills = simulator.last_market_fills()[1];
  check(result.status == issuer::CrossTickAnimalStatus::Bound &&
            result.obligation && result.obligation->player == 1 &&
            result.obligation->actor == 1 &&
            result.obligation->animal == Item::COW &&
            result.obligation->place_step == 106 &&
            result.obligation->target.row == 1 &&
            result.obligation->target.column == 2 &&
            result.obligation->place_action.op == Op::PLACE &&
            result.obligation->acquisition_step == 96 &&
            result.obligation->acquisition_market_slot == 3 &&
            result.obligation->acquisition_action.op == Op::BUY_ANIMAL &&
            result.obligation->acquisition_action.item == Item::COW &&
            result.obligation->acquisition_action.quantity == 1 &&
            result.obligation->acquisition_fill == 0 &&
            result.obligation->acquisition_seed == simulator.seed() &&
            fills.size() == 4 && fills[0] == 1 && fills[1] == 1 &&
            fills[2] == 1 && fills[3] == 0,
        "seed990034 seat1 cross-tick animal binding changed");
  std::cout << "cross_tick_seed990034 seat=1 submitted=96 actor="
            << result.obligation->actor
            << " animal=COW place_step=" << result.obligation->place_step
            << " tile=[" << result.obligation->target.column << ','
            << result.obligation->target.row << "] hire_fills=" << fills[0]
            << ',' << fills[1] << ',' << fills[2]
            << " buy_fill=" << fills[3] << '\n';
}

void one_real_g001_shadow_trace() {
  const std::string root = G001_ISSUER_REPO_ROOT;
  auto library = real_g001_library(root);
  const auto raw_tape = library.routes[0];
  fastkag::NativeTeammateExecutor executor(std::move(library));
  std::array<fastkag::NativeAgentState, 2> agent_state;
  Simulator simulator({}, 25772238701ULL);
  issuer::PersistentRouteIntentRegistry registry;
  int issued_days = 0;
  int move_tokens = 0;
  int typed_crop_sources = 0;
  int typed_animal_sources = 0;
  int final_short_day_rejections = 0;
  int provider_valid_days = 0;
  int provider_invalid_days = 0;
  int provider_mismatches = 0;
  std::array<int, 24> provider_mismatches_by_op{};
  std::vector<int> invalid_provider_days;
  std::optional<issuer::IssueResult> current_issue;
  std::vector<PlayerAction> actual_provider_day;

  while (!simulator.done()) {
    if (simulator.hour() == 0) {
      if (current_issue) {
        const auto audit = issuer::audit_final_provider_day(
            *current_issue, simulator.step_count() - 24, actual_provider_day);
        provider_valid_days += audit.day_valid;
        provider_invalid_days += !audit.day_valid;
        provider_mismatches += audit.mismatches;
        if (!audit.day_valid) {
          invalid_provider_days.push_back(simulator.day() - 1);
        }
        for (std::size_t operation = 0;
             operation < provider_mismatches_by_op.size(); ++operation) {
          provider_mismatches_by_op[operation] +=
              audit.mismatches_by_source_op[operation];
        }
        check(audit.mismatches ==
                  static_cast<int>(audit.mismatched_sources.size()),
              "provider mismatches were not marked unsupported");
      }
      current_issue.reset();
      actual_provider_day.clear();
      auto request = request_for(simulator, raw_tape,
                                 static_cast<std::uint64_t>(simulator.day() + 1));
      request.persistent_intents = &registry;
      const auto issued = issuer::issue_day_start(request);
      if (simulator.step_count() <= 672) {
        check(issued.issued(), "real G001 full day issuer rejected");
        ++issued_days;
        move_tokens += static_cast<int>(issued.moves.size());
        typed_crop_sources +=
            static_cast<int>(issued.crop_owner_obligations.size());
        typed_animal_sources +=
            static_cast<int>(issued.animal_owner_obligations.size());
        current_issue = issued;
        for (const auto& move : issued.moves) {
          check(move.source_step / 24 == simulator.day() &&
                    move.actor >= 0 &&
                    move.actor < static_cast<int>(
                                     raw_tape[move.source_step].units.size()) &&
                    raw_tape[move.source_step].units[move.actor].op ==
                        move.action.op,
                "real G001 MOVE source mismatch");
        }
      } else {
        check(issued.reject == issuer::IssueReject::RouteTooShort,
              "terminal short day did not fail closed");
        ++final_short_day_rejections;
      }
    }

    std::array<PlayerAction, 2> actions;
    for (int player = 0; player < 2; ++player) {
      actions[player] =
          executor.action_external(simulator, player, 0, agent_state[player]);
    }
    actual_provider_day.push_back(actions[0]);
    const auto& farm = simulator.farms()[0];
    std::vector<Position> positions{farm.farmer};
    positions.insert(positions.end(), farm.hands.begin(), farm.hands.end());
    for (std::size_t actor = 0;
         actor < actions[0].units.size() && actor < positions.size(); ++actor) {
      (void)registry.record_exact_source(
          static_cast<int>(actor), simulator.step_count(), positions[actor],
          actions[0].units[actor]);
    }
    simulator.step(actions);
  }

  check(issued_days == 29 && final_short_day_rejections == 1 &&
            provider_valid_days + provider_invalid_days == 29 &&
            move_tokens > 0 && typed_crop_sources > 0 &&
            typed_animal_sources > 0,
        "real G001 shadow coverage was empty");
  std::cout << "shadow_provider_valid_days=" << provider_valid_days
            << " invalid_days=" << provider_invalid_days
            << " source_mismatches=" << provider_mismatches << " days=";
  for (std::size_t index = 0; index < invalid_provider_days.size(); ++index) {
    std::cout << (index == 0 ? "" : ",") << invalid_provider_days[index];
  }
  std::cout << " ops=";
  bool first = true;
  for (std::size_t operation = 0;
       operation < provider_mismatches_by_op.size(); ++operation) {
    if (provider_mismatches_by_op[operation] == 0) {
      continue;
    }
    std::cout << (first ? "" : ",") << operation << ':'
              << provider_mismatches_by_op[operation];
    first = false;
  }
  std::cout << '\n';
}

}  // namespace

int main() {
  try {
    persistent_crop_lineage_survives_weed();
    itemless_crop_without_lineage_fails_closed();
    exact_action_shapes_emit_typed_abis();
    mismatched_authority_fails_closed();
    cross_tick_animal_binding_fails_closed();
    real_seed990034_seat1_cross_tick_animal_binding();
    one_real_g001_shadow_trace();
    std::cout << "g001_day_start_obligation_issuer_tests: 7 groups passed\n";
  } catch (const std::exception& error) {
    std::cerr << "g001_day_start_obligation_issuer_tests: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
