#include "joint_fixed_move_oracle.hpp"

#include "nt_trace_bank.hpp"
#include "replanner.hpp"
#include "route_loader.hpp"
#include "static_dependencies.hpp"
#include "weed_absorption.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace jfmo = joint_fixed_move_oracle;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using jfmo::Tape;

namespace {

struct Options {
  std::uint64_t seed_begin = 990001;
  int seeds = 32;
  int threads = std::max(1u, std::thread::hardware_concurrency());
  std::string output = "artifacts/report.json";
  std::string tapes = JFMO_TAPES;
  std::string library = JFMO_LIBRARY;
  std::string nt_root = JFMO_NT_ROOT;
};

struct Opponent {
  std::string name;
  const Tape* tape{};
  const public_ports::NtTraceBankAgent* native{};
};

struct OpponentState {
  public_ports::NtTraceBankState native;
};

struct GameMetrics {
  double own{};
  double opponent{};
  std::int64_t unit_failures{};
  std::int64_t purchase_failures{};
  std::int64_t overflow{};
  jfmo::TradePolicy trade_policy{jfmo::TradePolicy::Baseline};
  int candidate_index{};
  int weed_patches{};
  int debt_patches{};
  int crop_patches{};
  bool legacy_weed{true};
  int legacy_weed_triggers{};
  int legacy_dropped_move_risks{};
  int runtime_move_mismatches{};

  [[nodiscard]] int score() const noexcept {
    return own > opponent ? 2 : own == opponent ? 1 : 0;
  }
  [[nodiscard]] double margin() const noexcept { return own - opponent; }
};

struct VariantTape {
  Tape tape;
  int weed_patches{};
  int debt_patches{};
  int crop_patches{};
  bool legacy_weed{true};
};

struct LegacyWeedTransaction {
  bool active{};
  int start{-1};
  Action intended{};
};

struct LegacyWeedState {
  std::vector<LegacyWeedTransaction> actors;
};

struct WeedEvent {
  int step{};
  int actor{};
  fastkag::Position position{};
  Action intended{};
};

struct PurchaseFailure {
  int step{};
  int slot{};
  Action action{};
  int missing{};
};

struct Trace {
  std::vector<std::vector<fastkag::Position>> positions;
  std::vector<WeedEvent> weeds;
  std::vector<PurchaseFailure> purchases;
  std::vector<latecrop::Visit> visits;
  std::vector<int> market_room;
  std::array<std::vector<int>, fastkag::N_CROPS> crop_sales;
};

struct Job {
  int opponent{};
  std::uint64_t seed{};
  int seat{};
};

struct Row {
  std::string opponent;
  std::uint64_t seed{};
  int seat{};
  std::array<GameMetrics, 6> arms{};
};

bool purchase(Op op) {
  return op == Op::BUY_SEED || op == Op::BUY_PRODUCT ||
         op == Op::BUY_ANIMAL || op == Op::HIRE || op == Op::BUY_LAND;
}

int request_quantity(const Action& action) {
  return action.op == Op::HIRE || action.op == Op::BUY_LAND
             ? 1
             : std::max(0, static_cast<int>(action.quantity));
}

std::vector<fastkag::Position> actor_positions(const fastkag::Farm& farm) {
  std::vector<fastkag::Position> out{farm.farmer};
  out.insert(out.end(), farm.hands.begin(), farm.hands.end());
  return out;
}

int shed_total(const fastkag::PrivateState& state) {
  return std::accumulate(state.shed.begin(), state.shed.end(), 0);
}

const PlayerAction& frame(const Tape& tape, int step);

const fastkag::Tile* tile_at(const fastkag::Simulator& simulator, int player,
                             fastkag::Position position) {
  if (position.x < 0 || position.y < 0 ||
      position.x >= simulator.config().board_size ||
      position.y >= simulator.config().board_size)
    return nullptr;
  const int index = position.y * simulator.config().board_size + position.x;
  return &simulator.farms()[player].tiles[static_cast<std::size_t>(index)];
}

PlayerAction apply_legacy_weed(PlayerAction out, const Tape& tape,
                               const fastkag::Simulator& simulator, int player,
                               LegacyWeedState& state, int& triggers,
                               int& dropped_move_risks) {
  const int step = simulator.step_count();
  const auto positions = actor_positions(simulator.farms()[player]);
  state.actors.resize(out.units.size());
  int hires = 0;
  for (int ahead = 0; ahead < 3; ++ahead) {
    const auto& value = frame(tape, std::min(
        step + ahead, static_cast<int>(tape.size()) - 1));
    hires += std::count_if(value.market.begin(), value.market.end(),
                           [](const Action& action) {
                             return action.op == Op::HIRE;
                           });
  }
  const bool farmer_barrier = hires >= 5;
  for (std::size_t actor = 0; actor < state.actors.size(); ++actor) {
    auto& transaction = state.actors[actor];
    if (farmer_barrier && actor == 0) {
      transaction.active = false;
      continue;
    }
    if (!transaction.active) continue;
    const int age = step - transaction.start;
    if (age == 1) {
      out.units[actor] = transaction.intended;
    } else if (age >= 2 && age <= 9) {
      const auto& previous = frame(tape, std::max(0, step - 1));
      out.units[actor] = actor < previous.units.size()
                             ? previous.units[actor]
                             : Action{};
    } else {
      transaction.active = false;
    }
  }
  for (std::size_t actor = 0;
       actor < out.units.size() && actor < positions.size(); ++actor) {
    auto& transaction = state.actors[actor];
    if (transaction.active || (farmer_barrier && actor == 0)) continue;
    const auto op = out.units[actor].op;
    if (op != Op::PLANT && op != Op::BUILD_PASTURE) continue;
    const auto* tile = tile_at(simulator, player, positions[actor]);
    if (!tile || tile->kind != fastkag::TileKind::WEED) continue;
    transaction = {true, step, out.units[actor]};
    out.units[actor] = {Op::DIG};
    ++triggers;
    const auto& dropped = frame(tape, std::min(
        step + 9, static_cast<int>(tape.size()) - 1));
    if (actor < dropped.units.size() &&
        jfmo::is_move(dropped.units[actor].op))
      ++dropped_move_risks;
  }
  return out;
}

const PlayerAction& frame(const Tape& tape, int step) {
  if (tape.empty()) throw std::runtime_error("empty tape");
  return tape[std::min<std::size_t>(static_cast<std::size_t>(step),
                                    tape.size() - 1)];
}

PlayerAction opponent_action(const Opponent& opponent, OpponentState& state,
                             const fastkag::Simulator& simulator, int seat) {
  if (opponent.tape) return frame(*opponent.tape, simulator.step_count());
  if (!opponent.native) throw std::runtime_error("invalid opponent provider");
  return opponent.native->action(simulator, seat, state.native);
}

GameMetrics run_game(const Tape& baseline, const VariantTape& variant,
                     jfmo::TradePolicy policy, const Opponent& opponent,
                     std::uint64_t seed, int seat, int candidate_index = 0,
                     Trace* trace = nullptr) {
  const auto move_certificate =
      jfmo::certify_absolute_moves(baseline, variant.tape);
  if (!move_certificate.valid) {
    std::ostringstream message;
    message << "fixed MOVE candidate rejected: " << move_certificate.reason
            << ":weed=" << variant.weed_patches
            << ":debt=" << variant.debt_patches
            << ":crop=" << variant.crop_patches
            << ":legacy=" << variant.legacy_weed;
    throw std::runtime_error(message.str());
  }
  fastkag::Simulator simulator({}, seed);
  OpponentState opponent_state;
  GameMetrics result;
  result.trade_policy = policy;
  result.candidate_index = candidate_index;
  result.weed_patches = variant.weed_patches;
  result.debt_patches = variant.debt_patches;
  result.crop_patches = variant.crop_patches;
  result.legacy_weed = variant.legacy_weed;
  LegacyWeedState legacy_weed;
  if (trace) trace->positions.resize(
      static_cast<std::size_t>(simulator.config().episode_steps));

  while (!simulator.done()) {
    const int step = simulator.step_count();
    const auto& raw = frame(variant.tape, step);
    auto own = jfmo::apply_trade_policy(raw, simulator, seat, policy);
    if (variant.legacy_weed)
      own = apply_legacy_weed(std::move(own), variant.tape, simulator, seat,
                              legacy_weed, result.legacy_weed_triggers,
                              result.legacy_dropped_move_risks);
    const std::size_t move_slots = std::max(raw.units.size(), own.units.size());
    for (std::size_t actor = 0; actor < move_slots; ++actor) {
      const Action raw_action = actor < raw.units.size() ? raw.units[actor]
                                                         : Action{};
      const Action emitted = actor < own.units.size() ? own.units[actor]
                                                       : Action{};
      if ((jfmo::is_move(raw_action.op) || jfmo::is_move(emitted.op)) &&
          !jfmo::action_equal(raw_action, emitted))
        ++result.runtime_move_mismatches;
    }
    const auto positions = actor_positions(simulator.farms()[seat]);
    if (trace) {
      trace->positions[static_cast<std::size_t>(step)] = positions;
      if (raw.market.size() <
          static_cast<std::size_t>(simulator.config().max_market_orders))
        trace->market_room.push_back(step);
      for (const auto& order : raw.market) {
        const int item = static_cast<int>(order.item);
        if (order.op == Op::SELL && item >= 0 && item < fastkag::N_CROPS)
          trace->crop_sales[static_cast<std::size_t>(item)].push_back(step);
      }
    }

    for (std::size_t actor = 0;
         actor < own.units.size() && actor < positions.size(); ++actor) {
      const auto& action = own.units[actor];
      if (action.op == Op::PASS) continue;
      const bool succeeds = g001::weed_absorption::projected_unit_action_succeeds(
          simulator, seat, own, static_cast<int>(actor));
      if (!succeeds) ++result.unit_failures;
      if (trace) {
        const auto position = positions[actor];
        const int tile_index = position.y * simulator.config().board_size +
                               position.x;
        const auto& tile = simulator.farms()[seat].tiles[
            static_cast<std::size_t>(tile_index)];
        if (!succeeds && tile.kind == fastkag::TileKind::WEED &&
            (action.op == Op::PLANT || action.op == Op::BUILD_PASTURE))
          trace->weeds.push_back(
              {step, static_cast<int>(actor), position, action});
        if (latecrop::crop_rewritable(action.op)) {
          const int crop = std::clamp(static_cast<int>(tile.crop), 0,
                                      fastkag::N_CROPS - 1);
          trace->visits.push_back(
              {step, static_cast<int>(actor), tile_index, action.op, tile.kind,
               tile.crop, simulator.market().inventory[static_cast<std::size_t>(crop)],
               simulator.market().prices[static_cast<std::size_t>(crop)],
               shed_total(simulator.privates()[seat])});
        }
      }
    }

    std::array<PlayerAction, 2> actions;
    actions[seat] = std::move(own);
    actions[1 - seat] =
        opponent_action(opponent, opponent_state, simulator, 1 - seat);
    simulator.step(actions);
    const auto& fills = simulator.last_market_fills()[seat];
    for (std::size_t slot = 0; slot < actions[seat].market.size(); ++slot) {
      const auto& order = actions[seat].market[slot];
      if (!purchase(order.op)) continue;
      const int requested = request_quantity(order);
      const int filled = slot < fills.size() ? fills[slot] : 0;
      const int missing = std::max(0, requested - filled);
      result.purchase_failures += missing;
      if (trace && policy == jfmo::TradePolicy::Baseline && missing > 0 &&
          (order.op == Op::BUY_SEED || order.op == Op::BUY_ANIMAL))
        trace->purchases.push_back(
            {step, static_cast<int>(slot), order, missing});
    }
    result.overflow += simulator.last_end_of_day_overflow()[seat];
  }
  result.own = simulator.farms()[seat].money;
  result.opponent = simulator.farms()[1 - seat].money;
  return result;
}

bool better(const GameMetrics& left, const GameMetrics& right) {
  return std::tuple(left.score(), left.margin(), left.own,
                    -left.unit_failures - left.purchase_failures,
                    -left.overflow) >
         std::tuple(right.score(), right.margin(), right.own,
                    -right.unit_failures - right.purchase_failures,
                    -right.overflow);
}

GameMetrics exact_select(const Tape& baseline,
                         const std::vector<VariantTape>& tapes,
                         const std::vector<jfmo::TradePolicy>& policies,
                         const Opponent& opponent, std::uint64_t seed,
                         int seat) {
  std::optional<GameMetrics> best;
  int candidate = 0;
  for (const auto& tape : tapes)
    for (const auto policy : policies) {
      auto value = run_game(baseline, tape, policy, opponent, seed, seat,
                            candidate++);
      if (!best || better(value, *best)) best = value;
    }
  if (!best) throw std::runtime_error("oracle candidate set is empty");
  return *best;
}

std::vector<VariantTape> weed_variants(const Tape& baseline,
                                       const VariantTape& source,
                                       const Opponent& opponent,
                                       std::uint64_t seed, int seat) {
  Trace trace;
  auto discovery = source;
  discovery.legacy_weed = false;
  (void)run_game(baseline, discovery, jfmo::TradePolicy::Baseline, opponent,
                 seed, seat, 0, &trace);
  // The deployed baseline contains the legacy ten-step shift transaction.
  // Raw fixed-MOVE execution is itself a valid fallback when no productive
  // recompilation is safe, so the oracle must be allowed to decline repair.
  std::vector<VariantTape> out{source, discovery};
  // Each collision gets independent terminal-certified branches.  Bundling
  // every collision into one all-or-nothing tape hid useful repairs whenever
  // one unrelated collision displaced a valuable operation.
  std::vector<std::vector<jfmo::AbsoluteWeedPatch>> by_event;
  by_event.reserve(trace.weeds.size());
  for (const auto& event : trace.weeds) {
    auto patches = jfmo::enumerate_absolute_weed_patches(
        baseline, source.tape, trace.positions,
        {event.step, event.actor, event.position, event.intended}, 24, 24);
    for (auto& patch : patches) {
      VariantTape candidate{std::move(patch.tape), source.weed_patches + 1,
                            source.debt_patches, source.crop_patches, false};
      out.push_back(std::move(candidate));
    }
    by_event.push_back(std::move(patches));
  }

  // Also offer one low-displacement joint branch.  Full Cartesian expansion
  // is unnecessary for this bound and grows exponentially; individual event
  // branches above ensure that a bad second patch cannot mask a good first.
  auto combined = source;
  combined.legacy_weed = false;
  std::set<std::pair<int, int>> occupied;
  for (std::size_t event_index = 0; event_index < by_event.size();
       ++event_index) {
    const auto& patches = by_event[event_index];
    const int actor = trace.weeds[event_index].actor;
    for (const auto& patch : patches) {
      const auto event_slot = std::pair(patch.event_step, actor);
      const auto replay_slot = std::pair(patch.replay_step, actor);
      const auto water_slot = std::pair(patch.water_step, actor);
      if (occupied.contains(event_slot) || occupied.contains(replay_slot) ||
          (patch.water_step >= 0 && occupied.contains(water_slot)))
        continue;
      combined.tape[patch.event_step].units[actor] = {Op::DIG};
      combined.tape[patch.replay_step].units[actor] =
          trace.weeds[event_index].intended;
      if (patch.water_step >= 0)
        combined.tape[patch.water_step].units[actor] = {Op::WATER};
      occupied.insert(event_slot);
      occupied.insert(replay_slot);
      if (patch.water_step >= 0) occupied.insert(water_slot);
      ++combined.weed_patches;
      break;
    }
  }
  if (combined.weed_patches > source.weed_patches) {
    const auto certificate = jfmo::certify_absolute_moves(baseline, combined.tape);
    if (!certificate.valid)
      throw std::runtime_error("combined weed patch rejected:" +
                               certificate.reason);
    out.push_back(std::move(combined));
  }
  return out;
}

std::vector<VariantTape> debt_variants(const Tape& baseline,
                                       const VariantTape& source,
                                       const Opponent& opponent,
                                       std::uint64_t seed, int seat) {
  Trace trace;
  (void)run_game(baseline, source, jfmo::TradePolicy::Baseline, opponent,
                 seed, seat, 0, &trace);
  const auto dependencies =
      g001::failure_debt::extract_static_dependencies(source.tape);
  auto repaired = source;
  int patched = 0;
  for (const auto& failure : trace.purchases) {
    int deadline = -1;
    for (const auto& dependency : dependencies) {
      if (dependency.purchase_step == failure.step &&
          dependency.purchase_slot == failure.slot && dependency.complete &&
          dependency.first_unit_step > failure.step)
        deadline = deadline < 0 ? dependency.first_unit_step
                                : std::min(deadline, dependency.first_unit_step);
    }
    if (deadline < 0) continue;
    auto target = std::find_if(trace.market_room.begin(), trace.market_room.end(),
                               [&](int step) {
      return step > failure.step && step < deadline &&
             repaired.tape[step].market.size() < 10;
    });
    if (target == trace.market_room.end()) continue;
    auto retry = failure.action;
    retry.quantity = failure.missing;
    repaired.tape[*target].market.push_back(retry);
    ++patched;
  }
  std::vector<VariantTape> out{source};
  if (patched > 0) {
    repaired.debt_patches += patched;
    const auto certificate = jfmo::certify_absolute_moves(baseline, repaired.tape);
    if (!certificate.valid)
      throw std::runtime_error("debt patch rejected:" + certificate.reason);
    out.push_back(std::move(repaired));
  }
  return out;
}

int actor_at(const std::vector<latecrop::Visit>& visits, int tile, int step) {
  for (const auto& visit : visits)
    if (visit.tile == tile && visit.step == step) return visit.actor;
  throw std::runtime_error("late crop planned visit disappeared");
}

std::vector<VariantTape> crop_variants(const Tape& baseline,
                                       const VariantTape& source,
                                       const Opponent& opponent,
                                       std::uint64_t seed, int seat) {
  Trace trace;
  auto discovery = source;
  discovery.legacy_weed = false;
  (void)run_game(baseline, discovery, jfmo::TradePolicy::Baseline, opponent,
                 seed, seat, 0, &trace);
  latecrop::Candidate best;
  for (const auto target : {Item::TOMATO, Item::MELON, Item::WHEAT}) {
    auto candidate = latecrop::find_candidate(
        trace.visits, trace.market_room,
        trace.crop_sales[static_cast<std::size_t>(target)], Item::STRAWBERRY,
        target, 220, 12);
    if (candidate.feasible &&
        (!best.feasible || candidate.net_value > best.net_value))
      best = std::move(candidate);
  }
  std::vector<VariantTape> out{source};
  if (!best.feasible) return out;
  auto compiled = source;
  compiled.legacy_weed = false;
  for (const auto& visit : trace.visits) {
    if (visit.tile == best.tile && visit.step >= best.dig_step &&
        visit.step <= best.harvest_step &&
        latecrop::crop_rewritable(visit.baseline_op) &&
        visit.actor < static_cast<int>(compiled.tape[visit.step].units.size()))
      compiled.tape[visit.step].units[visit.actor] = Action{};
  }
  compiled.tape[best.dig_step].units[actor_at(trace.visits, best.tile,
                                              best.dig_step)] = {Op::DIG};
  compiled.tape[best.plant_step].units[actor_at(trace.visits, best.tile,
                                                best.plant_step)] =
      {Op::PLANT, best.target, 1};
  for (const int step : best.water_steps)
    compiled.tape[step].units[actor_at(trace.visits, best.tile, step)] =
        {Op::WATER};
  compiled.tape[best.harvest_step].units[actor_at(
      trace.visits, best.tile, best.harvest_step)] = {Op::HARVEST};
  if (compiled.tape[best.buy_step].market.size() >= 10) return out;
  compiled.tape[best.buy_step].market.push_back(
      {Op::BUY_SEED, best.target, 1});
  ++compiled.crop_patches;
  const auto certificate = jfmo::certify_absolute_moves(baseline, compiled.tape);
  if (!certificate.valid)
    throw std::runtime_error("crop patch rejected:" + certificate.reason);
  out.push_back(std::move(compiled));
  return out;
}

std::vector<VariantTape> expand(
    const std::vector<VariantTape>& input,
    const std::function<std::vector<VariantTape>(const VariantTape&)>& fn) {
  std::vector<VariantTape> out;
  for (const auto& value : input) {
    auto children = fn(value);
    out.insert(out.end(), std::make_move_iterator(children.begin()),
               std::make_move_iterator(children.end()));
  }
  return out;
}

std::array<GameMetrics, 6> evaluate_job(const Tape& baseline,
                                        const Opponent& opponent,
                                        std::uint64_t seed, int seat) {
  const VariantTape root{baseline};
  const std::vector<jfmo::TradePolicy> baseline_policy{
      jfmo::TradePolicy::Baseline};
  const std::vector<jfmo::TradePolicy> trade_policies{
      jfmo::TradePolicy::Baseline, jfmo::TradePolicy::HoldAll,
      jfmo::TradePolicy::ClearExisting,
      jfmo::TradePolicy::HoldThenClear360,
      jfmo::TradePolicy::HoldThenClear480,
      jfmo::TradePolicy::HoldThenClear600,
      jfmo::TradePolicy::TerminalClear};
  std::array<GameMetrics, 6> out;
  out[0] = exact_select(baseline, {root}, baseline_policy, opponent, seed, seat);

  const auto weeds = weed_variants(baseline, root, opponent, seed, seat);
  out[1] = exact_select(baseline, weeds, baseline_policy, opponent, seed, seat);

  const auto weed_debt = expand(weeds, [&](const VariantTape& value) {
    return debt_variants(baseline, value, opponent, seed, seat);
  });
  out[2] = exact_select(baseline, weed_debt, baseline_policy, opponent, seed,
                        seat);
  out[3] = exact_select(baseline, {root}, trade_policies, opponent, seed, seat);

  const auto crops = crop_variants(baseline, root, opponent, seed, seat);
  out[4] = exact_select(baseline, crops, trade_policies, opponent, seed, seat);

  const auto crop_weeds = expand(crops, [&](const VariantTape& value) {
    return weed_variants(baseline, value, opponent, seed, seat);
  });
  const auto all = expand(crop_weeds, [&](const VariantTape& value) {
    return debt_variants(baseline, value, opponent, seed, seat);
  });
  out[5] = exact_select(baseline, all, trade_policies, opponent, seed, seat);
  return out;
}

std::string json_escape(std::string_view text) {
  std::ostringstream out;
  out << '"';
  for (const char c : text) {
    if (c == '"' || c == '\\') out << '\\';
    out << c;
  }
  out << '"';
  return out.str();
}

Options parse_options(int argc, char** argv) {
  Options out;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() {
      if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
      return std::string(argv[i]);
    };
    if (arg == "--seed-begin") out.seed_begin = std::stoull(next());
    else if (arg == "--seeds") out.seeds = std::stoi(next());
    else if (arg == "--threads") out.threads = std::stoi(next());
    else if (arg == "--output") out.output = next();
    else if (arg == "--tapes") out.tapes = next();
    else if (arg == "--library") out.library = next();
    else if (arg == "--nt-root") out.nt_root = next();
    else throw std::invalid_argument("unknown option " + arg);
  }
  if (out.seeds <= 0 || out.threads <= 0)
    throw std::invalid_argument("seeds and threads must be positive");
  for (int i = 0; i < out.seeds; ++i)
    if (out.seed_begin + static_cast<std::uint64_t>(i) == 976)
      throw std::invalid_argument("seed 976 is forbidden");
  return out;
}

std::string bank_filename(std::string_view name) {
  if (name == "hasegawa") return "hasegawa_current_trace_bank_v1.npz";
  if (name == "rank04") return "rank04_arman_trace_bank_v1.npz";
  throw std::invalid_argument("unknown native opponent");
}

struct Aggregate {
  int games{};
  int original_losses{};
  int loss_to_win{};
  int baseline_wins{};
  int wins_retained{};
  double own{};
  double margin{};
  double score{};
  std::int64_t failures{};
  std::int64_t overflow{};
  int selected_weed{};
  int selected_debt{};
  int selected_crop{};
  int selected_fixed_weed{};
  std::int64_t legacy_triggers{};
  std::int64_t legacy_dropped_move_risks{};
  std::int64_t runtime_move_mismatches{};
};

Aggregate aggregate(const std::vector<Row>& rows, std::string_view opponent,
                    int arm) {
  Aggregate out;
  for (const auto& row : rows) {
    if (row.opponent != opponent) continue;
    const auto& baseline = row.arms[0];
    const auto& value = row.arms[static_cast<std::size_t>(arm)];
    ++out.games;
    out.original_losses += baseline.score() == 0;
    out.loss_to_win += baseline.score() == 0 && value.score() == 2;
    out.baseline_wins += baseline.score() == 2;
    out.wins_retained += baseline.score() == 2 && value.score() == 2;
    out.own += value.own;
    out.margin += value.margin();
    out.score += value.score() / 2.0;
    out.failures += value.unit_failures + value.purchase_failures;
    out.overflow += value.overflow;
    out.selected_weed += value.weed_patches > 0;
    out.selected_debt += value.debt_patches > 0;
    out.selected_crop += value.crop_patches > 0;
    out.selected_fixed_weed += !value.legacy_weed;
    out.legacy_triggers += value.legacy_weed_triggers;
    out.legacy_dropped_move_risks += value.legacy_dropped_move_risks;
    out.runtime_move_mismatches += value.runtime_move_mismatches;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) try {
  const auto options = parse_options(argc, argv);
  const auto started = std::chrono::steady_clock::now();
  const auto baseline = g001::repair::load_route(options.tapes, options.library,
                                                  "G001");
  const auto g001 = baseline;
  const auto g096 = g001::repair::load_route(options.tapes, options.library,
                                              "G096");
  const auto hasegawa_path =
      (std::filesystem::path(options.nt_root) / bank_filename("hasegawa")).string();
  const auto rank04_path =
      (std::filesystem::path(options.nt_root) / bank_filename("rank04")).string();
  const public_ports::NtTraceBankAgent hasegawa(
      hasegawa_path, public_ports::router_for_slug("hasegawa_current"));
  const public_ports::NtTraceBankAgent rank04(
      rank04_path, public_ports::router_for_slug("rank04_arman"));
  const std::array<Opponent, 4> opponents{{
      {"G001", &g001, nullptr}, {"G096", &g096, nullptr},
      {"hasegawa", nullptr, &hasegawa}, {"rank04", nullptr, &rank04}}};

  const auto baseline_certificate = jfmo::certify_absolute_moves(baseline, baseline);
  if (!baseline_certificate.valid)
    throw std::runtime_error("baseline failed its MOVE certificate");

  std::vector<Job> jobs;
  for (int opponent = 0; opponent < static_cast<int>(opponents.size()); ++opponent)
    for (int offset = 0; offset < options.seeds; ++offset)
      for (int seat = 0; seat < 2; ++seat)
        jobs.push_back({opponent, options.seed_begin +
                                      static_cast<std::uint64_t>(offset), seat});
  std::vector<Row> rows(jobs.size());
  std::atomic<std::size_t> cursor{};
  std::mutex error_mutex;
  std::exception_ptr failure;
  std::vector<std::thread> workers;
  const int worker_count =
      std::min<int>(options.threads, static_cast<int>(jobs.size()));
  for (int worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&] {
      try {
        while (true) {
          const auto index = cursor.fetch_add(1);
          if (index >= jobs.size()) break;
          const auto& job = jobs[index];
          rows[index] = {opponents[job.opponent].name, job.seed, job.seat,
                         evaluate_job(baseline, opponents[job.opponent],
                                      job.seed, job.seat)};
        }
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (!failure) failure = std::current_exception();
        cursor.store(jobs.size());
      }
    });
  }
  for (auto& worker : workers) worker.join();
  if (failure) std::rethrow_exception(failure);

  const auto parent = std::filesystem::path(options.output).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream report(options.output);
  if (!report) throw std::runtime_error("cannot open report output");
  report << std::fixed << std::setprecision(6);
  report << "{\n  \"schema\": \"joint-fixed-move-oracle-mvp-v2\",\n"
         << "  \"oracle_bound_not_deployable\": true,\n"
         << "  \"baseline_provider_scope\": \"compiled G001 route tape on FastKag plus the exact deployed legacy weed transaction; other stateful NativeTeammate overlays are not composed\",\n"
         << "  \"future_information_use\": \"complete-game terminal selection over a bounded finite branch portfolio\",\n"
         << "  \"market_oracle_scope\": \"FastKag exact complete-game branches; seven deterministic rolling/hold/clear tapes, not a proof of global market optimum\",\n"
         << "  \"failure_debt_scope\": \"failureDebtSchedulerCpp static dependency extraction plus exact-terminal-certified retry; full receding-horizon scheduler ABI is not composed\",\n"
         << "  \"weed_scope\": \"exact legacy ten-step transaction baseline versus absolute-MOVE production recompilation; PLANT branches require same-day PLANT plus WATER slots\",\n"
         << "  \"move_contract\": \"every actor at every absolute step has exact MOVE op/item/quantity identity\",\n"
         << "  \"move_certificate\": {\"valid\": true, \"checked_steps\": "
         << baseline_certificate.checked_steps << "},\n"
         << "  \"seed_begin\": " << options.seed_begin << ",\n"
         << "  \"seeds_per_opponent\": " << options.seeds << ",\n"
         << "  \"seats\": 2,\n  \"opponents\": [\"G001\", \"G096\", \"hasegawa\", \"rank04\"],\n"
         << "  \"arms\": {\n";
  for (int arm = 0; arm < 6; ++arm) {
    report << "    " << json_escape(jfmo::arm_name(static_cast<jfmo::Arm>(arm)))
           << ": {\n";
    for (std::size_t opponent = 0; opponent < opponents.size(); ++opponent) {
      const auto value = aggregate(rows, opponents[opponent].name, arm);
      const double games = std::max(1, value.games);
      report << "      " << json_escape(opponents[opponent].name) << ": {"
             << "\"games\": " << value.games
             << ", \"original_losses\": " << value.original_losses
             << ", \"loss_to_win\": " << value.loss_to_win
             << ", \"baseline_wins\": " << value.baseline_wins
             << ", \"wins_retained\": " << value.wins_retained
             << ", \"own_mean\": " << value.own / games
             << ", \"margin_mean\": " << value.margin / games
             << ", \"score_rate\": " << value.score / games
             << ", \"failure_mean\": " << double(value.failures) / games
             << ", \"overflow_mean\": " << double(value.overflow) / games
             << ", \"selected_weed_games\": " << value.selected_weed
             << ", \"selected_debt_games\": " << value.selected_debt
             << ", \"selected_crop_games\": " << value.selected_crop
             << ", \"selected_fixed_weed_games\": "
             << value.selected_fixed_weed
             << ", \"legacy_weed_triggers\": " << value.legacy_triggers
             << ", \"legacy_dropped_move_risks\": "
             << value.legacy_dropped_move_risks
             << ", \"runtime_move_mismatches\": "
             << value.runtime_move_mismatches << "}"
             << (opponent + 1 == opponents.size() ? "\n" : ",\n");
    }
    report << "    }" << (arm == 5 ? "\n" : ",\n");
  }
  report << "  },\n  \"synergy\": {\n";
  for (std::size_t opponent = 0; opponent < opponents.size(); ++opponent) {
    const auto all = aggregate(rows, opponents[opponent].name, 5);
    std::array<Aggregate, 4> single{
        aggregate(rows, opponents[opponent].name, 1),
        aggregate(rows, opponents[opponent].name, 2),
        aggregate(rows, opponents[opponent].name, 3),
        aggregate(rows, opponents[opponent].name, 4)};
    const auto best_loss_to_win = std::max_element(
        single.begin(), single.end(), [](const auto& a, const auto& b) {
          return a.loss_to_win < b.loss_to_win;
        })->loss_to_win;
    const auto best_margin = std::max_element(
        single.begin(), single.end(), [](const auto& a, const auto& b) {
          return a.margin / std::max(1, a.games) <
                 b.margin / std::max(1, b.games);
        });
    report << "    " << json_escape(opponents[opponent].name)
           << ": {\"joint_extra_loss_to_win_vs_best_component_arm\": "
           << all.loss_to_win - best_loss_to_win
           << ", \"joint_margin_mean_minus_best_component_arm\": "
           << all.margin / std::max(1, all.games) -
                  best_margin->margin / std::max(1, best_margin->games)
           << "}" << (opponent + 1 == opponents.size() ? "\n" : ",\n");
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  report << "  },\n  \"wall_seconds\": " << seconds << "\n}\n";

  std::ofstream detail(options.output + ".rows.jsonl");
  for (const auto& row : rows) {
    detail << "{\"opponent\":" << json_escape(row.opponent)
           << ",\"seed\":" << row.seed << ",\"seat\":" << row.seat
           << ",\"arms\":[";
    for (std::size_t arm = 0; arm < row.arms.size(); ++arm) {
      const auto& value = row.arms[arm];
      detail << "{\"name\":"
             << json_escape(jfmo::arm_name(static_cast<jfmo::Arm>(arm)))
             << ",\"own\":" << value.own
             << ",\"opponent_money\":" << value.opponent
             << ",\"margin\":" << value.margin()
             << ",\"score\":" << value.score() / 2.0
             << ",\"unit_failures\":" << value.unit_failures
             << ",\"purchase_failures\":" << value.purchase_failures
             << ",\"overflow\":" << value.overflow
             << ",\"trade_policy\":"
             << json_escape(jfmo::trade_policy_name(value.trade_policy))
             << ",\"weed_patches\":" << value.weed_patches
             << ",\"debt_patches\":" << value.debt_patches
             << ",\"crop_patches\":" << value.crop_patches
             << ",\"legacy_weed\":"
             << (value.legacy_weed ? "true" : "false")
             << ",\"legacy_weed_triggers\":"
             << value.legacy_weed_triggers
             << ",\"legacy_dropped_move_risks\":"
             << value.legacy_dropped_move_risks
             << ",\"runtime_move_mismatches\":"
             << value.runtime_move_mismatches << "}"
             << (arm + 1 == row.arms.size() ? "" : ",");
    }
    detail << "]}\n";
  }
  std::cout << "joint oracle games=" << rows.size()
            << " branches complete; report=" << options.output
            << " seconds=" << seconds << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "joint_fixed_move_oracle: " << error.what() << '\n';
  return 2;
}
