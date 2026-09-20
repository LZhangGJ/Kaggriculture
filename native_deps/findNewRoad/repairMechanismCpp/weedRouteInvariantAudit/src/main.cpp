#include "native_teammate.hpp"
#include "route_loader.hpp"
#include "scheduler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;

constexpr std::array<const char*, 6> kPolicyNames{
    "legacy9", "min_loss", "strict_slack", "move_guard", "day_tail_drop",
    "tail_preserve_move"};

struct Options {
  std::string tapes;
  std::string library;
  std::string refs;
  std::string output;
  int threads{std::max(1u, std::thread::hardware_concurrency())};
  int seeds{128};
  std::vector<std::uint64_t> blocks{997000, 998000};
};

template <typename T> T number(std::string_view text) {
  T value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("invalid number");
  return value;
}

Options parse(int argc, char** argv) {
  Options out;
  bool custom_block = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto next = [&]() -> std::string_view {
      if (++i >= argc) throw std::invalid_argument("missing argument value");
      return argv[i];
    };
    if (arg == "--tapes") out.tapes = next();
    else if (arg == "--library") out.library = next();
    else if (arg == "--refs") out.refs = next();
    else if (arg == "--output") out.output = next();
    else if (arg == "--threads") out.threads = number<int>(next());
    else if (arg == "--seeds") out.seeds = number<int>(next());
    else if (arg == "--block") {
      if (!custom_block) { out.blocks.clear(); custom_block = true; }
      out.blocks.push_back(number<std::uint64_t>(next()));
    }
    else throw std::invalid_argument("unknown argument: " + std::string(arg));
  }
  if (out.tapes.empty() || out.library.empty() || out.refs.empty() || out.output.empty())
    throw std::invalid_argument("--tapes --library --refs --output are required");
  if (out.threads <= 0 || out.seeds <= 0) throw std::invalid_argument("positive sizes required");
  // Supplying --block appends to defaults intentionally only when no custom
  // blocks are needed. Keep the mandated two blocks unique.
  std::sort(out.blocks.begin(), out.blocks.end());
  out.blocks.erase(std::unique(out.blocks.begin(), out.blocks.end()), out.blocks.end());
  return out;
}

const char* op_name(Op op) {
  switch (op) {
    case Op::PASS: return "PASS"; case Op::NORTH: return "NORTH";
    case Op::SOUTH: return "SOUTH"; case Op::EAST: return "EAST";
    case Op::WEST: return "WEST"; case Op::DROP: return "DROP";
    case Op::PICKUP: return "PICKUP"; case Op::PLACE: return "PLACE";
    case Op::PLANT: return "PLANT"; case Op::WATER: return "WATER";
    case Op::HARVEST: return "HARVEST"; case Op::FERTILIZE: return "FERTILIZE";
    case Op::DIG: return "DIG"; case Op::BUILD_COOP: return "BUILD_COOP";
    case Op::BUILD_PASTURE: return "BUILD_PASTURE"; case Op::FEED: return "FEED";
    case Op::COLLECT_FERTILIZER: return "COLLECT_FERTILIZER";
    case Op::CARE: return "CARE"; default: return "MARKET";
  }
}

bool production_op(Op op) {
  return op == Op::PLANT || op == Op::WATER || op == Op::HARVEST ||
         op == Op::FERTILIZE || op == Op::DIG || op == Op::BUILD_COOP ||
         op == Op::BUILD_PASTURE || op == Op::FEED ||
         op == Op::COLLECT_FERTILIZER || op == Op::CARE || op == Op::PLACE;
}

Position actor_position(const Simulator& env, int player, int actor) {
  const auto& farm = env.farms()[player];
  if (actor == 0) return farm.farmer;
  if (actor > 0 && actor <= static_cast<int>(farm.hands.size()))
    return farm.hands[static_cast<std::size_t>(actor - 1)];
  return {-99, -99};
}

const fastkag::Tile* tile_at(const Simulator& env, int player, Position p) {
  const int n = env.config().board_size;
  if (p.x < 0 || p.y < 0 || p.x >= n || p.y >= n) return nullptr;
  return &env.farms()[player].tiles[static_cast<std::size_t>(p.y * n + p.x)];
}

int total_inventory(const fastkag::PrivateState& pr, int actor, int item) {
  if (actor < 0 || actor >= static_cast<int>(pr.inventories.size()) || item < 0 ||
      item >= fastkag::N_ITEMS) return 0;
  return pr.inventories[static_cast<std::size_t>(actor)][static_cast<std::size_t>(item)];
}

bool shed_adjacent(Position p) {
  return (p.x == 4 || p.x == 5) && (p.y == 4 || p.y == 5);
}

bool production_legal(const Simulator& env, int player, int actor, const Action& a,
                      const PlayerAction& whole) {
  if (!production_op(a.op)) return true;
  const auto& farm = env.farms()[player];
  const auto& pr = env.privates()[player];
  if (actor < 0 || actor > static_cast<int>(farm.hands.size())) return false;
  const auto p = actor_position(env, player, actor);
  const auto* tile = tile_at(env, player, p);
  if (!tile) return false;
  const int item = static_cast<int>(a.item);
  if (a.op == Op::PLANT) {
    if (item < 0 || item >= fastkag::N_CROPS || tile->kind != fastkag::TileKind::EMPTY)
      return false;
    int demand = 0;
    for (const auto& other : whole.units)
      demand += other.op == Op::PLANT && other.item == a.item;
    return demand <= pr.seeds[static_cast<std::size_t>(item)];
  }
  if (a.op == Op::WATER)
    return tile->kind == fastkag::TileKind::PLANT && !tile->watered_today;
  if (a.op == Op::HARVEST) {
    if (tile->yield_units <= 0) return false;
    if (tile->kind == fastkag::TileKind::PLANT)
      return env.day() - tile->planted_day >=
          std::array<int, 5>{2, 2, 8, 10, 10}[static_cast<std::size_t>(tile->crop)];
    return tile->kind == fastkag::TileKind::ANIMAL;
  }
  if (a.op == Op::FERTILIZE)
    return tile->kind == fastkag::TileKind::PLANT &&
           total_inventory(pr, actor, static_cast<int>(Item::FERTILIZER)) > 0;
  if (a.op == Op::DIG)
    return tile->kind != fastkag::TileKind::EMPTY &&
           tile->kind != fastkag::TileKind::ANIMAL;
  if (a.op == Op::BUILD_COOP) return tile->kind == fastkag::TileKind::EMPTY;
  if (a.op == Op::BUILD_PASTURE) return tile->kind == fastkag::TileKind::EMPTY;
  if (a.op == Op::FEED)
    return tile->kind == fastkag::TileKind::ANIMAL && !tile->fed_today &&
           total_inventory(pr, actor, static_cast<int>(Item::WHEAT)) > 0;
  if (a.op == Op::COLLECT_FERTILIZER)
    return tile->kind == fastkag::TileKind::ANIMAL && tile->fertilizer_available;
  if (a.op == Op::CARE)
    return tile->kind == fastkag::TileKind::ANIMAL && !tile->cared_today;
  if (a.op == Op::PLACE) {
    if (item >= 9 && item < 12)
      return (tile->kind == fastkag::TileKind::COOP ||
              tile->kind == fastkag::TileKind::PASTURE) &&
             tile->animal == Item::NONE && total_inventory(pr, actor, item) > 0;
    return shed_adjacent(p) && item >= 0 && item < fastkag::N_ITEMS &&
           total_inventory(pr, actor, item) > 0;
  }
  return true;
}

// Exact pre-market shadow of Simulator::step's unit phase. This processes all
// actors in official order and applies the global all-actor PLANT demand gate,
// so legality is not inferred from an actor-isolated snapshot.
std::vector<int> official_unit_legality(const Simulator& env, int player,
                                        const PlayerAction& whole) {
  auto farm = env.farms()[player];
  auto pr = env.privates()[player];
  std::vector<int> legal(whole.units.size(), -1);
  std::array<int, fastkag::N_CROPS> demand{};
  for (const auto& a : whole.units)
    if (a.op == Op::PLANT && static_cast<int>(a.item) >= 0 &&
        static_cast<int>(a.item) < fastkag::N_CROPS)
      ++demand[static_cast<std::size_t>(a.item)];
  std::array<bool, fastkag::N_CROPS> blocked{};
  for (int item = 0; item < fastkag::N_CROPS; ++item)
    blocked[item] = demand[item] > pr.seeds[item];
  auto shed_sum = [&]() {
    return std::accumulate(pr.shed.begin(), pr.shed.end(), 0);
  };
  auto take = [&](int actor, int item, int quantity) {
    if (actor < 0 || actor >= static_cast<int>(pr.inventories.size()) || item < 0 ||
        item >= fastkag::N_ITEMS || quantity <= 0 ||
        pr.inventories[actor][item] < quantity) return false;
    pr.inventories[actor][item] -= quantity;
    return true;
  };
  auto add = [&](int actor, int item, int quantity) {
    if (actor >= 0 && actor < static_cast<int>(pr.inventories.size()) && item >= 0 &&
        item < fastkag::N_ITEMS && quantity > 0)
      pr.inventories[actor][item] += quantity;
  };
  constexpr std::array<int, 5> first_day{2, 2, 8, 10, 10};
  constexpr std::array<fastkag::TileKind, 3> animal_structure{
      fastkag::TileKind::COOP, fastkag::TileKind::PASTURE,
      fastkag::TileKind::PASTURE};
  for (int actor = 0; actor < static_cast<int>(whole.units.size()); ++actor) {
    Action a = whole.units[static_cast<std::size_t>(actor)];
    const bool audited = production_op(a.op);
    if (actor > static_cast<int>(farm.hands.size())) {
      if (audited) legal[actor] = 0;
      continue;
    }
    if (a.op == Op::PLANT && static_cast<int>(a.item) >= 0 &&
        static_cast<int>(a.item) < fastkag::N_CROPS && blocked[static_cast<int>(a.item)]) {
      legal[actor] = 0;
      continue;
    }
    auto& position = actor == 0 ? farm.farmer : farm.hands[actor - 1];
    auto move = [&](int dx, int dy) {
      const int x = position.x + dx, y = position.y + dy;
      if (x >= 0 && y >= 0 && x < env.config().board_size &&
          y < env.config().board_size) position = Position{static_cast<std::int16_t>(x),
                                                           static_cast<std::int16_t>(y)};
    };
    if (a.op == Op::NORTH) { move(0, -1); continue; }
    if (a.op == Op::SOUTH) { move(0, 1); continue; }
    if (a.op == Op::EAST) { move(1, 0); continue; }
    if (a.op == Op::WEST) { move(-1, 0); continue; }
    if (a.op == Op::PASS) continue;
    const int tile_index = position.y * env.config().board_size + position.x;
    auto& tile = farm.tiles[static_cast<std::size_t>(tile_index)];
    const int item = static_cast<int>(a.item);
    const int quantity = std::max(0, a.quantity);
    bool ok = true;
    if (a.op == Op::DROP) {
      ok = shed_adjacent(position);
      if (ok) for (int product : pr.inventory_order[actor]) {
        const int n = std::min(pr.inventories[actor][product],
                               std::max(0, env.config().shed_capacity - shed_sum()));
        pr.shed[product] += n; pr.inventories[actor][product] -= n;
      }
    } else if (a.op == Op::PICKUP) {
      ok = shed_adjacent(position) && item >= 0 && item < fastkag::N_ITEMS &&
           quantity > 0 && pr.shed[item] > 0;
      if (ok) { const int n = std::min(quantity, pr.shed[item]); pr.shed[item] -= n; add(actor,item,n); }
    } else if (a.op == Op::PLACE) {
      if (item >= 9 && item < 12) {
        ok = tile.kind == animal_structure[item - 9] && tile.animal == Item::NONE &&
             take(actor, item, 1);
        if (ok) { tile = {}; tile.kind = fastkag::TileKind::ANIMAL; tile.animal = a.item; tile.placed_day = env.day(); }
      } else {
        ok = shed_adjacent(position) && item >= 0 && item < fastkag::N_ITEMS &&
             quantity > 0 && pr.inventories[actor][item] > 0;
        if (ok) { const int n = std::min({quantity, pr.inventories[actor][item],
                                         std::max(0, env.config().shed_capacity-shed_sum())});
                  pr.inventories[actor][item]-=n; pr.shed[item]+=n; ok = n > 0; }
      }
    } else if (a.op == Op::PLANT) {
      ok = item >= 0 && item < fastkag::N_CROPS &&
           tile.kind == fastkag::TileKind::EMPTY && pr.seeds[item] > 0;
      if (ok) { --pr.seeds[item]; tile = {}; tile.kind=fastkag::TileKind::PLANT;
                tile.crop=a.item; tile.planted_day=env.day(); tile.consecutive_unwatered=1;
                tile.yield_units = item >= 2 && item <= 3 ? 0 : 1; }
    } else if (a.op == Op::WATER) {
      ok = tile.kind == fastkag::TileKind::PLANT && !tile.watered_today;
      if (ok) tile.watered_today = true;
    } else if (a.op == Op::HARVEST) {
      ok = tile.yield_units > 0 &&
          ((tile.kind == fastkag::TileKind::PLANT && static_cast<int>(tile.crop) >= 0 &&
            env.day()-tile.planted_day >= first_day[static_cast<int>(tile.crop)]) ||
           tile.kind == fastkag::TileKind::ANIMAL);
      if (ok) { const int product = tile.kind == fastkag::TileKind::PLANT
                    ? static_cast<int>(tile.crop) : static_cast<int>(tile.animal)-4;
                const bool clears = tile.kind == fastkag::TileKind::PLANT &&
                                    product != 2 && product != 3;
                add(actor, product, tile.yield_units); tile.yield_units=0;
                if (clears) tile = {}; }
    } else if (a.op == Op::FERTILIZE) {
      ok = tile.kind == fastkag::TileKind::PLANT && take(actor, 8, 1);
    } else if (a.op == Op::DIG) {
      ok = tile.kind != fastkag::TileKind::EMPTY && tile.kind != fastkag::TileKind::ANIMAL;
      if (ok) tile = {};
    } else if (a.op == Op::BUILD_COOP) {
      ok = tile.kind == fastkag::TileKind::EMPTY; if (ok) tile.kind=fastkag::TileKind::COOP;
    } else if (a.op == Op::BUILD_PASTURE) {
      ok = tile.kind == fastkag::TileKind::EMPTY; if (ok) tile.kind=fastkag::TileKind::PASTURE;
    } else if (a.op == Op::FEED) {
      ok = tile.kind == fastkag::TileKind::ANIMAL && !tile.fed_today && take(actor,0,1);
      if (ok) tile.fed_today=true;
    } else if (a.op == Op::COLLECT_FERTILIZER) {
      ok = tile.kind == fastkag::TileKind::ANIMAL && tile.fertilizer_available;
      if (ok) { tile.fertilizer_available=false; add(actor,8,1); }
    } else if (a.op == Op::CARE) {
      ok = tile.kind == fastkag::TileKind::ANIMAL && !tile.cared_today;
      if (ok) tile.cared_today=true;
    }
    if (audited) legal[actor] = ok;
  }
  return legal;
}

struct Metrics {
  double own{}, opponent{};
  int unit_failures{}, market_failures{}, overflow{};
  int production_attempts{}, production_legal{}, production_invalid{};
  std::array<int, fastkag::N_CROPS> crop_harvest{};
  int triggers{}, pass_skips{}, productive_skips{}, day_end_truncated{}, crossed_day{};
  int legacy_drop_move{}, legacy_drop_nonmove{};
  std::array<int, 18> skip_ops{};
  int move_invariant_failures{}, coordinate_realign_failures{};
  int source_position_mismatches{};
  int source_sequence_failures{};
  int declined_repairs{};
};

struct TraceStep {
  int step{}, source{};
  Position actual{}, planned{};
  Op raw_op{Op::PASS}, actual_op{Op::PASS};
  int legal{-1}, market_fills{};
};

struct TriggerTrace {
  std::uint64_t seed{};
  int seat{}, policy{}, actor{}, start{}, skip{}, realign{};
  Op skip_op{Op::PASS};
  bool crossed_day{}, day_end_truncated{}, move_ok{}, coordinate_ok{};
  bool source_positions_ok{true};
  bool source_sequence_ok{true};
  bool coordinate_checked{};
  std::vector<TraceStep> steps;
};

struct Active {
  bool active{};
  int trigger_index{-1};
  int start{}, skip{};
  Position planned{};
};

struct Monitor {
  bool active{};
  int trigger_index{-1};
  int day_end{};
  Position planned{};
};

std::optional<weed_audit::Choice> choice_for(
    int policy, const std::vector<PlayerAction>& tape, int step, int actor, Position pos) {
  if (policy == 0) {
    const int skip = std::min(static_cast<int>(tape.size()) - 1, step + 9);
    const auto op = weed_audit::tape_unit(tape, skip, actor).op;
    return weed_audit::Choice{skip, op, skip / 24 != step / 24,
                              skip == (step / 24 + 1) * 24 - 1, op == Op::PASS};
  }
  if (policy == 1) return weed_audit::choose_minimum_loss(tape, step, actor, pos, 16);
  if (policy == 2) return weed_audit::choose_strict_slack(tape, step, actor, 23);
  if (policy == 4) {
    const int skip = std::min(static_cast<int>(tape.size()) - 1,
                              (step / 24 + 1) * 24 - 1);
    const auto op = weed_audit::tape_unit(tape, skip, actor).op;
    return weed_audit::Choice{skip, op, false, true, op == Op::PASS};
  }
  if (policy == 5)
    return weed_audit::choose_tail_preserve_move(tape, step, actor);
  const auto legacy = weed_audit::tape_unit(tape, step + 9, actor).op;
  const int legacy_skip = std::min(static_cast<int>(tape.size()) - 1, step + 9);
  if (g001::repair::is_movement(legacy) || legacy_skip / 24 != step / 24)
    return weed_audit::choose_minimum_loss(tape, step, actor, pos, 16);
  const int skip = legacy_skip;
  return weed_audit::Choice{skip, weed_audit::tape_unit(tape, skip, actor).op,
                            false, skip == (step / 24 + 1) * 24 - 1,
                            weed_audit::tape_unit(tape, skip, actor).op == Op::PASS};
}

Metrics analyze_trace(const std::vector<std::array<PlayerAction, 2>>& trace,
                      std::uint64_t seed, int seat, int policy,
                      const std::vector<PlayerAction>& tape,
                      std::vector<TriggerTrace>& trigger_output) {
  Simulator env({}, seed);
  Metrics out;
  const auto first_trigger = trigger_output.size();
  std::vector<Active> active;
  std::vector<Monitor> monitors;
  for (int step = 0; step < static_cast<int>(trace.size()); ++step) {
    const auto& actions = trace[static_cast<std::size_t>(step)];
    const auto& focal = actions[seat];
    active.resize(focal.units.size());
    monitors.resize(focal.units.size());
    const auto official_legal = official_unit_legality(env, seat, focal);
    const int unit_fail = fastkag::native_macro_unit_failures(env, seat, focal);
    out.unit_failures += unit_fail;
    for (int actor = 0; actor < static_cast<int>(focal.units.size()); ++actor) {
      auto& transaction = active[static_cast<std::size_t>(actor)];
      auto& monitor = monitors[static_cast<std::size_t>(actor)];
      if (transaction.active && step > transaction.skip) transaction.active = false;
      if (monitor.active && step > monitor.day_end) monitor.active = false;
      const auto actual_position = actor_position(env, seat, actor);
      const auto raw = weed_audit::tape_unit(tape, step, actor);
      const auto* tile = tile_at(env, seat, actual_position);
      const bool collision = !transaction.active &&
          (raw.op == Op::PLANT || raw.op == Op::BUILD_PASTURE) && tile &&
          tile->kind == fastkag::TileKind::WEED &&
          (focal.units[actor].op == Op::DIG ||
           (policy == 5 && focal.units[actor].op == raw.op &&
            focal.units[actor].item == raw.item));
      if (collision) {
        const auto choice = choice_for(policy, tape, step, actor, actual_position);
        if (choice || policy == 5) {
          TriggerTrace record;
          record.seed = seed; record.seat = seat; record.policy = policy;
          record.actor = actor; record.start = step;
          record.skip = choice ? choice->skip_step : -1;
          record.realign = choice ? choice->skip_step + 1 : step;
          record.skip_op = choice ? choice->skip_op : Op::PASS;
          record.crossed_day = choice && choice->crossed_day;
          record.day_end_truncated = choice && choice->day_end_truncated;
          record.move_ok = !choice ? true
              : policy == 0 ? !g001::repair::is_movement(choice->skip_op)
              : weed_audit::exact_move_source_invariant(
                    step, choice->skip_step, tape, actor);
          trigger_output.push_back(record);
          const int trigger_index = static_cast<int>(trigger_output.size()) - 1;
          if (choice)
            transaction = {true, trigger_index, step, choice->skip_step, actual_position};
          monitor = {true, trigger_index,
                     std::min(static_cast<int>(tape.size()) - 1,
                              (step / 24 + 1) * 24 - 1), actual_position};
          ++out.triggers;
          out.declined_repairs += !choice;
          const auto legacy_drop = weed_audit::tape_unit(tape, step + 9, actor).op;
          out.legacy_drop_move += g001::repair::is_movement(legacy_drop);
          out.legacy_drop_nonmove += !g001::repair::is_movement(legacy_drop);
          if (choice) {
            out.pass_skips += choice->used_pass;
            out.productive_skips += !choice->used_pass;
            out.crossed_day += choice->crossed_day;
            out.day_end_truncated += choice->day_end_truncated;
            const int skip_op = static_cast<int>(choice->skip_op);
            if (skip_op >= 0 && skip_op < static_cast<int>(out.skip_ops.size()))
              ++out.skip_ops[static_cast<std::size_t>(skip_op)];
          }
        }
      }

      int source = step;
      if (monitor.active) {
        if (transaction.active && step == transaction.start) source = -1;
        else if (step <= transaction.skip) source = step - 1;
        auto& record = trigger_output[static_cast<std::size_t>(monitor.trigger_index)];
        const auto source_action = source >= 0
            ? weed_audit::tape_unit(tape, source, actor) : Action{Op::DIG};
        const int legal = production_op(focal.units[actor].op)
            ? official_legal[static_cast<std::size_t>(actor)] : -1;
        record.steps.push_back({step, source, actual_position, monitor.planned,
                                source_action.op, focal.units[actor].op, legal, 0});
        if (source >= 0 &&
            (focal.units[actor].op != source_action.op ||
             focal.units[actor].item != source_action.item ||
             focal.units[actor].quantity != source_action.quantity))
          record.source_sequence_ok = false;
        if (source >= 0 && g001::repair::is_movement(source_action.op) &&
            focal.units[actor].op != source_action.op)
          record.move_ok = false;
        if (source >= 0 && !g001::repair::is_movement(source_action.op) &&
            (actual_position.x != monitor.planned.x ||
             actual_position.y != monitor.planned.y))
          record.source_positions_ok = false;
        if (source >= 0) {
          if (source_action.op == Op::NORTH) --monitor.planned.y;
          else if (source_action.op == Op::SOUTH) ++monitor.planned.y;
          else if (source_action.op == Op::EAST) ++monitor.planned.x;
          else if (source_action.op == Op::WEST) --monitor.planned.x;
        }
      }

      const auto& action = focal.units[static_cast<std::size_t>(actor)];
      if (production_op(action.op)) {
        ++out.production_attempts;
        const bool legal = official_legal[static_cast<std::size_t>(actor)] > 0;
        out.production_legal += legal;
        out.production_invalid += !legal;
        if (legal && action.op == Op::HARVEST && tile &&
            tile->kind == fastkag::TileKind::PLANT &&
            static_cast<int>(tile->crop) >= 0 && static_cast<int>(tile->crop) < fastkag::N_CROPS)
          out.crop_harvest[static_cast<std::size_t>(tile->crop)] += tile->yield_units;
      }
    }
    env.step(actions);
    out.market_failures += fastkag::native_macro_market_failures(env, seat, focal);
    out.overflow += env.last_end_of_day_overflow()[seat];
    const auto& fills = env.last_market_fills()[seat];
    const int fill_total = std::accumulate(fills.begin(), fills.end(), 0);
    for (std::size_t actor = 0; actor < monitors.size(); ++actor) {
      auto& monitor = monitors[actor];
      if (!monitor.active) continue;
      auto& record = trigger_output[static_cast<std::size_t>(monitor.trigger_index)];
      if (!record.steps.empty() && record.steps.back().step == step)
        record.steps.back().market_fills = fill_total;
      if ((record.skip >= 0 && step == record.skip) ||
          (record.skip < 0 && step == monitor.day_end)) {
        // Coordinate at the beginning of the next aligned source turn. At
        // day close the simulator reset is intentional and is audited rather
        // than silently treated as a MOVE edit.
        const auto next_actual = actor_position(env, seat, record.actor);
        if ((step + 1) % 24 == 0) {
          // Official end_of_day resets farmer to (4,4) and destroys all
          // hands. This is the raw route's next-day lifecycle boundary too.
        record.coordinate_ok = record.actor == 0
              ? next_actual.x == 4 && next_actual.y == 4
              : next_actual.x == -99 && next_actual.y == -99;
        } else {
          record.coordinate_ok = next_actual.x == monitor.planned.x &&
                                 next_actual.y == monitor.planned.y;
        }
        record.coordinate_checked = true;
        out.coordinate_realign_failures += !record.coordinate_ok;
      }
    }
  }
  for (std::size_t index = first_trigger; index < trigger_output.size(); ++index) {
    auto& record = trigger_output[index];
    out.move_invariant_failures += !record.move_ok;
    out.source_position_mismatches += !record.source_positions_ok;
    out.source_sequence_failures += !record.source_sequence_ok;
    if (record.coordinate_checked) continue;
    record.coordinate_ok = false;
    ++out.coordinate_realign_failures;
  }
  out.own = env.farms()[seat].money;
  out.opponent = env.farms()[1 - seat].money;
  return out;
}

fastkag::NativeTapeLibrary load_library(const Options& o,
                                        std::vector<PlayerAction>& g001) {
  fastkag::NativeTapeLibrary out;
  g001 = g001::repair::load_route(o.tapes, o.library, "G001");
  out.routes.push_back(g001);
  out.r5_reference = g001::repair::load_route(o.refs, o.library, "R5");
  out.md_reference = g001::repair::load_route(o.refs, o.library, "MD");
  constexpr std::array<std::string_view, 5> labels{
      "10C4S_3Q", "8C6S_3Q", "6C8S_3Q", "6C12S_4Q_FIRST_YARN",
      "6C12S_4Q_SECOND_YARN"};
  for (std::size_t i = 0; i < labels.size(); ++i) {
    out.moon[i] = g001::repair::load_route(
        o.refs, o.library, "MOON_" + std::string(labels[i]));
    out.moon_legacy[i] = g001::repair::load_route(
        o.refs, o.library, "MOON_LEGACY_" + std::string(labels[i]));
  }
  return out;
}

fastkag::NativeMatchResult play_strict(const fastkag::NativeTeammateExecutor& executor,
                                       const std::vector<PlayerAction>& tape,
                                       std::uint64_t seed, int seat) {
  Simulator env({}, seed);
  std::array<fastkag::NativeAgentState, 2> states;
  fastkag::NativeMatchResult out;
  out.trace.reserve(719);
  while (!env.done()) {
    const int step = env.step_count();
    auto& focal_state = states[seat];
    for (auto& transaction : focal_state.experimental_realign)
      if (transaction.active && step > transaction.skipped_source_step)
        transaction.active = false;
    std::array<PlayerAction, 2> actions{
        executor.action_external(env, 0, 0, states[0]),
        executor.action_external(env, 1, 0, states[1])};
    auto& focal = actions[seat];
    focal_state.experimental_realign.resize(focal.units.size());
    focal_state.weed.resize(focal.units.size());
    for (int actor = 0; actor < static_cast<int>(focal.units.size()); ++actor) {
      auto& weed = focal_state.weed[static_cast<std::size_t>(actor)];
      if (!weed.active || weed.start != step) continue;
      const auto choice = weed_audit::choose_strict_slack(tape, step, actor, 23);
      if (!choice)
        throw std::runtime_error("strict same-day scheduler found no non-MOVE sink");
      focal_state.experimental_realign[static_cast<std::size_t>(actor)] =
          {true, step, choice->skip_step, Action{Op::DIG}};
      weed = {};
    }
    out.trace.push_back(actions);
    env.step(actions);
  }
  out.rewards = {env.farms()[0].money, env.farms()[1].money};
  return out;
}

fastkag::NativeMatchResult play_move_guard(
    const fastkag::NativeTeammateExecutor& executor,
    const std::vector<PlayerAction>& tape, std::uint64_t seed, int seat) {
  Simulator env({}, seed);
  std::array<fastkag::NativeAgentState, 2> states;
  fastkag::NativeMatchResult out;
  out.trace.reserve(719);
  while (!env.done()) {
    const int step = env.step_count();
    auto& focal_state = states[seat];
    for (auto& transaction : focal_state.experimental_realign)
      if (transaction.active && step > transaction.skipped_source_step)
        transaction.active = false;
    std::array<PlayerAction, 2> actions{
        executor.action_external(env, 0, 0, states[0]),
        executor.action_external(env, 1, 0, states[1])};
    auto& focal = actions[seat];
    focal_state.experimental_realign.resize(focal.units.size());
    focal_state.weed.resize(focal.units.size());
    for (int actor = 0; actor < static_cast<int>(focal.units.size()); ++actor) {
      auto& weed = focal_state.weed[static_cast<std::size_t>(actor)];
      if (!weed.active || weed.start != step) continue;
      const auto legacy_drop = weed_audit::tape_unit(tape, step + 9, actor).op;
      const int legacy_skip = std::min(static_cast<int>(tape.size()) - 1, step + 9);
      if (!g001::repair::is_movement(legacy_drop) &&
          legacy_skip / 24 == step / 24) continue;
      const auto choice = weed_audit::choose_minimum_loss(
          tape, step, actor, actor_position(env, seat, actor), 16);
      if (!choice)
        throw std::runtime_error("MOVE guard found no same-day non-MOVE sink");
      focal_state.experimental_realign[static_cast<std::size_t>(actor)] =
          {true, step, choice->skip_step, Action{Op::DIG}};
      weed = {};
    }
    out.trace.push_back(actions);
    env.step(actions);
  }
  out.rewards = {env.farms()[0].money, env.farms()[1].money};
  return out;
}

fastkag::NativeMatchResult play_day_tail(
    const fastkag::NativeTeammateExecutor& executor,
    const std::vector<PlayerAction>& tape, std::uint64_t seed, int seat) {
  Simulator env({}, seed);
  std::array<fastkag::NativeAgentState, 2> states;
  fastkag::NativeMatchResult out;
  out.trace.reserve(719);
  while (!env.done()) {
    const int step = env.step_count();
    auto& focal_state = states[seat];
    for (auto& transaction : focal_state.experimental_realign)
      if (transaction.active && step > transaction.skipped_source_step)
        transaction.active = false;
    std::array<PlayerAction, 2> actions{
        executor.action_external(env, 0, 0, states[0]),
        executor.action_external(env, 1, 0, states[1])};
    auto& focal = actions[seat];
    focal_state.experimental_realign.resize(focal.units.size());
    focal_state.weed.resize(focal.units.size());
    for (int actor = 0; actor < static_cast<int>(focal.units.size()); ++actor) {
      auto& weed = focal_state.weed[static_cast<std::size_t>(actor)];
      if (!weed.active || weed.start != step) continue;
      const int day_end = std::min(static_cast<int>(tape.size()) - 1,
                                   (step / 24 + 1) * 24 - 1);
      focal_state.experimental_realign[static_cast<std::size_t>(actor)] =
          {true, step, day_end, Action{Op::DIG}};
      weed = {};
    }
    out.trace.push_back(actions);
    env.step(actions);
  }
  out.rewards = {env.farms()[0].money, env.farms()[1].money};
  return out;
}

fastkag::NativeMatchResult play_tail_preserve(
    const fastkag::NativeTeammateExecutor& executor,
    const std::vector<PlayerAction>& tape, std::uint64_t seed, int seat) {
  Simulator env({}, seed);
  std::array<fastkag::NativeAgentState, 2> states;
  fastkag::NativeMatchResult out;
  out.trace.reserve(719);
  while (!env.done()) {
    const int step = env.step_count();
    auto& focal_state = states[seat];
    for (auto& transaction : focal_state.experimental_realign)
      if (transaction.active && step > transaction.skipped_source_step)
        transaction.active = false;
    std::array<PlayerAction, 2> actions{
        executor.action_external(env, 0, 0, states[0]),
        executor.action_external(env, 1, 0, states[1])};
    auto& focal = actions[seat];
    focal_state.experimental_realign.resize(focal.units.size());
    focal_state.weed.resize(focal.units.size());
    for (int actor = 0; actor < static_cast<int>(focal.units.size()); ++actor) {
      auto& weed = focal_state.weed[static_cast<std::size_t>(actor)];
      if (!weed.active || weed.start != step) continue;
      const auto choice = weed_audit::choose_tail_preserve_move(tape, step, actor);
      if (choice) {
        focal_state.experimental_realign[static_cast<std::size_t>(actor)] =
            {true, step, choice->skip_step, Action{Op::DIG}};
      } else {
        // The remainder is all MOVE: retain the raw failing macro rather than
        // changing any MOVE source index.
        focal.units[static_cast<std::size_t>(actor)] = weed.intended;
      }
      weed = {};
    }
    out.trace.push_back(actions);
    env.step(actions);
  }
  out.rewards = {env.farms()[0].money, env.farms()[1].money};
  return out;
}

struct CaseResult {
  std::uint64_t block{}, seed{};
  int seat{};
  std::array<Metrics, 6> arm;
  std::vector<TriggerTrace> triggers;
};

struct Aggregate {
  int games{}, triggers{}, pass{}, productive{}, truncated{}, crossed{};
  int declined{};
  int legacy_drop_move{}, legacy_drop_nonmove{};
  int move_fail{}, coord_fail{}, position_fail{}, sequence_fail{};
  int own_negative{}, margin_negative{}, pure_externality{};
  int margin_negative_with_own_loss{}, margin_negative_without_own_loss{};
  int base_w{}, base_t{}, base_l{}, cand_w{}, cand_t{}, cand_l{};
  int lost_wins{}, gained_wins{};
  double own_delta{}, opponent_delta{}, margin_delta{};
  int unit_delta{}, market_delta{}, production_legal_delta{}, production_invalid_delta{};
  int overflow_delta{};
  std::array<int, fastkag::N_CROPS> harvest_delta{};
  std::array<int, 18> skip_ops{};
};

void outcome(double own, double opp, int& w, int& t, int& l) {
  if (own > opp) ++w; else if (own < opp) ++l; else ++t;
}

void write_trigger(std::ostream& os, const TriggerTrace& t) {
  os << "{\"seed\":" << t.seed << ",\"seat\":" << t.seat
     << ",\"policy\":\"" << kPolicyNames[static_cast<std::size_t>(t.policy)]
     << "\",\"actor\":" << t.actor << ",\"start\":" << t.start
     << ",\"skip\":" << t.skip << ",\"skip_op\":\"" << op_name(t.skip_op)
     << "\",\"realign\":" << t.realign
     << ",\"crossed_day\":" << (t.crossed_day ? "true" : "false")
     << ",\"day_end_truncated\":" << (t.day_end_truncated ? "true" : "false")
     << ",\"move_source_invariant\":" << (t.move_ok ? "true" : "false")
     << ",\"source_positions_match\":" << (t.source_positions_ok ? "true" : "false")
     << ",\"source_sequence_exact\":" << (t.source_sequence_ok ? "true" : "false")
     << ",\"coordinate_realign\":" << (t.coordinate_ok ? "true" : "false")
     << ",\"move_source_indices\":[";
  bool first_move = true;
  for (const auto& s : t.steps) {
    if (s.source < 0 || !g001::repair::is_movement(s.raw_op)) continue;
    if (!first_move) os << ',';
    first_move = false;
    os << s.source;
  }
  os << "],\"steps\":[";
  for (std::size_t i = 0; i < t.steps.size(); ++i) {
    if (i) os << ',';
    const auto& s = t.steps[i];
    os << "{\"step\":" << s.step << ",\"raw_source\":" << s.source
       << ",\"actual_xy\":[" << s.actual.x << ',' << s.actual.y
       << "],\"planned_xy\":[" << s.planned.x << ',' << s.planned.y
       << "],\"raw_op\":\"" << op_name(s.raw_op) << "\",\"actual_op\":\""
       << op_name(s.actual_op) << "\",\"production_legal\":";
    if (s.legal < 0) os << "null"; else os << (s.legal ? "true" : "false");
    os << ",\"market_fills\":" << s.market_fills << '}';
  }
  os << "]}\n";
}

}  // namespace

int main(int argc, char** argv) try {
  const auto options = parse(argc, argv);
  std::vector<PlayerAction> tape;
  const fastkag::NativeTeammateExecutor executor(load_library(options, tape));
  std::vector<std::pair<std::uint64_t, int>> tasks;
  for (const auto block : options.blocks)
    for (int offset = 0; offset < options.seeds; ++offset)
      for (int seat = 0; seat < 2; ++seat) tasks.emplace_back(block + offset, seat);
  std::vector<CaseResult> results(tasks.size());
  std::atomic<std::size_t> cursor{};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < std::min<int>(options.threads, tasks.size()); ++worker)
    workers.emplace_back([&] {
      for (;;) {
        const auto index = cursor.fetch_add(1);
        if (index >= tasks.size()) break;
        const auto [seed, seat] = tasks[index];
        auto& result = results[index];
        result.seed = seed; result.seat = seat;
        result.block = *std::max_element(options.blocks.begin(), options.blocks.end(),
            [seed](auto a, auto b) { return std::llabs(static_cast<long long>(seed-a)) >
                                             std::llabs(static_cast<long long>(seed-b)); });
        {
          const auto match = executor.play(0, 0, seed, -1, -1, -1, -1, true);
          result.arm[0] = analyze_trace(
              match.trace, seed, seat, 0, tape, result.triggers);
        }
        {
          fastkag::NativeRepairOptions repair;
          repair.weed_min_loss_realign = true;
          const auto match = executor.play(0, 0, seed, -1, -1, -1, -1, true, true,
                                           false, repair, seat);
          result.arm[1] = analyze_trace(
              match.trace, seed, seat, 1, tape, result.triggers);
        }
        {
          const auto match = play_strict(executor, tape, seed, seat);
          result.arm[2] = analyze_trace(
              match.trace, seed, seat, 2, tape, result.triggers);
        }
        {
          const auto match = play_move_guard(executor, tape, seed, seat);
          result.arm[3] = analyze_trace(
              match.trace, seed, seat, 3, tape, result.triggers);
        }
        {
          const auto match = play_day_tail(executor, tape, seed, seat);
          result.arm[4] = analyze_trace(
              match.trace, seed, seat, 4, tape, result.triggers);
        }
        {
          const auto match = play_tail_preserve(executor, tape, seed, seat);
          result.arm[5] = analyze_trace(
              match.trace, seed, seat, 5, tape, result.triggers);
        }
      }
    });
  for (auto& worker : workers) worker.join();

  std::filesystem::create_directories(options.output);
  std::ofstream detail(std::filesystem::path(options.output) / "triggers.jsonl");
  for (const auto& result : results) for (const auto& trigger : result.triggers)
    write_trigger(detail, trigger);

  std::ofstream cases(std::filesystem::path(options.output) / "cases.jsonl");
  cases << std::fixed << std::setprecision(6);
  for (const auto& r : results) {
    cases << "{\"seed\":" << r.seed << ",\"seat\":" << r.seat << ",\"arms\":[";
    for (int p = 0; p < 6; ++p) {
      if (p) cases << ',';
      const auto& m = r.arm[p];
      cases << "{\"policy\":\"" << kPolicyNames[p] << "\",\"own\":" << m.own
            << ",\"opponent\":" << m.opponent << ",\"unit_failures\":"
            << m.unit_failures << ",\"market_failures\":" << m.market_failures
            << ",\"production_legal\":" << m.production_legal
            << ",\"production_invalid\":" << m.production_invalid
            << ",\"overflow\":" << m.overflow << ",\"crop_harvest\":[";
      for (int i = 0; i < fastkag::N_CROPS; ++i) { if (i) cases << ','; cases << m.crop_harvest[i]; }
      cases << "]}";
    }
    cases << "]}\n";
  }

  std::ofstream summary(std::filesystem::path(options.output) / "summary.json");
  summary << std::fixed << std::setprecision(6);
  summary << "{\n  \"schema\":\"weed_route_invariant_audit_v1\",\n"
          << "  \"seeds_per_block\":" << options.seeds << ",\n  \"blocks\":[";
  for (std::size_t b = 0; b < options.blocks.size(); ++b) {
    if (b) summary << ','; summary << options.blocks[b];
  }
  summary << "],\n  \"paired_seats_per_policy\":" << results.size() << ",\n"
          << "  \"legacy_audit\":[";
  for (std::size_t block_index = 0; block_index < options.blocks.size(); ++block_index) {
    const auto block = options.blocks[block_index];
    Metrics total;
    for (const auto& r : results) {
      if (r.seed < block || r.seed >= block + static_cast<std::uint64_t>(options.seeds))
        continue;
      const auto& m = r.arm[0];
      total.triggers += m.triggers;
      total.legacy_drop_move += m.legacy_drop_move;
      total.legacy_drop_nonmove += m.legacy_drop_nonmove;
      total.move_invariant_failures += m.move_invariant_failures;
      total.coordinate_realign_failures += m.coordinate_realign_failures;
    }
    if (block_index) summary << ',';
    summary << "{\"block\":" << block << ",\"triggers\":" << total.triggers
            << ",\"source9_move\":" << total.legacy_drop_move
            << ",\"source9_nonmove\":" << total.legacy_drop_nonmove
            << ",\"move_invariant_failures\":" << total.move_invariant_failures
            << ",\"coordinate_or_lifecycle_realign_failures\":"
            << total.coordinate_realign_failures << '}';
  }
  summary << "],\n  \"comparisons\":[\n";
  bool first = true;
  for (const auto block : options.blocks) for (int p = 1; p < 6; ++p) {
    Aggregate a;
    for (const auto& r : results) {
      if (r.seed < block || r.seed >= block + static_cast<std::uint64_t>(options.seeds)) continue;
      const auto& base = r.arm[0]; const auto& cand = r.arm[p];
      ++a.games; a.triggers += cand.triggers; a.pass += cand.pass_skips;
      a.declined += cand.declined_repairs;
      a.productive += cand.productive_skips; a.truncated += cand.day_end_truncated;
      a.crossed += cand.crossed_day; a.move_fail += cand.move_invariant_failures;
      a.legacy_drop_move += cand.legacy_drop_move;
      a.legacy_drop_nonmove += cand.legacy_drop_nonmove;
      a.coord_fail += cand.coordinate_realign_failures;
      a.position_fail += cand.source_position_mismatches;
      a.sequence_fail += cand.source_sequence_failures;
      for (std::size_t op = 0; op < a.skip_ops.size(); ++op)
        a.skip_ops[op] += cand.skip_ops[op];
      const double own_delta = cand.own - base.own;
      const double opponent_delta = cand.opponent - base.opponent;
      const double margin_delta = own_delta - opponent_delta;
      a.own_delta += own_delta; a.opponent_delta += opponent_delta; a.margin_delta += margin_delta;
      a.own_negative += own_delta < 0; a.margin_negative += margin_delta < 0;
      a.margin_negative_with_own_loss += margin_delta < 0 && own_delta < 0;
      a.margin_negative_without_own_loss += margin_delta < 0 && own_delta >= 0;
      a.pure_externality += margin_delta < 0 && own_delta >= 0 && opponent_delta > own_delta &&
          cand.production_legal == base.production_legal &&
          cand.crop_harvest == base.crop_harvest;
      outcome(base.own, base.opponent, a.base_w, a.base_t, a.base_l);
      outcome(cand.own, cand.opponent, a.cand_w, a.cand_t, a.cand_l);
      a.lost_wins += base.own > base.opponent && cand.own <= cand.opponent;
      a.gained_wins += base.own <= base.opponent && cand.own > cand.opponent;
      a.unit_delta += cand.unit_failures - base.unit_failures;
      a.market_delta += cand.market_failures - base.market_failures;
      a.production_legal_delta += cand.production_legal - base.production_legal;
      a.production_invalid_delta += cand.production_invalid - base.production_invalid;
      a.overflow_delta += cand.overflow - base.overflow;
      for (int crop = 0; crop < fastkag::N_CROPS; ++crop)
        a.harvest_delta[crop] += cand.crop_harvest[crop] - base.crop_harvest[crop];
    }
    if (!first) summary << ",\n"; first = false;
    summary << "    {\"block\":" << block << ",\"policy\":\"" << kPolicyNames[p]
            << "\",\"games\":" << a.games << ",\"triggers\":" << a.triggers
            << ",\"pass_skips\":" << a.pass << ",\"productive_skips\":" << a.productive
            << ",\"declined_repairs\":" << a.declined
            << ",\"day_end_truncated\":" << a.truncated << ",\"crossed_day\":" << a.crossed
            << ",\"legacy_source9_move\":" << a.legacy_drop_move
            << ",\"legacy_source9_nonmove\":" << a.legacy_drop_nonmove
            << ",\"move_invariant_failures\":" << a.move_fail
            << ",\"coordinate_realign_failures\":" << a.coord_fail
            << ",\"source_position_mismatches\":" << a.position_fail
            << ",\"source_sequence_failures\":" << a.sequence_fail
            << ",\"mean_own_delta\":" << a.own_delta / std::max(1, a.games)
            << ",\"mean_opponent_delta\":" << a.opponent_delta / std::max(1, a.games)
            << ",\"mean_margin_delta\":" << a.margin_delta / std::max(1, a.games)
            << ",\"own_negative_cases\":" << a.own_negative
            << ",\"margin_negative_cases\":" << a.margin_negative
            << ",\"margin_negative_with_own_loss\":"
            << a.margin_negative_with_own_loss
            << ",\"margin_negative_without_own_loss\":"
            << a.margin_negative_without_own_loss
            << ",\"pure_market_externality_negative_cases\":" << a.pure_externality
            << ",\"wtl_baseline\":[" << a.base_w << ',' << a.base_t << ',' << a.base_l
            << "],\"wtl_candidate\":[" << a.cand_w << ',' << a.cand_t << ',' << a.cand_l
            << "],\"score_delta\":"
            << ((a.cand_w + 0.5 * a.cand_t) - (a.base_w + 0.5 * a.base_t)) /
                   std::max(1, a.games)
            << ",\"lost_wins\":" << a.lost_wins
            << ",\"gained_wins\":" << a.gained_wins
            << ",\"unit_failure_delta\":" << a.unit_delta
            << ",\"market_failure_delta\":" << a.market_delta
            << ",\"production_legal_delta\":" << a.production_legal_delta
            << ",\"production_invalid_delta\":" << a.production_invalid_delta
            << ",\"overflow_delta\":" << a.overflow_delta << ",\"skip_op_counts\":{";
    bool first_op = true;
    for (int op = 0; op < static_cast<int>(a.skip_ops.size()); ++op) {
      if (a.skip_ops[static_cast<std::size_t>(op)] == 0) continue;
      if (!first_op) summary << ',';
      first_op = false;
      summary << '\"' << op_name(static_cast<Op>(op)) << "\":"
              << a.skip_ops[static_cast<std::size_t>(op)];
    }
    summary << "},\"crop_harvest_delta\":[";
    for (int crop = 0; crop < fastkag::N_CROPS; ++crop) {
      if (crop) summary << ','; summary << a.harvest_delta[crop];
    }
    summary << "]}";
  }
  summary << "\n  ],\n  \"case_strata\":[\n";
  bool first_stratum = true;
  constexpr std::array<const char*, 5> strata{
      "no_trigger", "pass_only", "has_productive", "day_end_truncated", "not_truncated"};
  for (const auto block : options.blocks) for (int p = 1; p < 6; ++p)
    for (int stratum = 0; stratum < static_cast<int>(strata.size()); ++stratum) {
      int games = 0, own_negative = 0, margin_negative = 0;
      double own_delta = 0, margin_delta = 0;
      for (const auto& r : results) {
        if (r.seed < block || r.seed >= block + static_cast<std::uint64_t>(options.seeds))
          continue;
        const auto& m = r.arm[p];
        const bool selected = stratum == 0 ? m.triggers == 0
            : stratum == 1 ? m.triggers > 0 && m.productive_skips == 0
            : stratum == 2 ? m.productive_skips > 0
            : stratum == 3 ? m.day_end_truncated > 0
            : m.triggers > 0 && m.day_end_truncated == 0;
        if (!selected) continue;
        const double own = m.own - r.arm[0].own;
        const double margin = own - (m.opponent - r.arm[0].opponent);
        ++games; own_delta += own; margin_delta += margin;
        own_negative += own < 0; margin_negative += margin < 0;
      }
      if (!first_stratum) summary << ",\n";
      first_stratum = false;
      summary << "    {\"block\":" << block << ",\"policy\":\""
              << kPolicyNames[p] << "\",\"stratum\":\"" << strata[stratum]
              << "\",\"games\":" << games << ",\"mean_own_delta\":"
              << (games ? own_delta / games : 0.0) << ",\"mean_margin_delta\":"
              << (games ? margin_delta / games : 0.0) << ",\"own_negative\":"
              << own_negative << ",\"margin_negative\":" << margin_negative << '}';
    }
  summary << "\n  ]\n}\n";
  std::cout << "wrote " << options.output << " with " << results.size() << " paired seats\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
