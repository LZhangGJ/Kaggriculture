#include "clairvoyant_economic_oracle.hpp"
#include "joint_fixed_move_oracle.hpp"

#include "nt_trace_bank.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace jfmo = joint_fixed_move_oracle;
namespace econ = joint_fixed_move_oracle::economic;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using jfmo::Tape;

namespace {

constexpr std::array<int, 5> kFirstDay{2, 2, 8, 10, 10};
constexpr std::array<int, 5> kMaxDay{4, 3, 8, 10, 12};
constexpr std::array<int, 5> kMaxYield{6, 4, 4, 4, 6};
constexpr std::array<int, 5> kSeedCost{10, 20, 50, 100, 80};
constexpr std::array<int, 5> kBasePrice{25, 35, 60, 120, 250};

struct Options {
  std::uint64_t seed_begin{995001};
  int seeds{8};
  int threads{static_cast<int>(
      std::max(1u, std::thread::hardware_concurrency()))};
  int beam{16};
  std::string output{"artifacts/clairvoyant/report.json"};
  std::string tapes{JFMO_TAPES};
  std::string library{JFMO_LIBRARY};
  std::string nt_root{JFMO_NT_ROOT};
};

struct Opponent {
  std::string name;
  const Tape* tape{};
  const public_ports::NtTraceBankAgent* native{};
};

struct OpponentState { public_ports::NtTraceBankState native; };

struct Metrics {
  double own{};
  double opponent{};
  std::int64_t overflow{};
  int move_checks{};
  int move_mismatches{};
  int crop_rewrites{};
  int seed_buys{};
  int sell_orders{};
  int seed_buy_requested{};
  int seed_buy_filled{};
  int plant_actions{};
  int harvest_actions{};
  int water_actions{};
  std::array<int, 5> planted{};
  std::array<int, fastkag::N_ITEMS> terminal_own_shed{};
  std::array<int, fastkag::N_ITEMS> terminal_opponent_shed{};
  std::array<int, fastkag::N_PRODUCTS> terminal_market_inventory{};
  std::array<int, fastkag::N_PRODUCTS> terminal_market_price{};

  [[nodiscard]] int score() const noexcept {
    return own > opponent ? 2 : own == opponent ? 1 : 0;
  }
  [[nodiscard]] double margin() const noexcept { return own - opponent; }
};

enum class CropMode : std::uint8_t {
  Disabled,
  Fixed,
  FutureDemand,
  PerTileCycle,
};
enum class MarketMode : std::uint8_t { Baseline, Legacy, FutureAware };

struct CropConfig {
  CropMode mode{CropMode::Disabled};
  econ::CropPolicy policy;
  int demand_weight{1};
  bool diversify_top_two{};
  // -1 preserves the macro plan; 0..4 choose a fixed crop; 5 chooses the
  // future-demand leader anew for every planting cycle; 6 alternates the
  // future-demand top two by tile.  Used by the coordinate beam only.
  std::array<std::int8_t, 100> per_tile_target = [] {
    std::array<std::int8_t, 100> value{};
    value.fill(-1);
    return value;
  }();
};

struct Candidate {
  CropConfig crop;
  MarketMode market{MarketMode::Baseline};
  jfmo::TradePolicy legacy{jfmo::TradePolicy::Baseline};
  econ::TradePolicy trade;
  int id{};
};

struct Reference {
  Metrics baseline;
  econ::FutureDemand future;
  std::vector<econ::ReferenceFrame> frames;
  std::vector<std::array<bool, 100>> farmland_by_step;
};

struct SearchResult {
  Metrics baseline;
  Metrics legacy;
  Metrics trade;
  Metrics joint;
  Candidate trade_candidate;
  Candidate joint_candidate;
  Metrics best_grid_trade;
  Metrics best_stage_crop;
  Candidate best_stage_candidate;
  int crop_candidates{};
  int trade_candidates{};
  int exact_branches{};
  int farmland_tiles{};
  int late_crop_stationary_slots{};
};

struct Job { int opponent{}; std::uint64_t seed{}; int seat{}; };
struct Row { std::string opponent; std::uint64_t seed{}; int seat{}; SearchResult value; };

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

std::vector<fastkag::Position> positions(const fastkag::Farm& farm) {
  std::vector<fastkag::Position> out{farm.farmer};
  out.insert(out.end(), farm.hands.begin(), farm.hands.end());
  return out;
}

bool crop_slot(Op op) {
  return op == Op::PASS || op == Op::DIG || op == Op::PLANT ||
         op == Op::WATER || op == Op::HARVEST || op == Op::FERTILIZE;
}

bool mature(const fastkag::Tile& tile, int day) {
  const int crop = static_cast<int>(tile.crop);
  return crop >= 0 && crop < 5 && day - tile.planted_day >= kFirstDay[crop];
}

bool harvest_ready(const fastkag::Tile& tile, int day) {
  const int crop = static_cast<int>(tile.crop);
  if (crop < 0 || crop >= 5 || tile.yield_units <= 0) return false;
  // Tomato and strawberry are ongoing crops: harvesting a ready pulse does
  // not destroy the plant.  Annual crops should stay through max_day so the
  // watering windows can accumulate their full yield.
  const bool ongoing = crop == 2 || crop == 3;
  return day - tile.planted_day >= (ongoing ? kFirstDay[crop] : kMaxDay[crop]);
}

bool better(const Metrics& left, const Metrics& right) {
  return std::tuple(left.score(), left.margin(), left.own, -left.overflow) >
         std::tuple(right.score(), right.margin(), right.own, -right.overflow);
}

std::array<int, 5> ranked_future_crops(const fastkag::Simulator& simulator,
                                       const Reference& reference,
                                       int step, int demand_weight) {
  std::array<int, 5> crops{0, 1, 2, 3, 4};
  const std::size_t index = std::min<std::size_t>(
      static_cast<std::size_t>(std::max(0, step)),
      reference.future.suffix_units.size() - 1);
  std::stable_sort(crops.begin(), crops.end(), [&](int left, int right) {
    auto value = [&](int crop) {
      const double demand = reference.future.suffix_units[index][crop];
      const double peak = reference.future.suffix_peak_price[index][crop];
      const double current = simulator.market().prices[crop];
      const double production = double(kMaxYield[crop]) /
                                std::max(1, kFirstDay[crop]);
      // Future refresh demand and its realized reference peak are explicit.
      // Current inventory/price is retained so counterfactual branches can
      // react when their own sales depart from the reference trajectory.
      return production * (current + peak + demand_weight * demand) -
             kSeedCost[crop] / double(std::max(1, kFirstDay[crop]));
    };
    return std::tuple(value(left), -left) > std::tuple(value(right), -right);
  });
  return crops;
}

int crop_target(const CropConfig& config, const fastkag::Simulator& simulator,
                const Reference& reference, int step, int tile) {
  if (config.mode == CropMode::PerTileCycle) {
    const int choice = config.per_tile_target[static_cast<std::size_t>(tile)];
    if (choice >= 0 && choice < 5) return choice;
    const auto ranked = ranked_future_crops(simulator, reference, step,
                                            config.demand_weight);
    if (choice == 5) return ranked[0];
    if (choice == 6) {
      econ::CropPolicy hash_policy = config.policy;
      hash_policy.primary = static_cast<Item>(ranked[0]);
      hash_policy.secondary = static_cast<Item>(ranked[1]);
      hash_policy.primary_share_percent = 50;
      return static_cast<int>(econ::target_for_tile(hash_policy, tile));
    }
    return -1;
  }
  if (config.mode == CropMode::Fixed)
    return static_cast<int>(econ::target_for_tile(config.policy, tile));
  const auto ranked = ranked_future_crops(simulator, reference, step,
                                          config.demand_weight);
  if (!config.diversify_top_two) return ranked[0];
  econ::CropPolicy hash_policy = config.policy;
  hash_policy.primary = static_cast<Item>(ranked[0]);
  hash_policy.secondary = static_cast<Item>(ranked[1]);
  hash_policy.primary_share_percent = 50;
  return static_cast<int>(econ::target_for_tile(hash_policy, tile));
}

void certify_step_moves(const PlayerAction& source, const PlayerAction& value,
                        Metrics& metrics) {
  const std::size_t actors = std::max(source.units.size(), value.units.size());
  for (std::size_t actor = 0; actor < actors; ++actor) {
    const Action left = actor < source.units.size() ? source.units[actor] : Action{};
    const Action right = actor < value.units.size() ? value.units[actor] : Action{};
    if (jfmo::is_move(left.op) || jfmo::is_move(right.op)) {
      ++metrics.move_checks;
      if (!jfmo::action_equal(left, right)) ++metrics.move_mismatches;
    }
  }
}

PlayerAction rewrite_crops(const PlayerAction& source,
                           const fastkag::Simulator& simulator, int player,
                           const Candidate& candidate, const Reference& reference,
                           Metrics& metrics,
                           std::array<int, 5>& planned_targets) {
  auto out = source;
  if (candidate.crop.mode == CropMode::Disabled ||
      simulator.step_count() < candidate.crop.policy.cutoff_step)
    return out;
  const int step = simulator.step_count();
  const int day = simulator.day();
  const auto actor_positions = positions(simulator.farms()[player]);
  std::array<int, 5> plant_requests{};
  for (std::size_t actor = 0;
       actor < out.units.size() && actor < actor_positions.size(); ++actor) {
    const auto original = out.units[actor];
    if (jfmo::is_move(original.op) || !crop_slot(original.op)) continue;
    const auto position = actor_positions[actor];
    const int tile_index = position.y * simulator.config().board_size + position.x;
    if (tile_index < 0 || tile_index >= 100 ||
        !reference.farmland_by_step[static_cast<std::size_t>(step)][tile_index] ||
        (candidate.crop.mode != CropMode::PerTileCycle &&
         !econ::tile_selected(candidate.crop.policy, tile_index)) ||
        (candidate.crop.mode == CropMode::PerTileCycle &&
         candidate.crop.per_tile_target[static_cast<std::size_t>(tile_index)] < 0))
      continue;
    const auto& tile = simulator.farms()[player].tiles[tile_index];
    const int target = crop_target(candidate.crop, simulator, reference,
                                   step, tile_index);
    if (target < 0 || target >= 5) continue;
    Action replacement = original;
    if (tile.kind == fastkag::TileKind::WEED) {
      replacement = {Op::DIG};
      planned_targets[target]++;
    } else if (tile.kind == fastkag::TileKind::EMPTY) {
      planned_targets[target]++;
      const int remaining_days = (simulator.config().episode_steps - 1 - step) /
                                 simulator.config().turns_per_day;
      const int available = simulator.privates()[player].seeds[target];
      if (remaining_days >= kFirstDay[target] + 1 &&
          plant_requests[target] < available) {
        replacement = {Op::PLANT, static_cast<Item>(target), 1};
        ++plant_requests[target];
        ++metrics.planted[target];
      } else {
        replacement = {Op::PASS};
      }
    } else if (tile.kind == fastkag::TileKind::PLANT) {
      const int current = static_cast<int>(tile.crop);
      if (current != target) {
        planned_targets[target]++;
        if (candidate.crop.policy.harvest_before_switch) {
          if (harvest_ready(tile, day))
            replacement = {Op::HARVEST};
          else if (!mature(tile, day)) {
            if (original.op == Op::FERTILIZE)
              replacement = original;
            else if (!tile.watered_today)
              replacement = {Op::WATER};
            else
              replacement = {Op::PASS};
          }
          else
            replacement = {Op::DIG};
        } else {
          replacement = {Op::DIG};
        }
      } else if (original.op == Op::FERTILIZE) {
        replacement = original;
      } else if (harvest_ready(tile, day)) {
        replacement = {Op::HARVEST};
      } else if (!tile.watered_today) {
        replacement = {Op::WATER};
      } else {
        replacement = {Op::PASS};
      }
    } else {
      continue;
    }
    if (!jfmo::action_equal(replacement, original)) ++metrics.crop_rewrites;
    out.units[actor] = replacement;
  }
  return out;
}

void erase_sells(std::vector<Action>& orders) {
  std::erase_if(orders, [](const Action& action) { return action.op == Op::SELL; });
}

PlayerAction rewrite_market(PlayerAction out,
                            const fastkag::Simulator& simulator, int player,
                            const Candidate& candidate, const Reference& reference,
                            const std::array<int, 5>& planned_targets,
                            Metrics& metrics) {
  if (candidate.market == MarketMode::Baseline) return out;
  if (candidate.market == MarketMode::Legacy)
    return jfmo::apply_trade_policy(out, simulator, player, candidate.legacy);
  const int step = simulator.step_count();
  if (step < candidate.trade.cutoff_step) return out;
  if (candidate.trade.baseline_mode == econ::BaselineSellMode::Replace) {
    erase_sells(out.market);
  } else if (candidate.trade.baseline_mode == econ::BaselineSellMode::Gate) {
    const auto& prices = simulator.market().prices;
    const std::size_t fi = std::min<std::size_t>(
        static_cast<std::size_t>(step),
        reference.future.suffix_peak_price.size() - 1);
    std::erase_if(out.market, [&](const Action& action) {
      if (action.op != Op::SELL) return false;
      const int product = static_cast<int>(action.item);
      if (product < 0 || product >= fastkag::N_PRODUCTS) return true;
      const int peak = std::max(1, reference.future.suffix_peak_price[fi][product]);
      return 10000LL * prices[product] <
             1LL * candidate.trade.future_peak_ratio_bps * peak;
    });
  }
  // Keep the macro plan's seed orders.  Unselected fields still execute their
  // baseline production schedule, so removing these orders silently destroys
  // the control arm.  Target-crop seed buffers are additive and use only free
  // market slots.

  if (candidate.crop.mode != CropMode::Disabled) {
    const auto& seeds = simulator.privates()[player].seeds;
    for (int crop = 0; crop < 5 && out.market.size() < 10; ++crop) {
      if (planned_targets[crop] <= 0) continue;
      const int wanted = std::max(candidate.crop.policy.seed_buffer,
                                  planned_targets[crop]);
      if (seeds[crop] < wanted) {
        out.market.push_back({Op::BUY_SEED, static_cast<Item>(crop),
                              wanted - seeds[crop]});
        ++metrics.seed_buys;
      }
    }
  }

  struct Sale { int product{}; int quantity{}; double priority{}; };
  std::vector<Sale> sales;
  const auto& own = simulator.privates()[player].shed;
  const auto& other = simulator.privates()[1 - player].shed;
  const auto& price = simulator.market().prices;
  const std::size_t future_index = std::min<std::size_t>(
      static_cast<std::size_t>(step), reference.future.suffix_peak_price.size() - 1);
  const bool terminal = step >= simulator.config().episode_steps -
                                candidate.trade.terminal_steps;
  for (int product = 0; product < fastkag::N_PRODUCTS; ++product) {
    if (own[product] <= 0) continue;
    const int peak = std::max(1, reference.future.suffix_peak_price[future_index][product]);
    const bool at_value = 10000LL * price[product] >=
                          1LL * candidate.trade.future_peak_ratio_bps * peak;
    if (!terminal && !at_value) continue;
    if (candidate.trade.baseline_mode == econ::BaselineSellMode::Gate &&
        !terminal)
      continue;
    int quantity = terminal ? own[product]
                            : econ::sell_quantity(candidate.trade.pace,
                                                  own[product], other[product]);
    if (quantity <= 0) continue;
    const double demand = reference.future.suffix_units[future_index][product];
    const double priority = double(price[product]) / peak + 0.001 * demand +
                            0.002 * other[product];
    sales.push_back({product, quantity, priority});
  }
  std::stable_sort(sales.begin(), sales.end(), [](const auto& left,
                                                   const auto& right) {
    return std::tuple(left.priority, left.quantity, -left.product) >
           std::tuple(right.priority, right.quantity, -right.product);
  });
  for (const auto& sale : sales) {
    if (out.market.size() >= 10) break;
    out.market.push_back({Op::SELL, static_cast<Item>(sale.product),
                          sale.quantity});
    ++metrics.sell_orders;
  }
  return out;
}

Metrics run_candidate(const Tape& baseline, const Opponent& opponent,
                      std::uint64_t seed, int seat, const Candidate& candidate,
                      const Reference* reference, Reference* record = nullptr) {
  fastkag::Simulator simulator({}, seed);
  OpponentState opponent_state;
  Metrics metrics;
  std::array<bool, 100> known_farmland{};
  if (record) {
    record->frames.resize(simulator.config().episode_steps);
    record->farmland_by_step.resize(simulator.config().episode_steps);
  }
  while (!simulator.done()) {
    const int step = simulator.step_count();
    const auto& source = frame(baseline, step);
    if (record) {
      record->frames[step].prices = simulator.market().prices;
      record->frames[step].shops = simulator.shops();
      const auto actor_positions = positions(simulator.farms()[seat]);
      for (std::size_t actor = 0;
           actor < source.units.size() && actor < actor_positions.size(); ++actor) {
        if (!crop_slot(source.units[actor].op) || jfmo::is_move(source.units[actor].op))
          continue;
        const auto position = actor_positions[actor];
        const int tile = position.y * simulator.config().board_size + position.x;
        if (tile < 0 || tile >= 100) continue;
        const auto kind = simulator.farms()[seat].tiles[tile].kind;
        if (source.units[actor].op == Op::PLANT ||
            source.units[actor].op == Op::WATER ||
            source.units[actor].op == Op::HARVEST ||
            source.units[actor].op == Op::FERTILIZE ||
            kind == fastkag::TileKind::PLANT)
          known_farmland[tile] = true;
      }
      record->farmland_by_step[step] = known_farmland;
    }
    PlayerAction own = source;
    std::array<int, 5> planned_targets{};
    if (reference)
      own = rewrite_crops(source, simulator, seat, candidate, *reference,
                          metrics, planned_targets);
    if (reference)
      own = rewrite_market(std::move(own), simulator, seat, candidate,
                           *reference, planned_targets, metrics);
    certify_step_moves(source, own, metrics);
    if (metrics.move_mismatches != 0)
      throw std::runtime_error("dynamic candidate violated exact MOVE identity");
    std::array<PlayerAction, 2> actions;
    actions[seat] = std::move(own);
    actions[1 - seat] = opponent_action(opponent, opponent_state, simulator,
                                        1 - seat);
    for (const auto& action : actions[seat].units) {
      metrics.plant_actions += action.op == Op::PLANT;
      metrics.harvest_actions += action.op == Op::HARVEST;
      metrics.water_actions += action.op == Op::WATER;
    }
    for (const auto& action : actions[seat].market)
      if (action.op == Op::BUY_SEED)
        metrics.seed_buy_requested += std::max(0, int(action.quantity));
    simulator.step(actions);
    const auto& fills = simulator.last_market_fills()[seat];
    for (std::size_t slot = 0; slot < actions[seat].market.size(); ++slot)
      if (actions[seat].market[slot].op == Op::BUY_SEED && slot < fills.size())
        metrics.seed_buy_filled += fills[slot];
    metrics.overflow += simulator.last_end_of_day_overflow()[seat];
  }
  metrics.own = simulator.farms()[seat].money;
  metrics.opponent = simulator.farms()[1 - seat].money;
  metrics.terminal_own_shed = simulator.privates()[seat].shed;
  metrics.terminal_opponent_shed = simulator.privates()[1 - seat].shed;
  metrics.terminal_market_inventory = simulator.market().inventory;
  metrics.terminal_market_price = simulator.market().prices;
  return metrics;
}

std::vector<econ::TradePolicy> trade_grid() {
  std::vector<econ::TradePolicy> out;
  // Production-only control: preserve every baseline sell, add no economic
  // sell, and merely replace stale seed orders when a crop plan is active.
  out.push_back({220, 1000000, 0, econ::SellPace::Cap4,
                 econ::BaselineSellMode::Augment});
  for (const int cutoff : {0, 72, 144, 220, 288, 360, 432})
    for (const int threshold : {6000, 7000, 8000, 9000, 10000, 11000})
      for (const int terminal : {8, 24, 48, 72, 120})
        for (const auto pace : {econ::SellPace::All, econ::SellPace::Half,
                                econ::SellPace::Cap1, econ::SellPace::Cap2,
                                econ::SellPace::Cap4, econ::SellPace::Cap8,
                                econ::SellPace::OpponentPressure})
          for (const auto mode : {econ::BaselineSellMode::Replace,
                                  econ::BaselineSellMode::Gate,
                                  econ::BaselineSellMode::Augment})
            out.push_back({cutoff, threshold, terminal, pace, mode});
  return out;
}

std::vector<CropConfig> crop_grid() {
  std::vector<CropConfig> out;
  for (const int cutoff : {216, 288, 360, 432, 504, 576})
    for (const int fraction : {10, 25, 50, 75, 100})
      for (const int salt : (fraction == 100 ? std::vector<int>{0}
                                             : std::vector<int>{0, 1, 2, 3}))
       for (const bool harvest_first : {false, true}) {
        for (int crop = 0; crop < 5; ++crop) {
          CropConfig config;
          config.mode = CropMode::Fixed;
          config.policy.cutoff_step = cutoff;
          config.policy.primary = static_cast<Item>(crop);
          config.policy.secondary = static_cast<Item>(crop);
          config.policy.selected_tile_percent = fraction;
          config.policy.tile_selector_salt = salt;
          config.policy.harvest_before_switch = harvest_first;
          out.push_back(config);
        }
        for (const int weight : {0, 1, 2, 4})
          for (const bool diversify : {false, true}) {
            CropConfig config;
            config.mode = CropMode::FutureDemand;
            config.policy.cutoff_step = cutoff;
            config.policy.selected_tile_percent = fraction;
            config.policy.tile_selector_salt = salt;
            config.policy.harvest_before_switch = harvest_first;
            config.demand_weight = weight;
            config.diversify_top_two = diversify;
            out.push_back(config);
          }
      }
  return out;
}

SearchResult search_game(const Tape& baseline, const Opponent& opponent,
                         std::uint64_t seed, int seat, int beam_width) {
  SearchResult out;
  Reference reference;
  Candidate baseline_candidate;
  out.baseline = run_candidate(baseline, opponent, seed, seat,
                               baseline_candidate, nullptr, &reference);
  reference.baseline = out.baseline;
  reference.future = econ::build_future_demand(
      reference.frames, 4, 24);
  std::array<bool, 100> all_known_farmland{};
  for (const auto& snapshot : reference.farmland_by_step)
    for (std::size_t tile = 0; tile < snapshot.size(); ++tile)
      all_known_farmland[tile] = all_known_farmland[tile] || snapshot[tile];
  out.farmland_tiles = static_cast<int>(std::count(
      all_known_farmland.begin(), all_known_farmland.end(), true));
  for (std::size_t step = 216; step < reference.farmland_by_step.size(); ++step) {
    const auto& source = frame(baseline, static_cast<int>(step));
    for (const auto& action : source.units)
      out.late_crop_stationary_slots += crop_slot(action.op) &&
                                        !jfmo::is_move(action.op);
  }

  out.legacy = out.baseline;
  for (const auto policy : {jfmo::TradePolicy::HoldAll,
                            jfmo::TradePolicy::ClearExisting,
                            jfmo::TradePolicy::HoldThenClear360,
                            jfmo::TradePolicy::HoldThenClear480,
                            jfmo::TradePolicy::HoldThenClear600,
                            jfmo::TradePolicy::TerminalClear}) {
    Candidate value;
    value.market = MarketMode::Legacy;
    value.legacy = policy;
    const auto result = run_candidate(baseline, opponent, seed, seat, value,
                                      &reference);
    ++out.exact_branches;
    if (better(result, out.legacy)) out.legacy = result;
  }

  const auto trades = trade_grid();
  out.trade_candidates = static_cast<int>(trades.size());
  out.trade = out.baseline;
  std::optional<Metrics> best_grid_trade;
  struct RankedTrade {
    Metrics metrics;
    econ::TradePolicy trade;
  };
  std::vector<RankedTrade> ranked_trades;
  ranked_trades.reserve(trades.size());
  int candidate_id = 0;
  for (const auto& trade : trades) {
    Candidate value;
    value.id = candidate_id++;
    value.market = MarketMode::FutureAware;
    value.trade = trade;
    const auto result = run_candidate(baseline, opponent, seed, seat, value,
                                      &reference);
    ++out.exact_branches;
    ranked_trades.push_back({result, trade});
    if (!best_grid_trade || better(result, *best_grid_trade))
      best_grid_trade = result;
    if (better(result, out.trade)) {
      out.trade = result;
      out.trade_candidate = value;
    }
  }
  out.best_grid_trade = *best_grid_trade;
  std::stable_sort(ranked_trades.begin(), ranked_trades.end(),
                   [](const auto& left, const auto& right) {
                     return better(left.metrics, right.metrics);
                   });

  const auto crops = crop_grid();
  out.crop_candidates = static_cast<int>(crops.size());
  struct BeamEntry { Metrics metrics; CropConfig crop; econ::TradePolicy trade; };
  std::vector<BeamEntry> stage;
  stage.reserve(crops.size());
  // Two reference-aware trade shapes rank production allocations.  The final
  // stage crosses the retained multi-field/multi-cycle plans with the complete
  // wide quantity/timing grid; no independent gains are added.
  const std::array<econ::TradePolicy, 3> stage_trades{{
      {220, 1000000, 0, econ::SellPace::Cap4,
       econ::BaselineSellMode::Augment},
      {220, 8500, 24, econ::SellPace::Cap8,
       econ::BaselineSellMode::Augment},
      {220, 10000, 72, econ::SellPace::Half,
       econ::BaselineSellMode::Gate},
  }};
  for (const auto& crop : crops) {
    std::optional<Metrics> best_stage;
    econ::TradePolicy best_stage_trade;
    for (const auto& trade : stage_trades) {
      Candidate value;
      value.crop = crop;
      value.market = MarketMode::FutureAware;
      value.trade = trade;
      const auto result = run_candidate(baseline, opponent, seed, seat, value,
                                        &reference);
      ++out.exact_branches;
      if (!best_stage || better(result, *best_stage)) {
        best_stage = result;
        best_stage_trade = trade;
      }
    }
    stage.push_back({*best_stage, crop, best_stage_trade});
  }

  // Per-field coordinate beam.  Each tile independently chooses preserve,
  // one of five fixed crops, or a future-demand target that is recomputed at
  // every new planting cycle.  Every accepted coordinate is selected by a
  // complete terminal game, so combinations are evaluated jointly rather
  // than by adding per-tile rewards.
  std::vector<int> farmland_tiles;
  for (int tile = 0; tile < 100; ++tile)
    if (all_known_farmland[static_cast<std::size_t>(tile)])
      farmland_tiles.push_back(tile);
  const int coordinate_trade_starts =
      std::min<int>(2, static_cast<int>(ranked_trades.size()));
  for (const int cutoff : {216, 288, 360, 432})
    for (const bool harvest_first : {false, true})
      for (int trade_start = 0; trade_start < coordinate_trade_starts;
           ++trade_start) {
        CropConfig plan;
        plan.mode = CropMode::PerTileCycle;
        plan.policy.cutoff_step = cutoff;
        plan.policy.harvest_before_switch = harvest_first;
        plan.demand_weight = 2;
        Candidate working;
        working.crop = plan;
        working.market = MarketMode::FutureAware;
        working.trade = ranked_trades[static_cast<std::size_t>(trade_start)].trade;
        Metrics working_metrics = run_candidate(
            baseline, opponent, seed, seat, working, &reference);
        ++out.exact_branches;
        for (const int tile : farmland_tiles) {
          Metrics tile_best = working_metrics;
          int best_choice = -1;
          for (int choice = 0; choice <= 6; ++choice) {
            Candidate trial = working;
            trial.crop.per_tile_target[static_cast<std::size_t>(tile)] =
                static_cast<std::int8_t>(choice);
            const auto result = run_candidate(baseline, opponent, seed, seat,
                                              trial, &reference);
            ++out.exact_branches;
            ++out.crop_candidates;
            if (better(result, tile_best)) {
              tile_best = result;
              best_choice = choice;
            }
          }
          if (best_choice >= 0) {
            working.crop.per_tile_target[static_cast<std::size_t>(tile)] =
                static_cast<std::int8_t>(best_choice);
            working_metrics = tile_best;
          }
        }
        if (working_metrics.crop_rewrites > 0)
          stage.push_back({working_metrics, working.crop, working.trade});
      }
  std::stable_sort(stage.begin(), stage.end(), [](const auto& left,
                                                   const auto& right) {
    return better(left.metrics, right.metrics);
  });
  out.best_stage_crop = stage.front().metrics;
  out.best_stage_candidate.crop = stage.front().crop;
  out.best_stage_candidate.market = MarketMode::FutureAware;
  out.best_stage_candidate.trade = stage.front().trade;
  if (static_cast<int>(stage.size()) > beam_width) stage.resize(beam_width);

  out.joint = better(out.trade, out.baseline) ? out.trade : out.baseline;
  out.joint_candidate = out.trade_candidate;
  const int retained_trade_profiles =
      std::min<int>(64, static_cast<int>(ranked_trades.size()));
  for (const auto& entry : stage)
    for (int trade_index = 0; trade_index < retained_trade_profiles;
         ++trade_index) {
      const auto& trade =
          ranked_trades[static_cast<std::size_t>(trade_index)].trade;
      Candidate value;
      value.id = candidate_id++;
      value.crop = entry.crop;
      value.market = MarketMode::FutureAware;
      value.trade = trade;
      const auto result = run_candidate(baseline, opponent, seed, seat, value,
                                        &reference);
      ++out.exact_branches;
      if (better(result, out.joint)) {
        out.joint = result;
        out.joint_candidate = value;
      }
    }
  return out;
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
    else if (arg == "--beam") out.beam = std::stoi(next());
    else if (arg == "--output") out.output = next();
    else if (arg == "--tapes") out.tapes = next();
    else if (arg == "--library") out.library = next();
    else if (arg == "--nt-root") out.nt_root = next();
    else throw std::invalid_argument("unknown option " + arg);
  }
  if (out.seeds <= 0 || out.threads <= 0 || out.beam <= 0)
    throw std::invalid_argument("seeds, threads and beam must be positive");
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

struct Aggregate {
  int games{};
  int losses{};
  int loss_to_win{};
  int wins{};
  int wins_retained{};
  double own{};
  double margin{};
  double score{};
};

const Metrics& arm(const SearchResult& value, int index) {
  switch (index) {
    case 0: return value.baseline;
    case 1: return value.legacy;
    case 2: return value.trade;
    default: return value.joint;
  }
}

Aggregate aggregate(const std::vector<Row>& rows, std::string_view opponent,
                    int index) {
  Aggregate out;
  for (const auto& row : rows) {
    if (row.opponent != opponent) continue;
    const auto& baseline = row.value.baseline;
    const auto& value = arm(row.value, index);
    ++out.games;
    out.losses += baseline.score() == 0;
    out.loss_to_win += baseline.score() == 0 && value.score() == 2;
    out.wins += baseline.score() == 2;
    out.wins_retained += baseline.score() == 2 && value.score() == 2;
    out.own += value.own;
    out.margin += value.margin();
    out.score += value.score() / 2.0;
  }
  return out;
}

const char* crop_mode_name(CropMode mode) {
  switch (mode) {
    case CropMode::Disabled: return "disabled";
    case CropMode::Fixed: return "fixed";
    case CropMode::FutureDemand: return "future-demand";
    case CropMode::PerTileCycle: return "per-tile-cycle";
  }
  return "unknown";
}

void write_candidate(std::ostream& out, const Candidate& value) {
  const int assigned_tiles = static_cast<int>(std::count_if(
      value.crop.per_tile_target.begin(), value.crop.per_tile_target.end(),
      [](std::int8_t choice) { return choice >= 0; }));
  out << "{\"crop_mode\":" << json_escape(crop_mode_name(value.crop.mode))
      << ",\"crop_cutoff\":" << value.crop.policy.cutoff_step
      << ",\"primary\":" << static_cast<int>(value.crop.policy.primary)
      << ",\"fraction\":" << value.crop.policy.selected_tile_percent
      << ",\"selector_salt\":" << value.crop.policy.tile_selector_salt
      << ",\"harvest_first\":"
      << (value.crop.policy.harvest_before_switch ? "true" : "false")
      << ",\"demand_weight\":" << value.crop.demand_weight
      << ",\"diversify\":" << (value.crop.diversify_top_two ? "true" : "false")
      << ",\"assigned_tiles\":" << assigned_tiles
      << ",\"trade_threshold_bps\":" << value.trade.future_peak_ratio_bps
      << ",\"terminal_steps\":" << value.trade.terminal_steps
      << ",\"pace\":" << json_escape(econ::sell_pace_name(value.trade.pace))
      << ",\"baseline_sell_mode\":"
      << json_escape(econ::baseline_sell_mode_name(value.trade.baseline_mode))
      << "}";
}

template <typename T, std::size_t N>
void write_numeric_array(std::ostream& out, const std::array<T, N>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) out << ',';
    out << values[index];
  }
  out << ']';
}

}  // namespace

int main(int argc, char** argv) try {
  const Options options = parse_options(argc, argv);
  const auto started = std::chrono::steady_clock::now();
  const auto baseline = g001::repair::load_route(options.tapes, options.library,
                                                  "G001");
  const auto g001 = baseline;
  const auto g096 = g001::repair::load_route(options.tapes, options.library,
                                              "G096");
  const public_ports::NtTraceBankAgent hasegawa(
      (std::filesystem::path(options.nt_root) / bank_filename("hasegawa")).string(),
      public_ports::router_for_slug("hasegawa_current"));
  const public_ports::NtTraceBankAgent rank04(
      (std::filesystem::path(options.nt_root) / bank_filename("rank04")).string(),
      public_ports::router_for_slug("rank04_arman"));
  const std::array<Opponent, 4> opponents{{
      {"G001", &g001, nullptr}, {"G096", &g096, nullptr},
      {"hasegawa", nullptr, &hasegawa}, {"rank04", nullptr, &rank04}}};

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
  const int worker_count = std::min<int>(options.threads, jobs.size());
  for (int worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&] {
      try {
        while (true) {
          const auto index = cursor.fetch_add(1);
          if (index >= jobs.size()) break;
          const auto& job = jobs[index];
          rows[index] = {opponents[job.opponent].name, job.seed, job.seat,
                         search_game(baseline, opponents[job.opponent], job.seed,
                                     job.seat, options.beam)};
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
  report << "{\n  \"schema\": \"joint-clairvoyant-economic-oracle-v2\",\n"
         << "  \"oracle_bound_not_deployable\": true,\n"
         << "  \"bound_kind\": \"lower bound on the fixed-MOVE clairvoyant optimum; exact terminal selection over a finite two-stage beam, not a proof of the global optimum\",\n"
         << "  \"future_information\": \"every realized reference shop set, future demand unit, future reference price peak, current market inventory/price, and exact opponent private shed inventory\",\n"
         << "  \"production_scope\": \"all recognized farmland, all remaining cycles, all five crops including wheat, state-dependent DIG/PLANT/WATER/HARVEST with exact baseline MOVE identity\",\n"
         << "  \"trade_scope\": \"broad refresh-aware timing/quantity grid, then the best 64 trade profiles crossed with crop beams; no sum of independent gains\",\n"
         << "  \"reference_caveat\": \"future refresh and peak features come from the realized baseline trajectory; branch-induced weed RNG can change later shop draws\",\n"
         << "  \"seed_begin\": " << options.seed_begin << ",\n"
         << "  \"seeds_per_opponent\": " << options.seeds << ",\n"
         << "  \"beam_width\": " << options.beam << ",\n"
         << "  \"opponents\": [\"G001\",\"G096\",\"hasegawa\",\"rank04\"],\n"
         << "  \"arms\": {\n";
  const std::array<const char*, 4> names{
      "baseline", "legacy-seven-policy-portfolio", "wide-trade-only",
      "multi-field-multi-cycle-crop+wide-trade"};
  for (int index = 0; index < 4; ++index) {
    report << "    " << json_escape(names[index]) << ": {\n";
    for (std::size_t opponent = 0; opponent < opponents.size(); ++opponent) {
      const auto value = aggregate(rows, opponents[opponent].name, index);
      const double games = std::max(1, value.games);
      report << "      " << json_escape(opponents[opponent].name) << ": {"
             << "\"games\":" << value.games
             << ",\"original_losses\":" << value.losses
             << ",\"loss_to_win\":" << value.loss_to_win
             << ",\"loss_reversal_rate\":"
             << (value.losses ? double(value.loss_to_win) / value.losses : 0.0)
             << ",\"baseline_wins\":" << value.wins
             << ",\"wins_retained\":" << value.wins_retained
             << ",\"own_mean\":" << value.own / games
             << ",\"margin_mean\":" << value.margin / games
             << ",\"score_rate\":" << value.score / games << "}"
             << (opponent + 1 == opponents.size() ? "\n" : ",\n");
    }
    report << "    }" << (index == 3 ? "\n" : ",\n");
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  report << "  },\n  \"jobs\": " << rows.size()
         << ",\n  \"wall_seconds\": " << seconds << "\n}\n";

  std::ofstream detail(options.output + ".rows.jsonl");
  for (const auto& row : rows) {
    detail << "{\"opponent\":" << json_escape(row.opponent)
           << ",\"seed\":" << row.seed << ",\"seat\":" << row.seat
           << ",\"crop_candidates\":" << row.value.crop_candidates
           << ",\"trade_candidates\":" << row.value.trade_candidates
           << ",\"exact_branches\":" << row.value.exact_branches
           << ",\"farmland_tiles\":" << row.value.farmland_tiles
           << ",\"late_crop_stationary_slots\":"
           << row.value.late_crop_stationary_slots
           << ",\"best_grid_trade_margin\":" << row.value.best_grid_trade.margin()
           << ",\"best_grid_trade_own\":" << row.value.best_grid_trade.own
           << ",\"best_stage_crop_margin\":" << row.value.best_stage_crop.margin()
           << ",\"best_stage_crop_own\":" << row.value.best_stage_crop.own
           << ",\"best_stage_crop_rewrites\":" << row.value.best_stage_crop.crop_rewrites
           << ",\"best_stage_seed_buys\":" << row.value.best_stage_crop.seed_buys
           << ",\"best_stage_sell_orders\":" << row.value.best_stage_crop.sell_orders
           << ",\"arms\":[";
    for (int index = 0; index < 4; ++index) {
      const auto& value = arm(row.value, index);
      detail << "{\"name\":" << json_escape(names[index])
             << ",\"own\":" << value.own
             << ",\"opponent_money\":" << value.opponent
             << ",\"margin\":" << value.margin()
             << ",\"score\":" << value.score() / 2.0
             << ",\"move_checks\":" << value.move_checks
             << ",\"move_mismatches\":" << value.move_mismatches
             << ",\"crop_rewrites\":" << value.crop_rewrites
             << ",\"seed_buys\":" << value.seed_buys
             << ",\"seed_buy_requested\":" << value.seed_buy_requested
             << ",\"seed_buy_filled\":" << value.seed_buy_filled
             << ",\"plant_actions\":" << value.plant_actions
             << ",\"harvest_actions\":" << value.harvest_actions
             << ",\"water_actions\":" << value.water_actions
             << ",\"sell_orders\":" << value.sell_orders
             << ",\"terminal_own_shed\":";
      write_numeric_array(detail, value.terminal_own_shed);
      detail << ",\"terminal_opponent_shed\":";
      write_numeric_array(detail, value.terminal_opponent_shed);
      detail << ",\"terminal_market_inventory\":";
      write_numeric_array(detail, value.terminal_market_inventory);
      detail << ",\"terminal_market_price\":";
      write_numeric_array(detail, value.terminal_market_price);
      detail << "}"
             << (index == 3 ? "" : ",");
    }
    detail << "],\"selected_trade\":";
    write_candidate(detail, row.value.trade_candidate);
    detail << ",\"selected_joint\":";
    write_candidate(detail, row.value.joint_candidate);
    detail << ",\"best_stage_candidate\":";
    write_candidate(detail, row.value.best_stage_candidate);
    detail << "}\n";
  }
  std::cout << "clairvoyant economic oracle jobs=" << rows.size()
            << " report=" << options.output << " seconds=" << seconds << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "joint_clairvoyant_economic_eval: " << error.what() << '\n';
  return 2;
}
