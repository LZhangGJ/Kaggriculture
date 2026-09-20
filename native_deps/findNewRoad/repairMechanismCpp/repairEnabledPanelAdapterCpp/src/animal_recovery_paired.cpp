#include "g001_day_start_obligation_issuer.hpp"
#include "native_final_action_commit.hpp"
#include "native_general_market.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;

bool same(Action left, Action right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool same_units(const std::vector<Action>& left,
                const std::vector<Action>& right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin(), same);
}

bool move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
         op == Op::WEST;
}

std::uint64_t generation(std::uint64_t seed, int seat, int step) {
  return (seed << 20U) ^ (static_cast<std::uint64_t>(seat + 1) << 16U) ^
         static_cast<std::uint64_t>(step + 1);
}

fastkag::NativeTapeLibrary tapes() {
  fastkag::NativeTapeLibrary library;
  library.routes.push_back(g001::repair::load_route(
      ANIMAL_PAIRED_TAPES, ANIMAL_PAIRED_LIBRARY, "G001"));
  library.routes.push_back(g001::repair::load_route(
      ANIMAL_PAIRED_TAPES, ANIMAL_PAIRED_LIBRARY, "G096"));
  library.r5_reference = g001::repair::load_route(
      ANIMAL_PAIRED_REFERENCES, ANIMAL_PAIRED_LIBRARY, "R5");
  library.md_reference = g001::repair::load_route(
      ANIMAL_PAIRED_REFERENCES, ANIMAL_PAIRED_LIBRARY, "MD");
  constexpr std::array<const char*, 5> labels{
      "10C4S_3Q", "8C6S_3Q", "6C8S_3Q", "6C12S_4Q_FIRST_YARN",
      "6C12S_4Q_SECOND_YARN"};
  for (std::size_t index = 0; index < labels.size(); ++index) {
    library.moon[index] = g001::repair::load_route(
        ANIMAL_PAIRED_REFERENCES, ANIMAL_PAIRED_LIBRARY,
        "MOON_" + std::string(labels[index]));
    library.moon_legacy[index] = g001::repair::load_route(
        ANIMAL_PAIRED_REFERENCES, ANIMAL_PAIRED_LIBRARY,
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

bool hire_and_animal_buy(const PlayerAction& action) {
  return std::any_of(action.market.begin(), action.market.end(),
                     [](const Action& order) { return order.op == Op::HIRE; }) &&
         std::any_of(action.market.begin(), action.market.end(),
                     [](const Action& order) {
                       return order.op == Op::BUY_ANIMAL;
                     });
}

bool commit_selected(const fastkag::NativeTeammateExecutor& executor,
                     const fastkag::Simulator& env, int player,
                     const fastkag::NativeAgentState& state_before,
                     const PlayerAction& raw, const PlayerAction& selected,
                     int actor, bool owns_market, std::uint64_t owner_generation,
                     std::uint64_t certificate,
                     fastkag::NativeAgentState& committed_state) {
  namespace commit = g001::native_final_commit;
  const commit::Request request{&executor, &env, player, 0, {}, {}};
  const auto proposal = commit::propose(request, state_before);
  if (!commit::same_action(proposal.action, raw)) return false;
  const auto binding = commit::bind_repair_final_action(
      proposal, 1ULL << actor, owns_market, owner_generation, certificate,
      selected);
  committed_state = state_before;
  const auto committed = commit::commit_repair_owner_finalized(
      request, proposal, binding, selected, committed_state);
  return committed.committed &&
         commit::same_action(committed.replayed_action, selected);
}

struct PendingSecond {
  int step{-1};
  int actor{-1};
  int buy_slot{-1};
  Item animal{Item::NONE};
  std::uint64_t obligation_id{};
  int place_step{-1};
  g001::persistent_production::TileKey target{};
  PlayerAction expected_raw;
  PlayerAction selected;
};

struct ActiveChain {
  int actor{-1};
  int place_step{-1};
  Item animal{Item::NONE};
  g001::persistent_production::TileKey target{};
};

struct Result {
  std::uint64_t seed{};
  int seat{};
  double baseline_own{};
  double repair_own{};
  double baseline_opponent{};
  double repair_opponent{};
  int staged{};
  int bound{};
  int zero_fills{};
  int eligible_swaps{};
  int activations{};
  int recovered_pickups{};
  int recoveries{};
  int binder_rejects{};
  int compiler_rejects{};
  int preview_rejects{};
  int first_receipt_failures{};
  int second_commit_rejects{};
  int second_receipt_failures{};
  int place_failures{};
  int funding_sell_orders{};
  int direct_move_edits{};
  int movement_sequence_mismatches{};
  int movement_lane_population_mismatches{};
};

Result run(const fastkag::NativeTeammateExecutor& executor,
           std::uint64_t seed, int seat) {
  namespace binder_ns = g001::day_start_issuer;
  fastkag::Simulator baseline({}, seed);
  fastkag::Simulator repair({}, seed);
  std::array<fastkag::NativeAgentState, 2> baseline_states;
  std::array<fastkag::NativeAgentState, 2> repair_states;
  binder_ns::CrossTickAnimalIntentBinder binder;
  PlayerAction staged_submission;
  int staged_step{-1};
  PendingSecond pending_second;
  std::vector<ActiveChain> chains;
  Result result;
  result.seed = seed;
  result.seat = seat;

  constexpr int lane_stride = 128;
  const int lanes =
      (baseline.config().episode_steps / baseline.config().turns_per_day + 1) *
      lane_stride;
  std::vector<std::vector<int>> baseline_moves(static_cast<std::size_t>(lanes));
  std::vector<std::vector<int>> repair_moves(static_cast<std::size_t>(lanes));
  std::vector<bool> baseline_seen(static_cast<std::size_t>(lanes));
  std::vector<bool> repair_seen(static_cast<std::size_t>(lanes));
  const auto record_moves = [&](const fastkag::Simulator& env,
                                const PlayerAction& action, auto& sequences,
                                auto& seen) {
    for (std::size_t actor = 0; actor < action.units.size(); ++actor) {
      const int lane = env.day() * lane_stride + static_cast<int>(actor);
      if (lane < 0 || lane >= lanes) continue;
      seen[static_cast<std::size_t>(lane)] = true;
      if (move(action.units[actor].op))
        sequences[static_cast<std::size_t>(lane)].push_back(
            static_cast<int>(action.units[actor].op));
    }
  };

  const auto& tape = executor.route_tape(0);
  while (!baseline.done()) {
    std::array<PlayerAction, 2> baseline_actions;
    baseline_actions[seat] = executor.action_external(
        baseline, seat, 0, baseline_states[seat]);
    baseline_actions[1 - seat] = executor.action_external(
        baseline, 1 - seat, 1, baseline_states[1 - seat]);
    record_moves(baseline, baseline_actions[seat], baseline_moves,
                 baseline_seen);
    baseline.step(baseline_actions);

    std::optional<g001::transactional_animal_repair::TypedAnimalObligation>
        obligation;
    if (binder.pending()) {
      const auto bound = binder.bind_next(
          repair, receipt_for(repair, seat, staged_step, staged_submission));
      if (bound.status == binder_ns::CrossTickAnimalStatus::Bound &&
          bound.obligation) {
        ++result.bound;
        if (bound.obligation->acquisition_fill == 0) {
          ++result.zero_fills;
          obligation = *bound.obligation;
        }
      } else {
        ++result.binder_rejects;
      }
    }

    std::array<PlayerAction, 2> repair_actions;
    const auto focal_state_before = repair_states[seat];
    repair_actions[seat] =
        executor.action_external(repair, seat, 0, repair_states[seat]);
    repair_actions[1 - seat] = executor.action_external(
        repair, 1 - seat, 1, repair_states[1 - seat]);
    const int step = repair.step_count();
    bool applied_second = false;
    int animal_before_second = 0;
    ActiveChain next_chain;

    if (pending_second.step >= 0) {
      const auto& fills = repair.last_market_fills()[seat];
      if (step != pending_second.step || pending_second.buy_slot < 0 ||
          pending_second.buy_slot >= static_cast<int>(fills.size()) ||
          fills[static_cast<std::size_t>(pending_second.buy_slot)] != 1) {
        ++result.first_receipt_failures;
      } else if (!g001::native_final_commit::same_action(
                     repair_actions[seat], pending_second.expected_raw)) {
        ++result.second_commit_rejects;
      } else {
        auto committed = focal_state_before;
        if (!commit_selected(
                executor, repair, seat, focal_state_before,
                repair_actions[seat], pending_second.selected,
                pending_second.actor, false, generation(seed, seat, step),
                pending_second.obligation_id, committed)) {
          ++result.second_commit_rejects;
        } else {
          const auto& private_state = repair.privates()[seat];
          animal_before_second =
              private_state.inventories[pending_second.actor]
                                       [static_cast<int>(pending_second.animal)];
          repair_states[seat] = std::move(committed);
          repair_actions[seat] = pending_second.selected;
          applied_second = true;
          next_chain = {pending_second.actor, pending_second.place_step,
                        pending_second.animal, pending_second.target};
        }
      }
      pending_second = {};
    } else if (obligation) {
      const int actor = obligation->actor;
      const bool shape = step + 1 < static_cast<int>(tape.size()) &&
          actor > 0 && actor < static_cast<int>(repair_actions[seat].units.size()) &&
          actor < static_cast<int>(tape[static_cast<std::size_t>(step)].units.size()) &&
          actor < static_cast<int>(tape[static_cast<std::size_t>(step + 1)].units.size()) &&
          same_units(repair_actions[seat].units,
                     tape[static_cast<std::size_t>(step)].units) &&
          same(repair_actions[seat].units[actor],
               {Op::PICKUP, obligation->animal, 1}) &&
          tape[static_cast<std::size_t>(step + 1)].units[actor].op == Op::PICKUP &&
          tape[static_cast<std::size_t>(step + 1)].units[actor].quantity == 1 &&
          tape[static_cast<std::size_t>(step + 1)].units[actor].item !=
              obligation->animal &&
          obligation->place_step > step + 1;
      if (shape) {
        ++result.eligible_swaps;
        auto certified_units = repair_actions[seat].units;
        certified_units[actor] =
            tape[static_cast<std::size_t>(step + 1)].units[actor];
        // ponytail: this research certificate ends at midnight; add a
        // cross-day cash reserve before promoting the mechanism to runtime.
        const int day_end = std::min(
            (repair.day() + 1) * repair.config().turns_per_day - 1,
            static_cast<int>(tape.size()) - 1);
        std::vector<fastkag::NativeFutureUnitFrame> future;
        future.reserve(static_cast<std::size_t>(day_end - step));
        for (int source = step + 1; source <= day_end; ++source) {
          auto units = tape[static_cast<std::size_t>(source)].units;
          if (source == step + 1)
            units[actor] = {Op::PICKUP, obligation->animal, 1};
          future.push_back({source, std::move(units)});
        }
        const auto compiled = fastkag::compile_native_general_market(
            repair, seat, certified_units, future);
        int buy_slot = -1;
        for (std::size_t slot = 0; slot < compiled.market.size(); ++slot) {
          if (same(compiled.market[slot],
                   {Op::BUY_ANIMAL, obligation->animal, 1})) {
            if (buy_slot >= 0) buy_slot = -2;
            else buy_slot = static_cast<int>(slot);
          }
        }
        if (!compiled.feasible || !compiled.audit.compiler_feasible ||
            !compiled.audit.allocator_feasible ||
            !compiled.current_unit_replacements.empty() || buy_slot < 0) {
          ++result.compiler_rejects;
        } else {
          auto selected_first = repair_actions[seat];
          selected_first.units = std::move(certified_units);
          selected_first.market = compiled.market;
          auto committed_first = focal_state_before;
          const bool first_exact = commit_selected(
              executor, repair, seat, focal_state_before,
              repair_actions[seat], selected_first, actor, true,
              generation(seed, seat, step), obligation->id, committed_first);
          bool preview_exact = false;
          PlayerAction expected_second;
          PlayerAction selected_second;
          if (first_exact) {
            auto preview = repair;
            auto preview_states = repair_states;
            preview_states[seat] = committed_first;
            auto preview_actions = repair_actions;
            preview_actions[seat] = selected_first;
            preview.step(preview_actions);
            const auto second_state_before = preview_states[seat];
            expected_second = executor.action_external(
                preview, seat, 0, preview_states[seat]);
            preview_actions[1 - seat] = executor.action_external(
                preview, 1 - seat, 1, preview_states[1 - seat]);
            selected_second = expected_second;
            if (preview.step_count() == step + 1 &&
                same_units(expected_second.units,
                           tape[static_cast<std::size_t>(step + 1)].units) &&
                actor < static_cast<int>(selected_second.units.size())) {
              selected_second.units[actor] =
                  {Op::PICKUP, obligation->animal, 1};
              auto committed_second = second_state_before;
              preview_exact = commit_selected(
                  executor, preview, seat, second_state_before,
                  expected_second, selected_second, actor, false,
                  generation(seed, seat, step + 1), obligation->id,
                  committed_second);
              if (preview_exact) {
                const int before = preview.privates()[seat]
                                       .inventories[actor]
                                                   [static_cast<int>(obligation->animal)];
                preview_actions[seat] = selected_second;
                preview.step(preview_actions);
                preview_exact =
                    preview.privates()[seat]
                            .inventories[actor]
                                        [static_cast<int>(obligation->animal)] ==
                    before + 1;
              }
            }
          }
          if (!first_exact || !preview_exact) {
            ++result.preview_rejects;
          } else {
            for (std::size_t actor_index = 0;
                 actor_index < selected_first.units.size(); ++actor_index) {
              const auto& raw = repair_actions[seat].units[actor_index];
              const auto& selected = selected_first.units[actor_index];
              if (!same(raw, selected) &&
                  (move(raw.op) || move(selected.op)))
                ++result.direct_move_edits;
            }
            repair_states[seat] = std::move(committed_first);
            repair_actions[seat] = std::move(selected_first);
            pending_second = {step + 1,
                              actor,
                              buy_slot,
                              obligation->animal,
                              obligation->id,
                              obligation->place_step,
                              obligation->target,
                              std::move(expected_second),
                              std::move(selected_second)};
            ++result.activations;
            result.funding_sell_orders +=
                compiled.audit.funding_sell_orders;
          }
        }
      }
    }

    if (hire_and_animal_buy(repair_actions[seat]) && !binder.pending()) {
      const auto staged = binder.stage_final_submission(
          repair, seat, repair_actions[seat], tape,
          generation(seed, seat, step));
      if (staged.status == binder_ns::CrossTickAnimalStatus::Staged) {
        staged_submission = repair_actions[seat];
        staged_step = step;
        ++result.staged;
      }
    }

    record_moves(repair, repair_actions[seat], repair_moves, repair_seen);
    repair.step(repair_actions);
    if (applied_second) {
      const int after = repair.privates()[seat]
                            .inventories[next_chain.actor]
                                        [static_cast<int>(next_chain.animal)];
      if (after == animal_before_second + 1) {
        ++result.recovered_pickups;
        chains.push_back(next_chain);
      } else {
        ++result.second_receipt_failures;
      }
    }
    for (auto chain = chains.begin(); chain != chains.end();) {
      if (step < chain->place_step) {
        ++chain;
        continue;
      }
      const auto index = static_cast<std::size_t>(
          chain->target.row * repair.config().board_size +
          chain->target.column);
      const auto& tile = repair.farms()[seat].tiles[index];
      if (step == chain->place_step && tile.kind == fastkag::TileKind::ANIMAL &&
          tile.animal == chain->animal)
        ++result.recoveries;
      else
        ++result.place_failures;
      chain = chains.erase(chain);
    }
  }

  if (pending_second.step >= 0 || !chains.empty())
    ++result.place_failures;
  result.baseline_own = baseline.farms()[seat].money;
  result.repair_own = repair.farms()[seat].money;
  result.baseline_opponent = baseline.farms()[1 - seat].money;
  result.repair_opponent = repair.farms()[1 - seat].money;
  for (int lane = 0; lane < lanes; ++lane) {
    const auto index = static_cast<std::size_t>(lane);
    if (baseline_seen[index] != repair_seen[index])
      ++result.movement_lane_population_mismatches;
    else if (baseline_seen[index] && baseline_moves[index] != repair_moves[index])
      ++result.movement_sequence_mismatches;
  }
  return result;
}

int main_impl(int argc, char** argv) {
  const int seeds = argc > 1 ? std::stoi(argv[1]) : 96;
  const std::uint64_t seed_begin =
      argc > 2 ? std::stoull(argv[2]) : 990000ULL;
  const int requested_threads = argc > 3
      ? std::stoi(argv[3])
      : static_cast<int>(std::thread::hardware_concurrency());
  if (seeds <= 0 || requested_threads <= 0)
    throw std::invalid_argument("seeds and threads must be positive");
  const int games = seeds * 2;
  const fastkag::NativeTeammateExecutor executor(tapes());
  std::vector<Result> results(static_cast<std::size_t>(games));
  std::atomic<int> next{};
  std::exception_ptr failure;
  std::mutex failure_mutex;
  const int threads = std::min(games, requested_threads);
  std::vector<std::jthread> workers;
  workers.reserve(static_cast<std::size_t>(threads));
  for (int worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&] {
      try {
        for (;;) {
          const int index = next.fetch_add(1);
          if (index >= games) break;
          results[static_cast<std::size_t>(index)] = run(
              executor, seed_begin + static_cast<std::uint64_t>(index / 2),
              index % 2);
        }
      } catch (...) {
        std::lock_guard lock(failure_mutex);
        if (!failure) failure = std::current_exception();
      }
    });
  }
  workers.clear();
  if (failure) std::rethrow_exception(failure);

  double baseline_own{};
  double repair_own{};
  double own_delta{};
  double margin_delta{};
  double activated_own_delta{};
  double activated_margin_delta{};
  double worst_own = std::numeric_limits<double>::infinity();
  double worst_margin = std::numeric_limits<double>::infinity();
  int staged{}, bound{}, zero_fills{}, eligible_swaps{}, activations{};
  int recovered_pickups{}, recoveries{}, binder_rejects{};
  int compiler_rejects{}, preview_rejects{}, first_receipt_failures{};
  int second_commit_rejects{}, second_receipt_failures{}, place_failures{};
  int funding_sell_orders{}, direct_move_edits{}, movement_mismatches{};
  int lane_population_mismatches{}, triggered_games{}, activated_games{};
  int baseline_wins{}, repair_wins{}, loss_to_win{}, win_to_loss{};
  int improved{}, worsened{}, unchanged{};
  for (const auto& value : results) {
    const double base_margin = value.baseline_own - value.baseline_opponent;
    const double fixed_margin = value.repair_own - value.repair_opponent;
    const double own = value.repair_own - value.baseline_own;
    const double margin = fixed_margin - base_margin;
    baseline_own += value.baseline_own;
    repair_own += value.repair_own;
    own_delta += own;
    margin_delta += margin;
    worst_own = std::min(worst_own, own);
    worst_margin = std::min(worst_margin, margin);
    baseline_wins += base_margin > 0;
    repair_wins += fixed_margin > 0;
    loss_to_win += base_margin <= 0 && fixed_margin > 0;
    win_to_loss += base_margin > 0 && fixed_margin <= 0;
    improved += own > 0;
    worsened += own < 0;
    unchanged += own == 0;
    triggered_games += value.zero_fills > 0;
    if (value.activations > 0) {
      ++activated_games;
      activated_own_delta += own;
      activated_margin_delta += margin;
    }
    staged += value.staged;
    bound += value.bound;
    zero_fills += value.zero_fills;
    eligible_swaps += value.eligible_swaps;
    activations += value.activations;
    recovered_pickups += value.recovered_pickups;
    recoveries += value.recoveries;
    binder_rejects += value.binder_rejects;
    compiler_rejects += value.compiler_rejects;
    preview_rejects += value.preview_rejects;
    first_receipt_failures += value.first_receipt_failures;
    second_commit_rejects += value.second_commit_rejects;
    second_receipt_failures += value.second_receipt_failures;
    place_failures += value.place_failures;
    funding_sell_orders += value.funding_sell_orders;
    direct_move_edits += value.direct_move_edits;
    movement_mismatches += value.movement_sequence_mismatches;
    lane_population_mismatches +=
        value.movement_lane_population_mismatches;
  }
  const int failures = compiler_rejects + preview_rejects +
      first_receipt_failures + second_commit_rejects +
      second_receipt_failures + place_failures;
  std::cout << "{\"schema\":1,\"default_off\":true,\"focal\":\"G001\","
               "\"opponent\":\"G096\",\"seed_begin\":"
            << seed_begin << ",\"seeds\":" << seeds << ",\"games\":"
            << games << ",\"threads\":" << threads
            << ",\"events\":{\"staged\":" << staged
            << ",\"bound\":" << bound << ",\"zero_fill\":" << zero_fills
            << ",\"eligible_swap\":" << eligible_swaps
            << ",\"activated\":" << activations
            << ",\"recovered_pickup\":" << recovered_pickups
            << ",\"placed\":" << recoveries
            << ",\"funding_sell_orders\":" << funding_sell_orders
            << "},\"failures\":{\"total\":" << failures
            << ",\"binder_noneligible\":" << binder_rejects
            << ",\"compiler\":" << compiler_rejects
            << ",\"preview\":" << preview_rejects
            << ",\"buy_receipt\":" << first_receipt_failures
            << ",\"second_commit\":" << second_commit_rejects
            << ",\"second_receipt\":" << second_receipt_failures
            << ",\"place\":" << place_failures
            << "},\"games_with_zero_fill\":" << triggered_games
            << ",\"games_activated\":" << activated_games
            << ",\"terminal\":{\"baseline_own_mean\":"
            << baseline_own / games << ",\"repair_own_mean\":"
            << repair_own / games << ",\"own_delta_mean\":"
            << own_delta / games << ",\"margin_delta_mean\":"
            << margin_delta / games << ",\"activated_own_delta_mean\":"
            << (activated_games ? activated_own_delta / activated_games : 0.0)
            << ",\"activated_margin_delta_mean\":"
            << (activated_games ? activated_margin_delta / activated_games
                                : 0.0)
            << ",\"worst_own_delta\":" << worst_own
            << ",\"worst_margin_delta\":" << worst_margin
            << ",\"baseline_wins\":" << baseline_wins
            << ",\"repair_wins\":" << repair_wins
            << ",\"loss_to_win\":" << loss_to_win
            << ",\"win_to_loss\":" << win_to_loss
            << ",\"improved\":" << improved << ",\"worsened\":"
            << worsened << ",\"unchanged\":" << unchanged
            << "},\"activated_cases\":[";
  int emitted_cases = 0;
  for (const auto& value : results) {
    if (value.activations == 0) continue;
    if (emitted_cases != 0) std::cout << ',';
    const double base_margin = value.baseline_own - value.baseline_opponent;
    const double fixed_margin = value.repair_own - value.repair_opponent;
    std::cout << "{\"seed\":" << value.seed << ",\"seat\":" << value.seat
              << ",\"own_delta\":" << value.repair_own - value.baseline_own
              << ",\"margin_delta\":" << fixed_margin - base_margin << '}';
    // ponytail: cap detail rows; aggregate counters remain complete.
    if (++emitted_cases == 8) break;
  }
  std::cout << "],\"movement\":{\"direct_move_edits\":"
            << direct_move_edits << ",\"sequence_mismatch_lanes\":"
            << movement_mismatches
            << ",\"population_mismatch_lanes\":"
            << lane_population_mismatches << "}}\n";
  if (direct_move_edits != 0)
    throw std::runtime_error("two-hand recovery edited a MOVE directly");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "animal_recovery_paired: " << error.what() << '\n';
    return 1;
  }
}
