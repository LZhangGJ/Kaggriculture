// Diagnostic-only native R1 self-play runner.  It does not enter the submitted
// agent.  Build with the same feature macros as policy/r1/build.sh, for example:
// g++ -std=c++20 -O3 -DNDEBUG -fopenmp -pthread \
//   -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 \
//   -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 \
//   -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=0 \
//   -DR2_SALE_CLOCK_MODE=0 -DP16_WORKING_CAPITAL_GATE=1 \
//   -DP16_LIVE_REMAINING_VALUE=1 -DT3_OBLIGATION_REPAIR=1 \
//   -DT3_CAPACITY_REPAIR=1 -DT3_RECEIPT_REPAIR=1 \
//   -DP16_JOINT_BUNDLES=1 -Ipolicy/r1 \
//   experiments/benchmark_r1_cpp_selfplay.cpp \
//   policy/r1/executor/vendor/simulator.cpp -o /tmp/benchmark_r1_cpp_selfplay

#include "search.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace {

triad::Settings deployed_settings() {
  triad::Settings s;
  s.competition = 2;
  s.supply = .85;
  s.future_shop = .7;
  s.capital_power = .4;
  s.labor_hours = 10;
  s.work_price = 4;
  s.animal_work = 1;
  s.reserve = 120;
  s.max_animals = 40;
  s.max_hands = 14;
  s.max_land = 3;
  s.feed_cover = 1;
  s.rotation = 0;
  s.preview = 1;
  s.delivery = 2;
  s.intraday = 1;
  s.service = 1;
  s.replant = .5;
  s.land_rent = 2;
  s.discount = .005;
  s.tour_dp = 0;
  s.layout = 0;
  s.repeat = 0;
  s.animal_bias = 1;
  s.crop_bias = 1;
  s.portfolio_passes = 6;
  s.crop_fert = 1;
  s.harvest_threshold = 1;
  s.delay_sale = 0;
  s.opening_budget = 1;
  s.scenario = 1;
  s.keep_commitments = 1;
  s.candidate_extra = 1;
  s.service_reconcile = 2;
  s.live_ledger = 0;
  s.delivery_calendar = 0;
  s.feed_finance = 0;
  s.batch_delivery = 1;
  return s;
}

dp7::View view(const fastkag::Simulator& env, int player) {
  return {env.step_count(), env.day(), env.hour(), env.farms()[player],
          env.farms()[1 - player], env.privates()[player], env.market(),
          env.shops()};
}

uint64_t mix(uint64_t hash, const fastkag::Action& action) {
  for (int value : {int(action.op), int(action.item), action.quantity}) {
    hash ^= uint32_t(value);
    hash *= 1099511628211ULL;
  }
  return hash;
}

struct Result {
  double left = 0, right = 0;
  uint64_t action_hash = 1469598103934665603ULL;
  int steps = 0;
};

Result play(uint64_t seed, int max_steps) {
  fastkag::Simulator env({}, seed);
  const auto settings = deployed_settings();
  triad::SearchController policies[2] = {
      triad::SearchController(settings), triad::SearchController(settings)};
  Result result;
  while (!env.done() && result.steps < max_steps) {
    std::array<fastkag::PlayerAction, 2> actions;
    for (int player = 0; player < 2; ++player) {
      actions[player] = policies[player].act(view(env, player));
      for (const auto& action : actions[player].units)
        result.action_hash = mix(result.action_hash, action);
      for (const auto& action : actions[player].market)
        result.action_hash = mix(result.action_hash, action);
    }
    env.step(actions);
    ++result.steps;
  }
  result.left = env.farms()[0].money;
  result.right = env.farms()[1].money;
  return result;
}

int integer(const char* text, const char* name, int minimum) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!end || *end || value < minimum || value > INT32_MAX)
    throw std::invalid_argument(std::string("invalid ") + name);
  return int(value);
}

}  // namespace

int main(int argc, char** argv) try {
  int games = 8, threads = omp_get_max_threads(), steps = 719;
  uint64_t seed = 3800000000ULL;
  bool self_check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--self-check") self_check = true;
    else if (arg == "--games" && i + 1 < argc) games = integer(argv[++i], "games", 1);
    else if (arg == "--threads" && i + 1 < argc) threads = integer(argv[++i], "threads", 1);
    else if (arg == "--steps" && i + 1 < argc) steps = integer(argv[++i], "steps", 1);
    else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
    else throw std::invalid_argument("unknown or incomplete argument: " + arg);
  }
  if (steps > 719) throw std::invalid_argument("steps must be <= 719");
  if (self_check) {
    const auto a = play(seed, 2), b = play(seed, 2);
    if (a.steps != 2 || a.action_hash != b.action_hash || a.left != b.left ||
        a.right != b.right)
      throw std::runtime_error("determinism check failed");
    std::cout << "{\"status\":\"PASS\",\"steps\":2,\"action_hash\":"
              << a.action_hash << "}\n";
    return 0;
  }

  omp_set_num_threads(threads);
  std::vector<Result> results(games);
  const auto begin = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
  for (int game = 0; game < games; ++game)
    results[game] = play(seed + game, steps);
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  uint64_t hash = 1469598103934665603ULL;
  double left = 0, right = 0;
  int completed_steps = 0;
  for (const auto& result : results) {
    hash ^= result.action_hash;
    hash *= 1099511628211ULL;
    left += result.left;
    right += result.right;
    completed_steps += result.steps;
  }
  std::cout << std::setprecision(17)
            << "{\"games\":" << games << ",\"threads\":" << threads
            << ",\"steps\":" << completed_steps
            << ",\"seconds\":" << seconds
            << ",\"games_per_second\":" << games / seconds
            << ",\"decisions_per_second\":" << 2. * completed_steps / seconds
            << ",\"mean_left_cash\":" << left / games
            << ",\"mean_right_cash\":" << right / games
            << ",\"action_hash\":" << hash << "}\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "benchmark_r1_cpp_selfplay: " << error.what() << '\n';
  return 2;
}
