#include "production_obligation.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace production_obligation;

int main() {
  CompilerInput input;
  input.current.step = 0;
  input.current.unlocked_mask = 15;
  input.current.actor_positions = {{4, 4}};
  input.current.carried.resize(1);
  input.current.tiles.assign(100, TileState{});
  input.current.product_price.fill(50);
  for (int step = 0; step < 719; ++step) {
    const int hour = step % 24;
    const int actors = hour == 0 ? 1 : 6;
    UnitFrame frame;
    frame.step = step;
    frame.actor_actions.resize(actors);
    for (int actor = 0; actor < actors; ++actor) {
      frame.actor_actions[actor] = {UnitOp::Pass, Item::None, 1};
    }
    input.future_units.push_back(std::move(frame));
  }

  constexpr int kWarmup = 200;
  constexpr int kRuns = 3000;
  std::uint64_t checksum = 0;
  for (int i = 0; i < kWarmup; ++i) {
    const CompileResult result = compile(input);
    checksum ^= static_cast<std::uint64_t>(result.nodes.size() + result.edges.size());
  }
  std::vector<double> microseconds;
  microseconds.reserve(kRuns);
  for (int i = 0; i < kRuns; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    CompileResult result = compile(input);
    const auto end = std::chrono::steady_clock::now();
    checksum ^= static_cast<std::uint64_t>(result.nodes.size() * 131u +
                                           result.edges.size() * 17u + i);
    microseconds.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  std::sort(microseconds.begin(), microseconds.end());
  const auto percentile = [&](double p) {
    const std::size_t index = static_cast<std::size_t>(p * (microseconds.size() - 1));
    return microseconds[index];
  };
  const double mean = std::accumulate(microseconds.begin(), microseconds.end(), 0.0) /
                      static_cast<double>(microseconds.size());
  std::cout << std::fixed << std::setprecision(3)
            << "scenario=719_steps_daily_6_actors runs=" << kRuns
            << " mean_us=" << mean
            << " p50_us=" << percentile(0.50)
            << " p95_us=" << percentile(0.95)
            << " p99_us=" << percentile(0.99)
            << " max_us=" << microseconds.back()
            << " checksum=" << checksum << '\n';
  return 0;
}
