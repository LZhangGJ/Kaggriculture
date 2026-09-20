#include "general_planner.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
using namespace g001;

void run(std::string_view name, rolling::CurrentState state, int ticks,
         int scenario_count, std::size_t beam_width, std::size_t depth,
         int repetitions) {
    rolling::FixedForecast forecast;
    forecast.shed_capacity = 100;
    forecast.liquidate_own_at_end = true;
    forecast.ticks.resize(ticks);
    for (int index = 0; index < ticks; ++index) {
        auto& tick = forecast.ticks[index];
        tick.step = 240 + index;
        tick.sale_window = index % 4 == 0;
        tick.decision_epoch = index == 0 || index == ticks / 3 ||
                              index == 2 * ticks / 3;
        if (index % 12 == 7)
            tick.town_drain[(index / 12) % int(market::product_count)] = 2;
    }
    const auto full_candidates = general::candidate_catalog(state, forecast, {});
    const auto candidates = general::search_candidate_catalog(state, forecast, {});
    std::vector<rolling::Scenario> scenarios(scenario_count);
    for (int s = 0; s < scenario_count; ++s) {
        scenarios[s].weight = 1.0 / scenario_count;
        for (int p = 0; p < int(market::product_count); ++p)
            scenarios[s].belief_stock[p] = (s * 7 + p * 3) % 25;
        if (s % 3 != 0)
            scenarios[s].dumps.push_back(
                {240 + ticks / 2, s % int(market::product_count), 1 + s});
    }
    rolling::Config config;
    config.threads = 1;
    config.beam_width = beam_width;
    config.max_decisions = depth;
    std::cout << name << " starting full_candidates=" << full_candidates.size()
              << " search_candidates=" << candidates.size() << '\n'
              << std::flush;
    std::vector<double> milliseconds;
    std::size_t evaluated = 0;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto begin = std::chrono::steady_clock::now();
        const auto result = rolling::optimize(
            state, forecast, scenarios, candidates, config);
        const auto end = std::chrono::steady_clock::now();
        milliseconds.push_back(
            std::chrono::duration<double, std::milli>(end - begin).count());
        evaluated = result.evaluated_sequences;
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    const auto quantile = [&](double q) {
        return milliseconds[static_cast<std::size_t>(
            q * static_cast<double>(milliseconds.size() - 1))];
    };
    std::cout << std::fixed << std::setprecision(3) << name
              << " ticks=" << ticks << " scenarios=" << scenario_count
              << " candidates=" << candidates.size()
              << " beam=" << beam_width << " depth=" << depth
              << " threads=1 reps=" << repetitions
              << " evaluated=" << evaluated
              << " p50_ms=" << quantile(.50)
              << " p95_ms=" << quantile(.95)
              << " p99_ms=" << quantile(.99)
              << " max_ms=" << milliseconds.back() << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    g001::rolling::CurrentState single;
    single.market_inventory.fill(10000);
    single.own_stock[3] = 100;
    const bool deploy_only = argc > 1 && std::string_view(argv[1]) == "--deploy-only";
    if (!deploy_only) run("stock100-single", single, 72, 15, 64, 3, 20);

    auto spread = single;
    spread.own_stock.fill(11);
    spread.own_stock[0] = 12;
    if (!deploy_only) run("stock100-spread9", spread, 72, 15, 64, 3, 20);

    run("deploy-budgeted", spread, 48, 9, 4, 2, 100);
}
