#include "adaptive_execution.hpp"
#include "full_market_takeover.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using namespace g001::general_econ;
    ExecutionState state{64, 8, 0, false, 0};
    ObservableExecutionInput input;
    input.product = g001::market::Product::Strawberry;
    input.own_stock = 64;
    input.sale_window = true;
    input.reservation_price = 40;
    input.marginal_impact_limit = 12;
    std::vector<TerminalScenario> scenarios{
        {100, 0, .25}, {80, 8, .25}, {120, 32, .25}, {60, 64, .25}
    };

    constexpr std::uint64_t iterations = 2'000'000;
    std::uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
        input.market_inventory = static_cast<int>(i % 2000);
        input.cash_shortfall = static_cast<std::int64_t>(i % 7);
        const auto proposal = propose(state, input);
        const auto terminal = choose_terminal_timing(
            input.product, 16, input.market_inventory, scenarios
        );
        checksum += static_cast<std::uint64_t>(proposal.requested) +
            static_cast<std::uint64_t>(terminal.timing == TerminalTiming::SellNow);
    }
    const auto stop = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(stop - start).count();
    TakeoverInput takeover;
    takeover.enabled = true;
    takeover.critical_obligations.push_back(
        {MarketOperation::BuySeed, 3, 8, OrderOrigin::CriticalObligation, 12, 10});
    takeover.continuation_orders.push_back(
        {MarketOperation::Sell, 3, 4, OrderOrigin::Continuation, 0, 0});
    SellSettlementInput settlement;
    settlement.previous_shed[3] = 20;
    settlement.current_shed[3] = 16;
    settlement.requested_total_sell[3] = 4;
    settlement.requested_option_sell[3] = 4;
    std::vector<double> latency_us;
    latency_us.reserve(20'000);
    for (int i = 0; i < 20'000; ++i) {
        const auto tick_begin = std::chrono::steady_clock::now();
        const auto fills = infer_sell_fills(settlement);
        const auto queue = compose_takeover_market(takeover);
        checksum += fills.option_fill[3].point + queue.orders.size();
        const auto tick_end = std::chrono::steady_clock::now();
        latency_us.push_back(
            std::chrono::duration<double, std::micro>(tick_end - tick_begin).count());
    }
    std::sort(latency_us.begin(), latency_us.end());
    const auto p99_us = latency_us[static_cast<std::size_t>(latency_us.size() * .99)];
    std::cout << std::fixed << std::setprecision(3)
              << "iterations=" << iterations
              << " elapsed_ms=" << elapsed * 1000.0
              << " ns_per_decision=" << elapsed * 1e9 / static_cast<double>(iterations)
              << " adapter_takeover_p99_us=" << p99_us
              << " checksum=" << checksum << '\n';
}
