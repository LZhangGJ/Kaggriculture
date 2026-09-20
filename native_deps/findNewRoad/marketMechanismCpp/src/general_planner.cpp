#include "general_planner.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace g001::general {
namespace {

using g001::market::Product;

bool same_option(const rolling::PersistentOption& a,
                 const rolling::PersistentOption& b) {
    return std::tie(a.kind, a.product, a.quota, a.window_steps,
                    a.reservation_price, a.target_inventory, a.impact_limit) ==
           std::tie(b.kind, b.product, b.quota, b.window_steps,
                    b.reservation_price, b.target_inventory, b.impact_limit);
}

void append_unique(std::vector<rolling::PersistentOption>& out,
                   rolling::PersistentOption option) {
    if (std::none_of(out.begin(), out.end(), [&](const auto& existing) {
            return same_option(existing, option);
        })) {
        out.push_back(option);
    }
}

rolling::PersistentOption make(option::Kind kind, int product) {
    rolling::PersistentOption out;
    out.kind = kind;
    out.product = product;
    return out;
}

int forecast_window_steps(const rolling::FixedForecast& forecast) {
    if (forecast.ticks.empty()) return 24;
    return std::max(1, forecast.ticks.back().step - forecast.ticks.front().step + 1);
}

// Prices above the current public quote can be reached causally when a future
// FixedForecast tick contains a known town drain.  Only future sale windows
// are inspected: the current observation is exact, and no rival action or
// realized future inventory is used.  At most one candidate is returned per
// distinct public quote, so growth is bounded by the 24..72 tick horizon.
std::vector<int> future_public_recovery_prices(
    const rolling::CurrentState& current,
    const rolling::FixedForecast& forecast,
    int product) {
    std::set<int> prices;
    int inventory = std::max(0, current.market_inventory[product]);
    const auto item = static_cast<Product>(product);
    const int current_price = market::price(item, inventory);
    // Official phase order is market, then town drain.  A drain on tick i is
    // therefore first visible to an order on tick i+1; never let a candidate
    // consume the recovered quote one tick early.
    for (std::size_t index = 0; index + 1 < forecast.ticks.size(); ++index) {
        const auto& tick = forecast.ticks[index];
        inventory = std::max(0, inventory -
            std::max(0, tick.town_drain[static_cast<std::size_t>(product)]));
        if (!forecast.ticks[index + 1].sale_window) continue;
        const int recovered = market::price(item, inventory);
        if (recovered > current_price) prices.insert(recovered);
    }
    return {prices.begin(), prices.end()};
}

void append_recovery_price_targets(
    std::vector<rolling::PersistentOption>& out,
    const rolling::CurrentState& current,
    const rolling::FixedForecast& forecast,
    int product,
    int window_steps) {
    for (const int recovered_price :
         future_public_recovery_prices(current, forecast, product)) {
        auto target = make(option::Kind::PriceTarget, product);
        target.reservation_price = recovered_price;
        // A recovery target may wait at the current low quote. Once reached,
        // sell only marginal units that remain on that recovered price band.
        target.impact_limit = 0;
        target.window_steps = window_steps;
        append_unique(out, target);
    }
}

}  // namespace

Config conservative_deployment_config() {
    Config result;
    result.rolling.beam_width = 4;
    // One option is still a receding-horizon decision: each persistent option
    // is simulated over the complete 24-tick causal forecast and only its
    // first order is committed before replanning.  Expanding a second option
    // here multiplied every live-tick search by roughly four while that second
    // option could never be committed by this call.
    result.rolling.max_decisions = 1;
    result.rolling.threads = 1;
    return result;
}

std::vector<rolling::PersistentOption> candidate_catalog(
    const rolling::CurrentState& current,
    const rolling::FixedForecast& forecast,
    const Mechanisms& mechanisms) {
    std::vector<rolling::PersistentOption> result;
    append_unique(result, make(option::Kind::Baseline, 0));
    const int window_steps = forecast_window_steps(forecast);
    int sale_windows = 0;
    for (const auto& tick : forecast.ticks) sale_windows += tick.sale_window;
    sale_windows = std::max(1, sale_windows);

    for (int product = 0; product < int(market::product_count); ++product) {
        const int stock = std::max(0, current.own_stock[product]);
        if (stock == 0) continue;
        append_unique(result, make(option::Kind::Hold, product));

        if (mechanisms.feedback_band) {
            // All integer quotas are represented.  Drip carries debt forward,
            // which implements pause/resume when a price-target turn sells 0.
            for (int quota = 1; quota <= stock; ++quota) {
                auto drip = make(option::Kind::Drip, product);
                drip.quota = quota;
                drip.window_steps = window_steps;
                append_unique(result, drip);

                // BandDrip is represented by the same persistent Drip ABI
                // with a nonzero reservation price.  Its per-window due is
                // derived from the exact integer quota, avoiding a product
                // quantity Cartesian action space.
                auto band_drip = drip;
                const int due = (quota + sale_windows - 1) / sale_windows;
                band_drip.reservation_price = market::price(
                    static_cast<Product>(product),
                    current.market_inventory[product] + std::max(0, due - 1));
                band_drip.impact_limit = std::max(
                    0,
                    market::price(static_cast<Product>(product),
                                  current.market_inventory[product]) -
                        band_drip.reservation_price);
                append_unique(result, band_drip);
            }
            // Enumerate exact reachable marginal price bands, deduplicating
            // flat regions of the official integer price curve.
            for (int quantity = 0; quantity <= stock; ++quantity) {
                auto target = make(option::Kind::PriceTarget, product);
                target.reservation_price = market::price(
                    static_cast<Product>(product),
                    current.market_inventory[product] + quantity);
                target.impact_limit = std::max(
                    0,
                    market::price(static_cast<Product>(product),
                                  current.market_inventory[product]) -
                        target.reservation_price);
                target.window_steps = window_steps;
                append_unique(result, target);
            }
            append_recovery_price_targets(
                result, current, forecast, product, window_steps);
            for (int target_stock = 0; target_stock < stock; ++target_stock) {
                auto inventory = make(option::Kind::InventoryTarget, product);
                inventory.target_inventory = target_stock;
                inventory.window_steps = window_steps;
                append_unique(result, inventory);
            }
        }

        if (mechanisms.terminal_robust) {
            for (int quota = 1; quota <= stock; ++quota) {
                auto predump = make(option::Kind::PreDump, product);
                predump.quota = quota;
                predump.window_steps = window_steps;
                append_unique(result, predump);
            }
            append_unique(result, make(option::Kind::Clear, product));
        }
    }
    return result;
}

std::vector<rolling::PersistentOption> search_candidate_catalog(
    const rolling::CurrentState& current,
    const rolling::FixedForecast& forecast,
    const Mechanisms& mechanisms) {
    std::vector<rolling::PersistentOption> result;
    append_unique(result, make(option::Kind::Baseline, 0));
    const int window_steps = forecast_window_steps(forecast);
    int sale_windows = 0;
    for (const auto& tick : forecast.ticks) sale_windows += tick.sale_window;
    sale_windows = std::max(1, sale_windows);

    std::int64_t cumulative_cash = 0;
    std::int64_t cash_shortfall = 0;
    int projected_units = std::accumulate(
        current.own_stock.begin(), current.own_stock.end(), 0);
    int peak_units = projected_units;
    for (const auto& tick : forecast.ticks) {
        for (const auto& requirement : tick.requirements)
            cumulative_cash += std::max<std::int64_t>(0, requirement.cash);
        cash_shortfall = std::max(
            cash_shortfall, cumulative_cash - current.own_money);
        projected_units += std::accumulate(
            tick.production.begin(), tick.production.end(), 0);
        for (const auto& requirement : tick.requirements)
            if (requirement.product >= 0 &&
                requirement.product < int(market::product_count))
                projected_units -= std::max(0, requirement.quantity);
        peak_units = std::max(peak_units, projected_units);
    }
    const int overflow_shortfall = std::max(0, peak_units - forecast.shed_capacity);

    for (int product = 0; product < int(market::product_count); ++product) {
        const int stock = std::max(0, current.own_stock[product]);
        if (stock == 0) continue;
        append_unique(result, make(option::Kind::Hold, product));
        std::set<int> quota_set{1, stock, std::min(stock, sale_windows)};

        int cash_quota = 0;
        while (cash_quota < stock &&
               market::sell_revenue(static_cast<Product>(product),
                                    current.market_inventory[product], cash_quota) <
                   cash_shortfall) ++cash_quota;
        if (cash_shortfall > 0) quota_set.insert(std::max(1, cash_quota));
        const int overflow_quota = std::min(stock, overflow_shortfall);
        if (overflow_quota > 0) quota_set.insert(overflow_quota);

        const int initial_price = market::price(
            static_cast<Product>(product), current.market_inventory[product]);
        int plateau = 0;
        int simulated_inventory = current.market_inventory[product];
        while (plateau < stock &&
               market::price(static_cast<Product>(product), simulated_inventory) ==
                   initial_price) {
            if (initial_price > 1) ++simulated_inventory;
            ++plateau;
        }
        quota_set.insert(std::min(stock, std::max(1, plateau * sale_windows)));

        if (mechanisms.feedback_band) {
            for (const int quota : quota_set) {
                auto drip = make(option::Kind::Drip, product);
                drip.quota = quota;
                drip.window_steps = window_steps;
                const int due = (quota + sale_windows - 1) / sale_windows;
                drip.reservation_price = market::price(
                    static_cast<Product>(product),
                    current.market_inventory[product] + std::max(0, due - 1));
                drip.impact_limit = std::max(0, initial_price - drip.reservation_price);
                append_unique(result, drip);

                auto target = make(option::Kind::PriceTarget, product);
                target.reservation_price = market::price(
                    static_cast<Product>(product),
                    current.market_inventory[product] + std::max(0, quota - 1));
                target.impact_limit = std::max(0, initial_price - target.reservation_price);
                target.window_steps = window_steps;
                append_unique(result, target);
            }
            append_recovery_price_targets(
                result, current, forecast, product, window_steps);
            for (const int mandatory : {cash_quota, overflow_quota}) {
                if (mandatory <= 0 || mandatory > stock) continue;
                auto unrestricted = make(option::Kind::Drip, product);
                unrestricted.quota = mandatory;
                unrestricted.window_steps = window_steps;
                append_unique(result, unrestricted);
            }
            if (overflow_quota > 0) {
                auto inventory = make(option::Kind::InventoryTarget, product);
                inventory.target_inventory = stock - overflow_quota;
                inventory.window_steps = window_steps;
                append_unique(result, inventory);
            }
        }
        if (mechanisms.terminal_robust) {
            for (const int quota : quota_set) {
                auto predump = make(option::Kind::PreDump, product);
                predump.quota = quota;
                predump.window_steps = window_steps;
                append_unique(result, predump);
            }
            append_unique(result, make(option::Kind::Clear, product));
        }
    }
    return result;
}

Result plan(const Input& input, const Config& config) {
    if (input.own_causal_forecast.ticks.size() < 24 ||
        input.own_causal_forecast.ticks.size() > 72) {
        throw std::runtime_error("general planner requires a 24..72-step own causal forecast");
    }

    rolling::FixedForecast forecast = input.own_causal_forecast;
    if (!config.mechanisms.solvency) {
        // This is an explicit ablation only.  The actual simulator still
        // audits failures against the untouched baseline after execution.
        for (auto& tick : forecast.ticks) {
            tick.requirements.erase(
                std::remove_if(tick.requirements.begin(), tick.requirements.end(),
                    [](const rolling::Requirement& requirement) {
                        return requirement.kind == rolling::RequirementKind::CriticalPurchase ||
                               requirement.kind == rolling::RequirementKind::Feed;
                    }),
                tick.requirements.end());
        }
    }

    dump::Result dump_audit;
    std::vector<rolling::Scenario> scenarios;
    if (config.mechanisms.terminal_robust) {
        dump::Input dump_input;
        dump_input.public_history = input.public_market_history;
        dump_input.belief_lower = input.opponent_stock_lower;
        dump_input.belief_point = input.opponent_stock_point;
        dump_input.belief_upper = input.opponent_stock_upper;
        dump_input.current_step = input.current.step;
        dump_input.current_day = input.current.step / 24;
        dump_input.steps_remaining = std::max(0, 719 - input.current.step);
        dump_input.horizon_steps = static_cast<int>(forecast.ticks.size());
        dump_input.remaining_sale_windows = input.remaining_sale_windows;
        dump_audit = dump::generate(dump_input);
        scenarios.reserve(dump_audit.scenarios.size());
        for (const auto& value : dump_audit.scenarios) scenarios.push_back(value.scenario);
    } else {
        rolling::Scenario point;
        point.belief_stock = input.opponent_stock_point;
        point.weight = 1.0;
        scenarios.push_back(point);
    }
    if (config.mechanisms.same_tick_order_robust) {
        for (auto& scenario : scenarios)
            scenario.same_tick_order = rolling::SameTickOrder::OpponentFirst;
        // The returned dump audit is the provenance surface consumed by the
        // robust-certificate compiler. Keep it identical to the scenario set
        // actually passed to optimize(); otherwise an actual robust plan would
        // be reported with dump::generate's default ordering and could neither
        // be audited nor certified.
        for (auto& weighted : dump_audit.scenarios)
            weighted.scenario.same_tick_order =
                rolling::SameTickOrder::OpponentFirst;
    }

    const auto full_candidates = candidate_catalog(
        input.current, forecast, config.mechanisms);
    auto candidates = search_candidate_catalog(
        input.current, forecast, config.mechanisms);
    auto rolling_config = config.rolling;
    rolling_config.prefer_fewer_failures_within_envelope =
        config.mechanisms.solvency;
    auto selected = rolling::optimize(
        input.current, forecast, scenarios, candidates, rolling_config);
    return {std::move(selected), std::move(dump_audit),
            full_candidates.size(), candidates.size()};
}

}  // namespace g001::general
