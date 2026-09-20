#pragma once

#include "dump_scenarios.hpp"
#include "rolling_optimizer.hpp"

#include <cstddef>
#include <vector>

namespace g001::general {

// Every input is either public in the current/past observation or belongs to
// the focal player.  The fixed forecast is compiled from the focal player's
// own production route; realized future state and opponent identity/route are
// deliberately absent from this ABI.
struct Input {
    rolling::CurrentState current;
    rolling::FixedForecast own_causal_forecast;
    std::vector<dump::PublicMarketFrame> public_market_history;
    market::Inventory opponent_stock_lower{};
    market::Inventory opponent_stock_point{};
    market::Inventory opponent_stock_upper{};
    std::vector<int> remaining_sale_windows;
};

struct Mechanisms {
    bool feedback_band{true};       // price target + inventory target + drip
    bool terminal_robust{true};     // belief bands x dump timing scenarios
    bool solvency{true};            // critical purchase/feed requirements
    // Until order probabilities are calibrated, deploy against the pointwise
    // worst causal ordering for our SELL: opponent-first.  The three exact
    // modes remain available through rolling::simulate_terminal for audit.
    bool same_tick_order_robust{true};
};

struct Config {
    Mechanisms mechanisms{};
    rolling::Config rolling{};
};

struct Result {
    rolling::Result plan;
    dump::Result dump_audit;
    std::size_t candidate_count{};
    std::size_t search_candidate_count{};
};

// Single-core deployment envelope.  The optimizer always retains Baseline as
// an exact fallback; the narrow beam bounds work independently of opponent.
[[nodiscard]] Config conservative_deployment_config();

// Enumerates every integer quota/target reachable from current own stock plus
// distinct higher reservation prices causally reachable at future sale windows
// from FixedForecast town drain. It never uses opponent identity, a realized
// future dump, or a product-quantity Cartesian action: each option controls
// exactly one product and the global rolling allocator chooses an option
// sequence.
[[nodiscard]] std::vector<rolling::PersistentOption> candidate_catalog(
    const rolling::CurrentState& current,
    const rolling::FixedForecast& forecast,
    const Mechanisms& mechanisms
);

// Solver catalog derived from causal sufficient statistics (cash and storage
// boundaries, current price plateaus, distinct town-drain recovery prices,
// available sale windows, clear/hold). Recovery candidates are bounded by the
// 24..72 forecast ticks. The full integer catalog above remains the action
// representation and offline audit surface; this reduced set prevents every q
// from entering every beam level.
[[nodiscard]] std::vector<rolling::PersistentOption> search_candidate_catalog(
    const rolling::CurrentState& current,
    const rolling::FixedForecast& forecast,
    const Mechanisms& mechanisms
);

[[nodiscard]] Result plan(const Input& input, const Config& config = {});

}  // namespace g001::general
