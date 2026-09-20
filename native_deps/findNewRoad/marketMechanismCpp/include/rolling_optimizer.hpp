#pragma once

#include "market.hpp"
#include "persistent_options.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace g001::rolling {

using Inventory = g001::market::Inventory;
// Signed own-shed changes.  Positive values enter the shed and negative values
// leave it.  Separate phase fields are required because official unit actions
// precede the market while end-of-day carried returns follow it.
using StockDelta = std::array<int, g001::market::product_count>;

enum class RequirementKind : std::uint8_t { CriticalPurchase, Feed, RouteHard };

struct Requirement {
    RequirementKind kind{RequirementKind::CriticalPurchase};
    std::int64_t cash{};
    int product{-1};
    int quantity{};
};

struct ForecastTick {
    int step{};
    bool sale_window{};
    bool decision_epoch{};
    // Validated before pre_market_unit_delta is applied.  These requirements
    // reserve stock for route-fixed PICKUP operations without subtracting it a
    // second time.
    std::vector<Requirement> pre_market_requirements;
    StockDelta pre_market_unit_delta{};
    Inventory production{};
    // Deterministic shed changes caused by the market phase, principally
    // production-critical BUY_PRODUCT obligations.  Purchase cash remains in
    // requirements.
    StockDelta post_market_delta{};
    // Potential carried-product return after the market.  Capacity/overflow is
    // evaluated only after this phase, so it cannot be sold one tick early.
    StockDelta end_of_day_return{};
    Inventory baseline_sale{};
    // Exact caller-selected legacy queue in official slot order.  When this
    // is present it is the authoritative market phase: non-SELL purchases are
    // executed at their original slot, and inferred cash requirements /
    // post_market_delta are not replayed a second time.  Empty preserves the
    // older aggregate-forecast ABI.
    std::vector<g001::market::Order> legacy_market_orders;
    bool legacy_market_queue_known{};
    Inventory town_drain{};
    std::vector<Requirement> requirements;
};

struct FixedForecast {
    // Dynamic carrier. optimize() currently accepts 24--72 ticks; independent
    // causal consumers such as Generation-3 may carry up to 96.
    std::vector<ForecastTick> ticks;
    int shed_capacity{100};
    // Ancillary private state required by the official queue simulator. Money
    // and product shed are overwritten from CurrentState at simulation start.
    g001::market::PlayerMarketState initial_own_market{};
    int official_shed_capacity{100};
    int hire_cost_multiplier{1};
    int turns_per_day{24};
    // Terminal mark-to-market for a receding horizon.  Both farms' remaining
    // causal inventories are liquidated, opponent first, so rival stock is
    // not assigned zero value while our own stock is counted as an asset.
    bool liquidate_own_at_end{};
};

struct CurrentState {
    int step{};
    std::int64_t own_money{};
    std::int64_t opponent_money{}; // public farm balance
    Inventory own_stock{};
    Inventory market_inventory{};
};

struct OpponentDump {
    int step{};
    int product{};
    int quantity{};
};

enum class SameTickOrder : std::uint8_t {
    OwnFirst,
    OpponentFirst,
    Lockstep,
};

// Scenarios must be constructed from public belief lower/point/upper and public
// dump windows only. realized_future is accepted solely by tests/evaluators and
// is rejected by optimize().
struct Scenario {
    Inventory belief_stock{};
    std::vector<OpponentDump> dumps;
    double weight{1.0};
    bool realized_future{};
    SameTickOrder same_tick_order{SameTickOrder::OwnFirst};
};

struct PersistentOption {
    g001::option::Kind kind{g001::option::Kind::Baseline};
    int product{};
    int quota{};
    int window_steps{24};
    int reservation_price{};
    int target_inventory{};
    int impact_limit{std::numeric_limits<int>::max()};
};

struct FailureCounts {
    int critical_purchase{};
    int feed{};
    int overflow{};
    int route_hard{};
};

struct FailureRisk {
    int worst{};
    double affected_scenario_fraction{};
    double mean_severity{};              // unconditional expectation
    double conditional_severity{};       // mean given the failure occurs
};

struct FailureRiskProfile {
    FailureRisk critical_purchase, feed, overflow, route_hard;
};

struct RobustScore {
    FailureCounts worst_failures{};
    bool nonnegative_margin_feasible{};
    std::int64_t worst_margin{};
    // Minimum paired candidate-minus-Baseline margin over the exact same
    // causal scenarios. Independent worst-case scalars cannot prove this.
    std::int64_t worst_margin_delta_vs_baseline{};
    double cvar_margin{};
    double expected_margin{};
    std::int64_t worst_own_terminal_money{};
    // Minimum, paired over the exact same causal scenario set, of
    // candidate own terminal money minus Baseline own terminal money.
    // This is stronger than comparing two independent minima.
    std::int64_t worst_own_terminal_delta_vs_baseline{};
    double expected_own_terminal_money{};
    std::int64_t worst_price_impact{};
    int complexity{};
    FailureRiskProfile risk{};
};

struct RelativeFailureBudget {
    int critical_purchase{}, feed{}, overflow{}, route_hard{};
};

struct CatastropheLimit {
    int critical_purchase{std::numeric_limits<int>::max()};
    int feed{std::numeric_limits<int>::max()};
    int overflow{std::numeric_limits<int>::max()};
    int route_hard{std::numeric_limits<int>::max()};
};

struct Config {
    std::size_t beam_width{64};
    std::size_t max_decisions{3};
    std::size_t threads{0};
    double cvar_fraction{0.25};
    // A solvency controller may require feasible plans to reduce actual
    // critical/feed failures before comparing money, even when the Baseline
    // reference already fails and both plans lie inside the relative envelope.
    bool prefer_fewer_failures_within_envelope{};
    RelativeFailureBudget relative_budget{}; // extra worst failures vs baseline
    CatastropheLimit catastrophe_limit{};    // absolute hard rejection
};

struct ParetoPoint {
    std::vector<PersistentOption> sequence;
    RobustScore score;
    FailureCounts excess_over_envelope{};
    bool inside_envelope{};
};

struct Result {
    PersistentOption first;
    std::vector<PersistentOption> planned_sequence;
    RobustScore score;
    RobustScore baseline_reference;
    bool selected_inside_envelope{};
    std::vector<ParetoPoint> pareto_frontier;
    std::size_t evaluated_sequences{};
};

// Read-only exact rollout surface used by official queue parity fixtures.
// Production optimization uses the same implementation.
struct TerminalState {
    std::int64_t own_money{};
    std::int64_t opponent_money{};
    Inventory own_stock{};
    Inventory opponent_stock{};
    Inventory market_inventory{};
    FailureCounts failures{};
    int hires_today{};
    int hands{};
};

[[nodiscard]] TerminalState simulate_terminal(
    const CurrentState& current,
    const FixedForecast& forecast,
    const Scenario& scenario,
    const std::vector<PersistentOption>& sequence
);

[[nodiscard]] bool better(const RobustScore& left, const RobustScore& right);

[[nodiscard]] Result optimize(
    const CurrentState& current,
    const FixedForecast& forecast,
    const std::vector<Scenario>& causal_scenarios,
    const std::vector<PersistentOption>& candidates,
    const Config& config = {}
);

}  // namespace g001::rolling
