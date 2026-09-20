#pragma once

#include "market.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace g001::general_econ {

// A product-local execution promise. It contains no opponent identity, route
// identity, or learned threshold. The outstanding quantity survives a pause,
// a partial fill, and a later replanning call.
struct ExecutionState {
    int outstanding_units{};
    int remaining_windows{};
    int arrears{};
    bool paused{};
    std::uint64_t generation{};
    // Observation-confirmed uncertainty.  Legacy callers that leave both at
    // zero get the point value from outstanding_units.
    int outstanding_lower{};
    int outstanding_upper{};
};

struct QuantityInterval {
    int lower{};
    int point{};
    int upper{};
    bool consistent{true};
};

// Net non-SELL change to the focal player's shed between two observations.
// It is compiled only from the focal player's legal unit transfers, own BUY
// bounds, and (on day boundaries) automatic DROP bounds.
struct CausalShedDelta {
    std::array<int, market::product_count> lower{};
    std::array<int, market::product_count> point{};
    std::array<int, market::product_count> upper{};
};

struct SellSettlementInput {
    market::Inventory previous_shed{};
    market::Inventory current_shed{};
    CausalShedDelta non_sell_delta{};
    market::Inventory requested_total_sell{};
    market::Inventory requested_option_sell{};
    // Existing baseline/special SELL units have queue priority when a general
    // request is merged into their slot.  This prevents their fills from
    // incorrectly repaying the general option.
    market::Inventory prior_sell{};
};

struct SellSettlement {
    std::array<QuantityInterval, market::product_count> total_fill{};
    std::array<QuantityInterval, market::product_count> option_fill{};
};

[[nodiscard]] SellSettlement infer_sell_fills(const SellSettlementInput& input);

struct ExecutionProposal;

// Update all three outstanding bounds.  A wide fill interval is retained as
// uncertainty; it is never silently replaced by the requested quantity.
[[nodiscard]] ExecutionState settle_interval(
    const ExecutionState& state,
    const ExecutionProposal& proposal,
    QuantityInterval confirmed_fill
);

struct ExecutionOptionKey {
    std::uint8_t kind{};
    std::uint8_t product{};
    int quota{};
    int window_steps{};
    int reservation_price{};
    int target_inventory{};
    int impact_limit{};

    friend bool operator==(const ExecutionOptionKey&, const ExecutionOptionKey&) = default;
};

enum class SwitchDirective : std::uint8_t {
    RequireSame,
    PausePrevious,
    CancelPrevious,
};

struct ReplanResult {
    bool retained{};
    bool resumed{};
    int cancelled_units{};
};

// Cross-call option lifecycle.  Replanning to the same option is a no-op;
// changing it must explicitly pause or cancel the old promise.
class ExecutionLedger {
 public:
    [[nodiscard]] ReplanResult replan(
        ExecutionOptionKey option,
        int initial_units,
        int initial_windows,
        SwitchDirective directive = SwitchDirective::RequireSame
    );
    [[nodiscard]] bool has_active() const { return active_.has_value(); }
    [[nodiscard]] const ExecutionOptionKey& active_option() const;
    [[nodiscard]] const ExecutionState& state() const;
    void replace_state(ExecutionState state);
    void pause_active();
    [[nodiscard]] int cancel_active();

 private:
    struct Saved {
        ExecutionOptionKey option{};
        ExecutionState state{};
    };
    std::optional<Saved> active_;
    std::vector<Saved> paused_;
};

enum class MarketOrderKind : std::uint8_t { Other, Sell };

struct MarketOrder {
    MarketOrderKind kind{MarketOrderKind::Other};
    market::Product product{market::Product::Wheat};
    int quantity{1};
};

struct OrderOverlayResult {
    int requested{};
    int prior_sell{};
    int slot{-1};
    bool inserted{};
    bool slot_limited{};
};

// Adds general execution without exceeding the official ten market slots and
// without selling feed/production reserves.  Existing same-product SELL is
// extended in its last slot, preserving every earlier order's queue identity.
[[nodiscard]] OrderOverlayResult overlay_sell_order(
    std::vector<MarketOrder>& orders,
    market::Product product,
    int requested,
    int own_shed,
    int protected_stock,
    int maximum_slots = 10
);

// Official capacity counts all shed products and shed animals.  Carried items
// are intentionally absent from this ABI and therefore cannot inflate usage.
[[nodiscard]] int official_shed_occupancy(
    const std::array<int, 12>& shed_products_and_animals
);
[[nodiscard]] int official_shed_free_capacity(
    const std::array<int, 12>& shed_products_and_animals,
    int capacity = 100
);

struct ObservableExecutionInput {
    market::Product product{market::Product::Wheat};
    int own_stock{};
    int protected_stock{};       // feed/production stock that must not be sold
    int market_inventory{};
    bool sale_window{};
    int reservation_price{};     // economic shadow value supplied by the planner
    int marginal_impact_limit{}; // maximum allowed quote drop for discretionary units
    std::int64_t cash_shortfall{};
    int assigned_overflow_units{};
    // Explicit global terminal liquidation mode. This is stronger than the
    // final window of a finite drip option: it clears every non-protected unit
    // and may therefore exceed outstanding_units.
    bool terminal_clear_window{};
};

enum class ExecutionReason : std::uint8_t {
    NoWindow,
    Completed,
    FeedbackPause,
    ScheduledDrip,
    CashEmergency,
    StorageEmergency,
    LastWindow,
};

struct ExecutionProposal {
    int requested{};
    int scheduled_due{};
    int mandatory_units{};
    int quote{};
    // Counterfactual public inventory if every requested unit fills. Official
    // $1 sales leave this unchanged; this makes floor parity auditable.
    int market_inventory_if_filled{};
    bool cash_feasible{true};
    bool storage_feasible{true};
    ExecutionReason reason{ExecutionReason::NoWindow};
};

struct ObservationSettlementResult {
    SellSettlement fills{};
    QuantityInterval active_option_fill{};
    ExecutionState state{};
};

// Stateful online adapter: one call stages the submitted order against the
// current observation; the next call supplies only the new own shed snapshot.
// No request is settled early and no simulator fill/debug field is accepted.
class ObservationConfirmedExecutionAdapter {
 public:
    [[nodiscard]] ReplanResult replan(
        ExecutionOptionKey option,
        int initial_units,
        int initial_windows,
        SwitchDirective directive = SwitchDirective::RequireSame
    );
    void stage_submission(
        ExecutionProposal proposal,
        const market::Inventory& previous_shed,
        const CausalShedDelta& non_sell_delta,
        const market::Inventory& requested_total_sell,
        const market::Inventory& requested_option_sell,
        const market::Inventory& prior_sell
    );
    [[nodiscard]] std::optional<ObservationSettlementResult> observe(
        const market::Inventory& current_shed
    );
    [[nodiscard]] bool awaiting_observation() const { return pending_.has_value(); }
    [[nodiscard]] const ExecutionLedger& ledger() const { return ledger_; }

 private:
    struct Pending {
        ExecutionOptionKey option{};
        ExecutionProposal proposal{};
        SellSettlementInput input{};
    };
    ExecutionLedger ledger_;
    std::optional<Pending> pending_;
};

// Pure proposal: the state changes only after the next observation confirms
// the actual fill. This prevents requested orders from being mistaken for fills.
[[nodiscard]] ExecutionProposal propose(
    const ExecutionState& state,
    const ObservableExecutionInput& input
);

// Consume exactly one legal sale window. A zero fill is a genuine pause or
// failed fill: outstanding work is retained and automatically redistributed.
[[nodiscard]] ExecutionState settle(
    const ExecutionState& state,
    const ExecutionProposal& proposal,
    int confirmed_fill
);

// Terminal timing is chosen from causal public-belief scenarios. Each scenario
// describes the market inventory at the next guaranteed window after town flow
// and a possible rival liquidation bounded by the public inventory belief.
struct TerminalScenario {
    int next_market_inventory{};
    int rival_sale_before_next{};
    double weight{1.0};
};

struct TimingScore {
    std::int64_t worst{};
    double cvar{};
    double expected{};
};

enum class TerminalTiming : std::uint8_t { SellNow, SellNext };

struct TerminalDecision {
    TerminalTiming timing{TerminalTiming::SellNext};
    TimingScore now{};
    TimingScore next{};
};

[[nodiscard]] TerminalDecision choose_terminal_timing(
    market::Product product,
    int quantity,
    int current_market_inventory,
    const std::vector<TerminalScenario>& causal_scenarios,
    double cvar_fraction = 0.25
);

}  // namespace g001::general_econ
