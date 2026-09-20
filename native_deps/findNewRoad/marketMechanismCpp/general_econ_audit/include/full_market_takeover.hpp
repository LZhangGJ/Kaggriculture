#pragma once

#include <cstdint>
#include <vector>

namespace g001::general_econ {

enum class MarketOperation : std::uint8_t {
    Pass,
    Hire,
    BuyLand,
    BuySeed,
    BuyProduct,
    BuyAnimal,
    Sell,
};

enum class OrderOrigin : std::uint8_t {
    LegacyMacro,
    LegacySpecial,
    CriticalObligation,
    Liquidity,
    Capacity,
    Continuation,
    NewPlan,
};

struct MarketInstruction {
    MarketOperation operation{MarketOperation::Pass};
    int item{-1};
    int quantity{1};
    OrderOrigin origin{OrderOrigin::LegacyMacro};
    int deadline{};
    int priority{};

    friend bool operator==(const MarketInstruction&, const MarketInstruction&) = default;
};

// The primary experiment starts at step zero whenever its explicit toggle is
// enabled.  There is no step, route, opponent, or cluster-dependent gate.
[[nodiscard]] bool should_enter_takeover(bool experiment_enabled);

enum class PlannedUnitOperation : std::uint8_t {
    Other,
    Plant,
    Feed,
    Fertilize,
    PlaceAnimal,
};

struct PlannedUnitAction {
    int step{};
    int actor{};       // farmer is zero; hired workers are one and above
    int quadrant{};    // zero is initially unlocked
    PlannedUnitOperation operation{PlannedUnitOperation::Other};
    int item{-1};
};

struct OwnProductionResources {
    int current_actor_count{1};
    std::uint8_t unlocked_quadrants_mask{1};
    int seeds[5]{};
    int owned_items[12]{}; // shed + carried, used only for obligation coverage
};

struct ObligationCompilation {
    std::vector<MarketInstruction> orders;
    int required_hires{};
    int required_land_purchases{};
};

// Reverse-compiles only the focal player's non-market production plan.  The
// legacy market vector is intentionally not an input.
[[nodiscard]] ObligationCompilation compile_production_obligations(
    const OwnProductionResources& resources,
    const std::vector<PlannedUnitAction>& own_nonmarket_plan
);

struct TakeoverInput {
    bool enabled{};
    int maximum_slots{10};
    std::vector<MarketInstruction> legacy_market;
    std::vector<MarketInstruction> critical_obligations;
    std::vector<MarketInstruction> liquidity_orders;
    std::vector<MarketInstruction> capacity_orders;
    std::vector<MarketInstruction> continuation_orders;
    std::vector<MarketInstruction> new_plan_orders;
};

struct TakeoverResult {
    std::vector<MarketInstruction> orders;
    int suppressed_legacy_sells{};
    int suppressed_legacy_buys{};
    int suppressed_legacy_orders{};
    int unscheduled_critical_obligations{};
    int dropped_lower_priority_orders{};
    int legacy_origin_leaks{};
};

// Disabled: exact legacy vector passthrough.  Enabled: start from an empty
// market queue.  Every legacy action is audit-only and suppressed, including
// BUY/HIRE/LAND.  Critical obligations must come from the independent reverse
// compiler above.  The general controller owns all ten slots.
[[nodiscard]] TakeoverResult compose_takeover_market(const TakeoverInput& input);

enum class EconomicArm : std::uint8_t {
    Baseline,
    SpecialOnly,
    GeneralOnly,
    SpecialPlusGeneral,
};

[[nodiscard]] bool arm_uses_special(EconomicArm arm);
[[nodiscard]] bool arm_uses_general(EconomicArm arm);

}  // namespace g001::general_econ
