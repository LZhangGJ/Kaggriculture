#pragma once

#include "persistent_options.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace g001::robust {

struct ScenarioOutcome {
    std::uint32_t option_id{};
    std::uint32_t scenario_id{};  // belief lower/point/upper x dump window
    int purchase_failures{};
    int feed_failures{};
    int overflow_units{};
    double own_money{};
    double opponent_money{};
};

struct OptionScore {
    std::uint32_t option_id{};
    int worst_purchase_failures{};
    int worst_feed_failures{};
    int worst_overflow_units{};
    double worst_margin{};
    double cvar_margin{};
    double worst_own_money{};
};

struct Plan {
    std::uint32_t first_option{};
    int horizon_steps{};
    OptionScore score{};
};

// Stateless on purpose: callers execute only first_option and invoke this again
// at the next decision using the new public belief and persistent continuation.
[[nodiscard]] Plan choose_first_option(
    const std::vector<ScenarioOutcome>& outcomes,
    int horizon_steps,
    double cvar_tail_fraction = 0.25
);

}  // namespace g001::robust
