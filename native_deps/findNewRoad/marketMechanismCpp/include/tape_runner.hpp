#pragma once

#include "planner.hpp"
#include "simulator.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace g001::market::tape {

using CompiledTape = std::vector<fastkag::PlayerAction>;

[[nodiscard]] CompiledTape load_compiled_route(
    const std::string& compressed_tapes_path,
    const std::string& route_library_path,
    const std::string& family_or_route_id
);

struct ProductTrades {
    Inventory sold{};
    Inventory bought{};
};

struct RunMetrics {
    double reward[2]{};
    std::int64_t purchase_failures[2]{};
    std::int64_t overflow_units[2]{};
    ProductTrades trades[2]{};
    std::uint64_t compiled_unit_actions[2]{};
    std::uint64_t unit_action_mutations[2]{};
};

struct PairedSummary {
    std::uint64_t seeds = 0;
    std::uint64_t games = 0;
    double baseline_reward = 0.0;
    double variant_reward = 0.0;
    double opponent_reward = 0.0;
    double paired_reward_delta = 0.0;
    std::uint64_t wins = 0;
    std::uint64_t losses = 0;
    std::uint64_t ties = 0;
    std::int64_t baseline_purchase_failures = 0;
    std::int64_t variant_purchase_failures = 0;
    std::int64_t baseline_overflow_units = 0;
    std::int64_t variant_overflow_units = 0;
    ProductTrades baseline_trades{};
    ProductTrades variant_trades{};
    std::uint64_t baseline_compiled_unit_actions = 0;
    std::uint64_t variant_compiled_unit_actions = 0;
    std::uint64_t variant_unit_action_mutations = 0;
};

struct RunnerOptions {
    std::string compressed_tapes_path;
    std::string route_library_path;
    std::string route = "G001";
    std::string opponent_route;
    std::uint64_t seed_begin = 0;
    std::uint64_t seed_count = 192;
    std::size_t threads = 192;
    MechanismMask mechanisms = all_mechanisms();
};

[[nodiscard]] PairedSummary run_paired(const RunnerOptions& options);
[[nodiscard]] std::string summary_json(
    const PairedSummary& summary,
    const RunnerOptions& options
);

}  // namespace g001::market::tape
