#pragma once

#include "production_obligation.hpp"
#include "rolling_optimizer.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace production_forecast {

inline constexpr int kMinimumHorizonTicks = 24;
inline constexpr int kMaximumHorizonTicks = 96;

struct Config {
  // Dynamic causal forecast width. The default preserves the original
  // one-day ABI; Generation-3 may request up to four explicit days.
  int horizon_ticks{24};
  int episode_steps{720};
  bool liquidate_own_at_end{};
  // Default false preserves legacy 24-tick missing-frame-as-PASS callers.
  // Certified deployable adapters set true and must provide exactly one
  // ordered frame for every requested tick, including terminal padding.
  bool require_complete_causal_unit_tape{};
};

// This boundary accepts only the current own state, raw future unit actions,
// and their production-obligation DAG.  It cannot accept legacy market actions,
// route/opponent identity, or a realized replay tail.
struct Input {
  // Runtime contract: future_units includes the already-selected own unit
  // vector for current.step.  Raw fixed-tape frames begin at current.step + 1.
  // Omitting the current frame means PASS/no unit flow for that phase; the
  // compiler never reads native output or a replay tail implicitly.
  production_obligation::CompilerInput production;
  production_obligation::CompileResult obligations;
  Config config;
};

struct TickAudit {
  int step{-1};
  bool terminal_padding{};
  bool exact_production{true};
  bool pre_market_shed_valid{};
  // Exact simulated shed after this tick's unit phase and before acquisitions.
  std::array<int, production_obligation::kItems> pre_market_shed{};
  // Product created by HARVEST during this tick; unlike shed_inflow, this
  // excludes purchases, DROP and end-of-day returns.
  std::array<int, production_obligation::kProducts> production_yield_lower{};
  std::array<int, production_obligation::kProducts> production_yield_upper{};
  std::array<int, production_obligation::kProducts> shed_inflow_lower{};
  std::array<int, production_obligation::kProducts> shed_inflow_upper{};
  std::array<int, production_obligation::kProducts> shed_outflow{};
  std::vector<std::string> conservative_reasons;
};

struct Audit {
  bool accepted{};
  bool cash_nodes_consistent{true};
  bool used_conservative_production_lower_bound{};
  int terminal_padding_ticks{};
  int critical_purchase_requirements{};
  int feed_requirements{};
  int route_hard_requirements{};
  int maximum_animals_in_shed{};
  std::uint64_t raw_unit_fingerprint{};
  std::vector<TickAudit> ticks;
  std::vector<std::string> diagnostics;
};

struct Result {
  g001::rolling::FixedForecast forecast;
  Audit audit;
};

[[nodiscard]] Result compile(const Input& input);

}  // namespace production_forecast
