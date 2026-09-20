#pragma once

#include "repair_whole_game_panel.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace g001::repair_enabled_panel_adapter {

namespace detail {
[[nodiscard]] bool disrupted_crop_source(
    fastkag::Action source, fastkag::Item desired,
    const fastkag::Tile& tile, int source_step, int current_step);
[[nodiscard]] bool crop_recovery_fits_pass_capacity(
    fastkag::Action source, fastkag::Item desired,
    const fastkag::Tile& tile, int source_step, int current_step,
    int remaining_raw_pass_slots);
}

struct Metrics {
  std::uint64_t seed{};
  int seat{-1};
  repair_whole_game_panel::Scenario scenario{};
  int steps{};
  // Same-observation provider proposal versus committed final action.
  int action_divergences{};
  std::vector<int> action_divergence_steps;
  std::vector<int> provider_unit_ops;
  std::vector<int> repair_unit_ops;
  // Independent baseline arm versus repair arm; may grow after a real repair.
  int paired_action_divergences{};
  int paired_move_mismatches{};
  int paired_common_move_mismatches{};
  int paired_farmer_move_mismatches{};
  int paired_actor_count_mismatches{};
  std::vector<int> paired_move_mismatch_steps;
  int first_focal_unit_action_divergence_step{-1};
  int first_focal_market_action_divergence_step{-1};
  int first_opponent_action_divergence_step{-1};
  int first_state_divergence_step{-1};
  int first_own_money_divergence_step{-1};
  int first_opponent_money_divergence_step{-1};
  int first_market_state_divergence_step{-1};
  int first_tile_divergence_index{-1};
  int first_tile_baseline_kind{-1};
  int first_tile_repair_kind{-1};
  int first_tile_baseline_animal{-1};
  int first_tile_repair_animal{-1};
  std::vector<int> first_state_baseline_unit_ops;
  std::vector<int> first_state_repair_unit_ops;
  int first_market_divergence_product{-1};
  int first_market_baseline_inventory{};
  int first_market_repair_inventory{};
  int first_market_baseline_price{};
  int first_market_repair_price{};
  std::vector<int> first_market_focal_orders;
  std::vector<int> first_market_opponent_orders;
  std::vector<int> first_market_baseline_focal_fills;
  std::vector<int> first_market_repair_focal_fills;
  std::vector<int> first_market_baseline_opponent_fills;
  std::vector<int> first_market_repair_opponent_fills;
  int first_legacy_trade_exposure_step{-1};
  int first_legacy_trade_exposure_product{-1};
  int first_legacy_trade_requested{};
  int first_legacy_trade_baseline_shed{};
  int first_legacy_trade_repair_shed{};
  int first_legacy_trade_incremental_fill{};
  int first_legacy_trade_price{};
  int first_legacy_trade_baseline_animals{};
  int first_legacy_trade_repair_animals{};
  double first_legacy_trade_baseline_asset_value{};
  double first_legacy_trade_repair_asset_value{};
  int first_common_move_mismatch_step{-1};
  int first_common_move_mismatch_actor{-1};
  int first_common_move_baseline_op{-1};
  int first_common_move_repair_op{-1};
  int route_move_mismatches{};
  std::vector<int> route_move_mismatch_steps;
  int movement_sequence_mismatches{};
  int movement_lane_population_mismatches{};
  std::vector<int> movement_sequence_mismatch_lanes;
  std::vector<int> movement_lane_population_mismatch_lanes;
  int causal_activations{};
  int crop_repair_activations{};
  int owned_actor{-1};
  int trigger_goal{-1};
  int trigger_item{-1};
  int trigger_day{-1};
  int trigger_remaining_slots{};
  double trigger_own_money{};
  double trigger_opponent_money{};
  int trigger_market_price{};
  int trigger_market_inventory{};
  int trigger_seed_inventory{};
  int trigger_crop_yield{};
  int trigger_crop_age{};
  int trigger_unwatered{};
  int trigger_shop_demand{};
  std::vector<int> trigger_future_ops;
  std::uint64_t trigger_obligation_id{};
  int trigger_source_step{-1};
  int trigger_x{-1};
  int trigger_y{-1};
  bool trigger_observed_weed{};
  bool trigger_effect_observed{};
  int trigger_final_tile_kind{-1};
  std::vector<int> trigger_transition_steps;
  std::vector<int> trigger_transition_ops;
  std::vector<int> trigger_transition_before_kinds;
  std::vector<int> trigger_transition_after_kinds;
  int triggers{};
  int attempts{};
  int commits{};
  int fallbacks{};
  int provider_state_commits{};
  int provider_state_leaks{};
  int commit_failures{};
  int unowned_unit_failures{};
  int market_exact_failures{};
  int purchase_failures{};
  int seed_purchase_failures{};
  int purchase_retries{};
  int purchase_retry_fills{};
  int purchase_finalize_failures{};
  int purchase_move_mismatches{};
  int joint_weed_purchase_commits{};
  int receipt_checks{};
  int receipts_accepted{};
  int receipts_failed{};
  int moves_expected{};
  int moves_emitted{};
  int move_failures{};
  int move_early{};
  int move_duplicates{};
  int move_drops{};
  int terminal_debts{};
  std::vector<std::uint64_t> terminal_debt_ids;
  std::vector<int> terminal_debt_source_steps;
  int baseline_unit_failures{};
  int repair_unit_failures{};
  int baseline_market_failures{};
  int repair_market_failures{};
  std::vector<int> baseline_unit_failure_steps;
  std::vector<int> repair_unit_failure_steps;
  std::vector<int> baseline_market_failure_steps;
  std::vector<int> repair_market_failure_steps;
  int forced_events{};
  int forced_baseline_applied{};
  int forced_repair_applied{};
  int forced_step{-1};
  int forced_x{-1};
  int forced_y{-1};
  int forced_kind{-1};
  long long hand_us_total{};
  long long hand_us_min{};
  long long hand_us_median{};
  long long hand_us_max{};
  repair_whole_game_panel::MoneyMetrics baseline_terminal;
  repair_whole_game_panel::MoneyMetrics repair_terminal;
};

struct Report {
  bool disabled_parity{};
  bool fail_stop_probe{};
  bool no_event_strict_parity{};
  Metrics fail_stop_metrics;
  Metrics no_event_metrics;
  std::vector<Metrics> games;
  [[nodiscard]] std::string json() const;
};

// Default-off mechanical fixture for the transactional crop callback.  It is
// deliberately separate from the economic panel: the fixture proves the
// native final-action transaction and one certified day, not reward lift.
struct TransactionalCropSmoke {
  int steps{};
  int forced_events{};
  int native_rejections{};
  int native_reject_ledger_leaks{};
  int same_hand_retries{};
  int native_commits{};
  int action_receipts_issued{};
  int action_receipts_acked{};
  int purchase_receipts_issued{};
  int purchase_receipts_acked{};
  int purchase_zero_fills{};
  int debt_continuations{};
  int moves_expected{};
  int moves_emitted{};
  int move_mismatches{};
  int crop_commits{};
  int crop_plots_closed{};
  int terminal_plot_debts{};
  int terminal_expected_receipts{};
  int terminal_delayed_moves{};
  bool lineage_from_past_plant{};
  bool dig_plant_water_committed{};
  bool crop_effect_observed{};
  std::vector<int> committed_ops;

  [[nodiscard]] bool passed() const noexcept;
};

[[nodiscard]] Report evaluate(std::string tapes, std::string library,
                              std::uint64_t seed = 970017);
[[nodiscard]] Report evaluate_vs_g096(std::string tapes, std::string library,
                                      std::uint64_t seed = 970017);
// One forced-weed arm for focused sanitizer execution. The full Release audit
// remains the artifact-producing authority.
[[nodiscard]] Metrics evaluate_focused(std::string tapes, std::string library,
                                       std::uint64_t seed = 970017);
[[nodiscard]] Report evaluate_forced_crop_at(
    std::string tapes, std::string library, std::uint64_t seed,
    int step, int x, int y, bool weed = true);
[[nodiscard]] TransactionalCropSmoke evaluate_transactional_crop_smoke(
    std::uint64_t seed = 970017);

}  // namespace g001::repair_enabled_panel_adapter
