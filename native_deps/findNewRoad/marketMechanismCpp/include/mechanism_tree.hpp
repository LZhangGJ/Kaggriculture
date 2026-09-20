#pragma once

#include "replay_format.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace g001::tree {

constexpr std::size_t action_count = 5;
constexpr std::size_t market_product_count = 9;

enum class MechanismAction : std::uint8_t {
    Baseline = 0,
    CashMinimum = 1,
    CapacityMinimum = 2,
    Hold = 3,
    PreDump = 4,
};

[[nodiscard]] const char* action_name(MechanismAction action);

// One row is the counterfactual result for the decision whose replay action is
// stored at (episode_id, turn, seat).  Its causal input state is MRA record turn-1.
// split_group must identify the opponent/route grouping used for validation.
#pragma pack(push, 1)
struct CounterfactualHeader {
    char magic[8];                    // "MCFLABEL"
    std::uint32_t version;            // 1
    std::uint32_t row_size;
    std::uint64_t row_count;
    std::uint8_t reserved[40];
};
#pragma pack(pop)

// Natural alignment is intentional: bootstrap training reads these fields very
// frequently, and packed float/int64 members would be undefined on strict-alignment
// targets. row_size in the header detects ABI mismatches.
struct CounterfactualRow {
    std::uint64_t episode_id{};
    std::uint64_t split_group{};
    std::uint16_t turn{};             // 1..719; action turn, not state turn
    std::uint8_t seat{};
    std::uint8_t valid_action_mask{0x1f};
    std::uint8_t phase{};             // 0 capital, 1 capacity, 2 liquidation
    std::uint8_t reserved0{};
    std::int16_t capacity_used{};
    std::int16_t capacity_free{};
    std::int32_t route_next_sale_step{-1}; // replay action-turn coordinate
    std::int32_t route_sale_slots_remaining{};
    std::int64_t cash_reserved{};
    float reward[action_count]{};     // higher is better; same units for all actions
    std::int16_t belief_point[market_product_count]{};
    std::int16_t belief_lower[market_product_count]{};
    std::int16_t belief_upper[market_product_count]{};
    std::int16_t recent_clearance[market_product_count]{};
};

static_assert(sizeof(CounterfactualHeader) == 64);
static_assert(sizeof(CounterfactualRow) == 144);

struct Example {
    std::uint64_t episode_id{};
    std::uint64_t split_group{};
    std::vector<float> feature;
    std::array<float, action_count> reward{};
    std::uint8_t valid_action_mask{0x1f};
};

// Feature order is stable and is also emitted into every JSON report.
[[nodiscard]] const std::vector<std::string>& feature_names();
[[nodiscard]] Example make_example(
    const g001::replay::Record& causal_state,
    const CounterfactualRow& label
);

[[nodiscard]] std::vector<Example> load_joined_examples(
    const std::string& mra_path,
    const std::string& counterfactual_path
);

struct TreeNode {
    int feature{-1};
    float threshold{};
    int left{-1};
    int right{-1};
    MechanismAction action{MechanismAction::Baseline};
    std::uint32_t sample_count{};
    float mean_selected_reward{};
};

struct TreeModel {
    std::vector<TreeNode> nodes;
    [[nodiscard]] MechanismAction predict(const std::vector<float>& feature) const;
};

struct TrainConfig {
    int max_depth{4};                 // constrained to 3..5 by CLI
    std::size_t min_leaf{512};
    std::size_t max_thresholds{128};
    double min_gain_per_sample{0.0};
    double ccp_alpha{0.001};          // reward units per sample per extra leaf
    double holdout_fraction{0.20};
    std::uint64_t seed{20260828};
    std::size_t bootstrap_rounds{100};
    std::size_t threads{0};              // 0 = hardware concurrency
};

struct Evaluation {
    std::size_t samples{};
    double selected_mean{};
    double baseline_mean{};
    double oracle_mean{};
    double gain_over_baseline{};
    double oracle_regret{};
};

struct BootstrapFeature {
    std::string name;
    double split_frequency{};
    double threshold_q25{};
    double threshold_median{};
    double threshold_q75{};
};

struct TrainReport {
    TreeModel model;
    Evaluation train;
    Evaluation holdout;
    std::size_t train_samples{};
    std::size_t holdout_samples{};
    std::size_t split_components{};
    std::vector<BootstrapFeature> bootstrap_features;
    double bootstrap_holdout_gain_q025{};
    double bootstrap_holdout_gain_median{};
    double bootstrap_holdout_gain_q975{};
    double bootstrap_leaf_action_agreement{};
};

[[nodiscard]] TrainReport train_grouped_tree(
    const std::vector<Example>& examples,
    const TrainConfig& config
);

void write_json_report(
    const std::string& path,
    const TrainReport& report,
    const TrainConfig& config,
    std::size_t total_examples
);

}  // namespace g001::tree
