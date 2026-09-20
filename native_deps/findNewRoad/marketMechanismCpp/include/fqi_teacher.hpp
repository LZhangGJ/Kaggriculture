#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace g001::fqi {

enum class OptionKind : std::uint8_t {
    Baseline, Hold, Drip, PriceTarget, InventoryTarget, PreDump, Clear
};

// Persistent semi-Markov options, not product-quantity Cartesian actions. A
// global exact allocator converts selected product-level options into quantities.
struct OptionSpec {
    std::uint32_t id{};
    std::string name;
    OptionKind kind{OptionKind::Baseline};
    int quota{};
    int window_steps{};
    int reservation_price{};
    int target_inventory{};
    int impact_limit{};
};

#pragma pack(push, 1)
struct TransitionHeader {
    char magic[8];                  // "MFQITRN1"
    std::uint32_t version;          // 1
    std::uint32_t row_size;
    std::uint32_t feature_count;
    std::uint32_t option_count;     // 1..64, never assumed to be five
    std::uint64_t row_count;
    float gamma;                    // required to be 1.0
    std::uint32_t reward_semantics; // 1 = delta(own_money-opponent_money)
    std::uint64_t option_schema_hash;
    std::uint8_t reserved[16];
};
#pragma pack(pop)

// Dynamic arrays follow: rewards[O], durations[O], padding, next_state[O], features[F].
struct TransitionRowPrefix {
    std::uint64_t episode_id{};
    std::uint64_t split_group{};
    std::uint64_t valid_option_mask{};
    std::uint64_t done_option_mask{};
    std::uint16_t product_id{};      // enables shared product-level Q
    std::uint8_t reserved[6]{};
};

static_assert(sizeof(TransitionHeader) == 64);
static_assert(sizeof(TransitionRowPrefix) == 40);

struct RowLayout {
    std::size_t reward_offset{}, duration_offset{}, next_offset{}, feature_offset{}, row_size{};
};
[[nodiscard]] RowLayout row_layout(std::size_t feature_count, std::size_t option_count);

struct Transition {
    std::uint64_t episode_id{};
    std::uint64_t split_group{};
    std::uint64_t valid_option_mask{};
    std::uint64_t done_option_mask{};
    std::uint16_t product_id{};
    std::vector<float> reward;
    std::vector<std::uint16_t> duration;
    std::vector<std::int64_t> next_state;
    std::vector<float> feature;
};

struct TransitionDataset {
    std::vector<Transition> rows;
    std::size_t feature_count{}, base_feature_count{}, option_count{};
    std::uint64_t option_schema_hash{};
};
[[nodiscard]] TransitionDataset load_transitions(const std::string& path);

struct FqiConfig {
    std::size_t iterations{8}, trees{32};
    int tree_depth{6};
    std::size_t min_leaf{32}, random_features{0}, random_thresholds{8};
    double holdout_fraction{0.20};
    std::uint64_t seed{20260828};
    std::size_t threads{0}, bootstrap_rounds{8};
};

struct RegressionNode { int feature{-1}; float threshold{}; int left{-1}, right{-1}; float value{}; };
struct RegressionTree { std::vector<RegressionNode> nodes; };
struct Forest { std::vector<RegressionTree> trees; };

struct Teacher {
    std::vector<Forest> option;
    [[nodiscard]] std::vector<float> q_values(const std::vector<float>& feature) const;
};

struct OptionTreeNode {
    int feature{-1}; float threshold{}; int left{-1}, right{-1};
    std::uint32_t option_id{}, samples{};
};
struct OptionTree { std::vector<OptionTreeNode> nodes; };

struct IterationMetric { std::size_t iteration{}; double bellman_rmse{}, max_q_change{}; };

struct FqiReport {
    Teacher teacher;
    OptionTree distilled;
    std::vector<IterationMetric> convergence;
    std::size_t train_states{}, holdout_states{}, split_components{};
    double train_bellman_rmse{}, holdout_bellman_rmse{};
    double holdout_policy_value{}, holdout_baseline_value{}, holdout_oracle_value{};
    double distilled_teacher_agreement{};
    double bootstrap_policy_q025{}, bootstrap_policy_median{}, bootstrap_policy_q975{};
    double bootstrap_action_agreement{};
};

[[nodiscard]] FqiReport train_fqi_and_distill(
    const TransitionDataset& dataset, const FqiConfig& config,
    int distill_depth = 4, std::size_t distill_min_leaf = 512,
    double distill_ccp_alpha = 0.001
);
[[nodiscard]] std::uint32_t predict(const OptionTree& tree, const std::vector<float>& feature);
void write_report(const std::string& path, const FqiReport& report,
                  const FqiConfig& config, const std::vector<OptionSpec>& options,
                  std::size_t total_states);

}  // namespace g001::fqi
