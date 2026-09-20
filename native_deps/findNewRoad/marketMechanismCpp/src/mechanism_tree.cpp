#include "mechanism_tree.hpp"

#include "market.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace g001::tree {
namespace {

using g001::market::Product;
using g001::market::price;
using g001::market::sell_revenue;

constexpr double neg_inf = -std::numeric_limits<double>::infinity();

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Key {
    std::uint64_t episode{};
    std::uint16_t turn{};
    std::uint8_t seat{};
    bool operator==(const Key&) const = default;
};

struct KeyHash {
    std::size_t operator()(const Key& key) const {
        return static_cast<std::size_t>(mix(key.episode ^ (static_cast<std::uint64_t>(key.turn) << 9) ^ key.seat));
    }
};

std::string product_name(std::size_t p) {
    static constexpr std::array<const char*, 9> names{
        "wheat", "carrot", "tomato", "strawberry", "melon",
        "egg", "milk", "wool", "fertilizer"
    };
    return names.at(p);
}

std::string stock_name(std::size_t p) {
    static constexpr std::array<const char*, 12> names{
        "wheat", "carrot", "tomato", "strawberry", "melon",
        "egg", "milk", "wool", "fertilizer", "cow", "sheep", "goose"
    };
    return names.at(p);
}

double action_total(const std::vector<Example>& examples,
                    const std::vector<std::size_t>& rows, std::size_t action) {
    double total = 0.0;
    const std::uint8_t bit = static_cast<std::uint8_t>(1u << action);
    for (auto row : rows) {
        if ((examples[row].valid_action_mask & bit) == 0) return neg_inf;
        total += examples[row].reward[action];
    }
    return total;
}

std::pair<MechanismAction, double> best_leaf(const std::vector<Example>& examples,
                                             const std::vector<std::size_t>& rows) {
    std::size_t best = 0;
    double value = neg_inf;
    for (std::size_t action = 0; action < action_count; ++action) {
        const double candidate = action_total(examples, rows, action);
        if (candidate > value) { value = candidate; best = action; }
    }
    if (!std::isfinite(value)) {
        throw std::runtime_error("node has no mechanism valid for all samples");
    }
    return {static_cast<MechanismAction>(best), value};
}

struct Built {
    int node{};
    double value{};
    std::size_t leaves{};
};

class Builder {
 public:
    Builder(const std::vector<Example>& examples, const TrainConfig& config)
        : examples_(examples), config_(config) {}

    TreeModel fit(const std::vector<std::size_t>& rows) {
        if (rows.empty()) throw std::runtime_error("cannot fit empty training set");
        model_.nodes.reserve((1u << (config_.max_depth + 1)) - 1);
        (void)build(rows, 0);
        return std::move(model_);
    }

 private:
    Built build(const std::vector<std::size_t>& rows, int depth) {
        const auto [leaf_action, leaf_value] = best_leaf(examples_, rows);
        const int node_index = static_cast<int>(model_.nodes.size());
        model_.nodes.push_back(TreeNode{
            .feature = -1,
            .action = leaf_action,
            .sample_count = static_cast<std::uint32_t>(rows.size()),
            .mean_selected_reward = static_cast<float>(leaf_value / rows.size()),
        });
        if (depth >= config_.max_depth || rows.size() < 2 * config_.min_leaf) {
            return {node_index, leaf_value, 1};
        }

        int best_feature = -1;
        float best_threshold = 0.0f;
        double best_value = leaf_value;
        const std::size_t feature_count = examples_[rows.front()].feature.size();
        std::vector<std::pair<float, std::size_t>> order;
        order.reserve(rows.size());
        for (std::size_t feature = 0; feature < feature_count; ++feature) {
            order.clear();
            for (auto row : rows) order.emplace_back(examples_[row].feature[feature], row);
            std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            if (order.front().first == order.back().first) continue;

            std::array<double, action_count> left_sum{};
            std::array<double, action_count> right_sum{};
            std::array<std::size_t, action_count> left_valid{};
            std::array<std::size_t, action_count> right_valid{};
            for (const auto& [unused, row] : order) {
                (void)unused;
                for (std::size_t a = 0; a < action_count; ++a) {
                    right_sum[a] += examples_[row].reward[a];
                    right_valid[a] += (examples_[row].valid_action_mask >> a) & 1u;
                }
            }
            std::size_t last_bucket = std::numeric_limits<std::size_t>::max();
            for (std::size_t i = 1; i < order.size(); ++i) {
                const auto row = order[i - 1].second;
                for (std::size_t a = 0; a < action_count; ++a) {
                    left_sum[a] += examples_[row].reward[a];
                    right_sum[a] -= examples_[row].reward[a];
                    const auto valid = (examples_[row].valid_action_mask >> a) & 1u;
                    left_valid[a] += valid;
                    right_valid[a] -= valid;
                }
                if (i < config_.min_leaf || order.size() - i < config_.min_leaf ||
                    order[i - 1].first == order[i].first) continue;
                const std::size_t bucket = i * std::max<std::size_t>(1, config_.max_thresholds) / order.size();
                if (bucket == last_bucket && i + 1 != order.size() - config_.min_leaf) continue;
                last_bucket = bucket;
                double left_best = neg_inf;
                double right_best = neg_inf;
                for (std::size_t a = 0; a < action_count; ++a) {
                    if (left_valid[a] == i) left_best = std::max(left_best, left_sum[a]);
                    if (right_valid[a] == order.size() - i) right_best = std::max(right_best, right_sum[a]);
                }
                if (!std::isfinite(left_best) || !std::isfinite(right_best)) continue;
                const double value = left_best + right_best;
                if (value > best_value) {
                    best_value = value;
                    best_feature = static_cast<int>(feature);
                    best_threshold = std::midpoint(order[i - 1].first, order[i].first);
                }
            }
        }
        if (best_feature < 0 ||
            (best_value - leaf_value) / rows.size() <= config_.min_gain_per_sample) {
            return {node_index, leaf_value, 1};
        }

        std::vector<std::size_t> left_rows;
        std::vector<std::size_t> right_rows;
        left_rows.reserve(rows.size());
        right_rows.reserve(rows.size());
        for (auto row : rows) {
            (examples_[row].feature[best_feature] <= best_threshold ? left_rows : right_rows).push_back(row);
        }
        if (left_rows.size() < config_.min_leaf || right_rows.size() < config_.min_leaf) {
            return {node_index, leaf_value, 1};
        }
        const auto left = build(left_rows, depth + 1);
        const auto right = build(right_rows, depth + 1);
        const auto leaves = left.leaves + right.leaves;
        const double subtree_value = left.value + right.value;
        const double complexity_cost = config_.ccp_alpha * static_cast<double>(leaves - 1) * rows.size();
        if (subtree_value - leaf_value <= complexity_cost) {
            // Descendants were appended contiguously and are now unreachable.
            // Remove them so exported models contain no misleading orphan splits.
            model_.nodes.resize(static_cast<std::size_t>(node_index) + 1);
            return {node_index, leaf_value, 1};
        }
        auto& node = model_.nodes[node_index];
        node.feature = best_feature;
        node.threshold = best_threshold;
        node.left = left.node;
        node.right = right.node;
        return {node_index, subtree_value, leaves};
    }

    const std::vector<Example>& examples_;
    const TrainConfig& config_;
    TreeModel model_;
};

Evaluation evaluate(const TreeModel& model, const std::vector<Example>& examples,
                    const std::vector<std::size_t>& rows) {
    Evaluation result;
    result.samples = rows.size();
    if (rows.empty()) return result;
    for (auto row : rows) {
        auto selected = static_cast<std::size_t>(model.predict(examples[row].feature));
        if (((examples[row].valid_action_mask >> selected) & 1u) == 0) selected = 0;
        result.selected_mean += examples[row].reward[selected];
        result.baseline_mean += examples[row].reward[0];
        double oracle = neg_inf;
        for (std::size_t a = 0; a < action_count; ++a) {
            if ((examples[row].valid_action_mask >> a) & 1u) oracle = std::max(oracle, static_cast<double>(examples[row].reward[a]));
        }
        result.oracle_mean += oracle;
    }
    result.selected_mean /= rows.size();
    result.baseline_mean /= rows.size();
    result.oracle_mean /= rows.size();
    result.gain_over_baseline = result.selected_mean - result.baseline_mean;
    result.oracle_regret = result.oracle_mean - result.selected_mean;
    return result;
}

class Dsu {
 public:
    explicit Dsu(std::size_t n) : parent_(n), size_(n, 1) { std::iota(parent_.begin(), parent_.end(), 0); }
    std::size_t find(std::size_t x) { return parent_[x] == x ? x : parent_[x] = find(parent_[x]); }
    void join(std::size_t a, std::size_t b) {
        a = find(a); b = find(b); if (a == b) return;
        if (size_[a] < size_[b]) std::swap(a, b);
        parent_[b] = a; size_[a] += size_[b];
    }
 private:
    std::vector<std::size_t> parent_, size_;
};

struct Partition {
    std::vector<std::size_t> train;
    std::vector<std::size_t> holdout;
    std::vector<std::vector<std::size_t>> train_components;
    std::size_t component_count{};
};

Partition grouped_partition(const std::vector<Example>& examples, const TrainConfig& config) {
    Dsu dsu(examples.size());
    std::unordered_map<std::uint64_t, std::size_t> by_episode;
    std::unordered_map<std::uint64_t, std::size_t> by_group;
    for (std::size_t i = 0; i < examples.size(); ++i) {
        auto [ei, eins] = by_episode.emplace(examples[i].episode_id, i);
        if (!eins) dsu.join(i, ei->second);
        if (examples[i].split_group != 0) {
            auto [gi, gins] = by_group.emplace(examples[i].split_group, i);
            if (!gins) dsu.join(i, gi->second);
        }
    }
    std::unordered_map<std::size_t, std::vector<std::size_t>> grouped;
    for (std::size_t i = 0; i < examples.size(); ++i) grouped[dsu.find(i)].push_back(i);
    if (grouped.size() < 2) throw std::runtime_error("grouped holdout requires at least two disconnected replay/context components");
    std::vector<std::vector<std::size_t>> components;
    components.reserve(grouped.size());
    for (auto& [root, rows] : grouped) { (void)root; components.push_back(std::move(rows)); }
    std::sort(components.begin(), components.end(), [&](const auto& a, const auto& b) {
        return mix(examples[a.front()].episode_id ^ config.seed) < mix(examples[b.front()].episode_id ^ config.seed);
    });
    Partition result;
    result.component_count = components.size();
    for (auto& rows : components) {
        const auto key = examples[rows.front()].episode_id ^ examples[rows.front()].split_group ^ config.seed;
        const double unit = static_cast<double>(mix(key) >> 11) / static_cast<double>(1ULL << 53);
        if (unit < config.holdout_fraction) result.holdout.insert(result.holdout.end(), rows.begin(), rows.end());
        else {
            result.train.insert(result.train.end(), rows.begin(), rows.end());
            result.train_components.push_back(rows);
        }
    }
    // Deterministically repair unlucky tiny partitions at component granularity.
    if (result.holdout.empty()) {
        auto rows = result.train_components.back();
        result.train_components.pop_back();
        result.holdout = rows;
        std::unordered_map<std::size_t, bool> moved;
        for (auto row : rows) moved[row] = true;
        result.train.erase(std::remove_if(result.train.begin(), result.train.end(), [&](auto r) { return moved.contains(r); }), result.train.end());
    }
    if (result.train.empty() || result.train_components.empty()) {
        throw std::runtime_error("grouped split left no training component; add more replay/context groups");
    }
    return result;
}

double quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = q * (values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

void collect_splits(const TreeModel& model, int node, std::vector<bool>& used,
                    std::vector<std::vector<double>>& thresholds) {
    if (node < 0 || static_cast<std::size_t>(node) >= model.nodes.size()) return;
    const auto& n = model.nodes[node];
    if (n.feature < 0) return;
    used[n.feature] = true;
    thresholds[n.feature].push_back(n.threshold);
    collect_splits(model, n.left, used, thresholds);
    collect_splits(model, n.right, used, thresholds);
}

std::string json_escape(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (c == '\n') out += "\\n";
        else out.push_back(c);
    }
    return out;
}

}  // namespace

const char* action_name(MechanismAction action) {
    switch (action) {
        case MechanismAction::Baseline: return "baseline";
        case MechanismAction::CashMinimum: return "cash-min";
        case MechanismAction::CapacityMinimum: return "capacity-min";
        case MechanismAction::Hold: return "hold";
        case MechanismAction::PreDump: return "pre-dump";
    }
    return "invalid";
}

const std::vector<std::string>& feature_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> out{
            "state_step", "action_turn", "day", "hour", "action_turns_remaining",
            "phase_capital", "phase_capacity", "phase_liquidation",
            "own_cash", "cash_reserved", "capacity_used", "capacity_free"
        };
        for (std::size_t p = 0; p < 9; ++p) out.push_back("market_inventory_" + product_name(p));
        for (std::size_t p = 0; p < 9; ++p) out.push_back("market_price_" + product_name(p));
        for (std::size_t p = 0; p < 9; ++p) out.push_back("sell_marginal_impact_" + product_name(p));
        for (std::size_t p = 0; p < 9; ++p) out.push_back("buy_marginal_impact_" + product_name(p));
        for (std::size_t p = 0; p < 9; ++p) out.push_back("own_clear_slippage_" + product_name(p));
        for (std::size_t p = 0; p < 12; ++p) out.push_back("own_stock_" + stock_name(p));
        for (const char* side : {"own", "rival"}) {
            for (std::size_t p = 0; p < 9; ++p) out.push_back(std::string("farm_ready_") + side + "_" + product_name(p));
            for (std::size_t p = 0; p < 9; ++p) out.push_back(std::string("farm_producers_") + side + "_" + product_name(p));
        }
        for (const char* field : {"belief_point", "belief_lower", "belief_upper", "recent_clearance"})
            for (std::size_t p = 0; p < 9; ++p) out.push_back(std::string(field) + "_" + product_name(p));
        out.push_back("route_steps_to_next_sale");
        out.push_back("route_sale_slots_remaining");
        return out;
    }();
    return names;
}

Example make_example(const g001::replay::Record& state, const CounterfactualRow& label) {
    Example out;
    out.episode_id = label.episode_id;
    out.split_group = label.split_group;
    out.valid_action_mask = label.valid_action_mask;
    std::copy(std::begin(label.reward), std::end(label.reward), out.reward.begin());
    auto& f = out.feature;
    f.reserve(feature_names().size());
    f.push_back(state.turn);
    f.push_back(label.turn);
    f.push_back(state.day);
    f.push_back(state.hour);
    f.push_back(720 - label.turn);
    for (int phase = 0; phase < 3; ++phase) f.push_back(label.phase == phase ? 1.0f : 0.0f);
    f.push_back(state.money);
    f.push_back(static_cast<float>(label.cash_reserved));
    f.push_back(label.capacity_used);
    f.push_back(label.capacity_free);
    for (int value : state.market_inventory) f.push_back(static_cast<float>(value));
    for (short value : state.market_price) f.push_back(value);
    for (std::size_t p = 0; p < 9; ++p) {
        const auto product = static_cast<Product>(p);
        f.push_back(static_cast<float>(price(product, state.market_inventory[p]) - price(product, state.market_inventory[p] + 1)));
    }
    for (std::size_t p = 0; p < 9; ++p) {
        const auto product = static_cast<Product>(p);
        f.push_back(static_cast<float>(price(product, state.market_inventory[p] - 1) - price(product, state.market_inventory[p])));
    }
    for (std::size_t p = 0; p < 9; ++p) {
        const int quantity = std::max<int>(0, state.own_stock[p]);
        const auto product = static_cast<Product>(p);
        const double average = quantity ? static_cast<double>(sell_revenue(product, state.market_inventory[p], quantity)) / quantity
                                        : price(product, state.market_inventory[p]);
        f.push_back(static_cast<float>(price(product, state.market_inventory[p]) - average));
    }
    for (short value : state.own_stock) f.push_back(value);
    const std::size_t own = state.seat;
    const std::size_t rival = 1 - own;
    for (auto side : {own, rival}) {
        for (short value : state.farm_ready[side]) f.push_back(value);
        for (short value : state.farm_producers[side]) f.push_back(value);
    }
    for (short value : label.belief_point) f.push_back(value);
    for (short value : label.belief_lower) f.push_back(value);
    for (short value : label.belief_upper) f.push_back(value);
    for (short value : label.recent_clearance) f.push_back(value);
    f.push_back(label.route_next_sale_step < 0 ? 720.0f : label.route_next_sale_step - label.turn);
    f.push_back(label.route_sale_slots_remaining);
    if (f.size() != feature_names().size()) throw std::logic_error("feature schema length mismatch");
    return out;
}

std::vector<Example> load_joined_examples(const std::string& mra_path,
                                          const std::string& counterfactual_path) {
    std::ifstream labels(counterfactual_path, std::ios::binary);
    CounterfactualHeader label_header{};
    labels.read(reinterpret_cast<char*>(&label_header), sizeof(label_header));
    if (!labels || std::memcmp(label_header.magic, "MCFLABEL", 8) != 0 || label_header.version != 1 ||
        label_header.row_size != sizeof(CounterfactualRow)) {
        throw std::runtime_error("invalid counterfactual label header");
    }
    std::unordered_map<Key, CounterfactualRow, KeyHash> by_key;
    by_key.reserve(static_cast<std::size_t>(label_header.row_count * 1.25));
    for (std::uint64_t i = 0; i < label_header.row_count; ++i) {
        CounterfactualRow row{};
        labels.read(reinterpret_cast<char*>(&row), sizeof(row));
        if (!labels || row.turn == 0 || row.turn >= 720 || row.seat >= 2 || row.phase >= 3 ||
            (row.valid_action_mask & 1u) == 0) throw std::runtime_error("invalid counterfactual row");
        if (!by_key.emplace(Key{row.episode_id, row.turn, row.seat}, row).second)
            throw std::runtime_error("duplicate counterfactual key");
    }

    std::ifstream mra(mra_path, std::ios::binary);
    g001::replay::FileHeader file{};
    mra.read(reinterpret_cast<char*>(&file), sizeof(file));
    if (!mra || std::memcmp(file.magic, g001::replay::file_magic, 8) != 0 ||
        file.version != g001::replay::format_version || file.record_size != sizeof(g001::replay::Record))
        throw std::runtime_error("invalid MRA header");
    std::vector<Example> examples;
    examples.reserve(by_key.size());
    for (std::uint64_t replay_index = 0; replay_index < file.replay_count; ++replay_index) {
        g001::replay::ReplayHeader replay{};
        mra.read(reinterpret_cast<char*>(&replay), sizeof(replay));
        if (!mra) throw std::runtime_error("truncated MRA replay header");
        mra.seekg(replay.team_name_bytes[0] + replay.team_name_bytes[1], std::ios::cur);
        std::array<g001::replay::Record, 2> prior{};
        std::array<bool, 2> have_prior{};
        for (std::uint32_t i = 0; i < replay.record_count; ++i) {
            g001::replay::Record current{};
            mra.read(reinterpret_cast<char*>(&current), sizeof(current));
            if (!mra || current.seat >= 2) throw std::runtime_error("invalid MRA turn record");
            const auto found = by_key.find(Key{replay.episode_id, current.turn, current.seat});
            if (found != by_key.end()) {
                if (!have_prior[current.seat] || prior[current.seat].turn + 1 != current.turn)
                    throw std::runtime_error("counterfactual label has no causal prior state");
                examples.push_back(make_example(prior[current.seat], found->second));
                by_key.erase(found);
            }
            prior[current.seat] = current;
            have_prior[current.seat] = true;
        }
    }
    if (!by_key.empty()) throw std::runtime_error("counterfactual labels did not match MRA records: " + std::to_string(by_key.size()));
    return examples;
}

MechanismAction TreeModel::predict(const std::vector<float>& feature) const {
    if (nodes.empty()) return MechanismAction::Baseline;
    int index = 0;
    for (std::size_t guard = 0; guard <= nodes.size(); ++guard) {
        const auto& node = nodes.at(index);
        if (node.feature < 0) return node.action;
        if (static_cast<std::size_t>(node.feature) >= feature.size()) return MechanismAction::Baseline;
        index = feature[node.feature] <= node.threshold ? node.left : node.right;
        if (index < 0 || static_cast<std::size_t>(index) >= nodes.size()) return MechanismAction::Baseline;
    }
    return MechanismAction::Baseline;
}

TrainReport train_grouped_tree(const std::vector<Example>& examples, const TrainConfig& config) {
    if (examples.empty()) throw std::runtime_error("no joined examples");
    if (config.max_depth < 1 || config.max_depth > 5 || config.min_leaf == 0 ||
        !(config.holdout_fraction > 0 && config.holdout_fraction < 1))
        throw std::runtime_error("invalid tree configuration");
    const auto dimensions = examples.front().feature.size();
    if (dimensions != feature_names().size()) throw std::runtime_error("unexpected feature dimension");
    for (const auto& example : examples)
        if (example.feature.size() != dimensions) throw std::runtime_error("ragged feature vectors");
    auto partition = grouped_partition(examples, config);
    TrainReport report;
    report.train_samples = partition.train.size();
    report.holdout_samples = partition.holdout.size();
    report.split_components = partition.component_count;
    report.model = Builder(examples, config).fit(partition.train);
    report.train = evaluate(report.model, examples, partition.train);
    report.holdout = evaluate(report.model, examples, partition.holdout);
    if (config.bootstrap_rounds == 0) return report;

    struct BootstrapRound {
        double gain{};
        double agreement_sum{};
        std::uint64_t agreement_n{};
        std::vector<bool> used;
        std::vector<std::vector<double>> thresholds;
    };
    std::vector<BootstrapRound> rounds(config.bootstrap_rounds);
    std::atomic<std::size_t> next_round{0};
    const auto requested_threads = config.threads ? config.threads : std::max(1u, std::thread::hardware_concurrency());
    const auto thread_count = std::min<std::size_t>(requested_threads, config.bootstrap_rounds);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t worker = 0; worker < thread_count; ++worker) {
        workers.emplace_back([&] {
            for (;;) {
                const auto round = next_round.fetch_add(1);
                if (round >= config.bootstrap_rounds) break;
                std::mt19937_64 rng(mix(config.seed ^ 0xa5a5a5a5ULL ^ round));
                std::uniform_int_distribution<std::size_t> choose(0, partition.train_components.size() - 1);
                std::vector<std::size_t> sample;
                for (std::size_t draw = 0; draw < partition.train_components.size(); ++draw) {
                    const auto& rows = partition.train_components[choose(rng)];
                    sample.insert(sample.end(), rows.begin(), rows.end());
                }
                auto model = Builder(examples, config).fit(sample);
                auto got = evaluate(model, examples, partition.holdout);
                BootstrapRound result;
                result.gain = got.gain_over_baseline;
                result.used.assign(dimensions, false);
                result.thresholds.resize(dimensions);
                collect_splits(model, 0, result.used, result.thresholds);
                for (auto row : partition.holdout) {
                    result.agreement_sum += model.predict(examples[row].feature) == report.model.predict(examples[row].feature);
                    ++result.agreement_n;
                }
                rounds[round] = std::move(result);
            }
        });
    }
    for (auto& worker : workers) worker.join();

    std::vector<std::size_t> feature_used(dimensions);
    std::vector<std::vector<double>> thresholds(dimensions);
    std::vector<double> gains;
    double agreement_sum = 0.0;
    std::uint64_t agreement_n = 0;
    gains.reserve(rounds.size());
    for (const auto& round : rounds) {
        gains.push_back(round.gain);
        agreement_sum += round.agreement_sum;
        agreement_n += round.agreement_n;
        for (std::size_t f = 0; f < dimensions; ++f) {
            feature_used[f] += round.used[f];
            thresholds[f].insert(thresholds[f].end(), round.thresholds[f].begin(), round.thresholds[f].end());
        }
    }
    for (std::size_t f = 0; f < dimensions; ++f) {
        if (feature_used[f] == 0) continue;
        report.bootstrap_features.push_back(BootstrapFeature{
            .name = feature_names()[f],
            .split_frequency = static_cast<double>(feature_used[f]) / config.bootstrap_rounds,
            .threshold_q25 = quantile(thresholds[f], 0.25),
            .threshold_median = quantile(thresholds[f], 0.50),
            .threshold_q75 = quantile(thresholds[f], 0.75),
        });
    }
    std::sort(report.bootstrap_features.begin(), report.bootstrap_features.end(), [](const auto& a, const auto& b) {
        return a.split_frequency > b.split_frequency;
    });
    report.bootstrap_holdout_gain_q025 = quantile(gains, 0.025);
    report.bootstrap_holdout_gain_median = quantile(gains, 0.50);
    report.bootstrap_holdout_gain_q975 = quantile(gains, 0.975);
    report.bootstrap_leaf_action_agreement = agreement_n ? agreement_sum / agreement_n : 0.0;
    return report;
}

void write_json_report(const std::string& path, const TrainReport& report,
                       const TrainConfig& config, std::size_t total_examples) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create report: " + path);
    out << std::setprecision(10);
    auto eval = [&](const char* name, const Evaluation& e) {
        out << "\"" << name << "\":{\"samples\":" << e.samples
            << ",\"selected_mean\":" << e.selected_mean
            << ",\"baseline_mean\":" << e.baseline_mean
            << ",\"oracle_mean\":" << e.oracle_mean
            << ",\"gain_over_baseline\":" << e.gain_over_baseline
            << ",\"oracle_regret\":" << e.oracle_regret << "}";
    };
    out << "{\n\"schema\":\"g001-mechanism-tree-v1\","
        << "\n\"label_semantics\":\"counterfactual mechanism rewards; not historical-action imitation\","
        << "\n\"causal_alignment\":\"features=observation[t-1], label=mechanism action/reward at replay turn t\","
        << "\n\"private_feature_policy\":\"focal-seat private state only; rival private inventory forbidden\","
        << "\n\"total_examples\":" << total_examples
        << ",\"train_samples\":" << report.train_samples
        << ",\"holdout_samples\":" << report.holdout_samples
        << ",\"two_way_split_components\":" << report.split_components
        << ",\"config\":{\"max_depth\":" << config.max_depth
        << ",\"min_leaf\":" << config.min_leaf
        << ",\"max_thresholds\":" << config.max_thresholds
        << ",\"min_gain_per_sample\":" << config.min_gain_per_sample
        << ",\"ccp_alpha\":" << config.ccp_alpha
        << ",\"holdout_fraction\":" << config.holdout_fraction
        << ",\"bootstrap_rounds\":" << config.bootstrap_rounds
        << ",\"threads\":" << (config.threads ? config.threads : std::thread::hardware_concurrency()) << "},\n";
    eval("train", report.train); out << ",\n"; eval("holdout", report.holdout);
    out << ",\n\"bootstrap\":{\"holdout_gain_q025\":" << report.bootstrap_holdout_gain_q025
        << ",\"holdout_gain_median\":" << report.bootstrap_holdout_gain_median
        << ",\"holdout_gain_q975\":" << report.bootstrap_holdout_gain_q975
        << ",\"leaf_action_agreement\":" << report.bootstrap_leaf_action_agreement
        << ",\"features\":[";
    for (std::size_t i = 0; i < report.bootstrap_features.size(); ++i) {
        if (i) out << ',';
        const auto& f = report.bootstrap_features[i];
        out << "{\"name\":\"" << json_escape(f.name) << "\",\"split_frequency\":" << f.split_frequency
            << ",\"threshold_iqr\":[" << f.threshold_q25 << ',' << f.threshold_q75
            << "],\"threshold_median\":" << f.threshold_median << '}';
    }
    out << "]},\n\"feature_names\":[";
    for (std::size_t i = 0; i < feature_names().size(); ++i) {
        if (i) out << ',';
        out << '"' << json_escape(feature_names()[i]) << '"';
    }
    out << "],\n\"tree\":[";
    for (std::size_t i = 0; i < report.model.nodes.size(); ++i) {
        if (i) out << ',';
        const auto& n = report.model.nodes[i];
        out << "{\"id\":" << i << ",\"samples\":" << n.sample_count
            << ",\"leaf_action\":\"" << action_name(n.action) << "\""
            << ",\"mean_selected_reward\":" << n.mean_selected_reward;
        if (n.feature >= 0) out << ",\"feature\":\"" << json_escape(feature_names()[n.feature])
                                << "\",\"threshold\":" << n.threshold
                                << ",\"left\":" << n.left << ",\"right\":" << n.right;
        out << '}';
    }
    out << "]\n}\n";
}

}  // namespace g001::tree
