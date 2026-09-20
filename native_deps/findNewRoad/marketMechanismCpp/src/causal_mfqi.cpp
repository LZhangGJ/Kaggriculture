#include "causal_mfqi.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace g001::causal_fqi {

CapacityDecision causal_inventory_target(
    const RouteCapacityForecast& forecast,
    int selected_product_stock
) {
    CapacityDecision result;
    result.predicted_peak = std::max(0, forecast.current_shed_items) +
        std::max(0, forecast.current_carried_items) +
        std::max(0, forecast.visible_ready_harvest) +
        std::max(0, forecast.planned_product_buys) +
        std::max(0, forecast.planned_animal_buys);
    result.overflow = std::max(0, result.predicted_peak - std::max(0, forecast.shed_capacity));
    const auto releasable = std::min(std::max(0, selected_product_stock), result.overflow);
    result.target_inventory = std::max(0, selected_product_stock - releasable);
    result.valid = result.overflow > 0 && selected_product_stock > 0;
    return result;
}

DumpWindow public_dump_window(
    const std::vector<market::InventoryBelief>& history,
    std::size_t history_end,
    std::size_t product,
    int current_step,
    int episode_steps,
    int lookback_steps
) {
    DumpWindow result;
    if (history.empty() || history_end >= history.size() ||
        product >= market::product_count || current_step >= episode_steps - 1) return result;
    const auto begin = history_end > static_cast<std::size_t>(std::max(0, lookback_steps))
        ? history_end - static_cast<std::size_t>(lookback_steps) : 0;
    int clearance = 0;
    int stock_before_or_after = 0;
    for (auto index = begin; index <= history_end; ++index) {
        clearance = std::max(clearance, history[index].recent_clearance[product]);
        stock_before_or_after = std::max(
            stock_before_or_after,
            history[index].total_interval.upper[product]
        );
    }
    // A public liquidation pulse plus a causal nonzero upper bound is the
    // minimum evidence used here. No realized future sale or private stock is
    // consulted. The likely window is deliberately later than the earliest
    // window so PRE_DUMP remains distinct from CLEAR.
    if (clearance <= 0 || stock_before_or_after <= 0) return result;
    result.valid = true;
    result.observed_clearance = clearance;
    result.earliest_step = std::min(episode_steps - 1, current_step + 1);
    result.likely_step = std::min(episode_steps - 1, current_step + 8);
    return result;
}

int predump_quantity(int stock, int remaining_safe_windows) {
    stock = std::max(0, stock);
    remaining_safe_windows = std::max(1, remaining_safe_windows);
    return (stock + remaining_safe_windows - 1) / remaining_safe_windows;
}

int clear_quantity(int stock) { return std::max(0, stock); }

int causal_rival_net_market_flow(
    int market_before, int market_after, int town_drain,
    int own_sell_fill, int own_buy_fill
) {
    const auto all_player_net_sale = market_after + town_drain - market_before;
    const auto own_net_sale = own_sell_fill - own_buy_fill;
    return all_player_net_sale - own_net_sale;
}

void validate_graph(const std::vector<fqi::Transition>& rows,
                    std::size_t option_count) {
    if (option_count == 0 || option_count > 64) {
        throw std::runtime_error("MFQI option count must be 1..64");
    }
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto& row = rows[index];
        if (!row.valid_option_mask) throw std::runtime_error("MFQI row has no valid option");
        if (row.product_id >= market::product_count) {
            throw std::runtime_error("MFQI product id out of range");
        }
        if (row.reward.size() != option_count || row.duration.size() != option_count ||
            row.next_state.size() != option_count) {
            throw std::runtime_error("MFQI dynamic option arrays have wrong width");
        }
        for (std::size_t action = 0; action < option_count; ++action) {
            if (((row.valid_option_mask >> action) & 1ULL) == 0) continue;
            if (row.duration[action] == 0) {
                throw std::runtime_error("MFQI valid option has zero duration");
            }
            const auto done = (row.done_option_mask >> action) & 1ULL;
            const auto next = row.next_state[action];
            if (done) {
                if (next != -1) throw std::runtime_error("MFQI done edge must use next_state=-1");
                continue;
            }
            if (next <= static_cast<std::int64_t>(index) ||
                next < 0 || static_cast<std::size_t>(next) >= rows.size()) {
                throw std::runtime_error("MFQI nonterminal edge is not a closed forward edge");
            }
            const auto& successor = rows[static_cast<std::size_t>(next)];
            if (successor.episode_id != row.episode_id ||
                successor.product_id != row.product_id) {
                throw std::runtime_error("MFQI continuation changed episode or product");
            }
            if (action == static_cast<std::size_t>(fqi::OptionKind::Drip) &&
                row.feature.size() > ActiveKind &&
                row.feature[ActiveKind] == static_cast<float>(fqi::OptionKind::Drip) &&
                row.feature.size() >= FeatureCount && successor.feature.size() >= FeatureCount) {
                const auto current_total = row.feature[RemainingQuota] + row.feature[Debt];
                const auto next_total = successor.feature[RemainingQuota] + successor.feature[Debt];
                if (successor.feature[ActiveKind] !=
                        static_cast<float>(fqi::OptionKind::Drip) ||
                    successor.feature[RemainingWindows] != row.feature[RemainingWindows] - 1 ||
                    successor.feature[HorizonRemaining] !=
                        std::max(0.0f, row.feature[HorizonRemaining] - row.duration[action]) ||
                    next_total < 0 || next_total > current_total) {
                    throw std::runtime_error(
                        "MFQI DRIP continuation state is not conserved at row " +
                        std::to_string(index) + " -> " + std::to_string(next) +
                        " windows=" + std::to_string(row.feature[RemainingWindows]) +
                        "->" + std::to_string(successor.feature[RemainingWindows]) +
                        " horizon=" + std::to_string(row.feature[HorizonRemaining]) +
                        "-" + std::to_string(row.duration[action]) + "->" +
                        std::to_string(successor.feature[HorizonRemaining]) +
                        " total=" + std::to_string(current_total) + "->" +
                        std::to_string(next_total) + " kind=" +
                        std::to_string(successor.feature[ActiveKind])
                    );
                }
            }
        }
    }
}

void write_mfqi(const std::string& path,
                const std::vector<fqi::Transition>& rows,
                std::size_t feature_count,
                std::size_t option_count,
                std::uint64_t option_schema_hash) {
    validate_graph(rows, option_count);
    const auto layout = fqi::row_layout(feature_count, option_count);
    fqi::TransitionHeader header{};
    std::memcpy(header.magic, "MFQITRN1", 8);
    header.version = 1;
    header.row_size = static_cast<std::uint32_t>(layout.row_size);
    header.feature_count = static_cast<std::uint32_t>(feature_count);
    header.option_count = static_cast<std::uint32_t>(option_count);
    header.row_count = rows.size();
    header.gamma = 1.0f;
    header.reward_semantics = 1;
    header.option_schema_hash = option_schema_hash;

    const auto output_path = std::filesystem::path(path);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create MFQI output");
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    std::vector<char> bytes(layout.row_size);
    for (const auto& row : rows) {
        if (row.feature.size() != feature_count) {
            throw std::runtime_error("MFQI feature width mismatch");
        }
        std::fill(bytes.begin(), bytes.end(), 0);
        fqi::TransitionRowPrefix prefix{};
        prefix.episode_id = row.episode_id;
        prefix.split_group = row.split_group;
        prefix.valid_option_mask = row.valid_option_mask;
        prefix.done_option_mask = row.done_option_mask;
        prefix.product_id = row.product_id;
        std::memcpy(bytes.data(), &prefix, sizeof(prefix));
        std::memcpy(bytes.data() + layout.reward_offset, row.reward.data(),
                    sizeof(float) * option_count);
        std::memcpy(bytes.data() + layout.duration_offset, row.duration.data(),
                    sizeof(std::uint16_t) * option_count);
        std::memcpy(bytes.data() + layout.next_offset, row.next_state.data(),
                    sizeof(std::int64_t) * option_count);
        std::memcpy(bytes.data() + layout.feature_offset, row.feature.data(),
                    sizeof(float) * feature_count);
        output.write(bytes.data(), bytes.size());
    }
    if (!output) throw std::runtime_error("failed writing MFQI rows");
}

}  // namespace g001::causal_fqi
