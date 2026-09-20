#include "belief.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace g001::market {
namespace {

int nonnegative(int value) {
    return std::max(0, value);
}

bool can_buy_from_market(Product product) {
    return product == Product::Wheat || product == Product::Fertilizer;
}

std::int64_t fibonacci_hire_cost(int already_hired) {
    std::int64_t a = 1;
    std::int64_t b = 1;
    while (already_hired-- > 0) {
        const auto next = a + b;
        a = b;
        b = next;
    }
    return a;
}

std::int64_t public_opponent_purchase_cost(
    const PublicStateSummary& previous,
    const PublicStateSummary& current
) {
    std::int64_t cost = 0;
    if (current.day == previous.day) {
        const int first = std::max(0, previous.opponent_hires_today);
        const int hire_count = std::max(
            nonnegative(current.opponent_hires_today - previous.opponent_hires_today),
            nonnegative(current.opponent_hands - previous.opponent_hands)
        );
        const int last = first + hire_count;
        for (int hired = first; hired < last; ++hired) {
            cost += current.farm_hand_cost_multiplier * fibonacci_hire_cost(hired);
        }
    }
    constexpr std::array<int, 3> land_cost{1000, 2000, 4000};
    const int first_land = std::clamp(previous.opponent_unlocked_quadrants, 1, 4);
    const int last_land = std::clamp(current.opponent_unlocked_quadrants, first_land, 4);
    for (int count = first_land; count < last_land; ++count) {
        cost += land_cost[static_cast<std::size_t>(count - 1)];
    }
    return cost;
}

}  // namespace

OpponentInventoryBelief::OpponentInventoryBelief(int shed_capacity)
    : shed_capacity_(shed_capacity) {
    if (shed_capacity < 0) {
        throw std::invalid_argument("shed capacity must be non-negative");
    }
}

InventoryBelief OpponentInventoryBelief::reset(
    const PublicStateSummary& state,
    const InitialOpponentStock& initial
) {
    initialized_ = true;
    shed_partition_ambiguous_ = false;
    previous_ = state;
    carried_point_.fill(0);
    ambiguous_.fill(false);

    for (std::size_t i = 0; i < product_count; ++i) {
        const int opponent_point = nonnegative(initial.point[i]);
        const int opponent_lower = nonnegative(initial.lower[i]);
        const int opponent_upper = std::max(opponent_lower, initial.upper[i]);
        joint_point_[i] = state.own_total[i] + opponent_point;
        joint_lower_[i] = state.own_total[i] + opponent_lower;
        joint_upper_[i] = state.own_total[i] + opponent_upper;
    }

    Inventory zero{};
    InventoryInterval zero_interval{};
    ProductFlags not_saturated{};
    return snapshot(state, zero, zero_interval, not_saturated);
}

InventoryBelief OpponentInventoryBelief::update(const PublicStateSummary& state) {
    if (!initialized_) return reset(state);
    if (state.step <= previous_.step) {
        throw std::invalid_argument("belief summaries must have increasing steps");
    }

    Inventory previous_total{};
    Inventory previous_upper{};
    for (std::size_t i = 0; i < product_count; ++i) {
        previous_total[i] = nonnegative(joint_point_[i] - previous_.own_total[i]);
        previous_upper[i] = nonnegative(joint_upper_[i] - previous_.own_total[i]);
    }

    Inventory clearance{};
    InventoryInterval clearance_interval{};
    Inventory floor_purchase{};
    InventoryInterval floor_purchase_interval{};
    ProductFlags saturated{};
    Inventory opponent_market_flow{};

    if (state.own_market_net_flow_bounds_valid) {
        for (std::size_t i = 0; i < product_count; ++i) {
            if (state.own_market_net_flow_lower[i] > state.own_market_net_flow_point[i]
                || state.own_market_net_flow_point[i]
                    > state.own_market_net_flow_upper[i]) {
                throw std::invalid_argument("invalid own market-flow interval");
            }
        }
    }

    for (std::size_t i = 0; i < product_count; ++i) {
        const int harvest_point =
            state.own_flow.harvest_point[i] + state.opponent_flow.harvest_point[i];
        const int harvest_upper =
            std::max(state.own_flow.harvest_point[i], state.own_flow.harvest_upper[i])
            + std::max(
                state.opponent_flow.harvest_point[i],
                state.opponent_flow.harvest_upper[i]
            );
        const int consumption_point =
            state.own_flow.consumption_point[i] + state.opponent_flow.consumption_point[i];
        const int consumption_upper =
            std::max(
                state.own_flow.consumption_point[i],
                state.own_flow.consumption_upper[i]
            ) + std::max(
                state.opponent_flow.consumption_point[i],
                state.opponent_flow.consumption_upper[i]
            );

        // market change + town drain is exactly the player-caused non-floor
        // net sell flow (sales stored in market minus purchases).
        const int player_market_flow =
            state.market_inventory[i] + state.town_consumption[i]
            - previous_.market_inventory[i];

        joint_point_[i] += harvest_point - consumption_point - player_market_flow;
        joint_lower_[i] = nonnegative(
            joint_lower_[i] + harvest_point - consumption_upper - player_market_flow
        );
        joint_upper_[i] = nonnegative(
            joint_upper_[i] + harvest_upper - consumption_point - player_market_flow
        );

        const auto product = static_cast<Product>(i);
        const int after_players = state.market_inventory[i] + state.town_consumption[i];
        saturated[i] = state.floor_sale_ambiguity[i]
            || previous_.market_price[i] == 1
            || price(product, after_players) == 1;
        if (saturated[i] || state.possible_private_discard[i]) {
            // Hidden floor sales and discards can remove any amount that was
            // available.  The causal upper bound remains valid, while the
            // lower bound must collapse to zero.
            joint_lower_[i] = 0;
            ambiguous_[i] = true;
        }

        // Infer the opponent's visible net market outflow by removing the
        // exact own-stock flow.  This avoids mistaking a town drain or our own
        // sale for an opponent clearance.
        const int own_market_flow = state.own_market_net_flow_bounds_valid
            ? state.own_market_net_flow_point[i]
            : previous_.own_total[i] + state.own_flow.harvest_point[i]
                - state.own_flow.consumption_point[i] - state.own_total[i];
        const int visible_opponent_outflow = player_market_flow - own_market_flow;
        opponent_market_flow[i] = visible_opponent_outflow;
        clearance[i] = nonnegative(visible_opponent_outflow);
        clearance_interval.lower[i] = clearance[i];

        // This is an interval for net liquidation of previously/just-produced
        // stock, not gross churn from buy-then-sell in one transition.
        const int available_upper = previous_upper[i]
            + state.opponent_flow.harvest_upper[i]
            - state.opponent_flow.consumption_point[i];
        clearance_interval.upper[i] = std::max(
            clearance_interval.lower[i], nonnegative(available_upper)
        );
        if (!saturated[i] && !state.possible_private_discard[i]) {
            clearance_interval.upper[i] = clearance_interval.lower[i];
        }

        carried_point_[i] = nonnegative(
            carried_point_[i] + state.opponent_carried_gain[i]
            - state.opponent_carried_use[i]
        );
    }

    // At the official $1 floor, every hidden sale raises public money by one
    // dollar without entering market inventory.  Money therefore identifies a
    // *joint* hidden-sale interval H, even though its allocation over products
    // is not unique.  BUY_PRODUCT products add a second correlated latent B:
    // H - B equals the public cash residual.  We project both joint intervals
    // to per-product marginals rather than pretending an ambiguous wash is
    // either exactly known or completely unknowable.
    ProductFlags cash_constraint_applied{};
    ProductFlags floor_purchase_constraint_applied{};
    if (state.money_evidence_valid && previous_.money_evidence_valid
        && state.trade_cash_bounds_valid) {
        if (state.opponent_private_purchase_cost_lower < 0
            || state.opponent_private_purchase_cost_upper
                < state.opponent_private_purchase_cost_lower
            || state.opponent_nonfloor_trade_cash_upper
                < state.opponent_nonfloor_trade_cash_lower) {
            throw std::invalid_argument("invalid opponent cash-evidence interval");
        }
        const std::int64_t known_public_cost = public_opponent_purchase_cost(
            previous_, state
        );
        // Add successful HIRE/LAND cost back before attributing any positive
        // residual to market activity.
        const std::int64_t unexplained = state.opponent_money
            - previous_.opponent_money - state.opponent_public_cash_delta
            + known_public_cost;
        std::int64_t raw_hidden_lower = unexplained
            + state.opponent_private_purchase_cost_lower
            - state.opponent_nonfloor_trade_cash_upper;
        std::int64_t raw_hidden_upper = unexplained
            + state.opponent_private_purchase_cost_upper
            - state.opponent_nonfloor_trade_cash_lower;
        const auto midpoint = [](std::int64_t lower, std::int64_t upper) {
            return lower + (upper - lower) / 2;
        };
        std::int64_t raw_hidden_point = unexplained
            + midpoint(
                state.opponent_private_purchase_cost_lower,
                state.opponent_private_purchase_cost_upper
            ) - midpoint(
                state.opponent_nonfloor_trade_cash_lower,
                state.opponent_nonfloor_trade_cash_upper
            );

        struct Candidate {
            std::size_t product{};
            int visible{};
            int capacity{};
            int original_point{};
            int hidden_lower{};
            int hidden_upper{};
            int hidden_point{};
        };
        std::array<Candidate, product_count> candidates{};
        std::size_t candidate_count = 0;
        std::int64_t total_capacity = 0;
        Inventory buy_lower{};
        Inventory buy_point{};
        Inventory buy_upper{};
        ProductFlags buy_candidate{};
        std::int64_t buy_lower_sum = 0;
        std::int64_t buy_point_sum = 0;
        bool has_buy_candidate = false;
        // A ten-slot queue can contain at most five full-capacity refill
        // cycles (BUY,SELL,... or SELL,BUY,...).  This is deliberately a broad
        // throughput cap; cash/sale conservation below usually tightens it.
        constexpr int market_order_slots = 10;
        const std::int64_t gross_buy_throughput =
            ((market_order_slots + 1) / 2) * static_cast<std::int64_t>(shed_capacity_);
        for (std::size_t i = 0; i < product_count; ++i) {
            if (!saturated[i]) continue;
            if (can_buy_from_market(static_cast<Product>(i))) {
                // N=visible non-floor SELL-BUY.  Hence BUY>=max(0,-N).
                // BUY can be larger when an offsetting visible sell occurred;
                // the queue-throughput and cash/stock equations cap that churn.
                // A stock-delta fallback is not proof here: FEED, harvest and
                // PICKUP/DROP can change own_total without a market fill.  Only
                // a caller-supplied causal own-market interval may establish a
                // positive BUY lower bound.
                if (state.own_market_net_flow_bounds_valid) {
                    const int player_market_flow = state.market_inventory[i]
                        + state.town_consumption[i] - previous_.market_inventory[i];
                    const int opponent_net_point = player_market_flow
                        - state.own_market_net_flow_point[i];
                    const int opponent_net_upper = player_market_flow
                        - state.own_market_net_flow_lower[i];
                    buy_lower[i] = nonnegative(-opponent_net_upper);
                    buy_point[i] = std::max(
                        buy_lower[i], nonnegative(-opponent_net_point)
                    );
                }
                buy_upper[i] = static_cast<int>(std::min<std::int64_t>(
                    std::numeric_limits<int>::max(), gross_buy_throughput
                ));
                buy_candidate[i] = true;
                has_buy_candidate = true;
                buy_lower_sum += buy_lower[i];
                buy_point_sum += buy_point[i];
            }
            const int visible = clearance[i];
            // Current post-visible-flow upper stock is the sale capacity.  For
            // WHEAT/FERTILIZER this deliberately includes public net BUY units,
            // which makes SELL->$1 BUY wash/rebase conservation close.
            const int capacity = nonnegative(
                joint_upper_[i] - state.own_total[i]
            );
            if (capacity == 0) continue;
            auto& candidate = candidates[candidate_count++];
            candidate.product = i;
            candidate.visible = visible;
            candidate.capacity = capacity;
            candidate.original_point = std::min(
                capacity, nonnegative(joint_point_[i] - state.own_total[i])
            );
            total_capacity += capacity;
        }

        // Feasible joint gross floor purchases B satisfy both the public net
        // market lower bounds and S=R+B within the causal gross-sale cap.
        std::int64_t joint_buy_lower = buy_lower_sum;
        std::int64_t joint_buy_upper = has_buy_candidate
            ? gross_buy_throughput : 0;
        joint_buy_lower = std::max(joint_buy_lower, -raw_hidden_upper);
        joint_buy_upper = std::min(
            joint_buy_upper, total_capacity - raw_hidden_lower
        );
        const bool buy_infeasible = joint_buy_lower > joint_buy_upper;
        const auto feasible_buy_point = std::clamp(
            buy_point_sum, joint_buy_lower, joint_buy_upper
        );

        const auto sale_lower_with_buy = raw_hidden_lower + joint_buy_lower;
        const auto sale_point_with_buy = raw_hidden_point + feasible_buy_point;
        const auto sale_upper_with_buy = raw_hidden_upper + joint_buy_upper;

        // An empty intersection between cash evidence and causal stock caps
        // means at least one upstream bound is inconsistent.  Falling back to
        // the pre-cash interval is sounder than fabricating a clearance.
        const bool infeasible = buy_infeasible || sale_upper_with_buy < 0
            || sale_lower_with_buy > total_capacity;
        if (candidate_count > 0 && !infeasible) {
            const std::int64_t hidden_upper = std::min(
                total_capacity, std::max<std::int64_t>(0, sale_upper_with_buy)
            );
            const std::int64_t hidden_lower = std::max<std::int64_t>(
                0, sale_lower_with_buy
            );

            if (hidden_lower <= hidden_upper) {
                std::int64_t sum_lower = 0;
                std::int64_t sum_upper = 0;
                for (std::size_t c = 0; c < candidate_count; ++c) {
                    auto& candidate = candidates[c];
                    candidate.hidden_lower = static_cast<int>(std::max<std::int64_t>(
                        0, hidden_lower - (total_capacity - candidate.capacity)
                    ));
                    candidate.hidden_upper = static_cast<int>(std::min<std::int64_t>(
                        candidate.capacity, hidden_upper
                    ));
                    candidate.hidden_point = candidate.hidden_lower;
                    sum_lower += candidate.hidden_lower;
                    sum_upper += candidate.hidden_upper;
                }

                std::int64_t hidden_point = std::clamp(
                    sale_point_with_buy, hidden_lower, hidden_upper
                );
                hidden_point = std::clamp(hidden_point, sum_lower, sum_upper);

                // Reproducible point allocation: project the old point weights
                // onto the joint point total, clamp to each marginal interval,
                // then deterministically redistribute rounding/cap residuals.
                // If every old point is zero, capacity is the fallback weight.
                std::int64_t total_point_weight = 0;
                for (std::size_t c = 0; c < candidate_count; ++c) {
                    total_point_weight += candidates[c].original_point;
                }
                const bool initial_capacity_weight = total_point_weight == 0;
                if (initial_capacity_weight) total_point_weight = total_capacity;
                std::int64_t allocated_point = 0;
                for (std::size_t c = 0; c < candidate_count; ++c) {
                    auto& candidate = candidates[c];
                    const std::int64_t weight = initial_capacity_weight
                        ? candidate.capacity : candidate.original_point;
                    const auto ideal = static_cast<std::int64_t>(
                        static_cast<long double>(hidden_point)
                            * static_cast<long double>(weight)
                            / static_cast<long double>(total_point_weight)
                    );
                    candidate.hidden_point = static_cast<int>(std::clamp<std::int64_t>(
                        ideal, candidate.hidden_lower, candidate.hidden_upper
                    ));
                    allocated_point += candidate.hidden_point;
                }

                std::int64_t remaining = hidden_point - allocated_point;
                while (remaining > 0) {
                    std::int64_t total_weight = 0;
                    for (std::size_t c = 0; c < candidate_count; ++c) {
                        const auto& candidate = candidates[c];
                        if (candidate.hidden_point >= candidate.hidden_upper) continue;
                        total_weight += candidate.original_point;
                    }
                    const bool capacity_weight = total_weight == 0;
                    if (capacity_weight) {
                        for (std::size_t c = 0; c < candidate_count; ++c) {
                            const auto& candidate = candidates[c];
                            total_weight += candidate.hidden_upper - candidate.hidden_point;
                        }
                    }
                    if (total_weight <= 0) break;

                    const auto round_remaining = remaining;
                    std::int64_t allocated = 0;
                    for (std::size_t c = 0; c < candidate_count; ++c) {
                        auto& candidate = candidates[c];
                        const int room = candidate.hidden_upper - candidate.hidden_point;
                        if (room <= 0) continue;
                        const std::int64_t weight = capacity_weight
                            ? room : candidate.original_point;
                        if (weight <= 0) continue;
                        const auto share = static_cast<std::int64_t>(
                            static_cast<long double>(round_remaining)
                                * static_cast<long double>(weight)
                                / static_cast<long double>(total_weight)
                        );
                        const int take = static_cast<int>(std::min<std::int64_t>(room, share));
                        candidate.hidden_point += take;
                        allocated += take;
                    }
                    remaining -= allocated;
                    if (allocated == 0) {
                        // Largest current weight, stable product-index tie break.
                        std::size_t best = candidate_count;
                        std::int64_t best_weight = -1;
                        for (std::size_t c = 0; c < candidate_count; ++c) {
                            const auto& candidate = candidates[c];
                            const int room = candidate.hidden_upper - candidate.hidden_point;
                            if (room <= 0) continue;
                            const std::int64_t weight = capacity_weight
                                ? room : candidate.original_point;
                            if (weight > best_weight) {
                                best = c;
                                best_weight = weight;
                            }
                        }
                        if (best == candidate_count) break;
                        ++candidates[best].hidden_point;
                        --remaining;
                    }
                }
                while (remaining < 0) {
                    const auto excess = -remaining;
                    std::int64_t total_removable = 0;
                    for (std::size_t c = 0; c < candidate_count; ++c) {
                        total_removable += candidates[c].hidden_point
                            - candidates[c].hidden_lower;
                    }
                    if (total_removable <= 0) break;
                    std::int64_t removed = 0;
                    for (std::size_t c = 0; c < candidate_count; ++c) {
                        auto& candidate = candidates[c];
                        const int removable = candidate.hidden_point
                            - candidate.hidden_lower;
                        if (removable <= 0) continue;
                        const auto share = static_cast<std::int64_t>(
                            static_cast<long double>(excess)
                                * static_cast<long double>(removable)
                                / static_cast<long double>(total_removable)
                        );
                        const int take = static_cast<int>(std::min<std::int64_t>(
                            removable, share
                        ));
                        candidate.hidden_point -= take;
                        removed += take;
                    }
                    remaining += removed;
                    if (removed == 0) {
                        for (std::size_t c = 0; c < candidate_count; ++c) {
                            auto& candidate = candidates[c];
                            if (candidate.hidden_point > candidate.hidden_lower) {
                                --candidate.hidden_point;
                                ++remaining;
                                break;
                            }
                        }
                    }
                }

                for (std::size_t c = 0; c < candidate_count; ++c) {
                    const auto& candidate = candidates[c];
                    const auto i = candidate.product;
                    joint_point_[i] = nonnegative(
                        joint_point_[i] - candidate.hidden_point
                    );
                    joint_upper_[i] = nonnegative(
                        joint_upper_[i] - candidate.hidden_lower
                    );
                    clearance[i] = candidate.visible + candidate.hidden_point;
                    clearance_interval.lower[i] = candidate.visible
                        + candidate.hidden_lower;
                    // A private discard can add further unpriced liquidation;
                    // cash still proves the lower bound but cannot cap it.
                    const int liquidation_upper = state.possible_private_discard[i]
                        ? candidate.capacity : candidate.hidden_upper;
                    clearance_interval.upper[i] = std::max(
                        clearance_interval.lower[i],
                        candidate.visible + liquidation_upper
                    );
                    cash_constraint_applied[i] = true;
                }

                // Project the correlated joint BUY interval.  A negative net
                // market flow proves a per-product minimum; the remaining
                // churn mass is distributed deterministically by that point
                // evidence and then product index.
                if (has_buy_candidate) {
                    const std::int64_t sum_buy_upper = std::accumulate(
                        buy_upper.begin(), buy_upper.end(), std::int64_t{0}
                    );
                    std::int64_t allocated_buy_point = 0;
                    for (std::size_t i = 0; i < product_count; ++i) {
                        if (!buy_candidate[i]) continue;
                        floor_purchase_interval.lower[i] = static_cast<int>(
                            std::max<std::int64_t>(
                                buy_lower[i],
                                joint_buy_lower - (sum_buy_upper - buy_upper[i])
                            )
                        );
                        floor_purchase_interval.upper[i] = static_cast<int>(
                            std::min<std::int64_t>(
                                buy_upper[i],
                                joint_buy_upper - (buy_lower_sum - buy_lower[i])
                            )
                        );
                        floor_purchase[i] = std::clamp(
                            buy_point[i],
                            floor_purchase_interval.lower[i],
                            floor_purchase_interval.upper[i]
                        );
                        allocated_buy_point += floor_purchase[i];
                        floor_purchase_constraint_applied[i] = true;
                    }
                    std::int64_t remaining_buy = feasible_buy_point
                        - allocated_buy_point;
                    while (remaining_buy > 0) {
                        bool moved = false;
                        for (std::size_t i = 0; i < product_count && remaining_buy > 0; ++i) {
                            if (!buy_candidate[i]
                                || floor_purchase[i]
                                    >= floor_purchase_interval.upper[i]) continue;
                            ++floor_purchase[i];
                            --remaining_buy;
                            moved = true;
                        }
                        if (!moved) break;
                    }
                    while (remaining_buy < 0) {
                        bool moved = false;
                        for (std::size_t i = product_count; i-- > 0 && remaining_buy < 0;) {
                            if (!buy_candidate[i]
                                || floor_purchase[i]
                                    <= floor_purchase_interval.lower[i]) continue;
                            --floor_purchase[i];
                            ++remaining_buy;
                            moved = true;
                        }
                        if (!moved) break;
                    }
                }
            }
        }
    }

    if (state.opponent_shed_access_ambiguity || state.day_rollover
        || state.day != previous_.day) {
        // DROP is the useful point hypothesis.  PICKUP/PASS and overflow are
        // publicly indistinguishable, so shed bounds remain conservative.
        carried_point_.fill(0);
        shed_partition_ambiguous_ = true;
        for (std::size_t i = 0; i < product_count; ++i) {
            joint_lower_[i] = 0;
            ambiguous_[i] = true;
        }
    }

    previous_ = state;
    auto result = snapshot(state, clearance, clearance_interval, saturated);
    result.cash_constraint_applied = cash_constraint_applied;
    result.recent_floor_purchase = floor_purchase;
    result.recent_floor_purchase_interval = floor_purchase_interval;
    result.floor_purchase_constraint_applied = floor_purchase_constraint_applied;
    return result;
}

InventoryBelief OpponentInventoryBelief::snapshot(
    const PublicStateSummary& state,
    const Inventory& clearance,
    const InventoryInterval& clearance_interval,
    const ProductFlags& saturated
) const {
    InventoryBelief result;
    result.step = state.step;
    result.recent_clearance = clearance;
    result.recent_clearance_interval = clearance_interval;
    result.saturated = saturated;
    result.ambiguous = ambiguous_;

    for (std::size_t i = 0; i < product_count; ++i) {
        result.price_inconsistent[i] = state.market_price[i]
            != price(static_cast<Product>(i), state.market_inventory[i]);
        result.total_interval.lower[i] = nonnegative(joint_lower_[i] - state.own_total[i]);
        result.total_interval.upper[i] = std::max(
            result.total_interval.lower[i],
            nonnegative(joint_upper_[i] - state.own_total[i])
        );
        // Later public-cash and saturated-market constraints can tighten the
        // certified interval without changing the heuristic joint point by
        // the same amount.  A point estimate is never allowed to escape its
        // own interval; project it after every tightening step.
        result.total[i] = std::clamp(
            nonnegative(joint_point_[i] - state.own_total[i]),
            result.total_interval.lower[i], result.total_interval.upper[i]
        );

        if (shed_partition_ambiguous_ || ambiguous_[i]) {
            result.shed_interval.lower[i] = 0;
            result.shed_interval.upper[i] = std::min(
                shed_capacity_, result.total_interval.upper[i]
            );
        } else {
            result.shed_interval.lower[i] = std::min(
                shed_capacity_,
                nonnegative(result.total_interval.lower[i] - carried_point_[i])
            );
            result.shed_interval.upper[i] = std::min(
                shed_capacity_,
                nonnegative(result.total_interval.upper[i] - carried_point_[i])
            );
        }
        result.shed[i] = std::clamp(
            std::min(shed_capacity_,
                     nonnegative(result.total[i] - carried_point_[i])),
            result.shed_interval.lower[i], result.shed_interval.upper[i]
        );

        result.likely_cleared[i] = clearance[i] > 0
            && result.total[i] == 0
            && result.total_interval.upper[i] == 0;
    }

    auto lower_sum = std::accumulate(
        result.shed_interval.lower.begin(), result.shed_interval.lower.end(),
        std::int64_t{0});
    if (lower_sum > shed_capacity_) {
        result.shed_interval.lower.fill(0);
        result.ambiguous.fill(true);
        lower_sum = 0;
    }
    const auto point_sum = std::accumulate(
        result.shed.begin(), result.shed.end(), std::int64_t{0});
    if (point_sum > shed_capacity_) {
        const auto original = result.shed;
        const std::int64_t budget = shed_capacity_ - lower_sum;
        const std::int64_t slack = point_sum - lower_sum;
        std::int64_t allocated = 0;
        for (std::size_t i = 0; i < product_count; ++i) {
            const std::int64_t room = original[i] - result.shed_interval.lower[i];
            const int kept = static_cast<int>(room * budget / slack);
            result.shed[i] = result.shed_interval.lower[i] + kept;
            allocated += kept;
        }
        for (std::size_t i = 0; allocated < budget; i = (i + 1) % product_count) {
            if (result.shed[i] >= original[i]) continue;
            ++result.shed[i];
            ++allocated;
        }
    }
    return result;
}

}  // namespace g001::market
