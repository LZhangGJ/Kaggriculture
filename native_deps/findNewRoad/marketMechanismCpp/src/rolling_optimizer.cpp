#include "rolling_optimizer.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace g001::rolling {
namespace {

using g001::market::Product;
using g001::market::price;
using g001::market::sell_revenue;

struct SimState {
    std::int64_t own_money{}, opponent_money{};
    Inventory stock{}, opponent_stock{}, market{};
    g001::market::PlayerMarketState own_market{};
    FailureCounts failures{};
    std::int64_t price_impact{};
};

void record_legacy_purchase_failure(SimState& state,
                                    g001::market::Operation operation,
                                    int missing) {
    if (missing <= 0) return;
    if (operation == g001::market::Operation::Hire ||
        operation == g001::market::Operation::BuyLand) {
        state.failures.route_hard += missing;
    } else {
        state.failures.critical_purchase += missing;
    }
}

struct ExactQueueAudit {
    bool attempted_critical{};
    bool attempted_route{};
};

ExactQueueAudit apply_exact_legacy_queue(SimState& state,
                                         const FixedForecast& forecast,
                                         const ForecastTick& tick,
                                         int replaced_product) {
    ExactQueueAudit audit;
    std::vector<g001::market::Order> queue = tick.legacy_market_orders;
    for (auto& order : queue) {
        if (replaced_product >= 0 &&
            order.operation == g001::market::Operation::Sell &&
            static_cast<int>(order.product) == replaced_product) {
            order.operation = g001::market::Operation::Pass;
            order.quantity = 0;
        }
    }

    state.own_market.money = state.own_money;
    state.own_market.shed = state.stock;
    std::array<g001::market::PlayerMarketState, 2> players{};
    players[0] = state.own_market;
    std::array<std::vector<g001::market::Order>, 2> queues{};
    queues[0] = queue;
    const auto replay = g001::market::simulate_queue(
        state.market, players, queues, 10,
        std::max(0, forecast.official_shed_capacity),
        std::max(1, forecast.hire_cost_multiplier));

    for (std::size_t slot = 0; slot < queue.size(); ++slot) {
        const auto& order = queue[slot];
        if (order.operation == g001::market::Operation::Pass ||
            order.operation == g001::market::Operation::Sell) continue;
        if (order.operation == g001::market::Operation::Hire ||
            order.operation == g001::market::Operation::BuyLand)
            audit.attempted_route = true;
        else
            audit.attempted_critical = true;
        const int requested = (order.operation == g001::market::Operation::Hire ||
                               order.operation == g001::market::Operation::BuyLand)
            ? static_cast<int>(order.quantity > 0) : std::max(0, order.quantity);
        const int committed = slot < replay.committed[0].size()
            ? replay.committed[0][slot] : 0;
        record_legacy_purchase_failure(
            state, order.operation, std::max(0, requested - committed));
    }
    state.own_market = replay.players[0];
    state.own_money = state.own_market.money;
    state.stock = state.own_market.shed;
    state.market = replay.market_inventory;
    return audit;
}

struct Active {
    PersistentOption option{};
    int remaining_quota{}, remaining_windows{}, debt{};
    bool acted{};
};

int stock_sum(const Inventory& stock) {
    return std::accumulate(stock.begin(), stock.end(), 0);
}

int apply_joint_sale(SimState& state, int product, int requested,
                     int opponent_requested, int impact_limit,
                     SameTickOrder order) {
    if (product < 0 || product >= int(g001::market::product_count)) return 0;
    requested = std::min(std::max(0, requested), state.stock[product]);
    opponent_requested = std::min(
        std::max(0, opponent_requested), state.opponent_stock[product]);
    if (requested == 0 && opponent_requested == 0) return 0;
    const auto item = static_cast<Product>(product);
    const int initial_quote = price(item, state.market[product]);
    int own_accepted = 0;
    int opponent_accepted = 0;
    auto own_unit = [&](int quote) {
        if (own_accepted >= requested ||
            initial_quote - quote > impact_limit) return false;
        state.own_money += quote;
        state.price_impact += initial_quote - quote;
        if (quote > 1) ++state.market[product];
        --state.stock[product];
        ++own_accepted;
        return true;
    };
    auto opponent_unit = [&](int quote) {
        if (opponent_accepted >= opponent_requested) return false;
        state.opponent_money += quote;
        if (quote > 1) ++state.market[product];
        --state.opponent_stock[product];
        ++opponent_accepted;
        return true;
    };
    if (order == SameTickOrder::Lockstep) {
        while (own_accepted < requested || opponent_accepted < opponent_requested) {
            const int shared_quote = price(item, state.market[product]);
            const bool own_active = own_accepted < requested &&
                initial_quote - shared_quote <= impact_limit;
            const bool opponent_active = opponent_accepted < opponent_requested;
            if (!own_active && !opponent_active) break;
            // Both quotes are captured before either inventory mutation.
            if (own_active) own_unit(shared_quote);
            if (opponent_active) opponent_unit(shared_quote);
        }
        return own_accepted;
    }
    auto drain_own = [&] {
      while (own_accepted < requested) {
        const int quote = price(item, state.market[product]);
        if (!own_unit(quote)) break;
      }
    };
    auto drain_opponent = [&] {
      while (opponent_accepted < opponent_requested)
        opponent_unit(price(item, state.market[product]));
    };
    if (order == SameTickOrder::OpponentFirst) {
        drain_opponent();
        drain_own();
    } else {
        drain_own();
        drain_opponent();
    }
    return own_accepted;
}

int quantity_inside_band(const SimState& state, int product, int requested,
                         int reservation_price, int impact_limit) {
    if (product < 0 || product >= int(g001::market::product_count) ||
        requested <= 0) return 0;
    requested = std::min(requested, state.stock[product]);
    const auto item = static_cast<Product>(product);
    const int initial_quote = price(item, state.market[product]);
    int accepted = 0;
    int simulated_inventory = state.market[product];
    while (accepted < requested) {
        const int marginal_quote = price(item, simulated_inventory);
        if (reservation_price > 0 && marginal_quote < reservation_price) break;
        if (initial_quote - marginal_quote > impact_limit) break;
        if (marginal_quote > 1) ++simulated_inventory;
        ++accepted;
    }
    return accepted;
}

void apply_option(SimState& state, Active& active, bool sale_window, int step,
                  const std::vector<OpponentDump>& dumps, bool pre_dump_now,
                  int concurrent_opponent, SameTickOrder same_tick_order) {
    if (!sale_window) {
        apply_joint_sale(state, active.option.product, 0, concurrent_opponent,
                         std::numeric_limits<int>::max(), same_tick_order);
        return;
    }
    auto& option = active.option;
    const int product = option.product;
    if (product < 0 || product >= int(g001::market::product_count)) return;
    int requested = 0;
    if (option.kind == g001::option::Kind::Baseline) {
        return;
    } else if (option.kind == g001::option::Kind::Hold) {
        apply_joint_sale(state, product, 0, concurrent_opponent,
                         std::numeric_limits<int>::max(), same_tick_order);
        active.acted = true;
        return;
    } else if (option.kind == g001::option::Kind::Drip && active.remaining_windows > 0) {
        const auto transition = g001::option::settle_drip(
            {active.remaining_quota, active.remaining_windows, active.debt},
            state.stock[product], 0
        );
        // A persistent DRIP may also carry a public price/inventory band.  If
        // the quote falls below it (for example after an observed rival dump),
        // the unfilled amount becomes debt and is retried at later sale
        // windows after town drain restores the quote.
        requested = quantity_inside_band(
            state, product, transition.requested,
            option.reservation_price, option.impact_limit);
        const int before = state.stock[product];
        const int filled = apply_joint_sale(
            state, product, requested, concurrent_opponent,
            option.impact_limit, same_tick_order);
        const auto settled = g001::option::settle_drip(
            {active.remaining_quota, active.remaining_windows, active.debt}, before, filled
        );
        active.remaining_quota = settled.next.remaining_quota;
        active.remaining_windows = settled.next.remaining_windows;
        active.debt = settled.next.debt;
        active.acted = true;
        return;
    } else if (option.kind == g001::option::Kind::PriceTarget) {
        while (requested < state.stock[product] &&
               price(static_cast<Product>(product), state.market[product] + requested) >=
                   option.reservation_price &&
               price(static_cast<Product>(product), state.market[product]) -
                   price(static_cast<Product>(product), state.market[product] + requested) <=
                   option.impact_limit) ++requested;
    } else if (option.kind == g001::option::Kind::InventoryTarget) {
        requested = std::max(0, state.stock[product] - option.target_inventory);
    } else if (option.kind == g001::option::Kind::Clear && !active.acted) {
        requested = state.stock[product];
    } else if (option.kind == g001::option::Kind::PreDump && !active.acted) {
        (void)step; (void)dumps;
        if (pre_dump_now) requested = std::min(option.quota > 0 ? option.quota : state.stock[product], state.stock[product]);
    }
    apply_joint_sale(state, product, requested, concurrent_opponent,
                     option.impact_limit, same_tick_order);
    active.acted |= requested > 0;
}

void apply_requirements(SimState& state,
                        const std::vector<Requirement>& requirements,
                        bool consume_product) {
    for (const auto& requirement : requirements) {
        bool failed = false;
        if (requirement.cash > state.own_money) failed = true;
        if (requirement.product >= 0) {
            if (requirement.product >= int(g001::market::product_count) ||
                state.stock[requirement.product] < requirement.quantity) failed = true;
        }
        if (failed) {
            if (requirement.kind == RequirementKind::CriticalPurchase) ++state.failures.critical_purchase;
            else if (requirement.kind == RequirementKind::Feed) ++state.failures.feed;
            else ++state.failures.route_hard;
            continue;
        }
        state.own_money -= requirement.cash;
        if (consume_product && requirement.product >= 0)
            state.stock[requirement.product] -= requirement.quantity;
    }
}

void apply_delta(SimState& state, const StockDelta& delta) {
    for (std::size_t product = 0; product < g001::market::product_count; ++product) {
        // A pre-market requirement records a causal underflow as a typed
        // failure.  Clamp here so a failed/partial official PICKUP never creates
        // physically negative private inventory.
        state.stock[product] = std::max(0, state.stock[product] + delta[product]);
    }
}

bool has_delta(const StockDelta& delta) {
    return std::any_of(delta.begin(), delta.end(), [](int value) { return value != 0; });
}

void enforce_capacity(SimState& state, const FixedForecast& forecast) {
    const int overflow = std::max(0, stock_sum(state.stock) - forecast.shed_capacity);
    state.failures.overflow += overflow;
    for (std::size_t p = g001::market::product_count;
         stock_sum(state.stock) > forecast.shed_capacity && p-- > 0;) {
        const int excess = stock_sum(state.stock) - forecast.shed_capacity;
        const int drop = std::min(excess, state.stock[p]);
        state.stock[p] -= drop;
    }
}

SimState simulate(const CurrentState& current, const FixedForecast& forecast,
                  const Scenario& scenario, const std::vector<PersistentOption>& sequence) {
    SimState state;
    state.own_money = current.own_money;
    state.opponent_money = current.opponent_money;
    state.stock = current.own_stock;
    state.opponent_stock = scenario.belief_stock;
    state.market = current.market_inventory;
    state.own_market = forecast.initial_own_market;
    state.own_market.money = current.own_money;
    state.own_market.shed = current.own_stock;
    std::size_t decision = 0;
    Active active{};
    auto reset_active = [&](const PersistentOption& option, std::size_t tick_index) {
        active = {}; active.option = option;
        const auto stop = std::min(forecast.ticks.size(), tick_index + std::max(1, option.window_steps));
        for (std::size_t i = tick_index; i < stop; ++i) active.remaining_windows += forecast.ticks[i].sale_window;
        active.remaining_windows = std::max(1, active.remaining_windows);
        active.remaining_quota = option.quota > 0 ? option.quota :
            (option.product >= 0 && option.product < int(g001::market::product_count)
                 ? state.stock[option.product] : 0);
    };
    if (!sequence.empty()) reset_active(sequence.front(), 0);
    for (std::size_t i = 0; i < forecast.ticks.size(); ++i) {
        const auto& tick = forecast.ticks[i];
        if (i > 0 && forecast.turns_per_day > 0 &&
            tick.step % forecast.turns_per_day == 0) {
            state.own_market.hires_today = 0;
        }
        if (i > 0 && tick.decision_epoch && decision + 1 < sequence.size()) {
            ++decision; reset_active(sequence[decision], i);
        }
        // Official causal order: route-fixed unit transfers happen before any
        // market option.  Validation and delta are split to avoid counting a
        // PICKUP/FEED reserve twice.
        apply_requirements(state, tick.pre_market_requirements, false);
        apply_delta(state, tick.pre_market_unit_delta);
        // Preserve the exact legacy zero-field behavior: its first capacity
        // check remains after the market/old production phase.
        if (has_delta(tick.pre_market_unit_delta)) enforce_capacity(state, forecast);
        Inventory opponent_requested{};
        for (const auto& dump : scenario.dumps) {
            if (dump.step == tick.step && dump.product >= 0 &&
                dump.product < int(g001::market::product_count)) {
                opponent_requested[dump.product] += dump.quantity;
            }
        }
        std::array<bool, g001::market::product_count> opponent_settled{};
        auto settle_joint = [&](int product, int own_requested, int impact_limit) {
            opponent_settled[product] = true;
            return apply_joint_sale(
                state, product, own_requested, opponent_requested[product],
                impact_limit, scenario.same_tick_order);
        };
        const bool exact_legacy = tick.legacy_market_queue_known;
        ExactQueueAudit exact_audit;
        if (exact_legacy) {
            // The abstract OpponentFirst scenario is deliberately stronger
            // than official same-slot lockstep.  It is applied before every
            // own legacy slot; OwnFirst/Lockstep pressure is settled after the
            // exact own queue below.
            if (scenario.same_tick_order == SameTickOrder::OpponentFirst) {
                for (std::size_t p = 0; p < g001::market::product_count; ++p) {
                    if (opponent_requested[p] > 0)
                        settle_joint(static_cast<int>(p), 0,
                                     std::numeric_limits<int>::max());
                }
            }
            int replaced_product = -1;
            if (!sequence.empty() &&
                active.option.kind != g001::option::Kind::Baseline)
                replaced_product = active.option.product;
            exact_audit = apply_exact_legacy_queue(
                state, forecast, tick, replaced_product);

            if (!sequence.empty() &&
                active.option.kind != g001::option::Kind::Baseline) {
                const int product = active.option.product;
                if (product < 0 || product >= int(g001::market::product_count))
                    throw std::runtime_error("invalid active option product");
                bool pre_dump_now = false;
                if (active.option.kind == g001::option::Kind::PreDump && tick.sale_window) {
                    int earliest = std::numeric_limits<int>::max();
                    for (const auto& dump : scenario.dumps)
                        if (dump.product == product) earliest = std::min(earliest, dump.step);
                    pre_dump_now = tick.step <= earliest;
                }
                const int concurrent = scenario.same_tick_order == SameTickOrder::OpponentFirst
                    ? 0 : opponent_requested[product];
                opponent_settled[product] = opponent_settled[product] || concurrent > 0;
                // Conservative cash-prefix contract: replacement SELL is
                // materialized only after every immutable legacy slot, so it
                // can never finance an earlier BUY in this tick.
                apply_option(state, active, tick.sale_window, tick.step,
                             scenario.dumps, pre_dump_now, concurrent,
                             scenario.same_tick_order);
            }
        } else if (!sequence.empty()) {
            for (std::size_t p = 0; p < g001::market::product_count; ++p)
                if (active.option.kind == g001::option::Kind::Baseline || int(p) != active.option.product) {
                    if (tick.baseline_sale[p] > 0 || opponent_requested[p] > 0)
                        settle_joint(int(p), tick.baseline_sale[p], std::numeric_limits<int>::max());
                }
            bool pre_dump_now = false;
            if (active.option.kind == g001::option::Kind::PreDump && tick.sale_window) {
                int earliest = std::numeric_limits<int>::max();
                for (const auto& dump : scenario.dumps) if (dump.product == active.option.product)
                    earliest = std::min(earliest, dump.step);
                pre_dump_now = tick.step <= earliest;
                for (std::size_t later = i + 1; later < forecast.ticks.size(); ++later) {
                    if (forecast.ticks[later].step > earliest) break;
                    if (forecast.ticks[later].sale_window) { pre_dump_now = false; break; }
                }
            }
            if (active.option.kind != g001::option::Kind::Baseline) {
                const int product = active.option.product;
                if (product < 0 || product >= int(g001::market::product_count))
                    throw std::runtime_error("invalid active option product");
                opponent_settled[product] = true;
                apply_option(state, active, tick.sale_window, tick.step,
                             scenario.dumps, pre_dump_now,
                             opponent_requested[product], scenario.same_tick_order);
            }
        } else for (std::size_t p = 0; p < g001::market::product_count; ++p) {
            if (tick.baseline_sale[p] > 0 || opponent_requested[p] > 0)
                settle_joint(int(p), tick.baseline_sale[p], std::numeric_limits<int>::max());
        }
        for (std::size_t product = 0; product < g001::market::product_count; ++product) {
            if (!opponent_settled[product] && opponent_requested[product] > 0) {
                settle_joint(int(product), 0, std::numeric_limits<int>::max());
            }
        }
        // Official phase order: both players' ten market slots settle first;
        // shops/town center consume public inventory only afterwards.  Moving
        // this drain before the queue fabricates a recovered quote for the
        // current SELL and can also underquote current BUY_PRODUCT costs.
        for (std::size_t p = 0; p < g001::market::product_count; ++p)
            state.market[p] = std::max(0, state.market[p] - tick.town_drain[p]);
        if (exact_legacy) {
            // Exact legacy non-SELL slots supersede the old inferred cash
            // reservation and purchase delta. Product-consumption obligations
            // remain active. This prevents a failed BUY from fabricating shed
            // inventory via post_market_delta.
            std::vector<Requirement> stock_requirements;
            stock_requirements.reserve(tick.requirements.size());
            for (const auto& requirement : tick.requirements)
                if (requirement.product >= 0) stock_requirements.push_back(requirement);
            apply_requirements(state, stock_requirements, true);
            for (const auto& requirement : tick.requirements) {
                if (requirement.product >= 0 || requirement.cash <= 0) continue;
                const bool represented = requirement.kind == RequirementKind::RouteHard
                    ? exact_audit.attempted_route : exact_audit.attempted_critical;
                if (!represented) {
                    if (requirement.kind == RequirementKind::RouteHard)
                        ++state.failures.route_hard;
                    else if (requirement.kind == RequirementKind::Feed)
                        ++state.failures.feed;
                    else
                        ++state.failures.critical_purchase;
                }
            }
        } else {
            apply_requirements(state, tick.requirements, true);
            apply_delta(state, tick.post_market_delta);
        }
        for (std::size_t p = 0; p < g001::market::product_count; ++p) state.stock[p] += tick.production[p];
        enforce_capacity(state, forecast);
        apply_delta(state, tick.end_of_day_return);
        enforce_capacity(state, forecast);
    }
    if (forecast.liquidate_own_at_end)
        for (std::size_t p = 0; p < g001::market::product_count; ++p)
            apply_joint_sale(
                state, static_cast<int>(p), state.stock[p],
                state.opponent_stock[p], std::numeric_limits<int>::max(),
                SameTickOrder::OpponentFirst);
    return state;
}

RobustScore aggregate(const std::vector<SimState>& states, const std::vector<Scenario>& scenarios,
                      int complexity, double cvar_fraction) {
    RobustScore score; score.worst_margin = std::numeric_limits<std::int64_t>::max();
    score.worst_own_terminal_money = std::numeric_limits<std::int64_t>::max();
    std::vector<std::pair<std::int64_t,double>> margins;
    for (std::size_t index = 0; index < states.size(); ++index) {
        const auto& state = states[index]; const double weight = scenarios[index].weight;
        score.worst_failures.critical_purchase = std::max(score.worst_failures.critical_purchase, state.failures.critical_purchase);
        score.worst_failures.feed = std::max(score.worst_failures.feed, state.failures.feed);
        score.worst_failures.overflow = std::max(score.worst_failures.overflow, state.failures.overflow);
        score.worst_failures.route_hard = std::max(score.worst_failures.route_hard, state.failures.route_hard);
        const auto margin = state.own_money - state.opponent_money; margins.emplace_back(margin,weight);
        score.worst_margin = std::min(score.worst_margin, margin);
        score.worst_own_terminal_money = std::min(score.worst_own_terminal_money, state.own_money);
        score.expected_margin += static_cast<double>(margin) * weight;
        score.expected_own_terminal_money += static_cast<double>(state.own_money) * weight;
        score.worst_price_impact = std::max(score.worst_price_impact, state.price_impact);
    }
    score.nonnegative_margin_feasible = score.worst_margin >= 0;
    std::sort(margins.begin(), margins.end(), [](const auto& a,const auto& b){return a.first<b.first;});
    double remaining=cvar_fraction,tail=0;
    for(const auto& [margin,weight]:margins){const double used=std::min(remaining,weight);tail+=used*margin;remaining-=used;if(remaining<=1e-15)break;}
    score.cvar_margin=tail/cvar_fraction;
    score.complexity = complexity;
    auto risk = [&](auto field) {
        FailureRisk result; int affected = 0; double total = 0;
        double affected_weight = 0;
        for (std::size_t index = 0; index < states.size(); ++index) {
            const int value = field(states[index].failures); result.worst = std::max(result.worst, value);
            const double weight=scenarios[index].weight; total += value*weight;
            affected += value > 0; affected_weight += value>0 ? weight : 0.0;
        }
        (void)affected;
        result.affected_scenario_fraction = affected_weight;
        result.mean_severity = total;
        result.conditional_severity = affected_weight > 0 ? total / affected_weight : 0.0;
        return result;
    };
    score.risk.critical_purchase = risk([](const FailureCounts& f){return f.critical_purchase;});
    score.risk.feed = risk([](const FailureCounts& f){return f.feed;});
    score.risk.overflow = risk([](const FailureCounts& f){return f.overflow;});
    score.risk.route_hard = risk([](const FailureCounts& f){return f.route_hard;});
    return score;
}

struct Plan { std::vector<PersistentOption> sequence; RobustScore score; };

bool economic_better(const RobustScore& a, const RobustScore& b) {
    if (a.nonnegative_margin_feasible != b.nonnegative_margin_feasible)
        return a.nonnegative_margin_feasible;
    if (a.nonnegative_margin_feasible) {
        if (a.worst_own_terminal_money != b.worst_own_terminal_money)
            return a.worst_own_terminal_money > b.worst_own_terminal_money;
        if (a.worst_margin != b.worst_margin) return a.worst_margin > b.worst_margin;
        if (a.cvar_margin != b.cvar_margin) return a.cvar_margin > b.cvar_margin;
        if (a.expected_own_terminal_money != b.expected_own_terminal_money)
            return a.expected_own_terminal_money > b.expected_own_terminal_money;
        if (a.expected_margin != b.expected_margin) return a.expected_margin > b.expected_margin;
    } else {
        if (a.worst_margin != b.worst_margin) return a.worst_margin > b.worst_margin;
        if (a.cvar_margin != b.cvar_margin) return a.cvar_margin > b.cvar_margin;
        if (a.worst_own_terminal_money != b.worst_own_terminal_money)
            return a.worst_own_terminal_money > b.worst_own_terminal_money;
        if (a.expected_margin != b.expected_margin) return a.expected_margin > b.expected_margin;
        if (a.expected_own_terminal_money != b.expected_own_terminal_money)
            return a.expected_own_terminal_money > b.expected_own_terminal_money;
    }
    if (a.complexity != b.complexity) return a.complexity < b.complexity;
    return a.worst_price_impact < b.worst_price_impact;
}

FailureCounts envelope_excess(const RobustScore& candidate, const RobustScore& baseline,
                              const RelativeFailureBudget& budget) {
    return {
        std::max(0, candidate.worst_failures.critical_purchase - baseline.worst_failures.critical_purchase - budget.critical_purchase),
        std::max(0, candidate.worst_failures.feed - baseline.worst_failures.feed - budget.feed),
        std::max(0, candidate.worst_failures.overflow - baseline.worst_failures.overflow - budget.overflow),
        std::max(0, candidate.worst_failures.route_hard - baseline.worst_failures.route_hard - budget.route_hard),
    };
}

bool zero(const FailureCounts& f) {
    return f.critical_purchase == 0 && f.feed == 0 && f.overflow == 0 && f.route_hard == 0;
}

bool catastrophic(const RobustScore& score, const CatastropheLimit& limit) {
    return score.worst_failures.critical_purchase > limit.critical_purchase ||
        score.worst_failures.feed > limit.feed || score.worst_failures.overflow > limit.overflow ||
        score.worst_failures.route_hard > limit.route_hard;
}

bool dominates(const ParetoPoint& a, const ParetoPoint& b) {
    const auto& x=a.excess_over_envelope; const auto& y=b.excess_over_envelope;
    const bool no_worse = x.critical_purchase<=y.critical_purchase && x.feed<=y.feed &&
        x.overflow<=y.overflow && x.route_hard<=y.route_hard &&
        a.score.worst_margin>=b.score.worst_margin && a.score.cvar_margin>=b.score.cvar_margin &&
        a.score.worst_margin_delta_vs_baseline>=
            b.score.worst_margin_delta_vs_baseline &&
        a.score.expected_margin>=b.score.expected_margin &&
        a.score.worst_own_terminal_money>=b.score.worst_own_terminal_money &&
        a.score.worst_own_terminal_delta_vs_baseline>=
            b.score.worst_own_terminal_delta_vs_baseline &&
        a.score.expected_own_terminal_money>=b.score.expected_own_terminal_money &&
        a.score.complexity<=b.score.complexity && a.score.worst_price_impact<=b.score.worst_price_impact;
    const bool strict = x.critical_purchase<y.critical_purchase || x.feed<y.feed ||
        x.overflow<y.overflow || x.route_hard<y.route_hard ||
        a.score.worst_margin>b.score.worst_margin || a.score.cvar_margin>b.score.cvar_margin ||
        a.score.worst_margin_delta_vs_baseline>
            b.score.worst_margin_delta_vs_baseline ||
        a.score.expected_margin>b.score.expected_margin ||
        a.score.worst_own_terminal_money>b.score.worst_own_terminal_money ||
        a.score.worst_own_terminal_delta_vs_baseline>
            b.score.worst_own_terminal_delta_vs_baseline ||
        a.score.expected_own_terminal_money>b.score.expected_own_terminal_money ||
        a.score.complexity<b.score.complexity || a.score.worst_price_impact<b.score.worst_price_impact;
    return no_worse && strict;
}

bool same_pareto_objectives(const ParetoPoint& a, const ParetoPoint& b) {
    const auto& x = a.excess_over_envelope;
    const auto& y = b.excess_over_envelope;
    return x.critical_purchase == y.critical_purchase && x.feed == y.feed &&
        x.overflow == y.overflow && x.route_hard == y.route_hard &&
        a.score.worst_margin == b.score.worst_margin &&
        a.score.worst_margin_delta_vs_baseline ==
            b.score.worst_margin_delta_vs_baseline &&
        a.score.cvar_margin == b.score.cvar_margin &&
        a.score.expected_margin == b.score.expected_margin &&
        a.score.worst_own_terminal_money == b.score.worst_own_terminal_money &&
        a.score.worst_own_terminal_delta_vs_baseline ==
            b.score.worst_own_terminal_delta_vs_baseline &&
        a.score.expected_own_terminal_money == b.score.expected_own_terminal_money &&
        a.score.complexity == b.score.complexity &&
        a.score.worst_price_impact == b.score.worst_price_impact;
}

}  // namespace

TerminalState simulate_terminal(
    const CurrentState& current,
    const FixedForecast& forecast,
    const Scenario& scenario,
    const std::vector<PersistentOption>& sequence
) {
    if (scenario.realized_future)
        throw std::runtime_error("realized future outcome is forbidden as rollout input");
    const auto state = simulate(current, forecast, scenario, sequence);
    return {state.own_money, state.opponent_money, state.stock,
            state.opponent_stock, state.market, state.failures,
            state.own_market.hires_today, state.own_market.hands};
}

bool better(const RobustScore& a, const RobustScore& b) {
    auto failures = [](const FailureCounts& f) { return std::array{f.critical_purchase, f.feed, f.overflow, f.route_hard}; };
    if (failures(a.worst_failures) != failures(b.worst_failures)) return failures(a.worst_failures) < failures(b.worst_failures);
    if (a.nonnegative_margin_feasible != b.nonnegative_margin_feasible) return a.nonnegative_margin_feasible;
    if (a.nonnegative_margin_feasible) {
        // Once robust survival is secured, preserve own economy rather than
        // destroying both farms merely to enlarge an already nonnegative gap.
        if (a.worst_own_terminal_money != b.worst_own_terminal_money) return a.worst_own_terminal_money > b.worst_own_terminal_money;
        if (a.worst_margin != b.worst_margin) return a.worst_margin > b.worst_margin;
        if (a.cvar_margin != b.cvar_margin) return a.cvar_margin > b.cvar_margin;
        if (a.expected_own_terminal_money != b.expected_own_terminal_money) return a.expected_own_terminal_money > b.expected_own_terminal_money;
        if (a.expected_margin != b.expected_margin) return a.expected_margin > b.expected_margin;
    } else {
        if (a.worst_margin != b.worst_margin) return a.worst_margin > b.worst_margin;
        if (a.cvar_margin != b.cvar_margin) return a.cvar_margin > b.cvar_margin;
        if (a.worst_own_terminal_money != b.worst_own_terminal_money) return a.worst_own_terminal_money > b.worst_own_terminal_money;
        if (a.expected_margin != b.expected_margin) return a.expected_margin > b.expected_margin;
        if (a.expected_own_terminal_money != b.expected_own_terminal_money) return a.expected_own_terminal_money > b.expected_own_terminal_money;
    }
    if (a.complexity != b.complexity) return a.complexity < b.complexity;
    return a.worst_price_impact < b.worst_price_impact;
}

Result optimize(const CurrentState& current, const FixedForecast& forecast,
                const std::vector<Scenario>& scenarios,
                const std::vector<PersistentOption>& candidates, const Config& config) {
    if (forecast.ticks.size() < 24 || forecast.ticks.size() > 72 || scenarios.empty() || candidates.empty())
        throw std::runtime_error("rolling optimizer requires 24..72 ticks, scenarios and options");
    if (config.beam_width == 0 || config.max_decisions == 0 ||
        !(config.cvar_fraction > 0.0 && config.cvar_fraction <= 1.0))
        throw std::runtime_error("invalid rolling beam/CVaR configuration");
    const auto& budget = config.relative_budget;
    if (budget.critical_purchase < 0 || budget.feed < 0 || budget.overflow < 0 || budget.route_hard < 0)
        throw std::runtime_error("relative failure budgets must be nonnegative");
    for (const auto& scenario : scenarios) if (scenario.realized_future)
        throw std::runtime_error("realized future outcome is forbidden as optimizer input");
    double scenario_weight_sum=0;
    for(const auto& scenario:scenarios){if(!std::isfinite(scenario.weight)||scenario.weight<=0)throw std::runtime_error("scenario weights must be finite and positive");scenario_weight_sum+=scenario.weight;}
    if(std::abs(scenario_weight_sum-1.0)>1e-9)throw std::runtime_error("scenario weights must sum to one");
    for (const auto& scenario : scenarios) for (const auto& dump : scenario.dumps)
        if (dump.quantity < 0 || dump.product < 0 || dump.product >= int(g001::market::product_count))
            throw std::runtime_error("invalid causal opponent dump scenario");
    std::size_t epochs = 1;
    for (std::size_t i = 1; i < forecast.ticks.size(); ++i) epochs += forecast.ticks[i].decision_epoch;
    const auto depth = std::min({config.max_decisions, epochs, std::size_t(6)});
    PersistentOption baseline_option; baseline_option.kind = g001::option::Kind::Baseline;
    std::vector<PersistentOption> baseline_sequence(depth, baseline_option);
    std::vector<SimState> baseline_outcomes; baseline_outcomes.reserve(scenarios.size());
    for (const auto& scenario : scenarios)
        baseline_outcomes.push_back(simulate(current, forecast, scenario, baseline_sequence));
    const auto baseline_score = aggregate(baseline_outcomes, scenarios, 0, config.cvar_fraction);
    auto paired_baseline_score = baseline_score;
    paired_baseline_score.worst_margin_delta_vs_baseline = 0;
    paired_baseline_score.worst_own_terminal_delta_vs_baseline = 0;
    Plan baseline_plan{baseline_sequence, paired_baseline_score};
    auto plan_better = [&](const Plan& a, const Plan& b) {
        const bool a_cat = catastrophic(a.score, config.catastrophe_limit);
        const bool b_cat = catastrophic(b.score, config.catastrophe_limit);
        if (a_cat != b_cat) return !a_cat;
        const auto ax = envelope_excess(a.score, baseline_score, budget);
        const auto bx = envelope_excess(b.score, baseline_score, budget);
        const bool a_inside = zero(ax), b_inside = zero(bx);
        if (a_inside != b_inside) return a_inside;
        if (a_inside) {
            if (config.prefer_fewer_failures_within_envelope) {
                const auto af = std::array{
                    a.score.worst_failures.critical_purchase,
                    a.score.worst_failures.feed,
                    a.score.worst_failures.route_hard,
                    a.score.worst_failures.overflow};
                const auto bf = std::array{
                    b.score.worst_failures.critical_purchase,
                    b.score.worst_failures.feed,
                    b.score.worst_failures.route_hard,
                    b.score.worst_failures.overflow};
                if (af != bf) return af < bf;
            }
            return economic_better(a.score, b.score);
        }
        const auto av = std::array{ax.critical_purchase,ax.feed,ax.overflow,ax.route_hard};
        const auto bv = std::array{bx.critical_purchase,bx.feed,bx.overflow,bx.route_hard};
        if (av != bv) return av < bv; // auditable lexicographic beam tie-break, never a weighted sum
        return economic_better(a.score,b.score);
    };
    std::vector<Plan> beam(1);
    std::size_t evaluated = 0;
    std::vector<Plan> final_candidates;
    for (std::size_t level = 0; level < depth; ++level) {
        std::vector<Plan> expanded;
        for (const auto& parent : beam) for (const auto& option : candidates) {
            Plan plan = parent; plan.sequence.push_back(option); expanded.push_back(std::move(plan));
        }
        std::vector<std::vector<SimState>> outcomes(
            expanded.size(), std::vector<SimState>(scenarios.size())
        );
        std::atomic<std::size_t> next{0};
        const auto task_count = expanded.size() * scenarios.size();
        const auto threads = std::min<std::size_t>(task_count, config.threads ? config.threads : std::max(1u, std::thread::hardware_concurrency()));
        std::vector<std::thread> workers;
        for (std::size_t worker = 0; worker < threads; ++worker) workers.emplace_back([&] {
            for (;;) { const auto task = next.fetch_add(1); if (task >= task_count) break;
                const auto plan = task / scenarios.size();
                const auto scenario = task % scenarios.size();
                outcomes[plan][scenario] = simulate(
                    current, forecast, scenarios[scenario], expanded[plan].sequence
                );
            }
        });
        for (auto& worker : workers) worker.join();
        for (std::size_t index = 0; index < expanded.size(); ++index) {
            int complexity = 0; for (std::size_t i = 0; i < expanded[index].sequence.size(); ++i)
                complexity += expanded[index].sequence[i].kind != g001::option::Kind::Baseline &&
                    (i == 0 || expanded[index].sequence[i].kind != expanded[index].sequence[i-1].kind);
            expanded[index].score = aggregate(outcomes[index], scenarios, complexity, config.cvar_fraction);
            expanded[index].score.worst_margin_delta_vs_baseline =
                std::numeric_limits<std::int64_t>::max();
            expanded[index].score.worst_own_terminal_delta_vs_baseline =
                std::numeric_limits<std::int64_t>::max();
            for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
                const auto candidate_margin = outcomes[index][scenario].own_money -
                    outcomes[index][scenario].opponent_money;
                const auto baseline_margin = baseline_outcomes[scenario].own_money -
                    baseline_outcomes[scenario].opponent_money;
                expanded[index].score.worst_margin_delta_vs_baseline =
                    std::min(
                        expanded[index].score.worst_margin_delta_vs_baseline,
                        candidate_margin - baseline_margin);
                expanded[index].score.worst_own_terminal_delta_vs_baseline =
                    std::min(
                        expanded[index].score.worst_own_terminal_delta_vs_baseline,
                        outcomes[index][scenario].own_money -
                            baseline_outcomes[scenario].own_money);
            }
        }
        evaluated += expanded.size();
        if (level + 1 == depth) { final_candidates = expanded; final_candidates.push_back(baseline_plan); }
        std::stable_sort(expanded.begin(), expanded.end(), plan_better);
        if (expanded.size() > config.beam_width) expanded.resize(config.beam_width);
        beam = std::move(expanded);
    }
    if (final_candidates.empty()) final_candidates.push_back(baseline_plan);
    std::stable_sort(final_candidates.begin(), final_candidates.end(), plan_better);
    auto selected = std::find_if(final_candidates.begin(), final_candidates.end(), [&](const Plan& plan) {
        return !catastrophic(plan.score, config.catastrophe_limit);
    });
    if (selected == final_candidates.end()) throw std::runtime_error("all rolling plans exceed catastrophe limits");
    std::vector<ParetoPoint> frontier;
    for (const auto& plan : final_candidates) {
        if (catastrophic(plan.score, config.catastrophe_limit)) continue;
        ParetoPoint point{plan.sequence,plan.score,envelope_excess(plan.score,baseline_score,budget),false};
        point.inside_envelope=zero(point.excess_over_envelope);
        // Incrementally maintain the exact nondominated set.  The former
        // all-pairs scan was quadratic in beam*candidate_count (hundreds of
        // millions of comparisons for stock=100) and dominated runtime.
        const bool is_dominated = std::any_of(
            frontier.begin(), frontier.end(), [&](const auto& existing) {
                return dominates(existing, point) ||
                       same_pareto_objectives(existing, point);
            });
        if (is_dominated) continue;
        frontier.erase(
            std::remove_if(frontier.begin(), frontier.end(),
                [&](const auto& existing) { return dominates(point, existing); }),
            frontier.end());
        frontier.push_back(std::move(point));
    }
    const auto selected_excess=envelope_excess(selected->score,baseline_score,budget);
    return {selected->sequence.front(), selected->sequence, selected->score, baseline_score,
            zero(selected_excess), std::move(frontier), evaluated + 1};
}

}  // namespace g001::rolling
