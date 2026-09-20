#include "adaptive_execution.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace g001::general_econ {
namespace {

int ceil_div(int numerator, int denominator) {
    return denominator <= 0 ? 0 : (numerator + denominator - 1) / denominator;
}

int units_for_cash(
    market::Product product,
    int market_inventory,
    int sellable,
    std::int64_t shortfall
) {
    if (shortfall <= 0) return 0;
    for (int quantity = 1; quantity <= sellable; ++quantity) {
        if (market::sell_revenue(product, market_inventory, quantity) >= shortfall) {
            return quantity;
        }
    }
    return sellable + 1;
}

int inventory_after_sale(market::Product product, int inventory, int quantity) {
    for (int unit = 0; unit < quantity; ++unit) {
        if (market::price(product, inventory) > 1) ++inventory;
    }
    return inventory;
}

TimingScore aggregate(
    std::vector<std::pair<std::int64_t, double>> outcomes,
    double cvar_fraction
) {
    TimingScore score;
    score.worst = std::numeric_limits<std::int64_t>::max();
    for (const auto& [value, weight] : outcomes) {
        score.worst = std::min(score.worst, value);
        score.expected += static_cast<double>(value) * weight;
    }
    std::sort(outcomes.begin(), outcomes.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    double remaining = cvar_fraction;
    for (const auto& [value, weight] : outcomes) {
        const double used = std::min(remaining, weight);
        score.cvar += used * static_cast<double>(value);
        remaining -= used;
        if (remaining <= 1e-15) break;
    }
    score.cvar /= cvar_fraction;
    return score;
}

bool score_better(const TimingScore& left, const TimingScore& right) {
    if (left.worst != right.worst) return left.worst > right.worst;
    if (left.cvar != right.cvar) return left.cvar > right.cvar;
    return left.expected > right.expected;
}

std::pair<int, int> outstanding_bounds(const ExecutionState& state) {
    if (state.outstanding_lower == 0 && state.outstanding_upper == 0 &&
        state.outstanding_units > 0) {
        return {state.outstanding_units, state.outstanding_units};
    }
    return {state.outstanding_lower, state.outstanding_upper};
}

}  // namespace

SellSettlement infer_sell_fills(const SellSettlementInput& input) {
    SellSettlement result;
    for (std::size_t product = 0; product < market::product_count; ++product) {
        const int previous = input.previous_shed[product];
        const int current = input.current_shed[product];
        const int requested = input.requested_total_sell[product];
        const int option_requested = input.requested_option_sell[product];
        const int prior = input.prior_sell[product];
        const int delta_lower = input.non_sell_delta.lower[product];
        const int delta_point = input.non_sell_delta.point[product];
        const int delta_upper = input.non_sell_delta.upper[product];
        if (previous < 0 || current < 0 || requested < 0 || option_requested < 0 ||
            prior < 0 || option_requested + prior > requested ||
            delta_lower > delta_point || delta_point > delta_upper) {
            throw std::invalid_argument("invalid observation-confirmed SELL input");
        }

        const int raw_lower = previous + delta_lower - current;
        const int raw_point = previous + delta_point - current;
        const int raw_upper = previous + delta_upper - current;
        QuantityInterval total;
        total.consistent = raw_upper >= 0 && raw_lower <= requested;
        if (total.consistent) {
            total.lower = std::clamp(raw_lower, 0, requested);
            total.upper = std::clamp(raw_upper, 0, requested);
            if (total.lower > total.upper) total.consistent = false;
        }
        if (!total.consistent) {
            // Fail open on uncertainty but never pretend the request filled.
            total.lower = 0;
            total.upper = requested;
        }
        total.point = std::clamp(raw_point, total.lower, total.upper);
        result.total_fill[product] = total;

        QuantityInterval option;
        option.consistent = total.consistent;
        option.lower = std::clamp(total.lower - prior, 0, option_requested);
        option.point = std::clamp(total.point - prior, option.lower, option_requested);
        option.upper = std::clamp(total.upper - prior, option.point, option_requested);
        result.option_fill[product] = option;
    }
    return result;
}

ExecutionState settle_interval(
    const ExecutionState& state,
    const ExecutionProposal& proposal,
    QuantityInterval confirmed_fill
) {
    if (confirmed_fill.lower < 0 || confirmed_fill.lower > confirmed_fill.point ||
        confirmed_fill.point > confirmed_fill.upper ||
        confirmed_fill.upper > proposal.requested) {
        throw std::invalid_argument("confirmed fill interval outside proposed quantity");
    }
    if (proposal.reason == ExecutionReason::NoWindow ||
        proposal.reason == ExecutionReason::Completed) {
        if (confirmed_fill.upper != 0)
            throw std::invalid_argument("fill without a sale window");
        return state;
    }
    const auto [old_lower, old_upper] = outstanding_bounds(state);
    if (old_lower < 0 || state.outstanding_units < 0 || old_upper < old_lower)
        throw std::invalid_argument("invalid outstanding interval");
    ExecutionState next = state;
    next.outstanding_lower = std::max(0, old_lower - confirmed_fill.upper);
    next.outstanding_units = std::max(0, state.outstanding_units - confirmed_fill.point);
    next.outstanding_upper = std::max(0, old_upper - confirmed_fill.lower);
    next.outstanding_units = std::clamp(
        next.outstanding_units, next.outstanding_lower, next.outstanding_upper);
    next.remaining_windows = std::max(0, state.remaining_windows - 1);
    next.arrears = std::max(0, proposal.scheduled_due - confirmed_fill.point);
    next.paused = confirmed_fill.upper == 0 && next.outstanding_upper > 0;
    ++next.generation;
    return next;
}

ReplanResult ExecutionLedger::replan(
    ExecutionOptionKey option,
    int initial_units,
    int initial_windows,
    SwitchDirective directive
) {
    if (initial_units < 0 || initial_windows < 0)
        throw std::invalid_argument("negative execution promise");
    if (active_ && active_->option == option) {
        active_->state.paused = false;
        return {.retained = true};
    }
    ReplanResult result;
    if (active_) {
        if (directive == SwitchDirective::RequireSame)
            throw std::logic_error("option switch requires explicit pause or cancel");
        if (directive == SwitchDirective::PausePrevious) {
            active_->state.paused = true;
            paused_.push_back(*active_);
        } else {
            result.cancelled_units = cancel_active();
        }
        active_.reset();
    }
    auto saved = std::find_if(paused_.begin(), paused_.end(), [&](const Saved& value) {
        return value.option == option;
    });
    if (saved != paused_.end()) {
        active_ = *saved;
        active_->state.paused = false;
        paused_.erase(saved);
        result.resumed = true;
        return result;
    }
    ExecutionState initial;
    initial.outstanding_units = initial_units;
    initial.remaining_windows = initial_windows;
    initial.outstanding_lower = initial_units;
    initial.outstanding_upper = initial_units;
    active_ = Saved{option, initial};
    return result;
}

const ExecutionOptionKey& ExecutionLedger::active_option() const {
    if (!active_) throw std::logic_error("no active execution option");
    return active_->option;
}

const ExecutionState& ExecutionLedger::state() const {
    if (!active_) throw std::logic_error("no active execution state");
    return active_->state;
}

void ExecutionLedger::replace_state(ExecutionState state) {
    if (!active_) throw std::logic_error("no active execution state");
    const auto [lower, upper] = outstanding_bounds(state);
    if (state.outstanding_units < 0 || lower < 0 || upper < lower ||
        state.outstanding_units < lower || state.outstanding_units > upper)
        throw std::invalid_argument("invalid replacement execution state");
    state.outstanding_lower = lower;
    state.outstanding_upper = upper;
    active_->state = state;
}

void ExecutionLedger::pause_active() {
    if (!active_) return;
    active_->state.paused = true;
    paused_.push_back(*active_);
    active_.reset();
}

int ExecutionLedger::cancel_active() {
    if (!active_) return 0;
    const auto [lower, upper] = outstanding_bounds(active_->state);
    (void)lower;
    active_.reset();
    return upper;
}

ReplanResult ObservationConfirmedExecutionAdapter::replan(
    ExecutionOptionKey option,
    int initial_units,
    int initial_windows,
    SwitchDirective directive
) {
    if (pending_)
        throw std::logic_error("settle pending submission before replanning");
    return ledger_.replan(option, initial_units, initial_windows, directive);
}

void ObservationConfirmedExecutionAdapter::stage_submission(
    ExecutionProposal proposal,
    const market::Inventory& previous_shed,
    const CausalShedDelta& non_sell_delta,
    const market::Inventory& requested_total_sell,
    const market::Inventory& requested_option_sell,
    const market::Inventory& prior_sell
) {
    if (pending_) throw std::logic_error("a submission is already awaiting observation");
    if (!ledger_.has_active()) throw std::logic_error("cannot submit without an active option");
    const auto product = std::size_t(ledger_.active_option().product);
    if (product >= market::product_count)
        throw std::invalid_argument("active execution product outside market range");
    if (proposal.requested != requested_option_sell[product])
        throw std::invalid_argument("proposal and submitted option quantity differ");
    for (std::size_t other = 0; other < market::product_count; ++other) {
        if (other != product && requested_option_sell[other] != 0)
            throw std::invalid_argument("product-local option submitted on multiple products");
    }
    Pending pending;
    pending.option = ledger_.active_option();
    pending.proposal = proposal;
    pending.input.previous_shed = previous_shed;
    pending.input.non_sell_delta = non_sell_delta;
    pending.input.requested_total_sell = requested_total_sell;
    pending.input.requested_option_sell = requested_option_sell;
    pending.input.prior_sell = prior_sell;
    pending_ = std::move(pending);
}

std::optional<ObservationSettlementResult>
ObservationConfirmedExecutionAdapter::observe(const market::Inventory& current_shed) {
    if (!pending_) return std::nullopt;
    if (!ledger_.has_active() || !(ledger_.active_option() == pending_->option))
        throw std::logic_error("active option changed before observation settlement");
    pending_->input.current_shed = current_shed;
    auto fills = infer_sell_fills(pending_->input);
    const auto product = std::size_t(pending_->option.product);
    const auto option_fill = fills.option_fill[product];
    auto next = settle_interval(ledger_.state(), pending_->proposal, option_fill);
    ledger_.replace_state(next);
    ObservationSettlementResult result{std::move(fills), option_fill, next};
    pending_.reset();
    return result;
}

OrderOverlayResult overlay_sell_order(
    std::vector<MarketOrder>& orders,
    market::Product product,
    int requested,
    int own_shed,
    int protected_stock,
    int maximum_slots
) {
    if (requested < 0 || own_shed < 0 || protected_stock < 0 || maximum_slots < 0 ||
        int(orders.size()) > maximum_slots) {
        throw std::invalid_argument("invalid market overlay input");
    }
    OrderOverlayResult result;
    int last_matching = -1;
    for (std::size_t slot = 0; slot < orders.size(); ++slot) {
        const auto& order = orders[slot];
        if (order.quantity < 0) throw std::invalid_argument("negative existing market order");
        if (order.kind == MarketOrderKind::Sell && order.product == product) {
            result.prior_sell += order.quantity;
            last_matching = int(slot);
        }
    }
    const int sellable = std::max(0, own_shed - protected_stock);
    result.requested = std::min(requested, std::max(0, sellable - result.prior_sell));
    if (result.requested == 0) return result;
    if (last_matching >= 0) {
        orders[std::size_t(last_matching)].quantity += result.requested;
        result.slot = last_matching;
        result.inserted = true;
        return result;
    }
    if (int(orders.size()) >= maximum_slots) {
        result.requested = 0;
        result.slot_limited = true;
        return result;
    }
    orders.push_back({MarketOrderKind::Sell, product, result.requested});
    result.slot = int(orders.size()) - 1;
    result.inserted = true;
    return result;
}

int official_shed_occupancy(const std::array<int, 12>& shed) {
    int total = 0;
    for (const int quantity : shed) {
        if (quantity < 0) throw std::invalid_argument("negative shed quantity");
        total += quantity;
    }
    return total;
}

int official_shed_free_capacity(const std::array<int, 12>& shed, int capacity) {
    if (capacity < 0) throw std::invalid_argument("negative shed capacity");
    return std::max(0, capacity - official_shed_occupancy(shed));
}

ExecutionProposal propose(
    const ExecutionState& state,
    const ObservableExecutionInput& input
) {
    if (state.outstanding_units < 0 || state.remaining_windows < 0 || state.arrears < 0 ||
        input.own_stock < 0 || input.protected_stock < 0 || input.market_inventory < 0 ||
        input.reservation_price < 0 || input.marginal_impact_limit < 0 ||
        input.cash_shortfall < 0 || input.assigned_overflow_units < 0) {
        throw std::invalid_argument("negative adaptive execution input");
    }

    ExecutionProposal result;
    result.quote = market::price(input.product, input.market_inventory);
    result.market_inventory_if_filled = input.market_inventory;
    if (!input.sale_window) return result;

    const int sellable = std::max(0, input.own_stock - input.protected_stock);
    if (state.outstanding_units == 0 && input.cash_shortfall == 0 &&
        input.assigned_overflow_units == 0) {
        result.reason = ExecutionReason::Completed;
        return result;
    }

    result.scheduled_due = state.remaining_windows > 0
        ? ceil_div(state.outstanding_units, state.remaining_windows)
        : state.outstanding_units;
    const int cash_units = units_for_cash(
        input.product, input.market_inventory, sellable, input.cash_shortfall
    );
    result.cash_feasible = cash_units <= sellable;
    result.storage_feasible = input.assigned_overflow_units <= sellable;
    result.mandatory_units = std::min(
        sellable, std::max(input.assigned_overflow_units, std::min(cash_units, sellable))
    );

    if (input.terminal_clear_window) {
        result.requested = sellable;
        result.market_inventory_if_filled = inventory_after_sale(
            input.product, input.market_inventory, result.requested
        );
        result.reason = ExecutionReason::LastWindow;
        return result;
    }

    int discretionary = 0;
    int simulated_inventory = input.market_inventory;
    const int initial_quote = result.quote;
    while (discretionary < std::min(sellable, result.scheduled_due)) {
        const int marginal_quote = market::price(input.product, simulated_inventory);
        if (marginal_quote < input.reservation_price ||
            initial_quote - marginal_quote > input.marginal_impact_limit) {
            break;
        }
        ++discretionary;
        if (marginal_quote > 1) ++simulated_inventory;
    }
    result.requested = std::max(result.mandatory_units, discretionary);
    result.market_inventory_if_filled = inventory_after_sale(
        input.product, input.market_inventory, result.requested
    );
    if (input.cash_shortfall > 0) result.reason = ExecutionReason::CashEmergency;
    else if (input.assigned_overflow_units > 0) result.reason = ExecutionReason::StorageEmergency;
    else if (result.requested == 0) result.reason = ExecutionReason::FeedbackPause;
    else result.reason = ExecutionReason::ScheduledDrip;
    return result;
}

ExecutionState settle(
    const ExecutionState& state,
    const ExecutionProposal& proposal,
    int confirmed_fill
) {
    if (confirmed_fill < 0 || confirmed_fill > proposal.requested) {
        throw std::invalid_argument("confirmed fill outside proposed quantity");
    }
    return settle_interval(
        state, proposal,
        QuantityInterval{confirmed_fill, confirmed_fill, confirmed_fill, true});
}

TerminalDecision choose_terminal_timing(
    market::Product product,
    int quantity,
    int current_market_inventory,
    const std::vector<TerminalScenario>& scenarios,
    double cvar_fraction
) {
    if (quantity < 0 || current_market_inventory < 0 || scenarios.empty() ||
        !(cvar_fraction > 0.0 && cvar_fraction <= 1.0)) {
        throw std::invalid_argument("invalid terminal timing input");
    }
    double weight_sum = 0.0;
    for (const auto& scenario : scenarios) {
        if (scenario.next_market_inventory < 0 || scenario.rival_sale_before_next < 0 ||
            !std::isfinite(scenario.weight) || scenario.weight <= 0.0) {
            throw std::invalid_argument("invalid causal terminal scenario");
        }
        weight_sum += scenario.weight;
    }
    if (std::abs(weight_sum - 1.0) > 1e-9) {
        throw std::invalid_argument("terminal scenario weights must sum to one");
    }

    const auto now_value = market::sell_revenue(product, current_market_inventory, quantity);
    std::vector<std::pair<std::int64_t, double>> now_outcomes;
    std::vector<std::pair<std::int64_t, double>> next_outcomes;
    now_outcomes.reserve(scenarios.size());
    next_outcomes.reserve(scenarios.size());
    for (const auto& scenario : scenarios) {
        now_outcomes.emplace_back(now_value, scenario.weight);
        next_outcomes.emplace_back(
            market::sell_revenue(
                product,
                scenario.next_market_inventory + scenario.rival_sale_before_next,
                quantity
            ),
            scenario.weight
        );
    }
    TerminalDecision result;
    result.now = aggregate(std::move(now_outcomes), cvar_fraction);
    result.next = aggregate(std::move(next_outcomes), cvar_fraction);
    // Exact ties wait. Selling early must have a positive robust reason.
    if (score_better(result.now, result.next)) result.timing = TerminalTiming::SellNow;
    return result;
}

}  // namespace g001::general_econ
