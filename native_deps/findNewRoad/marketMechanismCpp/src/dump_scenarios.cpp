#include "dump_scenarios.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <string>

namespace g001::dump {
namespace {

using g001::market::Product;
constexpr int floor_price = 1;

bool can_buy(Product product) {
    return product == Product::Wheat || product == Product::Fertilizer;
}

bool empty(const Inventory& inventory) {
    return std::all_of(inventory.begin(), inventory.end(), [](int value) { return value == 0; });
}

void validate_inventory(const Inventory& inventory, const char* name) {
    if (std::any_of(inventory.begin(), inventory.end(), [](int value) { return value < 0; }))
        throw std::invalid_argument(std::string(name) + " must be nonnegative");
}

TransitionEvidence infer_transition(
    const PublicMarketFrame& previous,
    const PublicMarketFrame& current,
    const Inventory& belief_upper
) {
    TransitionEvidence result; result.previous_step = previous.step; result.current_step = current.step;
    for (std::size_t p = 0; p < g001::market::product_count; ++p) {
        auto& evidence = result.product[p];
        const int delta = current.market_inventory[p] - previous.market_inventory[p];
        const int public_player_flow = delta + current.known_town_drain[p];
        // Town demand is applied after the two players' market queues.  Add it
        // back to recover the inventory at the end of market execution; the
        // public post-town quote can otherwise hide that the market touched
        // the $1 floor during this transition.
        const int inventory_before_town = current.market_inventory[p] +
            current.known_town_drain[p];
        const bool started_at_floor = previous.market_price[p] <= floor_price;
        const bool ended_market_at_floor = g001::market::price(
            static_cast<Product>(p), inventory_before_town) <= floor_price;
        evidence.price_floor = started_at_floor || ended_market_at_floor;
        evidence.price_change = current.market_price[p] - previous.market_price[p];
        evidence.price_inconsistent = current.market_price[p] !=
            g001::market::price(static_cast<Product>(p), current.market_inventory[p]);
        evidence.buy_ambiguity = can_buy(static_cast<Product>(p));

        // Filled own sales and their *visible market contribution* differ at
        // $1.  If the transition starts at the floor, every own filled unit is
        // invisible because inventory never increases and the quote remains
        // $1.  If it reaches the floor during the market queue, an unknown
        // prefix is visible.  Away from the floor, shed-derived fills are the
        // exact contribution interval.
        int own_visible_lower = current.own_sell_filled_lower[p];
        int own_visible_point = current.own_sell_filled_point[p];
        int own_visible_upper = current.own_sell_filled_upper[p];
        if (started_at_floor) {
            own_visible_lower = own_visible_point = own_visible_upper = 0;
        } else if (ended_market_at_floor) {
            own_visible_lower = 0;
            own_visible_point = current.own_sell_filled_point[p] / 2;
        }
        evidence.fill_ambiguity = own_visible_lower != own_visible_upper;
        const int net_lower = public_player_flow - own_visible_upper;
        const int net_upper = public_player_flow - own_visible_lower;
        evidence.opponent_net_flow_point = public_player_flow - own_visible_point;
        evidence.sale_lower = std::max(0, net_lower);
        evidence.sale_point = std::max(0, evidence.opponent_net_flow_point);
        evidence.sale_upper = std::max(evidence.sale_lower, std::max(0, net_upper));
        if (evidence.sale_lower > belief_upper[p]) evidence.inconsistent = true;

        if (evidence.buy_ambiguity || evidence.price_floor) {
            // Gross sales can be hidden by a simultaneous legal WHEAT/FERTILIZER
            // buy, or by the official rule that a $1 SELL never enters public
            // market inventory. Conservation still caps them by the causal
            // opponent-stock upper bound.
            evidence.sale_upper = std::max(evidence.sale_upper, belief_upper[p]);
        } else if (net_upper < 0) {
            evidence.inconsistent = true;
        }
        evidence.sale_upper = std::min(evidence.sale_upper, belief_upper[p]);
        evidence.sale_lower = std::min(evidence.sale_lower, evidence.sale_upper);
        evidence.sale_point = std::clamp(evidence.sale_point,
                                         evidence.sale_lower, evidence.sale_upper);

        // Market flow alone cannot see $1 sales, but it is not the only public
        // signal.  The upstream belief filter can decompose public money
        // changes (after visible HIRE/LAND and bounded private purchases and
        // non-floor receipts) into a much tighter hidden-clearance interval.
        // Fuse that interval here when the caller has proved it valid.
        if (current.opponent_clearance_valid[p]) {
            const int external_lower = current.opponent_clearance_lower[p];
            const int external_point = current.opponent_clearance_point[p];
            const int external_upper = current.opponent_clearance_upper[p];
            const int intersection_lower = std::max(evidence.sale_lower, external_lower);
            const int intersection_upper = std::min(evidence.sale_upper, external_upper);
            if (intersection_lower <= intersection_upper) {
                evidence.sale_lower = intersection_lower;
                evidence.sale_upper = intersection_upper;
                evidence.sale_point = std::clamp(
                    external_point, evidence.sale_lower, evidence.sale_upper);
            } else {
                // Preserve soundness if two independently constructed public
                // intervals disagree; surface the inconsistency and keep their
                // hull rather than silently trusting either point estimate.
                evidence.inconsistent = true;
                evidence.sale_lower = std::min(evidence.sale_lower, external_lower);
                evidence.sale_upper = std::max(evidence.sale_upper, external_upper);
                evidence.sale_point = std::clamp(
                    external_point, evidence.sale_lower, evidence.sale_upper);
            }
        }
    }
    return result;
}

std::uint32_t scenario_id(BeliefBand band, DumpTiming timing) {
    return (static_cast<std::uint32_t>(band) + 1U) * 16U +
        static_cast<std::uint32_t>(timing);
}

}  // namespace

DerivedOwnSellFill derive_own_sell_fill_interval(const OwnShedTransition& transition) {
    validate_inventory(transition.previous_shed, "previous own shed");
    validate_inventory(transition.current_shed, "current own shed");
    validate_inventory(transition.sell_requested, "own SELL requested");
    const auto& in = transition.known_nonmarket_inflow;
    const auto& out = transition.known_nonmarket_outflow;
    validate_inventory(in.lower, "nonmarket inflow lower");
    validate_inventory(in.point, "nonmarket inflow point");
    validate_inventory(in.upper, "nonmarket inflow upper");
    validate_inventory(out.lower, "nonmarket outflow lower");
    validate_inventory(out.point, "nonmarket outflow point");
    validate_inventory(out.upper, "nonmarket outflow upper");
    DerivedOwnSellFill result;
    for (std::size_t p = 0; p < g001::market::product_count; ++p) {
        if (in.lower[p] > in.point[p] || in.point[p] > in.upper[p] ||
            out.lower[p] > out.point[p] || out.point[p] > out.upper[p])
            throw std::invalid_argument("nonmarket shed flow interval is invalid");
        const int raw_lower = transition.previous_shed[p] + in.lower[p] - out.upper[p] -
            transition.current_shed[p];
        const int raw_point = transition.previous_shed[p] + in.point[p] - out.point[p] -
            transition.current_shed[p];
        const int raw_upper = transition.previous_shed[p] + in.upper[p] - out.lower[p] -
            transition.current_shed[p];
        const int capacity = std::min(transition.sell_requested[p],
            transition.previous_shed[p] + in.upper[p]);
        result.lower[p] = std::clamp(raw_lower, 0, capacity);
        result.upper[p] = std::clamp(raw_upper, result.lower[p], capacity);
        result.point[p] = std::clamp(raw_point, result.lower[p], result.upper[p]);
        result.clamped[p] = raw_lower < 0 || raw_upper > capacity || raw_point < result.lower[p] ||
            raw_point > result.upper[p];
    }
    return result;
}

Result generate(const Input& input) {
    validate_inventory(input.belief_lower, "belief lower");
    validate_inventory(input.belief_point, "belief point");
    validate_inventory(input.belief_upper, "belief upper");
    for (std::size_t p = 0; p < g001::market::product_count; ++p) {
        if (input.belief_lower[p] > input.belief_point[p] ||
            input.belief_point[p] > input.belief_upper[p])
            throw std::invalid_argument("belief must satisfy lower <= point <= upper");
    }
    if (input.steps_remaining < 0 || input.horizon_steps <= 0 ||
        input.horizon_steps > 72)
        throw std::invalid_argument("invalid remaining/planning horizon");
    for (std::size_t i = 0; i < input.public_history.size(); ++i) {
        const auto& frame = input.public_history[i];
        validate_inventory(frame.market_inventory, "market inventory");
        validate_inventory(frame.market_price, "market price");
        validate_inventory(frame.own_sell_requested, "own requested sells");
        validate_inventory(frame.own_sell_filled_lower, "own filled sell lower");
        validate_inventory(frame.own_sell_filled_point, "own filled sell point");
        validate_inventory(frame.own_sell_filled_upper, "own filled sell upper");
        validate_inventory(frame.known_town_drain, "town drain");
        validate_inventory(frame.opponent_clearance_lower, "opponent clearance lower");
        validate_inventory(frame.opponent_clearance_point, "opponent clearance point");
        validate_inventory(frame.opponent_clearance_upper, "opponent clearance upper");
        for (std::size_t p = 0; p < g001::market::product_count; ++p) {
            if (frame.own_sell_filled_lower[p] > frame.own_sell_filled_point[p] ||
                frame.own_sell_filled_point[p] > frame.own_sell_filled_upper[p] ||
                frame.own_sell_filled_upper[p] > frame.own_sell_requested[p])
                throw std::invalid_argument("own filled SELL interval is invalid");
            if (frame.opponent_clearance_valid[p] &&
                (frame.opponent_clearance_lower[p] > frame.opponent_clearance_point[p] ||
                 frame.opponent_clearance_point[p] > frame.opponent_clearance_upper[p]))
                throw std::invalid_argument("opponent clearance interval is invalid");
        }
        if (i > 0 && (frame.step <= input.public_history[i - 1].step ||
                      frame.day < input.public_history[i - 1].day))
            throw std::invalid_argument("public history must be causal and ordered");
    }
    if (!input.public_history.empty()) {
        const auto& last = input.public_history.back();
        if (last.step != input.current_step || last.day != input.current_day)
            throw std::invalid_argument("current step/day must match final public frame");
    }
    Result result;
    result.flow_history.reserve(input.public_history.size() > 0 ? input.public_history.size() - 1 : 0);
    for (std::size_t i = 1; i < input.public_history.size(); ++i)
        result.flow_history.push_back(infer_transition(
            input.public_history[i - 1], input.public_history[i], input.belief_upper));

    int pulse_count = 0;
    int recent_sale = 0;
    for (const auto& transition : result.flow_history) {
        int quantity = 0;
        for (const auto& product : transition.product) quantity += product.sale_point;
        pulse_count += quantity > 0;
        recent_sale = quantity;
    }
    const int transition_count = static_cast<int>(result.flow_history.size());
    const int point_stock = std::accumulate(
        input.belief_point.begin(), input.belief_point.end(), 0);
    result.intent.recent_sale_point = recent_sale;
    result.intent.pulse_frequency = static_cast<double>(pulse_count + 1) /
        static_cast<double>(transition_count + 2); // finite-sample smoothing
    result.intent.recent_quantity_ratio = static_cast<double>(recent_sale) /
        static_cast<double>(recent_sale + point_stock + 1);
    result.intent.terminal_urgency = 24.0 / static_cast<double>(input.steps_remaining + 24);

    std::vector<int> windows;
    windows.reserve(input.remaining_sale_windows.size());
    for (const int step : input.remaining_sale_windows) {
        if (step < input.current_step || step > input.current_step + input.steps_remaining)
            throw std::invalid_argument("sale window lies outside causal remaining horizon");
        if (windows.empty() || windows.back() != step) windows.push_back(step);
        else throw std::invalid_argument("remaining sale windows must be strictly increasing");
    }
    if (!std::is_sorted(windows.begin(), windows.end()))
        throw std::invalid_argument("remaining sale windows must be strictly increasing");

    result.intent.stock_pressure = static_cast<double>(point_stock) /
        static_cast<double>(point_stock + std::max<std::size_t>(1, windows.size()));
    const double no_recent_dump = 1.0 - result.intent.pulse_frequency;
    result.intent.dump_propensity = 1.0 - no_recent_dump *
        (1.0 - result.intent.terminal_urgency * result.intent.stock_pressure);

    if (empty(input.belief_upper) || windows.empty()) {
        WeightedScenario no_dump; no_dump.scenario.belief_stock = input.belief_point;
        no_dump.belief_band = BeliefBand::Point; no_dump.timing = DumpTiming::None;
        no_dump.id = scenario_id(no_dump.belief_band, no_dump.timing); no_dump.weight = 1.0;
        no_dump.scenario.weight = no_dump.weight;
        result.scenarios.push_back(std::move(no_dump));
        return result;
    }

    struct BeliefChoice { BeliefBand band; const Inventory* inventory; };
    const std::array<BeliefChoice, 3> all_beliefs{{
        {BeliefBand::Lower, &input.belief_lower},
        {BeliefBand::Point, &input.belief_point},
        {BeliefBand::Upper, &input.belief_upper},
    }};
    std::vector<BeliefChoice> beliefs;
    for (const auto& choice : all_beliefs) {
        if (empty(*choice.inventory)) continue;
        const bool duplicate = std::any_of(beliefs.begin(), beliefs.end(), [&](const auto& prior) {
            return *prior.inventory == *choice.inventory;
        });
        if (!duplicate) beliefs.push_back(choice);
    }
    struct TimingChoice { DumpTiming timing; int step; bool distributed{}; };
    const std::array<TimingChoice, 3> all_timings{{
        {DumpTiming::Now, windows.front()},
        {DumpTiming::Next, windows.size() > 1 ? windows[1] : windows.front()},
        {DumpTiming::Late, windows.back()},
    }};
    std::vector<TimingChoice> timings;
    for (const auto& choice : all_timings) {
        const bool duplicate = std::any_of(timings.begin(), timings.end(), [&](const auto& prior) {
            return prior.step == choice.step;
        });
        if (!duplicate) timings.push_back(choice);
    }
    std::vector<int> horizon_windows;
    const int horizon_end = input.current_step + input.horizon_steps - 1;
    for (const int step : windows)
        if (step <= horizon_end) horizon_windows.push_back(step);
    if (horizon_windows.empty()) horizon_windows.push_back(windows.front());
    if (horizon_windows.size() > 1)
        timings.push_back({DumpTiming::Distributed, horizon_windows.front(), true});

    const double now_score = (1.0 + result.intent.recent_quantity_ratio) *
        (1.0 + result.intent.pulse_frequency);
    const double next_score = 1.0 + result.intent.pulse_frequency;
    const double late_score = 1.0 + (1.0 - result.intent.pulse_frequency) *
        (1.0 - result.intent.terminal_urgency);
    auto timing_score = [&](DumpTiming timing) {
        if (timing == DumpTiming::Now) return now_score;
        if (timing == DumpTiming::Next) return next_score;
        if (timing == DumpTiming::Distributed)
            return 1.0 + 0.5 * (1.0 - result.intent.pulse_frequency);
        return late_score;
    };
    double timing_total = 0;
    for (const auto& timing : timings) timing_total += timing_score(timing.timing);
    for (const auto& belief : beliefs) for (const auto& timing : timings) {
        WeightedScenario weighted; weighted.scenario.belief_stock = *belief.inventory;
        weighted.belief_band = belief.band; weighted.timing = timing.timing;
        weighted.id = scenario_id(belief.band, timing.timing);
        for (std::size_t p = 0; p < g001::market::product_count; ++p) {
            const int quantity = (*belief.inventory)[p];
            if (quantity <= 0) continue;
            if (!timing.distributed) {
                weighted.scenario.dumps.push_back(
                    {timing.step, static_cast<int>(p), quantity});
                continue;
            }
            const int windows_count = static_cast<int>(horizon_windows.size());
            const int base = quantity / windows_count;
            const int remainder = quantity % windows_count;
            for (int index = 0; index < windows_count; ++index) {
                const int due = base + static_cast<int>(index < remainder);
                if (due > 0) weighted.scenario.dumps.push_back(
                    {horizon_windows[static_cast<std::size_t>(index)],
                     static_cast<int>(p), due});
            }
        }
        weighted.weight = result.intent.dump_propensity /
            static_cast<double>(beliefs.size()) * timing_score(timing.timing) / timing_total;
        weighted.scenario.weight = weighted.weight;
        result.scenarios.push_back(std::move(weighted));
    }
    WeightedScenario no_dump; no_dump.scenario.belief_stock = input.belief_point;
    no_dump.belief_band = BeliefBand::Point; no_dump.timing = DumpTiming::None;
    no_dump.id = scenario_id(no_dump.belief_band, no_dump.timing);
    no_dump.weight = 1.0 - result.intent.dump_propensity;
    no_dump.scenario.weight = no_dump.weight;
    result.scenarios.push_back(std::move(no_dump));
    if (result.scenarios.size() > max_scenario_count)
        throw std::logic_error("dump scenario cardinality exceeds ABI bound");
    return result;
}

}  // namespace g001::dump
