#include "full_market_takeover.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>

namespace g001::general_econ {
namespace {

bool is_buy_operation(MarketOperation operation) {
    return operation == MarketOperation::Hire ||
        operation == MarketOperation::BuyLand ||
        operation == MarketOperation::BuySeed ||
        operation == MarketOperation::BuyProduct ||
        operation == MarketOperation::BuyAnimal;
}

void validate(const MarketInstruction& order) {
    if (order.quantity < 0 || order.deadline < 0)
        throw std::invalid_argument("invalid market instruction");
}

}  // namespace

bool should_enter_takeover(bool experiment_enabled) {
    return experiment_enabled;
}

ObligationCompilation compile_production_obligations(
    const OwnProductionResources& resources,
    const std::vector<PlannedUnitAction>& own_nonmarket_plan
) {
    if (resources.current_actor_count < 1)
        throw std::invalid_argument("production plan must include a farmer");
    std::array<int, 5> seeds{};
    std::array<int, 12> items{};
    for (int item = 0; item < 5; ++item) {
        if (resources.seeds[item] < 0)
            throw std::invalid_argument("negative owned seed quantity");
        seeds[item] = resources.seeds[item];
    }
    for (int item = 0; item < 12; ++item) {
        if (resources.owned_items[item] < 0)
            throw std::invalid_argument("negative owned item quantity");
        items[item] = resources.owned_items[item];
    }
    auto plan = own_nonmarket_plan;
    std::stable_sort(plan.begin(), plan.end(), [](const auto& left, const auto& right) {
        return left.step < right.step;
    });
    struct Need { MarketOperation operation; int item; int quantity; int deadline; };
    std::vector<Need> needs;
    needs.reserve(plan.size());
    auto add_need = [&](MarketOperation operation, int item, int deadline) {
        auto found = std::find_if(needs.begin(), needs.end(), [&](const Need& need) {
            return need.operation == operation && need.item == item;
        });
        if (found == needs.end()) needs.push_back({operation, item, 1, deadline});
        else {
            ++found->quantity;
            found->deadline = std::min(found->deadline, deadline);
        }
    };
    int maximum_actor = resources.current_actor_count - 1;
    int maximum_quadrant = 0;
    int first_actor_deadline = 719;
    int first_land_deadline = 719;
    for (const auto& action : plan) {
        if (action.step < 0 || action.actor < 0 || action.quadrant < 0 ||
            action.quadrant > 3)
            throw std::invalid_argument("invalid non-market production plan action");
        if (action.actor >= resources.current_actor_count)
            first_actor_deadline = std::min(first_actor_deadline, action.step);
        maximum_actor = std::max(maximum_actor, action.actor);
        if ((resources.unlocked_quadrants_mask & (1U << action.quadrant)) == 0)
            first_land_deadline = std::min(first_land_deadline, action.step);
        maximum_quadrant = std::max(maximum_quadrant, action.quadrant);
        if (action.operation == PlannedUnitOperation::Plant) {
            if (action.item < 0 || action.item >= 5)
                throw std::invalid_argument("invalid crop in PLANT obligation");
            if (seeds[action.item] > 0) --seeds[action.item];
            else add_need(MarketOperation::BuySeed, action.item, action.step);
        } else if (action.operation == PlannedUnitOperation::Feed) {
            if (items[0] > 0) --items[0];
            else add_need(MarketOperation::BuyProduct, 0, action.step);
        } else if (action.operation == PlannedUnitOperation::Fertilize) {
            if (items[8] > 0) --items[8];
            else add_need(MarketOperation::BuyProduct, 8, action.step);
        } else if (action.operation == PlannedUnitOperation::PlaceAnimal) {
            if (action.item < 9 || action.item >= 12)
                throw std::invalid_argument("invalid animal in PLACE obligation");
            if (items[action.item] > 0) --items[action.item];
            else add_need(MarketOperation::BuyAnimal, action.item, action.step);
        }
    }

    ObligationCompilation result;
    result.orders.reserve(needs.size() + 2);
    result.required_hires = std::max(0, maximum_actor + 1 - resources.current_actor_count);
    const int unlocked_extras = std::max(
        0, int(std::popcount(static_cast<unsigned>(resources.unlocked_quadrants_mask))) - 1);
    result.required_land_purchases = std::max(0, maximum_quadrant - unlocked_extras);
    if (result.required_hires > 0)
        result.orders.push_back({MarketOperation::Hire, -1, result.required_hires,
                                 OrderOrigin::CriticalObligation,
                                 first_actor_deadline, 100});
    if (result.required_land_purchases > 0)
        result.orders.push_back({MarketOperation::BuyLand, -1,
                                 result.required_land_purchases,
                                 OrderOrigin::CriticalObligation,
                                 first_land_deadline, 90});
    for (const auto& need : needs)
        result.orders.push_back({need.operation, need.item, need.quantity,
                                 OrderOrigin::CriticalObligation,
                                 need.deadline, 80});
    std::stable_sort(result.orders.begin(), result.orders.end(), [](const auto& left, const auto& right) {
        if (left.deadline != right.deadline) return left.deadline < right.deadline;
        return left.priority > right.priority;
    });
    return result;
}

TakeoverResult compose_takeover_market(const TakeoverInput& input) {
    if (input.maximum_slots < 0)
        throw std::invalid_argument("negative maximum market slots");
    for (const auto& order : input.legacy_market) validate(order);
    if (!input.enabled) {
        TakeoverResult passthrough;
        passthrough.orders = input.legacy_market;
        return passthrough;
    }

    TakeoverResult result;
    std::vector<MarketInstruction> critical = input.critical_obligations;
    for (const auto& legacy : input.legacy_market) {
        if (legacy.operation != MarketOperation::Pass) ++result.suppressed_legacy_orders;
        if (legacy.operation == MarketOperation::Sell) {
            ++result.suppressed_legacy_sells;
        } else if (is_buy_operation(legacy.operation)) ++result.suppressed_legacy_buys;
    }
    std::stable_sort(critical.begin(), critical.end(), [](const auto& left, const auto& right) {
        if (left.deadline != right.deadline) return left.deadline < right.deadline;
        return left.priority > right.priority;
    });

    auto append = [&](const std::vector<MarketInstruction>& source,
                      OrderOrigin origin,
                      bool critical_category) {
        for (auto order : source) {
            validate(order);
            if (order.quantity == 0 || order.operation == MarketOperation::Pass) continue;
            if (int(result.orders.size()) >= input.maximum_slots) {
                if (critical_category) ++result.unscheduled_critical_obligations;
                else ++result.dropped_lower_priority_orders;
                continue;
            }
            order.origin = origin;
            result.orders.push_back(order);
        }
    };
    append(critical, OrderOrigin::CriticalObligation, true);
    append(input.liquidity_orders, OrderOrigin::Liquidity, true);
    append(input.capacity_orders, OrderOrigin::Capacity, true);
    append(input.continuation_orders, OrderOrigin::Continuation, false);
    append(input.new_plan_orders, OrderOrigin::NewPlan, false);

    result.legacy_origin_leaks = int(std::count_if(
        result.orders.begin(), result.orders.end(), [](const auto& order) {
            return order.origin == OrderOrigin::LegacyMacro ||
                order.origin == OrderOrigin::LegacySpecial;
        }));
    if (result.legacy_origin_leaks != 0)
        throw std::logic_error("legacy market action leaked through full takeover");
    return result;
}

bool arm_uses_special(EconomicArm arm) {
    return arm == EconomicArm::SpecialOnly || arm == EconomicArm::SpecialPlusGeneral;
}

bool arm_uses_general(EconomicArm arm) {
    return arm == EconomicArm::GeneralOnly || arm == EconomicArm::SpecialPlusGeneral;
}

}  // namespace g001::general_econ
