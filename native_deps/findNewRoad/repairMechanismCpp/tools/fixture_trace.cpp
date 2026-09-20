#include "repair.hpp"

#include <iostream>
#include <string_view>
#include <vector>

using namespace g001::repair;

namespace {

const char* op_name(fastkag::Op op) {
    switch (op) {
        case fastkag::Op::PASS: return "PASS";
        case fastkag::Op::NORTH: return "NORTH";
        case fastkag::Op::SOUTH: return "SOUTH";
        case fastkag::Op::EAST: return "EAST";
        case fastkag::Op::WEST: return "WEST";
        case fastkag::Op::PLANT: return "PLANT";
        case fastkag::Op::WATER: return "WATER";
        case fastkag::Op::HARVEST: return "HARVEST";
        case fastkag::Op::FEED: return "FEED";
        case fastkag::Op::CARE: return "CARE";
        case fastkag::Op::DIG: return "DIG";
        default: return "OTHER";
    }
}

TimedAction make(fastkag::Op op, int x, int y, bool fill = false,
                 int earliest = 0, int latest = 100, int value = 0) {
    TimedAction result;
    result.action.op = op; result.required_position = {
        static_cast<std::int16_t>(x), static_cast<std::int16_t>(y)};
    result.expected_fill = fill; result.earliest_step = earliest;
    result.latest_step = latest; result.economic_value = value;
    return result;
}

void emit_summary(std::string_view name, const AlignmentPlan& plan) {
    const auto& value = plan.metrics;
    std::cout << "{\"kind\":\"alignment_summary\",\"name\":\"" << name
              << "\",\"skipped_source\":" << value.skipped_source_index
              << ",\"movement_edits\":" << value.movement_edits
              << ",\"critical_fill_failures\":" << value.critical_fill_failures
              << ",\"economic_value_lost\":" << value.economic_value_lost
              << ",\"path_l1_area\":" << value.path_l1_area
              << ",\"delayed_action_steps\":" << value.delayed_action_steps
              << ",\"objective\":" << value.objective << "}\n";
}

void emit_steps(std::string_view name, const AlignmentPlan& plan,
                const std::vector<TimedAction>& source) {
    for (std::size_t step = 0; step < plan.source_index_by_actual_step.size(); ++step) {
        const int index = plan.source_index_by_actual_step[step];
        const char* operation = index == -1 ? "DIG" :
            index >= 0 && index < int(source.size()) ? op_name(source[index].action.op) : "PASS";
        std::cout << "{\"kind\":\"alignment_step\",\"name\":\"" << name
                  << "\",\"actual_step\":" << step
                  << ",\"source_step\":" << index
                  << ",\"operation\":\"" << operation << "\"}\n";
    }
}

}  // namespace

int main() {
    const std::vector<TimedAction> source{
        make(fastkag::Op::PLANT, 0, 0, true, 0, 2, 9),
        make(fastkag::Op::EAST, 0, 0),
        make(fastkag::Op::WATER, 1, 0, true, 2, 4, 7),
        make(fastkag::Op::WEST, 1, 0),
        make(fastkag::Op::HARVEST, 0, 0, true, 4, 6, 12),
        make(fastkag::Op::PASS, 0, 0),
        make(fastkag::Op::EAST, 0, 0),
        make(fastkag::Op::FEED, 1, 0, true, 7, 7, 10),
        make(fastkag::Op::WEST, 1, 0),
        make(fastkag::Op::EAST, 0, 0),
        make(fastkag::Op::CARE, 1, 0, true, 10, 10, 6),
        make(fastkag::Op::PASS, 1, 0),
    };
    const auto legacy = legacy_fixed_window(source, {0, 0});
    const auto searched = search_minimum_loss_realign(source, {0, 0}, 9);
    emit_summary("legacy_fixed8", legacy);
    emit_steps("legacy_fixed8", legacy, source);
    emit_summary("minimum_loss", searched);
    emit_steps("minimum_loss", searched, source);

    std::vector<TimedAction> long_buffer;
    for (int step = 0; step < 15; ++step) {
        long_buffer.push_back(make(step == 0 ? fastkag::Op::PLANT : fastkag::Op::WATER,
                                   0, 0, true, 0, 30, step == 0 ? 9 : 7));
    }
    long_buffer.push_back(make(fastkag::Op::PASS, 0, 0));
    long_buffer.push_back(make(fastkag::Op::EAST, 0, 0));
    emit_summary("minimum_loss_slack_after_15",
                 search_minimum_loss_realign(long_buffer, {0, 0}, 16));

    AnimalRetryController controller;
    AnimalObservation initial;
    initial.step = 20; initial.money = 100; initial.empty_structures[1] = 1;
    initial.next_pickup_step[1] = 22;
    initial.next_place_step[1] = 23;
    initial.pickup_place_feasible[1] = true;
    controller.observe(initial); controller.record_attempt(Animal::Cow, 1, 23);
    AnimalObservation failed = initial; failed.step = 21; failed.money = 500;
    controller.observe(failed); const auto retry = controller.decide(failed);
    std::cout << "{\"kind\":\"animal_retry\",\"step\":21,\"failure_inferred\":"
              << controller.snapshot(Animal::Cow).inferred_failures
              << ",\"retry_quantity\":" << (retry ? retry->quantity : 0)
              << ",\"reason\":\"" << (retry ? retry->reason : "none") << "\"}\n";

    AnimalRetryController partial_controller;
    AnimalObservation nearly_full;
    nearly_full.step = 30; nearly_full.money = 2000; nearly_full.shed_used = 99;
    nearly_full.empty_structures[1] = 3; nearly_full.next_pickup_step[1] = 33;
    nearly_full.next_place_step[1] = 34; nearly_full.pickup_place_feasible[1] = true;
    partial_controller.observe(nearly_full);
    partial_controller.record_attempt(Animal::Cow, 3, 34);
    auto partial = nearly_full; partial.step = 31; partial.own_total[1] = 1;
    partial.shed_used = 100; partial_controller.observe(partial);
    auto after_drop = partial; after_drop.step = 32; after_drop.shed_used = 98;
    partial_controller.observe(after_drop); const auto partial_retry = partial_controller.decide(after_drop);
    std::cout << "{\"kind\":\"animal_partial_drop_retry\",\"filled\":"
              << partial_controller.snapshot(Animal::Cow).inferred_fills
              << ",\"capacity_failures\":"
              << partial_controller.snapshot(Animal::Cow).capacity_limited_failures
              << ",\"retry_after_drop\":" << (partial_retry ? partial_retry->quantity : 0)
              << "}\n";

    AnimalRetryController cash_controller;
    AnimalObservation cash_before;
    cash_before.step = 40; cash_before.money = 100;
    cash_before.empty_structures[2] = 1; cash_before.next_pickup_step[2] = 43;
    cash_before.next_place_step[2] = 44; cash_before.pickup_place_feasible[2] = true;
    cash_controller.observe(cash_before);
    cash_controller.record_attempt(Animal::Sheep, 1, 44);
    auto cash_short = cash_before; cash_short.step = 41; cash_short.money = 300;
    cash_controller.observe(cash_short);
    const auto blocked_cash_retry = cash_controller.decide(cash_short);
    auto cash_recovered = cash_short; cash_recovered.step = 42; cash_recovered.money = 600;
    cash_controller.observe(cash_recovered);
    const auto cash_retry = cash_controller.decide(cash_recovered);
    std::cout << "{\"kind\":\"animal_cash_recovery_retry\",\"cash_failures\":"
              << cash_controller.snapshot(Animal::Sheep).cash_limited_failures
              << ",\"retry_while_short\":"
              << (blocked_cash_retry ? blocked_cash_retry->quantity : 0)
              << ",\"retry_after_cash\":" << (cash_retry ? cash_retry->quantity : 0)
              << ",\"deadline\":" << (cash_retry ? cash_retry->deadline : -1) << "}\n";

    AnimalRetryController success_controller;
    AnimalObservation success_before;
    success_before.step = 50; success_before.money = 1000;
    success_before.empty_structures[0] = 1; success_before.next_pickup_step[0] = 52;
    success_before.next_place_step[0] = 53; success_before.pickup_place_feasible[0] = true;
    success_controller.observe(success_before);
    success_controller.record_attempt(Animal::Goose, 1, 53);
    auto success_after = success_before; success_after.step = 51;
    success_after.money = 700; success_after.own_total[0] = 1;
    success_controller.observe(success_after);
    std::cout << "{\"kind\":\"animal_success_before_pickup_place\",\"filled\":"
              << success_controller.snapshot(Animal::Goose).inferred_fills
              << ",\"duplicate_retry\":"
              << (success_controller.decide(success_after).has_value() ? "true" : "false")
              << ",\"pickup_step\":52,\"place_step\":53}\n";

    AnimalRetryController same_step_controller;
    AnimalObservation same_step_before;
    same_step_before.step = 60; same_step_before.money = 100;
    same_step_before.empty_structures[0] = 1;
    same_step_before.next_pickup_step[0] = 61;
    same_step_before.next_place_step[0] = 64;
    same_step_before.pickup_place_feasible[0] = true;
    same_step_controller.observe(same_step_before);
    same_step_controller.record_attempt(Animal::Goose, 1, 64);
    auto same_step = same_step_before; same_step.step = 61; same_step.money = 500;
    same_step_controller.observe(same_step);
    const auto unsafe_same_step = same_step_controller.decide(same_step);
    same_step.allow_same_step_pickup_realign[0] = true;
    const auto safe_same_step = same_step_controller.decide(same_step);
    std::vector<TimedAction> pickup_window{
        make(fastkag::Op::PICKUP, 0, 0, true, 0, 2, 4),
        make(fastkag::Op::EAST, 0, 0),
        make(fastkag::Op::PASS, 1, 0),
        make(fastkag::Op::WEST, 1, 0),
        make(fastkag::Op::PLACE, 0, 0, true, 3, 5, 9),
    };
    const auto pickup_realign = search_minimum_loss_realign(pickup_window, {0, 0}, 4, 1);
    std::cout << "{\"kind\":\"animal_same_step_pickup\",\"retry_without_realign\":"
              << (unsafe_same_step ? unsafe_same_step->quantity : 0)
              << ",\"retry_with_realign\":" << (safe_same_step ? safe_same_step->quantity : 0)
              << ",\"requires_realign\":"
              << (safe_same_step && safe_same_step->requires_pickup_realign ? "true" : "false")
              << ",\"skipped_source\":" << pickup_realign.metrics.skipped_source_index
              << ",\"movement_edits\":" << pickup_realign.metrics.movement_edits
              << ",\"critical_fill_failures\":"
              << pickup_realign.metrics.critical_fill_failures << "}\n";

    AnimalRetryController late_controller;
    AnimalObservation late_before;
    late_before.step = 70; late_before.money = 100;
    late_before.empty_structures[1] = 1; late_before.next_pickup_step[1] = 71;
    late_before.next_place_step[1] = 71; late_before.pickup_place_feasible[1] = true;
    late_controller.observe(late_before);
    late_controller.record_attempt(Animal::Cow, 1, 71);
    auto late_failed = late_before; late_failed.step = 71;
    late_failed.money = 1000; late_failed.shed_used = 100;
    late_controller.observe(late_failed);
    auto after_deadline = late_failed; after_deadline.step = 73; after_deadline.shed_used = 0;
    late_controller.observe(after_deadline);
    std::cout << "{\"kind\":\"animal_too_late_to_place\",\"retired\":"
              << (late_controller.snapshot(Animal::Cow).retired ? "true" : "false")
              << ",\"reason\":\"" << late_controller.snapshot(Animal::Cow).retire_reason
              << "\",\"retry_quantity\":"
              << (late_controller.decide(after_deadline).has_value() ? 1 : 0) << "}\n";

    std::vector<TimedAction> future{
        make(fastkag::Op::EAST, 0, 0),
        make(fastkag::Op::WATER, 0, 0, true, 50, 55, 7),
    };
    const auto work = search_stationary_work_replacement({0, 0}, 50, future, 1);
    std::cout << "{\"kind\":\"empty_stall_recovery\",\"recovered\":"
              << (work.recovered ? "true" : "false")
              << ",\"source_index\":" << work.source_index
              << ",\"reason\":\"" << work.reason << "\"}\n";
    const auto no_work = search_stationary_work_replacement({1, 0}, 50, future, 1);
    std::cout << "{\"kind\":\"empty_stall_unresolved\",\"recovered\":"
              << (no_work.recovered ? "true" : "false")
              << ",\"source_index\":" << no_work.source_index
              << ",\"reason\":\"" << no_work.reason << "\"}\n";
}
