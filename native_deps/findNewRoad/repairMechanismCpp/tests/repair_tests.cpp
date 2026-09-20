#include "repair.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

using namespace g001::repair;

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

TimedAction action(fastkag::Op op, int x, int y, bool fill = false,
                   int earliest = 0, int latest = 100, int value = 0) {
    TimedAction result;
    result.action.op = op;
    result.required_position = {static_cast<std::int16_t>(x),
                                static_cast<std::int16_t>(y)};
    result.expected_fill = fill;
    result.earliest_step = earliest;
    result.latest_step = latest;
    result.economic_value = value;
    return result;
}

std::vector<TimedAction> weed_fixture() {
    return {
        action(fastkag::Op::PLANT, 0, 0, true, 0, 2, 9),
        action(fastkag::Op::EAST, 0, 0),
        action(fastkag::Op::WATER, 1, 0, true, 2, 4, 7),
        action(fastkag::Op::WEST, 1, 0),
        action(fastkag::Op::HARVEST, 0, 0, true, 4, 6, 12),
        action(fastkag::Op::PASS, 0, 0),
        action(fastkag::Op::EAST, 0, 0),
        action(fastkag::Op::FEED, 1, 0, true, 7, 7, 10),
        action(fastkag::Op::WEST, 1, 0),
        action(fastkag::Op::EAST, 0, 0),  // legacy fixed drop corrupts route
        action(fastkag::Op::CARE, 1, 0, true, 10, 10, 6),
        action(fastkag::Op::PASS, 1, 0),
    };
}

void test_alignment_search() {
    const auto fixture = weed_fixture();
    const auto legacy = legacy_fixed_window(fixture, {0, 0});
    const auto repaired = search_minimum_loss_realign(fixture, {0, 0}, 9);
    require(legacy.metrics.skipped_source_index == 9,
            "legacy eight-step replay discards source index nine");
    require(legacy.metrics.movement_edits == 1,
            "legacy can discard a route movement");
    require(legacy.metrics.critical_fill_failures >= 1,
            "route corruption makes downstream work fail");
    require(repaired.metrics.skipped_source_index == 5,
            "search consumes the zero-value PASS slack");
    require(repaired.metrics.movement_edits == 0,
            "search never edits movement order");
    require(repaired.metrics.critical_fill_failures == 0,
            "PASS catch-up retains fixture production fills");
    require(repaired.metrics.objective < legacy.metrics.objective,
            "searched realignment strictly improves the explicit objective");
}

void test_alignment_can_wait_beyond_legacy_window() {
    std::vector<TimedAction> fixture;
    for (int step = 0; step < 15; ++step) {
        fixture.push_back(action(step == 0 ? fastkag::Op::PLANT : fastkag::Op::WATER,
                                 0, 0, true, 0, 30, step == 0 ? 9 : 7));
    }
    fixture.push_back(action(fastkag::Op::PASS, 0, 0));
    fixture.push_back(action(fastkag::Op::EAST, 0, 0));
    const auto result = search_minimum_loss_realign(fixture, {0, 0}, 16);
    require(result.metrics.skipped_source_index == 15,
            "search may wait beyond the legacy nine-source window for real slack");
    require(result.metrics.critical_fill_failures == 0 && result.metrics.movement_edits == 0,
            "longer buffering retains both production actions and movement subsequence");
}

void test_animal_retry_causal_fill() {
    AnimalRetryController controller;
    AnimalObservation first;
    first.step = 10; first.money = 100; first.empty_structures[1] = 1;
    first.next_pickup_step[1] = 12;
    first.next_place_step[1] = 13;
    first.pickup_place_feasible[1] = true;
    controller.observe(first);
    controller.record_attempt(Animal::Cow, 1, 13);

    AnimalObservation failed = first;
    failed.step = 11; failed.money = 500;
    controller.observe(failed);
    const auto retry = controller.decide(failed);
    require(retry.has_value() && retry->animal == Animal::Cow && retry->quantity == 1,
            "failed observable cow purchase is retried when feasible");
    controller.record_attempt(retry->animal, retry->quantity, retry->deadline);

    AnimalObservation filled = failed;
    filled.step = 12; filled.own_total[1] = 1;
    controller.observe(filled);
    require(!controller.decide(filled).has_value(),
            "confirmed retry fill cannot duplicate the animal");
    const auto& state = controller.snapshot(Animal::Cow);
    require(state.inferred_failures == 1 && state.inferred_fills == 1 &&
            state.outstanding == 0 && state.attempts == 2,
            "causal fill/failure accounting is exact in the fixture");
}

void test_animal_retry_guards() {
    AnimalRetryController controller;
    AnimalObservation before;
    before.step = 1; before.money = 100; before.empty_structures[2] = 1;
    before.next_pickup_step[2] = 2;
    before.next_place_step[2] = 2;
    before.pickup_place_feasible[2] = true;
    controller.observe(before);
    controller.record_attempt(Animal::Sheep, 1, 2);
    AnimalObservation after = before;
    after.step = 2; after.money = 1000; after.shed_used = 100;
    controller.observe(after);
    require(!controller.decide(after).has_value(),
            "full shed blocks a retry without consuming the intent");
    after.step = 5; after.shed_used = 0;
    controller.observe(after);
    require(controller.snapshot(Animal::Sheep).retired,
            "intent retires after the downstream placement deadline");

    require(suppress_empty_stall_work(fastkag::Op::FEED,
                                      fastkag::TileKind::PASTURE, false),
            "empty pasture feed is recognized as futile");
    require(!suppress_empty_stall_work(fastkag::Op::EAST,
                                       fastkag::TileKind::PASTURE, false),
            "movement is never suppressed");
    require(!suppress_empty_stall_work(fastkag::Op::CARE,
                                       fastkag::TileKind::PASTURE, true),
            "occupied structure work is preserved");
}

void test_partial_fill_drop_then_retry() {
    AnimalRetryController controller;
    AnimalObservation before;
    before.step = 30; before.money = 2000; before.shed_used = 99;
    before.empty_structures[1] = 3; before.next_pickup_step[1] = 33;
    before.next_place_step[1] = 34; before.pickup_place_feasible[1] = true;
    controller.observe(before); controller.record_attempt(Animal::Cow, 3, 34);

    AnimalObservation partial = before;
    partial.step = 31; partial.own_total[1] = 1; partial.shed_used = 100;
    controller.observe(partial);
    require(controller.snapshot(Animal::Cow).inferred_fills == 1 &&
            controller.snapshot(Animal::Cow).outstanding == 2 &&
            controller.snapshot(Animal::Cow).capacity_limited_failures == 2,
            "partial animal fill is separated from the capacity-limited remainder");
    require(!controller.decide(partial).has_value(),
            "full shed defers rather than retires the outstanding quantity");

    AnimalObservation after_drop = partial;
    after_drop.step = 32; after_drop.shed_used = 98;
    controller.observe(after_drop);
    const auto retry = controller.decide(after_drop);
    require(retry && retry->quantity == 2 && retry->deadline == 33,
            "a causal DROP capacity release admits the partial remainder before PICKUP");
}

void test_cash_recovers_before_pickup() {
    AnimalRetryController controller;
    AnimalObservation before;
    before.step = 40; before.money = 100; before.empty_structures[2] = 1;
    before.next_pickup_step[2] = 43; before.next_place_step[2] = 44;
    before.pickup_place_feasible[2] = true;
    controller.observe(before); controller.record_attempt(Animal::Sheep, 1, 44);
    AnimalObservation poor = before; poor.step = 41; poor.money = 300;
    controller.observe(poor);
    require(controller.snapshot(Animal::Sheep).cash_limited_failures == 1,
            "cash shortfall is classified from pre-attempt public money");
    require(!controller.decide(poor).has_value(), "retry waits while protected cash is insufficient");
    AnimalObservation recovered = poor; recovered.step = 42; recovered.money = 600;
    controller.observe(recovered);
    const auto retry = controller.decide(recovered);
    require(retry && retry->quantity == 1 && retry->deadline == 43,
            "retry fires after cash recovery but before the fixed route pickup");
}

void test_same_step_pickup_requires_realign() {
    AnimalRetryController controller;
    AnimalObservation before;
    before.step = 60; before.money = 100; before.empty_structures[0] = 1;
    before.next_pickup_step[0] = 61; before.next_place_step[0] = 64;
    before.pickup_place_feasible[0] = true;
    controller.observe(before); controller.record_attempt(Animal::Goose, 1, 64);
    AnimalObservation detected = before;
    detected.step = 61; detected.money = 500;
    detected.allow_same_step_pickup_realign[0] = false;
    controller.observe(detected);
    require(!controller.decide(detected).has_value(),
            "market retry after a same-turn PICKUP cannot feed that already-run PICKUP");
    detected.allow_same_step_pickup_realign[0] = true;
    const auto retry = controller.decide(detected);
    require(retry && retry->requires_pickup_realign && retry->deadline == 61,
            "same-step retry is admitted only with an explicit delayed PICKUP transaction");

    std::vector<TimedAction> pickup_window{
        action(fastkag::Op::PICKUP, 0, 0, true, 0, 2, 4),
        action(fastkag::Op::EAST, 0, 0),
        action(fastkag::Op::PASS, 1, 0),
        action(fastkag::Op::WEST, 1, 0),
        action(fastkag::Op::PLACE, 0, 0, true, 3, 5, 9),
    };
    const auto alignment = search_minimum_loss_realign(pickup_window, {0, 0}, 4, 1);
    require(alignment.metrics.movement_edits == 0 &&
            alignment.metrics.critical_fill_failures == 0 &&
            alignment.metrics.skipped_source_index == 2,
            "delayed PICKUP catches up at PASS and keeps the PLACE sequence viable");
}

void test_stationary_work_recovery() {
    std::vector<TimedAction> future{
        action(fastkag::Op::EAST, 0, 0),
        action(fastkag::Op::WATER, 0, 0, true, 50, 55, 7),
        action(fastkag::Op::HARVEST, 0, 0, true, 50, 55, 12),
    };
    const auto replacement = search_stationary_work_replacement({0, 0}, 50, future, 2);
    require(replacement.recovered && replacement.source_index == 1 &&
            replacement.replacement.op == fastkag::Op::WATER,
            "empty-stall work can pull forward valid same-position crop work");
    const auto unresolved = search_stationary_work_replacement({1, 0}, 50, future, 2);
    require(!unresolved.recovered && unresolved.reason == "no_same_position_productive_work",
            "unrecoverable empty stall is explicitly reported, not called repaired");
}

}  // namespace

int main() {
    test_alignment_search();
    test_alignment_can_wait_beyond_legacy_window();
    test_animal_retry_causal_fill();
    test_animal_retry_guards();
    test_partial_fill_drop_then_retry();
    test_cash_recovers_before_pickup();
    test_same_step_pickup_requires_realign();
    test_stationary_work_recovery();
    std::cout << "repair mechanism tests passed\n";
}
