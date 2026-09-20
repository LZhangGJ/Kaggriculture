#include "weed_absorption.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string_view>

using namespace g001::weed_absorption;

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

fastkag::PlayerAction unit(fastkag::Action action) {
    fastkag::PlayerAction result; result.units.push_back(action); return result;
}

fastkag::Simulator weed_state() {
    fastkag::Simulator sim({}, 77331);
    for (int step = 0; step < 24; ++step) {
        std::array<fastkag::PlayerAction, 2> actions;
        actions[0] = unit({}); actions[1] = unit({});
        if (step == 0)
            actions[0].market.push_back(
                {fastkag::Op::BUY_SEED, fastkag::Item::CARROT, 2});
        if (step == 1)
            actions[0].units[0] = {fastkag::Op::PLANT, fastkag::Item::CARROT, 1};
        sim.step(actions);
    }
    const auto& farm = sim.farms()[0];
    const int index = farm.farmer.y * sim.config().board_size + farm.farmer.x;
    require(farm.tiles[static_cast<std::size_t>(index)].kind == fastkag::TileKind::WEED,
            "fixture must reach an observed weed at the current actor position");
    return sim;
}

SearchRequest request_fixture(CandidatePolicy policy) {
    static fastkag::Simulator sim = weed_state();
    static std::vector<fastkag::PlayerAction> tape(80, unit({}));
    tape[24] = unit({fastkag::Op::PLANT, fastkag::Item::CARROT, 1});
    tape[25] = unit({fastkag::Op::WATER});
    tape[26] = unit({fastkag::Op::EAST});
    tape[27] = unit({fastkag::Op::PASS});
    tape[28] = unit({fastkag::Op::WEST});
    SearchRequest request;
    request.state = &sim; request.player = 0; request.actor = 0;
    request.absolute_start_step = 24; request.own_tape = &tape;
    request.options = {24, 48, policy};
    request.fixed_own_scenario.assign(tape.begin() + 24, tape.begin() + 72);
    request.opponent_scenario.assign(48, unit({}));
    return request;
}

void test_pass_absorption_preserves_chain() {
    const auto request = request_fixture(CandidatePolicy::AnyNonMovement);
    const auto result = select_absorption(request);
    require(result.feasible, "exact replay must find a movement-safe absorption");
    require(result.selected.skipped_operation == fastkag::Op::PASS,
            "a true PASS must dominate dropping PLANT or WATER in the fixture");
    require(result.selected.movement_source_edits == 0 &&
            result.selected.movement_coordinate_violations == 0,
            "MOVE source order and source coordinates are hard invariants");
}

void test_policy_layers() {
    auto request = request_fixture(CandidatePolicy::SameDayPassOnly);
    auto candidates = enumerate_candidates(request);
    require(!candidates.empty() && candidates.front() == 3 &&
            std::all_of(candidates.begin(), candidates.end(), [](int source) {
                return source < 24;
            }),
            "same-day PASS tier excludes productive work and cross-day slack");

    request.options.candidate_policy = CandidatePolicy::CrossDayPassOnly;
    candidates = enumerate_candidates(request);
    require(candidates.size() > 1 && candidates.front() == 3 &&
            std::find(candidates.begin(), candidates.end(), 24) != candidates.end(),
            "cross-day PASS tier keeps same-day slack and can scan farther");
}

void test_dependency_and_inventory_metrics() {
    const auto request = request_fixture(CandidatePolicy::AnyNonMovement);
    const auto drop_plant = replay_candidate(request, 0);
    const auto drop_water = replay_candidate(request, 1);
    const auto drop_pass = replay_candidate(request, 3);
    require(drop_plant.dependencies.productive_failures >= drop_pass.dependencies.productive_failures,
            "dependency replay exposes downstream failures after dropping PLANT");
    require(drop_pass.economic_equity > 0 && drop_water.economic_equity > 0,
            "branch score includes cash, inventory, seed, and standing assets");
    require(drop_pass.objective < drop_plant.objective,
            "long-horizon cost prefers the intact production chain");
}

void test_day_end_unit_market_deposit_order() {
    fastkag::Simulator sim({}, 99117);
    for (int step = 0; step < 24; ++step) {
        std::array<fastkag::PlayerAction, 2> actions;
        actions[0] = unit({}); actions[1] = unit({});
        if (step == 0)
            actions[0].market.push_back(
                {fastkag::Op::BUY_PRODUCT, fastkag::Item::WHEAT, 1});
        if (step == 23) {
            actions[0].units[0] = {fastkag::Op::PICKUP, fastkag::Item::WHEAT, 1};
            actions[0].market.push_back(
                {fastkag::Op::SELL, fastkag::Item::WHEAT, 1});
        }
        sim.step(actions);
        if (step == 23)
            require(sim.last_market_fills()[0][0] == 0,
                    "unit PICKUP precedes market, so same-step shed SELL cannot fill");
    }
    require(sim.privates()[0].shed[0] == 1 &&
            sim.privates()[0].inventories.size() == 1 &&
            sim.privates()[0].inventories[0][0] == 0,
            "end-of-day runs after market and deposits carried inventory into shed");
    require(sim.farms()[0].hands.empty() && sim.farms()[0].hires_today == 0,
            "end-of-day dismisses hands and resets the daily hire boundary");
}

}  // namespace

int main() {
    test_pass_absorption_preserves_chain();
    test_policy_layers();
    test_dependency_and_inventory_metrics();
    test_day_end_unit_market_deposit_order();
    std::cout << "weed absorption tests passed\n";
}
