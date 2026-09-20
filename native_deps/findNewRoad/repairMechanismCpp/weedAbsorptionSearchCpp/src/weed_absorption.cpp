#include "weed_absorption.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace g001::weed_absorption {
namespace {

constexpr std::array<int, 5> seed_cost{10, 20, 50, 100, 80};
constexpr std::array<int, 5> crop_first_day{2, 2, 8, 10, 10};
constexpr std::array<int, 3> animal_cost{300, 400, 500};
constexpr std::array<int, 3> animal_product{5, 6, 7};

int quantity(const fastkag::Action& action) { return std::max(0, action.quantity); }

int shed_total(const fastkag::PrivateState& state) {
    return std::accumulate(state.shed.begin(), state.shed.end(), 0);
}

fastkag::Position actor_position(const fastkag::Farm& farm, int actor) {
    if (actor == 0) return farm.farmer;
    if (actor > 0 && actor <= static_cast<int>(farm.hands.size()))
        return farm.hands[static_cast<std::size_t>(actor - 1)];
    return {-1, -1};
}

fastkag::Position moved(fastkag::Position position, fastkag::Op operation) {
    if (operation == fastkag::Op::NORTH) --position.y;
    else if (operation == fastkag::Op::SOUTH) ++position.y;
    else if (operation == fastkag::Op::EAST) ++position.x;
    else if (operation == fastkag::Op::WEST) --position.x;
    return position;
}

fastkag::Action tape_unit(const std::vector<fastkag::PlayerAction>& tape,
                          int step, int actor) {
    if (step < 0 || step >= static_cast<int>(tape.size()) || actor < 0 ||
        actor >= static_cast<int>(tape[static_cast<std::size_t>(step)].units.size()))
        return {};
    return tape[static_cast<std::size_t>(step)].units[static_cast<std::size_t>(actor)];
}

void ensure_inventory(fastkag::PrivateState& state, int actor) {
    while (actor >= static_cast<int>(state.inventories.size())) {
        state.inventories.push_back({});
        state.inventory_order.push_back({});
    }
}

void inventory_add(fastkag::PrivateState& state, int actor, int item, int amount) {
    if (amount <= 0) return;
    ensure_inventory(state, actor);
    auto& inventory = state.inventories[static_cast<std::size_t>(actor)];
    if (inventory[item] == 0)
        state.inventory_order[static_cast<std::size_t>(actor)].push_back(
            static_cast<std::int8_t>(item));
    inventory[item] += amount;
}

bool inventory_take(fastkag::PrivateState& state, int actor, int item, int amount) {
    ensure_inventory(state, actor);
    auto& inventory = state.inventories[static_cast<std::size_t>(actor)];
    if (amount <= 0 || item < 0 || item >= fastkag::N_ITEMS || inventory[item] < amount)
        return false;
    inventory[item] -= amount;
    if (inventory[item] == 0) {
        auto& order = state.inventory_order[static_cast<std::size_t>(actor)];
        std::erase(order, static_cast<std::int8_t>(item));
    }
    return true;
}

bool shed_adjacent(const fastkag::Config& config, fastkag::Position position) {
    const int half = config.board_size / 2;
    return (position.x == half - 1 || position.x == half) &&
           (position.y == half - 1 || position.y == half);
}

// Local unit-phase projection used only to label exact sequential legality.
// The authoritative branch state is still advanced by fastkag::Simulator.
bool apply_projected(fastkag::Farm& farm, fastkag::PrivateState& private_state,
                     const fastkag::Config& config, int actor,
                     const fastkag::Action& action, int day) {
    if (actor < 0 || actor > static_cast<int>(farm.hands.size())) return false;
    ensure_inventory(private_state, actor);
    auto position = actor_position(farm, actor);
    auto set_position = [&](fastkag::Position value) {
        if (actor == 0) farm.farmer = value;
        else farm.hands[static_cast<std::size_t>(actor - 1)] = value;
    };
    if (is_movement(action.op)) {
        const auto target = moved(position, action.op);
        if (target.x < 0 || target.y < 0 || target.x >= config.board_size ||
            target.y >= config.board_size) return false;
        set_position(target); return true;
    }
    if (action.op == fastkag::Op::PASS) return true;
    const int tile_index = position.y * config.board_size + position.x;
    if (tile_index < 0 || tile_index >= static_cast<int>(farm.tiles.size())) return false;
    auto& tile = farm.tiles[static_cast<std::size_t>(tile_index)];
    auto& inventory = private_state.inventories[static_cast<std::size_t>(actor)];
    const int item = static_cast<int>(action.item), amount = quantity(action);
    if (action.op == fastkag::Op::DROP) {
        if (!shed_adjacent(config, position)) return false;
        const bool any = std::accumulate(inventory.begin(), inventory.end(), 0) > 0;
        for (const int ordered_item : private_state.inventory_order[static_cast<std::size_t>(actor)]) {
            const int room = std::max(0, config.shed_capacity - shed_total(private_state));
            const int take = std::min(inventory[ordered_item], room);
            private_state.shed[ordered_item] += take;
            inventory[ordered_item] = 0;
        }
        private_state.inventory_order[static_cast<std::size_t>(actor)].clear();
        return any;
    }
    if (action.op == fastkag::Op::PICKUP) {
        if (!shed_adjacent(config, position) || item < 0 || item >= fastkag::N_ITEMS ||
            amount <= 0 || private_state.shed[item] <= 0) return false;
        const int take = std::min(amount, private_state.shed[item]);
        private_state.shed[item] -= take; inventory_add(private_state, actor, item, take);
        return take > 0;
    }
    if (action.op == fastkag::Op::PLACE) {
        if (item >= 9 && item < 12) {
            const auto required = item == 9 ? fastkag::TileKind::COOP
                                            : fastkag::TileKind::PASTURE;
            if (tile.kind != required || tile.animal != fastkag::Item::NONE ||
                !inventory_take(private_state, actor, item, 1)) return false;
            tile = {}; tile.kind = fastkag::TileKind::ANIMAL;
            tile.animal = static_cast<fastkag::Item>(item); tile.placed_day = day;
            return true;
        }
        if (!shed_adjacent(config, position) || item < 0 || item >= fastkag::N_ITEMS ||
            amount <= 0 || inventory[item] <= 0 || shed_total(private_state) >= config.shed_capacity)
            return false;
        const int take = std::min({amount, inventory[item],
            std::max(0, config.shed_capacity - shed_total(private_state))});
        inventory_take(private_state, actor, item, take); private_state.shed[item] += take;
        return take > 0;
    }
    if (tile.kind == fastkag::TileKind::LOCKED) return false;
    if (action.op == fastkag::Op::PLANT) {
        if (item < 0 || item >= fastkag::N_CROPS ||
            tile.kind != fastkag::TileKind::EMPTY || private_state.seeds[item] <= 0) return false;
        --private_state.seeds[item]; tile = {}; tile.kind = fastkag::TileKind::PLANT;
        tile.crop = static_cast<fastkag::Item>(item); tile.planted_day = day;
        tile.consecutive_unwatered = 1; tile.yield_units = item == 2 || item == 3 ? 0 : 1;
        return true;
    }
    if (action.op == fastkag::Op::WATER) {
        if (tile.kind != fastkag::TileKind::PLANT || tile.watered_today) return false;
        tile.watered_today = true; return true;
    }
    if (action.op == fastkag::Op::HARVEST) {
        if (tile.yield_units <= 0) return false;
        if (tile.kind == fastkag::TileKind::PLANT) {
            const int crop = static_cast<int>(tile.crop);
            if (crop < 0 || crop >= fastkag::N_CROPS ||
                day - tile.planted_day < crop_first_day[crop]) return false;
            inventory_add(private_state, actor, crop, tile.yield_units);
            tile.yield_units = 0;
            if (crop != 2 && crop != 3) tile = {};
            return true;
        }
        if (tile.kind == fastkag::TileKind::ANIMAL) {
            const int animal = static_cast<int>(tile.animal) - 9;
            if (animal < 0 || animal >= 3) return false;
            inventory_add(private_state, actor, animal_product[animal], tile.yield_units);
            tile.yield_units = 0; return true;
        }
        return false;
    }
    if (action.op == fastkag::Op::FERTILIZE) {
        if (tile.kind != fastkag::TileKind::PLANT ||
            !inventory_take(private_state, actor, 8, 1)) return false;
        tile.fertilized_until_day = std::max<int>(tile.fertilized_until_day, day + 2);
        return true;
    }
    if (action.op == fastkag::Op::DIG) {
        if (tile.kind == fastkag::TileKind::EMPTY || tile.kind == fastkag::TileKind::ANIMAL)
            return false;
        tile = {}; return true;
    }
    if (action.op == fastkag::Op::BUILD_COOP ||
        action.op == fastkag::Op::BUILD_PASTURE) {
        if (tile.kind != fastkag::TileKind::EMPTY) return false;
        tile.kind = action.op == fastkag::Op::BUILD_COOP
            ? fastkag::TileKind::COOP : fastkag::TileKind::PASTURE;
        return true;
    }
    if (action.op == fastkag::Op::FEED) {
        if (tile.kind != fastkag::TileKind::ANIMAL || tile.fed_today ||
            !inventory_take(private_state, actor, 0, 1)) return false;
        tile.fed_today = true; return true;
    }
    if (action.op == fastkag::Op::COLLECT_FERTILIZER) {
        if (tile.kind != fastkag::TileKind::ANIMAL || !tile.fertilizer_available) return false;
        tile.fertilizer_available = false; inventory_add(private_state, actor, 8, 1);
        return true;
    }
    if (action.op == fastkag::Op::CARE) {
        if (tile.kind != fastkag::TileKind::ANIMAL || tile.cared_today) return false;
        tile.cared_today = true; return true;
    }
    return false;
}

int operation_penalty(fastkag::Op operation) {
    switch (operation) {
        case fastkag::Op::HARVEST: return 300;
        case fastkag::Op::PLANT: return 240;
        case fastkag::Op::PLACE: return 220;
        case fastkag::Op::WATER: return 160;
        case fastkag::Op::FEED: return 150;
        case fastkag::Op::CARE: return 120;
        case fastkag::Op::FERTILIZE: return 100;
        case fastkag::Op::PICKUP:
        case fastkag::Op::DROP: return 80;
        case fastkag::Op::BUILD_COOP:
        case fastkag::Op::BUILD_PASTURE: return 200;
        default: return 20;
    }
}

void finalize_assets(const fastkag::Simulator& sim, int player, BranchMetrics& metrics) {
    const auto& private_state = sim.privates()[player];
    const auto& farm = sim.farms()[player];
    metrics.cash = farm.money;
    std::array<int, fastkag::N_ITEMS> held = private_state.shed;
    for (const auto& inventory : private_state.inventories)
        for (int item = 0; item < fastkag::N_ITEMS; ++item) held[item] += inventory[item];
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item)
        metrics.inventory_value += held[item] * sim.market().prices[item];
    for (int animal = 0; animal < 3; ++animal)
        metrics.inventory_value += held[animal + 9] * animal_cost[animal];
    for (int crop = 0; crop < fastkag::N_CROPS; ++crop)
        metrics.seed_value += private_state.seeds[crop] * seed_cost[crop];
    for (const auto& tile : farm.tiles) {
        if (tile.kind == fastkag::TileKind::PLANT) {
            const int crop = static_cast<int>(tile.crop);
            if (crop >= 0 && crop < fastkag::N_CROPS)
                metrics.standing_asset_value += seed_cost[crop] +
                    tile.yield_units * sim.market().prices[crop];
        } else if (tile.kind == fastkag::TileKind::ANIMAL) {
            const int animal = static_cast<int>(tile.animal) - 9;
            if (animal >= 0 && animal < 3)
                metrics.standing_asset_value += animal_cost[animal] +
                    tile.yield_units * sim.market().prices[animal_product[animal]];
        } else if (tile.kind == fastkag::TileKind::COOP ||
                   tile.kind == fastkag::TileKind::PASTURE) {
            metrics.standing_asset_value += 100;
        }
    }
    metrics.economic_equity = metrics.cash + metrics.inventory_value + metrics.seed_value +
                              metrics.standing_asset_value;
    double failure_cost = metrics.dependencies.market_unfilled_units * 40.0 +
                          metrics.dependencies.end_of_day_overflow * 200.0;
    for (int operation = 0; operation < 24; ++operation)
        failure_cost += metrics.dependencies.failed_by_op[operation] *
                        operation_penalty(static_cast<fastkag::Op>(operation));
    metrics.objective = -metrics.economic_equity + failure_cost +
        metrics.movement_source_edits * 1e12 + metrics.movement_coordinate_violations * 1e10;
}

}  // namespace

bool is_movement(fastkag::Op operation) {
    return operation == fastkag::Op::NORTH || operation == fastkag::Op::SOUTH ||
           operation == fastkag::Op::EAST || operation == fastkag::Op::WEST;
}

bool is_productive(fastkag::Op operation) {
    return operation != fastkag::Op::PASS && !is_movement(operation);
}

bool projected_unit_action_succeeds(const fastkag::Simulator& state, int player,
                                    const fastkag::PlayerAction& action, int actor) {
    if (player < 0 || player > 1 || actor < 0 ||
        actor >= static_cast<int>(action.units.size())) return false;
    auto farm = state.farms()[player];
    auto private_state = state.privates()[player];
    std::array<int, fastkag::N_CROPS> plant_demand{};
    for (const auto& unit_action : action.units) {
        const int item = static_cast<int>(unit_action.item);
        if (unit_action.op == fastkag::Op::PLANT && item >= 0 && item < fastkag::N_CROPS)
            ++plant_demand[item];
    }
    bool target_succeeded = false;
    for (int index = 0; index <= actor; ++index) {
        auto current = action.units[static_cast<std::size_t>(index)];
        const int item = static_cast<int>(current.item);
        if (current.op == fastkag::Op::PLANT && item >= 0 && item < fastkag::N_CROPS &&
            plant_demand[item] > private_state.seeds[item]) current.op = fastkag::Op::PASS;
        const bool succeeded = apply_projected(farm, private_state, state.config(), index,
                                               current, state.day());
        if (index == actor) target_succeeded = succeeded && current.op != fastkag::Op::PASS;
    }
    return target_succeeded;
}

std::vector<int> enumerate_candidates(const SearchRequest& request) {
    if (!request.state || !request.own_tape) return {};
    const int start = request.absolute_start_step;
    const int maximum = std::min({request.options.maximum_absorbed_source,
        request.options.replay_horizon - 2,
        static_cast<int>(request.own_tape->size()) - start - 1});
    std::vector<int> result;
    for (int source = 0; source <= maximum; ++source) {
        const auto operation = tape_unit(*request.own_tape, start + source,
                                         request.actor).op;
        if (is_movement(operation)) continue;
        if (request.options.candidate_policy != CandidatePolicy::AnyNonMovement &&
            operation != fastkag::Op::PASS) continue;
        if (request.options.candidate_policy == CandidatePolicy::SameDayPassOnly &&
            (start + source) / request.state->config().turns_per_day !=
                start / request.state->config().turns_per_day) continue;
        result.push_back(source);
    }
    return result;
}

BranchMetrics replay_candidate(const SearchRequest& request, int skipped_source) {
    if (!request.state || !request.own_tape)
        throw std::invalid_argument("weed absorption replay requires state and own tape");
    BranchMetrics metrics;
    metrics.skipped_source = skipped_source;
    metrics.skipped_operation = tape_unit(*request.own_tape,
        request.absolute_start_step + skipped_source, request.actor).op;
    if (is_movement(metrics.skipped_operation)) metrics.movement_source_edits = 1;

    fastkag::Simulator branch = *request.state;
    std::vector<fastkag::Position> expected_before(
        static_cast<std::size_t>(request.options.replay_horizon), {-1, -1});
    // Reference source coordinates must include end-of-day farmer reset,
    // hand dismissal/re-hire, cash-dependent HIRE fills, and older independent
    // transactions. A movement-only prefix is incorrect across day boundaries.
    auto reference = *request.state;
    for (int offset = 0; offset < request.options.replay_horizon && !reference.done(); ++offset) {
        expected_before[static_cast<std::size_t>(offset)] =
            actor_position(reference.farms()[request.player], request.actor);
        const int absolute = request.absolute_start_step + offset;
        fastkag::PlayerAction own;
        if (offset < static_cast<int>(request.fixed_own_scenario.size()))
            own = request.fixed_own_scenario[static_cast<std::size_t>(offset)];
        else if (absolute < static_cast<int>(request.own_tape->size()))
            own = (*request.own_tape)[static_cast<std::size_t>(absolute)];
        if (request.actor >= static_cast<int>(own.units.size()))
            own.units.resize(static_cast<std::size_t>(request.actor + 1));
        own.units[static_cast<std::size_t>(request.actor)] = tape_unit(
            *request.own_tape, absolute, request.actor);
        fastkag::PlayerAction opponent;
        if (offset < static_cast<int>(request.opponent_scenario.size()))
            opponent = request.opponent_scenario[static_cast<std::size_t>(offset)];
        std::array<fastkag::PlayerAction, 2> actions;
        actions[request.player] = std::move(own);
        actions[1 - request.player] = std::move(opponent);
        reference.step(actions);
    }

    for (int offset = 0; offset < request.options.replay_horizon && !branch.done(); ++offset) {
        const int absolute = request.absolute_start_step + offset;
        fastkag::PlayerAction own;
        if (offset < static_cast<int>(request.fixed_own_scenario.size()))
            own = request.fixed_own_scenario[static_cast<std::size_t>(offset)];
        else if (absolute < static_cast<int>(request.own_tape->size()))
            own = (*request.own_tape)[static_cast<std::size_t>(absolute)];
        if (request.actor >= static_cast<int>(own.units.size()))
            own.units.resize(static_cast<std::size_t>(request.actor + 1));
        int source = -1;
        if (offset == 0) {
            own.units[static_cast<std::size_t>(request.actor)] =
                fastkag::Action{fastkag::Op::DIG};
        } else {
            source = offset - 1;
            if (source >= skipped_source) ++source;
            own.units[static_cast<std::size_t>(request.actor)] = tape_unit(
                *request.own_tape, request.absolute_start_step + source, request.actor);
        }
        if (source >= 0 && is_movement(own.units[static_cast<std::size_t>(request.actor)].op)) {
            const auto actual = actor_position(branch.farms()[request.player], request.actor);
            const auto expected = expected_before[static_cast<std::size_t>(source)];
            metrics.movement_coordinate_violations +=
                actual.x != expected.x || actual.y != expected.y;
        }

        auto projected_farm = branch.farms()[request.player];
        auto projected_private = branch.privates()[request.player];
        for (int actor = 0; actor < static_cast<int>(own.units.size()); ++actor) {
            const auto operation = own.units[static_cast<std::size_t>(actor)].op;
            if (!is_productive(operation)) {
                (void)apply_projected(projected_farm, projected_private, branch.config(),
                                      actor, own.units[static_cast<std::size_t>(actor)], branch.day());
                continue;
            }
            ++metrics.dependencies.productive_attempts;
            ++metrics.dependencies.attempted_by_op[static_cast<int>(operation)];
            if (!apply_projected(projected_farm, projected_private, branch.config(), actor,
                                 own.units[static_cast<std::size_t>(actor)], branch.day())) {
                ++metrics.dependencies.productive_failures;
                ++metrics.dependencies.failed_by_op[static_cast<int>(operation)];
            }
        }

        fastkag::PlayerAction opponent;
        if (offset < static_cast<int>(request.opponent_scenario.size()))
            opponent = request.opponent_scenario[static_cast<std::size_t>(offset)];
        std::array<fastkag::PlayerAction, 2> actions;
        actions[request.player] = std::move(own);
        actions[1 - request.player] = std::move(opponent);
        branch.step(actions);
        const auto& fills = branch.last_market_fills()[request.player];
        const auto& emitted = actions[request.player].market;
        for (std::size_t slot = 0; slot < emitted.size(); ++slot) {
            const int requested = (emitted[slot].op == fastkag::Op::HIRE ||
                                   emitted[slot].op == fastkag::Op::BUY_LAND)
                ? int(emitted[slot].quantity > 0) : quantity(emitted[slot]);
            const int filled = slot < fills.size() ? fills[slot] : 0;
            metrics.dependencies.market_unfilled_units += std::max(0, requested - filled);
        }
        metrics.dependencies.end_of_day_overflow +=
            branch.last_end_of_day_overflow()[request.player];
        ++metrics.simulated_steps;
    }
    finalize_assets(branch, request.player, metrics);
    return metrics;
}

SearchResult select_absorption(const SearchRequest& request) {
    const auto started = std::chrono::steady_clock::now();
    SearchResult result;
    const auto candidates = enumerate_candidates(request);
    result.candidates.reserve(candidates.size());
    for (const int source : candidates)
        result.candidates.push_back(replay_candidate(request, source));
    if (result.candidates.empty()) {
        result.reason = "no_policy_eligible_nonmovement_source";
    } else {
        const auto better = [](const BranchMetrics& left, const BranchMetrics& right) {
            return std::tie(left.objective, left.dependencies.productive_failures,
                            left.dependencies.market_unfilled_units, left.skipped_source) <
                   std::tie(right.objective, right.dependencies.productive_failures,
                            right.dependencies.market_unfilled_units, right.skipped_source);
        };
        const auto best = std::min_element(result.candidates.begin(), result.candidates.end(),
                                           [&](const auto& left, const auto& right) {
                                               return better(left, right);
                                           });
        result.feasible = best->movement_source_edits == 0 &&
                          best->movement_coordinate_violations == 0;
        result.selected_source = best->skipped_source;
        result.selected = *best;
        result.reason = result.feasible ? "minimum_exact_branch_cost"
                                        : "movement_invariant_failed";
    }
    result.elapsed_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace g001::weed_absorption
