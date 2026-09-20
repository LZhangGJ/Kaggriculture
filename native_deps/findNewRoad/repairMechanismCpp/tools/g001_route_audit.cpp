#include "repair.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

const char* op_name(fastkag::Op op) {
    switch (op) {
        case fastkag::Op::PASS: return "PASS";
        case fastkag::Op::NORTH: return "NORTH";
        case fastkag::Op::SOUTH: return "SOUTH";
        case fastkag::Op::EAST: return "EAST";
        case fastkag::Op::WEST: return "WEST";
        case fastkag::Op::DROP: return "DROP";
        case fastkag::Op::PICKUP: return "PICKUP";
        case fastkag::Op::PLACE: return "PLACE";
        case fastkag::Op::PLANT: return "PLANT";
        case fastkag::Op::WATER: return "WATER";
        case fastkag::Op::HARVEST: return "HARVEST";
        case fastkag::Op::FERTILIZE: return "FERTILIZE";
        case fastkag::Op::DIG: return "DIG";
        case fastkag::Op::BUILD_COOP: return "BUILD_COOP";
        case fastkag::Op::BUILD_PASTURE: return "BUILD_PASTURE";
        case fastkag::Op::FEED: return "FEED";
        case fastkag::Op::COLLECT_FERTILIZER: return "COLLECT_FERTILIZER";
        case fastkag::Op::CARE: return "CARE";
        case fastkag::Op::BUY_ANIMAL: return "BUY_ANIMAL";
        default: return "OTHER";
    }
}

bool collision_op(fastkag::Op op) {
    return op == fastkag::Op::PLANT || op == fastkag::Op::BUILD_PASTURE;
}

bool toil_op(fastkag::Op op) {
    return op == fastkag::Op::FEED || op == fastkag::Op::CARE ||
           op == fastkag::Op::HARVEST || op == fastkag::Op::COLLECT_FERTILIZER;
}

const char* animal_name(fastkag::Item item) {
    if (item == fastkag::Item::GOOSE) return "GOOSE";
    if (item == fastkag::Item::COW) return "COW";
    if (item == fastkag::Item::SHEEP) return "SHEEP";
    return "UNKNOWN";
}

fastkag::Action unit_at(const fastkag::PlayerAction& turn, int actor) {
    return actor >= 0 && actor < int(turn.units.size())
        ? turn.units[static_cast<std::size_t>(actor)] : fastkag::Action{};
}

fastkag::Position moved(fastkag::Position position, fastkag::Op op) {
    if (op == fastkag::Op::NORTH) --position.y;
    if (op == fastkag::Op::SOUTH) ++position.y;
    if (op == fastkag::Op::EAST) ++position.x;
    if (op == fastkag::Op::WEST) --position.x;
    return position;
}

fastkag::Position position_before(const std::vector<fastkag::PlayerAction>& tape,
                                  int actor, int target_step) {
    fastkag::Position position{};
    for (int step = 0; step < target_step && step < int(tape.size()); ++step) {
        position = moved(position, unit_at(tape[static_cast<std::size_t>(step)], actor).op);
    }
    return position;
}

std::vector<g001::repair::TimedAction> relative_window(
    const std::vector<fastkag::PlayerAction>& tape, int start, int actor, int length
) {
    std::vector<g001::repair::TimedAction> result;
    auto position = position_before(tape, actor, start);
    const int end = std::min(int(tape.size()), start + length);
    for (int step = start; step < end; ++step) {
        const auto action = unit_at(tape[static_cast<std::size_t>(step)], actor);
        g001::repair::TimedAction timed;
        timed.action = action;
        timed.required_position = position;
        timed.earliest_step = 0;
        timed.latest_step = 1'000'000;
        timed.economic_value = g001::repair::default_economic_value(action.op);
        timed.expected_fill = timed.economic_value > 0;
        result.push_back(timed);
        position = moved(position, action.op);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: g001_repair_route_audit ROUTE_ACTIONS_ZLIB ROUTE_LIBRARY_JSON\n";
        return 2;
    }
    const auto tape = g001::repair::load_route(argv[1], argv[2], "G001");
    std::uint64_t movements = 0, collision_sites = 0, drop_move_risks = 0;
    std::uint64_t drop_work_risks = 0, delayed_work_exposures = 0;
    std::uint64_t new_zero_move = 0, new_pass_absorptions = 0;
    std::uint64_t new_low_value_absorptions = 0, new_unresolved = 0;
    std::uint64_t new_after_legacy_window = 0;
    std::uint64_t legacy_critical_failures = 0, new_critical_failures = 0;
    std::uint64_t legacy_economic_loss = 0, new_economic_loss = 0;
    std::map<std::string, std::uint64_t> absorption_distribution;
    for (std::size_t step = 0; step < tape.size(); ++step) {
        for (const auto& action : tape[step].units) {
            movements += g001::repair::is_movement(action.op);
        }
        for (int actor = 0; actor < int(tape[step].units.size()); ++actor) {
            if (!collision_op(unit_at(tape[step], actor).op)) continue;
            ++collision_sites;
            const auto window = relative_window(tape, int(step), actor, 34);
            const auto start_position = position_before(tape, actor, int(step));
            const auto legacy_plan = g001::repair::legacy_fixed_window(window, start_position);
            const auto searched_plan = g001::repair::search_minimum_loss_realign(
                window, start_position, std::min(24, int(window.size()) - 1));
            legacy_critical_failures += legacy_plan.metrics.critical_fill_failures;
            legacy_economic_loss += legacy_plan.metrics.economic_value_lost;
            new_critical_failures += searched_plan.metrics.critical_fill_failures;
            new_economic_loss += searched_plan.metrics.economic_value_lost;
            if (searched_plan.metrics.objective == std::numeric_limits<std::int64_t>::max()) {
                ++new_unresolved;
            } else {
                new_zero_move += searched_plan.metrics.movement_edits == 0;
                const int skipped = searched_plan.metrics.skipped_source_index;
                const auto skipped_op = skipped >= 0 && skipped < int(window.size())
                    ? window[static_cast<std::size_t>(skipped)].action.op : fastkag::Op::PASS;
                ++absorption_distribution[op_name(skipped_op)];
                new_pass_absorptions += skipped_op == fastkag::Op::PASS;
                new_low_value_absorptions += skipped_op != fastkag::Op::PASS;
                new_after_legacy_window += skipped > 9;
            }
            for (int offset = 1; offset <= 8 && step + offset < tape.size(); ++offset) {
                delayed_work_exposures += g001::repair::default_economic_value(
                    unit_at(tape[step + offset], actor).op) > 0;
            }
            if (step + 9 >= tape.size()) continue;
            const auto dropped = unit_at(tape[step + 9], actor).op;
            drop_move_risks += g001::repair::is_movement(dropped);
            drop_work_risks += g001::repair::default_economic_value(dropped) > 0;
            if (g001::repair::is_movement(dropped) ||
                g001::repair::default_economic_value(dropped) > 0) {
                std::cout << "{\"kind\":\"weed_fixed_drop_risk\",\"collision_step\":"
                          << step << ",\"actor\":" << actor
                          << ",\"intended\":\"" << op_name(unit_at(tape[step], actor).op)
                          << "\",\"dropped_step\":" << step + 9
                          << ",\"dropped_op\":\"" << op_name(dropped) << "\"}\n";
            }
        }
    }

    int animal_orders = 0, no_retry_before_place = 0, toil_proxy = 0;
    int stationary_recovery_candidates = 0, stationary_unresolved = 0;
    for (int step = 0; step < int(tape.size()); ++step) {
        for (int slot = 0; slot < int(tape[step].market.size()); ++slot) {
            const auto& order = tape[step].market[static_cast<std::size_t>(slot)];
            if (order.op != fastkag::Op::BUY_ANIMAL) continue;
            ++animal_orders;
            int next_buy = -1, next_place = -1, place_actor = -1;
            for (int future = step + 1; future < int(tape.size()); ++future) {
                if (next_buy < 0) {
                    for (const auto& candidate : tape[future].market) {
                        if (candidate.op == fastkag::Op::BUY_ANIMAL &&
                            candidate.item == order.item) { next_buy = future; break; }
                    }
                }
                if (next_place < 0) {
                    for (int actor = 0; actor < int(tape[future].units.size()); ++actor) {
                        const auto candidate = unit_at(tape[future], actor);
                        if (candidate.op == fastkag::Op::PLACE && candidate.item == order.item) {
                            next_place = future; place_actor = actor; break;
                        }
                    }
                }
                if (next_buy >= 0 && next_place >= 0) break;
            }
            int next_pickup = -1;
            if (next_place >= 0 && place_actor >= 0) {
                for (int future = step + 1; future <= next_place; ++future) {
                    if (unit_at(tape[static_cast<std::size_t>(future)], place_actor).op ==
                        fastkag::Op::PICKUP) { next_pickup = future; break; }
                }
            }
            int toil = 0;
            int recoverable = 0, unresolved = 0;
            if (next_place >= 0 && place_actor >= 0) {
                const int end = std::min(int(tape.size()), next_place + 25);
                for (int future = next_place + 1; future < end; ++future) {
                    if (!toil_op(unit_at(tape[future], place_actor).op)) continue;
                    ++toil;
                    auto candidates = relative_window(tape, future + 1, place_actor, 13);
                    // This is a structural upper bound: mark only non-stall
                    // productive actions at the exact same route position.
                    const auto current = position_before(tape, place_actor, future);
                    for (auto& candidate : candidates) {
                        candidate.earliest_step = future;
                        candidate.latest_step = future;
                    }
                    const auto replacement = g001::repair::search_stationary_work_replacement(
                        current, future, candidates, 12);
                    if (replacement.recovered) ++recoverable;
                    else ++unresolved;
                }
            }
            toil_proxy += toil;
            stationary_recovery_candidates += recoverable;
            stationary_unresolved += unresolved;
            const int retry_deadline = next_pickup >= 0 ? next_pickup : next_place;
            bool pickup_realign_zero_move = false;
            int pickup_realign_skip = -1, pickup_realign_critical = -1;
            int pickup_realign_economic_loss = -1;
            const char* pickup_realign_skip_op = "NONE";
            if (next_pickup == step + 1 && place_actor >= 0) {
                const auto pickup_window = relative_window(tape, next_pickup, place_actor, 34);
                const auto pickup_position = position_before(tape, place_actor, next_pickup);
                const auto alignment = g001::repair::search_minimum_loss_realign(
                    pickup_window, pickup_position,
                    std::min(24, int(pickup_window.size()) - 1), 1);
                pickup_realign_zero_move = alignment.metrics.movement_edits == 0;
                pickup_realign_skip = alignment.metrics.skipped_source_index;
                pickup_realign_critical = alignment.metrics.critical_fill_failures;
                pickup_realign_economic_loss = alignment.metrics.economic_value_lost;
                if (pickup_realign_skip >= 0 && pickup_realign_skip < int(pickup_window.size()))
                    pickup_realign_skip_op = op_name(
                        pickup_window[static_cast<std::size_t>(pickup_realign_skip)].action.op);
            }
            const bool unprotected = retry_deadline >= 0 &&
                (next_buy < 0 || next_buy > retry_deadline);
            no_retry_before_place += unprotected;
            std::cout << "{\"kind\":\"animal_order\",\"step\":" << step
                      << ",\"slot\":" << slot << ",\"animal\":\""
                      << animal_name(order.item)
                      << "\",\"quantity\":" << order.quantity
                      << ",\"next_same_buy\":" << next_buy
                      << ",\"next_pickup\":" << next_pickup
                      << ",\"next_place\":" << next_place
                      << ",\"retry_deadline\":" << retry_deadline
                      << ",\"retry_window_steps\":"
                      << (retry_deadline >= 0 ? retry_deadline - step : -1)
                      << ",\"causal_retry_window_after_detection\":"
                      << (retry_deadline >= 0 ? retry_deadline - (step + 1) : -1)
                      << ",\"same_step_pickup_realign_zero_move\":"
                      << (pickup_realign_zero_move ? "true" : "false")
                      << ",\"pickup_realign_skipped_source\":" << pickup_realign_skip
                      << ",\"pickup_realign_skipped_op\":\"" << pickup_realign_skip_op
                      << "\",\"pickup_realign_critical_failures\":" << pickup_realign_critical
                      << ",\"pickup_realign_economic_loss_units\":" << pickup_realign_economic_loss
                      << ",\"place_actor\":" << place_actor
                      << ",\"no_retry_before_place\":" << (unprotected ? "true" : "false")
                      << ",\"post_place_24step_toil_proxy\":" << toil
                      << ",\"same_position_recovery_upper_bound\":" << recoverable
                      << ",\"stationary_unresolved\":" << unresolved << "}\n";
        }
    }
    std::cout << "{\"kind\":\"route_summary\",\"route\":\"G001\",\"steps\":"
              << tape.size() << ",\"movement_actions\":" << movements
              << ",\"weed_collision_sites\":" << collision_sites
              << ",\"fixed_drop_move_risks\":" << drop_move_risks
              << ",\"fixed_drop_work_risks\":" << drop_work_risks
              << ",\"delayed_work_exposures\":" << delayed_work_exposures
              << ",\"legacy_critical_fill_failures_structural\":" << legacy_critical_failures
              << ",\"legacy_economic_loss_units\":" << legacy_economic_loss
              << ",\"new_zero_move_repairs\":" << new_zero_move
              << ",\"new_pass_absorptions\":" << new_pass_absorptions
              << ",\"new_low_value_absorptions\":" << new_low_value_absorptions
              << ",\"new_catchup_after_source9\":" << new_after_legacy_window
              << ",\"new_unresolved\":" << new_unresolved
              << ",\"new_critical_fill_failures_structural\":" << new_critical_failures
              << ",\"new_economic_loss_units\":" << new_economic_loss
              << ",\"animal_orders\":" << animal_orders
              << ",\"animal_orders_without_retry_before_place\":" << no_retry_before_place
              << ",\"post_place_toil_proxy\":" << toil_proxy
              << ",\"same_position_recovery_upper_bound\":" << stationary_recovery_candidates
              << ",\"stationary_unresolved\":" << stationary_unresolved << "}\n";
    std::cout << "{\"kind\":\"realignment_absorption_distribution\",\"counts\":{";
    bool first = true;
    for (const auto& [operation, count] : absorption_distribution) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << '\"' << operation << "\":" << count;
    }
    std::cout << "}}\n";
}
