#include "belief.hpp"
#include "causal_mfqi.hpp"
#include "persistent_options.hpp"
#include "tape_runner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using g001::market::InventoryBelief;
using g001::market::OpponentInventoryBelief;
using g001::market::PublicStateSummary;
using g001::market::product_count;
using g001::market::tape::CompiledTape;

namespace {

constexpr std::size_t options_count = g001::option::option_count;
constexpr int option_horizon = 24;
constexpr int decision_interval = 4;

struct Options {
    std::string tapes, library, route{"G001"};
    fs::path output, report;
    std::uint64_t seed_begin{};
    std::size_t seeds{4}, threads{192};
};

template <class T> T parse_integer(std::string_view value) {
    T result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("invalid integer: " + std::string(value));
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        auto value = [&]() -> std::string_view {
            if (++index >= argc) throw std::runtime_error("missing option value");
            return argv[index];
        };
        if (argument == "--tapes") result.tapes = value();
        else if (argument == "--library") result.library = value();
        else if (argument == "--route") result.route = value();
        else if (argument == "--output") result.output = value();
        else if (argument == "--report") result.report = value();
        else if (argument == "--seed-begin") result.seed_begin = parse_integer<std::uint64_t>(value());
        else if (argument == "--seeds") result.seeds = parse_integer<std::size_t>(value());
        else if (argument == "--threads") result.threads = parse_integer<std::size_t>(value());
        else if (argument == "--help") {
            std::cout << "fixed_tape_mfqi --tapes FILE --library FILE --output FILE "
                         "--report FILE [--route G001 --seeds 4 --threads 192]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + std::string(argument));
    }
    if (result.tapes.empty() || result.library.empty() || result.output.empty() ||
        result.report.empty() || !result.seeds || !result.threads) {
        throw std::runtime_error("missing required fixed_tape_mfqi option");
    }
    return result;
}

const fastkag::PlayerAction& action_at(const CompiledTape& tape, int step) {
    return tape.at(static_cast<std::size_t>(std::clamp(step, 0, int(tape.size()) - 1)));
}

fastkag::PlayerAction without_product_sales(fastkag::PlayerAction action, int product) {
    std::erase_if(action.market, [=](const auto& item) {
        return item.op == fastkag::Op::SELL && int(item.item) == product;
    });
    return action;
}

bool insert_sale(fastkag::PlayerAction& action, const fastkag::Simulator& simulator,
                 int product, int quantity) {
    if (quantity <= 0 || int(action.market.size()) >= simulator.config().max_market_orders) return false;
    action.market.insert(action.market.begin(), {
        fastkag::Op::SELL, static_cast<fastkag::Item>(product), quantity
    });
    return true;
}

int own_product_stock(const fastkag::Simulator& simulator, int seat, int product) {
    int result = simulator.privates()[seat].shed[static_cast<std::size_t>(product)];
    for (const auto& carried : simulator.privates()[seat].inventories) {
        result += carried[static_cast<std::size_t>(product)];
    }
    return result;
}

int shed_items(const fastkag::Simulator& simulator, int seat) {
    return std::accumulate(simulator.privates()[seat].shed.begin(),
                           simulator.privates()[seat].shed.end(), 0);
}

int carried_items(const fastkag::Simulator& simulator, int seat) {
    int result = 0;
    for (const auto& carried : simulator.privates()[seat].inventories) {
        result += std::accumulate(carried.begin(), carried.end(), 0);
    }
    return result;
}

std::array<std::array<int, product_count>, 2> farm_ready(const fastkag::Simulator& simulator) {
    std::array<std::array<int, product_count>, 2> result{};
    for (int farm = 0; farm < 2; ++farm) {
        for (const auto& tile : simulator.farms()[farm].tiles) {
            int product = -1;
            if (tile.kind == fastkag::TileKind::PLANT) product = int(tile.crop);
            else if (tile.kind == fastkag::TileKind::ANIMAL) {
                if (tile.animal == fastkag::Item::COW) product = int(fastkag::Item::MILK);
                else if (tile.animal == fastkag::Item::SHEEP) product = int(fastkag::Item::WOOL);
                else if (tile.animal == fastkag::Item::GOOSE) product = int(fastkag::Item::EGG);
            }
            if (product >= 0 && product < int(product_count)) {
                result[static_cast<std::size_t>(farm)][static_cast<std::size_t>(product)] +=
                    std::max(0, int(tile.yield_units));
            }
        }
    }
    return result;
}

g001::market::Inventory town_consumption(const fastkag::Simulator& simulator, int step) {
    g001::market::Inventory result{};
    auto add = [&](int product, int quantity = 1) { result[static_cast<std::size_t>(product)] += quantity; };
    if (step >= 0 && step % std::max(1, simulator.config().town_shop_sell_interval) == 0) {
        for (const auto shop : simulator.shops()) {
            switch (shop) {
                case 0: add(5); add(0); break;
                case 1: add(5); add(0); add(3); break;
                case 2: add(0); add(1); add(2); add(3); break;
                case 3: add(3); add(6); add(0); break;
                case 4: add(1, 2); break;
                case 5: add(6); add(2); add(0); break;
                case 6: add(3); add(6); break;
                case 7: add(7, 2); break;
            }
        }
    }
    if (step >= 0 && step % std::max(1, simulator.config().town_center_sell_interval) == 0) {
        for (int product = 0; product < 8; ++product) add(product);
    }
    return result;
}

PublicStateSummary public_summary(
    const fastkag::Simulator& simulator, int seat,
    const PublicStateSummary* previous,
    const std::array<std::array<int, product_count>, 2>* previous_ready
) {
    PublicStateSummary result;
    result.step = simulator.step_count();
    result.day = simulator.day();
    result.own_money = static_cast<std::int64_t>(simulator.farms()[seat].money);
    result.opponent_money = static_cast<std::int64_t>(simulator.farms()[1 - seat].money);
    result.own_hands = simulator.farms()[seat].hands.size();
    result.opponent_hands = simulator.farms()[1 - seat].hands.size();
    result.own_hires_today = simulator.farms()[seat].hires_today;
    result.opponent_hires_today = simulator.farms()[1 - seat].hires_today;
    result.own_unlocked_quadrants = std::popcount(unsigned(simulator.farms()[seat].unlocked_mask));
    result.opponent_unlocked_quadrants = std::popcount(unsigned(simulator.farms()[1 - seat].unlocked_mask));
    result.farm_hand_cost_multiplier = simulator.config().farm_hand_cost_mult;
    for (std::size_t product = 0; product < product_count; ++product) {
        result.market_inventory[product] = simulator.market().inventory[product];
        result.market_price[product] = simulator.market().prices[product];
        result.own_total[product] = own_product_stock(simulator, seat, int(product));
        result.floor_sale_ambiguity[product] = simulator.market().prices[product] == 1;
    }
    result.town_consumption = town_consumption(simulator, simulator.step_count() - 1);
    if (previous && previous_ready) {
        result.day_rollover = result.day != previous->day;
        result.opponent_shed_access_ambiguity = result.day_rollover;
        const auto ready = farm_ready(simulator);
        for (std::size_t product = 0; product < product_count; ++product) {
            result.own_flow.harvest_upper[product] = std::max(
                0, (*previous_ready)[static_cast<std::size_t>(seat)][product] -
                   ready[static_cast<std::size_t>(seat)][product]
            );
            result.opponent_flow.harvest_upper[product] = std::max(
                0, (*previous_ready)[static_cast<std::size_t>(1 - seat)][product] -
                   ready[static_cast<std::size_t>(1 - seat)][product]
            );
        }
    }
    return result;
}

struct Episode {
    std::uint64_t id{}, seed{};
    std::vector<fastkag::Simulator> checkpoints;
    std::array<std::vector<InventoryBelief>, 2> belief;
};

InventoryBelief advance_public_belief(
    InventoryBelief belief, const fastkag::Simulator& before,
    const fastkag::Simulator& after, int focal,
    const g001::market::Inventory& own_sell_fill,
    const g001::market::Inventory& own_buy_fill
);

Episode run_baseline(std::uint64_t id, std::uint64_t seed, const CompiledTape& tape) {
    Episode result;
    result.id = id;
    result.seed = seed;
    fastkag::Simulator simulator({}, seed);
    std::array<OpponentInventoryBelief, 2> initializers{
        OpponentInventoryBelief(simulator.config().shed_capacity),
        OpponentInventoryBelief(simulator.config().shed_capacity)
    };
    std::array<InventoryBelief, 2> current_belief{};
    for (int seat = 0; seat < 2; ++seat) {
        current_belief[seat] = initializers[seat].reset(
            public_summary(simulator, seat, nullptr, nullptr)
        );
    }
    while (!simulator.done()) {
        result.checkpoints.push_back(simulator);
        for (int seat = 0; seat < 2; ++seat) result.belief[seat].push_back(current_belief[seat]);
        const auto before = simulator;
        const auto action = action_at(tape, simulator.step_count());
        simulator.step({action, action});
        for (int seat = 0; seat < 2; ++seat) {
            g001::market::Inventory own_sell_fill{};
            g001::market::Inventory own_buy_fill{};
            const auto& fills = simulator.last_market_fills()[seat];
            for (std::size_t order = 0; order < action.market.size() && order < fills.size(); ++order) {
                const auto item = int(action.market[order].item);
                if (item < 0 || item >= int(product_count)) continue;
                if (action.market[order].op == fastkag::Op::SELL) {
                    own_sell_fill[static_cast<std::size_t>(item)] += fills[order];
                } else if (action.market[order].op == fastkag::Op::BUY_PRODUCT) {
                    own_buy_fill[static_cast<std::size_t>(item)] += fills[order];
                }
            }
            current_belief[seat] = advance_public_belief(
                current_belief[seat], before, simulator, seat,
                own_sell_fill, own_buy_fill
            );
        }
    }
    return result;
}

g001::causal_fqi::RouteCapacityForecast route_forecast(
    const fastkag::Simulator& simulator, int seat, const CompiledTape& tape, int horizon
) {
    g001::causal_fqi::RouteCapacityForecast result;
    result.current_shed_items = shed_items(simulator, seat);
    result.current_carried_items = carried_items(simulator, seat);
    result.shed_capacity = simulator.config().shed_capacity;
    const auto stop = std::min<int>(tape.size(), simulator.step_count() + horizon);
    bool harvest_planned = false;
    for (int step = simulator.step_count(); step < stop; ++step) {
        for (const auto& unit : action_at(tape, step).units) {
            harvest_planned |= unit.op == fastkag::Op::HARVEST;
        }
        for (const auto& order : action_at(tape, step).market) {
            if (order.op == fastkag::Op::BUY_PRODUCT) result.planned_product_buys += std::max(0, order.quantity);
            else if (order.op == fastkag::Op::BUY_ANIMAL) result.planned_animal_buys += std::max(0, order.quantity);
        }
    }
    if (harvest_planned) {
        const auto ready = farm_ready(simulator);
        result.visible_ready_harvest = std::accumulate(
            ready[static_cast<std::size_t>(seat)].begin(),
            ready[static_cast<std::size_t>(seat)].end(), 0
        );
    }
    return result;
}

std::uint64_t split_group(std::string_view route, std::uint64_t seed) {
    std::uint64_t value = 1469598103934665603ULL;
    for (const auto byte : route) { value ^= static_cast<unsigned char>(byte); value *= 1099511628211ULL; }
    // One seed per group for the small smoke. Episode joining still keeps all
    // seats/products/roots from that seed together.
    for (int byte = 0; byte < 8; ++byte) {
        value ^= static_cast<unsigned char>((seed >> (byte * 8)) & 0xffu);
        value *= 1099511628211ULL;
    }
    return value;
}

double margin(const fastkag::Simulator& simulator, int focal) {
    return simulator.farms()[focal].money - simulator.farms()[1 - focal].money;
}

int marginal_price_quantity(const fastkag::Simulator& simulator, int product, int stock,
                            int reservation) {
    int quantity = 0;
    while (quantity < stock && g001::market::price(
        static_cast<g001::market::Product>(product),
        simulator.market().inventory[static_cast<std::size_t>(product)] + quantity
    ) >= reservation) ++quantity;
    return quantity;
}

std::vector<float> features(
    const fastkag::Simulator& simulator, int focal, int product,
    const InventoryBelief& belief, const g001::option::DripState& continuation,
    g001::option::Kind active_kind, int horizon_remaining,
    const g001::causal_fqi::CapacityDecision& capacity,
    const g001::causal_fqi::DumpWindow& dump
) {
    using namespace g001::causal_fqi;
    std::vector<float> result(FeatureCount);
    result[Step] = simulator.step_count(); result[Day] = simulator.day(); result[Hour] = simulator.hour();
    result[OwnMoney] = simulator.farms()[focal].money;
    result[OpponentMoney] = simulator.farms()[1 - focal].money;
    result[CapacityUsed] = shed_items(simulator, focal) + carried_items(simulator, focal);
    result[CapacityFree] = simulator.config().shed_capacity - result[CapacityUsed];
    result[ActiveKind] = static_cast<float>(active_kind);
    result[RemainingQuota] = continuation.remaining_quota;
    result[RemainingWindows] = continuation.remaining_windows;
    result[Debt] = continuation.debt;
    result[HorizonRemaining] = horizon_remaining;
    result[CausalOverflow] = capacity.overflow;
    result[DumpEarliestDistance] = dump.valid ? dump.earliest_step - simulator.step_count() : -1;
    result[DumpLikelyDistance] = dump.valid ? dump.likely_step - simulator.step_count() : -1;
    const auto ready = farm_ready(simulator);
    for (std::size_t item = 0; item < product_count; ++item) {
        result[MarketInventoryBegin + item] = simulator.market().inventory[item];
        result[MarketPriceBegin + item] = simulator.market().prices[item];
        result[OwnStockBegin + item] = own_product_stock(simulator, focal, int(item));
        result[OwnFarmReadyBegin + item] = ready[static_cast<std::size_t>(focal)][item];
        result[OpponentFarmReadyBegin + item] = ready[static_cast<std::size_t>(1 - focal)][item];
    }
    result[BeliefPoint] = belief.total[static_cast<std::size_t>(product)];
    result[BeliefLower] = belief.total_interval.lower[static_cast<std::size_t>(product)];
    result[BeliefUpper] = belief.total_interval.upper[static_cast<std::size_t>(product)];
    result[RecentClearance] = belief.recent_clearance[static_cast<std::size_t>(product)];
    return result;
}

g001::fqi::Transition blank_row(std::uint64_t episode, std::uint64_t group, int product) {
    g001::fqi::Transition row;
    row.episode_id = episode; row.split_group = group; row.product_id = product;
    row.reward.assign(options_count, 0);
    row.duration.assign(options_count, 0);
    row.next_state.assign(options_count, -1);
    return row;
}

struct Chain { std::vector<g001::fqi::Transition> rows; };

int next_epoch(const fastkag::Simulator& simulator, const CompiledTape& tape) {
    const auto current = simulator.step_count();
    for (int step = current + 1; step < simulator.config().episode_steps; ++step) {
        if (!action_at(tape, step).market.empty() || step % simulator.config().turns_per_day == 0) {
            return step;
        }
    }
    return simulator.config().episode_steps;
}

InventoryBelief advance_public_belief(
    InventoryBelief belief, const fastkag::Simulator& before,
    const fastkag::Simulator& after, int focal,
    const g001::market::Inventory& own_sell_fill,
    const g001::market::Inventory& own_buy_fill
) {
    const auto ready_before = farm_ready(before);
    const auto ready_after = farm_ready(after);
    belief.step = after.step_count();
    for (std::size_t product = 0; product < product_count; ++product) {
        int town = 0;
        for (int step = before.step_count(); step < after.step_count(); ++step) {
            town += town_consumption(after, step)[product];
        }
        const auto rival_net_market_flow = g001::causal_fqi::causal_rival_net_market_flow(
            before.market().inventory[product], after.market().inventory[product], town,
            own_sell_fill[product], own_buy_fill[product]
        );
        const auto rival_outflow = std::max(0, rival_net_market_flow);
        const auto rival_visible_harvest = std::max(
            0, ready_before[static_cast<std::size_t>(1 - focal)][product] -
               ready_after[static_cast<std::size_t>(1 - focal)][product]
        );
        belief.total[product] = std::max(
            0, belief.total[product] + rival_visible_harvest - rival_net_market_flow
        );
        belief.total_interval.lower[product] = std::max(
            0, belief.total_interval.lower[product] - rival_outflow
        );
        belief.total_interval.upper[product] = std::max(
            belief.total_interval.lower[product],
            belief.total_interval.upper[product] + rival_visible_harvest - rival_net_market_flow
        );
        belief.recent_clearance[product] = rival_outflow;
        belief.recent_clearance_interval.lower[product] = rival_outflow;
        belief.recent_clearance_interval.upper[product] = std::max(
            rival_outflow, belief.total_interval.upper[product]
        );
    }
    return belief;
}

g001::causal_fqi::DumpWindow advance_dump_window(
    g001::causal_fqi::DumpWindow prior, const InventoryBelief& belief,
    int product, int step, int episode_steps
) {
    if (belief.recent_clearance[static_cast<std::size_t>(product)] > 0 &&
        belief.total_interval.upper[static_cast<std::size_t>(product)] > 0) {
        prior.valid = true;
        prior.observed_clearance = belief.recent_clearance[static_cast<std::size_t>(product)];
        prior.earliest_step = std::min(episode_steps - 1, step + 1);
        prior.likely_step = std::min(episode_steps - 1, step + 8);
    } else if (prior.valid && step > prior.likely_step) {
        prior = {};
        prior.earliest_step = prior.likely_step = -1;
    }
    return prior;
}

struct BranchState {
    fastkag::Simulator simulator;
    InventoryBelief belief{};
    g001::causal_fqi::DumpWindow dump{};
    g001::option::Kind active{g001::option::Kind::Baseline};
    g001::option::DripState continuation{};
    int horizon_remaining{option_horizon};
};

struct IntervalOutcome { BranchState next; float reward{}; std::uint16_t duration{}; };

bool option_valid(const BranchState& state, int focal, int product,
                  const CompiledTape& tape, g001::option::Kind kind) {
    if (kind == g001::option::Kind::Baseline || kind == g001::option::Kind::Hold) return true;
    if (kind == state.active && state.horizon_remaining <= 0) return false;
    if (kind == g001::option::Kind::Drip && kind == state.active &&
        state.continuation.remaining_windows <= 0) return false;
    const auto held = state.simulator.privates()[focal].shed[static_cast<std::size_t>(product)];
    if (kind == g001::option::Kind::InventoryTarget) {
        return g001::causal_fqi::causal_inventory_target(
            route_forecast(state.simulator, focal, tape, option_horizon), held
        ).valid;
    }
    if (kind == g001::option::Kind::PreDump) {
        return state.dump.valid && held > 0 && state.simulator.step_count() < state.dump.likely_step;
    }
    return held > 0 || (kind == state.active &&
        state.continuation.remaining_quota + state.continuation.debt > 0);
}

IntervalOutcome apply_interval(BranchState state, int focal, int product,
                               const CompiledTape& tape, g001::option::Kind kind) {
    const auto before = state.simulator;
    const auto initial_margin = margin(state.simulator, focal);
    const auto stop = next_epoch(state.simulator, tape);
    const auto switching = kind != state.active || state.horizon_remaining <= 0;
    if (switching) {
        const auto held = state.simulator.privates()[focal].shed[static_cast<std::size_t>(product)];
        state.continuation = {
            held, std::max(1, option_horizon / decision_interval), 0
        };
        state.horizon_remaining = option_horizon;
    }
    const auto capacity = g001::causal_fqi::causal_inventory_target(
        route_forecast(state.simulator, focal, tape, option_horizon),
        state.simulator.privates()[focal].shed[static_cast<std::size_t>(product)]
    );
    const auto price_target = state.simulator.market().prices[static_cast<std::size_t>(product)];
    const auto drip_before = state.continuation;
    int requested = 0, filled = 0, available = 0;
    bool inserted = false, predump_executed = false;
    g001::market::Inventory own_sell_fill{};
    g001::market::Inventory own_buy_fill{};
    while (!state.simulator.done() && state.simulator.step_count() < stop) {
        const auto step = state.simulator.step_count();
        auto focal_action = action_at(tape, step);
        if (kind == g001::option::Kind::Hold) {
            focal_action = without_product_sales(std::move(focal_action), product);
        } else if (step == before.step_count() && kind != g001::option::Kind::Baseline &&
                   kind != g001::option::Kind::PreDump) {
            focal_action = without_product_sales(std::move(focal_action), product);
            available = state.simulator.privates()[focal].shed[static_cast<std::size_t>(product)];
            if (kind == g001::option::Kind::Drip) {
                requested = g001::option::settle_drip(drip_before, available, 0).requested;
            } else if (kind == g001::option::Kind::PriceTarget) {
                requested = marginal_price_quantity(state.simulator, product, available, price_target);
            } else if (kind == g001::option::Kind::InventoryTarget) {
                requested = std::max(0, available - capacity.target_inventory);
            } else if (kind == g001::option::Kind::Clear) {
                requested = g001::causal_fqi::clear_quantity(available);
            }
            inserted = insert_sale(focal_action, state.simulator, product, requested);
        } else if (kind == g001::option::Kind::PreDump && !predump_executed) {
            focal_action = without_product_sales(std::move(focal_action), product);
            if (step >= state.dump.earliest_step) {
                available = state.simulator.privates()[focal].shed[static_cast<std::size_t>(product)];
                const auto safe = std::max(2, (state.dump.likely_step - step +
                    decision_interval - 1) / decision_interval + 1);
                requested = g001::causal_fqi::predump_quantity(available, safe);
                inserted = insert_sale(focal_action, state.simulator, product, requested);
                predump_executed = true;
            }
        }
        std::array<fastkag::PlayerAction, 2> actions{
            action_at(tape, step), action_at(tape, step)
        };
        actions[focal] = std::move(focal_action);
        state.simulator.step(actions);
        const auto& fills = state.simulator.last_market_fills()[focal];
        for (std::size_t order = 0; order < actions[focal].market.size() && order < fills.size(); ++order) {
            const auto& request = actions[focal].market[order];
            const auto item = int(request.item);
            if (item < 0 || item >= int(product_count)) continue;
            if (request.op == fastkag::Op::SELL) {
                own_sell_fill[static_cast<std::size_t>(item)] += fills[order];
            } else if (request.op == fastkag::Op::BUY_PRODUCT) {
                own_buy_fill[static_cast<std::size_t>(item)] += fills[order];
            }
        }
        if (inserted && filled == 0 && !state.simulator.last_market_fills()[focal].empty()) {
            filled = state.simulator.last_market_fills()[focal].front();
            inserted = false;
        }
    }
    const auto elapsed = state.simulator.step_count() - before.step_count();
    if (kind == g001::option::Kind::Drip) {
        state.continuation = g001::option::settle_drip(
            drip_before, available, filled
        ).next;
    } else if (kind != g001::option::Kind::Baseline && kind != g001::option::Kind::Hold) {
        state.continuation.remaining_quota = std::max(
            0, state.continuation.remaining_quota - filled
        );
        state.continuation.remaining_windows = std::max(
            0, state.continuation.remaining_windows - 1
        );
        state.continuation.debt = 0;
    }
    state.horizon_remaining = std::max(0, state.horizon_remaining - elapsed);
    state.active = kind;
    state.belief = advance_public_belief(
        state.belief, before, state.simulator, focal, own_sell_fill, own_buy_fill
    );
    state.dump = advance_dump_window(
        state.dump, state.belief, product, state.simulator.step_count(),
        state.simulator.config().episode_steps
    );
    const auto reward = static_cast<float>(margin(state.simulator, focal) - initial_margin);
    return {std::move(state), reward, static_cast<std::uint16_t>(elapsed)};
}

g001::option::Kind switch_kind(g001::option::Kind kind) {
    switch (kind) {
        case g001::option::Kind::Hold: return g001::option::Kind::PriceTarget;
        case g001::option::Kind::PriceTarget: return g001::option::Kind::Drip;
        case g001::option::Kind::Drip: return g001::option::Kind::Clear;
        case g001::option::Kind::InventoryTarget: return g001::option::Kind::PriceTarget;
        case g001::option::Kind::PreDump: return g001::option::Kind::Clear;
        case g001::option::Kind::Clear: return g001::option::Kind::Hold;
        case g001::option::Kind::Baseline: return g001::option::Kind::Hold;
    }
    return g001::option::Kind::Baseline;
}

g001::fqi::Transition state_row(const BranchState& state, std::uint64_t episode,
                                std::uint64_t group, int focal, int product,
                                const CompiledTape& tape) {
    auto row = blank_row(episode, group, product);
    const auto capacity = g001::causal_fqi::causal_inventory_target(
        route_forecast(state.simulator, focal, tape, option_horizon),
        state.simulator.privates()[focal].shed[static_cast<std::size_t>(product)]
    );
    row.feature = features(
        state.simulator, focal, product, state.belief, state.continuation,
        state.active, state.horizon_remaining, capacity, state.dump
    );
    return row;
}

std::size_t append_tail(Chain& chain, BranchState state, std::uint64_t episode,
                        std::uint64_t group, int focal, int product,
                        const CompiledTape& tape, g001::option::Kind desired) {
    const auto first = chain.rows.size();
    while (!state.simulator.done()) {
        auto row = state_row(state, episode, group, focal, product, tape);
        auto kind = option_valid(state, focal, product, tape, desired)
            ? desired : g001::option::Kind::Baseline;
        const auto action = static_cast<std::size_t>(kind);
        auto outcome = apply_interval(state, focal, product, tape, kind);
        row.valid_option_mask = 1ULL << action;
        row.reward[action] = outcome.reward;
        row.duration[action] = outcome.duration;
        if (outcome.next.simulator.done()) {
            row.done_option_mask = 1ULL << action;
            row.next_state[action] = -1;
        } else {
            row.next_state[action] = static_cast<std::int64_t>(chain.rows.size() + 1);
        }
        chain.rows.push_back(std::move(row));
        state = std::move(outcome.next);
    }
    return first;
}

std::size_t append_replan_child(
    Chain& chain, BranchState state, std::uint64_t episode, std::uint64_t group,
    int focal, int product, const CompiledTape& tape, g001::option::Kind previous_kind
) {
    const auto child_index = chain.rows.size();
    chain.rows.push_back(state_row(state, episode, group, focal, product, tape));
    std::array<g001::option::Kind, 3> candidates{
        g001::option::Kind::Baseline, previous_kind, switch_kind(previous_kind)
    };
    std::array<bool, options_count> emitted{};
    for (const auto kind : candidates) {
        const auto action = static_cast<std::size_t>(kind);
        if (emitted[action] || !option_valid(state, focal, product, tape, kind)) continue;
        emitted[action] = true;
        auto outcome = apply_interval(state, focal, product, tape, kind);
        auto& row = chain.rows[child_index];
        row.valid_option_mask |= 1ULL << action;
        row.reward[action] = outcome.reward;
        row.duration[action] = outcome.duration;
        if (outcome.next.simulator.done()) {
            row.done_option_mask |= 1ULL << action;
            row.next_state[action] = -1;
        } else {
            const auto next_index = append_tail(
                chain, std::move(outcome.next), episode, group,
                focal, product, tape, kind
            );
            chain.rows[child_index].next_state[action] = static_cast<std::int64_t>(next_index);
        }
    }
    return child_index;
}

Chain build_chain(const Episode& episode, int focal, int product, int root_step,
                  const CompiledTape& tape, std::uint64_t group) {
    Chain chain;
    BranchState root{
        episode.checkpoints.at(static_cast<std::size_t>(root_step)),
        episode.belief[focal].at(static_cast<std::size_t>(root_step)),
        g001::causal_fqi::public_dump_window(
            episode.belief[focal], static_cast<std::size_t>(root_step),
            static_cast<std::size_t>(product), root_step,
            episode.checkpoints[static_cast<std::size_t>(root_step)].config().episode_steps
        ),
        g001::option::Kind::Baseline,
        {episode.checkpoints[static_cast<std::size_t>(root_step)].privates()[focal]
             .shed[static_cast<std::size_t>(product)],
         option_horizon / decision_interval, 0},
        option_horizon
    };
    chain.rows.push_back(state_row(root, episode.id, group, focal, product, tape));
    for (std::size_t action = 0; action < options_count; ++action) {
        const auto kind = static_cast<g001::option::Kind>(action);
        if (!option_valid(root, focal, product, tape, kind)) continue;
        auto outcome = apply_interval(root, focal, product, tape, kind);
        chain.rows[0].valid_option_mask |= 1ULL << action;
        chain.rows[0].reward[action] = outcome.reward;
        chain.rows[0].duration[action] = outcome.duration;
        if (outcome.next.simulator.done()) {
            chain.rows[0].done_option_mask |= 1ULL << action;
            chain.rows[0].next_state[action] = -1;
        } else {
            const auto next_index = append_replan_child(
                chain, std::move(outcome.next), episode.id,
                group, focal, product, tape, kind
            );
            chain.rows[0].next_state[action] = static_cast<std::int64_t>(next_index);
        }
    }
    return chain;
}

void write_report(const Options& option, const std::vector<g001::fqi::Transition>& rows,
                  std::size_t chains, std::size_t nonterminal_edges,
                  double generation_seconds) {
    std::array<std::uint64_t, options_count> coverage{};
    std::array<std::uint64_t, options_count> terminal{};
    for (const auto& row : rows) for (std::size_t action = 0; action < options_count; ++action) {
        if ((row.valid_option_mask >> action) & 1ULL) {
            ++coverage[action];
            terminal[action] += (row.done_option_mask >> action) & 1ULL;
        }
    }
    if (!option.report.parent_path().empty()) fs::create_directories(option.report.parent_path());
    std::ofstream output(option.report);
    output << std::fixed << std::setprecision(6)
           << "{\n  \"format\": \"MFQITRN1\",\n"
           << "  \"result_kind\": \"native_fixed_tape_causal_counterfactual_not_official_result\",\n"
           << "  \"route\": \"" << option.route << "\",\n"
           << "  \"seeds\": " << option.seeds << ",\n"
           << "  \"native_baseline_episodes\": " << option.seeds << ",\n"
           << "  \"stable_episode_seat_product_chains\": " << chains << ",\n"
           << "  \"rows\": " << rows.size() << ",\n"
           << "  \"mean_rows_per_root\": " << (chains ? double(rows.size()) / chains : 0) << ",\n"
           << "  \"generation_seconds\": " << generation_seconds << ",\n"
           << "  \"bounded_replan_depth\": 2,\n"
           << "  \"nonterminal_closed_edges\": " << nonterminal_edges << ",\n"
           << "  \"reward\": \"exact option-duration delta(own_money-opponent_money)\",\n"
           << "  \"inventory_target\": \"causal current shed/carried + visible ready harvest + fixed-own-tape planned purchases; no realized future\",\n"
           << "  \"predump\": \"historical public belief clearance pulse and public stock upper bound only; invalid without signal; delayed fractional sale distinct from immediate CLEAR\",\n"
           << "  \"continuation\": \"same episode/product with remaining_quota,remaining_windows,debt,horizon in successor features\",\n"
           << "  \"root_epochs\": \"all products at mid=336 and late=624; product-related route trade in macro buckets 240-359 and 600-719; causal capacity/dump risk in 600-719\",\n"
           << "  \"split\": \"one whole seed per group; same-route stability only\",\n"
           << "  \"coverage\": {\n";
    for (std::size_t action = 0; action < options_count; ++action) {
        output << "    \"" << g001::option::name(static_cast<g001::option::Kind>(action))
               << "\": {\"valid_rows\": " << coverage[action]
               << ", \"terminal_rows\": " << terminal[action] << "}"
               << (action + 1 == options_count ? "\n" : ",\n");
    }
    output << "  }\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto generation_start = std::chrono::steady_clock::now();
        const auto option = parse_options(argc, argv);
        const auto tape = g001::market::tape::load_compiled_route(
            option.tapes, option.library, option.route
        );
        std::vector<Episode> episodes(option.seeds);
        std::atomic<std::size_t> next_episode{};
        std::vector<std::thread> workers;
        for (std::size_t worker = 0; worker < std::min(option.threads, option.seeds); ++worker) {
            workers.emplace_back([&] {
                while (true) {
                    const auto index = next_episode.fetch_add(1);
                    if (index >= option.seeds) break;
                    episodes[index] = run_baseline(
                        option.seed_begin + index + 1, option.seed_begin + index, tape
                    );
                }
            });
        }
        for (auto& worker : workers) worker.join();

        constexpr std::array<int, 2> phase_roots{336, 624};
        struct Task { std::size_t episode; int seat, product, step; };
        std::vector<Task> tasks;
        for (std::size_t episode = 0; episode < episodes.size(); ++episode) {
            for (int seat = 0; seat < 2; ++seat) for (int product = 0; product < int(product_count); ++product) {
                std::vector<bool> selected(episodes[episode].checkpoints.size());
                std::array<int, 6> route_epoch{};
                std::array<int, 6> risk_epoch{};
                route_epoch.fill(-1);
                risk_epoch.fill(-1);
                for (const auto step : phase_roots) {
                    if (step < int(selected.size())) selected[static_cast<std::size_t>(step)] = true;
                }
                for (int step = 0; step < int(selected.size()); ++step) {
                    for (const auto& order : action_at(tape, step).market) {
                        int related_product = -1;
                        if ((order.op == fastkag::Op::SELL ||
                             order.op == fastkag::Op::BUY_PRODUCT) &&
                            int(order.item) >= 0 && int(order.item) < int(product_count)) {
                            related_product = int(order.item);
                        } else if (order.op == fastkag::Op::BUY_SEED &&
                                   int(order.item) >= 0 && int(order.item) < fastkag::N_CROPS) {
                            related_product = int(order.item);
                        } else if (order.op == fastkag::Op::BUY_ANIMAL) {
                            if (order.item == fastkag::Item::GOOSE) related_product = int(fastkag::Item::EGG);
                            else if (order.item == fastkag::Item::COW) related_product = int(fastkag::Item::MILK);
                            else if (order.item == fastkag::Item::SHEEP) related_product = int(fastkag::Item::WOOL);
                        }
                        if (related_product == product) {
                            const auto bucket = std::min(5, step / 120);
                            if (bucket == 2 || bucket == 5) {
                                auto& candidate = route_epoch[static_cast<std::size_t>(bucket)];
                                if (candidate < 0) candidate = step;
                            }
                        }
                    }
                    if (step % 24 == 0) {
                        const auto& simulator = episodes[episode].checkpoints[static_cast<std::size_t>(step)];
                        const auto capacity = g001::causal_fqi::causal_inventory_target(
                            route_forecast(simulator, seat, tape, option_horizon),
                            simulator.privates()[seat].shed[static_cast<std::size_t>(product)]
                        );
                        const auto dump = g001::causal_fqi::public_dump_window(
                            episodes[episode].belief[seat], static_cast<std::size_t>(step),
                            static_cast<std::size_t>(product), step, simulator.config().episode_steps
                        );
                        if (capacity.valid || dump.valid) {
                            const auto bucket = std::min(5, step / 120);
                            if (bucket == 5) {
                                auto& candidate = risk_epoch[static_cast<std::size_t>(bucket)];
                                if (candidate < 0) candidate = step;
                            }
                        }
                    }
                }
                for (const auto step : route_epoch) if (step >= 0) {
                    selected[static_cast<std::size_t>(step)] = true;
                }
                for (const auto step : risk_epoch) if (step >= 0) {
                    selected[static_cast<std::size_t>(step)] = true;
                }
                for (std::size_t step = 0; step < selected.size(); ++step) {
                    if (selected[step]) tasks.push_back({episode, seat, product, int(step)});
                }
            }
        }
        std::vector<Chain> chains(tasks.size());
        std::atomic<std::size_t> next_task{};
        workers.clear();
        for (std::size_t worker = 0; worker < std::min(option.threads, tasks.size()); ++worker) {
            workers.emplace_back([&] {
                while (true) {
                    const auto index = next_task.fetch_add(1);
                    if (index >= tasks.size()) break;
                    const auto task = tasks[index];
                    const auto& episode = episodes[task.episode];
                    chains[index] = build_chain(
                        episode, task.seat, task.product, task.step, tape,
                        split_group(option.route, episode.seed)
                    );
                }
            });
        }
        for (auto& worker : workers) worker.join();

        std::vector<g001::fqi::Transition> rows;
        std::size_t nonterminal_edges = 0;
        for (auto& chain : chains) {
            const auto base = rows.size();
            for (auto& row : chain.rows) {
                for (std::size_t action = 0; action < options_count; ++action) {
                    if (((row.valid_option_mask >> action) & 1ULL) &&
                        ((row.done_option_mask >> action) & 1ULL) == 0) {
                        row.next_state[action] += static_cast<std::int64_t>(base);
                        ++nonterminal_edges;
                    }
                }
                rows.push_back(std::move(row));
            }
        }
        g001::causal_fqi::validate_graph(rows, options_count);
        g001::causal_fqi::write_mfqi(
            option.output.string(), rows, g001::causal_fqi::FeatureCount,
            options_count, g001::option::schema_hash
        );
        // Round-trip through the exact trainer loader is part of generation.
        const auto loaded = g001::fqi::load_transitions(option.output.string());
        if (loaded.rows.size() != rows.size() ||
            loaded.option_schema_hash != g001::option::schema_hash) {
            throw std::runtime_error("MFQI trainer-loader roundtrip mismatch");
        }
        const auto generation_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - generation_start
        ).count();
        write_report(option, rows, tasks.size(), nonterminal_edges, generation_seconds);
        std::cout << "fixed_tape_mfqi episodes=" << episodes.size()
                  << " chains=" << tasks.size() << " rows=" << rows.size()
                  << " closed_edges=" << nonterminal_edges
                  << " output=" << option.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixed_tape_mfqi: " << error.what() << '\n';
        return 2;
    }
}
