#include "belief.hpp"
#include "mechanism_tree.hpp"
#include "persistent_options.hpp"
#include "replay_format.hpp"
#include "tape_runner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using g001::market::Inventory;
using g001::market::InventoryBelief;
using g001::market::OpponentInventoryBelief;
using g001::market::Product;
using g001::market::PublicStateSummary;
using g001::market::price;
using g001::market::product_count;
using g001::market::tape::CompiledTape;
using g001::replay::FileHeader;
using g001::replay::Record;
using g001::replay::ReplayHeader;
using g001::tree::CounterfactualHeader;
using g001::tree::CounterfactualRow;
using g001::tree::MechanismAction;

namespace {

struct Options {
    std::string tapes;
    std::string library;
    std::string route = "G001";
    std::uint64_t seed_begin = 0;
    std::size_t seeds = 64;
    std::size_t threads = 192;
    fs::path mra;
    fs::path labels;
    fs::path report;
    fs::path persistent;
    fs::path persistent_report;
};

template <class Integer>
Integer integer(std::string_view value) {
    Integer result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("invalid integer: " + std::string(value));
    }
    return result;
}

Options options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        auto next = [&]() -> std::string_view {
            if (++i >= argc) throw std::runtime_error("missing option value");
            return argv[i];
        };
        if (argument == "--tapes") result.tapes = next();
        else if (argument == "--library") result.library = next();
        else if (argument == "--route") result.route = next();
        else if (argument == "--seed-begin") result.seed_begin = integer<std::uint64_t>(next());
        else if (argument == "--seeds") result.seeds = integer<std::size_t>(next());
        else if (argument == "--threads") result.threads = integer<std::size_t>(next());
        else if (argument == "--mra") result.mra = next();
        else if (argument == "--labels") result.labels = next();
        else if (argument == "--report") result.report = next();
        else if (argument == "--persistent-options") result.persistent = next();
        else if (argument == "--persistent-report") result.persistent_report = next();
        else if (argument == "--help") {
            std::cout << "fixed_tape_labels --tapes FILE --library FILE --mra FILE "
                         "--labels FILE --report FILE [--persistent-options FILE "
                         "--persistent-report FILE] [--route G001] "
                         "[--seed-begin 0] [--seeds 64] [--threads 192]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + std::string(argument));
    }
    if (result.tapes.empty() || result.library.empty() || result.mra.empty() ||
        result.labels.empty() || result.report.empty() || result.seeds == 0 ||
        result.threads == 0) throw std::runtime_error("missing required option");
    if (result.persistent.empty() != result.persistent_report.empty()) {
        throw std::runtime_error("persistent output and report must be specified together");
    }
    return result;
}

CompiledTape::value_type action_at(const CompiledTape& tape, int step) {
    return tape[static_cast<std::size_t>(std::min(step, int(tape.size()) - 1))];
}

bool is_purchase(fastkag::Op op) {
    return op == fastkag::Op::BUY_PRODUCT || op == fastkag::Op::BUY_SEED ||
        op == fastkag::Op::BUY_ANIMAL || op == fastkag::Op::HIRE ||
        op == fastkag::Op::BUY_LAND;
}

bool has_purchase(const fastkag::PlayerAction& action) {
    return std::any_of(action.market.begin(), action.market.end(), [](const auto& item) {
        return is_purchase(item.op);
    });
}

bool has_sell(const fastkag::PlayerAction& action) {
    return std::any_of(action.market.begin(), action.market.end(), [](const auto& item) {
        return item.op == fastkag::Op::SELL && item.quantity > 0;
    });
}

fastkag::PlayerAction without_sales(fastkag::PlayerAction action) {
    std::erase_if(action.market, [](const auto& item) {
        return item.op == fastkag::Op::SELL;
    });
    return action;
}

int shortfall_orders(const fastkag::Simulator& simulator, int player) {
    int result = 0;
    for (const auto value : simulator.last_market_cash_shortfalls()[player]) {
        result += value > 0.0;
    }
    return result;
}

std::int64_t current_cash_reservation(
    const fastkag::Simulator& state,
    int player,
    const std::array<fastkag::PlayerAction, 2>& baseline
) {
    auto actions = baseline;
    actions[player] = without_sales(std::move(actions[player]));
    auto trial = state;
    const auto before = trial.farms()[player].money;
    trial.step(actions);
    const auto spent = std::max(0.0, before - trial.farms()[player].money);
    double maximum_shortfall = 0;
    for (const auto amount : trial.last_market_cash_shortfalls()[player]) {
        maximum_shortfall = std::max(maximum_shortfall, amount);
    }
    return static_cast<std::int64_t>(std::ceil(spent + maximum_shortfall));
}

fastkag::PlayerAction cash_minimum(
    const fastkag::Simulator& state,
    int player,
    const std::array<fastkag::PlayerAction, 2>& baseline
) {
    auto result = without_sales(baseline[player]);
    std::array<int, product_count> sale{};
    for (int guard = 0; guard <= state.config().shed_capacity; ++guard) {
        auto trial_actions = baseline;
        trial_actions[player] = result;
        auto trial = state;
        trial.step(trial_actions);
        if (shortfall_orders(trial, player) == 0) return result;
        int best = -1;
        int best_quote = -1;
        for (std::size_t product = 0; product < product_count; ++product) {
            if (sale[product] >= state.privates()[player].shed[product]) continue;
            const auto quote = price(
                static_cast<Product>(product),
                state.market().inventory[product] + sale[product]
            );
            if (quote > best_quote) { best_quote = quote; best = int(product); }
        }
        if (best < 0) return result;
        ++sale[static_cast<std::size_t>(best)];
        result = without_sales(baseline[player]);
        for (std::size_t product = 0; product < product_count; ++product) {
            if (sale[product] > 0) result.market.insert(result.market.begin(), {
                fastkag::Op::SELL, static_cast<fastkag::Item>(product), sale[product]
            });
        }
    }
    return result;
}

fastkag::PlayerAction sell_units(
    const fastkag::Simulator& state,
    int player,
    fastkag::PlayerAction baseline,
    int units,
    bool low_value_first,
    const Inventory* product_filter = nullptr
) {
    auto result = without_sales(std::move(baseline));
    struct ProductRow { int product, quote, available; };
    std::vector<ProductRow> products;
    for (std::size_t product = 0; product < product_count; ++product) {
        auto available = state.privates()[player].shed[product];
        if (product_filter && (*product_filter)[product] == 0) available = 0;
        if (available > 0) products.push_back({
            int(product),
            price(static_cast<Product>(product), state.market().inventory[product]),
            available
        });
    }
    std::sort(products.begin(), products.end(), [&](const auto& left, const auto& right) {
        return low_value_first ? left.quote < right.quote : left.quote > right.quote;
    });
    const auto order_slots = std::max(0, state.config().max_market_orders - int(result.market.size()));
    std::vector<fastkag::Action> sells;
    for (const auto& product : products) {
        if (units <= 0 || int(sells.size()) >= order_slots) break;
        const auto quantity = std::min(units, product.available);
        sells.push_back({
            fastkag::Op::SELL, static_cast<fastkag::Item>(product.product), quantity
        });
        units -= quantity;
    }
    result.market.insert(result.market.begin(), sells.begin(), sells.end());
    return result;
}

int exact_day_overflow_if_hold(
    const fastkag::Simulator& checkpoint,
    int focal,
    const CompiledTape& tape
) {
    auto simulator = checkpoint;
    int overflow = 0;
    const auto start_day = simulator.day();
    while (!simulator.done() && simulator.day() == start_day) {
        std::array<fastkag::PlayerAction, 2> actions{
            action_at(tape, simulator.step_count()), action_at(tape, simulator.step_count())
        };
        if (simulator.step_count() == checkpoint.step_count()) {
            actions[focal] = without_sales(std::move(actions[focal]));
        }
        simulator.step(actions);
        overflow += simulator.last_end_of_day_overflow()[focal];
    }
    return overflow;
}

double rollout(
    fastkag::Simulator simulator,
    int focal,
    const CompiledTape& tape,
    const fastkag::PlayerAction& intervention
) {
    std::array<fastkag::PlayerAction, 2> first{
        action_at(tape, simulator.step_count()), action_at(tape, simulator.step_count())
    };
    first[focal] = intervention;
    simulator.step(first);
    while (!simulator.done()) {
        const auto action = action_at(tape, simulator.step_count());
        simulator.step({action, action});
    }
    return simulator.farms()[focal].money;
}

Record record(const fastkag::Simulator& simulator, int seat, const fastkag::PlayerAction& action) {
    Record result{};
    result.turn = static_cast<std::uint16_t>(simulator.step_count());
    result.seat = static_cast<std::uint8_t>(seat);
    result.day = static_cast<std::uint8_t>(simulator.day());
    result.hour = static_cast<std::uint8_t>(simulator.hour());
    result.money = static_cast<float>(simulator.farms()[seat].money);
    result.farm_money[0] = static_cast<float>(simulator.farms()[0].money);
    result.farm_money[1] = static_cast<float>(simulator.farms()[1].money);
    for (std::size_t product = 0; product < product_count; ++product) {
        result.market_inventory[product] = simulator.market().inventory[product];
        result.market_price[product] = static_cast<std::int16_t>(simulator.market().prices[product]);
    }
    const auto& private_state = simulator.privates()[seat];
    for (std::size_t item = 0; item < product_count; ++item) {
        int quantity = private_state.shed[item];
        for (const auto& carried : private_state.inventories) quantity += carried[item];
        result.own_stock[item] = static_cast<std::int16_t>(std::clamp(quantity, 0, 32767));
    }
    // Native: GOOSE,COW,SHEEP. MRA schema: COW,SHEEP,GOOSE.
    constexpr std::array<int, 3> native_animal{10, 11, 9};
    for (std::size_t animal = 0; animal < native_animal.size(); ++animal) {
        const auto native = static_cast<std::size_t>(native_animal[animal]);
        int quantity = private_state.shed[native];
        for (const auto& carried : private_state.inventories) quantity += carried[native];
        result.own_stock[product_count + animal] = static_cast<std::int16_t>(
            std::clamp(quantity, 0, 32767)
        );
    }
    for (std::size_t crop = 0; crop < fastkag::N_CROPS; ++crop) {
        result.own_seeds[crop] = static_cast<std::int16_t>(private_state.seeds[crop]);
    }
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
                ++result.farm_producers[farm][product];
                result.farm_ready[farm][product] = static_cast<std::int16_t>(
                    std::clamp<int>(result.farm_ready[farm][product] + tile.yield_units, 0, 32767)
                );
            }
        }
    }
    for (const auto& order : action.market) {
        const auto item = int(order.item);
        const auto quantity = static_cast<std::int16_t>(std::clamp(order.quantity, 0, 32767));
        if (order.op == fastkag::Op::SELL && item >= 0 && item < 9) result.sell[item] += quantity;
        else if (order.op == fastkag::Op::BUY_PRODUCT && item >= 0 && item < 9) result.buy_product[item] += quantity;
        else if (order.op == fastkag::Op::BUY_SEED && item >= 0 && item < 5) result.buy_seed[item] += quantity;
        else if (order.op == fastkag::Op::BUY_ANIMAL && item >= 9 && item < 12) result.buy_animal[item - 9] += quantity;
        else if (order.op == fastkag::Op::HIRE) ++result.hire_count;
        else if (order.op == fastkag::Op::BUY_LAND) ++result.buy_land_count;
    }
    return result;
}

void verify_native_record_contract(
    const fastkag::Simulator& simulator,
    int seat,
    const fastkag::PlayerAction& producing_action,
    const Record& value
) {
    constexpr std::array<int, 3> native_animal{10, 11, 9};
    const auto& private_state = simulator.privates()[seat];
    for (std::size_t animal = 0; animal < native_animal.size(); ++animal) {
        const auto native = static_cast<std::size_t>(native_animal[animal]);
        int expected = private_state.shed[native];
        for (const auto& carried : private_state.inventories) expected += carried[native];
        if (value.own_stock[product_count + animal] != expected) {
            throw std::logic_error("native/MRA animal mapping mismatch");
        }
    }
    Inventory expected_sell{};
    for (const auto& order : producing_action.market) {
        if (order.op == fastkag::Op::SELL && int(order.item) >= 0 && int(order.item) < 9) {
            expected_sell[static_cast<std::size_t>(order.item)] += order.quantity;
        }
    }
    for (std::size_t product = 0; product < product_count; ++product) {
        if (value.sell[product] != expected_sell[product]) {
            throw std::logic_error("MRA producing-action alignment mismatch");
        }
    }
}

Inventory town_consumption(const fastkag::Simulator& simulator, int transition_step) {
    Inventory result{};
    auto take = [&](int product, int quantity = 1) { result[product] += quantity; };
    if (transition_step >= 0 &&
        transition_step % std::max(1, simulator.config().town_shop_sell_interval) == 0) {
        for (const auto shop : simulator.shops()) {
            switch (shop) {
                case 0: take(5); take(0); break;
                case 1: take(5); take(0); take(3); break;
                case 2: take(0); take(1); take(2); take(3); break;
                case 3: take(3); take(6); take(0); break;
                case 4: take(1, 2); break;
                case 5: take(6); take(2); take(0); break;
                case 6: take(3); take(6); break;
                case 7: take(7, 2); break;
            }
        }
    }
    if (transition_step >= 0 &&
        transition_step % std::max(1, simulator.config().town_center_sell_interval) == 0) {
        for (int product = 0; product < 8; ++product) ++result[product];
    }
    return result;
}

PublicStateSummary public_summary(
    const fastkag::Simulator& simulator,
    int seat,
    const Record& current,
    const Record* previous
) {
    PublicStateSummary result;
    result.step = simulator.step_count();
    result.day = simulator.day();
    result.own_money = static_cast<std::int64_t>(simulator.farms()[seat].money);
    result.opponent_money = static_cast<std::int64_t>(simulator.farms()[1 - seat].money);
    result.own_hands = static_cast<int>(simulator.farms()[seat].hands.size());
    result.opponent_hands = static_cast<int>(simulator.farms()[1 - seat].hands.size());
    result.own_hires_today = simulator.farms()[seat].hires_today;
    result.opponent_hires_today = simulator.farms()[1 - seat].hires_today;
    result.own_unlocked_quadrants = __builtin_popcount(unsigned(simulator.farms()[seat].unlocked_mask));
    result.opponent_unlocked_quadrants = __builtin_popcount(unsigned(simulator.farms()[1 - seat].unlocked_mask));
    for (std::size_t product = 0; product < product_count; ++product) {
        result.market_inventory[product] = current.market_inventory[product];
        result.market_price[product] = current.market_price[product];
        result.own_total[product] = current.own_stock[product];
        result.floor_sale_ambiguity[product] = current.market_price[product] == 1;
    }
    result.town_consumption = town_consumption(simulator, simulator.step_count() - 1);
    if (previous) {
        result.day_rollover = current.day != previous->day;
        result.opponent_shed_access_ambiguity = result.day_rollover;
        for (std::size_t product = 0; product < product_count; ++product) {
            const auto own_farm = static_cast<std::size_t>(seat);
            const auto rival_farm = static_cast<std::size_t>(1 - seat);
            result.own_flow.harvest_upper[product] = std::max<int>(
                0, previous->farm_ready[own_farm][product] - current.farm_ready[own_farm][product]
            );
            result.opponent_flow.harvest_upper[product] = std::max<int>(
                0, previous->farm_ready[rival_farm][product] - current.farm_ready[rival_farm][product]
            );
        }
    }
    return result;
}

struct Episode {
    std::uint64_t id{};
    std::uint64_t seed{};
    double reward[2]{};
    std::vector<fastkag::Simulator> checkpoints;
    std::vector<std::array<Record, 2>> records;
    std::vector<std::array<InventoryBelief, 2>> beliefs;
};

Episode baseline_episode(std::uint64_t seed, std::uint64_t id, const CompiledTape& tape) {
    Episode result;
    result.id = id;
    result.seed = seed;
    fastkag::Simulator simulator({}, seed);
    std::array<OpponentInventoryBelief, 2> filters{
        OpponentInventoryBelief(simulator.config().shed_capacity),
        OpponentInventoryBelief(simulator.config().shed_capacity)
    };
    while (!simulator.done()) {
        const auto action = action_at(tape, simulator.step_count());
        const auto producing_action = simulator.step_count() == 0
            ? fastkag::PlayerAction{}
            : action_at(tape, simulator.step_count() - 1);
        std::array<Record, 2> records{
            record(simulator, 0, producing_action), record(simulator, 1, producing_action)
        };
        verify_native_record_contract(simulator, 0, producing_action, records[0]);
        verify_native_record_contract(simulator, 1, producing_action, records[1]);
        std::array<InventoryBelief, 2> beliefs;
        for (int seat = 0; seat < 2; ++seat) {
            const Record* previous = result.records.empty()
                ? nullptr : &result.records.back()[seat];
            const auto summary = public_summary(simulator, seat, records[seat], previous);
            beliefs[seat] = previous ? filters[seat].update(summary) : filters[seat].reset(summary);
        }
        result.checkpoints.push_back(simulator);
        result.records.push_back(records);
        result.beliefs.push_back(beliefs);
        simulator.step({action, action});
    }
    result.reward[0] = simulator.farms()[0].money;
    result.reward[1] = simulator.farms()[1].money;
    // turn 719 is the post-action record that lets a label for simulator step
    // 718 join against causal prior record 718.
    const auto terminal_action = action_at(tape, simulator.step_count() - 1);
    std::array<Record, 2> terminal_records{
        record(simulator, 0, terminal_action), record(simulator, 1, terminal_action)
    };
    verify_native_record_contract(simulator, 0, terminal_action, terminal_records[0]);
    verify_native_record_contract(simulator, 1, terminal_action, terminal_records[1]);
    result.records.push_back(terminal_records);
    return result;
}

int next_sale_step(const CompiledTape& tape, int step) {
    for (int candidate = step; candidate < int(tape.size()); ++candidate) {
        if (has_sell(tape[static_cast<std::size_t>(candidate)])) return candidate;
    }
    return -1;
}

int sale_slots_remaining(const CompiledTape& tape, int step) {
    int result = 0;
    for (int candidate = step; candidate < int(tape.size()); ++candidate) {
        result += has_sell(tape[static_cast<std::size_t>(candidate)]);
    }
    return result;
}

std::int64_t split_group(std::string_view route, std::uint64_t seed_block) {
    // Self-play has one route pair, so eight-seed blocks are appended to create
    // leakage-free same-family stability groups for grouped holdout.
    std::uint64_t value = 1469598103934665603ULL;
    for (const auto character : route) {
        value ^= static_cast<unsigned char>(character);
        value *= 1099511628211ULL;
    }
    value ^= '#'; value *= 1099511628211ULL;
    for (int byte = 0; byte < 8; ++byte) {
        value ^= static_cast<unsigned char>((seed_block >> (byte * 8)) & 0xffu);
        value *= 1099511628211ULL;
    }
    return static_cast<std::int64_t>(value);
}

struct WorkRow {
    std::size_t episode{};
    int turn{};
    int seat{};
    int overflow{};
    bool predump{};
};

struct LabelResult { CounterfactualRow row{}; };

LabelResult label_row(
    const Episode& episode,
    const WorkRow& work,
    const CompiledTape& tape,
    std::uint64_t group
) {
    const auto& checkpoint = episode.checkpoints[static_cast<std::size_t>(work.turn)];
    const auto baseline_action = action_at(tape, work.turn);
    const std::array<fastkag::PlayerAction, 2> baseline{baseline_action, baseline_action};
    LabelResult result;
    auto& row = result.row;
    row.episode_id = episode.id;
    row.split_group = group;
    // Simulator action step s is represented by MCF action turn s+1. The tree
    // joiner then consumes prior MRA record s, exactly checkpoint s.
    row.turn = static_cast<std::uint16_t>(work.turn + 1);
    row.seat = static_cast<std::uint8_t>(work.seat);
    row.valid_action_mask = 1u;
    row.reward[static_cast<int>(MechanismAction::Baseline)] =
        static_cast<float>(episode.reward[work.seat]);

    const auto& causal_record = episode.records[static_cast<std::size_t>(work.turn)][work.seat];
    const auto& belief = episode.beliefs[static_cast<std::size_t>(work.turn)][work.seat];
    if (causal_record.turn != work.turn ||
        causal_record.money != static_cast<float>(checkpoint.farms()[work.seat].money)) {
        throw std::logic_error("causal record/checkpoint turn or money mismatch");
    }
    for (std::size_t product = 0; product < product_count; ++product) {
        if (causal_record.market_inventory[product] != checkpoint.market().inventory[product] ||
            causal_record.market_price[product] != checkpoint.market().prices[product]) {
            throw std::logic_error("causal record/checkpoint market mismatch");
        }
    }
    const auto capacity = checkpoint.config().shed_capacity;
    const auto used = std::accumulate(
        std::begin(causal_record.own_stock), std::end(causal_record.own_stock), 0
    );
    row.capacity_used = static_cast<std::int16_t>(std::clamp(used, -32768, 32767));
    row.capacity_free = static_cast<std::int16_t>(std::clamp(capacity - used, -32768, 32767));
    const auto next_sale = next_sale_step(tape, work.turn);
    row.route_next_sale_step = next_sale < 0 ? -1 : next_sale + 1;
    row.route_sale_slots_remaining = sale_slots_remaining(tape, work.turn);
    row.phase = work.turn >= 708 ? 2 : (has_purchase(baseline_action) ? 0 : 1);
    row.cash_reserved = current_cash_reservation(checkpoint, work.seat, baseline);
    for (std::size_t product = 0; product < product_count; ++product) {
        row.belief_point[product] = static_cast<std::int16_t>(std::clamp(belief.total[product], 0, 32767));
        row.belief_lower[product] = static_cast<std::int16_t>(std::clamp(belief.total_interval.lower[product], 0, 32767));
        row.belief_upper[product] = static_cast<std::int16_t>(std::clamp(belief.total_interval.upper[product], 0, 32767));
        row.recent_clearance[product] = static_cast<std::int16_t>(std::clamp(belief.recent_clearance[product], 0, 32767));
    }
    if (has_purchase(baseline_action)) {
        row.valid_action_mask |= 1u << static_cast<int>(MechanismAction::CashMinimum);
        const auto action = cash_minimum(checkpoint, work.seat, baseline);
        row.reward[static_cast<int>(MechanismAction::CashMinimum)] = static_cast<float>(
            rollout(checkpoint, work.seat, tape, action)
        );
    }
    if (work.overflow > 0) {
        const auto available = std::accumulate(
            checkpoint.privates()[work.seat].shed.begin(),
            checkpoint.privates()[work.seat].shed.begin() + product_count, 0
        );
        if (available >= work.overflow) {
            row.valid_action_mask |= 1u << static_cast<int>(MechanismAction::CapacityMinimum);
            const auto action = sell_units(
                checkpoint, work.seat, baseline_action, work.overflow, true
            );
            row.reward[static_cast<int>(MechanismAction::CapacityMinimum)] = static_cast<float>(
                rollout(checkpoint, work.seat, tape, action)
            );
        }
    }
    if (has_sell(baseline_action)) {
        row.valid_action_mask |= 1u << static_cast<int>(MechanismAction::Hold);
        row.reward[static_cast<int>(MechanismAction::Hold)] = static_cast<float>(
            rollout(checkpoint, work.seat, tape, without_sales(baseline_action))
        );
    }
    if (work.predump) {
        const auto stock = std::accumulate(
            checkpoint.privates()[work.seat].shed.begin(),
            checkpoint.privates()[work.seat].shed.begin() + product_count, 0
        );
        if (stock > 0) {
            row.valid_action_mask |= 1u << static_cast<int>(MechanismAction::PreDump);
            const auto action = sell_units(
                checkpoint, work.seat, baseline_action, stock, false
            );
            row.reward[static_cast<int>(MechanismAction::PreDump)] = static_cast<float>(
                rollout(checkpoint, work.seat, tape, action)
            );
        }
    }
    return result;
}

fastkag::PlayerAction without_product_sales(
    fastkag::PlayerAction action, int product
) {
    std::erase_if(action.market, [&](const auto& item) {
        return item.op == fastkag::Op::SELL && int(item.item) == product;
    });
    return action;
}

bool is_trade_decision(const fastkag::PlayerAction& action) {
    return has_sell(action) || has_purchase(action);
}

int option_windows(const CompiledTape& tape, int start, int horizon) {
    int count = 0;
    const auto stop = std::min<int>(tape.size(), start + horizon);
    for (int step = start; step < stop; ++step) {
        count += step == start || is_trade_decision(tape[static_cast<std::size_t>(step)]);
    }
    return std::max(1, count);
}

struct OptionOutcome {
    float terminal_own_money{};
    float terminal_opponent_money{};
    float terminal_margin{};
    float transition_reward{};
    std::uint16_t next_turn{};
    bool done{};
    g001::option::Continuation continuation{};
    Record next_state{};
};

OptionOutcome option_rollout(
    fastkag::Simulator simulator,
    int focal,
    const CompiledTape& tape,
    g001::option::Kind kind,
    int product,
    int horizon,
    int initial_quota,
    int initial_windows,
    int price_target,
    int inventory_target
) {
    const auto start = simulator.step_count();
    const auto deadline = std::min(simulator.config().episode_steps - 1, start + horizon);
    const auto initial_margin = simulator.farms()[focal].money - simulator.farms()[1 - focal].money;
    int quota = initial_quota;
    int windows = initial_windows;
    int debt = 0;
    OptionOutcome outcome;
    bool captured = false;
    fastkag::PlayerAction producing_focal_action{};

    while (!simulator.done()) {
        const auto step = simulator.step_count();
        if (!captured && step > start && is_trade_decision(action_at(tape, step))) {
            const auto margin = simulator.farms()[focal].money - simulator.farms()[1 - focal].money;
            outcome.transition_reward = static_cast<float>(margin - initial_margin);
            outcome.next_turn = static_cast<std::uint16_t>(step + 1);
            outcome.next_state = record(simulator, focal, producing_focal_action);
            outcome.continuation = {
                kind,
                static_cast<std::uint8_t>(product),
                static_cast<std::uint8_t>(
                    step < deadline && kind != g001::option::Kind::Baseline &&
                    kind != g001::option::Kind::Hold &&
                    kind != g001::option::Kind::PreDump &&
                    kind != g001::option::Kind::Clear
                ),
                0,
                static_cast<std::int16_t>(std::clamp(quota, 0, 32767)),
                static_cast<std::int16_t>(std::clamp(windows, 0, 32767)),
                static_cast<std::int16_t>(std::clamp(debt, 0, 32767)),
                static_cast<std::uint16_t>(std::max(0, deadline - step))
            };
            captured = true;
        }

        auto focal_action = action_at(tape, step);
        int requested = 0;
        int drip_available = 0;
        g001::option::DripState drip_before{quota, windows, debt};
        bool inserted = false;
        if (kind != g001::option::Kind::Baseline && step < deadline &&
            (kind != g001::option::Kind::Hold || step == start)) {
            focal_action = without_product_sales(std::move(focal_action), product);
            const auto held = simulator.privates()[focal].shed[static_cast<std::size_t>(product)];
            drip_available = held;
            const auto window = step == start || is_trade_decision(action_at(tape, step));
            if (kind == g001::option::Kind::Drip && window && windows > 0) {
                requested = g001::option::settle_drip(
                    drip_before, held, 0
                ).requested;
            } else if (kind == g001::option::Kind::PriceTarget &&
                       simulator.market().prices[static_cast<std::size_t>(product)] >= price_target) {
                while (requested < held && price(
                    static_cast<Product>(product),
                    simulator.market().inventory[static_cast<std::size_t>(product)] + requested
                ) >= price_target) ++requested;
            } else if (kind == g001::option::Kind::InventoryTarget) {
                requested = std::max(0, held - inventory_target);
            } else if ((kind == g001::option::Kind::PreDump ||
                        kind == g001::option::Kind::Clear) && step == start) {
                requested = held;
            }
            if (requested > 0 && int(focal_action.market.size()) < simulator.config().max_market_orders) {
                focal_action.market.insert(focal_action.market.begin(), {
                    fastkag::Op::SELL,
                    static_cast<fastkag::Item>(product),
                    requested
                });
                inserted = true;
            }
        }
        std::array<fastkag::PlayerAction, 2> actions{
            action_at(tape, step), action_at(tape, step)
        };
        actions[focal] = focal_action;
        producing_focal_action = focal_action;
        simulator.step(actions);
        if (kind == g001::option::Kind::Drip && step < deadline &&
            (step == start || is_trade_decision(action_at(tape, step)))) {
            const auto filled = inserted && !simulator.last_market_fills()[focal].empty()
                ? simulator.last_market_fills()[focal][0] : 0;
            const auto transition = g001::option::settle_drip(
                drip_before,
                drip_available,
                filled
            );
            quota = transition.next.remaining_quota;
            debt = transition.next.debt;
            windows = transition.next.remaining_windows;
        }
    }
    outcome.terminal_own_money = static_cast<float>(simulator.farms()[focal].money);
    outcome.terminal_opponent_money = static_cast<float>(simulator.farms()[1 - focal].money);
    outcome.terminal_margin = outcome.terminal_own_money - outcome.terminal_opponent_money;
    if (!captured) {
        const auto margin = simulator.farms()[focal].money - simulator.farms()[1 - focal].money;
        outcome.transition_reward = static_cast<float>(margin - initial_margin);
        outcome.next_turn = 719;
        outcome.next_state = record(simulator, focal, producing_focal_action);
        outcome.continuation = {
            kind, static_cast<std::uint8_t>(product), 0, 0,
            static_cast<std::int16_t>(std::clamp(quota, 0, 32767)),
            static_cast<std::int16_t>(std::clamp(windows, 0, 32767)),
            static_cast<std::int16_t>(std::clamp(debt, 0, 32767)), 0
        };
        outcome.done = true;
    }
    return outcome;
}

int selected_product(const fastkag::Simulator& checkpoint, int seat,
                     const fastkag::PlayerAction& baseline) {
    int selected = -1;
    int best = -1;
    for (const auto& order : baseline.market) {
        if (order.op == fastkag::Op::SELL && int(order.item) >= 0 && int(order.item) < 9 &&
            order.quantity > best) {
            selected = int(order.item);
            best = order.quantity;
        }
    }
    if (selected >= 0) return selected;
    for (std::size_t product = 0; product < product_count; ++product) {
        const auto held = checkpoint.privates()[seat].shed[product];
        const auto value = held * checkpoint.market().prices[product];
        if (held > 0 && value > best) { best = value; selected = int(product); }
    }
    return selected;
}

g001::market::TownState planner_town(const fastkag::Simulator& simulator) {
    g001::market::TownState result;
    auto add = [&](int product, int quantity = 1) {
        result.shop_demand_per_tick[static_cast<std::size_t>(product)] += quantity;
    };
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
    result.shop_interval = simulator.config().town_shop_sell_interval;
    result.center_interval = simulator.config().town_center_sell_interval;
    return result;
}

struct PersistentResult { g001::option::Row row{}; };

PersistentResult persistent_row(
    const Episode& episode,
    const WorkRow& work,
    const CompiledTape& tape,
    std::uint64_t group
) {
    PersistentResult result;
    auto& row = result.row;
    const auto& checkpoint = episode.checkpoints[static_cast<std::size_t>(work.turn)];
    const auto baseline = action_at(tape, work.turn);
    const auto product = selected_product(checkpoint, work.seat, baseline);
    row.episode_id = episode.id;
    row.split_group = group;
    row.turn = static_cast<std::uint16_t>(work.turn + 1);
    row.seat = static_cast<std::uint8_t>(work.seat);
    row.product_id = static_cast<std::uint8_t>(std::max(0, product));
    row.horizon = 24;
    if (product < 0) {
        row.valid_mask = 0;
        return result;
    }
    const auto held = checkpoint.privates()[work.seat].shed[static_cast<std::size_t>(product)];
    row.initial_remaining_quota = static_cast<std::int16_t>(held);
    row.initial_remaining_windows = static_cast<std::int16_t>(
        option_windows(tape, work.turn, row.horizon)
    );
    const auto projected_inventory = checkpoint.market().inventory[static_cast<std::size_t>(product)] -
        g001::market::town_drain(
            static_cast<Product>(product), work.turn, work.turn + row.horizon,
            planner_town(checkpoint)
        );
    row.price_target = static_cast<std::int16_t>(std::max(
        checkpoint.market().prices[static_cast<std::size_t>(product)],
        price(static_cast<Product>(product), projected_inventory)
    ));
    row.inventory_target = static_cast<std::int16_t>(std::max(0, held - work.overflow));

    row.valid_mask = 1u;
    if (has_sell(baseline)) {
        row.valid_mask |= 1u << static_cast<int>(g001::option::Kind::Hold);
    }
    if (held > 0) {
        row.valid_mask |= 1u << static_cast<int>(g001::option::Kind::Drip);
        row.valid_mask |= 1u << static_cast<int>(g001::option::Kind::PriceTarget);
        row.valid_mask |= 1u << static_cast<int>(g001::option::Kind::Clear);
    }
    if (held > 0 && work.overflow > 0) {
        row.valid_mask |= 1u << static_cast<int>(g001::option::Kind::InventoryTarget);
    }
    // PRE_DUMP is compiled above but remains invalid until a public causal
    // earliest/likely dump window is available. CLEAR is the honest immediate
    // liquidation option; emitting two identical valid actions would corrupt FQI.

    for (std::size_t index = 0; index < g001::option::option_count; ++index) {
        if (((row.valid_mask >> index) & 1u) == 0) continue;
        const auto kind = static_cast<g001::option::Kind>(index);
        const auto outcome = option_rollout(
            checkpoint, work.seat, tape, kind, product, row.horizon,
            row.initial_remaining_quota, row.initial_remaining_windows,
            row.price_target, row.inventory_target
        );
        row.terminal_own_money[index] = outcome.terminal_own_money;
        row.terminal_opponent_money[index] = outcome.terminal_opponent_money;
        row.terminal_margin[index] = outcome.terminal_margin;
        row.transition_margin_reward[index] = outcome.transition_reward;
        row.next_turn[index] = outcome.next_turn;
        row.continuation[index] = outcome.continuation;
        row.next_state[index] = outcome.next_state;
        if (outcome.done) row.done_mask |= 1u << index;
    }
    return result;
}

void write_mra(const fs::path& path, const std::vector<Episode>& episodes) {
    fs::create_directories(path.parent_path().empty() ? fs::path(".") : path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    FileHeader header{};
    std::memcpy(header.magic, g001::replay::file_magic, 8);
    header.version = g001::replay::format_version;
    header.record_size = sizeof(Record);
    header.replay_count = episodes.size();
    for (const auto& episode : episodes) header.record_count += episode.records.size() * 2;
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    constexpr std::string_view name = "G001-raw-fixed-tape";
    for (const auto& episode : episodes) {
        ReplayHeader replay{};
        replay.record_count = static_cast<std::uint32_t>(episode.records.size() * 2);
        replay.episode_id = episode.id;
        replay.seed = episode.seed;
        replay.reward[0] = episode.reward[0];
        replay.reward[1] = episode.reward[1];
        replay.team_name_bytes[0] = name.size();
        replay.team_name_bytes[1] = name.size();
        replay.chunk_bytes = sizeof(replay) + 2 * name.size() +
            replay.record_count * sizeof(Record);
        output.write(reinterpret_cast<const char*>(&replay), sizeof(replay));
        output.write(name.data(), name.size());
        output.write(name.data(), name.size());
        for (const auto& turn : episode.records) {
            output.write(reinterpret_cast<const char*>(&turn[0]), sizeof(Record));
            output.write(reinterpret_cast<const char*>(&turn[1]), sizeof(Record));
        }
    }
    if (!output) throw std::runtime_error("failed writing MRA");
}

void write_labels(const fs::path& path, const std::vector<LabelResult>& rows) {
    fs::create_directories(path.parent_path().empty() ? fs::path(".") : path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    CounterfactualHeader header{};
    std::memcpy(header.magic, "MCFLABEL", 8);
    header.version = 1;
    header.row_size = sizeof(CounterfactualRow);
    header.row_count = rows.size();
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (const auto& row : rows) {
        output.write(reinterpret_cast<const char*>(&row.row), sizeof(row.row));
    }
    if (!output) throw std::runtime_error("failed writing labels");
}

void write_persistent(
    const fs::path& path, const std::vector<PersistentResult>& rows
) {
    fs::create_directories(path.parent_path().empty() ? fs::path(".") : path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    g001::option::Header header{};
    std::memcpy(header.magic, "MCFOPTN1", 8);
    header.version = 1;
    header.row_size = sizeof(g001::option::Row);
    header.row_count = rows.size();
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    std::vector<char> serialized(sizeof(g001::option::Row));
    for (const auto& item : rows) {
        std::memcpy(serialized.data(), &item.row, sizeof(item.row));
        output.write(serialized.data(), serialized.size());
    }
    output.close();
    std::ifstream verify(path, std::ios::binary);
    std::array<char, sizeof(g001::option::Header)> header_bytes{};
    verify.read(header_bytes.data(), header_bytes.size());
    g001::option::Header loaded{};
    std::memcpy(&loaded, header_bytes.data(), sizeof(loaded));
    if (!verify || std::memcmp(loaded.magic, "MCFOPTN1", 8) != 0 ||
        loaded.version != 1 || loaded.row_size != sizeof(g001::option::Row) ||
        loaded.schema_hash_value != g001::option::schema_hash ||
        loaded.row_count != rows.size() || fs::file_size(path) !=
            sizeof(loaded) + rows.size() * sizeof(g001::option::Row)) {
        throw std::runtime_error("persistent option ABI verification failed");
    }
}

void write_persistent_report(
    const Options& option, const std::vector<PersistentResult>& rows
) {
    std::array<std::uint64_t, g001::option::option_count> valid{};
    std::array<double, g001::option::option_count> best{};
    std::array<double, g001::option::option_count> regret{};
    for (const auto& item : rows) {
        float oracle = -std::numeric_limits<float>::infinity();
        int ties = 0;
        for (std::size_t action = 0; action < g001::option::option_count; ++action) {
            if ((item.row.valid_mask >> action) & 1u) {
                ++valid[action];
                oracle = std::max(oracle, item.row.terminal_margin[action]);
            }
        }
        for (std::size_t action = 0; action < g001::option::option_count; ++action) {
            if (((item.row.valid_mask >> action) & 1u) &&
                item.row.terminal_margin[action] == oracle) ++ties;
        }
        for (std::size_t action = 0; action < g001::option::option_count; ++action) {
            if ((item.row.valid_mask >> action) & 1u) {
                if (item.row.terminal_margin[action] == oracle) best[action] += 1.0 / ties;
                regret[action] += oracle - item.row.terminal_margin[action];
            }
        }
    }
    fs::create_directories(
        option.persistent_report.parent_path().empty()
            ? fs::path(".") : option.persistent_report.parent_path()
    );
    std::ofstream output(option.persistent_report);
    output << std::fixed << std::setprecision(6)
           << "{\n  \"schema_version\": 1,\n"
           << "  \"format\": \"MCFOPTN1\",\n"
           << "  \"row_size\": " << sizeof(g001::option::Row) << ",\n"
           << "  \"schema_hash\": \"0x" << std::hex << g001::option::schema_hash
           << std::dec << "\",\n"
           << "  \"result_kind\": \"fixed_tape_persistent_option_counterfactual_not_official_reward\",\n"
           << "  \"rows\": " << rows.size() << ",\n"
           << "  \"product_selection\": \"one causal own-stock product per row; no 9-product Cartesian product\",\n"
           << "  \"drip_formula\": \"base=ceil(remaining_quota/remaining_windows); request=min(stock,base+debt); preserve quota+debt conservation\",\n"
           << "  \"price_target\": \"max(current quote, deterministic-town-drain horizon quote)\",\n"
           << "  \"inventory_target\": \"current product stock minus exact label-only overflow\",\n"
           << "  \"pre_dump_status\": \"compiler implemented; invalid until public earliest/likely dump window is available\",\n"
           << "  \"fqi_transition\": \"next raw trade causal state; reward=delta own-opponent money margin; gamma=1\",\n"
           << "  \"terminal_values\": \"own money, opponent money, and margin stored separately; option ranking uses margin\",\n"
           << "  \"continuation_state\": \"kind,product,active,remaining_quota,remaining_windows,debt,horizon_remaining\",\n"
           << "  \"actions\": {\n";
    for (std::size_t action = 0; action < g001::option::option_count; ++action) {
        output << "    \"" << g001::option::name(static_cast<g001::option::Kind>(action))
               << "\": {\"valid\": " << valid[action]
               << ", \"best_share_overall\": " << (rows.empty() ? 0 : best[action] / rows.size())
               << ", \"best_share_when_valid\": " << (valid[action] ? best[action] / valid[action] : 0)
               << ", \"mean_terminal_regret_when_valid\": "
               << (valid[action] ? regret[action] / valid[action] : 0) << "}"
               << (action + 1 == g001::option::option_count ? "\n" : ",\n");
    }
    output << "  }\n}\n";
}

void write_report(
    const Options& option,
    const std::vector<LabelResult>& rows,
    std::size_t episode_count,
    std::size_t simulator_rollouts
) {
    std::array<std::uint64_t, g001::tree::action_count> valid{};
    std::array<double, g001::tree::action_count> best{};
    std::array<double, g001::tree::action_count> regret{};
    double baseline_regret = 0;
    for (const auto& item : rows) {
        float oracle = -std::numeric_limits<float>::infinity();
        int ties = 0;
        for (std::size_t action = 0; action < g001::tree::action_count; ++action) {
            if ((item.row.valid_action_mask >> action) & 1u) {
                ++valid[action];
                oracle = std::max(oracle, item.row.reward[action]);
            }
        }
        for (std::size_t action = 0; action < g001::tree::action_count; ++action) {
            if (((item.row.valid_action_mask >> action) & 1u) &&
                item.row.reward[action] == oracle) ++ties;
        }
        for (std::size_t action = 0; action < g001::tree::action_count; ++action) {
            if ((item.row.valid_action_mask >> action) & 1u) {
                if (item.row.reward[action] == oracle) best[action] += 1.0 / ties;
                regret[action] += oracle - item.row.reward[action];
            }
        }
        baseline_regret += oracle - item.row.reward[0];
    }
    fs::create_directories(option.report.parent_path().empty() ? fs::path(".") : option.report.parent_path());
    std::ofstream output(option.report);
    output << std::fixed << std::setprecision(6)
           << "{\n  \"schema_version\": 1,\n"
           << "  \"result_kind\": \"fixed_tape_one_turn_counterfactual_not_official_reward\",\n"
           << "  \"route\": \"" << option.route << "\",\n"
           << "  \"seeds\": " << option.seeds << ",\n"
           << "  \"baseline_episodes\": " << episode_count << ",\n"
           << "  \"label_rows\": " << rows.size() << ",\n"
           << "  \"counterfactual_rollouts\": " << simulator_rollouts << ",\n"
           << "  \"threads\": " << option.threads << ",\n"
           << "  \"split_group\": \"route_pair_plus_8_seed_block_same_family_stability_only\",\n"
           << "  \"causal_alignment\": \"action_turn=s+1; MRA prior record s equals native checkpoint before simulator action s\",\n"
           << "  \"animal_stock_mapping\": \"native GOOSE,COW,SHEEP remapped to MRA COW,SHEEP,GOOSE\",\n"
           << "  \"belief_source\": \"public_causal_conservation_interval_no_private_opponent_state\",\n"
           << "  \"capacity_label_source\": \"exact_private_clone_outcome_label_only\",\n"
           << "  \"mean_baseline_oracle_regret\": "
           << (rows.empty() ? 0 : baseline_regret / rows.size()) << ",\n"
           << "  \"actions\": {\n";
    for (std::size_t action = 0; action < g001::tree::action_count; ++action) {
        output << "    \"" << g001::tree::action_name(static_cast<MechanismAction>(action))
               << "\": {\"valid\": " << valid[action]
               << ", \"best_share_overall\": " << (rows.empty() ? 0 : best[action] / rows.size())
               << ", \"best_share_when_valid\": "
               << (valid[action] ? best[action] / valid[action] : 0)
               << ", \"mean_regret_when_valid\": "
               << (valid[action] ? regret[action] / valid[action] : 0) << "}"
               << (action + 1 == g001::tree::action_count ? "\n" : ",\n");
    }
    output << "  }\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto option = options(argc, argv);
        const auto tape = g001::market::tape::load_compiled_route(
            option.tapes, option.library, option.route
        );
        std::vector<Episode> episodes(option.seeds);
        std::atomic<std::size_t> next_episode{0};
        const auto episode_workers = std::min(option.threads, option.seeds);
        std::vector<std::thread> workers;
        for (std::size_t worker = 0; worker < episode_workers; ++worker) {
            workers.emplace_back([&] {
                while (true) {
                    const auto index = next_episode.fetch_add(1, std::memory_order_relaxed);
                    if (index >= option.seeds) break;
                    episodes[index] = baseline_episode(
                        option.seed_begin + index,
                        option.seed_begin + index + 1,
                        tape
                    );
                }
            });
        }
        for (auto& worker : workers) worker.join();

        std::vector<WorkRow> work;
        for (std::size_t episode = 0; episode < episodes.size(); ++episode) {
            for (int turn = 0; turn < int(episodes[episode].checkpoints.size()); ++turn) {
                const auto action = action_at(tape, turn);
                for (int seat = 0; seat < 2; ++seat) {
                    auto overflow = 0;
                    if (turn % 24 >= 18) {
                        overflow = exact_day_overflow_if_hold(
                            episodes[episode].checkpoints[static_cast<std::size_t>(turn)],
                            seat,
                            tape
                        );
                    }
                    bool clearance = false;
                    const auto& belief = episodes[episode].beliefs[static_cast<std::size_t>(turn)][seat];
                    for (const auto amount : belief.recent_clearance) clearance |= amount > 0;
                    const auto predump = turn >= 696 || (turn >= 600 && clearance);
                    if (has_purchase(action) || has_sell(action) || overflow > 0 || predump) {
                        work.push_back({episode, turn, seat, overflow, predump});
                    }
                }
            }
        }

        std::vector<LabelResult> rows(work.size());
        std::atomic<std::size_t> next_row{0};
        workers.clear();
        const auto row_workers = std::min(option.threads, work.size());
        for (std::size_t worker = 0; worker < row_workers; ++worker) {
            workers.emplace_back([&] {
                while (true) {
                    const auto index = next_row.fetch_add(1, std::memory_order_relaxed);
                    if (index >= work.size()) break;
                    const auto& episode = episodes[work[index].episode];
                    const auto group = static_cast<std::uint64_t>(split_group(
                        option.route, (episode.seed - option.seed_begin) / 8
                    ));
                    rows[index] = label_row(episode, work[index], tape, group);
                }
            });
        }
        for (auto& worker : workers) worker.join();

        std::size_t rollouts = 0;
        for (const auto& row : rows) {
            rollouts += std::popcount(row.row.valid_action_mask) - 1;
        }
        write_mra(option.mra, episodes);
        write_labels(option.labels, rows);
        write_report(option, rows, episodes.size(), rollouts);

        std::vector<PersistentResult> persistent_rows;
        if (!option.persistent.empty()) {
            persistent_rows.resize(work.size());
            std::atomic<std::size_t> next_persistent{0};
            workers.clear();
            for (std::size_t worker = 0; worker < row_workers; ++worker) {
                workers.emplace_back([&] {
                    while (true) {
                        const auto index = next_persistent.fetch_add(
                            1, std::memory_order_relaxed
                        );
                        if (index >= work.size()) break;
                        const auto& episode = episodes[work[index].episode];
                        const auto group = static_cast<std::uint64_t>(split_group(
                            option.route, (episode.seed - option.seed_begin) / 8
                        ));
                        persistent_rows[index] = persistent_row(
                            episode, work[index], tape, group
                        );
                    }
                });
            }
            for (auto& worker : workers) worker.join();
            std::erase_if(persistent_rows, [](const auto& item) {
                return item.row.valid_mask == 0;
            });
            write_persistent(option.persistent, persistent_rows);
            write_persistent_report(option, persistent_rows);
        }
        // This also verifies causal turn alignment and binary compatibility.
        const auto joined = g001::tree::load_joined_examples(
            option.mra.string(), option.labels.string()
        );
        if (joined.size() != rows.size()) throw std::runtime_error("MRA/MCF join count mismatch");
        std::cout << "fixed_tape_labels rows=" << rows.size()
                  << " rollouts=" << rollouts
                  << " joined=" << joined.size()
                  << " persistent_rows=" << persistent_rows.size()
                  << " seeds=" << option.seeds
                  << " threads=" << option.threads << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixed_tape_labels: " << error.what() << '\n';
        return 2;
    }
}
