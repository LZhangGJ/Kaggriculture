#include "tape_runner.hpp"

#include "simulator.hpp"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace g001::market::tape {
namespace {

using Tape = std::vector<fastkag::PlayerAction>;

class JsonCursor {
public:
    explicit JsonCursor(std::string_view text) : text_(text) {}
    void whitespace() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\n' ||
                text_[position_] == '\r' || text_[position_] == '\t')) ++position_;
    }
    char peek() { whitespace(); return position_ < text_.size() ? text_[position_] : '\0'; }
    bool take(char value) {
        whitespace();
        if (position_ < text_.size() && text_[position_] == value) {
            ++position_;
            return true;
        }
        return false;
    }
    void expect(char value) {
        if (!take(value)) throw std::runtime_error("invalid route JSON");
    }
    std::string string() {
        whitespace();
        expect('"');
        std::string result;
        while (position_ < text_.size()) {
            const char value = text_[position_++];
            if (value == '"') return result;
            if (value != '\\') { result.push_back(value); continue; }
            if (position_ >= text_.size()) break;
            const char escaped = text_[position_++];
            switch (escaped) {
                case '"': case '\\': case '/': result.push_back(escaped); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default: throw std::runtime_error("unsupported JSON escape");
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }
    int integer() {
        whitespace();
        const auto* begin = text_.data() + position_;
        const auto* end = text_.data() + text_.size();
        int value = 0;
        const auto parsed = std::from_chars(begin, end, value);
        if (parsed.ec != std::errc{}) throw std::runtime_error("expected JSON integer");
        position_ = static_cast<std::size_t>(parsed.ptr - text_.data());
        return value;
    }
    void skip() {
        whitespace();
        const auto value = peek();
        if (value == '"') { (void)string(); return; }
        if (value == '{') {
            expect('{');
            if (take('}')) return;
            do { (void)string(); expect(':'); skip(); } while (take(','));
            expect('}'); return;
        }
        if (value == '[') {
            expect('[');
            if (take(']')) return;
            do { skip(); } while (take(','));
            expect(']'); return;
        }
        while (position_ < text_.size() &&
               text_[position_] != ',' && text_[position_] != ']' &&
               text_[position_] != '}' && text_[position_] != ' ' &&
               text_[position_] != '\n' && text_[position_] != '\r' &&
               text_[position_] != '\t') ++position_;
    }
private:
    std::string_view text_;
    std::size_t position_ = 0;
};

[[nodiscard]] fastkag::Op parse_op(std::string_view value) {
    static const std::unordered_map<std::string_view, fastkag::Op> values{
        {"PASS",fastkag::Op::PASS},{"NORTH",fastkag::Op::NORTH},
        {"SOUTH",fastkag::Op::SOUTH},{"EAST",fastkag::Op::EAST},
        {"WEST",fastkag::Op::WEST},{"DROP",fastkag::Op::DROP},
        {"PICKUP",fastkag::Op::PICKUP},{"PLACE",fastkag::Op::PLACE},
        {"PLANT",fastkag::Op::PLANT},{"WATER",fastkag::Op::WATER},
        {"HARVEST",fastkag::Op::HARVEST},{"FERTILIZE",fastkag::Op::FERTILIZE},
        {"DIG",fastkag::Op::DIG},{"BUILD_COOP",fastkag::Op::BUILD_COOP},
        {"BUILD_PASTURE",fastkag::Op::BUILD_PASTURE},{"FEED",fastkag::Op::FEED},
        {"COLLECT_FERTILIZER",fastkag::Op::COLLECT_FERTILIZER},
        {"CARE",fastkag::Op::CARE},{"HIRE",fastkag::Op::HIRE},
        {"BUY_LAND",fastkag::Op::BUY_LAND},{"BUY_SEED",fastkag::Op::BUY_SEED},
        {"BUY_PRODUCT",fastkag::Op::BUY_PRODUCT},
        {"BUY_ANIMAL",fastkag::Op::BUY_ANIMAL},{"SELL",fastkag::Op::SELL},
    };
    const auto found = values.find(value);
    if (found == values.end()) throw std::runtime_error("unknown route operation");
    return found->second;
}

[[nodiscard]] fastkag::Item parse_item(std::string_view value) {
    static const std::unordered_map<std::string_view, fastkag::Item> values{
        {"WHEAT",fastkag::Item::WHEAT},{"CARROT",fastkag::Item::CARROT},
        {"TOMATO",fastkag::Item::TOMATO},{"STRAWBERRY",fastkag::Item::STRAWBERRY},
        {"MELON",fastkag::Item::MELON},{"EGG",fastkag::Item::EGG},
        {"MILK",fastkag::Item::MILK},{"WOOL",fastkag::Item::WOOL},
        {"FERTILIZER",fastkag::Item::FERTILIZER},{"GOOSE",fastkag::Item::GOOSE},
        {"COW",fastkag::Item::COW},{"SHEEP",fastkag::Item::SHEEP},
    };
    const auto found = values.find(value);
    if (found == values.end()) throw std::runtime_error("unknown route item");
    return found->second;
}

[[nodiscard]] fastkag::Action parse_action(JsonCursor& json) {
    json.expect('[');
    fastkag::Action result;
    result.op = parse_op(json.string());
    if (json.take(',')) {
        result.item = parse_item(json.string());
        if (json.take(',')) result.quantity = json.integer();
    }
    while (json.take(',')) json.skip();
    json.expect(']');
    return result;
}

[[nodiscard]] std::vector<fastkag::Action> parse_actions(JsonCursor& json) {
    std::vector<fastkag::Action> result;
    json.expect('[');
    if (json.take(']')) return result;
    do { result.push_back(parse_action(json)); } while (json.take(','));
    json.expect(']');
    return result;
}

[[nodiscard]] fastkag::PlayerAction parse_turn(JsonCursor& json) {
    fastkag::PlayerAction result;
    fastkag::Action farmer;
    bool has_farmer = false;
    json.expect('{');
    if (!json.take('}')) {
        do {
            const auto key = json.string();
            json.expect(':');
            if (key == "farmer") { farmer = parse_action(json); has_farmer = true; }
            else if (key == "hands") result.units = parse_actions(json);
            else if (key == "market") result.market = parse_actions(json);
            else json.skip();
        } while (json.take(','));
        json.expect('}');
    }
    result.units.insert(result.units.begin(), has_farmer ? farmer : fastkag::Action{});
    return result;
}

[[nodiscard]] Tape parse_tape(JsonCursor& json) {
    Tape result;
    json.expect('[');
    if (json.take(']')) return result;
    do { result.push_back(parse_turn(json)); } while (json.take(','));
    json.expect(']');
    return result;
}

[[nodiscard]] std::string inflate_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open compressed route tapes: " + path);
    std::vector<unsigned char> compressed(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()
    );
    z_stream stream{};
    stream.next_in = compressed.data();
    stream.avail_in = static_cast<uInt>(compressed.size());
    if (inflateInit(&stream) != Z_OK) throw std::runtime_error("zlib init failed");
    std::string output;
    std::array<char, 1 << 16> buffer{};
    int status = Z_OK;
    while (status == Z_OK) {
        stream.next_out = reinterpret_cast<Bytef*>(buffer.data());
        stream.avail_out = static_cast<uInt>(buffer.size());
        status = inflate(&stream, Z_NO_FLUSH);
        output.append(buffer.data(), buffer.size() - stream.avail_out);
    }
    inflateEnd(&stream);
    if (status != Z_STREAM_END) throw std::runtime_error("route tape decompression failed");
    return output;
}

[[nodiscard]] Tape load_tape(const std::string& path, const std::string& route_id) {
    const auto payload = inflate_file(path);
    JsonCursor json(payload);
    json.expect('{');
    if (!json.take('}')) {
        do {
            const auto key = json.string();
            json.expect(':');
            if (key == route_id) return parse_tape(json);
            json.skip();
        } while (json.take(','));
        json.expect('}');
    }
    throw std::runtime_error("route id not found: " + route_id);
}

[[nodiscard]] std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open route library: " + path);
    return std::string(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()
    );
}

[[nodiscard]] std::string resolve_route(
    const std::string& library_path, const std::string& requested
) {
    if (requested.size() < 2 || requested.front() != 'G') return requested;
    const auto source = read_text(library_path);
    const std::regex pattern(
        "\\\"family\\\"\\s*:\\s*\\\"" + requested +
        "\\\"[^}]*\\\"route_id\\\"\\s*:\\s*\\\"([^\\\"]+)\\\""
    );
    std::smatch match;
    if (!std::regex_search(source, match, pattern)) {
        throw std::runtime_error("route family not found: " + requested);
    }
    return match[1].str();
}

[[nodiscard]] fastkag::PlayerAction tape_action(const Tape& tape, int step) {
    if (tape.empty()) return {};
    return tape[static_cast<std::size_t>(std::min(step, int(tape.size()) - 1))];
}

[[nodiscard]] int fibonacci(int index) {
    int a = 1, b = 1;
    while (index-- > 0) { const auto next = a + b; a = b; b = next; }
    return a;
}

[[nodiscard]] std::int64_t current_purchase_cost(
    const fastkag::Simulator& simulator,
    int player,
    const fastkag::PlayerAction& action
) {
    constexpr std::array<int, 5> seed_cost{10, 20, 50, 100, 80};
    constexpr std::array<int, 3> animal_cost{300, 400, 500};
    constexpr std::array<int, 3> land_cost{1000, 2000, 4000};
    auto hires = simulator.farms()[player].hires_today;
    auto land = __builtin_popcount(unsigned(simulator.farms()[player].unlocked_mask)) - 1;
    auto inventory = simulator.market().inventory;
    std::int64_t result = 0;
    for (const auto& order : action.market) {
        const auto quantity = std::max(0, order.quantity);
        const auto item = static_cast<int>(order.item);
        if (order.op == fastkag::Op::HIRE) {
            result += fibonacci(hires++);
        } else if (order.op == fastkag::Op::BUY_LAND && land < 3) {
            result += land_cost[static_cast<std::size_t>(land++)];
        } else if (order.op == fastkag::Op::BUY_SEED && item >= 0 && item < 5) {
            result += std::int64_t(quantity) * seed_cost[static_cast<std::size_t>(item)];
        } else if (order.op == fastkag::Op::BUY_ANIMAL && item >= 9 && item < 12) {
            result += std::int64_t(quantity) * animal_cost[static_cast<std::size_t>(item - 9)];
        } else if (order.op == fastkag::Op::BUY_PRODUCT && (item == 0 || item == 8)) {
            for (int unit = 0; unit < quantity; ++unit) {
                --inventory[static_cast<std::size_t>(item)];
                result += price(static_cast<Product>(item), inventory[static_cast<std::size_t>(item)]);
            }
        }
    }
    return result;
}

[[nodiscard]] std::vector<CashCommitment> future_route_commitments(
    const fastkag::Simulator& simulator,
    int player,
    const Tape& tape
) {
    constexpr std::array<int, 5> seed_cost{10, 20, 50, 100, 80};
    constexpr std::array<int, 3> animal_cost{300, 400, 500};
    constexpr std::array<int, 3> land_cost{1000, 2000, 4000};
    std::vector<CashCommitment> result;
    auto land = __builtin_popcount(unsigned(simulator.farms()[player].unlocked_mask)) - 1;
    auto current_day = simulator.day();
    auto hires = simulator.farms()[player].hires_today;
    for (int step = simulator.step_count(); step < int(tape.size()); ++step) {
        const auto day = step / simulator.config().turns_per_day;
        if (day != current_day) { current_day = day; hires = 0; }
        std::int64_t amount = 0;
        for (const auto& order : tape[static_cast<std::size_t>(step)].market) {
            const auto quantity = std::max(0, order.quantity);
            const auto item = static_cast<int>(order.item);
            if (order.op == fastkag::Op::HIRE) {
                amount += simulator.config().farm_hand_cost_mult * fibonacci(hires++);
            } else if (order.op == fastkag::Op::BUY_LAND && land < 3) {
                amount += land_cost[static_cast<std::size_t>(land++)];
            } else if (order.op == fastkag::Op::BUY_SEED && item >= 0 && item < 5) {
                amount += std::int64_t(quantity) * seed_cost[static_cast<std::size_t>(item)];
            } else if (order.op == fastkag::Op::BUY_ANIMAL && item >= 9 && item < 12) {
                amount += std::int64_t(quantity) * animal_cost[static_cast<std::size_t>(item - 9)];
            } else if (order.op == fastkag::Op::BUY_PRODUCT &&
                       (item == 0 || item == 8)) {
                amount += std::int64_t(quantity) * price(
                    static_cast<Product>(item),
                    simulator.market().inventory[static_cast<std::size_t>(item)] - 1
                );
            }
        }
        if (amount > 0) result.push_back({step, amount, true});
    }
    return result;
}

void fill_town(const fastkag::Simulator& simulator, TownState& town) {
    for (const auto shop : simulator.shops()) {
        auto add = [&](int product, int quantity = 1) {
            town.shop_demand_per_tick[static_cast<std::size_t>(product)] += quantity;
        };
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
    town.shop_interval = simulator.config().town_shop_sell_interval;
    town.center_interval = simulator.config().town_center_sell_interval;
}

[[nodiscard]] PlanningState planning_state(
    const fastkag::Simulator& simulator,
    int player,
    const fastkag::PlayerAction& original,
    const Tape& tape,
    MechanismMask mechanisms
) {
    PlanningState result;
    result.step = simulator.step_count();
    result.last_action_step = simulator.config().episode_steps - 2;
    result.shed_capacity = simulator.config().shed_capacity;
    result.money = static_cast<std::int64_t>(std::floor(simulator.farms()[player].money));
    result.mechanisms = mechanisms;
    const auto& own = simulator.privates()[player];
    const auto& rival = simulator.privates()[1 - player];
    for (std::size_t i = 0; i < product_count; ++i) {
        result.market_inventory[i] = simulator.market().inventory[i];
        result.own_shed[i] = own.shed[i];
        for (const auto& inventory : own.inventories) result.own_carried[i] += inventory[i];
        // Explicitly an exact-offline belief for this first runner. The CLI
        // reports this mode so results cannot be mistaken for deployable play.
        result.belief.rival_stock[i] = rival.shed[i];
        result.belief.rival_lower[i] = rival.shed[i];
        result.belief.rival_upper[i] = rival.shed[i];
        for (const auto& inventory : rival.inventories) {
            result.belief.rival_stock[i] += inventory[i];
            result.belief.rival_lower[i] += inventory[i];
            result.belief.rival_upper[i] += inventory[i];
        }
        for (const auto& order : original.market) {
            if (order.op == fastkag::Op::SELL && static_cast<int>(order.item) == int(i)) {
                result.baseline_sale[i] += std::max(0, order.quantity);
            }
        }
    }
    fill_town(simulator, result.town);
    result.purchase_cash_required = current_purchase_cost(simulator, player, original);
    result.cash_commitments = future_route_commitments(simulator, player, tape);
    if (!result.cash_commitments.empty() &&
        result.cash_commitments.front().step == result.step) {
        // Same-turn quote/order effects can be evaluated exactly and replace
        // the nominal future-tape estimate.
        result.cash_commitments.front().amount = result.purchase_cash_required;
    }
    return result;
}

[[nodiscard]] fastkag::PlayerAction variant_action(
    const fastkag::Simulator& simulator,
    int player,
    fastkag::PlayerAction original,
    const Tape& tape,
    MechanismMask mechanisms
) {
    const auto state = planning_state(simulator, player, original, tape, mechanisms);
    std::erase_if(original.market, [](const auto& order) {
        return order.op == fastkag::Op::SELL;
    });
    const auto plan = plan_sales(state);
    std::vector<fastkag::Action> sells;
    for (std::size_t i = 0; i < product_count; ++i) {
        if (plan.quantity[i] > 0) {
            sells.push_back({
                fastkag::Op::SELL,
                static_cast<fastkag::Item>(i),
                plan.quantity[i]
            });
        }
    }
    // Sales precede unchanged production purchases so that same-turn funding
    // works exactly like the original simulator's ordered market queue.
    original.market.insert(original.market.begin(), sells.begin(), sells.end());
    return original;
}

[[nodiscard]] bool same_units(
    const std::vector<fastkag::Action>& left,
    const std::vector<fastkag::Action>& right
) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i].op != right[i].op || left[i].item != right[i].item ||
            left[i].quantity != right[i].quantity) return false;
    }
    return true;
}

void collect_step_metrics(
    const fastkag::Simulator& simulator,
    const std::array<fastkag::PlayerAction, 2>& actions,
    RunMetrics& metrics
) {
    for (int player = 0; player < 2; ++player) {
        const auto& fills = simulator.last_market_fills()[player];
        const auto& shortfalls = simulator.last_market_cash_shortfalls()[player];
        for (std::size_t order = 0; order < actions[player].market.size(); ++order) {
            const auto& action = actions[player].market[order];
            const auto fill = order < fills.size() ? fills[order] : 0;
            const auto item = static_cast<int>(action.item);
            if (item >= 0 && item < int(product_count)) {
                if (action.op == fastkag::Op::SELL) {
                    metrics.trades[player].sold[static_cast<std::size_t>(item)] += fill;
                } else if (action.op == fastkag::Op::BUY_PRODUCT) {
                    metrics.trades[player].bought[static_cast<std::size_t>(item)] += fill;
                }
            }
            if (action.op != fastkag::Op::SELL && order < shortfalls.size() &&
                shortfalls[order] > 0.0) {
                ++metrics.purchase_failures[player];
            }
        }
        metrics.overflow_units[player] += simulator.last_end_of_day_overflow()[player];
    }
}

[[nodiscard]] RunMetrics run_game(
    std::uint64_t seed,
    const Tape& own_tape,
    const Tape& opponent_tape,
    int variant_player,
    MechanismMask mechanisms
) {
    fastkag::Simulator simulator({}, seed);
    RunMetrics metrics;
    while (!simulator.done()) {
        std::array<fastkag::PlayerAction, 2> actions{
            tape_action(own_tape, simulator.step_count()),
            tape_action(opponent_tape, simulator.step_count())
        };
        if (variant_player >= 0) {
            const auto original_units = actions[variant_player].units;
            actions[variant_player] = variant_action(
                simulator, variant_player, std::move(actions[variant_player]),
                variant_player == 0 ? own_tape : opponent_tape,
                mechanisms
            );
            if (!same_units(original_units, actions[variant_player].units)) {
                ++metrics.unit_action_mutations[variant_player];
            }
        }
        for (int player = 0; player < 2; ++player) {
            metrics.compiled_unit_actions[player] += actions[player].units.size();
        }
        simulator.step(actions);
        collect_step_metrics(simulator, actions, metrics);
    }
    metrics.reward[0] = simulator.farms()[0].money;
    metrics.reward[1] = simulator.farms()[1].money;
    return metrics;
}

void add_trades(ProductTrades& destination, const ProductTrades& source) {
    for (std::size_t i = 0; i < product_count; ++i) {
        destination.sold[i] += source.sold[i];
        destination.bought[i] += source.bought[i];
    }
}

}  // namespace

PairedSummary run_paired(const RunnerOptions& options) {
    if (options.seed_count == 0) return {};
    const auto route_id = resolve_route(options.route_library_path, options.route);
    const auto opponent_request = options.opponent_route.empty()
        ? options.route : options.opponent_route;
    const auto opponent_id = resolve_route(options.route_library_path, opponent_request);
    const auto own_tape = load_tape(options.compressed_tapes_path, route_id);
    const auto opponent_tape = route_id == opponent_id
        ? own_tape : load_tape(options.compressed_tapes_path, opponent_id);

    struct SeedResult { RunMetrics baseline0, baseline1, seat0, seat1; };
    std::vector<SeedResult> results(static_cast<std::size_t>(options.seed_count));
    const auto task_count = static_cast<std::size_t>(options.seed_count) * 4;
    std::atomic<std::size_t> next{0};
    auto threads = options.threads == 0 ? std::thread::hardware_concurrency() : options.threads;
    threads = std::clamp<std::size_t>(threads, 1, task_count);
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (std::size_t worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&] {
            while (true) {
                const auto task = next.fetch_add(1, std::memory_order_relaxed);
                if (task >= task_count) break;
                const auto seed_index = task / 4;
                const auto kind = task % 4;
                const auto seed = options.seed_begin + seed_index;
                if (kind == 0) results[seed_index].baseline0 = run_game(
                    seed, own_tape, opponent_tape, -1, options.mechanisms
                );
                else if (kind == 1) results[seed_index].baseline1 = run_game(
                    seed, opponent_tape, own_tape, -1, options.mechanisms
                );
                else if (kind == 2) results[seed_index].seat0 = run_game(
                    seed, own_tape, opponent_tape, 0, options.mechanisms
                );
                else results[seed_index].seat1 = run_game(
                    seed, opponent_tape, own_tape, 1, options.mechanisms
                );
            }
        });
    }
    for (auto& worker : workers) worker.join();

    PairedSummary summary;
    summary.seeds = options.seed_count;
    summary.games = options.seed_count * 2;
    for (const auto& result : results) {
        for (int seat = 0; seat < 2; ++seat) {
            const auto& variant = seat == 0 ? result.seat0 : result.seat1;
            const auto& baseline = seat == 0 ? result.baseline0 : result.baseline1;
            const auto baseline_reward = baseline.reward[seat];
            const auto variant_reward = variant.reward[seat];
            const auto opponent_reward = variant.reward[1 - seat];
            summary.baseline_reward += baseline_reward;
            summary.variant_reward += variant_reward;
            summary.opponent_reward += opponent_reward;
            summary.paired_reward_delta += variant_reward - baseline_reward;
            summary.wins += variant_reward > opponent_reward;
            summary.losses += variant_reward < opponent_reward;
            summary.ties += variant_reward == opponent_reward;
            summary.baseline_purchase_failures += baseline.purchase_failures[seat];
            summary.variant_purchase_failures += variant.purchase_failures[seat];
            summary.baseline_overflow_units += baseline.overflow_units[seat];
            summary.variant_overflow_units += variant.overflow_units[seat];
            add_trades(summary.baseline_trades, baseline.trades[seat]);
            add_trades(summary.variant_trades, variant.trades[seat]);
            summary.baseline_compiled_unit_actions += baseline.compiled_unit_actions[seat];
            summary.variant_compiled_unit_actions += variant.compiled_unit_actions[seat];
            summary.variant_unit_action_mutations += variant.unit_action_mutations[seat];
        }
    }
    const auto games = static_cast<double>(summary.games);
    summary.baseline_reward /= games;
    summary.variant_reward /= games;
    summary.opponent_reward /= games;
    summary.paired_reward_delta /= games;
    return summary;
}

CompiledTape load_compiled_route(
    const std::string& compressed_tapes_path,
    const std::string& route_library_path,
    const std::string& family_or_route_id
) {
    return load_tape(
        compressed_tapes_path,
        resolve_route(route_library_path, family_or_route_id)
    );
}

std::string summary_json(const PairedSummary& summary, const RunnerOptions& options) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6);
    output << "{\n  \"schema_version\": 1,\n"
           << "  \"route\": \"" << options.route << "\",\n"
           << "  \"opponent_route\": \""
           << (options.opponent_route.empty() ? options.route : options.opponent_route)
           << "\",\n  \"belief_mode\": \"exact_offline_upper_bound\",\n"
           << "  \"future_cash_commitments\": \"full_tape_nominal_with_exact_current_step\",\n"
           << "  \"future_production_model\": \"not_available; current carried stock is included\",\n"
           << "  \"result_kind\": \"fixed_tape_counterfactual_not_official_win_rate\",\n"
           << "  \"seed_begin\": " << options.seed_begin << ",\n"
           << "  \"seeds\": " << summary.seeds << ",\n"
           << "  \"paired_games\": " << summary.games << ",\n"
           << "  \"simulator_episodes\": " << summary.seeds * 4 << ",\n"
           << "  \"threads\": " << options.threads << ",\n"
           << "  \"mechanism_mask\": " << options.mechanisms << ",\n"
           << "  \"baseline_mean_reward\": " << summary.baseline_reward << ",\n"
           << "  \"variant_mean_reward\": " << summary.variant_reward << ",\n"
           << "  \"variant_opponent_mean_reward\": " << summary.opponent_reward << ",\n"
           << "  \"paired_mean_reward_delta\": " << summary.paired_reward_delta << ",\n"
           << "  \"wins\": " << summary.wins << ",\n"
           << "  \"losses\": " << summary.losses << ",\n"
           << "  \"ties\": " << summary.ties << ",\n"
           << "  \"baseline_purchase_failures\": " << summary.baseline_purchase_failures << ",\n"
           << "  \"variant_purchase_failures\": " << summary.variant_purchase_failures << ",\n"
           << "  \"baseline_overflow_units\": " << summary.baseline_overflow_units << ",\n"
           << "  \"variant_overflow_units\": " << summary.variant_overflow_units << ",\n";
    output << "  \"baseline_compiled_unit_actions\": "
           << summary.baseline_compiled_unit_actions << ",\n"
           << "  \"variant_compiled_unit_actions\": "
           << summary.variant_compiled_unit_actions << ",\n"
           << "  \"variant_unit_action_mutations\": "
           << summary.variant_unit_action_mutations << ",\n";
    constexpr std::array<const char*, product_count> names{
        "WHEAT","CARROT","TOMATO","STRAWBERRY","MELON",
        "EGG","MILK","WOOL","FERTILIZER"
    };
    auto trades = [&](std::string_view key, const ProductTrades& values) {
        output << "  \"" << key << "\": {";
        for (std::size_t i = 0; i < product_count; ++i) {
            if (i) output << ',';
            output << "\n    \"" << names[i] << "\": {\"sold\": "
                   << values.sold[i] << ", \"bought\": " << values.bought[i] << '}';
        }
        output << "\n  }";
    };
    trades("baseline_product_fills", summary.baseline_trades);
    output << ",\n";
    trades("variant_product_fills", summary.variant_trades);
    output << "\n}\n";
    return output.str();
}

}  // namespace g001::market::tape
