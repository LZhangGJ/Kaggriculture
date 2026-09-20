#include "repair.hpp"
#include "route_loader.hpp"
#include "weed_absorption.hpp"

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>
#include <thread>

namespace {

using fastkag::Action;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;

enum class Variant : int {
    CurrentMinLoss, SameDayPass, CrossDayPass, ExactBranch, DayTailDrop, Count
};
constexpr std::array<const char*, 5> variant_names{
    "current_min_loss", "same_day_pass_only", "cross_day_pass_only",
    "exact_branch_all", "day_tail_drop"};
constexpr std::array<const char*, 24> op_names{
    "PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE",
    "PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP",
    "BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE",
    "BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};

bool movement(Op op) { return g001::weed_absorption::is_movement(op); }

Action tape_unit(const std::vector<PlayerAction>& tape, int step, int actor) {
    if (step < 0 || step >= static_cast<int>(tape.size()) || actor < 0 ||
        actor >= static_cast<int>(tape[static_cast<std::size_t>(step)].units.size())) return {};
    return tape[static_cast<std::size_t>(step)].units[static_cast<std::size_t>(actor)];
}

Position actor_position(const fastkag::Farm& farm, int actor) {
    if (actor == 0) return farm.farmer;
    if (actor > 0 && actor <= static_cast<int>(farm.hands.size()))
        return farm.hands[static_cast<std::size_t>(actor - 1)];
    return {-1, -1};
}

const fastkag::Tile* tile_at(const fastkag::Simulator& sim, int player, Position p) {
    if (p.x < 0 || p.y < 0 || p.x >= sim.config().board_size ||
        p.y >= sim.config().board_size) return nullptr;
    return &sim.farms()[player].tiles[static_cast<std::size_t>(
        p.y * sim.config().board_size + p.x)];
}

Position moved(Position p, Op op) {
    if (op == Op::NORTH) --p.y; else if (op == Op::SOUTH) ++p.y;
    else if (op == Op::EAST) ++p.x; else if (op == Op::WEST) --p.x;
    return p;
}

struct Transaction {
    bool active{};
    int start_absolute{-1};
    int skipped_absolute{-1};
    Position start_position{};
};
struct Runtime {
    std::vector<Transaction> transactions;
    int triggers{}, pass_absorptions{}, productive_absorptions{}, unresolved{};
    int candidate_branches{}, search_calls{}, market_failures{}, overflow{};
    std::int64_t search_nanoseconds{};
    std::int64_t maximum_search_nanoseconds{};
    int swallowed_movements{}, delayed_move_sources{}, delayed_move_coordinate_mismatches{};
    int delayed_nonmove_sources{}, delayed_nonmove_coordinate_mismatches{};
    std::array<int, 24> swallowed_by_op{};
    std::array<int, 24> delayed_nonmove_attempted_by_op{};
    std::array<int, 24> delayed_nonmove_failed_by_op{};
};

Position expected_source_position(const std::vector<PlayerAction>& tape,
                                  const Transaction& transaction, int actor,
                                  int source_absolute) {
    auto position = transaction.start_position;
    for (int source = transaction.start_absolute; source < source_absolute; ++source)
        position = moved(position, tape_unit(tape, source, actor).op);
    return position;
}

std::vector<PlayerAction> fixed_scenario(const std::vector<PlayerAction>& tape,
                                         int start, int horizon,
                                         const Runtime& runtime) {
    std::vector<PlayerAction> result;
    result.reserve(static_cast<std::size_t>(horizon));
    for (int offset = 0; offset < horizon; ++offset) {
        const int step = start + offset;
        PlayerAction action;
        if (step >= 0 && step < static_cast<int>(tape.size()))
            action = tape[static_cast<std::size_t>(step)];
        for (int actor = 0; actor < static_cast<int>(runtime.transactions.size()); ++actor) {
            const auto& transaction = runtime.transactions[static_cast<std::size_t>(actor)];
            if (!transaction.active || step > transaction.skipped_absolute) continue;
            if (actor >= static_cast<int>(action.units.size()))
                action.units.resize(static_cast<std::size_t>(actor + 1));
            action.units[static_cast<std::size_t>(actor)] = tape_unit(tape, step - 1, actor);
        }
        result.push_back(std::move(action));
    }
    return result;
}

int current_min_loss_skip(const std::vector<PlayerAction>& tape, int start,
                          int actor, Position position) {
    const int last = std::min({static_cast<int>(tape.size()) - 1,
        (start / 24 + 1) * 24 - 1, start + 16});
    std::vector<g001::repair::TimedAction> window;
    for (int step = start; step <= last; ++step) {
        const auto action = tape_unit(tape, step, actor);
        g001::repair::TimedAction timed;
        timed.action = action; timed.required_position = position;
        timed.earliest_step = 0; timed.latest_step = last - start + 1;
        timed.economic_value = g001::repair::default_economic_value(action.op);
        timed.expected_fill = timed.economic_value > 0;
        window.push_back(timed);
        position = moved(position, action.op);
    }
    if (window.empty()) return -1;
    const auto plan = g001::repair::search_minimum_loss_realign(
        window, window.front().required_position, static_cast<int>(window.size()) - 1);
    if (plan.metrics.skipped_source_index < 0 || plan.metrics.movement_edits != 0) return -1;
    return plan.metrics.skipped_source_index;
}

int choose_skip(const fastkag::Simulator& sim, int player, int actor,
                const std::vector<PlayerAction>& tape,
                const std::vector<PlayerAction>& opponent_tape,
                Variant variant, Runtime& runtime) {
    const int start = sim.step_count();
    if (variant == Variant::DayTailDrop)
        return (start / sim.config().turns_per_day + 1) *
                   sim.config().turns_per_day - 1 - start;
    if (variant == Variant::CurrentMinLoss)
        return current_min_loss_skip(tape, start, actor,
                                     actor_position(sim.farms()[player], actor));
    g001::weed_absorption::SearchRequest request;
    request.state = &sim; request.player = player; request.actor = actor;
    request.absolute_start_step = start; request.own_tape = &tape;
    request.options.maximum_absorbed_source = variant == Variant::SameDayPass ? 23 : 46;
    request.options.replay_horizon = 48;
    request.options.candidate_policy = variant == Variant::SameDayPass
        ? g001::weed_absorption::CandidatePolicy::SameDayPassOnly
        : variant == Variant::CrossDayPass
            ? g001::weed_absorption::CandidatePolicy::CrossDayPassOnly
            : g001::weed_absorption::CandidatePolicy::AnyNonMovement;
    request.fixed_own_scenario = fixed_scenario(tape, start, 48, runtime);
    for (int offset = 0; offset < 48; ++offset) {
        const int step = start + offset;
        request.opponent_scenario.push_back(
            step >= 0 && step < static_cast<int>(opponent_tape.size())
                ? opponent_tape[static_cast<std::size_t>(step)] : PlayerAction{});
    }
    const auto selected = g001::weed_absorption::select_absorption(request);
    ++runtime.search_calls;
    runtime.search_nanoseconds += selected.elapsed_nanoseconds;
    runtime.maximum_search_nanoseconds = std::max(runtime.maximum_search_nanoseconds,
                                                  selected.elapsed_nanoseconds);
    runtime.candidate_branches += static_cast<int>(selected.candidates.size());
    return selected.feasible ? selected.selected_source : -1;
}

PlayerAction repaired_action(const fastkag::Simulator& sim, int player,
                             const std::vector<PlayerAction>& tape,
                             const std::vector<PlayerAction>& opponent_tape,
                             Variant variant, Runtime& runtime) {
    const int step = sim.step_count();
    PlayerAction result = step < static_cast<int>(tape.size())
        ? tape[static_cast<std::size_t>(step)] : PlayerAction{};
    if (runtime.transactions.size() < result.units.size())
        runtime.transactions.resize(result.units.size());
    for (int actor = 0; actor < static_cast<int>(runtime.transactions.size()); ++actor) {
        auto& transaction = runtime.transactions[static_cast<std::size_t>(actor)];
        if (!transaction.active) continue;
        if (step <= transaction.skipped_absolute) {
            if (actor >= static_cast<int>(result.units.size()))
                result.units.resize(static_cast<std::size_t>(actor + 1));
            result.units[static_cast<std::size_t>(actor)] = tape_unit(tape, step - 1, actor);
        } else transaction.active = false;
    }
    for (int actor = 0; actor < static_cast<int>(result.units.size()); ++actor) {
        auto& transaction = runtime.transactions[static_cast<std::size_t>(actor)];
        if (transaction.active) continue;
        const auto op = result.units[static_cast<std::size_t>(actor)].op;
        if (op != Op::PLANT && op != Op::BUILD_PASTURE) continue;
        const auto* tile = tile_at(sim, player, actor_position(sim.farms()[player], actor));
        if (!tile || tile->kind != fastkag::TileKind::WEED) continue;
        const int skip = choose_skip(sim, player, actor, tape, opponent_tape, variant, runtime);
        if (skip < 0) { ++runtime.unresolved; continue; }
        const auto swallowed = tape_unit(tape, step + skip, actor).op;
        transaction = {true, step, step + skip,
                       actor_position(sim.farms()[player], actor)};
        result.units[static_cast<std::size_t>(actor)] = Action{Op::DIG};
        ++runtime.triggers;
        ++runtime.swallowed_by_op[static_cast<int>(swallowed)];
        if (movement(swallowed)) ++runtime.swallowed_movements;
        else if (swallowed == Op::PASS)
            ++runtime.pass_absorptions;
        else ++runtime.productive_absorptions;
    }
    // Audit the exact source action that will execute this actual step. DIG at
    // the transaction start is excluded; it is inserted rather than delayed.
    for (int actor = 0; actor < static_cast<int>(runtime.transactions.size()); ++actor) {
        const auto& transaction = runtime.transactions[static_cast<std::size_t>(actor)];
        if (!transaction.active || step <= transaction.start_absolute ||
            step > transaction.skipped_absolute || actor >= static_cast<int>(result.units.size()))
            continue;
        const int source_absolute = step - 1;
        const auto operation = result.units[static_cast<std::size_t>(actor)].op;
        const auto expected = expected_source_position(tape, transaction, actor, source_absolute);
        const auto actual = actor_position(sim.farms()[player], actor);
        const bool coordinate_mismatch = expected.x != actual.x || expected.y != actual.y;
        if (movement(operation)) {
            ++runtime.delayed_move_sources;
            runtime.delayed_move_coordinate_mismatches += coordinate_mismatch;
        } else {
            ++runtime.delayed_nonmove_sources;
            runtime.delayed_nonmove_coordinate_mismatches += coordinate_mismatch;
            if (operation != Op::PASS) {
                ++runtime.delayed_nonmove_attempted_by_op[static_cast<int>(operation)];
                if (!g001::weed_absorption::projected_unit_action_succeeds(
                        sim, player, result, actor))
                    ++runtime.delayed_nonmove_failed_by_op[static_cast<int>(operation)];
            }
        }
    }
    return result;
}

struct GameResult {
    double own{}, opponent{};
    int triggers{}, pass_absorptions{}, productive_absorptions{}, unresolved{};
    int candidate_branches{}, search_calls{}, market_failures{}, overflow{};
    int crops{}, animals{}, weeds{}, standing_yield{}, held_products{};
    std::int64_t search_nanoseconds{};
    std::int64_t maximum_search_nanoseconds{};
    int swallowed_movements{}, delayed_move_sources{}, delayed_move_coordinate_mismatches{};
    int delayed_nonmove_sources{}, delayed_nonmove_coordinate_mismatches{};
    std::array<int, 24> swallowed_by_op{}, delayed_nonmove_attempted_by_op{},
                        delayed_nonmove_failed_by_op{};
};

GameResult play(std::uint64_t seed, int seat, Variant variant,
                const std::vector<PlayerAction>& focal,
                const std::vector<PlayerAction>& opponent) {
    fastkag::Simulator sim({}, seed); Runtime runtime;
    while (!sim.done()) {
        std::array<PlayerAction, 2> actions;
        actions[seat] = repaired_action(sim, seat, focal, opponent, variant, runtime);
        const int step = sim.step_count();
        actions[1 - seat] = step < static_cast<int>(opponent.size())
            ? opponent[static_cast<std::size_t>(step)] : PlayerAction{};
        sim.step(actions);
        const auto& fills = sim.last_market_fills()[seat];
        for (std::size_t slot = 0; slot < actions[seat].market.size(); ++slot) {
            const auto& order = actions[seat].market[slot];
            if (order.op != Op::BUY_SEED && order.op != Op::BUY_ANIMAL &&
                order.op != Op::HIRE && order.op != Op::BUY_LAND) continue;
            const int requested = order.op == Op::HIRE || order.op == Op::BUY_LAND
                ? int(order.quantity > 0) : std::max(0, order.quantity);
            const int filled = slot < fills.size() ? fills[slot] : 0;
            runtime.market_failures += std::max(0, requested - filled);
        }
        runtime.overflow += sim.last_end_of_day_overflow()[seat];
    }
    GameResult result;
    result.own = sim.farms()[seat].money; result.opponent = sim.farms()[1 - seat].money;
    result.triggers = runtime.triggers; result.pass_absorptions = runtime.pass_absorptions;
    result.productive_absorptions = runtime.productive_absorptions;
    result.unresolved = runtime.unresolved; result.candidate_branches = runtime.candidate_branches;
    result.search_calls = runtime.search_calls;
    result.market_failures = runtime.market_failures; result.overflow = runtime.overflow;
    result.search_nanoseconds = runtime.search_nanoseconds;
    result.maximum_search_nanoseconds = runtime.maximum_search_nanoseconds;
    result.swallowed_movements = runtime.swallowed_movements;
    result.delayed_move_sources = runtime.delayed_move_sources;
    result.delayed_move_coordinate_mismatches = runtime.delayed_move_coordinate_mismatches;
    result.delayed_nonmove_sources = runtime.delayed_nonmove_sources;
    result.delayed_nonmove_coordinate_mismatches = runtime.delayed_nonmove_coordinate_mismatches;
    result.swallowed_by_op = runtime.swallowed_by_op;
    result.delayed_nonmove_attempted_by_op = runtime.delayed_nonmove_attempted_by_op;
    result.delayed_nonmove_failed_by_op = runtime.delayed_nonmove_failed_by_op;
    const auto& farm = sim.farms()[seat];
    for (const auto& tile : farm.tiles) {
        result.crops += tile.kind == fastkag::TileKind::PLANT;
        result.animals += tile.kind == fastkag::TileKind::ANIMAL;
        result.weeds += tile.kind == fastkag::TileKind::WEED;
        if (tile.kind == fastkag::TileKind::PLANT || tile.kind == fastkag::TileKind::ANIMAL)
            result.standing_yield += tile.yield_units;
    }
    const auto& private_state = sim.privates()[seat];
    for (int item = 0; item < fastkag::N_PRODUCTS; ++item) {
        result.held_products += private_state.shed[item];
        for (const auto& inventory : private_state.inventories)
            result.held_products += inventory[item];
    }
    return result;
}

template <class T> T number(std::string_view value) {
    std::size_t used = 0; const auto parsed = std::stoull(std::string(value), &used);
    if (used != value.size()) throw std::runtime_error("invalid number");
    return static_cast<T>(parsed);
}

}  // namespace

int main(int argc, char** argv) try {
    std::string tapes = "../../../meta_agent_route_rl_submission_minimal/teammate_meta_route_submission_v1/route_actions.json.zlib";
    std::string library = "../../../meta_agent_route_rl_submission_minimal/teammate_meta_route_submission_v1/route_library.json";
    std::uint64_t begin = 3600000; std::size_t seeds = 16;
    std::string opponent_name = "G001";
    std::size_t threads = std::min<unsigned>(16, std::max(1U, std::thread::hardware_concurrency()));
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i]; auto next = [&] {
            if (++i >= argc) throw std::runtime_error("missing value"); return std::string_view(argv[i]); };
        if (arg == "--seed-begin") begin = number<std::uint64_t>(next());
        else if (arg == "--seeds") seeds = number<std::size_t>(next());
        else if (arg == "--threads") threads = number<std::size_t>(next());
        else if (arg == "--opponent") opponent_name = next();
        else if (arg == "--tapes") tapes = next(); else if (arg == "--library") library = next();
        else throw std::runtime_error("unknown option");
    }
    const auto focal = g001::repair::load_route(tapes, library, "G001");
    const auto opponent = g001::repair::load_route(
        tapes, library, opponent_name);
    constexpr std::size_t variants = static_cast<std::size_t>(Variant::Count);
    const std::size_t games = seeds * 2;
    std::array<std::vector<GameResult>, variants> results;
    for (auto& values : results) values.resize(games);
    std::atomic<std::size_t> next_task{};
    const std::size_t tasks = variants * games;
    std::vector<std::thread> workers;
    for (std::size_t worker = 0; worker < std::min(threads, tasks); ++worker)
        workers.emplace_back([&] {
            for (;;) {
                const auto task = next_task.fetch_add(1);
                if (task >= tasks) break;
                const auto variant = task / games, local = task % games;
                results[variant][local] = play(begin + local / 2, static_cast<int>(local % 2),
                    static_cast<Variant>(variant), focal, opponent);
            }
        });
    for (auto& worker : workers) worker.join();

    std::cout << std::fixed << std::setprecision(6)
              << "{\"schema\":\"weed-absorption-fresh-paired-v1\",\"seed_begin\":"
              << begin << ",\"seeds\":" << seeds << ",\"games_per_variant\":" << games
              << ",\"opponent\":\"raw_" << opponent_name
              << "\",\"variants\":{";
    for (std::size_t variant = 0; variant < variants; ++variant) {
        if (variant) std::cout << ',';
        double own = 0, margin = 0, score = 0, base_score = 0;
        double base_own_delta = 0, base_margin_delta = 0;
        std::int64_t triggers = 0, pass = 0, productive = 0, unresolved = 0;
        std::int64_t branches = 0, search_calls = 0, nanos = 0, max_nanos = 0;
        std::int64_t crops = 0, animals = 0, weeds = 0;
        std::int64_t yield = 0, held = 0, market_failures = 0, overflow = 0;
        int own_negative = 0, margin_negative = 0, lost_current_wins = 0;
        std::int64_t swallowed_moves = 0, delayed_moves = 0, delayed_move_mismatch = 0;
        std::int64_t delayed_nonmoves = 0, delayed_nonmove_mismatch = 0;
        std::array<std::int64_t, 24> swallowed_by_op{}, attempted_by_op{}, failed_by_op{};
        for (std::size_t game = 0; game < games; ++game) {
            const auto& value = results[variant][game]; const auto& base = results[0][game];
            own += value.own; margin += value.own - value.opponent;
            const double game_score = value.own > value.opponent ? 1.0
                : value.own == value.opponent ? .5 : 0.0;
            const double current_score = base.own > base.opponent ? 1.0
                : base.own == base.opponent ? .5 : 0.0;
            score += game_score; base_score += current_score;
            lost_current_wins += current_score == 1.0 && game_score < 1.0;
            base_own_delta += value.own - base.own;
            base_margin_delta += (value.own - value.opponent) - (base.own - base.opponent);
            own_negative += value.own < base.own; margin_negative +=
                value.own - value.opponent < base.own - base.opponent;
            triggers += value.triggers; pass += value.pass_absorptions;
            productive += value.productive_absorptions; unresolved += value.unresolved;
            branches += value.candidate_branches; search_calls += value.search_calls;
            nanos += value.search_nanoseconds;
            max_nanos = std::max(max_nanos, value.maximum_search_nanoseconds);
            crops += value.crops; animals += value.animals; weeds += value.weeds;
            yield += value.standing_yield; held += value.held_products;
            market_failures += value.market_failures; overflow += value.overflow;
            swallowed_moves += value.swallowed_movements;
            delayed_moves += value.delayed_move_sources;
            delayed_move_mismatch += value.delayed_move_coordinate_mismatches;
            delayed_nonmoves += value.delayed_nonmove_sources;
            delayed_nonmove_mismatch += value.delayed_nonmove_coordinate_mismatches;
            for (int operation = 0; operation < 24; ++operation) {
                swallowed_by_op[operation] += value.swallowed_by_op[operation];
                attempted_by_op[operation] += value.delayed_nonmove_attempted_by_op[operation];
                failed_by_op[operation] += value.delayed_nonmove_failed_by_op[operation];
            }
        }
        std::cout << '"' << variant_names[variant] << "\":{";
        std::cout << "\"own_mean\":" << own / games << ",\"margin_mean\":" << margin / games
                  << ",\"score\":" << score / games
                  << ",\"vs_current_score_delta\":" << (score - base_score) / games
                  << ",\"lost_current_wins\":" << lost_current_wins
                  << ",\"vs_current_own_delta\":" << base_own_delta / games
                  << ",\"vs_current_margin_delta\":" << base_margin_delta / games
                  << ",\"paired_negative_own\":" << own_negative
                  << ",\"paired_negative_margin\":" << margin_negative
                  << ",\"triggers\":" << triggers << ",\"pass_absorptions\":" << pass
                  << ",\"productive_absorptions\":" << productive
                  << ",\"unresolved\":" << unresolved << ",\"candidate_branches\":" << branches
                  << ",\"search_calls\":" << search_calls
                  << ",\"market_failures\":" << market_failures
                  << ",\"overflow\":" << overflow
                  << ",\"mean_search_microseconds_per_call\":"
                  << (search_calls ? double(nanos) / search_calls / 1000.0 : 0.0)
                  << ",\"maximum_search_microseconds\":" << double(max_nanos) / 1000.0
                  << ",\"movement_audit\":{\"swallowed_move_sources\":" << swallowed_moves
                  << ",\"delayed_move_sources\":" << delayed_moves
                  << ",\"delayed_move_coordinate_mismatches\":" << delayed_move_mismatch
                  << ",\"delayed_nonmove_sources\":" << delayed_nonmoves
                  << ",\"delayed_nonmove_coordinate_mismatches\":" << delayed_nonmove_mismatch
                  << "},\"swallowed_op_distribution\":{";
        bool first = true;
        for (int operation = 0; operation < 24; ++operation) {
            if (!swallowed_by_op[operation]) continue;
            if (!first) std::cout << ','; first = false;
            std::cout << '"' << op_names[operation] << "\":" << swallowed_by_op[operation];
        }
        std::cout << "},\"delayed_nonmove_legality\":{"; first = true;
        for (int operation = 0; operation < 24; ++operation) {
            if (!attempted_by_op[operation]) continue;
            if (!first) std::cout << ','; first = false;
            std::cout << '"' << op_names[operation] << "\":{\"attempted\":"
                      << attempted_by_op[operation] << ",\"failed\":"
                      << failed_by_op[operation] << '}';
        }
        std::cout << '}'
                  << ",\"final_production_mean\":{\"crops\":" << double(crops) / games
                  << ",\"animals\":" << double(animals) / games
                  << ",\"weeds\":" << double(weeds) / games
                  << ",\"standing_yield\":" << double(yield) / games
                  << ",\"held_products\":" << double(held) / games << "}}";
    }
    std::cout << "}}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "weed_absorption_eval: " << error.what() << '\n'; return 2;
}
