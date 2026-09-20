#include "g001_typed_intent_issuer.hpp"

#include "native_general_market.hpp"
#include "native_teammate.hpp"
#include "repair_fork_evaluator.hpp"
#include "tape_runner.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ti = g001::typed_intent;
namespace po = production_obligation;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;

namespace {

struct Options {
  std::string output{"g001-typed-intent-coverage.json"};
  std::uint64_t seed{970017};
  std::size_t seeds{4};
};

Options parse(int argc, char** argv) {
  Options out;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help") {
      std::cout << "usage: " << argv[0]
                << " [--output FILE] [--seed N] [--seeds N]\n";
      std::exit(0);
    }
    if (++i >= argc) throw std::invalid_argument("missing value for " + option);
    const std::string value = argv[i];
    if (option == "--output") out.output = value;
    else if (option == "--seed") {
      const auto parsed = std::from_chars(
          value.data(), value.data() + value.size(), out.seed);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument("invalid --seed");
    } else if (option == "--seeds") {
      const auto parsed = std::from_chars(
          value.data(), value.data() + value.size(), out.seeds);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
          out.seeds == 0)
        throw std::invalid_argument("invalid --seeds");
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  return out;
}

fastkag::NativeTapeLibrary load_g001() {
  namespace tape = g001::market::tape;
  fastkag::NativeTapeLibrary out;
  out.routes.push_back(tape::load_compiled_route(
      G001_TYPED_TAPES, G001_TYPED_LIBRARY, "G001"));
  out.r5_reference = tape::load_compiled_route(
      G001_TYPED_REFS, G001_TYPED_LIBRARY, "R5");
  out.md_reference = tape::load_compiled_route(
      G001_TYPED_REFS, G001_TYPED_LIBRARY, "MD");
  constexpr std::array<std::string_view, 5> labels{
      "10C4S_3Q", "8C6S_3Q", "6C8S_3Q", "6C12S_4Q_FIRST_YARN",
      "6C12S_4Q_SECOND_YARN"};
  for (std::size_t i = 0; i < labels.size(); ++i) {
    out.moon[i] = tape::load_compiled_route(
        G001_TYPED_REFS, G001_TYPED_LIBRARY,
        "MOON_" + std::string(labels[i]));
    out.moon_legacy[i] = tape::load_compiled_route(
        G001_TYPED_REFS, G001_TYPED_LIBRARY,
        "MOON_LEGACY_" + std::string(labels[i]));
  }
  return out;
}

fastkag::Position position(const fastkag::Simulator& simulator, int player,
                           int actor) {
  const auto& farm = simulator.farms()[player];
  if (actor == 0) return farm.farmer;
  if (actor > 0 && actor <= static_cast<int>(farm.hands.size()))
    return farm.hands[static_cast<std::size_t>(actor - 1)];
  return {-1, -1};
}

fastkag::TileKind target_kind(const fastkag::Simulator& simulator, int player,
                              int actor) {
  const auto p = position(simulator, player, actor);
  const int n = simulator.config().board_size;
  if (p.x < 0 || p.y < 0 || p.x >= n || p.y >= n)
    return fastkag::TileKind::LOCKED;
  return simulator.farms()[player].tiles[
      static_cast<std::size_t>(p.y * n + p.x)].kind;
}

bool dynamic_suffix(const fastkag::NativeAgentState& state) {
  return std::any_of(state.weed.begin(), state.weed.end(),
                     [](const auto& value) { return value.active; }) ||
      std::any_of(state.experimental_realign.begin(),
                  state.experimental_realign.end(),
                  [](const auto& value) { return value.active; }) ||
      state.room_evac.active || state.salvage.active;
}

bool effect(const fastkag::Simulator& simulator, int player,
            std::span<const Action> units, int actor) {
  std::array<PlayerAction, 2> prefix;
  prefix[player].units.assign(units.begin(), units.begin() + actor);
  const auto before = simulator.preview_unit_phase(prefix);
  prefix[player].units.push_back(units[static_cast<std::size_t>(actor)]);
  const auto after = simulator.preview_unit_phase(prefix);
  return g001::repair_fork::full_unit_phase_state_fingerprint(before) !=
      g001::repair_fork::full_unit_phase_state_fingerprint(after);
}

struct Counts {
  std::uint64_t steps{}, player_steps{}, dag_feasible{}, dag_infeasible{};
  std::uint64_t dynamic_suffixes{};
  std::uint64_t water{}, water_none{}, harvest{}, harvest_none{};
  std::uint64_t weed_triggers{}, no_effect{}, weed_or_no_effect{};
  std::uint64_t wh_exact{}, wh_dag_exact{}, wh_weed_or_no_effect{};
  std::uint64_t wh_exact_on_weed_or_no_effect{};
  std::uint64_t place{}, place_target_exact{}, place_animal_exact{};
  std::uint64_t crop_emitted{}, animal_emitted{};
  std::uint64_t dag_typed_nodes{}, dag_typed_nodes_missing_position{};
  std::uint64_t dag_animal_consume_nodes{};
  std::uint64_t dag_animal_consume_nodes_missing_position{};
  std::map<std::string, std::uint64_t> proof;
  std::map<std::string, std::uint64_t> unresolved;
  std::map<std::string, std::uint64_t> place_item;
};

void count_evidence(Counts& counts, const ti::Evidence& evidence,
                    bool weed_or_no_effect) {
  ++counts.proof[ti::proof_name(evidence.proof)];
  const bool wh = evidence.source_action.op == Op::WATER ||
                  evidence.source_action.op == Op::HARVEST;
  if (wh && evidence.exact) ++counts.wh_exact;
  if (wh && evidence.proof == ti::Proof::DagTypedPosition)
    ++counts.wh_dag_exact;
  if (wh && weed_or_no_effect) ++counts.wh_weed_or_no_effect;
  if (wh && evidence.exact && weed_or_no_effect)
    ++counts.wh_exact_on_weed_or_no_effect;
  if (!evidence.exact)
    ++counts.unresolved[ti::proof_name(evidence.proof)];
}

void write_counts(std::ostream& out,
                  const std::map<std::string, std::uint64_t>& values) {
  out << '{';
  bool first = true;
  for (const auto& [name, count] : values) {
    if (!first) out << ',';
    first = false;
    out << '"' << name << "\":" << count;
  }
  out << '}';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse(argc, argv);
    const auto library = load_g001();
    fastkag::NativeTeammateExecutor executor(library);
    const auto& raw_tape = executor.route_tape(0);
    Counts counts;
    for (std::size_t seed_offset = 0; seed_offset < options.seeds;
         ++seed_offset) {
      const auto seed = options.seed + seed_offset;
      fastkag::Simulator simulator({}, seed);
      std::array<fastkag::NativeAgentState, 2> states;
      while (!simulator.done()) {
      ++counts.steps;
      std::array<PlayerAction, 2> actions;
      for (int player = 0; player < 2; ++player) {
        ++counts.player_steps;
        actions[player] = executor.action_external(
            simulator, player, 0, states[player],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, false,
            {}, nullptr);
        const bool dynamic = dynamic_suffix(states[player]);
        counts.dynamic_suffixes += dynamic;
        const int step = simulator.step_count();
        const int end = std::min({step + 24, 718,
                                  static_cast<int>(raw_tape.size()) - 1});
        std::vector<fastkag::NativeFutureUnitFrame> future;
        for (int next = step + 1; next <= end; ++next)
          future.push_back({next, raw_tape[static_cast<std::size_t>(next)].units});
        const auto dag = fastkag::compile_native_production_obligations(
            simulator, player, actions[player].units, future);
        counts.dag_feasible += dag.feasible;
        counts.dag_infeasible += !dag.feasible;
        for (const auto& node : dag.nodes) {
          if (node.kind == po::NodeKind::Consume &&
              static_cast<int>(node.item) >= po::kProducts &&
              static_cast<int>(node.item) < po::kItems) {
            ++counts.dag_animal_consume_nodes;
            if (node.position.x < 0 || node.position.y < 0)
              ++counts.dag_animal_consume_nodes_missing_position;
          }
          if (node.kind != po::NodeKind::DownstreamAction ||
              static_cast<int>(node.item) < 0 ||
              static_cast<int>(node.item) >= po::kCrops)
            continue;
          ++counts.dag_typed_nodes;
          if (node.position.x < 0 || node.position.y < 0)
            ++counts.dag_typed_nodes_missing_position;
        }
        const auto issued = ti::issue({
            &simulator, player,
            (static_cast<std::uint64_t>(seed) << 12) |
                static_cast<std::uint64_t>(player + 1),
            actions[player].units, dag.nodes, !dynamic && dag.feasible});
        counts.crop_emitted += issued.crops.size();
        counts.animal_emitted += issued.animals.size();

        for (std::size_t actor = 0; actor < actions[player].units.size(); ++actor) {
          const auto& action = actions[player].units[actor];
          const bool wh = action.op == Op::WATER || action.op == Op::HARVEST;
          const bool place = action.op == Op::PLACE;
          if (!wh && !place && action.op != Op::PLANT) continue;
          const bool weed = target_kind(simulator, player,
                                        static_cast<int>(actor)) ==
                            fastkag::TileKind::WEED;
          const bool no_effect = !effect(simulator, player,
                                         actions[player].units,
                                         static_cast<int>(actor));
          if (action.op == Op::WATER) {
            ++counts.water;
            counts.water_none += action.item == Item::NONE;
          }
          if (action.op == Op::HARVEST) {
            ++counts.harvest;
            counts.harvest_none += action.item == Item::NONE;
          }
          counts.weed_triggers += weed && (wh || action.op == Op::PLANT);
          counts.no_effect += no_effect;
          counts.weed_or_no_effect += weed || no_effect;
          counts.place += place;
          if (place) {
            const int item = static_cast<int>(action.item);
            const std::string item_name = item >= 0 && item < fastkag::N_ITEMS
                ? fastkag::item_name(item) : "NONE";
            ++counts.place_item[item_name];
            counts.place_target_exact += position(
                simulator, player, static_cast<int>(actor)).x >= 0;
          }
          const auto found = std::find_if(
              issued.evidence.begin(), issued.evidence.end(),
              [&](const auto& value) {
                return value.actor == static_cast<int>(actor);
              });
          if (found != issued.evidence.end()) {
            count_evidence(counts, *found, weed || no_effect);
            if (place && found->exact) ++counts.place_animal_exact;
          }
        }
      }
        simulator.step(actions);
      }
    }

    std::ofstream out(options.output, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create audit output");
    const auto ratio = [](std::uint64_t numerator, std::uint64_t denominator) {
      return denominator ? static_cast<double>(numerator) / denominator : 0.0;
    };
    out << std::setprecision(12)
        << "{\n  \"schema\":\"g001-typed-current-intent-coverage-v1\",\n"
        << "  \"provider\":\"NativeTeammateExecutor::action_external G001 vs G001\",\n"
        << "  \"seed_begin\":" << options.seed
        << ",\n  \"seed_count\":" << options.seeds << ",\n"
        << "  \"native_repair_options_enabled\":false,\n"
        << "  \"native_bit_opened\":false,\n"
        << "  \"win_rate_panel\":false,\n"
        << "  \"steps\":" << counts.steps
        << ",\n  \"player_steps\":" << counts.player_steps
        << ",\n  \"dag\":{\"feasible\":" << counts.dag_feasible
        << ",\"infeasible\":" << counts.dag_infeasible
        << ",\"dynamic_suffix_uncertified\":" << counts.dynamic_suffixes
        << ",\"typed_downstream_nodes\":" << counts.dag_typed_nodes
        << ",\"typed_downstream_nodes_missing_position\":"
        << counts.dag_typed_nodes_missing_position
        << ",\"animal_consume_nodes\":" << counts.dag_animal_consume_nodes
        << ",\"animal_consume_nodes_missing_position\":"
        << counts.dag_animal_consume_nodes_missing_position << "},\n"
        << "  \"raw_actions\":{\"water\":" << counts.water
        << ",\"water_item_none\":" << counts.water_none
        << ",\"harvest\":" << counts.harvest
        << ",\"harvest_item_none\":" << counts.harvest_none
        << ",\"place\":" << counts.place << ",\"place_item\":";
    write_counts(out, counts.place_item);
    out << "},\n"
        << "  \"triggers\":{\"weed\":" << counts.weed_triggers
        << ",\"no_effect\":" << counts.no_effect
        << ",\"weed_or_no_effect\":" << counts.weed_or_no_effect << "},\n"
        << "  \"coverage\":{\"water_harvest_exact\":" << counts.wh_exact
        << ",\"water_harvest_total\":" << counts.water + counts.harvest
        << ",\"water_harvest_exact_ratio\":"
        << ratio(counts.wh_exact, counts.water + counts.harvest)
        << ",\"water_harvest_dag_exact\":" << counts.wh_dag_exact
        << ",\"water_harvest_dag_exact_ratio\":"
        << ratio(counts.wh_dag_exact, counts.water + counts.harvest)
        << ",\"weed_or_no_effect_water_harvest_total\":"
        << counts.wh_weed_or_no_effect
        << ",\"weed_or_no_effect_water_harvest_exact\":"
        << counts.wh_exact_on_weed_or_no_effect
        << ",\"weed_or_no_effect_water_harvest_exact_ratio\":"
        << ratio(counts.wh_exact_on_weed_or_no_effect,
                 counts.wh_weed_or_no_effect)
        << ",\"animal_place_target_dag_exact\":0"
        << ",\"animal_place_target_dag_exact_ratio\":0"
        << ",\"animal_place_target_exact\":" << counts.place_target_exact
        << ",\"animal_place_total\":" << counts.place
        << ",\"animal_place_target_exact_ratio\":"
        << ratio(counts.place_target_exact, counts.place)
        << ",\"animal_place_identity_exact\":" << counts.place_animal_exact
        << ",\"animal_place_identity_exact_ratio\":"
        << ratio(counts.place_animal_exact, counts.place)
        << ",\"crop_owner_payloads\":" << counts.crop_emitted
        << ",\"animal_owner_payloads\":" << counts.animal_emitted << "},\n"
        << "  \"proof_categories\":";
    write_counts(out, counts.proof);
    out << ",\n  \"unresolved_categories\":";
    write_counts(out, counts.unresolved);
    out << ",\n  \"serving_interface\":{\"input\":\"final current units + live observation + existing DAG + suffix authority bool\",\"crop_sink\":\"TransactionalCropRepairOwner::prepare(context, span<TypedCropObligation>)\",\"animal_sink\":\"TransactionalAnimalRepairOwner::prepare(context, span<TypedAnimalObligation>)\",\"fail_closed\":true}\n}\n";
    std::cout << "steps=" << counts.steps << " wh="
              << counts.water + counts.harvest << " exact=" << counts.wh_exact
              << " place=" << counts.place << " place_exact="
              << counts.place_target_exact << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "typed-intent audit failed: " << error.what() << '\n';
    return 1;
  }
}
