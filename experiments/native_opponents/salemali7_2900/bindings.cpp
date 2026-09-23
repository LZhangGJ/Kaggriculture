#include "salemali7_2900.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <string>

namespace py = pybind11;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;

namespace {

const char* op_name(Op op) {
  static constexpr const char* names[] = {
      "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
      "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
      "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
      "CARE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT",
      "BUY_ANIMAL", "SELL"};
  const int index = int(op);
  return index >= 0 && index < int(std::size(names)) ? names[index] : "PASS";
}

py::list action_list(const Action& action) {
  py::list result;
  result.append(op_name(action.op));
  if (action.item != Item::NONE) result.append(fastkag::item_name(int(action.item)));
  if (action.quantity != 1 || action.item != Item::NONE) result.append(action.quantity);
  return result;
}

py::dict action_dict(const PlayerAction& action) {
  py::dict result;
  result["farmer"] = action.units.empty() ? action_list({}) : action_list(action.units[0]);
  py::list hands;
  for (std::size_t i = 1; i < action.units.size(); ++i)
    hands.append(action_list(action.units[i]));
  result["hands"] = std::move(hands);
  py::list market;
  for (const auto& order : action.market) market.append(action_list(order));
  result["market"] = std::move(market);
  return result;
}

py::dict play(const std::string& asset_path, std::uint64_t seed, int seat,
              int steps) {
  if (seat < 0 || seat > 1) throw std::invalid_argument("seat must be 0 or 1");
  if (steps < 1 || steps > 719) throw std::invalid_argument("steps must be in [1,719]");
  fastkag::Simulator env({}, seed);
  salemali7_2900::Opponent opponent(asset_path);
  py::list trace;
  for (int step = 0; step < steps; ++step) {
    std::array<PlayerAction, 2> actions{};
    actions[seat] = opponent.action(env, seat);
    trace.append(action_dict(actions[seat]));
    env.step(actions);
  }
  py::dict result;
  result["trace"] = std::move(trace);
  result["rewards"] = py::make_tuple(env.farms()[0].money, env.farms()[1].money);
  return result;
}

py::dict play_self(const std::string& asset_path, std::uint64_t seed, int steps) {
  if (steps < 1 || steps > 719) throw std::invalid_argument("steps must be in [1,719]");
  fastkag::Simulator env({}, seed);
  salemali7_2900::Opponent opponent(asset_path);
  std::array<py::list, 2> traces;
  for (int step = 0; step < steps; ++step) {
    std::array<PlayerAction, 2> actions{
        opponent.action(env, 0), opponent.action(env, 1)};
    for (int seat = 0; seat < 2; ++seat) traces[seat].append(action_dict(actions[seat]));
    env.step(actions);
  }
  py::dict result;
  result["traces"] = py::make_tuple(std::move(traces[0]), std::move(traces[1]));
  result["rewards"] = py::make_tuple(env.farms()[0].money, env.farms()[1].money);
  return result;
}

}  // namespace

PYBIND11_MODULE(salemali7_2900_native, module) {
  module.doc() = "Offline pure-C++ port of public Salemali7 2900";
  py::class_<salemali7_2900::Opponent>(module, "Opponent")
      .def(py::init<const std::string&>(), py::arg("asset_path"))
      .def("action", [](salemali7_2900::Opponent& opponent,
                        const fastkag::Simulator& env, int seat) {
        return action_dict(opponent.action(env, seat));
      }, py::arg("env"), py::arg("seat"))
      .def("reset", &salemali7_2900::Opponent::reset);
  module.def("play", &play, py::arg("asset_path"), py::arg("seed"),
             py::arg("seat"), py::arg("steps") = 719);
  module.def("play_self", &play_self, py::arg("asset_path"), py::arg("seed"),
             py::arg("steps") = 719);
}
