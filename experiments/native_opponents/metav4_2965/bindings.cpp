#include "metav4_2965.hpp"

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

py::dict play(const std::string& asset_path, std::uint64_t seed, int seat, int steps) {
  if (seat < 0 || seat > 1) throw std::invalid_argument("seat must be 0 or 1");
  if (steps < 1 || steps > 719) throw std::invalid_argument("steps must be in [1,719]");
  fastkag::Simulator env({}, seed);
  metav4_2965::Opponent opponent(asset_path);
  py::list trace;
  for (int step = 0; step < steps; ++step) {
    std::array<PlayerAction, 2> actions{};
    actions[seat] = opponent.action(env, seat);
    trace.append(action_dict(actions[seat]));
    env.step(actions);
  }
  py::dict result;
  result["trace"] = std::move(trace);
  result["route"] = opponent.route(seat);
  result["rewards"] = py::make_tuple(env.farms()[0].money, env.farms()[1].money);
  return result;
}

py::dict play_vs_route(const std::string& asset_path, std::uint64_t seed,
                       int seat, int steps, int rival_route) {
  if (seat < 0 || seat > 1) throw std::invalid_argument("seat must be 0 or 1");
  if (steps < 1 || steps > 719)
    throw std::invalid_argument("steps must be in [1,719]");
  fastkag::Simulator env({}, seed);
  metav4_2965::Opponent opponent(asset_path);
  py::list trace;
  for (int step = 0; step < steps; ++step) {
    std::array<PlayerAction, 2> actions{};
    actions[seat] = opponent.action(env, seat);
    actions[1 - seat] = opponent.fixture_route_action(rival_route, step);
    trace.append(action_dict(actions[seat]));
    env.step(actions);
  }
  py::dict result;
  result["trace"] = std::move(trace);
  result["route"] = opponent.route(seat);
  result["rewards"] = py::make_tuple(env.farms()[0].money,
                                     env.farms()[1].money);
  return result;
}

py::list batch_vs_route(const std::string& asset_path,
                        std::uint64_t seed_start, int games,
                        int steps, int rival_route) {
  if (games < 1) throw std::invalid_argument("games must be positive");
  if (steps < 1 || steps > 719)
    throw std::invalid_argument("steps must be in [1,719]");
  metav4_2965::Opponent opponent(asset_path);
  py::list rewards;
  for (int game = 0; game < games; ++game) {
    fastkag::Simulator env({}, seed_start + game);
    opponent.reset();
    const int seat = game & 1;
    for (int step = 0; step < steps; ++step) {
      std::array<PlayerAction, 2> actions{};
      actions[seat] = opponent.action(env, seat);
      actions[1 - seat] = opponent.fixture_route_action(rival_route, step);
      env.step(actions);
    }
    rewards.append(py::make_tuple(env.farms()[0].money,
                                  env.farms()[1].money));
  }
  return rewards;
}

}  // namespace

PYBIND11_MODULE(metav4_2965_native, module) {
  module.doc() = "Offline native port of the public 2965 Master Hybrid opponent";
  py::class_<metav4_2965::Opponent>(module, "Opponent")
      .def(py::init<const std::string&>(), py::arg("asset_path"))
      .def("action", [](metav4_2965::Opponent& opponent,
                        const fastkag::Simulator& env, int seat) {
        if (seat < 0 || seat > 1)
          throw std::invalid_argument("seat must be 0 or 1");
        return action_dict(opponent.action(env, seat));
      }, py::arg("env"), py::arg("seat"))
      .def("route", &metav4_2965::Opponent::route, py::arg("seat"))
      .def("reset", &metav4_2965::Opponent::reset);
  module.def("play", &play, py::arg("asset_path"), py::arg("seed"),
             py::arg("seat"), py::arg("steps") = 719);
  module.def("play_vs_route", &play_vs_route, py::arg("asset_path"),
             py::arg("seed"), py::arg("seat"), py::arg("steps") = 719,
             py::arg("rival_route") = 0);
  module.def("batch_vs_route", &batch_vs_route, py::arg("asset_path"),
             py::arg("seed_start"), py::arg("games"),
             py::arg("steps") = 719, py::arg("rival_route") = 0);
}
