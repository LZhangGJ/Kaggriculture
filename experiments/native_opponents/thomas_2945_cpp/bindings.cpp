#include "thomas_2945.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <chrono>
#include <cstdint>

#include <omp.h>

namespace py = pybind11;

namespace {

const char* op_name(fastkag::Op op) {
  static constexpr const char* names[] = {
      "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
      "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
      "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
      "CARE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT",
      "BUY_ANIMAL", "SELL"};
  const int index = int(op);
  return index >= 0 && index < int(std::size(names)) ? names[index] : "PASS";
}

py::list action_list(const fastkag::Action& action) {
  py::list result;
  result.append(op_name(action.op));
  if (action.item != fastkag::Item::NONE)
    result.append(fastkag::item_name(int(action.item)));
  if (action.quantity != 1 || action.item != fastkag::Item::NONE)
    result.append(action.quantity);
  return result;
}

py::dict action_dict(const fastkag::PlayerAction& action) {
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
  if (steps < 1 || steps > 719)
    throw std::invalid_argument("steps must be in [1,719]");
  fastkag::Simulator env({}, seed);
  thomas_2945::Opponent opponent(asset_path);
  py::list trace;
  for (int step = 0; step < steps; ++step) {
    std::array<fastkag::PlayerAction, 2> actions{};
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
  thomas_2945::Opponent opponent(asset_path);
  py::list trace;
  for (int step = 0; step < steps; ++step) {
    std::array<fastkag::PlayerAction, 2> actions{};
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

py::dict benchmark_vs_route(const std::string& asset_path, std::uint64_t seed,
                            int games, int threads, int steps,
                            int rival_route) {
  if (games < 1) throw std::invalid_argument("games must be positive");
  if (threads < 1) throw std::invalid_argument("threads must be positive");
  if (steps < 1 || steps > 719)
    throw std::invalid_argument("steps must be in [1,719]");
  const int workers = std::min(games, threads);
  std::uint64_t checksum = 0;
  std::int64_t money0 = 0, money1 = 0;
  const auto begin = std::chrono::steady_clock::now();
  {
    py::gil_scoped_release release;
#pragma omp parallel num_threads(workers) reduction(+ : checksum, money0, money1)
    {
      // Mutable wrapper state is deliberately thread-local.  Frozen assets are
      // loaded once per worker, then reused across its games via reset().
      thomas_2945::Opponent opponent(asset_path);
#pragma omp for schedule(static)
      for (int game = 0; game < games; ++game) {
        opponent.reset();
        fastkag::Simulator env({}, seed + std::uint64_t(game));
        const int seat = game & 1;
        for (int step = 0; step < steps; ++step) {
          std::array<fastkag::PlayerAction, 2> actions{};
          actions[seat] = opponent.action(env, seat);
          actions[1 - seat] = opponent.fixture_route_action(rival_route, step);
          env.step(actions);
        }
        const auto left = std::int64_t(env.farms()[0].money);
        const auto right = std::int64_t(env.farms()[1].money);
        money0 += left;
        money1 += right;
        checksum += std::uint64_t(left) * 0x9e3779b185ebca87ULL +
                    std::uint64_t(right) * 0xc2b2ae3d27d4eb4fULL +
                    std::uint64_t(game + 1);
      }
    }
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  py::dict result;
  result["games"] = games;
  result["steps"] = std::int64_t(games) * steps;
  result["threads"] = workers;
  result["seconds"] = seconds;
  result["games_per_second"] = games / seconds;
  result["steps_per_second"] = (std::int64_t(games) * steps) / seconds;
  result["money0_sum"] = money0;
  result["money1_sum"] = money1;
  result["checksum"] = checksum;
  return result;
}

}  // namespace

PYBIND11_MODULE(thomas_2945_cpp_native, module) {
  module.doc() = "Offline pure-C++ port of public Thomas 2945";
  py::class_<thomas_2945::Opponent>(module, "Opponent")
      .def(py::init<const std::string&>(), py::arg("asset_path"))
      .def("action", [](thomas_2945::Opponent& opponent,
                        const fastkag::Simulator& env, int seat) {
        if (seat < 0 || seat > 1)
          throw std::invalid_argument("seat must be 0 or 1");
        return action_dict(opponent.action(env, seat));
      }, py::arg("env"), py::arg("seat"))
      .def("route", &thomas_2945::Opponent::route, py::arg("seat"))
      .def("reset", &thomas_2945::Opponent::reset);
  module.def("play", &play, py::arg("asset_path"), py::arg("seed"),
             py::arg("seat"), py::arg("steps") = 719);
  module.def("play_vs_route", &play_vs_route, py::arg("asset_path"),
             py::arg("seed"), py::arg("seat"), py::arg("steps") = 719,
             py::arg("rival_route") = 0);
  module.def("benchmark_vs_route", &benchmark_vs_route, py::arg("asset_path"),
             py::arg("seed"), py::arg("games"), py::arg("threads") = 1,
             py::arg("steps") = 719, py::arg("rival_route") = 0);
}
