// Licensed under the Apache License, Version 2.0.
#pragma once

#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace fastkag {

// Offline-only native executor for the frozen teammate route stack.  The
// submission remains Python; this class exists so counterfactual route search
// can keep the complete 719-turn inner loop in C++.
struct NativeTapeLibrary {
  std::vector<std::vector<PlayerAction>> routes;
  std::vector<PlayerAction> r5_reference;
  std::vector<PlayerAction> md_reference;
  // Five current and five legacy Moon tapes, in K320 label order.
  std::array<std::vector<PlayerAction>, 5> moon{};
  std::array<std::vector<PlayerAction>, 5> moon_legacy{};
};

struct NativeAgentState {
  struct WeedRepair {
    bool active = false;
    int start = -1;
    Action intended{};
  };
  struct RoomEvac {
    bool active = false;
    int actor = -1;
    Position target{};
    int day = -1;
  };
  struct Salvage {
    bool active = false;
    int actor = -1;
    Position target{};
    Item product = Item::NONE;
    int quantity = 0;
  };
  struct MoonRace {
    int last_step = -1;
    std::array<int, N_PRODUCTS> inventory{};
    std::array<int, N_PRODUCTS> prices{};
    std::array<int, 4> own_sells{};
    std::vector<int8_t> shops;
    std::array<std::array<double, 6>, 4> scores{};
    std::array<double, 4> evidence{};
    std::array<int, 4> horizon{1, 1, 1, 1};
    std::array<double, 6> policy_scores{};
    double policy_evidence = 0.0;
    int policy_horizon = 1;
  };

  int last_step = -1;
  std::vector<WeedRepair> weed;
  std::array<int, 4> k320_due{};
  bool r5_target = false;
  bool md_target = false;
  RoomEvac room_evac{};
  int wheat_credit = 0;
  Salvage salvage{};
  // Moon debts indexed by due step and premium product.
  std::array<std::array<int, 4>, 720> moon_debts{};
  MoonRace moon_race{};
  int moon_layout = -1;  // -1 undecided, 0 current, 1 legacy

  void reset();
};

struct NativeMatchResult {
  std::array<double, 2> rewards{};
  std::array<int32_t, 2> macro_unit_failures{};
  std::array<int32_t, 2> macro_market_failures{};
  std::array<int32_t, 2> first_macro_failure_step{-1, -1};
  std::vector<std::array<PlayerAction, 2>> trace;
};

class NativeTeammateExecutor {
 public:
  explicit NativeTeammateExecutor(NativeTapeLibrary library)
      : library_(std::move(library)) {}

  NativeMatchResult play(int route0, int route1, uint64_t seed,
                         int switch_step0 = -1, int switch_route0 = -1,
                         int switch_step1 = -1, int switch_route1 = -1,
                         bool capture_trace = false,
                         bool capture_audit = true) const;
  std::array<float, 147> features_at(int route0, int route1, uint64_t seed,
                                     int checkpoint, int player,
                                     int feature_route,
                                     int switch_step0 = -1,
                                     int switch_route0 = -1,
                                     int switch_step1 = -1,
                                     int switch_route1 = -1) const;
  PlayerAction action_at(const Simulator& env, int player, int route,
                          NativeAgentState& state) const {
    return action(env, player, route, state, nullptr);
  }
  PlayerAction action_at_with_raw_override(
      const Simulator& env, int player, int route, NativeAgentState& state,
      const PlayerAction& raw_action) const {
    return action(env, player, route, state, &raw_action);
  }
  PlayerAction action_at_with_unit_override(
      const Simulator& env, int player, int route, NativeAgentState& state,
      const std::vector<Action>& raw_units) const;
  int route_count() const { return int(library_.routes.size()); }

 private:
  NativeTapeLibrary library_;

  PlayerAction action(const Simulator& env, int player, int route,
                       NativeAgentState& state,
                       const PlayerAction* raw_override = nullptr) const;
};

}  // namespace fastkag
