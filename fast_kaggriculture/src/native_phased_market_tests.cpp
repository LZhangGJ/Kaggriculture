#include "native_phased_market.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

bool same_action(const fastkag::Action& left, const fastkag::Action& right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool same_queue(const std::vector<fastkag::Action>& left,
                const std::vector<fastkag::Action>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!same_action(left[index], right[index])) return false;
  return true;
}

fastkag::PlayerAction pass_action() {
  fastkag::PlayerAction value;
  value.units.push_back({});
  return value;
}

std::vector<native_selective_input::SelectedPlanFrame> plan(int step,
                                                             int count = 24) {
  std::vector<native_selective_input::SelectedPlanFrame> result;
  for (int offset = 1; offset <= count; ++offset)
    result.push_back({step + offset, pass_action(), true, false});
  return result;
}

}  // namespace

int main() {
  using namespace fastkag;
  {
    Simulator env(Config{}, 8101);
    NativePhasedRuntimeState state;
    PlayerAction legacy = pass_action();
    legacy.market = {{Op::SELL, Item::WHEAT, 5},
                     {Op::BUY_ANIMAL, Item::COW, 2},
                     {Op::PASS, Item::MELON, 7}};
    const auto result = compose_native_phased_market(
        env, 0, NativeMarketArm::LegacyDefault, legacy, {}, state);
    require(same_queue(result.market, legacy.market) &&
                !result.audit.builder_called && !state.observation.has_value(),
            "arm0 is byte exact and never initializes the selective layer");
  }
  {
    Simulator env(Config{}, 8102);
    NativePhasedRuntimeState state;
    PlayerAction legacy = pass_action();
    legacy.market.push_back({Op::BUY_ANIMAL, Item::COW, 1});
    const auto result = compose_native_phased_market(
        env, 0, NativeMarketArm::SelectiveProtected, legacy, {}, state);
    require(same_queue(result.market, legacy.market) &&
                result.audit.observation_reset && result.audit.observation_staged &&
                result.audit.builder_called && !result.audit.builder_accepted &&
                result.audit.exact_legacy_fallback,
            "invalid/missing future plan stages and returns exact legacy");
  }
  {
    Simulator env(Config{}, 8103);
    NativePhasedRuntimeState state;
    auto& own = const_cast<PrivateState&>(env.privates()[0]);
    own.shed[static_cast<int>(Item::STRAWBERRY)] = 100;
    auto& public_market = const_cast<Market&>(env.market());
    for (int product = 0; product < N_PRODUCTS; ++product) {
      public_market.inventory[product] = 100;
      public_market.prices[product] = g001::market::price(
          static_cast<g001::market::Product>(product), 100);
    }
    PlayerAction legacy = pass_action();
    PlayerAction opponent = pass_action();
    // With no opponent pressure and a flat public market, immediate sale and
    // terminal inventory liquidation are economically tied.  The strict gate
    // must keep the exact legacy queue instead of inventing an improvement.
    for (int iteration = 0; iteration < 260; ++iteration) {
      auto current = compose_native_phased_market(
          env, 0, NativeMarketArm::SelectiveProtected, legacy,
          plan(env.step_count()), state);
      require(current.audit.observation_staged,
              "every valid arm1 tick stages the returned action");
      if (env.step_count() == 0)
        require(current.audit.observation_reset,
                "step0 resets before staging the exact submission");
      else
        require(current.audit.observation_observed,
                "nonzero tick observes the prior staged submission");
      require(current.audit.runtime_called,
              "valid native input reaches selective runtime on every tick");
      require(!current.audit.runtime_selected &&
                  same_queue(current.market, legacy.market),
              "terminal inventory value prevents a flat-market low-value SELL");
      PlayerAction submitted = legacy;
      submitted.market = current.market;
      env.step({submitted, opponent});
    }
  }
  {
    Simulator env(Config{}, 8104);
    NativePhasedRuntimeState state;
    PlayerAction legacy = pass_action();
    const auto result = compose_native_phased_market(
        env, 0, NativeMarketArm::Phased, legacy, plan(0), state);
    require(!result.audit.full_selected && result.audit.mode != 2,
            "native arm2 never executes FullTakeover");
  }
  std::cout << "native_phased_market_tests: PASS\n";
}
