#include "joint_fixed_move_oracle.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace joint_fixed_move_oracle {

bool is_move(fastkag::Op op) noexcept {
  return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
         op == fastkag::Op::EAST || op == fastkag::Op::WEST;
}

bool action_equal(const fastkag::Action& left,
                  const fastkag::Action& right) noexcept {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

MoveCertificate certify_absolute_moves(const Tape& baseline,
                                       const Tape& candidate) {
  MoveCertificate out;
  if (baseline.size() != candidate.size()) {
    out.valid = false;
    out.reason = "tape_length_changed";
    out.mismatches = 1;
    return out;
  }
  out.checked_steps = static_cast<int>(baseline.size());
  for (std::size_t step = 0; step < baseline.size(); ++step) {
    const auto& left = baseline[step].units;
    const auto& right = candidate[step].units;
    if (left.size() != right.size()) {
      out.valid = false;
      ++out.mismatches;
      if (out.reason.empty()) out.reason = "actor_count_changed";
      continue;
    }
    out.checked_actor_slots += static_cast<int>(left.size());
    for (std::size_t actor = 0; actor < left.size(); ++actor) {
      if ((is_move(left[actor].op) || is_move(right[actor].op)) &&
          !action_equal(left[actor], right[actor])) {
        out.valid = false;
        ++out.mismatches;
        if (out.reason.empty()) {
          std::ostringstream reason;
          reason << "absolute_move_changed:step=" << step
                 << ":actor=" << actor;
          out.reason = reason.str();
        }
      }
    }
  }
  if (out.valid) out.reason = "exact_op_item_quantity_identity";
  return out;
}

void require_absolute_moves(const Tape& baseline, const Tape& candidate) {
  const auto certificate = certify_absolute_moves(baseline, candidate);
  if (!certificate.valid)
    throw std::runtime_error("fixed MOVE certificate rejected candidate: " +
                             certificate.reason);
}

namespace {

bool same_position(fastkag::Position left, fastkag::Position right) noexcept {
  return left.x == right.x && left.y == right.y;
}

int displacement_cost(fastkag::Op op) noexcept {
  switch (op) {
    case fastkag::Op::PASS: return 0;
    case fastkag::Op::DIG: return 1;
    case fastkag::Op::COLLECT_FERTILIZER: return 4;
    case fastkag::Op::CARE: return 5;
    case fastkag::Op::FERTILIZE: return 7;
    case fastkag::Op::FEED: return 8;
    case fastkag::Op::WATER: return 9;
    case fastkag::Op::DROP:
    case fastkag::Op::PICKUP: return 12;
    case fastkag::Op::HARVEST: return 16;
    case fastkag::Op::PLANT:
    case fastkag::Op::PLACE:
    case fastkag::Op::BUILD_COOP:
    case fastkag::Op::BUILD_PASTURE: return 20;
    default: return is_move(op) ? std::numeric_limits<int>::max() : 10;
  }
}

}  // namespace

std::vector<AbsoluteWeedPatch> enumerate_absolute_weed_patches(
    const Tape& baseline, const Tape& source,
    const std::vector<std::vector<fastkag::Position>>& positions,
    const WeedCollision& collision, int turns_per_day,
    int maximum_candidates) {
  std::vector<AbsoluteWeedPatch> out;
  if (baseline.size() != source.size() || positions.size() < source.size() ||
      turns_per_day <= 0 || maximum_candidates <= 0 ||
      collision.event_step < 0 ||
      collision.event_step >= static_cast<int>(source.size()) ||
      collision.actor < 0 ||
      collision.actor >= static_cast<int>(source[collision.event_step].units.size()) ||
      (collision.intended.op != fastkag::Op::PLANT &&
       collision.intended.op != fastkag::Op::BUILD_PASTURE))
    return out;

  const int day_end = std::min(
      (collision.event_step / turns_per_day + 1) * turns_per_day - 1,
      static_cast<int>(source.size()) - 1);
  auto eligible = [&](int step) {
    return step > collision.event_step && step <= day_end &&
           collision.actor < static_cast<int>(source[step].units.size()) &&
           collision.actor < static_cast<int>(positions[step].size()) &&
           same_position(positions[step][collision.actor], collision.position) &&
           !is_move(source[step].units[collision.actor].op);
  };

  struct Choice {
    int replay{};
    int water{-1};
    int cost{};
  };
  std::vector<Choice> choices;
  for (int replay = collision.event_step + 1; replay <= day_end; ++replay) {
    if (!eligible(replay)) continue;
    if (collision.intended.op == fastkag::Op::BUILD_PASTURE) {
      choices.push_back(
          {replay, -1, displacement_cost(source[replay].units[collision.actor].op)});
      continue;
    }
    for (int water = replay + 1; water <= day_end; ++water) {
      if (!eligible(water)) continue;
      choices.push_back(
          {replay, water,
           displacement_cost(source[replay].units[collision.actor].op) +
               displacement_cost(source[water].units[collision.actor].op)});
    }
  }
  std::sort(choices.begin(), choices.end(), [](const Choice& left,
                                                const Choice& right) {
    return std::tie(left.cost, left.replay, left.water) <
           std::tie(right.cost, right.replay, right.water);
  });
  if (choices.size() > static_cast<std::size_t>(maximum_candidates))
    choices.resize(static_cast<std::size_t>(maximum_candidates));

  out.reserve(choices.size());
  for (const auto& choice : choices) {
    AbsoluteWeedPatch patch;
    patch.tape = source;
    patch.event_step = collision.event_step;
    patch.replay_step = choice.replay;
    patch.water_step = choice.water;
    patch.displaced_replay =
        patch.tape[choice.replay].units[collision.actor];
    patch.tape[collision.event_step].units[collision.actor] =
        {fastkag::Op::DIG};
    patch.tape[choice.replay].units[collision.actor] = collision.intended;
    if (choice.water >= 0) {
      patch.displaced_water =
          patch.tape[choice.water].units[collision.actor];
      patch.tape[choice.water].units[collision.actor] = {fastkag::Op::WATER};
    }
    const auto certificate = certify_absolute_moves(baseline, patch.tape);
    if (!certificate.valid) {
      std::ostringstream message;
      message << "absolute weed patch rejected:" << certificate.reason
              << ":event=" << collision.event_step
              << ":actor=" << collision.actor
              << ":replay=" << choice.replay
              << ":water=" << choice.water;
      throw std::runtime_error(message.str());
    }
    out.push_back(std::move(patch));
  }
  return out;
}

fastkag::PlayerAction apply_trade_policy(
    const fastkag::PlayerAction& source, const fastkag::Simulator& simulator,
    int player, TradePolicy policy) {
  auto out = source;
  const int step = simulator.step_count();
  auto erase_sells = [&] {
    std::erase_if(out.market, [](const fastkag::Action& action) {
      return action.op == fastkag::Op::SELL;
    });
  };
  auto clear_existing = [&] {
    for (auto& action : out.market)
      if (action.op == fastkag::Op::SELL) action.quantity = 1000000;
  };
  switch (policy) {
    case TradePolicy::Baseline: break;
    case TradePolicy::HoldAll: erase_sells(); break;
    case TradePolicy::ClearExisting: clear_existing(); break;
    case TradePolicy::HoldThenClear360:
      if (step < 360) erase_sells(); else clear_existing();
      break;
    case TradePolicy::HoldThenClear480:
      if (step < 480) erase_sells(); else clear_existing();
      break;
    case TradePolicy::HoldThenClear600:
      if (step < 600) erase_sells(); else clear_existing();
      break;
    case TradePolicy::TerminalClear:
      if (step < simulator.config().episode_steps - 24) {
        erase_sells();
      } else {
        erase_sells();
        const auto& shed = simulator.privates()[static_cast<std::size_t>(player)].shed;
        for (int product = 0;
             product < fastkag::N_PRODUCTS &&
             out.market.size() <
                 static_cast<std::size_t>(simulator.config().max_market_orders);
             ++product) {
          if (shed[static_cast<std::size_t>(product)] > 0)
            out.market.push_back({fastkag::Op::SELL,
                                  static_cast<fastkag::Item>(product),
                                  shed[static_cast<std::size_t>(product)]});
        }
      }
      break;
  }
  return out;
}

const char* arm_name(Arm arm) noexcept {
  switch (arm) {
    case Arm::Baseline: return "baseline";
    case Arm::WeedRepair: return "weed-repair";
    case Arm::RepairAndDebt: return "repair+seed-animal-debt";
    case Arm::TradeOnlyClairvoyant: return "trade-only-clairvoyant";
    case Arm::CropAndTrade: return "crop+trade";
    case Arm::AllFour: return "all-four";
  }
  return "unknown";
}

const char* trade_policy_name(TradePolicy policy) noexcept {
  switch (policy) {
    case TradePolicy::Baseline: return "baseline";
    case TradePolicy::HoldAll: return "hold-all";
    case TradePolicy::ClearExisting: return "clear-existing";
    case TradePolicy::HoldThenClear360: return "hold-then-clear-360";
    case TradePolicy::HoldThenClear480: return "hold-then-clear-480";
    case TradePolicy::HoldThenClear600: return "hold-then-clear-600";
    case TradePolicy::TerminalClear: return "terminal-clear";
  }
  return "unknown";
}

}  // namespace joint_fixed_move_oracle
