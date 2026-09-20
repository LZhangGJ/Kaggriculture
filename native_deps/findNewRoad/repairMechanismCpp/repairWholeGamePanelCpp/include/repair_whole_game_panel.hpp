#pragma once

#include "native_teammate.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace g001::repair_whole_game_panel {

enum class Scenario : std::uint8_t {
  Normal = 0,
  ForcedWeed = 1,
  ForcedCropWeed = 2,
};

[[nodiscard]] const char* scenario_name(Scenario scenario) noexcept;

// Evidence supplied by a future repair adapter. These counters are reported,
// not trusted as simulator truth. Physical receipt implementations remain
// responsible for their own certificate and state-difference verification.
struct AdapterEvidence {
  int receipt_checks{};
  int receipts_accepted{};
  int receipts_failed{};
  int moves_expected{};
  int moves_emitted{};
  int move_failures{};
  int debts_opened{};
  int debts_closed{};
  int failures{};
};

struct FinalActionContext {
  const fastkag::Simulator& observation;
  const fastkag::NativeTeammateExecutor& executor;
  int player{-1};
  int route{-1};
  int step{-1};
  const std::array<fastkag::PlayerAction, 2>& provider_joint;
  const fastkag::PlayerAction& provider_action;
  const fastkag::NativeAgentState& provider_state_before;
  const fastkag::NativeAgentState& provider_state_proposal;
};

struct FinalActionDecision {
  // The final action submitted to Simulator::step. Its unit cardinality must
  // equal the provider proposal. An invalid envelope fails closed to proposal.
  fastkag::PlayerAction final_action;
  // Optional transactional provider state. This is the future seam for a
  // native final-action commit adapter; it is applied only with a valid action.
  std::optional<fastkag::NativeAgentState> committed_provider_state;
  AdapterEvidence evidence;
};

using FinalActionCallback =
    std::function<FinalActionDecision(const FinalActionContext&)>;

struct MoneyMetrics {
  double own{};
  double opponent{};
  double margin{};
  double score{};
};

struct HandMetrics {
  int step{-1};
  bool callback_enabled{};
  bool final_action_receipt_accepted{};
  bool action_equal{};
  bool state_equal{};
  bool reward_equal{};
  int baseline_moves{};
  int repair_moves{};
  int move_action_mismatches{};
  int outstanding_debts{};
  int callback_failures{};
  int baseline_unit_failures{};
  int repair_unit_failures{};
  int baseline_market_failures{};
  int repair_market_failures{};
  AdapterEvidence adapter;
  std::array<double, 2> baseline_reward{};
  std::array<double, 2> repair_reward{};
  MoneyMetrics baseline;
  MoneyMetrics repair;
};

struct GameMetrics {
  std::uint64_t seed{};
  int seat{-1};
  Scenario scenario{Scenario::Normal};
  int steps{};
  int final_action_receipt_checks{};
  int final_action_receipts_accepted{};
  int final_action_receipts_failed{};
  int action_mismatches{};
  int state_mismatches{};
  int reward_mismatches{};
  int baseline_moves{};
  int repair_moves{};
  int move_action_mismatches{};
  int baseline_unit_failures{};
  int repair_unit_failures{};
  int baseline_market_failures{};
  int repair_market_failures{};
  int adapter_receipt_checks{};
  int adapter_receipts_accepted{};
  int adapter_receipts_failed{};
  int adapter_moves_expected{};
  int adapter_moves_emitted{};
  int adapter_move_failures{};
  int debts_opened{};
  int debts_closed{};
  int terminal_debts{};
  int callback_failures{};
  int provider_state_commits{};
  MoneyMetrics baseline_terminal;
  MoneyMetrics repair_terminal;
  std::vector<HandMetrics> hands;

  [[nodiscard]] bool exact_parity() const noexcept;
};

struct Options {
  std::string tapes;
  std::string library;
  std::string route{"G001"};
  std::uint64_t seed_begin{970017};
  int seeds{1};
  std::vector<int> seats{0, 1};
  std::vector<Scenario> scenarios{Scenario::Normal, Scenario::ForcedWeed};
};

struct Report {
  bool callback_enabled{};
  std::vector<GameMetrics> games;
  int games_with_exact_parity{};
  int action_mismatches{};
  int state_mismatches{};
  int reward_mismatches{};
  int terminal_debts{};
  int failures{};

  [[nodiscard]] bool exact_parity() const noexcept;
  [[nodiscard]] std::string json() const;
};

// With an empty callback the repair arm is disabled/default-off and must be
// globally identical to baseline. A callback sees the provider proposal and
// may replace only the focal player's final submitted action and, optionally,
// its transactionally committed provider state.
[[nodiscard]] Report evaluate(const Options& options,
                              FinalActionCallback callback = {});

}  // namespace g001::repair_whole_game_panel
