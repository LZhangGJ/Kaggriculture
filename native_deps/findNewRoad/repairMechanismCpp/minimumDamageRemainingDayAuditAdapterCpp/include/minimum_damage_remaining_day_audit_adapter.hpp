#pragma once

#include "minimum_damage_scheduler_bridge.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace g001::minimum_damage_remaining_audit {

struct HandAudit {
  int step{-1};
  bool valid{};
  std::string diagnostic;
  std::uint64_t certificate_hash{};
  std::size_t selector_states{};
  std::size_t signed_debts{};
  long long elapsed_microseconds{};
  int remaining_move_tokens{};
  int signed_move_tokens{};
  int outstanding_obligations{};
  int outstanding_omissions{};
  fastkag::Action candidate_unit{};
  std::optional<fastkag::Action> candidate_market;
};

struct TerminalDebt {
  std::uint64_t obligation_id{};
  std::uint64_t effect_evidence_hash{};
  int bound_day_offset{1};
};

struct DayAudit {
  bool default_off{true};
  int hands{};
  int initial_move_tokens{};
  int observed_move_receipts{};
  int duplicate_move_receipts{};
  int failed_move_receipts{};
  int move_early_violations{};
  int outstanding_omissions{};
  int completed_obligations{};
  int outstanding_obligations{};
  std::vector<TerminalDebt> terminal_debts;
  std::size_t total_selector_states{};
  long long total_issue_microseconds{};
  long long min_hand_microseconds{};
  long long median_hand_microseconds{};
  long long max_hand_microseconds{};
  std::vector<HandAudit> hand_audits;
};

// Read-only/default-off: plan() only computes and verifies a signed candidate.
// The caller remains responsible for selecting the real provider action.
// observe_final() records concrete effects of that independently selected
// selected actor action; byte equality alone never completes a typed obligation.
class Adapter {
 public:
  Adapter(const day_start_issuer::IssueResult& issued, int player,
          std::uint64_t issuer_generation,
          std::vector<minimum_damage_bridge::ObligationPolicy> policies,
          int actor = 0,
          std::size_t maximum_states = 2'000'000);

  [[nodiscard]] HandAudit plan(const fastkag::Simulator& observation);
  [[nodiscard]] bool observe_final(const fastkag::Simulator& before,
                                   fastkag::Action final_actor_action,
                                   const fastkag::Simulator& after);
  [[nodiscard]] bool observe_passthrough(const fastkag::Simulator& before,
                                         fastkag::Action provider_action,
                                         const fastkag::Simulator& after);
  [[nodiscard]] DayAudit finish() const;

 private:
  struct Progress {
    bool observed{};
    bool completed{};
    std::uint64_t evidence_hash{};
  };

  [[nodiscard]] bool record_observation(
      const fastkag::Simulator& before, fastkag::Action action,
      const fastkag::Simulator& after, int move_source_step,
      std::uint64_t obligation_id);

  const day_start_issuer::IssueResult* issued_{};
  int player_{-1};
  int actor_{-1};
  std::uint64_t issuer_generation_{};
  std::vector<minimum_damage_bridge::ObligationPolicy> policies_;
  std::size_t maximum_states_{};
  std::map<std::uint64_t, Progress> progress_;
  std::set<int> observed_move_steps_;
  std::map<int, std::uint64_t> passed_move_evidence_;
  int duplicate_move_receipts_{};
  int failed_move_receipts_{};
  int move_early_violations_{};
  int outstanding_omissions_{};
  std::vector<HandAudit> hands_;
  std::optional<minimum_damage_bridge::SignedSlot> pending_slot_;
};

}  // namespace g001::minimum_damage_remaining_audit
