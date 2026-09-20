#pragma once

#include "modular_repair_adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace g001::repair_audit::two_phase {

enum class StagePhase : std::uint8_t { Unit, Market };

struct LifecycleStage {
  modular_agent_core::IntentRef intent{};
  fastkag::Action action{};
  StagePhase phase{StagePhase::Unit};
  bool consumes_source{true};
  bool first_yield{};
  std::string provenance;
};

struct LifecycleContract {
  std::uint64_t transaction_id{};
  failure_debt::LifecycleKind kind{failure_debt::LifecycleKind::Crop};
  std::vector<LifecycleStage> stages;
};

struct CommitAudit {
  int acquisitions{};
  int pickups{};
  int places{};
  int plants{};
  int waters{};
  int feeds{};
  int cares{};
  int harvest_effects{};
  int overlay_overrides{};
  int effect_failures{};
  int completed_transactions{};
  friend bool operator==(const CommitAudit&, const CommitAudit&) = default;
};

struct StateSnapshot {
  int source_cursor{};
  std::size_t next_stage{};
  bool awaiting_observation{};
  bool completed{};
  CommitAudit audit;
  friend bool operator==(const StateSnapshot&, const StateSnapshot&) = default;
};

enum class ProposalStatus : std::uint8_t {
  Accepted,
  Completed,
  AwaitingObservation,
  InvalidRepairDelta,
};

struct Proposal {
  ProposalStatus status{ProposalStatus::InvalidRepairDelta};
  std::string reason;
  modular_agent_core::RepairDelta delta;
  LifecycleStage expected;
  std::size_t stage_index{};
  int source_cursor_before{};
  int source_cursor_after{};
  [[nodiscard]] bool accepted() const noexcept {
    return status == ProposalStatus::Accepted;
  }
};

enum class FinalStatus : std::uint8_t {
  Selected,
  Overridden,
  StaleProposal,
  InvalidManifest,
};

enum class CommitStatus : std::uint8_t {
  Committed,
  EffectNotConfirmed,
  NoPendingSubmission,
  StaleReceipt,
};

// Transactional seam around the existing side-effect-free RepairDelta and
// stateless modular composer. No RouteCursor/audit state is advanced by
// propose(); the final manifest is staged, then the next observation receipt
// is the sole commit authority.
class Coordinator {
 public:
  explicit Coordinator(LifecycleContract contract, int source_cursor = 0);

  [[nodiscard]] Proposal propose(
      const failure_debt::modular_bridge::RepairDeltaProposal& repair) const;
  FinalStatus stage_final(const Proposal& proposal,
                          const modular_agent_core::ExecutionManifest& manifest);
  CommitStatus observe(const modular_agent_core::ExecutionReceipt& receipt);

  [[nodiscard]] StateSnapshot snapshot() const noexcept;
  [[nodiscard]] const LifecycleContract& contract() const noexcept {
    return contract_;
  }

 private:
  struct Pending {
    std::size_t stage_index{};
    LifecycleStage expected;
    std::uint64_t action_hash{};
    modular_agent_core::SharedKey key;
  };

  LifecycleContract contract_;
  int source_cursor_{};
  std::size_t next_stage_{};
  bool completed_{};
  CommitAudit audit_;
  std::optional<Pending> pending_;
};

[[nodiscard]] const char* proposal_status_name(ProposalStatus status) noexcept;
[[nodiscard]] const char* final_status_name(FinalStatus status) noexcept;
[[nodiscard]] const char* commit_status_name(CommitStatus status) noexcept;

}  // namespace g001::repair_audit::two_phase
