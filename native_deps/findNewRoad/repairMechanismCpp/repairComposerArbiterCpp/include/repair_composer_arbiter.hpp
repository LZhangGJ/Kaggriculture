#pragma once

#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace g001::repair_composer_arbiter {

enum class Source : std::uint8_t { WeedMinimumDamage, PurchaseFailure };
enum class Domain : std::uint8_t { Unit, Market };

struct Intent {
  Source source{Source::WeedMinimumDamage};
  std::uint64_t id{};
  std::uint64_t debt_id{};
  Domain domain{Domain::Unit};
  int actor{-1};
  int market_slot{-1};
  fastkag::Action expected_base{};
  fastkag::Action replacement{};
  fastkag::Item resource_item{fastkag::Item::NONE};
  int resource_consumption{};
  int cash_cost{};
  int shed_capacity_cost{};
  int deadline{-1};
  bool must_finish{};
  bool important{};
  int cascade_risk{};
  int displacement_damage{};
  bool certified_weed_move_owner{};
  std::vector<std::uint64_t> dependencies;
  std::string provenance;
};

struct ParticipantProposal {
  Source source{Source::WeedMinimumDamage};
  std::uint64_t generation{};
  std::uint64_t observation_hash{};
  std::uint64_t commit_token{};
  fastkag::PlayerAction base_action;
  fastkag::PlayerAction proposed_action;
  std::vector<Intent> intents;
};

enum class ConflictReason : std::uint8_t {
  SameActor,
  SameMarketSlot,
  ItemResource,
  Cash,
  ShedCapacity,
  Dependency,
  Deadline,
  PurchaseMoveEdit,
  UncertifiedWeedMoveEdit,
  UnboundActionDiff,
};

struct ConflictEdge {
  std::uint64_t left{};
  std::uint64_t right{};
  ConflictReason reason{ConflictReason::SameActor};
  bool excludes_joint_selection{true};
  std::string diagnostic;
};

struct ParticipantSelection {
  Source source{Source::WeedMinimumDamage};
  std::uint64_t generation{};
  std::uint64_t commit_token{};
};

struct Objective {
  int missed_must_finish{};
  int cascade_risk{};
  int deferred_important{};
  int total_displacement_damage{};
  std::uint32_t deterministic_mask{};
};

struct PrepareRequest {
  bool enabled{false};
  int step{-1};
  std::uint64_t observation_hash{};
  fastkag::PlayerAction base_action;
  std::optional<ParticipantProposal> weed;
  std::optional<ParticipantProposal> purchase;
  int available_cash{};
  int free_shed_capacity{};
  std::map<fastkag::Item, int> available_items;
  std::vector<std::uint64_t> satisfied_dependencies;
};

enum class PrepareStatus : std::uint8_t {
  Disabled,
  Accepted,
  InvalidInput,
  NoFeasibleJointSet,
};

struct Prepared {
  PrepareStatus status{PrepareStatus::InvalidInput};
  int step{-1};
  std::uint64_t generation{};
  std::uint64_t observation_hash{};
  std::uint64_t base_action_hash{};
  std::uint64_t final_action_hash{};
  std::uint64_t binding_hash{};
  fastkag::PlayerAction final_action;
  std::vector<Source> selected_sources;
  std::vector<ParticipantSelection> selected_participants;
  std::vector<std::uint64_t> selected_intents;
  std::vector<ConflictEdge> conflict_graph;
  Objective objective;
  std::string diagnostic;
  [[nodiscard]] bool accepted() const noexcept {
    return status == PrepareStatus::Disabled || status == PrepareStatus::Accepted;
  }
};

// Type-erased hook whose callable must be statically nothrow-invocable. This
// prevents an adapter from claiming rollback after an earlier commit throws.
class NoFailHook {
 public:
  NoFailHook() = default;
  template <class Callable>
    requires std::is_nothrow_invocable_v<Callable&>
  NoFailHook(Callable callable) : callback_(std::move(callable)) {}
  explicit operator bool() const noexcept {
    return static_cast<bool>(callback_);
  }
  void operator()() noexcept {
    if (callback_) callback_();
  }

 private:
  std::function<void()> callback_;
};

// preflight may stage arbitrary work, but only in an isolated clone. Commit
// and abort are statically no-fail hooks and must only publish/discard that
// already-prepared clone.
struct ParticipantTransaction {
  Source source{Source::WeedMinimumDamage};
  std::uint64_t generation{};
  std::uint64_t commit_token{};
  std::function<bool()> preflight;
  NoFailHook commit;
  NoFailHook abort;
};

enum class FinalizeStatus : std::uint8_t {
  Selected,
  Stale,
  Tampered,
  PartialCommit,
  ParticipantRejected,
  AlreadyFinalized,
};

class Composer {
 public:
  [[nodiscard]] Prepared prepare(const PrepareRequest& request) const;
  FinalizeStatus finalize(const Prepared& prepared,
                          const fastkag::PlayerAction& exact_selected_action,
                          std::vector<ParticipantTransaction> participants);
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

 private:
  std::uint64_t generation_{1};
  int last_step_{-1};
};

[[nodiscard]] std::uint64_t action_hash(
    const fastkag::PlayerAction& action) noexcept;
[[nodiscard]] const char* conflict_reason_name(ConflictReason reason) noexcept;
[[nodiscard]] const char* prepare_status_name(PrepareStatus status) noexcept;
[[nodiscard]] const char* finalize_status_name(FinalizeStatus status) noexcept;

}  // namespace g001::repair_composer_arbiter
