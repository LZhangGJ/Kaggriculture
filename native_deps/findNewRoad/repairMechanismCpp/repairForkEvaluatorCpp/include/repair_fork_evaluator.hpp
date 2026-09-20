#pragma once

#include "native_teammate.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace g001::repair_fork {

enum class Panel : std::uint8_t { Normal = 0, ForcedWeed = 1 };

// A production purchase is deliberately typed.  A repair owner cannot hide a
// seed or animal acquisition inside an unowned generic market rewrite.
struct RequiredPurchase {
  std::uint64_t debt_id{};
  fastkag::Op operation{fastkag::Op::PASS};  // BUY_SEED or BUY_ANIMAL only.
  fastkag::Item item{fastkag::Item::NONE};
  int quantity{};
  int required_by_step{-1};
};

enum class PurchaseCompileStatus : std::uint8_t {
  BoundExisting = 0,
  Appended = 1,
  RejectedInvalid = 2,
  RejectedNoSlot = 3,
};

struct PurchaseBinding {
  std::uint64_t debt_id{};
  int market_slot{-1};
  fastkag::Op operation{fastkag::Op::PASS};
  fastkag::Item item{fastkag::Item::NONE};
  int requested{};
  PurchaseCompileStatus status{PurchaseCompileStatus::RejectedInvalid};
};

struct PurchaseReceipt {
  std::uint64_t debt_id{};
  int submitted_step{-1};
  fastkag::Op operation{fastkag::Op::PASS};
  fastkag::Item item{fastkag::Item::NONE};
  int requested{};
  int filled{};
  int market_slot{-1};
  PurchaseCompileStatus compile_status{PurchaseCompileStatus::RejectedInvalid};
};

// Every emitted unit action names its raw G001 source.  Inserted repair work
// uses source_step=-1.  This lets the evaluator audit MOVE identity without
// interpreting a planner's private queue.
struct SourceBinding {
  int actor{-1};
  int source_step{-1};
  fastkag::Action source_action{};
};

struct ActorPrefixAuthority {
  int actor{-1};
  std::uint64_t manifest_generation{};
  std::uint64_t prefix_manifest_hash{};
  std::uint64_t post_prefix_state_fingerprint{};
};

// Issued by the evaluator only after the authorized manifest was submitted.
// The next decision must echo every field in receipt_acks; an owner cannot
// manufacture success from a local prediction or silently cross a day reset.
struct ActionReceipt {
  int submitted_step{-1};
  int actor{-1};
  std::uint64_t manifest_generation{};
  std::uint64_t prefix_manifest_hash{};
  std::uint64_t post_prefix_state_fingerprint{};
  fastkag::Action emitted{};
};

struct RepairTelemetry {
  int triggers{};
  int debts_opened{};
  int debts_closed{};
  int receipts_confirmed{};
  int receipts_failed{};
  int fail_closed{};
};

struct RepairContext {
  const fastkag::Simulator& phase_start;
  int player{};
  int step{};
  const fastkag::PlayerAction& raw_g001;
  const std::array<fastkag::PlayerAction, 2>& raw_joint;
  std::span<const PurchaseReceipt> previous_purchase_receipts;
  std::span<const ActionReceipt> previous_action_receipts;
};

struct RepairDecision {
  std::vector<fastkag::Action> units;
  std::vector<SourceBinding> sources;
  std::vector<RequiredPurchase> required_purchases;
  std::vector<ActorPrefixAuthority> prefix_authority;
  std::vector<ActionReceipt> receipt_acks;
  std::vector<PurchaseReceipt> purchase_receipt_acks;
  RepairTelemetry telemetry{};
};

class RepairOwner {
 public:
  virtual ~RepairOwner() = default;
  virtual std::string name() const = 0;
  virtual RepairDecision decide(const RepairContext& context) = 0;
};

class PassThroughOwner final : public RepairOwner {
 public:
  std::string name() const override;
  RepairDecision decide(const RepairContext& context) override;
};

struct MarketCompileResult {
  std::vector<fastkag::Action> market;
  std::vector<PurchaseBinding> bindings;
  int rejected_invalid{};
  int rejected_no_slot{};
};

// The only market seam exposed to a repair owner.  Legacy G001 market actions
// remain intact; required production purchases bind to an existing identical
// order or append into a free official order slot.  Fill receipts come only
// from Simulator::last_market_fills after the phase.
class ExactMarketCompiler {
 public:
  MarketCompileResult compile(const fastkag::Simulator& phase_start, int player,
                              std::span<const fastkag::Action> legacy_market,
                              std::span<const RequiredPurchase> required) const;
};

struct EvaluatorOptions {
  std::string tapes;
  std::string library;
  std::string references;
  std::string opponent{"G001"};
  std::uint64_t seed_begin{25772238701ULL};
  int seeds{1};
  int threads{1};
  int candidate_repair_mask{};
  std::vector<Panel> panels{Panel::Normal};
};

struct GameResult {
  std::uint64_t seed{};
  int seat{};
  Panel panel{Panel::Normal};
  int steps{};
  double baseline_own{}, baseline_opponent{};
  double candidate_own{}, candidate_opponent{};
  int baseline_unit_failures{}, candidate_unit_failures{};
  int baseline_market_failures{}, candidate_market_failures{};
  int baseline_overflow{}, candidate_overflow{};
  int action_mismatches{}, environment_mismatches{};
  int reward_mismatches{};
  int move_direction_failures{}, move_day_failures{}, move_slot_failures{};
  int move_source_failures{}, move_sequence_failures{}, move_route_failures{};
  int observed_legacy_weed_triggers{};
  int repair_triggers{}, purchase_required{}, purchase_zero_fills{};
  int purchase_partial_fills{}, purchase_full_fills{};
  int purchase_compile_rejected_invalid{}, purchase_compile_rejected_no_slot{};
  int debts_opened{}, debts_closed{}, debts_outstanding{};
  int receipts_confirmed{}, receipts_failed{};
  int action_receipts_issued{}, action_receipts_acked{};
  int purchase_receipts_issued{}, purchase_receipts_acked{};
  int terminal_unacked_action_receipts{};
  int terminal_unacked_purchase_receipts{};
  int exact_purchase_debts_outstanding{};
  int prefix_authority_checks{}, prefix_authority_failures{};
  int hour23_triggers{}, hour23_commits{}, hour23_fail_closed{};
};

struct EvaluationReport {
  std::string owner;
  std::string opponent;
  std::vector<GameResult> games;
  std::string cases_jsonl;
  std::string summary_json;
  std::string deterministic_payload_sha256;
};

EvaluationReport evaluate(const EvaluatorOptions& options,
                          const std::function<std::unique_ptr<RepairOwner>()>&
                              owner_factory);

std::string sha256_hex(std::string_view input);
std::uint64_t full_unit_phase_state_fingerprint(
    const fastkag::Simulator& env);
std::uint64_t phase_start_fingerprint(const fastkag::Simulator& env, int player);
std::uint64_t unit_prefix_manifest_hash(
    std::span<const fastkag::Action> final_units, int actor_exclusive);
std::uint64_t post_unit_prefix_state_fingerprint(
    const fastkag::Simulator& phase_start, int player,
    const std::array<fastkag::PlayerAction, 2>& final_joint,
    int actor_exclusive);
const char* panel_name(Panel panel);

}  // namespace g001::repair_fork
