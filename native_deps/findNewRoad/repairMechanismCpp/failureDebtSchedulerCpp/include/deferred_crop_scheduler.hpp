#pragma once

#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace g001::failure_debt::deferred_crop {

enum class ObligationState : std::uint8_t { Pending, InFlight, Confirmed };
enum class ProposalStatus : std::uint8_t {
  Disabled,
  NoDebtAtVisit,
  WaitingForReceipt,
  Blocked,
  ExistingEquivalent,
  EmitReplacement,
};

struct DeferredSource {
  int actor{-1};
  fastkag::Position tile{-1, -1};
  int source_step{-1};
  int origin_day{-1};
  int deadline_step{-1};
  fastkag::Action action{};
  // RouteCursor's action.item is NONE for WATER/HARVEST.  The adapter should
  // pass its last observation-confirmed crop here so a weed/empty tile can be
  // restored without guessing a crop.
  fastkag::Item remembered_crop{fastkag::Item::NONE};
  bool critical{true};
  std::string provenance;
};

struct CropSnapshot {
  fastkag::TileKind kind{fastkag::TileKind::EMPTY};
  fastkag::Item crop{fastkag::Item::NONE};
  int planted_day{-1};
  int yield_units{};
  int fertilized_until_day{-1};
  bool watered_today{};
  // Supplied by the official simulator/projector.  HARVEST is never guessed
  // from age constants inside this isolated ABI.
  bool harvest_legal{};
};

struct Visit {
  int step{-1};
  int day{-1};
  int actor{-1};
  fastkag::Position position{-1, -1};
  CropSnapshot tile;
  std::array<int, fastkag::N_CROPS> seeds{};
  int carried_fertilizer{};
  fastkag::Action base_action{};
  bool base_critical{};
  // Certified by the route compiler after reserving every remaining MOVE.
  bool absorbable_slack{};
  int remaining_action_slots{};
  int remaining_moves{};
  std::uint64_t movement_hash{};
};

struct Obligation {
  std::uint64_t id{};
  int source_step{-1};
  int deadline_step{-1};
  fastkag::Action action{};
  ObligationState state{ObligationState::Pending};
  int attempts{};
  int coalesced_sources{1};
  bool critical{true};
  std::string provenance;
  // Exact source provenance is retained even when equivalent consecutive
  // same-day operations are semantically coalesced.
  std::vector<int> source_steps;
  std::vector<std::string> source_provenances;
};

struct TileDebt {
  std::uint64_t id{};
  int actor{-1};
  fastkag::Position tile{-1, -1};
  int origin_day{-1};
  int created_step{-1};
  int earliest_deadline{-1};
  fastkag::Item desired_crop{fastkag::Item::NONE};
  bool completed{};
  std::vector<Obligation> obligations;
};

struct Proposal {
  ProposalStatus status{ProposalStatus::Disabled};
  std::string reason;
  std::uint64_t debt_id{};
  std::uint64_t obligation_id{};
  int step{-1};
  int actor{-1};
  fastkag::Position tile{-1, -1};
  fastkag::Action original{};
  fastkag::Action action{};
  std::uint64_t movement_hash{};
  bool synthetic_prerequisite{};
  bool consumes_obligation_on_success{};
  [[nodiscard]] bool actionable() const noexcept {
    return status == ProposalStatus::ExistingEquivalent ||
           status == ProposalStatus::EmitReplacement;
  }
};

struct Receipt {
  int step{-1};
  int actor{-1};
  std::uint64_t debt_id{};
  std::uint64_t obligation_id{};
  fastkag::Action emitted{};
  CropSnapshot before;
  CropSnapshot after;
  int crop_inventory_delta{};
  int fertilizer_inventory_delta{};
  // WATER at the last turn of a day resets watered_today before the next
  // observation.  The integration may instead provide a deterministic lower
  // bound from the exact final manifest plus official unit-phase preview.
  bool day_end_water_effect_lower_bound{};
  bool generic_effect_verified{};
  std::uint64_t movement_hash{};
  std::string provenance;
};

struct AuditCounters {
  int sources_enqueued{};
  int sources_coalesced{};
  int proposals{};
  int replacements{};
  int existing_equivalents{};
  int synthetic_digs{};
  int synthetic_plants{};
  int synthetic_waters{};
  int receipt_successes{};
  int receipt_failures{};
  int completed_obligations{};
  int completed_tile_debts{};
  int overdue_open_obligations{};
  int rejected_sources{};
};

struct Config {
  bool enabled{false};
  int turns_per_day{24};
};

// Isolated execution ABI for RouteCursor::DeferredNonMove.  It never emits or
// consumes MOVE.  Deadlines affect priority/audit only: an overdue obligation
// remains open until an operation-specific receipt confirms it.
class DeferredCropScheduler {
 public:
  explicit DeferredCropScheduler(Config config = {});
  void reset();
  [[nodiscard]] bool enqueue(const DeferredSource& source);
  [[nodiscard]] Proposal propose(const Visit& visit) const;
  // Stage only the exact final unit manifest.  State is not completed here.
  [[nodiscard]] bool commit(const Proposal& proposal,
                            const fastkag::Action& final_action,
                            std::uint64_t final_movement_hash);
  // Receipt success confirms either a synthetic prerequisite or the head
  // obligation.  Receipt failure reopens it; nothing is silently retired.
  [[nodiscard]] bool observe_receipt(const Receipt& receipt);

  [[nodiscard]] std::vector<TileDebt> open_debts(int current_step) const;
  [[nodiscard]] const std::vector<TileDebt>& all_debts() const noexcept {
    return debts_;
  }
  [[nodiscard]] AuditCounters audit(int current_step) const;

 private:
  struct PendingReceipt {
    Proposal proposal;
  };

  [[nodiscard]] TileDebt* find_debt(std::uint64_t id);
  [[nodiscard]] const TileDebt* find_debt(std::uint64_t id) const;
  [[nodiscard]] Obligation* find_obligation(TileDebt& debt, std::uint64_t id);

  Config config_;
  std::vector<TileDebt> debts_;
  std::vector<PendingReceipt> pending_;
  AuditCounters audit_;
  std::uint64_t next_debt_id_{1};
  std::uint64_t next_obligation_id_{1};
};

[[nodiscard]] bool is_crop_deferred_operation(fastkag::Op operation) noexcept;
[[nodiscard]] const char* proposal_status_name(ProposalStatus status) noexcept;

}  // namespace g001::failure_debt::deferred_crop
