#pragma once

#include "production_obligation_day_scheduler.hpp"
#include "transactional_animal_repair_owner.hpp"
#include "transactional_crop_repair_owner.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <vector>

namespace g001::day_start_issuer {

enum class UnsupportedReason : std::uint8_t {
  MissingCropIdentity = 0,
  MissingAnimalIdentity,
  ActorUnavailableAtDayStart,
  UnsupportedProductionAction,
  AnimalHarvestProductIdentityMissing,
  AuthorityMismatch,
  FinalProviderMismatch,
};

struct UnsupportedIntent {
  int actor{-1};
  int source_step{-1};
  fastkag::Position tile{};
  fastkag::Action source_action{};
  UnsupportedReason reason{UnsupportedReason::UnsupportedProductionAction};
};

struct RawSourceEntry {
  int actor{-1};
  int source_step{-1};
  fastkag::Action action{};
};

struct IntentLineage {
  int actor{-1};
  int source_step{-1};
  fastkag::Position tile{};
  fastkag::Item item{fastkag::Item::NONE};
  fastkag::Action exact_source{};
};

// Episode-persistent, route-owned memory.  Only an exact typed PLANT/PLACE
// source may create lineage.  A later weed observation deliberately does not
// erase crop lineage.  The registry contains no opponent or market data.
class PersistentRouteIntentRegistry {
 public:
  [[nodiscard]] bool record_exact_source(int actor, int source_step,
                                         fastkag::Position tile,
                                         fastkag::Action source);
  [[nodiscard]] std::optional<IntentLineage> crop_at(
      fastkag::Position tile) const;
  [[nodiscard]] std::optional<IntentLineage> animal_at(
      fastkag::Position tile) const;
  void clear() noexcept;

 private:
  std::map<std::pair<int, int>, IntentLineage> crops_;
  std::map<std::pair<int, int>, IntentLineage> animals_;
};

// route_tape is the immutable native route/production tape selected as G001.
// The optional typed spans are compiler-owned identity evidence already known
// at day start; they are never populated from later receipts or observations.
struct IssueRequest {
  const fastkag::Simulator* day_start{};
  const std::vector<fastkag::PlayerAction>* route_tape{};
  int player{-1};
  std::uint64_t issuer_generation{};
  const PersistentRouteIntentRegistry* persistent_intents{};
  std::span<const transactional_crop_repair::TypedCropObligation>
      exact_crop_bindings;
  std::span<const transactional_animal_repair::TypedAnimalObligation>
      exact_animal_bindings;
};

enum class IssueReject : std::uint8_t {
  None = 0,
  InvalidInput,
  NotDayStart,
  RouteTooShort,
};

struct IssueResult {
  IssueReject reject{IssueReject::None};
  std::vector<obligation_day::MoveSourceToken> moves;
  std::vector<RawSourceEntry> raw_sources;
  std::vector<obligation_day::ProductionObligation> obligations;
  std::vector<transactional_crop_repair::TypedCropObligation>
      crop_owner_obligations;
  std::vector<transactional_animal_repair::TypedAnimalObligation>
      animal_owner_obligations;
  std::vector<UnsupportedIntent> unsupported;

  [[nodiscard]] bool issued() const noexcept {
    return reject == IssueReject::None;
  }
  [[nodiscard]] bool fully_proven() const noexcept {
    return issued() && unsupported.empty();
  }
};

enum class CrossTickAnimalStatus : std::uint8_t {
  Staged = 0,
  Bound,
  InvalidSubmission,
  SecondWriter,
  NoPending,
  StaleReceipt,
  ReceiptMismatch,
  MissingHireEvidence,
  MissingBuyEvidence,
  ActorShapeMismatch,
  PlaceNotUnique,
};

struct FinalMarketReceipt {
  int player{-1};
  int submitted_step{-1};
  std::vector<fastkag::Action> submitted_market;
  std::vector<std::int32_t> fills;
};

struct CrossTickAnimalResult {
  CrossTickAnimalStatus status{CrossTickAnimalStatus::InvalidSubmission};
  std::optional<transactional_animal_repair::TypedAnimalObligation>
      obligation;
};

// HIRE executes after units, so a day-start source cannot safely bind a new
// actor. This seam stages only the exact finalized market request, then binds
// on the next observation after exact HIRE/BUY receipts and physical actor
// creation prove a unique matching PLACE on the copied immutable day suffix.
class CrossTickAnimalIntentBinder {
 public:
  [[nodiscard]] CrossTickAnimalResult stage_final_submission(
      const fastkag::Simulator& before, int player,
      const fastkag::PlayerAction& final_action,
      std::span<const fastkag::PlayerAction> immutable_route,
      std::uint64_t issuer_generation);
  [[nodiscard]] CrossTickAnimalResult bind_next(
      const fastkag::Simulator& after, const FinalMarketReceipt& receipt);
  void abort() noexcept { pending_.reset(); }
  [[nodiscard]] bool pending() const noexcept { return pending_.has_value(); }

 private:
  struct Pending {
    int player{-1};
    int submitted_step{-1};
    int prior_hands{};
    std::uint64_t seed{};
    fastkag::Item animal{fastkag::Item::NONE};
    int buy_slot{-1};
    std::vector<int> hire_slots;
    std::vector<fastkag::Action> submitted_market;
    std::vector<std::vector<fastkag::Action>> suffix_units;
    std::uint64_t obligation_id{};
  };
  std::optional<Pending> pending_;
};

struct ProviderDayAudit {
  bool day_valid{};
  int checked_sources{};
  int mismatches{};
  int first_mismatch_step{-1};
  std::array<int, 24> mismatches_by_source_op{};
  std::vector<UnsupportedIntent> mismatched_sources;
};

[[nodiscard]] IssueResult issue_day_start(const IssueRequest& request);
// Audit-only: actual_provider_actions must be collected from the final native
// provider path.  They never enter issue_day_start or its planning evidence.
[[nodiscard]] ProviderDayAudit audit_final_provider_day(
    const IssueResult& issued, int day_start,
    std::span<const fastkag::PlayerAction> actual_provider_actions);
[[nodiscard]] const char* unsupported_reason_name(UnsupportedReason reason);
[[nodiscard]] const char* cross_tick_animal_status_name(
    CrossTickAnimalStatus status);

}  // namespace g001::day_start_issuer
