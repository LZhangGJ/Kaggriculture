#pragma once

#include "production_obligation_day_scheduler.hpp"

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace g001::day_runtime_receipt {

// One evaluator-owned record for one actor in an actually emitted joint unit
// manifest.  Every authority-bearing field is echoed explicitly; none is
// inferred from a mutable overlay or from the action payload alone.
struct ActorExecution {
  int player{-1};
  int day{-1};
  int step{-1};
  int actor{-1};
  int source_step{-1};
  std::uint64_t obligation_id{};
  std::uint64_t certificate_hash{};
  std::uint64_t issuer_generation{};
  fastkag::Action source_action{};
  fastkag::Action emitted{};
};

// before/after bracket only the official unit phase.  They must be real
// Simulator snapshots owned by the evaluator.  Market, town, decay, RNG,
// midnight reset, and logical step advance are outside this receipt.
struct StepExecution {
  const fastkag::Simulator* before{};
  const fastkag::Simulator* after{};
  std::uint64_t claimed_before_fingerprint{};
  std::uint64_t claimed_after_fingerprint{};
  std::vector<ActorExecution> actors;
};

enum class Failure : std::uint8_t {
  None = 0,
  CertificateInvalid,
  SessionFailed,
  AlreadyClosed,
  NullSnapshot,
  DuplicateStep,
  SkippedStep,
  SlotShape,
  IdentityBinding,
  CertificateHashBinding,
  GenerationBinding,
  SourceBinding,
  ObligationBinding,
  EmittedActionMismatch,
  UnsupportedAction,
  BeforeFingerprintForgery,
  AfterFingerprintForgery,
  StateFingerprintFork,
  PhysicalNoEffect,
  PassHadEffect,
  PhysicalAfterMismatch,
  CertificatePostMismatch,
  IncompleteDay,
  MoveNotExactlyOnce,
  SlotNotClosed,
};

struct Result {
  bool accepted{};
  bool day_closed{};
  Failure failure{Failure::None};
  int checked_steps{};
  int checked_actor_slots{};
  int failure_step{-1};
  int failure_actor{-1};
};

// A single-use, fail-closed verifier.  Construction independently verifies
// the supplied DayScheduleCertificate against its immutable request.  It then
// accepts receipts strictly in step order.  Any rejection permanently poisons
// the session; there is intentionally no rewind/retry API.
class RuntimeReceiptVerifier {
 public:
  RuntimeReceiptVerifier(
      const obligation_day::DayPlanRequest& request,
      const obligation_day::DayScheduleCertificate& certificate);

  RuntimeReceiptVerifier(const RuntimeReceiptVerifier&) = delete;
  RuntimeReceiptVerifier& operator=(const RuntimeReceiptVerifier&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] Failure opening_failure() const noexcept;
  [[nodiscard]] Result accept(const StepExecution& execution);
  [[nodiscard]] Result close_midnight();

 private:
  [[nodiscard]] Result reject(Failure failure, int step = -1,
                              int actor = -1);
  [[nodiscard]] Result snapshot_result(bool accepted,
                                       bool day_closed = false) const;

  obligation_day::DayScheduleCertificate certificate_;
  int day_start_{};
  int next_tick_{};
  int checked_actor_slots_{};
  bool failed_{};
  bool closed_{};
  Failure opening_failure_{Failure::None};
  std::uint64_t previous_post_fingerprint_{};
  std::vector<bool> closed_slots_;
  std::set<std::pair<int, int>> seen_moves_;
};

[[nodiscard]] const char* failure_name(Failure failure) noexcept;

}  // namespace g001::day_runtime_receipt
