#pragma once

#include "protected_queue.hpp"
#include "robust_certificate.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace queue_invariant_proof {

enum class ExactRelation : std::uint8_t {
  NoPressure,
  OpponentBefore,
  Lockstep,
};

// One fact per protected_queue::Audit::scenarios entry. Relation describes
// actual slot placement, not a probabilistic or abstract ordering label.
struct ExactPressureFact {
  std::size_t scenario_index{};
  ExactRelation relation{ExactRelation::NoPressure};
  bool exact_replay_completed{};
  bool abstract_opponent_first_represented{};
};

struct Input {
  std::vector<g001::market::Order> legacy_queue;
  protected_queue::ResultV2 candidate;
  g001::market::Product selected_product{g001::market::Product::Wheat};
  int bridge_emitted_total_target{};
  bool previous_execution_observation_confirmed{};
  bool movement_unchanged{};
  std::vector<ExactPressureFact> exact_pressure_facts;
};

struct Check {
  std::string name;
  bool passed{};
  std::string reason;
};

struct Audit {
  bool accepted{};
  int required_funding_units{};
  int optional_units{};
  int selected_actual_units{};
  int selected_legacy_sell_units{};
  int preserved_nonselected_sell_units{};
  int protected_non_sell_slots{};
  int due_binding_count{};
  int opponent_before_scenarios{};
  int lockstep_scenarios{};
  int no_pressure_scenarios{};
  // True only for pressure scenarios placed strictly before our optional SELL.
  // Lockstep is never counted as abstract OpponentFirst.
  bool all_pressure_scenarios_abstract_opponent_first{};
  std::vector<Check> checks;
  std::string reason;
};

struct Result {
  robust_certificate::ExecutionProof execution;
  robust_certificate::SlotProof slots;
  Audit audit;
};

[[nodiscard]] Result derive(const Input& input);

}  // namespace queue_invariant_proof
