#pragma once

#include "production_obligation.hpp"
#include "transactional_animal_repair_owner.hpp"
#include "transactional_crop_repair_owner.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace g001::typed_intent {

enum class Proof : std::uint8_t {
  DirectActionItem = 0,
  LivePlantTile,
  DagTypedPosition,
  InvalidActor,
  ActionItemMissing,
  ActionItemNotAnimal,
  LiveTileHasNoCrop,
  DagNodeMissing,
  DagPositionMissing,
  DagAmbiguous,
  DynamicSuffixUncertified,
  UnsupportedAction,
};

struct Evidence {
  int player{-1};
  int actor{-1};
  int source_step{-1};
  fastkag::Position tile{-1, -1};
  fastkag::Action source_action{};
  Proof proof{Proof::UnsupportedAction};
  fastkag::Item resolved_item{fastkag::Item::NONE};
  bool exact{};
  bool emitted{};
};

// Narrow read-only seam.  It accepts the final current unit manifest already
// chosen by NativeTeammate plus the existing production-obligation DAG.  It
// does not own scheduling, receipts, purchases, routes, or native options.
struct Request {
  const fastkag::Simulator* observation{};
  int player{-1};
  std::uint64_t issuer_generation{};
  std::span<const fastkag::Action> final_current_units;
  std::span<const production_obligation::ObligationNode> dag_nodes;
  bool dag_suffix_certified{};
};

struct Result {
  std::vector<transactional_crop_repair::TypedCropObligation> crops;
  std::vector<transactional_animal_repair::TypedAnimalObligation> animals;
  std::vector<Evidence> evidence;
};

[[nodiscard]] Result issue(const Request& request);
[[nodiscard]] const char* proof_name(Proof proof) noexcept;

}  // namespace g001::typed_intent
