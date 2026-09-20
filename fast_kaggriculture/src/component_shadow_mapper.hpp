#pragma once

#include "simulator.hpp"
#include "effect_certificate_semantics.hpp"
#include "../../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/include/component_scoped_route_planner.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <string_view>
#include <vector>

namespace fastkag::component_shadow {

namespace local = g001::day_horizon_repair;

enum class CertificateKind : std::uint8_t {
  ExactNoEffect,
  MoveTransition,
  TileTransition,
  ScopedEffect,
  ReplaceableObjective,
  MaturityWait,
  TypedEffect,
  GlobalOpaque,
};

enum class TypedStateDomain : std::uint8_t {
  ActorInventory,
  Shed,
  Tile,
};

enum class TypedTileField : std::uint8_t {
  Kind,
  Crop,
  Animal,
  PlantedDay,
  PlacedDay,
  YieldUnits,
  ConsecutiveUnwatered,
  ConsecutiveUnfed,
  FertilizedUntilDay,
  PendingCareBonus,
  MaxLifespanStep,
  WateredToday,
  FedToday,
  CaredToday,
  FertilizerAvailable,
};

struct TypedStateCell {
  TypedStateDomain domain{TypedStateDomain::ActorInventory};
  int actor{-1};
  local::Position tile;
  int item{-1};
  TypedTileField tile_field{TypedTileField::Kind};
  int before{};
  int after{};
};

struct TypedEffectCertificate {
  int actor{-1};
  std::uint64_t source_id{};
  int transferred{};
  bool lower_slot_prefix_bound{};
  std::uint64_t prefix_hash{};
  std::uint64_t actor_generation{};
  std::vector<TypedStateCell> cells;
  // Exact read/write dependency keys. Cells carry write quantities; keys also
  // include unchanged reads such as shed capacity/fullness.
  std::set<int> dependency_keys;
  std::vector<int> inventory_order_before;
  std::vector<int> inventory_order_after;
};

struct ObjectiveDraft {
  local::Position tile;
  std::vector<local::Action> transitions;
  int value{};
};

struct MappedUnit {
  Op native_op{Op::PASS};
  CertificateKind kind{CertificateKind::ExactNoEffect};
  std::optional<local::SourcedRawAction> raw_source;
  std::optional<ObjectiveDraft> objective;
  std::optional<TypedEffectCertificate> typed_effect;
  g001::effect_certificate::Failure effect_failure{
      g001::effect_certificate::Failure::None};
  bool causal_effect{};
};

[[nodiscard]] MappedUnit map_current_unit(
    const Simulator& before, int player, int actor,
    const PlayerAction& player_raw, std::uint64_t source_id,
    int earliest_turn);

[[nodiscard]] const char* certificate_kind_name(CertificateKind kind);
[[nodiscard]] const char* unit_op_name(Op op);
[[nodiscard]] bool native_unit_op(Op op);

}  // namespace fastkag::component_shadow
