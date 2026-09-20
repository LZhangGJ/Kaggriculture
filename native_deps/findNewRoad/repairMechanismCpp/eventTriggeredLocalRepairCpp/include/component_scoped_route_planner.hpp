#pragma once

#include "day_horizon_planner.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace g001::day_horizon_repair {

enum class UnsupportedEffectScope : std::uint8_t {
  ActorLocal,
  TileLocal,
};

struct UnsupportedEffectCertificate {
  UnsupportedEffectScope scope{UnsupportedEffectScope::TileLocal};
  // Item ids whose inventory/state is observed but not consumed.
  std::set<int> resource_reads;
  // Positive quantities consumed or produced by item id. Production is not
  // credited as planning capacity until a receipt updates ResourceSnapshot.
  std::map<int, int> resource_consumption;
  std::map<int, int> resource_production;
  // False means the footprint is incomplete and therefore GlobalOpaque.
  bool no_unlisted_global_effects{};
};

struct RawTileTransitionCertificate {
  event_local_repair::TileObservation before;
  event_local_repair::TileObservation after;
};

struct BoardLegalityCertificate {
  int rows{};
  int columns{};
  bool complete{};
  std::set<std::pair<Position, Position>> blocked_edges;
  // Exact source-id causal transition. Repeated equal-direction actions must
  // still have distinct entries.
  std::map<std::uint64_t, std::pair<Position, Position>> move_transitions;
};

// Offline-only input for the component-scoped rolling-route experiment.  The
// vector order is the hard source order. earliest_turn is a release time, not
// permission to overtake an earlier source. source_id distinguishes repeated
// equal-direction MOVE actions.
struct SourcedRawAction {
  SourcedRawAction() = default;
  SourcedRawAction(std::uint64_t source, int release, Action raw,
                   bool is_unsupported)
      : source_id(source), earliest_turn(release), action(std::move(raw)),
        unsupported(is_unsupported) {}
  SourcedRawAction(std::uint64_t source, int release, Action raw,
                   bool is_unsupported, std::nullopt_t)
      : SourcedRawAction(source, release, std::move(raw), is_unsupported) {}
  SourcedRawAction(std::uint64_t source, int release, Action raw,
                   bool is_unsupported,
                   UnsupportedEffectCertificate certificate)
      : SourcedRawAction(source, release, std::move(raw), is_unsupported) {
    effect_certificate.emplace(std::move(certificate));
  }
  SourcedRawAction(std::uint64_t source, int release, Action raw,
                   bool is_unsupported, std::nullopt_t,
                   RawTileTransitionCertificate tile_transition)
      : SourcedRawAction(source, release, std::move(raw), is_unsupported) {
    tile_transition_certificate.emplace(std::move(tile_transition));
  }
  SourcedRawAction(std::uint64_t source, int release, Action raw,
                   bool is_unsupported,
                   UnsupportedEffectCertificate certificate,
                   RawTileTransitionCertificate tile_transition)
      : SourcedRawAction(source, release, std::move(raw), is_unsupported,
                         std::move(certificate)) {
    tile_transition_certificate.emplace(std::move(tile_transition));
  }
  std::uint64_t source_id{};
  int earliest_turn{};
  Action action;
  // The crop planner cannot interpret this action's state/resource effects.
  // Its whole conflict component is sealed to ordered raw execution.
  bool unsupported{};
  // Unsupported Other without a complete certificate is GlobalOpaque and
  // seals every repair component. Distance alone is not independence proof.
  std::optional<UnsupportedEffectCertificate> effect_certificate;
  // Required before a later objective may depend on this raw tile effect.
  std::optional<RawTileTransitionCertificate> tile_transition_certificate;
};

struct ComponentActorPlan {
  ComponentActorPlan() = default;
  ComponentActorPlan(
      int actor_id, Position initial, int horizon,
      std::vector<SourcedRawAction> sources,
      std::optional<BoardLegalityCertificate> legality = std::nullopt)
      : actor(actor_id), start(initial), turns(horizon),
        ordered_raw(std::move(sources)), board(std::move(legality)) {}
  int actor{-1};
  Position start;
  int turns{};
  std::vector<SourcedRawAction> ordered_raw;
  // Required for every MOVE. Missing, incomplete, out-of-bounds, blocked, or
  // source-mismatched proof makes the global proposal unsafe.
  std::optional<BoardLegalityCertificate> board;
};

struct ComponentScopedResult : TimeExpandedResult {
  // Zero denotes PASS or a repair assignment. Nonzero values identify the
  // exact raw source emitted in that manifest slot.
  std::vector<std::vector<std::uint64_t>> raw_source_manifest;
  std::vector<std::uint64_t> committed_objectives;
  int sealed_components{};
  int committed_components{};
  bool raw_source_order_exact{};
  bool global_tile_serialization{};
  bool global_resource_safe{};
  bool global_route_geometry_safe{};
  bool objective_conservation{};
  bool merge_safe{};
};

// Component-scoped proposal prototype. Actor reachability, shared tiles and
// shared crop-seed demand form conflict components. A component containing an
// unsupported raw action retains its complete source queue but does not accept
// repair objectives. Other components may still commit. The merged manifest
// is then revalidated globally for source/MOVE order, positions, tile effects,
// resources, and terminal raw debt. There is deliberately no native seam.
[[nodiscard]] ComponentScopedResult compile_component_scoped_route(
    const std::vector<ComponentActorPlan>& actors,
    std::vector<Objective> objectives,
    const ResourceSnapshot& resources = {},
    TimeExpandedConfig config = {});

}  // namespace g001::day_horizon_repair
