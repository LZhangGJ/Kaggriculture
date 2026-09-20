#pragma once

#include "atomic_day_admission.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace g001::state_target {

enum class PlotTargetKind : std::uint8_t { Crop = 0, Animal };

struct PlotTarget {
  std::uint64_t id{};
  fastkag::Position tile{};
  PlotTargetKind kind{PlotTargetKind::Crop};
  fastkag::Item item{fastkag::Item::NONE};
  // Crop targets accept -1 (any actor). Animal workflows are bound to one
  // actor so PICKUP and PLACE cannot split carried inventory.
  int actor{-1};
  double value{};
  fastkag::Position supply_position{};
  // maintain controls HARVEST and fertilizer collection. FEED and CARE are
  // explicit because callers may value them independently.
  bool maintain{true};
  bool feed{true};
  bool care{true};
};

enum class TargetGroupKind : std::uint8_t {
  CropLifecycle = 0,
  AnimalAcquisition,
  AnimalFeed,
  AnimalCare,
  AnimalHarvest,
  AnimalFertilizer,
};

struct TargetGroup {
  // id identifies this independently selectable atomic group. Several groups
  // may share one parent PlotTarget.
  std::uint64_t id{};
  std::uint64_t parent_target_id{};
  TargetGroupKind kind{TargetGroupKind::CropLifecycle};
  double value{};
  std::vector<std::uint64_t> obligation_ids;
  std::vector<std::uint64_t> dependency_group_ids;
};

enum class ResourceDemandReason : std::uint8_t {
  Seed = 0,
  Animal,
  FeedWheat,
};

struct ResourceDemand {
  std::uint64_t group_id{};
  std::uint64_t target_id{};
  double unfilled_value{};
  fastkag::Item item{fastkag::Item::NONE};
  int quantity{};
  int request_step{-1};
  // A market fill becomes usable by units on the following step.
  int latest_purchase_step{-1};
  ResourceDemandReason reason{ResourceDemandReason::Seed};
};

enum class DiagnosticCode : std::uint8_t {
  InvalidRequest = 0,
  InvalidTarget,
  DuplicateTarget,
  DuplicateTile,
  InvalidActor,
  InvalidSupplyPosition,
  CannotRemoveAnimal,
  UnsupportedTileState,
};

struct Diagnostic {
  std::uint64_t target_id{};
  DiagnosticCode code{DiagnosticCode::InvalidTarget};
  std::string message;
};

struct CompileResult {
  std::vector<obligation_day::ProductionObligation> obligations;
  std::vector<TargetGroup> groups;
  std::vector<ResourceDemand> demands;
  std::vector<Diagnostic> diagnostics;
};

// Compiles only from the supplied real observation. It never reads an old
// action tape and keeps no displaced-action/debt state between calls.
[[nodiscard]] CompileResult compile_state_targets(
    const fastkag::Simulator& observation, int player, int earliest_step,
    int deadline_step, std::span<const PlotTarget> targets);

[[nodiscard]] std::vector<obligation_day::AtomicGroup> atomic_groups(
    const CompileResult& compiled);

// Current hard buys for compile_native_general_market's acquisition overload.
// Only explicitly approved positive-value groups are submitted. Demands which
// can no longer fill before their unit deadline are omitted.
[[nodiscard]] std::vector<fastkag::Action> current_acquisitions(
    const CompileResult& compiled,
    std::span<const std::uint64_t> approved_group_ids);

}  // namespace g001::state_target
