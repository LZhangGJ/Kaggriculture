#pragma once

#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace g001::effect_certificate {

enum class Failure : std::uint8_t {
  None,
  UnsupportedOperation,
  ActorMissing,
  LockedTile,
  NotShedAdjacent,
  InvalidItemOrQuantity,
  ShedItemAbsent,
  ActorItemAbsent,
  ShedFull,
  WrongAnimalStructure,
  TileNotPlant,
  FertilizerAbsent,
  TileNotAnimal,
  AlreadyFed,
  WheatAbsent,
  FertilizerUnavailable,
  YieldAbsent,
  CropImmature,
  InvalidTileIdentity,
};

enum class PublicCertainty : std::uint8_t {
  Exact,
  Unobservable,
  GlobalOpaque,
};

struct LocalState {
  fastkag::Config config;
  int day{};
  int actor_slot{};
  bool actor_exists{true};
  fastkag::Position actor_position;
  bool shed_adjacent{};
  fastkag::Tile tile;
  std::array<int32_t, fastkag::N_ITEMS> shed{};
  std::array<int32_t, fastkag::N_ITEMS> inventory{};
  std::vector<int8_t> inventory_order;
  friend bool operator==(const LocalState&, const LocalState&);
};

struct Footprint {
  std::vector<std::string> reads;
  std::vector<std::string> writes;
  PublicCertainty lifecycle_public_certainty{PublicCertainty::Exact};
  std::string public_limitation;
};

struct Prediction {
  LocalState after;
  Failure failure{Failure::None};
  int transferred{};
  bool changed{};
  Footprint footprint;
};

// Predicts exactly one actor's immediate unit-phase effect. `before` must be
// the state after every lower actor slot in the same player manifest has run.
// It deliberately excludes market, town, decay, end-of-day and step advance.
[[nodiscard]] Prediction predict_unit_effect(
    const LocalState& before, const fastkag::Action& action);

[[nodiscard]] const char* failure_name(Failure failure) noexcept;

}  // namespace g001::effect_certificate
