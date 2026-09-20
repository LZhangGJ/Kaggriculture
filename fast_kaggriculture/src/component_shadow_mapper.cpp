#include "component_shadow_mapper.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

namespace fastkag::component_shadow {
namespace {

using local::RawTileTransitionCertificate;
using local::SourcedRawAction;
using local::UnsupportedEffectCertificate;
using local::UnsupportedEffectScope;
using LAction = local::Action;
using LOp = g001::event_local_repair::Op;
using LPosition = local::Position;
using LTileKind = g001::event_local_repair::TileKind;
using LTile = g001::event_local_repair::TileObservation;
namespace effect = g001::effect_certificate;

bool movement(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
      op == Op::WEST;
}

Position actor_position(const Simulator& simulator, int player, int actor) {
  if (actor == 0) return simulator.farms()[player].farmer;
  const auto& hands = simulator.farms()[player].hands;
  if (actor < 0 || actor > static_cast<int>(hands.size())) return {-1, -1};
  return hands[static_cast<std::size_t>(actor - 1)];
}

const Tile* tile_at(const Simulator& simulator, int player, Position position) {
  const int size = simulator.config().board_size;
  if (position.x < 0 || position.y < 0 || position.x >= size ||
      position.y >= size)
    return nullptr;
  return &simulator.farms()[player].tiles[
      static_cast<std::size_t>(position.y * size + position.x)];
}

bool same_tile(const Tile& left, const Tile& right) {
  return left.kind == right.kind && left.crop == right.crop &&
      left.animal == right.animal && left.planted_day == right.planted_day &&
      left.placed_day == right.placed_day &&
      left.yield_units == right.yield_units &&
      left.consecutive_unwatered == right.consecutive_unwatered &&
      left.consecutive_unfed == right.consecutive_unfed &&
      left.fertilized_until_day == right.fertilized_until_day &&
      left.pending_care_bonus == right.pending_care_bonus &&
      left.max_lifespan_step == right.max_lifespan_step &&
      left.watered_today == right.watered_today &&
      left.fed_today == right.fed_today &&
      left.cared_today == right.cared_today &&
      left.fertilizer_available == right.fertilizer_available;
}

LPosition local_position(Position position) {
  return {position.y, position.x};
}

LTile local_tile(const Tile& tile, bool harvest_legal = false) {
  LTile result;
  switch (tile.kind) {
    case TileKind::EMPTY: result.kind = LTileKind::Empty; break;
    case TileKind::WEED: result.kind = LTileKind::Weed; break;
    case TileKind::PLANT: result.kind = LTileKind::Crop; break;
    case TileKind::COOP:
    case TileKind::PASTURE: result.kind = LTileKind::Structure; break;
    default: result.kind = LTileKind::Other; break;
  }
  result.item = tile.kind == TileKind::PLANT ? static_cast<int>(tile.crop) : -1;
  result.watered_today = tile.watered_today;
  result.harvest_legal = harvest_legal;
  return result;
}

LAction local_action(const Action& action) {
  LAction result;
  result.item = static_cast<int>(action.item);
  result.quantity = action.quantity;
  switch (action.op) {
    case Op::PASS: result.op = LOp::Pass; break;
    case Op::NORTH: result.op = LOp::Move; result.arg0 = 0; break;
    case Op::SOUTH: result.op = LOp::Move; result.arg0 = 1; break;
    case Op::WEST: result.op = LOp::Move; result.arg0 = 2; break;
    case Op::EAST: result.op = LOp::Move; result.arg0 = 3; break;
    case Op::DIG: result.op = LOp::Dig; break;
    case Op::PLANT: result.op = LOp::Plant; break;
    case Op::WATER: result.op = LOp::Water; break;
    case Op::HARVEST: result.op = LOp::Harvest; break;
    case Op::BUILD_COOP:
    case Op::BUILD_PASTURE: result.op = LOp::Build; break;
    default:
      result.op = LOp::Other;
      result.arg0 = static_cast<int>(action.op);
      break;
  }
  return result;
}

bool private_actor_equal(const Simulator& left, const Simulator& right,
                         int player, int actor) {
  const auto& lhs = left.privates()[player];
  const auto& rhs = right.privates()[player];
  if (lhs.shed != rhs.shed || lhs.seeds != rhs.seeds) return false;
  if (actor < 0 || actor >= static_cast<int>(lhs.inventories.size()) ||
      actor >= static_cast<int>(rhs.inventories.size()))
    return false;
  return lhs.inventories[static_cast<std::size_t>(actor)] ==
      rhs.inventories[static_cast<std::size_t>(actor)];
}

std::vector<LAction> crop_recovery(const Tile& tile, const Action& raw) {
  const int item = static_cast<int>(raw.item);
  if (item < 0 || item >= N_CROPS) return {};
  if (tile.kind == TileKind::WEED)
    return {{LOp::Dig}, {LOp::Plant, item, 1}, {LOp::Water, item, 1}};
  if (tile.kind == TileKind::EMPTY)
    return {{LOp::Plant, item, 1}, {LOp::Water, item, 1}};
  return {};
}

bool tile_effect_changed(const Tile& before, const Tile& after) {
  return !same_tile(before, after);
}

bool predicted_operation(Op op) {
  return op == Op::DROP || op == Op::PICKUP || op == Op::PLACE ||
      op == Op::FERTILIZE || op == Op::FEED ||
      op == Op::COLLECT_FERTILIZER || op == Op::HARVEST;
}

bool shed_adjacent(const Simulator& simulator, Position position) {
  const int half = simulator.config().board_size / 2;
  return (position.x == half - 1 || position.x == half) &&
      (position.y == half - 1 || position.y == half);
}

effect::LocalState effect_state(const Simulator& simulator, int player,
                                int actor) {
  effect::LocalState state;
  state.config = simulator.config();
  state.day = simulator.day();
  state.actor_slot = actor;
  const auto& farm = simulator.farms()[player];
  state.actor_exists = actor >= 0 && actor <= static_cast<int>(farm.hands.size());
  if (!state.actor_exists) return state;
  state.actor_position = actor_position(simulator, player, actor);
  state.shed_adjacent = shed_adjacent(simulator, state.actor_position);
  if (const auto* tile = tile_at(simulator, player, state.actor_position))
    state.tile = *tile;
  const auto& private_state = simulator.privates()[player];
  state.shed = private_state.shed;
  if (actor < static_cast<int>(private_state.inventories.size()))
    state.inventory = private_state.inventories[static_cast<std::size_t>(actor)];
  if (actor < static_cast<int>(private_state.inventory_order.size()))
    state.inventory_order =
        private_state.inventory_order[static_cast<std::size_t>(actor)];
  return state;
}

void add_cell(TypedEffectCertificate& certificate, TypedStateDomain domain,
              int actor, LPosition tile, int item, TypedTileField field,
              int before, int after) {
  if (before == after) return;
  certificate.cells.push_back(
      {domain, actor, tile, item, field, before, after});
}

std::uint64_t prefix_hash(const effect::LocalState& state) {
  std::uint64_t hash = 1469598103934665603ULL;
  auto add = [&](std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  };
  add(static_cast<std::uint64_t>(state.day));
  add(static_cast<std::uint64_t>(state.actor_slot));
  add(static_cast<std::uint64_t>(state.actor_position.x));
  add(static_cast<std::uint64_t>(state.actor_position.y));
  add(static_cast<std::uint64_t>(state.tile.kind));
  add(static_cast<std::uint64_t>(static_cast<int>(state.tile.crop) + 1));
  add(static_cast<std::uint64_t>(static_cast<int>(state.tile.animal) + 1));
  add(static_cast<std::uint64_t>(state.tile.planted_day));
  add(static_cast<std::uint64_t>(state.tile.placed_day));
  add(static_cast<std::uint64_t>(state.tile.yield_units));
  add(static_cast<std::uint64_t>(state.tile.fertilized_until_day));
  add(static_cast<std::uint64_t>(state.tile.watered_today));
  add(static_cast<std::uint64_t>(state.tile.fed_today));
  add(static_cast<std::uint64_t>(state.tile.cared_today));
  add(static_cast<std::uint64_t>(state.tile.fertilizer_available));
  for (const int value : state.shed) add(static_cast<std::uint64_t>(value));
  for (const int value : state.inventory) add(static_cast<std::uint64_t>(value));
  for (const int value : state.inventory_order)
    add(static_cast<std::uint64_t>(value + 1));
  return hash == 0 ? 1 : hash;
}

TypedEffectCertificate typed_certificate(
    int actor, std::uint64_t source_id, const effect::LocalState& before,
    const effect::Prediction& prediction, const Action& raw) {
  TypedEffectCertificate certificate;
  certificate.actor = actor;
  certificate.source_id = source_id;
  certificate.transferred = prediction.transferred;
  certificate.lower_slot_prefix_bound = true;
  certificate.prefix_hash = prefix_hash(before);
  certificate.actor_generation = static_cast<std::uint64_t>(before.day + 1);
  const auto actor_key = [&](int item) {
    return 100'000 + actor * 100 + item;
  };
  const auto shed_key = [](int item) { return 200'000 + item; };
  const int tile_key = 300'000 +
      (before.actor_position.y * 1024 + before.actor_position.x) * 32;
  const int item = static_cast<int>(raw.item);
  if (raw.op == Op::DROP) {
    for (int index = 0; index < N_ITEMS; ++index) {
      certificate.dependency_keys.insert(actor_key(index));
      certificate.dependency_keys.insert(shed_key(index));
    }
  } else if (raw.op == Op::PICKUP || raw.op == Op::PLACE) {
    if (item >= 0 && item < N_ITEMS) {
      certificate.dependency_keys.insert(actor_key(item));
      certificate.dependency_keys.insert(shed_key(item));
    }
    if (raw.op == Op::PLACE)
      for (int index = 0; index < N_ITEMS; ++index)
        certificate.dependency_keys.insert(shed_key(index));
  } else if (raw.op == Op::FERTILIZE ||
             raw.op == Op::COLLECT_FERTILIZER) {
    certificate.dependency_keys.insert(actor_key(8));
  } else if (raw.op == Op::FEED) {
    certificate.dependency_keys.insert(actor_key(0));
  } else if (raw.op == Op::HARVEST) {
    int product = -1;
    if (before.tile.kind == TileKind::PLANT)
      product = static_cast<int>(before.tile.crop);
    else if (before.tile.kind == TileKind::ANIMAL) {
      static constexpr int animal_product[N_ANIMALS] = {5, 6, 7};
      const int animal = static_cast<int>(before.tile.animal) - 9;
      if (animal >= 0 && animal < N_ANIMALS) product = animal_product[animal];
    }
    if (product >= 0) certificate.dependency_keys.insert(actor_key(product));
  }
  if (raw.op == Op::PLACE || raw.op == Op::FERTILIZE ||
      raw.op == Op::FEED || raw.op == Op::COLLECT_FERTILIZER ||
      raw.op == Op::HARVEST)
    certificate.dependency_keys.insert(tile_key);
  const auto tile = local_position(before.actor_position);
  for (int item = 0; item < N_ITEMS; ++item) {
    add_cell(certificate, TypedStateDomain::ActorInventory, actor, {}, item,
             TypedTileField::Kind, before.inventory[item],
             prediction.after.inventory[item]);
    add_cell(certificate, TypedStateDomain::Shed, -1, {}, item,
             TypedTileField::Kind, before.shed[item],
             prediction.after.shed[item]);
  }
  const auto tile_cell = [&](TypedTileField field, int left, int right) {
    add_cell(certificate, TypedStateDomain::Tile, -1, tile, -1, field,
             left, right);
  };
  tile_cell(TypedTileField::Kind, static_cast<int>(before.tile.kind),
            static_cast<int>(prediction.after.tile.kind));
  tile_cell(TypedTileField::Crop, static_cast<int>(before.tile.crop),
            static_cast<int>(prediction.after.tile.crop));
  tile_cell(TypedTileField::Animal, static_cast<int>(before.tile.animal),
            static_cast<int>(prediction.after.tile.animal));
  tile_cell(TypedTileField::PlantedDay, before.tile.planted_day,
            prediction.after.tile.planted_day);
  tile_cell(TypedTileField::PlacedDay, before.tile.placed_day,
            prediction.after.tile.placed_day);
  tile_cell(TypedTileField::YieldUnits, before.tile.yield_units,
            prediction.after.tile.yield_units);
  tile_cell(TypedTileField::ConsecutiveUnwatered,
            before.tile.consecutive_unwatered,
            prediction.after.tile.consecutive_unwatered);
  tile_cell(TypedTileField::ConsecutiveUnfed,
            before.tile.consecutive_unfed,
            prediction.after.tile.consecutive_unfed);
  tile_cell(TypedTileField::FertilizedUntilDay,
            before.tile.fertilized_until_day,
            prediction.after.tile.fertilized_until_day);
  tile_cell(TypedTileField::PendingCareBonus,
            before.tile.pending_care_bonus,
            prediction.after.tile.pending_care_bonus);
  tile_cell(TypedTileField::MaxLifespanStep,
            before.tile.max_lifespan_step,
            prediction.after.tile.max_lifespan_step);
  tile_cell(TypedTileField::WateredToday, before.tile.watered_today,
            prediction.after.tile.watered_today);
  tile_cell(TypedTileField::FedToday, before.tile.fed_today,
            prediction.after.tile.fed_today);
  tile_cell(TypedTileField::CaredToday, before.tile.cared_today,
            prediction.after.tile.cared_today);
  tile_cell(TypedTileField::FertilizerAvailable,
            before.tile.fertilizer_available,
            prediction.after.tile.fertilizer_available);
  certificate.inventory_order_before.assign(before.inventory_order.begin(),
                                            before.inventory_order.end());
  certificate.inventory_order_after.assign(
      prediction.after.inventory_order.begin(),
      prediction.after.inventory_order.end());
  return certificate;
}

}  // namespace

MappedUnit map_current_unit(const Simulator& before, int player, int actor,
                            const PlayerAction& player_raw,
                            std::uint64_t source_id, int earliest_turn) {
  if (player < 0 || player > 1 || actor < 0 ||
      actor >= static_cast<int>(player_raw.units.size()))
    throw std::invalid_argument("invalid shadow mapper actor");
  const Action raw = player_raw.units[static_cast<std::size_t>(actor)];
  MappedUnit result;
  result.native_op = raw.op;
  if (!native_unit_op(raw.op)) {
    result.kind = CertificateKind::GlobalOpaque;
    result.raw_source.emplace(source_id, earliest_turn, local_action(raw), true);
    return result;
  }
  if (raw.op == Op::PASS) return result;

  Simulator actor_before = before;
  if (predicted_operation(raw.op) && actor > 0) {
    std::array<PlayerAction, 2> prefix{};
    prefix[static_cast<std::size_t>(player)].units.resize(
        player_raw.units.size());
    for (int lower = 0; lower < actor; ++lower)
      prefix[static_cast<std::size_t>(player)]
          .units[static_cast<std::size_t>(lower)] =
          player_raw.units[static_cast<std::size_t>(lower)];
    actor_before = before.preview_unit_phase(prefix);
  }
  std::array<PlayerAction, 2> full{};
  full[static_cast<std::size_t>(player)] = player_raw;
  const Simulator full_after = before.preview_unit_phase(full);
  std::array<PlayerAction, 2> causal{};
  causal[static_cast<std::size_t>(player)].units.resize(player_raw.units.size());
  causal[static_cast<std::size_t>(player)]
      .units[static_cast<std::size_t>(actor)] = raw;
  const Simulator causal_after = actor_before.preview_unit_phase(causal);
  const Position position_before = actor_position(actor_before, player, actor);
  const Position position_after = actor_position(causal_after, player, actor);
  const Position full_position_after = actor_position(full_after, player, actor);
  const Tile* tile_before = tile_at(actor_before, player, position_before);
  const Tile* tile_after = tile_at(causal_after, player, position_before);
  const Tile* full_tile_after = tile_at(full_after, player, position_before);

  if (movement(raw.op)) {
    const bool moved = position_after.x != position_before.x ||
        position_after.y != position_before.y;
    if (!moved) {
      result.kind = CertificateKind::ExactNoEffect;
      return result;
    }
    if (position_after.x != full_position_after.x ||
        position_after.y != full_position_after.y) {
      result.kind = CertificateKind::GlobalOpaque;
      result.raw_source.emplace(source_id, earliest_turn,
                                LAction{LOp::Other, -1, 1,
                                        static_cast<int>(raw.op), 0}, true);
      return result;
    }
    result.kind = CertificateKind::MoveTransition;
    result.causal_effect = true;
    result.raw_source.emplace(source_id, earliest_turn, local_action(raw), false);
    return result;
  }

  if (!tile_before || !tile_after || !full_tile_after) {
    result.kind = CertificateKind::GlobalOpaque;
    result.raw_source.emplace(source_id, earliest_turn, local_action(raw), true);
    return result;
  }
  const bool causal_tile_effect = tile_effect_changed(*tile_before, *tile_after);
  const bool causal_private_effect =
      !private_actor_equal(actor_before, causal_after, player, actor);
  result.causal_effect = causal_tile_effect || causal_private_effect;
  bool manifest_causal = predicted_operation(raw.op) ||
      same_tile(*tile_after, *full_tile_after);
  if (raw.op == Op::PLANT && static_cast<int>(raw.item) >= 0 &&
      static_cast<int>(raw.item) < N_CROPS)
    manifest_causal = manifest_causal &&
        causal_after.privates()[player].seeds[static_cast<int>(raw.item)] ==
        full_after.privates()[player].seeds[static_cast<int>(raw.item)];
  if (raw.op == Op::HARVEST && actor >= 0 &&
      actor < static_cast<int>(causal_after.privates()[player]
                                   .inventories.size()) &&
      actor < static_cast<int>(full_after.privates()[player]
                                   .inventories.size()))
    manifest_causal = manifest_causal &&
        causal_after.privates()[player]
                .inventories[static_cast<std::size_t>(actor)] ==
            full_after.privates()[player]
                .inventories[static_cast<std::size_t>(actor)];

  auto replaceable_crop = [&] {
    const auto transitions = crop_recovery(*tile_before, raw);
    if (transitions.empty()) return false;
    result.kind = CertificateKind::ReplaceableObjective;
    result.objective = ObjectiveDraft{
        local_position(position_before), transitions,
        raw.op == Op::HARVEST ? 1000 : 500};
    return true;
  };
  auto opaque = [&] {
    result.kind = CertificateKind::GlobalOpaque;
    result.raw_source.emplace(
        source_id, earliest_turn,
        LAction{LOp::Other, static_cast<int>(raw.item), raw.quantity,
                static_cast<int>(raw.op), 0}, true);
  };

  if (predicted_operation(raw.op)) {
    const auto exact_before = effect_state(actor_before, player, actor);
    const auto prediction = effect::predict_unit_effect(exact_before, raw);
    result.effect_failure = prediction.failure;
    if (!(prediction.after == effect_state(causal_after, player, actor))) {
      opaque();
      return result;
    }
    if (prediction.changed) {
      result.kind = CertificateKind::TypedEffect;
      result.typed_effect = typed_certificate(
          actor, source_id, exact_before, prediction, raw);
      UnsupportedEffectCertificate certificate;
      certificate.scope = tile_effect_changed(exact_before.tile,
                                              prediction.after.tile)
          ? UnsupportedEffectScope::TileLocal
          : UnsupportedEffectScope::ActorLocal;
      certificate.no_unlisted_global_effects = true;
      certificate.resource_reads.insert(
          result.typed_effect->dependency_keys.begin(),
          result.typed_effect->dependency_keys.end());
      for (const auto& cell : result.typed_effect->cells) {
        int key = 0;
        if (cell.domain == TypedStateDomain::ActorInventory)
          key = 100'000 + cell.actor * 100 + cell.item;
        else if (cell.domain == TypedStateDomain::Shed)
          key = 200'000 + cell.item;
        else
          key = 300'000 + (cell.tile.row * 1024 + cell.tile.column) * 32 +
              static_cast<int>(cell.tile_field);
        certificate.resource_reads.insert(key);
      }
      std::optional<RawTileTransitionCertificate> tile_transition;
      const bool harvested_to_empty = raw.op == Op::HARVEST &&
          exact_before.tile.kind == TileKind::PLANT &&
          prediction.after.tile.kind == TileKind::EMPTY;
      if (harvested_to_empty)
        tile_transition = RawTileTransitionCertificate{
            local_tile(exact_before.tile, true),
            local_tile(prediction.after.tile)};
      if (tile_transition.has_value())
        result.raw_source.emplace(source_id, earliest_turn, local_action(raw),
                                  true, std::move(certificate),
                                  *tile_transition);
      else
        result.raw_source.emplace(source_id, earliest_turn, local_action(raw),
                                  true, std::move(certificate));
      return result;
    }
  }

  if (!result.causal_effect) {
    if ((raw.op == Op::PLANT || raw.op == Op::WATER ||
         raw.op == Op::HARVEST) &&
        replaceable_crop())
      return result;
    if (raw.op == Op::HARVEST && tile_before->kind == TileKind::PLANT) {
      result.kind = CertificateKind::MaturityWait;
      return result;
    }
    result.kind = CertificateKind::ExactNoEffect;
    return result;
  }
  if (!manifest_causal) {
    opaque();
    return result;
  }

  const bool harvested_to_empty = raw.op == Op::HARVEST &&
      tile_before->kind == TileKind::PLANT &&
      tile_after->kind == TileKind::EMPTY;
  if (raw.op == Op::DIG || raw.op == Op::PLANT || raw.op == Op::WATER ||
      raw.op == Op::BUILD_COOP || raw.op == Op::BUILD_PASTURE ||
      harvested_to_empty) {
    const bool harvest_legal = harvested_to_empty;
    const RawTileTransitionCertificate certificate{
        local_tile(*tile_before, harvest_legal), local_tile(*tile_after)};
    result.kind = CertificateKind::TileTransition;
    result.raw_source.emplace(source_id, earliest_turn, local_action(raw), false,
                              std::nullopt, certificate);
    return result;
  }
  if (raw.op == Op::CARE) {
    UnsupportedEffectCertificate certificate;
    certificate.scope = UnsupportedEffectScope::TileLocal;
    certificate.no_unlisted_global_effects = true;
    result.kind = CertificateKind::ScopedEffect;
    result.raw_source.emplace(source_id, earliest_turn, local_action(raw), true,
                              std::move(certificate));
    return result;
  }
  opaque();
  return result;
}

const char* certificate_kind_name(CertificateKind kind) {
  switch (kind) {
    case CertificateKind::ExactNoEffect: return "exact_no_effect";
    case CertificateKind::MoveTransition: return "move_transition";
    case CertificateKind::TileTransition: return "tile_transition";
    case CertificateKind::ScopedEffect: return "scoped_effect";
    case CertificateKind::ReplaceableObjective: return "replaceable_objective";
    case CertificateKind::MaturityWait: return "maturity_wait";
    case CertificateKind::TypedEffect: return "typed_effect";
    case CertificateKind::GlobalOpaque: return "global_opaque";
  }
  return "invalid";
}

const char* unit_op_name(Op op) {
  switch (op) {
    case Op::PASS: return "PASS";
    case Op::NORTH: return "NORTH";
    case Op::SOUTH: return "SOUTH";
    case Op::EAST: return "EAST";
    case Op::WEST: return "WEST";
    case Op::DROP: return "DROP";
    case Op::PICKUP: return "PICKUP";
    case Op::PLACE: return "PLACE";
    case Op::PLANT: return "PLANT";
    case Op::WATER: return "WATER";
    case Op::HARVEST: return "HARVEST";
    case Op::FERTILIZE: return "FERTILIZE";
    case Op::DIG: return "DIG";
    case Op::BUILD_COOP: return "BUILD_COOP";
    case Op::BUILD_PASTURE: return "BUILD_PASTURE";
    case Op::FEED: return "FEED";
    case Op::COLLECT_FERTILIZER: return "COLLECT_FERTILIZER";
    case Op::CARE: return "CARE";
    default: return "MARKET_ONLY";
  }
}

bool native_unit_op(Op op) {
  return static_cast<int>(op) >= static_cast<int>(Op::PASS) &&
      static_cast<int>(op) <= static_cast<int>(Op::CARE);
}

}  // namespace fastkag::component_shadow
