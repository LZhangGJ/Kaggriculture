#include "g001_typed_intent_issuer.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace g001::typed_intent {
namespace {

bool crop(fastkag::Item item) {
  const int value = static_cast<int>(item);
  return value >= 0 && value < fastkag::N_CROPS;
}

bool animal(fastkag::Item item) {
  const int value = static_cast<int>(item);
  return value >= fastkag::N_PRODUCTS && value < fastkag::N_ITEMS;
}

fastkag::Position actor_position(const fastkag::Simulator& simulator,
                                 int player, int actor) {
  const auto& farm = simulator.farms().at(static_cast<std::size_t>(player));
  if (actor == 0) return farm.farmer;
  if (actor > 0 && actor <= static_cast<int>(farm.hands.size()))
    return farm.hands[static_cast<std::size_t>(actor - 1)];
  return {-1, -1};
}

const fastkag::Tile* tile_at(const fastkag::Simulator& simulator, int player,
                             fastkag::Position position) {
  const int n = simulator.config().board_size;
  if (player < 0 || player > 1 || position.x < 0 || position.x >= n ||
      position.y < 0 || position.y >= n)
    return nullptr;
  return &simulator.farms()[player].tiles[
      static_cast<std::size_t>(position.y * n + position.x)];
}

production_obligation::UnitOp unit_op(fastkag::Op op) {
  return static_cast<production_obligation::UnitOp>(static_cast<int>(op));
}

fastkag::Item native_item(production_obligation::Item item) {
  return static_cast<fastkag::Item>(static_cast<int>(item));
}

std::uint64_t id(const Request& request, int actor,
                 fastkag::Position position, fastkag::Op op,
                 fastkag::Item item) {
  std::uint64_t hash = 1469598103934665603ULL;
  auto add = [&](std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= static_cast<std::uint8_t>(value >> (8 * byte));
      hash *= 1099511628211ULL;
    }
  };
  add(request.issuer_generation);
  add(static_cast<std::uint64_t>(request.observation->step_count()));
  add(static_cast<std::uint64_t>(request.player));
  add(static_cast<std::uint64_t>(actor));
  add(static_cast<std::uint64_t>(position.x));
  add(static_cast<std::uint64_t>(position.y));
  add(static_cast<std::uint8_t>(op));
  add(static_cast<std::uint8_t>(item));
  return hash ? hash : 1;
}

struct Resolution {
  fastkag::Item item{fastkag::Item::NONE};
  Proof proof{Proof::ActionItemMissing};
  bool exact{};
};

Resolution resolve_crop(const Request& request, int actor,
                        const fastkag::Action& action,
                        fastkag::Position position) {
  if (crop(action.item))
    return {action.item, Proof::DirectActionItem, true};
  const auto* tile = tile_at(*request.observation, request.player, position);
  if (tile && tile->kind == fastkag::TileKind::PLANT && crop(tile->crop))
    return {tile->crop, Proof::LivePlantTile, true};
  if (!request.dag_suffix_certified)
    return {fastkag::Item::NONE, Proof::DynamicSuffixUncertified, false};

  std::vector<fastkag::Item> exact;
  bool typed_without_position = false;
  for (const auto& node : request.dag_nodes) {
    if (node.actor != actor ||
        node.consumer_step != request.observation->step_count() ||
        node.original_unit_op != unit_op(action.op))
      continue;
    const auto item = native_item(node.item);
    if (!crop(item)) continue;
    if (node.position.x == position.x && node.position.y == position.y)
      exact.push_back(item);
    else
      typed_without_position = true;
  }
  std::sort(exact.begin(), exact.end());
  exact.erase(std::unique(exact.begin(), exact.end()), exact.end());
  if (exact.size() == 1)
    return {exact.front(), Proof::DagTypedPosition, true};
  if (exact.size() > 1)
    return {fastkag::Item::NONE, Proof::DagAmbiguous, false};
  if (typed_without_position)
    return {fastkag::Item::NONE, Proof::DagPositionMissing, false};
  if (!tile || tile->kind == fastkag::TileKind::WEED ||
      tile->kind == fastkag::TileKind::EMPTY)
    return {fastkag::Item::NONE, Proof::LiveTileHasNoCrop, false};
  return {fastkag::Item::NONE, Proof::DagNodeMissing, false};
}

}  // namespace

Result issue(const Request& request) {
  if (!request.observation || request.player < 0 || request.player > 1 ||
      request.observation->done())
    throw std::invalid_argument("invalid typed-intent issuer request");
  Result out;
  const int step = request.observation->step_count();
  for (std::size_t index = 0; index < request.final_current_units.size(); ++index) {
    const int actor = static_cast<int>(index);
    const auto& action = request.final_current_units[index];
    if (action.op != fastkag::Op::PLANT && action.op != fastkag::Op::WATER &&
        action.op != fastkag::Op::HARVEST && action.op != fastkag::Op::PLACE)
      continue;
    Evidence evidence;
    evidence.player = request.player;
    evidence.actor = actor;
    evidence.source_step = step;
    evidence.source_action = action;
    evidence.tile = actor_position(*request.observation, request.player, actor);
    if (evidence.tile.x < 0) {
      evidence.proof = Proof::InvalidActor;
      out.evidence.push_back(evidence);
      continue;
    }

    if (action.op == fastkag::Op::PLACE) {
      if (animal(action.item)) {
        evidence.proof = Proof::DirectActionItem;
        evidence.resolved_item = action.item;
        evidence.exact = evidence.emitted = true;
        out.animals.push_back({
            id(request, actor, evidence.tile, action.op, action.item),
            request.player, actor, step,
            {evidence.tile.y, evidence.tile.x}, action, action.item});
      } else {
        evidence.proof = action.item == fastkag::Item::NONE
            ? Proof::ActionItemMissing : Proof::ActionItemNotAnimal;
      }
      out.evidence.push_back(evidence);
      continue;
    }

    const auto resolved = resolve_crop(request, actor, action, evidence.tile);
    evidence.proof = resolved.proof;
    evidence.resolved_item = resolved.item;
    evidence.exact = resolved.exact;
    if (resolved.exact) {
      evidence.emitted = true;
      out.crops.push_back({
          id(request, actor, evidence.tile, action.op, resolved.item),
          request.player, actor, step, evidence.tile, action, resolved.item});
    }
    out.evidence.push_back(evidence);
  }
  return out;
}

const char* proof_name(Proof proof) noexcept {
  switch (proof) {
    case Proof::DirectActionItem: return "direct_action_item";
    case Proof::LivePlantTile: return "live_plant_tile";
    case Proof::DagTypedPosition: return "dag_typed_position";
    case Proof::InvalidActor: return "invalid_actor";
    case Proof::ActionItemMissing: return "action_item_missing";
    case Proof::ActionItemNotAnimal: return "action_item_not_animal";
    case Proof::LiveTileHasNoCrop: return "live_tile_has_no_crop";
    case Proof::DagNodeMissing: return "dag_node_missing";
    case Proof::DagPositionMissing: return "dag_position_missing";
    case Proof::DagAmbiguous: return "dag_ambiguous";
    case Proof::DynamicSuffixUncertified: return "dynamic_suffix_uncertified";
    case Proof::UnsupportedAction: return "unsupported_action";
  }
  return "unknown";
}

}  // namespace g001::typed_intent
