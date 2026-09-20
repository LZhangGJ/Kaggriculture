#include "minimum_damage_scheduler_bridge.hpp"

#include "production_suffix_scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace g001::minimum_damage_bridge {
namespace {

namespace md = minimum_damage;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;

bool same_action(Action left, Action right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
         op == Op::WEST;
}

bool market(Op op) {
  return op == Op::BUY_SEED || op == Op::BUY_ANIMAL ||
         op == Op::BUY_PRODUCT;
}

bool crop(Item item) {
  const int value = static_cast<int>(item);
  return value >= 0 && value < fastkag::N_CROPS;
}

bool animal(Item item) {
  const int value = static_cast<int>(item);
  return value >= static_cast<int>(Item::GOOSE) &&
         value <= static_cast<int>(Item::SHEEP);
}

void add_hash(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= static_cast<std::uint8_t>(value >> (byte * 8));
    hash *= 1099511628211ULL;
  }
}

void add_action_hash(std::uint64_t& hash, Action action) {
  add_hash(hash, static_cast<std::uint8_t>(action.op));
  add_hash(hash, static_cast<std::uint8_t>(action.item));
  add_hash(hash, static_cast<std::uint32_t>(action.quantity));
}

md::Position to_md(fastkag::Position position) {
  return {position.x, position.y};
}

std::optional<md::TileKind> to_md(fastkag::TileKind kind) {
  switch (kind) {
    case fastkag::TileKind::EMPTY: return md::TileKind::Empty;
    case fastkag::TileKind::LOCKED: return md::TileKind::Blocked;
    case fastkag::TileKind::WEED: return md::TileKind::Weed;
    case fastkag::TileKind::PLANT: return md::TileKind::Crop;
    case fastkag::TileKind::COOP: return md::TileKind::OtherStructure;
    case fastkag::TileKind::PASTURE: return md::TileKind::Pasture;
    case fastkag::TileKind::ANIMAL: return md::TileKind::Animal;
  }
  return std::nullopt;
}

std::optional<md::Action> to_md(Action action) {
  md::Action output;
  output.item = static_cast<int>(action.item);
  output.quantity = action.quantity;
  switch (action.op) {
    case Op::PASS: output.kind = md::ActionKind::Pass; break;
    case Op::NORTH: output.kind = md::ActionKind::North; break;
    case Op::SOUTH: output.kind = md::ActionKind::South; break;
    case Op::EAST: output.kind = md::ActionKind::East; break;
    case Op::WEST: output.kind = md::ActionKind::West; break;
    case Op::DIG: output.kind = md::ActionKind::Dig; break;
    case Op::BUILD_PASTURE:
      output.kind = md::ActionKind::BuildPasture;
      break;
    case Op::BUILD_COOP: output.kind = md::ActionKind::BuildCoop; break;
    case Op::PICKUP: output.kind = md::ActionKind::Pickup; break;
    case Op::PLANT: output.kind = md::ActionKind::Plant; break;
    case Op::WATER: output.kind = md::ActionKind::Water; break;
    case Op::COLLECT_FERTILIZER:
      output.kind = md::ActionKind::CollectFertilizer;
      break;
    case Op::PLACE: output.kind = md::ActionKind::PlaceAnimal; break;
    case Op::FEED: output.kind = md::ActionKind::Feed; break;
    case Op::CARE: output.kind = md::ActionKind::Care; break;
    case Op::BUY_SEED: output.kind = md::ActionKind::BuySeed; break;
    case Op::BUY_ANIMAL: output.kind = md::ActionKind::BuyAnimal; break;
    case Op::BUY_PRODUCT:
      if (action.item != Item::WHEAT) return std::nullopt;
      output.kind = md::ActionKind::BuyFeed;
      break;
    default: return std::nullopt;
  }
  return output;
}

std::optional<Action> to_fast(md::Action action) {
  const auto item = action.item >= -1 && action.item < fastkag::N_ITEMS
                        ? static_cast<Item>(action.item)
                        : Item::NONE;
  if (action.item < -1 || action.item >= fastkag::N_ITEMS ||
      action.quantity <= 0)
    return std::nullopt;
  switch (action.kind) {
    case md::ActionKind::Pass: return Action{Op::PASS, Item::NONE, 1};
    case md::ActionKind::North: return Action{Op::NORTH, Item::NONE, 1};
    case md::ActionKind::South: return Action{Op::SOUTH, Item::NONE, 1};
    case md::ActionKind::East: return Action{Op::EAST, Item::NONE, 1};
    case md::ActionKind::West: return Action{Op::WEST, Item::NONE, 1};
    case md::ActionKind::Dig: return Action{Op::DIG, Item::NONE, 1};
    case md::ActionKind::BuildPasture:
      return Action{Op::BUILD_PASTURE, Item::NONE, 1};
    case md::ActionKind::BuildCoop:
      return Action{Op::BUILD_COOP, Item::NONE, 1};
    case md::ActionKind::Pickup:
      if (item == Item::NONE) return std::nullopt;
      return Action{Op::PICKUP, item, action.quantity};
    case md::ActionKind::Plant:
      if (!crop(item)) return std::nullopt;
      return Action{Op::PLANT, item, action.quantity};
    case md::ActionKind::Water:
      if (!crop(item)) return std::nullopt;
      return Action{Op::WATER, item, action.quantity};
    case md::ActionKind::CollectFertilizer:
      return Action{Op::COLLECT_FERTILIZER, Item::NONE, action.quantity};
    case md::ActionKind::PlaceAnimal:
      if (!animal(item)) return std::nullopt;
      return Action{Op::PLACE, item, action.quantity};
    case md::ActionKind::Feed:
      if (item != Item::WHEAT) return std::nullopt;
      return Action{Op::FEED, Item::WHEAT, action.quantity};
    case md::ActionKind::Care:
      if (!animal(item)) return std::nullopt;
      return Action{Op::CARE, item, action.quantity};
    case md::ActionKind::BuySeed:
      if (!crop(item)) return std::nullopt;
      return Action{Op::BUY_SEED, item, action.quantity};
    case md::ActionKind::BuyAnimal:
      if (!animal(item)) return std::nullopt;
      return Action{Op::BUY_ANIMAL, item, action.quantity};
    case md::ActionKind::BuyFeed:
      return Action{Op::BUY_PRODUCT, Item::WHEAT, action.quantity};
  }
  return std::nullopt;
}

const fastkag::Tile& tile_at(const fastkag::Simulator& simulator, int player,
                             fastkag::Position position) {
  return simulator.farms()[player].tiles[
      static_cast<std::size_t>(position.y * simulator.config().board_size +
                               position.x)];
}

struct Conversion {
  RejectReason reject{RejectReason::None};
  std::string diagnostic;
  md::Request selector;
  int start_step{-1};
  int end_step{-1};
  std::uint64_t input_hash{};
  std::map<int, Action> raw_actions;
  std::map<int, Action> raw_moves;
  std::map<int, int> selector_move_sources;
  std::map<std::uint64_t, const obligation_day::ProductionObligation*>
      native_obligations;
};

void fail(Conversion& output, RejectReason reason,
          const std::string& diagnostic) {
  if (output.reject == RejectReason::None) {
    output.reject = reason;
    output.diagnostic = diagnostic;
  }
}

const ObligationPolicy* policy_for(const Request& request,
                                   std::uint64_t id) {
  const ObligationPolicy* result = nullptr;
  for (const auto& policy : request.policies) {
    if (policy.obligation_id != id) continue;
    if (result) return nullptr;
    result = &policy;
  }
  return result;
}

bool raw_matches(const obligation_day::ProductionObligation& obligation,
                 Action raw) {
  if (raw.quantity <= 0 || obligation.quantity <= 0) return false;
  using Goal = obligation_day::GoalKind;
  switch (obligation.goal) {
    case Goal::Pickup:
      return raw.op == Op::PICKUP && raw.item == obligation.item &&
             raw.quantity == obligation.quantity &&
             obligation.item != Item::NONE;
    case Goal::Place:
      return raw.op == Op::PLACE && raw.item == obligation.item &&
             raw.quantity == obligation.quantity && animal(obligation.item);
    case Goal::Feed:
      return raw.op == Op::FEED && raw.quantity == 1 &&
             obligation.quantity == 1 &&
             (raw.item == Item::NONE || raw.item == Item::WHEAT) &&
             animal(obligation.item);
    case Goal::Care:
      return raw.op == Op::CARE && raw.quantity == 1 &&
             obligation.quantity == 1 &&
             (raw.item == Item::NONE || raw.item == obligation.item) &&
             animal(obligation.item);
    case Goal::BuildPasture:
      return raw.op == Op::BUILD_PASTURE && raw.item == Item::NONE &&
             raw.quantity == 1 && obligation.quantity == 1 &&
             obligation.item == Item::NONE;
    case Goal::BuildCoop:
      return raw.op == Op::BUILD_COOP && raw.item == Item::NONE &&
             raw.quantity == 1 && obligation.quantity == 1 &&
             obligation.item == Item::NONE;
    case Goal::CropReady:
      return (raw.op == Op::DIG || raw.op == Op::PLANT ||
              raw.op == Op::WATER) &&
             raw.quantity == obligation.quantity &&
             crop(obligation.item) &&
             ((raw.op == Op::PLANT && raw.item == obligation.item) ||
              (raw.op == Op::WATER &&
               (raw.item == Item::NONE || raw.item == obligation.item)) ||
              (raw.op == Op::DIG && raw.item == Item::NONE));
    case Goal::Harvest:
      return false;
    case Goal::CollectFertilizer:
      return raw.op == Op::COLLECT_FERTILIZER &&
             raw.item == Item::NONE && raw.quantity == obligation.quantity &&
             obligation.item == Item::FERTILIZER;
  }
  return false;
}

std::optional<md::TypedObligation> convert_obligation(
    const Request& request,
    const obligation_day::ProductionObligation& obligation,
    const ObligationPolicy& policy, Action raw, int start_step,
    int end_step, bool carry_in = false) {
  if (obligation.actor != request.actor || obligation.policy_deferred ||
      obligation.deadline_step > end_step ||
      obligation.deadline_step < start_step ||
      obligation.earliest_step > obligation.deadline_step ||
      !raw_matches(obligation, raw))
    return std::nullopt;
  md::TypedObligation output;
  output.id = obligation.id;
  output.actor = 0;
  output.release_slot =
      carry_in ? 0 : std::max(obligation.earliest_step, start_step) - start_step;
  output.deadline_slot = obligation.deadline_step - start_step;
  output.source_slot = carry_in ? -1 : obligation.source_step - start_step;
  output.must_finish = obligation.must_finish_today;
  output.important = policy.important;
  output.cascade_risk = policy.cross_day_cascade_risk;
  output.dependencies = obligation.dependencies;
  const auto tile = to_md(obligation.tile);
  const auto add = [&](md::Action action) {
    output.transitions.push_back(
        {md::Channel::Unit, action, true, tile, false, 0});
  };
  using Goal = obligation_day::GoalKind;
  switch (obligation.goal) {
    case Goal::Pickup:
      output.kind = md::ObligationKind::RawProduction;
      add({md::ActionKind::Pickup, static_cast<int>(obligation.item),
           obligation.quantity});
      break;
    case Goal::Place:
      output.kind = md::ObligationKind::AnimalRecovery;
      add({md::ActionKind::PlaceAnimal, static_cast<int>(obligation.item),
           obligation.quantity});
      break;
    case Goal::Feed:
      output.kind = md::ObligationKind::AnimalCareRecovery;
      add({md::ActionKind::Feed, static_cast<int>(Item::WHEAT), 1});
      break;
    case Goal::Care:
      output.kind = md::ObligationKind::AnimalCareRecovery;
      add({md::ActionKind::Care, static_cast<int>(obligation.item), 1});
      break;
    case Goal::BuildPasture: {
      output.kind = md::ObligationKind::WeedPastureRecovery;
      const auto kind = tile_at(*request.remaining_day_start, request.player,
                                obligation.tile)
                            .kind;
      if (kind == fastkag::TileKind::WEED)
        add({md::ActionKind::Dig, -1, 1});
      else if (kind != fastkag::TileKind::EMPTY)
        return std::nullopt;
      add({md::ActionKind::BuildPasture, -1, 1});
      break;
    }
    case Goal::BuildCoop: {
      output.kind = md::ObligationKind::WeedPastureRecovery;
      const auto kind = tile_at(*request.remaining_day_start, request.player,
                                obligation.tile)
                            .kind;
      if (kind == fastkag::TileKind::WEED)
        add({md::ActionKind::Dig, -1, 1});
      else if (kind != fastkag::TileKind::EMPTY)
        return std::nullopt;
      add({md::ActionKind::BuildCoop, -1, 1});
      break;
    }
    case Goal::CropReady: {
      output.kind = md::ObligationKind::ReplantRecovery;
      const auto& current = tile_at(*request.remaining_day_start,
                                    request.player, obligation.tile);
      if (raw.op == Op::DIG) {
        add({md::ActionKind::Dig, -1, 1});
        break;
      }
      const bool desired_crop = current.kind == fastkag::TileKind::PLANT &&
                                current.crop == obligation.item;
      if (!desired_crop && current.kind != fastkag::TileKind::EMPTY)
        add({md::ActionKind::Dig, -1, 1});
      if (!desired_crop)
        add({md::ActionKind::Plant, static_cast<int>(obligation.item),
             obligation.quantity});
      if (raw.op == Op::WATER)
        add({md::ActionKind::Water, static_cast<int>(obligation.item),
             obligation.quantity});
      break;
    }
    case Goal::Harvest:
      return std::nullopt;
    case Goal::CollectFertilizer:
      output.kind = md::ObligationKind::RawProduction;
      add({md::ActionKind::CollectFertilizer,
           static_cast<int>(Item::FERTILIZER), obligation.quantity});
      break;
  }
  return output.transitions.empty() ? std::nullopt
                                    : std::optional<md::TypedObligation>(output);
}

Conversion convert(const Request& request) {
  Conversion output;
  if (!request.remaining_day_start || !request.issued ||
      request.player < 0 || request.player >= 2 || request.actor < 0 ||
      request.issuer_generation == 0 || request.maximum_states == 0) {
    fail(output, request.actor < 0 ? RejectReason::ActorScope
                                  : RejectReason::InvalidInput,
         "invalid bridge envelope or actor scope");
    return output;
  }
  if (!request.issued->issued()) {
    fail(output, RejectReason::IssuerRejected, "day issuer did not issue");
    return output;
  }
  const auto& env = *request.remaining_day_start;
  const int day_start = env.day() * env.config().turns_per_day;
  output.start_step = env.step_count();
  output.end_step = day_start + env.config().turns_per_day - 1;
  const int horizon = output.end_step - output.start_step + 1;
  if (horizon <= 0 || env.done()) {
    fail(output, RejectReason::InvalidInput, "empty/terminal remaining day");
    return output;
  }
  const auto& farm = env.farms()[request.player];
  const auto& private_state = env.privates()[request.player];
  if (farm.hands.size() + 1 <= static_cast<std::size_t>(request.actor) ||
      private_state.inventories.size() != farm.hands.size() + 1 ||
      std::floor(farm.money) != farm.money || farm.money < 0 ||
      farm.money > std::numeric_limits<int>::max()) {
    fail(output, RejectReason::UnsupportedSnapshotState,
         "snapshot actor/inventory/cash cannot map losslessly");
    return output;
  }

  auto& selector = output.selector;
  selector.horizon = horizon;
  selector.maximum_states = request.maximum_states;
  selector.day_start.width = env.config().board_size;
  selector.day_start.height = env.config().board_size;
  selector.day_start.cash = static_cast<int>(farm.money);
  const auto actor_position = request.actor == 0
      ? farm.farmer
      : farm.hands[static_cast<std::size_t>(request.actor - 1)];
  selector.day_start.actor_positions.push_back(to_md(actor_position));
  selector.day_start.seed_inventory.assign(private_state.seeds.begin(),
                                           private_state.seeds.end());
  selector.day_start.animal_inventory.assign(fastkag::N_ITEMS, 0);
  selector.day_start.shed_inventory.assign(private_state.shed.begin(),
                                           private_state.shed.end());
  selector.day_start.actor_inventory.emplace_back(
      private_state.inventories[static_cast<std::size_t>(request.actor)].begin(),
      private_state.inventories[static_cast<std::size_t>(request.actor)].end());
  selector.raw_sources.assign(
      1, std::vector<md::SourceSlot>(static_cast<std::size_t>(horizon)));
  for (const auto& tile : farm.tiles) {
    const auto mapped = to_md(tile.kind);
    if (!mapped) {
      fail(output, RejectReason::UnsupportedSnapshotState,
           "unknown tile kind");
      return output;
    }
    selector.day_start.tiles.push_back(*mapped);
  }

  std::map<int, std::vector<const day_start_issuer::RawSourceEntry*>> all_raw;
  std::map<int, std::vector<const day_start_issuer::RawSourceEntry*>> raw;
  for (const auto& source : request.issued->raw_sources) {
    if (source.actor != request.actor) continue;
    all_raw[source.source_step].push_back(&source);
    if (source.source_step >= output.start_step &&
        source.source_step <= output.end_step)
      raw[source.source_step].push_back(&source);
  }
  std::map<int, std::vector<const obligation_day::MoveSourceToken*>> moves;
  for (const auto& token : request.issued->moves) {
    if (token.actor == request.actor && token.source_step >= output.start_step &&
        token.source_step <= output.end_step)
      moves[token.source_step].push_back(&token);
  }
  std::map<int, std::vector<const obligation_day::ProductionObligation*>>
      obligations_by_source;
  std::vector<const obligation_day::ProductionObligation*> prior_obligations;
  for (const auto& obligation : request.issued->obligations) {
    if (obligation.actor != request.actor) continue;
    if (obligation.source_step < output.start_step) {
      prior_obligations.push_back(&obligation);
      if (!output.native_obligations.emplace(obligation.id, &obligation)
               .second) {
        fail(output, RejectReason::MissingTypedIdentity,
             "duplicate production obligation identity");
        return output;
      }
      continue;
    }
    if (obligation.source_step > output.end_step) continue;
    obligations_by_source[obligation.source_step].push_back(&obligation);
    if (!output.native_obligations.emplace(obligation.id, &obligation).second) {
      fail(output, RejectReason::MissingTypedIdentity,
           "duplicate production obligation identity");
      return output;
    }
  }
  for (const auto& unsupported : request.issued->unsupported) {
    if (unsupported.actor == request.actor &&
        unsupported.source_step >= output.start_step &&
        unsupported.source_step <= output.end_step) {
      fail(output, RejectReason::UnsupportedSource,
           "issuer reported unsupported selected-actor source");
      return output;
    }
  }

  std::map<std::uint64_t, const PriorObligationProgress*> progress_by_id;
  for (const auto& progress : request.prior_obligation_progress) {
    if (progress.obligation_id == 0 || progress.original_source_step < 0 ||
        progress.effect_evidence_hash == 0 ||
        !progress_by_id.emplace(progress.obligation_id, &progress).second) {
      fail(output, RejectReason::PriorSourceRequiresReceipt,
           "malformed or duplicate prior obligation receipt");
      return output;
    }
  }
  if (progress_by_id.size() != prior_obligations.size()) {
    fail(output, RejectReason::PriorSourceRequiresReceipt,
         "prior obligation receipt coverage is not exact");
    return output;
  }
  for (const auto* obligation : prior_obligations) {
    const auto progress = progress_by_id.find(obligation->id);
    if (progress == progress_by_id.end() ||
        progress->second->original_source_step != obligation->source_step) {
      fail(output, RejectReason::PriorSourceRequiresReceipt,
           "prior obligation receipt identity/source mismatch");
      return output;
    }
    if (progress->second->completed) continue;
    const auto raw_source = all_raw.find(obligation->source_step);
    const auto* policy = policy_for(request, obligation->id);
    if (raw_source == all_raw.end() || raw_source->second.size() != 1 ||
        !policy || policy->cross_day_cascade_risk < 0) {
      fail(output, RejectReason::PriorSourceRequiresReceipt,
           "outstanding prior obligation lacks exact raw/policy identity");
      return output;
    }
    const auto mapped = convert_obligation(
        request, *obligation, *policy, raw_source->second[0]->action,
        output.start_step, output.end_step, true);
    if (!mapped) {
      std::string dependency_ids = "[";
      for (std::size_t index = 0; index < obligation->dependencies.size();
           ++index) {
        if (index) dependency_ids += ',';
        dependency_ids += std::to_string(obligation->dependencies[index]);
      }
      dependency_ids += ']';
      const auto& current_tile =
          tile_at(env, request.player, obligation->tile);
      fail(output, RejectReason::MissingTypedIdentity,
           "outstanding prior obligation cannot map losslessly id=" +
               std::to_string(obligation->id) + " goal=" +
               std::to_string(static_cast<int>(obligation->goal)) +
               " item=" +
               std::to_string(static_cast<int>(obligation->item)) +
               " quantity=" + std::to_string(obligation->quantity) +
               " actor=" + std::to_string(obligation->actor) +
               " tile=(" + std::to_string(obligation->tile.x) + "," +
               std::to_string(obligation->tile.y) + ")" +
               " raw=" + std::to_string(static_cast<int>(
                              raw_source->second[0]->action.op)) +
               " raw_item=" + std::to_string(static_cast<int>(
                                  raw_source->second[0]->action.item)) +
               " raw_quantity=" +
               std::to_string(raw_source->second[0]->action.quantity) +
               " tile_kind=" +
               std::to_string(static_cast<int>(current_tile.kind)) +
               " tile_animal=" +
               std::to_string(static_cast<int>(current_tile.animal)) +
               " earliest=" + std::to_string(obligation->earliest_step) +
               " deadline=" + std::to_string(obligation->deadline_step) +
               " dependencies=" + dependency_ids +
               " effect_failure=1 evidence_hash=" +
               std::to_string(progress->second->effect_evidence_hash));
      return output;
    }
    selector.repair_obligations.push_back(*mapped);
  }

  for (int step = output.start_step; step <= output.end_step; ++step) {
    if (raw[step].size() != 1) {
      fail(output, RejectReason::AmbiguousSourceMapping,
           "selected-actor raw source coverage is not exactly one per slot");
      return output;
    }
    const auto action = raw[step][0]->action;
    output.raw_actions.emplace(step, action);
    if (move(action.op)) {
      if (moves[step].size() != 1 || !same_action(moves[step][0]->action, action) ||
          !obligations_by_source[step].empty()) {
        fail(output, RejectReason::AmbiguousSourceMapping,
             "MOVE source/token/obligation classification mismatch");
        return output;
      }
      const auto mapped = to_md(action);
      if (!mapped) {
        fail(output, RejectReason::UnsupportedSource,
             "MOVE action cannot map");
        return output;
      }
      output.raw_moves.emplace(step, action);
    } else if (action.op == Op::PASS) {
      if (!moves[step].empty() || !obligations_by_source[step].empty()) {
        fail(output, RejectReason::AmbiguousSourceMapping,
             "PASS source has typed classifications");
        return output;
      }
    } else {
      if (!moves[step].empty() || obligations_by_source[step].size() != 1) {
        fail(output, RejectReason::AmbiguousSourceMapping,
             "production source lacks unique typed obligation");
        return output;
      }
      const auto& obligation = *obligations_by_source[step][0];
      const auto* policy = policy_for(request, obligation.id);
      if (!policy || policy->cross_day_cascade_risk < 0) {
        fail(output, RejectReason::MissingTypedIdentity,
             "production obligation lacks unique risk/importance policy");
        return output;
      }
      const auto mapped = convert_obligation(
          request, obligation, *policy, action, output.start_step,
          output.end_step);
      if (!mapped) {
        fail(output, RejectReason::MissingTypedIdentity,
             "production source/goal/item/tile cannot map losslessly");
        return output;
      }
      selector.repair_obligations.push_back(*mapped);
    }
  }

  std::map<int, const PriorMoveProgress*> prior_move_progress;
  for (const auto& progress : request.prior_move_progress) {
    if (progress.original_source_step < 0 ||
        progress.effect_evidence_hash == 0 ||
        !prior_move_progress.emplace(progress.original_source_step, &progress)
             .second) {
      fail(output, RejectReason::PriorSourceRequiresReceipt,
           "malformed or duplicate prior MOVE receipt");
      return output;
    }
  }
  int expected_prior_moves = 0;
  for (const auto& token : request.issued->moves) {
    if (token.actor != request.actor ||
        token.source_step >= output.start_step)
      continue;
    ++expected_prior_moves;
    const auto progress = prior_move_progress.find(token.source_step);
    const auto raw_source = all_raw.find(token.source_step);
    if (progress == prior_move_progress.end() ||
        raw_source == all_raw.end() || raw_source->second.size() != 1 ||
        !same_action(raw_source->second[0]->action, token.action)) {
      fail(output, RejectReason::PriorSourceRequiresReceipt,
           "prior MOVE receipt/source identity mismatch");
      return output;
    }
    if (!progress->second->completed)
      output.raw_moves.emplace(token.source_step, token.action);
  }
  if (expected_prior_moves != static_cast<int>(prior_move_progress.size())) {
    fail(output, RejectReason::PriorSourceRequiresReceipt,
         "prior MOVE receipt coverage is not exact");
    return output;
  }

  int assigned_move_slot = -1;
  for (const auto& [source_step, action] : output.raw_moves) {
    assigned_move_slot = std::max(
        assigned_move_slot + 1, std::max(0, source_step - output.start_step));
    const auto mapped = to_md(action);
    if (assigned_move_slot >= horizon || !mapped || !move(action.op)) {
      fail(output, RejectReason::SelectorRejected,
           "remaining horizon cannot close ordered prior/future MOVE tokens");
      return output;
    }
    selector.raw_sources[0][static_cast<std::size_t>(assigned_move_slot)]
        .action = *mapped;
    output.selector_move_sources.emplace(assigned_move_slot, source_step);
  }

  std::set<std::uint64_t> completed_prior_ids;
  for (const auto& [id, progress] : progress_by_id)
    if (progress->completed) completed_prior_ids.insert(id);
  for (auto& obligation : selector.repair_obligations) {
    std::erase_if(obligation.dependencies, [&](std::uint64_t dependency) {
      return completed_prior_ids.contains(dependency);
    });
  }

  std::set<std::uint64_t> ids;
  for (const auto& obligation : selector.repair_obligations)
    ids.insert(obligation.id);
  std::set<std::uint64_t> policy_ids;
  for (const auto& policy : request.policies) {
    if (policy.obligation_id == 0 || policy.cross_day_cascade_risk < 0 ||
        !ids.contains(policy.obligation_id) ||
        !policy_ids.insert(policy.obligation_id).second) {
      fail(output, RejectReason::MissingTypedIdentity,
           "obligation policy set is not a bijection for selected-actor sources");
      return output;
    }
  }
  if (policy_ids != ids) {
    fail(output, RejectReason::MissingTypedIdentity,
         "obligation policy set is not a bijection for selected-actor sources");
    return output;
  }
  for (const auto& retry : request.purchase_retries) {
    if (retry.id == 0 || ids.contains(retry.id) || retry.quantity <= 0 ||
        retry.release_step < output.start_step ||
        retry.release_step > output.end_step ||
        retry.worst_case_cash_cost < 0 ||
        retry.cross_day_cascade_risk < 0) {
      fail(output, RejectReason::MissingTypedIdentity,
           "invalid purchase retry identity/envelope");
      return output;
    }
    md::TypedObligation mapped;
    const int release = retry.release_step - output.start_step;
    if (retry.operation == Op::BUY_SEED && crop(retry.item)) {
      mapped = md::make_seed_purchase_retry(
          retry.id, static_cast<int>(retry.item), retry.quantity, release,
          retry.worst_case_cash_cost, retry.guaranteed_fill,
          retry.must_finish, retry.important,
          retry.cross_day_cascade_risk);
    } else if (retry.operation == Op::BUY_ANIMAL && animal(retry.item)) {
      mapped = md::make_animal_purchase_retry(
          retry.id, static_cast<int>(retry.item), retry.quantity, release,
          retry.worst_case_cash_cost, retry.guaranteed_fill,
          retry.must_finish, retry.important,
          retry.cross_day_cascade_risk);
    } else if (retry.operation == Op::BUY_PRODUCT &&
               retry.item == Item::WHEAT) {
      mapped = md::make_feed_purchase_retry(
          retry.id, retry.quantity, release, retry.worst_case_cash_cost,
          retry.guaranteed_fill, retry.must_finish, retry.important,
          retry.cross_day_cascade_risk);
    } else {
      fail(output, RejectReason::UnsupportedSource,
           "purchase retry operation/item is unsupported");
      return output;
    }
    mapped.deadline_slot = horizon - 1;
    selector.repair_obligations.push_back(mapped);
    ids.insert(mapped.id);
    for (const auto target : retry.unlocks_obligations) {
      const auto native_target = output.native_obligations.find(target);
      const bool typed_unlock =
          native_target != output.native_obligations.end() &&
          ((retry.operation == Op::BUY_SEED &&
            native_target->second->goal == obligation_day::GoalKind::CropReady &&
            native_target->second->item == retry.item) ||
           (retry.operation == Op::BUY_ANIMAL &&
            native_target->second->goal == obligation_day::GoalKind::Pickup &&
            native_target->second->item == retry.item) ||
           (retry.operation == Op::BUY_PRODUCT && retry.item == Item::WHEAT &&
            native_target->second->goal == obligation_day::GoalKind::Pickup &&
            native_target->second->item == Item::WHEAT));
      if (!typed_unlock) {
        fail(output, RejectReason::MissingTypedIdentity,
             "purchase retry unlock lacks exact resource-consumer identity");
        return output;
      }
      const auto found = std::find_if(
          selector.repair_obligations.begin(),
          selector.repair_obligations.end(),
          [&](const auto& obligation) { return obligation.id == target; });
      if (found == selector.repair_obligations.end()) {
        fail(output, RejectReason::MissingTypedIdentity,
             "purchase retry unlock target is unknown");
        return output;
      }
      found->dependencies.push_back(mapped.id);
    }
  }

  std::uint64_t hash = 1469598103934665603ULL;
  add_hash(hash, production_suffix::focal_unit_state_fingerprint(
                     env, request.player));
  add_hash(hash, request.player);
  add_hash(hash, request.actor);
  add_hash(hash, request.issuer_generation);
  for (const auto& [step, action] : output.raw_actions) {
    add_hash(hash, step);
    add_action_hash(hash, action);
  }
  for (const auto& progress : request.prior_obligation_progress) {
    add_hash(hash, progress.obligation_id);
    add_hash(hash, progress.original_source_step);
    add_hash(hash, progress.completed);
    add_hash(hash, progress.effect_evidence_hash);
  }
  for (const auto& progress : request.prior_move_progress) {
    add_hash(hash, progress.original_source_step);
    add_hash(hash, progress.completed);
    add_hash(hash, progress.effect_evidence_hash);
  }
  for (const auto& obligation : selector.repair_obligations) {
    add_hash(hash, obligation.id);
    add_hash(hash, obligation.actor);
    add_hash(hash, obligation.release_slot);
    add_hash(hash, obligation.deadline_slot);
    add_hash(hash, obligation.source_slot);
    add_hash(hash, obligation.must_finish);
    add_hash(hash, obligation.important);
    add_hash(hash, obligation.cascade_risk);
    for (const auto dependency : obligation.dependencies)
      add_hash(hash, dependency);
    for (const auto& transition : obligation.transitions) {
      add_hash(hash, static_cast<int>(transition.channel));
      add_hash(hash, transition.requires_position);
      add_hash(hash, transition.tile.x);
      add_hash(hash, transition.tile.y);
      add_hash(hash, transition.guaranteed_fill);
      add_hash(hash, transition.worst_case_cash_cost);
      const auto action = to_fast(transition.action);
      if (action) add_action_hash(hash, *action);
    }
  }
  output.input_hash = hash;
  output.diagnostic = "lossless selected-actor remaining-day conversion";
  return output;
}

obligation_day::DebtReason scheduler_reason(md::DebtReason reason) {
  switch (reason) {
    case md::DebtReason::Capacity:
    case md::DebtReason::PartialDayBound:
      return obligation_day::DebtReason::Capacity;
    case md::DebtReason::ResourceUnavailable:
    case md::DebtReason::UnreachableTile:
    case md::DebtReason::UnsupportedIdentity:
      return obligation_day::DebtReason::UnsupportedState;
  }
  return obligation_day::DebtReason::UnsupportedState;
}

bool same_signed_slot(const SignedSlot& left, const SignedSlot& right) {
  return left.step == right.step && same_action(left.unit, right.unit) &&
         left.market.has_value() == right.market.has_value() &&
         (!left.market || same_action(*left.market, *right.market)) &&
         left.raw_move_source_step == right.raw_move_source_step &&
         left.obligation_id == right.obligation_id &&
         left.transition_index == right.transition_index &&
         left.market_obligation_id == right.market_obligation_id &&
         left.market_transition_index == right.market_transition_index;
}

}  // namespace

std::uint64_t certificate_hash(const Certificate& certificate) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  add_hash(hash, certificate.player);
  add_hash(hash, certificate.actor);
  add_hash(hash, certificate.day);
  add_hash(hash, certificate.start_step);
  add_hash(hash, certificate.end_step);
  add_hash(hash, certificate.issuer_generation);
  add_hash(hash, certificate.input_hash);
  add_hash(hash, certificate.objective.unfinished_must_finish);
  add_hash(hash, certificate.objective.cross_day_cascade_risk);
  add_hash(hash, certificate.objective.deferred_important);
  add_hash(hash, certificate.objective.inserted_move_distance);
  add_hash(hash, certificate.objective.move_slot_displacement);
  add_hash(hash, certificate.objective.production_slot_disturbance);
  for (const auto& slot : certificate.slots) {
    add_hash(hash, slot.step);
    add_action_hash(hash, slot.unit);
    add_hash(hash, slot.market.has_value());
    if (slot.market) add_action_hash(hash, *slot.market);
    add_hash(hash, slot.raw_move_source_step);
    add_hash(hash, slot.obligation_id);
    add_hash(hash, slot.transition_index);
    add_hash(hash, slot.market_obligation_id);
    add_hash(hash, slot.market_transition_index);
  }
  for (const auto& debt : certificate.debts) {
    add_hash(hash, debt.obligation_id);
    add_hash(hash, static_cast<int>(debt.selector_reason));
    add_hash(hash, static_cast<int>(debt.scheduler_reason));
    add_hash(hash, debt.completed_transitions);
    add_hash(hash, debt.remaining_transitions);
    add_hash(hash, debt.bound_day_offset);
  }
  add_hash(hash, certificate.raw_move_tokens);
  add_hash(hash, certificate.emitted_raw_moves);
  add_hash(hash, certificate.selector_explored_states);
  return hash;
}

IssueResult issue(const Request& request) {
  IssueResult output;
  const auto converted = convert(request);
  if (converted.reject != RejectReason::None) {
    output.reject = converted.reject;
    output.diagnostic = converted.diagnostic;
    return output;
  }
  const auto plan = md::select_minimum_damage(converted.selector);
  if (!plan.planned()) {
    output.reject = RejectReason::SelectorRejected;
    output.diagnostic = plan.diagnostic;
    return output;
  }
  const auto selector_verified = md::verify_schedule(converted.selector, plan);
  if (!selector_verified.valid) {
    output.reject = RejectReason::InternalVerificationFailure;
    output.diagnostic = selector_verified.diagnostic;
    return output;
  }
  Certificate certificate;
  certificate.player = request.player;
  certificate.actor = request.actor;
  certificate.day = request.remaining_day_start->day();
  certificate.start_step = converted.start_step;
  certificate.end_step = converted.end_step;
  certificate.issuer_generation = request.issuer_generation;
  certificate.input_hash = converted.input_hash;
  certificate.objective = plan.objective;
  certificate.raw_move_tokens = static_cast<int>(converted.raw_moves.size());
  certificate.selector_explored_states = plan.explored_states;
  for (std::size_t tick = 0; tick < plan.slots.size(); ++tick) {
    const auto& selector_slot = plan.slots[tick];
    const auto& unit = selector_slot.units.at(0);
    SignedSlot slot;
    slot.step = converted.start_step + static_cast<int>(tick);
    if (unit.raw_source_slot >= 0) {
      const auto source =
          converted.selector_move_sources.find(unit.raw_source_slot);
      if (source == converted.selector_move_sources.end()) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "selector returned unknown MOVE token slot";
        return output;
      }
      slot.raw_move_source_step = source->second;
    }
    slot.obligation_id = unit.obligation_id;
    slot.transition_index = unit.transition_index;
    if (slot.raw_move_source_step >= 0) {
      const auto raw = converted.raw_moves.find(slot.raw_move_source_step);
      if (raw == converted.raw_moves.end()) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "selector returned unknown raw MOVE source";
        return output;
      }
      slot.unit = raw->second;
      ++certificate.emitted_raw_moves;
    } else {
      const auto mapped = to_fast(unit.action);
      if (!mapped) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "selector unit output cannot map to fastkag";
        return output;
      }
      slot.unit = *mapped;
    }
    if (selector_slot.market) {
      const auto mapped = to_fast(selector_slot.market->action);
      if (!mapped || !market(mapped->op)) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "selector market output cannot map to fastkag";
        return output;
      }
      slot.market = *mapped;
      slot.market_obligation_id = selector_slot.market->obligation_id;
      slot.market_transition_index = selector_slot.market->transition_index;
    }
    certificate.slots.push_back(slot);
  }
  for (const auto& debt : plan.debts) {
    certificate.debts.push_back(
        {debt.obligation_id, debt.reason, scheduler_reason(debt.reason),
         debt.completed_transitions, debt.remaining_transitions,
         debt.bound_day_offset});
  }
  certificate.content_hash = certificate_hash(certificate);
  output.certificate = std::move(certificate);
  output.diagnostic = "signed minimum-damage selected-actor remaining-day manifest";
  return output;
}

VerifyResult verify(const Request& request, const Certificate& certificate) {
  VerifyResult output;
  const auto converted = convert(request);
  if (converted.reject != RejectReason::None) {
    output.reject = converted.reject;
    output.diagnostic = converted.diagnostic;
    return output;
  }
  if (certificate.player != request.player ||
      certificate.actor != request.actor ||
      certificate.day != request.remaining_day_start->day() ||
      certificate.start_step != converted.start_step ||
      certificate.end_step != converted.end_step ||
      certificate.issuer_generation != request.issuer_generation ||
      certificate.input_hash != converted.input_hash ||
      certificate.content_hash != certificate_hash(certificate) ||
      certificate.slots.size() !=
          static_cast<std::size_t>(converted.selector.horizon)) {
    output.reject = RejectReason::InternalVerificationFailure;
    output.diagnostic = "certificate envelope/hash mismatch";
    return output;
  }

  md::PlanResult reconstructed;
  reconstructed.status = md::PlanStatus::Planned;
  reconstructed.objective = certificate.objective;
  reconstructed.slots.resize(certificate.slots.size());
  for (std::size_t tick = 0; tick < certificate.slots.size(); ++tick) {
    const auto& signed_slot = certificate.slots[tick];
    if (signed_slot.step != converted.start_step + static_cast<int>(tick)) {
      output.reject = RejectReason::InternalVerificationFailure;
      output.diagnostic = "signed slot step mismatch";
      return output;
    }
    auto& slot = reconstructed.slots[tick];
    slot.units.resize(1);
    if (signed_slot.raw_move_source_step >= 0) {
      const auto found = converted.raw_moves.find(
          signed_slot.raw_move_source_step);
      if (found == converted.raw_moves.end() ||
          !same_action(found->second, signed_slot.unit) ||
          signed_slot.obligation_id != 0) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "raw MOVE signature/source mapping mismatch";
        return output;
      }
      const auto selector_source = std::find_if(
          converted.selector_move_sources.begin(),
          converted.selector_move_sources.end(), [&](const auto& entry) {
            return entry.second == signed_slot.raw_move_source_step;
          });
      if (selector_source == converted.selector_move_sources.end()) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "raw MOVE lacks selector token mapping";
        return output;
      }
      slot.units[0] = {*to_md(signed_slot.unit),
                       selector_source->first,
                       0, -1};
      ++output.checked_move_tokens;
    } else if (signed_slot.obligation_id != 0) {
      const auto obligation = std::find_if(
          converted.selector.repair_obligations.begin(),
          converted.selector.repair_obligations.end(), [&](const auto& value) {
            return value.id == signed_slot.obligation_id;
          });
      if (obligation == converted.selector.repair_obligations.end() ||
          signed_slot.transition_index < 0 ||
          signed_slot.transition_index >=
              static_cast<int>(obligation->transitions.size())) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "unknown obligation/transition signature";
        return output;
      }
      const auto& transition = obligation->transitions[
          static_cast<std::size_t>(signed_slot.transition_index)];
      const auto expected = to_fast(transition.action);
      if (transition.channel != md::Channel::Unit || !expected ||
          !same_action(*expected, signed_slot.unit)) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "unit action signature mapping mismatch";
        return output;
      }
      slot.units[0] = {transition.action, -1, obligation->id,
                       signed_slot.transition_index};
    } else {
      if (signed_slot.unit.op != Op::PASS) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "unbound non-PASS signed action";
        return output;
      }
      slot.units[0] = {};
    }
    if (signed_slot.market) {
      const auto obligation = std::find_if(
          converted.selector.repair_obligations.begin(),
          converted.selector.repair_obligations.end(), [&](const auto& value) {
            return value.id == signed_slot.market_obligation_id;
          });
      if (obligation == converted.selector.repair_obligations.end() ||
          signed_slot.market_transition_index < 0 ||
          signed_slot.market_transition_index >=
              static_cast<int>(obligation->transitions.size())) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "unknown market obligation/transition signature";
        return output;
      }
      const auto& transition = obligation->transitions[
          static_cast<std::size_t>(signed_slot.market_transition_index)];
      const auto expected = to_fast(transition.action);
      if (transition.channel != md::Channel::Market || !expected ||
          !same_action(*expected, *signed_slot.market)) {
        output.reject = RejectReason::InternalVerificationFailure;
        output.diagnostic = "market action signature mapping mismatch";
        return output;
      }
      slot.market = md::ScheduledAction{
          transition.action, -1, obligation->id,
          signed_slot.market_transition_index};
    } else if (signed_slot.market_obligation_id != 0 ||
               signed_slot.market_transition_index != -1) {
      output.reject = RejectReason::InternalVerificationFailure;
      output.diagnostic = "market identity exists without market action";
      return output;
    }
    ++output.checked_slots;
  }
  for (const auto& debt : certificate.debts) {
    const auto obligation = std::find_if(
        converted.selector.repair_obligations.begin(),
        converted.selector.repair_obligations.end(),
        [&](const auto& value) { return value.id == debt.obligation_id; });
    if (obligation == converted.selector.repair_obligations.end() ||
        debt.scheduler_reason != scheduler_reason(debt.selector_reason) ||
        debt.bound_day_offset != 1) {
      output.reject = RejectReason::InternalVerificationFailure;
      output.diagnostic = "signed debt mapping mismatch";
      return output;
    }
    reconstructed.debts.push_back(
        {debt.obligation_id, debt.selector_reason,
         debt.completed_transitions, debt.remaining_transitions,
         obligation->must_finish, obligation->important,
         obligation->cascade_risk, debt.bound_day_offset});
  }
  const auto selector_verified =
      md::verify_schedule(converted.selector, reconstructed);
  if (!selector_verified.valid) {
    output.reject = RejectReason::InternalVerificationFailure;
    output.diagnostic = selector_verified.diagnostic;
    return output;
  }
  if (certificate.raw_move_tokens !=
          static_cast<int>(converted.raw_moves.size()) ||
      certificate.emitted_raw_moves != certificate.raw_move_tokens ||
      output.checked_move_tokens != certificate.raw_move_tokens) {
    output.reject = RejectReason::InternalVerificationFailure;
    output.diagnostic = "signed MOVE coverage count mismatch";
    return output;
  }

  const auto canonical = issue(request);
  if (!canonical.issued() ||
      canonical.certificate->content_hash != certificate.content_hash ||
      canonical.certificate->slots.size() != certificate.slots.size() ||
      !std::equal(canonical.certificate->slots.begin(),
                  canonical.certificate->slots.end(),
                  certificate.slots.begin(), same_signed_slot)) {
    output.reject = RejectReason::InternalVerificationFailure;
    output.diagnostic = "certificate differs from canonical exact optimum";
    return output;
  }
  output.valid = true;
  output.diagnostic =
      "bridge signature, selector replay, MOVE closure, and optimum agree";
  return output;
}

const char* reject_reason_name(RejectReason reason) noexcept {
  switch (reason) {
    case RejectReason::None: return "none";
    case RejectReason::InvalidInput: return "invalid_input";
    case RejectReason::ActorScope: return "actor_scope";
    case RejectReason::IssuerRejected: return "issuer_rejected";
    case RejectReason::PriorSourceRequiresReceipt:
      return "prior_source_requires_receipt";
    case RejectReason::UnsupportedSource: return "unsupported_source";
    case RejectReason::MissingTypedIdentity: return "missing_typed_identity";
    case RejectReason::AmbiguousSourceMapping:
      return "ambiguous_source_mapping";
    case RejectReason::UnsupportedSnapshotState:
      return "unsupported_snapshot_state";
    case RejectReason::SelectorRejected: return "selector_rejected";
    case RejectReason::InternalVerificationFailure:
      return "internal_verification_failure";
  }
  return "unknown";
}

}  // namespace g001::minimum_damage_bridge
