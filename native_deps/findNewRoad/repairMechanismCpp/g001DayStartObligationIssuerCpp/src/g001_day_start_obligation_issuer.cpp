#include "../include/g001_day_start_obligation_issuer.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <tuple>

namespace g001::day_start_issuer {
namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::TileKind;

constexpr int kTurns = 24;

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
         op == Op::WEST;
}

bool is_crop(Item item) {
  const int value = static_cast<int>(item);
  return value >= 0 && value < fastkag::N_CROPS;
}

bool is_animal(Item item) {
  const int value = static_cast<int>(item);
  return value >= static_cast<int>(Item::GOOSE) &&
         value <= static_cast<int>(Item::SHEEP);
}

bool action_equal(Action lhs, Action rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool valid_position(const Simulator& simulator, Position position) {
  return position.x >= 0 && position.y >= 0 &&
         position.x < simulator.config().board_size &&
         position.y < simulator.config().board_size;
}

Position moved(Position position, Op op) {
  if (op == Op::NORTH) {
    --position.y;
  } else if (op == Op::SOUTH) {
    ++position.y;
  } else if (op == Op::WEST) {
    --position.x;
  } else if (op == Op::EAST) {
    ++position.x;
  }
  return position;
}

using SourceKey = std::tuple<int, int>;
using TileKey = std::pair<int, int>;

TileKey tile_key(Position position) {
  return {static_cast<int>(position.x), static_cast<int>(position.y)};
}

struct SourceShape {
  int actor{-1};
  int step{-1};
  Position tile{};
  Action action{};
};

std::optional<Item> authoritative_crop(const IssueRequest& request,
                                       const SourceShape& shape,
                                       bool& mismatch) {
  const transactional_crop_repair::TypedCropObligation* unique = nullptr;
  for (const auto& binding : request.exact_crop_bindings) {
    if (binding.actor != shape.actor || binding.source_step != shape.step) {
      continue;
    }
    if (unique || binding.player != request.player ||
        binding.tile.x != shape.tile.x || binding.tile.y != shape.tile.y ||
        !action_equal(binding.source_action, shape.action) ||
        !is_crop(binding.desired)) {
      mismatch = true;
      return std::nullopt;
    }
    unique = &binding;
  }
  return unique ? std::optional<Item>(unique->desired) : std::nullopt;
}

std::optional<Item> authoritative_animal(const IssueRequest& request,
                                         const SourceShape& shape,
                                         bool& mismatch) {
  const transactional_animal_repair::TypedAnimalObligation* unique = nullptr;
  for (const auto& binding : request.exact_animal_bindings) {
    if (binding.actor != shape.actor || binding.place_step != shape.step) {
      continue;
    }
    if (unique || binding.player != request.player ||
        binding.target.column != shape.tile.x ||
        binding.target.row != shape.tile.y ||
        !action_equal(binding.place_action, shape.action) ||
        !is_animal(binding.animal)) {
      mismatch = true;
      return std::nullopt;
    }
    unique = &binding;
  }
  return unique ? std::optional<Item>(unique->animal) : std::nullopt;
}

obligation_day::ProductionObligation make_obligation(
    std::uint64_t id, const SourceShape& shape,
    obligation_day::GoalKind goal, Item item, int day_start, int day_end) {
  obligation_day::ProductionObligation obligation;
  obligation.id = id;
  obligation.actor = shape.actor;
  obligation.tile = shape.tile;
  obligation.goal = goal;
  obligation.item = item;
  obligation.quantity = std::max(1, static_cast<int>(shape.action.quantity));
  obligation.earliest_step = day_start;
  obligation.deadline_step = day_end;
  obligation.priority = 1000 - (shape.step - day_start);
  obligation.must_finish_today = true;
  obligation.repair = false;
  obligation.source_step = shape.step;
  return obligation;
}

void add_unsupported(IssueResult& output, const SourceShape& shape,
                     UnsupportedReason reason) {
  output.unsupported.push_back(
      {shape.actor, shape.step, shape.tile, shape.action, reason});
}

}  // namespace

bool PersistentRouteIntentRegistry::record_exact_source(
    int actor, int source_step, Position tile, Action source) {
  if (actor < 0 || source_step < 0) {
    return false;
  }
  IntentLineage lineage{actor, source_step, tile, source.item, source};
  if (source.op == Op::PLANT && is_crop(source.item)) {
    crops_[tile_key(tile)] = lineage;
    return true;
  }
  if (source.op == Op::PLACE && is_animal(source.item)) {
    animals_[tile_key(tile)] = lineage;
    return true;
  }
  return false;
}

std::optional<IntentLineage> PersistentRouteIntentRegistry::crop_at(
    Position tile) const {
  const auto found = crops_.find(tile_key(tile));
  return found == crops_.end() ? std::nullopt
                              : std::optional<IntentLineage>(found->second);
}

std::optional<IntentLineage> PersistentRouteIntentRegistry::animal_at(
    Position tile) const {
  const auto found = animals_.find(tile_key(tile));
  return found == animals_.end() ? std::nullopt
                                : std::optional<IntentLineage>(found->second);
}

void PersistentRouteIntentRegistry::clear() noexcept {
  crops_.clear();
  animals_.clear();
}

CrossTickAnimalResult CrossTickAnimalIntentBinder::stage_final_submission(
    const Simulator& before, int player, const fastkag::PlayerAction& final,
    std::span<const fastkag::PlayerAction> immutable_route,
    std::uint64_t issuer_generation) {
  if (pending_) {
    return {CrossTickAnimalStatus::SecondWriter, std::nullopt};
  }
  const int submitted = before.step_count();
  const int turns = before.config().turns_per_day;
  const int day_end = (submitted / turns + 1) * turns - 1;
  if (player < 0 || player >= 2 || issuer_generation == 0 || turns <= 0 ||
      before.done() || submitted >= day_end ||
      day_end >= static_cast<int>(immutable_route.size())) {
    return {CrossTickAnimalStatus::InvalidSubmission, std::nullopt};
  }

  Pending staged;
  staged.player = player;
  staged.submitted_step = submitted;
  staged.seed = before.seed();
  staged.prior_hands =
      static_cast<int>(before.farms()[player].hands.size());
  staged.submitted_market = final.market;
  for (std::size_t slot = 0; slot < final.market.size(); ++slot) {
    const auto& order = final.market[slot];
    if (order.op == Op::HIRE) {
      if (order.item != Item::NONE || order.quantity != 1) {
        return {CrossTickAnimalStatus::InvalidSubmission, std::nullopt};
      }
      staged.hire_slots.push_back(static_cast<int>(slot));
    } else if (order.op == Op::BUY_ANIMAL) {
      if (staged.buy_slot >= 0 || !is_animal(order.item) ||
          order.quantity != 1) {
        return {CrossTickAnimalStatus::InvalidSubmission, std::nullopt};
      }
      staged.buy_slot = static_cast<int>(slot);
      staged.animal = order.item;
    }
  }
  if (staged.hire_slots.empty() || staged.buy_slot < 0) {
    return {CrossTickAnimalStatus::InvalidSubmission, std::nullopt};
  }
  staged.suffix_units.reserve(day_end - submitted);
  for (int step = submitted + 1; step <= day_end; ++step) {
    staged.suffix_units.push_back(immutable_route[step].units);
  }
  std::uint64_t id = 1469598103934665603ULL;
  const auto add = [&](std::uint64_t value) {
    id ^= value;
    id *= 1099511628211ULL;
  };
  add(issuer_generation);
  add(static_cast<std::uint64_t>(player));
  add(static_cast<std::uint64_t>(submitted));
  add(static_cast<std::uint64_t>(staged.animal));
  staged.obligation_id = id | (1ULL << 63U);
  pending_ = std::move(staged);
  return {CrossTickAnimalStatus::Staged, std::nullopt};
}

CrossTickAnimalResult CrossTickAnimalIntentBinder::bind_next(
    const Simulator& after, const FinalMarketReceipt& receipt) {
  if (!pending_) {
    return {CrossTickAnimalStatus::NoPending, std::nullopt};
  }
  auto staged = std::move(*pending_);
  pending_.reset();
  if (receipt.player != staged.player ||
      receipt.submitted_step != staged.submitted_step ||
      after.step_count() != staged.submitted_step + 1 ||
      after.seed() != staged.seed) {
    return {CrossTickAnimalStatus::StaleReceipt, std::nullopt};
  }
  const auto same_market = [&]() {
    if (receipt.submitted_market.size() != staged.submitted_market.size()) {
      return false;
    }
    for (std::size_t index = 0; index < receipt.submitted_market.size();
         ++index) {
      if (!action_equal(receipt.submitted_market[index],
                        staged.submitted_market[index])) {
        return false;
      }
    }
    return true;
  };
  const auto& observed_fills = after.last_market_fills()[staged.player];
  if (!same_market() || receipt.fills != observed_fills ||
      receipt.fills.size() != staged.submitted_market.size()) {
    return {CrossTickAnimalStatus::ReceiptMismatch, std::nullopt};
  }
  for (int slot : staged.hire_slots) {
    if (slot < 0 || slot >= static_cast<int>(receipt.fills.size()) ||
        receipt.fills[slot] != 1) {
      return {CrossTickAnimalStatus::MissingHireEvidence, std::nullopt};
    }
  }
  if (staged.buy_slot < 0 ||
      staged.buy_slot >= static_cast<int>(receipt.fills.size()) ||
      receipt.fills[staged.buy_slot] < 0 ||
      receipt.fills[staged.buy_slot] > 1) {
    return {CrossTickAnimalStatus::MissingBuyEvidence, std::nullopt};
  }
  const auto& farm = after.farms()[staged.player];
  const int expected_hands =
      staged.prior_hands + static_cast<int>(staged.hire_slots.size());
  if (static_cast<int>(farm.hands.size()) != expected_hands ||
      after.privates()[staged.player].inventories.size() !=
          farm.hands.size() + 1) {
    return {CrossTickAnimalStatus::ActorShapeMismatch, std::nullopt};
  }

  struct Candidate {
    int actor{};
    int place_step{};
    Position tile{};
    Action action{};
  };
  std::vector<Candidate> candidates;
  const int first_new_actor = staged.prior_hands + 1;
  for (int actor = first_new_actor; actor <= expected_hands; ++actor) {
    Position position = farm.hands[static_cast<std::size_t>(actor - 1)];
    for (std::size_t offset = 0; offset < staged.suffix_units.size();
         ++offset) {
      const auto& units = staged.suffix_units[offset];
      const Action action = actor < static_cast<int>(units.size())
                                ? units[static_cast<std::size_t>(actor)]
                                : Action{};
      if (action.op == Op::PLACE && action.item == staged.animal &&
          action.quantity == 1) {
        candidates.push_back(
            {actor, staged.submitted_step + 1 + static_cast<int>(offset),
             position, action});
      }
      if (is_move(action.op)) {
        const auto next = moved(position, action.op);
        if (valid_position(after, next)) {
          position = next;
        }
      }
    }
  }
  if (candidates.size() != 1) {
    return {CrossTickAnimalStatus::PlaceNotUnique, std::nullopt};
  }
  const auto& match = candidates.front();
  transactional_animal_repair::TypedAnimalObligation obligation;
  obligation.id = staged.obligation_id;
  obligation.player = staged.player;
  obligation.actor = match.actor;
  obligation.place_step = match.place_step;
  obligation.target = {match.tile.y, match.tile.x};
  obligation.place_action = match.action;
  obligation.animal = staged.animal;
  obligation.acquisition_step = staged.submitted_step;
  obligation.acquisition_market_slot = staged.buy_slot;
  obligation.acquisition_action = staged.submitted_market[staged.buy_slot];
  obligation.acquisition_fill = receipt.fills[staged.buy_slot];
  obligation.acquisition_seed = staged.seed;
  return {CrossTickAnimalStatus::Bound, obligation};
}

IssueResult issue_day_start(const IssueRequest& request) {
  IssueResult output;
  if (!request.day_start || !request.route_tape || request.player < 0 ||
      request.player >= 2 || request.issuer_generation == 0) {
    output.reject = IssueReject::InvalidInput;
    return output;
  }
  const auto& state = *request.day_start;
  if (state.hour() != 0 || state.config().turns_per_day != kTurns) {
    output.reject = IssueReject::NotDayStart;
    return output;
  }
  const int day_start = state.step_count();
  const int day_end = day_start + kTurns - 1;
  if (day_end >= static_cast<int>(request.route_tape->size())) {
    output.reject = IssueReject::RouteTooShort;
    return output;
  }

  std::vector<Position> positions{state.farms()[request.player].farmer};
  positions.insert(positions.end(), state.farms()[request.player].hands.begin(),
                   state.farms()[request.player].hands.end());
  const int actors = static_cast<int>(positions.size());
  std::vector<SourceShape> shapes;
  shapes.reserve(actors * kTurns);
  for (int step = day_start; step <= day_end; ++step) {
    const auto& units = (*request.route_tape)[step].units;
    for (int actor = 0; actor < actors; ++actor) {
      const Action action = actor < static_cast<int>(units.size())
                                ? units[actor]
                                : Action{};
      shapes.push_back({actor, step, positions[actor], action});
      output.raw_sources.push_back({actor, step, action});
      if (is_move(action.op)) {
        output.moves.push_back({actor, step, action});
        positions[actor] = moved(positions[actor], action.op);
      }
    }
    for (int actor = actors; actor < static_cast<int>(units.size()); ++actor) {
      if (units[actor].op != Op::PASS) {
        add_unsupported(output, {actor, step, {}, units[actor]},
                        UnsupportedReason::ActorUnavailableAtDayStart);
      }
    }
  }

  std::map<TileKey, Item> known_crops;
  std::map<TileKey, Item> known_animals;
  const int board_size = state.config().board_size;
  const auto& farm = state.farms()[request.player];
  for (int y = 0; y < board_size; ++y) {
    for (int x = 0; x < board_size; ++x) {
      const auto& tile = farm.tiles[y * board_size + x];
      if (tile.kind == TileKind::PLANT && is_crop(tile.crop)) {
        known_crops[{x, y}] = tile.crop;
      }
      if (tile.kind == TileKind::ANIMAL && is_animal(tile.animal)) {
        known_animals[{x, y}] = tile.animal;
      }
    }
  }
  if (request.persistent_intents) {
    for (const auto& shape : shapes) {
      const auto crop = request.persistent_intents->crop_at(shape.tile);
      // A currently visible typed crop/animal is newer public evidence and
      // wins over stale lineage.  EMPTY/WEED retains the last exact route
      // intent so a failed PLANT or weed does not erase its identity.
      if (crop && !known_crops.contains(tile_key(shape.tile))) {
        known_crops[tile_key(shape.tile)] = crop->item;
      }
      const auto animal = request.persistent_intents->animal_at(shape.tile);
      if (animal && !known_animals.contains(tile_key(shape.tile))) {
        known_animals[tile_key(shape.tile)] = animal->item;
      }
    }
  }

  // A raw DIG is only a crop lifecycle obligation if a later same-day typed
  // crop source proves what must be planted.  Reverse evidence never reads a
  // future game observation; it only reads the immutable G001 tape.
  std::map<TileKey, Item> future_crop;
  std::map<SourceKey, Item> dig_desired;
  for (auto iterator = shapes.rbegin(); iterator != shapes.rend(); ++iterator) {
    const auto& shape = *iterator;
    bool mismatch = false;
    const auto authority = authoritative_crop(request, shape, mismatch);
    if (mismatch) {
      continue;
    }
    if (authority) {
      future_crop[tile_key(shape.tile)] = *authority;
    } else if (shape.action.op == Op::PLANT && is_crop(shape.action.item)) {
      future_crop[tile_key(shape.tile)] = shape.action.item;
    }
    if (shape.action.op == Op::DIG &&
        future_crop.contains(tile_key(shape.tile))) {
      dig_desired[{shape.actor, shape.step}] =
          future_crop[tile_key(shape.tile)];
    }
  }

  std::map<TileKey, std::uint64_t> last_tile_obligation;
  std::map<std::pair<int, int>, std::uint64_t> last_pickup;
  std::uint64_t next_id = request.issuer_generation * 1000ULL + 1ULL;
  for (const auto& shape : shapes) {
    if (!valid_position(state, shape.tile)) {
      if (shape.action.op != Op::PASS && !is_move(shape.action.op)) {
        add_unsupported(output, shape,
                        UnsupportedReason::UnsupportedProductionAction);
      }
      continue;
    }
    const auto key = tile_key(shape.tile);
    bool crop_mismatch = false;
    bool animal_mismatch = false;
    auto crop_identity =
        authoritative_crop(request, shape, crop_mismatch);
    auto animal_identity =
        authoritative_animal(request, shape, animal_mismatch);
    if (crop_mismatch || animal_mismatch) {
      add_unsupported(output, shape, UnsupportedReason::AuthorityMismatch);
      continue;
    }
    if (!crop_identity && is_crop(shape.action.item)) {
      crop_identity = shape.action.item;
    }
    if (!animal_identity && is_animal(shape.action.item)) {
      animal_identity = shape.action.item;
    }

    std::optional<obligation_day::ProductionObligation> obligation;
    if (shape.action.op == Op::PASS || is_move(shape.action.op)) {
      continue;
    } else if (shape.action.op == Op::DIG) {
      const auto desired = dig_desired.find({shape.actor, shape.step});
      if (desired == dig_desired.end()) {
        add_unsupported(output, shape,
                        UnsupportedReason::MissingCropIdentity);
        continue;
      }
      crop_identity = desired->second;
      obligation = make_obligation(next_id++, shape,
                                   obligation_day::GoalKind::CropReady,
                                   *crop_identity, day_start, day_end);
    } else if (shape.action.op == Op::PLANT) {
      if (!crop_identity) {
        add_unsupported(output, shape,
                        UnsupportedReason::MissingCropIdentity);
        continue;
      }
      known_crops[key] = *crop_identity;
      obligation = make_obligation(next_id++, shape,
                                   obligation_day::GoalKind::CropReady,
                                   *crop_identity, day_start, day_end);
      obligation->resource = {*crop_identity, 1, 0, 0};
    } else if (shape.action.op == Op::WATER) {
      if (!crop_identity && known_crops.contains(key)) {
        crop_identity = known_crops[key];
      }
      if (!crop_identity) {
        add_unsupported(output, shape,
                        UnsupportedReason::MissingCropIdentity);
        continue;
      }
      obligation = make_obligation(next_id++, shape,
                                   obligation_day::GoalKind::CropReady,
                                   *crop_identity, day_start, day_end);
    } else if (shape.action.op == Op::HARVEST) {
      if (!crop_identity && known_crops.contains(key)) {
        crop_identity = known_crops[key];
      }
      if (!crop_identity && known_animals.contains(key)) {
        add_unsupported(
            output, shape,
            UnsupportedReason::AnimalHarvestProductIdentityMissing);
        continue;
      }
      if (!crop_identity) {
        add_unsupported(output, shape,
                        UnsupportedReason::MissingCropIdentity);
        continue;
      }
      obligation = make_obligation(next_id++, shape,
                                   obligation_day::GoalKind::Harvest,
                                   *crop_identity, day_start, day_end);
    } else if (shape.action.op == Op::BUILD_PASTURE) {
      obligation = make_obligation(next_id++, shape,
                                   obligation_day::GoalKind::BuildPasture,
                                   Item::NONE, day_start, day_end);
    } else if (shape.action.op == Op::PICKUP) {
      if (shape.action.item == Item::NONE) {
        add_unsupported(output, shape,
                        UnsupportedReason::UnsupportedProductionAction);
        continue;
      }
      obligation = make_obligation(next_id++, shape,
                                   obligation_day::GoalKind::Pickup,
                                   shape.action.item, day_start, day_end);
      obligation->resource = {shape.action.item, 0,
                              obligation->quantity, 0};
      last_pickup[{shape.actor, static_cast<int>(shape.action.item)}] =
          obligation->id;
    } else if (shape.action.op == Op::PLACE) {
      if (!animal_identity) {
        add_unsupported(output, shape,
                        UnsupportedReason::MissingAnimalIdentity);
        continue;
      }
      known_animals[key] = *animal_identity;
      obligation = make_obligation(next_id++, shape,
                                   obligation_day::GoalKind::Place,
                                   *animal_identity, day_start, day_end);
      obligation->resource = {*animal_identity, 0, 0, 1};
      const auto pickup =
          last_pickup.find({shape.actor, static_cast<int>(*animal_identity)});
      if (pickup != last_pickup.end()) {
        obligation->dependencies.push_back(pickup->second);
      }
      transactional_animal_repair::TypedAnimalObligation typed;
      typed.id = obligation->id;
      typed.player = request.player;
      typed.actor = shape.actor;
      typed.place_step = shape.step;
      typed.target = {shape.tile.y, shape.tile.x};
      typed.place_action = shape.action;
      typed.animal = *animal_identity;
      output.animal_owner_obligations.push_back(typed);
    } else if (shape.action.op == Op::FEED || shape.action.op == Op::CARE) {
      if (!animal_identity && known_animals.contains(key)) {
        animal_identity = known_animals[key];
      }
      if (!animal_identity) {
        add_unsupported(output, shape,
                        UnsupportedReason::MissingAnimalIdentity);
        continue;
      }
      const auto goal = shape.action.op == Op::FEED
                            ? obligation_day::GoalKind::Feed
                            : obligation_day::GoalKind::Care;
      obligation = make_obligation(next_id++, shape, goal, *animal_identity,
                                   day_start, day_end);
      if (goal == obligation_day::GoalKind::Feed) {
        obligation->resource = {Item::WHEAT, 0, 0, 1};
      }
    } else if (shape.action.op == Op::COLLECT_FERTILIZER) {
      obligation = make_obligation(
          next_id++, shape, obligation_day::GoalKind::CollectFertilizer,
          Item::FERTILIZER, day_start, day_end);
    } else {
      add_unsupported(output, shape,
                      UnsupportedReason::UnsupportedProductionAction);
      continue;
    }

    if ((shape.action.op == Op::PLANT || shape.action.op == Op::WATER ||
         shape.action.op == Op::HARVEST) &&
        crop_identity) {
      transactional_crop_repair::TypedCropObligation typed;
      typed.id = obligation->id;
      typed.player = request.player;
      typed.actor = shape.actor;
      typed.source_step = shape.step;
      typed.tile = shape.tile;
      typed.source_action = shape.action;
      typed.desired = *crop_identity;
      output.crop_owner_obligations.push_back(typed);
    }
    if (last_tile_obligation.contains(key) &&
        std::find(obligation->dependencies.begin(),
                  obligation->dependencies.end(),
                  last_tile_obligation[key]) == obligation->dependencies.end()) {
      obligation->dependencies.push_back(last_tile_obligation[key]);
    }
    last_tile_obligation[key] = obligation->id;
    output.obligations.push_back(std::move(*obligation));
  }

  return output;
}

ProviderDayAudit audit_final_provider_day(
    const IssueResult& issued, int day_start,
    std::span<const fastkag::PlayerAction> actual_provider_actions) {
  ProviderDayAudit audit;
  if (!issued.issued() || actual_provider_actions.size() != kTurns) {
    return audit;
  }
  for (const auto& source : issued.raw_sources) {
    const int tick = source.source_step - day_start;
    bool mismatch = tick < 0 || tick >= kTurns;
    if (!mismatch) {
      const auto& units = actual_provider_actions[tick].units;
      if (source.actor < 0 || source.actor >= static_cast<int>(units.size()) ||
          !action_equal(units[source.actor], source.action)) {
        mismatch = true;
      }
    }
    if (mismatch) {
      ++audit.mismatches;
      const int operation = static_cast<int>(source.action.op);
      if (operation >= 0 &&
          operation < static_cast<int>(audit.mismatches_by_source_op.size())) {
        ++audit.mismatches_by_source_op[operation];
      }
      audit.mismatched_sources.push_back(
          {source.actor, source.source_step, {}, source.action,
           UnsupportedReason::FinalProviderMismatch});
    }
    ++audit.checked_sources;
    if (audit.mismatches > 0 && audit.first_mismatch_step < 0) {
      audit.first_mismatch_step = source.source_step;
    }
  }
  audit.day_valid = audit.checked_sources > 0 && audit.mismatches == 0;
  return audit;
}

const char* unsupported_reason_name(UnsupportedReason reason) {
  switch (reason) {
    case UnsupportedReason::MissingCropIdentity:
      return "missing_crop_identity";
    case UnsupportedReason::MissingAnimalIdentity:
      return "missing_animal_identity";
    case UnsupportedReason::ActorUnavailableAtDayStart:
      return "actor_unavailable_at_day_start";
    case UnsupportedReason::UnsupportedProductionAction:
      return "unsupported_production_action";
    case UnsupportedReason::AnimalHarvestProductIdentityMissing:
      return "animal_harvest_product_identity_missing";
    case UnsupportedReason::AuthorityMismatch:
      return "authority_mismatch";
    case UnsupportedReason::FinalProviderMismatch:
      return "final_provider_mismatch";
  }
  return "unknown";
}

const char* cross_tick_animal_status_name(CrossTickAnimalStatus status) {
  switch (status) {
    case CrossTickAnimalStatus::Staged:
      return "staged";
    case CrossTickAnimalStatus::Bound:
      return "bound";
    case CrossTickAnimalStatus::InvalidSubmission:
      return "invalid_submission";
    case CrossTickAnimalStatus::SecondWriter:
      return "second_writer";
    case CrossTickAnimalStatus::NoPending:
      return "no_pending";
    case CrossTickAnimalStatus::StaleReceipt:
      return "stale_receipt";
    case CrossTickAnimalStatus::ReceiptMismatch:
      return "receipt_mismatch";
    case CrossTickAnimalStatus::MissingHireEvidence:
      return "missing_hire_evidence";
    case CrossTickAnimalStatus::MissingBuyEvidence:
      return "missing_buy_evidence";
    case CrossTickAnimalStatus::ActorShapeMismatch:
      return "actor_shape_mismatch";
    case CrossTickAnimalStatus::PlaceNotUnique:
      return "place_not_unique";
  }
  return "unknown";
}

}  // namespace g001::day_start_issuer
