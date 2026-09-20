#include "purchase_failure_day_rolling_owner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace g001::purchase_failure_rolling {
namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::Position;
using fastkag::Tile;
using fastkag::TileKind;

constexpr std::array<int, fastkag::N_CROPS> kSeedCost{10, 20, 50, 100,
                                                       80};
constexpr std::array<int, fastkag::N_ANIMALS> kAnimalCost{300, 400, 500};

void hash_add(std::uint64_t& hash, std::uint64_t value) noexcept {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & 255U;
    hash *= 1099511628211ULL;
  }
}

bool same_action(Action lhs, Action rhs) noexcept {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool same_player_action(const fastkag::PlayerAction& lhs,
                        const fastkag::PlayerAction& rhs) noexcept {
  if (lhs.units.size() != rhs.units.size() ||
      lhs.market.size() != rhs.market.size())
    return false;
  for (std::size_t index = 0; index < lhs.units.size(); ++index)
    if (!same_action(lhs.units[index], rhs.units[index])) return false;
  for (std::size_t index = 0; index < lhs.market.size(); ++index)
    if (!same_action(lhs.market[index], rhs.market[index])) return false;
  return true;
}

bool is_move(Op operation) noexcept {
  return operation == Op::NORTH || operation == Op::SOUTH ||
         operation == Op::EAST || operation == Op::WEST;
}

bool valid_player(const fastkag::Simulator& observation, int player) {
  return player >= 0 && player < static_cast<int>(observation.farms().size());
}

Position actor_position(const fastkag::Simulator& observation, int player,
                        int actor) {
  const auto& farm = observation.farms()[player];
  if (actor == 0) return farm.farmer;
  const auto index = static_cast<std::size_t>(actor - 1);
  return actor > 0 && index < farm.hands.size() ? farm.hands[index]
                                               : Position{-1, -1};
}

const Tile* tile_at(const fastkag::Simulator& observation, int player,
                    Position position) {
  const auto board = observation.config().board_size;
  if (position.x < 0 || position.y < 0 || position.x >= board ||
      position.y >= board)
    return nullptr;
  const auto index = static_cast<std::size_t>(position.y) *
                         static_cast<std::size_t>(board) +
                     static_cast<std::size_t>(position.x);
  const auto& tiles = observation.farms()[player].tiles;
  return index < tiles.size() ? &tiles[index] : nullptr;
}

bool same_position(Position lhs, Position rhs) noexcept {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool shed_adjacent(const fastkag::Simulator& observation,
                   Position position) {
  const int half = observation.config().board_size / 2;
  return (position.x == half - 1 || position.x == half) &&
         (position.y == half - 1 || position.y == half);
}

int shed_used(const fastkag::Simulator& observation, int player) {
  int total = 0;
  for (const auto value : observation.privates()[player].shed) total += value;
  return total;
}

int purchase_cost(Op operation, Item item) {
  const int value = static_cast<int>(item);
  if (operation == Op::BUY_SEED && value >= 0 && value < fastkag::N_CROPS)
    return kSeedCost[static_cast<std::size_t>(value)];
  if (operation == Op::BUY_ANIMAL && value >= 9 && value < 12)
    return kAnimalCost[static_cast<std::size_t>(value - 9)];
  return -1;
}

DebtKind debt_kind(Action action) {
  switch (action.op) {
    case Op::BUY_SEED: return DebtKind::AcquireSeed;
    case Op::BUY_ANIMAL: return DebtKind::AcquireAnimal;
    case Op::PICKUP:
      return static_cast<int>(action.item) >= 9 ? DebtKind::PickupAnimal
                                               : DebtKind::PickupInput;
    case Op::PLANT: return DebtKind::Plant;
    case Op::PLACE: return DebtKind::Place;
    case Op::WATER: return DebtKind::Water;
    case Op::FEED: return DebtKind::Feed;
    case Op::CARE: return DebtKind::Care;
    case Op::HARVEST: return DebtKind::Harvest;
    default: break;
  }
  throw std::invalid_argument("action is not a supported rolling debt");
}

bool supported_unit(Action action) noexcept {
  switch (action.op) {
    case Op::PICKUP:
    case Op::PLANT:
    case Op::PLACE:
    case Op::WATER:
    case Op::FEED:
    case Op::CARE:
    case Op::HARVEST: return true;
    default: return false;
  }
}

std::uint64_t observation_hash(const fastkag::Simulator& observation,
                               int player) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, observation.seed());
  hash_add(hash, static_cast<std::uint32_t>(observation.step_count()));
  hash_add(hash, static_cast<std::uint32_t>(player));
  const auto& farm = observation.farms()[player];
  hash_add(hash, static_cast<std::uint64_t>(farm.money * 100.0));
  hash_add(hash, static_cast<std::uint32_t>(farm.farmer.x));
  hash_add(hash, static_cast<std::uint32_t>(farm.farmer.y));
  hash_add(hash, farm.hands.size());
  for (const auto position : farm.hands) {
    hash_add(hash, static_cast<std::uint32_t>(position.x));
    hash_add(hash, static_cast<std::uint32_t>(position.y));
  }
  const auto& private_state = observation.privates()[player];
  for (const auto value : private_state.seeds) hash_add(hash, value);
  for (const auto value : private_state.shed) hash_add(hash, value);
  for (const auto& inventory : private_state.inventories)
    for (const auto value : inventory) hash_add(hash, value);
  for (const auto& tile : farm.tiles) {
    hash_add(hash, static_cast<std::uint8_t>(tile.kind));
    hash_add(hash, static_cast<std::uint8_t>(tile.crop));
    hash_add(hash, static_cast<std::uint8_t>(tile.animal));
    hash_add(hash, static_cast<std::uint32_t>(tile.yield_units));
    hash_add(hash, tile.watered_today);
    hash_add(hash, tile.fed_today);
    hash_add(hash, tile.cared_today);
  }
  return hash;
}

std::uint64_t proposal_binding_hash(const Proposal& proposal) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, static_cast<std::uint32_t>(proposal.step));
  hash_add(hash, proposal.generation);
  hash_add(hash, proposal.observation_hash);
  hash_add(hash, proposal.base_action_hash);
  hash_add(hash, proposal.final_action_hash);
  return hash;
}

Debt* find_debt(std::vector<Debt>& debts, std::uint64_t id) {
  const auto found = std::find_if(debts.begin(), debts.end(),
                                  [id](const Debt& debt) {
                                    return debt.id == id;
                                  });
  return found == debts.end() ? nullptr : &*found;
}

const Debt* find_open_acquisition(const std::vector<Debt>& debts, Item item,
                                  bool animal) {
  const auto kind = animal ? DebtKind::AcquireAnimal : DebtKind::AcquireSeed;
  const auto found = std::find_if(debts.rbegin(), debts.rend(),
                                  [&](const Debt& debt) {
                                    return debt.kind == kind &&
                                           debt.item == item &&
                                           debt.status != DebtStatus::Completed &&
                                           debt.status != DebtStatus::Expired;
                                  });
  return found == debts.rend() ? nullptr : &*found;
}

const Debt* find_recent_acquisition(const std::vector<Debt>& debts, Item item,
                                    bool animal, int step) {
  const auto kind = animal ? DebtKind::AcquireAnimal : DebtKind::AcquireSeed;
  const auto found = std::find_if(debts.rbegin(), debts.rend(),
                                  [&](const Debt& debt) {
                                    return debt.kind == kind &&
                                           debt.item == item &&
                                           debt.source_step < step &&
                                           step <= debt.deadline_step;
                                  });
  return found == debts.rend() ? nullptr : &*found;
}

bool has_chain_debt(const std::vector<Debt>& debts, std::uint64_t root,
                    DebtKind kind, int source_step, int actor) {
  return std::any_of(debts.begin(), debts.end(), [&](const Debt& debt) {
    return debt.root_id == root && debt.kind == kind &&
           debt.source_step == source_step && debt.actor == actor;
  });
}

bool has_open_dependency(const std::vector<Debt>& debts, std::uint64_t root,
                         DebtKind kind) {
  return std::any_of(debts.begin(), debts.end(), [&](const Debt& debt) {
    return debt.root_id == root && debt.kind == kind &&
           debt.status != DebtStatus::Completed &&
           debt.status != DebtStatus::Expired;
  });
}

const Debt* find_open_predecessor(const std::vector<Debt>& debts,
                                  DebtKind kind, Item item,
                                  Position tile) {
  const auto found = std::find_if(debts.rbegin(), debts.rend(),
                                  [&](const Debt& debt) {
    const bool item_matches = item == Item::NONE || debt.item == item;
    return debt.kind == kind && item_matches &&
           (tile.x < 0 || same_position(debt.tile, tile)) &&
           debt.status != DebtStatus::Completed &&
           debt.status != DebtStatus::Expired;
  });
  return found == debts.rend() ? nullptr : &*found;
}

bool unit_effect(const fastkag::Simulator& before,
                 const fastkag::Simulator& after, int player, int actor,
                 Action action, Position position) {
  if (actor < 0 || actor >= static_cast<int>(before.privates()[player].inventories.size()))
    return false;
  const auto& before_private = before.privates()[player];
  const auto& after_private = after.privates()[player];
  const int item = static_cast<int>(action.item);
  const auto* before_tile = tile_at(before, player, position);
  const auto* after_tile = tile_at(after, player, position);
  switch (action.op) {
    case Op::PICKUP:
      return item >= 0 && item < fastkag::N_ITEMS &&
             actor < static_cast<int>(after_private.inventories.size()) &&
             after_private.inventories[static_cast<std::size_t>(actor)]
                     [static_cast<std::size_t>(item)] >
                 before_private.inventories[static_cast<std::size_t>(actor)]
                         [static_cast<std::size_t>(item)];
    case Op::PLANT:
      return after_tile && after_tile->kind == TileKind::PLANT &&
             after_tile->crop == action.item;
    case Op::PLACE:
      return after_tile && after_tile->kind == TileKind::ANIMAL &&
             after_tile->animal == action.item;
    case Op::WATER:
      return before_tile && after_tile && !before_tile->watered_today &&
             after_tile->watered_today;
    case Op::FEED:
      return before_tile && after_tile && !before_tile->fed_today &&
             after_tile->fed_today;
    case Op::CARE:
      return before_tile && after_tile && !before_tile->cared_today &&
             after_tile->cared_today;
    case Op::HARVEST:
      if (!before_tile || !after_tile) return false;
      return before_tile->yield_units > after_tile->yield_units ||
             after_private.inventories[static_cast<std::size_t>(actor)] !=
                 before_private.inventories[static_cast<std::size_t>(actor)];
    default: return true;
  }
}

bool executable(const fastkag::Simulator& observation, int player,
                const Debt& debt, int actor) {
  if (actor < 0 || actor >=
                       static_cast<int>(observation.privates()[player]
                                            .inventories.size()))
    return false;
  const auto position = actor_position(observation, player, actor);
  const auto* tile = tile_at(observation, player, position);
  const int item = static_cast<int>(debt.item);
  const auto& private_state = observation.privates()[player];
  switch (debt.kind) {
    case DebtKind::PickupAnimal:
    case DebtKind::PickupInput:
      return shed_adjacent(observation, position) && item >= 0 &&
             item < fastkag::N_ITEMS && private_state.shed[item] > 0;
    case DebtKind::Plant:
      return tile && same_position(position, debt.tile) &&
             tile->kind == TileKind::EMPTY && item >= 0 &&
             item < fastkag::N_CROPS && private_state.seeds[item] > 0;
    case DebtKind::Place:
      return tile && same_position(position, debt.tile) &&
             (tile->kind == TileKind::COOP ||
              tile->kind == TileKind::PASTURE) &&
             tile->animal == Item::NONE && item >= 9 && item < 12 &&
             private_state.inventories[static_cast<std::size_t>(actor)]
                                      [static_cast<std::size_t>(item)] > 0;
    case DebtKind::Water:
      return tile && same_position(position, debt.tile) &&
             tile->kind == TileKind::PLANT && !tile->watered_today;
    case DebtKind::Feed:
      return tile && same_position(position, debt.tile) &&
             tile->kind == TileKind::ANIMAL && !tile->fed_today &&
             private_state.inventories[static_cast<std::size_t>(actor)][0] > 0;
    case DebtKind::Care:
      return tile && same_position(position, debt.tile) &&
             tile->kind == TileKind::ANIMAL && !tile->cared_today;
    case DebtKind::Harvest:
      return tile && same_position(position, debt.tile) &&
             tile->yield_units > 0;
    case DebtKind::AcquireSeed:
    case DebtKind::AcquireAnimal: return false;
  }
  return false;
}

bool can_retry_purchase(const fastkag::Simulator& observation, int player,
                        const Debt& debt,
                        const fastkag::PlayerAction& planned,
                        int quantity) {
  const auto operation = debt.kind == DebtKind::AcquireSeed ? Op::BUY_SEED
                                                            : Op::BUY_ANIMAL;
  const auto cost = purchase_cost(operation, debt.item);
  quantity = std::max(1, quantity);
  if (cost < 0) return false;
  auto protected_state = observation;
  std::array<fastkag::PlayerAction, 2> actions;
  for (const auto order : planned.market)
    if (order.op != Op::SELL) actions[player].market.push_back(order);
  protected_state.step(actions);
  if (protected_state.farms()[player].money < cost * quantity) return false;
  int protected_shed = shed_used(observation, player);
  for (const auto order : planned.market)
    if (order.op == Op::BUY_ANIMAL || order.op == Op::BUY_PRODUCT)
      protected_shed += std::max(0, order.quantity);
  if (operation == Op::BUY_ANIMAL && protected_shed + quantity >
          observation.config().shed_capacity)
    return false;
  return true;
}

int uncovered_seed_demand(const fastkag::Simulator& observation, int player,
                          const std::vector<fastkag::PlayerAction>& tape,
                          const Debt& debt) {
  const int item = static_cast<int>(debt.item);
  if (item < 0 || item >= fastkag::N_CROPS) return 0;
  int stock = observation.privates()[player].seeds[
      static_cast<std::size_t>(item)];
  int missing = 0;
  const int end = std::min(debt.deadline_step,
                           static_cast<int>(tape.size()) - 1);
  for (int step = observation.step_count() + 1; step <= end; ++step) {
    const auto& action = tape[static_cast<std::size_t>(step)];
    for (const auto unit : action.units) {
      if (unit.op != Op::PLANT || unit.item != debt.item) continue;
      if (stock > 0)
        --stock;
      else
        ++missing;
    }
    for (const auto order : action.market)
      if (order.op == Op::BUY_SEED && order.item == debt.item)
        stock += std::max(0, order.quantity);
  }
  return missing;
}

int uncovered_animal_demand(const fastkag::Simulator& observation, int player,
                            const std::vector<fastkag::PlayerAction>& tape,
                            const Debt& debt) {
  const int item = static_cast<int>(debt.item);
  if (item < static_cast<int>(Item::GOOSE) ||
      item > static_cast<int>(Item::SHEEP))
    return 0;
  int stock = observation.privates()[player].shed[
      static_cast<std::size_t>(item)];
  for (const auto& inventory : observation.privates()[player].inventories)
    stock += inventory[static_cast<std::size_t>(item)];
  int missing = 0;
  const int end = std::min(debt.deadline_step,
                           static_cast<int>(tape.size()) - 1);
  for (int step = observation.step_count() + 1; step <= end; ++step) {
    const auto& action = tape[static_cast<std::size_t>(step)];
    for (const auto unit : action.units) {
      if (unit.op != Op::PLACE || unit.item != debt.item) continue;
      if (stock > 0)
        --stock;
      else
        ++missing;
    }
    for (const auto order : action.market)
      if (order.op == Op::BUY_ANIMAL && order.item == debt.item)
        stock += std::max(0, order.quantity);
  }
  return missing;
}

std::string debt_string(const Debt& debt) {
  std::ostringstream stream;
  stream << "id=" << debt.id << " root=" << debt.root_id
         << " kind=" << debt_kind_name(debt.kind)
         << " status=" << debt_status_name(debt.status)
         << " item=" << static_cast<int>(debt.item)
         << " op=" << static_cast<int>(debt.action.op)
         << " qty=" << debt.action.quantity << " actor=" << debt.actor
         << " tile=(" << debt.tile.x << ',' << debt.tile.y << ')'
         << " source=" << debt.source_step
         << " earliest=" << debt.earliest_step
         << " deadline=" << debt.deadline_step
         << " attempts=" << debt.attempts
         << " provenance=" << debt.provenance;
  return stream.str();
}

}  // namespace

int detail::uncovered_seed_demand(
    const fastkag::Simulator& observation, int player,
    const std::vector<fastkag::PlayerAction>& tape, const Debt& debt) {
  return ::g001::purchase_failure_rolling::uncovered_seed_demand(
      observation, player, tape, debt);
}

Owner::Owner(const fastkag::NativeTeammateExecutor& executor, int player,
             int route, Config config)
    : executor_(&executor), player_(player), route_(route), config_(config) {
  if (player < 0 || player > 1 || route < 0 || route >= executor.route_count())
    throw std::invalid_argument("invalid purchase failure owner binding");
  if (config.maximum_attempts < 1 || config.debt_days < 1 ||
      config.maximum_retry_quantity < 1)
    throw std::invalid_argument("invalid purchase failure owner config");
}

bool Owner::observe(const fastkag::Simulator& observation) {
  if (!valid_player(observation, player_)) return false;
  ++metrics_.observations;
  last_observed_step_ = observation.step_count();
  if (!pending_) return true;
  if (observation.step_count() != pending_->step + 1) return false;
  if (!config_.enabled) {
    pending_.reset();
    return true;
  }

  const auto& fills = observation.last_market_fills()[player_];
  for (std::size_t slot = 0; slot < pending_->final_action.market.size(); ++slot) {
    const auto order = pending_->final_action.market[slot];
    if (order.op != Op::BUY_SEED && order.op != Op::BUY_ANIMAL) continue;
    const int filled = slot < fills.size() ? fills[slot] : 0;
    const auto linked = std::find_if(
        pending_->receipt_bindings.begin(), pending_->receipt_bindings.end(),
        [&](const Proposal::ReceiptBinding& binding) {
          return binding.market && binding.index == static_cast<int>(slot);
        });
    if (linked != pending_->receipt_bindings.end()) {
      auto* debt = find_debt(debts_, linked->debt_id);
      if (!debt || debt->status != DebtStatus::AwaitingReceipt) continue;
      if (filled >= order.quantity) {
        debt->status = DebtStatus::Completed;
        ++metrics_.completed_debts;
        ++metrics_.purchase_retry_fills;
      } else {
        debt->status = DebtStatus::Open;
      }
      continue;
    }
    if (filled >= order.quantity) continue;
    const bool animal = order.op == Op::BUY_ANIMAL;
    if (find_open_acquisition(debts_, order.item, animal)) continue;
    if ((filled > 0 && !config_.retry_partial_purchases) ||
        (animal && !config_.retry_animal_purchases) ||
        (!animal && !config_.retry_seed_purchases))
      continue;
    Debt debt;
    debt.id = next_debt_id_++;
    debt.root_id = debt.id;
    debt.kind = animal ? DebtKind::AcquireAnimal : DebtKind::AcquireSeed;
    debt.item = order.item;
    debt.action = {order.op, order.item, order.quantity - filled};
    debt.source_step = pending_->step;
    debt.earliest_step = observation.step_count();
    debt.deadline_step = pending_->step +
                         config_.debt_days * observation.config().turns_per_day;
    debt.provenance = "physical_zero_or_partial_market_receipt";
    debts_.push_back(std::move(debt));
    ++metrics_.purchase_failures;
    if (animal)
      ++metrics_.animal_purchase_failures;
    else
      ++metrics_.seed_purchase_failures;
  }

  for (std::size_t actor = 0; actor < pending_->final_action.units.size(); ++actor) {
    const auto action = pending_->final_action.units[actor];
    if (!supported_unit(action)) continue;
    const auto position = actor_position(pending_->before, player_,
                                         static_cast<int>(actor));
    const bool confirmed = unit_effect(pending_->before, observation, player_,
                                       static_cast<int>(actor), action, position);
    const auto linked = std::find_if(
        pending_->receipt_bindings.begin(), pending_->receipt_bindings.end(),
        [&](const Proposal::ReceiptBinding& binding) {
          return !binding.market && binding.index == static_cast<int>(actor);
        });
    if (linked != pending_->receipt_bindings.end()) {
      auto* debt = find_debt(debts_, linked->debt_id);
      if (!debt || debt->status != DebtStatus::AwaitingReceipt) continue;
      if (confirmed) {
        debt->status = DebtStatus::Completed;
        ++metrics_.completed_debts;
      } else {
        debt->status = DebtStatus::Open;
        ++metrics_.unit_effect_failures;
      }
      continue;
    }
    if (confirmed || (action.op != Op::PLANT && action.op != Op::PLACE))
      continue;
    const bool animal = action.op == Op::PLACE;
    const auto* acquisition =
        find_recent_acquisition(debts_, action.item, animal, pending_->step);
    if (!acquisition) continue;
    const auto kind = animal ? DebtKind::Place : DebtKind::Plant;
    if (has_chain_debt(debts_, acquisition->root_id, kind, pending_->step,
                       static_cast<int>(actor)))
      continue;
    Debt debt;
    debt.id = next_debt_id_++;
    debt.root_id = acquisition->root_id;
    debt.kind = kind;
    debt.item = action.item;
    debt.action = action;
    debt.actor = static_cast<int>(actor);
    debt.tile = position;
    debt.source_step = pending_->step;
    debt.earliest_step = observation.step_count();
    debt.deadline_step = acquisition->deadline_step;
    debt.provenance = "physical_first_consumer_effect_failure";
    debts_.push_back(std::move(debt));
    ++metrics_.unit_effect_failures;
  }

  for (auto& debt : debts_) {
    if (debt.status == DebtStatus::Open &&
        observation.step_count() > debt.deadline_step) {
      debt.status = DebtStatus::Expired;
      ++metrics_.expired_debts;
    }
  }
  pending_.reset();
  return true;
}

Proposal Owner::propose(const fastkag::Simulator& observation) const {
  const auto begin = std::chrono::steady_clock::now();
  if (!valid_player(observation, player_) || pending_ ||
      (last_observed_step_ >= 0 &&
       last_observed_step_ != observation.step_count()))
    throw std::logic_error("owner must observe the exact current hand first");

  Proposal proposal;
  proposal.step = observation.step_count();
  proposal.generation = generation_;
  proposal.observation_hash = observation_hash(observation, player_);
  proposal.observation_before = observation;
  proposal.next_native_state = native_state_;
  proposal.base_action = executor_->action_external(
      observation, player_, route_, proposal.next_native_state);
  proposal.final_action = proposal.base_action;
  proposal.debts_after_finalize = debts_;
  proposal.base_action_hash = action_hash(proposal.base_action);

  if (config_.enabled) {
    const auto& tape = executor_->route_tape(route_);
    for (auto& debt : proposal.debts_after_finalize) {
      if (debt.status != DebtStatus::Open) continue;
      const bool covered_seed = debt.kind == DebtKind::AcquireSeed &&
          config_.require_route_seed_demand &&
          uncovered_seed_demand(observation, player_, tape, debt) == 0;
      const bool covered_animal = debt.kind == DebtKind::AcquireAnimal &&
          config_.require_route_animal_demand &&
          uncovered_animal_demand(observation, player_, tape, debt) == 0;
      if (covered_seed || covered_animal) debt.status = DebtStatus::Completed;
    }
    const int deadline = proposal.step +
                         config_.debt_days * observation.config().turns_per_day;
    // First, causally block a consumer whose acquisition has not received a
    // physical fill. This is current-action reasoning, not a future scan.
    if (config_.block_unready_consumers)
    for (std::size_t actor = 0; actor < proposal.base_action.units.size(); ++actor) {
      const auto original = proposal.base_action.units[actor];
      const bool animal_consumer = original.op == Op::PICKUP ||
                                   original.op == Op::PLACE;
      const bool seed_consumer = original.op == Op::PLANT;
      const bool crop_suffix = original.op == Op::WATER ||
                               original.op == Op::HARVEST;
      const bool animal_suffix = original.op == Op::FEED ||
                                 original.op == Op::CARE ||
                                 original.op == Op::HARVEST;
      if (!animal_consumer && !seed_consumer && !crop_suffix &&
          !animal_suffix)
        continue;
      const bool animal = animal_consumer &&
                          static_cast<int>(original.item) >= 9;
      const auto position = actor_position(observation, player_,
                                           static_cast<int>(actor));
      const Debt* predecessor = nullptr;
      if (seed_consumer || animal_consumer)
        predecessor = find_open_acquisition(
            proposal.debts_after_finalize, original.item, animal);
      if (!predecessor && original.op == Op::PLACE)
        predecessor = find_open_predecessor(proposal.debts_after_finalize,
                                            DebtKind::PickupAnimal,
                                            original.item, {-1, -1});
      if (!predecessor && crop_suffix)
        predecessor = find_open_predecessor(proposal.debts_after_finalize,
                                            DebtKind::Plant, Item::NONE,
                                            position);
      if (!predecessor && animal_suffix)
        predecessor = find_open_predecessor(proposal.debts_after_finalize,
                                            DebtKind::Place, Item::NONE,
                                            position);
      if (!predecessor) continue;
      const auto kind = debt_kind(original);
      if (!has_chain_debt(proposal.debts_after_finalize,
                          predecessor->root_id, kind, proposal.step,
                          static_cast<int>(actor))) {
        Debt debt;
        debt.id = next_debt_id_ + proposal.debts_after_finalize.size() -
                  debts_.size();
        debt.root_id = predecessor->root_id;
        debt.kind = kind;
        debt.item = original.item == Item::NONE ? predecessor->item
                                                : original.item;
        debt.action = original;
        debt.actor = static_cast<int>(actor);
        debt.tile = position;
        debt.source_step = proposal.step;
        debt.earliest_step = proposal.step + 1;
        debt.deadline_step = std::min(deadline, predecessor->deadline_step);
        debt.provenance = "consumer_blocked_until_physical_purchase_receipt";
        proposal.debts_after_finalize.push_back(std::move(debt));
      }
      proposal.final_action.units[actor] = {Op::PASS, Item::NONE, 1};
      proposal.diagnostics.push_back("causal_block actor=" +
                                     std::to_string(actor));
    }

    // A physically ready unit debt may consume only the same actor's current
    // non-MOVE slot. Displaced supported work becomes a typed debt; unknown
    // operations are never overwritten.
    if (config_.retry_unit_debts)
    for (std::size_t debt_index = 0;
         debt_index < proposal.debts_after_finalize.size(); ++debt_index) {
      auto& debt = proposal.debts_after_finalize[debt_index];
      if (debt.status != DebtStatus::Open ||
          debt.kind == DebtKind::AcquireSeed ||
          debt.kind == DebtKind::AcquireAnimal ||
          debt.attempts >= config_.maximum_attempts ||
          proposal.step < debt.earliest_step ||
          proposal.step > debt.deadline_step)
        continue;
      int actor = debt.actor;
      if (actor < 0 || actor >= static_cast<int>(proposal.final_action.units.size()))
        continue;
      const auto original = proposal.final_action.units[actor];
      if (is_move(original.op) || !executable(observation, player_, debt, actor))
        continue;
      const auto retry_action = debt.action;
      if (original.op != Op::PASS && !same_action(original, debt.action)) {
        if (!supported_unit(original)) {
          proposal.diagnostics.push_back("unsupported_displacement actor=" +
                                         std::to_string(actor) + " op=" +
                                         std::to_string(static_cast<int>(original.op)));
          continue;
        }
        const auto position = actor_position(observation, player_, actor);
        Debt displaced;
        displaced.id = next_debt_id_ + proposal.debts_after_finalize.size() -
                       debts_.size();
        displaced.root_id = debt.root_id;
        displaced.kind = debt_kind(original);
        displaced.item = original.item;
        displaced.action = original;
        displaced.actor = actor;
        displaced.tile = position;
        displaced.source_step = proposal.step;
        displaced.earliest_step = proposal.step + 1;
        displaced.deadline_step = debt.deadline_step;
        displaced.provenance = "typed_action_displaced_by_causal_repair";
        proposal.debts_after_finalize.push_back(std::move(displaced));
        proposal.diagnostics.push_back("typed_displacement actor=" +
                                       std::to_string(actor));
      }
      auto& selected_debt = proposal.debts_after_finalize[debt_index];
      proposal.final_action.units[actor] = retry_action;
      selected_debt.status = DebtStatus::AwaitingReceipt;
      ++selected_debt.attempts;
      proposal.attempted_debt_ids.push_back(selected_debt.id);
      proposal.receipt_bindings.push_back(
          {selected_debt.id, false, actor});
      proposal.diagnostics.push_back("unit_retry debt=" +
                                     std::to_string(selected_debt.id));
      break;
    }

    // Retry at most one purchase per hand. We only emit when cash/capacity is
    // physically sufficient now; no assumed opponent/future fill is used.
    if (config_.retry_purchases)
    for (auto& debt : proposal.debts_after_finalize) {
        int retry_quantity = std::max(1, debt.action.quantity);
        retry_quantity = std::min(retry_quantity,
                                  config_.maximum_retry_quantity);
        if (debt.kind == DebtKind::AcquireSeed &&
            config_.require_route_seed_demand)
          retry_quantity = std::min(
              retry_quantity,
              uncovered_seed_demand(observation, player_,
                                    executor_->route_tape(route_), debt));
        if (debt.kind == DebtKind::AcquireAnimal &&
            config_.require_route_animal_demand)
          retry_quantity = std::min(
              retry_quantity,
              uncovered_animal_demand(observation, player_,
                                      executor_->route_tape(route_), debt));
        if ((debt.kind != DebtKind::AcquireSeed &&
             debt.kind != DebtKind::AcquireAnimal) ||
            (debt.kind == DebtKind::AcquireSeed &&
             !config_.retry_seed_purchases) ||
            (debt.kind == DebtKind::AcquireAnimal &&
             !config_.retry_animal_purchases) ||
            debt.status != DebtStatus::Open ||
            debt.attempts >= config_.maximum_attempts ||
            proposal.step < debt.earliest_step ||
            proposal.step > debt.deadline_step ||
            retry_quantity <= 0 ||
            !can_retry_purchase(observation, player_, debt,
                                proposal.final_action, retry_quantity))
          continue;
        const auto op = debt.kind == DebtKind::AcquireSeed ? Op::BUY_SEED
                                                           : Op::BUY_ANIMAL;
        const Action order{op, debt.item, retry_quantity};
        const auto existing = std::find_if(
            proposal.final_action.market.begin(),
            proposal.final_action.market.end(),
            [&](Action value) { return same_action(value, order); });
        int slot = -1;
        if (existing != proposal.final_action.market.end()) {
          slot = static_cast<int>(
              std::distance(proposal.final_action.market.begin(), existing));
        } else if (proposal.final_action.market.size() <
                   static_cast<std::size_t>(
                       observation.config().max_market_orders)) {
          slot = static_cast<int>(proposal.final_action.market.size());
          proposal.final_action.market.push_back(order);
        } else {
          continue;
        }
        debt.status = DebtStatus::AwaitingReceipt;
        ++debt.attempts;
        proposal.attempted_debt_ids.push_back(debt.id);
        proposal.receipt_bindings.push_back({debt.id, true, slot});
        proposal.diagnostics.push_back("purchase_retry debt=" +
                                       std::to_string(debt.id));
        break;
    }

    // If a repaired acquisition is already confirmed, bind the first natural
    // consumer receipt into the same typed chain without altering the action.
    if (config_.monitor_natural_consumers)
    for (std::size_t actor = 0; actor < proposal.final_action.units.size(); ++actor) {
      const auto action = proposal.final_action.units[actor];
      const bool plant = action.op == Op::PLANT;
      const bool animal = (action.op == Op::PICKUP || action.op == Op::PLACE) &&
                          static_cast<int>(action.item) >= 9;
      if (!plant && !animal) continue;
      const auto* acquisition = find_recent_acquisition(
          proposal.debts_after_finalize, action.item, animal, proposal.step);
      if (!acquisition || acquisition->status != DebtStatus::Completed) continue;
      const auto kind = debt_kind(action);
      if (has_chain_debt(proposal.debts_after_finalize, acquisition->root_id,
                         kind, proposal.step, static_cast<int>(actor)))
        continue;
      if (kind == DebtKind::Place &&
          has_open_dependency(proposal.debts_after_finalize,
                              acquisition->root_id, DebtKind::PickupAnimal))
        continue;
      Debt monitored;
      monitored.id = next_debt_id_ + proposal.debts_after_finalize.size() -
                     debts_.size();
      monitored.root_id = acquisition->root_id;
      monitored.kind = kind;
      monitored.status = DebtStatus::AwaitingReceipt;
      monitored.item = action.item;
      monitored.action = action;
      monitored.actor = static_cast<int>(actor);
      monitored.tile = actor_position(observation, player_,
                                      static_cast<int>(actor));
      monitored.source_step = proposal.step;
      monitored.earliest_step = proposal.step;
      monitored.deadline_step = acquisition->deadline_step;
      monitored.attempts = 1;
      monitored.provenance = "natural_consumer_bound_to_repaired_acquisition";
      proposal.attempted_debt_ids.push_back(monitored.id);
      proposal.receipt_bindings.push_back(
          {monitored.id, false, static_cast<int>(actor)});
      proposal.debts_after_finalize.push_back(std::move(monitored));
    }
  }

  proposal.final_action_hash = action_hash(proposal.final_action);
  proposal.binding_hash = proposal_binding_hash(proposal);
  const auto elapsed = std::chrono::steady_clock::now() - begin;
  metrics_.proposal_time_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  ++metrics_.proposals;
  return proposal;
}

FinalizeStatus Owner::finalize(const Proposal& proposal,
                               const fastkag::PlayerAction& selected_action) {
  return finalize_impl(proposal, selected_action, proposal.next_native_state,
                       false);
}

FinalizeStatus Owner::finalize_composed(
    const Proposal& proposal, const fastkag::PlayerAction& selected_action,
    fastkag::NativeAgentState committed_native_state) {
  return finalize_impl(proposal, selected_action,
                       std::move(committed_native_state), true);
}

FinalizeStatus Owner::finalize_impl(
    const Proposal& proposal, const fastkag::PlayerAction& selected_action,
    fastkag::NativeAgentState committed_native_state, bool composed) {
  if (proposal.generation != generation_ || proposal.step != last_observed_step_ ||
      pending_ || proposal.binding_hash != proposal_binding_hash(proposal)) {
    ++metrics_.stale_finalize_rejections;
    return FinalizeStatus::StaleProposal;
  }
  bool purchase_edits_preserved =
      selected_action.units.size() == proposal.final_action.units.size() &&
      selected_action.market.size() == proposal.final_action.market.size();
  for (std::size_t actor = 0;
       purchase_edits_preserved && actor < proposal.base_action.units.size();
       ++actor)
    if (!same_action(proposal.base_action.units[actor],
                     proposal.final_action.units[actor]) &&
        !same_action(selected_action.units[actor],
                     proposal.final_action.units[actor]))
      purchase_edits_preserved = false;
  for (std::size_t slot = 0;
       purchase_edits_preserved && slot < proposal.final_action.market.size();
       ++slot) {
    const Action base = slot < proposal.base_action.market.size()
                            ? proposal.base_action.market[slot]
                            : Action{};
    if (!same_action(base, proposal.final_action.market[slot]) &&
        !same_action(selected_action.market[slot],
                     proposal.final_action.market[slot]))
      purchase_edits_preserved = false;
  }
  if ((!composed &&
       (!same_player_action(selected_action, proposal.final_action) ||
        action_hash(selected_action) != proposal.final_action_hash)) ||
      (composed && !purchase_edits_preserved)) {
    ++metrics_.second_writer_rejections;
    return FinalizeStatus::SecondWriter;
  }
  for (std::size_t actor = 0; actor < proposal.base_action.units.size(); ++actor) {
    const bool source_move = is_move(proposal.base_action.units[actor].op);
    const bool final_move = is_move(proposal.final_action.units[actor].op);
    metrics_.move_source += source_move;
    metrics_.move_final += final_move;
    if (source_move != final_move ||
        (source_move && !same_action(proposal.base_action.units[actor],
                                    proposal.final_action.units[actor])))
      ++metrics_.move_mismatch;
  }
  for (const auto& diagnostic : proposal.diagnostics) {
    if (diagnostic.starts_with("purchase_retry")) ++metrics_.purchase_retries;
    if (diagnostic.starts_with("causal_block")) ++metrics_.causal_blocks;
    if (diagnostic.starts_with("unit_retry")) ++metrics_.unit_retries;
    if (diagnostic.starts_with("typed_displacement"))
      ++metrics_.displaced_typed_actions;
    if (diagnostic.starts_with("unsupported_displacement"))
      ++metrics_.unsupported_displacements;
  }
  native_state_ = std::move(committed_native_state);
  debts_ = proposal.debts_after_finalize;
  for (const auto& debt : debts_)
    next_debt_id_ = std::max(next_debt_id_, debt.id + 1);
  Pending pending(proposal.observation_before);
  pending.step = proposal.step;
  pending.base_action = proposal.base_action;
  pending.final_action = selected_action;
  pending.receipt_bindings = proposal.receipt_bindings;
  pending_ = std::move(pending);
  ++generation_;
  ++metrics_.finalized;
  return FinalizeStatus::Selected;
}

std::vector<std::string> Owner::terminal_diagnostics(int terminal_step) const {
  std::vector<std::string> result;
  for (const auto& debt : debts_) {
    if (debt.status == DebtStatus::Completed) continue;
    result.push_back("terminal_step=" + std::to_string(terminal_step) + " " +
                     debt_string(debt));
  }
  return result;
}

const char* debt_kind_name(DebtKind kind) noexcept {
  switch (kind) {
    case DebtKind::AcquireSeed: return "acquire_seed";
    case DebtKind::AcquireAnimal: return "acquire_animal";
    case DebtKind::PickupAnimal: return "pickup_animal";
    case DebtKind::PickupInput: return "pickup_input";
    case DebtKind::Plant: return "plant";
    case DebtKind::Place: return "place";
    case DebtKind::Water: return "water";
    case DebtKind::Feed: return "feed";
    case DebtKind::Care: return "care";
    case DebtKind::Harvest: return "harvest";
  }
  return "unknown";
}

const char* debt_status_name(DebtStatus status) noexcept {
  switch (status) {
    case DebtStatus::Open: return "open";
    case DebtStatus::AwaitingReceipt: return "awaiting_receipt";
    case DebtStatus::Completed: return "completed";
    case DebtStatus::Expired: return "expired";
  }
  return "unknown";
}

const char* finalize_status_name(FinalizeStatus status) noexcept {
  switch (status) {
    case FinalizeStatus::Selected: return "selected";
    case FinalizeStatus::StaleProposal: return "stale_proposal";
    case FinalizeStatus::SecondWriter: return "second_writer";
  }
  return "unknown";
}

std::uint64_t action_hash(const fastkag::PlayerAction& action) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, action.units.size());
  for (const auto value : action.units) {
    hash_add(hash, static_cast<std::uint8_t>(value.op));
    hash_add(hash, static_cast<std::uint8_t>(value.item));
    hash_add(hash, static_cast<std::uint32_t>(value.quantity));
  }
  hash_add(hash, action.market.size());
  for (const auto value : action.market) {
    hash_add(hash, static_cast<std::uint8_t>(value.op));
    hash_add(hash, static_cast<std::uint8_t>(value.item));
    hash_add(hash, static_cast<std::uint32_t>(value.quantity));
  }
  return hash;
}

}  // namespace g001::purchase_failure_rolling
