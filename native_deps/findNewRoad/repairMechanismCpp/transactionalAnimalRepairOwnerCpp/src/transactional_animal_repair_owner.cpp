#include "transactional_animal_repair_owner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace g001::transactional_animal_repair {
namespace {
namespace persistent = persistent_production;
namespace tx = repair_owner_composer;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;

bool same_action(const Action &a, const Action &b) {
  return a.op == b.op && a.item == b.item && a.quantity == b.quantity;
}
bool move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST || op == Op::WEST;
}
int idx(Item x) { return static_cast<int>(x); }
Position actor_pos(const Simulator &s, int p, int a) {
  const auto &f = s.farms()[p];
  if (a == 0)
    return f.farmer;
  if (a < 0 || a > (int)f.hands.size())
    return {-1, -1};
  return f.hands[a - 1];
}
bool shed_adj(const Simulator &s, Position p) {
  int h = s.config().board_size / 2;
  return (p.x == h - 1 || p.x == h) && (p.y == h - 1 || p.y == h);
}
const fastkag::Tile *tile_at(const Simulator &s, int p, persistent::TileKey k) {
  int n = s.config().board_size;
  if (k.row < 0 || k.column < 0 || k.row >= n || k.column >= n)
    return nullptr;
  return &s.farms()[p].tiles[(std::size_t)(k.row * n + k.column)];
}
std::uint64_t generation(const Simulator &s, int a) {
  return (std::uint64_t(s.day() + 1) << 32U) | std::uint32_t(a + 1);
}
persistent::Observation observe(const Simulator &s, int p,
                                persistent::TileKey k, int a) {
  persistent::Observation o;
  o.step = s.step_count();
  o.day = s.day();
  o.target_key = k;
  if (auto *t = tile_at(s, p, k))
    o.target = *t;
  const auto &pr = s.privates()[p];
  o.shed = pr.shed;
  o.seeds = pr.seeds;
  o.actor.identity = {a, generation(s, a)};
  auto pos = actor_pos(s, p, a);
  o.actor.position = {pos.y, pos.x};
  o.actor.shed_adjacent = shed_adj(s, pos);
  if (a >= 0 && a < (int)pr.inventories.size())
    o.actor.inventory = pr.inventories[a];
  return o;
}
void hadd(std::uint64_t &h, std::uint64_t v) {
  for (int b = 0; b < 8; b++) {
    h ^= (v >> (b * 8)) & 255ULL;
    h *= 1099511628211ULL;
  }
}
bool same_receipt(const repair_fork::ActionReceipt &a,
                  const repair_fork::ActionReceipt &b) {
  return a.submitted_step == b.submitted_step && a.actor == b.actor &&
         a.manifest_generation == b.manifest_generation &&
         a.prefix_manifest_hash == b.prefix_manifest_hash &&
         a.post_prefix_state_fingerprint == b.post_prefix_state_fingerprint &&
         same_action(a.emitted, b.emitted);
}
bool purchase_shape(const repair_fork::PurchaseReceipt &a,
                    const repair_fork::PurchaseReceipt &b) {
  return a.debt_id == b.debt_id && a.submitted_step == b.submitted_step &&
         a.operation == b.operation && a.item == b.item &&
         a.requested == b.requested && a.market_slot == b.market_slot &&
         a.compile_status == b.compile_status;
}
} // namespace

struct TransactionalAnimalRepairOwner::Impl {
  struct PendingUnit {
    std::uint64_t objective{}, lease{}, prefix{};
    persistent::ActorIdentity actor;
    persistent::TileKey target;
    repair_fork::ActionReceipt receipt;
  };
  struct PendingBuy {
    std::uint64_t objective{};
    purchase_recovery::Proposal proposal;
    repair_fork::PurchaseReceipt shape;
    int player{};
    bool staged{};
  };
  struct Cached {
    tx::PreparedRepair pub;
    Simulator env;
    int player{}, step{};
    std::uint64_t state_version{};
    std::array<PlayerAction, 2> raw_joint;
    persistent::Ledger speculative;
    std::map<persistent::TileKey, std::uint64_t> owners;
    std::map<persistent::TileKey, std::uint64_t> typed_sources;
    std::vector<std::pair<std::uint64_t, purchase_recovery::Proposal>> buys;
    std::optional<persistent::ReadyTransition> ready;
    std::optional<persistent::Observation> before;
  };
  persistent::Ledger ledger;
  std::map<persistent::TileKey, std::uint64_t> owners;
  std::map<persistent::TileKey, std::uint64_t> typed_sources;
  std::vector<PendingUnit> pending_units;
  std::vector<PendingBuy> pending_buys;
  std::map<int, std::uint64_t> generations;
  mutable std::map<std::uint64_t, Cached> cache;
  std::uint64_t state_version{1};
  Audit audit;
};

namespace {
std::uint64_t token(const TransactionalAnimalRepairOwner::Impl::Cached &c) {
  std::uint64_t h = 1469598103934665603ULL;
  hadd(h, repair_fork::full_unit_phase_state_fingerprint(c.env));
  hadd(h, c.player);
  hadd(h, c.step);
  hadd(h, c.state_version);
  for (auto &a : c.pub.units) {
    hadd(h, (std::uint8_t)a.op);
    hadd(h, (std::uint8_t)a.item);
    hadd(h, a.quantity);
  }
  for (auto &p : c.pub.required_purchases) {
    hadd(h, p.debt_id);
    hadd(h, (std::uint8_t)p.item);
    hadd(h, p.quantity);
  }
  for (const auto &[target, source] : c.typed_sources) {
    hadd(h, static_cast<std::uint32_t>(target.row));
    hadd(h, static_cast<std::uint32_t>(target.column));
    hadd(h, source);
  }
  return h ? h : 1;
}
bool sealed(const repair_fork::RepairContext &c, int selected) {
  for (int a = 0; a < (int)c.raw_g001.units.size(); a++)
    if (a != selected) {
      auto op = c.raw_g001.units[a].op;
      if (op != Op::PASS && !move(op))
        return false;
    }
  return true;
}
bool compiled(repair_fork::PurchaseCompileStatus status) {
  return status == repair_fork::PurchaseCompileStatus::BoundExisting ||
         status == repair_fork::PurchaseCompileStatus::Appended;
}
} // namespace

TransactionalAnimalRepairOwner::TransactionalAnimalRepairOwner()
    : impl_(std::make_unique<Impl>()) {}
TransactionalAnimalRepairOwner::~TransactionalAnimalRepairOwner() = default;
TransactionalAnimalRepairOwner::TransactionalAnimalRepairOwner(
    TransactionalAnimalRepairOwner &&) noexcept = default;
TransactionalAnimalRepairOwner &TransactionalAnimalRepairOwner::operator=(
    TransactionalAnimalRepairOwner &&) noexcept = default;
std::string TransactionalAnimalRepairOwner::name() const {
  return "transactional_animal_repair_owner_v1";
}

tx::SettlementResult TransactionalAnimalRepairOwner::settle_owned(
    const repair_fork::RepairContext &c,
    std::span<const repair_fork::ActionReceipt> action_receipts,
    std::span<const repair_fork::PurchaseReceipt> purchase_receipts) {
  if (c.player < 0 || c.player > 1 || c.step != c.phase_start.step_count())
    throw std::invalid_argument("invalid transactional animal settle context");
  // Validate every owned shape before applying either action or purchase
  // observations. Protocol rejection must leave this owner unchanged.
  if (action_receipts.size() != impl_->pending_units.size())
    return tx::SettlementResult::ProtocolInvalid;
  std::vector<bool> action_preflight(action_receipts.size());
  for (const auto &pending : impl_->pending_units) {
    std::size_t match = action_receipts.size();
    for (std::size_t index = 0; index < action_receipts.size(); ++index)
      if (!action_preflight[index] &&
          same_receipt(action_receipts[index], pending.receipt)) {
        match = index;
        break;
      }
    if (match == action_receipts.size() ||
        pending.receipt.submitted_step + 1 != c.step)
      return tx::SettlementResult::ProtocolInvalid;
    action_preflight[match] = true;
  }
  if (purchase_receipts.size() != impl_->pending_buys.size()) {
    ++impl_->audit.unresolved_purchase_receipts;
    return tx::SettlementResult::ProtocolInvalid;
  }
  std::vector<bool> purchase_preflight(purchase_receipts.size());
  for (const auto &pending : impl_->pending_buys) {
    std::size_t match = purchase_receipts.size();
    for (std::size_t index = 0; index < purchase_receipts.size(); ++index) {
      if (!purchase_preflight[index] &&
          purchase_shape(purchase_receipts[index], pending.shape)) {
        match = index;
        break;
      }
    }
    if (match == purchase_receipts.size() ||
        pending.shape.submitted_step + 1 != c.step) {
      ++impl_->audit.unresolved_purchase_receipts;
      return tx::SettlementResult::ProtocolInvalid;
    }
    const auto &receipt = purchase_receipts[match];
    const bool compiled_receipt = compiled(receipt.compile_status);
    if (receipt.filled < 0 || receipt.filled > receipt.requested ||
        (compiled_receipt && (!pending.staged || receipt.market_slot < 0)) ||
        (!compiled_receipt &&
         (pending.staged || receipt.filled != 0 || receipt.market_slot != -1))) {
      ++impl_->audit.unresolved_purchase_receipts;
      return tx::SettlementResult::ProtocolInvalid;
    }
    purchase_preflight[match] = true;
  }
  const bool had_pending =
      !impl_->pending_units.empty() || !impl_->pending_buys.empty();
  bool physical_success = true;

  std::vector<bool> used_actions(action_receipts.size());
  for (const auto &pending : impl_->pending_units) {
    std::size_t match = action_receipts.size();
    for (std::size_t index = 0; index < action_receipts.size(); ++index) {
      if (!used_actions[index] &&
          same_receipt(action_receipts[index], pending.receipt)) {
        match = index;
        break;
      }
    }
    if (match != action_receipts.size())
      used_actions[match] = true;
    const auto after = observe(c.phase_start, c.player, pending.target,
                               pending.actor.actor_id);
    const auto settlement = impl_->ledger.observe_transition(
        pending.objective, pending.lease,
        match == action_receipts.size() ? 0 : pending.prefix, after);
    if (match == action_receipts.size() ||
        settlement.status != persistent::UnitReceiptStatus::Success) {
      physical_success = false;
      ++impl_->audit.unit_failure;
    } else {
      ++impl_->audit.unit_success;
    }
  }
  impl_->pending_units.clear();

  std::vector<bool> used_purchases(purchase_receipts.size());
  int largest_slot = -1;
  for (const auto &pending : impl_->pending_buys) {
    std::size_t match = purchase_receipts.size();
    for (std::size_t index = 0; index < purchase_receipts.size(); ++index) {
      if (!used_purchases[index] &&
          purchase_shape(purchase_receipts[index], pending.shape)) {
        match = index;
        break;
      }
    }
    if (match == purchase_receipts.size()) {
      physical_success = false;
      continue;
    }
    used_purchases[match] = true;
    const auto &receipt = purchase_receipts[match];
    if (compiled(receipt.compile_status)) {
      if (!pending.staged || receipt.market_slot < 0 || receipt.filled < 0) {
        physical_success = false;
      } else {
        largest_slot = std::max(largest_slot, receipt.market_slot);
      }
    } else if (pending.staged || receipt.filled != 0 ||
               receipt.market_slot != -1) {
      physical_success = false;
    }
  }

  if (largest_slot >= 0) {
    std::vector<std::int32_t> fills(static_cast<std::size_t>(largest_slot + 1));
    for (std::size_t index = 0; index < purchase_receipts.size(); ++index) {
      if (!used_purchases[index])
        continue;
      const auto &receipt = purchase_receipts[index];
      if (!compiled(receipt.compile_status) || receipt.market_slot < 0)
        continue;
      auto &fill = fills[static_cast<std::size_t>(receipt.market_slot)];
      if (fill != 0)
        physical_success = false;
      fill = receipt.filled;
    }
    purchase_recovery::ReceiptObservation observation;
    observation.step = c.step;
    observation.slot_fills = fills;
    const auto &state = c.phase_start.privates()[c.player];
    for (int animal = 0; animal < fastkag::N_ANIMALS; ++animal) {
      const int item = static_cast<int>(Item::GOOSE) + animal;
      int count = state.shed[static_cast<std::size_t>(item)];
      for (const auto &inventory : state.inventories)
        count += inventory[static_cast<std::size_t>(item)];
      observation.animals_after[static_cast<std::size_t>(animal)] = count;
    }
    const auto settlements = impl_->ledger.observe_purchases(observation);
    for (const auto &pending : impl_->pending_buys) {
      if (!pending.staged)
        continue;
      const auto found = std::find_if(
          settlements.begin(), settlements.end(), [&](const auto &value) {
            return value.objective_id == pending.objective;
          });
      if (found == settlements.end()) {
        physical_success = false;
        continue;
      }
      using Fill = purchase_recovery::FillStatus;
      if (found->settlement.status == Fill::Zero)
        ++impl_->audit.purchase_zero;
      else if (found->settlement.status == Fill::Partial)
        ++impl_->audit.purchase_partial;
      else if (found->settlement.status == Fill::Full)
        ++impl_->audit.purchase_full;
      else
        physical_success = false;
    }
  }
  impl_->pending_buys.clear();
  if (had_pending)
    ++impl_->state_version;
  return physical_success ? tx::SettlementResult::AppliedSuccess
                          : tx::SettlementResult::AppliedPhysicalFailure;
}

tx::SettlementResult TransactionalAnimalRepairOwner::validate_settle(
    const repair_fork::RepairContext &context,
    std::span<const repair_fork::ActionReceipt> action_receipts,
    std::span<const repair_fork::PurchaseReceipt> purchase_receipts) const {
  TransactionalAnimalRepairOwner snapshot;
  *snapshot.impl_ = *impl_;
  return snapshot.settle_owned(context, action_receipts, purchase_receipts);
}

tx::PreparedRepair TransactionalAnimalRepairOwner::prepare(
    const repair_fork::RepairContext &c) const {
  return prepare(c, std::span<const TypedAnimalObligation>{});
}

tx::PreparedRepair TransactionalAnimalRepairOwner::prepare(
    const repair_fork::RepairContext &c, tx::TypedRepairInput input) const {
  return prepare(c, input.animals);
}

tx::PreparedRepair TransactionalAnimalRepairOwner::prepare(
    const repair_fork::RepairContext &c,
    std::span<const TypedAnimalObligation> obligations) const {
  if (c.player < 0 || c.player > 1 || c.step < 0 ||
      c.step != c.phase_start.step_count())
    throw std::invalid_argument("invalid transactional animal prepare context");
  if (!impl_->pending_units.empty() || !impl_->pending_buys.empty())
    throw std::logic_error("settle animal receipts first");
  ++impl_->audit.prepares;
  Impl::Cached x;
  x.env = c.phase_start;
  x.player = c.player;
  x.step = c.step;
  x.state_version = impl_->state_version;
  x.raw_joint = c.raw_joint;
  x.speculative = impl_->ledger;
  x.owners = impl_->owners;
  x.typed_sources = impl_->typed_sources;
  x.pub.units = c.raw_g001.units;
  for (std::size_t a = 0; a < x.pub.units.size(); a++)
    x.pub.sources.push_back({(int)a, c.step, x.pub.units[a]});
  // BUY_ANIMAL does not identify a PLACE tile. Open only targets named by the
  // typed route/production compiler; board-order structure guessing is unsafe.
  for (const auto &m : c.raw_g001.market)
    if (m.op == Op::BUY_ANIMAL && m.quantity > 0 &&
        animal_lifecycle::definition(m.item)) {
      const auto definition = *animal_lifecycle::definition(m.item);
      std::vector<const TypedAnimalObligation *> candidates;
      for (const auto &value : obligations) {
        if (value.id == 0 || value.player != c.player || value.actor < 0 ||
            value.actor >= static_cast<int>(c.raw_g001.units.size()) ||
            value.place_step < c.step || value.animal != m.item ||
            value.place_action.op != Op::PLACE ||
            value.place_action.item != m.item ||
            value.place_action.quantity <= 0)
          continue;
        candidates.push_back(&value);
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const auto *left, const auto *right) {
                  return std::tuple{left->place_step, left->actor,
                                    left->target.row, left->target.column,
                                    left->id} <
                         std::tuple{right->place_step, right->actor,
                                    right->target.row, right->target.column,
                                    right->id};
                });
      int left = m.quantity;
      for (std::size_t index = 0; index < candidates.size() && left > 0;) {
        const auto *candidate = candidates[index];
        std::size_t end = index + 1;
        bool ambiguous = false;
        while (end < candidates.size() &&
               candidates[end]->target == candidate->target) {
          ambiguous = ambiguous || candidates[end]->id != candidate->id ||
                      candidates[end]->animal != candidate->animal ||
                      candidates[end]->actor != candidate->actor ||
                      candidates[end]->place_step != candidate->place_step ||
                      !same_action(candidates[end]->place_action,
                                   candidate->place_action);
          ++end;
        }
        const auto *target =
            tile_at(c.phase_start, c.player, candidate->target);
        if (ambiguous || !target ||
            target->kind != definition.required_structure ||
            target->animal != Item::NONE ||
            x.owners.contains(candidate->target)) {
          index = end;
          continue;
        }
        auto opened =
            x.speculative.open({{c.phase_start.day(), candidate->target},
                                persistent::Kind::Animal,
                                m.item,
                                1,
                                definition.purchase_cost,
                                c.phase_start.config().episode_steps /
                                        c.phase_start.config().turns_per_day -
                                    1,
                                true,
                                500,
                                "transactional-typed-animal"});
        if (opened.status == persistent::OpenStatus::Opened) {
          x.owners[candidate->target] = opened.objective_id;
          x.typed_sources[candidate->target] = candidate->id;
          left--;
        }
        index = end;
      }
    }
  // A failed BUY can only be observed on the following tick. Accept that
  // narrow cross-tick path from the receipt-bound issuer; absent, stale,
  // successful, or forged provenance remains inert.
  const auto &last_fills = c.phase_start.last_market_fills()[c.player];
  const auto cross_tick = [&](const TypedAnimalObligation &value) {
    return value.id != 0 && value.player == c.player && value.actor >= 0 &&
           value.actor < static_cast<int>(c.raw_g001.units.size()) &&
           value.place_step >= c.step && value.place_action.op == Op::PLACE &&
           value.place_action.item == value.animal &&
           value.place_action.quantity == 1 &&
           animal_lifecycle::definition(value.animal) &&
           value.acquisition_step == c.step - 1 &&
           value.acquisition_market_slot >= 0 &&
           value.acquisition_market_slot < static_cast<int>(last_fills.size()) &&
           value.acquisition_action.op == Op::BUY_ANIMAL &&
           value.acquisition_action.item == value.animal &&
           value.acquisition_action.quantity == 1 &&
           value.acquisition_fill == 0 &&
           last_fills[value.acquisition_market_slot] == value.acquisition_fill &&
           value.acquisition_seed == c.phase_start.seed();
  };
  for (const auto &candidate : obligations) {
    if (!cross_tick(candidate) || x.owners.contains(candidate.target))
      continue;
    const auto provenance_owners = std::count_if(
        obligations.begin(), obligations.end(), [&](const auto &other) {
          return cross_tick(other) &&
                 other.acquisition_step == candidate.acquisition_step &&
                 other.acquisition_market_slot ==
                     candidate.acquisition_market_slot &&
                 same_action(other.acquisition_action,
                             candidate.acquisition_action) &&
                 other.acquisition_fill == candidate.acquisition_fill &&
                 other.acquisition_seed == candidate.acquisition_seed;
        });
    const auto definition = *animal_lifecycle::definition(candidate.animal);
    const auto *target = tile_at(c.phase_start, c.player, candidate.target);
    if (provenance_owners != 1 || !target ||
        target->kind != definition.required_structure ||
        target->animal != Item::NONE)
      continue;
    auto opened = x.speculative.open(
        {{c.phase_start.day(), candidate.target},
         persistent::Kind::Animal,
         candidate.animal,
         1,
         definition.purchase_cost,
         c.phase_start.config().episode_steps /
                 c.phase_start.config().turns_per_day -
             1,
         true,
         500,
         "transactional-cross-tick-animal"});
    if (opened.status == persistent::OpenStatus::Opened) {
      x.owners[candidate.target] = opened.objective_id;
      x.typed_sources[candidate.target] = candidate.id;
    }
  }
  std::set<Item> buying;
  // A confirmed animal purchase is not a durable reservation: raw production
  // or another actor can consume the shed/inventory unit before this typed
  // objective reaches PLACE. Reopen only on a whole-player absence proof, and
  // only in the speculative copy. Commit adopts this copy; abort discards it.
  const auto objectives_before_allocation = x.speculative.active_objectives();
  for (int animal = static_cast<int>(Item::GOOSE);
       animal <= static_cast<int>(Item::SHEEP); ++animal) {
    const auto animal_item = static_cast<Item>(animal);
    const auto &private_state = c.phase_start.privates()[c.player];
    const int shed_count = private_state.shed[static_cast<std::size_t>(animal)];
    int inventory_count = 0;
    for (const auto &inventory : private_state.inventories)
      inventory_count += inventory[static_cast<std::size_t>(animal)];
    int board_count = 0;
    for (const auto &tile : c.phase_start.farms()[c.player].tiles)
      if (tile.kind == fastkag::TileKind::ANIMAL && tile.animal == animal_item)
        ++board_count;
    const int total_owned = shed_count + inventory_count + board_count;

    std::set<std::uint64_t> allocated;
    int remaining = total_owned;
    // A matching typed target has first claim on its own placed animal.
    for (const auto &objective : objectives_before_allocation) {
      if (objective.spec.kind != persistent::Kind::Animal ||
          objective.spec.item != animal_item || !objective.purchase_complete)
        continue;
      const auto *target =
          tile_at(c.phase_start, c.player, objective.spec.key.tile);
      if (target && target->kind == fastkag::TileKind::ANIMAL &&
          target->animal == animal_item && remaining > 0) {
        allocated.insert(objective.id);
        --remaining;
      }
    }
    // Fungible shed/inventory and wrong-target board units are assigned once,
    // by stable objective id. They may block a duplicate BUY even when the
    // current simulator ABI cannot relocate a placed wrong-target animal.
    for (const auto &objective : objectives_before_allocation) {
      if (remaining <= 0)
        break;
      if (objective.spec.kind != persistent::Kind::Animal ||
          objective.spec.item != animal_item || !objective.purchase_complete ||
          allocated.contains(objective.id))
        continue;
      allocated.insert(objective.id);
      --remaining;
    }
    for (const auto &objective : objectives_before_allocation) {
      if (objective.spec.kind != persistent::Kind::Animal ||
          objective.spec.item != animal_item || !objective.purchase_complete ||
          objective.pending_unit_receipt || allocated.contains(objective.id))
        continue;
      const auto *target =
          tile_at(c.phase_start, c.player, objective.spec.key.tile);
      const bool target_has = target &&
                              target->kind == fastkag::TileKind::ANIMAL &&
                              target->animal == animal_item;
      static_cast<void>(x.speculative.reopen_animal_acquisition(
          objective.id,
          {c.step, c.phase_start.day(), objective.spec.key.tile, animal_item,
           shed_count, inventory_count, board_count, total_owned, target_has}));
    }
  }
  for (const auto &o : x.speculative.active_objectives())
    if (o.spec.kind == persistent::Kind::Animal && !o.purchase_complete &&
        !buying.contains(o.spec.item)) {
      std::optional<purchase_recovery::Proposal> p;
      for (const auto &m : c.raw_g001.market)
        if (m.op == Op::BUY_ANIMAL && m.item == o.spec.item && m.quantity > 0) {
          p = x.speculative.describe_hard_purchase(o.id, m.quantity);
          break;
        }
      if (!p)
        p = x.speculative.propose_purchase(
            o.id, {c.step,
                   (int)std::floor(c.phase_start.farms()[c.player].money), 0});
      if (p) {
        x.buys.push_back({o.id, *p});
        x.pub.required_purchases.push_back({p->debt_id, p->order.op,
                                            p->order.item, p->order.quantity,
                                            c.step});
        buying.insert(o.spec.item);
      }
    }
  bool hour23 =
      c.phase_start.hour() == c.phase_start.config().turns_per_day - 1;
  for (const auto &o : x.speculative.active_objectives()) {
    if (x.ready || o.spec.kind != persistent::Kind::Animal ||
        !o.purchase_complete || o.pending_unit_receipt)
      continue;
    for (int a = 0; a < (int)x.pub.units.size(); a++) {
      if (hour23) {
        ++impl_->audit.hour23_blocks;
        break;
      }
      if (!sealed(c, a)) {
        ++impl_->audit.shared_blocks;
        break;
      }
      if (c.raw_g001.units[a].op != Op::PASS)
        continue;
      auto lower = c.raw_joint;
      lower[c.player].units = x.pub.units;
      for (int z = a; z < (int)lower[c.player].units.size(); z++)
        lower[c.player].units[z] = {};
      if (c.player == 0)
        for (auto &u : lower[1].units)
          u = {};
      auto prefix = c.phase_start.preview_unit_phase(lower);
      auto before = observe(prefix, c.player, o.spec.key.tile, a);
      x.speculative.reconcile(o.id, before);
      auto lease =
          x.speculative.lease_actor(o.id, before.actor.identity, c.step);
      if (!lease)
        continue;
      std::array<persistent::ReadyRequest, 1> rr{{{o.id, *lease, before}}};
      persistent::ResourceSnapshot rs;
      rs.seeds = before.seeds;
      rs.shed = before.shed;
      auto ready = x.speculative.ready_transitions(rr, rs);
      if (ready.size() != 1) {
        x.speculative.release_actor(*lease);
        continue;
      }
      x.ready = ready[0];
      x.before = before;
      x.pub.units[a] = ready[0].action;
      x.pub.sources[a] = {a, -1, ready[0].action};
      x.pub.claimed_actors.push_back(a);
      break;
    }
  }
  x.pub.token = token(x);
  auto t = x.pub.token;
  impl_->cache.insert_or_assign(t, std::move(x));
  return impl_->cache.at(t).pub;
}

bool TransactionalAnimalRepairOwner::commit(std::uint64_t token_value,
                                            const tx::CommitGrant &grant) {
  const auto found = impl_->cache.find(token_value);
  if (found == impl_->cache.end()) {
    ++impl_->audit.rejected_commits;
    return false;
  }
  const auto cached = found->second;
  impl_->cache.erase(found);
  auto reject = [&]() {
    ++impl_->audit.rejected_commits;
    return false;
  };
  if (cached.state_version != impl_->state_version ||
      grant.submitted_step != cached.step ||
      grant.final_units.size() != cached.pub.units.size() ||
      grant.actions.size() != cached.pub.claimed_actors.size() ||
      grant.purchases.size() != cached.pub.required_purchases.size() ||
      !grant.certified_move_shifts.empty())
    return reject();

  auto final_joint = cached.raw_joint;
  final_joint[static_cast<std::size_t>(cached.player)].units =
      grant.final_units;
  std::set<int> action_actors;
  for (const int actor : cached.pub.claimed_actors) {
    const auto committed =
        std::find_if(grant.actions.begin(), grant.actions.end(),
                     [&](const auto &value) { return value.actor == actor; });
    if (committed == grant.actions.end() ||
        !action_actors.insert(actor).second ||
        !same_action(committed->emitted,
                     cached.pub.units[static_cast<std::size_t>(actor)]) ||
        !same_action(grant.final_units[static_cast<std::size_t>(actor)],
                     cached.pub.units[static_cast<std::size_t>(actor)]) ||
        committed->final_authority.actor != actor ||
        committed->final_authority.manifest_generation <=
            impl_->generations[actor] ||
        committed->final_authority.prefix_manifest_hash !=
            repair_fork::unit_prefix_manifest_hash(grant.final_units, actor) ||
        committed->final_authority.post_prefix_state_fingerprint !=
            repair_fork::post_unit_prefix_state_fingerprint(
                cached.env, cached.player, final_joint, actor))
      return reject();
  }

  auto staged = cached.speculative;
  struct PurchaseSelection {
    const tx::CommittedPurchase *committed{};
    std::size_t index{};
    int holding{};
  };
  std::vector<PurchaseSelection> selections;
  std::set<std::uint64_t> external_debts;
  std::set<int> market_slots;
  int largest_slot = -1;
  for (std::size_t index = 0; index < cached.buys.size(); ++index) {
    const auto &required = cached.pub.required_purchases[index];
    const auto committed = std::find_if(
        grant.purchases.begin(), grant.purchases.end(), [&](const auto &value) {
          return value.local_debt_id == required.debt_id;
        });
    if (committed == grant.purchases.end() ||
        committed->external_debt_id == 0 ||
        !external_debts.insert(committed->external_debt_id).second ||
        committed->final_binding.debt_id != committed->external_debt_id ||
        committed->final_binding.operation != required.operation ||
        committed->final_binding.item != required.item ||
        committed->final_binding.requested != required.quantity)
      return reject();
    if (compiled(committed->final_binding.status)) {
      if (committed->final_binding.market_slot < 0 ||
          !market_slots.insert(committed->final_binding.market_slot).second)
        return reject();
      largest_slot =
          std::max(largest_slot, committed->final_binding.market_slot);
    } else if (committed->final_binding.market_slot != -1) {
      return reject();
    }
    const int item = idx(committed->final_binding.item);
    int holding = cached.env.privates()[cached.player]
                      .shed[static_cast<std::size_t>(item)];
    for (const auto &inventory :
         cached.env.privates()[cached.player].inventories)
      holding += inventory[static_cast<std::size_t>(item)];
    selections.push_back({&*committed, index, holding});
  }

  std::vector<Action> final_market(static_cast<std::size_t>(largest_slot + 1));
  for (const auto &selection : selections) {
    if (!compiled(selection.committed->final_binding.status))
      continue;
    final_market[static_cast<std::size_t>(
        selection.committed->final_binding.market_slot)] =
        cached.buys[selection.index].second.order;
  }
  std::vector<Impl::PendingBuy> pending_buys;
  for (const auto &selection : selections) {
    const auto &committed = *selection.committed;
    const bool was_staged = compiled(committed.final_binding.status);
    if (was_staged) {
      const auto status = staged.stage_purchase(
          cached.buys[selection.index].first,
          cached.buys[selection.index].second,
          {cached.step, committed.final_binding.market_slot, final_market,
           selection.holding,
           static_cast<int>(cached.env.farms()[cached.player].money), 0});
      if (status != purchase_recovery::StageStatus::Selected)
        return reject();
    }
    pending_buys.push_back(
        {cached.buys[selection.index].first,
         cached.buys[selection.index].second,
         {committed.external_debt_id, cached.step,
          committed.final_binding.operation, committed.final_binding.item,
          committed.final_binding.requested, 0,
          committed.final_binding.market_slot, committed.final_binding.status},
         cached.player,
         was_staged});
  }

  std::vector<Impl::PendingUnit> pending_units;
  if (cached.ready) {
    const int actor = cached.pub.claimed_actors.front();
    const auto committed =
        std::find_if(grant.actions.begin(), grant.actions.end(),
                     [&](const auto &value) { return value.actor == actor; });
    const auto authority = persistent::make_prefix_authority(
        cached.step, actor, committed->final_authority.manifest_generation,
        repair_fork::phase_start_fingerprint(cached.env, cached.player),
        grant.final_units, *cached.before);
    const auto status = staged.stage_transition(
        *cached.ready,
        {cached.step, actor, grant.final_units, *cached.before, authority,
         cached.env.privates()[cached.player].seeds, true});
    if (status != persistent::UnitStageStatus::Selected)
      return reject();
    pending_units.push_back(
        {cached.ready->objective_id,
         cached.ready->lease.token,
         persistent::prefix_authority_fingerprint(authority),
         cached.ready->lease.actor,
         cached.ready->key.tile,
         {cached.step, actor, committed->final_authority.manifest_generation,
          committed->final_authority.prefix_manifest_hash,
          committed->final_authority.post_prefix_state_fingerprint,
          committed->emitted}});
  }

  impl_->ledger = std::move(staged);
  impl_->owners = cached.owners;
  impl_->typed_sources = cached.typed_sources;
  impl_->pending_buys = std::move(pending_buys);
  impl_->pending_units = std::move(pending_units);
  for (const auto &action : grant.actions)
    impl_->generations[action.actor] =
        action.final_authority.manifest_generation;
  impl_->audit.objectives_opened = static_cast<int>(impl_->owners.size());
  ++impl_->audit.commits;
  ++impl_->state_version;
  return true;
}
bool TransactionalAnimalRepairOwner::validate_commit(
    std::uint64_t token_value, const tx::CommitGrant &grant) const {
  TransactionalAnimalRepairOwner snapshot;
  *snapshot.impl_ = *impl_;
  return snapshot.commit(token_value, grant);
}

void TransactionalAnimalRepairOwner::abort(std::uint64_t t) noexcept {
  if (impl_->cache.erase(t))
    ++impl_->audit.aborts;
}
const Audit &TransactionalAnimalRepairOwner::audit() const noexcept {
  return impl_->audit;
}
std::vector<persistent::ObjectiveView>
TransactionalAnimalRepairOwner::objectives() const {
  return impl_->ledger.active_objectives();
}
std::size_t TransactionalAnimalRepairOwner::expected_actions() const noexcept {
  return impl_->pending_units.size();
}
std::size_t
TransactionalAnimalRepairOwner::expected_purchases() const noexcept {
  return impl_->pending_buys.size();
}
std::size_t TransactionalAnimalRepairOwner::cached_proposals() const noexcept {
  return impl_->cache.size();
}
} // namespace g001::transactional_animal_repair
