#include "transactional_crop_repair_owner.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace g001::transactional_crop_repair {
namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;
namespace tx = repair_owner_composer;

bool same_action(const Action &left, const Action &right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool same_position(Position left, Position right) {
  return left.x == right.x && left.y == right.y;
}

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST || op == Op::WEST;
}

bool is_crop_action(Op op) {
  return op == Op::PLANT || op == Op::WATER || op == Op::HARVEST;
}

bool valid_crop(Item item) {
  const int value = static_cast<int>(item);
  return value >= 0 && value < fastkag::N_CROPS;
}

bool accepted(repair_fork::PurchaseCompileStatus status) {
  return status == repair_fork::PurchaseCompileStatus::BoundExisting ||
         status == repair_fork::PurchaseCompileStatus::Appended;
}

Position actor_position(const Simulator &env, int player, int actor) {
  const auto &farm = env.farms()[static_cast<std::size_t>(player)];
  if (actor == 0)
    return farm.farmer;
  if (actor < 0 || actor > static_cast<int>(farm.hands.size()))
    throw std::invalid_argument("transactional crop actor is absent");
  return farm.hands[static_cast<std::size_t>(actor - 1)];
}

const Tile &tile_at(const Simulator &env, int player, Position position) {
  const int size = env.config().board_size;
  if (position.x < 0 || position.y < 0 || position.x >= size ||
      position.y >= size)
    throw std::invalid_argument("transactional crop tile is outside board");
  return env.farms()[static_cast<std::size_t>(player)]
      .tiles[static_cast<std::size_t>(position.y * size + position.x)];
}

Simulator prefix_before_actor(const repair_fork::RepairContext &context,
                              int actor) {
  std::array<PlayerAction, 2> prefix;
  if (context.player == 1)
    prefix[0].units = context.raw_joint[0].units;
  prefix[static_cast<std::size_t>(context.player)].units.assign(
      context.raw_g001.units.size(), Action{});
  for (int lower = 0; lower < actor; ++lower)
    prefix[static_cast<std::size_t>(context.player)]
        .units[static_cast<std::size_t>(lower)] =
        context.raw_g001.units[static_cast<std::size_t>(lower)];
  return context.phase_start.preview_unit_phase(prefix);
}

bool exact_no_effect(const repair_fork::RepairContext &context, int actor,
                     const Simulator &prefix, const Action &raw) {
  if (raw.op == Op::PASS)
    return true;
  if (is_move(raw.op))
    return false;
  std::array<PlayerAction, 2> one;
  one[static_cast<std::size_t>(context.player)].units.assign(
      context.raw_g001.units.size(), Action{});
  one[static_cast<std::size_t>(context.player)]
      .units[static_cast<std::size_t>(actor)] = raw;
  const auto after = prefix.preview_unit_phase(one);
  return repair_fork::full_unit_phase_state_fingerprint(prefix) ==
         repair_fork::full_unit_phase_state_fingerprint(after);
}

std::optional<Action> transition(const Tile &tile, Item desired, int seeds) {
  if (tile.kind == TileKind::WEED)
    return Action{Op::DIG, Item::NONE, 1};
  if (tile.kind == TileKind::EMPTY) {
    if (seeds > 0)
      return Action{Op::PLANT, desired, 1};
    return std::nullopt;
  }
  if (tile.kind == TileKind::PLANT && tile.crop == desired &&
      !tile.watered_today)
    return Action{Op::WATER, desired, 1};
  return std::nullopt;
}

void hash_add(std::uint64_t &hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffULL;
    hash *= 1099511628211ULL;
  }
}

void hash_action(std::uint64_t &hash, const Action &action) {
  hash_add(hash, static_cast<std::uint8_t>(action.op));
  hash_add(hash, static_cast<std::uint8_t>(action.item));
  hash_add(hash, static_cast<std::uint32_t>(action.quantity));
}

bool same_action_receipt(const repair_fork::ActionReceipt &left,
                         const repair_fork::ActionReceipt &right) {
  return left.submitted_step == right.submitted_step &&
         left.actor == right.actor &&
         left.manifest_generation == right.manifest_generation &&
         left.prefix_manifest_hash == right.prefix_manifest_hash &&
         left.post_prefix_state_fingerprint ==
             right.post_prefix_state_fingerprint &&
         same_action(left.emitted, right.emitted);
}

bool same_purchase_shape(const repair_fork::PurchaseReceipt &receipt,
                         const repair_fork::PurchaseReceipt &shape) {
  return receipt.debt_id == shape.debt_id &&
         receipt.submitted_step == shape.submitted_step &&
         receipt.operation == shape.operation && receipt.item == shape.item &&
         receipt.requested == shape.requested &&
         receipt.market_slot == shape.market_slot &&
         receipt.compile_status == shape.compile_status;
}

} // namespace

struct TransactionalCropRepairOwner::Impl {
  struct Plot {
    PlotDebtView view;
    int purchase_target{1};
    int purchase_acquired{};
  };
  struct Attempt {
    std::uint64_t plot_id{};
    int player{-1};
    int actor{-1};
    Position tile{};
    Item desired{Item::NONE};
    Action emitted{};
    bool route_replay{};
    int route_source_step{-1};
  };
  struct ExpectedAction {
    repair_fork::ActionReceipt receipt;
    Attempt attempt;
  };
  struct ExpectedPurchase {
    repair_fork::PurchaseReceipt shape;
    int player{-1};
    int seeds_before{};
    std::uint64_t plot_id{};
  };
  struct DelayedMove {
    int player{-1};
    int actor{-1};
    int source_step{-1};
    int replay_step{-1};
    Action action{};
    std::uint64_t certificate_hash{};
    std::uint64_t issuer_generation{};
    bool pending_replay{};
  };
  struct Cached {
    tx::PreparedRepair public_proposal;
    Simulator phase_start;
    int player{-1};
    int step{-1};
    std::array<PlayerAction, 2> raw_joint;
    std::uint64_t state_version{};
    std::uint64_t typed_obligation_id{};
    int source_actor{-1};
    bool open_plot{};
    bool update_plot{};
    bool crop_identity_change{};
    std::uint64_t plot_id{};
    Position tile{};
    Item desired{Item::NONE};
    int origin_day{-1};
    int purchase_target{1};
    int effective_purchase_acquired{};
    bool reconcile_purchase_credit{};
    std::vector<std::pair<std::uint64_t, int>> credit_updates;
    std::optional<Attempt> attempt;
    std::optional<repair_fork::RequiredPurchase> purchase;
    bool replaces_move{};
    std::optional<DelayedMove> replay;
  };

  std::vector<Plot> plots;
  std::vector<ExpectedAction> expected_actions;
  std::vector<ExpectedPurchase> expected_purchases;
  std::deque<DelayedMove> delayed;
  std::map<int, std::uint64_t> actor_generations;
  mutable std::map<std::uint64_t, Cached> cache;
  Audit audit;
  std::uint64_t state_version{1};
  std::uint64_t next_plot_id{1};

  Plot *find_plot(Position tile) {
    const auto found =
        std::find_if(plots.begin(), plots.end(), [&](const Plot &plot) {
          return same_position(plot.view.tile, tile);
        });
    return found == plots.end() ? nullptr : &*found;
  }
  const Plot *find_plot(Position tile) const {
    const auto found =
        std::find_if(plots.begin(), plots.end(), [&](const Plot &plot) {
          return same_position(plot.view.tile, tile);
        });
    return found == plots.end() ? nullptr : &*found;
  }
};

namespace {

std::uint64_t
proposal_token(const TransactionalCropRepairOwner::Impl::Cached &cached) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, cached.state_version);
  hash_add(hash, cached.typed_obligation_id);
  hash_add(hash,
           repair_fork::full_unit_phase_state_fingerprint(cached.phase_start));
  hash_add(hash, static_cast<std::uint32_t>(cached.player));
  hash_add(hash, static_cast<std::uint32_t>(cached.step));
  hash_add(hash, cached.plot_id);
  hash_add(hash, static_cast<std::uint16_t>(cached.tile.x));
  hash_add(hash, static_cast<std::uint16_t>(cached.tile.y));
  hash_add(hash, static_cast<std::uint8_t>(cached.desired));
  if (cached.source_actor >= 0) {
    hash_add(hash, static_cast<std::uint32_t>(cached.source_actor));
    hash_action(hash,
                cached.raw_joint[static_cast<std::size_t>(cached.player)]
                    .units[static_cast<std::size_t>(cached.source_actor)]);
  }
  hash_add(hash, cached.open_plot);
  hash_add(hash, cached.update_plot);
  hash_add(hash, cached.crop_identity_change);
  hash_add(hash,
           static_cast<std::uint32_t>(cached.effective_purchase_acquired));
  hash_add(hash, cached.reconcile_purchase_credit);
  for (const auto &[plot, credit] : cached.credit_updates) {
    hash_add(hash, plot);
    hash_add(hash, static_cast<std::uint32_t>(credit));
  }
  for (const auto &action : cached.public_proposal.units)
    hash_action(hash, action);
  for (const auto &source : cached.public_proposal.sources) {
    hash_add(hash, static_cast<std::uint32_t>(source.actor));
    hash_add(hash, static_cast<std::uint32_t>(source.source_step));
    hash_action(hash, source.source_action);
  }
  for (const auto &purchase : cached.public_proposal.required_purchases) {
    hash_add(hash, purchase.debt_id);
    hash_add(hash, static_cast<std::uint8_t>(purchase.operation));
    hash_add(hash, static_cast<std::uint8_t>(purchase.item));
    hash_add(hash, static_cast<std::uint32_t>(purchase.quantity));
  }
  return hash == 0 ? 1 : hash;
}

std::uint64_t purchase_debt(std::uint64_t plot_id) {
  return (1ULL << 61U) | plot_id;
}

bool action_grant_valid(
    const TransactionalCropRepairOwner::Impl::Cached &cached,
    const tx::CommitGrant &grant,
    const std::map<int, std::uint64_t> &generations) {
  const auto &claims = cached.public_proposal.claimed_actors;
  if (grant.actions.size() != claims.size() ||
      grant.final_units.size() != cached.public_proposal.units.size())
    return false;
  auto final_joint = cached.raw_joint;
  final_joint[static_cast<std::size_t>(cached.player)].units =
      grant.final_units;
  for (int actor : claims) {
    if (actor < 0 || actor >= static_cast<int>(grant.final_units.size()) ||
        !same_action(
            grant.final_units[static_cast<std::size_t>(actor)],
            cached.public_proposal.units[static_cast<std::size_t>(actor)]))
      return false;
    const auto found = std::find_if(
        grant.actions.begin(), grant.actions.end(),
        [&](const tx::CommittedAction &value) { return value.actor == actor; });
    if (found == grant.actions.end() ||
        !same_action(
            found->emitted,
            cached.public_proposal.units[static_cast<std::size_t>(actor)]) ||
        found->final_authority.actor != actor ||
        !same_action(found->emitted,
                     grant.final_units[static_cast<std::size_t>(actor)]))
      return false;
    const auto prior = generations.find(actor);
    const std::uint64_t prior_generation =
        prior == generations.end() ? 0 : prior->second;
    if (found->final_authority.manifest_generation <= prior_generation ||
        found->final_authority.prefix_manifest_hash !=
            repair_fork::unit_prefix_manifest_hash(grant.final_units, actor) ||
        found->final_authority.post_prefix_state_fingerprint !=
            repair_fork::post_unit_prefix_state_fingerprint(
                cached.phase_start, cached.player, final_joint, actor))
      return false;
  }
  std::set<int> unique;
  for (const auto &action : grant.actions)
    if (!unique.insert(action.actor).second ||
        std::find(claims.begin(), claims.end(), action.actor) == claims.end())
      return false;
  return true;
}

bool purchase_grant_valid(
    const TransactionalCropRepairOwner::Impl::Cached &cached,
    const tx::CommitGrant &grant) {
  if (grant.purchases.size() !=
      cached.public_proposal.required_purchases.size())
    return false;
  for (const auto &required : cached.public_proposal.required_purchases) {
    const auto found =
        std::find_if(grant.purchases.begin(), grant.purchases.end(),
                     [&](const tx::CommittedPurchase &value) {
                       return value.local_debt_id == required.debt_id;
                     });
    if (found == grant.purchases.end() || found->external_debt_id == 0 ||
        found->final_binding.debt_id != found->external_debt_id ||
        found->final_binding.operation != required.operation ||
        found->final_binding.item != required.item ||
        found->final_binding.requested != required.quantity)
      return false;
  }
  return true;
}

} // namespace

TransactionalCropRepairOwner::TransactionalCropRepairOwner()
    : impl_(std::make_unique<Impl>()) {}
TransactionalCropRepairOwner::~TransactionalCropRepairOwner() = default;
TransactionalCropRepairOwner::TransactionalCropRepairOwner(
    TransactionalCropRepairOwner &&) noexcept = default;
TransactionalCropRepairOwner &TransactionalCropRepairOwner::operator=(
    TransactionalCropRepairOwner &&) noexcept = default;

std::string TransactionalCropRepairOwner::name() const {
  return "transactional_crop_repair_owner_v1";
}

tx::SettlementResult TransactionalCropRepairOwner::settle_owned(
    const repair_fork::RepairContext &context,
    std::span<const repair_fork::ActionReceipt> action_receipts,
    std::span<const repair_fork::PurchaseReceipt> purchase_receipts) {
  if (context.player < 0 || context.player > 1 ||
      context.step != context.phase_start.step_count())
    throw std::invalid_argument("transactional crop settle context invalid");
  // Protocol validation is all-or-nothing. Do it before changing any debt so a
  // sibling owner can reject the same receipt set without a partial publish.
  if (action_receipts.size() != impl_->expected_actions.size()) {
    ++impl_->audit.receipt_failures;
    return tx::SettlementResult::ProtocolInvalid;
  }
  std::vector<bool> action_preflight(action_receipts.size());
  for (const auto &expected : impl_->expected_actions) {
    std::size_t match = action_receipts.size();
    for (std::size_t index = 0; index < action_receipts.size(); ++index)
      if (!action_preflight[index] &&
          same_action_receipt(action_receipts[index], expected.receipt)) {
        match = index;
        break;
      }
    if (match == action_receipts.size() ||
        expected.receipt.submitted_step + 1 != context.step) {
      ++impl_->audit.receipt_failures;
      return tx::SettlementResult::ProtocolInvalid;
    }
    action_preflight[match] = true;
  }
  if (purchase_receipts.size() != impl_->expected_purchases.size()) {
    ++impl_->audit.receipt_failures;
    return tx::SettlementResult::ProtocolInvalid;
  }
  std::vector<bool> purchase_preflight(purchase_receipts.size());
  for (const auto &expected : impl_->expected_purchases) {
    std::size_t match = purchase_receipts.size();
    for (std::size_t index = 0; index < purchase_receipts.size(); ++index) {
      if (!purchase_preflight[index] &&
          same_purchase_shape(purchase_receipts[index], expected.shape)) {
        match = index;
        break;
      }
    }
    if (match == purchase_receipts.size() ||
        expected.shape.submitted_step + 1 != context.step) {
      ++impl_->audit.receipt_failures;
      return tx::SettlementResult::ProtocolInvalid;
    }
    const auto &receipt = purchase_receipts[match];
    if (receipt.filled < 0 || receipt.filled > receipt.requested ||
        (!accepted(receipt.compile_status) && receipt.filled != 0)) {
      ++impl_->audit.receipt_failures;
      return tx::SettlementResult::ProtocolInvalid;
    }
    purchase_preflight[match] = true;
  }
  bool physical_success = true;
  bool state_changed =
      !impl_->expected_actions.empty() || !impl_->expected_purchases.empty();
  std::vector<bool> used_actions(action_receipts.size());
  for (const auto &expected : impl_->expected_actions) {
    std::size_t match = action_receipts.size();
    for (std::size_t index = 0; index < action_receipts.size(); ++index)
      if (!used_actions[index] &&
          same_action_receipt(action_receipts[index], expected.receipt)) {
        match = index;
        break;
      }
    if (match == action_receipts.size() ||
        expected.receipt.submitted_step + 1 != context.step) {
      if (expected.attempt.route_replay) {
        const auto delayed = std::find_if(
            impl_->delayed.begin(), impl_->delayed.end(),
            [&](const Impl::DelayedMove &value) {
              return value.player == expected.attempt.player &&
                     value.actor == expected.attempt.actor &&
                     value.source_step == expected.attempt.route_source_step;
            });
        if (delayed != impl_->delayed.end()) {
          delayed->pending_replay = false;
          state_changed = true;
        }
      }
      physical_success = false;
      continue;
    }
    used_actions[match] = true;
    if (expected.attempt.route_replay) {
      const auto delayed = std::find_if(
          impl_->delayed.begin(), impl_->delayed.end(),
          [&](const Impl::DelayedMove &value) {
            return value.player == expected.attempt.player &&
                   value.actor == expected.attempt.actor &&
                   value.source_step == expected.attempt.route_source_step &&
                   value.pending_replay;
          });
      if (delayed == impl_->delayed.end()) {
        physical_success = false;
      } else {
        impl_->delayed.erase(delayed);
        ++impl_->audit.move_shifts_replayed;
        state_changed = true;
      }
      continue;
    }
    const auto &tile = tile_at(context.phase_start, expected.attempt.player,
                               expected.attempt.tile);
    const auto op = expected.attempt.emitted.op;
    const bool physical =
        (op == Op::DIG && tile.kind == TileKind::EMPTY) ||
        (op == Op::PLANT && tile.kind == TileKind::PLANT &&
         tile.crop == expected.attempt.desired) ||
        (op == Op::WATER && tile.kind == TileKind::PLANT &&
         tile.crop == expected.attempt.desired && tile.watered_today);
    physical_success = physical_success && physical;
  }
  impl_->expected_actions.clear();

  if (purchase_receipts.size() != impl_->expected_purchases.size())
    physical_success = false;
  std::vector<bool> used_purchases(purchase_receipts.size());
  for (const auto &expected : impl_->expected_purchases) {
    std::size_t match = purchase_receipts.size();
    for (std::size_t index = 0; index < purchase_receipts.size(); ++index)
      if (!used_purchases[index] &&
          same_purchase_shape(purchase_receipts[index], expected.shape)) {
        match = index;
        break;
      }
    if (match == purchase_receipts.size() ||
        expected.shape.submitted_step + 1 != context.step) {
      physical_success = false;
      continue;
    }
    used_purchases[match] = true;
    const auto &receipt = purchase_receipts[match];
    if (accepted(receipt.compile_status)) {
      const int crop = static_cast<int>(receipt.item);
      const int after =
          context.phase_start.privates()[expected.player].seeds[crop];
      if (receipt.filled < 0 || after < expected.seeds_before + receipt.filled)
        physical_success = false;
      if (receipt.filled <= 0)
        ++impl_->audit.purchase_zero_fills;
      else if (receipt.filled < receipt.requested)
        ++impl_->audit.purchase_partial_fills;
      else
        ++impl_->audit.purchase_full_fills;
      if (receipt.filled > 0) {
        const auto plot =
            std::find_if(impl_->plots.begin(), impl_->plots.end(),
                         [&](const Impl::Plot &value) {
                           return value.view.id == expected.plot_id;
                         });
        if (plot == impl_->plots.end()) {
          physical_success = false;
        } else {
          plot->purchase_acquired = std::min(
              plot->purchase_target, plot->purchase_acquired + receipt.filled);
          state_changed = true;
        }
      }
    } else if (receipt.filled != 0) {
      physical_success = false;
    }
  }
  impl_->expected_purchases.clear();

  for (auto it = impl_->plots.begin(); it != impl_->plots.end();) {
    const auto &tile =
        tile_at(context.phase_start, context.player, it->view.tile);
    if (tile.kind == TileKind::PLANT && tile.crop == it->view.desired &&
        tile.watered_today) {
      it = impl_->plots.erase(it);
      ++impl_->audit.plots_closed;
      state_changed = true;
    } else {
      ++it;
    }
  }
  for (const auto &delayed : impl_->delayed)
    if (context.step / context.phase_start.config().turns_per_day >
        delayed.source_step / context.phase_start.config().turns_per_day) {
      ++impl_->audit.midnight_failures;
      physical_success = false;
    }
  if (!physical_success)
    ++impl_->audit.receipt_failures;
  if (state_changed)
    ++impl_->state_version;
  return physical_success ? tx::SettlementResult::AppliedSuccess
                          : tx::SettlementResult::AppliedPhysicalFailure;
}

tx::SettlementResult TransactionalCropRepairOwner::validate_settle(
    const repair_fork::RepairContext &context,
    std::span<const repair_fork::ActionReceipt> action_receipts,
    std::span<const repair_fork::PurchaseReceipt> purchase_receipts) const {
  TransactionalCropRepairOwner snapshot;
  *snapshot.impl_ = *impl_;
  return snapshot.settle_owned(context, action_receipts, purchase_receipts);
}

tx::PreparedRepair TransactionalCropRepairOwner::prepare(
    const repair_fork::RepairContext &context) const {
  return prepare(context, std::span<const TypedCropObligation>{});
}

tx::PreparedRepair TransactionalCropRepairOwner::prepare(
    const repair_fork::RepairContext &context,
    tx::TypedRepairInput input) const {
  return prepare(context, input.crops);
}

tx::PreparedRepair TransactionalCropRepairOwner::prepare(
    const repair_fork::RepairContext &context,
    std::span<const TypedCropObligation> obligations) const {
  if (context.player < 0 || context.player > 1 || context.step < 0 ||
      context.step != context.phase_start.step_count())
    throw std::invalid_argument("transactional crop prepare context invalid");
  if (!impl_->expected_actions.empty() || !impl_->expected_purchases.empty())
    throw std::logic_error(
        "transactional crop prepare requires owned receipt settlement first");
  ++impl_->audit.prepares;
  Impl::Cached cached;
  cached.phase_start = context.phase_start;
  cached.player = context.player;
  cached.step = context.step;
  cached.raw_joint = context.raw_joint;
  cached.state_version = impl_->state_version;
  cached.origin_day = context.phase_start.day();
  cached.public_proposal.units = context.raw_g001.units;
  for (std::size_t actor = 0; actor < context.raw_g001.units.size(); ++actor)
    cached.public_proposal.sources.push_back(
        {static_cast<int>(actor), context.step, context.raw_g001.units[actor]});

  // Purchase credit is an item-scoped reservation, not a per-plot view of the
  // same global seed count. Allocate each physical seed at most once in stable
  // plot-id order; commit applies the complete reconciliation atomically.
  for (int crop = 0; crop < fastkag::N_CROPS; ++crop) {
    int available =
        std::max(0, context.phase_start.privates()[context.player].seeds[crop]);
    std::vector<const Impl::Plot *> candidates;
    for (const auto &plot : impl_->plots) {
      if (static_cast<int>(plot.view.desired) != crop ||
          plot.purchase_acquired <= 0)
        continue;
      const auto &physical =
          tile_at(context.phase_start, context.player, plot.view.tile);
      if (physical.kind == TileKind::EMPTY || physical.kind == TileKind::WEED)
        candidates.push_back(&plot);
      else
        cached.credit_updates.push_back({plot.view.id, 0});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto *left, const auto *right) {
                return left->view.id < right->view.id;
              });
    for (const auto *plot : candidates) {
      const int allocated = std::min(plot->purchase_acquired, available);
      available -= allocated;
      if (allocated != plot->purchase_acquired)
        cached.credit_updates.push_back({plot->view.id, allocated});
    }
  }

  const auto replay = std::find_if(impl_->delayed.begin(), impl_->delayed.end(),
                                   [&](const Impl::DelayedMove &value) {
                                     return value.player == context.player &&
                                            value.replay_step == context.step &&
                                            !value.pending_replay;
                                   });
  if (replay != impl_->delayed.end()) {
    const int actor = replay->actor;
    if (actor >= 0 &&
        actor < static_cast<int>(cached.public_proposal.units.size())) {
      const auto prefix = prefix_before_actor(context, actor);
      const auto &raw = context.raw_g001.units[static_cast<std::size_t>(actor)];
      if ((raw.op == Op::PASS ||
           exact_no_effect(context, actor, prefix, raw)) &&
          !same_action(raw, replay->action)) {
        cached.public_proposal.units[static_cast<std::size_t>(actor)] =
            replay->action;
        cached.public_proposal.sources[static_cast<std::size_t>(actor)] = {
            actor, replay->source_step, replay->action};
        cached.public_proposal.claimed_actors.push_back(actor);
        cached.replay = *replay;
        cached.attempt = Impl::Attempt{
            0,          context.player,
            actor,      actor_position(prefix, context.player, actor),
            Item::NONE, replay->action,
            true,       replay->source_step};
      }
    }
  } else {
    struct Candidate {
      int actor{-1};
      Simulator prefix;
      Position position{};
      const Impl::Plot *plot{};
      bool trigger{};
      Item desired{Item::NONE};
      std::uint64_t obligation_id{};
    };
    std::optional<Candidate> selected;
    for (int actor = static_cast<int>(context.raw_g001.units.size()) - 1;
         actor >= 0; --actor) {
      auto prefix = prefix_before_actor(context, actor);
      const auto position = actor_position(prefix, context.player, actor);
      const auto &tile = tile_at(prefix, context.player, position);
      const auto &raw = context.raw_g001.units[static_cast<std::size_t>(actor)];
      Item trigger_desired = Item::NONE;
      std::uint64_t obligation_id = 0;
      if (raw.op == Op::PLANT && valid_crop(raw.item)) {
        trigger_desired = raw.item;
      } else {
        const TypedCropObligation *typed = nullptr;
        bool ambiguous = false;
        for (const auto &value : obligations) {
          const bool exact_source =
              value.id != 0 && value.player == context.player &&
              value.actor == actor && value.source_step == context.step &&
              same_position(value.tile, position) &&
              same_action(value.source_action, raw) &&
              is_crop_action(value.source_action.op) &&
              valid_crop(value.desired);
          if (!exact_source)
            continue;
          if (!typed) {
            typed = &value;
          } else if (typed->id != value.id || typed->desired != value.desired) {
            ambiguous = true;
          }
        }
        if (typed && !ambiguous) {
          trigger_desired = typed->desired;
          obligation_id = typed->id;
        }
      }
      const bool trigger =
          valid_crop(trigger_desired) && is_crop_action(raw.op) &&
          exact_no_effect(context, actor, prefix, raw) &&
          (tile.kind == TileKind::WEED ||
           (raw.op == Op::PLANT && tile.kind == TileKind::EMPTY));
      const auto *plot = impl_->find_plot(position);
      if (!plot && !trigger)
        continue;
      bool higher_safe = true;
      for (int higher = actor + 1;
           higher < static_cast<int>(context.raw_g001.units.size()); ++higher) {
        const auto op =
            context.raw_g001.units[static_cast<std::size_t>(higher)].op;
        higher_safe = higher_safe && (op == Op::PASS || is_move(op));
      }
      if (!higher_safe)
        continue;
      selected = Candidate{
          actor,        std::move(prefix),
          position,     plot,
          trigger,      trigger ? trigger_desired : plot->view.desired,
          obligation_id};
      break;
    }
    if (selected) {
      cached.typed_obligation_id = selected->obligation_id;
      cached.source_actor = selected->actor;
      cached.tile = selected->position;
      cached.desired = selected->desired;
      cached.plot_id =
          selected->plot ? selected->plot->view.id : impl_->next_plot_id;
      cached.open_plot = selected->plot == nullptr;
      cached.update_plot =
          selected->plot && selected->trigger && selected->obligation_id != 0 &&
          (selected->plot->view.desired != selected->desired ||
           selected->plot->view.obligation_id != selected->obligation_id);
      cached.crop_identity_change =
          selected->plot && selected->plot->view.desired != selected->desired;
      cached.purchase_target = selected->plot && !cached.crop_identity_change
                                   ? selected->plot->purchase_target
                                   : 1;
      const auto &tile =
          tile_at(selected->prefix, context.player, selected->position);
      const int crop = static_cast<int>(selected->desired);
      const int seeds = selected->prefix.privates()[context.player].seeds[crop];
      const int persisted_acquired =
          selected->plot && !cached.crop_identity_change
              ? selected->plot->purchase_acquired
              : 0;
      cached.effective_purchase_acquired = persisted_acquired;
      if (const auto update = std::find_if(
              cached.credit_updates.begin(), cached.credit_updates.end(),
              [&](const auto &entry) { return entry.first == cached.plot_id; });
          update != cached.credit_updates.end())
        cached.effective_purchase_acquired = update->second;
      cached.reconcile_purchase_credit =
          selected->plot && cached.effective_purchase_acquired !=
                                selected->plot->purchase_acquired;
      const auto next = transition(tile, selected->desired, seeds);
      const auto &raw =
          context.raw_g001.units[static_cast<std::size_t>(selected->actor)];
      const bool sink =
          raw.op == Op::PASS ||
          exact_no_effect(context, selected->actor, selected->prefix, raw);
      if (next && (sink || is_move(raw.op)) && !same_action(*next, raw)) {
        cached.public_proposal
            .units[static_cast<std::size_t>(selected->actor)] = *next;
        cached.public_proposal
            .sources[static_cast<std::size_t>(selected->actor)] = {
            selected->actor, -1, *next};
        cached.public_proposal.claimed_actors.push_back(selected->actor);
        cached.attempt = Impl::Attempt{cached.plot_id,
                                       context.player,
                                       selected->actor,
                                       selected->position,
                                       selected->desired,
                                       *next,
                                       false,
                                       -1};
        cached.replaces_move = is_move(raw.op);
      }
      if (tile.kind == TileKind::EMPTY || tile.kind == TileKind::WEED) {
        int hard_quantity = 0;
        for (const auto &market : context.raw_g001.market)
          if (market.op == Op::BUY_SEED && market.item == selected->desired &&
              market.quantity > 0) {
            hard_quantity = market.quantity;
            break;
          }
        if ((cached.open_plot || cached.crop_identity_change) &&
            hard_quantity > 0)
          cached.purchase_target = hard_quantity;
        const int acquired = cached.effective_purchase_acquired;
        const int quantity = std::max(0, cached.purchase_target - acquired);
        if (quantity > 0 && (seeds <= 0 || acquired < cached.purchase_target)) {
          const repair_fork::RequiredPurchase purchase{
              purchase_debt(cached.plot_id), Op::BUY_SEED, selected->desired,
              quantity,
              context.phase_start.day() *
                      context.phase_start.config().turns_per_day +
                  context.phase_start.config().turns_per_day - 1};
          cached.purchase = purchase;
          cached.public_proposal.required_purchases.push_back(purchase);
        }
      }
    }
  }

  cached.public_proposal.token = proposal_token(cached);
  const auto token = cached.public_proposal.token;
  impl_->cache.insert_or_assign(token, cached);
  return impl_->cache.at(token).public_proposal;
}

bool TransactionalCropRepairOwner::commit(std::uint64_t token,
                                          const tx::CommitGrant &grant) {
  const auto found = impl_->cache.find(token);
  if (found == impl_->cache.end()) {
    ++impl_->audit.rejected_commits;
    return false;
  }
  const Impl::Cached cached = found->second;
  impl_->cache.erase(found);
  auto reject = [&]() {
    ++impl_->audit.rejected_commits;
    return false;
  };
  if (cached.state_version != impl_->state_version ||
      grant.submitted_step != cached.step ||
      !action_grant_valid(cached, grant, impl_->actor_generations) ||
      !purchase_grant_valid(cached, grant))
    return reject();

  std::optional<tx::CertifiedMoveShift> shift;
  if (cached.replaces_move) {
    if (grant.certified_move_shifts.size() != 1) {
      ++impl_->audit.uncertified_move_rejections;
      return reject();
    }
    shift = grant.certified_move_shifts.front();
    const int actor = cached.public_proposal.claimed_actors.front();
    const auto &raw = cached.raw_joint[static_cast<std::size_t>(cached.player)]
                          .units[static_cast<std::size_t>(actor)];
    const int turns = cached.phase_start.config().turns_per_day;
    if (shift->actor != actor || shift->source_step != cached.step ||
        shift->emitted_step <= cached.step ||
        shift->source_step / turns != shift->emitted_step / turns ||
        shift->emitted_step >= cached.phase_start.config().episode_steps ||
        shift->day_suffix_certificate_hash == 0 ||
        shift->issuer_generation == 0 ||
        !same_action(shift->source_action, raw) || !is_move(raw.op))
      return reject();
  } else if (!grant.certified_move_shifts.empty()) {
    return reject();
  }

  if (cached.replay) {
    const auto delayed = std::find_if(
        impl_->delayed.begin(), impl_->delayed.end(),
        [&](const Impl::DelayedMove &value) {
          return value.player == cached.replay->player &&
                 value.actor == cached.replay->actor &&
                 value.source_step == cached.replay->source_step &&
                 value.replay_step == cached.replay->replay_step &&
                 same_action(value.action, cached.replay->action) &&
                 !value.pending_replay;
        });
    if (delayed == impl_->delayed.end())
      return reject();
  }

  if (cached.open_plot) {
    impl_->plots.push_back(
        {{cached.plot_id, cached.tile, cached.desired, cached.origin_day,
          purchase_debt(cached.plot_id), cached.typed_obligation_id,
          cached.player, cached.source_actor, cached.step,
          cached.raw_joint[static_cast<std::size_t>(cached.player)]
              .units[static_cast<std::size_t>(cached.source_actor)]},
         cached.purchase_target,
         0});
    ++impl_->next_plot_id;
    ++impl_->audit.plots_opened;
  } else if (cached.update_plot) {
    auto *plot = impl_->find_plot(cached.tile);
    if (!plot || plot->view.id != cached.plot_id)
      return reject();
    plot->view.desired = cached.desired;
    plot->view.origin_day = cached.origin_day;
    plot->view.purchase_debt_id = purchase_debt(cached.plot_id);
    plot->view.obligation_id = cached.typed_obligation_id;
    plot->view.source_player = cached.player;
    plot->view.source_actor = cached.source_actor;
    plot->view.source_step = cached.step;
    plot->view.source_action =
        cached.raw_joint[static_cast<std::size_t>(cached.player)]
            .units[static_cast<std::size_t>(cached.source_actor)];
    if (cached.crop_identity_change) {
      plot->purchase_target = cached.purchase_target;
      plot->purchase_acquired = 0;
    }
  } else if (cached.reconcile_purchase_credit) {
    auto *plot = impl_->find_plot(cached.tile);
    if (!plot || plot->view.id != cached.plot_id)
      return reject();
    plot->purchase_acquired = cached.effective_purchase_acquired;
  }
  for (const auto &[plot_id, credit] : cached.credit_updates) {
    if (cached.crop_identity_change && plot_id == cached.plot_id)
      continue;
    const auto found = std::find_if(
        impl_->plots.begin(), impl_->plots.end(),
        [&](const Impl::Plot &plot) { return plot.view.id == plot_id; });
    if (found == impl_->plots.end())
      return reject();
    found->purchase_acquired = credit;
  }

  for (const auto &committed : grant.actions) {
    impl_->actor_generations[committed.actor] =
        committed.final_authority.manifest_generation;
    const repair_fork::ActionReceipt receipt{
        cached.step,
        committed.actor,
        committed.final_authority.manifest_generation,
        committed.final_authority.prefix_manifest_hash,
        committed.final_authority.post_prefix_state_fingerprint,
        committed.emitted};
    auto attempt = cached.attempt.value_or(Impl::Attempt{});
    impl_->expected_actions.push_back({receipt, attempt});
  }
  for (const auto &committed : grant.purchases) {
    const auto required =
        std::find_if(cached.public_proposal.required_purchases.begin(),
                     cached.public_proposal.required_purchases.end(),
                     [&](const repair_fork::RequiredPurchase &value) {
                       return value.debt_id == committed.local_debt_id;
                     });
    const int crop = static_cast<int>(required->item);
    impl_->expected_purchases.push_back(
        {{committed.external_debt_id, cached.step,
          committed.final_binding.operation, committed.final_binding.item,
          committed.final_binding.requested, 0,
          committed.final_binding.market_slot, committed.final_binding.status},
         cached.player,
         cached.phase_start.privates()[cached.player].seeds[crop],
         cached.plot_id});
  }
  if (shift) {
    impl_->delayed.push_back({cached.player, shift->actor, shift->source_step,
                              shift->emitted_step, shift->source_action,
                              shift->day_suffix_certificate_hash,
                              shift->issuer_generation, false});
    ++impl_->audit.move_shifts_committed;
  }
  if (cached.replay) {
    const auto delayed =
        std::find_if(impl_->delayed.begin(), impl_->delayed.end(),
                     [&](const Impl::DelayedMove &value) {
                       return value.player == cached.replay->player &&
                              value.actor == cached.replay->actor &&
                              value.source_step == cached.replay->source_step &&
                              !value.pending_replay;
                     });
    if (delayed == impl_->delayed.end())
      throw std::logic_error("validated delayed MOVE disappeared at commit");
    delayed->pending_replay = true;
  }
  ++impl_->state_version;
  ++impl_->audit.commits;
  return true;
}

bool TransactionalCropRepairOwner::validate_commit(
    std::uint64_t token, const tx::CommitGrant &grant) const {
  TransactionalCropRepairOwner snapshot;
  *snapshot.impl_ = *impl_;
  return snapshot.commit(token, grant);
}

void TransactionalCropRepairOwner::abort(std::uint64_t token) noexcept {
  if (impl_->cache.erase(token) > 0)
    ++impl_->audit.aborts;
}

const Audit &TransactionalCropRepairOwner::audit() const noexcept {
  return impl_->audit;
}

std::vector<PlotDebtView> TransactionalCropRepairOwner::plot_debts() const {
  std::vector<PlotDebtView> out;
  out.reserve(impl_->plots.size());
  for (const auto &plot : impl_->plots)
    out.push_back(plot.view);
  return out;
}

std::uint64_t
TransactionalCropRepairOwner::persistent_fingerprint() const noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, impl_->state_version);
  hash_add(hash, impl_->next_plot_id);
  for (const auto &plot : impl_->plots) {
    hash_add(hash, plot.view.id);
    hash_add(hash, static_cast<std::uint16_t>(plot.view.tile.x));
    hash_add(hash, static_cast<std::uint16_t>(plot.view.tile.y));
    hash_add(hash, static_cast<std::uint8_t>(plot.view.desired));
    hash_add(hash, static_cast<std::uint32_t>(plot.view.origin_day));
    hash_add(hash, plot.view.purchase_debt_id);
    hash_add(hash, plot.view.obligation_id);
    hash_add(hash, static_cast<std::uint32_t>(plot.view.source_player));
    hash_add(hash, static_cast<std::uint32_t>(plot.view.source_actor));
    hash_add(hash, static_cast<std::uint32_t>(plot.view.source_step));
    hash_action(hash, plot.view.source_action);
    hash_add(hash, static_cast<std::uint32_t>(plot.purchase_target));
    hash_add(hash, static_cast<std::uint32_t>(plot.purchase_acquired));
  }
  for (const auto &expected : impl_->expected_actions) {
    hash_add(hash, expected.receipt.manifest_generation);
    hash_add(hash, expected.receipt.prefix_manifest_hash);
    hash_add(hash, expected.receipt.post_prefix_state_fingerprint);
  }
  for (const auto &expected : impl_->expected_purchases) {
    hash_add(hash, expected.shape.debt_id);
    hash_add(hash, static_cast<std::uint32_t>(expected.shape.market_slot));
  }
  for (const auto &move : impl_->delayed) {
    hash_add(hash, static_cast<std::uint32_t>(move.actor));
    hash_add(hash, static_cast<std::uint32_t>(move.source_step));
    hash_add(hash, static_cast<std::uint32_t>(move.replay_step));
    hash_action(hash, move.action);
    hash_add(hash, move.certificate_hash);
    hash_add(hash, move.issuer_generation);
    hash_add(hash, move.pending_replay);
  }
  for (const auto &[actor, generation] : impl_->actor_generations) {
    hash_add(hash, static_cast<std::uint32_t>(actor));
    hash_add(hash, generation);
  }
  return hash;
}

std::size_t TransactionalCropRepairOwner::cached_proposals() const noexcept {
  return impl_->cache.size();
}
std::size_t
TransactionalCropRepairOwner::expected_action_receipts() const noexcept {
  return impl_->expected_actions.size();
}
std::size_t
TransactionalCropRepairOwner::expected_purchase_receipts() const noexcept {
  return impl_->expected_purchases.size();
}
std::size_t TransactionalCropRepairOwner::delayed_moves() const noexcept {
  return impl_->delayed.size();
}

} // namespace g001::transactional_crop_repair
