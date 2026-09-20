#include "online_elastic_repair_owner.hpp"

#include "purchase_recovery_ledger.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace g001::online_elastic {
namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;
using repair_fork::PurchaseBinding;
using repair_fork::PurchaseCompileStatus;
using repair_fork::PurchaseReceipt;
using purchase_recovery::FillStatus;

constexpr std::array<int, fastkag::N_CROPS> kSeedCost{10, 20, 50, 100, 80};

bool action_equal(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
      op == Op::WEST;
}

bool is_crop_action(Op op) {
  return op == Op::PLANT || op == Op::WATER || op == Op::HARVEST;
}

bool valid_crop(Item item) {
  const int value = static_cast<int>(item);
  return value >= 0 && value < fastkag::N_CROPS;
}

std::uint64_t actor_generation(int day, int actor) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(day + 1))
          << 32U) |
      static_cast<std::uint32_t>(actor + 1);
}

Position actor_position(const Simulator& env, int player, int actor) {
  const auto& farm = env.farms()[static_cast<std::size_t>(player)];
  if (actor == 0) return farm.farmer;
  if (actor < 0 || actor > static_cast<int>(farm.hands.size()))
    throw std::invalid_argument("actor is absent from phase start");
  return farm.hands[static_cast<std::size_t>(actor - 1)];
}

const Tile& tile_at(const Simulator& env, int player, Position position) {
  const int size = env.config().board_size;
  if (position.x < 0 || position.y < 0 || position.x >= size ||
      position.y >= size)
    throw std::invalid_argument("actor position is outside board");
  return env.farms()[static_cast<std::size_t>(player)]
      .tiles[static_cast<std::size_t>(position.y * size + position.x)];
}

bool same_position(Position left, Position right) {
  return left.x == right.x && left.y == right.y;
}

Simulator prefix_before_actor(const repair_fork::RepairContext& context,
                              int actor) {
  std::array<PlayerAction, 2> actions;
  // Simulator unit order is all player-0 slots followed by all player-1
  // slots. A player-1 authority therefore includes the complete player-0
  // manifest, not merely lower slots from its own farm.
  if (context.player == 1)
    actions[0].units = context.raw_joint[0].units;
  actions[static_cast<std::size_t>(context.player)].units.assign(
      context.raw_g001.units.size(), Action{});
  for (int lower = 0; lower < actor; ++lower)
    actions[static_cast<std::size_t>(context.player)]
        .units[static_cast<std::size_t>(lower)] =
        context.raw_g001.units[static_cast<std::size_t>(lower)];
  return context.phase_start.preview_unit_phase(actions);
}

bool exact_no_effect(const repair_fork::RepairContext& context, int actor,
                     const Simulator& prefix, const Action& raw) {
  if (raw.op == Op::PASS) return true;
  if (is_move(raw.op)) return false;
  std::array<PlayerAction, 2> actions;
  actions[static_cast<std::size_t>(context.player)].units.assign(
      context.raw_g001.units.size(), Action{});
  actions[static_cast<std::size_t>(context.player)]
      .units[static_cast<std::size_t>(actor)] = raw;
  const auto after = prefix.preview_unit_phase(actions);
  return repair_fork::phase_start_fingerprint(prefix, context.player) ==
      repair_fork::phase_start_fingerprint(after, context.player);
}

void hash_add(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffULL;
    hash *= 1099511628211ULL;
  }
}

bool accepted(PurchaseCompileStatus status) {
  return status == PurchaseCompileStatus::BoundExisting ||
      status == PurchaseCompileStatus::Appended;
}

}  // namespace

std::uint64_t frozen_suffix_hash(
    const FrozenDaySuffixCertificate& certificate) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, static_cast<std::uint32_t>(certificate.player));
  hash_add(hash, static_cast<std::uint32_t>(certificate.day));
  hash_add(hash, static_cast<std::uint32_t>(certificate.actor));
  hash_add(hash, certificate.actor_generation);
  hash_add(hash, static_cast<std::uint32_t>(certificate.suffix_start_step));
  hash_add(hash, certificate.issuer_generation);
  hash_add(hash, certificate.sinks_irrevocable);
  hash_add(hash, certificate.slots.size());
  for (const auto& slot : certificate.slots) {
    hash_add(hash, static_cast<std::uint32_t>(slot.source_step));
    hash_add(hash, static_cast<std::uint8_t>(slot.source_action.op));
    hash_add(hash, static_cast<std::uint8_t>(slot.source_action.item));
    hash_add(hash, static_cast<std::uint32_t>(slot.source_action.quantity));
    hash_add(hash, static_cast<std::uint8_t>(slot.kind));
  }
  return hash;
}

struct OnlineElasticRepairOwner::Impl {
  struct PlotDebt {
    PlotDebtView view;
  };
  struct UnitAttempt {
    std::uint64_t debt_id{};
    int submitted_step{-1};
    Op operation{Op::PASS};
    Position tile{};
    Item desired{Item::NONE};
  };
  struct ExpectedActionReceipt {
    repair_fork::ActionReceipt receipt;
    std::optional<UnitAttempt> semantic_attempt;
  };
  struct ExpectedPurchase {
    PurchaseBinding binding;
    int submitted_step{-1};
  };
  struct MoveToken {
    Action action{};
    int source_step{-1};
  };
  struct ActorState {
    int day{-1};
    std::uint64_t generation{};
    std::deque<MoveToken> delayed;
    int emitted_move_sources{};
    std::uint64_t manifest_generation{};
  };

  std::vector<PlotDebt> plots;
  purchase_recovery::Ledger purchases;
  std::vector<ExpectedActionReceipt> expected_action_receipts;
  std::vector<ExpectedPurchase> expected_purchases;
  std::map<std::pair<int, int>, FrozenDaySuffixCertificate> certificates;
  std::map<int, ActorState> actors;
  OnlineAuditView audit;
  std::uint64_t next_plot_id{1};
  int last_step{-1};
  bool poisoned{};

  PlotDebt* find_plot(Position tile) {
    const auto found = std::find_if(plots.begin(), plots.end(),
                                    [&](const PlotDebt& value) {
      return same_position(value.view.tile, tile);
    });
    return found == plots.end() ? nullptr : &*found;
  }

  PlotDebt* find_plot(std::uint64_t id) {
    const auto found = std::find_if(plots.begin(), plots.end(),
                                    [&](const PlotDebt& value) {
      return value.view.id == id;
    });
    return found == plots.end() ? nullptr : &*found;
  }
};

namespace {

repair_fork::RepairDecision baseline_decision(
    const repair_fork::RepairContext& context) {
  repair_fork::RepairDecision out;
  out.units = context.raw_g001.units;
  out.sources.reserve(out.units.size());
  for (std::size_t actor = 0; actor < out.units.size(); ++actor)
    out.sources.push_back({static_cast<int>(actor), context.step,
                           out.units[actor]});
  out.receipt_acks.assign(context.previous_action_receipts.begin(),
                          context.previous_action_receipts.end());
  out.purchase_receipt_acks.assign(
      context.previous_purchase_receipts.begin(),
      context.previous_purchase_receipts.end());
  return out;
}

bool valid_certificate_shape(const FrozenDaySuffixCertificate& certificate) {
  if (!certificate.sinks_irrevocable || certificate.player < 0 ||
      certificate.player >= 2 || certificate.day < 0 ||
      certificate.actor < 0 || certificate.actor_generation == 0 ||
      certificate.issuer_generation == 0 || certificate.slots.empty() ||
      certificate.suffix_start_step != certificate.day * 24 ||
      certificate.content_hash != frozen_suffix_hash(certificate))
    return false;
  const int end = certificate.day * 24 + 23;
  if (certificate.slots.back().source_step != end) return false;
  for (std::size_t index = 0; index < certificate.slots.size(); ++index) {
    const auto& slot = certificate.slots[index];
    if (slot.source_step != certificate.suffix_start_step +
            static_cast<int>(index) ||
        slot.source_step / 24 != certificate.day)
      return false;
    if ((slot.kind == FrozenSlotKind::MoveToken) !=
        is_move(slot.source_action.op))
      return false;
    if (slot.kind == FrozenSlotKind::CertifiedSink &&
        is_move(slot.source_action.op))
      return false;
  }
  return true;
}

const FrozenSlot* current_certified_slot(
    const OnlineElasticRepairOwner::Impl& state,
    const repair_fork::RepairContext& context, int actor,
    const Action& raw, bool& invalid) {
  invalid = false;
  const auto found = state.certificates.find({context.phase_start.day(), actor});
  if (found == state.certificates.end()) return nullptr;
  const auto& certificate = found->second;
  if (certificate.player != context.player ||
      certificate.actor_generation !=
          actor_generation(context.phase_start.day(), actor) ||
      context.step < certificate.suffix_start_step ||
      context.step >= certificate.suffix_start_step +
          static_cast<int>(certificate.slots.size())) {
    invalid = true;
    return nullptr;
  }
  const auto& slot = certificate.slots[static_cast<std::size_t>(
      context.step - certificate.suffix_start_step)];
  if (slot.source_step != context.step ||
      !action_equal(slot.source_action, raw)) {
    invalid = true;
    return nullptr;
  }
  return &slot;
}

int future_certified_sinks(const OnlineElasticRepairOwner::Impl& state,
                           int day, int actor, int after_step, int last_step) {
  const auto found = state.certificates.find({day, actor});
  if (found == state.certificates.end()) return 0;
  return static_cast<int>(std::count_if(
      found->second.slots.begin(), found->second.slots.end(),
      [&](const FrozenSlot& slot) {
        return slot.source_step > after_step &&
            slot.source_step <= last_step &&
            slot.kind == FrozenSlotKind::CertifiedSink;
      }));
}

int remaining_transitions(const Tile& tile, Item desired) {
  if (tile.kind == TileKind::WEED) return 3;
  if (tile.kind == TileKind::EMPTY) return 2;
  if (tile.kind != TileKind::PLANT || tile.crop != desired) return 3;
  return tile.watered_today ? 0 : 1;
}

std::optional<Action> required_transition(const Tile& tile, Item desired,
                                          int seed_count) {
  if (tile.kind == TileKind::WEED) return Action{Op::DIG, Item::NONE, 1};
  if (tile.kind == TileKind::EMPTY) {
    if (seed_count <= 0) return std::nullopt;
    return Action{Op::PLANT, desired, 1};
  }
  if (tile.kind == TileKind::PLANT && tile.crop == desired &&
      !tile.watered_today)
    return Action{Op::WATER, desired, 1};
  return std::nullopt;
}

bool receipt_equal(const PurchaseReceipt& receipt,
                   const OnlineElasticRepairOwner::Impl::ExpectedPurchase&
                       expected) {
  const auto& binding = expected.binding;
  return receipt.debt_id == binding.debt_id &&
      receipt.submitted_step == expected.submitted_step &&
      receipt.operation == binding.operation &&
      receipt.item == binding.item && receipt.requested == binding.requested &&
      receipt.market_slot == binding.market_slot &&
      receipt.compile_status == binding.status;
}

bool settle_purchase_receipts(
    OnlineElasticRepairOwner::Impl& state,
    const repair_fork::RepairContext& context,
    repair_fork::RepairTelemetry& telemetry) {
  if (context.previous_purchase_receipts.size() !=
      state.expected_purchases.size()) {
    if (!context.previous_purchase_receipts.empty() ||
        !state.expected_purchases.empty()) {
      ++state.audit.purchase_receipt_failures;
      ++state.audit.fail_closed;
      ++telemetry.receipts_failed;
      return false;
    }
    return true;
  }
  bool has_accepted = false;
  for (std::size_t index = 0; index < state.expected_purchases.size(); ++index) {
    const auto& expected = state.expected_purchases[index];
    const auto& receipt = context.previous_purchase_receipts[index];
    if (!receipt_equal(receipt, expected) ||
        receipt.submitted_step + 1 != context.step) {
      ++state.audit.purchase_receipt_failures;
      ++state.audit.fail_closed;
      ++telemetry.receipts_failed;
      return false;
    }
    if (!accepted(receipt.compile_status)) continue;
    has_accepted = true;
    if (receipt.filled <= 0) ++state.audit.purchase_zero_fills;
    else if (receipt.filled < receipt.requested)
      ++state.audit.purchase_partial_fills;
    else
      ++state.audit.purchase_full_fills;
  }
  if (has_accepted) {
    purchase_recovery::ReceiptObservation observation;
    observation.step = context.step;
    observation.slot_fills =
        context.phase_start.last_market_fills()[context.player];
    observation.seeds_after =
        context.phase_start.privates()[context.player].seeds;
    const auto settlements = state.purchases.observe(observation);
    for (const auto& expected : state.expected_purchases) {
      if (!accepted(expected.binding.status)) continue;
      const auto found = std::find_if(
          settlements.begin(), settlements.end(), [&](const auto& value) {
            return value.debt_id == expected.binding.debt_id;
          });
      const auto receipt = std::find_if(
          context.previous_purchase_receipts.begin(),
          context.previous_purchase_receipts.end(), [&](const auto& value) {
            return value.debt_id == expected.binding.debt_id;
          });
      if (found == settlements.end() ||
          receipt == context.previous_purchase_receipts.end() ||
          found->filled != receipt->filled ||
          found->status == FillStatus::Ambiguous) {
        ++state.audit.purchase_receipt_failures;
        ++state.audit.fail_closed;
        ++telemetry.receipts_failed;
        return false;
      }
      ++telemetry.receipts_confirmed;
    }
    static_cast<void>(state.purchases.drain_handoffs());
  }
  state.expected_purchases.clear();
  return true;
}

bool unit_attempt_confirmed(const Simulator& env, int player,
                            const OnlineElasticRepairOwner::Impl::UnitAttempt&
                                attempt) {
  const auto& tile = tile_at(env, player, attempt.tile);
  if (attempt.operation == Op::DIG) return tile.kind == TileKind::EMPTY;
  if (attempt.operation == Op::PLANT)
    return tile.kind == TileKind::PLANT && tile.crop == attempt.desired;
  if (attempt.operation == Op::WATER)
    return tile.kind == TileKind::PLANT && tile.crop == attempt.desired &&
        tile.watered_today;
  return false;
}

bool action_receipt_equal(const repair_fork::ActionReceipt& left,
                          const repair_fork::ActionReceipt& right) {
  return left.submitted_step == right.submitted_step &&
      left.actor == right.actor &&
      left.manifest_generation == right.manifest_generation &&
      left.prefix_manifest_hash == right.prefix_manifest_hash &&
      left.post_prefix_state_fingerprint ==
          right.post_prefix_state_fingerprint &&
      action_equal(left.emitted, right.emitted);
}

bool settle_unit_attempts(OnlineElasticRepairOwner::Impl& state,
                          const repair_fork::RepairContext& context,
                          repair_fork::RepairTelemetry& telemetry) {
  if (context.previous_action_receipts.size() !=
      state.expected_action_receipts.size()) {
    ++state.audit.unit_receipts_failed;
    ++state.audit.fail_closed;
    ++telemetry.receipts_failed;
    return false;
  }
  for (std::size_t index = 0; index < state.expected_action_receipts.size();
       ++index) {
    const auto& expected = state.expected_action_receipts[index];
    const auto& receipt = context.previous_action_receipts[index];
    if (!action_receipt_equal(receipt, expected.receipt) ||
        receipt.submitted_step + 1 != context.step) {
      ++state.audit.unit_receipts_failed;
      ++state.audit.fail_closed;
      ++telemetry.receipts_failed;
      return false;
    }
    if (!expected.semantic_attempt.has_value()) continue;
    const auto& attempt = *expected.semantic_attempt;
    const bool adjacent = attempt.submitted_step + 1 == context.step;
    if (adjacent && unit_attempt_confirmed(context.phase_start, context.player,
                                           attempt)) {
      ++state.audit.unit_receipts_confirmed;
      ++telemetry.receipts_confirmed;
    } else {
      ++state.audit.unit_receipts_failed;
      ++telemetry.receipts_failed;
    }
  }
  state.expected_action_receipts.clear();

  for (auto it = state.plots.begin(); it != state.plots.end();) {
    const auto& tile = tile_at(context.phase_start, context.player,
                               it->view.tile);
    if (tile.kind == TileKind::PLANT && tile.crop == it->view.desired &&
        tile.watered_today) {
      ++state.audit.plot_debts_closed;
      ++telemetry.debts_closed;
      it = state.plots.erase(it);
    } else {
      ++it;
    }
  }
  return true;
}

int protected_cash(const PlayerAction& raw, int cash) {
  const bool has_hard_buy = std::any_of(
      raw.market.begin(), raw.market.end(), [](const Action& action) {
        return action.op == Op::BUY_SEED || action.op == Op::BUY_ANIMAL ||
            action.op == Op::BUY_PRODUCT;
      });
  return has_hard_buy ? std::max(0, cash) : 0;
}

std::optional<purchase_recovery::Proposal> purchase_proposal(
    OnlineElasticRepairOwner::Impl& state,
    const repair_fork::RepairContext& context,
    OnlineElasticRepairOwner::Impl::PlotDebt& plot, int& protected_amount) {
  const int crop = static_cast<int>(plot.view.desired);
  if (!plot.view.purchase_debt.has_value()) {
    plot.view.purchase_debt = state.purchases.open(
        {Op::BUY_SEED, plot.view.desired, 1, kSeedCost[crop],
         "online-elastic plot debt"});
  }
  const auto debt = *plot.view.purchase_debt;
  for (const auto& order : context.raw_g001.market)
    if (order.op == Op::BUY_SEED && order.item == plot.view.desired &&
        order.quantity > 0) {
      protected_amount = 0;
      return state.purchases.describe_hard_order(debt, order.quantity);
    }
  protected_amount = protected_cash(
      context.raw_g001,
      static_cast<int>(context.phase_start.farms()[context.player].money));
  return state.purchases.propose_recovery(
      debt, {context.step,
             static_cast<int>(context.phase_start.farms()[context.player].money),
             protected_amount});
}

void maybe_request_seed(OnlineElasticRepairOwner::Impl& state,
                        const repair_fork::RepairContext& context,
                        repair_fork::RepairDecision& decision) {
  for (auto& plot : state.plots) {
    const int crop = static_cast<int>(plot.view.desired);
    const auto& tile = tile_at(context.phase_start, context.player,
                               plot.view.tile);
    if ((tile.kind != TileKind::EMPTY && tile.kind != TileKind::WEED) ||
        context.phase_start.privates()[context.player].seeds[crop] > 0)
      continue;
    int protected_amount = 0;
    const auto proposal = purchase_proposal(state, context, plot,
                                             protected_amount);
    if (!proposal.has_value()) continue;
    repair_fork::RequiredPurchase required{
        proposal->debt_id, proposal->order.op, proposal->order.item,
        proposal->order.quantity,
        context.phase_start.day() * context.phase_start.config().turns_per_day +
            context.phase_start.config().turns_per_day - 1};
    repair_fork::ExactMarketCompiler compiler;
    const std::array required_array{required};
    const auto compiled = compiler.compile(
        context.phase_start, context.player, context.raw_g001.market,
        required_array);
    if (compiled.bindings.size() != 1) {
      ++state.audit.fail_closed;
      return;
    }
    const auto& binding = compiled.bindings.front();
    if (accepted(binding.status)) {
      const int holding =
          context.phase_start.privates()[context.player].seeds[crop];
      const auto status = state.purchases.stage_final(
          *proposal,
          {context.step, binding.market_slot, compiled.market, holding,
           static_cast<int>(context.phase_start.farms()[context.player].money),
           protected_amount});
      if (status != purchase_recovery::StageStatus::Selected) {
        ++state.audit.fail_closed;
        return;
      }
    }
    state.expected_purchases.push_back({binding, context.step});
    decision.required_purchases.push_back(required);
    return;
  }
}

repair_fork::RepairTelemetry telemetry_delta(const OnlineAuditView& before,
                                              const OnlineAuditView& after) {
  repair_fork::RepairTelemetry out;
  out.triggers = after.triggers - before.triggers;
  out.debts_opened = after.plot_debts_opened - before.plot_debts_opened;
  out.debts_closed = after.plot_debts_closed - before.plot_debts_closed;
  out.receipts_confirmed =
      after.unit_receipts_confirmed + after.purchase_zero_fills +
      after.purchase_partial_fills + after.purchase_full_fills -
      before.unit_receipts_confirmed - before.purchase_zero_fills -
      before.purchase_partial_fills - before.purchase_full_fills;
  out.receipts_failed = after.unit_receipts_failed +
      after.purchase_receipt_failures - before.unit_receipts_failed -
      before.purchase_receipt_failures;
  out.fail_closed = after.fail_closed - before.fail_closed;
  return out;
}

}  // namespace

OnlineElasticRepairOwner::OnlineElasticRepairOwner()
    : impl_(std::make_unique<Impl>()) {}
OnlineElasticRepairOwner::~OnlineElasticRepairOwner() = default;
OnlineElasticRepairOwner::OnlineElasticRepairOwner(
    OnlineElasticRepairOwner&&) noexcept = default;
OnlineElasticRepairOwner& OnlineElasticRepairOwner::operator=(
    OnlineElasticRepairOwner&&) noexcept = default;

std::string OnlineElasticRepairOwner::name() const {
  return "online_elastic_repair_owner";
}

bool OnlineElasticRepairOwner::install_frozen_suffix(
    FrozenDaySuffixCertificate certificate) {
  if (!valid_certificate_shape(certificate)) return false;
  impl_->certificates[{certificate.day, certificate.actor}] =
      std::move(certificate);
  return true;
}

repair_fork::RepairDecision OnlineElasticRepairOwner::decide(
    const repair_fork::RepairContext& context) {
  auto baseline = baseline_decision(context);
  if (context.player < 0 || context.player >= 2 || context.step < 0 ||
      context.step != context.phase_start.step_count())
    throw std::invalid_argument("online owner context is not phase-aligned");
  if (impl_->last_step >= 0 && context.step != impl_->last_step + 1) {
    ++impl_->audit.fail_closed;
    impl_->poisoned = true;
  }
  if (impl_->poisoned) {
    baseline.telemetry.fail_closed = 1;
    return baseline;
  }

  const auto before_audit = impl_->audit;
  Impl staged = *impl_;
  repair_fork::RepairTelemetry receipt_telemetry;
  if (!settle_purchase_receipts(staged, context, receipt_telemetry)) {
    staged.poisoned = true;
    staged.last_step = context.step;
    baseline.telemetry = telemetry_delta(before_audit, staged.audit);
    *impl_ = std::move(staged);
    return baseline;
  }
  if (!settle_unit_attempts(staged, context, receipt_telemetry)) {
    staged.poisoned = true;
    staged.last_step = context.step;
    baseline.telemetry = telemetry_delta(before_audit, staged.audit);
    *impl_ = std::move(staged);
    return baseline;
  }

  const int day = context.phase_start.day();
  for (auto& [actor, runtime] : staged.actors) {
    if (runtime.day == day) continue;
    if (!runtime.delayed.empty()) {
      ++staged.audit.midnight_fail_closed;
      ++staged.audit.fail_closed;
      staged.poisoned = true;
    }
    const auto manifest_generation = runtime.manifest_generation;
    runtime = {day, actor_generation(day, actor), {}, 0,
               manifest_generation};
  }
  if (staged.poisoned) {
    staged.last_step = context.step;
    baseline.telemetry = telemetry_delta(before_audit, staged.audit);
    *impl_ = std::move(staged);
    return baseline;
  }

  struct Candidate {
    int actor{-1};
    Simulator prefix;
    Position position{};
    bool sink{};
    bool trigger{};
    Impl::PlotDebt* plot{};
    const FrozenSlot* certified_slot{};
    bool certificate_invalid{};
  };
  std::vector<Candidate> candidates;
  candidates.reserve(context.raw_g001.units.size());

  for (int actor = 0;
       actor < static_cast<int>(context.raw_g001.units.size()); ++actor) {
    auto prefix = prefix_before_actor(context, actor);
    const auto position = actor_position(prefix, context.player, actor);
    const auto& tile = tile_at(prefix, context.player, position);
    const auto& raw = context.raw_g001.units[static_cast<std::size_t>(actor)];
    const bool no_effect = exact_no_effect(context, actor, prefix, raw);
    const bool trigger = valid_crop(raw.item) && is_crop_action(raw.op) &&
        no_effect && (tile.kind == TileKind::WEED ||
                      (raw.op == Op::PLANT && tile.kind == TileKind::EMPTY));
    auto* plot = staged.find_plot(position);
    if (trigger) {
      if (!plot) {
        staged.plots.push_back({{staged.next_plot_id++, position, raw.item,
                                 day, std::nullopt}});
        plot = &staged.plots.back();
        ++staged.audit.plot_debts_opened;
      } else if (plot->view.desired != raw.item) {
        plot->view.desired = raw.item;
        plot->view.origin_day = day;
        plot->view.purchase_debt.reset();
      }
      ++staged.audit.triggers;
    }
    bool invalid_certificate = false;
    const auto* slot = current_certified_slot(
        staged, context, actor, raw, invalid_certificate);
    candidates.push_back({actor, std::move(prefix), position,
                          raw.op == Op::PASS || no_effect, trigger, plot, slot,
                          invalid_certificate});
  }

  Candidate* selected = nullptr;
  for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
    const auto runtime = staged.actors.find(it->actor);
    const bool has_delayed = runtime != staged.actors.end() &&
        !runtime->second.delayed.empty();
    if ((!it->plot || !same_position(it->plot->view.tile, it->position)) &&
        !has_delayed)
      continue;
    bool higher_safe = true;
    for (int higher = it->actor + 1;
         higher < static_cast<int>(context.raw_g001.units.size()); ++higher) {
      const auto op = context.raw_g001.units[static_cast<std::size_t>(higher)].op;
      higher_safe = higher_safe && (op == Op::PASS || is_move(op));
    }
    if (higher_safe) {
      selected = &*it;
      break;
    }
  }

  auto decision = baseline;
  std::optional<Impl::UnitAttempt> semantic_attempt;
  if (selected) {
    const bool final_hour = context.phase_start.hour() ==
        context.phase_start.config().turns_per_day - 1;
    const bool terminal_action = context.step >=
        context.phase_start.config().episode_steps - 2;
    auto& runtime = staged.actors[selected->actor];
    if (runtime.day != day)
      runtime = {day, actor_generation(day, selected->actor), {}, 0, 0};
    if (terminal_action && !runtime.delayed.empty()) {
      ++staged.audit.fail_closed;
      staged.poisoned = true;
    }
    if (selected->certificate_invalid) {
      ++staged.audit.invalid_capacity_certificate;
      ++staged.audit.fail_closed;
    } else {
      const auto& tile = tile_at(selected->prefix, context.player,
                                 selected->position);
      std::optional<Action> transition;
      if (selected->plot && !final_hour && !terminal_action) {
        const int crop = static_cast<int>(selected->plot->view.desired);
        const int seeds =
            selected->prefix.privates()[context.player].seeds[crop];
        transition = required_transition(tile, selected->plot->view.desired,
                                         seeds);
      }
      const auto& raw = context.raw_g001.units[
          static_cast<std::size_t>(selected->actor)];
      bool may_service = selected->sink;
      bool delaying_move = false;
      if (is_move(raw.op)) {
        may_service = false;
        if (selected->certified_slot &&
            selected->certified_slot->kind == FrozenSlotKind::MoveToken &&
            transition.has_value()) {
          const int after_needed = std::max(
              0, remaining_transitions(tile, selected->plot->view.desired) - 1);
          const int future_sinks = future_certified_sinks(
              staged, day, selected->actor, context.step,
              context.phase_start.config().episode_steps - 2);
          may_service = future_sinks >=
              static_cast<int>(runtime.delayed.size()) + 1 + after_needed;
          delaying_move = may_service;
        } else if (transition.has_value()) {
          ++staged.audit.missing_capacity_certificate;
          ++staged.audit.fail_closed;
        }
      }
      if (may_service && transition.has_value() &&
          !runtime.delayed.empty()) {
        const int after_needed = std::max(
            0, remaining_transitions(tile, selected->plot->view.desired) - 1);
        const int future_sinks = future_certified_sinks(
            staged, day, selected->actor, context.step,
            context.phase_start.config().episode_steps - 2);
        may_service = selected->certified_slot &&
            selected->certified_slot->kind ==
                FrozenSlotKind::CertifiedSink &&
            future_sinks >= static_cast<int>(runtime.delayed.size()) +
                after_needed;
      }
      if (may_service && transition.has_value() &&
          transition->op == Op::PLANT) {
        const int crop = static_cast<int>(transition->item);
        int manifest_demand = 1;
        for (int actor = 0;
             actor < static_cast<int>(context.raw_g001.units.size()); ++actor) {
          if (actor == selected->actor) continue;
          const auto& action = context.raw_g001.units[
              static_cast<std::size_t>(actor)];
          manifest_demand += action.op == Op::PLANT &&
              static_cast<int>(action.item) == crop;
        }
        may_service = manifest_demand <=
            context.phase_start.privates()[context.player].seeds[crop];
      }
      if (transition.has_value() && may_service) {
        if (delaying_move) {
          runtime.delayed.push_back({raw, context.step});
          ++staged.audit.move_tokens_delayed;
        }
        decision.units[static_cast<std::size_t>(selected->actor)] = *transition;
        decision.sources[static_cast<std::size_t>(selected->actor)] =
            {selected->actor, -1, *transition};
        semantic_attempt = Impl::UnitAttempt{
            selected->plot->view.id, context.step, transition->op,
            selected->position, selected->plot->view.desired};
      } else if (!terminal_action && !runtime.delayed.empty() &&
                 selected->certified_slot &&
                 (selected->certified_slot->kind ==
                      FrozenSlotKind::CertifiedSink ||
                  selected->certified_slot->kind ==
                      FrozenSlotKind::MoveToken)) {
        auto token = runtime.delayed.front();
        runtime.delayed.pop_front();
        if (selected->certified_slot->kind == FrozenSlotKind::MoveToken)
          runtime.delayed.push_back({raw, context.step});
        decision.units[static_cast<std::size_t>(selected->actor)] = token.action;
        decision.sources[static_cast<std::size_t>(selected->actor)] =
            {selected->actor, token.source_step, token.action};
        ++staged.audit.move_tokens_replayed;
      }
    }
  }

  maybe_request_seed(staged, context, decision);
  const bool units_changed = !std::equal(
      decision.units.begin(), decision.units.end(),
      context.raw_g001.units.begin(), context.raw_g001.units.end(),
      action_equal);
  if (units_changed) {
    int changed_actor = -1;
    for (int actor = 0; actor < static_cast<int>(decision.units.size()); ++actor)
      if (!action_equal(decision.units[static_cast<std::size_t>(actor)],
                        context.raw_g001.units[static_cast<std::size_t>(actor)])) {
        if (changed_actor >= 0)
          throw std::logic_error("online owner changed more than one actor");
        changed_actor = actor;
      }
    auto& runtime = staged.actors[changed_actor];
    ++runtime.manifest_generation;
    auto final_joint = context.raw_joint;
    final_joint[static_cast<std::size_t>(context.player)].units = decision.units;
    const repair_fork::ActorPrefixAuthority authority{
        changed_actor, runtime.manifest_generation,
        repair_fork::unit_prefix_manifest_hash(decision.units, changed_actor),
        repair_fork::post_unit_prefix_state_fingerprint(
            context.phase_start, context.player, final_joint, changed_actor)};
    decision.prefix_authority.push_back(authority);
    const repair_fork::ActionReceipt expected{
        context.step, changed_actor, authority.manifest_generation,
        authority.prefix_manifest_hash,
        authority.post_prefix_state_fingerprint,
        decision.units[static_cast<std::size_t>(changed_actor)]};
    staged.expected_action_receipts.push_back(
        {expected, std::move(semantic_attempt)});
  }
  staged.last_step = context.step;
  decision.telemetry = telemetry_delta(before_audit, staged.audit);
  *impl_ = std::move(staged);
  return decision;
}

OnlineAuditView OnlineElasticRepairOwner::audit() const { return impl_->audit; }

std::vector<PlotDebtView> OnlineElasticRepairOwner::plot_debts() const {
  std::vector<PlotDebtView> out;
  out.reserve(impl_->plots.size());
  for (const auto& plot : impl_->plots) out.push_back(plot.view);
  return out;
}

void OnlineElasticRepairOwner::reset() {
  impl_ = std::make_unique<Impl>();
}

}  // namespace g001::online_elastic
