#include "transactional_animal_repair_owner.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
namespace own = g001::transactional_animal_repair;
namespace api = g001::repair_fork;
namespace tx = g001::repair_owner_composer;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using fastkag::TileKind;
namespace {
void ck(bool x, const char *m) {
  if (!x)
    throw std::runtime_error(m);
}
fastkag::Config cfg(int cash = 3000) {
  fastkag::Config c;
  c.starting_money = cash;
  c.weed_spawn_chance = 0;
  c.episode_steps = 240;
  return c;
}
PlayerAction raw(int n = 1) {
  PlayerAction a;
  a.units.resize(n);
  return a;
}
std::array<PlayerAction, 2> joint(PlayerAction a, int p = 0) {
  std::array<PlayerAction, 2> j{raw(), raw()};
  j[p] = std::move(a);
  return j;
}
fastkag::Tile &tile(Simulator &s, int r, int c, int p = 0) {
  return const_cast<fastkag::Farm &>(s.farms()[p])
      .tiles[r * s.config().board_size + c];
}
api::RepairContext ctx(const Simulator &s, int p,
                       const std::array<PlayerAction, 2> &j) {
  return {s, p, s.step_count(), j[p], j, {}, {}};
}
fastkag::PrivateState &priv(Simulator &s, int player = 0) {
  return const_cast<fastkag::PrivateState &>(s.privates()[player]);
}
fastkag::Farm &farm(Simulator &s, int player = 0) {
  return const_cast<fastkag::Farm &>(s.farms()[player]);
}
struct G {
  tx::CommitGrant g;
  api::MarketCompileResult m;
};
G grant(const Simulator &s, int p, const std::array<PlayerAction, 2> &j,
        const tx::PreparedRepair &q, std::uint64_t gen = 1) {
  G z;
  z.g.submitted_step = s.step_count();
  z.g.final_units = q.units;
  auto fj = j;
  fj[p].units = q.units;
  for (int a : q.claimed_actors) {
    api::ActorPrefixAuthority au{
        a, gen, api::unit_prefix_manifest_hash(q.units, a),
        api::post_unit_prefix_state_fingerprint(s, p, fj, a)};
    z.g.actions.push_back({a, q.units[a], au});
  }
  std::vector<api::RequiredPurchase> ex;
  for (std::size_t i = 0; i < q.required_purchases.size(); i++) {
    auto r = q.required_purchases[i];
    r.debt_id = (1ULL << 62) + r.debt_id + i;
    ex.push_back(r);
  }
  api::ExactMarketCompiler mc;
  z.m = mc.compile(s, p, j[p].market, ex);
  for (std::size_t i = 0; i < ex.size(); i++)
    z.g.purchases.push_back(
        {q.required_purchases[i].debt_id, ex[i].debt_id, z.m.bindings[i]});
  return z;
}
struct R {
  std::vector<api::ActionReceipt> a;
  std::vector<api::PurchaseReceipt> p;
};
R run(Simulator &s, int p, std::array<PlayerAction, 2> j,
      const tx::PreparedRepair &q, const G &g) {
  j[p].units = q.units;
  j[p].market = g.m.market;
  int st = s.step_count();
  s.step(j);
  R r;
  for (auto &x : g.g.actions)
    r.a.push_back({st, x.actor, x.final_authority.manifest_generation,
                   x.final_authority.prefix_manifest_hash,
                   x.final_authority.post_prefix_state_fingerprint, x.emitted});
  auto f = s.last_market_fills()[p];
  for (auto &b : g.m.bindings) {
    int n = b.market_slot >= 0 && b.market_slot < (int)f.size()
                ? std::min(b.requested, (int)f[b.market_slot])
                : 0;
    r.p.push_back({b.debt_id, st, b.operation, b.item, b.requested, n,
                   b.market_slot, b.status});
  }
  return r;
}
bool settle(own::TransactionalAnimalRepairOwner &o, const Simulator &s, int p,
            const std::array<PlayerAction, 2> &j, const R &r = {}) {
  auto c = ctx(s, p, j);
  return o.settle_owned(c, r.a, r.p) ==
         tx::SettlementResult::AppliedSuccess;
}
tx::PreparedRepair
typed_prepare(own::TransactionalAnimalRepairOwner &owner,
              const Simulator &simulator, int player,
              const std::array<PlayerAction, 2> &actions, Item animal,
              std::span<const own::TypedAnimalObligation> obligations) {
  static_cast<void>(animal);
  return owner.prepare(ctx(simulator, player, actions), obligations);
}
tx::PreparedRepair typed_prepare(own::TransactionalAnimalRepairOwner &owner,
                                 const Simulator &simulator, int player,
                                 const std::array<PlayerAction, 2> &actions,
                                 Item animal, int row, int column,
                                 std::uint64_t id = 1, int actor = 0) {
  const own::TypedAnimalObligation obligation{id,
                                              player,
                                              actor,
                                              simulator.step_count() + 1,
                                              {row, column},
                                              {Op::PLACE, animal, 1},
                                              animal};
  return typed_prepare(owner, simulator, player, actions, animal,
                       std::span{&obligation, std::size_t{1}});
}
void abort_then_commit() {
  Simulator s(cfg(), 1);
  tile(s, 0, 0).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner o;
  PlayerAction a = raw();
  a.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto j = joint(a);
  for (int i = 0; i < 3; i++) {
    auto q = typed_prepare(o, s, 0, j, Item::GOOSE, 0, 0);
    o.abort(q.token);
    ck(o.objectives().empty() && o.expected_purchases() == 0,
       "abort mutated animal state");
    std::array<PlayerAction, 2> p{raw(), raw()};
    s.step(p);
    j = joint(a);
  }
  auto q = typed_prepare(o, s, 0, j, Item::GOOSE, 0, 0);
  auto g = grant(s, 0, j, q);
  ck(o.commit(q.token, g.g) && o.objectives().size() == 1,
     "commit after abort failed");
}
void retry_zero_rejected() {
  for (int mode = 0; mode < 2; mode++) {
    auto c = cfg(mode ? 3000 : 0);
    if (mode)
      c.max_market_orders = 1;
    Simulator s(c, 10 + mode);
    tile(s, 0, 0).kind = TileKind::COOP;
    own::TransactionalAnimalRepairOwner o;
    PlayerAction a = raw();
    a.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
    auto j = joint(a);
    auto q = typed_prepare(o, s, 0, j, Item::GOOSE, 0, 0);
    auto g = grant(s, 0, j, q);
    if (mode) {
      g.g.purchases[0].final_binding.status =
          api::PurchaseCompileStatus::RejectedNoSlot;
      g.g.purchases[0].final_binding.market_slot = -1;
      g.m.bindings[0] = g.g.purchases[0].final_binding;
      g.m.market.clear();
    }
    ck(o.commit(q.token, g.g), "retry commit");
    auto r = run(s, 0, j, q, g);
    ck(settle(o, s, 0, j, r), "retry settle");
    auto q2 = o.prepare(ctx(s, 0, j));
    ck(!q2.required_purchases.empty(),
       mode ? "rejected buy not retried" : "zero buy not retried");
    o.abort(q2.token);
  }
}
void multiple_partial() {
  Simulator s(cfg(300), 20);
  tile(s, 0, 0).kind = TileKind::COOP;
  tile(s, 0, 1).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner o;
  PlayerAction a = raw();
  a.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 2});
  auto j = joint(a);
  const std::array<own::TypedAnimalObligation, 2> targets{
      own::TypedAnimalObligation{
          1, 0, 0, 2, {0, 0}, {Op::PLACE, Item::GOOSE, 1}, Item::GOOSE},
      own::TypedAnimalObligation{
          2, 0, 0, 3, {0, 1}, {Op::PLACE, Item::GOOSE, 1}, Item::GOOSE}};
  auto q = typed_prepare(o, s, 0, j, Item::GOOSE, targets);
  ck(q.required_purchases.size() == 1 && q.required_purchases[0].quantity == 2,
     "multi-target hard order absent");
  auto g = grant(s, 0, j, q);
  ck(o.commit(q.token, g.g), "multi commit");
  auto r = run(s, 0, j, q, g);
  ck(r.p[0].filled == 1 && settle(o, s, 0, j, r), "multi partial settle");
  PlayerAction remaining = raw();
  remaining.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto remaining_joint = joint(remaining);
  auto q2 = o.prepare(ctx(s, 0, remaining_joint));
  ck(o.objectives().size() == 2 && q2.required_purchases.size() == 1 &&
         q2.required_purchases[0].quantity == 1,
     "partial fill blindly repeated quantity two for one remaining target");
  o.abort(q2.token);
}
void player1_and_move() {
  Simulator s(cfg(), 30);
  tile(s, 0, 0, 1).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner o;
  PlayerAction a = raw();
  a.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto j = joint(a, 1);
  j[0].units[0] = {Op::EAST, Item::NONE, 1};
  auto q = typed_prepare(o, s, 1, j, Item::GOOSE, 0, 0);
  auto g = grant(s, 1, j, q);
  ck(o.commit(q.token, g.g), "player1 final prefix");
  own::TransactionalAnimalRepairOwner m;
  PlayerAction mv = raw();
  mv.units[0] = {Op::EAST, Item::NONE, 1};
  auto mj = joint(mv);
  auto mq = m.prepare(ctx(s, 0, mj));
  ck(mq.units[0].op == Op::EAST && mq.claimed_actors.empty(),
     "animal changed MOVE");
  m.abort(mq.token);
}
void hour23() {
  Simulator s(cfg(), 40);
  tile(s, 4, 4).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner o;
  PlayerAction a = raw();
  a.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto actions = joint(a);
  auto q = typed_prepare(o, s, 0, actions, Item::GOOSE, 4, 4);
  auto g = grant(s, 0, actions, q);
  ck(o.commit(q.token, g.g), "hour23 purchase commit");
  auto receipts = run(s, 0, actions, q, g);
  ck(settle(o, s, 0, joint(raw()), receipts), "hour23 purchase settle");
  while (s.step_count() < 23) {
    std::array<PlayerAction, 2> p{raw(), raw()};
    s.step(p);
  }
  actions = joint(raw());
  q = o.prepare(ctx(s, 0, actions));
  ck(q.claimed_actors.empty(), "hour23 unit claim");
  o.abort(q.token);
  std::array<PlayerAction, 2> pass{raw(), raw()};
  s.step(pass);
  ck(s.day() == 1, "hour23 fixture did not cross midnight");
  q = o.prepare(ctx(s, 0, joint(raw())));
  ck(q.units[0].op == Op::PICKUP, "cross-day objective did not rebind");
  g = grant(s, 0, joint(raw()), q, 100);
  ck(o.commit(q.token, g.g), "cross-day generation commit failed");
}

void shared_resource_action_blocks_takeover() {
  Simulator s(cfg(), 50);
  tile(s, 4, 4).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner o;
  PlayerAction buy = raw(2);
  buy.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto buy_joint = joint(buy);
  auto proposal = typed_prepare(o, s, 0, buy_joint, Item::GOOSE, 4, 4);
  auto selected = grant(s, 0, buy_joint, proposal);
  ck(o.commit(proposal.token, selected.g), "shared setup commit");
  auto receipts = run(s, 0, buy_joint, proposal, selected);
  ck(settle(o, s, 0, joint(raw(2)), receipts), "shared setup settle");

  PlayerAction contested = raw(2);
  contested.units[1] = {Op::PICKUP, Item::WHEAT, 1};
  auto contested_joint = joint(contested);
  const auto blocked = o.prepare(ctx(s, 0, contested_joint));
  ck(blocked.claimed_actors.empty() && blocked.units[1].op == Op::PICKUP,
     "shared shed/wheat obligation was overwritten");
  o.abort(blocked.token);
}

void wrong_place_returns_to_pickup() {
  Simulator s(cfg(), 60);
  tile(s, 4, 4).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner o;
  PlayerAction buy = raw();
  buy.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto buy_joint = joint(buy);
  auto proposal = typed_prepare(o, s, 0, buy_joint, Item::GOOSE, 4, 4);
  auto selected = grant(s, 0, buy_joint, proposal);
  ck(o.commit(proposal.token, selected.g), "wrong-place BUY commit");
  auto receipts = run(s, 0, buy_joint, proposal, selected);
  ck(settle(o, s, 0, joint(raw()), receipts), "wrong-place BUY settle");

  auto pass_joint = joint(raw());
  proposal = o.prepare(ctx(s, 0, pass_joint));
  ck(proposal.units[0].op == Op::PICKUP, "BUY did not hand off to PICKUP");
  selected = grant(s, 0, pass_joint, proposal, 2);
  ck(o.commit(proposal.token, selected.g), "PICKUP commit");
  receipts = run(s, 0, pass_joint, proposal, selected);
  ck(settle(o, s, 0, joint(raw()), receipts), "PICKUP settle");

  proposal = o.prepare(ctx(s, 0, pass_joint));
  ck(proposal.units[0].op == Op::PLACE, "PICKUP did not hand off to PLACE");
  selected = grant(s, 0, pass_joint, proposal, 3);
  ck(o.commit(proposal.token, selected.g), "PLACE commit");
  tile(s, 4, 4) = {};
  receipts = run(s, 0, pass_joint, proposal, selected);
  ck(!settle(o, s, 0, joint(raw()), receipts),
     "wrong-structure PLACE was incorrectly reported as success");
  const auto retry = o.prepare(ctx(s, 0, pass_joint));
  ck(retry.units[0].op == Op::PICKUP,
     "wrong PLACE fallback did not recover from shed via PICKUP");
  o.abort(retry.token);
}

void exact_lifecycle_reaches_first_yield() {
  Simulator s(cfg(), 70);
  tile(s, 4, 4).kind = TileKind::COOP;
  priv(s).shed[static_cast<int>(Item::WHEAT)] = 16;
  own::TransactionalAnimalRepairOwner o;
  PlayerAction buy = raw();
  buy.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto actions = joint(buy);
  auto proposal = typed_prepare(o, s, 0, actions, Item::GOOSE, 4, 4);
  auto selected = grant(s, 0, actions, proposal, 1);
  ck(o.commit(proposal.token, selected.g), "lifecycle BUY commit");
  auto receipts = run(s, 0, actions, proposal, selected);
  ck(settle(o, s, 0, joint(raw()), receipts), "lifecycle BUY settle");

  bool pickup = false;
  bool place = false;
  bool feed = false;
  bool care = false;
  bool harvest = false;
  while (s.step_count() < 120 && !o.objectives().empty()) {
    PlayerAction route = raw();
    if (s.hour() == 0 && tile(s, 4, 4).kind == TileKind::ANIMAL &&
        tile(s, 4, 4).yield_units == 0)
      route.units[0] = {Op::PICKUP, Item::WHEAT, 1};
    actions = joint(route);
    proposal = o.prepare(ctx(s, 0, actions));
    const auto op = proposal.units[0].op;
    pickup = pickup || op == Op::PICKUP;
    place = place || op == Op::PLACE;
    feed = feed || op == Op::FEED;
    care = care || op == Op::CARE;
    harvest = harvest || op == Op::HARVEST;
    selected = grant(s, 0, actions, proposal,
                     static_cast<std::uint64_t>(s.step_count() + 1));
    ck(o.commit(proposal.token, selected.g), "lifecycle unit commit");
    receipts = run(s, 0, actions, proposal, selected);
    ck(settle(o, s, 0, joint(raw()), receipts), "lifecycle unit settle");
  }
  ck(pickup && place && feed && care && harvest && o.objectives().empty() &&
         priv(s).inventories[0][static_cast<int>(Item::EGG)] > 0,
     "BUY->PICKUP->PLACE->FEED/CARE->FIRST_YIELD did not complete");
}
void typed_target_is_required_and_board_order_is_ignored() {
  Simulator s(cfg(), 80);
  tile(s, 0, 0).kind = TileKind::COOP;
  tile(s, 4, 4).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner o;
  PlayerAction buy = raw();
  buy.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto actions = joint(buy);
  auto untyped = o.prepare(ctx(s, 0, actions));
  ck(untyped.required_purchases.empty() && o.objectives().empty(),
     "untyped BUY guessed the first matching structure");
  o.abort(untyped.token);
  auto typed = typed_prepare(o, s, 0, actions, Item::GOOSE, 4, 4, 81);
  auto selected = grant(s, 0, actions, typed);
  ck(typed.required_purchases.size() == 1 &&
         o.commit(typed.token, selected.g) && o.objectives().size() == 1 &&
         o.objectives()[0].spec.key.tile ==
             g001::persistent_production::TileKey{4, 4},
     "typed target did not bind the original PLACE structure");
}
void missing_purchase_receipt_is_not_cleared_as_retryable() {
  Simulator s(cfg(), 90);
  tile(s, 4, 4).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner o;
  PlayerAction buy = raw();
  buy.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto actions = joint(buy);
  auto proposal = typed_prepare(o, s, 0, actions, Item::GOOSE, 4, 4, 91);
  auto selected = grant(s, 0, actions, proposal);
  ck(o.commit(proposal.token, selected.g), "missing receipt setup commit");
  const auto receipts = run(s, 0, actions, proposal, selected);
  ck(!settle(o, s, 0, joint(raw())) && o.expected_purchases() == 1 &&
         o.audit().unresolved_purchase_receipts == 1,
     "missing staged purchase receipt was silently cleared");
  bool prepare_blocked = false;
  try {
    static_cast<void>(o.prepare(ctx(s, 0, joint(raw()))));
  } catch (const std::logic_error &) {
    prepare_blocked = true;
  }
  ck(prepare_blocked, "unresolved purchase allowed a duplicate retry");
  ck(settle(o, s, 0, joint(raw()), receipts) && o.expected_purchases() == 0,
     "late exact staged purchase receipt did not settle");
}
void conflicting_typed_place_bytes_fail_closed_in_both_orders() {
  Simulator s(cfg(), 100);
  tile(s, 4, 4).kind = TileKind::COOP;
  PlayerAction buy = raw();
  buy.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  const auto actions = joint(buy);
  std::array<own::TypedAnimalObligation, 2> obligations{
      own::TypedAnimalObligation{
          101, 0, 0, 2, {4, 4}, {Op::PLACE, Item::GOOSE, 1}, Item::GOOSE},
      own::TypedAnimalObligation{
          101, 0, 0, 2, {4, 4}, {Op::PLACE, Item::GOOSE, 2}, Item::GOOSE}};
  for (int reversed = 0; reversed < 2; ++reversed) {
    if (reversed)
      std::swap(obligations[0], obligations[1]);
    own::TransactionalAnimalRepairOwner owner;
    const auto proposal =
        typed_prepare(owner, s, 0, actions, Item::GOOSE, obligations);
    ck(proposal.required_purchases.empty() && owner.objectives().empty(),
       "conflicting typed PLACE bytes were resolved by input order");
    owner.abort(proposal.token);
  }
}

void lost_acquisition_reopens_transactionally() {
  Simulator s(cfg(), 110);
  tile(s, 0, 0).kind = TileKind::COOP;
  tile(s, 1, 1).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner owner;

  PlayerAction buy = raw(2);
  buy.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  auto actions = joint(buy);
  auto proposal =
      typed_prepare(owner, s, 0, actions, Item::GOOSE, 0, 0, 111, 0);
  auto selected = grant(s, 0, actions, proposal);
  ck(owner.commit(proposal.token, selected.g), "loss setup BUY commit");
  auto receipts = run(s, 0, actions, proposal, selected);
  ck(settle(owner, s, 0, joint(raw(2)), receipts) &&
         owner.objectives()[0].purchase_complete,
     "loss setup BUY did not complete");

  // A different actor takes the acquired goose, then places it on a structure
  // that is not owned by the typed objective. The intended target is still an
  // empty coop, while shed and every own inventory are now empty.
  farm(s).hands.push_back({4, 4});
  priv(s).inventories.push_back({});
  priv(s).inventory_order.push_back({});
  PlayerAction steal = raw(2);
  steal.units[1] = {Op::PICKUP, Item::GOOSE, 1};
  s.step(joint(steal));
  ck(priv(s).inventories[1][static_cast<int>(Item::GOOSE)] == 1,
     "second actor did not PICKUP acquired animal");
  farm(s).hands[0] = {1, 1};
  PlayerAction misplace = raw(2);
  misplace.units[1] = {Op::PLACE, Item::GOOSE, 1};
  s.step(joint(misplace));
  ck(tile(s, 1, 1).kind == TileKind::ANIMAL &&
         tile(s, 0, 0).kind == TileKind::COOP &&
         priv(s).shed[static_cast<int>(Item::GOOSE)] == 0 &&
         priv(s).inventories[0][static_cast<int>(Item::GOOSE)] == 0 &&
         priv(s).inventories[1][static_cast<int>(Item::GOOSE)] == 0,
     "wrong PLACE loss witness is not exact");

  // A wrong-target board animal remains an owned economic resource. Retain
  // the typed target, but do not duplicate BUY or claim that placed animals
  // can be moved between structures by the current simulator ABI.
  auto retained = owner.prepare(ctx(s, 0, joint(raw(2))));
  ck(retained.required_purchases.empty() && retained.claimed_actors.empty() &&
         owner.objectives()[0].purchase_complete,
     "wrong-target board animal was misclassified as lost inventory");
  owner.abort(retained.token);

  // Explicit removal models genuine external loss. Only then may prepare
  // reopen its copied ledger. Multiple conflict aborts retain the completed
  // persistent objective and reproduce the same stable debt.
  tile(s, 1, 1) = {};
  std::uint64_t stable_debt = 0;
  for (int attempt = 0; attempt < 3; ++attempt) {
    const auto recovery = owner.prepare(ctx(s, 0, joint(raw(2))));
    ck(recovery.required_purchases.size() == 1,
       "lost animal did not propose replacement acquisition");
    if (attempt == 0)
      stable_debt = recovery.required_purchases[0].debt_id;
    ck(recovery.required_purchases[0].debt_id == stable_debt &&
           owner.objectives()[0].purchase_complete &&
           owner.expected_purchases() == 0,
       "prepare/abort mutated or duplicated persistent acquisition debt");
    owner.abort(recovery.token);
    ck(owner.objectives()[0].purchase_complete,
       "abort persisted speculative reopen");
  }

  // Commit the reopen with no cash: an exact zero fill keeps the same debt
  // open and the next real prepare retries it.
  farm(s).money = 0;
  PlayerAction hard_zero = raw(2);
  hard_zero.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  actions = joint(hard_zero);
  proposal = owner.prepare(ctx(s, 0, actions));
  selected = grant(s, 0, actions, proposal, 20);
  ck(owner.commit(proposal.token, selected.g), "zero recovery commit");
  receipts = run(s, 0, actions, proposal, selected);
  ck(receipts.p.size() == 1 && receipts.p[0].filled == 0 &&
         settle(owner, s, 0, joint(raw(2)), receipts),
     "zero recovery receipt did not preserve reopened debt");
  proposal = owner.prepare(ctx(s, 0, actions));
  ck(proposal.required_purchases.size() == 1 &&
         proposal.required_purchases[0].debt_id == stable_debt,
     "zero recovery did not retry the same objective debt");
  owner.abort(proposal.token);

  // A hard quantity-two order with funds for one is a partial market order,
  // but one arrived unit completely satisfies this quantity-one objective.
  // The owner must not blindly buy the unneeded second unit next turn.
  farm(s).money = 300;
  PlayerAction hard_partial = raw(2);
  hard_partial.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 2});
  actions = joint(hard_partial);
  proposal = owner.prepare(ctx(s, 0, actions));
  ck(proposal.required_purchases.size() == 1 &&
         proposal.required_purchases[0].quantity == 2,
     "reopened debt did not bind exact hard quantity-two order");
  selected = grant(s, 0, actions, proposal, 21);
  ck(owner.commit(proposal.token, selected.g), "partial recovery commit");
  receipts = run(s, 0, actions, proposal, selected);
  ck(receipts.p.size() == 1 && receipts.p[0].filled == 1 &&
         settle(owner, s, 0, joint(raw(2)), receipts),
     "partial market recovery did not credit the required one unit");
  proposal = owner.prepare(ctx(s, 0, joint(raw(2))));
  ck(proposal.required_purchases.empty(),
     "one-unit recovered objective attempted to overbuy hard suffix");
  owner.abort(proposal.token);
}

void two_goose_credits_share_one_wrong_board_unit_once() {
  Simulator s(cfg(), 120);
  tile(s, 0, 0).kind = TileKind::COOP;
  tile(s, 0, 1).kind = TileKind::COOP;
  tile(s, 2, 2).kind = TileKind::COOP;
  own::TransactionalAnimalRepairOwner owner;
  const std::array<own::TypedAnimalObligation, 2> targets{
      own::TypedAnimalObligation{
          121, 0, 0, 2, {0, 0}, {Op::PLACE, Item::GOOSE, 1}, Item::GOOSE},
      own::TypedAnimalObligation{
          122, 0, 0, 3, {0, 1}, {Op::PLACE, Item::GOOSE, 1}, Item::GOOSE}};

  PlayerAction first = raw();
  first.units[0] = {Op::EAST, Item::NONE, 1};
  first.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 2});
  auto actions = joint(first);
  auto proposal = typed_prepare(owner, s, 0, actions, Item::GOOSE, targets);
  auto selected = grant(s, 0, actions, proposal, 30);
  ck(owner.commit(proposal.token, selected.g), "two-credit first commit");
  auto receipts = run(s, 0, actions, proposal, selected);
  ck(settle(owner, s, 0, joint(raw()), receipts), "two-credit first settle");

  PlayerAction second = raw();
  second.units[0] = {Op::EAST, Item::NONE, 1};
  second.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  actions = joint(second);
  proposal = owner.prepare(ctx(s, 0, actions));
  selected = grant(s, 0, actions, proposal, 31);
  ck(proposal.required_purchases.size() == 1 &&
         owner.commit(proposal.token, selected.g),
     "second objective acquisition commit failed");
  receipts = run(s, 0, actions, proposal, selected);
  ck(settle(owner, s, 0, joint(raw()), receipts) &&
         owner.objectives().size() == 2 &&
         owner.objectives()[0].purchase_complete &&
         owner.objectives()[1].purchase_complete,
     "two objective credits did not complete");

  // Leave exactly one physical goose, placed on neither typed target. Stable
  // item allocation may retain one credit, but the same unit cannot satisfy
  // both objectives: exactly one replacement debt must be exposed.
  priv(s).shed[static_cast<int>(Item::GOOSE)] = 0;
  for (auto &inventory : priv(s).inventories)
    inventory[static_cast<int>(Item::GOOSE)] = 0;
  tile(s, 2, 2) = {};
  tile(s, 2, 2).kind = TileKind::ANIMAL;
  tile(s, 2, 2).animal = Item::GOOSE;
  proposal = owner.prepare(ctx(s, 0, joint(raw())));
  ck(proposal.required_purchases.size() == 1 &&
         proposal.required_purchases[0].operation == Op::BUY_ANIMAL &&
         proposal.required_purchases[0].item == Item::GOOSE &&
         proposal.required_purchases[0].quantity == 1 &&
         owner.objectives()[0].purchase_complete &&
         owner.objectives()[1].purchase_complete,
     "one wrong-board goose was double-claimed or caused two BUY debts");
  owner.abort(proposal.token);
}

void cross_tick_failed_buy_requires_live_provenance() {
  auto setup = [](int cash, std::uint64_t seed) {
    Simulator simulator(cfg(cash), seed);
    tile(simulator, 0, 0).kind = TileKind::PASTURE;
    PlayerAction submitted = raw();
    submitted.market = {{Op::HIRE, Item::NONE, 1},
                        {Op::BUY_ANIMAL, Item::COW, 1}};
    simulator.step(joint(submitted));
    return simulator;
  };
  auto obligation_for = [](const Simulator &simulator, int fill) {
    own::TypedAnimalObligation value{
        130, 0, 1, 2, {0, 0}, {Op::PLACE, Item::COW, 1}, Item::COW};
    value.acquisition_step = 0;
    value.acquisition_market_slot = 1;
    value.acquisition_action = {Op::BUY_ANIMAL, Item::COW, 1};
    value.acquisition_fill = fill;
    value.acquisition_seed = simulator.seed();
    return value;
  };

  auto failed = setup(2, 130);
  ck(failed.step_count() == 1 && failed.farms()[0].hands.size() == 1 &&
         failed.last_market_fills()[0] == std::vector<std::int32_t>({1, 0}),
     "cross-tick failed BUY fixture changed");
  const auto current = joint(raw(2));
  auto exact = obligation_for(failed, 0);
  own::TransactionalAnimalRepairOwner owner;
  auto proposal = typed_prepare(owner, failed, 0, current, Item::COW,
                                std::span{&exact, std::size_t{1}});
  auto selected = grant(failed, 0, current, proposal, 130);
  ck(owner.commit(proposal.token, selected.g) &&
         owner.objectives().size() == 1 &&
         owner.objectives()[0].spec.item == Item::COW,
     "exact previous-tick failed BUY did not open the typed objective");
  const auto negative_fixture = failed;
  failed.step(current);
  ck(settle(owner, failed, 0, current),
     "cross-tick objective no-op hand did not settle");
  farm(failed).money = 1000;
  auto retry = owner.prepare(ctx(failed, 0, current));
  ck(retry.required_purchases.size() == 1 &&
         retry.required_purchases[0].operation == Op::BUY_ANIMAL &&
         retry.required_purchases[0].item == Item::COW &&
         retry.required_purchases[0].quantity == 1,
     "cross-tick objective did not retry BUY_ANIMAL when cash recovered");
  owner.abort(retry.token);

  auto rejected = [&](own::TypedAnimalObligation forged) {
    own::TransactionalAnimalRepairOwner candidate;
    const auto output = typed_prepare(candidate, negative_fixture, 0, current,
                                      Item::COW,
                                      std::span{&forged, std::size_t{1}});
    const bool inert = output.required_purchases.empty() &&
                       candidate.objectives().empty();
    candidate.abort(output.token);
    return inert;
  };
  auto forged = exact;
  forged.acquisition_step = -1;
  ck(rejected(forged), "stale acquisition step opened an objective");
  forged = exact;
  forged.acquisition_market_slot = 0;
  ck(rejected(forged), "wrong market slot opened an objective");
  forged = exact;
  forged.acquisition_action.item = Item::SHEEP;
  ck(rejected(forged), "wrong acquisition item opened an objective");
  forged = exact;
  ++forged.acquisition_seed;
  ck(rejected(forged), "wrong episode seed opened an objective");
  auto second = exact;
  second.id = 131;
  second.target = {0, 1};
  auto ambiguous_fixture = negative_fixture;
  tile(ambiguous_fixture, 0, 1).kind = TileKind::PASTURE;
  std::array duplicated{exact, second};
  own::TransactionalAnimalRepairOwner ambiguous;
  proposal = typed_prepare(ambiguous, ambiguous_fixture, 0, current, Item::COW,
                           duplicated);
  ck(proposal.required_purchases.empty() && ambiguous.objectives().empty(),
     "one failed BUY provenance opened multiple target debts");
  ambiguous.abort(proposal.token);

  auto succeeded = setup(1000, 131);
  ck(succeeded.last_market_fills()[0] ==
         std::vector<std::int32_t>({1, 1}),
     "successful BUY fixture changed");
  auto filled = obligation_for(succeeded, 1);
  own::TransactionalAnimalRepairOwner ordinary;
  proposal = typed_prepare(ordinary, succeeded, 0, current, Item::COW,
                           std::span{&filled, std::size_t{1}});
  ck(proposal.required_purchases.empty() && ordinary.objectives().empty(),
     "successful ordinary G001 BUY was mislabeled as repair debt");
  ordinary.abort(proposal.token);
}
} // namespace
int main() {
  try {
    abort_then_commit();
    retry_zero_rejected();
    multiple_partial();
    player1_and_move();
    hour23();
    shared_resource_action_blocks_takeover();
    wrong_place_returns_to_pickup();
    exact_lifecycle_reaches_first_yield();
    typed_target_is_required_and_board_order_is_ignored();
    missing_purchase_receipt_is_not_cleared_as_retryable();
    conflicting_typed_place_bytes_fail_closed_in_both_orders();
    lost_acquisition_reopens_transactionally();
    two_goose_credits_share_one_wrong_board_unit_once();
    cross_tick_failed_buy_requires_live_provenance();
    std::cout << "transactional animal owner: 14 deterministic groups passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
