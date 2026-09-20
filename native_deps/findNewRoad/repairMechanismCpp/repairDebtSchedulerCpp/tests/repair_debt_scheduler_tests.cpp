#include "repair_debt_scheduler.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace rd = repair_debt;

namespace {

rd::ResourceKey carried(int actor, int item) {
  return {rd::ResourceScope::ActorCarried, actor, item};
}

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

rd::DebtNode node(std::uint64_t id, std::uint64_t origin, int day, int expiry,
                  int actor = 0, int x = 2, int y = 3) {
  rd::DebtNode n;
  n.identity = {id, origin, actor, 4, 9, x, y, 0};
  n.required_action = {17, 9, 1, x, y};
  n.admitted_day = day;
  n.expires_day = expiry;
  n.priority = 5;
  n.value = 20;
  n.cascade_value = 3;
  n.resources = {{carried(actor, 9), 1}};
  n.target_must_exist = true;
  n.identity.dependency_set_hash = rd::dependency_set_hash(n.dependencies);
  n.authorization_hash = rd::debt_authorization_hash(n);
  return n;
}

rd::TickInput input(int day, int hour, std::uint64_t physical,
                    std::vector<rd::MoveToken> moves = {}, int resource = 1,
                    std::set<std::pair<int, int>> targets = {{2, 3}},
                    std::set<int> actors = {0}) {
  rd::TickInput in;
  in.observation = {day, hour, 0, false, std::move(targets),
                    std::move(actors), physical};
  in.immutable_day_moves = std::move(moves);
  in.resources.holdings[carried(0, 9)] = resource;
  in.resources.issuer_generation = 7;
  in.resources.physical_evidence_hash = physical;
  in.resources.observation_hash = rd::observation_hash(in.observation);
  in.resources.content_hash = rd::resource_certificate_hash(in.resources);
  return in;
}

rd::DebtLedger ledger(std::vector<rd::DebtNode> nodes) {
  rd::DebtLedger l;
  l.registry_generation = 41;
  for (auto& n : nodes) {
    rd::DebtRecord record; record.node = n;
    l.records.emplace(n.identity.id, std::move(record));
  }
  l.content_hash = rd::ledger_hash(l);
  return l;
}

rd::FinalizeResult commit(rd::RepairDebtScheduler& scheduler, rd::TickInput in,
                          const rd::Proposal& p, bool effect = true) {
  rd::PhysicalReceipt receipt;
  receipt.proposal_binding_hash = p.binding_hash;
  receipt.committed_action = p.action;
  receipt.physically_applied = p.action.kind != rd::ActionKind::Pass;
  receipt.target_effect_observed = effect;
  if (p.action.kind == rd::ActionKind::Obligation) {
    const auto& obligation = scheduler.ledger().records.at(p.action.debt_id).node;
    for (const auto& need : obligation.resources)
      receipt.resource_deltas[need.key] -= need.quantity;
    for (const auto& produced : obligation.produces)
      receipt.resource_deltas[produced.key] += produced.quantity;
  }
  receipt.before_hash = in.observation.physical_hash;
  receipt.after_hash = receipt.before_hash +
                       (p.action.kind == rd::ActionKind::Pass ? 0 : 1);
  return scheduler.finalize(in, p, p.action, receipt);
}

std::vector<rd::MoveToken> moves(int day, int count) {
  std::vector<rd::MoveToken> out;
  for (int i = 0; i < count; ++i)
    out.push_back({static_cast<std::uint64_t>(1000 + day * 31 + i), day, 0, i,
                   i % 4});
  return out;
}

void translation_and_permutation() {
  for (int shift : {0, 11, 97}) {
    auto n = node(100 + shift, 700 + shift, 4 + shift, 6 + shift, 0,
                  2 + shift, 3 - shift);
    rd::RepairDebtScheduler s(ledger({n}));
    auto in = input(4 + shift, 0, 55 + shift, {}, 1,
                    {{2 + shift, 3 - shift}});
    auto p = s.prepare(in);
    require(p.accepted && p.action.kind == rd::ActionKind::Obligation,
            "translation changes decision kind");
    require(p.action.debt_id == static_cast<std::uint64_t>(100 + shift),
            "identity permutation not preserved");
    require(commit(s, in, p).committed, "translated commit failed");
  }
}

void multi_day_and_moves() {
  for (int delay = 1; delay <= 5; ++delay) {
    auto n = node(200 + delay, 900 + delay, 10, 10 + delay);
    n.authorization_hash = rd::debt_authorization_hash(n);
    rd::RepairDebtScheduler s(ledger({n}));
    int physical = 100;
    for (int day = 10; day <= 10 + delay; ++day) {
      auto daily = moves(day, 3);
      for (int hour = 0; hour < 24; ++hour) {
        auto in = input(day, hour, physical, daily,
                        day == 10 + delay ? 1 : 0);
        in.resources.certified_available_day[carried(0, 9)] = 10 + delay;
        in.resources.content_hash = rd::resource_certificate_hash(in.resources);
        auto p = s.prepare(in);
        require(p.accepted, "multi-day prepare failed");
        auto result = commit(s, in, p);
        require(result.committed, "multi-day commit failed");
        if (p.action.kind != rd::ActionKind::Pass) ++physical;
      }
    }
    require(s.ledger().records.at(200 + delay).state == rd::DebtState::Completed,
            "resource arrival did not close debt");
    require(s.metrics().moves_emitted == 3 * (delay + 1),
            "MOVE not exactly once across days");
    require(s.metrics().move_early == 0 && s.metrics().move_duplicate == 0 &&
                s.metrics().move_drop == 0,
            "MOVE order/coverage violation");
    require(s.ledger().records.size() == 1, "ledger grew while rebasing");
  }
}

void capacity_min_loss_drop() {
  auto cheap = node(301, 3010, 3, 3); cheap.value = 1; cheap.cascade_value = 0;
  cheap.authorization_hash = rd::debt_authorization_hash(cheap);
  auto valuable = node(302, 3020, 3, 3); valuable.value = 100;
  valuable.authorization_hash = rd::debt_authorization_hash(valuable);
  rd::RepairDebtScheduler s(ledger({cheap, valuable}));
  auto in = input(3, 22, 77, moves(3, 1));
  auto p = s.prepare(in);
  require(p.accepted && p.action.kind == rd::ActionKind::Obligation &&
              p.action.debt_id == 302,
          "minimum-damage kept set is not scheduled before hard MOVE");
  require(s.ledger().records.at(301).state == rd::DebtState::DroppedDominated,
          "minimum-loss capacity root not dropped");
  require(s.ledger().records.at(302).state == rd::DebtState::Active,
          "valuable feasible obligation was dropped");
}

void causal_drop_has_no_ghost() {
  auto root = node(401, 4010, 5, 6);
  auto child = node(402, 4020, 5, 6);
  child.dependencies = {401};
  child.identity.dependency_set_hash = rd::dependency_set_hash(child.dependencies);
  child.authorization_hash = rd::debt_authorization_hash(child);
  rd::RepairDebtScheduler s(ledger({root, child}));
  auto in = input(5, 0, 88, {}, 1, {});
  auto p = s.prepare(in);
  require(p.accepted && p.action.kind == rd::ActionKind::Pass,
          "ghost action emitted after target vanished");
  require(s.ledger().records.at(401).state == rd::DebtState::DroppedInfeasible &&
              s.ledger().records.at(402).state == rd::DebtState::DroppedInfeasible,
          "causal descendants not dropped");
  require(s.ledger().records.at(402).drop_reason == rd::DropReason::AncestorDropped,
          "descendant lacks typed cancellation reason");
}

void dominated_and_terminal() {
  auto expensive = node(501, 5010, 8, 12);
  expensive.value = 2; expensive.cascade_value = 1;
  expensive.estimated_repair_damage = 9;
  expensive.authorization_hash = rd::debt_authorization_hash(expensive);
  rd::RepairDebtScheduler s(ledger({expensive}));
  auto in = input(8, 0, 90);
  auto p = s.prepare(in);
  require(p.accepted && p.action.kind == rd::ActionKind::Pass,
          "dominated debt emitted");
  require(s.ledger().records.at(501).state == rd::DebtState::DroppedDominated,
          "dominated debt not disposed");

  auto live = node(502, 5020, 8, 12);
  rd::RepairDebtScheduler terminal(ledger({live}));
  auto end = input(8, 1, 91); end.observation.terminal = true;
  end.resources.observation_hash = rd::observation_hash(end.observation);
  end.resources.content_hash = rd::resource_certificate_hash(end.resources);
  auto ep = terminal.prepare(end);
  require(ep.accepted && terminal.ledger().records.at(502).state ==
                             rd::DebtState::ExpiredTerminal,
          "terminal did not explicitly close debt");
}

void hard_receipt_and_move_fail_closed() {
  rd::RepairDebtScheduler s(ledger({}));
  auto in = input(2, 23, 33, moves(2, 1));
  auto p = s.prepare(in);
  require(p.action.kind == rd::ActionKind::Move, "hard MOVE not chosen");
  auto bad = p.action; bad.direction ^= 1;
  rd::PhysicalReceipt receipt{p.binding_hash, bad, true, false, {}, 33, 34};
  auto result = s.finalize(in, p, bad, receipt);
  require(!result.committed && result.reject == rd::Reject::FinalActionMismatch &&
              s.fail_stopped(),
          "non-exact final action did not fail closed");
}

void typed_action_rollback_and_scopes() {
  auto obligation = node(650, 6500, 2, 3);
  rd::RepairDebtScheduler exact(ledger({obligation}));
  const auto original_hash = exact.ledger().content_hash;
  auto in = input(2, 0, 60);
  auto p = exact.prepare(in);
  auto tampered = p.action; ++tampered.production.op;
  rd::PhysicalReceipt receipt;
  receipt.proposal_binding_hash = p.binding_hash;
  receipt.committed_action = tampered;
  receipt.physically_applied = true;
  receipt.target_effect_observed = true;
  receipt.before_hash = 60; receipt.after_hash = 61;
  auto rejected = exact.finalize(in, p, tampered, receipt);
  require(!rejected.committed && exact.ledger().content_hash == original_hash,
          "typed action mismatch leaked staged ledger state");

  auto carried_node = node(651, 6510, 4, 5);
  auto seed_node = node(652, 6520, 4, 5);
  seed_node.resources = {{{rd::ResourceScope::FarmSeed, 0, 9}, 1}};
  // Invalid crop item proves scope validation is not an untyped alias.
  seed_node.authorization_hash = rd::debt_authorization_hash(seed_node);
  rd::RepairDebtScheduler invalid_scope(ledger({seed_node}));
  require(invalid_scope.prepare(input(4, 0, 61)).reject == rd::Reject::InvalidLedger,
          "invalid scoped seed certificate accepted");

  seed_node.resources = {{{rd::ResourceScope::FarmSeed, 0, 2}, 1}};
  seed_node.authorization_hash = rd::debt_authorization_hash(seed_node);
  rd::RepairDebtScheduler separate(ledger({carried_node, seed_node}));
  auto scoped = input(4, 0, 62);
  scoped.resources.certified_available_day[{rd::ResourceScope::FarmSeed, 0, 2}] = 5;
  scoped.resources.content_hash = rd::resource_certificate_hash(scoped.resources);
  auto selected = separate.prepare(scoped);
  require(selected.accepted && selected.action.debt_id == 651 &&
              separate.ledger().records.at(652).state == rd::DebtState::Deferred,
          "FarmSeed and ActorCarried resources aliased");
}

void typed_validation_properties() {
  {
    rd::RepairDebtScheduler zero_generation;
    auto in = input(1, 0, 10);
    require(!zero_generation.prepare(in).accepted && zero_generation.fail_stopped(),
            "zero registry generation accepted");
  }
  {
    rd::RepairDebtScheduler duplicate_identity(ledger({}));
    auto in = input(1, 0, 11, {{70, 1, 0, 0, 1}, {70, 1, 1, 0, 2}},
                    1, {{2, 3}}, {0, 1});
    require(duplicate_identity.prepare(in).reject == rd::Reject::InvalidMoveTokens,
            "duplicate MOVE identity accepted");
  }
  {
    rd::RepairDebtScheduler duplicate_ordinal(ledger({}));
    auto in = input(1, 0, 12, {{71, 1, 0, 0, 1}, {72, 1, 0, 0, 2}});
    require(duplicate_ordinal.prepare(in).reject == rd::Reject::InvalidMoveTokens,
            "duplicate actor ordinal accepted");
  }
  {
    auto a = node(710, 7100, 2, 4); auto b = node(711, 7110, 2, 4);
    a.dependencies = {711}; b.dependencies = {710};
    a.identity.dependency_set_hash = rd::dependency_set_hash(a.dependencies);
    b.identity.dependency_set_hash = rd::dependency_set_hash(b.dependencies);
    a.authorization_hash = rd::debt_authorization_hash(a);
    b.authorization_hash = rd::debt_authorization_hash(b);
    rd::RepairDebtScheduler cycle(ledger({a, b}));
    require(cycle.prepare(input(2, 0, 13)).reject == rd::Reject::InvalidLedger,
            "cyclic DAG accepted");
  }
  {
    auto expired = node(712, 7120, 1, 2);
    rd::RepairDebtScheduler deadline(ledger({expired}));
    auto p = deadline.prepare(input(3, 0, 14));
    require(p.accepted && deadline.ledger().records.at(712).drop_reason ==
                              rd::DropReason::DeadlineElapsed,
            "resource-bearing expired debt escaped deadline");
  }
}

void multi_actor_capacity_and_move_closure() {
  auto blocked = node(801, 8010, 3, 3, 0);
  blocked.value = 4; blocked.authorization_hash = rd::debt_authorization_hash(blocked);
  auto feasible = node(802, 8020, 3, 3, 1);
  feasible.value = 40; feasible.authorization_hash = rd::debt_authorization_hash(feasible);
  rd::RepairDebtScheduler separate(ledger({blocked, feasible}));
  auto in = input(3, 23, 101, {{810, 3, 0, 0, 1}}, 1,
                  {{2, 3}}, {0, 1});
  in.observation.actor = 1;
  in.resources.holdings[carried(1, 9)] = 1;
  in.resources.observation_hash = rd::observation_hash(in.observation);
  in.resources.content_hash = rd::resource_certificate_hash(in.resources);
  auto p = separate.prepare(in);
  require(p.accepted && p.action.kind == rd::ActionKind::Obligation &&
              p.action.debt_id == 802,
          "one actor's MOVE capacity deleted another actor's work");

  rd::RepairDebtScheduler closure(ledger({}));
  std::vector<rd::MoveToken> daily{{901, 6, 0, 0, 1}, {902, 6, 0, 1, 2},
                                   {903, 6, 1, 0, 3}, {904, 6, 1, 1, 0}};
  int physical = 201;
  for (int hour : {22, 23}) {
    for (int actor : {0, 1}) {
      auto tick = input(6, hour, physical, daily, 1, {{2, 3}}, {0, 1});
      tick.observation.actor = actor;
      tick.resources.holdings[carried(1, 9)] = 1;
      tick.resources.observation_hash = rd::observation_hash(tick.observation);
      tick.resources.physical_evidence_hash = physical;
      tick.resources.content_hash = rd::resource_certificate_hash(tick.resources);
      auto move = closure.prepare(tick);
      require(move.accepted && move.action.kind == rd::ActionKind::Move &&
                  commit(closure, tick, move).committed,
              "per-actor MOVE closure failed");
      ++physical;
    }
  }
  auto next = input(7, 0, physical, {}, 1, {{2, 3}}, {0, 1});
  require(closure.prepare(next).accepted && closure.metrics().moves_emitted == 4 &&
              closure.metrics().move_drop == 0,
          "midnight did not close every actor's MOVE set");
}

void jump_bound_and_production_chain() {
  auto waiting = node(901, 9010, 1, 20);
  rd::RepairDebtScheduler jump(ledger({waiting}));
  auto first = input(1, 0, 301, {}, 0);
  first.resources.certified_available_day[carried(0, 9)] = 20;
  first.resources.content_hash = rd::resource_certificate_hash(first.resources);
  auto p1 = jump.prepare(first); require(commit(jump, first, p1).committed,
                                         "pre-jump finalize failed");
  auto fourth = input(4, 0, 301, {}, 0);
  fourth.resources.certified_available_day[carried(0, 9)] = 20;
  fourth.resources.content_hash = rd::resource_certificate_hash(fourth.resources);
  auto p4 = jump.prepare(fourth);
  require(p4.accepted && jump.ledger().records.at(901).rebases == 3,
          "arbitrary multi-day jump not rebased");

  auto bounded_node = node(902, 9020, 1, 20);
  rd::RepairDebtScheduler bounded(ledger({bounded_node}));
  int physical = 401;
  for (int day = 1; day <= 7 && !bounded.fail_stopped(); ++day) {
    auto tick = input(day, 0, physical, {}, 0);
    tick.resources.certified_available_day[carried(0, 9)] = 20;
    tick.resources.content_hash = rd::resource_certificate_hash(tick.resources);
    auto proposal = bounded.prepare(tick);
    require(proposal.accepted, "bounded ledger prepare failed");
    require(commit(bounded, tick, proposal).committed, "bounded ledger commit failed");
  }
  require(bounded.ledger().records.at(902).state == rd::DebtState::DroppedInfeasible,
          "debt carried beyond bounded age");

  auto producer = node(903, 9030, 9, 9); producer.resources.clear();
  producer.produces = {{carried(0, 5), 1}};
  producer.authorization_hash = rd::debt_authorization_hash(producer);
  auto consumer = node(904, 9040, 9, 9);
  consumer.resources = {{carried(0, 5), 1}};
  consumer.dependencies = {903};
  consumer.identity.dependency_set_hash = rd::dependency_set_hash(consumer.dependencies);
  consumer.authorization_hash = rd::debt_authorization_hash(consumer);
  rd::RepairDebtScheduler chain(ledger({producer, consumer}));
  auto before = input(9, 0, 501, {}, 0);
  auto produce = chain.prepare(before);
  require(produce.action.debt_id == 903 && commit(chain, before, produce).committed,
          "resource producer not scheduled first");
  auto after = input(9, 1, 502, {}, 0);
  after.resources.holdings[carried(0, 5)] = 1;
  after.resources.content_hash = rd::resource_certificate_hash(after.resources);
  auto consume = chain.prepare(after);
  require(consume.action.debt_id == 904 && commit(chain, after, consume).committed,
          "certified produced resource did not close successor");
}

void black_box_regression() {
  // The case number is fixture metadata only. All scheduling inputs flow
  // through the same generic API and no core branch can observe it.
  constexpr std::uint64_t black_box_case = 970017;
  auto prerequisite = node(601, black_box_case ^ 0x111, 14, 16, 0, 6, 3);
  prerequisite.resources = {{carried(0, 9), 1}};
  prerequisite.authorization_hash = rd::debt_authorization_hash(prerequisite);
  auto leaf = node(602, black_box_case ^ 0x222, 14, 16, 0, 6, 3);
  leaf.dependencies = {601};
  leaf.identity.dependency_set_hash = rd::dependency_set_hash(leaf.dependencies);
  leaf.authorization_hash = rd::debt_authorization_hash(leaf);
  rd::RepairDebtScheduler s(ledger({prerequisite, leaf}));
  auto in0 = input(14, 0, black_box_case, {}, 1, {{6, 3}});
  auto p0 = s.prepare(in0);
  require(p0.action.debt_id == 601 && commit(s, in0, p0).committed,
          "black-box prerequisite failed");
  auto in1 = input(14, 1, black_box_case + 1, {}, 1, {{6, 3}});
  auto p1 = s.prepare(in1);
  require(p1.action.debt_id == 602 && commit(s, in1, p1).committed,
          "black-box dependent failed");
}

}  // namespace

int main() {
  try {
    translation_and_permutation();
    multi_day_and_moves();
    capacity_min_loss_drop();
    causal_drop_has_no_ghost();
    dominated_and_terminal();
    hard_receipt_and_move_fail_closed();
    typed_action_rollback_and_scopes();
    typed_validation_properties();
    multi_actor_capacity_and_move_closure();
    jump_bound_and_production_chain();
    black_box_regression();
    std::cout << "PASS repair_debt_scheduler properties\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
