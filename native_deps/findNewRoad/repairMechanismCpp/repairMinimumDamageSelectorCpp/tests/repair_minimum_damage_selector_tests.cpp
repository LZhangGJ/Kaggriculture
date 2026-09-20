#include "repair_minimum_damage_selector.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace md = g001::minimum_damage;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

md::Request base_request(int horizon, int width = 2, int actors = 1) {
  md::Request request;
  request.horizon = horizon;
  request.maximum_states = 500'000;
  request.day_start.width = width;
  request.day_start.height = 1;
  request.day_start.tiles.assign(static_cast<std::size_t>(width),
                                 md::TileKind::Empty);
  request.day_start.actor_positions.assign(static_cast<std::size_t>(actors),
                                           md::Position{});
  request.day_start.seed_inventory.assign(2, 0);
  request.day_start.animal_inventory.assign(2, 0);
  request.raw_sources.assign(
      static_cast<std::size_t>(actors),
      std::vector<md::SourceSlot>(static_cast<std::size_t>(horizon)));
  return request;
}

void set_deadline(md::TypedObligation& obligation, int horizon) {
  obligation.deadline_slot = horizon - 1;
}

const md::DayBoundDebt* debt(const md::PlanResult& plan, std::uint64_t id) {
  for (const auto& value : plan.debts)
    if (value.obligation_id == id) return &value;
  return nullptr;
}

void weed_recovery_delays_but_never_drops_moves() {
  auto request = base_request(4);
  request.day_start.tiles[0] = md::TileKind::Weed;
  request.raw_sources[0][0].action = {md::ActionKind::East, -1, 1};
  request.raw_sources[0][2].action = {md::ActionKind::West, -1, 1};
  auto weed = md::make_weed_pasture_obligation(
      1, 0, {0, 0}, 0, true, true, 9);
  set_deadline(weed, request.horizon);
  request.repair_obligations.push_back(weed);

  const auto plan = md::select_minimum_damage(request);
  require(plan.planned(), "weed+MOVE plan was not produced");
  require(plan.debts.empty(), "weed must-finish work became debt");
  require(plan.objective.move_slot_displacement == 1,
          "minimum MOVE displacement changed: " +
              std::to_string(plan.objective.move_slot_displacement));
  require(plan.slots[0].units[0].action.kind == md::ActionKind::Dig &&
              plan.slots[1].units[0].raw_source_slot == 0 &&
              plan.slots[2].units[0].raw_source_slot == 2 &&
              plan.slots[3].units[0].action.kind ==
                  md::ActionKind::BuildPasture,
          "weed recovery did not preserve ordered MOVE identities");
  const auto verified = md::verify_schedule(request, plan);
  require(verified.valid && verified.checked_move_tokens == 2,
          "weed schedule certificate failed");
}

void seed_purchase_and_replant_share_one_causal_day_model() {
  auto request = base_request(4, 1);
  request.day_start.tiles[0] = md::TileKind::Weed;
  request.day_start.cash = 20;
  auto purchase = md::make_seed_purchase_retry(
      10, 0, 1, 0, 7, true, true, true, 4);
  auto replant = md::make_replant_obligation(
      11, 0, {0, 0}, 0, true, 0, true, true, 8);
  replant.dependencies = {10};
  set_deadline(purchase, request.horizon);
  set_deadline(replant, request.horizon);
  request.repair_obligations = {purchase, replant};

  const auto plan = md::select_minimum_damage(request);
  require(plan.planned() && plan.debts.empty(),
          "guaranteed purchase/replant did not close");
  int dig_slot = -1;
  int buy_slot = -1;
  int plant_slot = -1;
  for (int slot = 0; slot < request.horizon; ++slot) {
    if (plan.slots[static_cast<std::size_t>(slot)].units[0].action.kind ==
        md::ActionKind::Dig)
      dig_slot = slot;
    if (plan.slots[static_cast<std::size_t>(slot)].units[0].action.kind ==
        md::ActionKind::Plant)
      plant_slot = slot;
    if (plan.slots[static_cast<std::size_t>(slot)].market &&
        plan.slots[static_cast<std::size_t>(slot)].market->action.kind ==
            md::ActionKind::BuySeed)
      buy_slot = slot;
  }
  require(dig_slot >= 0 && buy_slot >= 0 && plant_slot > buy_slot,
          "market fill was not constrained to later unit use");
  require(md::verify_schedule(request, plan).valid,
          "purchase/replant verification failed");

  request.repair_obligations[0].transitions[0].guaranteed_fill = false;
  const auto uncertain = md::select_minimum_damage(request);
  const auto* purchase_debt = debt(uncertain, 10);
  const auto* replant_debt = debt(uncertain, 11);
  require(uncertain.planned() && uncertain.debts.size() == 2 &&
              purchase_debt && replant_debt &&
              purchase_debt->reason == md::DebtReason::ResourceUnavailable &&
              replant_debt->reason == md::DebtReason::ResourceUnavailable,
          std::string("non-guaranteed future market result was treated as evidence: ") +
              (purchase_debt ? md::debt_reason_name(purchase_debt->reason)
                             : "missing") +
              "/" +
              (replant_debt ? md::debt_reason_name(replant_debt->reason)
                            : "missing"));

  request.repair_obligations[0].transitions[0].guaranteed_fill = true;
  request.day_start.cash = 0;
  const auto cash_short = md::select_minimum_damage(request);
  require(cash_short.planned() &&
              debt(cash_short, 10)->reason ==
                  md::DebtReason::ResourceUnavailable &&
              debt(cash_short, 11)->reason ==
                  md::DebtReason::ResourceUnavailable,
          "worst-case cash shortage was classified as slot capacity");
}

void animal_purchase_place_feed_and_care_are_typed() {
  auto request = base_request(5, 1);
  request.day_start.tiles[0] = md::TileKind::Pasture;
  request.day_start.cash = 30;
  request.day_start.feed_inventory = 1;
  auto purchase = md::make_animal_purchase_retry(
      20, 1, 1, 0, 12, true, true, true, 5);
  auto animal = md::make_animal_recovery_obligation(
      21, 0, {0, 0}, 1, false, true, true, 0, true, true, 7, {20});
  set_deadline(purchase, request.horizon);
  set_deadline(animal, request.horizon);
  request.repair_obligations = {purchase, animal};

  const auto plan = md::select_minimum_damage(request);
  require(plan.planned() && plan.debts.empty(),
          "animal recovery chain did not close");
  require(plan.slots[0].market &&
              plan.slots[0].market->action.kind ==
                  md::ActionKind::BuyAnimal &&
              plan.slots[1].units[0].action.kind ==
                  md::ActionKind::PlaceAnimal &&
              plan.slots[2].units[0].action.kind == md::ActionKind::Feed &&
              plan.slots[3].units[0].action.kind == md::ActionKind::Care,
          "animal typed transition order changed");
  require(md::verify_schedule(request, plan).valid,
          "animal schedule verification failed");
}

void explicit_day_bound_debt_beats_move_loss() {
  auto request = base_request(2);
  request.day_start.tiles[0] = md::TileKind::Weed;
  request.raw_sources[0][0].action = {md::ActionKind::East, -1, 1};
  request.raw_sources[0][1].action = {md::ActionKind::West, -1, 1};
  auto weed = md::make_weed_pasture_obligation(
      30, 0, {0, 0}, 0, true, true, 11);
  set_deadline(weed, request.horizon);
  request.repair_obligations.push_back(weed);

  const auto plan = md::select_minimum_damage(request);
  require(plan.planned() && plan.debts.size() == 1 &&
              plan.objective.unfinished_must_finish == 1 &&
              plan.slots[0].units[0].raw_source_slot == 0 &&
              plan.slots[1].units[0].raw_source_slot == 1,
          "capacity shortage swallowed a hard MOVE");
  const auto* value = debt(plan, 30);
  require(value && value->reason == md::DebtReason::Capacity &&
              value->bound_day_offset == 1,
          "capacity shortage was not explicit day-bound debt");
  require(md::verify_schedule(request, plan).valid,
          "debt-bearing MOVE schedule did not verify");
}

void objective_is_lexicographic_not_scalar_reward() {
  auto request = base_request(1, 1);
  request.day_start.tiles[0] = md::TileKind::Weed;
  md::TypedObligation must_feed{
      40, md::ObligationKind::WeedPastureRecovery, 0, 0, 0, -1, true,
      false, 1, {},
      {{md::Channel::Unit, {md::ActionKind::Dig, -1, 1}, true, {0, 0},
        false, 0}}};
  md::TypedObligation high_risk_care{
      41, md::ObligationKind::WeedPastureRecovery, 0, 0, 0, -1, false,
      true, 100, {},
      {{md::Channel::Unit, {md::ActionKind::Dig, -1, 1}, true, {0, 0},
        false, 0}}};
  request.repair_obligations = {must_feed, high_risk_care};
  const auto plan = md::select_minimum_damage(request);
  require(plan.planned() && plan.slots[0].units[0].obligation_id == 40 &&
              debt(plan, 41) && !debt(plan, 40),
          "cascade scalar overrode must-finish lexicographic priority");
}

void invalid_hard_route_fails_closed() {
  auto request = base_request(2, 1);
  request.raw_sources[0][0].action = {md::ActionKind::West, -1, 1};
  const auto plan = md::select_minimum_damage(request);
  require(plan.status == md::PlanStatus::HardMoveInfeasible &&
              plan.slots.empty(),
          "out-of-map immutable MOVE did not fail closed");
}

void raw_production_is_typed_and_verifier_rejects_move_tamper() {
  auto request = base_request(3);
  request.day_start.tiles[0] = md::TileKind::Animal;
  request.day_start.feed_inventory = 1;
  request.raw_sources[0][0] = {
      {md::ActionKind::Feed, -1, 1}, true, true, 4};
  request.raw_sources[0][1].action = {md::ActionKind::East, -1, 1};
  const auto plan = md::select_minimum_damage(request);
  require(plan.planned() && plan.debts.empty() &&
              plan.slots[0].units[0].obligation_id != 0 &&
              plan.slots[1].units[0].raw_source_slot == 1 &&
              md::verify_schedule(request, plan).valid,
          "raw production was not converted to a typed obligation");

  auto tampered = plan;
  tampered.slots[1].units[0].raw_source_slot = 0;
  require(!md::verify_schedule(request, tampered).valid,
          "verifier accepted a changed MOVE source identity");
}

void feed_purchase_retry_enables_later_care_without_future_fill() {
  auto request = base_request(3, 1);
  request.day_start.tiles[0] = md::TileKind::Animal;
  request.day_start.cash = 9;
  auto purchase = md::make_feed_purchase_retry(
      45, 1, 0, 5, true, true, true, 3);
  md::TypedObligation feed{
      46, md::ObligationKind::AnimalCareRecovery, 0, 0, 1, -1, true, true,
      4, {45},
      {{md::Channel::Unit, {md::ActionKind::Feed, -1, 1}, true, {0, 0},
        false, 0}}};
  set_deadline(purchase, request.horizon);
  request.repair_obligations = {purchase, feed};
  const auto plan = md::select_minimum_damage(request);
  require(plan.planned() && plan.debts.empty() && plan.slots[0].market &&
              plan.slots[1].units[0].action.kind == md::ActionKind::Feed &&
              md::verify_schedule(request, plan).valid,
          "guaranteed feed retry did not enable later feed transition");
}

void midnight_transient_receipts_become_explicit_debt() {
  for (const auto kind : {md::ActionKind::Water, md::ActionKind::Feed,
                          md::ActionKind::Care}) {
    auto request = base_request(1, 1);
    request.day_start.tiles[0] = kind == md::ActionKind::Water
                                     ? md::TileKind::Crop
                                     : md::TileKind::Animal;
    request.day_start.feed_inventory = 1;
    request.raw_sources[0][0] = {{kind, -1, 1}, true, true, 9};
    const auto plan = md::select_minimum_damage(request);
    require(plan.planned() && plan.debts.size() == 1 &&
                plan.slots[0].units[0].action.kind == md::ActionKind::Pass &&
                plan.debts[0].remaining_transitions == 1 &&
                plan.debts[0].bound_day_offset == 1 &&
                md::verify_schedule(request, plan).valid,
            std::string("midnight transient action lacked explicit debt: ") +
                md::action_kind_name(kind));
  }
}

void explicit_repair_detour_is_counted_without_impersonating_raw_move() {
  auto request = base_request(2);
  md::TypedObligation detour{
      47, md::ObligationKind::WeedPastureRecovery, 0, 0, 1, -1, true, true,
      2, {},
      {{md::Channel::Unit, {md::ActionKind::East, -1, 1}, false, {}, false,
        0},
       {md::Channel::Unit, {md::ActionKind::West, -1, 1}, false, {}, false,
        0}}};
  request.repair_obligations.push_back(detour);
  const auto plan = md::select_minimum_damage(request);
  const auto verified = md::verify_schedule(request, plan);
  require(plan.planned() && plan.debts.empty() &&
              plan.objective.inserted_move_distance == 2 && verified.valid &&
              verified.checked_move_tokens == 0 &&
              plan.slots[0].units[0].raw_source_slot == -1 &&
              plan.slots[0].units[0].obligation_id == 47,
          "typed detour was confused with an immutable raw MOVE token");
}

void memoized_dp_matches_small_exhaustive_oracle() {
  for (int mask = 0; mask < 16; ++mask) {
    auto request = base_request(4);
    request.day_start.tiles[0] = md::TileKind::Weed;
    if ((mask & 1) != 0)
      request.raw_sources[0][0].action = {md::ActionKind::East, -1, 1};
    if ((mask & 2) != 0)
      request.raw_sources[0][2].action = {md::ActionKind::West, -1, 1};
    if ((mask & 4) != 0) request.day_start.seed_inventory[0] = 1;
    auto obligation = (mask & 8) != 0
                          ? md::make_replant_obligation(
                                50, 0, {0, 0}, 0, true, 0, true, true, 6)
                          : md::make_weed_pasture_obligation(
                                50, 0, {0, 0}, 0, true, true, 6);
    set_deadline(obligation, request.horizon);
    request.repair_obligations.push_back(obligation);
    const auto dp = md::select_minimum_damage(request);
    const auto oracle = md::exhaustive_oracle_for_testing(request);
    require(dp.status == oracle.status && dp.objective == oracle.objective &&
                dp.debts.size() == oracle.debts.size(),
            "memoized DP diverged from exhaustive oracle mask=" +
                std::to_string(mask));
    if (dp.planned())
      require(md::verify_schedule(request, dp).valid,
              "oracle-comparison DP schedule failed verification");
  }
}

void print_single_core_24_slot_timing() {
  auto request = base_request(24, 3);
  request.maximum_states = 2'000'000;
  request.day_start.tiles[0] = md::TileKind::Weed;
  request.raw_sources[0][3].action = {md::ActionKind::East, -1, 1};
  request.raw_sources[0][7].action = {md::ActionKind::East, -1, 1};
  request.raw_sources[0][12].action = {md::ActionKind::West, -1, 1};
  request.raw_sources[0][17].action = {md::ActionKind::West, -1, 1};
  request.repair_obligations.push_back(md::make_weed_pasture_obligation(
      60, 0, {0, 0}, 0, true, true, 10));
  const auto start = std::chrono::steady_clock::now();
  const auto plan = md::select_minimum_damage(request);
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start);
  require(plan.planned() && plan.debts.empty() &&
              md::verify_schedule(request, plan).valid,
          "24-slot timing fixture failed");
  std::cout << "single_core_24_slot_us=" << elapsed.count()
            << " explored_states=" << plan.explored_states << '\n';
}

}  // namespace

int main() {
  try {
    weed_recovery_delays_but_never_drops_moves();
    seed_purchase_and_replant_share_one_causal_day_model();
    animal_purchase_place_feed_and_care_are_typed();
    explicit_day_bound_debt_beats_move_loss();
    objective_is_lexicographic_not_scalar_reward();
    invalid_hard_route_fails_closed();
    raw_production_is_typed_and_verifier_rejects_move_tamper();
    feed_purchase_retry_enables_later_care_without_future_fill();
    midnight_transient_receipts_become_explicit_debt();
    explicit_repair_detour_is_counted_without_impersonating_raw_move();
    memoized_dp_matches_small_exhaustive_oracle();
    print_single_core_24_slot_timing();
    std::cout << "repair_minimum_damage_selector_tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "repair_minimum_damage_selector_tests: " << error.what()
              << '\n';
    return 1;
  }
}
