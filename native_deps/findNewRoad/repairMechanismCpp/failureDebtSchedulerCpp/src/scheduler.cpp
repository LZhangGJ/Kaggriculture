#include "failure_debt_scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace g001::failure_debt {
namespace {

constexpr int kInfinity = std::numeric_limits<int>::max() / 8;

std::size_t op_index(fastkag::Op operation) {
  return static_cast<std::size_t>(operation);
}

int item_index(fastkag::Item item) { return static_cast<int>(item); }

bool same_position(fastkag::Position left, fastkag::Position right) {
  return left.x == right.x && left.y == right.y;
}

bool same_action_kind(const fastkag::Action& left,
                      const fastkag::Action& right) {
  return left.op == right.op && left.item == right.item &&
         std::max(0, left.quantity) >= std::max(0, right.quantity);
}

bool complete_chain_shape(const PurchaseIntent& intent) {
  for (int unit = 0; unit < intent.quantity; ++unit) {
    const auto& requirements =
        intent.chains[static_cast<std::size_t>(unit)].requirements;
    if (requirements.empty()) return false;
    for (const auto& requirement : requirements) {
      if (requirement.actor < 0 || is_movement(requirement.action.op) ||
          requirement.deadline < requirement.earliest_step) return false;
    }
    if (intent.operation == fastkag::Op::BUY_SEED) {
      if (std::none_of(requirements.begin(), requirements.end(),
                       [&](const UnitRequirement& requirement) {
                         return requirement.action.op == fastkag::Op::PLANT &&
                                requirement.action.item == intent.item;
                       })) return false;
      continue;
    }
    // A BUY_ANIMAL debt is accepted only with an explicit first lifecycle,
    // not just transport. Additional PICKUP(WHEAT) or other prerequisite
    // actions may appear between these ordered mandatory stages.
    int stage = 0;
    for (const auto& requirement : requirements) {
      const auto operation = requirement.action.op;
      if (stage == 0 && operation == fastkag::Op::PICKUP &&
          requirement.action.item == intent.item) ++stage;
      else if (stage == 1 && operation == fastkag::Op::PLACE &&
               requirement.action.item == intent.item) ++stage;
      else if (stage == 2 && operation == fastkag::Op::FEED) ++stage;
      else if (stage == 3 && operation == fastkag::Op::CARE) ++stage;
      else if (stage == 4 && operation == fastkag::Op::HARVEST) ++stage;
    }
    if (stage != 5) return false;
  }
  return true;
}

std::string escape_json(const std::string& text) {
  std::string result;
  result.reserve(text.size() + 8);
  for (const char value : text) {
    switch (value) {
      case '\\': result += "\\\\"; break;
      case '"': result += "\\\""; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default: result.push_back(value); break;
    }
  }
  return result;
}

std::uint64_t observation_fingerprint(const OwnObservation& observation) {
  std::uint64_t hash = 1469598103934665603ULL;
  auto add = [&](int value) {
    hash ^= static_cast<std::uint32_t>(value);
    hash *= 1099511628211ULL;
  };
  add(observation.step); add(observation.money); add(observation.protected_cash);
  add(observation.shed_used); add(observation.shed_capacity);
  add(observation.market_slots_used); add(observation.maximum_market_slots);
  add(observation.turns_per_day);
  for (const int value : observation.seeds) add(value);
  for (const int value : observation.animals) add(value);
  for (const int value : observation.shed_items) add(value);
  add(static_cast<int>(observation.actor_inventory.size()));
  for (const auto& inventory : observation.actor_inventory)
    for (const int value : inventory) add(value);
  return hash;
}

struct PurchaseCandidate {
  int step{};
  bool existing{};
  int existing_slot{-1};
  int marginal_cost{};
};

struct FlatSlot {
  int turn_index{};
  int slot_index{};
  int step{};
  const PlannedUnitSlot* slot{};
};

struct Choice {
  int flat_index{};
  bool existing{};
  int cost{};
};

struct Assignment {
  bool feasible{};
  int absorbed{};
  int score{kInfinity};
  std::vector<Choice> choices;
};

struct HistoricUnitEvent {
  fastkag::Action action{};
  fastkag::Position position{};
  int step{};
  int actor{};
  int chain_id{};
};

struct CropLifecycleDef {
  int first_day;
  int max_day;
  int interval;
  int max_yield;
  bool ongoing;
};

constexpr std::array<CropLifecycleDef, fastkag::N_CROPS> kCropLifecycle{{
    {2, 4, 0, 6, false}, {2, 3, 0, 4, false},
    {8, 8, 1, 4, true}, {10, 10, 2, 4, true},
    {10, 12, 0, 6, false},
}};

struct AnimalLifecycleDef {
  int first_day;
  int interval;
  int max_held;
};

constexpr std::array<AnimalLifecycleDef, fastkag::N_ANIMALS> kAnimalLifecycle{{
    {4, 1, 4}, {8, 2, 6}, {6, 3, 6},
}};

bool compatible_replacement(const PlannedUnitSlot& slot,
                            const UnitRequirement& requirement) {
  if (is_movement(requirement.action.op) || !slot.absorbable || slot.critical ||
      is_movement(slot.original.op) ||
      slot.economic_value > requirement.maximum_absorbed_value ||
      slot.actor != requirement.actor ||
      !same_position(slot.position, requirement.position)) {
    return false;
  }
  const auto index = op_index(requirement.action.op);
  return index < slot.legal_replacements.size() &&
         slot.legal_replacements[index];
}

Assignment assign_requirements(const std::vector<UnitRequirement>& requirements,
                               const std::vector<int>& chain_ids,
                               const std::vector<FlatSlot>& slots,
                               const std::vector<int>& acquisition_steps) {
  if (requirements.empty()) return Assignment{true, 0, 0, {}};
  if (chain_ids.size() != requirements.size() ||
      acquisition_steps.size() != requirements.size()) return {};
  std::vector<std::vector<Choice>> choices(requirements.size());
  for (std::size_t requirement_index = 0;
       requirement_index < requirements.size(); ++requirement_index) {
    const auto& requirement = requirements[requirement_index];
    const int earliest = std::max(requirement.earliest_step,
                                  acquisition_steps[requirement_index] + 1);
    for (int flat_index = 0; flat_index < static_cast<int>(slots.size());
         ++flat_index) {
      const auto& flat = slots[static_cast<std::size_t>(flat_index)];
      const auto& slot = *flat.slot;
      if (flat.step < earliest || flat.step > requirement.deadline ||
          slot.actor != requirement.actor ||
          !same_position(slot.position, requirement.position)) {
        continue;
      }
      if (requirement.allow_existing_equivalent && slot.expected_success &&
          same_action_kind(slot.original, requirement.action)) {
        choices[requirement_index].push_back({flat_index, true, 0});
      } else if (compatible_replacement(slot, requirement)) {
        choices[requirement_index].push_back(
            {flat_index, false, std::max(0, slot.economic_value)});
      }
    }
    if (choices[requirement_index].empty()) return {};
  }

  for (std::size_t requirement = 0; requirement < choices.size(); ++requirement) {
    std::sort(choices[requirement].begin(), choices[requirement].end(),
              [&](const Choice& left, const Choice& right) {
      const auto score = [&](const Choice& choice) {
        const int delay = slots[static_cast<std::size_t>(choice.flat_index)].step -
            std::max(requirements[requirement].earliest_step,
                     acquisition_steps[requirement] + 1);
        return choice.cost * 100000 + std::max(0, delay);
      };
      return std::tuple{score(left), left.flat_index} <
             std::tuple{score(right), right.flat_index};
    });
  }

  std::vector<int> unique_chains;
  for (const int chain : chain_ids) {
    if (std::find(unique_chains.begin(), unique_chains.end(), chain) ==
        unique_chains.end()) unique_chains.push_back(chain);
  }
  std::sort(unique_chains.begin(), unique_chains.end(), [&](int left, int right) {
    const auto count = [&](int chain) {
      std::size_t total = 0;
      for (std::size_t i = 0; i < chain_ids.size(); ++i)
        if (chain_ids[i] == chain) total += choices[i].size();
      return total;
    };
    return count(left) < count(right);
  });

  std::vector<std::vector<int>> requirements_by_chain(unique_chains.size());
  for (std::size_t requirement = 0; requirement < chain_ids.size(); ++requirement) {
    const auto found = std::find(unique_chains.begin(), unique_chains.end(),
                                 chain_ids[requirement]);
    requirements_by_chain[static_cast<std::size_t>(found - unique_chains.begin())]
        .push_back(static_cast<int>(requirement));
  }

  std::vector<bool> used(slots.size(), false);
  std::vector<Choice> current(requirements.size());
  Assignment best;
  std::uint64_t expanded = 0;
  constexpr std::uint64_t kMaximumExpanded = 1'000'000;
  const auto local_score = [&](int requirement, const Choice& choice) {
    const int delay = slots[static_cast<std::size_t>(choice.flat_index)].step -
        std::max(requirements[static_cast<std::size_t>(requirement)].earliest_step,
                 acquisition_steps[static_cast<std::size_t>(requirement)] + 1);
    return choice.cost * 100000 + std::max(0, delay);
  };

  auto search = [&](auto&& self, std::size_t chain_position,
                    std::size_t within_chain, int previous_step,
                    int score, int absorbed) -> void {
    if (++expanded > kMaximumExpanded || score >= best.score) return;
    if (chain_position == requirements_by_chain.size()) {
      best.feasible = true;
      best.score = score;
      best.absorbed = absorbed;
      best.choices = current;
      return;
    }
    const auto& chain = requirements_by_chain[chain_position];
    if (within_chain == chain.size()) {
      self(self, chain_position + 1, 0, -1, score, absorbed);
      return;
    }
    const int requirement = chain[within_chain];
    for (const auto& choice : choices[static_cast<std::size_t>(requirement)]) {
      const auto& flat = slots[static_cast<std::size_t>(choice.flat_index)];
      // Causality is per purchased unit only. Independent animal/crop chains
      // may interleave or use different actors in the same turn.
      if (used[static_cast<std::size_t>(choice.flat_index)] ||
          flat.step <= previous_step) continue;
      used[static_cast<std::size_t>(choice.flat_index)] = true;
      current[static_cast<std::size_t>(requirement)] = choice;
      self(self, chain_position, within_chain + 1, flat.step,
           score + local_score(requirement, choice), absorbed + choice.cost);
      used[static_cast<std::size_t>(choice.flat_index)] = false;
    }
  };
  search(search, 0, 0, -1, 0, 0);
  return best;
}

bool validate_day_lifecycle(const OwnObservation& observation,
                            const PlanWindow& window,
                            const std::vector<UnitRequirement>& requirements,
                            const std::vector<int>& chain_ids,
                            const std::vector<FlatSlot>& slots,
                            const Assignment& assignment,
                            const std::vector<HistoricUnitEvent>& history,
                            std::string& reason) {
  const int turns_per_day = std::max(1, observation.turns_per_day);
  const int observed_day = observation.step / turns_per_day;
  std::vector<int> validated_chains;
  for (const int chain_id : chain_ids) {
    if (std::find(validated_chains.begin(), validated_chains.end(), chain_id) !=
        validated_chains.end()) continue;
    validated_chains.push_back(chain_id);
    const auto proof = std::find_if(window.day_lifecycles.begin(),
        window.day_lifecycles.end(), [&](const DayLifecycleSpec& candidate) {
          return candidate.purchase_unit_index == chain_id;
        });
    if (proof == window.day_lifecycles.end()) {
      reason = "missing_day_lifecycle_proof";
      return false;
    }
    if (proof->tile_observed_step != observation.step ||
        proof->required_harvest_cycles <= 0) {
      reason = "stale_or_invalid_tile_lifecycle_proof";
      return false;
    }

    struct Event {
      fastkag::Action action;
      fastkag::Position position;
      int step;
      int actor;
    };
    std::vector<Event> events;
    for (const auto& historic : history) {
      if (historic.chain_id == chain_id)
        events.push_back({historic.action, historic.position, historic.step,
                          historic.actor});
    }
    for (std::size_t index = 0; index < chain_ids.size(); ++index) {
      if (chain_ids[index] != chain_id) continue;
      const auto& choice = assignment.choices[index];
      const auto& flat = slots[static_cast<std::size_t>(choice.flat_index)];
      const auto& turn = window.turns[static_cast<std::size_t>(flat.turn_index)];
      if (requirements[index].actor > 0 &&
          (!turn.actor_availability_certified ||
           turn.certified_actor_count <= requirements[index].actor)) {
        reason = "future_hand_not_rehired_or_day_scoped_actor_uncertified";
        return false;
      }
      events.push_back({requirements[index].action, requirements[index].position,
                        flat.step,
                        requirements[index].actor});
    }
    std::sort(events.begin(), events.end(),
              [](const Event& left, const Event& right) {
                return left.step < right.step;
              });
    const auto steps_for = [&](fastkag::Op operation) {
      std::vector<int> result;
      for (const auto& event : events)
        if (event.action.op == operation &&
            same_position(event.position, proof->tile)) result.push_back(event.step);
      return result;
    };
    const auto has_action_on_day = [&](fastkag::Op operation, int day) {
      return std::any_of(events.begin(), events.end(), [&](const Event& event) {
        return event.action.op == operation &&
               same_position(event.position, proof->tile) &&
               event.step / turns_per_day == day;
      });
    };

    if (proof->kind == LifecycleKind::Crop) {
      const int crop = item_index(proof->item);
      if (crop < 0 || crop >= fastkag::N_CROPS ||
          proof->observed_kind != fastkag::TileKind::EMPTY) {
        reason = "crop_repair_requires_fresh_observed_empty_tile";
        return false;
      }
      const auto plants = steps_for(fastkag::Op::PLANT);
      const auto harvests = steps_for(fastkag::Op::HARVEST);
      if (plants.size() != 1 ||
          events[static_cast<std::size_t>(std::find_if(events.begin(), events.end(),
              [](const Event& event) { return event.action.op == fastkag::Op::PLANT; }) -
              events.begin())].action.item != proof->item) {
        reason = "crop_lifecycle_missing_exact_plant";
        return false;
      }
      const int plant_step = plants.front();
      const int plant_day = plant_step / turns_per_day;
      if (plant_day > observed_day) {
        const bool has_dig = std::any_of(events.begin(), events.end(),
            [&](const Event& event) {
            return event.action.op == fastkag::Op::DIG &&
                     same_position(event.position, proof->tile) &&
                     event.step < plant_step &&
                     event.step / turns_per_day == plant_day;
            });
        if (!proof->future_empty_tile_weed_risk_resolved || !has_dig) {
          reason = "future_empty_tile_weed_spawn_uncertified";
          return false;
        }
      }
      const auto& definition = kCropLifecycle[static_cast<std::size_t>(crop)];
      if ((!definition.ongoing && proof->required_harvest_cycles != 1) ||
          (definition.ongoing &&
           proof->required_harvest_cycles > definition.max_yield) ||
          harvests.size() < static_cast<std::size_t>(proof->required_harvest_cycles)) {
        reason = "crop_harvest_cycle_count_invalid";
        return false;
      }
      int final_harvest_step = -1;
      for (int cycle = 0; cycle < proof->required_harvest_cycles; ++cycle) {
        const int harvest_step = harvests[static_cast<std::size_t>(cycle)];
        const int harvest_day = harvest_step / turns_per_day;
        if (!definition.ongoing) {
          if (harvest_day < plant_day + definition.first_day ||
              harvest_day > plant_day + definition.max_day) {
            reason = "nonongoing_crop_harvest_outside_recomputed_maturity_decay_window";
            return false;
          }
        } else {
          const int maturity = plant_day + definition.first_day +
                               cycle * definition.interval;
          const int latest = maturity + std::max(1, definition.interval) - 1;
          if (harvest_day < maturity || harvest_day > latest) {
            reason = "ongoing_crop_harvest_cycle_not_recomputed_from_plant_day";
            return false;
          }
        }
        final_harvest_step = harvest_step;
      }
      const int last_water_day = final_harvest_step / turns_per_day - 1;
      for (int day = plant_day; day <= last_water_day; ++day) {
        if (!has_action_on_day(fastkag::Op::WATER, day)) {
          reason = day == plant_day ? "same_day_plant_missing_water_before_day_end"
                                    : "crop_lifecycle_has_unwatered_day_bucket";
          return false;
        }
      }
      const bool same_day_water_after_plant = std::any_of(
          events.begin(), events.end(), [&](const Event& event) {
            return event.action.op == fastkag::Op::WATER &&
                   same_position(event.position, proof->tile) &&
                   event.step > plant_step &&
                   event.step / turns_per_day == plant_day;
          });
      if (!same_day_water_after_plant) {
        reason = "same_day_plant_missing_water_before_day_end";
        return false;
      }
      if (proof->next_occupancy_step >= 0) {
        if (definition.ongoing || proof->next_occupancy_step <= final_harvest_step) {
          reason = "crop_tile_next_occupancy_conflicts_with_rebuilt_lifecycle";
          return false;
        }
      }
    } else {
      const int animal = item_index(proof->item) - 9;
      if (animal < 0 || animal >= fastkag::N_ANIMALS ||
          (proof->observed_kind != fastkag::TileKind::COOP &&
           proof->observed_kind != fastkag::TileKind::PASTURE)) {
        reason = "animal_repair_requires_observed_empty_structure";
        return false;
      }
      const auto places = steps_for(fastkag::Op::PLACE);
      const auto harvests = steps_for(fastkag::Op::HARVEST);
      if (places.size() != 1 ||
          harvests.size() < static_cast<std::size_t>(proof->required_harvest_cycles)) {
        reason = "animal_lifecycle_missing_place_or_harvest_cycle";
        return false;
      }
      const int place_day = places.front() / turns_per_day;
      const auto& definition = kAnimalLifecycle[static_cast<std::size_t>(animal)];
      if (proof->required_harvest_cycles > definition.max_held) {
        reason = "animal_harvest_cycle_count_exceeds_held_yield";
        return false;
      }
      int final_harvest_step = -1;
      for (int cycle = 0; cycle < proof->required_harvest_cycles; ++cycle) {
        const int expected_day = place_day + definition.first_day +
                                 cycle * definition.interval;
        const int harvest_day = harvests[static_cast<std::size_t>(cycle)] /
                                turns_per_day;
        if (harvest_day < expected_day ||
            harvest_day > expected_day + definition.interval - 1) {
          reason = "animal_yield_harvest_not_recomputed_from_placed_day";
          return false;
        }
        final_harvest_step = harvests[static_cast<std::size_t>(cycle)];
      }
      int cared_and_fed_days = 0;
      for (int day = place_day;
           day < final_harvest_step / turns_per_day; ++day) {
        const bool fed = has_action_on_day(fastkag::Op::FEED, day);
        const bool cared = has_action_on_day(fastkag::Op::CARE, day);
        if (!fed) {
          reason = "animal_lifecycle_feed_gap_risks_death_or_lost_yield";
          return false;
        }
        if (fed && cared) ++cared_and_fed_days;
      }
      if (cared_and_fed_days < proof->required_care_days) {
        reason = "animal_care_bonus_not_backed_by_same_day_feed_and_care";
        return false;
      }
      if (proof->next_occupancy_step >= 0) {
        reason = "animal_structure_next_occupancy_conflicts_with_live_animal";
        return false;
      }
    }
  }
  return true;
}

int market_cost(const PlannedMarketOrder& order) {
  const int quantity = std::max(0, order.action.quantity);
  return quantity * purchase_unit_cost(order.action.op, order.action.item,
                                       order.conservative_unit_cost);
}

const UnitPatch* patch_for(const std::vector<UnitPatch>& patches, int step,
                           int source_slot) {
  const auto found = std::find_if(patches.begin(), patches.end(),
      [&](const UnitPatch& patch) {
        return patch.step == step && patch.source_slot == source_slot;
      });
  return found == patches.end() ? nullptr : &*found;
}

bool validate_resources(const OwnObservation& observation,
                        const PlanWindow& window,
                        const PurchaseCandidate& purchase,
                        const fastkag::Action& repair_order,
                        const std::vector<UnitPatch>& patches,
                        std::string& reason) {
  int cash = observation.money;
  int shed = observation.shed_used;
  auto seeds = observation.seeds;
  auto shed_items = observation.shed_items;
  auto carried = observation.actor_inventory;
  if (carried.empty()) carried.resize(1);
  // `available_items` is retained for observation adapters, but exact unit
  // certification never credits it to an actor. FEED/PLACE require the item
  // in that actor's carried inventory after an explicit PICKUP.
  const int permanent_reserve = observation.protected_cash +
                                window.future_hard_purchase_reserve;
  int projected_day = observation.step / std::max(1, observation.turns_per_day);

  for (const auto& turn : window.turns) {
    const int turn_day = turn.step / std::max(1, observation.turns_per_day);
    while (projected_day < turn_day) {
      for (auto& inventory : carried) {
        for (int item = 0; item < fastkag::N_ITEMS; ++item) {
          const int quantity = inventory[static_cast<std::size_t>(item)];
          if (quantity <= 0) continue;
          if (shed + quantity > observation.shed_capacity) {
            reason = "end_of_day_inventory_would_overflow";
            return false;
          }
          shed += quantity;
          shed_items[static_cast<std::size_t>(item)] += quantity;
          inventory[static_cast<std::size_t>(item)] = 0;
        }
      }
      carried.resize(1);
      ++projected_day;
    }
    int required_actor_count = 1;
    for (const auto& slot : turn.units) {
      required_actor_count = std::max(required_actor_count, slot.actor + 1);
    }
    if (static_cast<int>(carried.size()) < required_actor_count)
      carried.resize(static_cast<std::size_t>(required_actor_count));

    cash += std::max(0, turn.guaranteed_cash_income_before_market);
    shed += turn.guaranteed_shed_delta_before_market;
    if (shed < 0 || shed > observation.shed_capacity) {
      reason = "baseline_or_patch_shed_projection_invalid";
      return false;
    }

    std::array<int, fastkag::N_CROPS> plant_demand{};
    for (int slot_index = 0; slot_index < static_cast<int>(turn.units.size());
         ++slot_index) {
      const auto& slot = turn.units[static_cast<std::size_t>(slot_index)];
      const auto* patch = patch_for(patches, turn.step, slot_index);
      if (!patch && !slot.expected_success) continue;
      const auto& action = patch ? patch->replacement : slot.original;
      const int item = item_index(action.item);
      if (action.op == fastkag::Op::PLANT && item >= 0 &&
          item < fastkag::N_CROPS)
        ++plant_demand[static_cast<std::size_t>(item)];
    }
    for (int crop = 0; crop < fastkag::N_CROPS; ++crop) {
      if (plant_demand[static_cast<std::size_t>(crop)] >
          seeds[static_cast<std::size_t>(crop)]) {
        reason = "same_tick_crop_plant_batch_underfunded_all_actions_would_pass";
        return false;
      }
    }

    for (int slot_index = 0; slot_index < static_cast<int>(turn.units.size());
         ++slot_index) {
      const auto& slot = turn.units[static_cast<std::size_t>(slot_index)];
      const auto* patch = patch_for(patches, turn.step, slot_index);
      if (!patch && !slot.expected_success) continue;
      const auto& action = patch ? patch->replacement : slot.original;
      const int item = item_index(action.item);
      const int quantity = std::max(1, int(action.quantity));
      if (slot.actor < 0 || slot.actor >= static_cast<int>(carried.size())) {
        reason = "unit_actor_inventory_unavailable";
        return false;
      }
      auto& inventory = carried[static_cast<std::size_t>(slot.actor)];
      if (action.op == fastkag::Op::PLANT && item >= 0 &&
          item < fastkag::N_CROPS) {
        if (--seeds[static_cast<std::size_t>(item)] < 0) {
          reason = "seed_chain_not_funded_before_plant";
          return false;
        }
      } else if (action.op == fastkag::Op::FEED) {
        if (--inventory[static_cast<std::size_t>(fastkag::Item::WHEAT)] < 0) {
          reason = "animal_chain_missing_feed_wheat";
          return false;
        }
      } else if (action.op == fastkag::Op::FERTILIZE) {
        if (--inventory[static_cast<std::size_t>(fastkag::Item::FERTILIZER)] < 0) {
          reason = "fertilize_chain_missing_carried_fertilizer";
          return false;
        }
      } else if (action.op == fastkag::Op::PICKUP && item >= 0 &&
                 item < fastkag::N_ITEMS) {
        if (shed_items[static_cast<std::size_t>(item)] < quantity) {
          reason = "pickup_not_backed_by_item_specific_shed_inventory";
          return false;
        }
        shed_items[static_cast<std::size_t>(item)] -= quantity;
        inventory[static_cast<std::size_t>(item)] += quantity;
        shed -= quantity;
        if (shed < 0) {
          reason = "pickup_not_backed_by_shed_inventory";
          return false;
        }
      } else if (action.op == fastkag::Op::PLACE && item >= 9 && item < 12) {
        if (--inventory[static_cast<std::size_t>(item)] < 0) {
          reason = "animal_place_not_backed_by_carried_animal";
          return false;
        }
      } else if (action.op == fastkag::Op::DROP) {
        for (int carried_item = 0; carried_item < fastkag::N_ITEMS;
             ++carried_item) {
          const int amount = inventory[static_cast<std::size_t>(carried_item)];
          if (amount <= 0) continue;
          if (shed + amount > observation.shed_capacity) {
            reason = "drop_would_overflow_shed";
            return false;
          }
          shed += amount;
          shed_items[static_cast<std::size_t>(carried_item)] += amount;
          inventory[static_cast<std::size_t>(carried_item)] = 0;
        }
      }
    }

    const int occupied_market_slots = turn.step == observation.step
        ? std::max(static_cast<int>(turn.market.size()), observation.market_slots_used)
        : static_cast<int>(turn.market.size());
    if (occupied_market_slots > observation.maximum_market_slots) {
      reason = "baseline_market_slot_overflow";
      return false;
    }
    for (const auto& order : turn.market) {
      if (!order.guaranteed_fill) continue;
      const int quantity = std::max(0, int(order.action.quantity));
      const int item = item_index(order.action.item);
      cash -= market_cost(order);
      if (order.action.op == fastkag::Op::BUY_SEED && item >= 0 &&
          item < fastkag::N_CROPS) {
        seeds[static_cast<std::size_t>(item)] += quantity;
      } else if (order.action.op == fastkag::Op::BUY_ANIMAL) {
        shed += quantity;
        if (item >= 9 && item < 12)
          shed_items[static_cast<std::size_t>(item)] += quantity;
      } else if (order.action.op == fastkag::Op::BUY_PRODUCT) {
        shed += quantity;
        if (item >= 0 && item < fastkag::N_ITEMS)
          shed_items[static_cast<std::size_t>(item)] += quantity;
      }
      if (cash < permanent_reserve + turn.extra_cash_reserve) {
        reason = "repair_would_starve_existing_or_future_purchase";
        return false;
      }
      if (shed > observation.shed_capacity) {
        reason = "repair_would_overflow_shed";
        return false;
      }
    }

    if (!purchase.existing && turn.step == purchase.step) {
      if (occupied_market_slots >= observation.maximum_market_slots) {
        reason = "no_market_slot_for_repair_purchase";
        return false;
      }
      const int quantity = std::max(0, int(repair_order.quantity));
      const int item = item_index(repair_order.item);
      cash -= purchase.marginal_cost;
      if (repair_order.op == fastkag::Op::BUY_SEED && item >= 0 &&
          item < fastkag::N_CROPS) {
        seeds[static_cast<std::size_t>(item)] += quantity;
      } else if (repair_order.op == fastkag::Op::BUY_ANIMAL) {
        shed += quantity;
        if (item >= 9 && item < 12)
          shed_items[static_cast<std::size_t>(item)] += quantity;
      }
      if (cash < permanent_reserve + turn.extra_cash_reserve) {
        reason = "repair_would_starve_existing_or_future_purchase";
        return false;
      }
      if (shed > observation.shed_capacity) {
        reason = "repair_would_overflow_shed";
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool is_movement(fastkag::Op operation) {
  return operation == fastkag::Op::NORTH || operation == fastkag::Op::SOUTH ||
         operation == fastkag::Op::EAST || operation == fastkag::Op::WEST;
}

int purchase_unit_cost(fastkag::Op operation, fastkag::Item item, int fallback) {
  const int index = item_index(item);
  if (operation == fastkag::Op::BUY_SEED) {
    constexpr std::array<int, fastkag::N_CROPS> costs{10, 20, 50, 100, 80};
    return index >= 0 && index < fastkag::N_CROPS
        ? costs[static_cast<std::size_t>(index)] : std::max(0, fallback);
  }
  if (operation == fastkag::Op::BUY_ANIMAL) {
    constexpr std::array<int, fastkag::N_ANIMALS> costs{300, 400, 500};
    return index >= 9 && index < 12
        ? costs[static_cast<std::size_t>(index - 9)] : std::max(0, fallback);
  }
  return std::max(0, fallback);
}

std::string audit_json(const AuditEvent& event) {
  std::ostringstream output;
  output << "{\"step\":" << event.step
         << ",\"transaction_id\":" << event.transaction_id
         << ",\"decision\":\"" << escape_json(event.decision)
         << "\",\"reason\":\"" << escape_json(event.reason)
         << "\",\"provenance\":\"" << escape_json(event.provenance)
         << "\",\"requested\":" << event.requested
         << ",\"filled\":" << event.filled
         << ",\"missing\":" << event.missing
         << ",\"absorbed_value\":" << event.absorbed_value << '}';
  return output.str();
}

FailureDebtScheduler::FailureDebtScheduler(SchedulerConfig config)
    : config_(config) {}

void FailureDebtScheduler::reset() {
  attempts_.clear();
  transactions_.clear();
  debts_.clear();
  audit_.clear();
  next_transaction_id_ = 1;
  next_debt_id_ = 1;
  latest_step_ = -1;
}

int FailureDebtScheduler::holding(const OwnObservation& observation,
                                  fastkag::Op operation,
                                  fastkag::Item item) const {
  const int index = item_index(item);
  if (operation == fastkag::Op::BUY_SEED && index >= 0 &&
      index < fastkag::N_CROPS) {
    return observation.seeds[static_cast<std::size_t>(index)];
  }
  if (operation == fastkag::Op::BUY_ANIMAL && index >= 9 && index < 12) {
    return observation.animals[static_cast<std::size_t>(index - 9)];
  }
  return -1;
}

void FailureDebtScheduler::append_audit(AuditEvent event) {
  audit_.push_back(std::move(event));
}

bool FailureDebtScheduler::record_attempt(const PurchaseIntent& intent,
                                          const OwnObservation& before_attempt) {
  if ((intent.operation != fastkag::Op::BUY_SEED &&
       intent.operation != fastkag::Op::BUY_ANIMAL) ||
      intent.quantity <= 0 || intent.deadline < before_attempt.step ||
      intent.chains.size() < static_cast<std::size_t>(intent.quantity) ||
      !complete_chain_shape(intent)) {
    append_audit({before_attempt.step, 0, "attempt_rejected",
                  "invalid_purchase_intent", intent.provenance,
                  intent.quantity, 0, intent.quantity, 0});
    return false;
  }
  const bool ambiguous = std::any_of(attempts_.begin(), attempts_.end(),
      [&](const AttemptState& attempt) {
        return attempt.awaiting && attempt.intent.operation == intent.operation &&
               attempt.intent.item == intent.item;
      });
  if (ambiguous) {
    append_audit({before_attempt.step, 0, "attempt_rejected",
                  "ambiguous_concurrent_same_item_attempt", intent.provenance,
                  intent.quantity, 0, intent.quantity, 0});
    return false;
  }
  const int before = holding(before_attempt, intent.operation, intent.item);
  if (before < 0) return false;
  AttemptState attempt;
  attempt.intent = intent;
  attempt.step = before_attempt.step;
  attempt.holding_before = before;
  attempt.attempts = 1;
  attempt.requested_quantity = intent.quantity;
  for (int unit = 0; unit < intent.quantity; ++unit)
    attempt.purchase_unit_indices.push_back(unit);
  attempt.awaiting = true;
  attempts_.push_back(std::move(attempt));
  append_audit({before_attempt.step, 0, "attempt_recorded",
                "awaiting_next_observation", intent.provenance,
                intent.quantity, 0, intent.quantity, 0});
  return true;
}

void FailureDebtScheduler::materialize_transaction(const AttemptState& attempt,
                                                   int filled, int missing,
                                                   int step) {
  TransactionState transaction;
  transaction.id = next_transaction_id_++;
  transaction.intent = attempt.intent;
  transaction.filled = filled;
  transaction.missing = missing;
  transaction.attempts = attempt.attempts;
  transaction.remaining_purchase = missing;

  Debt purchase;
  purchase.id = next_debt_id_++;
  purchase.transaction_id = transaction.id;
  purchase.source_attempt_id = attempt.intent.attempt_id;
  purchase.kind = DebtKind::MarketPurchase;
  purchase.action = {attempt.intent.operation, attempt.intent.item, missing};
  purchase.earliest_step = step;
  purchase.deadline = attempt.intent.deadline;
  purchase.economic_value = attempt.intent.economic_value;
  purchase.cash_required = missing * purchase_unit_cost(
      attempt.intent.operation, attempt.intent.item);
  purchase.shed_capacity_required =
      attempt.intent.operation == fastkag::Op::BUY_ANIMAL ? missing : 0;
  purchase.provenance = attempt.intent.provenance + ":settlement_shortfall";
  transaction.debt_ids.push_back(purchase.id);
  debts_.push_back(purchase);

  for (int unit = filled; unit < attempt.intent.quantity; ++unit) {
    // Separate missing units share only the purchase predecessor. Unit 2 is
    // not semantically dependent on completion of unit 1's lifecycle.
    std::uint64_t predecessor = purchase.id;
    const auto& chain = attempt.intent.chains[static_cast<std::size_t>(unit)];
    for (const auto& requirement : chain.requirements) {
      Debt action;
      action.id = next_debt_id_++;
      action.transaction_id = transaction.id;
      action.source_attempt_id = attempt.intent.attempt_id;
      action.kind = DebtKind::UnitAction;
      action.purchase_unit_index = unit;
      action.action = requirement.action;
      action.actor = requirement.actor;
      action.position = requirement.position;
      action.earliest_step = requirement.earliest_step;
      action.deadline = requirement.deadline;
      action.economic_value = requirement.economic_value;
      action.wheat_required = requirement.action.op == fastkag::Op::FEED ? 1 : 0;
      action.dependencies.push_back(predecessor);
      action.provenance = attempt.intent.provenance + ":" + requirement.provenance;
      predecessor = action.id;
      transaction.debt_ids.push_back(action.id);
      debts_.push_back(std::move(action));
    }
  }
  transactions_.push_back(std::move(transaction));
}

std::vector<SettlementRecord> FailureDebtScheduler::observe(
    const OwnObservation& observation) {
  if (latest_step_ >= 0 && observation.step < latest_step_) reset();
  if (latest_step_ == observation.step) return {};
  latest_step_ = observation.step;
  std::vector<SettlementRecord> result;
  for (auto& attempt : attempts_) {
    if (!attempt.awaiting || observation.step <= attempt.step) continue;
    const int item = item_index(attempt.intent.item);
    const int current = holding(observation, attempt.intent.operation,
                                attempt.intent.item);
    if (current < 0 || item < 0 || item >= fastkag::N_ITEMS) {
      result.push_back({attempt.intent.attempt_id, FillClass::Ambiguous,
                        attempt.intent.quantity, 0, attempt.intent.quantity,
                        "unsupported_or_invalid_holding"});
      attempt.awaiting = false;
      continue;
    }
    const int raw = current + observation.known_outflow[static_cast<std::size_t>(item)] -
                    observation.known_other_inflow[static_cast<std::size_t>(item)] -
                    attempt.holding_before;
    if (raw < 0 || raw > attempt.requested_quantity) {
      result.push_back({attempt.intent.attempt_id, FillClass::Ambiguous,
                        attempt.requested_quantity, 0, attempt.requested_quantity,
                        "inconsistent_holding_attribution"});
      if (attempt.transaction_id != 0) {
        const auto transaction = std::find_if(transactions_.begin(), transactions_.end(),
            [&](const TransactionState& candidate) {
              return candidate.id == attempt.transaction_id;
            });
        if (transaction != transactions_.end()) {
          transaction->retired = true;
          transaction->retire_reason = "ambiguous_retry_settlement";
          for (auto& debt : debts_) {
            if (debt.transaction_id == transaction->id)
              debt.status = DebtStatus::Retired;
          }
        }
      }
      append_audit({observation.step, attempt.transaction_id,
                    "settlement_ambiguous",
                    "inconsistent_holding_attribution",
                    attempt.intent.provenance, attempt.requested_quantity,
                    0, attempt.requested_quantity, 0});
      attempt.awaiting = false;
      continue;
    }
    const int filled = std::clamp(raw, 0, attempt.requested_quantity);
    const int missing = attempt.requested_quantity - filled;
    const auto classification = missing == 0 ? FillClass::Full
                              : filled == 0 ? FillClass::Zero
                                            : FillClass::Partial;
    const std::string reason = classification == FillClass::Full
        ? "causal_holding_delta_full"
        : classification == FillClass::Zero
            ? "causal_holding_delta_zero"
            : "causal_holding_delta_partial";
    result.push_back({attempt.intent.attempt_id, classification,
                      attempt.requested_quantity, filled, missing, reason});
    if (attempt.transaction_id != 0) {
      const auto transaction = std::find_if(transactions_.begin(), transactions_.end(),
          [&](const TransactionState& candidate) {
            return candidate.id == attempt.transaction_id;
          });
      if (transaction == transactions_.end()) {
        attempt.awaiting = false;
        continue;
      }
      transaction->filled += filled;
      transaction->missing = std::max(0, transaction->missing - filled);
      transaction->remaining_purchase =
          std::max(0, transaction->remaining_purchase - filled);
      Debt* purchase = nullptr;
      for (auto& debt : debts_) {
        if (debt.transaction_id == transaction->id &&
            debt.kind == DebtKind::MarketPurchase) {
          purchase = &debt;
          break;
        }
      }
      if (purchase != nullptr) {
        purchase->action.quantity = transaction->remaining_purchase;
        purchase->cash_required = transaction->remaining_purchase *
            purchase_unit_cost(purchase->action.op, purchase->action.item);
        purchase->shed_capacity_required =
            purchase->action.op == fastkag::Op::BUY_ANIMAL
                ? transaction->remaining_purchase : 0;
        purchase->status = transaction->remaining_purchase == 0
            ? DebtStatus::Confirmed : DebtStatus::Open;
      }
      // Unit nodes are deliberately not marked Scheduled here. Settlement
      // confirms inventory only; each due unit action requires a later
      // current-step commit and observed result.
    } else if (missing > 0) {
      materialize_transaction(attempt, filled, missing, observation.step);
      attempt.transaction_id = transactions_.back().id;
    }
    append_audit({observation.step, attempt.transaction_id, "settlement", reason,
                  attempt.intent.provenance, attempt.requested_quantity,
                  filled, missing, 0});
    attempt.awaiting = false;
  }
  for (auto& transaction : transactions_) {
    if (transaction.retired || transaction.completed) continue;
    const auto expired = std::find_if(debts_.begin(), debts_.end(),
        [&](const Debt& debt) {
          if (debt.transaction_id != transaction.id ||
              debt.status == DebtStatus::Consumed ||
              debt.status == DebtStatus::Retired) return false;
          if (debt.kind == DebtKind::MarketPurchase &&
              transaction.remaining_purchase == 0) return false;
          const int observation_grace =
              (debt.status == DebtStatus::Scheduled ||
               debt.status == DebtStatus::Attempted) ? 1 : 0;
          return observation.step > debt.deadline + observation_grace;
        });
    if (expired == debts_.end()) continue;
    transaction.retired = true;
    transaction.retire_reason = "debt_deadline_expired";
    for (auto& debt : debts_)
      if (debt.transaction_id == transaction.id &&
          debt.status != DebtStatus::Consumed)
        debt.status = DebtStatus::Retired;
    append_audit({observation.step, transaction.id, "transaction_retired",
                  "debt_deadline_expired", expired->provenance,
                  0, 0, transaction.remaining_purchase, 0});
  }
  return result;
}

std::vector<std::uint64_t> FailureDebtScheduler::open_transactions() const {
  std::vector<std::uint64_t> result;
  for (const auto& transaction : transactions_) {
    if (!transaction.retired && !transaction.completed)
      result.push_back(transaction.id);
  }
  return result;
}

RepairPlan FailureDebtScheduler::plan(std::uint64_t transaction_id,
                                      const OwnObservation& observation,
                                      const PlanWindow& window) {
  RepairPlan rejected;
  rejected.transaction_id = transaction_id;
  rejected.observation_fingerprint = observation_fingerprint(observation);
  auto transaction_it = std::find_if(transactions_.begin(), transactions_.end(),
      [&](const TransactionState& transaction) {
        return transaction.id == transaction_id;
      });
  if (transaction_it == transactions_.end()) {
    rejected.reason = "unknown_transaction";
    return rejected;
  }
  const auto& transaction = *transaction_it;
  const auto reject = [&](const std::string& reason) {
    rejected.reason = reason;
    rejected.audit.push_back({observation.step, transaction_id, "repair_rejected",
                              reason, transaction.intent.provenance,
                              transaction.intent.quantity, transaction.filled,
                              transaction.missing, 0});
    return rejected;
  };
  if (!config_.enabled) return reject("default_off");
  if (transaction.retired) return reject("transaction_retired");
  if (transaction.completed) return reject("transaction_completed");
  if (transaction.remaining_purchase > 0 &&
      transaction.attempts >= config_.maximum_attempts) {
    return reject("attempt_limit");
  }
  if (window.turns.empty()) return reject("empty_plan_window");
  int previous_step = -1;
  for (const auto& turn : window.turns) {
    if (turn.step <= previous_step || turn.step < observation.step ||
        turn.step - observation.step > config_.maximum_horizon) {
      return reject("invalid_or_out_of_horizon_plan_window");
    }
    if (config_.require_day_lifecycle &&
        ((previous_step < 0 && turn.step != observation.step) ||
         (previous_step >= 0 && turn.step != previous_step + 1))) {
      return reject("day_lifecycle_plan_window_has_missing_step");
    }
    if (config_.require_day_lifecycle) {
      for (const auto& slot : turn.units) {
        if (slot.actor > 0 &&
            (!turn.actor_availability_certified ||
             turn.certified_actor_count <= slot.actor)) {
          return reject("future_hand_not_rehired_or_day_scoped_actor_uncertified");
        }
      }
    }
    previous_step = turn.step;
  }

  std::vector<Debt*> transaction_debts;
  for (auto& debt : debts_) {
    if (debt.transaction_id == transaction_id) transaction_debts.push_back(&debt);
  }
  if (transaction_debts.empty() ||
      transaction_debts.front()->kind != DebtKind::MarketPurchase) {
    return reject("malformed_debt_graph");
  }
  if (transaction_debts.front()->status == DebtStatus::Attempted) {
    return reject("awaiting_repair_settlement");
  }
  std::vector<UnitRequirement> requirements;
  std::vector<Debt*> requirement_debts;
  std::vector<int> requirement_chain_ids;
  for (std::size_t index = 1; index < transaction_debts.size(); ++index) {
    const auto& debt = *transaction_debts[index];
    const bool open = debt.status == DebtStatus::Open ||
                      debt.status == DebtStatus::Failed;
    if (!open) continue;
    UnitRequirement requirement;
    requirement.action = debt.action;
    requirement.actor = debt.actor;
    requirement.position = debt.position;
    requirement.earliest_step = debt.earliest_step;
    requirement.deadline = debt.deadline;
    requirement.economic_value = debt.economic_value;
    requirement.provenance = debt.provenance;
    // The source requirement carries its configured ceiling in the original
    // chain; recover it by matching provenance/order.
    int configured_ceiling = 4;
    for (const auto& chain : transaction.intent.chains) {
      for (const auto& source : chain.requirements) {
        if (transaction.intent.provenance + ":" + source.provenance ==
            debt.provenance) configured_ceiling = source.maximum_absorbed_value;
      }
    }
    requirement.maximum_absorbed_value = configured_ceiling;
    requirements.push_back(std::move(requirement));
    requirement_debts.push_back(transaction_debts[index]);
    requirement_chain_ids.push_back(debt.purchase_unit_index);
  }
  if (requirements.empty()) {
    return reject(transaction.remaining_purchase > 0
        ? "purchase_has_no_complete_recovery_chain" : "no_pending_unit_debt");
  }

  std::vector<FlatSlot> slots;
  for (int turn_index = 0; turn_index < static_cast<int>(window.turns.size());
       ++turn_index) {
    const auto& turn = window.turns[static_cast<std::size_t>(turn_index)];
    for (int slot_index = 0; slot_index < static_cast<int>(turn.units.size());
         ++slot_index) {
      slots.push_back({turn_index, slot_index, turn.step,
                       &turn.units[static_cast<std::size_t>(slot_index)]});
    }
  }

  const auto repair_order = transaction_debts.front()->action;
  const bool purchase_required = transaction.remaining_purchase > 0;
  std::vector<HistoricUnitEvent> lifecycle_history;
  for (const auto& executed : transaction.executed_units) {
    const auto debt = std::find_if(transaction_debts.begin(), transaction_debts.end(),
        [&](const Debt* candidate) { return candidate->id == executed.debt_id; });
    if (debt != transaction_debts.end()) {
      lifecycle_history.push_back({executed.action, executed.position,
                                   executed.step, executed.actor,
                                   (*debt)->purchase_unit_index});
    }
  }
  std::vector<PurchaseCandidate> purchase_candidates;
  if (!purchase_required) {
    purchase_candidates.push_back({observation.step - 1, true, -1, 0});
  } else {
    for (const auto& turn : window.turns) {
      for (int slot = 0; slot < static_cast<int>(turn.market.size()); ++slot) {
        const auto& order = turn.market[static_cast<std::size_t>(slot)];
        const int surplus = std::max(0, int(order.action.quantity) -
                                        order.reserved_quantity);
        if (order.guaranteed_fill && order.action.op == repair_order.op &&
            order.action.item == repair_order.item &&
            surplus >= repair_order.quantity) {
          purchase_candidates.push_back({turn.step, true, slot, 0});
        }
      }
      const int occupied_slots = turn.step == observation.step
          ? std::max(static_cast<int>(turn.market.size()), observation.market_slots_used)
          : static_cast<int>(turn.market.size());
      if (occupied_slots < observation.maximum_market_slots) {
        purchase_candidates.push_back(
            {turn.step, false, -1,
             repair_order.quantity * purchase_unit_cost(repair_order.op,
                                                         repair_order.item)});
      }
    }
  }
  if (purchase_candidates.empty()) return reject("no_market_purchase_candidate");

  RepairPlan best;
  int best_score = kInfinity;
  std::string last_resource_rejection = "no_complete_unit_chain";
  for (const auto& candidate : purchase_candidates) {
    std::vector<int> acquisition_steps;
    acquisition_steps.reserve(requirement_debts.size());
    for (const auto* debt : requirement_debts) {
      acquisition_steps.push_back(debt->purchase_unit_index < transaction.filled
          ? observation.step - 1 : candidate.step);
    }
    const auto assignment = assign_requirements(
        requirements, requirement_chain_ids, slots, acquisition_steps);
    if (!assignment.feasible ||
        assignment.absorbed > config_.maximum_total_absorbed_value) {
      continue;
    }
    RepairPlan current;
    current.accepted = true;
    current.transaction_id = transaction_id;
    current.observation_fingerprint = rejected.observation_fingerprint;
    current.reason = !purchase_required
        ? "confirmed_inventory_unit_suffix_constraints_satisfied"
        : candidate.existing
        ? "declared_constraints_use_unreserved_existing_purchase"
        : "declared_constraints_satisfied";
    current.purchase_cost = candidate.marginal_cost;
    current.absorbed_value = assignment.absorbed;
    current.market = {purchase_required, candidate.step, repair_order, candidate.existing,
                      candidate.existing_slot};
    if (config_.require_day_lifecycle) {
      std::string lifecycle_reason;
      if (!validate_day_lifecycle(observation, window,
                                  requirements, requirement_chain_ids, slots,
                                  assignment, lifecycle_history,
                                  lifecycle_reason)) {
        last_resource_rejection = lifecycle_reason;
        continue;
      }
      current.day_lifecycle_complete = true;
    }
    for (std::size_t index = 0; index < assignment.choices.size(); ++index) {
      const auto& choice = assignment.choices[index];
      current.scheduled_debt_ids.push_back(requirement_debts[index]->id);
      const auto& flat = slots[static_cast<std::size_t>(choice.flat_index)];
      const auto& slot = *flat.slot;
      current.unit_schedule.push_back({requirement_debts[index]->id,
                                       flat.step, slot.actor, flat.slot_index,
                                       choice.existing});
      if (!choice.existing) {
        current.units.push_back({flat.step, slot.actor, flat.slot_index,
                                 slot.original, requirements[index].action,
                                 slot.economic_value,
                                 requirement_debts[index]->id});
      }
    }
    std::vector<int> valued_chains;
    for (const int chain : requirement_chain_ids)
      if (std::find(valued_chains.begin(), valued_chains.end(), chain) ==
          valued_chains.end()) valued_chains.push_back(chain);
    const int recovered_value = transaction.intent.quantity > 0
        ? transaction.intent.economic_value *
              static_cast<int>(valued_chains.size()) /
              transaction.intent.quantity
        : 0;
    current.net_value = recovered_value - current.purchase_cost -
                        current.absorbed_value;
    if (current.net_value < config_.minimum_net_value) {
      last_resource_rejection = "negative_conservative_transaction_value";
      continue;
    }
    std::string resource_reason;
    if (!validate_resources(observation, window, candidate, repair_order,
                            current.units, resource_reason)) {
      last_resource_rejection = resource_reason;
      continue;
    }
    const int score = assignment.score + candidate.marginal_cost * 10 +
                      candidate.step - observation.step;
    if (score < best_score) {
      best_score = score;
      best = std::move(current);
    }
  }
  if (!best.accepted) return reject(last_resource_rejection);

  if (best.market.required) {
    best.audit.push_back({observation.step, transaction_id, "market_debt_planned",
                        best.market.uses_existing_surplus
                            ? "existing_unreserved_purchase_surplus"
                            : "new_purchase_slot",
                        transaction_debts.front()->provenance,
                        transaction.missing, 0, transaction.missing, 0});
  }
  for (const auto debt_id : best.scheduled_debt_ids) {
    const auto debt = std::find_if(transaction_debts.begin(),
        transaction_debts.end(), [&](const Debt* candidate) {
          return candidate->id == debt_id;
        });
    const auto patch = std::find_if(best.units.begin(), best.units.end(),
        [&](const UnitPatch& candidate) { return candidate.debt_id == debt_id; });
    best.audit.push_back({patch == best.units.end() ? observation.step : patch->step,
                          transaction_id, "unit_debt_planned",
                          patch == best.units.end()
                              ? "existing_equivalent_action"
                              : "low_loss_non_move_absorption",
                          debt == transaction_debts.end()
                              ? transaction.intent.provenance
                              : (*debt)->provenance,
                          1, 0, 1,
                          patch == best.units.end() ? 0 : patch->absorbed_value});
  }
  best.audit.push_back({observation.step, transaction_id, "repair_accepted",
                        best.reason, transaction.intent.provenance,
                        transaction.intent.quantity, transaction.filled,
                        transaction.missing, best.absorbed_value});
  return best;
}

bool FailureDebtScheduler::commit(const RepairPlan& plan,
                                  const OwnObservation& before_execution) {
  if (!plan.accepted) return false;
  if (plan.observation_fingerprint !=
      observation_fingerprint(before_execution)) return false;
  const auto transaction = std::find_if(transactions_.begin(), transactions_.end(),
      [&](const TransactionState& candidate) {
        return candidate.id == plan.transaction_id;
      });
  if (transaction == transactions_.end() || transaction->retired ||
      transaction->completed || plan.scheduled_debt_ids.empty()) {
    return false;
  }
  std::vector<Debt*> scheduled;
  std::vector<std::uint64_t> seen_debt_ids;
  for (const auto debt_id : plan.scheduled_debt_ids) {
    if (std::find(seen_debt_ids.begin(), seen_debt_ids.end(), debt_id) !=
        seen_debt_ids.end()) return false;
    seen_debt_ids.push_back(debt_id);
    const auto debt = std::find_if(debts_.begin(), debts_.end(),
        [&](const Debt& candidate) {
          return candidate.id == debt_id &&
                 candidate.transaction_id == plan.transaction_id &&
                 candidate.kind == DebtKind::UnitAction;
        });
    if (debt == debts_.end() ||
        (debt->status != DebtStatus::Open &&
         debt->status != DebtStatus::Failed)) return false;
    scheduled.push_back(&*debt);
  }
  if (plan.unit_schedule.size() != plan.scheduled_debt_ids.size()) return false;
  for (const auto debt_id : plan.scheduled_debt_ids) {
    if (std::count_if(plan.unit_schedule.begin(), plan.unit_schedule.end(),
                      [&](const UnitSchedule& schedule) {
                        return schedule.debt_id == debt_id;
                      }) != 1) return false;
  }

  Debt* purchase = nullptr;
  for (auto& debt : debts_) {
    if (debt.transaction_id == plan.transaction_id &&
        debt.kind == DebtKind::MarketPurchase) {
      purchase = &debt;
      break;
    }
  }
  if (purchase == nullptr) return false;

  std::vector<Debt*> due_debts;
  for (const auto& schedule : plan.unit_schedule) {
    if (schedule.step != before_execution.step) continue;
    const auto debt = std::find_if(scheduled.begin(), scheduled.end(),
        [&](const Debt* candidate) { return candidate->id == schedule.debt_id; });
    if (debt == scheduled.end()) return false;
    for (const auto predecessor_id : (*debt)->dependencies) {
      const auto predecessor = std::find_if(debts_.begin(), debts_.end(),
          [&](const Debt& candidate) { return candidate.id == predecessor_id; });
      if (predecessor == debts_.end()) return false;
      const bool ready = predecessor->kind == DebtKind::MarketPurchase
          ? (predecessor->status == DebtStatus::Confirmed ||
             (*debt)->purchase_unit_index < transaction->filled)
          : predecessor->status == DebtStatus::Consumed;
      if (!ready) return false;
    }
    due_debts.push_back(*debt);
  }

  if (plan.market.required) {
    if (plan.market.step != before_execution.step ||
        transaction->remaining_purchase <= 0 ||
        purchase->status != DebtStatus::Open ||
        plan.market.addition.op != purchase->action.op ||
        plan.market.addition.item != purchase->action.item ||
        plan.market.addition.quantity != transaction->remaining_purchase ||
        transaction->attempts >= config_.maximum_attempts) {
      return false;
    }
    if (std::any_of(attempts_.begin(), attempts_.end(),
                    [&](const AttemptState& attempt) {
                      return attempt.awaiting &&
                             attempt.transaction_id == plan.transaction_id;
                    })) return false;

    AttemptState attempt;
    attempt.intent = transaction->intent;
    attempt.intent.quantity = transaction->remaining_purchase;
    attempt.intent.attempt_id =
        (transaction->intent.attempt_id << 8) ^
        static_cast<std::uint64_t>(transaction->attempts + 1);
    attempt.step = before_execution.step;
    attempt.holding_before = holding(before_execution, purchase->action.op,
                                     purchase->action.item);
    if (attempt.holding_before < 0) return false;
    attempt.attempts = transaction->attempts + 1;
    attempt.requested_quantity = transaction->remaining_purchase;
    for (const auto* debt : scheduled) {
      if (debt->purchase_unit_index < transaction->filled) continue;
      if (std::find(attempt.purchase_unit_indices.begin(),
                    attempt.purchase_unit_indices.end(),
                    debt->purchase_unit_index) ==
          attempt.purchase_unit_indices.end()) {
        attempt.purchase_unit_indices.push_back(debt->purchase_unit_index);
      }
    }
    std::sort(attempt.purchase_unit_indices.begin(),
              attempt.purchase_unit_indices.end());
    if (static_cast<int>(attempt.purchase_unit_indices.size()) !=
        attempt.requested_quantity) return false;
    attempt.scheduled_debt_ids = plan.scheduled_debt_ids;
    attempt.awaiting = true;
    attempt.transaction_id = plan.transaction_id;
    attempts_.push_back(std::move(attempt));
    purchase->status = DebtStatus::Attempted;
    ++transaction->attempts;
    // Official order is unit phase then market phase. Confirmed partial-fill
    // chains due now are atomically registered with the emitted market retry.
    for (auto* due_debt : due_debts)
      due_debt->status = DebtStatus::Scheduled;
  } else {
    if (transaction->remaining_purchase != 0 ||
        purchase->status != DebtStatus::Confirmed) return false;
    // A future certificate is preview-only. Replan when its first action is
    // actually due instead of reserving future nodes prematurely.
    if (due_debts.empty()) return false;
    for (auto* due_debt : due_debts)
      due_debt->status = DebtStatus::Scheduled;
  }

  for (const auto& event : plan.audit) append_audit(event);
  append_audit({before_execution.step, plan.transaction_id, "repair_committed",
                plan.market.required ? "purchase_emitted_awaiting_observation"
                                     : "unit_suffix_scheduled",
                transaction->intent.provenance,
                plan.market.required ? transaction->remaining_purchase : 0,
                0, plan.market.required ? transaction->remaining_purchase : 0,
                plan.absorbed_value});
  return true;
}

bool FailureDebtScheduler::observe_unit_result(
    std::uint64_t debt_id, const UnitExecutionEvidence& evidence) {
  const auto debt = std::find_if(debts_.begin(), debts_.end(),
      [&](const Debt& candidate) { return candidate.id == debt_id; });
  if (debt == debts_.end() || debt->kind != DebtKind::UnitAction ||
      debt->status != DebtStatus::Scheduled) return false;
  const auto transaction = std::find_if(transactions_.begin(), transactions_.end(),
      [&](const TransactionState& candidate) {
        return candidate.id == debt->transaction_id;
      });
  if (transaction == transactions_.end() || transaction->retired ||
      transaction->completed) return false;

  if (evidence.emitted.op != debt->action.op ||
      evidence.emitted.item != debt->action.item) return false;
  bool success = false;
  switch (debt->action.op) {
    case fastkag::Op::PLANT:
      success = evidence.observed_tile_kind == fastkag::TileKind::PLANT &&
                evidence.observed_tile_item == debt->action.item;
      break;
    case fastkag::Op::PLACE:
      success = evidence.observed_tile_kind == fastkag::TileKind::ANIMAL &&
                evidence.observed_tile_item == debt->action.item;
      break;
    case fastkag::Op::WATER: success = evidence.watered_today; break;
    case fastkag::Op::FEED: success = evidence.fed_today; break;
    case fastkag::Op::CARE: success = evidence.cared_today; break;
    case fastkag::Op::HARVEST:
      success = evidence.own_inventory_delta > 0 ||
                evidence.yield_after < evidence.yield_before;
      break;
    case fastkag::Op::PICKUP:
      success = evidence.own_inventory_delta >=
                std::max(1, int(debt->action.quantity));
      break;
    case fastkag::Op::DIG:
      success = evidence.observed_tile_kind == fastkag::TileKind::EMPTY;
      break;
    default: success = evidence.generic_effect_verified; break;
  }

  if (success) {
    debt->status = DebtStatus::Consumed;
    transaction->executed_units.push_back(
        {debt->id, debt->action, debt->actor, debt->position, evidence.step});
  } else {
    debt->status = DebtStatus::Failed;
    std::vector<std::uint64_t> frontier{debt->id};
    while (!frontier.empty()) {
      const auto predecessor = frontier.back();
      frontier.pop_back();
      for (auto& candidate : debts_) {
        if (candidate.transaction_id != transaction->id ||
            candidate.kind != DebtKind::UnitAction ||
            candidate.purchase_unit_index != debt->purchase_unit_index ||
            std::find(candidate.dependencies.begin(), candidate.dependencies.end(),
                      predecessor) == candidate.dependencies.end()) {
          continue;
        }
        if (candidate.status != DebtStatus::Retired)
          candidate.status = DebtStatus::Open;
        frontier.push_back(candidate.id);
      }
    }
  }

  const bool all_consumed = std::all_of(debts_.begin(), debts_.end(),
      [&](const Debt& candidate) {
        return candidate.transaction_id != transaction->id ||
               candidate.kind != DebtKind::UnitAction ||
               candidate.status == DebtStatus::Consumed;
      });
  if (transaction->remaining_purchase == 0 && all_consumed)
    transaction->completed = true;
  append_audit({evidence.step, transaction->id,
                success ? "unit_debt_consumed" : "unit_debt_reopened",
                evidence.provenance.empty()
                    ? (success ? "verified_success" : "verified_failure")
                    : evidence.provenance,
                debt->provenance, 1, success ? 1 : 0, success ? 0 : 1, 0});
  return true;
}

}  // namespace g001::failure_debt
